"""An out-of-memory failure has to say which ceiling it hit.

The case this exists for: Boltz-2 at 3012 tokens dies with "Out of memory while
trying to allocate 60.75GiB" on a 96 GiB card, which reads as a model that does
not fit. It fits. The run needs 73.32 GiB and JAX's default pool is 0.75 of the
device, so it was 1.6 GiB short of a ceiling with 24 GiB of the card behind it.
"""

from __future__ import annotations

import pytest

from foldjax import oom

GIB = 2**30
CARD = 96 * GIB


def test_a_shortfall_inside_the_uncovered_device_is_named_as_the_pool(monkeypatch):
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
    monkeypatch.setattr(
        oom, "_pool_card_and_used", lambda: (int(0.75 * CARD), CARD, int(12 * GIB))
    )
    message = oom.diagnose(
        RuntimeError(
            "RESOURCE_EXHAUSTED: Out of memory while trying to allocate 60.75GiB."
        )
    )
    assert message is not None
    assert "72.0 GiB" in message, "the pool, not the card"
    assert "96.0 GiB" in message
    assert "not the card's" in message


def test_a_request_larger_than_the_device_is_not_blamed_on_the_pool(monkeypatch):
    """Raising the fraction would waste another run before failing the same way.

    OpenDDE at 3012 is this case: it needs several times the card, and the pool
    is not what stopped it.
    """
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
    monkeypatch.setattr(
        oom, "_pool_card_and_used", lambda: (int(0.75 * CARD), CARD, int(12 * GIB))
    )
    message = oom.diagnose(
        RuntimeError(
            "RESOURCE_EXHAUSTED: Out of memory while trying to allocate 300.00GiB."
        )
    )
    assert message is not None
    assert "not the card's" not in message
    assert "will not help" in message


def test_nothing_is_said_when_the_pool_cannot_be_read(monkeypatch):
    """Silence beats a guess. A wrong explanation of a failure costs more."""
    monkeypatch.setattr(oom, "_pool_card_and_used", lambda: (None, None, None))
    assert (
        oom.diagnose(RuntimeError("RESOURCE_EXHAUSTED: Out of memory")) is None
    )


def test_other_failures_are_left_alone():
    assert oom.diagnose(ValueError("shape mismatch")) is None
    assert not oom.is_out_of_memory(ValueError("shape mismatch"))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("allocate 60.75GiB.", 60.75 * GIB),
        ("allocate 512.00MiB.", 512 * 2**20),
        ("allocate 16.11GiB. [tf-allocator", 16.11 * GIB),
    ],
)
def test_the_requested_size_is_read_out_of_the_message(text, expected):
    assert oom._requested_bytes(text) == pytest.approx(expected, rel=1e-6)


def test_preallocation_off_means_there_is_no_pool_to_be_short_of(monkeypatch):
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    assert oom._pool_card_and_used() == (None, None, None)


def test_an_unparseable_fraction_falls_back_to_jax_s_default(monkeypatch):
    # Importing any model sets the current spelling process-wide, and it is
    # read first, so pinning only the deprecated one leaves this test reading
    # somebody else's import.
    monkeypatch.delenv(oom.CURRENT_FRACTION_ENV, raising=False)
    monkeypatch.setenv(oom.FRACTION_ENV, "not-a-number")
    assert oom.mem_fraction() == oom.DEFAULT_MEM_FRACTION


def test_the_cli_asks_for_more_of_the_card_than_jax_would(monkeypatch):
    """0.75 is JAX's default for a shared device. One prediction is not that.

    Boltz-2 at 3012 tokens needs 73.33 GiB; 0.75 of this 95.6 GiB card is 71.7,
    so the job died 1.6 GiB short with a quarter of the device held in reserve.
    """
    from foldjax import cli

    monkeypatch.delenv(oom.CURRENT_FRACTION_ENV, raising=False)
    monkeypatch.delenv(oom.FRACTION_ENV, raising=False)
    cli._apply_mem_fraction(None)
    # The property, not the spelling: jaxlib has two names for this, and
    # adding ours beside a caller's choice costs them the GPU -- so which
    # name gets written is exactly the part that may change.
    assert oom.mem_fraction() == oom.PREDICT_MEM_FRACTION
    assert oom.PREDICT_MEM_FRACTION > oom.DEFAULT_MEM_FRACTION


def test_an_explicit_environment_setting_is_not_overridden(monkeypatch):
    """Someone who exported it is sharing the card, and knows better than a default."""
    from foldjax import cli

    monkeypatch.delenv(oom.CURRENT_FRACTION_ENV, raising=False)
    monkeypatch.setenv(oom.FRACTION_ENV, "0.4")
    cli._apply_mem_fraction(None)
    assert oom.mem_fraction() == 0.4


def test_the_flag_beats_both(monkeypatch):
    from foldjax import cli

    monkeypatch.delenv(oom.CURRENT_FRACTION_ENV, raising=False)
    monkeypatch.setenv(oom.FRACTION_ENV, "0.4")
    cli._apply_mem_fraction(0.55)
    assert oom.mem_fraction() == 0.55
    # Replaced in place, not added beside it: the pair is what breaks CUDA.
    assert oom.CURRENT_FRACTION_ENV not in __import__("os").environ


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.01, 2.0])
def test_a_fraction_outside_the_unit_interval_is_refused(bad, monkeypatch):
    from foldjax import cli

    monkeypatch.delenv(oom.CURRENT_FRACTION_ENV, raising=False)
    monkeypatch.delenv(oom.FRACTION_ENV, raising=False)
    with pytest.raises(ValueError, match="must be in"):
        cli._apply_mem_fraction(bad)


def test_pool_fraction_is_never_set_as_a_second_spelling() -> None:
    """Both spellings at once cost the caller their GPU, quietly.

    jaxlib renamed this knob. Either name works alone; together they raise
    inside the CUDA plugin's `initialize()`, so the plugin does not load and
    JAX reports "an NVIDIA GPU may be present on this machine, but a
    CUDA-enabled jaxlib is not installed" and runs on the CPU. The jaxlib is
    installed, and nothing in that message points at the pair of variables.
    FoldJAX used to add its own spelling with `setdefault`, so a caller who
    followed jaxlib's current documentation lost the GPU by importing us.
    """
    import os

    from foldjax import oom

    for present, absent in (
        (oom.CURRENT_FRACTION_ENV, oom.FRACTION_ENV),
        (oom.FRACTION_ENV, oom.CURRENT_FRACTION_ENV),
    ):
        environment = {present: "0.95"}
        original = dict(os.environ)
        try:
            os.environ.clear()
            os.environ.update(environment)
            oom.set_mem_fraction(0.90)
            assert os.environ[present] == "0.95", "an explicit choice must stand"
            assert absent not in os.environ, "the other spelling must stay absent"
            assert oom.mem_fraction() == 0.95
        finally:
            os.environ.clear()
            os.environ.update(original)


def test_importing_a_model_keeps_a_chosen_pool_fraction_alone() -> None:
    """The import that used to break it: a real interpreter, not a monkeypatch."""
    import os
    import subprocess
    import sys

    program = (
        "import os, foldjax.models.openfold3;"
        "print(os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION'))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env={
            key: value
            for key, value in os.environ.items()
            if key != "XLA_PYTHON_CLIENT_MEM_FRACTION"
        }
        | {"XLA_CLIENT_MEM_FRACTION": "0.95", "JAX_PLATFORMS": "cpu"},
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.strip() == "None"


@pytest.mark.parametrize(
    ("used_gib", "requested_gib", "expected"),
    [
        (2.0, 120.0, "past the device itself"),
        (2.0, 88.0, "past the pool"),
        # Fits both, so the answer is placement rather than capacity. Asserted
        # on what the reader is told to do -- that a bigger fraction is not the
        # lever -- rather than on the sentence that says it, which has been
        # reworded once already for being read as classic fragmentation when
        # the pool was nearly empty.
        (2.0, 60.0, "a larger fraction will not help"),
    ],
)
def test_the_explainer_separates_a_pool_limit_from_fragmentation(
    monkeypatch, used_gib, requested_gib, expected
) -> None:
    """It used to call all three "past the pool" and advise a bigger fraction.

    The branch compared the total against the *card* and then reported the
    *pool*, so a request that fits both printed "This is the pool's limit" and
    sent the reader to raise a fraction that cannot help. Measured: a 90.5 GiB
    need failed inside a 93.1 GiB pool, and failed again at 0.98.
    """
    gib = 2**30
    monkeypatch.setattr(
        oom,
        "_pool_card_and_used",
        lambda: (90.0 * gib, 95.0 * gib, used_gib * gib),
    )
    error = RuntimeError(
        f"RESOURCE_EXHAUSTED: Out of memory while trying to "
        f"allocate {requested_gib:.2f}GiB."
    )

    explanation = oom.diagnose(error)

    assert explanation is not None
    assert expected in explanation
