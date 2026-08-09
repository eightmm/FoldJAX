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
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "not-a-number")
    assert oom.mem_fraction() == oom.DEFAULT_MEM_FRACTION


def test_the_cli_asks_for_more_of_the_card_than_jax_would(monkeypatch):
    """0.75 is JAX's default for a shared device. One prediction is not that.

    Boltz-2 at 3012 tokens needs 73.33 GiB; 0.75 of this 95.6 GiB card is 71.7,
    so the job died 1.6 GiB short with a quarter of the device held in reserve.
    """
    from foldjax import cli

    monkeypatch.delenv(oom.FRACTION_ENV, raising=False)
    cli._apply_mem_fraction(None)
    assert float(__import__("os").environ[oom.FRACTION_ENV]) == oom.PREDICT_MEM_FRACTION
    assert oom.PREDICT_MEM_FRACTION > oom.DEFAULT_MEM_FRACTION


def test_an_explicit_environment_setting_is_not_overridden(monkeypatch):
    """Someone who exported it is sharing the card, and knows better than a default."""
    from foldjax import cli

    monkeypatch.setenv(oom.FRACTION_ENV, "0.4")
    cli._apply_mem_fraction(None)
    assert __import__("os").environ[oom.FRACTION_ENV] == "0.4"


def test_the_flag_beats_both(monkeypatch):
    from foldjax import cli

    monkeypatch.setenv(oom.FRACTION_ENV, "0.4")
    cli._apply_mem_fraction(0.55)
    assert __import__("os").environ[oom.FRACTION_ENV] == "0.55"


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.01, 2.0])
def test_a_fraction_outside_the_unit_interval_is_refused(bad, monkeypatch):
    from foldjax import cli

    monkeypatch.delenv(oom.FRACTION_ENV, raising=False)
    with pytest.raises(ValueError, match="must be in"):
        cli._apply_mem_fraction(bad)
