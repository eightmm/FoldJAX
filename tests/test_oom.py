"""An out-of-memory failure has to say which ceiling it hit.

The case this exists for: Boltz-2 at 3012 tokens dies with "Out of memory while
trying to allocate 60.75GiB" on a 96 GiB card, which reads as a model that does
not fit. It fits. The run needs 73.32 GiB and JAX's default pool is 0.75 of the
device, so it was 1.6 GiB short of a ceiling with 24 GiB of the card behind it.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from foldjax import oom

GIB = 2**30
CARD = 96 * GIB


def test_post_failure_usage_does_not_become_a_required_peak(monkeypatch):
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
    assert "in use 12.0 GiB" in message
    assert "failed allocation was 60.8 GiB" in message
    assert "needed" not in message
    assert "72.8 GiB" not in message


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
    assert oom.diagnose(RuntimeError("RESOURCE_EXHAUSTED: Out of memory")) is None


def test_other_failures_are_left_alone():
    assert oom.diagnose(ValueError("shape mismatch")) is None
    assert not oom.is_out_of_memory(ValueError("shape mismatch"))


@pytest.mark.parametrize(
    "message",
    [
        "RESOURCE_EXHAUSTED: too many open files",
        "RESOURCE_EXHAUSTED: failed to create a worker thread",
        "RESOURCE_EXHAUSTED: disk quota exceeded",
        "RESOURCE_EXHAUSTED",
    ],
)
def test_non_allocator_resource_exhaustion_is_not_gpu_oom(message, monkeypatch):
    def unexpected_stats():
        pytest.fail("non-memory resource failures must not read GPU memory statistics")

    monkeypatch.setattr(oom, "_pool_card_and_used", unexpected_stats)
    assert not oom.is_out_of_memory(RuntimeError(message))
    assert oom.diagnose(RuntimeError(message)) is None


@pytest.mark.parametrize(
    "message",
    ["CUDA_ERROR_OUT_OF_MEMORY", "cudaErrorMemoryAllocation", "hipErrorOutOfMemory"],
)
def test_driver_allocator_oom_codes_are_recognized(message):
    assert oom.is_out_of_memory(RuntimeError(message))


def test_nested_oom_group_uses_the_leaf_allocation(monkeypatch):
    monkeypatch.setenv(oom.CURRENT_FRACTION_ENV, "0.75")
    monkeypatch.setattr(
        oom, "_pool_card_and_used", lambda: (int(0.75 * CARD), CARD, int(12 * GIB))
    )
    error = ExceptionGroup(
        "kernel failed",
        [
            ExceptionGroup(
                "triton",
                [
                    RuntimeError(
                        "RESOURCE_EXHAUSTED: Out of memory trying to allocate 60.75GiB."
                    )
                ],
            )
        ],
    )
    assert oom.is_out_of_memory(error)
    explanation = oom.diagnose(error)
    assert "failed allocation was 60.8 GiB" in explanation
    assert "needed" not in explanation


def test_mixed_kernel_failures_are_not_reported_as_only_gpu_oom():
    error = ExceptionGroup(
        "kernel failed",
        [RuntimeError("Out of memory"), ValueError("unsupported architecture")],
    )
    assert not oom.is_out_of_memory(error)
    assert oom.diagnose(error) is None


def test_multiple_oom_attempts_do_not_claim_one_allocation_total(monkeypatch):
    monkeypatch.setattr(
        oom, "_pool_card_and_used", lambda: (int(0.75 * CARD), CARD, int(12 * GIB))
    )
    error = ExceptionGroup(
        "kernel failed",
        [
            RuntimeError("Out of memory trying to allocate 60.75GiB."),
            RuntimeError("Out of memory trying to allocate 1.00GiB."),
        ],
    )
    explanation = oom.diagnose(error)
    assert explanation is not None
    assert "at that moment" not in explanation


def test_memory_budget_reads_the_explicitly_selected_device(monkeypatch):
    import jax

    device = SimpleNamespace(
        memory_stats=lambda: {"bytes_limit": 16 * GIB, "bytes_in_use": 4 * GIB}
    )
    monkeypatch.setenv(oom.CURRENT_FRACTION_ENV, "0.5")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    monkeypatch.setattr(jax, "config", SimpleNamespace(jax_default_device=device))
    monkeypatch.setattr(jax, "devices", lambda: [])
    monkeypatch.setattr(
        "foldjax._cuda_memory.total_device_bytes",
        lambda chosen: 32 * GIB if chosen is device else None,
    )
    assert oom._pool_card_and_used() == (16 * GIB, 32 * GIB, 4 * GIB)


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
        (2.0, 92.0, "past the pool"),
        (2.0, 60.0, "do not establish the total memory requirement"),
    ],
)
def test_only_the_failed_block_can_prove_a_capacity_limit(
    monkeypatch, used_gib, requested_gib, expected
) -> None:
    """A post-failure in-use snapshot cannot prove a co-live requirement."""
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


def test_the_advice_names_the_sample_axis_lever_a_caller_can_actually_type(
    monkeypatch,
) -> None:
    """`--num-samples` was the only sample-axis advice, and it drops predictions.

    The option name is the contract here: the message tells a caller what to
    type, so renaming the knob has to reach this string too. The caveat is
    measured -- the same option rescued OpenFold3 at 4,100 tokens and moved
    Protenix by one mebibyte -- so the sentence promises a lever to try, not a
    lever that works.
    """
    gib = 2**30
    monkeypatch.setattr(
        oom, "_pool_card_and_used", lambda: (90.0 * gib, 95.0 * gib, 2 * gib)
    )
    message = oom.diagnose(
        RuntimeError(
            "RESOURCE_EXHAUSTED: Out of memory while trying to allocate 88.00GiB."
        )
    )
    assert message is not None
    assert "diffusion_chunk_size" in message
    assert "model-specific" in message


# ---------------------------------------------------------------------------
# The rendezvous bound: an OOM that hangs cannot be diagnosed at all
# ---------------------------------------------------------------------------


@pytest.fixture
def mesh_record():
    """A clean CP record before and after, since it outlives its context."""
    oom.clear_mesh_record()
    yield oom.record_mesh
    oom.clear_mesh_record()


def test_the_rendezvous_bound_reaches_xla_flags_exactly_once(monkeypatch) -> None:
    """Twice called, once composed -- and the caller's own flags survive.

    Two callers ask for this bound, `foldjax.cli` while it resolves arguments
    and `models/_cp.context_parallel` when it finds no backend yet, and a CLI
    run reaches both. A second `--xla_gpu_nccl_termination_timeout_seconds` in
    `XLA_FLAGS` is not a duplicate XLA ignores: the later value wins, so a
    composer that appended blindly would decide which one that is by accident.
    """
    monkeypatch.delenv(oom.RENDEZVOUS_ENV, raising=False)
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

    first = oom.set_rendezvous_timeout()
    second = oom.set_rendezvous_timeout()

    flags = os.environ["XLA_FLAGS"]
    assert first == flags
    assert second is None, "nothing left to write"
    assert flags.count(oom.RENDEZVOUS_FLAG) == 1
    assert f"--{oom.RENDEZVOUS_FLAG}={oom.CP_RENDEZVOUS_SECONDS}" in flags
    assert "--xla_force_host_platform_device_count=4" in flags


def test_a_rendezvous_bound_the_caller_chose_is_left_alone(monkeypatch) -> None:
    """Someone who set this flag has a reason, as with the pool fraction."""
    monkeypatch.delenv(oom.RENDEZVOUS_ENV, raising=False)
    chosen = f"--{oom.RENDEZVOUS_FLAG}=30"
    monkeypatch.setenv("XLA_FLAGS", chosen)

    assert oom.set_rendezvous_timeout() is None
    assert os.environ["XLA_FLAGS"] == chosen


@pytest.mark.parametrize(
    ("value", "seconds"),
    [("45", 45), ("", oom.CP_RENDEZVOUS_SECONDS), ("-1", None)],
)
def test_the_rendezvous_bound_is_overridable_and_can_be_declined(
    monkeypatch, value: str, seconds: int | None
) -> None:
    """A negative value is XLA's own spelling of "wait forever"."""
    monkeypatch.setenv(oom.RENDEZVOUS_ENV, value)
    monkeypatch.setenv("XLA_FLAGS", "")

    assert oom.rendezvous_timeout_seconds() == seconds

    oom.set_rendezvous_timeout()
    if seconds is None:
        assert oom.RENDEZVOUS_FLAG not in os.environ["XLA_FLAGS"]
    else:
        assert f"--{oom.RENDEZVOUS_FLAG}={seconds}" in os.environ["XLA_FLAGS"]


@pytest.mark.parametrize("bad", ["0", "soon", "600s", "6.5"])
def test_a_rendezvous_bound_that_cannot_mean_seconds_is_refused(
    monkeypatch, bad: str
) -> None:
    """Including `0`, which to XLA is a zero-second timeout and not "off"."""
    monkeypatch.setenv(oom.RENDEZVOUS_ENV, bad)
    monkeypatch.setenv("XLA_FLAGS", "")

    with pytest.raises(ValueError, match=oom.RENDEZVOUS_ENV):
        oom.set_rendezvous_timeout()
    assert os.environ["XLA_FLAGS"] == ""


@pytest.mark.parametrize(
    ("platforms", "possible"),
    [(None, True), ("", True), ("cuda", True), ("cuda,cpu", True), ("cpu", False)],
)
def test_only_a_process_that_could_reach_a_gpu_is_given_a_gpu_flag(
    monkeypatch, platforms: str | None, possible: bool
) -> None:
    """`XLA_FLAGS` is inherited by every child, so a CPU run is left alone.

    The flag itself is inert off GPU. What is not inert is the variable: a
    CPU-pinned test process that composed it would hand a `--xla_gpu_*` flag to
    every probe subprocess it spawns.
    """
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    if platforms is None:
        monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    else:
        monkeypatch.setenv("JAX_PLATFORMS", platforms)

    assert oom.gpu_is_possible() is possible


def test_an_oom_under_a_mesh_is_named_as_one_device_of_it(
    monkeypatch, mesh_record
) -> None:
    """The figures are one device's budget, and the levers are CP's own.

    `_pool_card_and_used` reads device 0, so under a mesh the statistics need
    not even come from the allocator that failed. Saying so is the difference
    between a per-device number and a wrong total.
    """
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    monkeypatch.setattr(
        oom, "_pool_card_and_used", lambda: (int(0.9 * CARD), CARD, int(60 * GIB))
    )
    mesh_record(layout="2d", devices=4, grid=(2, 2))

    message = oom.diagnose(
        RuntimeError("RESOURCE_EXHAUSTED: Out of memory trying to allocate 56.78GiB.")
    )

    assert message is not None
    assert "one device of a 4-device context-parallel mesh" in message
    assert "2d layout, 2x2 grid" in message
    assert "cp_devices" in message and "cp_layout" in message


def test_a_serial_oom_says_nothing_about_a_mesh(monkeypatch, mesh_record) -> None:
    """The record outlives its context, so a stale one would describe this."""
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    monkeypatch.setattr(
        oom, "_pool_card_and_used", lambda: (int(0.9 * CARD), CARD, int(60 * GIB))
    )

    message = oom.diagnose(
        RuntimeError("RESOURCE_EXHAUSTED: Out of memory trying to allocate 56.78GiB.")
    )

    assert message is not None
    assert "context-parallel" not in message


@pytest.mark.parametrize(
    "message",
    [
        "RESOURCE_EXHAUSTED: Shared memory size limit exceeded",
        "RESOURCE_EXHAUSTED: Shared memory requested 110592 exceeds device resources",
        "RESOURCE_EXHAUSTED: out of resource: shared memory, "
        "Required: 110592, Hardware limit: 101376",
    ],
)
def test_shared_memory_kernel_resource_failures_are_not_allocator_oom(
    message, monkeypatch
):
    monkeypatch.setattr(oom, "_pool_card_and_used", lambda: (int(0.9 * CARD), CARD, 0))
    error = RuntimeError(message)
    assert not oom.is_out_of_memory(error)
    assert oom.diagnose(error) is None


@pytest.mark.parametrize(
    "call",
    ["cuModuleGetFunction", "cuModuleLoadDataEx", "before calling cuModule"],
)
@pytest.mark.parametrize("readable_stats", [False, True])
def test_cuda_module_oom_leaves_headroom_outside_the_pool(
    monkeypatch, call, readable_stats
):
    monkeypatch.setenv(oom.CURRENT_FRACTION_ENV, "0.9")
    monkeypatch.setattr(
        oom,
        "_pool_card_and_used",
        lambda: (int(0.9 * CARD), CARD, 0) if readable_stats else (None, None, None),
    )
    error = RuntimeError(f"cudaErrorMemoryAllocation: out of memory before {call}")
    assert oom.is_outside_pool_oom(error)
    message = oom.diagnose(error)
    assert "CUDA context/module memory outside" in message
    assert f"Lower {oom.CURRENT_FRACTION_ENV}" in message
    assert "0.85" in message and "restart the process" in message
    assert "=0.95" not in message


def test_module_names_without_an_oom_do_not_change_the_classification():
    assert not oom.is_outside_pool_oom(RuntimeError("cuModuleLoad invalid PTX"))
    assert not oom.is_outside_pool_oom(RuntimeError("Out of memory allocating array"))


@pytest.mark.parametrize(
    "launch",
    (
        "Kernel launch failed",
        "kernel-launch failed",
        "kernel_launch failed",
        "failed to launch kernel",
        "Failed to launch GPU kernel",
        "Failed to launch CUDA kernel",
        "cuLaunchKernel",
        "cudaLaunchKernel",
    ),
)
@pytest.mark.parametrize(
    "code", ("CUDA_ERROR_OUT_OF_MEMORY", "cudaErrorMemoryAllocation")
)
def test_cuda_launch_oom_is_outside_the_pool_even_when_pool_usage_is_low(
    monkeypatch, launch, code
):
    monkeypatch.setenv(oom.CURRENT_FRACTION_ENV, "0.9")
    monkeypatch.setattr(
        oom,
        "_pool_card_and_used",
        lambda: (int(21.2 * GIB), int(23.5 * GIB), 2 * GIB),
    )
    error = RuntimeError(f"{launch}: {code}")
    assert oom.is_outside_pool_oom(error)
    message = oom.diagnose(error)
    assert "outside the allocator's pool" in message
    assert "loading or launching" in message
    assert "0.8-0.85" in message and "restart the process" in message
    assert "=0.95" not in message and "Raise" not in message


def test_launch_words_without_cuda_oom_do_not_steal_allocator_failures(monkeypatch):
    assert not oom.is_outside_pool_oom(RuntimeError("kernel launch invalid argument"))
    assert not oom.is_outside_pool_oom(
        RuntimeError("Out of memory allocating kernel launch buffer")
    )
    mixed = ExceptionGroup(
        "failures",
        [
            RuntimeError("kernel launch: CUDA_ERROR_OUT_OF_MEMORY"),
            ValueError("invalid kernel signature"),
        ],
    )
    assert not oom.is_outside_pool_oom(mixed)
    monkeypatch.setattr(
        oom, "_pool_card_and_used", lambda: (21 * GIB, 24 * GIB, 2 * GIB)
    )
    assert "Raise it" in oom.diagnose(RuntimeError("CUDA_ERROR_OUT_OF_MEMORY"))


def test_pool_size_is_not_reported_as_the_required_memory(monkeypatch):
    monkeypatch.setattr(
        oom,
        "_pool_card_and_used",
        lambda: (int(21.2 * GIB), int(23.5 * GIB), int(12.9 * GIB)),
    )
    message = oom.diagnose(RuntimeError("CUDA_ERROR_OUT_OF_MEMORY"))
    assert "Pool 21.2 GiB, in use 12.9 GiB" in message
    assert "the failed allocation did not fit" in message
    assert "needed" not in message
