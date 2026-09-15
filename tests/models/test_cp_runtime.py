"""Context-local CP runtime and semantic entry-placement gates."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import jax
import pytest

from foldjax import oom
from foldjax.models import _cp
from tests.models.cp_probe_env import inherited_environment

_PROBE = textwrap.dedent(
    r"""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    import jax
    import jax.numpy as jnp

    from foldjax.models._cp import (
        context_parallel,
        cp_identity,
        cp_layout,
        feature_spec,
        replicate_tree,
        resolve_cp_layout,
    )

    assert jax.device_count() == 4, jax.devices()
    assert cp_identity() == ("serial", 1, (1, 1), ())

    barrier = Barrier(2)

    def worker(layout):
        with context_parallel(4, layout=layout):
            barrier.wait(timeout=30)
            return cp_layout(), cp_identity()

    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(worker, "1d")
        two = pool.submit(worker, "2d")
        assert one.result(timeout=60)[0] == "1d"
        assert two.result(timeout=60)[0] == "2d"
    assert cp_layout() is None

    with context_parallel(4, layout="2d"):
        relp = jnp.zeros((1, 8, 8, 3), dtype=jnp.float32)
        atom = jnp.zeros((1, 8, 3), dtype=jnp.float32)
        odd = jnp.zeros((1, 7, 7, 3), dtype=jnp.float32)
        placed = replicate_tree({"relp": relp, "ref_pos": atom, "odd": odd})
        assert placed["relp"].sharding.shard_shape(relp.shape) == (1, 4, 4, 3)
        assert placed["ref_pos"].sharding.shard_shape(atom.shape) == atom.shape
        assert placed["odd"].sharding.shard_shape(odd.shape) == odd.shape
        assert feature_spec("relp", odd) is None

        atom_map = jnp.zeros((1, 8, 8), dtype=jnp.float32)
        coords = jnp.zeros((1, 2, 8, 3), dtype=jnp.float32)
        atom_placed = replicate_tree(
            {
                "ref_pos": atom,
                "coords": coords,
                "atom_to_token": atom_map,
            },
            shard_atom_features=True,
        )
        assert atom_placed["ref_pos"].sharding.shard_shape(atom.shape) == (1, 4, 3)
        assert atom_placed["coords"].sharding.shard_shape(coords.shape) == (1, 2, 4, 3)
        # Coupled dense atom/token maps have no generic independent sharding.
        assert (
            atom_placed["atom_to_token"].sharding.shard_shape(atom_map.shape)
            == atom_map.shape
        )
        assert feature_spec(
            "ref_pos", atom, shard_atom_features=True
        ) is not None

        strict = replicate_tree(
            {"relp": relp},
            shard_pair_features=False,
        )
        assert strict["relp"].sharding.shard_shape(relp.shape) == relp.shape

        try:
            with context_parallel(1):
                pass
        except RuntimeError as error:
            assert "does not nest" in str(error)
        else:
            raise AssertionError("a nested serial context was accepted")

    assert resolve_cp_layout("auto", 4) == "1d"
    assert resolve_cp_layout("2d", 4) == "2d"
    try:
        resolve_cp_layout("2d", 3)
    except ValueError as error:
        assert "perfect-square" in str(error)
    else:
        raise AssertionError("a non-square 2-D mesh was accepted")

    print("CP_RUNTIME_OK")
    """
)


def test_runtime_is_context_local_and_places_only_safe_pair_features() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
            **inherited_environment(),
        },
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "CP_RUNTIME_OK" in completed.stdout


# ---------------------------------------------------------------------------
# What a distributed entry does about XLA's collective rendezvous
# ---------------------------------------------------------------------------


def test_a_distributed_entry_bounds_the_rendezvous_while_flags_are_still_read(
    monkeypatch,
) -> None:
    """The bound has to be composed before the entry's own device query.

    That query is what initialises a backend when nothing else has, and
    `XLA_FLAGS` is parsed there -- so a bound written after it is a bound XLA
    never sees. Asking for more devices than exist stops the entry at exactly
    that call, with the composition already behind it.

    The initialised-state check is mocked because the pytest process has a
    backend, and the platform gate because this process is pinned to the CPU.
    Both are pinned on their own in `tests/test_oom.py`.
    """
    monkeypatch.setattr(_cp, "backends_are_initialized", lambda: False)
    monkeypatch.setattr(oom, "gpu_is_possible", lambda: True)
    monkeypatch.delenv(oom.RENDEZVOUS_ENV, raising=False)
    monkeypatch.setenv("XLA_FLAGS", "")

    with pytest.raises(ValueError, match="visible"):
        with _cp.context_parallel(jax.device_count() + 1):
            pass

    assert (
        f"--{oom.RENDEZVOUS_FLAG}={oom.CP_RENDEZVOUS_SECONDS}"
        in os.environ["XLA_FLAGS"]
    )


def test_a_one_device_context_bounds_nothing_and_records_no_mesh(monkeypatch) -> None:
    """A serial null context has no rendezvous and no topology to report."""
    monkeypatch.setattr(_cp, "backends_are_initialized", lambda: False)
    monkeypatch.setattr(oom, "gpu_is_possible", lambda: True)
    monkeypatch.delenv(oom.RENDEZVOUS_ENV, raising=False)
    monkeypatch.setenv("XLA_FLAGS", "")
    oom.clear_mesh_record()

    with _cp.context_parallel(1) as mesh:
        assert mesh is None

    assert os.environ["XLA_FLAGS"] == ""
    assert oom.recorded_mesh() is None


def test_a_context_entered_too_late_says_the_wait_is_unbounded(monkeypatch) -> None:
    """When the bound can no longer be set, say what that costs and how to fix.

    Silence would leave the two-hour hang exactly as it was found, with nothing
    in the log connecting it to a variable that had to be set earlier.
    """
    monkeypatch.delenv(oom.RENDEZVOUS_ENV, raising=False)
    monkeypatch.setenv("XLA_FLAGS", "")

    message = _cp.unbounded_rendezvous_warning(platform="gpu", devices=2)

    assert message is not None
    assert oom.RENDEZVOUS_FLAG in message
    assert "XLA_FLAGS" in message
    assert "cp_devices" in message
    # And not `FOLDJAX_CP_RENDEZVOUS_TIMEOUT`, which would be advice that does
    # nothing: it is read where FoldJAX composes the flag, and this warning is
    # exactly the case where that moment has passed.
    assert oom.RENDEZVOUS_ENV not in message

    # Nothing to say off GPU, where there is no NCCL rendezvous at all.
    assert _cp.unbounded_rendezvous_warning(platform="cpu", devices=2) is None

    # Nor when the bound is already in hand, or was declined.
    monkeypatch.setenv("XLA_FLAGS", f"--{oom.RENDEZVOUS_FLAG}=30")
    assert _cp.unbounded_rendezvous_warning(platform="gpu", devices=2) is None
    monkeypatch.setenv("XLA_FLAGS", "")
    monkeypatch.setenv(oom.RENDEZVOUS_ENV, "-1")
    assert _cp.unbounded_rendezvous_warning(platform="gpu", devices=2) is None


_RECORD_PROBE = textwrap.dedent(
    r"""
    import os
    import warnings

    import jax

    from foldjax import oom
    from foldjax.models._cp import context_parallel

    assert jax.device_count() == 4, jax.devices()
    inherited = os.environ["XLA_FLAGS"]
    expected = {"layout": "2d", "devices": 4, "grid": [2, 2]}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with context_parallel(4, layout="2d") as mesh:
            assert mesh is not None
            assert oom.recorded_mesh() == expected, oom.recorded_mesh()

    # The mesh is gone and the record is not: an OOM under it is diagnosed in
    # `foldjax.api`, after this context has unwound.
    assert oom.recorded_mesh() == expected, oom.recorded_mesh()
    assert [str(w.message) for w in caught] == [], "a CPU mesh has no rendezvous"
    assert os.environ["XLA_FLAGS"] == inherited, "a CPU-pinned process keeps its flags"

    oom.clear_mesh_record()
    assert oom.recorded_mesh() is None

    print("CP_RECORD_OK")
    """
)


def test_the_entered_topology_outlives_the_context_that_built_it() -> None:
    """Four real devices, because the record has to describe the mesh that ran."""
    completed = subprocess.run(
        [sys.executable, "-c", _RECORD_PROBE],
        capture_output=True,
        text=True,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
            **inherited_environment(),
        },
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "CP_RECORD_OK" in completed.stdout


_SPELLING_PROBE = textwrap.dedent(
    """
    import jax

    assert jax.devices(), "no devices"
    print("CP_FLAG_OK")
    """
)


def test_xla_accepts_the_rendezvous_flag_this_jaxlib_was_built_with(
    monkeypatch,
) -> None:
    """A misspelled flag does not lose the bound; it kills the run.

    XLA parses `XLA_FLAGS` with `ParseFlagsFromEnvAndDieIfUnknown`, and its
    flag list is global rather than per-platform -- the `--help` probe that
    found this spelling died with "Check failed: tsl::Flags::Parse ... Flag
    parsing failed" on a CPU backend. So the failure mode of a jaxlib that
    renames `xla_gpu_nccl_termination_timeout_seconds` is every context-parallel
    run refusing to start, and a child that initialises a backend with the
    composed flags is the cheap guard for it.
    """
    monkeypatch.delenv(oom.RENDEZVOUS_ENV, raising=False)
    monkeypatch.setenv("XLA_FLAGS", "")
    flags = oom.set_rendezvous_timeout()
    assert flags is not None

    completed = subprocess.run(
        [sys.executable, "-c", _SPELLING_PROBE],
        capture_output=True,
        text=True,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": flags,
            **inherited_environment(),
        },
        timeout=180,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "CP_FLAG_OK" in completed.stdout
