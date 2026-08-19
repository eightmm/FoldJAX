"""Context-local CP runtime and semantic entry-placement gates."""

from __future__ import annotations

import subprocess
import sys
import textwrap

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
            "PATH": "/usr/bin",
        },
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "CP_RUNTIME_OK" in completed.stdout
