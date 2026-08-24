"""Context parallelism must not change what the ESMFold2 trunk computes.

ESMFold2's CP path is constraint-only -- its pair trunk has triangle
multiplicative updates but no triangle attention, so no ``shard_map`` and no
kernel gating. The property to hold is numerical parity of the pair trunk
against the unsharded program on a mesh whose size does not divide the token
count. The model is stochastic *end to end* (random initial pair state,
per-loop LM dropout), so parity is checked at module level with fixed inputs,
where the computation is deterministic. A mesh needs more than one device and
the device count is fixed at process start, so the parity check runs in a
subprocess with four forced CPU devices.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from foldjax.models._cp import context_parallel, cp_shards
from tests.models.cp_probe_env import inherited_environment


def test_single_shard_context_is_a_no_op() -> None:
    with context_parallel(1) as mesh:
        assert mesh is None
        assert cp_shards() == 1


def test_shard_count_must_match_the_active_mesh() -> None:
    """The static ``cp_shards`` and the ambient mesh are one decision.

    The guard runs before the model touches anything, so garbage inputs are
    fine.
    """
    from foldjax.models.esmfold2.inference import _run

    with pytest.raises(RuntimeError, match="cp_shards=2"):
        _run(None, {}, {}, None, None, 1, False, 2)


def test_compact_token_bond_static_choice_crosses_the_mesh_context(
    monkeypatch,
) -> None:
    """The host proof and ambient CP route must reach one graph decision."""
    from foldjax.models.esmfold2 import inference

    seen = []

    def fake_predict(*args, **kwargs):
        del args
        seen.append(kwargs["compact_token_bond_encoding"])
        return {}

    monkeypatch.setattr(inference.structure_model, "predict", fake_predict)
    with context_parallel(1):
        inference._run(
            None, {}, {}, None, None, 1, False, compact_token_bond_encoding=True
        )
        # A direct low-level call that does not supply the proof stays generic.
        inference._run(None, {}, {}, None, None, 1, False)

    assert seen == [True, False]


_DISTOGRAM_ROUTE_PROBE = textwrap.dedent(
    """
    import jax

    from foldjax.models._cp import context_parallel
    from foldjax.models.esmfold2 import inference

    assert jax.device_count() == 4, jax.devices()
    seen = []
    original = inference.structure_model.predict

    def fake_predict(*args, **kwargs):
        del args
        seen.append(kwargs["return_distogram_logits"])
        return {}

    inference.structure_model.predict = fake_predict
    try:
        with context_parallel(4):
            inference._run(
                None,
                {},
                {},
                None,
                None,
                1,
                False,
                4,
                return_distogram_logits=False,
            )
    finally:
        inference.structure_model.predict = original

    assert seen == [False], seen
    print("CP_DISTOGRAM_ROUTE_OK")
    """
)


def test_distogram_choice_crosses_a_forced_four_device_cpu_mesh() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _DISTOGRAM_ROUTE_PROBE],
        capture_output=True,
        text=True,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
            **inherited_environment(),
        },
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "CP_DISTOGRAM_ROUTE_OK" in completed.stdout


_PARITY_PROBE = textwrap.dedent(
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel
    from foldjax.models.esmfold2.models.trunk import folding_trunk

    assert jax.device_count() == 4, jax.devices()
    # `jax.jit` caches its jaxpr on the *callable*, and the mesh lives in a
    # module global the trace reads, so a second `jit` of the same function
    # object replays the unsharded program -- the comparison below was the
    # unsharded program against itself until `clear_caches` was added here.
    # The giveaway was an exactly-zero difference, which a real `shard_map`
    # cannot produce because it reorders float32 accumulation. Removing these
    # calls makes every parity assertion below vacuous.


    C, N, LAYERS = 8, 13, 2  # 13 rows over 4 shards: uneven-shard path
    rng = np.random.default_rng(0)

    def arr(*shape):
        return jnp.asarray(rng.normal(size=shape, scale=0.5), dtype=jnp.float32)

    params = {}
    for index in range(LAYERS):
        block = f"blocks.{index}"
        for tri in ("tri_mul_out", "tri_mul_in"):
            engine = f"{block}.{tri}._engine"
            params[f"{engine}.norm_start.weight"] = arr(C) * 0.1 + 1.0
            params[f"{engine}.norm_start.bias"] = arr(C) * 0.1
            params[f"{engine}.proj_bundle.weight"] = arr(4 * C, C)
            params[f"{engine}.proj_bundle.bias"] = arr(4 * C)
            params[f"{engine}.norm_mix.weight"] = arr(C) * 0.1 + 1.0
            params[f"{engine}.norm_mix.bias"] = arr(C) * 0.1
            params[f"{engine}.proj_emit.weight"] = arr(C, C)
            params[f"{engine}.proj_emit.bias"] = arr(C)
            params[f"{engine}.proj_gate.weight"] = arr(C, C)
            params[f"{engine}.proj_gate.bias"] = arr(C)
        transition = f"{block}.pair_transition"
        params[f"{transition}.norm.weight"] = arr(C) * 0.1 + 1.0
        params[f"{transition}.norm.bias"] = arr(C) * 0.1
        params[f"{transition}.ffn.w12.weight"] = arr(4 * C, C)
        params[f"{transition}.ffn.w3.weight"] = arr(C, 2 * C)

    pair = arr(1, N, N, C)
    mask_np = rng.random(N) > 0.15
    mask = jnp.asarray((mask_np[:, None] & mask_np[None, :])[None].astype(np.float32))

    def run(pair_in):
        return folding_trunk(pair_in, params, n_layers=LAYERS, mask=mask)

    ref = jax.device_get(jax.jit(run)(pair))
    jax.clear_caches()
    with context_parallel(4):
        got = jax.device_get(jax.jit(run)(pair))
    np.testing.assert_allclose(ref, got, atol=3e-5, rtol=3e-5)
    print("CP_PARITY_OK", float(np.abs(ref - got).max()))
    """
)


def test_context_parallel_matches_the_unsharded_trunk() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _PARITY_PROBE],
        capture_output=True,
        text=True,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
            **inherited_environment(),
        },
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "CP_PARITY_OK" in completed.stdout
