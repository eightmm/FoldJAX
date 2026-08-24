"""Context parallelism must not change what the model computes.

The CP path re-routes three things -- triangle attention through ``shard_map``,
triangle multiplication through the unchunked einsum, and the role-pair
projection through a per-role sum instead of a row scan -- so the property to
hold is numerical parity against the unsharded program, on a mesh whose size
does not divide the token count. A mesh needs more than one device, and the
device count is fixed at process start, so the parity checks run in a
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


def test_mesh_larger_than_the_device_pool_is_rejected() -> None:
    with pytest.raises(ValueError, match="are visible"):
        with context_parallel(4096):
            pass


def test_shard_count_must_match_the_active_mesh() -> None:
    """The static ``cp_shards`` and the ambient mesh are one decision.

    The guard runs before any feature validation, so garbage inputs are fine.
    """
    from foldjax.models.opendde.models.model import opendde_infer_static

    with pytest.raises(RuntimeError, match="cp_shards=2"):
        opendde_infer_static({}, None, None, key=None, n_sample=1, cp_shards=2)


_PARITY_PROBE = textwrap.dedent(
    """
    import os
    os.environ["PROTENIX_TRIANGLE_BACKEND"] = "xla"
    os.environ["PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND"] = "xla"

    import jax
    import jax.numpy as jnp
    import numpy as np
    import pytest

    from foldjax.models._cp import context_parallel
    from foldjax.models.protenix.models.primitives.attention import (
        AttentionPairBiasParams,
        AttentionParams,
    )
    from foldjax.models.protenix.models.primitives.primitives import (
        LayerNormParams,
        LinearParams,
        TransitionParams,
    )
    from foldjax.models.protenix.models.triangle.triangle import (
        TriangleAttentionParams,
        TriangleMultiplicationParams,
        triangle_attention,
    )
    from foldjax.models.protenix.models.trunk_blocks.pairformer import (
        PairformerBlockParams,
        PairformerStackParams,
        pairformer_stack,
    )
    from foldjax.models.opendde.models.structural_tokens import (
        StructuralTokenExpanderParams,
        structural_token_expand,
    )

    assert jax.device_count() == 4, jax.devices()
    # `jax.jit` caches its jaxpr on the *callable*, and the mesh lives in a
    # module global the trace reads, so a second `jit` of the same function
    # object replays the unsharded program -- the comparison below was the
    # unsharded program against itself until `clear_caches` was added here.
    # The giveaway was an exactly-zero difference, which a real `shard_map`
    # cannot produce because it reorders float32 accumulation. Removing these
    # calls makes every parity assertion below vacuous.


    C, HEADS, N = 8, 2, 13  # 13 rows over 4 shards: padding path exercised
    rng = np.random.default_rng(0)

    def arr(*shape):
        return jnp.asarray(rng.normal(size=shape, scale=0.5), dtype=jnp.float32)

    def lin(o, i):
        return LinearParams(weight=arr(o, i), bias=arr(o))

    def ln(c):
        return LayerNormParams(weight=arr(c) * 0.1 + 1.0, bias=arr(c) * 0.1)

    def tri_mult():
        return TriangleMultiplicationParams(
            layer_norm_in=ln(C), layer_norm_out=ln(C),
            linear_a_p=lin(C, C), linear_a_g=lin(C, C),
            linear_b_p=lin(C, C), linear_b_g=lin(C, C),
            linear_z=lin(C, C), linear_g=lin(C, C),
        )

    def attn():
        return AttentionParams(lin(C, C), lin(C, C), lin(C, C), lin(C, C), lin(C, C))

    def tri_att():
        return TriangleAttentionParams(
            layer_norm=ln(C),
            linear=LinearParams(weight=arr(HEADS, C), bias=None),
            attention=attn(),
        )

    def transition():
        return TransitionParams(
            layer_norm=ln(C),
            linear_a=lin(2 * C, C), linear_b=lin(2 * C, C),
            linear_out=lin(C, 2 * C),
        )

    def block():
        return PairformerBlockParams(
            tri_mul_out=tri_mult(), tri_mul_in=tri_mult(),
            tri_att_start=tri_att(), tri_att_end=tri_att(),
            pair_transition=transition(),
            attention_pair_bias=AttentionPairBiasParams(
                layernorm_a=ln(C), layernorm_kv=None, attention=attn(),
                layernorm_z=ln(C),
                linear_z=LinearParams(weight=arr(HEADS, C), bias=None),
                has_s=False, cross_attention_mode=False,
            ),
            single_transition=transition(),
        )

    params = PairformerStackParams(blocks=(block(), block()))
    s, z = arr(N, C), arr(N, N, C)
    mask_np = rng.random(N) > 0.15
    pair_mask = jnp.asarray(mask_np[:, None] & mask_np[None, :])

    def run(s_in, z_in):
        return pairformer_stack(
            s_in, z_in, pair_mask, params, use_scan=True,
            single_attention_backend="xla", triangle_attention_backend="xla",
        )

    ref_s, ref_z = map(jax.device_get, jax.jit(run)(s, z))
    jax.clear_caches()
    with context_parallel(4):
        cp_s, cp_z = map(jax.device_get, jax.jit(run)(s, z))
    np.testing.assert_allclose(ref_s, cp_s, atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(ref_z, cp_z, atol=3e-5, rtol=3e-5)

    NR, NS, ROLES = 7, 13, 7
    features = {
        "parent_residue_idx": jnp.asarray(rng.integers(0, NR, NS).astype(np.int32)),
        "subtoken_role_id": jnp.asarray(rng.integers(0, ROLES, NS).astype(np.int32)),
        "asym_id": jnp.asarray(rng.integers(0, 2, NR).astype(np.int32)),
        "residue_index": jnp.asarray(np.arange(NR, dtype=np.int32)),
    }
    exp_params = StructuralTokenExpanderParams(
        single_split_layer_norm=ln(C),
        single_split_linear_in=lin(C, C), single_split_linear_out=lin(C, C),
        single_input_role_embedding=arr(ROLES, C), single_role_embedding=arr(ROLES, C),
        pair_block_proj=arr(ROLES, ROLES, C, C),
        same_parent_embedding=arr(2, C), same_residue_twin_embedding=arr(2, C),
        prev_bb_chain_embedding=arr(2, C), next_bb_chain_embedding=arr(2, C),
        role_pair_type_embedding=arr(8, C),
        attn_bias_same_parent=arr(), attn_bias_same_residue_twin=arr(),
        attn_bias_prev_bb_chain=arr(), attn_bias_next_bb_chain=arr(),
        attn_bias_role_pair_type=arr(8),
    )
    s_inputs_res, s_res, z_res = arr(NR, C), arr(NR, C), arr(NR, NR, C)

    def expand():
        return structural_token_expand(
            features, s_inputs_res, s_res, z_res, exp_params
        )

    ref = jax.device_get(jax.jit(expand)())
    jax.clear_caches()
    with context_parallel(4):
        got = jax.device_get(jax.jit(expand)())
    for r, g in zip(ref[:3], got[:3]):
        np.testing.assert_allclose(r, g, atol=3e-5, rtol=3e-5)

    # The fused triangle-attention kernel is accepted under a mesh -- it runs
    # per-shard inside `shard_map`, where each device holds whole rows. Only
    # backends with no per-shard story are refused.
    jax.clear_caches()
    with context_parallel(4):
        with pytest.raises(ValueError, match="supports the XLA and"):
            triangle_attention(
                z, None, tri_att(), num_heads=HEADS, attention_backend="tokamax"
            )

    print("CP_PARITY_OK")
    """
)


_GRID_PROBE = textwrap.dedent(
    """
    import os
    os.environ["PROTENIX_TRIANGLE_BACKEND"] = "xla"
    os.environ["PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND"] = "xla"

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import (
        col_skew_perm, context_parallel, cp_grid, grid_axes, permute,
        ring_perm, row_skew_perm, transpose_perm,
    )
    from jax.sharding import NamedSharding, PartitionSpec

    assert jax.device_count() == 4, jax.devices()
    # `jax.jit` caches its jaxpr on the *callable*, and the mesh lives in a
    # module global the trace reads, so a second `jit` of the same function
    # object replays the unsharded program -- the comparison below was the
    # unsharded program against itself until `clear_caches` was added here.
    # The giveaway was an exactly-zero difference, which a real `shard_map`
    # cannot produce because it reorders float32 accumulation. Removing these
    # calls makes every parity assertion below vacuous.

    SIDE = 2
    ids = np.asarray([[10.0 * i + j for j in range(SIDE)] for i in range(SIDE)])

    def move(build):
        jax.clear_caches()
        with context_parallel(4, layout="2d") as mesh:
            assert cp_grid() == (2, 2)
            spec = PartitionSpec(*grid_axes())
            x = jax.device_put(jnp.asarray(ids), NamedSharding(mesh, spec))
            body = lambda tile: permute(tile, build(SIDE))
            return np.asarray(jax.device_get(jax.jit(jax.shard_map(
                body, mesh=mesh, in_specs=(spec,), out_specs=spec))(x)))

    # A grid transpose puts (j, i)'s tile on (i, j).
    np.testing.assert_array_equal(move(transpose_perm), ids.T)
    # Cannon's skews leave device (i, j) holding A[i, i+j] and B[i+j, j].
    np.testing.assert_array_equal(
        move(row_skew_perm), np.stack([np.roll(ids[i], -i) for i in range(SIDE)])
    )
    np.testing.assert_array_equal(
        move(col_skew_perm),
        np.stack([np.roll(ids[:, j], -j) for j in range(SIDE)], axis=1),
    )
    # One ring hop along each axis.
    np.testing.assert_array_equal(
        move(lambda s: ring_perm(s, axis=grid_axes()[1], delta=1)),
        np.roll(ids, -1, axis=1),
    )
    np.testing.assert_array_equal(
        move(lambda s: ring_perm(s, axis=grid_axes()[0], delta=1)),
        np.roll(ids, -1, axis=0),
    )
    print("GRID_PRIMITIVES_OK")
    """
)


_CANNON_PROBE = textwrap.dedent(
    """
    import os
    os.environ["PROTENIX_TRIANGLE_BACKEND"] = "xla"
    os.environ["PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND"] = "xla"

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel, cp_layout
    from foldjax.models.protenix.models.primitives.primitives import (
        LayerNormParams, LinearParams,
    )
    from foldjax.models.protenix.models.triangle.triangle import (
        TriangleMultiplicationParams, triangle_multiplication,
    )

    # `clear_caches` above is what forces the sharded retrace, but nothing
    # about it is guaranteed by JAX's contract. This records what the mesh
    # looked like at trace time and asserts the sharded trace happened, so a
    # future change in cache behaviour fails loudly instead of quietly making
    # every comparison below vacuous again.
    traced = []

    # `jax.jit` caches its jaxpr on the *callable*, and the mesh lives in a
    # module global the trace reads, so a second `jit` of the same function
    # object replays the unsharded program -- the comparison below was the
    # unsharded program against itself until `clear_caches` was added here.
    # The giveaway was an exactly-zero difference, which a real `shard_map`
    # cannot produce because it reorders float32 accumulation. Removing these
    # calls makes every parity assertion below vacuous.

    total = int(os.environ.get("FOLDJAX_CP_PROBE_DEVICES", "4"))
    assert jax.device_count() == total, jax.devices()
    C, N = 8, 12  # 12 divides a 2x2 and a 3x3 grid on both pair axes
    rng = np.random.default_rng(0)
    arr = lambda *s: jnp.asarray(rng.normal(size=s, scale=0.5), dtype=jnp.float32)
    lin = lambda o, i: LinearParams(weight=arr(o, i), bias=arr(o))
    ln = lambda c: LayerNormParams(weight=arr(c) * 0.1 + 1.0, bias=arr(c) * 0.1)

    params = TriangleMultiplicationParams(
        layer_norm_in=ln(C), layer_norm_out=ln(C),
        linear_a_p=lin(C, C), linear_a_g=lin(C, C),
        linear_b_p=lin(C, C), linear_b_g=lin(C, C),
        linear_z=lin(C, C), linear_g=lin(C, C),
    )
    z = arr(N, N, C)
    keep = rng.random(N) > 0.15
    mask = jnp.asarray(keep[:, None] & keep[None, :])

    def _witness(z_in, d):
        traced.append(cp_layout())
        return triangle_multiplication(z_in, mask, params, d)

    for direction in ("outgoing", "incoming"):
        run = lambda z_in, d=direction: _witness(z_in, d)
        ref = jax.device_get(jax.jit(run)(z))
        jax.clear_caches()
        with context_parallel(total, layout="2d"):
            got = jax.device_get(jax.jit(run)(z))
        np.testing.assert_allclose(ref, got, atol=3e-5, rtol=3e-5)
        assert traced[-2:] == [None, "2d"], traced
    print("CANNON_PARITY_OK")
    """
)


_SAMPLED_MSA_PLACEMENT_PROBE = textwrap.dedent(
    """
    import jax
    import numpy as np

    from foldjax.models._cp import context_parallel, replicate_tree
    from foldjax.models.opendde.models.msa_sampling import (
        drop_sampled_msa_source_features,
    )

    assert jax.device_count() == 4, jax.devices()
    M, N = 32, 16
    raw_names = ("msa", "has_deletion", "deletion_value", "msa_mask")
    features = {
        "msa": np.zeros((M, N), dtype=np.int32),
        "has_deletion": np.zeros((M, N), dtype=np.float32),
        "deletion_value": np.zeros((M, N), dtype=np.float32),
        "msa_mask": np.ones((M, N), dtype=np.float32),
        "profile": np.zeros((N, 32), dtype=np.float32),
        "deletion_mean": np.zeros((N,), dtype=np.float32),
        "constraint_feature": {"contact": np.ones((N, N), dtype=np.float32)},
        "template_aatype": np.ones((1, N), dtype=np.int32),
        "custom_metadata": "kept",
    }
    cycle = {
        "msa": np.zeros((4, N), dtype=np.int32),
        "has_deletion": np.zeros((4, N), dtype=np.float32),
        "deletion_value": np.zeros((4, N), dtype=np.float32),
        "msa_mask": np.ones((4, N), dtype=np.float32),
    }
    pruned = drop_sampled_msa_source_features(features, (cycle, cycle))
    assert set(features) - set(pruned) == set(raw_names)
    assert pruned["profile"] is features["profile"]
    assert pruned["deletion_mean"] is features["deletion_mean"]
    assert pruned["constraint_feature"] is features["constraint_feature"]
    assert pruned["template_aatype"] is features["template_aatype"]
    assert pruned["custom_metadata"] == "kept"

    def physical_bytes(values):
        return sum(
            shard.data.size * shard.data.dtype.itemsize
            for value in values
            for shard in value.addressable_shards
        )

    for layout in ("1d", "2d"):
        with context_parallel(4, layout=layout):
            raw_placed = replicate_tree(features)
            pruned_placed = replicate_tree(pruned)
        for name in raw_names:
            assert "P()" in str(raw_placed[name].sharding)
            assert name not in pruned_placed
        removed_bytes = physical_bytes([raw_placed[name] for name in raw_names])
        assert removed_bytes > 0
        assert set(pruned_placed) == set(pruned)
        print(layout, removed_bytes)
    print("CP_SAMPLED_MSA_PRUNE_OK")
    """
)


def _run_probe(source: str, *, devices: int = 4) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
            "FOLDJAX_CP_PROBE_DEVICES": str(devices),
            **inherited_environment(),
        },
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def test_grid_primitives_move_tiles_where_cannon_needs_them() -> None:
    """The 2-D schedules are static permutations; check each one moves right."""
    assert "GRID_PRIMITIVES_OK" in _run_probe(_GRID_PROBE)


def test_cannon_triangle_multiplication_matches_the_unsharded_form() -> None:
    """Both directions, on a mesh that splits rows *and* columns."""
    assert "CANNON_PARITY_OK" in _run_probe(_CANNON_PROBE)


def test_cannon_holds_on_a_three_by_three_grid() -> None:
    """A 2x2 grid cannot falsify a skew or hop *sign*.

    Every shift in Cannon's schedule is +-1 or +-coord, and modulo 2 a
    forward hop and a backward hop are the same permutation -- so a sign
    error in `row_skew_perm`, `col_skew_perm` or the ring direction passes a
    four-device test and fails on any real mesh. Nine devices is the smallest
    grid that separates them.
    """
    assert "CANNON_PARITY_OK" in _run_probe(_CANNON_PROBE, devices=9)


def test_sampled_raw_msa_never_reaches_cp_placement() -> None:
    assert "CP_SAMPLED_MSA_PRUNE_OK" in _run_probe(_SAMPLED_MSA_PLACEMENT_PROBE)


_WHOLE_MODEL_PROBE = textwrap.dedent(
    """
    import os
    os.environ["PROTENIX_TRIANGLE_BACKEND"] = "xla"
    os.environ["PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND"] = "xla"

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel, cp_layout
    from foldjax.models.protenix.models.primitives.attention import (
        AttentionPairBiasParams, AttentionParams,
    )
    from foldjax.models.protenix.models.primitives.primitives import (
        LayerNormParams, LinearParams, TransitionParams,
    )
    from foldjax.models.protenix.models.triangle.triangle import (
        TriangleAttentionParams, TriangleMultiplicationParams,
    )
    from foldjax.models.protenix.models.trunk_blocks.pairformer import (
        PairformerBlockParams, PairformerStackParams, pairformer_stack,
    )

    # Triangle *attention* stays row-sharded under a 2-D mesh, so its
    # `shard_map` specs have to name the mesh's row axis rather than the 1-D
    # literal. They did not, and no module-level probe caught it because the
    # attention was only ever run under a 1-D mesh; the whole-stack run below
    # is what fails when the two layouts disagree about axis names.
    total = int(os.environ.get("FOLDJAX_CP_PROBE_DEVICES", "4"))
    assert jax.device_count() == total, jax.devices()
    C, HEADS, N = 8, 2, 12
    rng = np.random.default_rng(0)
    arr = lambda *s: jnp.asarray(rng.normal(size=s, scale=0.5), dtype=jnp.float32)
    lin = lambda o, i: LinearParams(weight=arr(o, i), bias=arr(o))
    ln = lambda c: LayerNormParams(weight=arr(c) * 0.1 + 1.0, bias=arr(c) * 0.1)
    mult = lambda: TriangleMultiplicationParams(
        ln(C), ln(C), lin(C, C), lin(C, C), lin(C, C), lin(C, C), lin(C, C), lin(C, C)
    )
    attn = lambda: AttentionParams(
        lin(C, C), lin(C, C), lin(C, C), lin(C, C), lin(C, C)
    )
    tri_att = lambda: TriangleAttentionParams(
        ln(C), LinearParams(weight=arr(HEADS, C), bias=None), attn()
    )
    trans = lambda: TransitionParams(
        ln(C), lin(2 * C, C), lin(2 * C, C), lin(C, 2 * C)
    )
    params = PairformerStackParams(blocks=(PairformerBlockParams(
        tri_mul_out=mult(), tri_mul_in=mult(),
        tri_att_start=tri_att(), tri_att_end=tri_att(),
        pair_transition=trans(),
        attention_pair_bias=AttentionPairBiasParams(
            layernorm_a=ln(C), layernorm_kv=None, attention=attn(),
            layernorm_z=ln(C),
            linear_z=LinearParams(weight=arr(HEADS, C), bias=None),
            has_s=False, cross_attention_mode=False,
        ),
        single_transition=trans(),
    ),) * 2)
    s_in, z = arr(N, C), arr(N, N, C)
    keep = rng.random(N) > 0.15
    pair_mask = jnp.asarray(keep[:, None] & keep[None, :])
    traced = []

    def run(s_arg, z_arg):
        traced.append(cp_layout())
        return pairformer_stack(
            s_arg, z_arg, pair_mask, params, use_scan=True,
            single_attention_backend="xla", triangle_attention_backend="xla",
        )

    ref_s, ref_z = map(jax.device_get, jax.jit(run)(s_in, z))
    jax.clear_caches()
    with context_parallel(total, layout="2d"):
        got_s, got_z = map(jax.device_get, jax.jit(run)(s_in, z))
    assert traced == [None, "2d"], traced
    np.testing.assert_allclose(ref_s, got_s, atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(ref_z, got_z, atol=3e-5, rtol=3e-5)
    print("WHOLE_STACK_2D_OK")
    """
)


def test_the_whole_pair_stack_runs_under_the_square_grid() -> None:
    """Attention and multiplication together, which module probes miss.

    The triangle-attention `shard_map` names a mesh axis; under the 2-D
    layout that axis has a different name, and a spec that hardcodes the 1-D
    one raises only when attention actually runs on the grid.
    """
    assert "WHOLE_STACK_2D_OK" in _run_probe(_WHOLE_MODEL_PROBE)


def test_the_whole_pair_stack_runs_on_a_three_by_three_grid() -> None:
    assert "WHOLE_STACK_2D_OK" in _run_probe(_WHOLE_MODEL_PROBE, devices=9)


def test_square_layout_requires_a_square_device_count() -> None:
    with pytest.raises(ValueError, match="square device count"):
        with context_parallel(3, layout="2d"):
            pass


def test_context_parallel_matches_the_unsharded_program() -> None:
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
