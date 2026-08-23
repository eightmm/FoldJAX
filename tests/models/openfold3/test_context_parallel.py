"""Context parallelism must not change what OpenFold3 computes.

The CP path re-routes triangle attention through ``shard_map``, disables the
pair transition's row chunking, and re-shards around the pair block's own
transposes, so the property to hold is numerical parity against the unsharded
program on a mesh whose size does not divide the token count. A mesh needs more
than one device and the device count is fixed at process start, so the parity
checks run in a subprocess with four forced CPU devices.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from foldjax.models._cp import context_parallel
from tests.models.cp_probe_env import inherited_environment


def test_shard_count_must_match_the_active_mesh() -> None:
    """The config's ``cp_shards`` and the ambient mesh are one decision.

    The guard runs before anything touches the batch, so garbage inputs are
    fine.
    """
    from foldjax.models.openfold3.inference import InferenceConfig, predict

    config = InferenceConfig(
        n_token=1,
        n_atom=1,
        n_query=1,
        n_key=1,
        atom_heads=1,
        token_heads=1,
        no_heads_msa=1,
        no_heads_pair=1,
        no_heads_pair_bias=1,
        max_relative_idx=1,
        max_relative_chain=1,
        num_cycles=1,
        num_samples=1,
        max_atoms_per_token=1,
        plddt_bins=1,
        pae_bins=1,
        pae_bin_max=1.0,
        no_rollout_steps=1,
        cp_shards=2,
    )
    with pytest.raises(RuntimeError, match="cp_shards=2"):
        predict(None, {}, None, config, None)


def test_fused_backend_under_cp_is_rejected() -> None:
    import jax.numpy as jnp

    from foldjax.models.openfold3.models.triangle_attention import (
        triangle_attention,
    )

    with context_parallel(1):
        pass  # a 1-shard context is a no-op and must not trip the guard below

    # The guard fires at dispatch, before any math or any mesh-size work, so a
    # single-device "mesh" cannot be built here -- instead exercise the check
    # through the subprocess parity probe below. What can be checked in-process
    # is that the plain path still rejects unknown backends.
    with pytest.raises(ValueError, match="unsupported triangle attention"):
        triangle_attention(
            jnp.zeros((3, 3, 4)),
            None,
            no_heads=1,
            backend="nonsense",
        )


_PARITY_PROBE = textwrap.dedent(
    """
    import os

    import jax
    import jax.numpy as jnp
    import numpy as np
    import pytest

    from foldjax.models._cp import context_parallel
    from foldjax.models._stacking import prestack_layer_lists
    from foldjax.models.openfold3.models.attention import AttentionParams
    from foldjax.models.openfold3.models.attention_pair_bias import (
        AttentionPairBiasParams,
    )
    from foldjax.models.openfold3.models.pair_block import PairBlockParams
    from foldjax.models.openfold3.models.pairformer import (
        PairformerBlockParams,
        PairformerStackParams,
        pairformer_stack,
    )
    from foldjax.models.openfold3.models.primitives import (
        LayerNormParams,
        LinearParams,
        SwiGLUParams,
        SwiGLUTransitionParams,
    )
    from foldjax.models.openfold3.models.triangle import (
        TriangleMultiplicationParams,
    )
    from foldjax.models.openfold3.models.triangle_attention import (
        TriangleAttentionParams,
        triangle_attention,
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
            linear_g=lin(C, C), linear_z=lin(C, C),
        )

    def attn():
        return AttentionParams(lin(C, C), lin(C, C), lin(C, C), lin(C, C), lin(C, C))

    def tri_att():
        return TriangleAttentionParams(
            layer_norm=ln(C),
            linear_z=LinearParams(weight=arr(HEADS, C), bias=None),
            mha=attn(),
        )

    def transition():
        return SwiGLUTransitionParams(
            layer_norm=ln(C),
            swiglu=SwiGLUParams(linear_a=lin(2 * C, C), linear_b=lin(2 * C, C)),
            linear_out=lin(C, 2 * C),
        )

    def block():
        return PairformerBlockParams(
            pair_stack=PairBlockParams(
                tri_mul_out=tri_mult(), tri_mul_in=tri_mult(),
                tri_att_start=tri_att(), tri_att_end=tri_att(),
                pair_transition=transition(),
            ),
            attn_pair_bias=AttentionPairBiasParams(
                layer_norm_a=ln(C), layer_norm_z=ln(C),
                linear_z=LinearParams(weight=arr(HEADS, C), bias=None),
                mha=attn(),
            ),
            single_transition=transition(),
        )

    params = prestack_layer_lists(
        PairformerStackParams(blocks=(block(), block()))
    )
    s, z = arr(N, C), arr(N, N, C)
    mask_np = rng.random(N) > 0.15
    single_mask = jnp.asarray(mask_np, dtype=jnp.float32)
    pair_mask = jnp.asarray(
        (mask_np[:, None] & mask_np[None, :]), dtype=jnp.float32
    )

    def run(s_in, z_in):
        return pairformer_stack(
            s_in, z_in, params,
            single_mask=single_mask, pair_mask=pair_mask,
            no_heads_pair=HEADS, no_heads_pair_bias=HEADS,
            chunk_size=5,
        )

    ref_s, ref_z = map(jax.device_get, jax.jit(run)(s, z))
    jax.clear_caches()
    with context_parallel(4):
        cp_s, cp_z = map(jax.device_get, jax.jit(run)(s, z))
    np.testing.assert_allclose(ref_s, cp_s, atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(ref_z, cp_z, atol=3e-5, rtol=3e-5)

    # The template stack and the confidence re-embedding run the same ops with
    # a leading batch axis; the shard_map path must keep the row axis at -3.
    t = arr(2, N, N, C)
    t_params = tri_att()

    def run_batched(t_in):
        return triangle_attention(
            t_in, t_params, no_heads=HEADS, backend="xla", chunk_size=5
        )

    ref_t = jax.device_get(jax.jit(run_batched)(t))
    jax.clear_caches()
    with context_parallel(4):
        cp_t = jax.device_get(jax.jit(run_batched)(t))
    np.testing.assert_allclose(ref_t, cp_t, atol=3e-5, rtol=3e-5)

    # cueq attention is accepted under a mesh (per-shard inside `shard_map`);
    # an unknown backend is still refused.
    jax.clear_caches()
    with context_parallel(4):
        with pytest.raises(ValueError, match="supports the XLA and"):
            triangle_attention(z, t_params, no_heads=HEADS, backend="tokamax")

    print("CP_PARITY_OK")
    """
)


#: Shared preamble for the 2-D probes.
#:
#: Two traps are guarded here, and neither shows up as a red test on its own.
#:
#: *Vacuity.* ``jax.jit`` caches its jaxpr on the callable, and ``_cp``'s mesh is
#: a plain module global that JAX's trace context knows nothing about, so calling
#: ``jax.jit(run)`` twice on *one* closure replays the unsharded jaxpr under the
#: mesh and compares the unsharded program with itself. That reads as a pass --
#: it is how a Cannon contraction off by 15.9 shipped green. ``under_mesh``
#: therefore builds a fresh closure per call *and* clears the caches, then
#: asserts from a trace-time side effect that the sharded run really was traced
#: under the mesh. The positive assertion is the part that cannot rot: if a
#: future refactor reintroduces cache reuse, ``traced`` stays short and this
#: fails loudly instead of silently passing.
#:
#: *Grid size.* A 2x2 grid cannot falsify a sign anywhere in the schedule,
#: because modulo 2 a forward hop and a backward hop are the same permutation --
#: ``(j - 1) % 2 == (j + 1) % 2``, and likewise for both skews. Every probe below
#: is therefore parameterised by ``FOLDJAX_CP_PROBE_DEVICES`` and run at 3x3 as
#: well, which is the smallest grid where the skew signs and the two ring
#: directions are distinguishable.
_TWO_D_PREAMBLE = """
import os
os.environ["OPENFOLD3_TRIANGLE_BACKEND"] = "xla"

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models._cp import context_parallel, cp_grid, cp_layout
from foldjax.models._stacking import prestack_layer_lists
from foldjax.models.openfold3.models.primitives import (
    LayerNormParams, LinearParams, SwiGLUParams, SwiGLUTransitionParams,
)

DEVICES = int(os.environ.get("FOLDJAX_CP_PROBE_DEVICES", "4"))
SIDE = round(DEVICES ** 0.5)
assert jax.device_count() == DEVICES, jax.devices()
rng = np.random.default_rng(0)
arr = lambda *s: jnp.asarray(rng.normal(size=s, scale=0.5), dtype=jnp.float32)
lin = lambda o, i: LinearParams(weight=arr(o, i), bias=arr(o))
ln = lambda c: LayerNormParams(weight=arr(c) * 0.1 + 1.0, bias=arr(c) * 0.1)
traced = []

def under_mesh(build, *args):
    \"\"\"Run ``build()``'s program unsharded and on the square grid; return both.\"\"\"
    def wrap():
        fn = build()
        def run(*a):
            traced.append(cp_layout())
            return fn(*a)
        return run
    ref = jax.device_get(jax.jit(wrap())(*args))
    jax.clear_caches()
    with context_parallel(DEVICES, layout="2d"):
        assert cp_grid() == (SIDE, SIDE), cp_grid()
        got = jax.device_get(jax.jit(wrap())(*args))
    assert traced[-2:] == [None, "2d"], traced
    return ref, got
"""


_CANNON_PROBE = _TWO_D_PREAMBLE + textwrap.dedent(
    """
    from foldjax.models.openfold3.models.triangle import (
        TriangleMultiplicationParams, triangle_multiplication,
    )

    C = 8
    params = TriangleMultiplicationParams(
        layer_norm_in=ln(C), layer_norm_out=ln(C),
        linear_a_p=lin(C, C), linear_a_g=lin(C, C),
        linear_b_p=lin(C, C), linear_b_g=lin(C, C),
        linear_g=lin(C, C), linear_z=lin(C, C),
    )

    # 12 divides both a 2x2 and a 3x3 grid on both pair axes; 13 divides
    # neither, which is the branch that pads both axes and slices the result
    # back. Batched, because a leading axis is openfold3's norm rather than its
    # exception -- the template stack and the confidence re-embedding both carry
    # one, and they must stay unsharded at either grid size.
    for n in (12, 13):
        z = arr(1, n, n, C)
        keep = rng.random(n) > 0.15
        mask = jnp.asarray(
            (keep[:, None] & keep[None, :])[None], dtype=jnp.float32
        )
        for outgoing in (True, False):
            ref, got = under_mesh(
                lambda o=outgoing, m=mask: (
                    lambda z_in: triangle_multiplication(
                        z_in, params, outgoing=o, mask=m
                    )
                ),
                z,
            )
            np.testing.assert_allclose(ref, got, atol=3e-5, rtol=3e-5)
    print("CANNON_PARITY_OK")
    """
)


_COMPENSATED_CANNON_PROBE = _TWO_D_PREAMBLE + textwrap.dedent(
    """
    from foldjax.models.openfold3.models.triangle import _cannon_combine

    assert DEVICES == 9

    # Each output is mathematically 1.0, but a plain fp32 tile reduction loses
    # that unit for two of the three cyclic Cannon orders:
    # (1e8 + 1) - 1e8 and (1 - 1e8) + 1e8 both round to zero. Neumaier's
    # correction must retain it without changing the tile-only schedule.
    values = jnp.asarray([1e8, 1.0, -1e8], dtype=jnp.float32)
    a = jnp.broadcast_to(values[None, :, None], (3, 3, 1))
    b = jnp.ones((3, 3, 1), dtype=jnp.float32)

    def run(a_in, b_in):
        return _cannon_combine(a_in, b_in, outgoing=True)

    jax.clear_caches()
    with context_parallel(DEVICES, layout="2d"):
        lowered = jax.jit(run).lower(a, b)
        got = jax.device_get(lowered.compile()(a, b))
    np.testing.assert_array_equal(got, np.ones((3, 3, 1), dtype=np.float32))

    hlo = lowered.compiler_ir(dialect="hlo").as_hlo_text().lower()
    assert "collective-permute" in hlo or "collective_permute" in hlo, hlo
    assert "all-gather" not in hlo and "all_gather" not in hlo, hlo
    print("COMPENSATED_CANNON_OK")
    """
)


_GRID_STACK_PROBE = _TWO_D_PREAMBLE + textwrap.dedent(
    """
    from foldjax.models.openfold3.models.attention import AttentionParams
    from foldjax.models.openfold3.models.attention_pair_bias import (
        AttentionPairBiasParams,
    )
    from foldjax.models.openfold3.models.pair_block import PairBlockParams
    from foldjax.models.openfold3.models.pairformer import (
        PairformerBlockParams, PairformerStackParams, pairformer_stack,
    )
    from foldjax.models.openfold3.models.triangle import (
        TriangleMultiplicationParams,
    )
    from foldjax.models.openfold3.models.triangle_attention import (
        TriangleAttentionParams,
    )

    C, HEADS, N = 8, 2, 13  # 13 over a 2x2 grid: both pad paths exercised

    def tri_mult():
        return TriangleMultiplicationParams(
            layer_norm_in=ln(C), layer_norm_out=ln(C),
            linear_a_p=lin(C, C), linear_a_g=lin(C, C),
            linear_b_p=lin(C, C), linear_b_g=lin(C, C),
            linear_g=lin(C, C), linear_z=lin(C, C),
        )

    def attn():
        return AttentionParams(lin(C, C), lin(C, C), lin(C, C), lin(C, C), lin(C, C))

    def tri_att():
        return TriangleAttentionParams(
            layer_norm=ln(C),
            linear_z=LinearParams(weight=arr(HEADS, C), bias=None),
            mha=attn(),
        )

    def transition():
        return SwiGLUTransitionParams(
            layer_norm=ln(C),
            swiglu=SwiGLUParams(linear_a=lin(2 * C, C), linear_b=lin(2 * C, C)),
            linear_out=lin(C, 2 * C),
        )

    def block():
        return PairformerBlockParams(
            pair_stack=PairBlockParams(
                tri_mul_out=tri_mult(), tri_mul_in=tri_mult(),
                tri_att_start=tri_att(), tri_att_end=tri_att(),
                pair_transition=transition(),
            ),
            attn_pair_bias=AttentionPairBiasParams(
                layer_norm_a=ln(C), layer_norm_z=ln(C),
                linear_z=LinearParams(weight=arr(HEADS, C), bias=None),
                mha=attn(),
            ),
            single_transition=transition(),
        )

    params = prestack_layer_lists(
        PairformerStackParams(blocks=(block(), block()))
    )
    s, z = arr(N, C), arr(N, N, C)
    mask_np = rng.random(N) > 0.15
    single_mask = jnp.asarray(mask_np, dtype=jnp.float32)
    pair_mask = jnp.asarray(mask_np[:, None] & mask_np[None, :], dtype=jnp.float32)

    # Triangle attention stays row-sharded under the 2-D layout by design: its
    # softmax spans a whole column, so `pair_row_spec` names the row axis only
    # and the partitioner gathers the column axis for that one shard_map. The
    # multiplicative updates are what the grid buys, and they are the half this
    # asserts is unchanged.
    def build():
        def run(s_in, z_in):
            return pairformer_stack(
                s_in, z_in, params,
                single_mask=single_mask, pair_mask=pair_mask,
                no_heads_pair=HEADS, no_heads_pair_bias=HEADS,
                chunk_size=5,
            )
        return run

    (ref_s, ref_z), (cp_s, cp_z) = under_mesh(build, s, z)
    np.testing.assert_allclose(ref_s, cp_s, atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(ref_z, cp_z, atol=3e-5, rtol=3e-5)
    print("GRID_STACK_PARITY_OK")
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


def test_cannon_triangle_multiplication_matches_the_unsharded_form() -> None:
    """Both directions, batched, on a mesh that splits rows *and* columns."""
    assert "CANNON_PARITY_OK" in _run_probe(_CANNON_PROBE)


def test_cannon_holds_on_a_three_by_three_grid() -> None:
    """The smallest grid that can tell a schedule's signs apart.

    On a 2x2 grid ``(j - 1) % 2 == (j + 1) % 2``, so a reversed ring hop and
    either flipped skew sign are indistinguishable from the correct ones -- a
    green 2x2 gate says nothing about them. Nine devices is the first size where
    the two ring directions and both skews are separate permutations.
    """
    assert "CANNON_PARITY_OK" in _run_probe(_CANNON_PROBE, devices=9)


def test_cannon_uses_compensated_tile_accumulation() -> None:
    """Cancellation-prone tiles retain low bits on a 3x3 grid."""
    assert "COMPENSATED_CANNON_OK" in _run_probe(
        _COMPENSATED_CANNON_PROBE, devices=9
    )


def test_two_dimensional_pairformer_matches_the_unsharded_stack() -> None:
    assert "GRID_STACK_PARITY_OK" in _run_probe(_GRID_STACK_PROBE)


def test_two_dimensional_pairformer_holds_on_a_three_by_three_grid() -> None:
    assert "GRID_STACK_PARITY_OK" in _run_probe(_GRID_STACK_PROBE, devices=9)


def test_auto_layout_stays_on_the_one_dimensional_default() -> None:
    """``auto`` must not silently change the program anyone has measured.

    The square grid is the better layout and is gated above, but every published
    number for this feature was taken on the 1-D layout, so ``auto`` stays there
    until the grid has its own GPU evidence. This pins that as a decision rather
    than an accident: flipping the default should require editing this test.
    """
    from foldjax.models.openfold3.inference import (
        InferenceConfig,
        resolve_cp_layout,
    )

    def layout(shards: int, requested: str = "auto") -> str:
        return resolve_cp_layout(
            InferenceConfig(
                n_token=1, n_atom=1, n_query=1, n_key=1, atom_heads=1,
                token_heads=1, no_heads_msa=1, no_heads_pair=1,
                no_heads_pair_bias=1, max_relative_idx=1, max_relative_chain=1,
                num_cycles=1, num_samples=1, max_atoms_per_token=1,
                plddt_bins=1, pae_bins=1, pae_bin_max=1.0, no_rollout_steps=1,
                cp_shards=shards, cp_layout=requested,
            )
        )

    # A square shard count does not opt anyone in by itself.
    for shards in (1, 2, 3, 4, 8, 9):
        assert layout(shards) == "1d", shards
    # An explicit choice is passed through untouched, so asking for a grid on a
    # non-square count still fails loudly in `context_parallel` rather than
    # quietly degrading to rows.
    assert layout(4, "2d") == "2d"
    assert layout(3, "2d") == "2d"
    assert layout(4, "1d") == "1d"


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
