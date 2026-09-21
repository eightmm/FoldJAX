"""CPU proof gates for `cp_fused_attention`, Boltz-2's two local CP attentions.

Under a mesh every fused kernel in this port resolves to -- or is refused in
favour of -- an XLA path, because a kernel that consumes a whole token axis
cannot be partitioned. Two diffusion attentions are not in that position: the
atom-window one runs entirely inside its own `shard_map` on halo-exchanged
keys, and the token one runs on a grid-transposed key tile whose softmax is
finished by collectives. `cp_fused_attention` opens those two and nothing else.

What the card decides -- which tokamax implementation it selects, and what that
costs -- is not here. What is here is everything a CPU can settle:

* **dispatch isolation** -- the fused callable is reached at the site the
  option names, once per attention, and at no other site; `off` reaches it
  nowhere and compiles the program it compiled before;
* **the sharding contract** -- with the kernel resolved to tokamax's own
  portable XLA implementation, the option-on program equals the option-off
  program on deliberately asymmetric per-rank inputs, on 2x2 and on 3x3, and
  the token merge produces byte-identical output on every `cp_col` replica,
  which is the promise the unchecked `shard_map` stops verifying;
* **empty-tile semantics** -- a column tile with no valid key contributes
  nothing rather than outvoting a neighbour that has keys, in either order,
  and a globally empty row is zeros rather than NaN;
* **refusal** -- every way of asking for a site nothing would honour raises.

Each probe runs in a subprocess because a forced device count has to be set
before JAX initialises. Every arm builds its own `jax.jit` object: two arms
over one function object share one trace, and an exactly-zero difference
between them is what that looks like.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from tests.models.cp_probe_env import inherited_environment


def _run(source: str, *, devices: int) -> str:
    env = {
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
        **inherited_environment(),
    }
    completed = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


#: A one-layer diffusion transformer with the native parameter schema, small
#: enough to trace quickly and wide enough that the attention is not degenerate.
_ATOM_PRELUDE = r"""
import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models._cp import context_parallel
from foldjax.models._cp_atom import place_atoms
from foldjax.models._cp_attention import cp_fused_attention_scope
from foldjax.models.boltz2.models.diffusion import diffusion_transformer as dt
from foldjax.models.boltz2.models.diffusion.atom import atom_transformer_forward

rng = np.random.default_rng(20260921)
DIM = COND = 8
HEADS = 2
W, HK = 32, 128
BATCH, ATOMS = 1, 128
WINDOWS = ATOMS // W


def weight(rows, cols, scale=0.2):
    return jnp.asarray(rng.normal(scale=scale, size=(rows, cols)), jnp.float32)


def bvec(size, scale=0.1):
    return jnp.asarray(rng.normal(scale=scale, size=(size,)), jnp.float32)


def norm(size, with_bias=True):
    out = {"scale": jnp.ones((size,), jnp.float32)}
    if with_bias:
        out["bias"] = jnp.zeros((size,), jnp.float32)
    return out


def adaln(cond, act):
    return {
        "s_norm": norm(cond, with_bias=False),
        "s_scale": {"kernel": weight(cond, act), "bias": bvec(act)},
        "s_bias": {"kernel": weight(cond, act)},
    }


def transformer_layer(dim, cond):
    hidden = 2 * dim
    return {
        "adaln": adaln(cond, dim),
        "pair_bias_attn": {
            "num_heads": HEADS,
            "proj_q": {"kernel": weight(dim, dim), "bias": bvec(dim)},
            "proj_k": {"kernel": weight(dim, dim)},
            "proj_v": {"kernel": weight(dim, dim)},
            "proj_g": {"kernel": weight(dim, dim)},
            "proj_o": {"kernel": weight(dim, dim)},
        },
        "output_projection": {"kernel": weight(cond, dim), "bias": bvec(dim)},
        "transition": {
            "adaln": adaln(cond, dim),
            "swish_gate": {"kernel": weight(dim, 2 * hidden)},
            "a_to_b": {"kernel": weight(dim, hidden)},
            "b_to_a": {"kernel": weight(hidden, dim)},
            "output_projection": {"kernel": weight(cond, dim), "bias": bvec(dim)},
        },
    }


params = {"diffusion_transformer": {"layers": [transformer_layer(DIM, COND)]}}
# Asymmetric on purpose: each atom shard is scaled differently, so a body that
# read a neighbour's window, or a merge that mixed the shards up, cannot come
# back equal by symmetry.
atom_scale = jnp.repeat(
    jnp.asarray([1.0, 30.0], jnp.float32), ATOMS // 2
)[None, :, None]
q = jnp.asarray(rng.normal(size=(BATCH, ATOMS, DIM)), jnp.float32) * atom_scale
c = jnp.asarray(rng.normal(size=(BATCH, ATOMS, COND)), jnp.float32)
bias = jnp.asarray(rng.normal(size=(BATCH, WINDOWS, W, HK, HEADS)), jnp.float32)
mask = jnp.asarray(rng.random((BATCH, ATOMS)) > 0.1, jnp.float32)


def make_run():
    # One fresh jitted object per arm. Two arms over one function object share
    # a trace and the second never sees the scope.
    def run(qv, cv, bv, mv):
        return atom_transformer_forward(
            params,
            q=qv,
            c=cv,
            bias=bv,
            to_keys=lambda x: x,
            mask=mv,
            attn_window_queries=W,
            attn_window_keys=HK,
            multiplicity=1,
            attention_backend="xla",
            atom_context_parallel=True,
        )

    return jax.jit(run)


calls = []
_real_fused = dt.tokamax_dot_product_attention


def census(qa, ka, va, ba, ma, *, scale, backend="xla", implementation=None):
    calls.append((backend, tuple(qa.shape), tuple(ka.shape)))
    return _real_fused(
        qa, ka, va, ba, ma,
        scale=scale, backend=backend, implementation=implementation,
    )


dt.tokamax_dot_product_attention = census
"""


_ATOM_SITE_PROBE = _ATOM_PRELUDE + textwrap.dedent(
    r"""
    assert jax.device_count() == 4

    with context_parallel(4, layout="2d"):
        qd = place_atoms(q, atom_axis=1)
        cd = place_atoms(c, atom_axis=1)
        md = place_atoms(mask, atom_axis=1)
        off_fn = make_run()
        off = jax.device_get(off_fn(qd, cd, bias, md))
        off_hlo = off_fn.lower(qd, cd, bias, md).compiler_ir(
            dialect="hlo"
        ).as_hlo_text().lower()
        assert not calls, calls
        with cp_fused_attention_scope("atom"):
            on_fn = make_run()
            on = jax.device_get(on_fn(qd, cd, bias, md))
            on_hlo = on_fn.lower(qd, cd, bias, md).compiler_ir(
                dialect="hlo"
            ).as_hlo_text().lower()

    # One layer, one attention, one fused call -- and the shapes are the
    # window-local ones: 32 queries per window against the 128 halo-exchanged
    # keys, never a token axis.
    assert len(calls) == 1, calls
    backend, q_shape, k_shape = calls[0]
    assert backend == "tokamax", calls
    assert q_shape[-3] == W and k_shape[-3] == HK, calls
    assert q_shape[0] == k_shape[0], calls

    np.testing.assert_allclose(off, on, atol=1e-5, rtol=1e-5)
    assert np.isfinite(on).all()
    # The halo is the only communication either arm needs, and the fused arm
    # adds none: no gather appeared, and the collective-permute count is the
    # halo's, unchanged.
    assert "all-gather" not in on_hlo and "all_gather" not in on_hlo
    for name in ("collective-permute", "collective_permute"):
        assert on_hlo.count(name) == off_hlo.count(name), name
    print("ATOM_SITE_MAXDIFF", float(np.abs(off - on).max()))
    print("ATOM_SITE_OK")
    """
)


_ATOM_ISOLATION_PROBE = _ATOM_PRELUDE + textwrap.dedent(
    r"""
    assert jax.device_count() == 4

    with context_parallel(4, layout="2d"):
        qd = place_atoms(q, atom_axis=1)
        cd = place_atoms(c, atom_axis=1)
        md = place_atoms(mask, atom_axis=1)
        # The token site named alone must not open the atom one. The diffusion
        # atom transformer is the only consumer here, so the census must stay
        # empty through a `token` run as it does through an `off` one.
        with cp_fused_attention_scope("token"):
            make_run()(qd, cd, bias, md).block_until_ready()
        assert not calls, calls
        with cp_fused_attention_scope("atom+token"):
            make_run()(qd, cd, bias, md).block_until_ready()
        assert len(calls) == 1, calls
    print("ATOM_ISOLATION_OK")
    """
)


#: The token site. `pair_bias_attention_2d` is not a ring: one grid transpose
#: puts a key tile on each column device, and `pmax`/`psum` over `cp_col`
#: finish the softmax. So a "tile" here is a column device, and the merge is
#: the collective.
_TOKEN_PRELUDE = r"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec

from foldjax.models import _cp_atom
from foldjax.models._cp import (
    CP_COL_AXIS,
    CP_ROW_AXIS,
    context_parallel,
    cp_grid,
    cp_mesh,
    permute,
    transpose_perm,
)
from foldjax.models._cp_atom import (
    _fused_token_tile_2d,
    atom_spec,
    pair_bias_attention_2d,
)
from foldjax.models._cp_attention import (
    cp_fused_attention_scope,
    tile_attention_tokamax,
)

#: Tokamax's portable implementation, addressed through the same entry point
#: the Triton one goes through (`base.DotProductAttention.__call__`), so the
#: operand conversions, the residual convention and the merge are the ones the
#: card will run -- only the kernel behind them differs.
XLA_TILE = functools.partial(tile_attention_tokamax, implementation="xla")


def case(side, *, mask_values=None, seed=20260921):
    rng = np.random.default_rng(seed)
    tokens = 4 * side
    heads, dim = 2, 3
    # Asymmetric per rank: each row shard's magnitude differs by 30x, so the
    # per-tile maxima are far apart and a merge that skipped a rescale, or
    # paired a tile with the wrong bias columns, cannot pass by symmetry.
    scale_rows = jnp.repeat(
        jnp.asarray([1.0 + 29.0 * i for i in range(side)], jnp.float32),
        tokens // side,
    )[None, :, None]
    q = jnp.asarray(rng.normal(size=(1, tokens, heads, dim)), jnp.float32)
    k = jnp.asarray(rng.normal(size=(1, tokens, heads, dim)), jnp.float32)
    v = jnp.asarray(rng.normal(size=(1, tokens, heads, dim)), jnp.float32)
    q = q * scale_rows[..., None]
    k = k * scale_rows[..., None]
    bias = jnp.asarray(
        rng.normal(scale=3.0, size=(1, heads, tokens, tokens)), jnp.float32
    )
    if mask_values is None:
        keep = rng.random((1, tokens)) > 0.2
        # At least one key per column tile, so this case is the ordinary one
        # and the empty-tile cases stay the empty-tile probe's subject.
        keep[:, ::4] = True
        mask = jnp.asarray(keep.astype(np.float32), jnp.float32)
    else:
        mask = jnp.asarray([mask_values], jnp.float32)
    return q, k, v, bias, mask, float(dim) ** -0.5


def arm(q, k, v, bias, mask, scale, request):
    # A fresh closure per arm, for the reason the atom probe states.
    fn = jax.jit(
        lambda a, b, c, d, e: pair_bias_attention_2d(a, b, c, d, e, scale=scale)
    )
    if request is None:
        return jax.device_get(fn(q, k, v, bias, mask)), fn
    with cp_fused_attention_scope(request):
        return jax.device_get(fn(q, k, v, bias, mask)), fn
"""


_TOKEN_CONTRACT_PROBE = _TOKEN_PRELUDE + textwrap.dedent(
    r"""
    side = PROBE_SIDE
    devices = side * side
    assert jax.device_count() == devices, jax.devices()

    q, k, v, bias, mask, scale = case(side)

    # A tripwire on the arm, not a label on it: at 3x3 the two arms agree to
    # the last bit, and without this an arm that never selected the fused tile
    # would read as a pass rather than as nothing measured.
    fired = []

    def counted(*tile, **keywords):
        fired.append(len(fired))
        return XLA_TILE(*tile, **keywords)

    _cp_atom.tile_attention_tokamax = counted
    with context_parallel(devices, layout="2d"):
        assert cp_grid() == (side, side), cp_grid()
        off, off_fn = arm(q, k, v, bias, mask, scale, None)
        assert not fired, fired
        on, on_fn = arm(q, k, v, bias, mask, scale, "token")
        assert len(fired) == 1, fired
        off_hlo = off_fn.lower(q, k, v, bias, mask).compiler_ir(
            dialect="hlo"
        ).as_hlo_text().lower()
        on_hlo = on_fn.lower(q, k, v, bias, mask).compiler_ir(
            dialect="hlo"
        ).as_hlo_text().lower()

    np.testing.assert_allclose(off, on, atol=1e-5, rtol=1e-5)
    assert np.isfinite(on).all()
    assert "all-gather" not in on_hlo and "all_gather" not in on_hlo, on_hlo[:400]
    # The collectives are the same ones in the same number: one pmax and two
    # psums per call, on both arms.
    for name in ("all-reduce", "all_reduce"):
        assert on_hlo.count(name) == off_hlo.count(name), (
            name, on_hlo.count(name), off_hlo.count(name)
        )
    print("TOKEN_CONTRACT_MAXDIFF", side, float(np.abs(off - on).max()))
    print("TOKEN_CONTRACT_OK", side)
    """
)


_TOKEN_REPLICATION_PROBE = _TOKEN_PRELUDE + textwrap.dedent(
    r"""
    assert jax.device_count() == 4

    side = 2
    q, k, v, bias, mask, scale = case(side)
    qkv_spec = None

    with context_parallel(4, layout="2d") as mesh:
        qkv_spec = atom_spec(4, atom_axis=1)
        mask_spec = atom_spec(2, atom_axis=1)
        bias_spec = PartitionSpec(None, None, CP_ROW_AXIS, CP_COL_AXIS)

        def column_view(q_local, k_local, v_local, bias_local, mask_local):
            # The production body's own transpose, then the production merge,
            # returned with the column axis materialised instead of promised.
            transpose = transpose_perm(side)
            k_local = permute(k_local, transpose)
            v_local = permute(v_local, transpose)
            mask_local = permute(mask_local, transpose)
            out = _fused_token_tile_2d(
                q_local,
                k_local,
                v_local,
                bias_local,
                mask_local,
                scale=scale,
                out_dtype=v_local.dtype,
            )
            return out[None]

        _cp_atom.tile_attention_tokamax = XLA_TILE
        per_column = jax.device_get(
            jax.jit(
                lambda *a: jax.shard_map(
                    column_view,
                    mesh=mesh,
                    in_specs=(
                        qkv_spec, qkv_spec, qkv_spec, bias_spec, mask_spec
                    ),
                    out_specs=PartitionSpec(
                        CP_COL_AXIS, None, CP_ROW_AXIS, None, None
                    ),
                    check_vma=False,
                )(*a)
            )(q, k, v, bias, mask)
        )

    # `out_specs` in production names only `cp_row`, which promises the result
    # is replicated along `cp_col`; `check_vma=False` stops that promise being
    # verified, so it is verified here instead. Bit-equality, not a tolerance:
    # a psum hands every participant the same bytes.
    assert per_column.shape[0] == side, per_column.shape
    for column in range(1, side):
        np.testing.assert_array_equal(per_column[0], per_column[column])
    assert np.isfinite(per_column).all()
    print("TOKEN_REPLICATION_OK")
    """
)


_TOKEN_EMPTY_PROBE = _TOKEN_PRELUDE + textwrap.dedent(
    r"""
    assert jax.device_count() == 4

    side = 2
    tokens = 4 * side
    half = tokens // side
    # With side=2 the grid transpose gives column `c` the key block row `c`
    # owned, so masking one half of the tokens empties exactly one column's
    # tile. An empty tile is not an empty global row: the other column still
    # has keys and must be the whole answer.
    cases = {
        # first tile empty, second has keys
        "empty_then_valid": [0.0] * half + [1.0] * half,
        # first tile has keys, second empty
        "valid_then_empty": [1.0] * half + [0.0] * half,
        # no keys anywhere: the globally empty row
        "all_empty": [0.0] * tokens,
        # one key in each tile, so neither the empty nor the full case
        "sparse": ([1.0] + [0.0] * (half - 1)) * side,
    }

    # A tripwire, because every case below agrees with the checked arm
    # exactly: without it an arm that silently never selected the fused tile
    # would read as a pass rather than as nothing measured.
    fired = []

    def counted(*tile, **keywords):
        fired.append(len(fired))
        return XLA_TILE(*tile, **keywords)

    _cp_atom.tile_attention_tokamax = counted
    for name, values in cases.items():
        q, k, v, bias, mask, scale = case(side, mask_values=values)
        before = len(fired)
        with context_parallel(4, layout="2d"):
            off, _ = arm(q, k, v, bias, mask, scale, None)
            assert len(fired) == before, (name, fired)
            on, _ = arm(q, k, v, bias, mask, scale, "token")
        assert len(fired) == before + 1, (name, fired)
        assert np.isfinite(on).all(), (name, on)
        assert not np.isnan(on).any(), (name, on)
        if name == "all_empty":
            np.testing.assert_array_equal(on, np.zeros_like(on))
        np.testing.assert_allclose(on, off, atol=1e-5, rtol=1e-5, err_msg=name)
        print("EMPTY_CASE", name, float(np.abs(on - off).max()))
    print("TOKEN_EMPTY_OK")
    """
)


_TOKEN_CENSUS_PROBE = _TOKEN_PRELUDE + textwrap.dedent(
    r"""
    assert jax.device_count() == 4

    side = 2
    q, k, v, bias, mask, scale = case(side)
    seen = []

    def census(*tile, **keywords):
        seen.append(tuple(operand.shape for operand in tile[:3]))
        return XLA_TILE(*tile, **keywords)

    _cp_atom.tile_attention_tokamax = census
    with context_parallel(4, layout="2d"):
        arm(q, k, v, bias, mask, scale, None)
        assert not seen, seen
        arm(q, k, v, bias, mask, scale, "atom")
        assert not seen, seen
        arm(q, k, v, bias, mask, scale, "token")
        assert len(seen) == 1, seen
        arm(q, k, v, bias, mask, scale, "atom+token")
        assert len(seen) == 2, seen
    # The tile the kernel sees is the local one: queries this row owns against
    # the keys this column owns, never a full token axis.
    tokens = 4 * side
    for q_shape, k_shape, v_shape in seen:
        assert q_shape[-2] == tokens // side, seen
        assert k_shape == v_shape == q_shape, seen
    print("TOKEN_CENSUS_OK")
    """
)


_TOKEN_PLATFORM_PROBE = _TOKEN_PRELUDE + textwrap.dedent(
    r"""
    assert jax.device_count() == 4

    # The token site pins tokamax's Triton implementation for the reason the
    # ring pins its own: only that entry point returns the residuals the merge
    # needs, and a sequence of implementations would be a silent fallback
    # chain. So the site is GPU-only, and what a CPU gets is a refusal out of
    # tokamax rather than a quiet lowering to something else. Asserted rather
    # than described, because "GPU-only" in the documentation is a claim about
    # what happens here.
    q, k, v, bias, mask, scale = case(2)
    with context_parallel(4, layout="2d"):
        try:
            arm(q, k, v, bias, mask, scale, "token")
        except NotImplementedError as error:
            assert "cpu" in str(error).lower(), error
        else:
            raise AssertionError("the pinned Triton tile lowered on a CPU")
    print("TOKEN_PLATFORM_OK")
    """
)


_REFUSAL_PROBE = textwrap.dedent(
    r"""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import pytest

    from foldjax.models import _cp_attention
    from foldjax.models._cp import context_parallel
    from foldjax.models._cp_attention import (
        CP_FUSED_ATTENTION_REQUESTS,
        cp_fused_attention,
        cp_fused_attention_scope,
        cp_fused_attention_sites,
        cp_fused_shard_map_options,
        resolve_cp_fused_attention,
    )

    assert CP_FUSED_ATTENTION_REQUESTS == ("off", "atom", "token", "atom+token")
    assert cp_fused_attention() == "off"
    assert cp_fused_attention_sites("off") == frozenset()
    assert cp_fused_attention_sites("atom+token") == frozenset({"atom", "token"})

    # The scope restores, including through a failure.
    with cp_fused_attention_scope("atom") as name:
        assert name == "atom"
        assert cp_fused_attention() == "atom"
    assert cp_fused_attention() == "off"
    with cp_fused_attention_scope(None) as name:
        assert name == "off"
    try:
        with cp_fused_attention_scope("atom"):
            raise RuntimeError("deliberate")
    except RuntimeError:
        pass
    assert cp_fused_attention() == "off"

    # A misspelling is refused where it is spelled, not at the site.
    for bad in ("tokamax", "on", "token+atom", "atom+atom", ""):
        try:
            with cp_fused_attention_scope(bad):
                raise AssertionError(bad)
        except ValueError as error:
            assert "cp_fused_attention" in str(error), error
    for bad in ("trunk", "msa"):
        try:
            resolve_cp_fused_attention(bad)
            raise AssertionError(bad)
        except ValueError as error:
            assert "cp_fused_attention site" in str(error), error

    # Off is off everywhere, mesh or no mesh, and adds no `shard_map` keyword.
    for site in ("atom", "token"):
        assert resolve_cp_fused_attention(site) is False
    assert cp_fused_shard_map_options(False) == {}
    assert cp_fused_shard_map_options(True) == {"check_vma": False}

    # Refused, not downgraded: no mesh.
    with cp_fused_attention_scope("atom+token"):
        for site in ("atom", "token"):
            try:
                resolve_cp_fused_attention(site)
                raise AssertionError(site)
            except RuntimeError as error:
                assert "no mesh active" in str(error), error

    # Refused, not downgraded: no tokamax.
    _cp_attention.tokamax_available = lambda: False
    with context_parallel(4, layout="2d"):
        with cp_fused_attention_scope("atom+token"):
            for site in ("atom", "token"):
                try:
                    resolve_cp_fused_attention(site)
                    raise AssertionError(site)
                except RuntimeError as error:
                    assert "tokamax package" in str(error), error
    print("REFUSAL_OK")
    """
)


_ENTRY_REFUSAL_PROBE = textwrap.dedent(
    r"""
    import jax
    import jax.numpy as jnp

    from foldjax.models._cp import context_parallel
    from foldjax.models._cp_attention import cp_fused_attention_scope
    from foldjax.models.boltz2.models import predict as predict_module

    assert jax.device_count() == 4


    def entry(**options):
        # The model entry validates its knobs before it touches features, so a
        # call with nothing but the knobs reaches the refusal and stops there.
        return predict_module.boltz2_predict(
            params={},
            feats={},
            key=jax.random.key(0),
            **options,
        )


    def refused(fragment, **options):
        try:
            entry(**options)
        except ValueError as error:
            assert fragment in str(error), (fragment, str(error))
            return
        except Exception as error:  # got past the refusal
            raise AssertionError(f"{fragment}: reached {error!r}") from error
        raise AssertionError(fragment)


    # No mesh at all.
    with cp_fused_attention_scope("atom"):
        refused("needs an active mesh")
    with context_parallel(4, layout="1d"):
        # The token site exists only under the square grid.
        with cp_fused_attention_scope("token"):
            refused("only under the 2-D layout", atom_context_parallel=True)
        # The atom site exists only where the atom graph is distributed.
        with cp_fused_attention_scope("atom"):
            refused("leaves it replicated", atom_context_parallel=False)
    print("ENTRY_REFUSAL_OK")
    """
)


_OFF_IDENTITY_PROBE = _ATOM_PRELUDE + textwrap.dedent(
    r"""
    import functools
    import hashlib
    import re

    from foldjax.models import _cp_atom
    from foldjax.models._cp_atom import pair_bias_attention_2d
    from foldjax.models._cp_attention import tile_attention_tokamax

    assert jax.device_count() == 4

    # The token site pins tokamax's Triton implementation, which raises
    # `NotImplementedError: Not supported on cpu` here rather than quietly
    # lowering to something else -- refused, not downgraded, by tokamax
    # itself. Resolve it to tokamax's portable implementation so the token
    # program can be fingerprinted at all; the selection, the unchecked
    # `shard_map` and the merge are the same either way.
    _cp_atom.tile_attention_tokamax = functools.partial(
        tile_attention_tokamax, implementation="xla"
    )


    def fingerprint(compiled, *operands):
        lowered = compiled.lower(*operands)
        text = lowered.compiler_ir(dialect="hlo").as_hlo_text()
        text = re.sub(r",?\s*metadata=\{[^}]*\}", "", text)
        text = re.sub(r",?\s*stack_frame_id=\d+", "", text)
        digest = hashlib.sha256(
            re.sub(r"\s+", " ", text).strip().encode()
        ).hexdigest()
        return digest, lowered.compile().memory_analysis().temp_size_in_bytes


    with context_parallel(4, layout="2d"):
        qd = place_atoms(q, atom_axis=1)
        cd = place_atoms(c, atom_axis=1)
        md = place_atoms(mask, atom_axis=1)
        absent = fingerprint(make_run(), qd, cd, bias, md)
        with cp_fused_attention_scope("off"):
            spelled = fingerprint(make_run(), qd, cd, bias, md)
        with cp_fused_attention_scope("token"):
            other_site = fingerprint(make_run(), qd, cd, bias, md)
        with cp_fused_attention_scope("atom"):
            fused = fingerprint(make_run(), qd, cd, bias, md)

    # Spelling the released value, and spelling a site this program does not
    # contain, both compile the program an omitted option compiles -- hash and
    # temp arena alike.
    assert spelled == absent, (spelled, absent)
    assert other_site == absent, (other_site, absent)
    # And the site this program does contain compiles a different one, which
    # is what says the branch carrying the new selection was traced rather
    # than skipped.
    assert fused != absent, (fused, absent)

    # The same three questions at the token site.
    heads, dim, tokens = 2, 3, 8
    q2 = jnp.asarray(rng.normal(size=(1, tokens, heads, dim)), jnp.float32)
    k2 = jnp.asarray(rng.normal(size=(1, tokens, heads, dim)), jnp.float32)
    v2 = jnp.asarray(rng.normal(size=(1, tokens, heads, dim)), jnp.float32)
    b2 = jnp.asarray(rng.normal(size=(1, heads, tokens, tokens)), jnp.float32)
    m2 = jnp.asarray(rng.random((1, tokens)) > 0.2, jnp.float32)


    def token_arm():
        return jax.jit(
            lambda *operands: pair_bias_attention_2d(
                *operands, scale=float(dim) ** -0.5
            )
        )


    with context_parallel(4, layout="2d"):
        token_absent = fingerprint(token_arm(), q2, k2, v2, b2, m2)
        with cp_fused_attention_scope("off"):
            token_spelled = fingerprint(token_arm(), q2, k2, v2, b2, m2)
        with cp_fused_attention_scope("atom"):
            token_other = fingerprint(token_arm(), q2, k2, v2, b2, m2)
        with cp_fused_attention_scope("token"):
            token_fused = fingerprint(token_arm(), q2, k2, v2, b2, m2)

    assert token_spelled == token_absent, (token_spelled, token_absent)
    assert token_other == token_absent, (token_other, token_absent)
    assert token_fused != token_absent, (token_fused, token_absent)
    print("OFF_IDENTITY_OK", absent[0][:16], token_absent[0][:16])
    """
)


def test_the_option_off_compiles_the_program_it_compiled_before() -> None:
    """Both routes, hash and temp arena, against an omitted option.

    The cross-tree half of this gate -- the same seven programs fingerprinted
    against the tree before the change -- is
    `scripts/cp_fused_attention_fingerprints.py`, which takes a `PYTHONPATH`
    and prints one line per program.
    """

    assert "OFF_IDENTITY_OK" in _run(_OFF_IDENTITY_PROBE, devices=4)


def test_the_atom_site_reaches_the_fused_kernel_and_nothing_else() -> None:
    assert "ATOM_SITE_OK" in _run(_ATOM_SITE_PROBE, devices=4)


def test_the_token_request_does_not_open_the_atom_site() -> None:
    assert "ATOM_ISOLATION_OK" in _run(_ATOM_ISOLATION_PROBE, devices=4)


def test_the_token_site_reaches_the_fused_tile_and_nothing_else() -> None:
    assert "TOKEN_CENSUS_OK" in _run(_TOKEN_CENSUS_PROBE, devices=4)


@pytest.mark.parametrize("side", [2, 3])
def test_the_token_site_is_the_same_attention_on_asymmetric_ranks(side: int) -> None:
    """Both grid sides. A 2x2 cannot see a sign or ownership error that a 3x3
    can: with two columns the two hops are each other's inverse."""

    assert f"TOKEN_CONTRACT_OK {side}" in _run(
        f"PROBE_SIDE = {side}\n" + _TOKEN_CONTRACT_PROBE,
        devices=side * side,
    )


def test_the_token_output_is_identical_on_every_column_replica() -> None:
    assert "TOKEN_REPLICATION_OK" in _run(_TOKEN_REPLICATION_PROBE, devices=4)


def test_an_empty_column_tile_contributes_nothing_in_either_order() -> None:
    assert "TOKEN_EMPTY_OK" in _run(_TOKEN_EMPTY_PROBE, devices=4)


def test_the_token_tile_refuses_a_cpu_rather_than_lowering_to_something_else(
) -> None:
    assert "TOKEN_PLATFORM_OK" in _run(_TOKEN_PLATFORM_PROBE, devices=4)


def test_the_option_is_refused_rather_than_downgraded() -> None:
    assert "REFUSAL_OK" in _run(_REFUSAL_PROBE, devices=4)


def test_the_model_entry_refuses_a_site_nothing_would_honour() -> None:
    assert "ENTRY_REFUSAL_OK" in _run(_ENTRY_REFUSAL_PROBE, devices=4)
