import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from foldjax.models.boltz2.bridge.torch_checkpoint import load_checkpoint_state_dict
from foldjax.models.boltz2.bridge.torch_mapping import map_msa_module_state_dict
from foldjax.models.boltz2.models.trunk_blocks.msa import (
    _OPM_BUDGET_BYTES,
    _PWA_BUDGET_BYTES,
    _auto_outer_product_chunk,
    _auto_pair_averaging_chunk,
    msa_module_forward,
    outer_product_mean_forward,
    pair_weighted_averaging_forward,
)

CHECKPOINT = Path(__file__).resolve().parents[4] / "boltz/.cache/boltz/boltz2_conf.ckpt"
BOLTZ_SRC = Path(__file__).resolve().parents[4] / "boltz/src"
PREFIX = "msa_module"


@pytest.fixture(scope="module")
def checkpoint_state() -> dict[str, torch.Tensor]:
    if not CHECKPOINT.exists():
        pytest.skip(f"Boltz-2 checkpoint not found: {CHECKPOINT}")
    return load_checkpoint_state_dict(CHECKPOINT)


def test_checkpoint_msa_module_matches_boltz_torch(
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    num_layers = 2
    torch_module = _load_torch_msa_module(checkpoint_state, num_layers)
    params = map_msa_module_state_dict(checkpoint_state, PREFIX, num_layers=num_layers)
    z, emb, feats = _msa_inputs()

    with torch.no_grad():
        expected = torch_module(z, emb, feats, use_kernels=False)
    actual = msa_module_forward(
        params,
        jnp.asarray(z.numpy()),
        jnp.asarray(emb.numpy()),
        _jax_feats(feats),
    )

    np.testing.assert_allclose(
        np.asarray(actual),
        expected.detach().numpy(),
        rtol=2e-3,
        atol=2e-3,
    )


def test_masked_msa_depth_padding_preserves_jax_output(
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    params = map_msa_module_state_dict(checkpoint_state, PREFIX, num_layers=2)
    z, emb, feats_torch = _msa_inputs()
    feats = _jax_feats(feats_torch)
    expected = msa_module_forward(
        params, jnp.asarray(z.numpy()), jnp.asarray(emb.numpy()), feats
    )
    padded = dict(feats)
    for key in ("msa", "has_deletion", "deletion_value", "msa_paired", "msa_mask"):
        padded[key] = jnp.pad(feats[key], ((0, 0), (0, 5), (0, 0)))

    actual = msa_module_forward(
        params, jnp.asarray(z.numpy()), jnp.asarray(emb.numpy()), padded
    )

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=2e-5)


def test_msa_subsample_without_a_key_takes_the_first_rows(
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    """`msa_key=None` is plain truncation, which is what it must stay."""

    params = map_msa_module_state_dict(checkpoint_state, PREFIX, num_layers=2)
    z, emb, feats_torch = _msa_inputs()
    feats = _jax_feats(feats_torch)
    z_jax, emb_jax = jnp.asarray(z.numpy()), jnp.asarray(emb.numpy())

    actual = msa_module_forward(
        params, z_jax, emb_jax, feats, subsample_msa=True, num_subsampled_msa=2
    )
    truncated = dict(feats)
    for key in ("msa", "has_deletion", "deletion_value", "msa_paired", "msa_mask"):
        truncated[key] = feats[key][:, :2]
    expected = msa_module_forward(params, z_jax, emb_jax, truncated)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=2e-5)


def test_msa_subsample_with_a_key_draws_a_different_slice_per_key(
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    """Upstream re-draws its 1,024 rows inside every forward pass.

    `boltz predict` overrides its own `MSAModuleArgs` default to
    `subsample_msa=True`, and `MSAModule.forward` then calls
    `torch.randperm(n_msa)[:1024]` -- a *different* slice each pass, so a
    recycled trunk ensembles over the whole alignment. Taking the top rows
    every pass instead shows the model the same 1,024 sequences eleven times,
    and on a 2,372-row alignment that cost ~0.13 of complex_plddt against
    upstream. Two keys must therefore disagree, and neither may equal
    truncation.
    """

    params = map_msa_module_state_dict(checkpoint_state, PREFIX, num_layers=2)
    # The module sums over MSA rows, so only the *set* of rows matters. With a
    # handful of rows two independent draws collide often enough to make this
    # test flaky, so give it enough to choose from.
    z, emb, feats_torch = _msa_inputs(msa_rows=24)
    feats = _jax_feats(feats_torch)
    z_jax, emb_jax = jnp.asarray(z.numpy()), jnp.asarray(emb.numpy())
    kwargs = dict(subsample_msa=True, num_subsampled_msa=8)

    left = msa_module_forward(
        params, z_jax, emb_jax, feats, msa_key=jax.random.PRNGKey(0), **kwargs
    )
    right = msa_module_forward(
        params, z_jax, emb_jax, feats, msa_key=jax.random.PRNGKey(7), **kwargs
    )
    first_rows = msa_module_forward(params, z_jax, emb_jax, feats, **kwargs)

    assert not np.allclose(np.asarray(left), np.asarray(right), atol=1e-6)
    assert not np.allclose(np.asarray(left), np.asarray(first_rows), atol=1e-6)


def test_msa_subsample_with_explicit_rows_replays_that_exact_gather(
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    """`msa_rows` is the matched-tape injection point, so it must be exact.

    The parity harness records upstream's `torch.randperm(n_msa)[:k]` and hands
    the indices back here. If this path selected anything but those rows the
    two trunks would read different alignments, and no amount of matched
    sampler noise could then make their coordinates agree -- the comparison
    would silently measure the wrong thing. It also has to beat `msa_key`,
    which the production path always supplies.
    """

    params = map_msa_module_state_dict(checkpoint_state, PREFIX, num_layers=2)
    z, emb, feats_torch = _msa_inputs(msa_rows=24)
    feats = _jax_feats(feats_torch)
    z_jax, emb_jax = jnp.asarray(z.numpy()), jnp.asarray(emb.numpy())
    rows = jnp.asarray([19, 3, 11, 0, 23, 7, 15, 2], dtype=jnp.int32)

    actual = msa_module_forward(
        params,
        z_jax,
        emb_jax,
        feats,
        subsample_msa=True,
        num_subsampled_msa=int(rows.size),
        msa_key=jax.random.PRNGKey(0),
        msa_rows=rows,
    )
    gathered = dict(feats)
    for key in ("msa", "has_deletion", "deletion_value", "msa_paired", "msa_mask"):
        gathered[key] = jnp.take(feats[key], rows, axis=1)
    expected = msa_module_forward(params, z_jax, emb_jax, gathered)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=2e-5)


def test_msa_rows_of_the_wrong_length_is_rejected(
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    """A short row list would silently shrink the alignment; refuse it."""

    params = map_msa_module_state_dict(checkpoint_state, PREFIX, num_layers=2)
    z, emb, feats_torch = _msa_inputs(msa_rows=24)
    feats = _jax_feats(feats_torch)

    with pytest.raises(ValueError, match="msa_rows"):
        msa_module_forward(
            params,
            jnp.asarray(z.numpy()),
            jnp.asarray(emb.numpy()),
            feats,
            subsample_msa=True,
            num_subsampled_msa=8,
            msa_rows=jnp.arange(4, dtype=jnp.int32),
        )


def test_pair_weighted_averaging_preserves_bfloat16_activation_dtype() -> None:
    rng = np.random.default_rng(0)
    c_m, c_z, heads, c_h = 8, 12, 4, 3

    def w(*shape):
        return jnp.asarray(rng.standard_normal(shape) * 0.1, dtype=jnp.bfloat16)

    params = {
        "norm_m": {"scale": w(c_m), "bias": w(c_m)},
        "norm_z": {"scale": w(c_z), "bias": w(c_z)},
        "proj_m": {"kernel": w(c_m, heads * c_h)},
        "proj_z": {"kernel": w(c_z, heads)},
        "proj_g": {"kernel": w(c_m, heads * c_h)},
        "proj_o": {"kernel": w(heads * c_h, c_m)},
    }
    m = jnp.asarray(rng.standard_normal((1, 2, 5, c_m)), dtype=jnp.bfloat16)
    z = jnp.asarray(rng.standard_normal((1, 5, 5, c_z)), dtype=jnp.bfloat16)
    mask = jnp.ones((1, 5, 5), dtype=jnp.bfloat16)

    out = pair_weighted_averaging_forward(params, m, z, mask)

    assert out.dtype == jnp.bfloat16


def _pair_averaging_case(n_msa: int, n_token: int, c_m: int = 8):
    """Small PairWeightedAveraging inputs with a fixed, nonzero mask."""
    rng = np.random.default_rng(7)
    c_z, heads, c_h = 12, 4, 3

    def w(*shape):
        return jnp.asarray(rng.standard_normal(shape) * 0.1, dtype=jnp.float32)

    params = {
        "norm_m": {"scale": w(c_m), "bias": w(c_m)},
        "norm_z": {"scale": w(c_z), "bias": w(c_z)},
        "proj_m": {"kernel": w(c_m, heads * c_h)},
        "proj_z": {"kernel": w(c_z, heads)},
        "proj_g": {"kernel": w(c_m, heads * c_h)},
        "proj_o": {"kernel": w(heads * c_h, c_m)},
    }
    m = jnp.asarray(rng.standard_normal((1, n_msa, n_token, c_m)), dtype=jnp.float32)
    z = jnp.asarray(rng.standard_normal((1, n_token, n_token, c_z)), dtype=jnp.float32)
    mask = jnp.ones((1, n_token, n_token), dtype=jnp.float32)
    return params, m, z, mask


@pytest.mark.parametrize("row_chunk_size", [1, 2, 3, 5])
def test_pair_weighted_averaging_row_chunk_matches_whole_tensor(
    row_chunk_size: int,
) -> None:
    """Splitting the MSA axis must not change what the block computes.

    Rows of the MSA are a pure batch axis here, so a block boundary splits no
    reduction: every chunk size, including ones that do not divide the depth,
    has to reproduce the single-op result. Without this the chunk could silently
    drop or double-count the ragged final block.
    """
    params, m, z, mask = _pair_averaging_case(n_msa=7, n_token=5)

    whole = pair_weighted_averaging_forward(params, m, z, mask, row_chunk_size=0)
    chunked = pair_weighted_averaging_forward(
        params, m, z, mask, row_chunk_size=row_chunk_size
    )

    assert chunked.shape == whole.shape
    np.testing.assert_allclose(chunked, whole, rtol=1e-6, atol=1e-6)


def test_pair_averaging_chunk_leaves_small_alignments_on_the_single_op_path() -> None:
    """The budget must not engage where the port already matched upstream.

    A chunk is exact but not free: XLA tiles the smaller GEMMs differently, and
    that moved the matched-tape RMSD when the 132-token benchmark job split.
    That job's widened form is 132 * 256 * 4 bytes a row over 2,371 rows, and it
    has to stay under the budget -- which is also what upstream runs there,
    since 132 is below the 384-token threshold that turns on its own chunking.
    """
    params, m, _, _ = _pair_averaging_case(n_msa=2371, n_token=132, c_m=64)
    params["proj_m"] = {"kernel": jnp.zeros((64, 256), jnp.float32)}

    assert 2371 * 132 * 256 * 4 < _PWA_BUDGET_BYTES
    assert _auto_pair_averaging_chunk(m, params) is None

    # Not just "the chunk is off" but "the program is the one that shipped":
    # lowering with the auto default has to produce the same HLO, character for
    # character, as forcing the single-op path.
    shapes = (
        jax.ShapeDtypeStruct((1, 2371, 132, 64), jnp.float32),
        jax.ShapeDtypeStruct((1, 132, 132, 12), jnp.float32),
        jax.ShapeDtypeStruct((1, 132, 132), jnp.float32),
    )
    params = {**params, "norm_m": {"scale": jnp.zeros(64), "bias": jnp.zeros(64)}}
    params["proj_g"] = {"kernel": jnp.zeros((64, 256), jnp.float32)}
    params["proj_o"] = {"kernel": jnp.zeros((256, 64), jnp.float32)}

    def lower(chunk):
        return (
            jax.jit(
                pair_weighted_averaging_forward, static_argnames=("row_chunk_size",)
            )
            .lower(params, *shapes, row_chunk_size=chunk)
            .as_text()
        )

    assert lower(None) == lower(0)


def test_pair_averaging_chunk_splits_a_deep_alignment_under_the_budget() -> None:
    """Above the budget the block must be small enough to have been worth it."""
    params, _, _, _ = _pair_averaging_case(n_msa=2, n_token=2, c_m=64)
    params["proj_m"] = {"kernel": jnp.zeros((64, 256), jnp.float32)}
    # The 490-token benchmark job's real MSA, as a shape rather than 3.7 GiB of
    # zeros: the block size is decided from shapes alone.
    m = jax.ShapeDtypeStruct((1, 7917, 490, 64), jnp.float32)

    rows = _auto_pair_averaging_chunk(m, params)

    assert rows is not None
    assert 0 < rows < 7917
    assert rows * 490 * 256 * 4 <= _PWA_BUDGET_BYTES


def _outer_product_case(n_msa: int, n_token: int, dtype=jnp.float32):
    """Small OuterProductMean inputs with a fixed, partly-zero mask."""
    rng = np.random.default_rng(11)
    c_m, c_hidden, c_z = 64, 32, 128

    def w(*shape, scale=0.1):
        return jnp.asarray(rng.standard_normal(shape) * scale, dtype=dtype)

    params = {
        "norm": {"scale": w(c_m, scale=1.0), "bias": w(c_m)},
        "proj_a": {"kernel": w(c_m, c_hidden)},
        "proj_b": {"kernel": w(c_m, c_hidden)},
        "proj_o": {"kernel": w(c_hidden * c_hidden, c_z), "bias": w(c_z)},
    }
    m = w(1, n_msa, n_token, c_m, scale=1.0)
    mask = jnp.asarray(
        (rng.random((1, n_msa, n_token)) > 0.2).astype(np.float32), dtype=dtype
    )
    return params, m, mask


def _shipped_outer_product_mean(params, m, mask, eps=1e-5, chunk_size=128):
    """OuterProductMean as it stood before the float32 operands came out.

    Kept here rather than described, because the claim under test is that the
    rework computes the same numbers -- and the only way to check that is to
    still have the thing it replaced.
    """
    from foldjax.models.boltz2.models.primitives._common import layer_norm, linear

    mask = mask[..., None].astype(m.dtype)
    m = layer_norm(m, params["norm"]["scale"], params["norm"]["bias"], eps)
    a = (linear(m, params["proj_a"]["kernel"]) * mask).astype(jnp.float32)
    b = (linear(m, params["proj_b"]["kernel"]) * mask).astype(jnp.float32)
    pair_mask = mask[:, :, None, :] * mask[:, :, :, None]
    num_mask = jnp.maximum(jnp.sum(pair_mask, axis=1), 1.0)

    proj_o = params["proj_o"]
    n = a.shape[2]
    out = jnp.zeros((a.shape[0], n, n, proj_o["kernel"].shape[-1]), dtype=m.dtype)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        z = jnp.einsum("bsic,bsjd->bijcd", a[:, :, start:end], b)
        z = jnp.reshape(z, (*z.shape[:3], -1)) / num_mask[:, start:end]
        out = out.at[:, start:end].set(
            linear(z.astype(m.dtype), proj_o["kernel"], proj_o["bias"])
        )
    return out


def test_outer_product_mean_matches_the_widened_operand_form() -> None:
    """Feeding the float32 accumulator narrow operands is the same arithmetic.

    Upstream writes ``torch.einsum(..., a.float(), b.float())``. Both operands
    come out of a linear in the compute dtype, so widening them changes no
    value that enters the product -- only whether a second, wider copy of each
    is materialised, which at 8,192 alignment rows is f32[8192,32096], 1,003
    MiB. Under the float32 default there is nothing to widen and the result has
    to be identical bit for bit.
    """
    params, m, mask = _outer_product_case(n_msa=37, n_token=132)

    reworked = outer_product_mean_forward(params, m, mask)
    shipped = _shipped_outer_product_mean(params, m, mask)

    assert reworked.dtype == shipped.dtype == jnp.float32
    assert jnp.array_equal(reworked, shipped)


def test_outer_product_mean_num_mask_contraction_is_exact() -> None:
    """The pair count is a contraction, not a [b, s, i, j] broadcast product.

    Written as a broadcast the intermediate is one float per (row, token,
    token): 16.5 GiB of elementwise work at 8,192 rows, fused away by XLA but
    still computed. The mask is 0/1, so summing it as a dot in float32 is exact
    at any depth this model can be given -- which is the only reason the
    rewrite is allowed to be a rewrite rather than an approximation.
    """
    for depth in (7, 512, 8192):
        _, _, mask = _outer_product_case(n_msa=depth, n_token=9)
        broadcast = jnp.sum(
            mask[..., None][:, :, None, :] * mask[..., None][:, :, :, None], axis=1
        )
        contracted = jnp.einsum(
            "bsi,bsj->bij", mask, mask, preferred_element_type=jnp.float32
        )[..., None]

        assert jnp.array_equal(broadcast, contracted)


def test_outer_product_chunk_leaves_the_parity_job_on_its_own_program() -> None:
    """The budget must only ever tighten the chunk the caller asked for.

    At 132 tokens the widened float32 product is 132 * 1024 * 4 bytes a row,
    so the budget would allow far more rows than the 128 the trunk passes --
    and raising it there would take the matched-tape job off the two-block
    program it already runs. Below the budget the lowered HLO has to be the one
    that shipped, character for character.
    """
    assert _auto_outer_product_chunk(132, 32 * 32, 128) == 128

    params, m, mask = _outer_product_case(n_msa=37, n_token=132)

    def lower(chunk_size):
        return (
            jax.jit(outer_product_mean_forward, static_argnames=("chunk_size",))
            .lower(params, m, mask, chunk_size=chunk_size)
            .as_text()
        )

    assert lower(128) == lower(_auto_outer_product_chunk(132, 32 * 32, 128))


def test_outer_product_chunk_splits_a_long_sequence_under_the_budget() -> None:
    """Above the budget the block has to be small enough to have been worth it."""
    for tokens in (1003, 3012):
        chunk = _auto_outer_product_chunk(tokens, 32 * 32, 128)

        assert 0 < chunk < 128
        assert chunk * tokens * 32 * 32 * 4 <= _OPM_BUDGET_BYTES


def test_outer_product_chunking_changes_nothing_it_should_not() -> None:
    """Row blocks are independent: no sum is split, so the values must agree."""
    params, m, mask = _outer_product_case(n_msa=13, n_token=64)

    whole = outer_product_mean_forward(params, m, mask, chunk_size=64)
    for chunk_size in (1, 7, 16, 63):
        blocked = outer_product_mean_forward(params, m, mask, chunk_size=chunk_size)
        np.testing.assert_allclose(blocked, whole, rtol=1e-6, atol=1e-6)


def _load_torch_msa_module(
    state: dict[str, torch.Tensor],
    num_layers: int,
) -> torch.nn.Module:
    sys.path.insert(0, str(BOLTZ_SRC))
    from boltz.model.modules.trunkv2 import MSAModule

    module = MSAModule(
        msa_s=64,
        token_z=128,
        token_s=384,
        msa_blocks=num_layers,
        msa_dropout=0.15,
        z_dropout=0.25,
        pairwise_head_width=32,
        pairwise_num_heads=4,
        use_paired_feature=True,
    ).eval()
    module_state = {}
    for key, value in state.items():
        if not key.startswith(f"{PREFIX}."):
            continue
        local_key = key.removeprefix(f"{PREFIX}.")
        if local_key.startswith("layers."):
            index = int(local_key.split(".")[1])
            if index >= num_layers:
                continue
        module_state[local_key] = value
    module.load_state_dict(module_state)
    return module


def _msa_inputs(
    msa_rows: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    tokens = 4
    z = torch.linspace(-0.1, 0.1, steps=tokens * tokens * 128).reshape(
        1, tokens, tokens, 128
    )
    emb = torch.linspace(0.2, -0.2, steps=tokens * 384).reshape(1, tokens, 384)
    feats = {
        "msa": (torch.arange(msa_rows * tokens).reshape(1, msa_rows, tokens) % 33),
        "has_deletion": torch.zeros(1, msa_rows, tokens),
        "deletion_value": torch.linspace(0.0, 1.0, steps=msa_rows * tokens).reshape(
            1, msa_rows, tokens
        ),
        "msa_paired": torch.ones(1, msa_rows, tokens),
        "msa_mask": torch.ones(1, msa_rows, tokens),
        "token_pad_mask": torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
    }
    feats["msa_mask"][:, -1, -1] = 0.0
    return z, emb, feats


def _jax_feats(feats: dict[str, torch.Tensor]) -> dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value.numpy()) for key, value in feats.items()}
