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
    msa_module_forward,
    pair_weighted_averaging_forward,
)

CHECKPOINT = (
    Path(__file__).resolve().parents[4] / "boltz/.cache/boltz/boltz2_conf.ckpt"
)
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
