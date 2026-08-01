import sys
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from foldjax.models.boltz2.bridge.torch_checkpoint import load_checkpoint_state_dict
from foldjax.models.boltz2.bridge.torch_mapping import map_atom_transformer_state_dict
from foldjax.models.boltz2.models.diffusion.atom import (
    atom_transformer_forward,
    gather_rep_atoms_to_tokens,
    gather_token_pairs_to_atom_windows,
    gather_tokens_to_atoms,
    get_indexing_matrix,
    scatter_atoms_to_tokens_mean,
    single_to_keys,
)

CHECKPOINT = (
    Path(__file__).resolve().parents[4] / "boltz/.cache/boltz/boltz2_conf.ckpt"
)
BOLTZ_SRC = Path(__file__).resolve().parents[4] / "boltz/src"
PREFIX = "structure_module.score_model.atom_attention_encoder.atom_encoder"


@pytest.fixture(scope="module")
def checkpoint_state() -> dict[str, torch.Tensor]:
    if not CHECKPOINT.exists():
        pytest.skip(f"Boltz-2 checkpoint not found: {CHECKPOINT}")
    return load_checkpoint_state_dict(CHECKPOINT)


def test_atom_window_indexing_matches_boltz_torch() -> None:
    if not BOLTZ_SRC.is_dir():
        pytest.skip("upstream Boltz source is not available")
    sys.path.insert(0, str(BOLTZ_SRC))
    from boltz.model.modules.encodersv2 import get_indexing_matrix as torch_index

    actual = get_indexing_matrix(k=2, w=32, h_keys=128)
    expected = torch_index(K=2, W=32, H=128, device="cpu")

    np.testing.assert_array_equal(np.asarray(actual), expected.numpy())


def test_single_to_keys_matches_boltz_torch() -> None:
    if not BOLTZ_SRC.is_dir():
        pytest.skip("upstream Boltz source is not available")
    sys.path.insert(0, str(BOLTZ_SRC))
    from boltz.model.modules.encodersv2 import (
        get_indexing_matrix as torch_index,
    )
    from boltz.model.modules.encodersv2 import (
        single_to_keys as torch_to_keys,
    )

    single = torch.arange(1 * 64 * 5, dtype=torch.float32).reshape(1, 64, 5)
    expected = torch_to_keys(
        single,
        indexing_matrix=torch_index(K=2, W=32, H=128, device="cpu"),
        W=32,
        H=128,
    )

    indexing = get_indexing_matrix(k=2, w=32, h_keys=128)
    actual = single_to_keys(
        jnp.asarray(single.numpy()), indexing, w=32, h_keys=128
    )

    np.testing.assert_array_equal(np.asarray(actual), expected.numpy())


def test_single_to_keys_preserves_zero_key_columns() -> None:
    single = jnp.arange(1 * 64 * 3, dtype=jnp.float32).reshape(1, 64, 3)
    indexing = jnp.zeros((4, 8), dtype=jnp.float32)
    indexing = indexing.at[2, 0].set(1.0)
    indexing = indexing.at[1, 3].set(1.0)

    actual = np.asarray(single_to_keys(single, indexing, w=32, h_keys=64))
    assert np.all(actual[:, 0, 16:48] == 0)
    np.testing.assert_array_equal(actual[:, 0, 0:16], np.asarray(single[:, 32:48]))
    np.testing.assert_array_equal(actual[:, 0, 48:64], np.asarray(single[:, 16:32]))


def test_dense_atom_token_helpers_match_matmul_reference() -> None:
    atom_to_token = np.zeros((2, 5, 3), dtype=np.float32)
    tok_idx = np.asarray([[0, 1, 1, 2, 0], [2, 2, 1, 0, 0]])
    for b in range(atom_to_token.shape[0]):
        atom_to_token[b, np.arange(atom_to_token.shape[1]), tok_idx[b]] = 1.0
    atom_to_token[0, 4] = 0.0

    token_values = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    atom_values = np.arange(2 * 5 * 4, dtype=np.float32).reshape(2, 5, 4)

    gathered = gather_tokens_to_atoms(
        jnp.asarray(atom_to_token), jnp.asarray(token_values)
    )
    expected_gather = np.einsum("bat,btd->bad", atom_to_token, token_values)
    np.testing.assert_array_equal(np.asarray(gathered), expected_gather)

    scattered = scatter_atoms_to_tokens_mean(
        jnp.asarray(atom_to_token), jnp.asarray(atom_values)
    )
    denom = atom_to_token.sum(axis=1, keepdims=True) + 1e-6
    expected_scatter = np.einsum("bat,bad->btd", atom_to_token / denom, atom_values)
    np.testing.assert_allclose(np.asarray(scattered), expected_scatter, atol=1e-6)


def test_token_pair_window_gather_matches_dense_einsum() -> None:
    batch, windows, q_atoms, k_atoms, tokens, dim = 2, 2, 3, 4, 5, 6
    rng = np.random.default_rng(0)
    z = rng.standard_normal((batch, tokens, tokens, dim)).astype(np.float32)
    q_idx = rng.integers(0, tokens, size=(batch, windows, q_atoms))
    k_idx = rng.integers(0, tokens, size=(batch, windows, k_atoms))
    q_map = np.zeros((batch, windows, q_atoms, tokens), dtype=np.float32)
    k_map = np.zeros((batch, windows, k_atoms, tokens), dtype=np.float32)
    for b in range(batch):
        for w in range(windows):
            q_map[b, w, np.arange(q_atoms), q_idx[b, w]] = 1.0
            k_map[b, w, np.arange(k_atoms), k_idx[b, w]] = 1.0
    q_map[1, 0, 2] = 0.0

    actual = gather_token_pairs_to_atom_windows(
        jnp.asarray(z), jnp.asarray(q_map), jnp.asarray(k_map)
    )
    expected = np.einsum("bijd,bwki,bwlj->bwkld", z, q_map, k_map)
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_rep_atom_gather_matches_matmul_reference() -> None:
    token_to_rep_atom = np.zeros((1, 4, 6), dtype=np.float32)
    rep_idx = np.asarray([1, 3, 5, 0])
    token_to_rep_atom[0, np.arange(4), rep_idx] = 1.0
    token_to_rep_atom[0, 3] = 0.0
    atom_values = np.arange(1 * 6 * 3, dtype=np.float32).reshape(1, 6, 3)

    actual = gather_rep_atoms_to_tokens(
        jnp.asarray(token_to_rep_atom), jnp.asarray(atom_values)
    )
    expected = token_to_rep_atom @ atom_values
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_checkpoint_atom_transformer_matches_boltz_torch(
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    torch_module = _load_torch_atom_transformer(checkpoint_state)
    params = map_atom_transformer_state_dict(
        checkpoint_state,
        PREFIX,
        num_heads=4,
    )
    q, c, bias, mask = _atom_transformer_inputs()

    sys.path.insert(0, str(BOLTZ_SRC))
    from boltz.model.modules.encodersv2 import (
        get_indexing_matrix as torch_index,
    )
    from boltz.model.modules.encodersv2 import (
        single_to_keys as torch_to_keys,
    )

    torch_keys = torch_index(K=2, W=32, H=128, device="cpu")
    torch_to_keys_fn = partial(torch_to_keys, indexing_matrix=torch_keys, W=32, H=128)
    jax_keys = get_indexing_matrix(k=2, w=32, h_keys=128)

    with torch.no_grad():
        expected = torch_module(
            q=q,
            c=c,
            bias=bias,
            to_keys=torch_to_keys_fn,
            mask=mask,
            multiplicity=1,
        )
    actual = atom_transformer_forward(
        params,
        q=jnp.asarray(q.numpy()),
        c=jnp.asarray(c.numpy()),
        bias=jnp.asarray(bias.numpy()),
        to_keys=lambda x: single_to_keys(x, jax_keys, w=32, h_keys=128),
        mask=jnp.asarray(mask.numpy()),
        attn_window_queries=32,
        attn_window_keys=128,
        multiplicity=1,
    )

    np.testing.assert_allclose(
        np.asarray(actual),
        expected.detach().numpy(),
        rtol=2e-3,
        atol=2e-3,
    )


def _load_torch_atom_transformer(
    state: dict[str, torch.Tensor],
) -> torch.nn.Module:
    sys.path.insert(0, str(BOLTZ_SRC))
    from boltz.model.modules.transformersv2 import AtomTransformer

    module = AtomTransformer(
        attn_window_queries=32,
        attn_window_keys=128,
        depth=3,
        heads=4,
        dim=128,
        dim_single_cond=128,
        post_layer_norm=False,
    ).eval()
    module_state = {
        key.removeprefix(f"{PREFIX}."): value
        for key, value in state.items()
        if key.startswith(f"{PREFIX}.")
    }
    module.load_state_dict(module_state)
    return module


def _atom_transformer_inputs() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    atoms = 64
    q = torch.linspace(-0.2, 0.2, steps=atoms * 128).reshape(1, atoms, 128)
    c = torch.linspace(0.25, -0.25, steps=atoms * 128).reshape(1, atoms, 128)
    bias = torch.linspace(-0.1, 0.1, steps=2 * 32 * 128 * 12)
    bias = bias.reshape(1, 2, 32, 128, 12)
    mask = torch.ones(1, atoms, dtype=torch.float32)
    mask[:, -3:] = 0.0
    return q, c, bias, mask
