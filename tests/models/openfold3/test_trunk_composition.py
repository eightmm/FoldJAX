"""The trunk's recycling composition.

Every sub-module is gated against upstream individually. This checks the wiring:
the recycling projections, the number of cycles actually mattering, and the mask
construction.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.models.trunk import trunk

pytestmark = pytest.mark.torch_parity

C_ATOM, C_ATOM_PAIR, C_TOKEN = 8, 6, 8
C_S, C_Z = 8, 4
C_ELEMENT, C_CHARS = 12, 5
C_RESTYPE, C_PROFILE, C_MSA = 32, 32, 32
N_QUERY, N_KEY, HEADS = 4, 8, 2
MAX_IDX, MAX_CHAIN = 4, 2
COUNTS = [4, 3, 5]
N_ATOM, N_TOKEN, N_MSA = sum(COUNTS), len(COUNTS), 3
C_S_INPUT = C_TOKEN + C_RESTYPE + C_PROFILE + 1


def _torch():
    import torch

    torch.manual_seed(0)
    return torch


def _params(torch, randomized):
    """Assemble trunk params from individually gated sub-modules."""
    from ml_collections import ConfigDict
    from openfold3.core.model.feature_embedders.input_embedders import (
        InputEmbedderAllAtom,
        MSAModuleEmbedder,
    )
    from openfold3.core.model.latent.msa_module import MSAModuleStack
    from openfold3.core.model.latent.pairformer import PairFormerStack
    from openfold3.core.model.primitives import LayerNorm, Linear

    from foldjax.models.openfold3.bridge.torch_mapping import (
        map_input_embedder,
        map_layer_norm,
        map_linear,
        map_msa_embedder,
        map_msa_module_stack,
        map_pairformer_stack,
    )
    from foldjax.models.openfold3.models.trunk import TrunkParams

    enc_cfg = {
        "c_atom_ref": ConfigDict({"element": C_ELEMENT, "name_chars": 4 * C_CHARS}),
        "c_atom": C_ATOM,
        "c_atom_pair": C_ATOM_PAIR,
        "c_token": C_TOKEN,
        "c_hidden": 4,
        "no_heads": HEADS,
        "no_blocks": 1,
        "n_transition": 2,
        "n_query": N_QUERY,
        "n_key": N_KEY,
        "use_ada_layer_norm": False,
    }
    embedder = randomized(
        InputEmbedderAllAtom(
            c_s_input=C_S_INPUT,
            c_s=C_S,
            c_z=C_Z,
            max_relative_idx=MAX_IDX,
            max_relative_chain=MAX_CHAIN,
            atom_attn_enc=enc_cfg,
        )
    )
    msa_emb = randomized(
        MSAModuleEmbedder(
            c_m_feats=C_MSA + 2,
            c_m=C_ATOM,
            c_s_input=C_S_INPUT,
            subsample_main_msa=False,
            subsample_all_msa=False,
            min_subsampled_all_msa=1,
            max_subsampled_all_msa=N_MSA,
        )
    )
    msa_stack = randomized(
        MSAModuleStack(
            c_m=C_ATOM, c_z=C_Z, c_hidden_msa_att=4, c_hidden_opm=4,
            c_hidden_mul=6, c_hidden_pair_att=4, no_heads_msa=HEADS,
            no_heads_pair=HEADS, no_blocks=1, transition_type="swiglu",
            transition_n=2, msa_dropout=0.0, pair_dropout=0.0, opm_first=True,
            fuse_projection_weights=False, blocks_per_ckpt=None, inf=1e9, eps=1e-3,
        )
    )
    pf = randomized(
        PairFormerStack(
            c_s=C_S, c_z=C_Z, c_hidden_pair_bias=4, no_heads_pair_bias=HEADS,
            c_hidden_mul=6, c_hidden_pair_att=4, no_heads_pair=HEADS, no_blocks=1,
            transition_type="swiglu", transition_n=2, pair_dropout=0.0,
            fuse_projection_weights=False, blocks_per_ckpt=None, inf=1e9,
        )
    )
    return TrunkParams(
        input_embedder=map_input_embedder(dict(embedder.state_dict())),
        msa_module_embedder=map_msa_embedder(dict(msa_emb.state_dict())),
        msa_module=map_msa_module_stack(dict(msa_stack.state_dict())),
        pairformer_stack=map_pairformer_stack(dict(pf.state_dict())),
        layer_norm_z=map_layer_norm(dict(randomized(LayerNorm(C_Z)).state_dict())),
        linear_z=map_linear(
            dict(randomized(Linear(C_Z, C_Z, bias=False)).state_dict()), bias=False
        ),
        layer_norm_s=map_layer_norm(dict(randomized(LayerNorm(C_S)).state_dict())),
        linear_s=map_linear(
            dict(randomized(Linear(C_S, C_S, bias=False)).state_dict()), bias=False
        ),
    )


def _batch(torch) -> dict:
    atom_to_token = torch.cat(
        [torch.full((c,), i, dtype=torch.long) for i, c in enumerate(COUNTS)]
    )
    raw = {
        "ref_pos": torch.randn(1, N_ATOM, 3),
        "ref_charge": torch.randn(1, N_ATOM),
        "ref_mask": torch.ones(1, N_ATOM),
        "ref_element": torch.rand(1, N_ATOM, C_ELEMENT),
        "ref_atom_name_chars": torch.rand(1, N_ATOM, 4, C_CHARS),
        "ref_space_uid": (torch.arange(N_ATOM).float() // 4).reshape(1, N_ATOM),
        "atom_mask": torch.ones(1, N_ATOM),
        "atom_to_token_index": atom_to_token.reshape(1, N_ATOM),
        "token_mask": torch.ones(1, N_TOKEN),
        "num_atoms_per_token": torch.tensor([COUNTS], dtype=torch.float32),
        "restype": torch.rand(1, N_TOKEN, C_RESTYPE),
        "profile": torch.rand(1, N_TOKEN, C_PROFILE),
        "deletion_mean": torch.rand(1, N_TOKEN),
        "token_bonds": torch.zeros(1, N_TOKEN, N_TOKEN),
        "residue_index": torch.arange(N_TOKEN).reshape(1, N_TOKEN),
        "asym_id": torch.tensor([[0, 0, 1]]),
        "entity_id": torch.tensor([[0, 0, 0]]),
        "token_index": torch.arange(N_TOKEN).reshape(1, N_TOKEN),
        "sym_id": torch.tensor([[0, 0, 1]]),
        "msa": torch.rand(1, N_MSA, N_TOKEN, C_MSA),
        "has_deletion": torch.zeros(1, N_MSA, N_TOKEN),
        "deletion_value": torch.rand(1, N_MSA, N_TOKEN),
        "msa_mask": torch.ones(1, N_MSA, N_TOKEN),
    }
    return {k: jnp.asarray(v.numpy()) for k, v in raw.items()}


def _kwargs(num_cycles: int) -> dict:
    return {
        "num_cycles": num_cycles,
        "n_query": N_QUERY,
        "n_key": N_KEY,
        "atom_heads": HEADS,
        "n_token": N_TOKEN,
        "max_relative_idx": MAX_IDX,
        "max_relative_chain": MAX_CHAIN,
        "no_heads_msa": HEADS,
        "no_heads_pair": HEADS,
        "no_heads_pair_bias": HEADS,
    }


def test_trunk_runs_and_shapes_are_right(openfold3_source, randomized) -> None:
    torch = _torch()
    params = _params(torch, randomized)
    batch = _batch(torch)
    s_input, s, z = trunk(batch, params, **_kwargs(1))
    assert s_input.shape == (1, N_TOKEN, C_S_INPUT)
    assert s.shape == (1, N_TOKEN, C_S)
    assert z.shape == (1, N_TOKEN, N_TOKEN, C_Z)
    for array in (s_input, s, z):
        assert np.isfinite(np.asarray(array)).all()


def test_more_cycles_change_the_result(openfold3_source, randomized) -> None:
    """Recycling must actually feed back; otherwise cycles would be no-ops."""
    torch = _torch()
    params = _params(torch, randomized)
    batch = _batch(torch)
    _si1, s1, z1 = trunk(batch, params, **_kwargs(1))
    _si2, s2, z2 = trunk(batch, params, **_kwargs(2))
    assert not np.allclose(np.asarray(s1), np.asarray(s2), rtol=1e-4)
    assert not np.allclose(np.asarray(z1), np.asarray(z2), rtol=1e-4)


def test_s_input_is_cycle_independent(openfold3_source, randomized) -> None:
    """s_input comes from the input embedder, which runs once before the loop."""
    torch = _torch()
    params = _params(torch, randomized)
    batch = _batch(torch)
    first, _s, _z = trunk(batch, params, **_kwargs(1))
    second, _s2, _z2 = trunk(batch, params, **_kwargs(3))
    np.testing.assert_allclose(
        np.asarray(first), np.asarray(second), rtol=1e-6, atol=1e-6
    )


def test_rejects_zero_cycles(openfold3_source, randomized) -> None:
    torch = _torch()
    params = _params(torch, randomized)
    with pytest.raises(ValueError, match="at least 1"):
        trunk(_batch(torch), params, **_kwargs(0))
