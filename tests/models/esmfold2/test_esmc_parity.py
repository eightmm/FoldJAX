"""ESMC against torch: the stack, and the chain packing ESMFold2 wraps it in.

Random weights are enough here in a way they would not be for the trunk: what
can be wrong in a plain transformer is structural -- a rotary convention, a
normalisation applied per head instead of per model, a residual scale dropped
-- and every one of those shows on any weights at all.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
esmc_modeling = pytest.importorskip("transformers.models.esmc.modeling_esmc")
esmc_config = pytest.importorskip("transformers.models.esmc.configuration_esmc")
common = pytest.importorskip("transformers.models.esmfold2.modeling_esmfold2_common")

from foldjax.models.esmfold2.models import esmc  # noqa: E402
from tests.models.esmfold2.parity import torch_state_to_numpy  # noqa: E402

D_MODEL, N_HEADS, N_LAYERS, VOCAB = 32, 4, 3, 64
N_TOKENS = 7


def _model():
    torch.manual_seed(0)
    config = esmc_config.ESMCConfig(
        vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS
    )
    model = esmc_modeling.ESMCModel(config).eval()
    # The initialiser leaves the fused LN+MLP weights near zero, which would
    # hide a wrong split of `fc1_weight`; give every parameter real spread.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.05)
        for name, parameter in model.named_parameters():
            if name.endswith("layer_norm_weight") or name.endswith("ln.weight"):
                parameter.add_(1.0)
    return model


def _settings():
    return esmc.ESMCSettings(
        vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS
    )


def test_the_rotary_tables_match() -> None:
    module = esmc_modeling.RotaryEmbedding(D_MODEL // N_HEADS)
    module._update_cos_sin_cache(
        N_TOKENS, device=torch.device("cpu"), dtype=torch.float32
    )
    cos, sin = esmc.rotary_tables(N_TOKENS, D_MODEL // N_HEADS)
    np.testing.assert_allclose(np.asarray(cos), module._cos_cached.numpy(), atol=1e-6)
    np.testing.assert_allclose(np.asarray(sin), module._sin_cached.numpy(), atol=1e-6)


def test_the_whole_stack_matches() -> None:
    """Every one of the `n_layers + 1` hidden states, not only the last."""
    model = _model()
    ids = np.random.default_rng(1).integers(4, 30, (2, N_TOKENS)).astype(np.int64)
    sequence_id = np.zeros((2, N_TOKENS), dtype=np.int64)
    sequence_id[1, -2:] = -1

    with torch.no_grad():
        expected = model(
            input_ids=torch.from_numpy(ids),
            sequence_id=torch.from_numpy(sequence_id),
            output_hidden_states=True,
        ).hidden_states

    got = esmc.encode(
        ids, sequence_id, torch_state_to_numpy(model), settings=_settings()
    )
    assert got.shape == expected.shape == (N_LAYERS + 1, 2, N_TOKENS, D_MODEL)
    np.testing.assert_allclose(
        np.asarray(got), expected.numpy(), atol=2e-5, rtol=2e-5
    )


def test_the_chain_packing_matches() -> None:
    """`[BOS] chain [EOS BOS] chain [EOS]`, and the atom-token collapse."""
    ids = np.array([[10, 11, 12, 13, 14, 1]], dtype=np.int64)
    asym = np.array([[0, 0, 0, 1, 1, 1]], dtype=np.int64)
    # Tokens 1 and 2 are one atom-tokenised residue: same chain, same index.
    residues = np.array([[0, 1, 1, 0, 1, 2]], dtype=np.int64)
    mol_type = np.zeros((1, 6), dtype=np.int64)
    token_mask = np.array([[1, 1, 1, 1, 1, 0]], dtype=np.int64)

    packed, sequence_id, expand = esmc.pack_lm_inputs(
        ids, asym, residues, mol_type, token_mask
    )
    np.testing.assert_array_equal(packed[0], [0, 10, 11, 2, 0, 13, 14, 2])
    np.testing.assert_array_equal(sequence_id[0], [0, 0, 0, 0, 1, 1, 1, 1])
    # The two tokens of the collapsed residue read the same LM position, and
    # the masked-out token reads none.
    np.testing.assert_array_equal(expand[0], [1, 2, 2, 5, 6, -1])


def test_the_hidden_states_esmfold2_reads_match() -> None:
    """Packing, the stack, and the scatter back, against upstream's own helper."""
    model = _model()
    rng = np.random.default_rng(2)
    ids = rng.integers(4, 30, (1, N_TOKENS)).astype(np.int64)
    asym = (np.arange(N_TOKENS, dtype=np.int64) // 4)[None]
    residues = np.arange(N_TOKENS, dtype=np.int64)[None]
    mol_type = np.zeros((1, N_TOKENS), dtype=np.int64)
    token_mask = np.ones((1, N_TOKENS), dtype=np.int64)

    with torch.no_grad():
        expected = common.compute_lm_hidden_states(
            model,
            torch.from_numpy(ids),
            torch.from_numpy(asym),
            torch.from_numpy(residues),
            torch.from_numpy(mol_type),
            torch.from_numpy(token_mask).bool(),
        )
    got = esmc.lm_hidden_states(
        ids,
        asym,
        residues,
        mol_type,
        token_mask,
        torch_state_to_numpy(model),
        settings=_settings(),
    )
    assert got.shape == expected.shape == (1, N_TOKENS, N_LAYERS + 1, D_MODEL)
    np.testing.assert_allclose(
        np.asarray(got), expected.numpy(), atol=2e-5, rtol=2e-5
    )
