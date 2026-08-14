"""The confidence head against torch, including the parts that look like bugs."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
common = pytest.importorskip("transformers.models.esmfold2.modeling_esmfold2_common")
modeling = pytest.importorskip("transformers.models.esmfold2.modeling_esmfold2")
configuration = pytest.importorskip(
    "transformers.models.esmfold2.configuration_esmfold2"
)

from foldjax.models.esmfold2.models import heads  # noqa: E402
from tests.models.esmfold2.parity import torch_state_to_numpy  # noqa: E402

D_SINGLE, D_PAIR, D_INPUTS = 24, 16, 20
N_TOKENS, N_ATOMS, N_CHAINS = 6, 18, 2


def _config():
    return configuration.ESMFold2Config(
        d_single=D_SINGLE,
        d_pair=D_PAIR,
        inputs={"d_inputs": D_INPUTS},
        confidence_head={
            "num_plddt_bins": 8,
            "num_pae_bins": 8,
            "num_pde_bins": 8,
            "distogram_bins": 12,
            "folding_trunk": {"n_layers": 1},
        },
    )


def _inputs(seed=0):
    rng = np.random.default_rng(seed)
    atom_to_token = (np.arange(N_ATOMS, dtype=np.int64) // 3)[None]
    return {
        "s_inputs": rng.standard_normal((1, N_TOKENS, D_INPUTS)).astype(np.float32),
        "z": rng.standard_normal((1, N_TOKENS, N_TOKENS, D_PAIR)).astype(np.float32),
        "x_pred": (rng.standard_normal((1, N_ATOMS, 3)) * 8).astype(np.float32),
        "distogram_atom_idx": (np.arange(N_TOKENS, dtype=np.int64) * 3)[None],
        "token_attention_mask": np.ones((1, N_TOKENS), dtype=np.float32),
        "atom_to_token": atom_to_token,
        "atom_attention_mask": np.ones((1, N_ATOMS), dtype=np.float32),
        # Two chains, so the inter-chain terms and the per-chain matrix are
        # exercised rather than divided by their own epsilon.
        "asym_id": (np.arange(N_TOKENS, dtype=np.int64) // 3)[None],
        "mol_type": np.zeros((1, N_TOKENS), dtype=np.int64),
    }


def test_the_confidence_head_matches() -> None:
    module = modeling.ConfidenceHead(_config()).eval()
    module.folding_trunk.set_chunk_size(None)
    params = torch_state_to_numpy(module)
    params["boundaries"] = module.boundaries.numpy()
    data = _inputs(3)

    with torch.no_grad():
        expected = module(
            **{k: torch.from_numpy(v) for k, v in data.items()},
        )
    got = heads.confidence_head(
        data["s_inputs"],
        data["z"],
        data["x_pred"],
        params,
        distogram_atom_idx=data["distogram_atom_idx"],
        token_mask=data["token_attention_mask"],
        atom_to_token=data["atom_to_token"],
        atom_mask=data["atom_attention_mask"],
        asym_id=data["asym_id"],
        mol_type=data["mol_type"],
        n_layers=1,
        n_chains=N_CHAINS,
    )

    for name, reference in expected.items():
        np.testing.assert_allclose(
            np.asarray(got[name]),
            reference.float().numpy(),
            atol=2e-4,
            rtol=2e-4,
            err_msg=name,
        )


def test_plddt_bins_run_zero_to_one() -> None:
    """A head scaled to 0..100 would rank identically and read a hundred-fold high."""
    logits = np.zeros((1, 4, 10), dtype=np.float32)
    mean = np.asarray(heads.categorical_mean(logits, 0.0, 1.0))
    np.testing.assert_allclose(mean, np.full((1, 4), 0.5), atol=1e-6)


def test_the_atom_rank_within_a_token_matches() -> None:
    atom_to_token = np.array([[0, 0, 0, 1, 1, 2]], dtype=np.int64)
    with torch.no_grad():
        expected = common._compute_intra_token_idx(torch.from_numpy(atom_to_token))
    got = heads.intra_token_index(atom_to_token)
    np.testing.assert_array_equal(np.asarray(got), expected.numpy())
