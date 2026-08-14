"""Embedders against torch, including the encodings whose order is load-bearing."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
common = pytest.importorskip("transformers.models.esmfold2.modeling_esmfold2_common")
modeling = pytest.importorskip("transformers.models.esmfold2.modeling_esmfold2")

from foldjax.models.esmfold2.models import embedders  # noqa: E402
from tests.models.esmfold2.parity import (  # noqa: E402
    random_inputs,
    torch_state_to_numpy,
)

ATOL = 2e-5


def _compare(module, run, arrays, **extra):
    module = module.eval()
    with torch.no_grad():
        expected = module(
            **{k: torch.from_numpy(v) for k, v in arrays.items()}, **extra
        )
    if isinstance(expected, tuple):
        expected = expected[-1]
    got = np.asarray(run(torch_state_to_numpy(module), **arrays))
    np.testing.assert_allclose(got, expected.numpy(), atol=ATOL, rtol=ATOL)


def test_relative_position_encoding_matches() -> None:
    """Two chains, repeated residue numbering: every branch is exercised."""
    module = common.ResIdxAsymIdSymIdEntityIdEncoding(
        n_relative_residx_bins=4, n_relative_chain_bins=2, d_pair=16
    )
    arrays = {
        "residue_index": np.array([[0, 1, 2, 0, 1, 2]], dtype=np.int64),
        "asym_id": np.array([[0, 0, 0, 1, 1, 1]], dtype=np.int64),
        "sym_id": np.array([[0, 0, 0, 1, 1, 1]], dtype=np.int64),
        "entity_id": np.array([[1, 1, 1, 1, 1, 1]], dtype=np.int64),
        "token_index": np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64),
    }
    module = module.eval()
    with torch.no_grad():
        expected = module(**{k: torch.from_numpy(v) for k, v in arrays.items()})
    got = np.asarray(
        embedders.relative_position_encoding(
            **{k: np.asarray(v) for k, v in arrays.items()},
            params=torch_state_to_numpy(module),
            n_residue_bins=4,
            n_chain_bins=2,
        )
    )
    np.testing.assert_allclose(got, expected.numpy(), atol=ATOL, rtol=ATOL)


def test_single_to_pair_keeps_product_before_difference() -> None:
    """The difference is antisymmetric; swapping halves mirrors the pair."""
    _compare(
        common.SingleToPair(input_dim=16, downproject_dim=16, output_dim=16),
        lambda p, x: embedders.single_to_pair(x, p),
        random_inputs({"x": (1, 5, 16)}),
    )


def test_row_attention_pooling_matches() -> None:
    module = common.RowAttentionPooling(d_pair=16, d_single=12).eval()
    z = np.random.default_rng(0).standard_normal((1, 5, 5, 16)).astype(np.float32)
    mask = np.array([[1, 1, 1, 0, 0]], dtype=np.float32)
    with torch.no_grad():
        expected = module(torch.from_numpy(z), torch.from_numpy(mask)).numpy()
    got = np.asarray(
        embedders.row_attention_pooling(z, mask, torch_state_to_numpy(module))
    )
    np.testing.assert_allclose(got, expected, atol=ATOL, rtol=ATOL)


def test_the_msa_encoder_matches_including_its_final_block() -> None:
    """The last block has no MSA-side update and no keys for one."""
    module = modeling.MSAEncoder(
        d_msa=12, d_pair=16, d_inputs=10, d_hidden=4, n_layers=2, n_heads_msa=2,
        msa_head_width=4,
    ).eval()
    rng = np.random.default_rng(7)
    arrays = {
        "x_pair": rng.standard_normal((1, 4, 4, 16)).astype(np.float32),
        "x_inputs": rng.standard_normal((1, 4, 10)).astype(np.float32),
        "msa_oh": rng.standard_normal((1, 4, 3, 33)).astype(np.float32),
        "has_deletion": rng.standard_normal((1, 4, 3)).astype(np.float32),
        "deletion_value": rng.standard_normal((1, 4, 3)).astype(np.float32),
        "msa_attention_mask": np.ones((1, 4, 3), dtype=np.float32),
    }
    with torch.no_grad():
        expected = module(**{k: torch.from_numpy(v) for k, v in arrays.items()})
    got = np.asarray(
        embedders.msa_encoder(
            arrays["x_pair"],
            arrays["x_inputs"],
            arrays["msa_oh"],
            arrays["has_deletion"],
            arrays["deletion_value"],
            arrays["msa_attention_mask"],
            torch_state_to_numpy(module),
            n_layers=2,
        )
    )
    np.testing.assert_allclose(got, expected.numpy(), atol=ATOL, rtol=ATOL)
