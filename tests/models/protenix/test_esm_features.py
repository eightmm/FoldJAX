from __future__ import annotations

import numpy as np
import pytest

from foldjax.models.protenix.data.esm import add_esm_embeddings


def test_add_esm_embeddings_maps_entities_and_reuses_duplicate_sequences() -> None:
    job = {
        "sequences": [
            {"proteinChain": {"sequence": "AG", "count": 2}},
            {"ligand": {"ligand": "CCD_ATP", "count": 1}},
        ]
    }
    features = {
        "token_entity_id": np.array([1, 1, 1, 1, 2]),
        "residue_index": np.array([0, 1, 0, 1, 0]),
    }
    calls: list[str] = []

    def provider(sequence: str) -> np.ndarray:
        calls.append(sequence)
        return np.arange(6, dtype=np.float32).reshape(2, 3)

    result = add_esm_embeddings(features, job, provider=provider, embedding_dim=3)

    assert calls == ["AG"]
    np.testing.assert_array_equal(
        result["esm_token_embedding"],
        np.array(
            [[0, 1, 2], [3, 4, 5], [0, 1, 2], [3, 4, 5], [0, 0, 0]],
            dtype=np.float32,
        ),
    )


def test_add_esm_embeddings_rejects_wrong_provider_shape() -> None:
    features = {
        "token_entity_id": np.array([1, 1]),
        "residue_index": np.array([0, 1]),
    }
    job = {"sequences": [{"proteinChain": {"sequence": "AG"}}]}

    with pytest.raises(ValueError, match="embedding shape"):
        add_esm_embeddings(
            features,
            job,
            provider=lambda _: np.zeros((1, 3), dtype=np.float32),
            embedding_dim=3,
        )


def test_add_esm_embeddings_requires_entity_metadata() -> None:
    with pytest.raises(KeyError, match="token_entity_id"):
        add_esm_embeddings(
            {"residue_index": np.array([0])},
            {"sequences": []},
            provider=lambda _: np.zeros((0, 3), dtype=np.float32),
            embedding_dim=3,
        )
