"""Optional ESM/ISM preprocessing for original Protenix model variants."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

ESM_MODELS = {
    "esm2-3b": "esm2_t36_3B_UR50D.pt",
    "esm2-3b-ism": "esm2_t36_3B_UR50D_ism.pt",
}


class FairEsmProvider:
    """Lazily run the same fair-esm checkpoint used by upstream Protenix."""

    def __init__(
        self,
        model_name: str,
        *,
        checkpoint_dir: str | Path,
        max_sequence_length: int = 4094,
    ) -> None:
        if model_name not in ESM_MODELS:
            raise ValueError(f"unsupported ESM model: {model_name}")
        self.model_name = model_name
        self.checkpoint = Path(checkpoint_dir) / ESM_MODELS[model_name]
        self.max_sequence_length = max_sequence_length
        self._model: Any = None
        self._alphabet: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from esm import pretrained
        except ImportError as exc:
            raise RuntimeError(
                "ESM preprocessing requires the optional 'fair-esm' and 'torch' "
                "packages"
            ) from exc
        if self.checkpoint.is_file():
            model, alphabet = pretrained.load_model_and_alphabet_local(
                str(self.checkpoint)
            )
        elif self.model_name == "esm2-3b-ism":
            raise FileNotFoundError(
                f"ISM checkpoint is not available from fair-esm: {self.checkpoint}"
            )
        else:
            model, alphabet = pretrained.load_model_and_alphabet(
                self.checkpoint.stem
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.eval().to(device)
        self._alphabet = alphabet

    def __call__(self, sequence: str) -> np.ndarray:
        if len(sequence) > self.max_sequence_length:
            raise ValueError(
                f"ESM sequence length {len(sequence)} exceeds "
                f"{self.max_sequence_length}"
            )
        self._load()
        import torch

        converter = self._alphabet.get_batch_converter(self.max_sequence_length)
        _, _, tokens = converter([("query", sequence)])
        device = next(self._model.parameters()).device
        with torch.no_grad():
            output = self._model(
                tokens.to(device),
                repr_layers=[self._model.num_layers],
                return_contacts=False,
            )
        embedding = output["representations"][self._model.num_layers][0, 1:-1]
        return embedding.float().cpu().numpy()


def add_esm_embeddings(
    features: Mapping[str, Any],
    job: Mapping[str, Any],
    *,
    provider: Callable[[str], np.ndarray],
    embedding_dim: int = 2560,
) -> dict[str, Any]:
    """Attach residue-level ESM values to protein tokens in static features."""

    if "token_entity_id" not in features:
        raise KeyError("ESM features require token_entity_id")
    if "residue_index" not in features:
        raise KeyError("ESM features require residue_index")
    entity_ids = np.asarray(features["token_entity_id"])
    residue_indices = np.asarray(features["residue_index"])
    if entity_ids.shape != residue_indices.shape or entity_ids.ndim != 1:
        raise ValueError("ESM token entity and residue metadata must be rank-1")

    result = dict(features)
    token_embedding = np.zeros((len(entity_ids), embedding_dim), dtype=np.float32)
    sequence_cache: dict[str, np.ndarray] = {}
    for entity_number, wrapper in enumerate(job.get("sequences", []), start=1):
        protein = wrapper.get("proteinChain") if isinstance(wrapper, Mapping) else None
        if not isinstance(protein, Mapping):
            continue
        sequence = str(protein.get("sequence", "")).upper()
        if not sequence:
            raise ValueError(f"protein entity {entity_number} has an empty sequence")
        embedding = sequence_cache.get(sequence)
        if embedding is None:
            embedding = np.asarray(provider(sequence), dtype=np.float32)
            expected = (len(sequence), embedding_dim)
            if embedding.shape != expected:
                raise ValueError(
                    f"ESM embedding shape for entity {entity_number} is "
                    f"{embedding.shape}, expected {expected}"
                )
            sequence_cache[sequence] = embedding
        mask = entity_ids == entity_number
        positions = residue_indices[mask]
        if np.any(positions < 0) or np.any(positions >= len(sequence)):
            raise ValueError(f"invalid ESM residue index for entity {entity_number}")
        token_embedding[mask] = embedding[positions]
    result["esm_token_embedding"] = token_embedding
    return result
