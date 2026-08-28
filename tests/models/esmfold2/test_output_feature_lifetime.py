"""Managed ESMFold2 keeps only the feature leaves its writer reads."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy as np
import pytest

from foldjax.models.esmfold2 import output
from foldjax.models.esmfold2.data.features import build_features, pad_features


class _PoisonOutsideWriterView(Mapping[str, object]):
    """Expose the complete schema while rejecting an unproven value read."""

    def __init__(self, source: Mapping[str, object]) -> None:
        self._source = source

    def __getitem__(self, name: str) -> object:
        if name not in output.ESMFOLD2_GENERATED_OUTPUT_FEATURE_FIELDS:
            raise AssertionError(f"writer read feature outside its view: {name}")
        return self._source[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._source)

    def __len__(self) -> int:
        return len(self._source)


def _all_biomolecule_metadata(
    features: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    result = dict(features)
    tokens = int(features["token_attention_mask"].shape[-1])
    chain_names = np.full((1, tokens, 1), ord("A"), dtype=np.uint8)
    residue_names = np.zeros((1, tokens, 3), dtype=np.uint8)
    names = ("ALA", "GLY")
    for index in range(tokens):
        residue_names[0, index] = np.frombuffer(
            names[index % len(names)].encode("ascii"), dtype=np.uint8
        )
    result["token_chain_id_chars"] = chain_names
    result["token_residue_name_chars"] = residue_names
    return result


def _prediction(features: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    samples = 2
    tokens = int(features["token_attention_mask"].shape[-1])
    atoms = int(features["atom_attention_mask"].shape[-1])
    return {
        "sample_atom_coords": np.arange(
            samples * atoms * 3, dtype=np.float32
        ).reshape(samples, atoms, 3)
        / np.float32(10.0),
        "plddt": np.linspace(
            np.float32(0.1), np.float32(0.9), samples * tokens
        ).reshape(samples, tokens),
        "plddt_per_atom": np.linspace(
            np.float32(0.2), np.float32(0.8), samples * atoms
        ).reshape(samples, atoms),
        "complex_plddt": np.asarray([0.25, 0.75], dtype=np.float32),
        "complex_iplddt": np.asarray([0.5, 0.6], dtype=np.float32),
        "ptm": np.asarray([0.2, 0.3], dtype=np.float32),
        "iptm": np.asarray([0.4, 0.5], dtype=np.float32),
    }


def _feature_case(name: str) -> dict[str, np.ndarray]:
    features = build_features([("AG", "A", 0, 0)])
    if name in {"all_biomolecule", "padded"}:
        features = _all_biomolecule_metadata(features)
    if name == "padded":
        features = pad_features(
            features,
            n_token=features["token_attention_mask"].shape[-1] + 3,
            n_atom=features["atom_attention_mask"].shape[-1] + 32,
            n_msa=features["msa_attention_mask"].shape[-2] + 2,
        )
    return features


@pytest.mark.parametrize("case", ["legacy", "all_biomolecule", "padded"])
def test_poison_writer_view_is_cif_and_json_byte_exact(tmp_path, case: str) -> None:
    features = _feature_case(case)
    prediction = _prediction(features)
    projected = output.project_generated_output_features(features)

    assert projected is not features
    assert set(projected) == (
        output.ESMFOLD2_GENERATED_OUTPUT_FEATURE_FIELDS & features.keys()
    )
    assert all(projected[name] is features[name] for name in projected)

    full = output.write_prediction_outputs(
        prediction, features, tmp_path / "full", name="same"
    )
    poison = output.write_prediction_outputs(
        prediction,
        _PoisonOutsideWriterView(features),
        tmp_path / "poison",
        name="same",
    )
    projected_written = output.write_prediction_outputs(
        prediction, projected, tmp_path / "projected", name="same"
    )

    assert full["summary"] == poison["summary"] == projected_written["summary"]
    assert full["scores"].read_bytes() == poison["scores"].read_bytes()
    assert full["scores"].read_bytes() == projected_written["scores"].read_bytes()
    for full_path, poison_path, projected_path in zip(
        full["structures"],
        poison["structures"],
        projected_written["structures"],
        strict=True,
    ):
        assert full_path.read_bytes() == poison_path.read_bytes()
        assert full_path.read_bytes() == projected_path.read_bytes()


@pytest.mark.parametrize(
    "damage",
    [
        "missing_base_key",
        "wrong_atom_shape",
        "partial_all_biomolecule_metadata",
        "wrong_all_biomolecule_shape",
        "unbatched_custom_mapping",
        "mapping_wrapper",
    ],
)
def test_incomplete_or_custom_writer_schema_keeps_original_identity(
    damage: str,
) -> None:
    features = build_features([("AG", "A", 0, 0)])
    if damage == "missing_base_key":
        del features["res_type"]
    elif damage == "wrong_atom_shape":
        features["ref_element"] = features["ref_element"][:, :-1]
    elif damage == "partial_all_biomolecule_metadata":
        features["token_chain_id_chars"] = np.full(
            (1, 2, 1), ord("A"), dtype=np.uint8
        )
    elif damage == "wrong_all_biomolecule_shape":
        features = _all_biomolecule_metadata(features)
        features["token_chain_id_chars"] = np.full(
            (1, 3, 1), ord("A"), dtype=np.uint8
        )
    elif damage == "unbatched_custom_mapping":
        features = {name: value[0] for name, value in features.items()}
    elif damage == "mapping_wrapper":
        features = _PoisonOutsideWriterView(features)
    else:  # pragma: no cover - parameter list and implementation stay paired.
        raise AssertionError(damage)

    assert output.project_generated_output_features(features) is features
