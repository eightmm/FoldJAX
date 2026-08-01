"""Seed ownership and stochastic preprocessing regression tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import foldjax.models.chai.inference as inference
from foldjax.models.chai.bridge.bundle_io import NativeBundle
from foldjax.models.chai.bridge.conformer_io import ConformerData
from foldjax.models.chai.data.features import generate_features

from .test_features import _inputs


def _conformer(names: tuple[str, ...]) -> ConformerData:
    count = len(names)
    return ConformerData(
        position=np.arange(count * 3, dtype=np.float32).reshape(count, 3),
        element=np.full(count, 6, np.int32),
        charge=np.zeros(count, np.int32),
        atom_names=names,
        bonds=tuple((index, index + 1) for index in range(count - 1)),
        symmetries=np.arange(count, dtype=np.int64)[:, None],
    )


def _assets() -> inference.InferenceAssets:
    return inference.InferenceAssets(
        bundle=NativeBundle(manifest={}, components={}),
        conformers={
            "ALA": _conformer(("N", "CA", "C", "O", "CB")),
            "MSE": _conformer(("N", "CA", "C", "O", "CB", "SE")),
        },
    )


def _prepare(tmp_path: Path, seed: int | None) -> inference.PreparedInference:
    fasta = tmp_path / "query.fasta"
    fasta.write_text(">protein|name=target\nA(MSE)\n", encoding="utf-8")
    return inference.prepare_inference_from_assets(
        fasta,
        _assets(),
        config=inference.InferenceConfig(seed=seed, use_esm_embeddings=False),
    )


def test_explicit_seed_reproduces_all_prepared_arrays(tmp_path: Path) -> None:
    left = _prepare(tmp_path, 1729)
    right = _prepare(tmp_path, 1729)

    assert left.realized_seed == right.realized_seed == 1729
    for name in left.structure_context.to_dict():
        left_value = getattr(left.structure_context, name)
        right_value = getattr(right.structure_context, name)
        if isinstance(left_value, np.ndarray):
            np.testing.assert_array_equal(left_value, right_value, err_msg=name)
    assert left.padded_inputs.keys() == right.padded_inputs.keys()
    for name in left.padded_inputs:
        left_value = left.padded_inputs[name]
        right_value = right.padded_inputs[name]
        if isinstance(left_value, np.ndarray):
            np.testing.assert_array_equal(left_value, right_value, err_msg=name)
    for name in left.features:
        np.testing.assert_array_equal(
            left.features[name], right.features[name], err_msg=name
        )


def test_different_seeds_change_modified_conformer_augmentation(tmp_path: Path) -> None:
    left = _prepare(tmp_path, 17)
    right = _prepare(tmp_path, 18)

    assert left.realized_seed == 17
    assert right.realized_seed == 18
    assert not np.array_equal(
        left.structure_context.atom_ref_pos, right.structure_context.atom_ref_pos
    )


def test_none_seed_is_realized_and_never_silently_becomes_zero(
    tmp_path: Path, monkeypatch
) -> None:
    seeds = iter((101, 202))
    monkeypatch.setattr(inference, "_random_seed", lambda: next(seeds))

    left = _prepare(tmp_path, None)
    right = _prepare(tmp_path, None)

    assert (left.realized_seed, right.realized_seed) == (101, 202)
    assert not np.array_equal(
        left.structure_context.atom_ref_pos, right.structure_context.atom_ref_pos
    )


def test_ordinary_inputs_are_seed_invariant(tmp_path: Path) -> None:
    fasta = tmp_path / "ordinary.fasta"
    fasta.write_text(">protein|name=target\nAA\n", encoding="utf-8")
    assets = _assets()

    left = inference.prepare_inference_from_assets(
        fasta,
        assets,
        config=inference.InferenceConfig(seed=1, use_esm_embeddings=False),
    )
    right = inference.prepare_inference_from_assets(
        fasta,
        assets,
        config=inference.InferenceConfig(seed=2, use_esm_embeddings=False),
    )

    for name in left.features:
        np.testing.assert_array_equal(
            left.features[name], right.features[name], err_msg=name
        )


def test_docking_noise_and_dropout_use_the_supplied_rng() -> None:
    inputs = _inputs()
    inputs["token_asym_id"] = np.asarray([[1, 1, 2, 0]], np.int32)
    inputs["subchain_id"] = np.asarray(
        [[[65, 255, 255, 255], [65, 255, 255, 255], [66, 255, 255, 255], [255] * 4]],
        np.uint8,
    )
    inputs["docking_constraints"] = [
        [
            {
                "subchain_ids": ["A", "B"],
                "noise_sigma": 5.0,
                "dropout_prob": 0.4,
                "atom_center_mask": [
                    np.asarray([True, True]),
                    np.asarray([True]),
                ],
                "atom_center_coords": [
                    np.asarray([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], np.float32),
                    np.asarray([[10.0, 0.0, 0.0]], np.float32),
                ],
            }
        ]
    ]

    left = generate_features(inputs, rng=np.random.default_rng(31))[
        "DockingConstraintGenerator"
    ]
    repeated = generate_features(inputs, rng=np.random.default_rng(31))[
        "DockingConstraintGenerator"
    ]
    different = generate_features(inputs, rng=np.random.default_rng(32))[
        "DockingConstraintGenerator"
    ]

    np.testing.assert_array_equal(left, repeated)
    assert not np.array_equal(left, different)
