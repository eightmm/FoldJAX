"""Inference-to-public-output integration for the MSA coverage artifact."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import foldjax.models.chai.inference as inference
from foldjax.models.chai.output import StructureCandidates


def test_run_inference_writes_msa_depth_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = SimpleNamespace(
        realized_seed=0,
        structure_context=SimpleNamespace(
            num_tokens=2,
            token_residue_type=np.asarray([0, 1], dtype=np.int32),
        ),
        padded_inputs={
            "msa_tokens": np.asarray([[[0, 1], [0, 31]]], dtype=np.int32),
            "msa_mask": np.asarray([[[True, True], [True, True]]]),
        },
    )
    assets = SimpleNamespace(bundle=object())
    candidates = StructureCandidates(
        cif_paths=[],
        ranking_data=[],
        msa_coverage_plot_path=None,
        pae=np.empty((0, 2, 2), dtype=np.float32),
        pde=np.empty((0, 2, 2), dtype=np.float32),
        plddt=np.empty((0, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        inference, "prepare_inference", lambda *_args, **_kwargs: (prepared, assets)
    )
    monkeypatch.setattr(inference, "map_model_components", lambda _bundle: object())
    monkeypatch.setattr(
        inference, "execute_prepared_inference", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        inference, "write_prediction_outputs", lambda *_args, **_kwargs: candidates
    )

    result = inference.run_inference(
        tmp_path / "query.fasta",
        output_dir=tmp_path / "outputs",
        bundle_path=tmp_path / "bundle",
        conformer_path=tmp_path / "conformers.npz",
    )

    assert result.msa_coverage_plot_path == tmp_path / "outputs/msa_depth.pdf"
    assert result.msa_coverage_plot_path.read_bytes().startswith(b"%PDF-1.4")


def test_multiple_trunks_write_msa_plot_in_each_trunk_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = SimpleNamespace(
        realized_seed=0,
        structure_context=SimpleNamespace(
            num_tokens=1,
            token_residue_type=np.asarray([0], dtype=np.int32),
        ),
        padded_inputs={
            "msa_tokens": np.asarray([[[0]]], dtype=np.int32),
            "msa_mask": np.asarray([[[True]]]),
        },
    )
    assets = SimpleNamespace(bundle=object())
    candidates = StructureCandidates(
        cif_paths=[],
        ranking_data=[],
        msa_coverage_plot_path=None,
        pae=np.empty((0, 1, 1), dtype=np.float32),
        pde=np.empty((0, 1, 1), dtype=np.float32),
        plddt=np.empty((0, 1), dtype=np.float32),
    )
    monkeypatch.setattr(
        inference, "prepare_inference", lambda *_args, **_kwargs: (prepared, assets)
    )
    monkeypatch.setattr(inference, "map_model_components", lambda _bundle: object())
    monkeypatch.setattr(
        inference, "execute_prepared_inference", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        inference, "write_prediction_outputs", lambda *_args, **_kwargs: candidates
    )

    result = inference.run_inference(
        tmp_path / "query.fasta",
        output_dir=tmp_path / "outputs",
        bundle_path=tmp_path / "bundle",
        conformer_path=tmp_path / "conformers.npz",
        config=inference.InferenceConfig(num_trunk_samples=2),
    )

    first = tmp_path / "outputs/trunk_0/msa_depth.pdf"
    second = tmp_path / "outputs/trunk_1/msa_depth.pdf"
    assert result.msa_coverage_plot_path == first
    assert first.is_file()
    assert second.is_file()
