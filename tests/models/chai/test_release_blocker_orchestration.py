"""End-to-end regressions for standalone CLI output orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import foldjax.models.chai.inference as inference
import foldjax.models.chai.output as output
from foldjax.models.chai.bridge.bundle_io import NativeBundle
from foldjax.models.chai.bridge.conformer_io import ConformerData
from foldjax.models.chai.data.search.msa import MsaPayload


def _assets() -> inference.InferenceAssets:
    return inference.InferenceAssets(
        bundle=NativeBundle(manifest={}, components={}),
        conformers={
            "ALA": ConformerData(
                position=np.asarray(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0],
                    ],
                    np.float32,
                ),
                element=np.asarray([7, 6, 6, 6], np.int32),
                charge=np.zeros(4, np.int32),
                atom_names=("N", "CA", "C", "CB"),
                bonds=((0, 1), (1, 2), (1, 3)),
                symmetries=np.arange(4, dtype=np.int64)[:, None],
            )
        },
    )


class _MockRemoteBackend:
    name = "mock-remote"
    version = "test-v1"
    server_url = "https://mock.invalid"
    cache_identity: dict[str, object] = {}

    def search(self, sequence: str) -> MsaPayload:
        query = f">query\n{sequence}\n"
        return MsaPayload(query, query, source={"mock": True})


def test_default_remote_msa_cache_coexists_with_written_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fasta = tmp_path / "query.fasta"
    fasta.write_text(">protein|name=PROA\nAA\n", encoding="utf-8")
    output_dir = tmp_path / "outputs"
    monkeypatch.setattr(inference, "load_inference_assets", lambda *a, **k: _assets())
    monkeypatch.setattr(inference, "map_model_components", lambda _bundle: object())
    monkeypatch.setattr(
        inference,
        "RemoteMMseqs2Backend",
        lambda *_args, **_kwargs: _MockRemoteBackend(),
    )
    monkeypatch.setattr(output, "get_scores", lambda _ranking: {"score": 0.0})

    def execute(prepared, _components, _config):
        token_count = prepared.model_size
        atom_count = prepared.padded_inputs["atom_exists_mask"].shape[1]
        return output.TrunkPrediction(
            atom_coords=np.zeros((1, atom_count, 3), np.float32),
            atom_plddt=np.full((1, atom_count), 0.5, np.float32),
            atom_mask=prepared.padded_inputs["atom_exists_mask"][0],
            token_mask=prepared.padded_inputs["token_exists_mask"][0],
            ranking_data=[object()],
            pae=np.zeros((1, token_count, token_count), np.float32),
            pde=np.zeros((1, token_count, token_count), np.float32),
            plddt=np.zeros((1, token_count), np.float32),
            structure_context=prepared.structure_context.pad(token_count, atom_count),
        )

    monkeypatch.setattr(inference, "execute_prepared_inference", execute)

    candidates = inference.run_inference(
        fasta,
        output_dir=output_dir,
        bundle_path="bundle",
        conformer_path="conformers",
        config=inference.InferenceConfig(
            use_msa_server=True,
            use_esm_embeddings=False,
            fasta_names_as_cif_chains=True,
            num_diffusion_samples=1,
        ),
    )

    assert (output_dir / "msas").is_dir()
    assert candidates.cif_paths == [output_dir / "pred.model_idx_0.cif"]
    assert "PROA" in candidates.cif_paths[0].read_text(encoding="utf-8")
    assert candidates.msa_coverage_plot_path == output_dir / "msa_depth.pdf"
