"""Standalone Chai public result and file-output contract tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from foldjax.models.chai.bridge.conformer_io import ConformerData
from foldjax.models.chai.data.input import EntityType, Input
from foldjax.models.chai.data.structure import tokenize_inputs
from foldjax.models.chai.output import (
    OutputDirectoryNotEmptyError,
    StructureCandidates,
    TrunkPrediction,
    write_msa_coverage_pdf,
    write_prediction_outputs,
)
from foldjax.models.chai.ranking.clashes import ClashScores
from foldjax.models.chai.ranking.plddt import PLDDTScores
from foldjax.models.chai.ranking.ptm import PTMScores
from foldjax.models.chai.ranking.rank import SampleRanking

SCORE_KEYS = {
    "aggregate_score",
    "ptm",
    "iptm",
    "per_chain_ptm",
    "per_chain_pair_iptm",
    "has_inter_chain_clashes",
    "chain_chain_clashes",
}


def test_msa_coverage_pdf_is_dependency_free_and_valid(tmp_path: Path) -> None:
    output = tmp_path / "msa_depth.pdf"
    result = write_msa_coverage_pdf(
        input_tokens=np.asarray([0, 1, 2, 3], dtype=np.int32),
        msa_tokens=np.asarray(
            [[0, 1, 2, 3], [0, 31, 2, 31], [5, 1, 32, 3]], dtype=np.int32
        ),
        msa_mask=np.asarray(
            [[True, True, True, True], [True, False, True, False], [True] * 4]
        ),
        output_path=output,
    )

    assert result == output
    data = output.read_bytes()
    assert data.startswith(b"%PDF-1.4")
    assert data.endswith(b"%%EOF\n")
    assert b"MSA sequence coverage" in data


def _ranking(score: float) -> SampleRanking:
    return SampleRanking(
        asym_ids=np.asarray([1, 2]),
        aggregate_score=np.asarray([score], dtype=np.float32),
        ptm_scores=PTMScores(
            complex_ptm=np.asarray([0.4], dtype=np.float32),
            interface_ptm=np.asarray([0.5], dtype=np.float32),
            per_chain_ptm=np.asarray([[0.3, 0.4]], dtype=np.float32),
            per_chain_pair_iptm=np.asarray(
                [[[0.0, 0.5], [0.6, 0.0]]], dtype=np.float32
            ),
        ),
        clash_scores=ClashScores(
            total_clashes=np.asarray([0]),
            total_inter_chain_clashes=np.asarray([0]),
            chain_chain_clashes=np.zeros((1, 2, 2), dtype=np.int32),
            has_inter_chain_clashes=np.asarray([False]),
        ),
        plddt_scores=PLDDTScores(
            complex_plddt=np.asarray([0.7], dtype=np.float32),
            per_chain_plddt=np.asarray([[0.7, 0.8]], dtype=np.float32),
            per_atom_plddt=np.asarray([[0.6, 0.7]], dtype=np.float32),
        ),
    )


def _prediction(scores: tuple[float, ...], offset: float = 0.0) -> TrunkPrediction:
    count = len(scores)
    coords = np.arange(count * 5 * 3, dtype=np.float32).reshape(count, 5, 3)
    coords += offset
    atom_plddt = np.broadcast_to(np.linspace(0.1, 0.9, 5, dtype=np.float32), (count, 5))
    pae = np.arange(count * 4 * 4, dtype=np.float32).reshape(count, 4, 4)
    return TrunkPrediction(
        atom_coords=coords,
        atom_plddt=atom_plddt,
        atom_mask=np.asarray([True, True, False, True, False]),
        token_mask=np.asarray([True, True, False, True]),
        ranking_data=[_ranking(score) for score in scores],
        pae=pae,
        pde=pae + 100.0,
        plddt=np.arange(count * 4, dtype=np.float32).reshape(count, 4),
    )


def _conformer(atom_names: tuple[str, ...]) -> ConformerData:
    count = len(atom_names)
    return ConformerData(
        position=np.arange(count * 3, dtype=np.float32).reshape(count, 3),
        element=np.full(count, 6, np.int32),
        charge=np.zeros(count, np.int32),
        atom_names=atom_names,
        bonds=(),
        symmetries=np.arange(count, dtype=np.int64)[:, None],
    )


def test_structure_candidates_sort_is_stable_and_reorders_scores() -> None:
    candidate = StructureCandidates(
        cif_paths=[Path("low.cif"), Path("high-a.cif"), Path("high-b.cif")],
        ranking_data=[_ranking(0.1), _ranking(0.9), _ranking(0.9)],
        msa_coverage_plot_path=Path("msa_depth.pdf"),
        pae=np.asarray([[[1.0]], [[2.0]], [[3.0]]]),
        pde=np.asarray([[[11.0]], [[12.0]], [[13.0]]]),
        plddt=np.asarray([[21.0], [22.0], [23.0]]),
    ).sorted()
    assert [path.name for path in candidate.cif_paths] == [
        "high-a.cif",
        "high-b.cif",
        "low.cif",
    ]
    np.testing.assert_array_equal(candidate.pae[:, 0, 0], [2.0, 3.0, 1.0])
    np.testing.assert_array_equal(candidate.pde[:, 0, 0], [12.0, 13.0, 11.0])
    np.testing.assert_array_equal(candidate.plddt[:, 0], [22.0, 23.0, 21.0])


def test_single_trunk_writes_exact_names_keys_and_unpads(tmp_path: Path) -> None:
    calls = []

    def writer(coords: np.ndarray, b_factors: np.ndarray | None, index: int) -> str:
        calls.append((coords.copy(), None if b_factors is None else b_factors.copy()))
        return f"data_model_{index}\n"

    result = write_prediction_outputs(
        tmp_path / "outputs",
        [_prediction((0.2, 0.8))],
        cif_text_writer=writer,
    )
    output_dir = tmp_path / "outputs"
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "pred.model_idx_0.cif",
        "pred.model_idx_1.cif",
        "scores.model_idx_0.npz",
        "scores.model_idx_1.npz",
    ]
    assert [path.name for path in result.cif_paths] == [
        "pred.model_idx_0.cif",
        "pred.model_idx_1.cif",
    ]
    assert result.pae.shape == (2, 3, 3)
    assert result.pde.shape == (2, 3, 3)
    assert result.plddt.shape == (2, 3)
    assert [call[0].shape for call in calls] == [(3, 3), (3, 3)]
    np.testing.assert_allclose(calls[0][1], [10.0, 30.0, 70.0], atol=1e-5)
    with np.load(output_dir / "scores.model_idx_0.npz") as scores:
        assert set(scores.files) == SCORE_KEYS
    assert (output_dir / "pred.model_idx_1.cif").read_text() == "data_model_1\n"


def test_multi_trunk_directory_names_and_concat_order(tmp_path: Path) -> None:
    result = write_prediction_outputs(
        tmp_path / "outputs",
        [_prediction((0.1,), 0.0), _prediction((0.9, 0.2), 100.0)],
        cif_text_writer=lambda coords, b_factors, index: f"data_{coords[0, 0]}\n",
    )
    relative_paths = [
        path.relative_to(tmp_path / "outputs").as_posix() for path in result.cif_paths
    ]
    assert relative_paths == [
        "trunk_0/pred.model_idx_0.cif",
        "trunk_1/pred.model_idx_0.cif",
        "trunk_1/pred.model_idx_1.cif",
    ]
    assert (tmp_path / "outputs" / "trunk_0" / "scores.model_idx_0.npz").is_file()
    assert (tmp_path / "outputs" / "trunk_1" / "scores.model_idx_1.npz").is_file()
    assert result.pae.shape == (3, 3, 3)


def test_nonempty_output_directory_fails_before_writes(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("owned by user")
    called = False

    def writer(coords: np.ndarray, b_factors: np.ndarray | None, index: int) -> str:
        nonlocal called
        called = True
        return "data\n"

    with pytest.raises(OutputDirectoryNotEmptyError, match="not empty"):
        write_prediction_outputs(
            output_dir,
            [_prediction((0.1,))],
            cif_text_writer=writer,
        )
    assert not called
    assert sentinel.read_text() == "owned by user"
    assert list(output_dir.iterdir()) == [sentinel]


def test_trunk_with_no_candidates_is_rejected_before_writes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one candidate"):
        write_prediction_outputs(
            tmp_path / "outputs",
            [_prediction(())],
            cif_text_writer=lambda coords, b_factors, index: "data\n",
        )
    assert not (tmp_path / "outputs").exists()


def test_native_modelcif_writer_preserves_structure_metadata(tmp_path: Path) -> None:
    gemmi = pytest.importorskip("gemmi")
    context = tokenize_inputs(
        [
            Input("A", EntityType.PROTEIN.value, "alpha"),
            Input("A", EntityType.PROTEIN.value, "alpha-copy"),
            Input("A", EntityType.RNA.value, "rna"),
        ],
        {
            "ALA": _conformer(("N", "CA", "C", "O", "CB")),
            "A": _conformer(("P", "C1'", "C3'", "C4'", "C4")),
        },
    )
    count = context.num_atoms
    prediction = TrunkPrediction(
        atom_coords=context.atom_ref_pos[None],
        atom_plddt=np.linspace(0.1, 0.9, count, dtype=np.float32)[None],
        atom_mask=np.ones(count, bool),
        token_mask=np.ones(context.num_tokens, bool),
        ranking_data=[_ranking(0.5)],
        pae=np.zeros((1, context.num_tokens, context.num_tokens), np.float32),
        pde=np.zeros((1, context.num_tokens, context.num_tokens), np.float32),
        plddt=np.zeros((1, context.num_tokens), np.float32),
        structure_context=context,
        asym_entity_names={1: "A", 2: "B", 3: "RNA"},
    )

    result = write_prediction_outputs(tmp_path / "outputs", [prediction])

    document = gemmi.cif.read_file(str(result.cif_paths[0]))
    block = document.sole_block()
    atom_site = block.find_mmcif_category("_atom_site.")
    assert len(atom_site) == count
    asym_ids = list(block.find_values("_atom_site.label_asym_id"))
    entity_ids = list(block.find_values("_atom_site.label_entity_id"))
    assert set(asym_ids) == {"A", "B", "RNA"}
    assert set(block.find_values("_atom_site.type_symbol")) == {"C"}
    assert [
        float(value) for value in block.find_values("_atom_site.B_iso_or_equiv")
    ] == pytest.approx(np.linspace(10.0, 90.0, count), abs=0.01)
    # Identical protein sequences share one entity; RNA is a different entity.
    assert entity_ids[0] == entity_ids[5]
    assert entity_ids[0] != entity_ids[10]
    qa = block.find_mmcif_category("_ma_qa_metric_local.")
    assert len(qa) == 3


def test_fasta_entity_names_flow_through_prediction_to_modelcif(
    tmp_path: Path,
) -> None:
    gemmi = pytest.importorskip("gemmi")
    context = tokenize_inputs(
        [
            Input("A", EntityType.PROTEIN.value, "PROA"),
            Input("A", EntityType.RNA.value, "RNA1"),
        ],
        {
            "ALA": _conformer(("N", "CA", "C", "O", "CB")),
            "A": _conformer(("P", "C1'", "C3'", "C4'", "C4")),
        },
        entity_name_as_subchain=True,
    )
    count = context.num_atoms
    prediction = TrunkPrediction(
        atom_coords=context.atom_ref_pos[None],
        atom_plddt=np.full((1, count), 0.5, np.float32),
        atom_mask=np.ones(count, bool),
        token_mask=np.ones(context.num_tokens, bool),
        ranking_data=[_ranking(0.5)],
        pae=np.zeros((1, context.num_tokens, context.num_tokens), np.float32),
        pde=np.zeros((1, context.num_tokens, context.num_tokens), np.float32),
        plddt=np.zeros((1, context.num_tokens), np.float32),
        structure_context=context,
    )

    assert prediction.asym_entity_names == {1: "PROA", 2: "RNA1"}

    result = write_prediction_outputs(tmp_path / "outputs", [prediction])
    block = gemmi.cif.read_file(str(result.cif_paths[0])).sole_block()
    assert set(block.find_values("_atom_site.label_asym_id")) == {"PROA", "RNA1"}


def test_prediction_output_preserves_reserved_default_msa_cache(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "outputs"
    msa_cache = output_dir / "msas" / "cache-key"
    msa_cache.mkdir(parents=True)
    provenance = msa_cache / "provenance.json"
    provenance.write_text("{}\n", encoding="utf-8")

    result = write_prediction_outputs(
        output_dir,
        [_prediction((0.5,))],
        cif_text_writer=lambda coords, b_factors, index: "data\n",
    )

    assert result.cif_paths == [output_dir / "pred.model_idx_0.cif"]
    assert provenance.read_text(encoding="utf-8") == "{}\n"


def test_native_modelcif_requires_context(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    with pytest.raises(ValueError, match="structure_context"):
        write_prediction_outputs(output_dir, [_prediction((0.5,))])
    assert not output_dir.exists()


def test_native_modelcif_makes_ligand_atom_ids_unique(tmp_path: Path) -> None:
    gemmi = pytest.importorskip("gemmi")
    context = tokenize_inputs(
        [Input("A", EntityType.PROTEIN.value, "ligand")],
        {"ALA": _conformer(("N", "CA", "C", "O", "CB"))},
    )
    context = replace(
        context,
        token_entity_type=np.asarray([EntityType.LIGAND.value], np.int32),
        token_residue_name=np.asarray(
            [[ord(char) for char in "LIG"] + [255] * 5], np.uint8
        ),
        atom_ref_name=("C", "C", "C", "C", "C"),
    )
    count = context.num_atoms
    prediction = TrunkPrediction(
        atom_coords=context.atom_ref_pos[None],
        atom_plddt=np.full((1, count), 0.5, np.float32),
        atom_mask=np.ones(count, bool),
        token_mask=np.ones(1, bool),
        ranking_data=[_ranking(0.5)],
        pae=np.zeros((1, 1, 1), np.float32),
        pde=np.zeros((1, 1, 1), np.float32),
        plddt=np.zeros((1, 1), np.float32),
        structure_context=context,
    )

    result = write_prediction_outputs(tmp_path / "outputs", [prediction])

    block = gemmi.cif.read_file(str(result.cif_paths[0])).sole_block()
    assert list(block.find_values("_atom_site.label_atom_id")) == [
        "C_1",
        "C_2",
        "C_3",
        "C_4",
        "C_5",
    ]
    assert set(block.find_values("_atom_site.group_PDB")) == {"HETATM"}
    assert not block.find_values("_ma_qa_metric_local.metric_value")
