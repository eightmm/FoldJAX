"""`confidence_full.npz`: the confidence arrays a model already computes, on disk.

What is checked here is the contract a reader relies on: the arrays come back
by name with their axes, unit, scale and stored dtype; the archive lands in the
canonical sample directory beside `confidence.json`; the manifest and
capabilities say which arrays a model provides; and each backend routes the
arrays its program returned without inventing any.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from foldjax import confidence_arrays
from foldjax.confidence_arrays import FILENAME, load_confidence_arrays
from foldjax.schema import PredictionRequest, PredictionResult, PredictionSample

MODELS = ("alphafold3", "boltz2", "esmfold2", "opendde", "openfold3", "protenix")


def _write(path: Path, **overrides):
    rng = np.random.default_rng(0)
    arrays = {
        "pae": rng.uniform(0, 31.9, size=(3, 3)).astype(np.float32),
        "token_plddt": np.asarray([0.1, 0.5, 0.9], dtype=np.float32),
        "token_chain_id": np.asarray(["A", "A", "B"]),
        "token_residue_index": np.asarray([1, 2, 1]),
    }
    arrays.update(overrides.pop("arrays", {}))
    return confidence_arrays.write(
        path,
        model="boltz2",
        arrays=arrays,
        scales={"token_plddt": "0-1"},
        sources={"pae": "pae", "token_plddt": "plddt"},
        unavailable={"pae": "never", "chain_pair_iptm": "not computed"},
        sample={"sample": 0},
        **overrides,
    ), arrays


def test_round_trip_keeps_values_axes_units_and_dtypes(tmp_path: Path) -> None:
    record, arrays = _write(tmp_path / FILENAME)
    loaded = load_confidence_arrays(tmp_path)

    assert loaded.model == "boltz2"
    assert set(loaded) == set(arrays)
    # Token-pair maps are float16; the rounding is below 0.008 A under 32 A.
    assert loaded["pae"].dtype == np.float16
    np.testing.assert_allclose(loaded["pae"], arrays["pae"], atol=0.008)
    assert loaded["token_plddt"].dtype == np.float32
    np.testing.assert_array_equal(loaded["token_plddt"], arrays["token_plddt"])
    assert loaded["token_chain_id"].tolist() == ["A", "A", "B"]
    assert loaded["token_residue_index"].dtype == np.int32

    pae = loaded.describe("pae")
    assert pae["axes"] == ["token", "token"]
    assert pae["unit"] == "angstrom"
    assert pae["dtype"] == "float16"
    assert loaded.describe("token_plddt")["scale"] == "0-1"
    assert loaded.describe("token_plddt")["source"] == "plddt"
    # A reason is kept only for what is actually absent.
    assert loaded.unavailable == {"chain_pair_iptm": "not computed"}
    assert record["arrays"] == ["pae", "token_plddt"]
    assert record["index_arrays"] == ["token_chain_id", "token_residue_index"]


def test_the_archive_needs_no_pickle(tmp_path: Path) -> None:
    _write(tmp_path / FILENAME)
    with np.load(tmp_path / FILENAME, allow_pickle=False) as archive:
        meta = json.loads(str(archive["_meta"][()]))
    assert meta["schema_version"] == confidence_arrays.SCHEMA_VERSION
    assert set(meta["axes"]) == {"token", "atom", "chain"}


@pytest.mark.parametrize(
    ("arrays", "message"),
    [
        ({"made_up": np.zeros(3)}, "unknown"),
        ({"pae": np.zeros(3)}, "axes"),
    ],
)
def test_unknown_names_and_wrong_axes_are_refused(
    tmp_path: Path, arrays, message
) -> None:
    with pytest.raises(ValueError, match=message):
        _write(tmp_path / FILENAME, arrays=arrays)


def test_plddt_without_a_scale_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scale"):
        confidence_arrays.write(
            tmp_path / FILENAME, model="x", arrays={"atom_plddt": np.ones(2)}
        )


def test_missing_archive_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_confidence_arrays(tmp_path)


def test_normalize_moves_the_staged_archive_beside_confidence_json(
    tmp_path: Path,
) -> None:
    from foldjax.output import normalize

    native = tmp_path / "native"
    native.mkdir()
    structure = native / "job_model_0.cif"
    structure.write_text("data_x\n")
    record, _arrays = _write(confidence_arrays.staged_path(structure))
    result = PredictionResult(
        model="boltz2",
        samples=(
            PredictionSample(
                seed=3,
                structure_path=structure,
                metadata={"confidence_arrays": record},
            ),
        ),
        output_dir=tmp_path,
    )

    placed = normalize(result, job="job")

    directory = tmp_path / "seed-3_sample-00"
    assert (directory / FILENAME).is_file()
    assert (directory / "confidence.json").is_file()
    assert not confidence_arrays.staged_path(structure).exists()
    entry = placed.samples[0].metadata["confidence_arrays"]
    assert entry["file"] == FILENAME and "path" not in entry
    assert set(load_confidence_arrays(directory)) >= {"pae", "token_plddt"}

    summary = confidence_arrays.manifest_record(placed.samples)
    assert summary == {
        "file": FILENAME,
        "schema_version": confidence_arrays.SCHEMA_VERSION,
        "samples": 1,
        "arrays": ["pae", "token_plddt"],
        "unavailable": {"chain_pair_iptm": "not computed"},
    }
    # The per-sample record survives the manifest's redaction unchanged.
    assert placed.samples[0].summary()["metadata"]["confidence_arrays"] == entry


def test_a_sample_without_arrays_is_left_alone(tmp_path: Path) -> None:
    sample = PredictionSample(seed=1, metadata={"job": "x"})
    assert confidence_arrays.place(sample, tmp_path) is sample
    assert confidence_arrays.manifest_record([sample]) is None


def test_a_run_places_the_archive_and_records_it_in_the_manifest(
    tmp_path: Path,
) -> None:
    import foldjax
    from foldjax.manifest import MANIFEST_NAME
    from foldjax.registry import backend_override
    from tests.test_manifest import _job, _Recorder, _weights

    class _WithArrays(_Recorder):
        def predict(self, request):
            result = super().predict(request)
            (sample,) = result.samples
            confidence_arrays.write(
                confidence_arrays.staged_path(sample.structure_path),
                model="opendde",
                arrays={"atom_plddt": np.full(4, 0.5)},
                scales={"atom_plddt": "0-1"},
                unavailable=confidence_arrays.AVAILABILITY["opendde"]["unavailable"],
            )
            metadata = confidence_arrays.sample_metadata(sample.structure_path)
            return dataclasses.replace(
                result, samples=(dataclasses.replace(sample, metadata=metadata),)
            )

    out = tmp_path / "out"
    request = PredictionRequest(
        model="opendde",
        input=_job(tmp_path),
        weights=_weights(tmp_path),
        output_dir=out,
        seed=17,
        use_compile_cache=False,
    )
    with backend_override("opendde", _WithArrays):
        result = foldjax.predict(request)

    directory = out / "seed-17_sample-00"
    assert load_confidence_arrays(directory)["atom_plddt"].tolist() == [0.5] * 4
    assert result.samples[0].metadata["confidence_arrays"]["file"] == FILENAME
    manifest = json.loads((out / MANIFEST_NAME).read_text())
    assert manifest["confidence_arrays"]["arrays"] == ["atom_plddt"]
    assert "pae" in manifest["confidence_arrays"]["unavailable"]
    recorded = manifest["samples"][0]["metadata"]["confidence_arrays"]
    assert recorded["file"] == FILENAME and "path" not in recorded


@pytest.mark.parametrize("model", MODELS)
def test_capabilities_report_each_models_default_arrays(model: str) -> None:
    from foldjax import capabilities

    described = capabilities(model)
    assert described.confidence_arrays == confidence_arrays.default_arrays(model)
    assert described.confidence_arrays
    for name in described.confidence_arrays:
        assert name in confidence_arrays.SPECS
        assert name not in confidence_arrays.INDEX_ARRAYS
    reasons = confidence_arrays.AVAILABILITY[model]["unavailable"]
    assert not set(reasons) & set(described.confidence_arrays)


# --- Boltz-2 --------------------------------------------------------------------


def _boltz_request(tmp_path: Path) -> PredictionRequest:
    input_path = tmp_path / "job.yaml"
    input_path.write_text("{}")
    weights = tmp_path / "weights"
    weights.mkdir()
    mols = tmp_path / "mols"
    mols.mkdir()
    return PredictionRequest(
        model="boltz2",
        input=input_path,
        weights=weights,
        output_dir=tmp_path / "out",
        seed=5,
        cache_dir=tmp_path / "cache",
        options={"mols": mols},
    )


def test_boltz2_routes_pae_pde_and_token_plddt_per_sample(
    tmp_path: Path, monkeypatch
) -> None:
    from foldjax.backends.boltz2 import Boltz2Backend

    out = tmp_path / "out"
    out.mkdir()
    paths = [out / f"job_model_{index}.cif" for index in range(2)]
    for path in paths:
        path.write_text("data_x\n")
    pae = np.stack([np.full((3, 3), 1.5), np.full((3, 3), 7.25)])
    pde = pae + 0.5

    def native_predict(**kwargs):
        return {
            "coords": np.zeros((2, 4, 3)),
            "plddt": np.asarray([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]]),
            "out_paths": paths,
            "raw": {
                "pae": pae,
                "pde": pde,
                "complex_pde": np.asarray([0.4, 0.6]),
                "complex_ipde": np.asarray([1.4, 1.6]),
            },
            "confidence_index": {
                "token_chain_id": np.asarray(["A", "A", "B"]),
                "token_residue_index": np.asarray([1, 2, 1], dtype=np.int32),
                "atom_token_index": np.asarray([0, 0, 1, 2], dtype=np.int32),
            },
        }

    monkeypatch.setattr(
        "foldjax.backends.boltz2.import_module",
        lambda name: SimpleNamespace(predict=native_predict),
    )
    result = Boltz2Backend().predict(_boltz_request(tmp_path))

    assert [sample.scores["complex_pde"] for sample in result.samples] == (
        pytest.approx([0.4, 0.6])
    )
    assert [sample.scores["complex_ipde"] for sample in result.samples] == (
        pytest.approx([1.4, 1.6])
    )
    assert "confidence_index" not in result.raw
    for index, sample in enumerate(result.samples):
        entry = sample.metadata["confidence_arrays"]
        assert entry["arrays"] == ["pae", "pde", "token_plddt"]
        loaded = load_confidence_arrays(Path(entry["path"]))
        np.testing.assert_array_equal(loaded["pae"], pae[index])
        np.testing.assert_array_equal(loaded["pde"], pde[index])
        np.testing.assert_allclose(
            loaded["token_plddt"], [[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]][index]
        )
        assert loaded["token_chain_id"].tolist() == ["A", "A", "B"]
        assert loaded["atom_token_index"].tolist() == [0, 0, 1, 2]
        assert loaded.describe("token_plddt")["scale"] == "0-1"
        assert "chain_pair_iptm" in loaded.unavailable


def test_boltz2_index_maps_agree_with_the_written_mmcif(tmp_path: Path) -> None:
    """Token chain/residue maps, read through atom_token_index, name each CIF atom."""
    gemmi = pytest.importorskip("gemmi")
    from foldjax.paths import weights_dir

    mols = weights_dir() / "boltz2" / "mols"
    if not mols.is_dir():
        pytest.skip("Boltz-2 CCD molecule directory is not in the weight store")
    from foldjax.models.boltz2.api import _confidence_index, featurize
    from foldjax.models.boltz2.data.write.structure import write_prediction

    features, record_id, struct_dir = featurize(
        seq=["MKV", "MKV"], ligand_ccd=["ZN", "ATP"], mols=mols, out_dir=tmp_path
    )
    structure_npz = struct_dir / f"{record_id}.npz"
    index = _confidence_index(features, structure_npz)
    atom_mask = np.asarray(features["atom_pad_mask"]).reshape(-1)
    path = write_prediction(
        structure_npz=structure_npz,
        coords=np.zeros((atom_mask.size, 3), dtype=np.float32),
        atom_pad_mask=atom_mask,
        out_path=tmp_path / "out.cif",
        fmt="cif",
    )
    site = gemmi.cif.read_file(str(path)).sole_block().get_mmcif_category(
        "_atom_site."
    )
    owners = index["atom_token_index"]
    assert len(owners) == len(site["auth_asym_id"])
    assert index["token_chain_id"][owners].tolist() == site["auth_asym_id"]
    assert index["token_residue_index"][owners].tolist() == [
        int(value) for value in site["auth_seq_id"]
    ]


# --- Protenix and OpenDDE (one writer) ----------------------------------------


def _protenix_features():
    from foldjax.models.protenix.data.featurize_json import featurize_protein_json

    return featurize_protein_json(
        {
            "name": "arrays",
            "sequences": [
                {"proteinChain": {"sequence": "CAG", "count": 2, "id": ["P", "Q"]}},
                {"ion": {"ion": "MG", "count": 1, "id": ["M"]}},
            ],
        },
        n_queries=2,
        n_keys=2,
    )


def _protenix_output(n_atom: int, n_token: int, *, details: bool) -> dict:
    samples, chains = 2, 3
    output = {
        "coordinate": np.zeros((samples, n_atom, 3), dtype=np.float32),
        "atom_plddt": np.stack(
            [np.linspace(0.1, 0.9, n_atom), np.linspace(0.9, 0.1, n_atom)]
        ).astype(np.float32),
        # Sample 1 ranks first, so the native suffixes are reversed.
        "summary_ranking_score": np.asarray([0.2, 0.8], dtype=np.float32),
        "chain_ptm": np.arange(samples * chains, dtype=np.float32).reshape(2, 3),
        "chain_pair_iptm": np.ones((samples, chains, chains), dtype=np.float32),
    }
    if details:
        output["token_pair_pae"] = np.stack(
            [np.full((n_token, n_token), 2.0), np.full((n_token, n_token), 9.0)]
        ).astype(np.float32)
        output["contact_probs"] = np.full((n_token, n_token), 0.25, np.float32)
    return output


@pytest.mark.parametrize("details", [False, True])
def test_protenix_writer_stages_arrays_in_cif_atom_order(
    tmp_path: Path, details: bool
) -> None:
    gemmi = pytest.importorskip("gemmi")
    from foldjax.models.protenix.data.output import write_protenix_outputs

    features = _protenix_features()
    owners = np.asarray(features["atom_to_token_idx"])
    n_atom, n_token = owners.size, int(owners.max()) + 1
    output = _protenix_output(n_atom, n_token, details=details)
    written = write_protenix_outputs(
        tmp_path, job_name="arrays", seed=1, output=output, features=features
    )
    cifs = [path for path in written if path.suffix == ".cif"]
    assert [path.stem[-1] for path in cifs] == ["1", "0"]  # diffusion order
    assert not [path for path in written if path.suffix == ".npz"]

    for sample_index, cif in enumerate(cifs):
        entry = confidence_arrays.sample_metadata(cif)["confidence_arrays"]
        loaded = load_confidence_arrays(Path(entry["path"]))
        assert loaded.model == "protenix"
        assert loaded.meta["sample"] == {
            "sample": sample_index,
            "native_rank": 1 - sample_index,
        }
        np.testing.assert_allclose(
            loaded["atom_plddt"], output["atom_plddt"][sample_index]
        )
        assert loaded.describe("atom_plddt")["scale"] == "0-1"
        np.testing.assert_array_equal(
            loaded["chain_ptm"], output["chain_ptm"][sample_index]
        )
        assert loaded["chain_id"].tolist() == ["P", "Q", "M"]
        site = gemmi.cif.read_file(str(cif)).sole_block().get_mmcif_category(
            "_atom_site."
        )
        assert loaded["atom_chain_id"].tolist() == site["auth_asym_id"]
        assert loaded["atom_residue_index"].tolist() == [
            int(value) for value in site["auth_seq_id"]
        ]
        if details:
            np.testing.assert_array_equal(
                loaded["pae"], output["token_pair_pae"][sample_index]
            )
            assert loaded["contact_probs"].shape == (n_token, n_token)
            owners_read = loaded["atom_token_index"]
            assert loaded["token_chain_id"][owners_read].tolist() == (
                site["auth_asym_id"]
            )
            assert "pae" not in loaded.unavailable
        else:
            assert "pae" not in loaded
            assert "output_format" in loaded.unavailable["pae"]


def _assert_maps_name_every_cif_atom(loaded, cif: Path) -> None:
    gemmi = pytest.importorskip("gemmi")
    site = gemmi.cif.read_file(str(cif)).sole_block().get_mmcif_category(
        "_atom_site."
    )
    owners = loaded["atom_token_index"]
    assert len(owners) == len(site["auth_asym_id"])
    assert loaded["token_chain_id"][owners].tolist() == site["auth_asym_id"]
    assert loaded["token_residue_index"][owners].tolist() == [
        int(value) for value in site["auth_seq_id"]
    ]


# --- ESMFold2 -------------------------------------------------------------------


def _esmfold2_output(built, samples: int = 2) -> dict:
    n_atom = int(np.asarray(built["atom_attention_mask"]).sum())
    n_token = int(np.asarray(built["token_attention_mask"]).sum())
    return {
        "sample_atom_coords": np.zeros((samples, n_atom, 3), dtype=np.float32),
        "plddt": np.stack([np.full(n_token, 0.25), np.full(n_token, 0.75)]),
        "plddt_per_atom": np.stack([np.full(n_atom, 0.2), np.full(n_atom, 0.8)]),
        "ptm": np.asarray([0.5, 0.6]),
    }


def test_esmfold2_writer_stages_plddt_and_reports_withheld_pae(
    tmp_path: Path,
) -> None:
    from foldjax.models.esmfold2.data import features
    from foldjax.models.esmfold2.output import write_prediction_outputs

    built = features.build_features([("ACDK", "A", 0, 0), ("GHK", "B", 0, 1)])
    written = write_prediction_outputs(
        _esmfold2_output(built), built, tmp_path, name="j"
    )

    for index, cif in enumerate(written["structures"]):
        entry = confidence_arrays.sample_metadata(cif)["confidence_arrays"]
        assert entry["arrays"] == ["atom_plddt", "token_plddt"]
        loaded = load_confidence_arrays(Path(entry["path"]))
        assert loaded.model == "esmfold2"
        assert loaded["token_plddt"].tolist() == [[0.25, 0.75][index]] * 7
        assert loaded.describe("atom_plddt")["scale"] == "0-1"
        assert "return_confidence_logits" in loaded.unavailable["pae"]
        _assert_maps_name_every_cif_atom(loaded, cif)


def test_esmfold2_all_biomolecule_maps_and_direct_pae(
    tmp_path: Path, monkeypatch
) -> None:
    """A direct caller that compiled `pae` gets it; ligands map atom by atom."""
    from foldjax.models.esmfold2.data import all_atom
    from foldjax.models.esmfold2.output import write_prediction_outputs
    from tests.models.esmfold2.test_all_atom_features import (
        _FakeCCD,
        _mixed_document,
    )

    monkeypatch.setattr(all_atom, "get_ccd_store", _FakeCCD)
    built = all_atom.build_job_features(
        _mixed_document(), base_dir=".", ccd_path="unused.pkl", seed=7
    )
    output = _esmfold2_output(built)
    n_token = output["plddt"].shape[-1]
    output["pae"] = np.full((2, n_token, n_token), 3.5, dtype=np.float32)
    written = write_prediction_outputs(output, built, tmp_path, name="mixed")

    loaded = load_confidence_arrays(
        confidence_arrays.staged_path(written["structures"][1])
    )
    assert loaded["pae"].shape == (n_token, n_token)
    assert "pae" not in loaded.unavailable
    assert set(loaded["token_chain_id"].tolist()) == {
        "PROT",
        "DNA",
        "RNA",
        "ATPCHAIN",
        "SMILES",
    }
    _assert_maps_name_every_cif_atom(loaded, written["structures"][1])


@pytest.mark.parametrize("multimer", [True, False])
def test_openfold3_writer_stages_atom_plddt_and_chain_pair_iptm(
    tmp_path: Path, multimer: bool
) -> None:
    from foldjax.models.openfold3.output import write_prediction_outputs
    from tests.test_mmcif_label_fields import openfold3_case

    chain_pair = (
        np.asarray([[[0.0, 0.3], [0.3, 0.0]], [[0.0, 0.8], [0.8, 0.0]]])
        if multimer
        else None
    )
    prediction, features, metadata = openfold3_case(
        samples=2, chain_pair_iptm=chain_pair
    )
    written = write_prediction_outputs(
        prediction, features, tmp_path, output_metadata=metadata
    )

    for index, cif in enumerate(written["structures"]):
        entry = confidence_arrays.sample_metadata(cif)["confidence_arrays"]
        loaded = load_confidence_arrays(Path(entry["path"]))
        assert loaded.model == "openfold3"
        np.testing.assert_allclose(loaded["atom_plddt"], prediction.plddt[index])
        assert loaded.describe("atom_plddt")["scale"] == "0-1"
        assert loaded["chain_id"].tolist() == ["P", "L"]
        assert loaded["atom_chain_id"].tolist() == ["P"] * 4 + ["L"]
        assert "all_arrays" in loaded.unavailable["pae"]
        if multimer:
            np.testing.assert_allclose(loaded["chain_pair_iptm"], chain_pair[index])
        else:
            assert "chain_pair_iptm" in loaded.unavailable
        _assert_maps_name_every_cif_atom(loaded, cif)


# --- AlphaFold 3 ----------------------------------------------------------------


def _alphafold3_structure():
    try:
        from foldjax.models.alphafold3 import build

        build.register_runtime()
        from alphafold3 import structure
        from alphafold3.constants import chemical_components

        ccd = chemical_components.Ccd()
    except Exception as error:  # noqa: BLE001 - any missing native piece skips
        pytest.skip(f"AlphaFold 3 native runtime is not available: {error}")
    built = structure.from_sequences_and_bonds(
        sequences=["MKV", "GA", "ZN"],
        chain_types=["polypeptide(L)", "polypeptide(L)", "non-polymer"],
        sequence_formats=[
            structure.SequenceFormat.FASTA,
            structure.SequenceFormat.FASTA,
            structure.SequenceFormat.CCD_CODES,
        ],
        bonded_atom_pairs=None,
        ccd=ccd,
    )
    plddt = np.linspace(10.0, 90.0, built.num_atoms).astype(np.float32)
    return built.copy_and_update_atoms(atom_b_factor=plddt)


def test_alphafold3_routes_the_inference_result_arrays(tmp_path: Path) -> None:
    from foldjax.backends.alphafold3 import _samples

    predicted = _alphafold3_structure()
    n_token, n_chain = 6, 3
    pae = np.arange(n_token * n_token, dtype=np.float32).reshape(n_token, n_token) / 2
    result = SimpleNamespace(
        predicted_structure=predicted,
        numerical_data={
            "full_pae": pae,
            "full_pde": pae + 1,
            "contact_probs": np.full((n_token, n_token), 0.5, np.float32),
        },
        metadata={
            "ranking_score": 0.9,
            "chain_pair_iptm": np.eye(n_chain, dtype=np.float32),
            "chain_pair_pae_min": np.ones((n_chain, n_chain), np.float32),
            "iptm_ichain": np.asarray([0.1, 0.2, 0.3], np.float32),
            "iptm_xchain": np.asarray([0.4, 0.5, 0.6], np.float32),
            "token_chain_ids": ["A", "A", "A", "B", "B", "C"],
            "token_res_ids": np.asarray([1, 2, 3, 1, 2, 1]),
        },
    )
    sample_dir = tmp_path / "seed-5_sample-0"
    sample_dir.mkdir()
    cif = sample_dir / "job_seed-5_sample-0_model.cif"
    cif.write_text(predicted.to_mmcif())

    (sample,) = _samples(
        (SimpleNamespace(seed=5, inference_results=(result,)),), tmp_path, "job"
    )

    entry = sample.metadata["confidence_arrays"]
    loaded = load_confidence_arrays(Path(entry["path"]))
    assert loaded.model == "alphafold3"
    np.testing.assert_allclose(loaded["pae"], pae, atol=0.008)
    assert loaded.describe("pae")["source"] == "full_pae"
    np.testing.assert_allclose(loaded["chain_ptm"], [0.1, 0.2, 0.3], rtol=1e-6)
    assert loaded.describe("chain_ptm")["source"] == "iptm_ichain"
    assert loaded["chain_id"].tolist() == ["A", "B", "C"]
    assert loaded["token_chain_id"].tolist() == ["A", "A", "A", "B", "B", "C"]
    assert loaded.describe("atom_plddt")["scale"] == "0-100"
    assert "atom_token_index" in loaded.unavailable

    gemmi = pytest.importorskip("gemmi")
    site = gemmi.cif.read_file(str(cif)).sole_block().get_mmcif_category(
        "_atom_site."
    )
    assert loaded["atom_chain_id"].tolist() == site["auth_asym_id"]
    assert loaded["atom_residue_index"].tolist() == [
        int(value) for value in site["auth_seq_id"]
    ]
    np.testing.assert_allclose(
        loaded["atom_plddt"],
        [float(value) for value in site["B_iso_or_equiv"]],
        atol=0.01,
    )


def test_opendde_writer_names_its_own_model(tmp_path: Path) -> None:
    from foldjax.models.opendde import runner

    features = _protenix_features()
    owners = np.asarray(features["atom_to_token_idx"])
    output = _protenix_output(owners.size, int(owners.max()) + 1, details=False)
    output.pop("chain_pair_iptm")
    written = runner._write(
        tmp_path, job_name="arrays", seed=1, output=output, features=features
    )
    cif = next(path for path in written if path.suffix == ".cif")
    loaded = load_confidence_arrays(confidence_arrays.staged_path(cif))
    assert loaded.model == "opendde"
    assert "include_raw" in loaded.unavailable["pae"]
