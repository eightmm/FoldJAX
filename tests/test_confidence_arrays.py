"""`confidence_full.npz`: the confidence arrays a model already computes, on disk.

What is checked here is the contract a reader relies on: the arrays come back
by name with their axes, unit, scale and stored dtype; the archive lands in the
canonical sample directory beside `confidence.json`; the manifest and
capabilities say which arrays a model provides; and each backend routes the
arrays its program returned without inventing any.
"""

from __future__ import annotations

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
