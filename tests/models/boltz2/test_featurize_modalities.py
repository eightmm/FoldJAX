from pathlib import Path

import numpy as np
import pytest

from foldjax.models.boltz2.api import featurize
from foldjax.models.boltz2.data.featurize import (
    featurize_affinity_from_prediction,
    featurize_yaml,
)
from foldjax.models.boltz2.data.types import StructureV2

ROOT = Path(__file__).resolve().parents[4]
MOLS = ROOT / "boltz/.cache/boltz/mols"
TEMPLATE = (
    ROOT
    / "boltz/benchmark_results/real_pdb_eval_expanded_2026-06-04"
    / "references/1UBQ.cif"
)
UBIQUITIN = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)


@pytest.fixture(scope="module", autouse=True)
def require_setup_assets() -> None:
    if not MOLS.is_dir() or not TEMPLATE.is_file():
        pytest.skip("setup molecule DB and template fixture are not available")


@pytest.mark.slow
def test_raw_mixed_entities_cover_all_supported_molecule_types(tmp_path) -> None:
    feats, record_id, struct_dir = featurize(
        seq=["ACDE", "FGHI"],
        dna=["ACGT"],
        rna=["ACGU"],
        ligand_ccd=["ATP"],
        ligand_smiles=["CC(=O)O"],
        mols=MOLS,
        out_dir=tmp_path,
    )

    assert record_id == "job"
    assert (struct_dir / "job.npz").is_file()
    assert set(np.unique(feats["mol_type"])) == {0, 1, 2, 3}
    assert int(feats["token_pad_mask"].sum()) > 16
    assert int(feats["atom_pad_mask"].sum()) > int(feats["token_pad_mask"].sum())
    assert np.isfinite(feats["ref_pos"]).all()


@pytest.mark.slow
def test_precomputed_msa_and_feature_cache_are_deterministic(tmp_path) -> None:
    a3m = tmp_path / "query.a3m"
    a3m.write_text(">query\nACDEFG\n>homolog\nACDEYG\n")
    job = tmp_path / "msa.yaml"
    job.write_text(
        "version: 1\nsequences:\n  - protein:\n      id: A\n"
        f"      sequence: ACDEFG\n      msa: {a3m}\n"
    )
    cache = tmp_path / "cache"

    first, first_id, _ = featurize(
        input=job, mols=MOLS, out_dir=tmp_path / "first", feature_cache=cache
    )
    second, second_id, _ = featurize(
        input=job, mols=MOLS, out_dir=tmp_path / "second", feature_cache=cache
    )

    assert first_id == second_id == "msa"
    assert first["msa"].shape[1] == 2
    assert np.any(first["msa"][0, 0] != first["msa"][0, 1])
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key], err_msg=key)


@pytest.mark.slow
def test_remote_msa_server_reaches_model_features(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_run_mmseqs2(sequences, prefix, **kwargs):
        calls.append((sequences, Path(prefix), kwargs))
        assert sequences == ["ACDEFG"]
        return [">101\nACDEFG\n>homolog\nACDEYG\n"]

    monkeypatch.setattr(
        "foldjax.models.boltz2.data.msa.mmseqs2.run_mmseqs2", fake_run_mmseqs2
    )
    job = tmp_path / "remote.yaml"
    job.write_text(
        "version: 1\nsequences:\n  - protein:\n      id: A\n"
        "      sequence: ACDEFG\n"
    )

    feats, record_id, _ = featurize(
        input=job,
        mols=MOLS,
        out_dir=tmp_path / "out",
        use_msa_server=True,
        msa_server_url="https://msa.example.test",
        msa_server_username="user",
        msa_server_password="password",
    )

    assert record_id == "remote"
    assert feats["msa"].shape[1] == 2
    assert np.any(feats["msa"][0, 0] != feats["msa"][0, 1])
    assert len(calls) == 1  # a monomer needs only the unpaired search
    assert calls[0][2]["use_pairing"] is False
    assert calls[0][2]["host_url"] == "https://msa.example.test"
    assert calls[0][2]["msa_server_username"] == "user"
    assert calls[0][2]["msa_server_password"] == "password"
    generated = tmp_path / "out" / "msa" / "remote_0.csv"
    assert generated.read_text().splitlines() == [
        "key,sequence",
        "-1,ACDEFG",
        "-1,ACDEYG",
    ]


@pytest.mark.slow
def test_remote_msa_server_pairs_multimer_entities(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_run_mmseqs2(sequences, prefix, **kwargs):
        calls.append(kwargs)
        if kwargs["use_pairing"]:
            return [
                ">101\nACDE\n>paired\nACDF\n",
                ">102\nFGHI\n>paired\nFGHV\n",
            ]
        return [
            ">101\nACDE\n>unpaired\nACDG\n",
            ">102\nFGHI\n>unpaired\nFGHK\n",
        ]

    monkeypatch.setattr(
        "foldjax.models.boltz2.data.msa.mmseqs2.run_mmseqs2", fake_run_mmseqs2
    )
    job = tmp_path / "complex.yaml"
    job.write_text(
        "version: 1\nsequences:\n"
        "  - protein:\n      id: A\n      sequence: ACDE\n"
        "  - protein:\n      id: B\n      sequence: FGHI\n"
    )

    feats, _, _ = featurize(
        input=job,
        mols=MOLS,
        out_dir=tmp_path / "out",
        use_msa_server=True,
        msa_pairing_strategy="complete",
    )

    assert [call["use_pairing"] for call in calls] == [True, False]
    assert calls[0]["pairing_strategy"] == "complete"
    assert float(feats["msa_paired"].sum()) > 0
    assert feats["msa"].shape[1] >= 3


@pytest.mark.slow
def test_template_and_affinity_inputs_reach_model_features(tmp_path) -> None:
    template_job = tmp_path / "template.yaml"
    template_job.write_text(
        "version: 1\nsequences:\n  - protein:\n      id: A\n"
        f"      sequence: {UBIQUITIN}\n      msa: empty\n"
        f"templates:\n  - cif: {TEMPLATE}\n"
    )
    template_feats, _, _ = featurize_yaml(
        template_job, tmp_path / "template", MOLS
    )

    affinity_job = ROOT / "boltz_jax/outputs/prep/affinity_no_msa.yaml"
    affinity_feats, affinity_manifest, affinity_struct_dir = featurize_yaml(
        affinity_job, tmp_path / "affinity", MOLS
    )

    assert float(template_feats["template_mask"].sum()) > 0
    assert float(template_feats["template_mask_frame"].sum()) > 0
    assert int(affinity_feats["affinity_token_mask"].sum()) > 0
    assert 3 in set(np.unique(affinity_feats["mol_type"]))

    structure = StructureV2.load(
        affinity_struct_dir / f"{affinity_manifest.records[0].id}.npz"
    )
    predicted = np.zeros((*affinity_feats["atom_pad_mask"].shape, 3), np.float32)
    predicted[0, affinity_feats["atom_pad_mask"][0].astype(bool)] = structure.coords[
        "coords"
    ]
    cropped = featurize_affinity_from_prediction(
        manifest=affinity_manifest,
        processed_dir=affinity_struct_dir.parent,
        mol_dir=MOLS,
        predicted_coords=predicted,
        atom_pad_mask=affinity_feats["atom_pad_mask"],
        out_dir=tmp_path / "affinity_second_stage",
    )

    assert "profile_affinity" in cropped
    assert "deletion_mean_affinity" in cropped
    assert "affinity_mw" in cropped
    assert cropped["token_pad_mask"].shape[1] <= 256
    assert cropped["atom_pad_mask"].shape[1] <= 2048
