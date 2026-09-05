"""Template featurization parity tests against torch Protenix goldens.

Goldens were produced by ``InferenceTemplateFeaturizer`` (JSON template path,
``parse_json_templates`` -> ``Templates.as_protenix_dict``) on the embedded
template and query, and stored in ``tests/fixtures``.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest

from foldjax.models.protenix.data.featurize_json import featurize_protein_json
from foldjax.models.protenix.data.template_features import (
    ATOM37_ORDER,
    _parse_template_mmcif,
)

_FIXTURES = Path(__file__).parent / "fixtures"
_TEMPLATE_JSON = _FIXTURES / "template_small.json"
_GOLDEN_B64 = _FIXTURES / "template_small_golden.b64"
_QUERY = "AGSRF"
_TEMPLATE_KEYS = (
    "template_aatype",
    "template_atom_positions",
    "template_atom_mask",
    "template_pseudo_beta_mask",
    "template_distogram",
    "template_unit_vector",
    "template_backbone_frame_mask",
)


def _load_golden() -> dict[str, np.ndarray]:
    raw = base64.b64decode(_GOLDEN_B64.read_text())
    with np.load(io.BytesIO(raw)) as data:
        return {k: data[k] for k in data.files}


def test_template_features_match_torch_golden() -> None:
    job = {
        "name": "tpl",
        "sequences": [
            {
                "proteinChain": {
                    "sequence": _QUERY,
                    "count": 1,
                    "templatesPath": str(_TEMPLATE_JSON),
                },
            },
        ],
    }
    features = featurize_protein_json(job)
    golden = _load_golden()

    for key in _TEMPLATE_KEYS:
        assert key in features, key
        got = np.asarray(features[key]).astype(np.float64)
        ref = golden[key].astype(np.float64)
        assert got.shape == ref.shape, (key, got.shape, ref.shape)
        np.testing.assert_allclose(got, ref, atol=1e-5, err_msg=key)

    assert features["template_aatype"].shape == (4, 5)
    assert features["template_distogram"].shape == (4, 5, 5, 39)
    assert features["template_unit_vector"].shape == (4, 5, 5, 3)


def test_dummy_template_emitted_without_templates() -> None:
    # torch always embeds a single fully-masked gap template when none is
    # provided (make_dummy_feature / empty_template_features): aatype slot 0 is
    # the gap residue (31), remaining slots zero-padded, all 2D features zero.
    # The trunk TemplateEmbedder runs over these, so the features must be
    # present (a previous return-None skip diverged from torch by ~128 in z).
    job = {
        "name": "no_tpl",
        "sequences": [
            {"proteinChain": {"sequence": "AGSRF", "count": 1}},
        ],
    }
    features = featurize_protein_json(job)
    for key in _TEMPLATE_KEYS:
        assert key in features, key
    aatype = np.asarray(features["template_aatype"])
    assert aatype.shape == (4, 5)
    assert (aatype[0] == 31).all()  # gap template
    assert (aatype[1:] == 0).all()  # zero padding
    for key in ("template_distogram", "template_unit_vector"):
        assert not np.asarray(features[key]).any()


def test_template_short_chain_skipped_emits_dummy() -> None:
    # Length <= 4 protein chains are skipped, so no real template remains; torch
    # still emits the single fully-masked gap template placeholder.
    job = {
        "name": "short",
        "sequences": [
            {
                "proteinChain": {
                    "sequence": "AGSR",
                    "count": 1,
                    "templatesPath": str(_TEMPLATE_JSON),
                },
            },
        ],
    }
    features = featurize_protein_json(job)
    aatype = np.asarray(features["template_aatype"])
    assert aatype.shape == (4, 4)
    assert (aatype[0] == 31).all()


def _write_local_template_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, observed_start: int = 1
) -> Path:
    database = tmp_path / "mmcif"
    database.mkdir()
    sequence = "AGSRFAGSRF"
    residue_names = {"A": "ALA", "G": "GLY", "S": "SER", "R": "ARG", "F": "PHE"}
    mmcif = [
        "data_1abc",
        "_entry.id 1abc",
        "loop_",
        "_entity.id",
        "_entity.type",
        "1 polymer",
        "loop_",
        "_entity_poly.entity_id",
        "_entity_poly.type",
        "_entity_poly.pdbx_strand_id",
        "_entity_poly.pdbx_seq_one_letter_code",
        f"1 'polypeptide(L)' X {sequence}",
        "loop_",
        "_entity_poly_seq.entity_id",
        "_entity_poly_seq.num",
        "_entity_poly_seq.mon_id",
        *[
            f"1 {index} {residue_names[residue]}"
            for index, residue in enumerate(sequence, start=1)
        ],
        "loop_",
        "_struct_asym.id",
        "_struct_asym.entity_id",
        "X 1",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.pdbx_formal_charge",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_asym_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    atom_id = 1
    for residue_id, residue in enumerate(sequence, start=1):
        if residue_id < observed_start:
            continue
        x = residue_id * 3.8
        for element, name, dx, dy in (
            ("N", "N", 0.0, 0.0),
            ("C", "CA", 1.46, 0.0),
            ("C", "C", 2.0, 1.4),
            ("O", "O", 1.3, 2.4),
        ):
            mmcif.append(
                f"ATOM {atom_id} {element} {name} . {residue_names[residue]} "
                f"X 1 {residue_id} ? {x + dx} {dy} 0 1 20 ? "
                f"{residue_id} X 1"
            )
            atom_id += 1
    (database / "1abc.cif").write_text("\n".join(mmcif) + "\n")
    release_dates = tmp_path / "release_dates.json"
    release_dates.write_text(json.dumps({"1abc": {"release_date": "2020-01-01"}}))
    obsolete = tmp_path / "obsolete.json"
    obsolete.write_text("{}")
    monkeypatch.setenv("PROTENIX_TEMPLATE_MMCIF_DIR", str(database))
    monkeypatch.setenv("PROTENIX_TEMPLATE_RELEASE_DATES_FILE", str(release_dates))
    monkeypatch.setenv("PROTENIX_TEMPLATE_OBSOLETE_FILE", str(obsolete))
    binary = tmp_path / "kalign"
    binary.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib\n"
        "import sys\n"
        "if '--version' in sys.argv:\n"
        "    print('kalign 3.3.5')\n"
        "    raise SystemExit(0)\n"
        "source = pathlib.Path(sys.argv[sys.argv.index('-i') + 1])\n"
        "target = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])\n"
        "target.write_text(source.read_text())\n"
    )
    binary.chmod(0o755)
    monkeypatch.setenv("PROTENIX_KALIGN_BINARY", str(binary))
    return database


def test_template_a3m_resolves_local_mmcif_coordinates(tmp_path, monkeypatch) -> None:
    _write_local_template_db(tmp_path, monkeypatch)
    artifact = tmp_path / "tpl.a3m"
    artifact.write_text(">query\nAGSRFAGSRF\n>1abc_X mol:protein 1-10\nCCCCCCCCCC\n")
    job = {
        "name": "a3m",
        "sequences": [
            {
                "proteinChain": {
                    "sequence": "AGSRFAGSRF",
                    "count": 1,
                    "templatesPath": str(artifact),
                },
            },
        ],
    }
    features = featurize_protein_json(job)
    assert features["template_atom_mask"][0].any()
    assert features["template_aatype"][0].tolist() == [0, 7, 15, 1, 13, 0, 7, 15, 1, 13]


def test_template_hhr_realigns_query_against_mmcif_seqres(
    tmp_path, monkeypatch
) -> None:
    _write_local_template_db(tmp_path, monkeypatch)
    artifact = tmp_path / "tpl.hhr"
    artifact.write_text(
        "No 1\n>1abc_X template\n"
        "Probab=99 Aligned_cols=9 Sum_probs=8.8\n"
        "Q query 1 AG-RFAGSRF 9 (10)\n"
        "T 1abc_X 1 CCCCCCCCCC 10 (10)\n"
    )
    features = featurize_protein_json(
        {
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": "AGSRFAGSRF",
                        "templatesPath": str(artifact),
                    }
                }
            ]
        }
    )
    assert features["template_atom_mask"][0].any(axis=1).all()


def test_template_mmcif_seqres_indices_preserve_missing_residue_masks(
    tmp_path, monkeypatch
) -> None:
    _write_local_template_db(tmp_path, monkeypatch, observed_start=3)
    artifact = tmp_path / "tpl.a3m"
    artifact.write_text(">query\nAGSRFAGSRF\n>1abc_X mol:protein 1-10\nCCCCCCCCCC\n")
    features = featurize_protein_json(
        {
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": "AGSRFAGSRF",
                        "templatesPath": str(artifact),
                    }
                }
            ]
        }
    )
    residue_mask = features["template_atom_mask"][0].any(axis=1)
    assert residue_mask.tolist() == [False, False, *([True] * 8)]


def test_template_mmcif_uses_first_equal_occupancy_alt_conformer(
    tmp_path, monkeypatch
) -> None:
    database = _write_local_template_db(tmp_path, monkeypatch)
    mmcif = (database / "1abc.cif").read_text()
    original = "ATOM 2 C CA . ALA X 1 1 ? 5.26 0.0 0 1 20 ? 1 X 1"
    alternates = (
        "ATOM 2 C CA A ALA X 1 1 ? 5.26 0.0 0 0.5 20 ? 1 X 1\n"
        "ATOM 999 C CA B ALA X 1 1 ? 99.0 0.0 0 0.5 20 ? 1 X 1"
    )
    assert original in mmcif
    _, positions, _ = _parse_template_mmcif(
        mmcif.replace(original, alternates), chain_id="X"
    )
    ca_minus_n = positions[0, ATOM37_ORDER["CA"]] - positions[0, ATOM37_ORDER["N"]]
    np.testing.assert_allclose(ca_minus_n, [1.46, 0.0, 0.0], atol=1e-6)


def test_mapped_json_uses_observed_indices_even_with_seqres(tmp_path, monkeypatch):
    database = _write_local_template_db(tmp_path, monkeypatch, observed_start=3)
    artifact = tmp_path / "mapped.json"
    artifact.write_text(
        json.dumps(
            [
                {
                    "mmcif": (database / "1abc.cif").read_text(),
                    # Native parse_simple_cif ignores even an absent chainId.
                    "chainId": "not-the-first-chain",
                    "queryIndices": list(range(10)),
                    "templateIndices": list(range(10)),
                }
            ]
        )
    )
    features = featurize_protein_json(
        {
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": "AGSRFAGSRF",
                        "templatesPath": str(artifact),
                    }
                }
            ]
        }
    )
    assert features["template_atom_mask"][0].any(axis=1).tolist() == [
        *([True] * 8),
        False,
        False,
    ]
    assert features["template_aatype"][0, :8].tolist() == [15, 1, 13, 0, 7, 15, 1, 13]


def test_mapped_json_merges_disjoint_auth_chain_segments(tmp_path, monkeypatch):
    database = _write_local_template_db(tmp_path, monkeypatch)
    mmcif = (database / "1abc.cif").read_text()
    mmcif += (
        "ATOM 999 N N . ALA Y 2 1 ? 100 0 0 1 20 ? 1 Y 1\n"
        "HETATM 1000 O O . HOH Z 3 . ? 200 0 0 1 20 ? 100 X 1\n"
    )
    sequence, positions, mask = _parse_template_mmcif(mmcif, observed_residues=True)
    assert sequence == "AGSRFAGSRFX"
    assert mask[-1, ATOM37_ORDER["O"]] == 1
    np.testing.assert_allclose(positions[mask.astype(bool)].mean(axis=0), 0, atol=1e-5)


def test_template_search_artifact_requires_local_mmcif_db(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("PROTENIX_TEMPLATE_MMCIF_DIR", raising=False)
    artifact = tmp_path / "tpl.a3m"
    artifact.write_text(">query\nAGSRF\n>1abc_X mol:protein 1-5\nAGSRF\n")
    try:
        featurize_protein_json(
            {
                "sequences": [
                    {
                        "proteinChain": {
                            "sequence": "AGSRF",
                            "templatesPath": str(artifact),
                        }
                    }
                ]
            }
        )
    except ValueError as exc:
        assert "PROTENIX_TEMPLATE_MMCIF_DIR" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing local template database error")


def test_template_json_fixture_is_self_contained() -> None:
    data = json.loads(_TEMPLATE_JSON.read_text())
    assert data[0]["mmcif"]
    assert len(data[0]["queryIndices"]) == len(data[0]["templateIndices"])
