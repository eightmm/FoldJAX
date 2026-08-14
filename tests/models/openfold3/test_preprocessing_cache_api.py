"""Public, pickle-free OpenFold3 preprocessing cache contracts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from foldjax.models.openfold3.data import (
    _numpy_featurization as numpy_features,
)
from foldjax.models.openfold3.data import (
    featurize_query,
    save_preparsed_msas,
    save_template_cache,
)

SEQUENCE = "ACDEFGHIK"


def _spec(**chain_extra) -> dict:
    return {
        "queries": {
            "query": {
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": ["A"],
                        "sequence": SEQUENCE,
                        **chain_extra,
                    }
                ]
            }
        }
    }


def test_save_preparsed_msas_is_public_pickle_free_and_roundtrips(
    tmp_path: Path,
) -> None:
    archive = save_preparsed_msas(
        {
            "colabfold_main": {
                "msa": [SEQUENCE, "A" * len(SEQUENCE)],
                "deletion_matrix": np.zeros((2, len(SEQUENCE)), dtype=np.int32),
                "metadata": ["query", "hit"],
            }
        },
        tmp_path / "representative",
    )

    assert archive == tmp_path / "representative.npz"
    with np.load(archive, allow_pickle=False) as stored:
        assert set(stored.files) == {
            "colabfold_main__msa",
            "colabfold_main__deletion_matrix",
            "colabfold_main__metadata",
        }
        assert all(stored[name].dtype.kind != "O" for name in stored.files)

    features = featurize_query(_spec(main_msa_file_paths=[str(archive)]))
    assert features["msa"].shape[1] > 1


def test_save_preparsed_msas_rejects_inconsistent_shapes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="deletion_matrix"):
        save_preparsed_msas(
            {
                "colabfold_main": {
                    "msa": [SEQUENCE],
                    "deletion_matrix": np.zeros((1, 2), dtype=np.int32),
                }
            },
            tmp_path / "bad.npz",
        )


def test_template_cache_skips_invalid_requested_ids_until_four_valid(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    entries = {}
    for name in ("one_A", "two_A", "three_A", "four_A", "five_A"):
        cif_path = tmp_path / f"{name.removesuffix('_A')}.cif"
        cif_path.write_text("local fixture")
        entries[name] = {
            "idx_map": [[1, 1]],
            "release_date": "2000-01-01",
            "cif_path": cif_path,
        }
    cache = save_template_cache(entries, tmp_path / "templates.npz")

    with np.load(cache, allow_pickle=False) as archive:
        manifest = json.loads(str(np.asarray(archive["entries_json"]).item()))
    manifest["broken_A"] = {
        "idx_map": [[1.5, 1]],
        "release_date": "2000-01-01",
        "cif_path": str(tmp_path / "one.cif"),
    }
    np.savez(cache, entries_json=np.asarray(json.dumps(manifest)))

    requested = [
        "missing_A",
        "broken_A",
        "five_A",
        "four_A",
        "three_A",
        "two_A",
        "one_A",
    ]
    with caplog.at_level(logging.WARNING):
        selected, _ = numpy_features._preprocessed_template_entries(
            cache, requested
        )

    assert list(selected) == ["five_A", "four_A", "three_A", "two_A"]
    assert [entry.index for entry in selected.values()] == [0, 1, 2, 3]
    assert "missing_A" in caplog.text
    assert "broken_A" in caplog.text


def test_template_cache_all_unknown_requested_ids_fail_clearly(
    tmp_path: Path,
) -> None:
    cif_path = tmp_path / "one.cif"
    cif_path.write_text("local fixture")
    cache = save_template_cache(
        {"one_A": {"idx_map": [[1, 1]], "cif_path": cif_path}},
        tmp_path / "templates.npz",
    )

    with pytest.raises(ValueError, match="none of the requested.*missing_A"):
        numpy_features._preprocessed_template_entries(cache, ["missing_A"])


def test_malformed_alignment_is_not_reinterpreted_as_requested_cifs(
    tmp_path: Path,
) -> None:
    alignment = tmp_path / "templates.a3m"
    alignment.write_text(f">query\n{SEQUENCE}\n")

    with pytest.raises(
        ValueError, match="failed to parse OpenFold3 template alignment"
    ):
        numpy_features._alignment_template_entries(
            alignment, SEQUENCE, ["1ubq_A"]
        )


def test_direct_template_runtime_error_is_not_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_systemically(*_args, **_kwargs):
        raise RuntimeError("parser installation is broken")

    monkeypatch.setattr(
        numpy_features, "_template_entry_from_cif", fail_systemically
    )
    with pytest.raises(RuntimeError, match="parser installation is broken"):
        numpy_features._rank_direct_template_entries(
            [Path("template.cif")], [], SEQUENCE
        )


def test_template_precursor_reports_all_malformed_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import biotite.structure as struc

    def malformed_template(_template):
        raise ValueError("malformed local template")

    monkeypatch.setattr(struc, "get_residue_starts", malformed_template)
    template_slice = SimpleNamespace(
        atom_array=object(),
        template_residue_repeats=[1],
        query_token_positions=[0],
    )

    with pytest.raises(ValueError, match="none of the aligned.*malformed local"):
        numpy_features._template_precursor({"A": [template_slice]}, 4, 1)


def test_template_precursor_does_not_hide_systemic_runtime_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import biotite.structure as struc

    def unavailable_parser(_template):
        raise RuntimeError("structure runtime unavailable")

    monkeypatch.setattr(struc, "get_residue_starts", unavailable_parser)
    template_slice = SimpleNamespace(
        atom_array=object(),
        template_residue_repeats=[1],
        query_token_positions=[0],
    )

    with pytest.raises(RuntimeError, match="structure runtime unavailable"):
        numpy_features._template_precursor({"A": [template_slice]}, 4, 1)
