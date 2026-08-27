"""Production preprocessing must stay usable when PyTorch cannot be imported."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from foldjax.models.openfold3.data import MODEL_FEATURES, featurize_query

SEQUENCE = "ACDEFGHIK"
TEMPLATE_SEQUENCE = "GAAG"
LOCAL_TEMPLATE_CIF = Path(__file__).with_name("fixtures") / "gaag.cif"


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


def test_seed_deterministically_controls_reference_conformer() -> None:
    first = featurize_query(_spec(), seed=7)
    repeated = featurize_query(_spec(), seed=7)
    changed = featurize_query(_spec(), seed=8)

    np.testing.assert_array_equal(first["ref_pos"], repeated["ref_pos"])
    assert not np.array_equal(first["ref_pos"], changed["ref_pos"])
    assert set(MODEL_FEATURES).issubset(first)


def test_raw_query_and_msa_execute_with_a_hard_torch_import_block(
    tmp_path: Path,
) -> None:
    alignment = tmp_path / "colabfold_main.a3m"
    alignment.write_text(
        f">query\n{SEQUENCE}\n>hit\n{'A' * len(SEQUENCE)}\n"
    )
    script = r"""
import builtins
import sys

original_import = builtins.__import__
def import_without_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError(f"production preprocessing imported {name}")
    return original_import(name, *args, **kwargs)
builtins.__import__ = import_without_torch

from foldjax.models.openfold3.data import MODEL_FEATURES, featurize_query

sequence = "ACDEFGHIK"
features = featurize_query({
    "queries": {"query": {"chains": [{
        "molecule_type": "protein",
        "chain_ids": ["A"],
        "sequence": sequence,
        "main_msa_file_paths": [sys.argv[1]],
    }]}}
}, seed=3)
assert set(MODEL_FEATURES).issubset(features)
assert features["msa"].shape[1] > 1
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(alignment)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_direct_cif_template_is_torch_free_and_nonempty() -> None:
    cif = LOCAL_TEMPLATE_CIF
    assert cif.is_file()

    script = r"""
import builtins
import sys

original_import = builtins.__import__
def import_without_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError(f"template preprocessing imported {name}")
    return original_import(name, *args, **kwargs)
builtins.__import__ = import_without_torch

import numpy as np
from foldjax.models.openfold3.data import featurize_query

sequence = "GAAG"
features = featurize_query({"queries": {"ubq": {"chains": [{
    "molecule_type": "protein",
    "chain_ids": ["A"],
    "sequence": sequence,
    "template_cif_paths": [sys.argv[1]],
    "template_cif_chain_ids": ["A"],
}]}}})
assert float(features["template_pseudo_beta_mask"].sum()) > 0
assert float(features["template_backbone_frame_mask"].sum()) > 0
assert float(features["template_distogram"].sum()) > 0
assert float(np.abs(features["template_unit_vector"]).sum()) > 0
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(cif)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_preprocessed_template_cache_and_adjacent_structures_are_torch_free(
    tmp_path: Path,
) -> None:
    fixture = LOCAL_TEMPLATE_CIF
    assert fixture.is_file()

    base = tmp_path / "template_data"
    cache_directory = base / "template_cache"
    structure_directory = base / "template_structures"
    cache_directory.mkdir(parents=True)
    structure_directory.mkdir()
    shutil.copy2(fixture, structure_directory / "gaag.cif")
    cache = cache_directory / "ubq.npz"
    index_map = np.stack(
        (np.arange(1, 5, dtype=np.int64), np.arange(1, 5, dtype=np.int64)),
        axis=-1,
    )
    np.savez(
        cache,
        entries_json=np.asarray(
            json.dumps(
                {
                    "gaag_A": {
                        "index": 0,
                        "release_date": "1987-01-02",
                        "idx_map": index_map.tolist(),
                    }
                }
            )
        ),
    )

    script = r"""
import builtins
import sys

original_import = builtins.__import__
def import_without_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError(f"cached template preprocessing imported {name}")
    return original_import(name, *args, **kwargs)
builtins.__import__ = import_without_torch

from foldjax.models.openfold3.data import featurize_query

sequence = "GAAG"
features = featurize_query({"queries": {"ubq": {"chains": [{
    "molecule_type": "protein",
    "chain_ids": ["A"],
    "sequence": sequence,
    "template_alignment_file_path": sys.argv[1],
    "template_entry_chain_ids": ["gaag_A"],
}]}}})
assert float(features["template_pseudo_beta_mask"].sum()) > 0
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(cache)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_template_alignment_never_fetches_a_missing_cif(tmp_path: Path) -> None:
    alignment = tmp_path / "templates.a3m"
    alignment.write_text(f">query_X/1-{len(SEQUENCE)}\n{SEQUENCE}\n")

    with pytest.raises(FileNotFoundError, match="never fetches templates implicitly"):
        featurize_query(
            _spec(
                template_alignment_file_path=str(alignment),
                template_entry_chain_ids=["1ubq_A"],
            )
        )


def test_template_a3m_reads_only_the_records_the_parser_can_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from foldjax.models.openfold3._upstream.openfold3.core.data.io.sequence import (
        template as template_io,
    )
    from foldjax.models.openfold3.data import _numpy_featurization as numpy_features

    alignment = tmp_path / "templates.a3m"
    alignment.write_text(
        ">query/1-4\nACDE\n"
        ">first_A/1-4\nACDE\n"
        ">second_A/1-4\nACDE\n"
        ">tail-must-not-be-read_A/1-4\nACDE\n",
        encoding="utf-8",
    )
    captured: list[str] = []

    class RecordingParser:
        def __init__(self, max_sequences: int) -> None:
            assert max_sequences == 2

        def __call__(self, source: str, query_sequence: str) -> dict:
            assert query_sequence == "ACDE"
            captured.append(source)
            return {}

    monkeypatch.setattr(numpy_features, "_MAX_TEMPLATE_ALIGNMENT_HITS", 2)
    monkeypatch.setattr(template_io, "A3mParser", RecordingParser)

    assert numpy_features._alignment_templates(alignment, "ACDE") == []
    assert "second_A" in captured[0]
    assert "tail-must-not-be-read" not in captured[0]


def test_bounded_template_a3m_matches_the_full_parser_prefix(tmp_path: Path) -> None:
    from foldjax.models.openfold3._upstream.openfold3.core.data.io.sequence.template import (  # noqa: E501
        A3mParser,
    )
    from foldjax.models.openfold3.data import _numpy_featurization as numpy_features

    alignment = tmp_path / "templates.a3m"
    alignment.write_text(
        ">query_X/1-4\nACDE\n"
        + "".join(f">{index:04d}_A/1-4\nACDE\n" for index in range(300)),
        encoding="utf-8",
    )
    parsed = A3mParser(max_sequences=200)(alignment.read_text(), "ACDE")
    expected = list(parsed.values())[1:]

    actual = numpy_features._alignment_templates(alignment, "ACDE")

    assert len(actual) == len(expected) == 200
    for observed, reference in zip(actual, expected, strict=True):
        assert observed.entry_id == reference.entry_id
        assert observed.chain_id == reference.chain_id
        assert observed.seq == reference.seq
        np.testing.assert_array_equal(observed.query_aln_pos, reference.query_aln_pos)
        np.testing.assert_array_equal(observed.aln_pos, reference.aln_pos)
        assert observed.seq_id == reference.seq_id
        assert observed.q_cov == reference.q_cov


def test_direct_cif_candidates_are_quality_ranked_filtered_and_reindexed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from foldjax.models.openfold3._upstream.openfold3.core.data.primitives.structure.template import (  # noqa: E501
        TemplateCacheEntry,
    )
    from foldjax.models.openfold3.data import _numpy_featurization as numpy_features

    scores = {
        "low": 0.2,
        "best": 0.99,
        "invalid": None,
        "third": 0.7,
        "second": 0.8,
        "fourth": 0.6,
    }

    def fake_candidate(path, _query_sequence, *, preferred_chain_id):
        del preferred_chain_id
        score = scores[path.stem]
        if score is None:
            return None
        return numpy_features._TemplateCandidate(
            name=f"{path.stem}_A",
            entry=TemplateCacheEntry(
                index=0,
                release_date="2000-01-01",
                idx_map=np.asarray([[1, 1]], dtype=np.int64),
                cif_path=path,
            ),
            score=score,
        )

    monkeypatch.setattr(numpy_features, "_template_entry_from_cif", fake_candidate)
    entries = numpy_features._rank_direct_template_entries(
        [Path(f"{name}.cif") for name in scores], [], SEQUENCE
    )

    assert list(entries) == ["best_A", "second_A", "third_A", "fourth_A"]
    assert [entry.index for entry in entries.values()] == [0, 1, 2, 3]


def test_alignment_indel_map_is_preserved_without_canonical_realign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cif = LOCAL_TEMPLATE_CIF
    assert cif.is_file()

    from foldjax.models.openfold3._upstream.openfold3.core.data.io.sequence.template import (  # noqa: E501
        TemplateData,
    )
    from foldjax.models.openfold3.data import _numpy_featurization as numpy_features

    template = TemplateData(
        index=7,
        entry_id="gaag",
        chain_id="A",
        query_aln_pos=np.asarray([1, 2, 3, 4]),
        aln_pos=np.asarray([1, -1, 2, 3]),
        seq_id=0.75,
        q_cov=0.75,
        seq=TEMPLATE_SEQUENCE,
    )

    def unexpected_realign(*_args, **_kwargs):
        raise AssertionError("complete supplied alignment must not be realigned")

    monkeypatch.setattr(
        numpy_features, "_template_entry_from_cif", unexpected_realign
    )
    candidate = numpy_features._template_entry_from_alignment(
        cif, template, TEMPLATE_SEQUENCE
    )

    assert candidate is not None
    np.testing.assert_array_equal(
        candidate.entry.idx_map,
        np.asarray([[1, 1], [3, 2], [4, 3]], dtype=np.int64),
    )
    assert (
        numpy_features._template_entry_from_alignment(
            cif, template._replace(seq="SEQUENCE-NOT-IN-CIF"), TEMPLATE_SEQUENCE
        )
        is None
    )


def test_a3m_indel_position_reaches_template_mask(tmp_path: Path) -> None:
    fixture = LOCAL_TEMPLATE_CIF
    assert fixture.is_file()
    shutil.copy2(fixture, tmp_path / "gaag.cif")

    insertion = 2
    query_sequence = (
        TEMPLATE_SEQUENCE[:insertion] + "A" + TEMPLATE_SEQUENCE[insertion:]
    )
    template_alignment = (
        TEMPLATE_SEQUENCE[:insertion] + "-" + TEMPLATE_SEQUENCE[insertion:]
    )
    alignment = tmp_path / "templates.a3m"
    alignment.write_text(
        f">query_X/1-{len(query_sequence)}\n{query_sequence}\n"
        f">gaag_A/1-{len(TEMPLATE_SEQUENCE)}\n{template_alignment}\n"
    )

    features = featurize_query(
        {
            "queries": {
                "query": {
                    "chains": [
                        {
                            "molecule_type": "protein",
                            "chain_ids": ["A"],
                            "sequence": query_sequence,
                            "template_alignment_file_path": str(alignment),
                        }
                    ]
                }
            }
        }
    )
    mask = features["template_pseudo_beta_mask"][0, 0]

    assert int(mask.sum()) == len(TEMPLATE_SEQUENCE)
    assert mask[insertion] == 0
    assert mask[insertion - 1] == 1
    assert mask[insertion + 1] == 1


def test_object_valued_template_cache_is_never_unpickled(tmp_path: Path) -> None:
    from foldjax.models.openfold3.data import _numpy_featurization as numpy_features

    cache = tmp_path / "unsafe.npz"
    np.savez(cache, legacy=np.asarray({"idx_map": [[1, 1]]}, dtype=object))

    with pytest.raises(ValueError, match="Object-valued NumPy archives are not loaded"):
        numpy_features._preprocessed_template_entries(cache, None)


def test_safe_preparsed_msa_npz_reaches_features(tmp_path: Path) -> None:
    archive = tmp_path / "representative.npz"
    msa = np.asarray([list(SEQUENCE), list("A" * len(SEQUENCE))], dtype="U1")
    np.savez(
        archive,
        colabfold_main__msa=msa,
        colabfold_main__deletion_matrix=np.zeros(msa.shape, dtype=np.int32),
        colabfold_main__metadata=np.asarray(["query", "hit"]),
    )

    features = featurize_query(_spec(main_msa_file_paths=[str(archive)]))

    assert features["msa"].shape[1] > 1


def test_object_valued_msa_npz_is_never_unpickled(tmp_path: Path) -> None:
    archive = tmp_path / "legacy.npz"
    np.savez(
        archive,
        colabfold_main=np.asarray(
            {"msa": np.asarray([list(SEQUENCE)], dtype="U1")}, dtype=object
        ),
    )

    with pytest.raises(ValueError, match="Object-valued upstream archives"):
        featurize_query(_spec(main_msa_file_paths=[str(archive)]))
