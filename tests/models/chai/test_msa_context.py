from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from foldjax.models.chai.data.input import EntityType, Input
from foldjax.models.chai.data.msa import (
    NO_PAIRING_KEY,
    RESIDUE_ORDER,
    MSAContext,
    MSADataSource,
    expected_aligned_pqt_basename,
    load_msa_contexts,
    msa_context_from_a3m,
    msa_context_from_search_results,
    pair_and_merge_msas,
    parse_aligned_pqt,
    tokenize_aligned_sequences,
)
from foldjax.models.chai.data.structure import StructureContext


def _write_aligned_pqt(path: Path, rows: list[dict[str, str]]) -> None:
    pq.write_table(
        pa.table(
            {
                name: [row[name] for row in rows]
                for name in ("sequence", "source_database", "pairing_key", "comment")
            }
        ),
        path,
    )


def _minimal_structure(
    token_residue_type: list[int],
    token_residue_index: list[int],
    token_asym_id: list[int],
) -> StructureContext:
    """Construct only fields consumed by MSA assembly."""
    context = object.__new__(StructureContext)
    object.__setattr__(context, "token_residue_type", np.asarray(token_residue_type))
    object.__setattr__(context, "token_residue_index", np.asarray(token_residue_index))
    object.__setattr__(context, "token_asym_id", np.asarray(token_asym_id))
    return context


def test_a3m_tokenization_matches_chai_insertions_and_deletion_counts() -> None:
    tokens, deletions = tokenize_aligned_sequences(["ACD-E", "AaaCD-E", "ACDbb-E"])
    expected = np.asarray(
        [[RESIDUE_ORDER[x] for x in "ACD-E"]] * 3,
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(tokens, expected)
    np.testing.assert_array_equal(
        deletions,
        [[0, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 0, 2, 0]],
    )


def test_a3m_unknown_skip_and_deletion_saturation() -> None:
    insertion = "a" * 300
    tokens, deletions = tokenize_aligned_sequences([f"A.{insertion}Z-"])
    np.testing.assert_array_equal(
        tokens[0], [RESIDUE_ORDER["A"], RESIDUE_ORDER["X"], RESIDUE_ORDER["-"]]
    )
    np.testing.assert_array_equal(deletions[0], [0, 255, 0])


def test_a3m_rows_must_have_same_aligned_length() -> None:
    with pytest.raises(ValueError, match="aligned length"):
        tokenize_aligned_sequences(["ACD", "AC"])


def test_msa_context_single_empty_padding_and_mask() -> None:
    tokens, _ = tokenize_aligned_sequences(["ACD"])
    single = MSAContext.create_single_sequence(MSADataSource.QUERY, tokens[0])
    assert single.depth == 1
    assert single.num_tokens == 3
    assert single.mask.all()
    assert (single.pairing_key_hash == NO_PAIRING_KEY).all()
    assert (single.sequence_source == 5).all()

    padded = single.pad(max_num_tokens=5, max_msa_depth=3)
    assert padded.tokens.shape == (3, 5)
    assert (padded.tokens[1:] == RESIDUE_ORDER[":"]).all()
    assert not padded.mask[1:].any()
    assert (padded.sequence_source[1:] == 4).all()

    empty = MSAContext.create_empty(4, depth=2)
    assert empty.tokens.shape == (2, 4)
    assert not empty.mask.any()


def test_search_a3m_converts_to_chai_context_and_deduplicates() -> None:
    context = msa_context_from_a3m(
        "ACDE",
        paired=">101\nACDE\n>pair-a\nAC-E\n",
        unpaired=(">101\nACDE\n>UniRef_hit\nACdDE\n>duplicate\nAC-E\n>other\nA-DE\n"),
    )
    assert context.tokens.shape == (4, 4)
    assert context.sequence_source[:, 0].tolist() == [5, 0, 2, 0]
    assert context.pairing_key_hash[0, 0] != NO_PAIRING_KEY
    assert context.pairing_key_hash[1, 0] != NO_PAIRING_KEY
    assert (context.pairing_key_hash[2:] == NO_PAIRING_KEY).all()


def test_pair_and_merge_places_shared_pairs_before_unpaired_rows() -> None:
    first = msa_context_from_a3m(
        "AC",
        paired=">101\nAC\n>pair\nA-\n",
        unpaired=">101\nAC\n>first-only\n-C\n",
    )
    second = msa_context_from_a3m(
        "DE",
        paired=">101\nDE\n>pair\nD-\n",
        unpaired=">101\nDE\n>second-only\n-E\n",
    )

    merged = pair_and_merge_msas([first, second])

    assert merged.tokens.shape == (3, 4)
    np.testing.assert_array_equal(merged.tokens[0], [RESIDUE_ORDER[x] for x in "ACDE"])
    np.testing.assert_array_equal(merged.tokens[1], [RESIDUE_ORDER[x] for x in "A-D-"])
    np.testing.assert_array_equal(merged.tokens[2], [RESIDUE_ORDER[x] for x in "-C-E"])


def test_cached_search_paths_feed_the_paired_context(tmp_path: Path) -> None:
    paths = []
    for index, (query, hit) in enumerate((("AC", "A-"), ("DE", "D-"))):
        directory = tmp_path / str(index)
        directory.mkdir()
        paired = directory / "pairing.a3m"
        unpaired = directory / "non_pairing.a3m"
        paired.write_text(f">101\n{query}\n>pair\n{hit}\n")
        unpaired.write_text(f">101\n{query}\n")
        paths.append(
            {
                "pairedMsaPath": str(paired),
                "unpairedMsaPath": str(unpaired),
            }
        )

    merged = msa_context_from_search_results(["AC", "DE"], paths)

    assert merged.tokens.shape == (2, 4)
    np.testing.assert_array_equal(merged.tokens[1], [RESIDUE_ORDER[x] for x in "A-D-"])


def test_pair_rank_assignment_matches_upstream_row_order() -> None:
    tokens = np.asarray(
        [
            [RESIDUE_ORDER[value] for value in row]
            for row in ("AAAA", "----", "AA--", "A---")
        ],
        dtype=np.uint8,
    )
    shape = tokens.shape
    pairing = np.full(shape, NO_PAIRING_KEY, np.int32)
    pairing[1:] = np.asarray([17, 18, 17], np.int32)[:, None]
    context = MSAContext(
        tokens=tokens,
        pairing_key_hash=pairing,
        deletion_matrix=np.zeros(shape, np.uint8),
        mask=np.ones(shape, np.bool_),
        sequence_source=np.full(shape, 3, np.uint8),
    )

    actual = pair_and_merge_msas([context])

    np.testing.assert_array_equal(actual.tokens, tokens)


@pytest.mark.official_parity
def test_aligned_parquet_round_trip_matches_upstream_chai(
    tmp_path: Path,
    upstream_chai_dir: Path,
    upstream_chai_python: Path,
) -> None:
    sequence = "ACDE"
    path = tmp_path / expected_aligned_pqt_basename(sequence)
    _write_aligned_pqt(
        path,
        [
            {
                "sequence": "ACDE",
                "source_database": "query",
                "pairing_key": "",
                "comment": "query",
            },
            {
                "sequence": "AaaC-E",
                "source_database": "uniprot",
                "pairing_key": "9606",
                "comment": "paired",
            },
            {
                "sequence": "ACD-",
                "source_database": "uniref90",
                "pairing_key": "",
                "comment": "unpaired",
            },
            {
                "sequence": "A-DE",
                "source_database": "bfd_uniclust",
                "pairing_key": "",
                "comment": "bfd",
            },
        ],
    )

    actual = parse_aligned_pqt(path, query_sequence=sequence)
    assert actual.sequence_source[:, 0].tolist() == [5, 2, 3, 0]
    assert actual.deletion_matrix[2, 1] == 2

    output = tmp_path / "upstream.npz"
    script = """
import numpy as np
from pathlib import Path
from chai_lab.data.parsing.msas.aligned_pqt import parse_aligned_pqt_to_msa_context
ctx = parse_aligned_pqt_to_msa_context(Path(__import__('sys').argv[1]))
np.savez(__import__('sys').argv[2], **{
    name: getattr(ctx, name).cpu().numpy()
    for name in (
        'tokens', 'pairing_key_hash', 'deletion_matrix', 'mask', 'sequence_source'
    )
})
"""
    subprocess.run(
        [str(upstream_chai_python), "-c", script, str(path), str(output)],
        check=True,
        env={**os.environ, "PYTHONPATH": str(upstream_chai_dir)},
    )
    with np.load(output) as reference:
        for name in reference.files:
            np.testing.assert_array_equal(getattr(actual, name), reference[name])


@pytest.mark.official_parity
def test_aligned_parquet_repeated_source_order_matches_upstream_chai(
    tmp_path: Path,
    upstream_chai_dir: Path,
    upstream_chai_python: Path,
) -> None:
    sequence = "ACDE"
    residues = "ARNDCQEGHILKMFPSTWYV"
    rows = [
        {
            "sequence": sequence,
            "source_database": "query",
            "pairing_key": "",
            "comment": "query",
        }
    ]
    for index in range(64):
        rows.append(
            {
                "sequence": "".join(
                    residues[(index + offset * 7) % len(residues)]
                    for offset in range(len(sequence))
                ),
                "source_database": ("uniref90", "uniprot", "bfd_uniclust")[index % 3],
                "pairing_key": str(index) if index % 3 == 1 else "",
                "comment": f"hit-{index}",
            }
        )
    path = tmp_path / expected_aligned_pqt_basename(sequence)
    _write_aligned_pqt(path, rows)

    output = tmp_path / "upstream-repeated.npz"
    script = """
import numpy as np
from pathlib import Path
from chai_lab.data.parsing.msas.aligned_pqt import parse_aligned_pqt_to_msa_context
ctx = parse_aligned_pqt_to_msa_context(Path(__import__('sys').argv[1]))
np.savez(__import__('sys').argv[2], **{
    name: getattr(ctx, name).cpu().numpy()
    for name in (
        'tokens', 'pairing_key_hash', 'deletion_matrix', 'mask', 'sequence_source'
    )
})
"""
    subprocess.run(
        [str(upstream_chai_python), "-c", script, str(path), str(output)],
        check=True,
        env={**os.environ, "PYTHONPATH": str(upstream_chai_dir)},
    )
    actual = parse_aligned_pqt(path, query_sequence=sequence)
    with np.load(output) as reference:
        for name in reference.files:
            np.testing.assert_array_equal(getattr(actual, name), reference[name])


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [
                {
                    "sequence": "AC",
                    "source_database": "uniref90",
                    "pairing_key": "",
                    "comment": "not-query",
                }
            ],
            "first row",
        ),
        (
            [
                {
                    "sequence": "AC",
                    "source_database": "query",
                    "pairing_key": "",
                    "comment": "query",
                },
                {
                    "sequence": "AC",
                    "source_database": "unknown-db",
                    "pairing_key": "",
                    "comment": "bad",
                },
            ],
            "source_database",
        ),
    ],
)
def test_aligned_parquet_schema_fails_explicitly(
    tmp_path: Path, rows: list[dict[str, str]], message: str
) -> None:
    path = tmp_path / "bad.aligned.pqt"
    _write_aligned_pqt(path, rows)
    with pytest.raises(ValueError, match=message):
        parse_aligned_pqt(path, query_sequence="AC")


def test_aligned_parquet_missing_column_reports_schema_error(tmp_path: Path) -> None:
    path = tmp_path / "missing.aligned.pqt"
    pq.write_table(
        pa.table(
            {
                "sequence": ["AC"],
                "source_database": ["query"],
                "pairing_key": [""],
            }
        ),
        path,
    )

    with pytest.raises(ValueError, match="missing columns.*comment"):
        parse_aligned_pqt(path, query_sequence="AC")


def test_load_msa_contexts_matches_chain_pairing_and_profile_contract(
    tmp_path: Path,
) -> None:
    inputs = [
        Input("AC", EntityType.PROTEIN.value, "first"),
        Input("DE", EntityType.PROTEIN.value, "second"),
    ]
    for sequence, hit in (("AC", "A-"), ("DE", "D-")):
        _write_aligned_pqt(
            tmp_path / expected_aligned_pqt_basename(sequence),
            [
                {
                    "sequence": sequence,
                    "source_database": "query",
                    "pairing_key": "",
                    "comment": "query",
                },
                {
                    "sequence": hit,
                    "source_database": "uniprot",
                    "pairing_key": "shared",
                    "comment": "hit",
                },
            ],
        )
    structure = _minimal_structure([0, 4, 3, 6], [0, 1, 0, 1], [1, 1, 2, 2])

    joined, profile = load_msa_contexts(inputs, structure, msa_directory=tmp_path)

    assert joined.tokens.shape == (2, 4)
    np.testing.assert_array_equal(joined.tokens[1], [0, 31, 3, 31])
    assert profile.tokens.shape == (2, 4)


def test_missing_protein_parquet_uses_query_row_without_silent_empty_msa(
    tmp_path: Path,
) -> None:
    inputs = [Input("AC", EntityType.PROTEIN.value, "target")]
    structure = _minimal_structure([0, 4], [0, 1], [1, 1])

    with pytest.warns(RuntimeWarning, match="query row only"):
        joined, profile = load_msa_contexts(inputs, structure, msa_directory=tmp_path)

    np.testing.assert_array_equal(joined.tokens, [[0, 4]])
    assert joined.mask.all()
    np.testing.assert_array_equal(profile.tokens, joined.tokens)
