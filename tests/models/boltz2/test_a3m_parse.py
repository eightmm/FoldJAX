"""Byte-table Boltz-2 A3M parser parity, block-budget, and file-reader tests."""

from __future__ import annotations

import gzip
import io
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest

from foldjax.models.boltz2.data import const
from foldjax.models.boltz2.data.parse import a3m
from foldjax.models.boltz2.data.parse.a3m import _parse_a3m, parse_a3m
from foldjax.models.boltz2.data.types import MSA, MSADeletion, MSAResidue, MSASequence


def _oracle_parse_a3m(
    lines: Iterable[str],
    taxonomy: dict[str, str] | None,
    max_seqs: int | None = None,
) -> MSA:
    """Reference copy of the pre-vectorisation character-by-character A3M parser."""
    visited: set[str] = set()
    sequences = []
    deletions = []
    residues = []

    seq_idx = 0
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith(">"):
            header = line.split()[0]
            if taxonomy and header.startswith(">UniRef100"):
                uniref_id = header.split("_")[1]
                taxonomy_id = taxonomy.get(uniref_id)
                if taxonomy_id is None:
                    taxonomy_id = -1
            else:
                taxonomy_id = -1
            continue

        str_seq = line.replace("-", "").upper()
        if str_seq not in visited:
            visited.add(str_seq)
        else:
            continue

        residue = []
        deletion = []
        count = 0
        res_idx = 0
        for c in line:
            if c != "-" and c.islower():
                count += 1
                continue
            residue.append(const.token_ids[const.prot_letter_to_token[c]])
            if count > 0:
                deletion.append((res_idx, count))
                count = 0
            res_idx += 1

        res_start = len(residues)
        res_end = res_start + len(residue)
        del_start = len(deletions)
        del_end = del_start + len(deletion)

        sequences.append((seq_idx, taxonomy_id, res_start, res_end, del_start, del_end))
        residues.extend(residue)
        deletions.extend(deletion)

        seq_idx += 1
        if (max_seqs is not None) and (seq_idx >= max_seqs):
            break

    return MSA(
        residues=np.array(residues, dtype=MSAResidue),
        deletions=np.array(deletions, dtype=MSADeletion),
        sequences=np.array(sequences, dtype=MSASequence),
    )


def _assert_msa_equal(actual: MSA, expected: MSA) -> None:
    assert isinstance(actual, MSA)
    for attr in ("residues", "deletions", "sequences"):
        got = getattr(actual, attr)
        want = getattr(expected, attr)
        assert got.dtype == want.dtype, attr
        assert got.shape == want.shape, attr
        assert got.flags.writeable, attr
        assert np.array_equal(got, want), attr
        assert want.dtype.names is not None
        for field_name in want.dtype.names:
            assert got[field_name].dtype == want[field_name].dtype, (attr, field_name)
            assert np.array_equal(got[field_name], want[field_name]), (attr, field_name)


def _outcome(fn):
    try:
        return ("ok", fn())
    except Exception as exc:  # noqa: BLE001
        return ("err", type(exc), exc.args)


def test_ascii_byte_table_matches_character_oracle() -> None:
    """Checks ``_ASCII_TABLE`` directly and every ASCII byte against the oracle."""
    assert a3m._ASCII_TABLE.shape == (128,)
    assert a3m._ASCII_TABLE.dtype == np.int8

    for code in range(128):
        text = f">r\nA{chr(code)}C\n"
        got = _outcome(lambda: _parse_a3m(io.StringIO(text), taxonomy=None))
        want = _outcome(lambda: _oracle_parse_a3m(io.StringIO(text), taxonomy=None))
        assert got[0] == want[0], code
        if got[0] == "err":
            assert got[1:] == want[1:], code
        else:
            _assert_msa_equal(got[1], want[1])


@pytest.mark.parametrize(
    "bad_row",
    [
        "AC.DE",
        "AC0DE",
        "AC9DE",
        "AC*DE",
        "AC DE",
        "AC_DE",
        "AÉC",
        "aé.C",
    ],
)
def test_unknown_characters_raise_key_error_like_oracle(bad_row: str) -> None:
    """Oracle-equality check: invalid ASCII and Unicode characters raise KeyError."""
    text = f">query\nACDEF\n>bad\n{bad_row}\n>tail\nGHIKL\n"
    got = _outcome(lambda: _parse_a3m(io.StringIO(text), taxonomy=None))
    want = _outcome(lambda: _oracle_parse_a3m(io.StringIO(text), taxonomy=None))
    assert got[0] == "err"
    assert got == want
    assert got[1] is KeyError


def test_malformed_header_after_invalid_row_preserves_exception_order() -> None:
    """Buffered invalid ASCII row still raises KeyError before later header error."""
    taxonomy = {"P12345": "9606"}
    bad_row_first = ">UniRef100_P12345\nAC.DE\n>UniRef100\nFGHIJ\n"
    got = _outcome(lambda: _parse_a3m(io.StringIO(bad_row_first), taxonomy=taxonomy))
    want = _outcome(
        lambda: _oracle_parse_a3m(io.StringIO(bad_row_first), taxonomy=taxonomy)
    )
    assert got == want
    assert got[1] is KeyError

    bad_header_only = ">UniRef100_P12345\nACDEF\n>UniRef100\nFGHIJ\n"
    got_hdr = _outcome(
        lambda: _parse_a3m(io.StringIO(bad_header_only), taxonomy=taxonomy)
    )
    want_hdr = _outcome(
        lambda: _oracle_parse_a3m(io.StringIO(bad_header_only), taxonomy=taxonomy)
    )
    assert got_hdr == want_hdr
    assert got_hdr[1] is IndexError


def test_edge_cases_and_max_seqs_limits_match_oracle() -> None:
    """Oracle-equality check over headers, duplicates, insertions, Unicode, max_seqs."""
    for empty_text in (
        "",
        "   \n\t  \r\n",
        "# a3m comment\n  # indented\n",
        ">q\n>h\n",
    ):
        _assert_msa_equal(
            _parse_a3m(io.StringIO(empty_text), taxonomy={"P1": "9606"}),
            _oracle_parse_a3m(io.StringIO(empty_text), taxonomy={"P1": "9606"}),
        )

    taxonomy = {"P12345": "9606", "Q8N158": "10090"}
    alignment = "\n".join(
        [
            "# leading comment",
            "",
            "   ",
            ">query extra header words",
            "  ACDEFGHIKLMNPQRSTVWYXJBZOU-  ",
            "# duplicate with gaps and lowercase is skipped before max_seqs:",
            ">UniRef100_P12345",
            "acdefghiklmnpqrstvwyxjbzou",
            ">UniRef100_P12345_HUMAN/1-27 metadata",
            "aaACdddEF-GHIKLMNPQRSTVWYXJBZOUzzz",
            ">UniRef100_UNKNOWN",
            "-----",
            "# second all-gap row normalises to empty string and is skipped:",
            ">UniRef90_P12345",
            "---",
            ">insertion_only",
            "abcdefg",
            ">unicode_lowercase_insertions",
            "AéCüDßE",
            "   \t   ",
            ">multi_line_under_one_header",
            "zA-C-D",
            "F-G-Hz",
        ]
    )

    full_expected = _oracle_parse_a3m(io.StringIO(alignment), taxonomy=taxonomy)
    kept_rows = len(full_expected.sequences)
    assert kept_rows == 7

    # Note: in the vendored parser, `seq_idx >= max_seqs` runs after keeping the
    # first sequence, so `max_seqs=0` yields 1 row on non-empty input (0 on empty).
    for limit in (None, 0, 1, 2, kept_rows - 1, kept_rows, kept_rows + 1):
        actual = _parse_a3m(io.StringIO(alignment), taxonomy=taxonomy, max_seqs=limit)
        expected = _oracle_parse_a3m(
            io.StringIO(alignment), taxonomy=taxonomy, max_seqs=limit
        )
        _assert_msa_equal(actual, expected)

    actual_zero_empty = _parse_a3m(io.StringIO(""), taxonomy=taxonomy, max_seqs=0)
    expected_zero_empty = _oracle_parse_a3m(
        io.StringIO(""), taxonomy=taxonomy, max_seqs=0
    )
    _assert_msa_equal(actual_zero_empty, expected_zero_empty)
    assert len(actual_zero_empty.sequences) == 0


def test_block_budgets_split_rows_across_row_and_byte_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercises block-closing boundaries (1, exact, +1, oversized, Unicode split)."""
    rows = [
        ("P1", "ACDE"),  # 4 bytes
        ("P2", "aaFGHIzz"),  # 8 bytes
        ("P3", "klmno"),  # 5 bytes, insertion-only
        ("P4", "pqrst"),  # 5 bytes, insertion-only
        ("P5", "KLMN"),  # 4 bytes
        ("P6", "AéBC"),  # non-ASCII flushes preceding ASCII block
        (
            "P7",
            "a" * 30 + "PQR" + "b" * 15,
        ),  # 48 bytes: over a 12-byte budget, the loop
        ("P8", "ST-UV"),  # 5 bytes
    ]
    taxonomy = {"P1": "9606", "P5": "10090", "P8": "7227"}
    text = "".join(f">UniRef100_{uid}\n{seq}\n" for uid, seq in rows)
    expected = _oracle_parse_a3m(io.StringIO(text), taxonomy=taxonomy)

    real_flush = a3m._flush_ascii_block
    flushed_sizes: list[tuple[int, int]] = []

    def spy_flush(
        block_lines: list[str],
        block_taxonomies: list[int | str],
        seq_idx_start: int,
        res_offset: int,
        del_offset: int,
    ):
        flushed_sizes.append((len(block_lines), sum(map(len, block_lines))))
        return real_flush(
            block_lines, block_taxonomies, seq_idx_start, res_offset, del_offset
        )

    monkeypatch.setattr(a3m, "_flush_ascii_block", spy_flush)

    budgets = [
        (1, 1 << 22),  # minimum row budget: every ASCII row is its own block
        (
            2,
            1 << 22,
        ),  # pair blocks, including an insertion-only pair ("klmno", "pqrst")
        (5, 1 << 22),  # exact ASCII prefix count before the non-ASCII row
        (1024, 1),  # byte budget smaller than every row
        (1024, 12),  # exact byte match on rows 0+1 (4 + 8 == 12), flushes before row 2
        (1024, 11),  # one below 4 + 8: flushes between rows 0 and 1
        (1024, 13),  # one above 4 + 8: rows 0+1 stay together, flushes before row 2
    ]
    for row_block, byte_block in budgets:
        flushed_sizes.clear()
        monkeypatch.setattr(a3m, "_ROW_BLOCK", row_block)
        monkeypatch.setattr(a3m, "_BYTE_BLOCK", byte_block)
        actual = _parse_a3m(io.StringIO(text), taxonomy=taxonomy)
        _assert_msa_equal(actual, expected)

        if (row_block, byte_block) == (1, 1 << 22):
            assert [n for n, _ in flushed_sizes] == [1, 1, 1, 1, 1, 1, 1]
        elif (row_block, byte_block) == (2, 1 << 22):
            # Rows 2 & 3 ("klmno", "pqrst") form an all-insertion block.
            assert [n for n, _ in flushed_sizes] == [2, 2, 1, 2]
        elif (row_block, byte_block) == (5, 1 << 22):
            assert [n for n, _ in flushed_sizes] == [5, 2]
        elif (row_block, byte_block) == (1024, 12):
            assert flushed_sizes == [(2, 12), (2, 10), (1, 4), (1, 5)]
        elif (row_block, byte_block) == (1024, 11):
            assert flushed_sizes == [(1, 4), (1, 8), (2, 10), (1, 4), (1, 5)]
        elif (row_block, byte_block) == (1024, 1):
            # Every row is larger than the budget, so none is vectorised.
            assert flushed_sizes == []


def _random_a3m(
    rng: np.random.Generator, n_records: int, base_cols: int
) -> tuple[str, dict[str, str]]:
    upper_alphabet = list("ACDEFGHIKLMNPQRSTVWYXJBZOU-")
    lower_alphabet = list("abcdefghijklmnopqrstuvwxyz")
    unicode_lower = ["é", "ü", "α", "ß"]
    taxonomy = {f"U{idx:04d}": str(1000 + idx) for idx in range(25)}

    out_lines = ["# synthetic random A3M"]
    seen_seqs: list[str] = []
    for rec in range(n_records):
        if rng.random() < 0.15:
            out_lines.append("")
        if rng.random() < 0.10:
            out_lines.append(f"# note {rec}")

        hdr_kind = int(rng.integers(0, 4))
        if hdr_kind == 0:
            uid = f"U{int(rng.integers(0, 35)):04d}"
            out_lines.append(f">UniRef100_{uid} sub=1")
        elif hdr_kind == 1:
            uid = f"U{int(rng.integers(0, 35)):04d}"
            out_lines.append(f">UniRef100_{uid}_SPECIES/1-{base_cols}")
        elif hdr_kind == 2:
            out_lines.append(f">UniRef90_U{rec:04d}")
        else:
            out_lines.append(f">seq_{rec}")

        row_kind = rng.random()
        if row_kind < 0.06 and seen_seqs:
            # Duplicate via gap/case perturbation.
            chosen = seen_seqs[int(rng.integers(0, len(seen_seqs)))]
            seq = "-" + chosen.lower() if rng.random() < 0.5 else chosen
        elif row_kind < 0.10:
            # Insertion-only row.
            ins_len = int(rng.integers(1, 8))
            seq = "".join(rng.choice(lower_alphabet, size=ins_len))
        elif row_kind < 0.14:
            # All-gap row.
            seq = "-" * base_cols
        else:
            cols = base_cols + int(rng.integers(-2, 3))
            cols = max(1, cols)
            pieces: list[str] = []
            if rng.random() < 0.35:
                lead = int(rng.integers(1, 5))
                pieces.extend(rng.choice(lower_alphabet, size=lead))
            for _ in range(cols):
                if rng.random() < 0.12:
                    ins_n = int(rng.integers(1, 6))
                    if rng.random() < 0.08:
                        pieces.append(
                            unicode_lower[int(rng.integers(0, len(unicode_lower)))]
                        )
                    else:
                        pieces.extend(rng.choice(lower_alphabet, size=ins_n))
                pieces.append(upper_alphabet[int(rng.integers(0, len(upper_alphabet)))])
            if rng.random() < 0.30:
                trail = int(rng.integers(1, 5))
                pieces.extend(rng.choice(lower_alphabet, size=trail))
            seq = "".join(pieces)

        seen_seqs.append(seq)
        if rng.random() < 0.2:
            seq = f"  {seq}  "
        out_lines.append(seq)

    return "\n".join(out_lines) + "\n", taxonomy


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_random_alignments_match_oracle_across_block_budgets(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Oracle-equality check with small block budgets forcing multi-block splits."""
    rng = np.random.default_rng(seed)
    text, taxonomy = _random_a3m(rng, n_records=180, base_cols=35)
    expected = _oracle_parse_a3m(io.StringIO(text), taxonomy=taxonomy)
    expected_no_tax = _oracle_parse_a3m(io.StringIO(text), taxonomy=None)

    _assert_msa_equal(_parse_a3m(io.StringIO(text), taxonomy=taxonomy), expected)
    _assert_msa_equal(_parse_a3m(io.StringIO(text), taxonomy=None), expected_no_tax)

    monkeypatch.setattr(a3m, "_ROW_BLOCK", 11)
    monkeypatch.setattr(a3m, "_BYTE_BLOCK", 250)
    _assert_msa_equal(_parse_a3m(io.StringIO(text), taxonomy=taxonomy), expected)

    half = max(1, len(expected.sequences) // 2)
    _assert_msa_equal(
        _parse_a3m(io.StringIO(text), taxonomy=taxonomy, max_seqs=half),
        _oracle_parse_a3m(io.StringIO(text), taxonomy=taxonomy, max_seqs=half),
    )


def test_parse_a3m_plain_and_gzip_files(tmp_path: Path) -> None:
    """Oracle-equality check through ``parse_a3m`` for ``.a3m`` and ``.a3m.gz``."""
    rng = np.random.default_rng(123)
    text, taxonomy = _random_a3m(rng, n_records=60, base_cols=24)
    expected = _oracle_parse_a3m(io.StringIO(text), taxonomy=taxonomy, max_seqs=30)

    plain_path = tmp_path / "sample.a3m"
    plain_path.write_text(text, encoding="utf-8")
    gz_path = tmp_path / "sample.a3m.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as handle:
        handle.write(text)

    _assert_msa_equal(parse_a3m(plain_path, taxonomy=taxonomy, max_seqs=30), expected)
    _assert_msa_equal(parse_a3m(gz_path, taxonomy=taxonomy, max_seqs=30), expected)

    bad_gz = tmp_path / "invalid.a3m.gz"
    with gzip.open(bad_gz, "wt", encoding="utf-8") as handle:
        handle.write(">q\nAC.DE\n")
    with pytest.raises(KeyError):
        parse_a3m(bad_gz, taxonomy=None)


@pytest.mark.parametrize(
    "text",
    [
        # 32,769 distinct rows: seq_idx 32,768 does not fit the int16 field.
        "".join(
            f">s\nC{'A' * (i // 20)}{'ACDEFGHIKLMNPQRSTVWY'[i % 20]}\n"
            for i in range(32769)
        ),
        # An insertion run of 40,000 does not fit the int16 deletion count.
        ">a\n" + "a" * 40000 + "C\n",
    ],
    ids=["seq_idx", "deletion"],
)
def test_int16_overflow_raises_like_oracle(text: str) -> None:
    with pytest.raises(OverflowError) as expected:
        _oracle_parse_a3m(io.StringIO(text), None)
    with pytest.raises(OverflowError) as actual:
        _parse_a3m(io.StringIO(text), None)
    assert str(actual.value) == str(expected.value)


def _first_error(parse, text: str) -> str:
    try:
        parse(io.StringIO(text), None)
    except Exception as error:  # noqa: BLE001 -- the comparison is the point
        return f"{type(error).__name__}: {error}"
    return "no error"


@pytest.mark.parametrize(
    "text",
    [
        # The overflow is only refused when the arrays are built, after the
        # whole file, so the later unknown character is what the loop reports.
        ">a\n" + "a" * 40000 + "C\n>b\nAC.D\n",
        # The same with a malformed taxonomy header after the overflowing row.
        ">a\n" + "a" * 40000 + "C\n>UniRef100\nACD\n",
    ],
    ids=["overflow-then-key", "overflow-then-header"],
)
def test_overflow_is_reported_after_every_row_like_oracle(text: str) -> None:
    taxonomy = {"x": "1"}

    def oracle(lines, _):
        return _oracle_parse_a3m(lines, taxonomy)

    def actual(lines, _):
        return _parse_a3m(lines, taxonomy)

    assert _first_error(actual, text) == _first_error(oracle, text)


def test_unknown_character_is_raised_before_later_rows_are_read() -> None:
    def lines():
        yield ">a\n"
        yield "AC.D\n"
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    with pytest.raises(KeyError) as expected:
        _oracle_parse_a3m(lines(), None)
    with pytest.raises(KeyError) as actual:
        _parse_a3m(lines(), None)
    assert actual.value.args == expected.value.args


def test_row_before_any_header_fails_like_oracle() -> None:
    def first_error(parse, text):
        try:
            parse(io.StringIO(text), None)
        except Exception as error:  # noqa: BLE001 -- the comparison is the point
            return f"{type(error).__name__}: {error}"
        return "no error"

    for text in ("ACD\n>a\nEFG\n", "AC.D\n", "\nACD\n"):
        assert first_error(_parse_a3m, text) == first_error(_oracle_parse_a3m, text)
