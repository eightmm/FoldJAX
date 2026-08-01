"""NumPy implementation of Chai's aligned-MSA token and context contract."""

from __future__ import annotations

import hashlib
import string
import warnings
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from foldjax.models.chai.data.input import EntityType, Input
from foldjax.models.chai.data.structure import StructureContext

RESIDUE_TYPES: Final[tuple[str, ...]] = (
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
    "X",
    "RA",
    "RC",
    "RG",
    "RU",
    "RX",
    "DA",
    "DC",
    "DG",
    "DT",
    "DX",
    "-",
    ":",
)
RESIDUE_ORDER: Final[dict[str, int]] = {
    residue: index for index, residue in enumerate(RESIDUE_TYPES)
}
NO_PAIRING_KEY: Final[int] = -999991
MAX_PAIRED_DEPTH: Final[int] = 8192
FULL_DEPTH: Final[int] = 16384
_QUERY_PAIR_KEY: Final[tuple[int, int]] = (-999, -999)
_ALIGNED_PQT_COLUMNS: Final[tuple[str, ...]] = (
    "sequence",
    "source_database",
    "pairing_key",
    "comment",
)


class MSADataSource(Enum):
    QUERY = "query"
    UNIPROT = "uniprot"
    UNIREF90 = "uniref90"
    BFD = "BFD"
    MGNIFY = "mgnify"
    PAIRED = "paired"
    MAIN = "main"
    BFD_UNICLUST = "bfd_uniclust"
    SINGLETON = "singleton"
    NONE = "none"
    PDB70 = "pdb70"
    UNIPROT_N3 = "uniprot_n3"
    UNIREF90_N3 = "uniref90_n3"
    MGNIFY_N3 = "mgnify_n3"


SOURCE_TO_INT: Final[dict[MSADataSource, int]] = {
    MSADataSource.BFD_UNICLUST: 0,
    MSADataSource.MGNIFY: 1,
    MSADataSource.UNIREF90: 2,
    MSADataSource.UNIPROT: 3,
    MSADataSource.NONE: 4,
    MSADataSource.UNIPROT_N3: 3,
    MSADataSource.UNIREF90_N3: 2,
    MSADataSource.MGNIFY_N3: 1,
    MSADataSource.QUERY: 5,
}
_SOURCE_QUOTAS: Final[dict[MSADataSource, int]] = {
    MSADataSource.UNIREF90: 10_000,
    MSADataSource.UNIPROT: 50_000,
    MSADataSource.BFD_UNICLUST: 1_000_000,
    MSADataSource.BFD: 5_000,
    MSADataSource.MGNIFY: 5_000,
    MSADataSource.UNIREF90_N3: 10_000,
    MSADataSource.UNIPROT_N3: 50_000,
    MSADataSource.MGNIFY_N3: 5_000,
    MSADataSource.PDB70: 5_000,
}
_SOURCE_PRIORITY: Final[dict[MSADataSource, int]] = {
    source: index for index, source in enumerate(_SOURCE_QUOTAS)
}
_SOURCE_PRIORITY[MSADataSource.QUERY] = -1
_RECOGNIZED_ALIGNED_SOURCES: Final[set[MSADataSource]] = {
    MSADataSource.QUERY,
    MSADataSource.UNIPROT,
    MSADataSource.UNIREF90,
    MSADataSource.MGNIFY,
    MSADataSource.BFD_UNICLUST,
}


def _numpy_126_quicksort_indices(values: Sequence[int]) -> list[int]:
    """Reproduce NumPy 1.26's indirect quicksort tie ordering.

    Official chai_lab prepares aligned Parquet rows with pandas backed by
    NumPy 1.26. Its default unstable quicksort makes equal-priority row order
    observable to the model. NumPy 2 changed that permutation, so keep the
    small legacy indirect partition here instead of adding a pandas runtime
    dependency or pinning an obsolete NumPy.
    """

    order = list(range(len(values)))
    if len(order) < 2:
        return order
    stack: list[tuple[int, int, int]] = [
        (0, len(order) - 1, (len(order).bit_length() - 1) * 2)
    ]
    while stack:
        low, high, depth = stack.pop()
        if depth < 0:
            raise RuntimeError("legacy MSA priority quicksort exceeded depth limit")
        while high - low > 15:
            middle = low + ((high - low) >> 1)
            if values[order[middle]] < values[order[low]]:
                order[middle], order[low] = order[low], order[middle]
            if values[order[high]] < values[order[middle]]:
                order[high], order[middle] = order[middle], order[high]
            if values[order[middle]] < values[order[low]]:
                order[middle], order[low] = order[low], order[middle]
            pivot = values[order[middle]]
            left = low
            right = high - 1
            order[middle], order[right] = order[right], order[middle]
            while True:
                left += 1
                while values[order[left]] < pivot:
                    left += 1
                right -= 1
                while pivot < values[order[right]]:
                    right -= 1
                if left >= right:
                    break
                order[left], order[right] = order[right], order[left]
            order[left], order[high - 1] = order[high - 1], order[left]
            next_depth = depth - 1
            if left - low < high - left:
                stack.append((left + 1, high, next_depth))
                high = left - 1
            else:
                stack.append((low, left - 1, next_depth))
                low = left + 1
            depth = next_depth

        for index in range(low + 1, high + 1):
            selected = order[index]
            position = index
            while position > low and values[selected] < values[order[position - 1]]:
                order[position] = order[position - 1]
                position -= 1
            order[position] = selected
    return order


def msa_subsample_indices_and_mask(
    mask: np.ndarray,
    *,
    select_n_rows: int = 4096,
    random_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return Chai-compatible MSA row order and selected-plus-padded mask.

    ``random_values`` is explicit so callers can use a framework-owned PRNG
    without silently depending on NumPy global state. The ranking, stable
    top-to-bottom selected-row order, and mask padding match chai_lab.
    """

    msa_mask = np.asarray(mask, dtype=np.bool_)
    if msa_mask.ndim != 3 or msa_mask.shape[0] != 1:
        raise ValueError("mask must have shape (1, depth, token)")
    depth = msa_mask.shape[1]
    draws = np.asarray(random_values, dtype=np.float16)
    if draws.shape != (depth,):
        raise ValueError("random_values must have shape (depth,)")
    msa_sizes = msa_mask.sum(axis=(0, 2), dtype=np.int64)
    nonnull = msa_sizes > 0
    input_depth = int(nonnull.sum())
    if select_n_rows <= 0 or input_depth <= select_n_rows:
        return None

    rankings = msa_sizes.astype(np.float16) * draws
    selected = np.argsort(rankings)[-select_n_rows:]
    if np.any(~nonnull[selected]):
        raise AssertionError("MSA subsampling selected an empty row")
    selection_mask = np.zeros(depth, dtype=np.bool_)
    selection_mask[selected] = True
    selected_indices = np.flatnonzero(selection_mask)
    unselected_indices = np.flatnonzero(~selection_mask)
    row_order = np.concatenate([selected_indices, unselected_indices])
    selected_mask = msa_mask[:, selected_indices]
    padded_mask = np.zeros_like(msa_mask)
    padded_mask[:, : selected_indices.size] = selected_mask
    return row_order, padded_mask


def subsample_and_reorder_msa_features(
    features: np.ndarray,
    mask: np.ndarray,
    *,
    select_n_rows: int = 4096,
    random_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Chai's recycle-MSA feature reorder and mask subsampling."""

    feature_array = np.asarray(features)
    mask_array = np.asarray(mask)
    if feature_array.ndim != 4 or feature_array.shape[:3] != mask_array.shape:
        raise ValueError("features/mask must have matching (batch, depth, token) axes")
    plan = msa_subsample_indices_and_mask(
        mask_array,
        select_n_rows=select_n_rows,
        random_values=random_values,
    )
    if plan is None:
        return features, mask
    row_order, sampled_mask = plan
    return np.take(feature_array, row_order, axis=1), sampled_mask


def _token_mapping() -> np.ndarray:
    skip = -1
    insertion = -2
    mapping = np.full(256, skip, dtype=np.int32)
    for character in string.ascii_uppercase:
        mapping[ord(character)] = RESIDUE_ORDER["X"]
    for character, value in RESIDUE_ORDER.items():
        if len(character) == 1:
            mapping[ord(character)] = value
    for character in string.ascii_lowercase:
        mapping[ord(character)] = insertion
    return mapping


_TOKEN_MAPPING = _token_mapping()


def tokenize_aligned_sequences(
    sequences: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize A3M rows, dropping insertions and recording deletion counts."""
    if not sequences:
        raise ValueError("at least one aligned sequence is required")
    aligned_length = sum(
        character in string.ascii_uppercase or character == "-"
        for character in sequences[0]
    )
    tokens = np.zeros((len(sequences), aligned_length), dtype=np.uint8)
    deletions = np.zeros_like(tokens)
    for row_index, sequence in enumerate(sequences):
        position = 0
        skipped = 0
        try:
            encoded = sequence.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError(
                "aligned sequences must contain ASCII characters"
            ) from error
        for character in encoded:
            mapped = int(_TOKEN_MAPPING[character])
            if mapped == -1:
                continue
            if mapped == -2:
                skipped += 1
                continue
            if position >= aligned_length:
                raise ValueError("MSA rows do not share the same aligned length")
            tokens[row_index, position] = mapped
            deletions[row_index, position] = min(skipped, 255)
            position += 1
            skipped = 0
        if position != aligned_length:
            raise ValueError("MSA rows do not share the same aligned length")
    return tokens, deletions


def write_aligned_pqt_from_a3m(
    a3m_text: str,
    directory: str | Path,
    *,
    source_database: str = "uniref90",
) -> Path:
    """Write an A3M alignment as the aligned-Parquet file Chai reads.

    Chai is the one backend here whose native input is a FASTA, with no room
    for an alignment path: it takes user-supplied MSAs as ``.aligned.pqt``
    files in a directory, addressed by a hash of the query sequence. An A3M is
    what every MSA server returns and what the other four backends consume
    directly, so this is the bridge between them -- it lets one alignment be
    held fixed across models instead of Chai silently running on a different
    one.

    The first A3M record is the query, and is written as the query row Chai
    requires. Lowercase insertion columns are preserved verbatim, since that is
    how the A3M encodes deletions and Chai's reader expects them.
    """
    records = _a3m_records(a3m_text)
    if not records:
        raise ValueError("A3M alignment contains no records")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    query_aligned = records[0][1]
    query = "".join(
        character
        for character in query_aligned
        if character.isupper() and character != "-"
    )
    sources = [MSADataSource.QUERY.value] + [source_database] * (len(records) - 1)
    table = pa.table(
        {
            "sequence": pa.array([sequence for _, sequence in records], pa.string()),
            "source_database": pa.array(sources, pa.string()),
            # Chai pairs chains by this key; an unpaired alignment has none.
            "pairing_key": pa.array([""] * len(records), pa.string()),
            "comment": pa.array([comment for comment, _ in records], pa.string()),
        }
    )
    path = directory / expected_aligned_pqt_basename(query)
    pq.write_table(table, path)
    return path


def expected_aligned_pqt_basename(query_sequence: str) -> str:
    """Return Chai's public SHA256-addressed aligned-Parquet basename."""
    if not isinstance(query_sequence, str) or not query_sequence:
        raise ValueError("MSA query sequence must be a non-empty string")
    digest = hashlib.sha256(query_sequence.upper().encode()).hexdigest()
    return f"{digest}.aligned.pqt"


@dataclass(frozen=True)
class MSAContext:
    tokens: np.ndarray
    pairing_key_hash: np.ndarray
    deletion_matrix: np.ndarray
    mask: np.ndarray
    sequence_source: np.ndarray

    def __post_init__(self) -> None:
        shape = self.tokens.shape
        if len(shape) != 2 or any(
            value.shape != shape
            for value in (
                self.pairing_key_hash,
                self.deletion_matrix,
                self.mask,
                self.sequence_source,
            )
        ):
            raise ValueError(
                "all MSA context arrays must share a two-dimensional shape"
            )

    @property
    def depth(self) -> int:
        return self.tokens.shape[0]

    @property
    def num_tokens(self) -> int:
        return self.tokens.shape[1]

    def pad(
        self,
        max_num_tokens: int | None = None,
        max_msa_depth: int | None = None,
    ) -> MSAContext:
        token_count = self.num_tokens if max_num_tokens is None else max_num_tokens
        depth = self.depth if max_msa_depth is None else max_msa_depth
        if token_count < self.num_tokens or depth < self.depth:
            raise ValueError("MSA padding targets cannot crop the context")
        widths = ((0, depth - self.depth), (0, token_count - self.num_tokens))

        def padded(value: np.ndarray, fill: int | bool) -> np.ndarray:
            return np.pad(value, widths, constant_values=fill)

        return MSAContext(
            tokens=padded(self.tokens, RESIDUE_ORDER[":"]),
            pairing_key_hash=padded(self.pairing_key_hash, NO_PAIRING_KEY),
            deletion_matrix=padded(self.deletion_matrix, 0),
            mask=padded(self.mask, False),
            sequence_source=padded(
                self.sequence_source, SOURCE_TO_INT[MSADataSource.NONE]
            ),
        )

    def select_rows_with_padding(self, row_indices: list[int | None]) -> MSAContext:
        padded = self.pad(max_msa_depth=self.depth + 1)
        indices = np.asarray(
            [-1 if index is None else index for index in row_indices], np.int64
        )
        return MSAContext(
            tokens=padded.tokens[indices],
            pairing_key_hash=padded.pairing_key_hash[indices],
            deletion_matrix=padded.deletion_matrix[indices],
            mask=padded.mask[indices],
            sequence_source=padded.sequence_source[indices],
        )

    @classmethod
    def concatenate(cls, contexts: list[MSAContext], axis: int) -> MSAContext:
        if not contexts or axis not in {0, 1, -1}:
            raise ValueError("MSA concatenation requires contexts and axis 0 or 1")
        return cls(
            tokens=np.concatenate([value.tokens for value in contexts], axis=axis),
            pairing_key_hash=np.concatenate(
                [value.pairing_key_hash for value in contexts], axis=axis
            ),
            deletion_matrix=np.concatenate(
                [value.deletion_matrix for value in contexts], axis=axis
            ),
            mask=np.concatenate([value.mask for value in contexts], axis=axis),
            sequence_source=np.concatenate(
                [value.sequence_source for value in contexts], axis=axis
            ),
        )

    @classmethod
    def create_single_sequence(
        cls, source: MSADataSource, tokens: np.ndarray
    ) -> MSAContext:
        values = np.asarray(tokens, dtype=np.uint8)[None]
        return cls(
            tokens=values,
            pairing_key_hash=np.full(values.shape, NO_PAIRING_KEY, np.int32),
            deletion_matrix=np.zeros(values.shape, np.uint8),
            mask=np.ones(values.shape, np.bool_),
            sequence_source=np.full(values.shape, SOURCE_TO_INT[source], np.uint8),
        )

    @classmethod
    def create_empty(cls, n_tokens: int, depth: int = 0) -> MSAContext:
        if n_tokens < 0 or depth < 0:
            raise ValueError("MSA dimensions cannot be negative")
        shape = (depth, n_tokens)
        return cls(
            tokens=np.full(shape, RESIDUE_ORDER[":"], np.uint8),
            pairing_key_hash=np.full(shape, NO_PAIRING_KEY, np.int32),
            deletion_matrix=np.zeros(shape, np.uint8),
            mask=np.zeros(shape, np.bool_),
            sequence_source=np.full(shape, SOURCE_TO_INT[MSADataSource.NONE], np.uint8),
        )


def _a3m_records(text: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    sequence: list[str] = []
    for raw_line in text.replace("\x00", "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(sequence)))
            header = line[1:]
            sequence = []
        elif header is None:
            raise ValueError("A3M content must start with a header")
        else:
            sequence.append(line)
    if header is not None:
        records.append((header, "".join(sequence)))
    if not records or any(not value for _, value in records):
        raise ValueError("A3M content contains an empty record")
    return records


def _stable_pair_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:7], 16)


def msa_context_from_a3m(
    query_sequence: str,
    *,
    paired: str,
    unpaired: str,
) -> MSAContext:
    """Convert cached ColabFold paired/unpaired A3M text to Chai arrays."""
    paired_records = _a3m_records(paired)
    unpaired_records = _a3m_records(unpaired)
    paired_sequences = [sequence for _, sequence in paired_records]
    paired_set = set(paired_sequences)
    single_records = [
        (header, sequence)
        for header, sequence in unpaired_records
        if sequence not in paired_set
    ]
    all_records = [*paired_records, *single_records]
    sequences = [sequence for _, sequence in all_records]
    normalized_query = "".join(
        character
        for character in sequences[0]
        if character.isupper() and character != "-"
    )
    if normalized_query != "".join(query_sequence.split()).upper():
        raise ValueError("paired A3M query does not match the requested sequence")
    tokens, deletions = tokenize_aligned_sequences(sequences)
    paired_count = len(paired_records)
    pairing_keys = [
        _stable_pair_hash(str(index)) if index < paired_count else NO_PAIRING_KEY
        for index in range(len(sequences))
    ]
    sources = [SOURCE_TO_INT[MSADataSource.QUERY]]
    for header, _sequence in all_records[1:]:
        source = (
            MSADataSource.UNIREF90
            if header.startswith("UniRef")
            else MSADataSource.BFD_UNICLUST
        )
        sources.append(SOURCE_TO_INT[source])
    shape = tokens.shape
    return MSAContext(
        tokens=tokens,
        pairing_key_hash=np.broadcast_to(
            np.asarray(pairing_keys, np.int32)[:, None], shape
        ).copy(),
        deletion_matrix=deletions,
        mask=np.ones(shape, np.bool_),
        sequence_source=np.broadcast_to(
            np.asarray(sources, np.uint8)[:, None], shape
        ).copy(),
    )


def _pair_rows(context: MSAContext) -> dict[tuple[int, int], int]:
    distances = np.sum(context.tokens[0] != context.tokens, axis=1)
    grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row, (key, distance) in enumerate(
        zip(context.pairing_key_hash[:, 0], distances, strict=True)
    ):
        if row and int(key) != NO_PAIRING_KEY:
            grouped[int(key)].append((int(distance), row))
    row_to_key: dict[int, tuple[int, int]] = {}
    for key, values in grouped.items():
        order = np.argsort([distance for distance, _ in values], kind="stable")
        # Preserve chai_lab's observable rank assignment: torch.argsort returns
        # row positions, and upstream zips those positions to the original rows
        # rather than computing the inverse permutation.
        for (_, row), rank in zip(values, order, strict=True):
            row_to_key[row] = (key, int(rank))
    result = {_QUERY_PAIR_KEY: 0}
    for row in range(1, context.depth):
        key = int(context.pairing_key_hash[row, 0])
        if key != NO_PAIRING_KEY:
            result[row_to_key[row]] = row
    return result


def _drop_duplicate_rows(context: MSAContext) -> MSAContext:
    if context.depth == 0 or context.num_tokens == 0:
        return context
    _, first = np.unique(context.tokens, axis=0, return_index=True)
    indices = np.sort(first)
    return MSAContext(
        tokens=context.tokens[indices],
        pairing_key_hash=context.pairing_key_hash[indices],
        deletion_matrix=context.deletion_matrix[indices],
        mask=context.mask[indices],
        sequence_source=context.sequence_source[indices],
    )


def parse_aligned_pqt(
    path: str | Path,
    *,
    query_sequence: str | None = None,
    quota_sizes: Mapping[MSADataSource, int] | None = _SOURCE_QUOTAS,
) -> MSAContext:
    """Read Chai's `.aligned.pqt` schema directly with Arrow, without Torch."""
    parquet_path = Path(path)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"aligned MSA is not a regular file: {parquet_path}")
    try:
        table = pq.read_table(parquet_path)
    except (pa.ArrowException, OSError) as error:
        raise ValueError(f"invalid aligned-Parquet file: {parquet_path}") from error
    missing = [name for name in _ALIGNED_PQT_COLUMNS if name not in table.column_names]
    if missing:
        raise ValueError(f"aligned-Parquet schema is missing columns: {missing}")
    columns = {name: table[name].to_pylist() for name in _ALIGNED_PQT_COLUMNS}
    row_count = table.num_rows
    if row_count == 0:
        raise ValueError("aligned-Parquet MSA must contain at least one row")
    for name, values in columns.items():
        if len(values) != row_count or any(
            not isinstance(value, str) for value in values
        ):
            raise ValueError(f"aligned-Parquet column {name!r} must contain strings")

    source_values: list[MSADataSource] = []
    for value in columns["source_database"]:
        try:
            source = MSADataSource(value)
        except ValueError as error:
            raise ValueError(
                f"aligned-Parquet source_database contains unsupported value {value!r}"
            ) from error
        if source not in _RECOGNIZED_ALIGNED_SOURCES:
            raise ValueError(
                f"aligned-Parquet source_database contains unsupported value {value!r}"
            )
        source_values.append(source)
    if source_values[0] is not MSADataSource.QUERY:
        raise ValueError("aligned-Parquet first row must be the query")
    if source_values.count(MSADataSource.QUERY) != 1:
        raise ValueError("aligned-Parquet must contain exactly one query row")
    normalized_query = "".join(
        character
        for character in columns["sequence"][0]
        if character.isupper() and character != "-"
    )
    if query_sequence is not None and normalized_query != query_sequence.upper():
        raise ValueError(
            "aligned-Parquet query sequence does not match the requested protein"
        )

    if quota_sizes is None:
        selected = list(range(row_count))
    else:
        # chai_lab applies a sorted pandas groupby/head before sorting source
        # priorities. Preserve that exact intermediate order: the following
        # priority sort is intentionally NumPy quicksort, matching pandas'
        # unstable default and therefore its within-source row permutation.
        selected = []
        for source in sorted(set(source_values), key=lambda value: value.value):
            quota = quota_sizes.get(source, 1_000_000) - 1
            if quota < 0:
                raise ValueError("MSA source quotas must be positive")
            group_indices = [
                index for index, value in enumerate(source_values) if value is source
            ]
            selected.extend(group_indices[:quota])
    priorities = [
        _SOURCE_PRIORITY.get(source_values[index], 1_000_000) for index in selected
    ]
    selected = [selected[index] for index in _numpy_126_quicksort_indices(priorities)]

    sequences = [columns["sequence"][index] for index in selected]
    tokens, deletions = tokenize_aligned_sequences(sequences)
    pairing_keys = [
        _stable_pair_hash(columns["pairing_key"][index])
        if columns["pairing_key"][index]
        else NO_PAIRING_KEY
        for index in selected
    ]
    source_ids = [
        SOURCE_TO_INT.get(source_values[index], SOURCE_TO_INT[MSADataSource.NONE])
        for index in selected
    ]
    shape = tokens.shape
    return MSAContext(
        tokens=tokens,
        pairing_key_hash=np.broadcast_to(
            np.asarray(pairing_keys, np.int32)[:, None], shape
        ).copy(),
        deletion_matrix=deletions,
        mask=np.ones(shape, np.bool_),
        sequence_source=np.broadcast_to(
            np.asarray(source_ids, np.uint8)[:, None], shape
        ).copy(),
    )


def _merge_profile_msas(contexts: list[MSAContext]) -> MSAContext:
    contexts = [context for context in contexts if context.num_tokens > 0]
    if not contexts:
        raise ValueError("at least one tokenized chain is required for an MSA")
    depth = max(context.depth for context in contexts)
    return MSAContext.concatenate(
        [context.pad(max_msa_depth=depth) for context in contexts], axis=1
    )


def _query_context_for_chain(
    structure: StructureContext, chain_index: int
) -> tuple[MSAContext, np.ndarray]:
    chain_mask = np.asarray(structure.token_asym_id) == chain_index + 1
    residue_indices = np.asarray(structure.token_residue_index)[chain_mask].astype(
        np.int64, copy=False
    )
    residue_types = np.asarray(structure.token_residue_type)[chain_mask]
    if residue_indices.size == 0:
        return MSAContext.create_empty(0), residue_indices
    if residue_indices.min() < 0:
        raise ValueError("MSA token residue indices must be non-negative")
    query = np.empty(int(residue_indices.max()) + 1, np.uint8)
    seen = np.zeros(query.shape, np.bool_)
    for residue_index, residue_type in zip(residue_indices, residue_types, strict=True):
        if not seen[residue_index]:
            query[residue_index] = residue_type
            seen[residue_index] = True
    if not seen.all():
        raise ValueError("MSA token residue indices must be contiguous")
    return (
        MSAContext.create_single_sequence(MSADataSource.QUERY, query),
        residue_indices,
    )


def load_msa_contexts(
    inputs: Sequence[Input],
    structure: StructureContext,
    *,
    msa_directory: str | Path | None = None,
    search_results: Sequence[Mapping[str, str]] | None = None,
) -> tuple[MSAContext, MSAContext]:
    """Build Chai's joined and profile contexts from public local/search paths."""
    if msa_directory is not None and search_results is not None:
        raise ValueError("msa_directory and search_results are mutually exclusive")
    protein_inputs = [
        item for item in inputs if item.entity_type == EntityType.PROTEIN.value
    ]
    if search_results is not None and len(search_results) != len(protein_inputs):
        raise ValueError("MSA search results must align with the protein inputs")
    result_iterator = iter(search_results or ())
    directory = None if msa_directory is None else Path(msa_directory)
    if directory is not None and not directory.is_dir():
        raise NotADirectoryError(f"MSA directory is not a directory: {directory}")

    contexts: list[MSAContext] = []
    for chain_index, item in enumerate(inputs):
        query_context, residue_indices = _query_context_for_chain(
            structure, chain_index
        )
        context = query_context
        if item.entity_type == EntityType.PROTEIN.value:
            if not item.sequence.isascii() or not item.sequence.isalpha():
                if directory is not None or search_results is not None:
                    raise ValueError(
                        "searched MSAs for modified proteins require a canonical "
                        "one-letter query sequence"
                    )
            elif directory is not None:
                path = directory / expected_aligned_pqt_basename(item.sequence)
                if path.is_file():
                    context = parse_aligned_pqt(
                        path, query_sequence=item.sequence.upper()
                    )
                else:
                    warnings.warn(
                        f"no aligned MSA found for protein sequence {item.sequence!r}; "
                        "using the query row only",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            elif search_results is not None:
                context = msa_context_from_search_results(
                    [item.sequence], [next(result_iterator)]
                )
        if context.num_tokens <= int(residue_indices.max(initial=-1)):
            raise ValueError("MSA width does not cover the structure residue indices")
        contexts.append(
            MSAContext(
                tokens=context.tokens[:, residue_indices],
                pairing_key_hash=context.pairing_key_hash[:, residue_indices],
                deletion_matrix=context.deletion_matrix[:, residue_indices],
                mask=context.mask[:, residue_indices],
                sequence_source=context.sequence_source[:, residue_indices],
            )
        )

    deduplicated = [_drop_duplicate_rows(context) for context in contexts]
    profile = _merge_profile_msas(deduplicated)
    joined = pair_and_merge_msas(contexts)
    return _drop_duplicate_rows(joined), profile


def pair_and_merge_msas(contexts: list[MSAContext]) -> MSAContext:
    """Apply Chai's shared-pair ranking, padding, merge, and depth limits."""
    if not contexts:
        raise ValueError("at least one MSA context is required")
    row_maps = [_pair_rows(context) for context in contexts]
    counts = Counter(key for mapping in row_maps for key in mapping)
    nonempty_count = sum(context.depth > 1 for context in contexts)
    selected = [key for key, count in counts.items() if count >= nonempty_count][
        :MAX_PAIRED_DEPTH
    ]
    if not selected or selected[0] != _QUERY_PAIR_KEY:
        raise ValueError("paired MSA query row is missing")

    reordered = []
    for mapping, context in zip(row_maps, contexts, strict=True):
        paired_rows = [mapping.get(key) for key in selected]
        paired_set = set(paired_rows)
        rows = paired_rows + [
            row for row in range(context.depth) if row not in paired_set
        ]
        reordered.append(context.select_rows_with_padding(rows[:FULL_DEPTH]))

    max_depth = max(context.depth for context in reordered)
    merged = MSAContext.concatenate(
        [context.pad(max_msa_depth=max_depth) for context in reordered], axis=1
    )
    return _drop_duplicate_rows(merged)


def msa_context_from_search_results(
    protein_sequences: Sequence[str],
    search_results: Sequence[Mapping[str, str]],
) -> MSAContext:
    """Connect content-addressed search files to Chai's paired model context."""
    if not protein_sequences or len(protein_sequences) != len(search_results):
        raise ValueError(
            "protein sequences and search results must be non-empty and aligned"
        )
    contexts = []
    for sequence, result in zip(protein_sequences, search_results, strict=True):
        try:
            paired_path = Path(result["pairedMsaPath"])
            unpaired_path = Path(result["unpairedMsaPath"])
        except KeyError as error:
            raise ValueError("MSA search result paths are incomplete") from error
        try:
            paired = paired_path.read_text(encoding="utf-8")
            unpaired = unpaired_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise ValueError("MSA search result is unreadable") from error
        contexts.append(
            msa_context_from_a3m(sequence, paired=paired, unpaired=unpaired)
        )
    return pair_and_merge_msas(contexts)
