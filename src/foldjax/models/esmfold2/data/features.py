"""ESMFold2's feature dictionary for a whole job, in NumPy.

Upstream ships `prepare_protein_features`, which covers a single protein chain
with no alignment; everything else its `forward` accepts -- several chains, a
real MSA, symmetry -- is left to the caller. This is that assembly, written
against upstream's conventions rather than around them:

* residue types index a 33-entry vocabulary and are *not* one-hot -- the model
  one-hots them itself;
* atoms are each residue's heavy atoms in upstream's order, carrying its
  reference conformer coordinates, and the atom axis is padded to a multiple
  of 32;
* elements are atomic numbers and atom names are the four-character `ord - 32`
  encoding, neither of which is the AlphaFold-3 one-hot the other ports build;
* the MSA carries residue-type indices with gaps at `MSA_GAP_TOKEN_ID`.

Nothing here imports torch. The tables come from `chemistry`, which is
generated from upstream rather than transcribed, and
`tests/models/esmfold2/test_features_parity.py` checks the single-chain,
no-alignment case against `prepare_protein_features` tensor for tensor.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from foldjax.models.esmfold2.data import chemistry

#: The atom axis is padded to a multiple of this. Upstream's sliding-window
#: attention is happier on a round length, and the padding is masked anyway.
ATOM_BLOCK = 32

# ESMC's tokenizer reserves 1 for right padding.  Keeping the sentinel here
# avoids importing the 6B model implementation into the NumPy featurizer.
ESMC_PAD_TOKEN_ID = 1


# Explicit semantic axes for the feature dictionary returned below.  Shape
# matching is not safe: a perfectly ordinary target can have 32 tokens, which
# is also the atom block size and a fixed channel width elsewhere in the model.
_TOKEN_AXES: dict[str, tuple[int, ...]] = {
    "token_index": (-1,),
    "residue_index": (-1,),
    "asym_id": (-1,),
    "sym_id": (-1,),
    "entity_id": (-1,),
    "mol_type": (-1,),
    "res_type": (-1,),
    "input_ids": (-1,),
    "token_bonds": (-3, -2),
    "token_attention_mask": (-1,),
    "distogram_atom_idx": (-1,),
    "msa": (-1,),
    "msa_attention_mask": (-1,),
    "has_deletion": (-1,),
    "deletion_value": (-1,),
    "deletion_mean": (-1,),
    # Writer-only all-biomolecule metadata. Inference removes these leaves
    # before constructing the JAX PyTree, but serving padding must still keep
    # them aligned with their tokens.
    "token_chain_id_chars": (-2,),
    "token_residue_name_chars": (-2,),
    # Host-side deep-MSA normalization preserves the profile of the complete
    # alignment here before selecting the bounded set of encoder rows.
    "msa_profile": (-2,),
}

_ATOM_AXES: dict[str, tuple[int, ...]] = {
    "ref_pos": (-2,),
    "ref_element": (-1,),
    "ref_charge": (-1,),
    "ref_atom_name_chars": (-2,),
    "ref_space_uid": (-1,),
    "atom_attention_mask": (-1,),
    "atom_to_token": (-1,),
}

_MSA_AXES: dict[str, tuple[int, ...]] = {
    "msa": (-2,),
    "msa_attention_mask": (-2,),
    "has_deletion": (-2,),
    "deletion_value": (-2,),
}

_MSA_TAPE_FEATURES = {
    "msa": "msa_loop_tape",
    "msa_attention_mask": "msa_attention_mask_loop_tape",
    "has_deletion": "has_deletion_loop_tape",
    "deletion_value": "deletion_value_loop_tape",
}
_MSA_ROW_FEATURES = frozenset(_MSA_TAPE_FEATURES)
for _tape_name in _MSA_TAPE_FEATURES.values():
    _TOKEN_AXES[_tape_name] = (-1,)
    _MSA_AXES[_tape_name] = (-2,)
_NUM_RES_TYPES = 33


def build_features(
    chains: Sequence[tuple[str, str, int, int]],
    alignments: dict[int, Path] | None = None,
    *,
    msa_depth: int | None = None,
) -> dict[str, np.ndarray]:
    """Featurize a job, batch axis included.

    `chains` is one entry per chain copy: `(sequence, chain id, entity index,
    symmetry index)`. `alignments` maps an entity index to its a3m; a chain
    whose entity has none contributes only the query row.
    """
    alignments = alignments or {}

    res_type: list[int] = []
    input_ids: list[int] = []
    asym_id: list[int] = []
    sym_id: list[int] = []
    entity_id: list[int] = []
    residue_index: list[int] = []
    atom_records: list[tuple[int, str, str, int, tuple[float, float, float]]] = []
    distogram_atom_idx: list[int] = []
    # Where each chain's tokens start, so its alignment lands in its own
    # columns and nowhere else.
    chain_spans: list[tuple[int, int, int]] = []

    for chain_index, (sequence, _chain_id, entity, symmetry) in enumerate(chains):
        start = len(res_type)
        for position, letter in enumerate(sequence):
            residue = chemistry.PROTEIN_1TO3.get(letter, "UNK")
            names = chemistry.PROTEIN_HEAVY_ATOMS[residue]
            token = len(res_type)
            atom_start = len(atom_records)
            for name in names:
                atom_records.append(
                    (
                        token,
                        name,
                        name[0],
                        chemistry.PROTEIN_CHARGED_ATOMS.get((residue, name), 0),
                        chemistry.PROTEIN_REF_POS[residue][name],
                    )
                )
            representative = "CB" if "CB" in names else "CA"
            distogram_atom_idx.append(atom_start + names.index(representative))
            res_type.append(
                chemistry.PROTEIN_RESIDUE_TO_RES_TYPE.get(
                    residue, chemistry.PROTEIN_UNK_RES_TYPE
                )
            )
            input_ids.append(
                chemistry.ESM_PROTEIN_VOCAB.get(
                    letter, chemistry.ESM_PROTEIN_VOCAB["X"]
                )
            )
            # Chain identity, following upstream's single-chain values: asym_id
            # counts chain copies from zero, entity_id is one-based so that
            # zero stays available as padding, sym_id counts copies within an
            # entity -- which is what tells a homodimer from a heterodimer.
            asym_id.append(chain_index)
            sym_id.append(symmetry)
            entity_id.append(entity + 1)
            # Restarts per chain: the index is a position in its own polymer.
            residue_index.append(position)
        chain_spans.append((start, len(res_type), entity))

    n_token = len(res_type)
    if n_token == 0:
        raise ValueError("the job names no protein residues")
    n_atoms = -(-len(atom_records) // ATOM_BLOCK) * ATOM_BLOCK

    ref_pos = np.zeros((n_atoms, 3), dtype=np.float32)
    ref_element = np.zeros(n_atoms, dtype=np.int64)
    ref_charge = np.zeros(n_atoms, dtype=np.int8)
    ref_atom_name_chars = np.zeros((n_atoms, 4), dtype=np.int64)
    ref_space_uid = np.zeros(n_atoms, dtype=np.int64)
    atom_attention_mask = np.zeros(n_atoms, dtype=bool)
    atom_to_token = np.zeros(n_atoms, dtype=np.int64)
    for index, (token, name, element, charge, position) in enumerate(atom_records):
        ref_pos[index] = position
        ref_element[index] = chemistry.PROTEIN_ELEMENT_TO_ATOMIC_NUM[element]
        ref_charge[index] = charge
        ref_atom_name_chars[index] = chemistry.encode_atom_name(name)
        ref_space_uid[index] = token
        atom_attention_mask[index] = True
        atom_to_token[index] = token

    msa, msa_mask, has_deletion, deletion_value = build_msa(
        chain_spans, res_type, alignments, msa_depth=msa_depth
    )

    features = {
        "token_index": np.arange(n_token, dtype=np.int64),
        "residue_index": np.asarray(residue_index, dtype=np.int64),
        "asym_id": np.asarray(asym_id, dtype=np.int64),
        "sym_id": np.asarray(sym_id, dtype=np.int64),
        "entity_id": np.asarray(entity_id, dtype=np.int64),
        "mol_type": np.full(n_token, chemistry.MOL_TYPE_PROTEIN, dtype=np.int64),
        "res_type": np.asarray(res_type, dtype=np.int64),
        "input_ids": np.asarray(input_ids, dtype=np.int64),
        "token_bonds": np.zeros((n_token, n_token, 1), dtype=np.float32),
        "token_attention_mask": np.ones(n_token, dtype=bool),
        "ref_pos": ref_pos,
        "ref_element": ref_element,
        "ref_charge": ref_charge,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_space_uid": ref_space_uid,
        "atom_attention_mask": atom_attention_mask,
        "atom_to_token": atom_to_token,
        "distogram_atom_idx": np.asarray(distogram_atom_idx, dtype=np.int64),
        "msa": msa,
        "msa_attention_mask": msa_mask,
        "has_deletion": has_deletion,
        "deletion_value": deletion_value,
        "deletion_mean": deletion_value.mean(axis=0),
    }
    return {name: value[None] for name, value in features.items()}


def pad_features(
    features: dict[str, np.ndarray],
    *,
    n_token: int,
    n_atom: int,
    n_msa: int,
) -> dict[str, np.ndarray]:
    """Right-pad all dynamic axes while preserving the real feature prefix.

    ESMFold2 already masks every one of these axes, but padding only the obvious
    token vector is insufficient: the pair features carry two token axes, the
    alignment carries token and row axes, and atom features have their own
    independently compiled length.  This helper owns that schema in one place.

    ``msa`` uses its gap token rather than zero for padded cells.  Its mask is
    still false there, so the choice is numerically inert, while keeping the
    archive meaningful to feature inspection tools.
    """

    tokens = int(np.asarray(features["token_attention_mask"]).shape[-1])
    atoms = int(np.asarray(features["atom_attention_mask"]).shape[-1])
    msa_rows = int(np.asarray(features["msa_attention_mask"]).shape[-2])
    requested = {"tokens": n_token, "atoms": n_atom, "msa": n_msa}
    current = {"tokens": tokens, "atoms": atoms, "msa": msa_rows}
    for axis, target in requested.items():
        if target < current[axis]:
            raise ValueError(
                f"cannot pad ESMFold2 {axis} down from {current[axis]} to {target}"
            )

    padded: dict[str, np.ndarray] = {}
    for name, value in features.items():
        array = np.asarray(value)
        widths = [(0, 0) for _ in array.shape]
        for kind, axes, source, target in (
            ("token", _TOKEN_AXES.get(name, ()), tokens, n_token),
            ("atom", _ATOM_AXES.get(name, ()), atoms, n_atom),
            ("MSA", _MSA_AXES.get(name, ()), msa_rows, n_msa),
        ):
            for declared_axis in axes:
                if not -array.ndim <= declared_axis < array.ndim:
                    raise ValueError(
                        f"ESMFold2 feature {name!r} has shape {array.shape}, "
                        f"which has no declared {kind} axis {declared_axis}"
                    )
                axis = declared_axis % array.ndim
                if array.shape[axis] != source:
                    raise ValueError(
                        f"ESMFold2 feature {name!r} has {array.shape[axis]} entries "
                        f"on {kind} axis {axis}, expected {source}"
                    )
                widths[axis] = (0, target - source)

        fill = (
            chemistry.MSA_GAP_TOKEN_ID
            if name in {"msa", "msa_loop_tape"}
            else ESMC_PAD_TOKEN_ID
            if name == "input_ids"
            else 0
        )
        padded[name] = np.pad(array, widths, constant_values=fill)
    return padded


def normalize_msa_features(
    features: dict[str, np.ndarray],
    *,
    n_msa: int,
    row_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Materialize exact per-loop MSA rows before serving-bucket padding.

    The released model samples at most ``max_msa_depth`` rows separately in
    every trunk loop.  A serving bucket cannot retain an arbitrarily deep raw
    alignment, however, and padding invalid rows above the sampling cap would
    let the random permutation select dummy rows. On the opt-in padding path,
    the caller therefore supplies the exact query-preserving row indices that
    the original model would select for each loop. They become a fixed-shape
    tape; the compiled loop still performs every original key split, but reads
    the already selected rows instead of permuting the padded storage.

    ``msa_profile`` captures the profile of the complete alignment before any
    rows are removed; ``deletion_mean`` is already a token-only aggregate and
    is preserved as-is.  The helper refuses feature layouts it cannot crop
    coherently rather than guessing which dimension is the MSA axis.
    """

    if n_msa < 1:
        raise ValueError(f"ESMFold2 MSA target must be positive; got {n_msa}")

    missing = sorted(_MSA_ROW_FEATURES - features.keys())
    if "token_attention_mask" not in features:
        missing.append("token_attention_mask")
    if "deletion_mean" not in features:
        missing.append("deletion_mean")
    if missing:
        raise ValueError(
            "cannot normalize ESMFold2 MSA without row features: "
            + ", ".join(missing)
        )

    mask = np.asarray(features["msa_attention_mask"])
    if mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError(
            "ESMFold2 MSA normalization requires one batched alignment with "
            f"shape [1, rows, tokens]; got {mask.shape}"
        )
    rows, tokens = mask.shape[1:]
    token_mask = np.asarray(features["token_attention_mask"]).astype(bool)
    if token_mask.shape != (1, tokens):
        raise ValueError(
            "ESMFold2 token mask does not match the MSA token axis: "
            f"{token_mask.shape} versus {(1, tokens)}"
        )
    bool_mask = mask.astype(bool)
    if not np.array_equal(bool_mask[:, 0], token_mask) or np.any(
        bool_mask & ~token_mask[:, None, :]
    ):
        raise ValueError(
            "ESMFold2 MSA normalization requires query row 0 to match the "
            "token mask and every hit row to stay within it"
        )
    for name in _MSA_ROW_FEATURES:
        value = np.asarray(features[name])
        if (
            value.ndim != 3
            or value.shape[:2] != (1, rows)
            or value.shape[2] != tokens
        ):
            raise ValueError(
                f"ESMFold2 MSA row feature {name!r} has shape {value.shape}, "
                f"expected {(1, rows, tokens)}"
            )
    if "msa_profile" in features:
        profile_shape = np.asarray(features["msa_profile"]).shape
        expected_profile = (1, tokens, _NUM_RES_TYPES)
        if profile_shape != expected_profile:
            raise ValueError(
                f"ESMFold2 feature 'msa_profile' has shape {profile_shape}, "
                f"expected {expected_profile}"
            )

    active_rows = np.flatnonzero(np.any(bool_mask, axis=-1)[0])
    if active_rows.size == 0 or active_rows[0] != 0:
        raise ValueError(
            "ESMFold2 MSA normalization requires a valid query in row 0"
        )
    normalized = dict(features)
    if "msa_profile" not in normalized:
        msa = np.asarray(features["msa"])
        active_ids = msa[bool_mask]
        if np.any(active_ids < 0) or np.any(active_ids >= _NUM_RES_TYPES):
            raise ValueError(
                "cannot normalize ESMFold2 MSA with residue ids outside "
                f"[0, {_NUM_RES_TYPES - 1}]"
            )
        safe_ids = np.where(bool_mask, msa, 0)
        one_hot = np.eye(_NUM_RES_TYPES, dtype=np.float32)[safe_ids]
        one_hot *= bool_mask[..., None].astype(np.float32)
        counts = np.clip(bool_mask.astype(np.float32).sum(axis=1), 1.0, None)
        normalized["msa_profile"] = one_hot.sum(axis=1) / counts[..., None]

    indices = np.asarray(row_indices)
    if (
        indices.ndim != 2
        or indices.shape[0] < 1
        or indices.shape[1] < 1
        or indices.shape[1] > n_msa
    ):
        raise ValueError(
            "ESMFold2 MSA loop indices must have shape [loops, selected_rows] "
            f"with selected_rows <= {n_msa}; got {indices.shape}"
        )
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("ESMFold2 MSA loop indices must be integers")
    if np.any(indices < 0) or np.any(indices >= rows):
        raise ValueError(
            f"ESMFold2 MSA loop indices exceed stored depth {rows}"
        )
    valid_rows = np.any(bool_mask, axis=-1)[0]
    if np.any(~valid_rows[indices]):
        raise ValueError(
            "ESMFold2 MSA loop indices select stored dummy rows; refusing an "
            "alignment layout that cannot be normalized safely"
        )
    if np.any(indices[:, 0] != 0):
        raise ValueError("ESMFold2 MSA loop indices must keep query row 0")
    if np.any(np.diff(indices, axis=1) <= 0):
        raise ValueError(
            "ESMFold2 MSA loop indices must be unique and sorted per loop"
        )

    first_loop = indices[0]
    for name, tape_name in _MSA_TAPE_FEATURES.items():
        value = np.asarray(features[name])
        normalized[name] = np.take(value, first_loop, axis=1)
        # np.take inserts both index axes after batch; put loops first so lax.scan
        # consumes exactly one [batch, rows, tokens] block at each iteration.
        tape = np.take(value, indices, axis=1)
        normalized[tape_name] = np.swapaxes(tape, 0, 1)
    return normalized


def build_msa(
    chain_spans: Sequence[tuple[int, int, int]],
    res_type: Sequence[int],
    alignments: dict[int, Path],
    *,
    msa_depth: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble one alignment block per chain, gaps everywhere else.

    Row 0 is the query: every chain's own sequence, which is what the model
    reads when there is no alignment at all. Each further row belongs to one
    chain and is a gap over the others -- the unpaired arrangement, which is
    what a per-chain a3m can honestly support. Pairing rows across chains would
    claim the alignments were searched together, and they were not.
    """
    n_token = len(res_type)
    rows: list[list[int]] = [list(res_type)]
    deletions: list[list[float]] = [[0.0] * n_token]

    for start, end, entity in chain_spans:
        path = alignments.get(entity)
        if path is None:
            continue
        query = "".join(
            chemistry.RES_TYPE_TO_LETTER.get(value, "X")
            for value in res_type[start:end]
        )
        for sequence, counts in read_a3m(path, query=query):
            row = [chemistry.MSA_GAP_TOKEN_ID] * n_token
            deletion = [0.0] * n_token
            for offset, (letter, count) in enumerate(zip(sequence, counts)):
                residue = (
                    chemistry.PROTEIN_1TO3.get(letter, "UNK") if letter != "-" else None
                )
                row[start + offset] = (
                    chemistry.MSA_GAP_TOKEN_ID
                    if residue is None
                    else chemistry.PROTEIN_RESIDUE_TO_RES_TYPE.get(
                        residue, chemistry.PROTEIN_UNK_RES_TYPE
                    )
                )
                deletion[start + offset] = float(count)
            rows.append(row)
            deletions.append(deletion)
            if msa_depth is not None and len(rows) >= msa_depth:
                break

    msa = np.asarray(rows, dtype=np.int64)
    values = np.asarray(deletions, dtype=np.float32)
    return msa, np.ones(msa.shape, dtype=bool), values > 0, values


def read_a3m(path: Path, *, query: str) -> list[tuple[str, list[int]]]:
    """Aligned rows and their insertion counts, without the query row.

    a3m writes insertions as lowercase letters, which occupy no query column;
    each is counted against the column that follows it, which is what
    `deletion_value` means. A row whose aligned length is not the query's is
    dropped rather than padded -- it is not an alignment to this sequence --
    and a row identical to the query is dropped because row 0 already is it.
    Identity is the test rather than position: a search that does not repeat
    the query first would otherwise lose a real hit.
    """
    rows: list[tuple[str, list[int]]] = []
    for block in Path(path).read_text(encoding="utf-8").split(">")[1:]:
        if not block.strip():
            continue
        body = "".join(block.split("\n")[1:])
        aligned: list[str] = []
        counts: list[int] = []
        pending = 0
        for character in body:
            if character.islower():
                pending += 1
                continue
            aligned.append(character.upper() if character != "-" else "-")
            counts.append(pending)
            pending = 0
        if len(aligned) != len(query):
            continue
        sequence = "".join(aligned)
        if sequence == query and not any(counts):
            continue
        rows.append((sequence, counts))
    return rows


__all__ = [
    "ATOM_BLOCK",
    "ESMC_PAD_TOKEN_ID",
    "build_features",
    "build_msa",
    "normalize_msa_features",
    "pad_features",
    "read_a3m",
]
