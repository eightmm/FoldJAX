"""Torch-free OpenFold3 query featurization.

The vendored OpenFold3 data pipeline mixes otherwise NumPy/RDKit preprocessing
with a thin PyTorch tensorization layer and a Lightning ``Dataset`` wrapper.
Inference needs neither abstraction.  This module keeps the upstream chemistry,
tokenization, and MSA parsing code, then materializes the released model features
directly as NumPy arrays.

Only one query is handled at a time.  Arrays returned here are unbatched; the public
adapter in :mod:`foldjax.models.openfold3.data.featurize` adds the leading batch axis.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

logger = logging.getLogger(__name__)
_MAX_TEMPLATE_ALIGNMENT_HITS = 200


class RawFeatures(NamedTuple):
    """One featurized query and the exact atom array it was built from."""

    features: dict[str, np.ndarray]
    atom_array: Any


def _one_hot(indices: np.ndarray, classes: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= classes):
        raise ValueError(f"one-hot index outside [0, {classes}): {indices}")
    return np.eye(classes, dtype=np.int32)[indices]


def _pad_axis0(array: np.ndarray, size: int) -> np.ndarray:
    """Pad the first axis to ``size`` with zeros."""
    array = np.asarray(array)
    if array.shape[0] > size:
        raise ValueError(
            f"OpenFold3 feature has {array.shape[0]} tokens, larger than {size}"
        )
    widths = [(0, size - array.shape[0]), *((0, 0),) * (array.ndim - 1)]
    return np.pad(array, widths)


def _sym_ids(atom_array: Any, token_starts: np.ndarray) -> np.ndarray:
    """Implement upstream ``create_sym_id`` without its Torch import closure."""
    import biotite.structure as struc

    chain_starts = struc.get_chain_starts(atom_array, add_exclusive_stop=True)
    entity_per_chain = atom_array.entity_id[chain_starts[:-1]]
    counters: dict[int, int] = {}
    sym_per_chain = np.empty(entity_per_chain.shape, dtype=np.int32)
    for index, entity in enumerate(entity_per_chain):
        key = int(entity)
        counters[key] = counters.get(key, 0) + 1
        sym_per_chain[index] = counters[key]
    sym_per_atom = np.repeat(sym_per_chain, np.diff(chain_starts))
    return sym_per_atom[token_starts]


def _token_bonds(atom_array: Any, token_index: np.ndarray) -> np.ndarray:
    """Create the SI Table 5 token-bond matrix using NumPy."""
    result = np.zeros((len(token_index), len(token_index)), dtype=np.int32)
    if atom_array.bonds is None:
        return result
    bonds = np.asarray(atom_array.bonds.as_array())
    if bonds.size == 0:
        return result

    endpoints = bonds[:, :2].astype(np.int64, copy=False)
    endpoints = endpoints[atom_array.is_atomized[endpoints].all(axis=1)]
    if endpoints.size == 0:
        return result

    token_to_position = {int(token): index for index, token in enumerate(token_index)}
    token_endpoints = atom_array.token_id[endpoints]
    positions = np.asarray(
        [
            (token_to_position[int(left)], token_to_position[int(right)])
            for left, right in token_endpoints
        ],
        dtype=np.int64,
    )
    result[positions[:, 0], positions[:, 1]] = 1
    result[positions[:, 1], positions[:, 0]] = 1
    return result


def _structure_features(atom_array: Any, n_tokens: int) -> dict[str, np.ndarray]:
    """NumPy equivalent of ``featurize_structure_of3(..., is_gt=False)``."""
    import biotite.structure as struc

    from foldjax.models.openfold3._upstream.openfold3.core.data.primitives.structure.labels import (  # noqa: E501
        get_token_starts,
    )
    from foldjax.models.openfold3._upstream.openfold3.core.data.resources.residues import (  # noqa: E501
        STANDARD_RESIDUES_WITH_GAP_3,
        MoleculeType,
        get_with_unknown_3_to_idx,
    )

    starts_with_stop = get_token_starts(atom_array, add_exclusive_stop=True)
    starts = starts_with_stop[:-1]
    token_index = np.asarray(atom_array.token_id[starts], dtype=np.int32)
    counts = np.asarray(np.diff(starts_with_stop), dtype=np.int32)

    chain_ids = atom_array.chain_id[starts]
    _, renumbered = np.unique(chain_ids, return_inverse=True)
    chain_starts = struc.get_chain_starts(atom_array)
    segment_chain_ids = atom_array.chain_id[chain_starts]
    if len(segment_chain_ids) != len(np.unique(segment_chain_ids)):
        logger.warning("Chain IDs are not unique within complex.")

    molecule_type = atom_array.molecule_type_id[starts]
    features: dict[str, np.ndarray] = {
        "token_index": token_index,
        "restype": _one_hot(
            get_with_unknown_3_to_idx(atom_array.res_name[starts]),
            len(STANDARD_RESIDUES_WITH_GAP_3),
        ).astype(np.int32, copy=False),
        "is_protein": (molecule_type == MoleculeType.PROTEIN).astype(np.int32),
        "is_rna": (molecule_type == MoleculeType.RNA).astype(np.int32),
        "is_dna": (molecule_type == MoleculeType.DNA).astype(np.int32),
        "is_ligand": (molecule_type == MoleculeType.LIGAND).astype(np.int32),
        "is_atomized": np.asarray(atom_array.is_atomized[starts], dtype=np.int32),
        "token_mask": np.ones(len(starts), dtype=np.float32),
        "num_atoms_per_token": counts,
        "start_atom_index": np.asarray(starts, dtype=np.int32),
        "atom_mask": np.ones(int(counts.sum()), dtype=np.float32),
        "residue_index": np.asarray(atom_array.res_id[starts], dtype=np.int32),
        "asym_id": np.asarray(renumbered + 1, dtype=np.int32),
        "entity_id": np.asarray(atom_array.entity_id[starts], dtype=np.int32),
        "sym_id": _sym_ids(atom_array, starts).astype(np.int32, copy=False),
        "token_bonds": _token_bonds(atom_array, token_index),
        "atom_to_token_index": np.repeat(
            np.arange(len(starts), dtype=np.int32), counts
        ),
    }

    token_features = (
        "token_index",
        "restype",
        "is_protein",
        "is_rna",
        "is_dna",
        "is_ligand",
        "is_atomized",
        "token_mask",
        "num_atoms_per_token",
        "start_atom_index",
        "residue_index",
        "asym_id",
        "entity_id",
        "sym_id",
    )
    for name in token_features:
        features[name] = _pad_axis0(features[name], n_tokens)
    if features["token_bonds"].shape != (n_tokens, n_tokens):
        pad = n_tokens - features["token_bonds"].shape[0]
        features["token_bonds"] = np.pad(features["token_bonds"], ((0, pad), (0, pad)))
    return features


def _quaternion_rotation(quaternion: np.ndarray) -> np.ndarray:
    """Match OpenFold3's scalar-first ``quat_to_rot`` convention."""
    a, b, c, d = np.asarray(quaternion, dtype=np.float32)
    return np.asarray(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d],
        ],
        dtype=np.float32,
    )


def _augment_reference_positions(
    positions: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Deterministic NumPy implementation of AF3 Algorithm 19."""
    positions = np.asarray(positions, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    quaternion = rng.standard_normal(4, dtype=np.float32)
    quaternion /= np.linalg.norm(quaternion)
    rotation = _quaternion_rotation(quaternion)
    translation = rng.standard_normal(3, dtype=np.float32)
    centre = np.sum(positions * mask[:, None], axis=-2, keepdims=True) / np.sum(mask)
    augmented = (positions - centre) @ rotation.T + translation[None, :]
    return np.asarray(augmented * mask[:, None], dtype=np.float32)


def _conformer_features(
    processed_ref_molecules: list[Any], *, seed: int
) -> dict[str, np.ndarray]:
    """NumPy equivalent of OpenFold3 reference-conformer tensorization."""
    from rdkit import Chem

    rng = np.random.default_rng(seed)
    periodic_table = Chem.GetPeriodicTable()
    positions: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    elements: list[int] = []
    charges: list[int] = []
    atom_name_chars: list[list[int]] = []
    space_uids: list[int] = []

    for molecule_index, processed in enumerate(processed_ref_molecules):
        molecule = processed.mol
        conformer = molecule.GetConformer()
        molecule_positions: list[tuple[float, float, float]] = []
        molecule_masks: list[int] = []

        for atom, in_crop in zip(
            molecule.GetAtoms(), processed.in_crop_mask, strict=True
        ):
            if not in_crop:
                continue
            point = conformer.GetAtomPosition(atom.GetIdx())
            molecule_positions.append((point.x, point.y, point.z))
            molecule_masks.append(int(atom.GetBoolProp("annot_used_atom_mask")))
            symbol = atom.GetSymbol()
            elements.append(
                118 if symbol == "R" else periodic_table.GetAtomicNumber(symbol) - 1
            )
            charges.append(atom.GetFormalCharge())
            space_uids.append(molecule_index)
            atom_name = atom.GetProp("annot_atom_name").ljust(4)
            atom_name_chars.append(
                [ord(character) - 32 for character in atom_name]
            )

        molecule_mask = np.asarray(molecule_masks, dtype=np.int32)
        molecule_position = np.asarray(
            molecule_positions, dtype=np.float32
        ).reshape(-1, 3)
        if np.any(molecule_mask):
            molecule_position = _augment_reference_positions(
                molecule_position, molecule_mask, rng
            )
        positions.append(molecule_position)
        masks.append(molecule_mask)

    if positions:
        ref_pos = np.concatenate(positions, axis=0).astype(np.float32, copy=False)
        ref_mask = np.concatenate(masks, axis=0).astype(np.int32, copy=False)
    else:
        ref_pos = np.empty((0, 3), dtype=np.float32)
        ref_mask = np.empty((0,), dtype=np.int32)

    return {
        "ref_pos": ref_pos,
        "ref_mask": ref_mask,
        "ref_element": _one_hot(np.asarray(elements, dtype=np.int64), 119),
        "ref_charge": np.asarray(charges, dtype=np.float32),
        "ref_atom_name_chars": _one_hot(
            np.asarray(atom_name_chars, dtype=np.int64).reshape(-1, 4), 64
        ),
        "ref_space_uid": np.asarray(space_uids, dtype=np.int32),
    }


class _MsaPrecursor(NamedTuple):
    msa_index: np.ndarray
    deletion_matrix: np.ndarray
    n_rows_paired: int
    msa_mask: np.ndarray
    profile: np.ndarray
    deletion_mean: np.ndarray


def _msa_precursor(
    atom_array: Any, collection: Any, n_tokens: int
) -> _MsaPrecursor:
    """Map processed upstream MSA arrays onto query token positions."""
    from foldjax.models.openfold3._upstream.openfold3.core.data.primitives.structure.labels import (  # noqa: E501
        get_token_starts,
    )
    from foldjax.models.openfold3._upstream.openfold3.core.data.resources.residues import (  # noqa: E501
        STANDARD_RESIDUES_WITH_GAP_1,
        map_str_array_to_idx_array,
    )

    gap_index = STANDARD_RESIDUES_WITH_GAP_1.index("-")
    if not collection.chain_id_to_query_seq:
        return _MsaPrecursor(
            msa_index=np.full((1, n_tokens), gap_index, dtype=np.int64),
            deletion_matrix=np.zeros((1, n_tokens), dtype=np.int64),
            n_rows_paired=1,
            msa_mask=np.ones((1, n_tokens), dtype=np.float32),
            profile=np.zeros(
                (n_tokens, len(STANDARD_RESIDUES_WITH_GAP_1)), dtype=np.float32
            ),
            deletion_mean=np.zeros(n_tokens, dtype=np.float32),
        )

    rows = int(collection.row_counts.n_rows_total)
    msa_index = np.full((rows, n_tokens), gap_index, dtype=np.int64)
    deletion_matrix = np.zeros((rows, n_tokens), dtype=np.int64)
    msa_mask = np.ones((rows, n_tokens), dtype=np.float32)
    profile = np.zeros(
        (n_tokens, len(STANDARD_RESIDUES_WITH_GAP_1)), dtype=np.float32
    )
    deletion_mean = np.zeros(n_tokens, dtype=np.float32)

    for chain_id in collection.chain_id_to_rep_id:
        stacked = collection.chain_id_to_query_seq[chain_id]
        if collection.row_counts.n_rows_paired_subsampled > 0:
            stacked = stacked.concatenate(
                collection.chain_id_to_paired_msa[chain_id], axis=0
            )
        if collection.row_counts.n_rows_main_subsampled:
            stacked = stacked.concatenate(
                collection.chain_id_to_main_msa[chain_id], axis=0
            )
        stacked, _stacked_mask = stacked.pad(target_length=rows, axis=0)

        chain_array = atom_array[atom_array.chain_id == chain_id]
        chain_starts = get_token_starts(chain_array)
        token_positions = chain_array[chain_starts].token_position
        columns = chain_array[chain_starts].res_id - 1
        mapped_msa = stacked.msa[:, columns]
        msa_index[:, token_positions] = map_str_array_to_idx_array(
            msa_array=mapped_msa,
            molecule_type=collection.chain_id_to_mol_type[chain_id],
        )
        deletion_matrix[:, token_positions] = stacked.deletion_matrix[:, columns]
        profile[token_positions] = collection.chain_id_to_profile[chain_id][columns]
        deletion_mean[token_positions] = collection.chain_id_to_deletion_mean[chain_id][
            columns
        ]

    token_mask = np.zeros(n_tokens, dtype=np.float32)
    token_mask[: len(get_token_starts(atom_array))] = 1
    msa_mask *= token_mask[None, :]
    return _MsaPrecursor(
        msa_index=msa_index,
        deletion_matrix=deletion_matrix,
        n_rows_paired=int(collection.row_counts.n_rows_paired_subsampled) + 1,
        msa_mask=msa_mask,
        profile=profile,
        deletion_mean=deletion_mean,
    )


def _subsample_msa_precursor(
    precursor: _MsaPrecursor,
    msa_depth: int | None,
) -> _MsaPrecursor:
    """Apply the inference row selection before expanding categorical MSA IDs."""

    if msa_depth is None or precursor.msa_index.shape[0] <= msa_depth:
        return precursor
    if msa_depth < 1:
        raise ValueError(f"msa_depth must be at least 1; got {msa_depth}")
    valid = precursor.msa_mask.sum(axis=-1) > 0
    order = np.argsort(~valid, kind="stable")[:msa_depth]
    return precursor._replace(
        msa_index=precursor.msa_index[order],
        deletion_matrix=precursor.deletion_matrix[order],
        msa_mask=precursor.msa_mask[order],
    )


def _tensorize_msa_precursor(
    precursor: _MsaPrecursor,
    *,
    compact_msa: bool = False,
) -> dict[str, np.ndarray]:
    """Tensorize a selected precursor for archives or private prediction.

    Portable/direct callers retain the released 32-channel int32 one-hot. The
    managed prediction path stores the same categorical IDs as uint8 and lets the
    first model consumer reconstruct that one-hot inside the compiled graph.
    """

    from foldjax.models.openfold3._upstream.openfold3.core.data.resources.residues import (  # noqa: E501
        STANDARD_RESIDUES_WITH_GAP_1,
    )

    deletion = precursor.deletion_matrix
    classes = len(STANDARD_RESIDUES_WITH_GAP_1)
    msa_index = np.asarray(precursor.msa_index)
    if np.any(msa_index < 0) or np.any(msa_index >= classes):
        raise ValueError(f"one-hot index outside [0, {classes}): {msa_index}")
    features = {
        "has_deletion": (deletion != 0).astype(np.float32),
        "deletion_value": (
            np.arctan(deletion / 3.0) * (8.0 / np.pi)
        ).astype(np.float32),
        "deletion_mean": precursor.deletion_mean.astype(np.float32, copy=False),
        "profile": precursor.profile.astype(np.float32, copy=False),
        "num_paired_seqs": np.asarray([precursor.n_rows_paired], dtype=np.int32),
        "msa_mask": precursor.msa_mask.astype(np.float32, copy=False),
    }
    if compact_msa:
        from foldjax.models.openfold3.data.featurize import (
            _COMPACT_MSA_INDICES,
            _COMPACT_MSA_MARKER,
        )

        features[_COMPACT_MSA_INDICES] = msa_index.astype(np.uint8, copy=False)
        features[_COMPACT_MSA_MARKER] = np.zeros((), dtype=np.float32)
    else:
        features["msa"] = _one_hot(msa_index, classes)
    return features


def _msa_features(
    query: Any,
    atom_array: Any,
    n_tokens: int,
    settings: Any,
    *,
    msa_depth: int | None = None,
    compact_msa: bool = False,
) -> dict[str, np.ndarray]:
    """Run upstream parsing/pairing, then tensorize the result with NumPy."""
    from foldjax.models.openfold3._upstream.openfold3.core.config.msa_pipeline_configs import (  # noqa: E501
        MsaSampleProcessorInputInference,
    )
    from foldjax.models.openfold3._upstream.openfold3.core.data.pipelines.sample_processing.msa import (  # noqa: E501
        MsaSampleProcessorInference,
    )
    create_input = MsaSampleProcessorInputInference.create_from_inference_query_entry
    processor_input = create_input(inference_query=query)
    collection = MsaSampleProcessorInference(config=settings)(input=processor_input)
    precursor = _msa_precursor(atom_array, collection, n_tokens)
    precursor = _subsample_msa_precursor(precursor, msa_depth)
    return _tensorize_msa_precursor(precursor, compact_msa=compact_msa)


def _empty_template_features(
    n_tokens: int,
    *,
    n_templates: int = 4,
    lazy_pair_features: bool = False,
) -> dict[str, np.ndarray]:
    """Released-shape template features for a query with no templates."""
    from foldjax.models.openfold3._upstream.openfold3.core.data.resources.residues import (  # noqa: E501
        STANDARD_RESIDUES_WITH_GAP_3,
    )

    gap = STANDARD_RESIDUES_WITH_GAP_3.index("GAP")
    indices = np.full((n_templates, n_tokens), gap, dtype=np.int64)

    def zero(shape: tuple[int, ...]) -> np.ndarray:
        if not lazy_pair_features:
            return np.zeros(shape, dtype=np.float32)
        # Raw prediction compacts these exact-zero fields immediately after
        # validation.  A zero-stride view preserves their complete shape and
        # dtype contract without materializing the quadratic storage first.
        return np.broadcast_to(np.zeros((), dtype=np.float32), shape)

    return {
        "template_restype": _one_hot(indices, len(STANDARD_RESIDUES_WITH_GAP_3)),
        "template_pseudo_beta_mask": zero((n_templates, n_tokens)),
        "template_backbone_frame_mask": zero((n_templates, n_tokens)),
        "template_distogram": zero((n_templates, n_tokens, n_tokens, 39)),
        "template_unit_vector": zero((n_templates, n_tokens, n_tokens, 3)),
    }


class _TemplateCandidate(NamedTuple):
    name: str
    entry: Any
    score: float


def _cif_template_metadata(path: Path) -> tuple[Any, dict[str, str], str]:
    """Load the local structure metadata required by template filtering."""
    from foldjax.models.openfold3._upstream.openfold3.core.data.io.structure.cif import (  # noqa: E501
        _load_ciffile,
    )
    from foldjax.models.openfold3._upstream.openfold3.core.data.primitives.structure.metadata import (  # noqa: E501
        get_asym_id_to_canonical_seq_dict,
        get_cif_block,
        get_release_date,
    )

    if not path.is_file():
        raise FileNotFoundError(f"OpenFold3 template CIF does not exist: {path}")
    cif_file = _load_ciffile(path)
    sequence_by_chain = get_asym_id_to_canonical_seq_dict(cif_file)
    release_date = get_release_date(get_cif_block(cif_file)).strftime("%Y-%m-%d")
    return cif_file, sequence_by_chain, release_date


def _aligned_residue_map(template: Any) -> np.ndarray | None:
    """Return only residue-to-residue columns from a supplied alignment."""
    if template.query_aln_pos is None or template.aln_pos is None:
        return None
    query_positions = np.asarray(template.query_aln_pos, dtype=np.int64)
    template_positions = np.asarray(template.aln_pos, dtype=np.int64)
    if query_positions.shape != template_positions.shape:
        raise ValueError(
            f"template {template.entry_id}_{template.chain_id} has mismatched "
            "query/template alignment coordinate shapes"
        )
    aligned = (query_positions > 0) & (template_positions > 0)
    return np.stack(
        (query_positions[aligned], template_positions[aligned]), axis=-1
    )


def _template_entry_from_cif(
    path: Path,
    query_sequence: str,
    *,
    preferred_chain_id: str | None,
) -> _TemplateCandidate | None:
    """Align one local CIF to a query and return an upstream cache entry."""
    from foldjax.models.openfold3._upstream.openfold3.core.data.io.sequence.template import (  # noqa: E501
        CifDirectParser,
    )
    from foldjax.models.openfold3._upstream.openfold3.core.data.primitives.structure.template import (  # noqa: E501
        TemplateCacheEntry,
    )

    _cif_file, sequence_by_chain, release_date = _cif_template_metadata(path)
    parsed = CifDirectParser(max_sequences=None, min_score_threshold=0.1)(
        cif_file_path=path,
        query_seq_str=query_sequence,
        chain_id_seq_map=sequence_by_chain,
        entry_id=path.stem,
        specified_chain_id=preferred_chain_id,
    )
    if not parsed:
        return None
    template = parsed[0]
    index_map = _aligned_residue_map(template)
    if index_map is None or not len(index_map):
        return None
    name = f"{template.entry_id}_{template.chain_id}"
    return _TemplateCandidate(
        name=name,
        entry=TemplateCacheEntry(
            index=int(template.index),
            release_date=release_date,
            idx_map=index_map,
            cif_path=path,
        ),
        score=float(template.seq_id) * float(template.q_cov or 0.0),
    )


def _rank_direct_template_entries(
    paths: list[Path], chain_ids: list[str | None], query_sequence: str
) -> dict[str, Any]:
    """Apply CIF-direct quality filtering, rank globally, and keep four."""
    candidates: list[_TemplateCandidate] = []
    failures: list[str] = []
    for index, path in enumerate(paths):
        preferred = chain_ids[index] if index < len(chain_ids) else None
        try:
            candidate = _template_entry_from_cif(
                path,
                query_sequence,
                preferred_chain_id=preferred,
            )
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f"{path}: {error}")
            logger.warning(
                "Skipping invalid OpenFold3 template CIF %s: %s", path, error
            )
            continue
        if candidate is not None:
            candidates.append(candidate)
    if not candidates and failures:
        raise ValueError(
            "none of the local OpenFold3 template CIFs passed structure/release "
            f"validation: {'; '.join(failures)}"
        )
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return {
        candidate.name: dataclasses.replace(candidate.entry, index=rank)
        for rank, candidate in enumerate(candidates[:4])
    }


def _candidate_cif_path(source: Path, entry_id: str) -> Path | None:
    candidates = (
        source.parent / f"{entry_id}.cif",
        source.parent / "template_structures" / f"{entry_id}.cif",
        source.parent.parent / "template_structures" / f"{entry_id}.cif",
    )
    return next((path for path in candidates if path.is_file()), None)


def _read_a3m_template_prefix(path: Path, *, max_hits: int) -> str:
    """Read only the query and hit records an A3M parser can return."""

    lines: list[str] = []
    records = 0
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip().startswith(">"):
                if records >= max_hits + 1:
                    break
                records += 1
            lines.append(line)
    return "".join(lines)


def _alignment_templates(path: Path, query_sequence: str) -> list[Any]:
    """Read ordered template hits while preserving residue-wise coordinates."""
    from foldjax.models.openfold3._upstream.openfold3.core.data.io.sequence.template import (  # noqa: E501
        A3mParser,
        StoParser,
    )

    parsers = {".a3m": A3mParser, ".sto": StoParser}
    parser_type = parsers.get(path.suffix.lower())
    if parser_type is None:
        raise ValueError(
            "OpenFold3 template alignment must be .a3m, .sto, or a preprocessed "
            f".npz cache, got {path}"
        )
    alignment_source = (
        _read_a3m_template_prefix(path, max_hits=_MAX_TEMPLATE_ALIGNMENT_HITS)
        if path.suffix.lower() == ".a3m"
        else path.read_text()
    )
    parsed = parser_type(max_sequences=_MAX_TEMPLATE_ALIGNMENT_HITS)(
        alignment_source, query_sequence
    )
    templates = list(parsed.values())
    if templates and templates[0].seq == query_sequence:
        templates = templates[1:]
    return templates


def _matching_structure_chain(
    template: Any, sequences: Mapping[str, str]
) -> str | None:
    """Match the alignment sequence to a local CIF chain without realigning it."""
    if template.seq is None:
        return None
    original = sequences.get(template.chain_id)
    if original is not None and template.seq in original:
        return str(template.chain_id)
    return next(
        (
            str(chain_id)
            for chain_id, sequence in sequences.items()
            if chain_id != template.chain_id and template.seq in sequence
        ),
        None,
    )


def _template_entry_from_alignment(
    path: Path,
    template: Any,
    query_sequence: str,
) -> _TemplateCandidate | None:
    """Create a local entry, retaining the alignment's residue correspondence."""
    from foldjax.models.openfold3._upstream.openfold3.core.data.primitives.structure.template import (  # noqa: E501
        TemplateCacheEntry,
    )

    index_map = _aligned_residue_map(template)
    mapping_is_complete = (
        bool(template.seq)
        and index_map is not None
        and bool(template.q_cov)
    )
    if not mapping_is_complete:
        return _template_entry_from_cif(
            path,
            query_sequence,
            preferred_chain_id=template.chain_id,
        )

    _cif_file, sequences, release_date = _cif_template_metadata(path)
    matched_chain = _matching_structure_chain(template, sequences)
    if matched_chain is None or not len(index_map):
        return None
    name = f"{template.entry_id}_{matched_chain}"
    return _TemplateCandidate(
        name=name,
        entry=TemplateCacheEntry(
            index=int(template.index),
            release_date=release_date,
            idx_map=index_map,
            cif_path=path,
        ),
        score=float(template.seq_id) * float(template.q_cov),
    )


def _local_cif_or_error(alignment_path: Path, entry_id: str) -> Path:
    cif_path = _candidate_cif_path(alignment_path, entry_id)
    if cif_path is None:
        raise FileNotFoundError(
            f"OpenFold3 template {entry_id!r} has no local CIF. Place "
            f"{entry_id}.cif beside {alignment_path}, in a template_structures/ "
            "sibling directory, or use template_cif_paths. FoldJAX prediction "
            "never fetches templates implicitly."
        )
    return cif_path


def _alignment_template_entries(
    path: Path,
    query_sequence: str,
    requested_ids: list[str],
) -> dict[str, Any]:
    try:
        parsed = _alignment_templates(path, query_sequence)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"failed to parse OpenFold3 template alignment {path}: {error}"
        ) from error
    by_name = {
        f"{template.entry_id}_{template.chain_id}": template for template in parsed
    }
    order = requested_ids or list(by_name)
    entries: dict[str, Any] = {}
    failures: list[str] = []
    for template_id in order:
        if len(entries) == 4:
            break
        template = by_name.get(template_id)
        try:
            if template is None:
                try:
                    entry_id, preferred_chain = template_id.split("_", maxsplit=1)
                except ValueError:
                    entry_id, preferred_chain = template_id, None
                candidate = _template_entry_from_cif(
                    _local_cif_or_error(path, entry_id),
                    query_sequence,
                    preferred_chain_id=preferred_chain,
                )
            else:
                candidate = _template_entry_from_alignment(
                    _local_cif_or_error(path, template.entry_id),
                    template,
                    query_sequence,
                )
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f"{template_id}: {error}")
            logger.warning(
                "Skipping invalid OpenFold3 alignment template %s: %s",
                template_id,
                error,
            )
            continue
        if candidate is not None:
            entries[candidate.name] = dataclasses.replace(
                candidate.entry, index=len(entries)
            )
    if not entries and failures:
        raise ValueError(
            "none of the OpenFold3 alignment templates passed local structure "
            f"validation: {'; '.join(failures)}"
        )
    return entries


def _preprocessed_template_entries(
    path: Path, template_ids: list[str] | None
) -> tuple[dict[str, Any], Path | None]:
    """Load up to four valid cache entries without enabling NumPy pickle.

    Requested IDs retain caller order. Unknown, duplicated, or malformed
    candidates are reported and skipped while later IDs are still considered.
    If candidates were requested but none are usable, the failure is explicit.
    """
    from foldjax.models.openfold3._upstream.openfold3.core.data.primitives.structure.template import (  # noqa: E501
        TemplateCacheEntry,
    )

    with np.load(path, allow_pickle=False) as archive:
        if "entries_json" not in archive.files:
            raise ValueError(
                f"OpenFold3 template cache {path} is not a safe FoldJAX cache: "
                "expected a plain-text entries_json array. Object-valued NumPy "
                "archives are not loaded because they require pickle."
            )
        encoded = np.asarray(archive["entries_json"])
        if encoded.dtype.kind not in {"U", "S"} or encoded.size != 1:
            raise ValueError(
                f"OpenFold3 template cache {path} entries_json must be one string"
            )
        try:
            text = encoded.reshape(()).item()
            if isinstance(text, bytes):
                text = text.decode("utf-8")
            stored = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"OpenFold3 template cache {path} entries_json is not valid JSON"
            ) from error
    if not isinstance(stored, dict):
        raise ValueError(f"OpenFold3 template cache {path} manifest must be an object")
    ordered_ids = template_ids or list(stored)
    structure_directory = path.parent.parent / "template_structures"
    if not structure_directory.is_dir():
        structure_directory = None

    entries: dict[str, Any] = {}
    failures: list[str] = []
    seen: set[str] = set()
    for template_id in ordered_ids:
        if len(entries) == 4:
            break
        if template_id in seen:
            failures.append(f"{template_id}: duplicate requested ID")
            continue
        seen.add(template_id)
        if template_id not in stored:
            failures.append(f"{template_id}: ID is absent from the cache")
            continue
        try:
            if template_id.count("_") != 1:
                raise ValueError("ID must be '<entry_id>_<chain_id>'")
            entry_id, chain_id = template_id.split("_")
            if not entry_id or not chain_id:
                raise ValueError("ID must be '<entry_id>_<chain_id>'")

            value = stored[template_id]
            if not isinstance(value, dict):
                raise TypeError("manifest value must be an object")
            index_map = np.asarray(value["idx_map"])
            if index_map.dtype.kind not in {"i", "u"} or (
                index_map.ndim != 2 or index_map.shape[1:] != (2,)
            ):
                raise ValueError("idx_map must be an integer array with shape (N, 2)")
            index_map = index_map[(index_map > 0).all(axis=1)]
            if not len(index_map):
                raise ValueError("idx_map has no aligned residue pairs")

            release_date = value.get("release_date", "")
            if not isinstance(release_date, str):
                raise TypeError("release_date must be a string")
            raw_cif_path = value.get("cif_path")
            cif_path: Path | None = None
            if raw_cif_path is not None:
                if not isinstance(raw_cif_path, str) or not raw_cif_path:
                    raise TypeError("cif_path must be a non-empty string")
                cif_path = Path(raw_cif_path)
                if not cif_path.is_absolute():
                    cif_path = path.parent / cif_path
                if not cif_path.is_file():
                    raise FileNotFoundError(
                        f"local CIF {cif_path} does not exist; FoldJAX never "
                        "fetches templates implicitly"
                    )
            else:
                if structure_directory is None:
                    raise FileNotFoundError(
                        "no cif_path and no adjacent template_structures/ directory"
                    )
                expected_cif = structure_directory / f"{entry_id}.cif"
                if not expected_cif.is_file():
                    raise FileNotFoundError(
                        f"adjacent local CIF {expected_cif} does not exist; FoldJAX "
                        "never fetches templates implicitly"
                    )

            entries[template_id] = TemplateCacheEntry(
                index=len(entries),
                release_date=release_date,
                idx_map=index_map.astype(np.int64, copy=False),
                cif_path=cif_path,
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            failures.append(f"{template_id}: {error}")

    if failures:
        logger.warning(
            "Skipped invalid OpenFold3 template cache candidates from %s: %s",
            path,
            "; ".join(failures),
        )
    if not entries and failures:
        scope = "requested" if template_ids else "cached"
        raise ValueError(
            f"none of the {scope} OpenFold3 templates in {path} are valid local "
            f"entries: {'; '.join(failures)}"
        )
    return entries, structure_directory


def _template_slices(
    query: Any,
    atom_array: Any,
    *,
    ccd: Any,
) -> dict[str, list[Any]]:
    """Create per-chain template slices from every public query input form."""
    from foldjax.models.openfold3._upstream.openfold3.core.data.primitives.structure.template import (  # noqa: E501
        align_template_to_query,
    )

    result: dict[str, list[Any]] = {}
    cif_cache: dict[str, tuple] = {}
    for chain in query.chains:
        if chain.sequence is None:
            continue
        direct_paths = list(chain.template_cif_paths or ())
        direct_chain_ids = list(chain.template_cif_chain_ids or ())
        alignment_path = (
            None
            if chain.template_alignment_file_path is None
            else Path(chain.template_alignment_file_path)
        )
        requested_ids = list(chain.template_entry_chain_ids or ())

        for query_chain_id in chain.chain_ids:
            entries: dict[str, Any] = {}
            structure_directory: Path | None = None

            if direct_paths:
                entries = _rank_direct_template_entries(
                    [Path(path) for path in direct_paths],
                    direct_chain_ids,
                    chain.sequence,
                )

            elif alignment_path is not None and alignment_path.suffix.lower() == ".npz":
                entries, structure_directory = _preprocessed_template_entries(
                    alignment_path, requested_ids or None
                )

            elif alignment_path is not None:
                entries = _alignment_template_entries(
                    alignment_path,
                    chain.sequence,
                    requested_ids,
                )

            if not entries:
                result[query_chain_id] = []
                continue
            chain_slices: list[Any] = []
            failures: list[str] = []
            for template_name, entry in entries.items():
                try:
                    chain_slices.extend(
                        align_template_to_query(
                            sampled_template_data={template_name: entry},
                            template_structures_directory=structure_directory,
                            template_structure_array_directory=None,
                            template_file_format="cif",
                            ccd=ccd,
                            atom_array_query_chain=atom_array[
                                atom_array.chain_id == query_chain_id
                            ],
                            cif_assembly_cache=cif_cache,
                        )
                    )
                except (IndexError, KeyError, TypeError, ValueError) as error:
                    failures.append(f"{template_name}: {error}")
                    logger.warning(
                        "Skipping malformed OpenFold3 template %s: %s",
                        template_name,
                        error,
                    )
            if not chain_slices and failures:
                raise ValueError(
                    "none of the selected OpenFold3 templates could be aligned: "
                    f"{'; '.join(failures)}"
                )
            result[query_chain_id] = chain_slices
    return result


def _template_precursor(
    template_slices: Mapping[str, list[Any]], n_templates: int, n_tokens: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract residue, pseudo-beta, and backbone-frame arrays from slices."""
    import biotite.structure as struc

    residue_names = np.full((n_templates, n_tokens), "GAP", dtype="U3")
    pseudo_beta = np.full((n_templates, n_tokens, 3), np.nan, dtype=np.float32)
    frame = np.full((n_templates, n_tokens, 3, 3), np.nan, dtype=np.float32)

    input_count = 0
    accepted_count = 0
    failures: list[str] = []
    for slices in template_slices.values():
        template_index = 0
        for template_slice in slices:
            if template_index == n_templates:
                break
            input_count += 1
            try:
                template = template_slice.atom_array
                repeats = np.asarray(
                    template_slice.template_residue_repeats, dtype=np.int64
                )
                positions = np.asarray(
                    template_slice.query_token_positions, dtype=np.int64
                )
                if repeats.ndim != 1 or positions.ndim != 1:
                    raise ValueError("residue repeats and token positions must be 1D")
                if (repeats < 0).any():
                    raise ValueError("template residue repeats cannot be negative")
                if ((positions < 0) | (positions >= n_tokens)).any():
                    raise IndexError("template query token position is out of range")
                residue_starts = struc.get_residue_starts(template)
                candidate_residue_names = np.repeat(
                    template[residue_starts].res_name, repeats
                )
                if len(candidate_residue_names) != len(positions):
                    raise ValueError(
                        "template residue names do not match query token positions"
                    )

                is_gly = template.res_name == "GLY"
                is_ca = template.atom_name == "CA"
                is_cb = template.atom_name == "CB"
                pseudo_mask = (is_gly & is_ca) | (~is_gly & is_cb)
                if int(pseudo_mask.sum()) != len(residue_starts):
                    raise ValueError("template lacks one pseudo-beta atom per residue")
                candidate_pseudo_beta = np.repeat(
                    template[pseudo_mask].coord, repeats, axis=0
                )
                if candidate_pseudo_beta.shape != (len(positions), 3):
                    raise ValueError(
                        "template pseudo-beta coordinates do not match token positions"
                    )

                frame_coordinates = []
                for atom_name in ("N", "CA", "C"):
                    coordinates = template[template.atom_name == atom_name].coord
                    repeated = np.repeat(coordinates, repeats, axis=0)
                    if repeated.shape != (len(positions), 3):
                        raise ValueError(
                            f"template {atom_name} coordinates do not match token "
                            "positions"
                        )
                    frame_coordinates.append(repeated)

                residue_names[template_index, positions] = candidate_residue_names
                pseudo_beta[template_index, positions] = candidate_pseudo_beta
                for frame_index, coordinates in enumerate(frame_coordinates):
                    frame[template_index, positions, frame_index] = coordinates
                template_index += 1
                accepted_count += 1
            except (
                AttributeError,
                IndexError,
                KeyError,
                TypeError,
                ValueError,
            ) as error:
                failures.append(str(error))
                logger.warning(
                    "Skipping invalid OpenFold3 template slice: %s", error
                )

    if input_count and not accepted_count and failures:
        raise ValueError(
            "none of the aligned OpenFold3 template slices were structurally "
            f"usable: {'; '.join(failures)}"
        )
    return residue_names, pseudo_beta, frame


def _dot3(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Three-component dot product with upstream's fixed operation order."""
    return (
        left[..., 0] * right[..., 0]
        + left[..., 1] * right[..., 1]
        + left[..., 2] * right[..., 2]
    )


def _normalize(vector: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    squared_norm = _dot3(vector, vector)[..., None]
    norm = np.sqrt(np.maximum(squared_norm, epsilon**2))
    return vector / norm


def _template_features(
    query: Any,
    atom_array: Any,
    n_tokens: int,
    *,
    ccd_file_path: str | None,
    n_templates: int = 4,
    lazy_empty_pair_features: bool = False,
) -> dict[str, np.ndarray]:
    """Featurize optional templates with local NumPy geometry."""
    from biotite.structure.io import pdbx

    from foldjax.models.openfold3._upstream.openfold3.core.data.primitives.structure.component import (  # noqa: E501
        BiotiteCCDWrapper,
    )
    from foldjax.models.openfold3._upstream.openfold3.core.data.primitives.structure.labels import (  # noqa: E501
        get_token_starts,
    )
    from foldjax.models.openfold3._upstream.openfold3.core.data.resources.residues import (  # noqa: E501
        STANDARD_RESIDUES_WITH_GAP_3,
        get_with_unknown_3_to_idx,
    )

    has_input = any(
        chain.template_alignment_file_path is not None
        or chain.template_cif_paths is not None
        for chain in query.chains
    )
    if not has_input:
        return _empty_template_features(
            n_tokens,
            n_templates=n_templates,
            lazy_pair_features=lazy_empty_pair_features,
        )

    ccd = (
        BiotiteCCDWrapper()
        if ccd_file_path is None
        else pdbx.CIFFile.read(ccd_file_path)
    )
    slices = _template_slices(query, atom_array, ccd=ccd)
    residue_names, pseudo_beta, frame = _template_precursor(
        slices, n_templates, n_tokens
    )
    pseudo_mask = (~np.isnan(pseudo_beta).any(axis=-1)).astype(np.float32)
    frame_mask = (~np.isnan(frame).any(axis=(-2, -1))).astype(np.float32)
    template_restype = _one_hot(
        get_with_unknown_3_to_idx(residue_names),
        len(STANDARD_RESIDUES_WITH_GAP_3),
    )

    squared_distance = np.sum(
        (pseudo_beta[..., :, None, :] - pseudo_beta[..., None, :, :]) ** 2,
        axis=-1,
        keepdims=True,
    )
    lower = np.linspace(3.25, 50.75, 39, dtype=np.float32) ** 2
    upper = np.concatenate((lower[1:], np.asarray([1e8], dtype=np.float32)))
    distogram = ((squared_distance > lower) & (squared_distance < upper)).astype(
        np.float32
    )

    starts = get_token_starts(atom_array)
    _, asym = np.unique(atom_array.chain_id[starts], return_inverse=True)
    same_chain = (asym[:, None] == asym[None, :]).astype(np.float32)
    pair_mask = pseudo_mask[..., :, None] * pseudo_mask[..., None, :] * same_chain
    distogram *= pair_mask[..., None]

    frame_zero = np.nan_to_num(frame, nan=0.0)
    n_position = frame_zero[..., 0, :]
    ca_position = frame_zero[..., 1, :]
    c_position = frame_zero[..., 2, :]
    axis_x = _normalize(c_position - ca_position)
    axis_y_seed = n_position - ca_position
    axis_y = _normalize(
        axis_y_seed - _dot3(axis_y_seed, axis_x)[..., None] * axis_x
    )
    axis_z = np.cross(axis_x, axis_y)
    delta = ca_position[..., None, :, :] - ca_position[..., :, None, :]
    local = np.stack(
        (
            _dot3(delta, axis_x[..., :, None, :]),
            _dot3(delta, axis_y[..., :, None, :]),
            _dot3(delta, axis_z[..., :, None, :]),
        ),
        axis=-1,
    )
    unit_vector = _normalize(local).astype(np.float32)
    frame_pair_mask = (
        frame_mask[..., :, None] * frame_mask[..., None, :] * same_chain
    )
    unit_vector *= frame_pair_mask[..., None]

    return {
        "template_restype": template_restype.astype(np.int32, copy=False),
        "template_pseudo_beta_mask": pseudo_mask,
        "template_backbone_frame_mask": frame_mask,
        "template_distogram": distogram,
        "template_unit_vector": unit_vector,
    }


def featurize_query_numpy(
    query: Any,
    *,
    seed: int,
    msa_settings: Any,
    ccd_file_path: str | None = None,
    msa_depth: int | None = None,
    lazy_empty_template_pairs: bool = False,
    compact_msa: bool = False,
) -> RawFeatures:
    """Build released inference features without importing PyTorch.

    ``msa_depth`` is an inference-only host cap applied before categorical MSA
    storage. ``compact_msa`` keeps those selected IDs as private uint8 graph
    input; the default expands to the 32-channel archive ABI. ``None`` depth
    preserves every row for standalone feature archives and direct callers.
    """
    from foldjax.models.openfold3._upstream.openfold3.core.data.primitives.structure.query import (  # noqa: E501
        structure_with_ref_mols_from_query,
    )
    from foldjax.models.openfold3._upstream.openfold3.core.data.primitives.structure.tokenization import (  # noqa: E501
        add_token_positions,
        get_token_count,
        tokenize_atom_array,
    )

    # Upstream's RDKit conformer helper draws its embedding seed from Python's
    # module-level RNG.  Scope that draw to the public seed while restoring the
    # caller's RNG state afterwards, so repeated preprocessing is reproducible
    # without perturbing application-level randomness.
    random_state = random.getstate()
    random.seed(int(seed))
    try:
        atom_array, processed_reference_molecules = structure_with_ref_mols_from_query(
            query=query
        )
    finally:
        random.setstate(random_state)
    tokenize_atom_array(atom_array)
    add_token_positions(atom_array)
    n_tokens = get_token_count(atom_array)

    features = _structure_features(atom_array, n_tokens)
    features.update(
        _conformer_features(processed_reference_molecules, seed=int(seed))
    )
    features.update(
        _msa_features(
            query,
            atom_array,
            n_tokens,
            msa_settings,
            msa_depth=msa_depth,
            compact_msa=compact_msa,
        )
    )
    features.update(
        _template_features(
            query,
            atom_array,
            n_tokens,
            ccd_file_path=ccd_file_path,
            lazy_empty_pair_features=lazy_empty_template_pairs,
        )
    )
    return RawFeatures(features=features, atom_array=atom_array)
