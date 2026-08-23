"""Template featurization from provided mmCIF templates (torch-parity).

Implements the inference JSON-template path of Protenix: parse a provided
template structure, map query residues to template residues via supplied
``queryIndices``/``templateIndices``, and build the exact feature arrays the
JAX ``template_embedder`` consumes (``template_aatype``,
``template_pseudo_beta_mask``, ``template_distogram``, ``template_unit_vector``,
``template_backbone_frame_mask``), stacked over ``max_templates``.

Ground truth: ``protenix.data.template`` (``parse_json_templates`` ->
``package``/``fix``/``reduce`` -> ``Templates.as_protenix_dict``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

ATOM37_NUM = 37
# torch ``ATOM37_ORDER``.
ATOM37_ORDER = {
    "N": 0,
    "CA": 1,
    "C": 2,
    "CB": 3,
    "O": 4,
    "CG": 5,
    "CG1": 6,
    "CG2": 7,
    "OG": 8,
    "OG1": 9,
    "SG": 10,
    "CD": 11,
    "CD1": 12,
    "CD2": 13,
    "ND1": 14,
    "ND2": 15,
    "OD1": 16,
    "OD2": 17,
    "SD": 18,
    "CE": 19,
    "CE1": 20,
    "CE2": 21,
    "CE3": 22,
    "NE": 23,
    "NE1": 24,
    "NE2": 25,
    "OE1": 26,
    "OE2": 27,
    "CH2": 28,
    "NH1": 29,
    "NH2": 30,
    "OH": 31,
    "CZ": 32,
    "CZ2": 33,
    "CZ3": 34,
    "NZ": 35,
    "OXT": 36,
}
# BioPython ``PDBData.protein_letters_3to1`` (20 standard residues only).
_PROTEIN_3TO1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}
# torch ``TEMPLATE_PROTEIN_SEQ_TO_ID``.
_TEMPLATE_PROTEIN_SEQ_TO_ID = {
    "A": 0,
    "B": 3,
    "C": 4,
    "D": 3,
    "E": 6,
    "F": 13,
    "G": 7,
    "H": 8,
    "I": 9,
    "J": 20,
    "K": 11,
    "L": 10,
    "M": 12,
    "N": 2,
    "O": 20,
    "P": 14,
    "Q": 5,
    "R": 1,
    "S": 15,
    "T": 16,
    "U": 4,
    "V": 19,
    "W": 17,
    "X": 20,
    "Y": 18,
    "Z": 6,
    "-": 31,
}
_GAP_AATYPE = 31  # torch ``STD_RESIDUES_WITH_GAP["-"]``.
_DENSE_NUM = 24
# torch ``PROTEIN_AATYPE_DENSE_ATOM_TO_ATOM37`` (32, 24): per-restype atom37
# indices for each dense slot, right-padded with 0 to 24. Restypes 20-31
# (gap/non-standard) map every dense slot to atom37 index 0.
_DENSE_TO_ATOM37_RAGGED: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 4, 3),
    (0, 1, 2, 4, 3, 5, 11, 23, 32, 29, 30),
    (0, 1, 2, 4, 3, 5, 16, 15),
    (0, 1, 2, 4, 3, 5, 16, 17),
    (0, 1, 2, 4, 3, 10),
    (0, 1, 2, 4, 3, 5, 11, 26, 25),
    (0, 1, 2, 4, 3, 5, 11, 26, 27),
    (0, 1, 2, 4),
    (0, 1, 2, 4, 3, 5, 14, 13, 20, 25),
    (0, 1, 2, 4, 3, 6, 7, 12),
    (0, 1, 2, 4, 3, 5, 12, 13),
    (0, 1, 2, 4, 3, 5, 11, 19, 35),
    (0, 1, 2, 4, 3, 5, 18, 19),
    (0, 1, 2, 4, 3, 5, 12, 13, 20, 21, 32),
    (0, 1, 2, 4, 3, 5, 11),
    (0, 1, 2, 4, 3, 8),
    (0, 1, 2, 4, 3, 9, 7),
    (0, 1, 2, 4, 3, 5, 12, 13, 24, 21, 22, 33, 34, 28),
    (0, 1, 2, 4, 3, 5, 12, 13, 20, 21, 32, 31),
    (0, 1, 2, 4, 3, 6, 7),
)
_DENSE_TO_ATOM37 = np.zeros((32, 24), dtype=np.int64)
for _r, _slots in enumerate(_DENSE_TO_ATOM37_RAGGED):
    _DENSE_TO_ATOM37[_r, : len(_slots)] = _slots
# torch ``RESTYPE_PSEUDOBETA_INDEX`` (dense-atom index per restype).
_PSEUDOBETA_INDEX = np.array(
    [
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        1,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        0,
        22,
        23,
        14,
        14,
        0,
        21,
        22,
        13,
        13,
        0,
        0,
    ],
    dtype=np.int64,
)
# torch ``RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX[:, 0]`` backbone frame (C, CA, N).
_BACKBONE_FRAME = np.array(
    [[2, 1, 0]] * 20
    + [
        [0, 0, 0],
        [12, 8, 6],
        [12, 8, 6],
        [12, 8, 6],
        [12, 8, 6],
        [0, 0, 0],
        [12, 8, 6],
        [12, 8, 6],
        [12, 8, 6],
        [12, 8, 6],
        [0, 0, 0],
        [0, 0, 0],
    ],
    dtype=np.int64,
)
_DGRAM_MIN_BIN = 3.25
_DGRAM_MAX_BIN = 50.75
_DGRAM_NUM_BINS = 39
MAX_TEMPLATES = 4

#: Every field ``_as_protenix_dict`` produces with a leading template axis.
#: Named here rather than in the padding module because this is where the
#: template axis is created; padding imports it so the two cannot drift.
TEMPLATE_FIELDS = (
    "template_aatype",
    "template_atom_positions",
    "template_atom_mask",
    "template_pseudo_beta_mask",
    "template_distogram",
    "template_unit_vector",
    "template_backbone_frame_mask",
)


def dedup_templates(features: Mapping[str, Any]) -> Mapping[str, Any]:
    """Keep one copy of each distinct template, with how many it stands for.

    ``TemplateEmbedder`` runs the whole pairformer stack once per row and
    divides the sum by the row count, so identical rows cost their multiple to
    produce the value one already gives. That is not hypothetical: a query with
    fewer than ``MAX_TEMPLATES`` hits is padded up to four, and the padding is
    *zeros* rather than the gap restype (``_pad_templates``, and torch's
    ``pad_to`` before it). A query with no hits at all therefore has

        row 0   all-gap restype 31, everything else zero
        rows 1-3   restype 0, everything else zero -- identical to each other

    which is **two** distinct templates, not one and not four. Collapsing all
    four would be wrong: ``template_aatype`` feeds ``jnp.eye(32)[aatype]``, so
    restype 31 and restype 0 are different inputs and give different
    embeddings. Keeping all four is four stack evaluations to compute two
    values.

    So this deduplicates rather than collapses, and emits
    ``template_multiplicity`` alongside the survivors.
    ``template_embedder`` weights each row by its multiplicity and divides by
    their sum, which is the original row count -- so the value is preserved and
    the work is not. "Every row identical" is simply the case where the distinct
    set has size one.

    NOT BIT-IDENTICAL, and the gate must be a tolerance. Two separate reasons,
    and **the obvious third reason is false** -- see below, because a reader who
    checks it, finds it false, and concludes the tolerance is unnecessary would
    tighten this into a gate that passes by luck.

    1. The divisor is ``1e-7 + n`` and that epsilon does not scale with ``n``,
       so ``m * x / (1e-7 + m)`` is not ``x / (1e-7 + 1)``.
    2. The **association** of the running sum changes once a second distinct
       row exists. The loop computes ``((f0 + f1) + f1) + f1``; this computes
       ``f0 + 3*f1``. Measured on random float32 pairs, those disagree 44.7% of
       the time.

    What is *not* a reason: that ``m * f`` rounds differently from ``f + f +
    f``. It does not. ``x + x`` is exact, so ``(x + x) + x`` rounds the same
    exact value ``3 * x`` does, and over 300,000 random float32 values at
    ``m = 2, 3, 4`` **zero** disagreed. This docstring asserted the opposite
    until it was measured.

    Must run *after* padding: ``pad_protenix_features`` requires the native
    template depth of four and raises otherwise.

    Args:
        features: a feature mapping. Template fields are found through
            :data:`TEMPLATE_FIELDS`, so a subset without them is returned
            unchanged.

    Returns:
        A new ``dict`` carrying only the distinct template rows plus
        ``template_multiplicity``. When there is nothing to deduplicate the
        input mapping is returned **unchanged and by identity** -- which is why
        the return is annotated ``Mapping`` rather than ``dict``: on those paths
        the caller gets back exactly what it passed, whatever type that was -- a no-op
        that returns a copy is not a no-op, and OpenDDE's CLI wiring test
        asserts the featurizer's own dict reaches the model.
    """
    present = {
        name: np.asarray(features[name])
        for name in TEMPLATE_FIELDS
        if name in features
    }
    if not present:
        return features

    rows = {array.shape[0] for array in present.values()}
    if len(rows) != 1:
        raise ValueError(
            f"template features disagree on their row count: {sorted(rows)}"
        )
    (n_templates,) = rows
    if n_templates <= 1:
        return features

    # Smallest first, so a distinct pair of restypes -- the field that actually
    # discriminates on the default path -- answers the question before the
    # gigabytes of distogram are ever compared. Measured at 3012 tokens: 7 us
    # to reject, against 185 ms if every field had to be walked.
    ordered = sorted(present.items(), key=lambda item: item[1].nbytes)

    keep: list[int] = []
    multiplicity: list[int] = []
    for index in range(n_templates):
        for slot, representative in enumerate(keep):
            if all(
                np.array_equal(array[representative], array[index])
                for _name, array in ordered
            ):
                multiplicity[slot] += 1
                break
        else:
            keep.append(index)
            multiplicity.append(1)

    if len(keep) == n_templates:
        return features

    out = dict(features)
    for name, array in present.items():
        out[name] = array[keep]
    out["template_multiplicity"] = np.asarray(multiplicity, dtype=np.float32)
    return out


def _encode_template_restype(sequence: str) -> np.ndarray:
    return np.array(
        [
            _TEMPLATE_PROTEIN_SEQ_TO_ID.get(r, _TEMPLATE_PROTEIN_SEQ_TO_ID["X"])
            for r in sequence
        ],
        dtype=np.int32,
    )


def _parse_template_mmcif(
    mmcif_string: str,
    *,
    chain_id: str | None = None,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Parse a simplified template mmCIF chain, torch ``parse_simple_cif``.

    Returns ``(template_seq, atom37_positions, atom37_mask)``, zero-centered
    across masked atoms like the torch JSON path (``_zero_center=True``).
    """

    import gemmi

    doc = gemmi.cif.read_string(mmcif_string)
    structure = gemmi.make_structure_from_block(doc[0])
    model = structure[0]
    chains = list(model)
    if not chains:
        raise ValueError("no chains found in simplified mmCIF")
    from foldjax.models.protenix.data.search.templates import (
        _mmcif_chain_sequences,
        _template_chain_sequence,
    )

    seqres_chains = _mmcif_chain_sequences(mmcif_string)
    if seqres_chains:
        actual_chain_id, chain_sequence = _template_chain_sequence(
            mmcif_string, chain_id
        )
        chain = next(
            (candidate for candidate in chains if candidate.name == actual_chain_id),
            None,
        )
        if chain is None:
            available = [candidate.name for candidate in chains]
            raise ValueError(
                f"template chain {actual_chain_id!r} not found; "
                f"available chains: {available}"
            )
        template_seq = chain_sequence.sequence
        num_res = len(template_seq)
        residue_indices = (
            (
                residue,
                int(residue.label_seq) - chain_sequence.seqres_start,
            )
            for residue in chain
            if residue.label_seq is not None
        )
    else:
        if chain_id is None:
            chain = chains[0]
        else:
            chain = next(
                (candidate for candidate in chains if candidate.name == chain_id),
                None,
            )
            if chain is None:
                available = [candidate.name for candidate in chains]
                raise ValueError(
                    f"template chain {chain_id!r} not found; "
                    f"available chains: {available}"
                )
        residues = list(chain)
        num_res = len(residues)
        template_seq = "".join(_PROTEIN_3TO1.get(r.name, "X") for r in residues)
        residue_indices = ((residue, index) for index, residue in enumerate(residues))

    all_pos = np.zeros((num_res, ATOM37_NUM, 3), dtype=np.float32)
    all_mask = np.zeros((num_res, ATOM37_NUM), dtype=np.float32)
    for res, i in residue_indices:
        if not 0 <= i < num_res:
            continue
        selected_occupancy: dict[int, float] = {}
        for atom in res:
            name = atom.name
            coord = (atom.pos.x, atom.pos.y, atom.pos.z)
            if name in ATOM37_ORDER:
                idx = ATOM37_ORDER[name]
            elif name.upper() == "SE" and res.name == "MSE":
                idx = ATOM37_ORDER["SD"]
            else:
                continue
            occupancy = float(atom.occ)
            if idx in selected_occupancy and occupancy <= selected_occupancy[idx]:
                continue
            selected_occupancy[idx] = occupancy
            all_pos[i, idx] = coord
            all_mask[i, idx] = 1.0
        cd, nh1, nh2 = ATOM37_ORDER["CD"], ATOM37_ORDER["NH1"], ATOM37_ORDER["NH2"]
        if res.name == "ARG" and all(all_mask[i, [cd, nh1, nh2]]):
            if np.linalg.norm(all_pos[i, nh1] - all_pos[i, cd]) > np.linalg.norm(
                all_pos[i, nh2] - all_pos[i, cd]
            ):
                all_pos[i, [nh1, nh2]] = all_pos[i, [nh2, nh1]]
                all_mask[i, [nh1, nh2]] = all_mask[i, [nh2, nh1]]

    mask_bool = all_mask.astype(bool)
    if np.any(mask_bool):
        center = all_pos[mask_bool].mean(axis=0)
        all_pos[mask_bool] -= center
    return template_seq, all_pos, all_mask


def _single_json_template(
    template_info: dict[str, Any], num_query: int
) -> dict[str, np.ndarray]:
    """Port of one iteration of torch ``parse_json_templates``."""

    mmcif_str = template_info.get("mmcif", "")
    q_indices = template_info.get("queryIndices", [])
    t_indices = template_info.get("templateIndices", [])
    if not mmcif_str or len(q_indices) != len(t_indices):
        raise ValueError("invalid template info: missing mmcif or index mismatch")
    mapping = dict(zip(q_indices, t_indices))
    template_seq, all_pos, all_mask = _parse_template_mmcif(
        mmcif_str, chain_id=template_info.get("chainId")
    )

    out_pos = np.zeros((num_query, ATOM37_NUM, 3), dtype=np.float32)
    out_mask = np.zeros((num_query, ATOM37_NUM), dtype=np.float32)
    out_seq = ["-"] * num_query
    for q_idx, t_idx in mapping.items():
        if t_idx != -1 and t_idx < len(all_pos) and q_idx < num_query:
            out_pos[q_idx] = all_pos[t_idx]
            out_mask[q_idx] = all_mask[t_idx]
            out_seq[q_idx] = template_seq[t_idx] if t_idx < len(template_seq) else "-"
    out_seq_str = "".join(out_seq)
    return {
        "template_all_atom_positions": out_pos,
        "template_all_atom_masks": out_mask,
        "template_aatype": _encode_template_restype(out_seq_str),
    }


def _fix_to_dense(
    aatype: np.ndarray, atom_positions: np.ndarray, atom_masks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """atom37 -> dense-24 gather; torch ``fix_template_features``."""

    dense_idx = _DENSE_TO_ATOM37[aatype]  # (T, R, 24)
    atom_mask = np.take_along_axis(atom_masks, dense_idx, axis=2)
    atom_pos = np.take_along_axis(atom_positions, dense_idx[..., None], axis=2)
    atom_pos *= atom_mask[..., None]
    return atom_pos.astype(np.float32), atom_mask.astype(np.int32)


def _dgram_from_positions(positions: np.ndarray) -> np.ndarray:
    """torch ``dgram_from_positions``."""

    lower = np.square(
        np.linspace(_DGRAM_MIN_BIN, _DGRAM_MAX_BIN, _DGRAM_NUM_BINS, dtype=np.float32)
    )
    upper = np.empty_like(lower)
    upper[:-1] = lower[1:]
    upper[-1] = 1e8
    diff = positions[:, None, :] - positions[None, :, :]
    dist2 = np.einsum("ijk,ijk->ij", diff, diff)[..., None]
    return ((dist2 > lower) & (dist2 < upper)).astype(np.float32)


def _unit_vector(
    aatype: np.ndarray, atom_positions: np.ndarray, atom_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """torch ``compute_template_unit_vector``."""

    eps = 1e-6
    bb = _BACKBONE_FRAME[aatype]
    c_idx, ca_idx, n_idx = bb[:, 0], bb[:, 1], bb[:, 2]
    r = np.arange(aatype.shape[0])
    c_pos = atom_positions[r, c_idx].astype(np.float32)
    ca_pos = atom_positions[r, ca_idx].astype(np.float32)
    n_pos = atom_positions[r, n_idx].astype(np.float32)
    mask = (atom_mask[r, c_idx] * atom_mask[r, ca_idx] * atom_mask[r, n_idx]).astype(
        np.float32
    )
    v1 = c_pos - ca_pos
    v2 = n_pos - ca_pos
    e1 = v1 / (np.sqrt(np.einsum("ij,ij->i", v1, v1))[:, None] + eps)
    e2 = v2 - np.einsum("ij,ij->i", v2, e1)[:, None] * e1
    e2 = e2 / (np.sqrt(np.einsum("ij,ij->i", e2, e2))[:, None] + eps)
    e3 = np.cross(e1, e2)
    rot = np.stack([e1, e2, e3], axis=-1)
    diff = ca_pos[None, :, :] - ca_pos[:, None, :]
    uv = np.einsum("ilk,ijl->ijk", rot, diff)
    uv = uv / (np.sqrt(np.einsum("ijk,ijk->ij", uv, uv))[..., None] + eps)
    return uv, mask[:, None] * mask[None, :]


def _pseudo_beta(
    aatype: np.ndarray, atom_positions: np.ndarray, atom_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """torch ``pseudo_beta_fn`` (no ligand)."""

    pb_idx = _PSEUDOBETA_INDEX[aatype].astype(np.int32)
    pb_pos = np.take_along_axis(
        atom_positions, pb_idx[..., None, None], axis=-2
    ).squeeze(-2)
    pb_mask = (
        np.take_along_axis(atom_mask, pb_idx[..., None], axis=-1)
        .astype(np.float32)
        .squeeze(-1)
    )
    return pb_pos, pb_mask


def _as_protenix_dict(
    aatype: np.ndarray, atom_positions: np.ndarray, atom_mask: np.ndarray
) -> dict[str, np.ndarray]:
    """torch ``Templates.as_protenix_dict``."""

    num_t, num_res = aatype.shape
    pb_masks = np.empty((num_t, num_res, num_res), dtype=np.float32)
    dgrams = np.empty((num_t, num_res, num_res, _DGRAM_NUM_BINS), dtype=np.float32)
    unit_vectors = np.empty((num_t, num_res, num_res, 3), dtype=np.float32)
    bb_masks = np.empty((num_t, num_res, num_res), dtype=np.float32)
    bool_mask = atom_mask.astype(bool)
    for i in range(num_t):
        pos = atom_positions[i] * bool_mask[i][..., None]
        pb_pos, pb_mask = _pseudo_beta(aatype[i], pos, bool_mask[i])
        pb_mask_2d = pb_mask[:, None] * pb_mask[None, :]
        dgrams[i] = _dgram_from_positions(pb_pos) * pb_mask_2d[..., None]
        pb_masks[i] = pb_mask_2d
        uv, bb_mask_2d = _unit_vector(aatype[i], pos, bool_mask[i])
        unit_vectors[i] = uv * bb_mask_2d[..., None]
        bb_masks[i] = bb_mask_2d
    return {
        "template_aatype": aatype,
        "template_atom_positions": atom_positions,
        "template_atom_mask": atom_mask.astype(bool),
        "template_pseudo_beta_mask": pb_masks,
        "template_distogram": dgrams,
        "template_unit_vector": unit_vectors,
        "template_backbone_frame_mask": bb_masks,
    }


def _pad_templates(
    aatype: np.ndarray, atom_positions: np.ndarray, atom_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce to and pad with empty templates up to ``MAX_TEMPLATES``."""

    num_t = aatype.shape[0]
    num_res = aatype.shape[1]
    keep = min(num_t, MAX_TEMPLATES)
    aatype = aatype[:keep]
    atom_positions = atom_positions[:keep]
    atom_mask = atom_mask[:keep]
    pad = MAX_TEMPLATES - keep
    if pad > 0:
        # torch ``pad_to`` zero-pads extra templates (not the gap restype).
        aatype = np.concatenate(
            [aatype, np.zeros((pad, num_res), dtype=np.int32)], axis=0
        )
        atom_positions = np.concatenate(
            [atom_positions, np.zeros((pad, num_res, _DENSE_NUM, 3), np.float32)],
            axis=0,
        )
        atom_mask = np.concatenate(
            [atom_mask, np.zeros((pad, num_res, _DENSE_NUM), np.int32)], axis=0
        )
    return aatype, atom_positions, atom_mask


def _empty_chain_dense(num_res: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """torch ``empty_template_features`` (single fully-masked template)."""

    aatype = np.full((1, num_res), _GAP_AATYPE, dtype=np.int32)
    atom_positions = np.zeros((1, num_res, _DENSE_NUM, 3), dtype=np.float32)
    atom_mask = np.zeros((1, num_res, _DENSE_NUM), dtype=np.int32)
    return aatype, atom_positions, atom_mask


def chain_template_dense(
    templates_path: str | Path | None,
    *,
    sequence: str,
    skip: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-chain dense template features padded to ``MAX_TEMPLATES``.

    Returns ``(aatype, atom_positions, atom_mask)`` of shapes
    ``(MAX_TEMPLATES, num_res[, 24[, 3]])``. ``skip`` (non-protein or
    ``len <= 4``) or no templates -> a single empty template padded out.
    """

    num_res = len(sequence)
    if skip or not templates_path:
        return _pad_templates(*_empty_chain_dense(num_res))
    path = Path(templates_path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        template_list = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in (".a3m", ".hhr"):
        from foldjax.models.protenix.data.search.templates import (
            resolve_template_search_hits,
        )

        template_list = resolve_template_search_hits(
            path, query_sequence=sequence, max_templates=MAX_TEMPLATES
        )
    else:
        raise ValueError("templatesPath must end in .json, .a3m, or .hhr")
    hits = [_single_json_template(t, num_res) for t in template_list]
    if not hits:
        return _pad_templates(*_empty_chain_dense(num_res))
    aatype = np.stack([h["template_aatype"] for h in hits], axis=0)
    all_pos = np.stack([h["template_all_atom_positions"] for h in hits], axis=0)
    all_mask = np.stack([h["template_all_atom_masks"] for h in hits], axis=0)
    atom_pos, atom_mask = _fix_to_dense(aatype, all_pos, all_mask)
    return _pad_templates(aatype, atom_pos, atom_mask)


def assemble_template_features(
    chain_dense: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray] | None:
    """Concatenate per-chain dense features over residues, derive 2D features.

    ``chain_dense`` items are ``(aatype, atom_positions, atom_mask)`` each
    ``(MAX_TEMPLATES, num_res, ...)``. Returns ``None`` only when no chains are
    present. When chains exist but carry no real templates, torch still embeds
    a single fully-masked gap template (see ``make_dummy_feature`` /
    ``empty_template_features``): ``template_aatype`` slot 0 = gap (31), the
    remaining slots zero-padded, all 2D features zero. The trunk
    ``TemplateEmbedder`` runs over these and its ``linear_no_bias_z(z_norm)``
    path contributes a dense, non-trivial delta to ``z``. Returning ``None``
    here drops that contribution and diverges from torch, so the dummy
    placeholder must be emitted rather than skipped.
    """

    if not chain_dense:
        return None
    aatype = np.concatenate([c[0] for c in chain_dense], axis=1)
    atom_positions = np.concatenate([c[1] for c in chain_dense], axis=1)
    atom_mask = np.concatenate([c[2] for c in chain_dense], axis=1)
    return _as_protenix_dict(aatype, atom_positions, atom_mask)
