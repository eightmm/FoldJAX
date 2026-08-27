"""Write predictions as structures and scores.

``predict`` returns arrays. Everything downstream of a prediction -- looking at it,
scoring it, comparing it to a deposition -- needs a structure file and a summary,
which is what the other ports in this package emit
(``foldjax.models.protenix.data.output`` and the rest). This brings OpenFold3 to
the same level: one mmCIF per diffusion sample, one confidence JSON, and the raw
arrays.

New feature archives carry exact atom labels and bonds from upstream's ``AtomArray``
under a reserved, non-model prefix. Older archives remain writable through the
canonical feature decoder, but that fallback cannot recover ligand residue names or
connectivity that ``restype`` intentionally collapses to ``UNK``.
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from foldjax.models._output_validation import require_finite_coordinates

# Upstream's 32-token residue vocabulary, in order; ``restype`` is a one-hot over
# it. Embedded rather than imported so writing output needs none of the data stack,
# and gated against upstream in the tests.
RESIDUE_NAMES: tuple[str, ...] = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "UNK", "A", "G", "C", "U", "N", "DA", "DG", "DC", "DT", "DN", "GAP",
)

# ``ref_element`` is a one-hot over 119 bins where bin *k* means atomic number
# *k + 1*, and the last bin is upstream's "unknown element" catch-all.
ELEMENT_SYMBOLS: tuple[str, ...] = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)
UNKNOWN_ELEMENT_BIN = 118

# gemmi is a base FoldJAX dependency, not an extra as it was in the standalone
# package, so reaching this means a broken environment rather than a missing
# option. The guard stays because write_scores/write_arrays genuinely do not
# need it, and losing the structures is not a reason to lose the scores too.
_GEMMI_MISSING = (
    "writing mmCIF needs gemmi, which is a base FoldJAX dependency -- this "
    "environment is incomplete. Reinstall it (`uv sync`), or "
    "use write_scores/write_arrays, which do not need it."
)


def _safe_output_name(name: str) -> str:
    """Require one portable filename component for a prediction label."""
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
    ):
        raise ValueError(
            "OpenFold3 output name must be one non-empty filename component"
        )
    return name


def _output_path(root: Path, filename: str) -> Path:
    """Keep generated outputs below ``root`` and refuse existing symlinks."""
    target = root / filename
    try:
        contained = target.resolve(strict=False).is_relative_to(root.resolve())
    except (OSError, ValueError):
        contained = False
    if target.is_symlink() or not contained:
        raise ValueError(
            f"OpenFold3 output path escapes its output directory: {target}"
        )
    return target


class AtomMetadata(NamedTuple):
    """Per-atom identity, decoded from the features.

    Every array is ``[n_atom]`` over the *real* atoms only: padding is dropped, so
    these line up with a prediction sliced to the same mask.
    """

    name: np.ndarray
    element: np.ndarray
    residue_name: np.ndarray
    residue_id: np.ndarray
    chain_id: np.ndarray
    keep: np.ndarray
    entity_id: np.ndarray | None = None
    molecule_type_id: np.ndarray | None = None
    bonds: np.ndarray | None = None
    bond_type: np.ndarray | None = None


def _first(array: np.ndarray, ndim: int) -> np.ndarray:
    """Drop leading batch axes until ``ndim`` remain."""
    result = np.asarray(array)
    while result.ndim > ndim:
        result = result[0]
    return result


def _chain_label(position: int) -> str:
    """A, ..., Z, AA, AB, ... from a 0-based position."""
    letters = ""
    position = int(position)
    while True:
        letters = chr(ord("A") + position % 26) + letters
        position = position // 26 - 1
        if position < 0:
            return letters


def atom_metadata(
    features: Mapping[str, Any], output_metadata: Any | None = None
) -> AtomMetadata:
    """Decode atom names, elements, residues and chains from the features.

    Raises:
        KeyError: a feature needed to identify atoms is absent.
        ValueError: an encoding has an unexpected width, which would otherwise
            decode into plausible-looking nonsense.
    """
    if output_metadata is not None:
        from foldjax.models.openfold3.data.featurize import validate_output_metadata

        exact = validate_output_metadata(output_metadata, features)
        keep = _first(features["atom_mask"], 1) > 0
        return AtomMetadata(
            name=exact.atom_name,
            element=exact.element,
            residue_name=exact.residue_name,
            residue_id=exact.residue_id,
            chain_id=exact.chain_id,
            keep=keep,
            entity_id=exact.entity_id,
            molecule_type_id=exact.molecule_type_id,
            bonds=exact.bonds,
            bond_type=exact.bond_type,
        )

    required = (
        "ref_atom_name_chars",
        "ref_element",
        "atom_mask",
        "atom_to_token_index",
        "restype",
        "residue_index",
        "asym_id",
    )
    missing = [name for name in required if name not in features]
    if missing:
        raise KeyError(f"cannot identify atoms without {missing}")

    chars = _first(features["ref_atom_name_chars"], 3)
    if chars.shape[1:] != (4, 64):
        raise ValueError(
            f"ref_atom_name_chars must be (n_atom, 4, 64), got {chars.shape}"
        )
    element = _first(features["ref_element"], 2)
    if element.shape[-1] != len(ELEMENT_SYMBOLS) + 1:
        raise ValueError(
            f"ref_element must have {len(ELEMENT_SYMBOLS) + 1} bins, "
            f"got {element.shape[-1]}"
        )

    atom_mask = _first(features["atom_mask"], 1)
    keep = np.asarray(atom_mask) > 0
    atom_to_token = _first(features["atom_to_token_index"], 1).astype(np.int64)
    restype = _first(features["restype"], 2)
    residue_index = _first(features["residue_index"], 1).astype(np.int64)
    asym_id = _first(features["asym_id"], 1).astype(np.int64)

    # Names are four characters offset by 32, right-padded; elements are one-hot
    # over atomic number minus one.
    name_codes = np.argmax(chars, axis=-1)[keep]
    names = np.asarray(
        ["".join(chr(int(code) + 32) for code in row).strip() for row in name_codes],
        dtype="U4",
    )
    element_bins = np.argmax(element, axis=-1)[keep]
    elements = np.asarray(
        [
            "X" if int(bin_) >= UNKNOWN_ELEMENT_BIN else ELEMENT_SYMBOLS[int(bin_)]
            for bin_ in element_bins
        ],
        dtype="U2",
    )

    tokens = atom_to_token[keep]
    residue_bins = np.argmax(restype, axis=-1)[tokens]
    residue_names = np.asarray(
        [
            RESIDUE_NAMES[int(b)] if int(b) < len(RESIDUE_NAMES) else "UNK"
            for b in residue_bins
        ],
        dtype="U3",
    )
    # ``residue_index`` is already 1-based in the features -- upstream's featurizer
    # carries the deposition's numbering -- so it is used as it is. An earlier
    # version added one, having been written against a hand-built batch that
    # numbered from zero; every residue in every output file was off by one.
    residue_ids = residue_index[tokens]
    # Chain labels belong to emitted atoms. Padded token IDs are arbitrary
    # sentinel values and must not shift a real chain from A to B (or create a
    # chain that has no atom in the structure at all).
    emitted_asym_id = asym_id[tokens]
    order = {
        value: rank
        for rank, value in enumerate(sorted(set(emitted_asym_id.tolist())))
    }
    chain_ids = np.asarray(
        [_chain_label(order[int(value)]) for value in emitted_asym_id], dtype="U4"
    )

    return AtomMetadata(
        name=names,
        element=elements,
        residue_name=residue_names,
        residue_id=residue_ids,
        chain_id=chain_ids,
        keep=keep,
    )


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return None
    result = float(array[0])
    return result if np.isfinite(result) else None


def _plddt_percent(values: Any) -> np.ndarray:
    """Convert the model's fractional pLDDT to the public 0--100 scale.

    ``Prediction.plddt`` is an internal array and currently follows upstream's
    [0, 1] head output. Writers are the public boundary. Keeping the conversion
    here makes the raw archive lossless and, by accepting already-percent values,
    prevents a second boundary call from multiplying them twice.
    """
    result = np.asarray(values, dtype=np.float64)
    finite = result[np.isfinite(result)]
    if finite.size and np.all((finite >= 0.0) & (finite <= 1.0 + 1e-6)):
        result = result * 100.0
    return result


def _ranking_atom_features(
    features: Mapping[str, Any], *, n_atom: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return atom mask/chain/polymer flags and whether the input has protein."""
    required = (
        "atom_mask",
        "atom_to_token_index",
        "token_mask",
        "asym_id",
        "is_protein",
        "is_rna",
        "is_dna",
    )
    missing = [name for name in required if name not in features]
    if missing:
        raise KeyError(f"cannot compute OpenFold3 sample ranking without {missing}")

    atom_mask = _first(features["atom_mask"], 1).astype(bool)
    atom_to_token = _first(features["atom_to_token_index"], 1).astype(np.int64)
    token_mask = _first(features["token_mask"], 1).astype(bool)
    asym_id = _first(features["asym_id"], 1).astype(np.int64)
    protein = _first(features["is_protein"], 1).astype(bool)
    rna = _first(features["is_rna"], 1).astype(bool)
    dna = _first(features["is_dna"], 1).astype(bool)
    if atom_mask.size != n_atom or atom_to_token.size != n_atom:
        raise ValueError(
            "OpenFold3 ranking features and predicted coordinates disagree: "
            f"{atom_mask.size} feature atoms versus {n_atom} predicted atoms"
        )
    n_token = token_mask.size
    if not (
        asym_id.size == protein.size == rna.size == dna.size == n_token
    ):
        raise ValueError("OpenFold3 ranking token features have inconsistent lengths")
    if np.any((atom_to_token < 0) | (atom_to_token >= n_token)):
        raise ValueError("atom_to_token_index is out of bounds for sample ranking")

    polymer_token = (protein | rna | dna) & token_mask
    return (
        atom_mask,
        asym_id[atom_to_token],
        polymer_token[atom_to_token],
        bool(np.any(protein & token_mask)),
    )


def _has_clash_blocked(
    coordinates: np.ndarray,
    atom_mask: np.ndarray,
    atom_asym_id: np.ndarray,
    is_polymer: np.ndarray,
    *,
    threshold: float = 1.1,
    violation_abs: int = 100,
    violation_frac: float = 0.5,
    block_size: int = 512,
) -> np.ndarray:
    """Upstream inter-polymer-chain clash test without an atom-squared array.

    Only one ``block_size`` by ``block_size`` distance tile is live at a time.
    This matters at output-writing time: the model has already consumed most of
    the device budget, and forming the old full atom-pair tensor could turn a
    successful large prediction into an output-stage OOM.
    """
    positions = np.asarray(coordinates, dtype=np.float64)
    if positions.ndim == 2:
        positions = positions[None, ...]
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(
            "OpenFold3 coordinates must have shape (samples, atoms, 3) for ranking"
        )

    # Match upstream exactly: a chain is eligible only when every atom assigned
    # to that chain is marked polymer; atom_mask is applied to its clash count.
    polymer_chains = [
        chain
        for chain in np.unique(atom_asym_id)
        if np.all((atom_asym_id != chain) | is_polymer)
    ]
    chain_indices = [
        np.flatnonzero((atom_asym_id == chain) & atom_mask)
        for chain in polymer_chains
    ]
    result = np.zeros(positions.shape[0], dtype=np.float64)
    threshold_squared = float(threshold) ** 2
    for sample, sample_positions in enumerate(positions):
        if not np.isfinite(sample_positions[atom_mask]).all():
            result[sample] = np.nan
            continue
        clashing = False
        for first_index, first_atoms in enumerate(chain_indices):
            if first_atoms.size == 0:
                continue
            first = sample_positions[first_atoms]
            for second_atoms in chain_indices[first_index + 1 :]:
                if second_atoms.size == 0:
                    continue
                second = sample_positions[second_atoms]
                limit = min(
                    violation_abs + 1,
                    int(
                        np.floor(
                            violation_frac * min(first.shape[0], second.shape[0])
                        )
                    )
                    + 1,
                )
                count = 0
                for first_start in range(0, first.shape[0], block_size):
                    first_block = first[first_start : first_start + block_size]
                    for second_start in range(0, second.shape[0], block_size):
                        second_block = second[
                            second_start : second_start + block_size
                        ]
                        delta = first_block[:, None, :] - second_block[None, :, :]
                        squared = np.einsum("...i,...i->...", delta, delta)
                        count += int(np.count_nonzero(squared < threshold_squared))
                        if count >= limit:
                            clashing = True
                            break
                    if clashing:
                        break
                if clashing:
                    break
            if clashing:
                break
        result[sample] = float(clashing)
    return result


def confidence_summary(
    prediction: Any, features: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Summarize a prediction into JSON-serializable scores.

    Ranking follows upstream's
    ``0.8*ipTM + 0.2*pTM + 0.5*disorder - 100*has_clash`` formula. FoldJAX can
    compute clashes from prediction coordinates without an optional dependency,
    but protein disorder requires the upstream RASA/SASA pipeline. Protein runs
    therefore report a clearly partial ``sample_ranking_score_no_disorder`` and
    make no ranked/best claim. For inputs without protein, disorder is exactly
    zero upstream too, so ``sample_ranking_score`` is exact and may rank samples.

    pLDDT in this public JSON is 0--100. The raw prediction archive intentionally
    retains the model's fractional values.
    """
    plddt = _plddt_percent(prediction.plddt)
    if plddt.ndim == 1:
        plddt = plddt[None, :]
    atom_mask = None
    if features is not None and "atom_mask" in features:
        atom_mask = _first(features["atom_mask"], 1) > 0

    ptm = np.asarray(prediction.ptm, dtype=np.float64).reshape(-1)
    iptm = (
        None
        if prediction.iptm is None
        else np.asarray(prediction.iptm, dtype=np.float64).reshape(-1)
    )

    coordinates = np.asarray(prediction.coordinates, dtype=np.float64)
    if coordinates.ndim == 2:
        coordinates = coordinates[None, ...]
    has_clash = None
    has_protein = None
    if features is not None:
        ranking_features = _ranking_atom_features(
            features, n_atom=coordinates.shape[-2]
        )
        atom_mask_for_clash, atom_asym_id, is_polymer, has_protein = ranking_features
        has_clash = _has_clash_blocked(
            coordinates, atom_mask_for_clash, atom_asym_id, is_polymer
        )

    samples = []
    for index in range(plddt.shape[0]):
        row = plddt[index]
        if atom_mask is not None and atom_mask.shape[-1] == row.shape[-1]:
            row = row[atom_mask]
        entry: dict[str, Any] = {
            "sample": index,
            "mean_plddt": _as_float(row.mean()) if row.size else None,
            "ptm": _as_float(ptm[index]) if index < ptm.size else None,
        }
        if iptm is not None and index < iptm.size:
            entry["iptm"] = _as_float(iptm[index])
        if has_clash is not None and index < has_clash.size:
            entry["has_clash"] = _as_float(has_clash[index])
        components = (entry.get("ptm"), entry.get("iptm"), entry.get("has_clash"))
        if has_protein is not None and all(value is not None for value in components):
            ptm_value, iptm_value, clash_value = components
            if not has_protein:
                entry["disorder"] = 0.0
            score = (
                0.8 * iptm_value
                + 0.2 * ptm_value
                - 100.0 * clash_value
            )
            entry[
                "sample_ranking_score_no_disorder"
                if has_protein
                else "sample_ranking_score"
            ] = float(score)
        samples.append(entry)

    summary: dict[str, Any] = {
        "num_samples": len(samples),
        "samples": samples,
    }
    exact = [entry.get("sample_ranking_score") for entry in samples]
    if samples and all(value is not None and np.isfinite(value) for value in exact):
        # Python's sort is stable, so equal scores retain diffusion sample order.
        ranked = sorted(samples, key=lambda item: -item["sample_ranking_score"])
        summary["ranked_samples"] = [entry["sample"] for entry in ranked]
    if prediction.chain_pair_iptm is not None:
        summary["chain_pair_iptm"] = np.asarray(
            prediction.chain_pair_iptm, dtype=np.float64
        ).tolist()
    return summary


def write_scores(
    prediction: Any,
    path: str | os.PathLike[str],
    *,
    features: Mapping[str, Any] | None = None,
) -> Path:
    """Write :func:`confidence_summary` as JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(confidence_summary(prediction, features), indent=2))
    return target


#: Size above which :func:`write_arrays` leaves the per-bin pair logits out. They
#: are quadratic in token count *and* carry a bin axis, so they dominate the file:
#: PAE and PDE together are 51 GiB at 2076 tokens against 1 MB of coordinates.
DEFAULT_ARRAY_BUDGET_BYTES = 4 * 2**30

#: Arrays that are per-bin distributions over token pairs. Everything else a
#: prediction carries is small enough that its size never decides anything.
_PAIR_LOGIT_NAMES = ("pae_logits", "pde_logits", "distogram_logits")


def crop_prediction(prediction: Any, features: Mapping[str, Any]) -> Any:
    """Crop masked token/atom suffixes from the portable prediction arrays.

    The compiled program deliberately returns its bucket shape. Structures and
    scores already apply the masks, but a raw ``.npz`` should describe the
    molecule rather than expose serving-only filler rows.
    """

    token_mask = _first(features["token_mask"], 1).astype(bool)
    atom_mask = _first(features["atom_mask"], 1).astype(bool)
    n_token = int(token_mask.sum())
    n_atom = int(atom_mask.sum())
    if not np.array_equal(token_mask, np.arange(token_mask.size) < n_token):
        raise ValueError("OpenFold3 token padding must be a suffix")
    if not np.array_equal(atom_mask, np.arange(atom_mask.size) < n_atom):
        raise ValueError("OpenFold3 atom padding must be a suffix")

    values = prediction._asdict()
    for name in _PAIR_LOGIT_NAMES:
        value = values.get(name)
        if value is not None:
            values[name] = value[..., :n_token, :n_token, :]
    for name in ("coordinates", "plddt", "experimentally_resolved_logits"):
        value = values.get(name)
        if value is None:
            continue
        values[name] = (
            value[..., :n_atom, :]
            if name == "coordinates" or value.ndim >= 3
            else value[..., :n_atom]
        )
    return type(prediction)(**values)


def plan_returned_pair_logits(
    *,
    n_token: int,
    num_samples: int,
    pae_bins: int = 64,
    pde_bins: int = 64,
    distogram_bins: int = 64,
    max_bytes: int | None = DEFAULT_ARRAY_BUDGET_BYTES,
) -> tuple[str, ...]:
    """Which pair logits `write_arrays` would keep, decided before the run.

    Same rule as `write_arrays` -- drop largest-first until the total fits --
    with a 64 MiB allowance standing in for the small arrays it also counts,
    so this plan is strictly more conservative than the writer. The writer
    still applies its own budget to whatever arrives, so a prediction built
    from this plan never writes an array the old behaviour would have dropped;
    the only divergence is inside that 64 MiB band, where an array narrowly
    inside the budget is not returned at all.

    The point is where the decision lands: an excluded array is never an entry
    output of the compiled program, instead of being hauled out -- 21.6 GiB at
    3,012 tokens and five samples -- and then discarded unwritten.
    """
    if max_bytes is None:
        return _PAIR_LOGIT_NAMES
    element = 4  # float32
    pair = n_token * n_token
    sizes = {
        "pae_logits": num_samples * pair * pae_bins * element,
        "pde_logits": num_samples * pair * pde_bins * element,
        "distogram_logits": pair * distogram_bins * element,
    }
    budget = max_bytes - 64 * 2**20
    total = sum(sizes.values())
    kept = dict(sizes)
    for name in sorted(sizes, key=sizes.__getitem__, reverse=True):
        if total <= budget:
            break
        total -= kept.pop(name)
    return tuple(name for name in _PAIR_LOGIT_NAMES if name in kept)


def write_arrays(
    prediction: Any,
    path: str | os.PathLike[str],
    *,
    max_bytes: int | None = DEFAULT_ARRAY_BUDGET_BYTES,
) -> tuple[Path, tuple[str, ...]]:
    """Write a prediction's arrays to ``.npz``, and say what was left out.

    The pair logits are `[num_samples, N_token, N_token, n_bin]`, so at 2076 tokens PAE
    and PDE are 25.7 GiB each and the distogram another 5.1. Writing them
    unconditionally turns a prediction that takes eight minutes to compute into one
    that spends longer than that saving a file most callers will not read. Above
    ``max_bytes`` they are omitted; ``None`` writes everything.

    The scalar and per-atom outputs -- coordinates, pLDDT, pTM, ipTM -- are always
    written, and so is anything the confidence JSON is derived from, so omitting the
    logits never costs a reported number.

    Returns:
        ``(path, omitted_names)``.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        name: np.asarray(value)
        for name, value in prediction._asdict().items()
        if value is not None
    }

    omitted: list[str] = []
    if max_bytes is not None:
        total = sum(array.nbytes for array in arrays.values())
        # Dropped largest-first, so the smallest set of omissions gets under the
        # budget rather than the first ones encountered.
        for name in sorted(
            (n for n in _PAIR_LOGIT_NAMES if n in arrays),
            key=lambda n: arrays[n].nbytes,
            reverse=True,
        ):
            if total <= max_bytes:
                break
            total -= arrays.pop(name).nbytes
            omitted.append(name)

    np.savez(target, **arrays)
    return target, tuple(omitted)


def write_structure(
    coordinates: np.ndarray,
    metadata: AtomMetadata,
    path: str | os.PathLike[str],
    *,
    name: str = "prediction",
    b_factors: np.ndarray | None = None,
) -> Path:
    """Write one sample's coordinates as mmCIF.

    Args:
        coordinates: ``[n_atom, 3]`` for the real atoms, in metadata order.
        metadata: from :func:`atom_metadata`.
        path: output file.
        name: structure name recorded in the file.
        b_factors: per-atom values for the B-factor column, conventionally pLDDT.

    Raises:
        ImportError: gemmi is not installed.
        ValueError: the coordinate count does not match the metadata.
    """
    try:
        import gemmi
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(_GEMMI_MISSING) from error

    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.shape != (metadata.name.size, 3):
        raise ValueError(
            f"expected ({metadata.name.size}, 3) coordinates, got {coordinates.shape}"
        )
    if b_factors is not None:
        b_factors = np.asarray(b_factors, dtype=np.float64).reshape(-1)
        if b_factors.size != metadata.name.size:
            raise ValueError(
                f"expected {metadata.name.size} b-factors, got {b_factors.size}"
            )

    structure = gemmi.Structure()
    structure.name = name
    model = gemmi.Model("1")
    for chain_id in dict.fromkeys(metadata.chain_id.tolist()):
        chain = gemmi.Chain(str(chain_id))
        indices = np.flatnonzero(metadata.chain_id == chain_id)
        residue = None
        current: tuple[int, str] | None = None
        for index in indices:
            key = (int(metadata.residue_id[index]), str(metadata.residue_name[index]))
            if key != current:
                if residue is not None:
                    chain.add_residue(residue)
                residue = gemmi.Residue()
                residue.name = key[1]
                residue.seqid = gemmi.SeqId(key[0], " ")
                residue.subchain = str(chain_id)
                if metadata.molecule_type_id is not None:
                    residue_mask = (
                        (metadata.residue_id[indices] == key[0])
                        & (metadata.residue_name[indices] == key[1])
                    )
                    residue_indices = indices[residue_mask]
                    residue_types = np.unique(
                        metadata.molecule_type_id[residue_indices]
                    )
                    entity_ids = np.unique(metadata.entity_id[residue_indices])
                    if len(residue_types) != 1 or len(entity_ids) != 1:
                        raise ValueError(
                            f"chain {chain_id!r} has inconsistent output provenance"
                        )
                    residue.entity_id = str(entity_ids[0])
                    if int(residue_types[0]) == 3:
                        residue.het_flag = "H"
                    else:
                        residue.het_flag = "A"
                        residue.label_seq = key[0]
                current = key
            atom = gemmi.Atom()
            atom.name = str(metadata.name[index])
            atom.element = gemmi.Element(str(metadata.element[index]))
            atom.pos = gemmi.Position(*coordinates[index])
            atom.b_iso = 0.0 if b_factors is None else float(b_factors[index])
            residue.add_atom(atom)
        if residue is not None:
            chain.add_residue(residue)
        model.add_chain(chain)
    structure.add_model(model)
    if metadata.entity_id is None:
        structure.setup_entities()
    else:
        _add_entities(structure, metadata)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    document = structure.make_mmcif_document()
    _set_bond_categories(document, metadata)
    document.write_file(str(target))
    return target


_BOND_ORDERS = {
    "ANY": (False, False),
    "SINGLE": ("SING", "N"),
    "DOUBLE": ("DOUB", "N"),
    "TRIPLE": ("TRIP", "N"),
    "QUADRUPLE": ("QUAD", "N"),
    "AROMATIC_SINGLE": ("SING", "Y"),
    "AROMATIC_DOUBLE": ("DOUB", "Y"),
    "AROMATIC_TRIPLE": ("TRIP", "Y"),
    "COORDINATION": ("SING", "N"),
    "AROMATIC": ("AROM", "Y"),
}
_CANONICAL_LINK_RESIDUES = frozenset(
    (*RESIDUE_NAMES[:20], "A", "G", "C", "U", "DA", "DG", "DC", "DT")
)


def _is_canonical_polymer_link(
    metadata: AtomMetadata, first: int, second: int
) -> bool:
    """Match Biotite's suppression of implicit backbone links in struct_conn."""
    if metadata.chain_id[first] != metadata.chain_id[second]:
        return False
    if metadata.molecule_type_id is not None and (
        int(metadata.molecule_type_id[first]) == 3
        or int(metadata.molecule_type_id[second]) == 3
    ):
        return False
    if int(metadata.residue_id[first]) > int(metadata.residue_id[second]):
        first, second = second, first
    return (
        str(metadata.residue_name[first]) in _CANONICAL_LINK_RESIDUES
        and str(metadata.residue_name[second]) in _CANONICAL_LINK_RESIDUES
        and str(metadata.name[first]) in {"C", "O3'"}
        and str(metadata.name[second]) in {"N", "P"}
        and int(metadata.residue_id[second]) - int(metadata.residue_id[first]) == 1
    )


def _add_entities(structure: Any, metadata: AtomMetadata) -> None:
    """Populate entity/subchain metadata from exact upstream annotations."""
    import gemmi

    assert metadata.entity_id is not None
    assert metadata.molecule_type_id is not None
    entity_ids = metadata.entity_id.astype(str)
    molecule_types = metadata.molecule_type_id.astype(np.int64)
    chain_ids = metadata.chain_id.astype(str)
    polymer_types = {
        0: gemmi.PolymerType.PeptideL,
        1: gemmi.PolymerType.Rna,
        2: gemmi.PolymerType.Dna,
    }
    for entity_id in dict.fromkeys(entity_ids.tolist()):
        entity_mask = entity_ids == entity_id
        unique_types = np.unique(molecule_types[entity_mask])
        if unique_types.size != 1:
            raise ValueError(f"entity {entity_id!r} has inconsistent molecule types")
        molecule_type = int(unique_types[0])
        entity = gemmi.Entity(str(entity_id))
        entity.subchains = list(dict.fromkeys(chain_ids[entity_mask].tolist()))
        if molecule_type == 3:
            entity.entity_type = gemmi.EntityType.NonPolymer
        else:
            entity.entity_type = gemmi.EntityType.Polymer
            entity.polymer_type = polymer_types[molecule_type]
            sequences = []
            for chain_id in entity.subchains:
                indices = np.flatnonzero(entity_mask & (chain_ids == chain_id))
                residues = list(
                    dict.fromkeys(
                        (
                            int(metadata.residue_id[index]),
                            str(metadata.residue_name[index]),
                        )
                        for index in indices
                    )
                )
                sequences.append(
                    [residue_name for _residue_id, residue_name in residues]
                )
            if any(sequence != sequences[0] for sequence in sequences[1:]):
                raise ValueError(
                    f"copies of entity {entity_id!r} have inconsistent sequences"
                )
            entity.full_sequence = sequences[0]
        structure.entities.append(entity)


def _set_bond_categories(document: Any, metadata: AtomMetadata) -> None:
    """Write upstream's intra- and inter-residue covalent graph to mmCIF."""
    if (
        metadata.bonds is None
        or metadata.bond_type is None
        or not len(metadata.bonds)
    ):
        return

    intra: dict[str, list[str | bool]] = {
        "pdbx_ordinal": [],
        "comp_id": [],
        "atom_id_1": [],
        "atom_id_2": [],
        "value_order": [],
        "pdbx_aromatic_flag": [],
        "pdbx_stereo_config": [],
    }
    inter: dict[str, list[str | bool]] = {
        "id": [],
        "conn_type_id": [],
        "pdbx_value_order": [],
        "ptnr1_label_asym_id": [],
        "ptnr2_label_asym_id": [],
        "ptnr1_label_comp_id": [],
        "ptnr2_label_comp_id": [],
        "ptnr1_label_seq_id": [],
        "ptnr2_label_seq_id": [],
        "ptnr1_label_atom_id": [],
        "ptnr2_label_atom_id": [],
        "pdbx_ptnr1_PDB_ins_code": [],
        "pdbx_ptnr2_PDB_ins_code": [],
    }
    seen_intra: set[tuple[str, str, str]] = set()
    for atom1, atom2, raw_type in zip(
        metadata.bonds[:, 0],
        metadata.bonds[:, 1],
        metadata.bond_type,
        strict=True,
    ):
        first, second = int(atom1), int(atom2)
        bond_type = str(raw_type)
        first_residue = (
            str(metadata.chain_id[first]),
            int(metadata.residue_id[first]),
            str(metadata.residue_name[first]),
        )
        second_residue = (
            str(metadata.chain_id[second]),
            int(metadata.residue_id[second]),
            str(metadata.residue_name[second]),
        )
        order, aromatic = _BOND_ORDERS[bond_type]
        if first_residue == second_residue:
            key = (
                first_residue[2],
                str(metadata.name[first]),
                str(metadata.name[second]),
            )
            if key in seen_intra:
                continue
            seen_intra.add(key)
            intra["pdbx_ordinal"].append(str(len(intra["pdbx_ordinal"]) + 1))
            intra["comp_id"].append(first_residue[2])
            intra["atom_id_1"].append(str(metadata.name[first]))
            intra["atom_id_2"].append(str(metadata.name[second]))
            intra["value_order"].append(order)
            intra["pdbx_aromatic_flag"].append(aromatic)
            intra["pdbx_stereo_config"].append(False)
            continue

        if _is_canonical_polymer_link(metadata, first, second):
            continue

        inter["id"].append(str(len(inter["id"]) + 1))
        inter["conn_type_id"].append(
            "metalc" if bond_type == "COORDINATION" else "covale"
        )
        inter["pdbx_value_order"].append(
            False
            if bond_type in {"ANY", "AROMATIC", "COORDINATION"}
            else str(order).lower()
        )
        for side, index in ((1, first), (2, second)):
            inter[f"ptnr{side}_label_asym_id"].append(str(metadata.chain_id[index]))
            inter[f"ptnr{side}_label_comp_id"].append(
                str(metadata.residue_name[index])
            )
            is_ligand = (
                metadata.molecule_type_id is not None
                and int(metadata.molecule_type_id[index]) == 3
            )
            inter[f"ptnr{side}_label_seq_id"].append(
                False if is_ligand else str(int(metadata.residue_id[index]))
            )
            inter[f"ptnr{side}_label_atom_id"].append(str(metadata.name[index]))
            inter[f"pdbx_ptnr{side}_PDB_ins_code"].append(False)

    block = document.sole_block()
    if intra["pdbx_ordinal"]:
        block.set_mmcif_category("_chem_comp_bond.", intra)
    if inter["id"]:
        block.set_mmcif_category("_struct_conn.", inter)


def write_prediction_outputs(
    prediction: Any,
    features: Mapping[str, Any],
    directory: str | os.PathLike[str],
    *,
    name: str = "prediction",
    max_array_bytes: int | None = DEFAULT_ARRAY_BUDGET_BYTES,
    output_metadata: Any | None = None,
) -> dict[str, Any]:
    """Write every sample's structure plus the scores and raw arrays.

    ``max_array_bytes`` caps the ``.npz``; see :func:`write_arrays` for what it drops
    and why. ``None`` writes everything, which at 2000 tokens is over 50 GiB.

    Returns a mapping of what was written, so a caller does not have to reconstruct
    the filenames. ``omitted_arrays`` names anything the cap left out.
    """
    name = _safe_output_name(name)
    if output_metadata is None:
        warnings.warn(
            "OpenFold3 features have no exact output metadata; using the canonical "
            "feature fallback, which may label ligands or modified residues as UNK "
            "and cannot emit covalent bonds. Re-create the archive with "
            "openfold3-jax-featurize for chemically faithful output.",
            RuntimeWarning,
            stacklevel=2,
        )
    metadata = atom_metadata(features, output_metadata)

    coordinates = np.asarray(prediction.coordinates, dtype=np.float64)
    if coordinates.ndim == 2:
        coordinates = coordinates[None, ...]
    require_finite_coordinates(
        coordinates,
        model="OpenFold3",
        atom_mask=features["atom_mask"],
    )
    # mmCIF B factors and public confidence JSON conventionally carry pLDDT on
    # a 0--100 scale. The Prediction/raw archive remains fractional.
    plddt = _plddt_percent(prediction.plddt)
    if plddt.ndim == 1:
        plddt = plddt[None, :]

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    structures = []
    for index in range(coordinates.shape[0]):
        # Predictions cover padded atoms too; only the real ones are written.
        sample_coordinates = coordinates[index][metadata.keep]
        row = plddt[min(index, plddt.shape[0] - 1)]
        usable = row.size == metadata.keep.size
        structures.append(
            write_structure(
                sample_coordinates,
                metadata,
                _output_path(root, f"{name}_sample_{index}.cif"),
                name=f"{name}_sample_{index}",
                b_factors=row[metadata.keep] if usable else None,
            )
        )

    arrays, omitted = write_arrays(
        crop_prediction(prediction, features),
        _output_path(root, f"{name}_raw.npz"),
        max_bytes=max_array_bytes,
    )
    return {
        "structures": structures,
        "scores": write_scores(
            prediction,
            _output_path(root, f"{name}_confidences.json"),
            features=features,
        ),
        "arrays": arrays,
        "omitted_arrays": omitted,
        "num_atoms": int(metadata.name.size),
    }
