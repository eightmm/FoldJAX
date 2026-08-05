"""Write predictions as structures and scores.

``predict`` returns arrays. Everything downstream of a prediction -- looking at it,
scoring it, comparing it to a deposition -- needs a structure file and a summary,
which is what the other ports in this package emit
(``foldjax.models.protenix.data.output``, ``foldjax.models.chai.output``). This
brings OpenFold3 to the same level: one mmCIF per diffusion sample, one
confidence JSON, and the raw arrays.

The atom-level metadata comes back out of the features rather than being tracked
alongside them. That is deliberate: the features are the only thing guaranteed to
have survived padding, device transfers and an ``.npz`` round trip, so decoding them
cannot go out of sync with what the model was actually given.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

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
    "environment is incomplete. Reinstall it (`uv sync --extra cuda13`), or "
    "use write_scores/write_arrays, which do not need it."
)


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


def atom_metadata(features: Mapping[str, Any]) -> AtomMetadata:
    """Decode atom names, elements, residues and chains from the features.

    Raises:
        KeyError: a feature needed to identify atoms is absent.
        ValueError: an encoding has an unexpected width, which would otherwise
            decode into plausible-looking nonsense.
    """
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
    # ``asym_id`` is 1-based and need not be dense, so chains are labelled by
    # position in sorted order rather than by the id itself.
    order = {value: rank for rank, value in enumerate(sorted(set(asym_id.tolist())))}
    chain_ids = np.asarray(
        [_chain_label(order[int(value)]) for value in asym_id[tokens]], dtype="U4"
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
    return None if array.size == 0 else float(array[0])


def confidence_summary(
    prediction: Any, features: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Summarize a prediction into JSON-serializable scores.

    Per-sample entries are ordered by the AF3 ranking score when there is enough
    information to compute it, and by mean pLDDT otherwise -- an unranked list is
    the one thing a caller cannot recover for themselves.
    """
    from foldjax.models.openfold3.models.clash import sample_ranking_score

    plddt = np.asarray(prediction.plddt, dtype=np.float64)
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

    samples = []
    for index in range(plddt.shape[0]):
        row = plddt[index]
        if atom_mask is not None and atom_mask.shape[-1] == row.shape[-1]:
            row = row[atom_mask]
        entry: dict[str, Any] = {
            "sample": index,
            "mean_plddt": float(row.mean()) if row.size else None,
            "ptm": float(ptm[index]) if index < ptm.size else None,
        }
        if iptm is not None and index < iptm.size:
            entry["iptm"] = float(iptm[index])
            # A Prediction carries no clash or disorder term, so both are zero
            # here. The key says so: the full score vetoes clashing samples with a
            # weight of 100, and calling this the ranking score would hide that the
            # veto never fired.
            entry["ranking_score_no_clash"] = float(
                np.asarray(
                    sample_ranking_score(
                        np.asarray([entry["ptm"]]),
                        np.asarray([entry["iptm"]]),
                        np.zeros(1),
                    )
                ).reshape(-1)[0]
            )
        samples.append(entry)

    key = (
        (lambda item: -(item.get("ranking_score_no_clash") or -np.inf))
        if any("ranking_score_no_clash" in s for s in samples)
        else (lambda item: -(item.get("mean_plddt") or -np.inf))
    )
    ranked = sorted(samples, key=key)

    summary: dict[str, Any] = {
        "num_samples": len(samples),
        "ranked_samples": [entry["sample"] for entry in ranked],
        "samples": samples,
    }
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


def write_arrays(
    prediction: Any,
    path: str | os.PathLike[str],
    *,
    max_bytes: int | None = DEFAULT_ARRAY_BUDGET_BYTES,
) -> tuple[Path, tuple[str, ...]]:
    """Write a prediction's arrays to ``.npz``, and say what was left out.

    The pair logits are `[n_sample, N_token, N_token, n_bin]`, so at 2076 tokens PAE
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
    structure.setup_entities()

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    structure.make_mmcif_document().write_file(str(target))
    return target


def write_prediction_outputs(
    prediction: Any,
    features: Mapping[str, Any],
    directory: str | os.PathLike[str],
    *,
    name: str = "prediction",
    max_array_bytes: int | None = DEFAULT_ARRAY_BUDGET_BYTES,
) -> dict[str, Any]:
    """Write every sample's structure plus the scores and raw arrays.

    ``max_array_bytes`` caps the ``.npz``; see :func:`write_arrays` for what it drops
    and why. ``None`` writes everything, which at 2000 tokens is over 50 GiB.

    Returns a mapping of what was written, so a caller does not have to reconstruct
    the filenames. ``omitted_arrays`` names anything the cap left out.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    metadata = atom_metadata(features)

    coordinates = np.asarray(prediction.coordinates, dtype=np.float64)
    if coordinates.ndim == 2:
        coordinates = coordinates[None, ...]
    plddt = np.asarray(prediction.plddt, dtype=np.float64)
    if plddt.ndim == 1:
        plddt = plddt[None, :]

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
                root / f"{name}_sample_{index}.cif",
                name=f"{name}_sample_{index}",
                b_factors=row[metadata.keep] if usable else None,
            )
        )

    arrays, omitted = write_arrays(
        prediction, root / f"{name}_raw.npz", max_bytes=max_array_bytes
    )
    return {
        "structures": structures,
        "scores": write_scores(
            prediction, root / f"{name}_confidences.json", features=features
        ),
        "arrays": arrays,
        "omitted_arrays": omitted,
        "num_atoms": int(metadata.name.size),
    }
