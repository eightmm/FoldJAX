"""Protein disorder for OpenFold3's sample ranking score, on the host.

Upstream ranks samples by ``0.8*ipTM + 0.2*pTM + 0.5*disorder - 100*has_clash``
(``core/metrics/sample_ranking.py``). ``disorder`` is the fraction of protein
residues whose smoothed relative solvent accessibility exceeds 0.581
(``core/metrics/rasa.py``, ``compute_disorder``/``process_disorder``): every
residue counts as unresolved at inference, each protein chain's atom SASA is
taken on its own with biotite's Shrake-Rupley ``sasa`` and ProtOr radii,
summed per residue, divided by the Sander maximum (113 for a residue the scale
does not list), clipped to [0, 1], smoothed by a 25-residue moving average with
reflect padding, and the residues of all chains are pooled before the mean.

This is the same arithmetic in NumPy, with the same biotite calls, so it needs
biotite -- an optional dependency (the ``openfold3-preprocess`` extra). Without
it :func:`unavailable_reason` says why, and the writer keeps the partial
``sample_ranking_score_no_disorder`` instead of inventing a disorder term.

The scale, window and threshold are embedded rather than imported so that
writing output needs none of the vendored data package; a test pins them
against upstream's ``RESIDUE_SASA_SCALES["Sander"]`` and upstream's own CPU
outputs (``tests/models/openfold3/fixtures/full_confidence_upstream.npz``).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: Sander & Rost 1994 maximum accessible surface areas, upstream's default
#: ``residue_sasa_scale`` (``RESIDUE_SASA_SCALES["Sander"]``).
SANDER_MAX_ACC: dict[str, float] = {
    "ALA": 106.0,
    "ARG": 248.0,
    "ASN": 157.0,
    "ASP": 163.0,
    "CYS": 135.0,
    "GLN": 198.0,
    "GLU": 194.0,
    "GLY": 84.0,
    "HIS": 184.0,
    "ILE": 169.0,
    "LEU": 164.0,
    "LYS": 205.0,
    "MET": 188.0,
    "PHE": 197.0,
    "PRO": 136.0,
    "SER": 130.0,
    "THR": 142.0,
    "TRP": 227.0,
    "TYR": 222.0,
    "VAL": 142.0,
}
#: ``compute_disorder``'s ``default_max_acc``, used for residues outside the scale.
DEFAULT_MAX_ACC = 113.0
#: ``compute_disorder``'s smoothing window and ``vdw_radii``.
WINDOW = 25
VDW_RADII = "ProtOr"
#: ``confidence.sample_ranking.full_complex.disorder_threshold``.
DISORDER_THRESHOLD = 0.581


def unavailable_reason() -> str | None:
    """Why disorder cannot be computed here, or None when it can."""
    try:
        import biotite.structure  # noqa: F401
    except ImportError:
        return (
            "OpenFold3's disorder term needs biotite's SASA (upstream "
            "core/metrics/rasa.py), which is not installed; install the "
            "openfold3-preprocess extra"
        )
    return None


def smooth_rasa(res_rasa: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Upstream ``_smooth_rasa``: reflect-padded moving average."""
    half_w = (window - 1) // 2
    padded = np.pad(res_rasa, (half_w, half_w), mode="reflect")
    return np.convolve(padded, np.ones(window), mode="valid") / window


def _chain_rasa(chain) -> np.ndarray:
    import biotite.structure as struc

    atom_sasa = struc.sasa(chain, vdw_radii=VDW_RADII)
    res_sasa = struc.apply_residue_wise(chain, atom_sasa, np.sum)
    _, res_names = struc.get_residues(chain)
    max_acc = np.array(
        [SANDER_MAX_ACC.get(str(name), DEFAULT_MAX_ACC) for name in res_names]
    )
    return smooth_rasa(np.clip(res_sasa / max_acc, 0, 1))


def protein_residue_rasa(
    coordinates: np.ndarray,
    *,
    atom_name: Sequence[str],
    element: Sequence[str],
    residue_name: Sequence[str],
    residue_id: Sequence[int],
    chain_id: Sequence[str],
    is_protein: np.ndarray,
) -> list[np.ndarray | None]:
    """Smoothed RASA of every protein residue, chains pooled, per sample.

    ``coordinates`` is ``[num_samples, n_atom, 3]``. The per-atom labels describe
    the structure's atoms in the order upstream's ``AtomArray`` holds them
    (chains contiguous, as ``struc.chain_iter`` expects). A sample with no
    scorable protein chain gives None. A chain whose SASA fails is skipped
    with a warning, as upstream does.
    """
    import biotite.structure as struc

    positions = np.asarray(coordinates, dtype=np.float32)
    if positions.ndim == 2:
        positions = positions[None]
    protein = np.asarray(is_protein, dtype=bool).reshape(-1)
    if positions.shape[1] != protein.size:
        raise ValueError(
            f"{positions.shape[1]} coordinates against {protein.size} atom labels"
        )
    template = struc.AtomArray(int(protein.sum()))
    template.chain_id = np.asarray(chain_id, dtype=str)[protein]
    template.res_id = np.asarray(residue_id, dtype=np.int64)[protein]
    template.res_name = np.asarray(residue_name, dtype=str)[protein]
    template.atom_name = np.asarray(atom_name, dtype=str)[protein]
    template.element = np.char.upper(np.asarray(element, dtype=str)[protein])
    template.hetero = np.zeros(template.array_length(), dtype=bool)

    pooled_samples: list[np.ndarray | None] = []
    for sample_positions in positions:
        if template.array_length() == 0:
            pooled_samples.append(None)
            continue
        template.coord = sample_positions[protein]
        pooled: list[np.ndarray] = []
        for chain in struc.chain_iter(template):
            try:
                pooled.append(_chain_rasa(chain))
            except Exception as error:  # noqa: BLE001 - upstream logs and skips
                logger.warning("RASA computation failed: %s", error)
        pooled_samples.append(np.concatenate(pooled) if pooled else None)
    return pooled_samples


def protein_disorder(
    coordinates: np.ndarray,
    *,
    threshold: float = DISORDER_THRESHOLD,
    **labels,
) -> np.ndarray:
    """Upstream ``compute_disorder``: ``[num_samples]`` fractions of disordered
    protein residues; NaN where no protein residue could be scored, as upstream
    returns. ``labels`` are :func:`protein_residue_rasa`'s keyword arguments.
    """
    pooled = protein_residue_rasa(coordinates, **labels)
    return np.asarray(
        [
            np.nan if values is None else float(np.mean(values > threshold))
            for values in pooled
        ],
        dtype=np.float64,
    )


__all__ = [
    "DEFAULT_MAX_ACC",
    "DISORDER_THRESHOLD",
    "SANDER_MAX_ACC",
    "VDW_RADII",
    "WINDOW",
    "protein_disorder",
    "protein_residue_rasa",
    "smooth_rasa",
    "unavailable_reason",
]
