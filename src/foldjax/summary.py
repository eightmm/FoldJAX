"""The common confidence block written beside each model's own scores.

Six models report confidence under five pLDDT names on two scales, and one of
them (AlphaFold 3) reports no mean pLDDT at all. `confidence.json` keeps every
native score exactly as the model named it, under ``scores``; this module adds
a ``summary`` next to it that says, for four quantities -- pLDDT, pTM, ipTM and
the model's own ranking score -- which native number it came from, what was
done to it, and over what it was averaged.

Common fields standardize names and numerical scales. They retain
model-specific definitions and calibration and do not establish comparable
accuracy probabilities or authorize pooled cross-model ranking.

A field is either a value with its provenance or ``{"value": null, "reason":
...}``. It is never 0 by default, and never filled from a native key with a
different definition: a monomer's ipTM, which several models report as 0.0,
is null here with the reason, because 0.0 would read as "no confidence in an
interface" when there is no interface.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Version of the `confidence.json` and `foldjax_run.json` contract. A minor
#: version only adds optional fields; a reader written for 1.x accepts any 1.y.
#: A change that removes, renames or reinterprets a field is a new major.
SCHEMA_VERSION = "1.0"

#: The sentence every description of the common block carries. One constant so
#: the docstrings, the JSON Schemas and the files themselves cannot drift.
COMMON_FIELDS_NOTE = (
    "Common fields standardize names and numerical scales. They retain "
    "model-specific definitions and calibration and do not establish comparable "
    "accuracy probabilities or authorize pooled cross-model ranking."
)

RANKING_SCOPE = "within one model run"


@dataclass(frozen=True, slots=True)
class _Plddt:
    """Where one model's whole-structure pLDDT comes from."""

    key: str | None  # native score key; None means derived from the structure
    native_scale: str
    granularity: str
    population: str


#: Verified against each writer and against finished runs (scales observed on
#: 9,085 `confidence.json` files): Boltz-2 and ESMFold2 report 0-1, Protenix,
#: OpenDDE and OpenFold3 0-100.
#:
#: - Boltz-2 ``complex_plddt``: the confidence head's token mean masked to real
#:   tokens (``backends/boltz2.py`` ``_CONFIDENCE_FIELDS``), upstream's own JSON
#:   key. Not ``mean_plddt``, an unmasked plain mean this port also reports.
#: - ESMFold2 ``complex_plddt``: the head's atom mean over real atoms
#:   (``models/esmfold2/models/heads.py`` ``complex_plddt``). Not ``plddt``,
#:   FoldJAX's per-token mean, which is what ESMFold2's samples are *ranked* by.
#: - Protenix / OpenDDE ``plddt``: ``summary_plddt``, the atom mean over real
#:   atoms times 100 (``models/protenix/models/heads/confidence.py``).
#: - OpenFold3 ``mean_plddt``: the atom mean over real atoms, 0-100
#:   (``models/openfold3/output.py`` ``confidence_summary``).
#: - AlphaFold 3 has no such key; see `_ALPHAFOLD3_PLDDT`.
_PLDDT: dict[str, _Plddt] = {
    "boltz2": _Plddt(
        "complex_plddt", "0-1", "token", "real tokens (each ligand atom is a token)"
    ),
    "esmfold2": _Plddt("complex_plddt", "0-1", "atom", "real atoms"),
    "opendde": _Plddt("plddt", "0-100", "atom", "real atoms"),
    "openfold3": _Plddt("mean_plddt", "0-100", "atom", "real atoms"),
    "protenix": _Plddt("plddt", "0-100", "atom", "real atoms"),
}

#: AlphaFold 3 writes per-atom pLDDT into the mmCIF B-factor column (the same
#: numbers as ``atom_plddts`` in its native ``*_confidences.json``, rounded to
#: two decimals) and no mean. The mean over every atom of the written structure
#: is the documented aggregation, and ``transform`` says so.
_ALPHAFOLD3_PLDDT = _Plddt(None, "0-100", "atom", "all atoms in the written structure")

#: The score each model ranks its own samples by. ESMFold2's is FoldJAX's
#: choice -- upstream writes one structure and ranks nothing -- and the block
#: says so through ``defined_by``. See `foldjax.output._RANKING_SCORE`.
_RANKING_DEFINED_BY = {
    "alphafold3": "upstream",
    "boltz2": "upstream",
    "esmfold2": "foldjax",
    "opendde": "upstream",
    "openfold3": "upstream",
    "protenix": "upstream",
}

#: Notes a native writer attaches to its scores that the scores themselves do
#: not carry. ESMFold2's confidence JSON says its pLDDT is 0-1 while its
#: structure's B-factors are 0-100; that sentence is kept verbatim.
SCORE_NOTES: dict[str, dict[str, str]] = {
    "esmfold2": {
        "plddt_scale": "0-1 here; the structures' b-factor column is 0-100",
    },
}


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _missing(reason: str) -> dict[str, Any]:
    return {"value": None, "reason": reason}


def structure_facts(path: Path | None) -> dict[str, Any] | None:
    """Chain count and mean B-factor of a written structure, or None.

    Read once per sample: the chain count decides whether an interface score
    means anything, and the B-factors are AlphaFold 3's only per-atom pLDDT
    on disk. Chains are counted by author chain name, so a ligand in a chain
    of its own counts, as it does for every model's ipTM.
    """
    if path is None:
        return None
    try:
        import gemmi

        structure = gemmi.read_structure(str(path))
        if len(structure) == 0:
            return None
        model = structure[0]
        chains = {chain.name for chain in model if len(chain)}
        factors = [
            atom.b_iso for chain in model for residue in chain for atom in residue
        ]
    except Exception:  # noqa: BLE001 - an unreadable file only loses the summary
        return None
    mean = math.fsum(factors) / len(factors) if factors else None
    return {"chains": len(chains), "atoms": len(factors), "mean_b_factor": mean}


def _plddt(
    model: str, scores: Mapping[str, Any], facts: Mapping[str, Any] | None
) -> dict[str, Any]:
    if model == "alphafold3":
        spec = _ALPHAFOLD3_PLDDT
        mean = None if facts is None else _finite(facts.get("mean_b_factor"))
        if mean is None:
            return _missing(
                "AlphaFold 3 reports no mean pLDDT, and the per-atom pLDDT in "
                "the structure's B-factor column could not be read"
            )
        return {
            "value": mean,
            "scale": "0-100",
            "source": "structure:_atom_site.B_iso_or_equiv",
            "transform": (
                "arithmetic mean of the per-atom pLDDT AlphaFold 3 writes as "
                "B-factors, over every atom of the written structure"
            ),
            "granularity": spec.granularity,
            "population": spec.population,
        }
    spec = _PLDDT.get(model)
    if spec is None:
        return _missing(f"no pLDDT mapping is defined for model {model!r}")
    assert spec.key is not None
    value = _finite(scores.get(spec.key))
    if value is None:
        return _missing(f"{model} reported no {spec.key!r}")
    scale = 100.0 if spec.native_scale == "0-1" else 1.0
    return {
        "value": value * scale,
        "scale": "0-100",
        "source": f"scores.{spec.key}",
        "transform": "x100 (native 0-1)" if scale != 1.0 else "identity (native 0-100)",
        "granularity": spec.granularity,
        "population": spec.population,
    }


def _ptm(model: str, scores: Mapping[str, Any]) -> dict[str, Any]:
    value = _finite(scores.get("ptm"))
    if value is None:
        return _missing(f"{model} reported no 'ptm'")
    return {
        "value": value,
        "scale": "0-1",
        "source": "scores.ptm",
        "transform": "identity (native 0-1)",
        "granularity": "token pair",
        "population": "all tokens",
    }


def _iptm(
    model: str, scores: Mapping[str, Any], facts: Mapping[str, Any] | None
) -> dict[str, Any]:
    native = scores.get("iptm")
    if facts is None:
        return _missing(
            "the structure could not be read, so whether it has an interface "
            "is unknown; a native ipTM of 0 would be indistinguishable from an "
            "undefined one"
        )
    if facts["chains"] < 2:
        detail = (
            f" (the native value {native!r} is kept in scores)"
            if native is not None
            else ""
        )
        return _missing(
            "single chain: ipTM scores interfaces between chains and this "
            f"structure has none{detail}"
        )
    value = _finite(native)
    if value is None:
        return _missing(f"{model} reported no 'iptm'")
    return {
        "value": value,
        "scale": "0-1",
        "source": "scores.iptm",
        "transform": "identity (native 0-1)",
        "granularity": "token pair",
        "population": "inter-chain token pairs",
    }


def _ranking(model: str, scores: Mapping[str, Any]) -> dict[str, Any]:
    from foldjax.output import _RANKING_SCORE

    key = _RANKING_SCORE.get(model)
    if key is None:
        return _missing(f"no ranking score is defined for model {model!r}")
    value = _finite(scores.get(key))
    if value is None:
        reason = f"{model} reported no {key!r}"
        if model == "openfold3":
            reason += (
                "; OpenFold3's score needs a protein disorder term, which needs "
                "biotite's SASA and the atoms' identities; without them the "
                "run reports 'sample_ranking_score_no_disorder' instead -- a "
                "different quantity, so it is not used here"
            )
        return {**_missing(reason), "key": key}
    return {
        "key": key,
        "value": value,
        "higher_is_better": True,
        "scope": RANKING_SCOPE,
        "defined_by": _RANKING_DEFINED_BY.get(model, "upstream"),
    }


def common_summary(
    model: str,
    scores: Mapping[str, Any],
    *,
    structure: Path | None = None,
    facts: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """The common block for one sample.

    ``facts`` (from `structure_facts`) may be passed when the caller already
    read the structure; otherwise ``structure`` is read here. Without either,
    the fields that need the structure are null with a reason.

    Common fields standardize names and numerical scales. They retain
    model-specific definitions and calibration and do not establish comparable
    accuracy probabilities or authorize pooled cross-model ranking.
    """
    if facts is None and structure is not None:
        facts = structure_facts(Path(structure))
    return {
        "plddt": _plddt(model, scores, facts),
        "ptm": _ptm(model, scores),
        "iptm": _iptm(model, scores, facts),
        "ranking": _ranking(model, scores),
    }


def plddt_source(model: str) -> str | None:
    """The native key a model's common pLDDT is read from, or None if derived."""
    if model == "alphafold3":
        return None
    spec = _PLDDT.get(model)
    return spec.key if spec is not None else None


def load_schema(name: str) -> dict[str, Any]:
    """The published JSON Schema ``"confidence"`` or ``"run"``."""
    import json
    from importlib.resources import files

    if name not in {"confidence", "run"}:
        raise ValueError(f"no FoldJAX schema named {name!r}; use 'confidence' or 'run'")
    resource = files("foldjax").joinpath("schemas", f"{name}.schema.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def mapping_table() -> list[dict[str, Any]]:
    """The per-model mapping, as rows a document or test can print."""
    from foldjax.output import _RANKING_SCORE

    rows = []
    for model in sorted(_RANKING_SCORE):
        spec = _ALPHAFOLD3_PLDDT if model == "alphafold3" else _PLDDT[model]
        rows.append(
            {
                "model": model,
                "plddt_source": spec.key or "structure B-factors",
                "plddt_native_scale": spec.native_scale,
                "plddt_granularity": spec.granularity,
                "ranking_key": _RANKING_SCORE[model],
                "ranking_defined_by": _RANKING_DEFINED_BY[model],
            }
        )
    return rows
