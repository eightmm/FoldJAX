"""Shared confidence-score normalization.

AlphaFold 3 and Protenix both write a per-sample summary JSON beside each
structure. Their key names differ and are left untranslated here, because a
``ptm`` from one model is not the same quantity as a ``ptm`` from another; only
the scalar/array split is common. The names and scales that *are* common live in
:mod:`foldjax.summary`, beside these native scores rather than instead of them.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

#: Protenix and OpenDDE both write "<name>_sample_<rank>.cif" next to
#: "<name>_summary_confidence_sample_<rank>.json" in one predictions directory.
_CONFIDENCE_INFIX = "_summary_confidence_sample_"

#: Numbers a native summary JSON carries that describe how the run executed,
#: not how confident the model is in the structure. They travel in sample
#: metadata (and so in `confidence.json`'s ``execution`` block) instead of among
#: the scores, where a table of confidences would otherwise list a recycle
#: count as if it were one.
EXECUTION_FIELDS = frozenset({"num_recycles"})


def sample_summary_scores(structure_path: Path) -> dict[str, float]:
    """Read the summary confidence JSON written beside ``structure_path``.

    A structure whose name does not carry the sample suffix has no matching
    summary and yields no scores.
    """
    name, separator, rank = structure_path.stem.rpartition("_sample_")
    if not separator:
        return {}
    return scalar_scores(
        structure_path.with_name(f"{name}{_CONFIDENCE_INFIX}{rank}.json")
    )


def native_rank(structure_path: Path) -> int | None:
    """The rank a Protenix-style writer put in the file name, or None."""
    _name, separator, rank = Path(structure_path).stem.rpartition("_sample_")
    if not separator:
        return None
    try:
        return int(rank)
    except ValueError:
        return None


def ranked_native_samples(
    paths: Iterable[Path],
) -> list[tuple[Path, dict[str, float], dict[str, Any]]]:
    """Scores and sample identity for structures a Protenix-style writer made.

    Protenix and OpenDDE write their samples in diffusion order but *name* each
    one by its rank (``protenix/data/output.py`` ``_sample_ranks``), so the
    number in a native file name is not the sample's position. Here the sample
    number is the diffusion index -- the order the writer returned the files in,
    restarting for each job's prediction directory -- and the rank in the name
    is kept as ``native_rank``. A run spanning several jobs records each one's
    name, because the diffusion index alone repeats across them.
    """
    structures = [Path(path) for path in paths if Path(path).suffix == ".cif"]
    several_jobs = len({path.parent for path in structures}) > 1
    seen: dict[Path, int] = {}
    records: list[tuple[Path, dict[str, float], dict[str, Any]]] = []
    for path in structures:
        index = seen.get(path.parent, 0)
        seen[path.parent] = index + 1
        scores = sample_summary_scores(path)
        metadata: dict[str, Any] = {"sample": index}
        rank = native_rank(path)
        if rank is not None:
            metadata["native_rank"] = rank
        for key in sorted(EXECUTION_FIELDS & set(scores)):
            value = scores.pop(key)
            metadata[key] = int(value) if float(value).is_integer() else value
        if several_jobs:
            # <root>/<job>/seed_<seed>/predictions/<job>_sample_<rank>.cif
            metadata["job"] = path.parent.parent.parent.name
        records.append((path, scores, metadata))
    return records


def scalar_scores(path: Path) -> dict[str, float]:
    """Return the scalar fields of a summary confidence JSON, or ``{}``.

    A missing or non-object file yields no scores rather than an error: the
    structure is the primary result and confidence output is optional in several
    of the native runners. A boolean flag such as Protenix's ``has_clash`` is
    kept as 0.0/1.0 -- the encoding AlphaFold 3 and OpenFold3 already use for the
    same flag -- rather than dropped for not being a number.
    """
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {
        key: float(value)
        for key, value in payload.items()
        if isinstance(value, (int, float))
    }
