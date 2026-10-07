"""``foldjax msa prefetch``: search and cache alignments, and nothing else.

A batch on a GPU node should not spend its allocation waiting on an MSA
server, and a node without network access cannot reach one at all. Prefetch
runs the same search `foldjax predict --msa auto` runs, into the same
model-independent cache (`foldjax.paths.msa_cache_dir`), so the later
prediction reads every alignment from the cache. No weights load, no output
directory is written, and no job is translated into a native input.

Without ``--model`` each protein chain gets the per-chain search every model
shares (and, under ``--msa-pairing greedy``/``complete``, the complex pairing
search). With ``--model`` the search is the one `predict` would run for that
model -- OpenFold3's complex pairing included -- after the job is validated
for it exactly as `predict` validates it.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any


def _jobs(paths: list[Path], scratch: Path) -> list[tuple[str, Path]]:
    """``(label, single-job file)`` for every job of every input."""
    from foldjax.input import expand_jobs_file, is_jobs_file

    found: list[tuple[str, Path]] = []
    for path in paths:
        if is_jobs_file(path):
            for target, source in expand_jobs_file(path, root=scratch / "split"):
                found.append((f"{path.name}:{source.name}", target))
        else:
            found.append((path.name, path))
    return found


def _generic(job: dict[str, Any], pairing: str) -> list[dict[str, Any]]:
    """The model-independent search: per chain, plus the complex if asked."""
    from foldjax.msa_search import (
        _msa_pipeline,
        _pair_complex,
        _rna_msa_pipeline,
        _run_search,
    )
    from foldjax.search.msa import COMPLETE_PAIRING_MODE

    records: list[dict[str, Any]] = []
    wanted = [
        entity
        for entity in job["entities"]
        if entity.get("type") == "protein" and not entity.get("unpaired_msa")
    ]
    if wanted:
        pipeline = _msa_pipeline()
        if pairing == "none":
            pipeline = pipeline.without_per_chain_pairing()
        records.extend(
            _run_search(pipeline, wanted, policy="auto", paired=False, prefetch=True)
        )
        if pairing in ("greedy", "complete"):
            _pair_complex(
                pipeline,
                job,
                wanted,
                records,
                policy="auto",
                model="prefetch",
                mode=COMPLETE_PAIRING_MODE if pairing == "complete" else None,
                prefetch=True,
            )
    rna = [
        entity
        for entity in job["entities"]
        if entity.get("type") == "rna" and not entity.get("unpaired_msa")
    ]
    rna_pipeline = _rna_msa_pipeline() if rna else None
    if rna_pipeline is not None:
        records.extend(
            _run_search(rna_pipeline, rna, policy="auto", paired=False, prefetch=True)
        )
    return records


def prefetch(
    inputs: list[Path],
    *,
    models: list[str] | None = None,
    pairing: str = "model",
) -> list[dict[str, Any]]:
    """Search and cache the alignments ``inputs`` need; one record per chain.

    A chain whose search failed carries ``error``; nothing raises for it, so
    one bad sequence does not stop the rest of a batch.
    """
    from foldjax.input import (
        _TARGETS,
        _checked_common_job,
        _nucleic_msa_read,
        assign_chain_ids,
        read_job_document,
    )
    from foldjax.msa_search import _search_alignments
    from foldjax.registry import capabilities, get_backend
    from foldjax.schema import MSA_PAIRINGS

    if pairing not in MSA_PAIRINGS:
        raise ValueError(f"msa_pairing must be one of {', '.join(MSA_PAIRINGS)}")
    names = [get_backend(model).name for model in models or []]
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="foldjax-prefetch-") as raw:
        for label, path in _jobs(list(inputs), Path(raw)):
            if not names:
                job = read_job_document(path)
                if not isinstance(job, dict) or not isinstance(
                    job.get("entities"), list
                ):
                    raise ValueError(f"{label} is not a FoldJAX job document")
                assign_chain_ids(job["entities"])
                for record in _generic(job, pairing):
                    records.append({"job": label, "model": None, **record})
                continue
            for model in names:
                job = _checked_common_job(
                    path,
                    capabilities(model),
                    msa="auto",
                    options=None,
                    templates="none",
                    msa_pairing=pairing,
                )
                read = _nucleic_msa_read(model, use_rna_msa=False)
                searched = _search_alignments(
                    job,
                    _TARGETS[model],
                    policy="auto",
                    model=model,
                    pairing=pairing,
                    search_rna=read is None or "rna" in read,
                    prefetch=True,
                )
                records.extend(
                    {"job": label, "model": model, **record} for record in searched
                )
    return records


def render(records: list[dict[str, Any]]) -> str:
    return json.dumps(records, indent=2, sort_keys=True)
