"""Attach MSAs to a query specification.

Search itself is shared: :mod:`foldjax.search` holds the ColabFold MMseqs2 client
that the Chai and Protenix ports each used to carry a copy of. What is specific to
OpenFold3 is how alignments are *named*: upstream selects them by stem and only
parses database names, so the cached files -- which the search writes under its
own fixed names -- are linked to accepted stems here rather than renamed there.
Renaming them inside the cache would break every later cache hit, since the cache
key does not include filenames.
"""

from __future__ import annotations

import copy
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Upstream's accepted stems for a ColabFold search; both are keys of
# ``MSASettings.max_seq_counts``, which is what makes them parsed rather than
# silently skipped.
PAIRED_STEM = "colabfold_paired"
MAIN_STEM = "colabfold_main"


def _link(source: Path, target: Path) -> Path:
    """Point ``target`` at ``source``, preferring a symlink over a copy."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".foldjax-msa-link-", dir=target.parent
    ) as scratch:
        staged = Path(scratch) / target.name
        try:
            staged.symlink_to(source.resolve())
        except OSError:
            # Some filesystems refuse symlinks; the file is small enough to copy.
            shutil.copyfile(source, staged)
        os.replace(staged, target)
    return target


def _safe_directory(root: Path, *parts: str) -> Path:
    """Create an internal directory without following user-planted symlinks."""
    if root.is_symlink():
        raise ValueError(f"alignment directory is a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True)
    root_resolved = root.resolve()
    directory = root
    for part in parts:
        directory /= part
        # Resolve before mkdir as well: an intermediate symlink must not let the
        # mkdir itself create a directory outside ``root``.
        if not directory.resolve().is_relative_to(root_resolved):
            raise ValueError(
                f"alignment subdirectory escapes output root: {directory}"
            )
        if directory.is_symlink():
            raise ValueError(f"alignment subdirectory is a symlink: {directory}")
        directory.mkdir(exist_ok=True)
        if not directory.resolve().is_relative_to(root_resolved):
            raise ValueError(
                f"alignment subdirectory escapes output root: {directory}"
            )
    return directory


def attach_msas(
    spec: Mapping[str, Any],
    *,
    alignment_dir: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str] | None = None,
    backend: Any = None,
    paired: bool = False,
    query_id: str | None = None,
) -> dict[str, Any]:
    """Search for each protein chain's MSA and record the paths in ``spec``.

    Args:
        spec: upstream's query mapping. Only protein chains are searched; anything
            else is left alone.
        alignment_dir: where the per-chain alignment files are linked. One
            subdirectory per query and chain group, so two chains never collide.
        cache_dir: the search's own cache. Defaults to ``alignment_dir/cache``.
        backend: an ``MsaBackend``; ``RemoteMMseqs2Client`` when omitted, which
            calls the public ColabFold server.
        paired: also record the paired alignment. Off by default, and that default
            is not conservatism: pairing needs taxonomy annotations in the
            alignment headers, and an alignment that cannot be paired does not
            degrade gracefully -- upstream collapses the MSA to the query sequence
            alone, with no error. Turn it on for a multimer whose backend returns a
            genuinely paired search.
        query_id: search only this query. Required by callers that select one
            member of a multi-query document, so unused queries never trigger
            network work or write alignments.

    Returns:
        A copy of ``spec`` with ``main_msa_file_paths`` -- and
        ``paired_msa_file_paths`` when ``paired`` -- set on every protein chain
        that has a sequence.

    """
    # In-package since the search was vendored; it was an unpublished sibling
    # package before, so this import could fail and the caller had to be told
    # how to install something that was not on any index.
    from foldjax.search import MsaSearchPipeline, RemoteMMseqs2Client

    root = Path(alignment_dir)
    queries = spec.get("queries", {})
    if not isinstance(queries, Mapping):
        raise ValueError("OpenFold3 specification requires a 'queries' mapping")
    if query_id is not None and query_id not in queries:
        raise KeyError(f"{query_id!r} is not in the specification: {list(queries)}")
    indexed_queries = list(enumerate(queries.items()))
    selected = [
        (index, selected_id)
        for index, (selected_id, _) in indexed_queries
        if query_id is None or selected_id == query_id
    ]

    _safe_directory(root)
    pipeline_cache = (
        Path(cache_dir)
        if cache_dir is not None
        else _safe_directory(root, "cache")
    )
    pipeline = MsaSearchPipeline(
        cache_dir=pipeline_cache,
        backend=backend if backend is not None else RemoteMMseqs2Client(),
    )

    updated = copy.deepcopy(dict(spec))
    for query_index, selected_id in selected:
        query = updated["queries"][selected_id]
        chains = [
            chain
            for chain in query.get("chains", [])
            if str(chain.get("molecule_type", "")).lower() == "protein"
            and chain.get("sequence")
        ]
        if not chains:
            continue
        query_directory = _safe_directory(root, f"query_{query_index:04d}")
        results = pipeline.search([chain["sequence"] for chain in chains])
        for index, (chain, result) in enumerate(zip(chains, results, strict=True)):
            # Keyed by position rather than chain id: a chain entry can name
            # several ids, and the sequence is what was searched.
            # Query names are arbitrary document keys and may contain path
            # separators. Only deterministic positional identifiers reach disk.
            directory = _safe_directory(query_directory, f"chain_{index:04d}")
            chain["main_msa_file_paths"] = [
                str(
                    _link(
                        Path(result["unpairedMsaPath"]),
                        directory / f"{MAIN_STEM}.a3m",
                    )
                )
            ]
            if paired:
                chain["paired_msa_file_paths"] = [
                    str(
                        _link(
                            Path(result["pairedMsaPath"]),
                            directory / f"{PAIRED_STEM}.a3m",
                        )
                    )
                ]
    return updated
