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
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(source)
    except OSError:
        # Some filesystems refuse symlinks; the file is small enough to copy.
        target.write_bytes(source.read_bytes())
    return target


def attach_msas(
    spec: Mapping[str, Any],
    *,
    alignment_dir: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str] | None = None,
    backend: Any = None,
    paired: bool = False,
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
    pipeline = MsaSearchPipeline(
        cache_dir=Path(cache_dir) if cache_dir is not None else root / "cache",
        backend=backend if backend is not None else RemoteMMseqs2Client(),
    )

    updated = copy.deepcopy(dict(spec))
    for query_id, query in updated.get("queries", {}).items():
        chains = [
            chain
            for chain in query.get("chains", [])
            if str(chain.get("molecule_type", "")).lower() == "protein"
            and chain.get("sequence")
        ]
        if not chains:
            continue
        results = pipeline.search([chain["sequence"] for chain in chains])
        for index, (chain, result) in enumerate(zip(chains, results, strict=True)):
            # Keyed by position rather than chain id: a chain entry can name
            # several ids, and the sequence is what was searched.
            directory = root / str(query_id) / f"chain_{index}"
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
