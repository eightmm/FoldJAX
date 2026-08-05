"""Protenix's own glue on top of the shared MSA search.

The search itself is :mod:`foldjax.search`, which was extracted verbatim from
this file back when the ports were separate packages -- so the 642 lines that
used to sit here were a byte-identical second copy of it, differing only in
whether one private helper carried a ``@staticmethod`` decorator.

What is genuinely Protenix's stays: the two functions below map search results
onto Protenix's own inference JSON, which spells its chains ``proteinChain``
and ``rnaSequence`` and wants ``unpairedMsaPath`` written back into them. That
is a consumer's requirement, not the search's, which is why the search does not
know about it.

The re-exports keep this module's import surface unchanged, so
``data/search/__init__.py`` and every caller through it are untouched.
"""

from collections.abc import Mapping
from typing import Any

from foldjax.search import (
    HttpResponse,
    LocalMsaClient,
    LocalRnaMsaClient,
    MsaBackend,
    MsaPayload,
    MsaSearchPipeline,
    RemoteMMseqs2Client,
    RnaMsaBackend,
    RnaMsaPayload,
    RnaMsaSearchPipeline,
    SearchError,
)

__all__ = [
    "HttpResponse",
    "LocalMsaClient",
    "LocalRnaMsaClient",
    "MsaBackend",
    "MsaPayload",
    "MsaSearchPipeline",
    "RemoteMMseqs2Client",
    "RnaMsaBackend",
    "RnaMsaPayload",
    "RnaMsaSearchPipeline",
    "SearchError",
    "apply_msa_paths",
    "apply_rna_msa_paths",
]


def apply_rna_msa_paths(
    inference: Mapping[str, Any], pipeline: RnaMsaSearchPipeline
) -> dict[str, Any]:
    """Return a copy with cached automatic RNA MSA paths added when absent."""

    import copy

    updated = copy.deepcopy(inference)
    chains = [
        item["rnaSequence"]
        for item in updated.get("sequences", [])
        if isinstance(item, dict)
        and isinstance(item.get("rnaSequence"), dict)
        and not item["rnaSequence"].get("unpairedMsa")
        and not item["rnaSequence"].get("unpairedMsaPath")
    ]
    if not chains:
        return updated
    paths = pipeline.search([chain.get("sequence", "") for chain in chains])
    for chain, result in zip(chains, paths, strict=True):
        chain["unpairedMsaPath"] = result["unpairedMsaPath"]
    return updated


def apply_msa_paths(
    inference: Mapping[str, Any], pipeline: MsaSearchPipeline
) -> dict[str, Any]:
    """Return an inference mapping with MSA paths added to protein chains."""
    import copy

    updated = copy.deepcopy(inference)
    chains = [
        item["proteinChain"]
        for item in updated.get("sequences", [])
        if isinstance(item, dict) and isinstance(item.get("proteinChain"), dict)
    ]
    paths = pipeline.search([chain.get("sequence", "") for chain in chains])
    for chain, result in zip(chains, paths, strict=True):
        chain.update(
            pairedMsaPath=result["pairedMsaPath"],
            unpairedMsaPath=result["unpairedMsaPath"],
        )
    return updated
