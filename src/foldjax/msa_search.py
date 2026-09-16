"""The MSA search behind ``msa='auto'`` and ``msa='required'``.

`foldjax.input` calls into here while materializing a native input, and
`foldjax doctor` reports what a search would use. It is kept apart from the
dialect writers because it is the one part of the input layer that reaches
outside the process: every environment variable that selects a search backend
is declared here, and searching a protein chain against the public endpoint
sends the query sequence to it.

Nothing here imports JAX or a port, and the search clients in
`foldjax.search.msa` are imported inside the functions that use them: a
`doctor` report and a job that needs no search must not pay for either.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from foldjax.input import _Target

#: The public ColabFold MMseqs2 endpoint, and the label its results are cached
#: under. The label is part of the cache identity, so pointing FoldJAX at a
#: different server does not read another server's alignments back.
#:
#: Searching **sends the query sequence to that server**. That is the reason the
#: policy is opt-in rather than a better default: for some of the sequences this
#: package is used on, leaving the machine is the part that matters, not the
#: alignment.
_MSA_SERVER_ENV = "FOLDJAX_MSA_SERVER_URL"
_MSA_VERSION_ENV = "FOLDJAX_MSA_SERVER_VERSION"
_DEFAULT_MSA_SERVER = "https://api.colabfold.com"
_DEFAULT_MSA_VERSION = "colabfold-mmseqs2"

#: A locally installed search, for sequences that must not leave the machine
#: and for RNA, which no public endpoint answers. The wrapper is called as
#: ``<command> --input query.fasta --output DIR`` and writes ``pairing.a3m``
#: and ``non_pairing.a3m`` (``rna_msa.a3m`` for the RNA one) -- the contract
#: `foldjax.search.msa.LocalMsaClient` already implements. The version string
#: is part of the cache identity, so upgrading a database invalidates the
#: alignments it produced instead of silently mixing generations.
_MSA_COMMAND_ENV = "FOLDJAX_MSA_COMMAND"
_RNA_MSA_COMMAND_ENV = "FOLDJAX_RNA_MSA_COMMAND"
_LOCAL_VERSION_ENV = "FOLDJAX_MSA_LOCAL_VERSION"
_DEFAULT_LOCAL_VERSION = "local"


def _local_command(name: str) -> list[str] | None:
    """A configured local search command, split the way a shell would."""
    import shlex

    raw = os.environ.get(name, "").strip()
    return shlex.split(raw) if raw else None


def _msa_pipeline() -> Any:
    """The protein search: a local wrapper when configured, else the server."""
    from foldjax.paths import msa_cache_dir
    from foldjax.search.msa import (
        LocalMsaClient,
        MsaSearchPipeline,
        RemoteMMseqs2Client,
    )

    version = os.environ.get(_LOCAL_VERSION_ENV, "").strip() or _DEFAULT_LOCAL_VERSION
    command = _local_command(_MSA_COMMAND_ENV)
    if command is not None:
        # Nothing leaves the machine on this path, which is the whole point of
        # configuring it.
        return MsaSearchPipeline(
            msa_cache_dir(),
            LocalMsaClient(command, version=version),
            options={"command": command},
        )

    host = os.environ.get(_MSA_SERVER_ENV, _DEFAULT_MSA_SERVER).strip()
    if not host:
        raise ValueError(f"{_MSA_SERVER_ENV} is set to an empty value")
    remote_version = (
        os.environ.get(_MSA_VERSION_ENV, "").strip() or _DEFAULT_MSA_VERSION
    )
    return MsaSearchPipeline(
        msa_cache_dir(),
        RemoteMMseqs2Client(host, version=remote_version),
        options={"host": host, "pairing": "paircomplete"},
    )


def _rna_msa_pipeline() -> Any | None:
    """The RNA search, or None when no local workflow is configured.

    There is no remote fallback on purpose: the ColabFold endpoint answers for
    protein, and pretending otherwise would send an RNA sequence to a service
    that cannot search it.
    """
    from foldjax.paths import msa_cache_dir
    from foldjax.search.msa import LocalRnaMsaClient, RnaMsaSearchPipeline

    command = _local_command(_RNA_MSA_COMMAND_ENV)
    if command is None:
        return None
    version = os.environ.get(_LOCAL_VERSION_ENV, "").strip() or _DEFAULT_LOCAL_VERSION
    return RnaMsaSearchPipeline(
        msa_cache_dir(),
        LocalRnaMsaClient(command, version=version),
        options={"command": command},
    )


def msa_search_backend() -> dict[str, Any]:
    """What a search would use right now, for `foldjax doctor` to report."""
    protein_command = _local_command(_MSA_COMMAND_ENV)
    return {
        "protein": (
            {"kind": "local", "command": protein_command}
            if protein_command
            else {
                "kind": "remote",
                "host": os.environ.get(_MSA_SERVER_ENV, _DEFAULT_MSA_SERVER),
            }
        ),
        "rna": (
            {"kind": "local", "command": _local_command(_RNA_MSA_COMMAND_ENV)}
            if _local_command(_RNA_MSA_COMMAND_ENV)
            else {"kind": "unavailable", "setup": _RNA_MSA_COMMAND_ENV}
        ),
    }


def _search_alignments(
    job: dict[str, Any],
    target: _Target,
    *,
    policy: str,
    model: str,
) -> list[dict[str, str]]:
    """Fill in missing alignments, and report what was searched.

    Only protein chains: the remote MMseqs2 endpoint answers for protein, and
    the RNA pipeline in `foldjax.search.msa` needs a locally installed nhmmer
    workflow that this package does not ship. An RNA chain therefore keeps the
    behaviour it had, and ``required`` says why rather than pretending.
    """
    from foldjax.input import _ids

    if policy == "none":
        return []
    if "unpaired_msa" not in target.features:
        raise ValueError(
            f"{model} cannot take a searched alignment; run it with msa='none'"
        )
    wanted = [
        entity
        for entity in job["entities"]
        if entity.get("type") == "protein" and not entity.get("unpaired_msa")
    ]
    rna = [
        entity
        for entity in job["entities"]
        if entity.get("type") == "rna" and not entity.get("unpaired_msa")
    ]
    rna_pipeline = _rna_msa_pipeline() if rna else None
    if policy == "required" and rna and rna_pipeline is None:
        raise ValueError(
            f"RNA entity {_ids(rna[0])[0]!r} has no alignment and no RNA search "
            f"is configured; set {_RNA_MSA_COMMAND_ENV} to a local nhmmer "
            "workflow, or supply unpaired_msa for it"
        )
    searched: list[dict[str, str]] = []
    if wanted:
        searched.extend(
            _run_search(
                _msa_pipeline(),
                wanted,
                policy=policy,
                paired="paired_msa" in target.features,
            )
        )
    if rna and rna_pipeline is not None:
        searched.extend(_run_search(rna_pipeline, rna, policy=policy, paired=False))
    return searched


def _run_search(
    pipeline: Any,
    entities: list[dict[str, Any]],
    *,
    policy: str,
    paired: bool,
) -> list[dict[str, str]]:
    """Search for these chains and attach what came back."""
    from foldjax.input import _ids
    from foldjax.search.msa import SearchError

    try:
        found = pipeline.search([str(entity["sequence"]) for entity in entities])
    except (SearchError, TimeoutError, OSError, ValueError) as error:
        if policy == "required":
            raise ValueError(
                f"MSA search failed and msa='required': {error}"
            ) from error
        # `auto` is a convenience, not a promise. A search that could not run
        # must not destroy a job that would have folded from single sequence --
        # but it must also not do so quietly, so the caller sees the reason.
        import warnings

        warnings.warn(
            f"MSA search failed ({error}); folding from single sequence",
            UserWarning,
            stacklevel=3,
        )
        return []

    searched = []
    for entity, result in zip(entities, found, strict=True):
        entity["unpaired_msa"] = result["unpairedMsaPath"]
        if paired and "pairedMsaPath" in result:
            entity["paired_msa"] = result["pairedMsaPath"]
        searched.append(
            {
                "chain": _ids(entity)[0],
                "unpaired_msa": result["unpairedMsaPath"],
                "provenance": result["provenancePath"],
            }
        )
    return searched


def _warn_single_sequence(job: dict[str, Any], model: str) -> None:
    """Say out loud that a protein chain is being folded without an alignment.

    This is the one failure this layer used to have no answer for: the job is
    valid, the run succeeds, the structure is worse, and nothing anywhere says
    why. Boltz-2's ``msa: empty`` and AlphaFold 3's ``unpairedMsa: ""`` are both
    written from here, so here is where the sentence belongs.
    """
    from foldjax.input import _ids

    bare = [
        _ids(entity)[0]
        for entity in job["entities"]
        if entity.get("type") == "protein" and not entity.get("unpaired_msa")
    ]
    if not bare:
        return
    import warnings

    chains = ", ".join(bare)
    warnings.warn(
        f"{model}: protein chain(s) {chains} have no alignment; predicting from "
        "a single sequence. Pass msa='auto' (--msa auto) to search for one.",
        UserWarning,
        stacklevel=3,
    )
