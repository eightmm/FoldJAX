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

#: Backends whose upstream pairs a complex in one search instead of pairing each
#: chain on its own. AlphaFold 3 keeps the per-chain ``paircomplete``
#: alignment this search inherited from the Protenix port.
#:
#: Boltz-2's server search (``main.py`` ``compute_msa``) submits its protein
#: entities together as one ``pairgreedy-env`` job when there are two or more,
#: then writes each entity a CSV whose paired rows share a ``key``. Its common
#: schema takes no ``paired_msa`` -- only the search attaches one, after
#: validation -- and the Boltz-2 writer turns the pair into that CSV.
#:
#: OpenDDE submits its protein entries as one ``pairgreedy`` job whenever
#: there is more than one (`_PAIRS_EVERY_ENTRY`;
#: opendde/data/msa/msa_service_client.py:390-400).
#:
#: Protenix is not here. Its default server mode (``protenix``) pairs by NCBI
#: taxonomy on Protenix's own server, which FoldJAX does not use. Its
#: ColabFold mode submits the same ``pairgreedy`` job
#: (web_service/colab_request_parser.py:303-318) but writes the blocks to
#: ``msa/complex/<i>/pairing.a3m`` while the runner attaches only what is under
#: ``msa/<i>/<i>/`` (runner/msa_search.py:177-185), so a ColabFold-mode run
#: pairs nothing, and neither does ``msa_pairing="model"``. ``greedy`` /
#: ``complete`` opt into that complex search, read by row.
_COMPLEX_PAIRING = frozenset({"openfold3", "boltz2", "opendde"})

#: The ones whose search asks for the pairing without the environmental
#: databases (``pairgreedy`` / ``paircomplete``): OpenDDE's upstream, and the
#: search Protenix's ColabFold mode submits.
_PLAIN_PAIRING = frozenset({"opendde", "protenix"})

#: Backends whose upstream pairs whenever a job lists more than one protein
#: entry, even one sequence listed twice, and submits the entries sorted, a
#: repeat kept (the server is sent each distinct sequence once). Upstream
#: OpenDDE's ``update_seq_msa`` sorts the ``proteinChain`` sequences
#: (runner/msa_search.py:146-151) and ``search_and_build_msa`` runs the
#: pairing ticket when ``len(seqs) > 1``. One FoldJAX entity is one such entry:
#: its copies are the entry's ``count``.
_PAIRS_EVERY_ENTRY = frozenset({"opendde"})

#: Backends whose upstream writes a searched unpaired alignment environmental
#: hits first, then UniRef's: Protenix's ColabFold mode
#: (web_service/colab_request_utils.py:290-310). The cache keeps the server's
#: UniRef-then-environmental layout the other ports read, and the Protenix
#: writer reorders it (`foldjax.input.env_first_a3m`).
_ENV_FIRST_UNPAIRED = frozenset({"protenix"})

#: Backends whose common route reads a per-chain ``paired_msa`` under
#: ``msa_pairing="model"``. That alignment pairs nothing on a heteromer: the
#: chains are searched apart, so their rows are not aligned, and AlphaFold 3
#: re-pairs by the UniProt species in each header (``msa_identifiers.py``
#: ``_UNIPROT_PATTERN``), which a ColabFold ``>UniRef100_<accession>`` header
#: does not carry. AlphaFold 3 has no upstream reader of ColabFold output; its
#: own advice for a pre-paired alignment is ``unpairedMsa`` with ``pairedMsa``
#: empty and ``--resolve_msa_overlaps=false`` (docs/input.md "MSA Pairing").
_PER_CHAIN_PAIRING = frozenset({"alphafold3"})

#: Backends that read a complex search's row-paired alignment *as rows*:
#: OpenFold3 v0.5.0 takes the blocks row-aligned (sample_processing/msa.py:
#: 239-246), and Boltz-2 takes upstream's keyed CSV built from the same search
#: (boltz/main.py:500-520), whose rows sharing a key are paired. Protenix and
#: OpenDDE pair by the species `featurize_json._species_id` reads from each
#: header, and a ColabFold header carries none; under an explicit
#: ``greedy``/``complete`` their writer gives each row the species upstream
#: Protenix's ColabFold mode writes into its ``pairing.a3m``, its row number
#: (`foldjax.input.row_species_a3m`). AlphaFold 3 is refused
#: ``greedy``/``complete``: a complex-paired block would reach it unpaired.
_ROW_PAIRED = frozenset({"boltz2", "openfold3", "opendde", "protenix"})

#: Backends whose ``msa_pairing="model"`` delivers the complex search's blocks
#: with the server's own headers, as their released upstream does, so the
#: species re-pairing pairs no row beyond the query; the rows still join each
#: chain's unpaired stack (``msa_pair_as_unpair``). Upstream OpenDDE writes
#: the ``pairgreedy`` blocks as they come (msa_service_client.py:406-410,
#: ``_write_query_leading_a3m``) and its ``_UNIREF_REGEX`` (msa_utils.py:33)
#: reads no species from them. An explicit ``greedy``/``complete`` opts into
#: the row-number rewrite.
_MODEL_PAIRS_BY_SPECIES = frozenset({"opendde"})

#: Set on an entity whose ``paired_msa`` is one block of a complex search, so
#: row *i* of every chain's block is one paired row. The search sets it after
#: validation, as it attaches the alignment; a caller cannot.
ROW_PAIRED_MSA = "_row_paired_msa"

#: Set the same way on an entity of an `_ENV_FIRST_UNPAIRED` model whose
#: ``unpaired_msa`` the search attached from a backend that lays it out as
#: ColabFold's UniRef block then its environmental block
#: (``unpaired_blocks``), so the writer can put the environmental hits first.
ENV_FIRST_UNPAIRED_MSA = "_env_first_unpaired_msa"

#: Said once per input when a Protenix heteromer is folded unpaired under the
#: default ``msa_pairing="model"``.
_PROTENIX_UNPAIRED_NOTE = (
    "protenix: the chains of this heteromer are not paired, as in upstream's "
    "ColabFold mode; upstream's default taxonomy pairing needs Protenix's own "
    "MSA server, which FoldJAX does not use. --msa-pairing greedy (or "
    "msa_pairing='greedy') opts into pairing the ColabFold rows by number"
)


def complex_queries(model: str, sequences: list[str]) -> list[str] | None:
    """What one complex pairing search submits for these protein entities.

    The entities' sequences in submission order, or None when ``model``'s
    upstream pairs none of them: two or more distinct sequences in job order,
    or, for `_PAIRS_EVERY_ENTRY`, two or more entries sorted with repeats kept.
    """
    from foldjax.search.msa import _normalize_sequence

    normalized = [_normalize_sequence(sequence) for sequence in sequences]
    if model in _PAIRS_EVERY_ENTRY:
        return sorted(normalized) if len(normalized) > 1 else None
    return normalized if len(set(normalized)) > 1 else None


def resolve_pairing(model: str, pairing: str = "model") -> dict[str, Any]:
    """What ``msa_pairing`` means for ``model``, as the manifest records it.

    ``resolved`` is ``greedy``/``complete`` (one ColabFold pairing search over
    the complex, ``mode`` its ColabFold mode), ``per_chain`` (each chain's own
    ``paircomplete`` alignment) or ``none`` (no paired alignment delivered).
    ``paired_by`` is how the model joins chains' rows: ``row`` (row *i* of
    every block), ``species`` (re-paired by each header's UniProt species,
    which a ColabFold header does not carry, so no row beyond the query is
    paired: AlphaFold 3's per-chain alignment and OpenDDE's default, as their
    upstreams) or None (no paired alignment: Protenix's default, as its
    ColabFold mode, and ESMFold2).
    """
    from foldjax.schema import MSA_PAIRINGS
    from foldjax.search.msa import (
        COMPLETE_PAIRING_MODE,
        COMPLEX_PAIRING_MODE,
        PLAIN_COMPLETE_PAIRING_MODE,
        PLAIN_GREEDY_PAIRING_MODE,
    )

    if pairing not in MSA_PAIRINGS:
        raise ValueError(
            f"msa_pairing must be one of {', '.join(MSA_PAIRINGS)}; got {pairing!r}"
        )
    resolved = pairing
    if pairing == "model":
        if model in _COMPLEX_PAIRING:
            resolved = "greedy"
        elif model in _PER_CHAIN_PAIRING:
            resolved = "per_chain"
        else:
            resolved = "none"
    plain = model in _PLAIN_PAIRING
    mode = {
        "greedy": PLAIN_GREEDY_PAIRING_MODE if plain else COMPLEX_PAIRING_MODE,
        "complete": PLAIN_COMPLETE_PAIRING_MODE if plain else COMPLETE_PAIRING_MODE,
        "per_chain": "paircomplete",
    }.get(resolved)
    paired_by = {"greedy": "row", "complete": "row", "per_chain": "species"}.get(
        resolved
    )
    if pairing == "model" and model in _MODEL_PAIRS_BY_SPECIES:
        paired_by = "species"
    return {
        "requested": pairing,
        "resolved": resolved,
        "mode": mode,
        "paired_by": paired_by,
    }


def refuse_msa_pairing(model: str, pairing: str) -> None:
    """Refuse a pairing whose alignment ``model`` would not read as paired."""
    if pairing not in ("greedy", "complete") or model in _ROW_PAIRED:
        return
    if model in _PER_CHAIN_PAIRING:
        raise ValueError(
            f"{model} pairs a paired_msa again by the species in each row's "
            "UniProt header, and a ColabFold pairing search's rows carry none, "
            f"so msa_pairing={pairing!r} would reach it unpaired. "
            f"{pairing!r} is for openfold3, boltz2, protenix and opendde; use "
            "'model' (its per-chain alignment, which pairs no heteromer rows "
            "either) or 'none'"
        )
    raise ValueError(
        f"{model} reads no paired alignment, so msa_pairing={pairing!r} has "
        "nothing to deliver; use 'model' or 'none'"
    )


def report_search_failure(message: str) -> None:
    """Say that a search under ``auto`` failed, once for this input.

    In the CLI it is a progress line. ``warnings.warn`` stays the channel for
    library callers, but Python's warning registry prints an identical message
    once per call site, so in a CLI batch every input after the first failed
    silently.
    """
    _report(message, "warning")


def report_notice(message: str) -> None:
    """Say how this input is searched, once, the way `report_search_failure` does."""
    _report(message, "note")


def _report(message: str, label: str) -> None:
    from foldjax import progress

    if progress.enabled():
        progress.message(f"  {label}: {message}")
        return
    import warnings

    warnings.warn(message, UserWarning, stacklevel=5)


def _local_command(name: str) -> list[str] | None:
    """A configured local search command, split the way a shell would."""
    import shlex

    raw = os.environ.get(name, "").strip()
    return shlex.split(raw) if raw else None


def remote_msa_server() -> tuple[str, str]:
    """The remote MMseqs2 endpoint and its cache label, from the environment.

    One reading for every caller that builds a `RemoteMMseqs2Client`, so a
    port's own search helper honours the same variables as ``msa='auto'``.
    """
    host = os.environ.get(_MSA_SERVER_ENV, _DEFAULT_MSA_SERVER).strip()
    if not host:
        raise ValueError(f"{_MSA_SERVER_ENV} is set to an empty value")
    version = os.environ.get(_MSA_VERSION_ENV, "").strip() or _DEFAULT_MSA_VERSION
    return host, version


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

    host, remote_version = remote_msa_server()
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
    """What a search would use right now, for `foldjax doctor` to report.

    Redacted: a command's argv or a server URL can carry a credential.
    """
    from foldjax.redaction import redact

    protein_command = _local_command(_MSA_COMMAND_ENV)
    report = {
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
    return redact(report)


def _search_alignments(
    job: dict[str, Any],
    target: _Target,
    *,
    policy: str,
    model: str,
    search_rna: bool = True,
    pairing: str = "model",
    prefetch: bool = False,
) -> list[dict[str, str]]:
    """Fill in missing alignments, and report what was searched.

    Only protein chains: the remote MMseqs2 endpoint answers for protein, and
    the RNA pipeline in `foldjax.search.msa` needs a locally installed nhmmer
    workflow that this package does not ship. An RNA chain therefore keeps the
    behaviour it had, and ``required`` says why rather than pretending.
    ``search_rna=False`` is for a backend that does not read RNA alignments:
    nothing is searched for its RNA chains, and ``required`` does not demand it.
    ``pairing`` is ``msa_pairing`` (`resolve_pairing`). ``prefetch`` is
    `foldjax.msa_prefetch`'s call, which folds nothing, so a failure is
    reported as an alignment not cached rather than a chain folded without one.
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
        if search_rna
        and entity.get("type") == "rna"
        and not entity.get("unpaired_msa")
    ]
    rna_pipeline = _rna_msa_pipeline() if rna else None
    if policy == "required" and rna and rna_pipeline is None:
        raise ValueError(
            f"RNA entity {_ids(rna[0])[0]!r} has no alignment and no RNA search "
            f"is configured; set {_RNA_MSA_COMMAND_ENV} to a local nhmmer "
            "workflow, or supply unpaired_msa for it"
        )
    searched: list[dict[str, str]] = []
    refuse_msa_pairing(model, pairing)
    resolved = resolve_pairing(model, pairing)
    # Boltz-2's common schema takes no paired_msa; the search attaches one
    # after validation and its writer turns the pair into upstream's CSV.
    pairs_complex = resolved["resolved"] in ("greedy", "complete") and (
        "paired_msa" in target.features or model in _ROW_PAIRED
    )
    if wanted:
        pipeline = _msa_pipeline()
        # Only an asked-for "none" narrows the search: ESMFold2 resolves
        # "model" to "none" too, and keeps the cache entries it had.
        if pairing == "none":
            narrow = getattr(pipeline, "without_per_chain_pairing", None)
            pipeline = narrow() if callable(narrow) else pipeline
        searched.extend(
            _run_search(
                pipeline,
                wanted,
                policy=policy,
                paired="paired_msa" in target.features
                and resolved["resolved"] == "per_chain",
                prefetch=prefetch,
            )
        )
        backend = getattr(pipeline, "backend", None)
        if model in _ENV_FIRST_UNPAIRED and getattr(
            backend, "unpaired_blocks", None
        ) == ("uniref", "env"):
            for entity in wanted:
                if entity.get("unpaired_msa"):
                    entity[ENV_FIRST_UNPAIRED_MSA] = True
        if (
            model == "protenix"
            and pairing == "model"
            and not prefetch
            # A caller's own paired_msa is passed through, so it is paired.
            and not any(
                entity.get("paired_msa")
                for entity in job["entities"]
                if entity.get("type") == "protein"
            )
            and complex_queries(
                model,
                [
                    str(entity["sequence"])
                    for entity in job["entities"]
                    if entity.get("type") == "protein"
                ],
            )
        ):
            report_notice(_PROTENIX_UNPAIRED_NOTE)
        if pairs_complex:
            from foldjax.search.msa import COMPLEX_PAIRING_MODE

            mode = resolved["mode"]
            _pair_complex(
                pipeline,
                job,
                wanted,
                searched,
                policy=policy,
                model=model,
                mode=None if mode == COMPLEX_PAIRING_MODE else mode,
                by_row=resolved["paired_by"] == "row",
                prefetch=prefetch,
            )
    if rna and rna_pipeline is not None:
        searched.extend(
            _run_search(
                rna_pipeline, rna, policy=policy, paired=False, prefetch=prefetch
            )
        )
    return searched


def _run_search(
    pipeline: Any,
    entities: list[dict[str, Any]],
    *,
    policy: str,
    paired: bool,
    prefetch: bool = False,
) -> list[dict[str, str]]:
    """Search for these chains and attach what came back.

    Each chain keeps what its own search found: one chain's failure used to
    drop every chain's alignment. A failed chain is recorded with its
    ``error`` and reported once for this input.
    """
    from foldjax.input import _ids
    from foldjax.search.msa import SearchError

    sequences = [str(entity["sequence"]) for entity in entities]
    try:
        search_each = getattr(pipeline, "search_each", None)
        if callable(search_each):
            found = search_each(sequences)
        else:
            found = pipeline.search(sequences)
    except (SearchError, TimeoutError, OSError, ValueError) as error:
        found = [error] * len(entities)
    failed = [
        (entity, result)
        for entity, result in zip(entities, found, strict=True)
        if isinstance(result, Exception)
    ]
    if failed and policy == "required":
        error = failed[0][1]
        raise ValueError(f"MSA search failed and msa='required': {error}") from error

    searched = []
    if failed:
        # `auto` is a convenience, not a promise. A search that could not run
        # must not destroy a job that would have folded from single sequence --
        # but it must also not do so quietly, so the caller sees the reason.
        reasons = dict.fromkeys(str(result) for _, result in failed)
        consequence = (
            "no alignment was cached for them"
            if prefetch
            else "folding them from single sequence"
        )
        report_search_failure(
            f"MSA {'prefetch' if prefetch else 'search'} failed for chain(s) "
            f"{', '.join(_ids(entity)[0] for entity, _ in failed)} "
            f"({'; '.join(reasons)}); {consequence}"
        )
    for entity, result in zip(entities, found, strict=True):
        if isinstance(result, Exception):
            searched.append({"chain": _ids(entity)[0], "error": str(result)})
            continue
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


def _pair_complex(
    pipeline: Any,
    job: dict[str, Any],
    wanted: list[dict[str, Any]],
    searched: list[dict[str, str]],
    *,
    policy: str,
    model: str,
    mode: str | None = None,
    by_row: bool = True,
    prefetch: bool = False,
) -> None:
    """Pair the whole complex in one search, as OpenFold3 v0.5.0, Boltz-2 and
    OpenDDE do (and Protenix under an explicit ``greedy``/``complete``).

    ``mode`` None is the backend's own (``pairgreedy-env``, OpenFold3's);
    otherwise `resolve_pairing`'s mode for the model and strategy.
    ``by_row`` marks the blocks for the row-number rewrite
    (`ROW_PAIRED_MSA`); without it they reach the writer as the server wrote
    them.

    Upstream submits one ColabFold ``pairgreedy-env`` job per query, over its
    distinct protein sequences, and only when there is more than one of them
    (colabfold_msa_server.py:642-648, 940-975); a homomer gets no paired MSA.
    Its v0.5.0 featurizer requires every chain's paired block at one depth
    (sample_processing/msa.py:239-246), which per-chain pair jobs do not give.
    Boltz-2 submits the same job over its protein *entities* (``main.py``
    ``compute_msa``); here a common job's entities with one sequence are paired
    as one query, as OpenFold3's are. OpenDDE pairs every job with more than
    one protein entity, in sorted order, as its upstream does
    (`complex_queries`); its cache entry is keyed on that ordered list, repeats
    included.

    Only a job whose protein chains were all searched here is paired: pairing
    submits every chain's sequence, and a chain that arrived with its own
    alignment may have done so to keep its sequence off the server.
    """
    import warnings

    from foldjax.input import _ids
    from foldjax.search.msa import SearchError, _normalize_sequence

    proteins = [
        entity for entity in job["entities"] if entity.get("type") == "protein"
    ]
    queries = complex_queries(model, [str(entity["sequence"]) for entity in proteins])
    if queries is None:
        return
    searched_ids = {id(entity) for entity in wanted}
    if not all(
        id(entity) in searched_ids
        and entity.get("unpaired_msa")
        and not entity.get("paired_msa")
        for entity in proteins
    ):
        # A failed search already warned; a caller-supplied alignment did not.
        supplied = [
            _ids(entity)[0]
            for entity in proteins
            if id(entity) not in searched_ids or entity.get("paired_msa")
        ]
        if supplied:
            remedy = (
                "a native Boltz YAML whose msa fields name paired .csv files"
                if model == "boltz2"
                else "paired_msa for every protein chain"
            )
            warnings.warn(
                f"{model}: not pairing the complex because chain(s) "
                f"{', '.join(supplied)} carry their own alignment; supply "
                f"{remedy} to pair it",
                UserWarning,
                stacklevel=3,
            )
        return
    if not getattr(pipeline, "pairs_complexes", False):
        consequence = (
            "no paired MSA was cached for this heteromer"
            if prefetch
            else "this heteromer is folded without a paired MSA"
        )
        warnings.warn(
            f"{model}: the configured MSA search cannot pair a complex in one "
            f"job, so {consequence}",
            UserWarning,
            stacklevel=3,
        )
        return
    keyword: dict[str, Any] = {} if mode is None else {"mode": mode}
    if model in _PAIRS_EVERY_ENTRY:
        keyword["entries"] = True
    try:
        found = pipeline.search_complex(queries, **keyword)
    except (SearchError, TimeoutError, OSError, ValueError) as error:
        if policy == "required":
            raise ValueError(
                f"paired MSA search failed and msa='required': {error}"
            ) from error
        report_search_failure(
            f"paired MSA {'prefetch' if prefetch else 'search'} failed ({error}); "
            + (
                "no paired MSA was cached"
                if prefetch
                else "folding without a paired MSA"
            )
        )
        for record in searched:
            record.setdefault("paired_error", str(error))
        return
    records = {record["chain"]: record for record in searched}
    by_sequence = dict(zip(queries, found, strict=True))
    for entity in proteins:
        result = by_sequence[_normalize_sequence(str(entity["sequence"]))]
        entity["paired_msa"] = result["pairedMsaPath"]
        if by_row:
            entity[ROW_PAIRED_MSA] = True
        record = records.get(_ids(entity)[0])
        if record is not None:
            record["paired_msa"] = result["pairedMsaPath"]
            record["paired_provenance"] = result["provenancePath"]


def _warn_single_sequence(
    job: dict[str, Any], model: str, *, asked: bool = False
) -> None:
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
    # ``asked``: msa='single' chose this, so the way out is not news to them.
    advice = (
        "as msa='single' asked; the structure is usually worse without one."
        if asked
        else "Pass msa='auto' (--msa auto) to search for one."
    )
    warnings.warn(
        f"{model}: protein chain(s) {chains} have no alignment; predicting from "
        f"a single sequence{',' if asked else '.'} {advice}",
        UserWarning,
        stacklevel=3,
    )
