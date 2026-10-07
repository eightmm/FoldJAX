"""The structural-template search behind ``templates='auto'`` and ``'required'``.

`foldjax.input` calls into here while materializing a native input, after the
common document is validated and before the dialect is written. The hits come
from the ColabFold MMseqs2 server's PDB70 search (or a local command), the
structures from RCSB (or a local mirror), and everything model-specific -- the
date cutoff, the hit filters, how many templates a chain keeps, and which form
the backend reads -- is the selected upstream's released inference default,
tabulated in `POLICIES` with the line it comes from.

A searched template is attached to its entity in the common schema, so the
dialect writers translate it exactly as they translate one the caller wrote.
Its form is the backend's: AlphaFold 3, Protenix, OpenDDE and OpenFold3 take a
residue map (0-based query and template-residue indices); Boltz-2 aligns a
structure file itself and takes the file and chain. Upstream Boltz-2 has no
template search at all, so for it this is a FoldJAX convenience, not parity.

Like `foldjax.msa_search`, searching **sends the query sequence** to the server
unless a local command is configured, and nothing here runs unless asked for.
Under ``auto`` a search that cannot run warns and leaves the chain without
templates, and the run manifest records why; under ``required`` it fails the run.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from foldjax.redaction import redact

#: A locally installed template search, for sequences that must not leave the
#: machine. Called as ``<command> --input query.fasta --output DIR``; writes
#: ``DIR/pdb70.m8`` (BLAST-tabular hits named ``<pdb id>_<author chain>``).
#: The version string is part of the cache identity.
_TEMPLATE_COMMAND_ENV = "FOLDJAX_TEMPLATE_COMMAND"
_TEMPLATE_LOCAL_VERSION_ENV = "FOLDJAX_TEMPLATE_LOCAL_VERSION"
#: A local mmCIF mirror (flat, or PDB-divided ``ab/1abc.cif``) read before
#: anything is downloaded.
_TEMPLATE_MMCIF_DIR_ENV = "FOLDJAX_TEMPLATE_MMCIF_DIR"
#: Where structures are downloaded from; an empty value disables downloading.
_TEMPLATE_STRUCTURE_URL_ENV = "FOLDJAX_TEMPLATE_STRUCTURE_URL"
#: Every variable the search reads, for `foldjax doctor`.
TEMPLATE_ENVIRONMENT = (
    _TEMPLATE_COMMAND_ENV,
    _TEMPLATE_LOCAL_VERSION_ENV,
    _TEMPLATE_MMCIF_DIR_ENV,
    _TEMPLATE_STRUCTURE_URL_ENV,
)


@dataclass(frozen=True)
class TemplatePolicy:
    """One upstream's released template selection, as FoldJAX reproduces it."""

    #: The released default release-date cutoff; None means no cutoff.
    max_date: date | None
    #: Whether a template released *on* the cutoff date is kept.
    keep_on_cutoff: bool
    cutoff_source: str
    #: How many templates one chain keeps.
    max_templates: int
    #: True: a residue map is attached; False: only the file and chain.
    mapped: bool
    #: AlphaFold 3's hit filters (``TemplateFilterConfig`` in
    #: ``data/pipeline.py:456-462``): drop a near-copy of the query longer than
    #: 0.95 of it, a hit shorter than 10 residues, one aligning to 0.1 of the
    #: query or less, one with no resolved aligned residue, and duplicates.
    hit_filters: bool
    #: Hits considered after the cheap filters, before realignment.
    max_candidates: int | None
    #: Write each template as a single chain of observed residues (see
    #: ``_observed_chain_file``).
    observed_chain_file: bool
    selection_source: str
    #: Skip a hit whose author chain spans several polymer label chains: the
    #: AlphaFold 3 writer filters the file to that author chain and AlphaFold 3
    #: requires exactly one polymer in a template (`input.py`
    #: ``_filter_alphafold3_template_mmcif``).
    single_polymer_author: bool = False


_AF3_SELECTION = (
    "AlphaFold 3 TemplateFilterConfig(max_subsequence_ratio=0.95, "
    "min_align_ratio=0.1, min_hit_length=10, deduplicate_sequences=True, "
    "max_hits=4) (data/pipeline.py:456-462); hits in e-value order"
)
_PROTENIX_LIKE = (
    "{name} TemplateHitFeaturizer(max_hits=4, _max_template_candidates_num=20) "
    "with the AlphaFold 3 hit filters ({where}); hits in e-value order"
)

POLICIES: dict[str, TemplatePolicy] = {
    "alphafold3": TemplatePolicy(
        max_date=date(2021, 9, 30),
        keep_on_cutoff=True,
        cutoff_source=(
            "AlphaFold 3 run_alphafold.py:295-297 --max_template_date; a hit is "
            "dropped when released after it (data/templates.py:336-337)"
        ),
        max_templates=4,
        mapped=True,
        hit_filters=True,
        max_candidates=None,
        observed_chain_file=False,
        selection_source=_AF3_SELECTION,
        single_polymer_author=True,
    ),
    "protenix": TemplatePolicy(
        max_date=date(2021, 9, 30),
        keep_on_cutoff=True,
        cutoff_source=(
            "Protenix protenix/data/inference/infer_dataloader.py:107 "
            'max_template_date="2021-09-30"'
        ),
        max_templates=4,
        mapped=True,
        hit_filters=True,
        max_candidates=20,
        observed_chain_file=True,
        selection_source=_PROTENIX_LIKE.format(
            name="Protenix", where="protenix/data/inference/infer_dataloader.py:103-114"
        ),
    ),
    "opendde": TemplatePolicy(
        max_date=date(2021, 9, 30),
        keep_on_cutoff=True,
        cutoff_source=(
            "OpenDDE opendde/data/inference/infer_dataloader.py:196 "
            'max_template_date="2021-09-30"'
        ),
        max_templates=4,
        mapped=True,
        hit_filters=True,
        max_candidates=20,
        observed_chain_file=True,
        selection_source=_PROTENIX_LIKE.format(
            name="OpenDDE", where="opendde/data/inference/infer_dataloader.py:192-203"
        ),
    ),
    "openfold3": TemplatePolicy(
        max_date=None,
        keep_on_cutoff=False,
        cutoff_source=(
            "OpenFold3 v0.5.0 TemplatePreprocessorSettings.max_release_date=None "
            "(core/data/pipelines/preprocessing/template.py:1677; inference "
            "builds it with mode='predict' only, entry_points/validator.py:435-437); "
            "a set cutoff drops a template released on or after it (template.py:2698)"
        ),
        max_templates=4,
        mapped=True,
        hit_filters=False,
        max_candidates=200,
        observed_chain_file=False,
        selection_source=(
            "OpenFold3 v0.5.0: .m8 hits in e-value order (M8Parser), "
            "max_sequences_parse=200, no sequence filters, Kalign realignment "
            "to the template chain, first n_templates=4 at inference "
            "(dataset_config_components.py:128)"
        ),
    ),
    "boltz2": TemplatePolicy(
        max_date=None,
        keep_on_cutoff=True,
        cutoff_source=(
            "Boltz-2 has no template search upstream, so no released cutoff "
            "exists; none is applied unless template_max_date is set"
        ),
        max_templates=4,
        mapped=False,
        hit_filters=False,
        max_candidates=None,
        observed_chain_file=False,
        selection_source=(
            "FoldJAX convenience (upstream Boltz-2 has no template search): the "
            "first 4 hits in e-value order whose structure has the hit chain; "
            "Boltz-2 aligns each file itself"
        ),
    ),
}

#: The most ``.m8`` rows ever read, OpenFold3's ``max_sequences_parse``.
_MAX_PARSED_HITS = 200

#: What may appear in a generated file name or template id built from data.
_ENTRY_ID = re.compile(r"[^0-9A-Za-z]")


def refuse_template_search(
    model: str,
    templates: str,
    options: Mapping[str, Any] | None,
    *,
    input_format: str = "foldjax",
    template_dir: Path | None = None,
) -> None:
    """Refuse a template search where its result would be discarded.

    The search fills in FoldJAX-format jobs while they are translated; a
    native document is passed through untouched, so it would do nothing.
    A private ``template_dir`` is searched on this machine only, so it is
    refused here -- where `foldjax plan` sees it -- when neither aligner it
    can use is installed (`folder_aligner`).
    """
    if templates == "none":
        return
    if input_format != "foldjax":
        raise ValueError(
            f"templates={templates!r} searches templates for FoldJAX-format jobs while "
            f"they are translated; this {input_format!r} input is passed to "
            f"{model} untouched, so nothing would be searched. Put templates in "
            "the native document, or run a FoldJAX job"
        )
    if model not in POLICIES:
        raise ValueError(
            f"{model} has no structural-template input, so templates={templates!r} has "
            "nothing to deliver; run it with templates='none' (--templates none)"
        )
    from foldjax.input import _USE_TEMPLATE_MODELS

    if (
        model in _USE_TEMPLATE_MODELS
        and (options or {}).get("use_template") is not True
    ):
        raise ValueError(
            f"{model} reads templates only with use_template=true, and "
            "upstream's released default is false, which would discard a "
            f"searched template; templates={templates!r} does not turn it on. Set "
            "--option use_template=true to search and read them, or run with "
            "--templates none"
        )
    if template_dir is not None:
        if not _folder_files(Path(template_dir)):
            raise ValueError(
                f"template folder {template_dir} holds no "
                f"{', '.join(_FOLDER_SUFFIXES)} file"
            )
        folder_aligner(POLICIES[model])
    elif templates == "required":
        # `search_templates` fails a `required` run on this before anything is
        # sent; refused here too, so `foldjax plan` does not pass it. (`auto`
        # folds without templates and warns: `missing_template_aligner`.)
        missing = missing_template_aligner(model)
        if missing is not None:
            raise ValueError(missing)


def missing_template_aligner(model: str) -> str | None:
    """Why a server template search for ``model`` would keep no hit, if so.

    Every hit of a mapped policy is realigned with Kalign; without it the
    search is skipped before anything is sent (`search_templates`).
    """
    from foldjax.search.msa import SearchError

    if model not in POLICIES or not POLICIES[model].mapped:
        return None
    try:
        _require_kalign()
    except SearchError as error:
        return str(error)
    return None


def _local_command(name: str) -> list[str] | None:
    import shlex

    raw = os.environ.get(name, "").strip()
    return shlex.split(raw) if raw else None


def _hits_pipeline() -> tuple[Any, dict[str, Any]]:
    """The hits search: a local wrapper when configured, else the server."""
    from foldjax.msa_search import (
        _DEFAULT_MSA_SERVER,
        _DEFAULT_MSA_VERSION,
        _MSA_SERVER_ENV,
        _MSA_VERSION_ENV,
    )
    from foldjax.paths import template_cache_dir
    from foldjax.search.templates import (
        HITS_MEMBER,
        LocalTemplateHitsClient,
        RemoteTemplateHitsClient,
        TemplateHitsPipeline,
    )

    cache = template_cache_dir() / "hits"
    command = _local_command(_TEMPLATE_COMMAND_ENV)
    if command is not None:
        version = os.environ.get(_TEMPLATE_LOCAL_VERSION_ENV, "").strip() or "local"
        return (
            TemplateHitsPipeline(
                cache,
                LocalTemplateHitsClient(command, version=version),
                options={"command": command},
            ),
            {"kind": "local", "command": command, "version": version},
        )
    host = os.environ.get(_MSA_SERVER_ENV, _DEFAULT_MSA_SERVER).strip()
    if not host:
        raise ValueError(f"{_MSA_SERVER_ENV} is set to an empty value")
    version = os.environ.get(_MSA_VERSION_ENV, "").strip() or _DEFAULT_MSA_VERSION
    return (
        TemplateHitsPipeline(
            cache,
            RemoteTemplateHitsClient(host, version=version),
            options={"host": host, "mode": "env", "member": HITS_MEMBER},
        ),
        {"kind": "remote", "host": host, "version": version, "member": HITS_MEMBER},
    )


def _structure_store() -> Any:
    from foldjax.paths import template_cache_dir
    from foldjax.search.templates import DEFAULT_STRUCTURE_URL, StructureStore

    url = os.environ.get(_TEMPLATE_STRUCTURE_URL_ENV)
    return StructureStore(
        template_cache_dir() / "mmcif",
        local_dir=os.environ.get(_TEMPLATE_MMCIF_DIR_ENV, "").strip() or None,
        base_url=DEFAULT_STRUCTURE_URL if url is None else (url.strip() or None),
    )


def template_search_backend() -> dict[str, Any]:
    """What a template search would use right now, for `foldjax doctor`."""
    import importlib.util

    from foldjax.msa_search import _DEFAULT_MSA_SERVER, _MSA_SERVER_ENV

    command = _local_command(_TEMPLATE_COMMAND_ENV)
    try:
        structures = _structure_store().describe()
    except ValueError as error:  # e.g. a plain-http structure URL
        structures = {
            "local_dir": os.environ.get(_TEMPLATE_MMCIF_DIR_ENV, "").strip() or None,
            "url": None,
            "cache_dir": None,
            "error": str(error),
        }
    report = {
        "hits": (
            {"kind": "local", "command": command}
            if command
            else {
                "kind": "remote",
                "host": os.environ.get(_MSA_SERVER_ENV, _DEFAULT_MSA_SERVER),
            }
        ),
        "structures": structures,
        "aligner": (
            "kalign-python"
            if importlib.util.find_spec("kalign") is not None
            else "unavailable (install the templates extra: kalign-python)"
        ),
    }
    # Printed by `foldjax doctor`: a command's argv or a URL can carry a secret.
    return redact(report)


# --------------------------------------------------------------------------
# Private template folders (``templates_dir`` / ``--templates DIR``)

#: The suffixes a private folder's structures may have.
_FOLDER_SUFFIXES = (".cif", ".mmcif", ".cif.gz")
#: Kalign fallback: the least identity over the aligned columns, and the
#: fewest aligned residues, a folder chain needs to count as a hit. FoldJAX's
#: choice (no upstream searches a private folder); 25% is the conventional
#: edge of the twilight zone, below which a pairwise alignment no longer
#: implies homology. The mmseqs path keeps mmseqs's own e-value (1e-3).
_KALIGN_MIN_IDENTITY = 0.25
_KALIGN_MIN_ALIGNED = 10


def folder_aligner(policy: TemplatePolicy) -> str:
    """``mmseqs`` or ``kalign``: what searches a private folder here.

    mmseqs (on ``PATH``) finds the hits when installed, Kalign otherwise; a
    mapped policy also needs Kalign to realign each hit, as for any searched
    template. Raises ``ValueError`` -- the refusal `foldjax plan` shows --
    when what the policy needs is missing.
    """
    import importlib.util
    import shutil

    has_kalign = importlib.util.find_spec("kalign") is not None
    has_mmseqs = shutil.which("mmseqs") is not None
    if policy.mapped and not has_kalign:
        raise ValueError(
            "searching a private template folder realigns each hit with "
            "Kalign, which is not installed; install kalign-python (the "
            "templates extra)"
        )
    if has_mmseqs:
        return "mmseqs"
    if has_kalign:
        return "kalign"
    raise ValueError(
        "searching a private template folder needs a local aligner: mmseqs on "
        "PATH, or kalign-python (the templates extra); neither is installed, "
        "and a private folder is never sent anywhere to be searched"
    )


def folder_digest(directory: Path) -> dict[str, Any]:
    """The folder's structures by name and SHA-256, for the run manifest."""
    import hashlib

    files = {}
    for path in _folder_files(Path(directory)):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        files[str(path.relative_to(directory))] = digest.hexdigest()
    combined = hashlib.sha256(
        "\n".join(f"{name}\t{sha}" for name, sha in sorted(files.items())).encode()
    ).hexdigest()
    return {
        "path": str(Path(directory).resolve()),
        "files": len(files),
        "sha256": combined,
    }


def _folder_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and not path.name.startswith(".")
        and path.name.lower().endswith(_FOLDER_SUFFIXES)
    )


class FolderTemplateSource:
    """A private folder of mmCIF files, as the structure store of a search.

    Each file gets a key built from its name (letters and digits only, made
    unique), which stands where a PDB id stands for a searched hit: the hit
    target is ``<key>_<author chain>`` and `path` maps the key back. A
    ``.cif.gz`` is unpacked once into ``scratch``, because the readers
    downstream take a plain file.
    """

    def __init__(self, directory: Path, scratch: Path) -> None:
        self.directory = Path(directory)
        self.scratch = Path(scratch)
        self._paths: dict[str, Path] = {}
        for path in _folder_files(self.directory):
            stem = path.name
            for suffix in _FOLDER_SUFFIXES:
                if stem.lower().endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            base = _ENTRY_ID.sub("", stem).lower() or "template"
            key, index = base, 1
            while key in self._paths:
                index += 1
                key = f"{base}{index}"
            self._paths[key] = path
        if not self._paths:
            raise ValueError(
                f"template folder {self.directory} holds no "
                f"{', '.join(_FOLDER_SUFFIXES)} file"
            )

    def keys(self) -> list[str]:
        return list(self._paths)

    def describe(self) -> dict[str, Any]:
        return {
            "local_dir": str(self.directory),
            "url": None,
            "cache_dir": None,
            "kind": "private folder",
        }

    def path(self, key: str) -> Path:
        source = self._paths.get(key)
        if source is None:
            from foldjax.search.msa import SearchError

            raise SearchError(f"template folder has no structure {key!r}")
        if not source.name.lower().endswith(".gz"):
            return source.resolve()
        import gzip

        target = self.scratch / f"{key}.cif"
        if not target.is_file():
            self.scratch.mkdir(parents=True, exist_ok=True)
            from foldjax.input import _write_text_atomic

            _write_text_atomic(
                target, gzip.decompress(source.read_bytes()).decode("utf-8")
            )
        return target.resolve()


class _FolderHits:
    """`search` like a hits pipeline, answered from a private folder."""

    def __init__(self, source: FolderTemplateSource, aligner: str, scratch: Path):
        self.source = source
        self.aligner = aligner
        self.scratch = scratch

    def search(self, sequence: str) -> dict[str, Any]:
        query = "".join(sequence.split()).upper()
        return {
            "hits": folder_hits(query, self.source, self.aligner, self.scratch),
            "hitsPath": None,
            "provenancePath": None,
        }


def _folder_chains(source: FolderTemplateSource) -> list[tuple[str, str, str]]:
    """``(key, author chain, sequence)`` for every protein chain of the folder."""
    chains = []
    for key in source.keys():
        try:
            structure = read_template_structure(source.path(key))
        except (OSError, ValueError):
            continue
        for chain in structure.chains.values():
            if chain.sequence:
                chains.append((key, chain.author_id, chain.sequence))
    return chains


def folder_hits(
    query: str, source: FolderTemplateSource, aligner: str, scratch: Path
) -> list[Any]:
    """Hits for ``query`` among the folder's protein chains, best first."""
    from foldjax.search.templates import TemplateHit

    chains = _folder_chains(source)
    if aligner == "mmseqs":
        rows = _mmseqs_hits(query, chains, scratch)
    else:
        rows = _kalign_hits(query, chains)
    return [
        TemplateHit(rank=rank, **row) for rank, row in enumerate(rows)
    ]


def _mmseqs_hits(
    query: str, chains: list[tuple[str, str, str]], scratch: Path
) -> list[dict[str, Any]]:
    """``mmseqs easy-search`` of the query against the folder's chains.

    The default BLAST-tabular columns, ordered by e-value as `parse_m8` orders
    a server's hits. Targets are named ``<key>_<author chain>``; keys hold no
    underscore, so the first one splits them.
    """
    import subprocess
    import tempfile

    from foldjax.search.msa import SearchError
    from foldjax.search.templates import parse_m8_rows

    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch, prefix="mmseqs-") as raw:
        work = Path(raw)
        (work / "query.fasta").write_text(f">query\n{query}\n", encoding="utf-8")
        (work / "targets.fasta").write_text(
            "".join(f">{key}_{chain}\n{sequence}\n" for key, chain, sequence in chains),
            encoding="utf-8",
        )
        command = [
            "mmseqs",
            "easy-search",
            str(work / "query.fasta"),
            str(work / "targets.fasta"),
            str(work / "hits.m8"),
            str(work / "tmp"),
            "-v",
            "1",
        ]
        try:
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True
            )
        except OSError as error:
            raise SearchError(f"failed to start mmseqs: {error}") from error
        if completed.returncode:
            detail = (completed.stderr or completed.stdout or "").strip()[-500:]
            raise SearchError(
                f"mmseqs easy-search exited {completed.returncode}: {detail}"
            )
        hits = work / "hits.m8"
        text = hits.read_text(encoding="utf-8") if hits.is_file() else ""
    return parse_m8_rows(text)


def _kalign_hits(
    query: str, chains: list[tuple[str, str, str]]
) -> list[dict[str, Any]]:
    """Every folder chain Kalign aligns to the query well enough, best first.

    Ranked by identical residues (then input order); a chain needs
    `_KALIGN_MIN_IDENTITY` over its aligned columns and `_KALIGN_MIN_ALIGNED`
    aligned residues. No e-value exists on this path, so ``e_value`` is 0.0
    and ``bit_score`` the identical-residue count.
    """
    from foldjax.search.msa import SearchError

    scored = []
    for order, (key, chain, sequence) in enumerate(chains):
        try:
            mapping = _kalign_mapping(query, sequence)
        except SearchError:
            continue
        if len(mapping) < _KALIGN_MIN_ALIGNED:
            continue
        identical = sum(query[q] == sequence[t] for q, t in mapping.items())
        identity = identical / len(mapping)
        if identity < _KALIGN_MIN_IDENTITY:
            continue
        queries, targets = sorted(mapping), sorted(mapping.values())
        scored.append(
            (
                -identical,
                order,
                {
                    "query": "query",
                    "target": f"{key}_{chain}",
                    "pdb_id": key,
                    "chain_id": chain,
                    "identity": round(identity, 4),
                    "alignment_length": len(mapping),
                    "query_start": queries[0] + 1,
                    "query_end": queries[-1] + 1,
                    "target_start": targets[0] + 1,
                    "target_end": targets[-1] + 1,
                    "e_value": 0.0,
                    "bit_score": float(identical),
                },
            )
        )
    scored.sort(key=lambda item: (item[0], item[1]))
    return [row for _, _, row in scored]


# --------------------------------------------------------------------------
# Template structures


@dataclass(frozen=True)
class TemplateChain:
    """One protein chain of a template structure.

    ``sequence`` is the chain's full polymer sequence (``_entity_poly_seq``),
    unresolved residues included; ``numbers`` is the ``label_seq_id`` of each
    of its positions, and ``observed`` those that have coordinates.
    """

    author_id: str
    label_id: str
    sequence: str
    numbers: tuple[int, ...]
    observed: frozenset[int]


@dataclass(frozen=True)
class TemplateStructure:
    path: Path
    entry_id: str
    release_date: date | None
    chains: Mapping[str, TemplateChain] = field(default_factory=dict)
    by_author: Mapping[str, str] = field(default_factory=dict)
    #: Every polymer label chain (any polymer type) of each author chain.
    polymer_labels: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def chain(self, chain_id: str | None) -> TemplateChain | None:
        """The protein chain an author id names; a label id is the fallback.

        Without an id, the structure's only protein chain, if it has one.
        """
        if chain_id is None:
            return next(iter(self.chains.values())) if len(self.chains) == 1 else None
        label = self.by_author.get(chain_id)
        if label is not None:
            return self.chains[label]
        return self.chains.get(chain_id)


def _one_letter(name: str) -> str:
    import gemmi

    info = gemmi.find_tabulated_residue(name)
    if info is None or not info.is_amino_acid():
        return "X"
    code = info.one_letter_code.strip()
    return code.upper() if code else "X"


def _cif_text(value: str) -> str:
    import gemmi

    return gemmi.cif.as_string(value)


def read_template_structure(path: str | Path) -> TemplateStructure:
    """Read what template selection needs from an mmCIF, without coordinates."""
    path = Path(path).resolve()
    stat = path.stat()
    return _read_template_structure(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=64)
def _read_template_structure(path: str, _mtime: int, _size: int) -> TemplateStructure:
    import gemmi

    try:
        block = gemmi.cif.read(path).sole_block()
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"template mmCIF {path} could not be read: {error}") from error
    entry = _cif_text(block.find_value("_entry.id") or "") or block.name
    released: date | None = None
    revisions = []
    for value in block.find_values("_pdbx_audit_revision_history.revision_date"):
        try:
            revisions.append(date.fromisoformat(_cif_text(value)))
        except ValueError:
            continue
    if revisions:
        # The earliest revision is the release, as OpenFold3's get_release_date
        # reads it (primitives/structure/metadata.py:63-79).
        released = min(revisions)

    polymer_types = {
        _cif_text(row[0]): _cif_text(row[1]).lower()
        for row in block.find("_entity_poly.", ["entity_id", "type"])
    }
    protein_entities = {
        entity
        for entity, kind in polymer_types.items()
        if kind.startswith("polypeptide")
    }
    residues: dict[str, dict[int, str]] = {}
    for row in block.find("_entity_poly_seq.", ["entity_id", "num", "mon_id"]):
        entity = _cif_text(row[0])
        if entity not in protein_entities:
            continue
        try:
            number = int(_cif_text(row[1]))
        except ValueError:
            continue
        # A heterogeneous position lists several monomers; keep the first.
        residues.setdefault(entity, {}).setdefault(number, _cif_text(row[2]))
    asym_entity = {
        _cif_text(row[0]): _cif_text(row[1])
        for row in block.find("_struct_asym.", ["id", "entity_id"])
    }

    authors: dict[str, str] = {}
    polymer_labels: dict[str, list[str]] = {}
    observed: dict[str, set[int]] = {}
    sites = block.find(
        "_atom_site.",
        ["label_asym_id", "auth_asym_id", "label_seq_id", "?pdbx_PDB_model_num"],
    )
    first_model: str | None = None
    for row in sites:
        if row.has(3):
            model = _cif_text(row[3])
            first_model = first_model or model
            if model != first_model:
                continue
        label = _cif_text(row[0])
        author = _cif_text(row[1])
        if asym_entity.get(label) in polymer_types:
            labels = polymer_labels.setdefault(author, [])
            if label not in labels:
                labels.append(label)
        if asym_entity.get(label) not in residues:
            continue
        authors.setdefault(label, author)
        try:
            observed.setdefault(label, set()).add(int(_cif_text(row[2])))
        except ValueError:
            continue

    chains: dict[str, TemplateChain] = {}
    by_author: dict[str, str] = {}
    for label, author in authors.items():
        sequence = residues[asym_entity[label]]
        numbers = tuple(sorted(sequence))
        chains[label] = TemplateChain(
            author_id=author,
            label_id=label,
            sequence="".join(_one_letter(sequence[number]) for number in numbers),
            numbers=numbers,
            observed=frozenset(observed.get(label, ())),
        )
        by_author.setdefault(author, label)
    return TemplateStructure(
        path=Path(path),
        entry_id=str(entry),
        release_date=released,
        chains=chains,
        by_author=by_author,
        polymer_labels={key: tuple(value) for key, value in polymer_labels.items()},
    )


def _kalign_mapping(query: str, target: str) -> dict[int, int]:
    """0-based query position -> 0-based target position, from Kalign.

    The same aligner, through the same Python binding, as OpenFold3's template
    realignment (``core/data/tools/kalign.py``). Protenix and OpenDDE call the
    Kalign executable instead; the engine is the same, bitwise agreement across
    builds is not asserted.
    """
    from foldjax.search.msa import SearchError

    try:
        import kalign
    except ImportError as error:
        raise SearchError(
            "template search realigns each hit with Kalign; install "
            "kalign-python (the templates extra)"
        ) from error
    try:
        aligned = kalign.align([query, target])
    except Exception as error:  # noqa: BLE001 - the binding raises bare errors
        raise SearchError(f"Kalign failed: {error}") from error
    if len(aligned) != 2 or len(aligned[0]) != len(aligned[1]):
        raise SearchError("Kalign returned an unexpected alignment")
    mapping: dict[int, int] = {}
    query_index = target_index = -1
    for left, right in zip(aligned[0], aligned[1], strict=True):
        if left != "-":
            query_index += 1
        if right != "-":
            target_index += 1
        if left != "-" and right != "-":
            mapping[query_index] = target_index
    return mapping


def _observed_chain_file(
    structure: TemplateStructure, chain: TemplateChain, target: Path
) -> dict[int, int]:
    """Write one chain's observed polymer residues as their own mmCIF.

    Protenix and OpenDDE read a mapped template as upstream's
    ``parse_simple_cif`` does: the *first* chain of the file, indexed by its
    observed residues, ignoring any chain id (Protenix
    ``protenix/data/template/template_parser.py:370-391``, port
    ``models/protenix/data/template_features.py:413-417``). AlphaFold 3 indexes
    the chain's full sequence instead. A file holding only the selected
    chain's resolved residues, whose polymer sequence is exactly those
    residues, means the same under both readings, so the searched template is
    an ordinary common-schema template. Returns ``label_seq_id -> index``.
    """
    import gemmi

    source = gemmi.make_structure_from_block(
        gemmi.cif.read(str(structure.path)).sole_block()
    )
    selected: dict[int, Any] = {}
    for model_chain in source[0]:
        for residue in model_chain:
            if residue.subchain != chain.label_id or residue.label_seq is None:
                continue
            selected.setdefault(int(residue.label_seq), residue)
    numbers = sorted(selected)
    if not numbers:
        raise ValueError(
            f"template {structure.entry_id} chain {chain.author_id} has no "
            "resolved residues"
        )
    output = gemmi.Structure()
    output.name = structure.entry_id
    new_chain = gemmi.Chain(chain.author_id)
    for index, number in enumerate(numbers, start=1):
        residue = selected[number].clone()
        residue.label_seq = index
        residue.subchain = chain.label_id
        residue.entity_id = "1"
        new_chain.add_residue(residue)
    model = gemmi.Model(1)
    model.add_chain(new_chain)
    output.add_model(model)
    entity = gemmi.Entity("1")
    entity.entity_type = gemmi.EntityType.Polymer
    entity.polymer_type = gemmi.PolymerType.PeptideL
    entity.subchains = [chain.label_id]
    entity.full_sequence = [selected[number].name for number in numbers]
    output.entities.append(entity)
    target.parent.mkdir(parents=True, exist_ok=True)
    from foldjax.input import _write_text_atomic

    _write_text_atomic(target, output.make_mmcif_document().as_string())
    return {number: index for index, number in enumerate(numbers)}


# --------------------------------------------------------------------------
# Selection


@dataclass(frozen=True)
class _Selected:
    common: dict[str, Any]
    record: dict[str, Any]


def _cutoff(policy: TemplatePolicy, override: str | None) -> tuple[date | None, dict]:
    if override is not None:
        cutoff = date.fromisoformat(override)
        source = "template_max_date"
    else:
        cutoff = policy.max_date
        source = policy.cutoff_source
    return cutoff, {
        "max_template_date": cutoff.isoformat() if cutoff else None,
        "keeps_cutoff_date": policy.keep_on_cutoff,
        "source": source,
    }


def _passes_date(
    released: date | None, cutoff: date | None, policy: TemplatePolicy
) -> bool:
    if cutoff is None:
        return True
    if released is None:
        # A cutoff that cannot be checked is not met.
        return False
    return released <= cutoff if policy.keep_on_cutoff else released < cutoff


def _select(
    query: str,
    hits: list[Any],
    policy: TemplatePolicy,
    cutoff: date | None,
    store: Any,
    destination: Path,
) -> tuple[list[_Selected], dict[str, int]]:
    from foldjax.search.msa import SearchError

    skipped: Counter[str] = Counter()
    chosen: list[_Selected] = []
    seen_hits: set[tuple[str, str]] = set()
    seen_projections: set[str] = set()
    candidates = 0
    for hit in hits[:_MAX_PARSED_HITS]:
        if len(chosen) == policy.max_templates:
            break
        if policy.max_candidates is not None and candidates >= policy.max_candidates:
            break
        if (hit.pdb_id, hit.chain_id) in seen_hits:
            skipped["duplicate"] += 1
            continue
        seen_hits.add((hit.pdb_id, hit.chain_id))
        try:
            path = store.path(hit.pdb_id)
            structure = read_template_structure(path)
        except (SearchError, OSError, ValueError):
            skipped["no_structure"] += 1
            continue
        if not _passes_date(structure.release_date, cutoff, policy):
            skipped["release_date"] += 1
            continue
        chain = structure.chain(hit.chain_id)
        if chain is None:
            # OpenFold3 drops a hit whose author chain the entry does not have
            # (colabfold_msa_server.py:778-787).
            skipped["chain"] += 1
            continue
        if (
            policy.single_polymer_author
            and len(structure.polymer_labels.get(chain.author_id, ())) > 1
        ):
            skipped["chain"] += 1
            continue
        record = {
            **hit.summary(),
            "pdb_id": hit.pdb_id,
            "chain_id": chain.author_id,
            "label_chain_id": chain.label_id,
            "release_date": (
                structure.release_date.isoformat() if structure.release_date else None
            ),
            "mmcif": str(path),
        }
        if not policy.mapped:
            chosen.append(
                _Selected({"mmcif": str(path), "chain_id": chain.author_id}, record)
            )
            continue
        candidates += 1
        try:
            mapping = _kalign_mapping(query, chain.sequence)
        except SearchError:
            skipped["alignment"] += 1
            continue
        if not mapping:
            skipped["alignment"] += 1
            continue
        if policy.hit_filters:
            span = sorted(mapping.values())
            matching = chain.sequence[span[0] : span[-1] + 1]
            resolved = any(chain.numbers[t] in chain.observed for t in span)
            if (
                (len(matching) / len(query) > 0.95 and matching in query)
                or len(matching) < 10
                or not resolved
                or len(mapping) / len(query) <= 0.1
            ):
                skipped["filter"] += 1
                continue
            projection = "".join(
                chain.sequence[mapping[q]] if q in mapping else "-"
                for q in range(len(query))
            )
            if projection in seen_projections:
                skipped["duplicate"] += 1
                continue
            seen_projections.add(projection)
        pairs = sorted(mapping.items())
        mmcif = path
        if policy.observed_chain_file:
            # Named from the validated hit, never from the downloaded file: an
            # mmCIF's own `_entry.id` or chain ids are data, not path parts.
            label = _ENTRY_ID.sub("", chain.label_id) or "x"
            target = destination / f"{_ENTRY_ID.sub('', hit.pdb_id)}_{label}.cif"
            try:
                ordinals = _observed_chain_file(structure, chain, target)
            except (OSError, RuntimeError, ValueError):
                skipped["no_structure"] += 1
                continue
            pairs = [
                (q, ordinals[chain.numbers[t]])
                for q, t in pairs
                if chain.numbers[t] in ordinals
            ]
            if not pairs:
                skipped["filter"] += 1
                continue
            mmcif = target
            record["observed_chain_file"] = str(target)
        record["aligned_residues"] = len(pairs)
        chosen.append(
            _Selected(
                {
                    "mmcif": str(mmcif),
                    "chain_id": chain.author_id,
                    "query_indices": [q for q, _ in pairs],
                    "template_indices": [t for _, t in pairs],
                },
                record,
            )
        )
    return chosen, dict(skipped)


def _generated_directory(root: Path, directory: Path) -> None:
    """Create a generated-file directory that cannot lead outside ``root``.

    The same checks `foldjax.input` applies to its generated ``msa/``: a
    planted symlink would otherwise receive the template files written here.
    """
    if directory.is_symlink():
        raise ValueError(f"generated template directory is a symlink: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    if not directory.resolve().is_relative_to(root.resolve()):
        raise ValueError(
            f"generated template directory escapes output root: {directory}"
        )


def search_templates(
    job: dict[str, Any],
    model: str,
    *,
    max_date: str | None,
    destination: Path,
    required: bool = False,
    template_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Attach searched templates to protein chains that name none.

    ``template_dir`` searches that private folder of mmCIF files on this
    machine instead of PDB70 (`folder_hits`), and applies no release-date
    cutoff unless ``max_date`` is given: the model's cutoff exists to keep a
    PDB search clear of its training set, and a private structure usually
    carries no release date at all, which the cutoff would read as "too new".

    Returns one record per searched entity for the run manifest. A chain that
    already names templates keeps exactly those, as a chain with its own
    alignment keeps it under ``msa='auto'``.

    ``required`` (``templates='required'``) turns every warning below into a
    ``ValueError``: a search that cannot run, a chain whose search fails, and a
    chain that keeps no template. A job with no protein chain at all is refused
    too, since nothing could be searched for it.
    """
    from foldjax.input import _ids
    from foldjax.search.msa import SearchError
    from foldjax.search.templates import parse_m8

    policy = POLICIES[model]
    cutoff, cutoff_record = _cutoff(policy, max_date)
    if template_dir is not None and max_date is None:
        cutoff = None
        cutoff_record = {
            "max_template_date": None,
            "keeps_cutoff_date": policy.keep_on_cutoff,
            "source": (
                "private template folder: no release-date cutoff unless "
                "template_max_date is set"
            ),
        }
    wanted = [
        entity
        for entity in job["entities"]
        if entity.get("type") == "protein" and not entity.get("templates")
    ]
    if not wanted:
        if required and not any(
            entity.get("type") == "protein" for entity in job["entities"]
        ):
            raise ValueError(
                "templates='required' but the job has no protein chain to search "
                "templates for"
            )
        return []
    records: list[dict[str, Any]] = []
    generated = destination / "template_search"
    try:
        if policy.mapped:
            # Checked once, before anything is sent: without the aligner every
            # hit would fail on its own and the chain would fold template-free
            # with nothing but a skip count to show for it.
            _require_kalign()
        if template_dir is not None:
            aligner = folder_aligner(policy)
            _generated_directory(destination, generated)
            store = FolderTemplateSource(template_dir, generated / "private")
            pipeline = _FolderHits(store, aligner, generated)
            source = {
                "kind": "private folder",
                "directory": str(Path(template_dir).resolve()),
                "aligner": aligner,
                "hit_filter": (
                    "mmseqs easy-search default e-value (1e-3)"
                    if aligner == "mmseqs"
                    else f"Kalign identity >= {_KALIGN_MIN_IDENTITY} over "
                    f">= {_KALIGN_MIN_ALIGNED} aligned residues; ranked by "
                    "identical residues, no e-value (0.0)"
                ),
            }
        else:
            pipeline, source = _hits_pipeline()
            store = _structure_store()
        if policy.observed_chain_file:
            _generated_directory(destination, generated)
    except (SearchError, ValueError) as error:
        chains = [_ids(entity)[0] for entity in wanted]
        if required:
            _refuse_required(model, chains, error)
        _warn_failed(model, chains, error)
        return [
            {"chains": _ids(entity), "error": str(error), "cutoff": cutoff_record}
            for entity in wanted
        ]
    memo: dict[str, tuple[dict[str, str], int, list[_Selected], dict[str, int]]] = {}
    for entity in wanted:
        chains = _ids(entity)
        sequence = str(entity["sequence"])
        record: dict[str, Any] = {
            "chains": chains,
            # Written to template_search.json and the run manifest: a search
            # command's argv or a server URL can carry a credential.
            "source": redact(source),
            "structures": redact(store.describe()),
            "cutoff": cutoff_record,
            "selection": policy.selection_source,
        }
        try:
            if sequence not in memo:
                found = pipeline.search(sequence)
                hits = (
                    found["hits"]
                    if "hits" in found
                    else parse_m8(Path(found["hitsPath"]).read_text(encoding="utf-8"))
                )
                chosen, skipped = _select(
                    sequence,
                    hits,
                    policy,
                    cutoff,
                    store,
                    generated,
                )
                memo[sequence] = (found, len(hits), chosen, skipped)
            found, total, chosen, skipped = memo[sequence]
        except (SearchError, TimeoutError, OSError, ValueError) as error:
            if required:
                _refuse_required(model, chains, error)
            _warn_failed(model, chains, error)
            record["error"] = str(error)
            records.append(record)
            continue
        record.update(
            hits_path=found["hitsPath"],
            provenance=found["provenancePath"],
            hits=total,
            templates=[dict(item.record) for item in chosen],
            skipped=dict(skipped),
        )
        if chosen:
            entity["templates"] = [dict(item.common) for item in chosen]
        elif required:
            reasons = ", ".join(f"{name} {count}" for name, count in skipped.items())
            _refuse_required(
                model,
                chains,
                f"none of {total} template hits was kept ({reasons or 'none tried'})",
            )
        elif total:
            # The search ran and found hits, and every one of them was dropped:
            # a filter, a cutoff, or a download or realignment that failed.
            reasons = ", ".join(f"{name} {count}" for name, count in skipped.items())
            message = (
                f"none of {total} template hits was kept ({reasons or 'none tried'})"
            )
            record["error"] = message
            _warn_failed(model, chains, message)
        records.append(record)
    return records


def _require_kalign() -> None:
    import importlib.util

    from foldjax.search.msa import SearchError

    if importlib.util.find_spec("kalign") is None:
        raise SearchError(
            "template search realigns each hit with Kalign, which is not "
            "installed; install kalign-python (the templates extra)"
        )


def _refuse_required(model: str, chains: list[str], error: BaseException | str) -> None:
    raise ValueError(
        f"{model}: template search for chain(s) {', '.join(chains)} kept no "
        f"template and templates='required': {error}"
    ) from (error if isinstance(error, BaseException) else None)


def _warn_failed(model: str, chains: list[str], error: BaseException | str) -> None:
    # `auto` is a convenience: a search that could not run must not destroy a
    # job that folds without templates, but it must not do so quietly either.
    # The record carries the reason into the manifest; this says it now.
    from foldjax.msa_search import report_search_failure

    report_search_failure(
        f"{model}: template search for chain(s) {', '.join(chains)} failed "
        f"({error}); folding without searched templates"
    )


def entry_name(structure: TemplateStructure, fallback: str) -> str:
    """An entry id with no ``_`` -- OpenFold3 splits template ids on it."""
    name = _ENTRY_ID.sub("", structure.entry_id).lower()
    return name or fallback
