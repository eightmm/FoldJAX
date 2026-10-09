"""FoldJAX common-input to model-native input conversion.

The common schema is deliberately model-neutral: modifications and covalent
bonds are expressed once here and translated into each backend's own key names.
A field a backend cannot express is rejected, because silently discarding an
MSA or a modification changes the science without changing the exit code. The
one exception is an input the upstream model itself never reads (a nucleic
alignment, a template under ``use_template=false``, or OpenDDE's pocket or
contact constraint): it is dropped as upstream drops it, with a warning and a record
in the manifest's ``ignored_msas`` / ``ignored_templates`` /
``ignored_constraints``, and ``ignore_nucleic_msa=false`` /
``ignore_templates=false`` / ``ignore_constraints=false`` refuse the job
instead.
"""

from __future__ import annotations

import difflib
import json
import math
import operator
import os
import shutil
import string
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The search moved to `foldjax.msa_search`; the three aliased names are
# re-exported because the Colab notebook and `foldjax doctor` import them from
# here, which is the name they have always had.
from foldjax.msa_search import _msa_pipeline as _msa_pipeline
from foldjax.msa_search import _rna_msa_pipeline as _rna_msa_pipeline
from foldjax.msa_search import _search_alignments, _warn_single_sequence
from foldjax.msa_search import msa_search_backend as msa_search_backend
from foldjax.pocket_selection import FEATURE as POCKET_SELECTION_FEATURE
from foldjax.portspec import PORTS, provider
from foldjax.schema import (
    MSA_POLICIES,
    TEMPLATE_POLICIES,
    ModelCapabilities,
    _strict_boolean,
)

_ENTITY_TYPES = ("protein", "dna", "rna", "ligand")
_JOB_KEYS = frozenset({"name", "entities", "bonds", "properties", "constraints"})
_POLYMER_KEYS = frozenset(
    {
        "type",
        "id",
        "sequence",
        "unpaired_msa",
        "paired_msa",
        "modifications",
        "templates",
    }
)
_LIGAND_KEYS = frozenset({"type", "id", "ccd", "smiles"})
_PROTENIX_ENTITY_NAMES = {
    "protein": "proteinChain",
    "dna": "dnaSequence",
    "rna": "rnaSequence",
    "ligand": "ligand",
}


@dataclass(frozen=True, slots=True)
class _Target:
    """What one backend's native input can actually express."""

    suffix: str
    features: frozenset[str]


_ALL_FEATURES = frozenset(
    {
        "unpaired_msa",
        "paired_msa",
        "modifications",
        "ligand_ccd",
        "ligand_smiles",
        "bonds",
        # A ligand chain of several CCD components, such as a glycan:
        # ``ccd: [NAG, NAG, BMA]``, one residue per code, numbered 1, 2, ... in
        # ``bonds`` (`_ccd_codes`). AlphaFold 3 ``ccdCodes`` (folding_input.py
        # Ligand.from_dict), Boltz-2 a ``ccd`` list (data/parse/schema.py
        # parse_boltz_schema), Protenix/OpenDDE ``CCD_A_B``
        # (data/inference/json_parser.py build_ligand, res_id = index + 1) and
        # ESMFold2 ``LigandInput.ccd`` (prepare_input.py tokenize_ligand_ccd).
        # OpenFold3 v0.5.0 raises NotImplementedError for more than one code
        # (core/data/primitives/structure/query.py), so it does not list it.
        "multi_residue_ligand",
        # A structural template, in the two forms the dialects actually take.
        # AlphaFold 3 and Protenix require an explicit query-residue ->
        # template-residue map and refuse a bare file (`folding_input.py:378`,
        # `template_features.py:329`); Boltz-2 takes a path and aligns it itself
        # (`parse/schema.py:986`). OpenDDE's shipped inference configuration
        # disables its template path unless the request explicitly enables it.
        # These are different inputs, not one input in two spellings, so they
        # are separate features and a document is refused by whichever model
        # cannot use its form.
        "templates",
        "templates_unmapped",
        # Binding affinity. Only Boltz-2 has the head, and its schema addresses
        # the binder by chain id (`parse/schema.py:988`).
        "affinity",
        # A pocket restraint: a binder chain near listed polymer residues
        # (`_pocket_constraints`). Boltz-2 ``constraints: - pocket``, OpenFold3
        # ``pocket_constraint`` and Protenix ``constraint.pocket``; AlphaFold 3
        # and ESMFold2 have no such field upstream, and OpenDDE's upstream
        # ignores it (``_IGNORED_CONSTRAINT_MODELS``).
        "pocket_constraints",
        # A contact restraint between two residues (`_contact_constraints`):
        # Boltz-2 ``constraints: - contact`` and Protenix ``constraint.contact``.
        # AlphaFold 3, ESMFold2 and OpenFold3 v0.5.0 have no such field
        # upstream; OpenDDE's upstream ignores it, as it ignores a pocket.
        "contact_constraints",
    }
)
_RESTRAINTS = frozenset({"pocket_constraints", "contact_constraints"})

# Boltz derives pairing from a single per-chain a3m, so a separate paired MSA has
# nowhere to go.
_TARGETS = {
    "alphafold3": _Target(
        ".json",
        _ALL_FEATURES - {"templates_unmapped", "affinity"} - _RESTRAINTS,
    ),
    # foldjax.models.boltz2 dispatches its parser on the file suffix and rejects .json
    # outright; JSON is a YAML subset, so the document is written as .yaml.
    "boltz2": _Target(".yaml", _ALL_FEATURES - {"paired_msa", "templates"}),
    # OpenDDE consumes most of the Protenix list-of-jobs dialect, including its
    # modified-polymer and entity/copy-addressed covalent-bond representations.
    # OpenDDE accepts Protenix-style mapped templates and exposes the dormant
    # upstream path behind ``use_template=true``.  At the released default the
    # validation below drops the field with an ``ignored_templates`` record.
    # A common pocket or contact restraint is dropped for OpenDDE (with an
    # ``ignored_constraints`` record), as its upstream drops a constraint.
    "opendde": _Target(
        ".json",
        _ALL_FEATURES - {"templates_unmapped", "affinity"} - _RESTRAINTS,
    ),
    # Protenix carries a pocket into ``constraint.pocket`` and a contact into
    # ``constraint.contact``; only weights with a constraint embedder read
    # them, and the released default profile has none, so such a run is
    # refused at the embedder exactly as a native one is.
    "protenix": _Target(".json", _ALL_FEATURES - {"templates_unmapped", "affinity"}),
    # OpenFold3 expresses everything, but with two constraints its own layer
    # enforces and this writer therefore has to: alignment files are selected by
    # *stem* and only database names are parsed, and the chains' paired MSAs must
    # be row-aligned at one depth, which `msa=auto` gets from one complex search.
    # The released OpenFold3 query schema declares covalent bonds but its
    # featurizer never applies them. Advertising the field would silently drop
    # chemistry, so reject it until the upstream pipeline consumes the contract.
    # OpenFold3 takes templates in either form: a mapped one becomes an entry of
    # the template cache its reader loads from ``template_alignment_file_path``
    # (`_openfold3_templates`), a bare file ``template_cif_paths``, which it
    # aligns itself (upstream's CIF-direct mode).
    "openfold3": _Target(
        ".json",
        _ALL_FEATURES
        - {"bonds", "affinity", "multi_residue_ligand", "contact_constraints"},
    ),
    # ESMFold2 has no second native dialect: its NumPy adapter reads the common
    # document and implements Biohub's all-biomolecule tokenizer directly.
    # Taxonomy-paired MSAs and structural templates are not part of this route;
    # every chemistry feature below is represented without loss.
    "esmfold2": _Target(
        ".json",
        {
            "unpaired_msa",
            "modifications",
            "ligand_ccd",
            "ligand_smiles",
            "bonds",
            "multi_residue_ligand",
        },
    ),
}

#: Backends whose native ligand string joins CCD codes with "_" (``CCD_A_B``,
#: Protenix and OpenDDE json_parser.py ``build_ligand``).
_CCD_JOINED_MODELS = frozenset({"opendde", "protenix"})

#: The per-job option governing a nucleic-acid alignment a backend does not
#: read. Its default, true, does what every such upstream does -- folds the
#: chain without it -- but not silently: the drop is warned about and recorded
#: in the run manifest's ``ignored_msas``. ``false`` refuses the job instead.
IGNORE_NUCLEIC_MSA = "ignore_nucleic_msa"

#: Nucleic-acid entity types whose ``unpaired_msa`` each backend's featurizer
#: actually reads. Every other nucleic alignment would reach a native document
#: and be discarded there without a word:
#:
#: - Boltz-2 keeps an ``msa`` only for protein entities; RNA and DNA chains get
#:   ``msa_id=-1`` (models/boltz2/data/parse/schema.py:1143-1146).
#: - ESMFold2 reads alignments for protein chains only
#:   (models/esmfold2/data/all_atom.py `_msa`).
#: - Protenix reads ``unpairedMsaPath`` on ``rnaSequence`` alone, and only
#:   with ``use_rna_msa=true``; a ``dnaSequence`` path is never opened
#:   (models/protenix/data/featurize_json.py `_build_nucleic_chain`). The
#:   flag is upstream's, released false (Protenix 2.0.0
#:   configs/configs_inference.py:37), and upstream then ignores the path
#:   without a word (protenix/data/msa/msa_featurizer.py:633-640).
#: - OpenDDE shares that featurizer and the same rule
#:   (models/opendde/data/featurize_json.py:47-50).
#: - OpenFold3 parses MSAs for ``MSASettings.moltypes``, PROTEIN and RNA
#:   (openfold3 dataset_config_components.py:76-79, io/sequence/msa.py:626).
#:
#: AlphaFold 3 is absent on purpose: it reads RNA alignments, and its own
#: parser already refuses ``unpairedMsaPath`` on a DNA chain
#: (alphafold3/common/folding_input.py `DnaChain.from_dict`), so nothing is
#: dropped there and the option does not apply to it.
_NUCLEIC_MSA_READ: dict[str, frozenset[str]] = {
    "boltz2": frozenset(),
    "esmfold2": frozenset(),
    "opendde": frozenset(),
    "openfold3": frozenset({"rna"}),
    "protenix": frozenset(),
}

#: Backends that read an RNA ``unpaired_msa`` only under ``use_rna_msa=true``,
#: and refuse RNA ``paired_msa`` outright: their upstreams have no field for it.
_USE_RNA_MSA_MODELS = frozenset({"opendde", "protenix"})

#: The same as ``_NUCLEIC_MSA_READ`` for ``paired_msa``, on the backends whose
#: target takes one:
#:
#: - Protenix writes ``pairedMsaPath`` on a ``dnaSequence`` too, but
#:   ``_build_nucleic_chain`` never reads it and pairs the query row alone
#:   (models/protenix/data/featurize_json.py `_build_nucleic_chain`,
#:   `_assemble_msa_features`); upstream's msa_featurizer.py does the same.
#:   OpenDDE shares that builder. Both refuse an RNA ``paired_msa`` outright
#:   (``_USE_RNA_MSA_MODELS``), so only the DNA one is governed here.
#: - OpenFold3 maps ``paired_msa_file_paths`` only for ``MSASettings.moltypes``
#:   (io/sequence/msa.py:626), so a DNA one is dropped and an RNA one read.
#:
#: Boltz-2 and ESMFold2 express no ``paired_msa`` at all, and AlphaFold 3's
#: parser refuses ``pairedMsaPath`` on an RNA or DNA chain.
_NUCLEIC_PAIRED_MSA_READ: dict[str, frozenset[str]] = {
    "opendde": frozenset(),
    "openfold3": frozenset({"rna"}),
    "protenix": frozenset(),
}

#: Backends whose upstream reads a chain's templates only under
#: ``use_template=true``, released false: Protenix
#: (``configs/configs_inference.py:36``, ``template_featurizer.py:710``) and
#: OpenDDE (``config/inference_defaults.py:28``). Without it a common-schema
#: template is discarded as upstream discards it, with a warning and a record.
_USE_TEMPLATE_MODELS = frozenset({"opendde", "protenix"})

#: Backends whose native query takes either a residue-mapped template source
#: or bare files per chain, never both: OpenFold3 refuses a chain with both
#: ``template_alignment_file_path`` and ``template_cif_paths``
#: (``inference_query_format.py:112-119``) and reads protein templates only.
_ONE_TEMPLATE_FORM = frozenset({"openfold3"})

#: The template counterpart of ``IGNORE_NUCLEIC_MSA``: true by default (drop,
#: warn, record under ``ignored_templates``), ``false`` refuses the job.
IGNORE_TEMPLATES = "ignore_templates"


#: The constraint counterpart of ``IGNORE_TEMPLATES``. It governs a native
#: document's ``constraint`` and a common job's ``constraints`` alike:
#: true by default (drop, warn, record under ``ignored_constraints``),
#: ``false`` refuses it.
IGNORE_CONSTRAINTS = "ignore_constraints"

#: Backends whose upstream reads no native ``constraint`` at inference. OpenDDE
#: shares Protenix's featurizer, which builds ``constraint_feature``, but its
#: model has no constraint embedder, and upstream's inference build warns and
#: ignores the field (OpenDDE 1.1.1 ``opendde/data/inference/
#: json_to_feature.py:28-32``; ``docs/infer_json_format.md`` "Unsupported
#: `constraint`"; ``config/model_registry.py`` lists Constraint as x).
_IGNORED_CONSTRAINT_MODELS = frozenset({"opendde"})


def accepts_ignore_constraints(model: str) -> bool:
    """Whether ``IGNORE_CONSTRAINTS`` is a meaningful option for ``model``."""
    return model in _IGNORED_CONSTRAINT_MODELS


def native_ignored_constraints(path: Path, model: str) -> list[dict[str, Any]] | None:
    """One record per native job whose ``constraint`` ``model`` never reads.

    None when ``model`` reads constraints (or has no such field), so the
    manifest keeps "not inspected" distinct from "inspected, nothing dropped".
    An empty constraint carries nothing and is not recorded, the rule
    ``_drop_fields_opendde_ignores`` applies to an empty path.
    """
    if model not in _IGNORED_CONSTRAINT_MODELS:
        return None
    try:
        document = read_job_document(Path(path))
    except (OSError, ValueError):
        return []
    jobs = document if isinstance(document, list) else [document]
    records: list[dict[str, Any]] = []
    for index, job in enumerate(jobs):
        if not isinstance(job, Mapping):
            continue
        constraint = job.get("constraint")
        if constraint in (None, {}, [], ""):
            continue
        records.append(
            {
                "job": str(job.get("name") or index),
                "field": "constraint",
                "keys": (
                    sorted(map(str, constraint))
                    if isinstance(constraint, Mapping)
                    else []
                ),
                "reason": (
                    f"{model} reads no constraint at inference (upstream's "
                    "inference build warns and ignores it); ignored, as "
                    "upstream does"
                ),
            }
        )
    return records


#: Native fields a backend's released defaults never read and its featurizer
#: drops with a warning (OpenDDE ``featurize_json._OPENDDE_IGNORED_FIELDS``):
#: entity kind -> (field, the option that reads it, manifest record list).
_NATIVE_IGNORED_FIELDS: dict[str, dict[str, tuple[tuple[str, str, str], ...]]] = {
    "opendde": {
        "proteinChain": (("templatesPath", "use_template", "templates"),),
        "rnaSequence": (("unpairedMsaPath", "use_rna_msa", "msas"),),
    },
}


def native_ignored_inputs(
    path: Path, model: str, options: Mapping[str, Any]
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]] | None]:
    """``(ignored_msas, ignored_templates)`` for a native document's drops.

    The records of what the featurizer drops, as ``native_ignored_constraints``
    records a dropped constraint: a template or RNA alignment the released
    ``use_template=false`` / ``use_rna_msa=false`` never reads. ``(None, None)``
    when ``model`` drops nothing from native input. An empty path carries
    nothing and is not recorded, as the featurizer drops it without a word.
    """
    fields = _NATIVE_IGNORED_FIELDS.get(model)
    if fields is None:
        return None, None
    records: dict[str, list[dict[str, Any]]] = {"msas": [], "templates": []}
    try:
        document = read_job_document(Path(path))
    except (OSError, ValueError):
        return records["msas"], records["templates"]
    jobs = document if isinstance(document, list) else [document]
    for index, job in enumerate(jobs):
        if not isinstance(job, Mapping):
            continue
        sequences = job.get("sequences")
        if not isinstance(sequences, list):
            continue
        for number, entry in enumerate(sequences, start=1):
            if not isinstance(entry, Mapping) or len(entry) != 1:
                continue
            kind, info = next(iter(entry.items()))
            if not isinstance(info, Mapping):
                continue
            for field, option, bucket in fields.get(kind, ()):
                value = info.get(field)
                if options.get(option) is True or not value:
                    continue
                chains = info.get("id")
                records[bucket].append(
                    {
                        "job": str(job.get("name") or index),
                        "entity": number,
                        "chains": (
                            [chains]
                            if isinstance(chains, str)
                            else list(chains)
                            if isinstance(chains, list)
                            else []
                        ),
                        "type": "protein" if kind == "proteinChain" else "rna",
                        "field": field,
                        "path": str(value),
                        "reason": (
                            f"{model} reads {kind}.{field} only with "
                            f"{option}=true, and upstream's released default is "
                            "false; ignored, as upstream does"
                        ),
                    }
                )
    return records["msas"], records["templates"]


def refuse_ignored_constraints(path: Path, model: str) -> None:
    """Refuse a native job whose constraint ``model`` would discard."""
    records = native_ignored_constraints(path, model) or []
    if not records:
        return
    jobs = ", ".join(repr(record["job"]) for record in records)
    _reject(
        model,
        "a constraint",
        f"job(s) {jobs} in {path} carry one, but upstream {model}'s inference "
        "build ignores the constraint field and only covalent_bonds reach the "
        "model. Remove it, or unset "
        f"{IGNORE_CONSTRAINTS}=false to run without it as upstream does; the "
        "run manifest then records the drop under ignored_constraints",
    )


def accepts_ignore_nucleic_msa(model: str) -> bool:
    """Whether ``IGNORE_NUCLEIC_MSA`` is a meaningful option for ``model``."""
    return model in _NUCLEIC_MSA_READ


def accepts_ignore_templates(model: str) -> bool:
    """Whether ``IGNORE_TEMPLATES`` is a meaningful option for ``model``."""
    return model in _USE_TEMPLATE_MODELS


def _nucleic_msa_read(model: str, *, use_rna_msa: bool) -> frozenset[str] | None:
    """Nucleic entity types ``model`` reads an alignment for; None if not governed."""
    kinds = _NUCLEIC_MSA_READ.get(model)
    if kinds is not None and model in _USE_RNA_MSA_MODELS and use_rna_msa:
        return kinds | {"rna"}
    return kinds


def _nucleic_msa_readers(kind: str) -> str:
    """Name the backends that would use this alignment, for the refusal."""
    if kind == "dna":
        return "no FoldJAX backend reads a DNA alignment"
    return (
        "RNA alignments are read by alphafold3 and openfold3, and by protenix "
        "and opendde with use_rna_msa=true"
    )


def _ids(entity: dict[str, Any]) -> list[str]:
    value = entity.get("id")
    if isinstance(value, str):
        identifiers = [value]
    if isinstance(value, list) and value:
        if not all(isinstance(item, str) for item in value):
            raise ValueError("every entity id must be a string")
        identifiers = value
    elif not isinstance(value, str):
        raise ValueError("every entity requires a non-empty id")
    normalized = [identifier.strip() for identifier in identifiers]
    if any(not identifier for identifier in normalized):
        raise ValueError("every entity requires a non-empty id")
    return normalized


def _reject_unknown(unknown: set[str], allowed: frozenset[str], what: str) -> None:
    """Refuse unrecognized keys, naming the field the writer probably meant.

    A misspelled ``unpaired_msa`` used to produce ``unsupported protein entity
    fields: ['unpared_msa']`` -- correct, and no help at all when the two
    spellings differ by one character in the middle of a long document.
    """
    if not unknown:
        return
    hints = []
    for name in sorted(unknown):
        close = difflib.get_close_matches(name, sorted(allowed), n=1, cutoff=0.7)
        hints.append(f"{name!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
    raise ValueError(f"unsupported {what}: {', '.join(hints)}")


def chain_labels() -> Iterator[str]:
    """A, B, ... Z, AA, AB, ... -- the order a person writes chains in."""
    letters = string.ascii_uppercase
    width = 1
    while True:
        index = 0
        total = len(letters) ** width
        while index < total:
            label = ""
            remainder = index
            for _ in range(width):
                label = letters[remainder % len(letters)] + label
                remainder //= len(letters)
            yield label
            index += 1
        width += 1


def assign_chain_ids(entities: list[dict[str, Any]]) -> None:
    """Give every entity that did not name itself the next free chain id.

    Requiring an ``id`` made the smallest possible job -- one protein -- carry a
    field whose value cannot matter, and the error for leaving it out said only
    that it was required. Explicit ids still win everywhere, and the assignment
    is positional, so the same document always produces the same chains.

    Only an *absent* id is filled in. ``id: ""`` and ``id: [""]`` keep raising:
    a field that was written and left blank is a mistake, and inventing a chain
    name for it would hide the typo rather than the ceremony.
    """
    taken: set[str] = set()
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        value = entity.get("id")
        if isinstance(value, str) and value.strip():
            taken.add(value.strip())
        elif isinstance(value, list):
            taken.update(
                item.strip() for item in value if isinstance(item, str) and item.strip()
            )
    labels = chain_labels()
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        if entity.get("id") is not None:
            continue
        label = next(labels)
        while label in taken:
            label = next(labels)
        taken.add(label)
        entity["id"] = label


#: What each polymer alphabet may contain.
#:
#: Nucleic acids are checked against IUPAC, which is complete and unambiguous --
#: a protein sequence pasted into a ``dna`` entity is a real mistake and this
#: catches it. Protein deliberately accepts every letter: which non-canonical
#: residues a model tolerates differs by model, and rejecting one here would
#: refuse a job that its backend would have run. What is refused for every
#: polymer is a character that is not a letter at all -- a digit, a gap dash, a
#: FASTA header that came along with the sequence.
_NUCLEIC_ALPHABETS = {
    "dna": frozenset("ACGTNRYKMSWBDHV"),
    "rna": frozenset("ACGUNRYKMSWBDHV"),
}


def _normalize_sequence(value: Any, *, kind: str, chain: str) -> str:
    """Return the sequence as the featurizers need it, or say exactly what is wrong.

    Whitespace is removed rather than trimmed. A YAML block scalar is how people
    paste a sequence -- ``sequence: |`` keeps the newlines and ``>`` turns them
    into spaces -- and both used to travel all the way into a featurizer, which
    reported the damage as an index error somewhere else entirely.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{kind} entity requires a non-empty string sequence")
    sequence = "".join(value.split()).upper()
    if not sequence:
        raise ValueError(f"{kind} entity requires a non-empty string sequence")
    allowed = _NUCLEIC_ALPHABETS.get(kind)
    for position, residue in enumerate(sequence, start=1):
        if residue.isalpha() if allowed is None else residue in allowed:
            continue
        window = sequence[max(0, position - 12) : position + 11]
        caret = " " * (position - max(1, position - 11)) + "^"
        expected = (
            "letters only" if allowed is None else "IUPAC " + "".join(sorted(allowed))
        )
        raise ValueError(
            f"{kind} entity {chain!r} has an unsupported residue {residue!r} at "
            f"position {position} ({expected})\n  {window}\n  {caret}"
        )
    return sequence


def _strict_position(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error


def _path(value: Any, base: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("MSA references must be non-empty path strings")
    path = Path(value.strip())
    return str(path if path.is_absolute() else (base / path).resolve())


def _reject(model: str, feature: str, detail: str) -> None:
    raise ValueError(f"{model} cannot express {feature}: {detail}")


def _ccd_codes(entity: Mapping[str, Any]) -> list[str]:
    """A CCD ligand's components in order: one code, or a ``ccd`` list.

    Residue ``i`` (1-based, as ``bonds`` and contacts number it) is code
    ``i - 1``. Empty for a SMILES ligand.
    """
    value = entity.get("ccd")
    if value is None:
        return []
    return [str(code) for code in value] if isinstance(value, list) else [str(value)]


def _residue_counts(entities: list[dict[str, Any]]) -> dict[str, int]:
    """Residues per chain: a polymer's length, a ligand's CCD codes (else 1)."""
    return {
        chain: (
            len(_ccd_codes(entity)) or 1
            if entity["type"] == "ligand"
            else len(entity["sequence"])
        )
        for entity in entities
        for chain in _ids(entity)
    }


def _modifications(entity: dict[str, Any]) -> list[tuple[str, int]]:
    """Return validated ``(ccd, position)`` pairs in the common representation."""
    value = entity.get("modifications")
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("modifications must be a list")
    length = len(str(entity["sequence"]))
    pairs: list[tuple[str, int]] = []
    seen: set[int] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {"ccd", "position"}:
            raise ValueError(
                "each modification requires exactly a ccd and a position field"
            )
        if not isinstance(item["ccd"], str):
            raise ValueError("modification ccd must be a string")
        ccd = item["ccd"].strip()
        position = _strict_position(item["position"], name="modification position")
        if not ccd:
            raise ValueError("modification ccd must be non-empty")
        if not 1 <= position <= length:
            raise ValueError(
                f"modification position {position} outside sequence length {length}"
            )
        if position in seen:
            raise ValueError(f"duplicate modification position: {position}")
        seen.add(position)
        pairs.append((ccd, position))
    return pairs


_TEMPLATE_KEYS = frozenset({"mmcif", "query_indices", "template_indices", "chain_id"})


def _templates(entity: dict[str, Any]) -> list[dict[str, Any]]:
    """Return validated structural templates for one chain.

    ``mmcif`` is a path. ``query_indices``/``template_indices`` are the
    residue map three of the five dialects require; supplying one without the
    other, or lists of different lengths, is refused here rather than producing
    a silently truncated mapping inside a featurizer.

    Both index lists are 0-based, as AlphaFold 3's ``queryIndices`` and
    ``templateIndices`` are (its ``docs/input.md``, "Structural Templates"),
    and reach AlphaFold 3 and Protenix verbatim. A template index counts the
    residues of the template chain's polymer sequence, unresolved ones
    included, as AlphaFold 3 and OpenFold3 read it; Protenix and OpenDDE
    count the observed residues of the file's first chain instead, which is
    the same thing only for a single-chain file with every residue resolved
    -- the form `foldjax.template_search` writes for them. ``chain_id`` is the
    template's author chain (``auth_asym_id``). A query index past the end of
    the sequence -- what a 1-based map has at its last position -- is refused
    rather than shifted.
    """
    value = entity.get("templates")
    if value is None:
        return []
    if not isinstance(value, list) or not value:
        raise ValueError("templates must be a non-empty list")
    templates: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("each template must be an object")
        _reject_unknown(set(item) - _TEMPLATE_KEYS, _TEMPLATE_KEYS, "template fields")
        path = item.get("mmcif")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("template mmcif must be a non-empty path string")
        query = item.get("query_indices")
        target = item.get("template_indices")
        if (query is None) != (target is None):
            raise ValueError(
                "template query_indices and template_indices go together; "
                "supply both or neither"
            )
        mapping: list[tuple[int, int]] = []
        if query is not None:
            if not isinstance(query, list) or not isinstance(target, list):
                raise ValueError("template indices must be lists of integers")
            if len(query) != len(target):
                raise ValueError(
                    f"template indices differ in length: {len(query)} query "
                    f"positions against {len(target)} template positions"
                )
            mapping = [
                (
                    _strict_position(left, name="template query index"),
                    _strict_position(right, name="template index"),
                )
                for left, right in zip(query, target, strict=True)
            ]
            length = len(str(entity.get("sequence") or ""))
            for left, right in mapping:
                if left < 0 or right < 0:
                    raise ValueError(
                        "template indices are 0-based and must not be negative"
                    )
                if length and left >= length:
                    raise ValueError(
                        f"template query index {left} is outside the "
                        f"{length}-residue sequence; indices are 0-based "
                        f"(0..{length - 1})"
                    )
        chain_id = item.get("chain_id")
        if chain_id is not None and (
            not isinstance(chain_id, str) or not chain_id.strip()
        ):
            raise ValueError("template chain_id must be a non-empty string")
        templates.append(
            {
                "mmcif": path.strip(),
                "mapping": mapping,
                "chain_id": chain_id.strip() if isinstance(chain_id, str) else None,
            }
        )
    return templates


def _affinity_binder(job: dict[str, Any], chains: set[str]) -> str | None:
    """Return the chain whose binding affinity is requested, if any."""
    value = job.get("properties")
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError("properties must be a non-empty list")
    binder: str | None = None
    for item in value:
        if not isinstance(item, dict) or set(item) != {"affinity"}:
            raise ValueError(
                "each property must be an object with exactly an affinity field"
            )
        body = item["affinity"]
        if not isinstance(body, dict) or set(body) != {"binder"}:
            raise ValueError("affinity requires exactly a binder field")
        name = body["binder"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("affinity binder must be a non-empty chain id")
        name = name.strip()
        if chains and name not in chains:
            raise ValueError(f"affinity binder is not a chain in this job: {name!r}")
        if binder is not None:
            raise ValueError("only one affinity binder is supported per job")
        binder = name
    return binder


_POCKET_KEYS = frozenset({"binder", "contacts", "max_distance", "force"})

#: Each upstream's own ``max_distance`` (Å) for a pocket that omits it.
#: Boltz-2 ``parse/schema.py:1573`` (``.get("max_distance", 6.0)``); OpenFold3
#: ``core/config/pocket_sampling_config.py:25``
#: (``DEFAULT_POCKET_CONSTRAINT_MAX_DISTANCE = 4.0``). Protenix has none: its
#: featurizer reads ``pocket["max_distance"]`` unconditionally
#: (``protenix/data/constraint/constraint_featurizer.py:323``), so a common
#: pocket without one is refused there rather than given a FoldJAX value.
_POCKET_MAX_DISTANCE_DEFAULTS: dict[str, float] = {"boltz2": 6.0, "openfold3": 4.0}


#: The kinds a common ``constraints`` item may be, each its own one-key object.
_CONSTRAINT_KINDS = ("pocket", "contact")


def _constraint_items(job: Mapping[str, Any], kind: str) -> list[Any]:
    """The bodies of the job's ``constraints`` items of ``kind``, in order."""
    value = job.get("constraints")
    if value is None:
        return []
    if not isinstance(value, list) or not value:
        raise ValueError("constraints must be a non-empty list")
    bodies = []
    for item in value:
        if (
            not isinstance(item, dict)
            or len(item) != 1
            or next(iter(item)) not in _CONSTRAINT_KINDS
        ):
            raise ValueError(
                "each constraint must be an object with exactly one field, "
                "pocket or contact"
            )
        if kind in item:
            bodies.append(item[kind])
    return bodies


def _pocket_constraints(
    job: dict[str, Any],
    chains: Mapping[str, str] | None = None,
    lengths: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Return validated pocket restraints: ``binder``, ``contacts``, ``max_distance``.

    ``chains`` maps every chain id to its entity type and ``lengths`` every
    polymer chain to its residue count; both omitted skips those checks, for
    writers re-reading a document that was already validated. Contacts are
    polymer residues in the 1-based numbering modifications and bonds use.
    ``max_distance`` is ``None`` when the job leaves it to the model. ``force``
    defaults false; only Boltz-2 reads it (`_validate_pockets`).
    """
    pockets: list[dict[str, Any]] = []
    for body in _constraint_items(job, "pocket"):
        if not isinstance(body, dict):
            raise ValueError("pocket constraint must be an object")
        _reject_unknown(set(body) - _POCKET_KEYS, _POCKET_KEYS, "pocket fields")
        binder = body.get("binder")
        if not isinstance(binder, str) or not binder.strip():
            raise ValueError("pocket binder must be a non-empty chain id")
        binder = binder.strip()
        if chains is not None and binder not in chains:
            raise ValueError(f"pocket binder is not a chain in this job: {binder!r}")
        contacts = body.get("contacts")
        if not isinstance(contacts, list) or not contacts:
            raise ValueError("pocket contacts must be a non-empty list")
        residues: list[tuple[str, int]] = []
        for contact in contacts:
            if not isinstance(contact, (list, tuple)) or len(contact) != 2:
                raise ValueError(
                    "each pocket contact must be [chain_id, residue_index]"
                )
            chain, residue = contact
            if not isinstance(chain, str) or not chain.strip():
                raise ValueError("pocket contact chain id must be a non-empty string")
            chain = chain.strip()
            residue = _strict_position(residue, name="pocket contact residue index")
            if chains is not None:
                if chain not in chains:
                    raise ValueError(
                        f"pocket contact references unknown chain id: {chain!r}"
                    )
                if chains[chain] == "ligand":
                    raise ValueError(
                        f"pocket contact {chain!r} is a ligand; contacts are "
                        "polymer residues"
                    )
                if chain == binder:
                    raise ValueError("pocket contacts cannot be on the binder chain")
            if residue < 1:
                raise ValueError(
                    "pocket contact residue index is 1-based and must be positive"
                )
            if lengths is not None and chain in lengths and residue > lengths[chain]:
                raise ValueError(
                    f"pocket contact residue {residue} outside chain {chain!r} "
                    f"length {lengths[chain]}"
                )
            residues.append((chain, residue))
        distance = body.get("max_distance")
        if distance is not None:
            if isinstance(distance, bool) or not isinstance(distance, (int, float)):
                raise ValueError("pocket max_distance must be a number")
            distance = float(distance)
            if not math.isfinite(distance) or distance <= 0:
                raise ValueError("pocket max_distance must be positive and finite")
        force = _strict_boolean(body.get("force", False), name="pocket force")
        pockets.append(
            {
                "binder": binder,
                "contacts": residues,
                "max_distance": distance,
                "force": force,
            }
        )
    return pockets


def _pocket_max_distance(model: str, pocket: Mapping[str, Any]) -> float:
    """The distance ``model`` runs ``pocket`` at: the job's, else upstream's."""
    if pocket["max_distance"] is not None:
        return float(pocket["max_distance"])
    return _POCKET_MAX_DISTANCE_DEFAULTS[model]


_CONTACT_KEYS = frozenset({"token1", "token2", "max_distance", "force"})

#: Each upstream's own ``max_distance`` (Å) for a contact that omits it.
#: Boltz-2 ``data/parse/schema.py:1574`` (``.get("max_distance", 6.0)``).
#: Protenix has none: ``_canonicalize_contact_format`` reads
#: ``pair["max_distance"]`` unconditionally
#: (``protenix/data/constraint/constraint_featurizer.py:83``), so a common
#: contact without one is refused there, as a pocket is.
_CONTACT_MAX_DISTANCE_DEFAULTS: dict[str, float] = {"boltz2": 6.0}


def _contact_constraints(
    job: Mapping[str, Any],
    chains: Mapping[str, str] | None = None,
    residues: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Return validated contact restraints: ``token1``, ``token2``, ``max_distance``.

    ``force`` defaults false; only Boltz-2 reads it (`_validate_contacts`). A
    token is ``(chain_id, residue_index)``, 1-based as in ``bonds``: a
    polymer residue, or residue *i* of a ligand (code *i* of a ``ccd`` list;
    1 for any other ligand). ``chains`` maps chain ids to entity types and
    ``residues`` every chain to its residue count; omitted, those checks are
    skipped, as for ``_pocket_constraints``. Whether a model can address a
    ligand token is ``_validate_pocket_constraints``'s question. Two equal
    tokens pass, as upstream Boltz-2's parser lets them (its featurizer marks
    that token's diagonal); Protenix refuses them as a same-chain pair.
    """
    contacts: list[dict[str, Any]] = []
    for body in _constraint_items(job, "contact"):
        if not isinstance(body, dict):
            raise ValueError("contact constraint must be an object")
        _reject_unknown(set(body) - _CONTACT_KEYS, _CONTACT_KEYS, "contact fields")
        tokens: list[tuple[str, int]] = []
        for name in ("token1", "token2"):
            token = body.get(name)
            if not isinstance(token, (list, tuple)) or len(token) != 2:
                raise ValueError(f"contact {name} must be [chain_id, residue_index]")
            chain, residue = token
            if not isinstance(chain, str) or not chain.strip():
                raise ValueError(f"contact {name} chain id must be a non-empty string")
            chain = chain.strip()
            residue = _strict_position(residue, name=f"contact {name} residue index")
            if chains is not None and chain not in chains:
                raise ValueError(
                    f"contact {name} references unknown chain id: {chain!r}"
                )
            if residue < 1:
                raise ValueError(
                    f"contact {name} residue index is 1-based and must be positive"
                )
            if residues is not None and chain in residues and residue > residues[chain]:
                raise ValueError(
                    f"contact {name} residue {residue} outside chain {chain!r}, "
                    f"which has {residues[chain]} residue(s)"
                )
            tokens.append((chain, residue))
        distance = body.get("max_distance")
        if distance is not None:
            if isinstance(distance, bool) or not isinstance(distance, (int, float)):
                raise ValueError("contact max_distance must be a number")
            distance = float(distance)
            if not math.isfinite(distance) or distance <= 0:
                raise ValueError("contact max_distance must be positive and finite")
        force = _strict_boolean(body.get("force", False), name="contact force")
        contacts.append(
            {
                "token1": tokens[0],
                "token2": tokens[1],
                "max_distance": distance,
                "force": force,
            }
        )
    return contacts


def _with_force(record: dict[str, Any], force: bool) -> dict[str, Any]:
    """``record`` plus ``force: true`` when set; unchanged otherwise.

    Keeps an unforced manifest ``constraints`` record exactly the shape it
    had before ``force`` existed.
    """
    return {**record, "force": True} if force else record


def _contact_max_distance(model: str, contact: Mapping[str, Any]) -> float:
    """The distance ``model`` runs ``contact`` at: the job's, else upstream's."""
    if contact["max_distance"] is not None:
        return float(contact["max_distance"])
    return _CONTACT_MAX_DISTANCE_DEFAULTS[model]


_Endpoint = tuple[str, int, str]


def _bonds(
    job: dict[str, Any],
    chains: set[str],
    lengths: Mapping[str, int] | None = None,
) -> list[tuple[_Endpoint, _Endpoint]]:
    """Return validated ``[chain, residue, atom]`` endpoint pairs.

    ``chains`` is the set of known chain ids, or empty to skip that check when
    the caller does not need chain resolution. ``lengths`` maps a chain id to
    its residue count, so an index past the chain's end is refused here for
    every backend, as AlphaFold 3's own parser does
    (``common/folding_input.py:1242-1246``), instead of as a Boltz ``KeyError``
    or a featurizer error after the model has loaded. Atom names are left to
    each backend's chemistry: which atoms a residue has (leaving atoms, SMILES
    atom naming) differs between them.
    """
    value = job.get("bonds")
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("bonds must be a list")
    pairs = []
    for bond in value:
        if not isinstance(bond, (list, tuple)) or len(bond) != 2:
            raise ValueError("each bond must be a pair of atom references")
        endpoints: list[_Endpoint] = []
        for endpoint in bond:
            if not isinstance(endpoint, (list, tuple)) or len(endpoint) != 3:
                raise ValueError(
                    "each bond atom must be [chain_id, residue_index, atom_name]"
                )
            chain, residue, atom = endpoint
            if not isinstance(chain, str) or not isinstance(atom, str):
                raise ValueError("bond chain id and atom name must be strings")
            chain, atom = chain.strip(), atom.strip()
            if not chain or not atom:
                raise ValueError("bond chain id and atom name must be non-empty")
            residue = _strict_position(residue, name="bond residue index")
            if chains and chain not in chains:
                raise ValueError(f"bond references unknown chain id: {chain!r}")
            if residue < 1:
                raise ValueError("bond residue index is 1-based and must be positive")
            if lengths is not None and chain in lengths and residue > lengths[chain]:
                raise ValueError(
                    f"bond residue index {residue} is outside chain {chain!r}, "
                    f"which has {lengths[chain]} residue(s)"
                )
            endpoints.append((chain, residue, atom))
        pairs.append((endpoints[0], endpoints[1]))
    return pairs


def _require_component(code: str, *, what: str) -> None:
    """Refuse a CCD code the installed Chemical Component Dictionary lacks."""
    from foldjax import ccd

    # Exact: every backend passes the code through as written, and the
    # dictionary's codes are uppercase, so 'atp' fails in all of them.
    found = ccd.lookup(code.strip())
    if found is not None and not found[0]:
        upper = code.strip().upper()
        hint = (
            f" (CCD codes are uppercase: {upper!r})"
            if upper != code.strip() and (ccd.lookup(upper) or (False,))[0]
            else ""
        )
        raise ValueError(
            f"{what} {code!r} is not in the wwPDB Chemical Component "
            f"Dictionary{hint}"
        )


def _require_smiles(smiles: str, *, chain: str) -> None:
    """Refuse a SMILES string RDKit cannot read, when RDKit is installed.

    Every backend builds a SMILES ligand with RDKit; a string it cannot parse
    surfaced as a Boost.Python ``ArgumentError`` from inside a featurizer,
    after the model had loaded.
    """
    try:
        from rdkit import Chem, rdBase
    except ImportError:
        return
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles, sanitize=False)
        if molecule is None:
            raise ValueError(
                f"ligand {chain!r} SMILES {smiles!r} is not a valid SMILES "
                "string (RDKit cannot parse it)"
            )
        try:
            Chem.SanitizeMol(molecule)
        except Exception as error:  # noqa: BLE001 - RDKit raises several types
            detail = str(error).strip().splitlines()
            raise ValueError(
                f"ligand {chain!r} SMILES {smiles!r} is not a valid molecule: "
                f"{detail[0] if detail else type(error).__name__}"
            ) from None


def _require_input_files(entities: list[dict[str, Any]], base: Path) -> None:
    """Refuse an alignment or template path that names no file."""
    for entity in entities:
        if entity.get("type") == "ligand":
            continue
        chain = _ids(entity)[0]
        named = [
            (field, entity[field])
            for field in ("unpaired_msa", "paired_msa")
            if isinstance(entity.get(field), str)
        ]
        named.extend(
            ("template mmcif", template["mmcif"])
            for template in entity.get("templates") or []
            if isinstance(template, dict) and isinstance(template.get("mmcif"), str)
        )
        for field, value in named:
            path = Path(value.strip())
            resolved = path if path.is_absolute() else (base / path).absolute()
            if not resolved.is_file():
                raise FileNotFoundError(
                    f"entity {chain!r} names {field} {value.strip()!r}, and there "
                    f"is no such file: {resolved}"
                )


def _residue_component(entity: dict[str, Any], residue: int) -> str | None:
    """The CCD component at a bond endpoint, or None when it has no fixed one."""
    from foldjax import ccd

    kind = entity["type"]
    if kind == "ligand":
        # Residue ``i`` of a CCD ligand is its ``i``-th code; a SMILES
        # ligand's atom names are each backend's own.
        codes = _ccd_codes(entity)
        return codes[residue - 1] if 0 < residue <= len(codes) else None
    modified = {position: code for code, position in _modifications(entity)}
    if residue in modified:
        return modified[residue]
    table = {
        "protein": ccd.PROTEIN_RESIDUES,
        "dna": ccd.DNA_RESIDUES,
        "rna": ccd.RNA_RESIDUES,
    }[kind]
    return table.get(entity["sequence"][residue - 1])


def _check_bond_atoms(
    bonds: list[tuple[_Endpoint, _Endpoint]], entities: list[dict[str, Any]]
) -> None:
    """Refuse a bond atom its residue's CCD component does not have.

    Existence only: whether a backend strips that atom (a leaving group, a
    hydrogen) is its own chemistry. Skipped where no dictionary is installed.
    The residue is named by the job's own 1-based index.
    """
    if not bonds:
        return
    from foldjax import ccd

    by_chain = {chain: entity for entity in entities for chain in _ids(entity)}
    for bond in bonds:
        for chain, residue, atom in bond:
            entity = by_chain.get(chain)
            code = None if entity is None else _residue_component(entity, residue)
            if code is None:
                continue
            found = ccd.lookup(code.strip())
            if found is None or not found[0] or not found[1] or atom in found[1]:
                continue
            shown = ", ".join(found[1][:40]) + (", ..." if len(found[1]) > 40 else "")
            raise ValueError(
                f"bond atom {atom!r} is not an atom of residue {residue} "
                f"({code}) of chain {chain!r}; {code} has {shown}"
            )


def _validate(
    job: dict[str, Any],
    model: str,
    target: _Target,
    entity_types: tuple[str, ...],
    *,
    options: Mapping[str, Any] | None = None,
    ignored: list[dict[str, Any]] | None = None,
    ignored_templates: list[dict[str, Any]] | None = None,
    ignored_constraints: list[dict[str, Any]] | None = None,
    base: Path | None = None,
    pocket_conditioning: bool | None = None,
    selection_pockets: list[dict[str, Any]] | None = None,
) -> None:
    """Check the common document against what ``model`` can express.

    ``base`` is the job file's directory. Given, every alignment and template
    path the backend will read must exist; omitted, paths are not checked.

    A nucleic-acid alignment the backend does not read is removed from ``job``
    and described in ``ignored``, and a template a backend would discard at
    ``use_template=false`` in ``ignored_templates``, as their upstreams drop
    them; ``ignore_nucleic_msa=false`` or ``ignore_templates=false`` refuses
    the job instead. A pocket or contact restraint OpenDDE's upstream ignores
    goes to ``ignored_constraints`` the same way (``ignore_constraints=false``).
    """
    options = options or {}
    # Only Boltz-2 resolves a CCD code against its own chemistry archive, and
    # only it therefore validates the identifier here. The validator is named by
    # the port table and imported at this call rather than at module import, so
    # the model-neutral input layer does not load a port to translate for
    # another one.
    spec = PORTS.get(model)
    validate_ccd = (
        provider(spec.input_ccd_validator)
        if spec is not None and spec.input_ccd_validator is not None
        else None
    )
    use_template = False
    use_rna_msa = False
    ignore_templates = False
    if model in _USE_TEMPLATE_MODELS:
        use_template = _strict_boolean(
            options.get("use_template", False), name="use_template"
        )
        explicit = options.get(IGNORE_TEMPLATES)
        ignore_templates = explicit is None or _strict_boolean(
            explicit, name=IGNORE_TEMPLATES
        )
        if use_template and explicit is not None and ignore_templates:
            raise ValueError(
                f"use_template=true reads the job's templates and "
                f"{IGNORE_TEMPLATES}=true drops them; set one of the two"
            )
    if model in _USE_RNA_MSA_MODELS:
        use_rna_msa = _strict_boolean(
            options.get("use_rna_msa", False), name="use_rna_msa"
        )
    nucleic_msa_read = _nucleic_msa_read(model, use_rna_msa=use_rna_msa)
    ignore_nucleic_msa = nucleic_msa_read is not None and _strict_boolean(
        options.get(IGNORE_NUCLEIC_MSA, True), name=IGNORE_NUCLEIC_MSA
    )
    _reject_unknown(set(job) - _JOB_KEYS, _JOB_KEYS, "top-level fields")
    entities = job.get("entities")
    if not isinstance(entities, list) or not entities:
        raise ValueError("FoldJAX input requires a non-empty entities list")
    assign_chain_ids(entities)

    chains: set[str] = set()
    for entity in entities:
        if not isinstance(entity, dict):
            raise ValueError("each entity must be an object")
        kind = entity.get("type")
        if kind not in _ENTITY_TYPES:
            raise ValueError(f"unsupported entity type: {kind!r}")
        if kind not in entity_types:
            _reject(model, "an entity type", f"{kind} is not a supported entity")
        allowed = _LIGAND_KEYS if kind == "ligand" else _POLYMER_KEYS
        _reject_unknown(set(entity) - allowed, allowed, f"{kind} entity fields")
        for chain_id in _ids(entity):
            if chain_id in chains:
                raise ValueError(f"duplicate chain id: {chain_id!r}")
            chains.add(chain_id)

        if kind == "ligand":
            codes = entity.get("ccd")
            if isinstance(codes, list):
                if not codes or not all(
                    isinstance(code, str) and code.strip() for code in codes
                ):
                    raise ValueError(
                        "ligand ccd list must hold one or more non-empty CCD codes"
                    )
                entity["ccd"] = [code.strip() for code in codes]
            for field_name in ("ccd", "smiles"):
                field_value = entity.get(field_name)
                if field_name == "ccd" and isinstance(field_value, list):
                    continue
                if field_value is not None and (
                    not isinstance(field_value, str) or not field_value.strip()
                ):
                    raise ValueError(f"ligand {field_name} must be a non-empty string")
                if field_value is not None:
                    entity[field_name] = field_value.strip()
            if not (entity.get("ccd") or entity.get("smiles")):
                raise ValueError("ligand entity requires ccd or smiles")
            if entity.get("ccd") and entity.get("smiles"):
                raise ValueError("ligand entity accepts either ccd or smiles, not both")
            if validate_ccd is not None and isinstance(entity.get("ccd"), list):
                entity["ccd"] = [
                    validate_ccd(code, field="ligand CCD code")
                    for code in entity["ccd"]
                ]
            elif validate_ccd is not None and entity.get("ccd"):
                entity["ccd"] = validate_ccd(entity["ccd"], field="ligand CCD code")
            codes = _ccd_codes(entity)
            if len(codes) > 1 and "multi_residue_ligand" not in target.features:
                _reject(
                    model,
                    "a ligand of several CCD codes",
                    f"entity {_ids(entity)[0]!r} lists {len(codes)}, and upstream "
                    f"{model} builds a ligand chain from one code (OpenFold3 "
                    "v0.5.0 raises NotImplementedError for more)",
                )
            if model in _CCD_JOINED_MODELS and any("_" in code for code in codes):
                # build_ligand splits ``CCD_A_B`` on "_" into its components.
                _reject(
                    model,
                    "a CCD code containing '_'",
                    f"its ligand string joins codes with '_' (CCD_A_B), so "
                    f"{codes!r} would be read as other components",
                )
            for code in codes:
                _require_component(
                    code, what=f"ligand {_ids(entity)[0]!r} CCD code"
                )
            if entity.get("smiles"):
                _require_smiles(entity["smiles"], chain=_ids(entity)[0])
            feature = "ligand_ccd" if entity.get("ccd") else "ligand_smiles"
            if feature not in target.features:
                _reject(model, feature, "supply the other ligand representation")
            continue

        entity["sequence"] = _normalize_sequence(
            entity.get("sequence"), kind=kind, chain=_ids(entity)[0]
        )
        for feature in ("unpaired_msa", "paired_msa"):
            value = entity.get(feature)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{feature} must be a non-empty path string")
            if value is not None and feature not in target.features:
                _reject(model, feature, f"remove it from entity {_ids(entity)[0]!r}")
            if (
                value is not None
                and model == "boltz2"
                and Path(value.strip()).suffix.lower() == ".csv"
            ):
                # Boltz dispatches on the exact suffix: `.csv` is its paired
                # alignment format, whose rows sharing a `key` are paired
                # (`data/parse/csv.py`), so this "unpaired" alignment would be
                # paired after all -- the input refused above as paired_msa.
                # Any other spelling of the suffix it does not read at all.
                named = f"entity {_ids(entity)[0]!r} names {value.strip()!r}"
                _reject(
                    model,
                    "a .csv unpaired_msa",
                    (
                        f"{named}, and .csv is Boltz's paired-alignment format: "
                        "rows that share a key are paired. Give an .a3m here, or "
                        "a native Boltz YAML whose msa field names the CSV to "
                        "pair it"
                        if Path(value.strip()).suffix == ".csv"
                        else f"{named}, a suffix Boltz does not read as an "
                        "alignment. Give an .a3m, or a native Boltz YAML for a "
                        "paired .csv"
                    ),
                )
            if value is not None and model in _USE_RNA_MSA_MODELS and kind == "rna":
                if feature == "paired_msa":
                    _reject(
                        model,
                        "RNA paired_msa",
                        "upstream accepts only rnaSequence.unpairedMsaPath",
                    )
                if not use_rna_msa and not ignore_nucleic_msa:
                    _reject(
                        model,
                        "RNA unpaired_msa",
                        f"entity {_ids(entity)[0]!r} names {value.strip()!r}, "
                        f"but {model} reads RNA alignments only with "
                        "use_rna_msa=true, and upstream's released default is "
                        "false, which would discard it. Set --option "
                        "use_rna_msa=true to read it, or unset "
                        f"{IGNORE_NUCLEIC_MSA}=false to fold without it as "
                        "upstream does",
                    )
            paired = feature == "paired_msa"
            read = _NUCLEIC_PAIRED_MSA_READ.get(model) if paired else nucleic_msa_read
            if (
                value is not None
                and kind in ("dna", "rna")
                and read is not None
                and kind not in read
            ):
                chain_ids = _ids(entity)
                if not ignore_nucleic_msa:
                    _reject(
                        model,
                        f"a {kind.upper()} {feature}",
                        f"entity {chain_ids[0]!r} names {value.strip()!r}, but "
                        f"{model} would discard it and fold that chain "
                        f"{'without pairing' if paired else 'from its sequence alone'} "
                        f"({_nucleic_msa_readers(kind)}). "
                        f"Remove it, or unset {IGNORE_NUCLEIC_MSA}=false to run "
                        "without it as upstream does; the run manifest then "
                        "records the drop under ignored_msas",
                    )
                if ignored is not None:
                    ignored.append(
                        {
                            "chains": chain_ids,
                            "type": kind,
                            "field": feature,
                            "path": value.strip(),
                            "reason": (
                                f"{model} does not read {kind.upper()} "
                                f"{'paired ' if paired else ''}alignments; "
                                "ignored, as upstream does"
                            ),
                        }
                    )
                del entity[feature]
                continue
            if value is not None:
                entity[feature] = value.strip()
        if entity.get("modifications") and "modifications" not in target.features:
            _reject(
                model,
                "modifications",
                f"remove it from entity {_ids(entity)[0]!r}",
            )
        modifications = _modifications(entity)
        if validate_ccd is not None:
            for ccd, _ in modifications:
                validate_ccd(ccd, field="modification CCD code")
        for ccd, position in modifications:
            _require_component(
                ccd,
                what=f"modification at residue {position} of chain "
                f"{_ids(entity)[0]!r}: CCD code",
            )

        templates = _templates(entity)
        if (
            templates
            and model in _USE_TEMPLATE_MODELS
            and not use_template
            and all(template["mapping"] for template in templates)
        ):
            chain_ids = _ids(entity)
            if not ignore_templates:
                _reject(
                    model,
                    "templates in the common schema",
                    f"entity {chain_ids[0]!r} names {len(templates)} "
                    f"template(s), but {model} reads templates only with "
                    "use_template=true, and upstream's released default is "
                    "false, which would discard them. Set --option "
                    "use_template=true to read them, or unset "
                    f"{IGNORE_TEMPLATES}=false to fold without them as upstream "
                    "does; the run manifest then records the drop under "
                    "ignored_templates",
                )
            if ignored_templates is not None:
                ignored_templates.extend(
                    {
                        "chains": chain_ids,
                        "type": kind,
                        "field": "templates",
                        "path": template["mmcif"],
                        "reason": (
                            f"{model} reads templates only with "
                            "use_template=true; ignored, as upstream does"
                        ),
                    }
                    for template in templates
                )
            del entity["templates"]
            templates = []
        for template in templates:
            feature = "templates" if template["mapping"] else "templates_unmapped"
            if feature in target.features:
                continue
            if feature == "templates_unmapped" and "templates" in target.features:
                _reject(
                    model,
                    "a template without a residue map",
                    "it requires query_indices and template_indices; Boltz-2 and "
                    "OpenFold3 align a bare mmCIF themselves",
                )
            if feature == "templates" and "templates_unmapped" in target.features:
                _reject(
                    model,
                    "a template residue map",
                    "it aligns the mmCIF itself; drop query_indices and "
                    "template_indices",
                )
            _reject(
                model,
                "templates",
                "this backend has no per-job template field; use its own "
                "template pipeline",
            )
        if templates and model in _ONE_TEMPLATE_FORM:
            if kind != "protein":
                _reject(
                    model,
                    f"templates on a {kind} chain",
                    "it preprocesses templates for protein chains only "
                    "(TemplatePreprocessorSettings.moltypes)",
                )
            if len({bool(template["mapping"]) for template in templates}) > 1:
                _reject(
                    model,
                    "mapped and unmapped templates on one chain",
                    f"entity {_ids(entity)[0]!r} mixes them, and it reads one "
                    "template source per chain (a residue-mapped template cache "
                    "or bare files it aligns itself); give every template the "
                    "same form",
                )

    if base is not None:
        _require_input_files(entities, base)
    if job.get("bonds") and "bonds" not in target.features:
        _reject(
            model,
            "bonds",
            "its featurizer never applies covalent bonds, upstream or here",
        )
    # A ligand has one residue per CCD code; a SMILES ligand has one.
    _check_bond_atoms(_bonds(job, chains, _residue_counts(entities)), entities)
    if job.get("properties") and "affinity" not in target.features:
        _reject(
            model,
            "binding affinity",
            "only Boltz-2 carries an affinity head",
        )
    binder = _affinity_binder(job, chains)
    for entity in entities:
        if binder in _ids(entity) and len(_ccd_codes(entity)) > 1:
            # data/parse/schema.py:1190-1192 "Cannot compute affinity for
            # multi residue ligands!", raised while parsing; refused here first.
            _reject(
                model,
                "binding affinity for a ligand of several CCD codes",
                f"upstream Boltz-2 cannot compute affinity for the multi-residue "
                f"ligand {binder!r}",
            )
    _validate_pocket_constraints(
        job,
        model,
        target,
        entities,
        options,
        ignored_constraints,
        pocket_conditioning=pocket_conditioning,
        selection_pockets=selection_pockets,
    )


def pocket_conditioned(model: str) -> bool:
    """Whether ``model``'s native input conditions on a common pocket.

    The translation table's answer; a backend whose checkpoint decides
    (Protenix) refines it in `Backend.pocket_conditioning`.
    """
    target = _TARGETS.get(model)
    return target is not None and "pocket_constraints" in target.features


def _validate_pocket_constraints(
    job: dict[str, Any],
    model: str,
    target: _Target,
    entities: list[dict[str, Any]],
    options: Mapping[str, Any],
    ignored_constraints: list[dict[str, Any]] | None,
    *,
    pocket_conditioning: bool | None = None,
    selection_pockets: list[dict[str, Any]] | None = None,
) -> None:
    """Check the job's pocket and contact restraints against what ``model`` reads.

    OpenDDE's upstream ignores a constraint, so there the field is removed
    from ``job`` and recorded, as a native OpenDDE constraint is; every other
    backend either carries it or refuses it.

    Under ``pocket_sampling=select`` (`foldjax.pocket_selection`) a pocket the
    native input cannot condition on -- ``pocket_conditioning`` false; None
    asks the translation table -- is taken out of the job for the selection
    alone and appended to ``selection_pockets`` instead of being refused or
    dropped. Contacts follow the model's own rules either way.
    """
    from foldjax.pocket_selection import POCKET_SAMPLING, requested

    kinds = {chain: entity["type"] for entity in entities for chain in _ids(entity)}
    lengths = _residue_counts(entities)
    pockets = _pocket_constraints(job, kinds, lengths)
    contacts = _contact_constraints(job, kinds, lengths)
    if not pockets and not contacts:
        return
    # Ahead of ``pocket_sampling=select``'s carve-out, which could otherwise
    # take a forced pocket out of the job silently (as one the native input
    # cannot condition on at all) rather than refuse what force asked for.
    # Gated on the model actually having the field (`target.features`): a
    # model with none reports that instead, in `_validate_pockets` or
    # `_validate_contacts` below.
    if model != "boltz2" and "pocket_constraints" in target.features:
        for pocket in pockets:
            if pocket["force"]:
                _reject(
                    model,
                    "a pocket constraint with force",
                    "only Boltz-2's native pocket field reads force and "
                    "steers its sampler toward it (parse/schema.py:1607); "
                    f"upstream {model} has no such potential",
                )
    if model != "boltz2" and "contact_constraints" in target.features:
        for contact in contacts:
            if contact["force"]:
                _reject(
                    model,
                    "a contact constraint with force",
                    "only Boltz-2's native contact field reads force and "
                    "steers its sampler toward it (parse/schema.py:1639); "
                    f"upstream {model} has no such potential",
                )
    if pockets and requested(options) == "select":
        conditioned = (
            pocket_conditioned(model)
            if pocket_conditioning is None
            else bool(pocket_conditioning)
        )
        if not conditioned:
            for pocket in pockets:
                if pocket["max_distance"] is None:
                    _reject(
                        model,
                        "a pocket constraint without max_distance",
                        f"{POCKET_SAMPLING}=select scores its samples against the "
                        f"pocket and upstream {model} has no pocket distance of "
                        "its own to fall back on; set max_distance",
                    )
            if selection_pockets is not None:
                selection_pockets.extend(pockets)
            job["constraints"] = [
                item
                for item in job["constraints"]
                if not (isinstance(item, dict) and "pocket" in item)
            ]
            if not job["constraints"]:
                del job["constraints"]
            pockets = []
            if not contacts:
                return
    if model in _IGNORED_CONSTRAINT_MODELS:
        if not _strict_boolean(
            options.get(IGNORE_CONSTRAINTS, True), name=IGNORE_CONSTRAINTS
        ):
            _reject(
                model,
                " and ".join(
                    f"a {kind} constraint"
                    for kind, present in (("pocket", pockets), ("contact", contacts))
                    if present
                ),
                f"upstream {model}'s inference build ignores constraints. "
                f"Remove them, or unset {IGNORE_CONSTRAINTS}=false to run without "
                "them as upstream does; the run manifest then records the drop "
                "under ignored_constraints",
            )
        if ignored_constraints is not None:
            record: dict[str, Any] = {
                "job": str(job.get("name") or 0),
                "field": "constraints",
                "keys": [
                    kind
                    for kind, present in (("pocket", pockets), ("contact", contacts))
                    if present
                ],
                "binders": [pocket["binder"] for pocket in pockets],
                "reason": (
                    f"{model} reads no constraint at inference (upstream's "
                    "inference build warns and ignores it); ignored, as "
                    "upstream does"
                ),
            }
            if contacts:
                record["contacts"] = [
                    [list(contact["token1"]), list(contact["token2"])]
                    for contact in contacts
                ]
            ignored_constraints.append(record)
        del job["constraints"]
        return
    if pockets:
        _validate_pockets(model, target, pockets, kinds)
    if contacts:
        _validate_contacts(model, target, contacts, entities, kinds)


def _validate_pockets(
    model: str,
    target: _Target,
    pockets: list[dict[str, Any]],
    kinds: Mapping[str, str],
) -> None:
    if "pocket_constraints" not in target.features:
        _reject(
            model,
            "a pocket constraint",
            f"upstream {model} has no pocket or restraint field",
        )
    if model in ("openfold3", "protenix") and len(pockets) > 1:
        _reject(
            model,
            "more than one pocket constraint",
            "its query takes a single pocket",
        )
    if model == "openfold3" and kinds[pockets[0]["binder"]] != "ligand":
        # inference_query_format.py Query.validate_pocket_constraint
        _reject(
            model,
            "a pocket constraint on a polymer binder",
            "its pocket_constraint names a ligand chain",
        )
    if model == "protenix" and pockets[0]["max_distance"] is None:
        _reject(
            model,
            "a pocket constraint without max_distance",
            "upstream Protenix requires max_distance and has no default; set it",
        )


#: Why a model cannot address a ligand residue as a contact token.
_LIGAND_CONTACT_REASONS = {
    # data/parse/schema.py token_spec_to_ids: a NONPOLYMER chain is addressed
    # by the atom name of its first residue, never by residue index.
    "boltz2": (
        "upstream Boltz-2 addresses a ligand in a contact by atom name, not "
        "by residue; write the native Boltz-2 contact instead"
    ),
    # constraint_featurizer.py ContactFeaturizer.generate_spec_constraint
    # draws one token of a multi-token residue with torch.randint on the
    # global generator; a ligand residue is one token per atom.
    "protenix": (
        "upstream Protenix draws one random atom token of a ligand residue "
        "for a residue contact (torch.randint), which no common job can "
        "reproduce; write the native Protenix contact with an atom instead"
    ),
}


def _validate_contacts(
    model: str,
    target: _Target,
    contacts: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    kinds: Mapping[str, str],
) -> None:
    if "contact_constraints" not in target.features:
        _reject(
            model,
            "a contact constraint",
            f"upstream {model} has no contact restraint field",
        )
    modified = {
        (chain, position)
        for entity in entities
        if entity["type"] != "ligand"
        for _, position in _modifications(entity)
        for chain in _ids(entity)
    }
    for contact in contacts:
        for token in (contact["token1"], contact["token2"]):
            if kinds[token[0]] == "ligand":
                _reject(
                    model,
                    f"a contact on ligand {token[0]!r}",
                    _LIGAND_CONTACT_REASONS[model],
                )
            if model == "protenix" and token in modified:
                # Same draw as a ligand: a modified residue is atom-tokenized.
                _reject(
                    model,
                    f"a contact on modified residue {token[0]}:{token[1]}",
                    "upstream Protenix draws one random atom token of a "
                    "modified residue (torch.randint), which no common job "
                    "can reproduce",
                )
        if model == "protenix" and contact["token1"][0] == contact["token2"][0]:
            # constraint_featurizer.py _canonicalize_contact_format
            _reject(
                model,
                "a contact within one chain",
                "upstream Protenix refuses a contact pair on the same chain",
            )
        if model == "protenix" and contact["max_distance"] is None:
            _reject(
                model,
                "a contact constraint without max_distance",
                "upstream Protenix requires max_distance and has no default; set it",
            )


def _filter_alphafold3_template_mmcif(source: Path, chain_id: str) -> str:
    """Return an mmCIF containing the selected author polymer chain.

    AlphaFold 3's input dialect has no template-chain field. Its featurizer
    instead requires each template file to contain exactly one polymer chain,
    so preserving a common-schema ``chain_id`` means filtering the structure
    before the native document is written.
    """

    from foldjax.models.alphafold3 import build

    build.register_runtime()
    from alphafold3 import structure
    from alphafold3.constants import mmcif_names

    parsed = structure.from_mmcif(source.read_bytes(), include_other=True)
    selected = parsed.filter(chain_auth_asym_id=chain_id)
    polymer_count = sum(
        chain_type in mmcif_names.POLYMER_CHAIN_TYPES
        for chain_type in selected.chains_table.type
    )
    if polymer_count != 1:
        available = sorted(set(map(str, parsed.chains_table.auth_asym_id)))
        raise ValueError(
            f"AlphaFold 3 template chain_id {chain_id!r} selected "
            f"{polymer_count} polymer chains in {source}; available author "
            f"chain IDs: {available}"
        )
    return selected.to_mmcif()


def _alphafold3_template_path(
    template: dict[str, Any],
    base: Path,
    destination: Path,
    entity_index: int,
    template_index: int,
) -> str:
    source = Path(_path(template["mmcif"], base))
    chain_id = template["chain_id"]
    if chain_id is None:
        return str(source)

    directory = destination / "templates"
    if directory.is_symlink():
        raise ValueError(f"generated template directory is a symlink: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (
        f"entity_{entity_index:04d}_template_{template_index:04d}.cif"
    )
    _write_text_atomic(
        target,
        _filter_alphafold3_template_mmcif(source, chain_id),
    )
    return str(target)


def _alphafold3(
    job: dict[str, Any], base: Path, *, seed: int, destination: Path
) -> dict[str, Any]:
    sequences = []
    for entity_index, entity in enumerate(job["entities"]):
        kind = entity["type"]
        body: dict[str, Any] = {"id": _ids(entity)}
        if kind == "ligand":
            if entity.get("ccd"):
                # Bonds address residue i of the ligand as code i - 1
                # (docs/input.md "Defining Glycans").
                body["ccdCodes"] = _ccd_codes(entity)
            else:
                body["smiles"] = str(entity["smiles"])
        else:
            body["sequence"] = str(entity["sequence"])
            unpaired = entity.get("unpaired_msa")
            paired = entity.get("paired_msa")
            if unpaired:
                body["unpairedMsaPath"] = _path(unpaired, base)
            if paired:
                body["pairedMsaPath"] = _path(paired, base)
            # AlphaFold 3 refuses to featurise a protein or RNA chain whose MSA
            # is *absent*: it assumes its own genetic-search pipeline will fill
            # it in, which needs hundreds of gigabytes of databases. An **empty**
            # MSA is a different thing and is accepted -- it means single
            # sequence, which is what the other backends already fall back to
            # when a job carries no alignment. Say so explicitly rather than
            # failing a job the other MSA-capable models accept.
            if kind in ("protein", "rna") and not unpaired:
                body["unpairedMsa"] = ""
            if kind == "protein":
                if not paired:
                    body["pairedMsa"] = ""
                body.setdefault("templates", [])
            # The alphafold3 dialect strips a CCD_ prefix, so bare codes are
            # emitted to match AlphaFold 3's own serialization.
            modifications = _modifications(entity)
            if kind == "protein":
                body["modifications"] = [
                    {"ptmType": ccd, "ptmPosition": position}
                    for ccd, position in modifications
                ]
            elif modifications:
                body["modifications"] = [
                    {"modificationType": ccd, "basePosition": position}
                    for ccd, position in modifications
                ]
            templates = _templates(entity)
            if templates:
                # AlphaFold 3 reads the file itself and wants the residue map
                # split into two parallel lists (`folding_input.py:389`).
                body["templates"] = [
                    {
                        "mmcifPath": _alphafold3_template_path(
                            template,
                            base,
                            destination,
                            entity_index,
                            template_index,
                        ),
                        "queryIndices": [pair[0] for pair in template["mapping"]],
                        "templateIndices": [pair[1] for pair in template["mapping"]],
                    }
                    for template_index, template in enumerate(templates)
                ]
        sequences.append({kind: body})
    native: dict[str, Any] = {
        "name": str(job.get("name", "foldjax_job")),
        "modelSeeds": [seed],
        "sequences": sequences,
        "dialect": "alphafold3",
        "version": 4,
    }
    bonds = _bonds(job, set())
    if bonds:
        native["bondedAtomPairs"] = [[list(left), list(right)] for left, right in bonds]
    return native


#: Boltz's ``const.max_paired_seqs`` and ``const.max_msa_seqs``, the two caps
#: its server search writes a CSV under.
_BOLTZ_MAX_PAIRED_SEQS = 8192
_BOLTZ_MAX_MSA_SEQS = 16384


def _a3m_rows(text: str) -> list[str]:
    """An A3M's sequences in order, a record's continuation lines joined.

    Boltz reads ColabFold's output as strict header/sequence line pairs
    (``lines[1::2]``); reading by header gives the same rows for that output
    and does not shift on a wrapped sequence.
    """
    rows: list[str] = []
    current: list[str] | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current is not None:
                rows.append("".join(current))
            current = []
        elif current is not None:
            current.append(line)
    if current is not None:
        rows.append("".join(current))
    return rows


def row_species_a3m(paired: str) -> str:
    """One chain's block of a complex pairing search, as Protenix's ColabFold
    mode writes its ``pairing.a3m``.

    Upstream (web_service/colab_request_utils.py:313-345) names the query
    ``>query`` and suffixes every hit's accession with its row number,
    ``>UniRef100_<accession>_<row>/<rest>``, so the species its featurizer
    reads (``_UNIREF_REGEX``, ``^UniRef100_[^_]+_([^_/]+)``) is the row: rows
    with one number are paired across chains, which is how the server aligned
    them. Upstream's runner never reads that file in ColabFold mode (it is
    written under ``msa/complex/``, runner/msa_search.py:177-185), so this is
    FoldJAX's ``greedy``/``complete`` opt-in, not a reproduction of a run.

    Two departures. Upstream numbers the entries of a dict keyed by header
    text (``parse_fasta_string``): a repeated header keeps its first slot
    with the later record's sequence and drops a row, so every later row's
    number shifts down by one and pairs with a different row of the other
    chains than the server aligned. Here the number is the record's
    position, which keeps the server's alignment; on a block with a repeated
    header the two pair different rows. And a header without the
    ``UniRef100_`` prefix gets it, with ``_`` and ``/`` in the accession
    replaced so the regex cannot read another field.
    """
    out: list[str] = []
    for index, (header, row) in enumerate(_a3m_records(paired)):
        if index == 0:
            out.append(f">query\n{row}\n")
            continue
        accession, tab, rest = header.partition("\t")
        accession = accession.split()[0] if accession.split() else ""
        accession = accession.removeprefix("UniRef100_")
        accession = accession.replace("_", "-").replace("/", "-") or "hit"
        out.append(f">UniRef100_{accession}_{index}/{tab}{rest}\n{row}\n")
    return "".join(out)


def _a3m_records(text: str) -> list[tuple[str, str]]:
    """An A3M's (header without ``>``, sequence) records, as `_a3m_rows` reads them."""
    records: list[tuple[str, str]] = []
    header: str | None = None
    current: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(current)))
            header, current = line[1:], []
        elif header is not None:
            current.append(line)
    if header is not None:
        records.append((header, "".join(current)))
    return records


def _protenix_row_paired(
    job: dict[str, Any], base: Path, destination: Path
) -> dict[int, str]:
    """Write each searched complex block as `row_species_a3m`; entity index -> path.

    The search refused blocks of different depths before caching them
    (`MsaSearchPipeline._complex_materialize`), so row *i* is one paired row.
    """
    from foldjax.msa_search import ROW_PAIRED_MSA

    texts = {
        index: Path(_path(entity["paired_msa"], base)).read_text(encoding="utf-8")
        for index, entity in enumerate(job["entities"])
        if entity.get(ROW_PAIRED_MSA) and entity.get("paired_msa")
    }
    if not texts:
        return {}
    msa_root = _generated_msa_root(destination)
    paths: dict[int, str] = {}
    for index, text in texts.items():
        target = msa_root / f"entity_{index:04d}_pairing.a3m"
        _write_text_atomic(target, row_species_a3m(text))
        paths[index] = str(target.resolve())
    return paths


def env_first_a3m(unpaired: str) -> str | None:
    """A ColabFold unpaired alignment in the order Protenix's ColabFold mode
    writes its ``non_pairing.a3m``, or None when it is not one.

    The search caches the server's ``uniref.a3m`` block and then its
    ``bfd.mgnify30.metaeuk30.smag30.a3m`` block, each led by the query record.
    Upstream (web_service/colab_request_utils.py:290-310) writes ``>query``,
    then the environmental hits, then the UniRef hits. Anything that does not
    split into exactly those two query-led blocks is left as it is.

    One departure: upstream reads each file into a dict keyed by header text
    (``parse_fasta_string``), so a repeated header keeps its first slot with
    the later record's sequence and the earlier sequence is dropped. Every
    record is kept here; Protenix's featurizer drops repeated *sequences*
    either way.
    """
    records = _a3m_records(unpaired)
    if not records:
        return None
    query = records[0]
    starts = [index for index, record in enumerate(records) if record == query]
    if len(starts) != 2:
        return None
    uniref, env = records[1 : starts[1]], records[starts[1] + 1 :]
    return f">query\n{query[1]}\n" + "".join(
        f">{header}\n{row}\n" for header, row in (*env, *uniref)
    )


def _protenix_env_first(
    job: dict[str, Any], base: Path, destination: Path
) -> dict[int, str]:
    """Write each searched unpaired alignment as `env_first_a3m`; index -> path."""
    from foldjax.msa_search import ENV_FIRST_UNPAIRED_MSA

    paths: dict[int, str] = {}
    for index, entity in enumerate(job["entities"]):
        if not (entity.get(ENV_FIRST_UNPAIRED_MSA) and entity.get("unpaired_msa")):
            continue
        text = Path(_path(entity["unpaired_msa"], base)).read_text(encoding="utf-8")
        reordered = env_first_a3m(text)
        if reordered is None:
            continue
        msa_root = _generated_msa_root(destination)
        target = msa_root / f"entity_{index:04d}_non_pairing.a3m"
        _write_text_atomic(target, reordered)
        paths[index] = str(target.resolve())
    return paths


def _generated_msa_root(destination: Path) -> Path:
    """``destination/msa``, refused when a symlink could send it elsewhere."""
    msa_root = destination / "msa"
    if msa_root.is_symlink():
        raise ValueError(f"generated MSA directory is a symlink: {msa_root}")
    msa_root.mkdir(parents=True, exist_ok=True)
    if not msa_root.resolve().is_relative_to(destination.resolve()):
        raise ValueError(f"generated MSA directory escapes output root: {msa_root}")
    return msa_root


def boltz_server_msa_csv(paired: str, unpaired: str) -> str:
    """Upstream Boltz's per-entity CSV from a paired and an unpaired A3M.

    The body of ``compute_msa`` (``boltz/main.py:496-522``): the paired rows
    first, capped at 8,192, all-gap rows dropped and the rest keyed by their
    row number so rows sharing a key across entities are paired; then the
    unpaired rows, capped so the total stays at 16,384, minus the query when a
    paired block already holds it, keyed ``-1``.
    """
    paired_rows = _a3m_rows(paired)[:_BOLTZ_MAX_PAIRED_SEQS]
    keys = [index for index, row in enumerate(paired_rows) if row != "-" * len(row)]
    paired_rows = [row for row in paired_rows if row != "-" * len(row)]
    unpaired_rows = _a3m_rows(unpaired)[: _BOLTZ_MAX_MSA_SEQS - len(paired_rows)]
    if paired_rows:
        unpaired_rows = unpaired_rows[1:]
    rows = paired_rows + unpaired_rows
    keys = keys + [-1] * len(unpaired_rows)
    return "\n".join(
        ["key,sequence"] + [f"{key},{row}" for key, row in zip(keys, rows, strict=True)]
    )


def _boltz_server_csv(
    entity: Mapping[str, Any], base: Path, destination: Path, entity_index: int
) -> str:
    """Write one entity's searched pair as the CSV Boltz reads; return its path."""
    paired = Path(_path(entity["paired_msa"], base)).read_text(encoding="utf-8")
    unpaired = Path(_path(entity["unpaired_msa"], base)).read_text(encoding="utf-8")
    # Positional, like OpenFold3's links: chain ids are document data, not
    # path components.
    msa_root = destination / "msa"
    if msa_root.is_symlink():
        raise ValueError(f"generated MSA directory is a symlink: {msa_root}")
    msa_root.mkdir(parents=True, exist_ok=True)
    if not msa_root.resolve().is_relative_to(destination.resolve()):
        raise ValueError(f"generated MSA directory escapes output root: {msa_root}")
    target = msa_root / f"entity_{entity_index:04d}.csv"
    _write_text_atomic(target, boltz_server_msa_csv(paired, unpaired))
    return str(target.resolve())


def _boltz(
    job: dict[str, Any],
    base: Path,
    *,
    seed: int = 0,
    destination: Path | None = None,
) -> dict[str, Any]:
    sequences = []
    server_csvs: dict[str, str] = {}
    for entity_index, entity in enumerate(job["entities"]):
        kind = entity["type"]
        body: dict[str, Any] = {"id": _ids(entity)}
        if kind == "ligand":
            if entity.get("ccd"):
                # A list is one multi-residue chain (parse/schema.py
                # parse_boltz_schema, residue index = list index).
                codes = _ccd_codes(entity)
                body["ccd"] = codes[0] if isinstance(entity["ccd"], str) else codes
            else:
                body["smiles"] = str(entity["smiles"])
        else:
            body["sequence"] = str(entity["sequence"])
            if entity.get("unpaired_msa") and entity.get("paired_msa"):
                # Only `msa='auto'` attaches a paired alignment here (the
                # common schema refuses one for Boltz-2); Boltz reads the pair
                # as one CSV whose paired rows share a key.
                if destination is None:
                    raise ValueError(
                        "a paired Boltz-2 alignment needs a destination directory"
                    )
                # One CSV per sequence, named after the first entity carrying
                # it: Boltz refuses two chains of one sequence naming two
                # alignments ("All proteins with the same sequence must share
                # the same MSA!", parse/schema.py), and upstream's search
                # writes one per sequence too.
                sequence = str(entity["sequence"])
                if sequence not in server_csvs:
                    server_csvs[sequence] = _boltz_server_csv(
                        entity, base, destination, entity_index
                    )
                body["msa"] = server_csvs[sequence]
            elif entity.get("unpaired_msa"):
                body["msa"] = _path(entity["unpaired_msa"], base)
            elif kind == "protein":
                # Boltz refuses a protein chain with neither an alignment nor an
                # explicit opt-out: "Missing MSA's in input and --use-msa-server not
                # set. Use `msa: empty` per protein chain". Omitting the key is not
                # the same as asking for single-sequence prediction, so the document
                # has to say so. Nucleic acid chains take no MSA and must not carry
                # the key at all.
                body["msa"] = "empty"
            modifications = _modifications(entity)
            if modifications:
                body["modifications"] = [
                    {"ccd": ccd, "position": position}
                    for ccd, position in modifications
                ]
        sequences.append({kind: body})
    native: dict[str, Any] = {"version": 1, "sequences": sequences}
    bonds = _bonds(job, set())
    if bonds:
        native["constraints"] = [
            {"bond": {"atom1": list(left), "atom2": list(right)}}
            for left, right in bonds
        ]
    # Boltz's own pocket form (`parse/schema.py:1561-1595`): polymer contacts
    # are [chain, 1-based residue]. ``force`` (upstream's own field, default
    # off) is carried through when the job asks for it; omitted otherwise, so
    # a job that leaves it off writes the same document it always has.
    for pocket in _pocket_constraints(job):
        body: dict[str, Any] = {
            "binder": pocket["binder"],
            "contacts": [list(contact) for contact in pocket["contacts"]],
            "max_distance": _pocket_max_distance("boltz2", pocket),
        }
        if pocket["force"]:
            body["force"] = True
        native.setdefault("constraints", []).append({"pocket": body})
    # And its contact form (`parse/schema.py:1562-1597`): validation leaves
    # polymer tokens, [chain, 1-based residue]; ``force`` as the pocket form.
    for contact in _contact_constraints(job):
        body = {
            "token1": list(contact["token1"]),
            "token2": list(contact["token2"]),
            "max_distance": _contact_max_distance("boltz2", contact),
        }
        if contact["force"]:
            body["force"] = True
        native.setdefault("constraints", []).append({"contact": body})
    # Boltz keeps templates at the top level and aligns each one itself
    # (`parse/schema.py:1633`), so there is no residue map to carry across.
    templates = []
    for entity in job["entities"]:
        query_chain_ids = _ids(entity)
        for template in _templates(entity):
            entry: dict[str, Any] = {
                "cif": _path(template["mmcif"], base),
                # Boltz stores templates at the top level, so scope this one
                # back to the common-schema entity it came from.
                "chain_id": query_chain_ids,
            }
            if template["chain_id"] is not None:
                # Its ``template_id`` is the source-structure chain, while
                # ``chain_id`` above names query chains. One entity can denote
                # several copies, each using the same source chain. Boltz names
                # a structure's chains by ``label_asym_id``
                # (`parse/mmcif.py` subchains), the common field by author id.
                label = _template_label_chain(Path(entry["cif"]), template["chain_id"])
                entry["template_id"] = [label] * len(query_chain_ids)
            templates.append(entry)
    if templates:
        native["templates"] = templates
    binder = _affinity_binder(job, set())
    if binder is not None:
        native["properties"] = [{"affinity": {"binder": binder}}]
    return native


def _template_label_chain(path: Path, chain_id: str) -> str:
    """The ``label_asym_id`` of the protein chain ``chain_id`` names in ``path``.

    The common ``chain_id`` is an author chain; Boltz-2 and OpenFold3 address
    a structure's chains by label id. An id that is already a label id of a
    protein chain is kept, and so is one the file does not resolve, so the
    backend's own parser reports it in its own words.
    """
    from foldjax.template_search import read_template_structure

    try:
        structure = read_template_structure(path)
    except (OSError, ValueError):
        return chain_id
    chain = structure.chain(chain_id)
    if chain is None:
        return chain_id
    if chain.label_id != chain_id and chain_id in structure.chains:
        # The id names two different chains depending on how it is read. The
        # common field is the author id, so that reading wins -- but a caller
        # who meant the label id gets another chain, so say so.
        import warnings

        warnings.warn(
            f"template {path}: chain_id {chain_id!r} is author chain "
            f"{chain_id!r} (label {chain.label_id!r}) and also the label id of "
            f"another chain; using the author chain, as the common schema "
            f"defines chain_id",
            UserWarning,
            stacklevel=3,
        )
    return chain.label_id


def _openfold3_templates(
    templates: list[dict[str, Any]],
    base: Path,
    destination: Path,
    entity_index: int,
) -> dict[str, Any]:
    """One chain's templates as fields of OpenFold3's native query.

    Bare files go to ``template_cif_paths``, which OpenFold3 aligns and ranks
    itself (upstream's CIF-direct mode). A residue map has no field of its own
    in the query, so it is written as the template cache the reader loads from
    ``template_alignment_file_path``: one entry per template, keyed
    ``<entry>_<label chain>`` and listed in order in
    ``template_entry_chain_ids``, whose ``idx_map`` pairs 1-based query
    positions with the template residues' ``label_seq_id`` -- the form
    upstream's preprocessor writes (``build_residue_idx_map``) and the port
    reads (`_preprocessed_template_entries`). The reader keeps the first four
    that resolve, as upstream's inference featurizer takes the top four.
    """
    from foldjax.template_search import entry_name, read_template_structure

    if not templates[0]["mapping"]:
        paths = [Path(_path(template["mmcif"], base)) for template in templates]
        fields: dict[str, Any] = {"template_cif_paths": [str(path) for path in paths]}
        if any(template["chain_id"] is not None for template in templates):
            fields["template_cif_chain_ids"] = [
                None
                if template["chain_id"] is None
                else _template_label_chain(path, template["chain_id"])
                for path, template in zip(paths, templates, strict=True)
            ]
        return fields

    entries: dict[str, dict[str, Any]] = {}
    for index, template in enumerate(templates):
        path = Path(_path(template["mmcif"], base))
        structure = read_template_structure(path)
        chain = structure.chain(template["chain_id"])
        if chain is None:
            named = template["chain_id"]
            raise ValueError(
                f"openfold3 template {path}: "
                + (
                    f"chain {named!r} is not a protein chain of it"
                    if named is not None
                    else "it has more than one protein chain; name one with chain_id"
                )
            )
        pairs = []
        for query_index, template_index in template["mapping"]:
            if template_index >= len(chain.numbers):
                raise ValueError(
                    f"openfold3 template {path}: template index {template_index} "
                    f"is outside chain {chain.author_id!r}'s "
                    f"{len(chain.numbers)} residues"
                )
            pairs.append([query_index + 1, chain.numbers[template_index]])
        name = f"{entry_name(structure, f't{index}')}_{chain.label_id}"
        if name in entries:
            name = f"t{index}_{chain.label_id}"
        entries[name] = {
            "idx_map": pairs,
            "release_date": (
                structure.release_date.isoformat() if structure.release_date else ""
            ),
            "cif_path": str(path),
        }
    directory = destination / "templates"
    if directory.is_symlink():
        raise ValueError(f"generated template directory is a symlink: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"entity_{entity_index:04d}.npz"
    import numpy as np

    with tempfile.TemporaryDirectory(
        prefix=".foldjax-templates-", dir=directory
    ) as scratch:
        staged = Path(scratch) / target.name
        with staged.open("wb") as handle:
            np.savez(handle, entries_json=np.array(json.dumps(entries)))
        os.replace(staged, target)
    return {
        "template_alignment_file_path": str(target),
        "template_entry_chain_ids": list(entries),
    }


def _protenix_templates(
    templates: list[dict[str, Any]],
    base: Path,
    destination: Path,
    entity_index: int,
) -> str:
    """Write one chain's templates as the sidecar JSON Protenix reads.

    Protenix takes a per-chain ``templatesPath`` pointing at a JSON list whose
    entries hold the mmCIF *contents* rather than a path
    (`template_features.py:328`). The structure is inlined here, into a file of
    its own, so the generated job document stays readable at the size a
    multi-megabyte mmCIF would otherwise give it.
    """
    payload = []
    for template in templates:
        path = Path(_path(template["mmcif"], base))
        mmcif, mapping = _protenix_observed_template(
            path.read_text(encoding="utf-8"),
            template["chain_id"],
            template["mapping"],
            path,
        )
        # The three keys upstream's `parse_json_templates` reads; the named
        # chain is already first in `mmcif`, which is how both readers pick it.
        payload.append(
            {
                "mmcif": mmcif,
                "queryIndices": [pair[0] for pair in mapping],
                "templateIndices": [pair[1] for pair in mapping],
            }
        )
    directory = destination / "templates"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"entity_{entity_index:04d}.json"
    _write_text_atomic(target, json.dumps(payload))
    return str(target)


def _protenix_observed_template(
    text: str,
    chain_id: str | None,
    mapping: list[tuple[int, int]],
    source: Path,
) -> tuple[str, list[tuple[int, int]]]:
    """Restate a common template map in the indices Protenix and OpenDDE read.

    The common ``template_indices`` are AlphaFold 3's: 0-based over the named
    author chain's full ``_entity_poly_seq``, unresolved residues included.
    Protenix's mapped-JSON reader (upstream ``parse_simple_cif``; port
    ``template_features.py:413-416``) instead takes the file's *first* chain,
    after merging its parts, and counts its observed residues, ignoring any
    chain id. So the named chain is moved first -- the file is passed verbatim
    when it already is -- and each template index becomes that residue's
    ordinal in the chain Protenix will read. A pair whose template residue is
    unresolved is dropped: Protenix cannot address it, and it carries no
    coordinates under AlphaFold 3 either. With no ``chain_id`` the first chain
    is the template, as Protenix reads it; a file with no ``label_seq_id`` has
    no full sequence to count, and its map is passed through.
    """
    import gemmi

    def parse(content: str) -> Any:
        try:
            structure = gemmi.make_structure_from_block(
                gemmi.cif.read_string(content)[0]
            )
        except (RuntimeError, ValueError, IndexError) as error:
            raise ValueError(f"cannot read template mmCIF {source}: {error}") from None
        structure.merge_chain_parts()
        return structure

    structure = parse(text)
    names = [chain.name for chain in structure[0]] if len(structure) else []
    if not names:
        raise ValueError(f"template mmCIF {source} has no chains")
    selected = names[0] if chain_id is None else chain_id
    if selected not in names:
        raise ValueError(
            f"template chain_id {chain_id!r} is not an author chain of {source}; "
            f"available: {sorted(set(names))}"
        )
    original = text
    if selected != names[0]:
        for model in structure:
            for name in {name for name in names if name != selected}:
                while model.find_chain(name) is not None:
                    model.remove_chain(name)
        text = structure.make_mmcif_document().as_string()
        structure = parse(text)
    residues = list(structure[0][0])
    polymer = [residue for residue in residues if residue.label_seq is not None]
    if not polymer:
        return text, list(mapping)

    block = gemmi.cif.read_string(original)[0]
    # Atoms need not carry label_entity_id; the chain's _struct_asym row then
    # names its entity, as `template_search._read_template_structure` reads it.
    asym_entity = {
        row.str(0): row.str(1)
        for row in block.find("_struct_asym.", ["id", "entity_id"])
    }

    def entity_of(residue: Any) -> str:
        return residue.entity_id or asym_entity.get(residue.subchain, "")

    entity = entity_of(polymer[0])
    rows = list(block.find("_entity_poly_seq.", ["entity_id", "num"]))
    numbers: dict[int, int] = {}
    for row in rows:
        if row.str(0) == entity:
            numbers.setdefault(int(row.str(1)), len(numbers))
    if rows and not numbers:
        raise ValueError(
            f"template chain {selected!r} of {source} has no _entity_poly_seq rows "
            f"(entity {entity or 'unknown'!r}), so its template indices have no "
            "full sequence to count"
        )
    if not numbers:
        # No declared sequence at all: what the chain resolves is all of it.
        for number in sorted({residue.label_seq for residue in polymer}):
            numbers[number] = len(numbers)
    ordinals: dict[int, int] = {}
    for ordinal, residue in enumerate(residues):
        if residue.label_seq is None or entity_of(residue) != entity:
            continue
        position = numbers.get(residue.label_seq)
        if position is not None:
            ordinals.setdefault(position, ordinal)
    converted = []
    for query, position in mapping:
        if not 0 <= position < len(numbers):
            raise ValueError(
                f"template index {position} is outside chain {selected!r} of "
                f"{source}, whose sequence has {len(numbers)} residues; template "
                "indices are 0-based over its _entity_poly_seq"
            )
        if position in ordinals:
            converted.append((query, ordinals[position]))
    return text, converted


def _protenix(
    job: dict[str, Any], base: Path, *, seed: int, destination: Path
) -> list[dict[str, Any]]:
    sequences = []
    # Protenix addresses covalent bonds by 1-based entity number and copy index
    # rather than by chain id, and derives both from this sequences list.
    endpoints: dict[str, tuple[int, int]] = {}
    # A searched complex block is written as upstream's ColabFold mode writes
    # it, so its rows pair by number; a caller's paired_msa passes untouched.
    row_paired = _protenix_row_paired(job, base, destination)
    # A searched unpaired alignment, for Protenix, environmental hits first as
    # its ColabFold mode writes it; OpenDDE's upstream keeps the server order.
    env_first = _protenix_env_first(job, base, destination)
    for entity_number, entity in enumerate(job["entities"], start=1):
        kind = entity["type"]
        ids = _ids(entity)
        for copy_id, chain_id in enumerate(ids, start=1):
            endpoints[chain_id] = (entity_number, copy_id)
        body: dict[str, Any] = {"id": ids, "count": len(ids)}
        if kind == "ligand":
            # Several codes join as CCD_A_B; upstream numbers them res_id 1, 2,
            # ..., which covalent_bonds' position addresses.
            body["ligand"] = (
                f"CCD_{'_'.join(_ccd_codes(entity))}"
                if entity.get("ccd")
                else str(entity["smiles"])
            )
        else:
            body["sequence"] = str(entity["sequence"])
            if entity_number - 1 in env_first:
                body["unpairedMsaPath"] = env_first[entity_number - 1]
            elif entity.get("unpaired_msa"):
                body["unpairedMsaPath"] = _path(entity["unpaired_msa"], base)
            if entity_number - 1 in row_paired:
                body["pairedMsaPath"] = row_paired[entity_number - 1]
            elif entity.get("paired_msa"):
                body["pairedMsaPath"] = _path(entity["paired_msa"], base)
            # Protenix requires the CCD_ prefix on modification types.
            modifications = _modifications(entity)
            if kind == "protein":
                body["modifications"] = [
                    {"ptmType": f"CCD_{ccd}", "ptmPosition": position}
                    for ccd, position in modifications
                ]
            elif modifications:
                body["modifications"] = [
                    {"modificationType": f"CCD_{ccd}", "basePosition": position}
                    for ccd, position in modifications
                ]
            templates = _templates(entity)
            if templates:
                body["templatesPath"] = _protenix_templates(
                    templates, base, destination, entity_number - 1
                )
        sequences.append({_PROTENIX_ENTITY_NAMES[kind]: body})
    native: dict[str, Any] = {
        "name": str(job.get("name", "foldjax_job")),
        "modelSeeds": [seed],
        "sequences": sequences,
    }
    bonds = _bonds(job, set(endpoints))
    if bonds:
        native["covalent_bonds"] = [
            {
                "entity1": endpoints[left[0]][0],
                "copy1": endpoints[left[0]][1],
                "position1": left[1],
                "atom1": left[2],
                "entity2": endpoints[right[0]][0],
                "copy2": endpoints[right[0]][1],
                "position2": right[1],
                "atom2": right[2],
            }
            for left, right in bonds
        ]
    # Validation leaves Protenix at most one pocket, with an explicit distance;
    # OpenDDE shares this writer but its constraints were dropped there.
    for pocket in _pocket_constraints(job):
        binder_entity, binder_copy = endpoints[pocket["binder"]]
        native.setdefault("constraint", {})["pocket"] = {
            "binder_chain": {"entity": binder_entity, "copy": binder_copy},
            "contact_residues": [
                {
                    "entity": endpoints[chain][0],
                    "copy": endpoints[chain][1],
                    "position": residue,
                }
                for chain, residue in pocket["contacts"]
            ],
            "max_distance": float(pocket["max_distance"]),
        }
    # A token contact (constraint_featurizer.py `_canonicalize_contact_format`):
    # entity/copy/position per side, no atom; validation leaves residues of
    # one token on different chains, with an explicit distance.
    contacts = [
        {
            **{
                f"{key}{side}": value
                for side, (chain, residue) in (
                    (1, contact["token1"]),
                    (2, contact["token2"]),
                )
                for key, value in zip(
                    ("entity", "copy", "position"), (*endpoints[chain], residue)
                )
            },
            "max_distance": float(contact["max_distance"]),
        }
        for contact in _contact_constraints(job)
    ]
    if contacts:
        native.setdefault("constraint", {})["contact"] = contacts
    return [native]


# Upstream parses alignment files by stem and skips every other name in silence,
# so an unrecognized one has to be refused here rather than discovered as an
# IndexError inside the MSA parser.
_OPENFOLD3_MSA_STEMS = frozenset(
    {
        "uniref90_hits",
        "uniprot_hits",
        "bfd_uniclust_hits",
        "bfd_uniref_hits",
        "cfdb_uniref30",
        "mgnify_hits",
        "rfam_hits",
        "rnacentral_hits",
        "nt_hits",
        "concat_cfdb_uniref100_filtered",
        "mmseqs_colabfold",
        "colabfold_main",
        "colabfold_paired",
    }
)


#: The stem OpenFold3 reads a generic alignment under. It is not cosmetic: the
#: stem selects the row cap in `max_seq_counts` (dataset_config_components.py:80)
#: -- 16,384 for `colabfold_main` against 10,000 for `uniref90_hits` and 5,000
#: for `mgnify_hits`. `colabfold_main` is the most permissive main-MSA source and
#: the one whose format an `.a3m` from a colabfold-style search already is, so a
#: file that arrives with no source of its own loses the fewest rows there.
_OPENFOLD3_MAIN_STEM = "colabfold_main"
_OPENFOLD3_PAIRED_STEM = "colabfold_paired"


def _link_or_copy_atomic(source: Path, target: Path) -> None:
    """Publish a generated alignment link without following ``target``."""
    with tempfile.TemporaryDirectory(
        prefix=".foldjax-msa-link-", dir=target.parent
    ) as scratch:
        staged = Path(scratch) / target.name
        try:
            staged.symlink_to(source.resolve())
        except OSError:
            shutil.copyfile(source, staged)
        os.replace(staged, target)


def _openfold3_msa(
    value: Any, base: Path, field: str, destination: Path, entity_index: int
) -> list[str]:
    """The alignment, under a name OpenFold3 will actually read.

    OpenFold3 identifies an alignment's source by the file's stem and ignores
    anything it does not recognise. This used to raise on an unrecognised name,
    which is honest but leaves the backend unable to take the one thing users
    have -- an `.a3m` named after the target. Linking it under an accepted stem
    keeps the refusal's real point (nothing is silently dropped) while letting
    the run happen, and the link goes in a per-entity directory because two
    entities with different alignments would otherwise both want the same name.
    """
    path = Path(_path(value, base))
    if path.is_dir():
        return [str(path)]

    if not path.is_file():
        raise FileNotFoundError(f"MSA path is not a file or directory: {path}")

    suffix = path.suffix.lower()
    supported_suffixes = {".a3m", ".sto"}
    if suffix in supported_suffixes and path.stem in _OPENFOLD3_MSA_STEMS:
        return [str(path)]

    if suffix not in supported_suffixes:
        # Extension-less files are common when an alignment comes from a
        # workflow artifact store. OpenFold3 itself dispatches on extension,
        # so infer only the two unambiguous text formats and otherwise fail at
        # this boundary instead of passing a file it will silently ignore.
        with path.open(encoding="utf-8", errors="replace") as source:
            prefix = source.read(4096)
        content = prefix.lstrip("\ufeff \t\r\n")
        if content.startswith("# STOCKHOLM"):
            suffix = ".sto"
        elif content.startswith(">"):
            suffix = ".a3m"
        else:
            raise ValueError(
                f"cannot infer MSA format for {path}; use an A3M/FASTA file "
                "starting with '>' or Stockholm starting with '# STOCKHOLM'"
            )

    # Equality, not `"paired" in field`: the other caller passes "unpaired_msa",
    # which contains it, and would have linked an unpaired alignment under the
    # paired stem -- a silent halving of the row cap, 16,384 to 8,192.
    stem = _OPENFOLD3_PAIRED_STEM if field == "paired_msa" else _OPENFOLD3_MAIN_STEM
    # User-controlled chain identifiers are document data, not path components.
    # A fixed positional directory is stable for one materialized job and cannot
    # turn values such as ``../../outside`` into a traversal.
    msa_root = destination / "msa"
    if msa_root.is_symlink():
        raise ValueError(f"generated MSA directory is a symlink: {msa_root}")
    msa_root.mkdir(parents=True, exist_ok=True)
    if not msa_root.resolve().is_relative_to(destination.resolve()):
        raise ValueError(f"generated MSA directory escapes output root: {msa_root}")
    directory = msa_root / f"entity_{entity_index:04d}"
    if directory.is_symlink():
        raise ValueError(f"generated MSA directory is a symlink: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    if not directory.resolve().is_relative_to(destination.resolve()):
        raise ValueError(f"generated MSA directory escapes output root: {directory}")
    linked = directory / f"{stem}{suffix}"
    _link_or_copy_atomic(path, linked)
    return [str(linked)]


def _write_text_atomic(path: Path, text: str) -> None:
    """Replace a generated input file without following an existing symlink."""
    with tempfile.TemporaryDirectory(
        prefix=".foldjax-input-", dir=path.parent
    ) as scratch:
        staged = Path(scratch) / path.name
        staged.write_text(text, encoding="utf-8")
        os.replace(staged, path)


def _openfold3(
    job: dict[str, Any], base: Path, *, seed: int = 0, destination: Path
) -> dict[str, Any]:
    """Build OpenFold3's inference query document."""
    chains: list[dict[str, Any]] = []
    for entity_index, entity in enumerate(job["entities"]):
        kind = entity["type"]
        ids = _ids(entity)
        body: dict[str, Any] = {"molecule_type": kind, "chain_ids": ids}
        if kind == "ligand":
            if entity.get("ccd"):
                # Validation leaves one code: upstream refuses more.
                body["ccd_codes"] = _ccd_codes(entity)
            else:
                body["smiles"] = str(entity["smiles"])
        else:
            body["sequence"] = str(entity["sequence"])
            if entity.get("unpaired_msa"):
                body["main_msa_file_paths"] = _openfold3_msa(
                    entity["unpaired_msa"],
                    base,
                    "unpaired_msa",
                    destination,
                    entity_index,
                )
            if entity.get("paired_msa"):
                body["paired_msa_file_paths"] = _openfold3_msa(
                    entity["paired_msa"],
                    base,
                    "paired_msa",
                    destination,
                    entity_index,
                )
            modifications = _modifications(entity)
            if modifications:
                # OpenFold3 names these by residue rather than by PTM type.
                body["non_canonical_residues"] = {
                    position: ccd for ccd, position in modifications
                }
            templates = _templates(entity)
            if templates:
                body.update(
                    _openfold3_templates(templates, base, destination, entity_index)
                )
        chains.append(body)

    query: dict[str, Any] = {"chains": chains}
    # Validation leaves one pocket on a ligand binder (upstream's single
    # ``pocket_constraint``); its residue_id is the 1-based query position.
    for pocket in _pocket_constraints(job):
        query["pocket_constraint"] = {
            "ligand_chain_id": pocket["binder"],
            "pocket_residues": [list(contact) for contact in pocket["contacts"]],
            "max_distance": _pocket_max_distance("openfold3", pocket),
        }
    return {"queries": {str(job.get("name", "query")): query}}


def _esmfold2(
    job: dict[str, Any],
    base: Path,
    *,
    seed: int = 0,
    destination: Path | None = None,
) -> dict[str, Any]:
    """ESMFold2 has no dialect to translate into: the adapter reads this schema.

    Unlike every generated native dialect this is otherwise a direct copy, so
    MSA paths are resolved now: relocating the generated document must not
    change which alignment it names.
    """
    for entity in job["entities"]:
        if entity.get("unpaired_msa"):
            entity["unpaired_msa"] = _path(entity["unpaired_msa"], base)
    return job


def common_schema_features(model: str) -> tuple[str, ...]:
    """Which common-schema fields this backend's native dialect can carry."""
    target = _TARGETS.get(model)
    if target is None:
        return ()
    return tuple(sorted(target.features))


#: Routes FoldJAX adds on top of every port and no upstream has, kept apart
#: from ``common_schema_features`` so a capability record never reads as
#: upstream support: ``pocket_selection`` is ``pocket_sampling=select``
#: (`foldjax.pocket_selection`), which scores and ranks samples against a
#: common pocket after prediction on all six models.
_FOLDJAX_ONLY_FEATURES = (POCKET_SELECTION_FEATURE,)


def foldjax_only_features(model: str) -> tuple[str, ...]:
    """FoldJAX-only routes available on ``model``; none for an unknown one."""
    if model not in _TARGETS:
        return ()
    return _FOLDJAX_ONLY_FEATURES


#: Common-schema features (``_TARGETS``) that a model's *released* weight
#: profile cannot reach, though its dialect carries them and some other
#: named profile reads them: Protenix's pocket and contact constraints are
#: embedded by ``trunk_blocks/embedders.py`` ``constraint_embedder``, which
#: only the managed ``--profile base-constraint-v0.5.0`` (upstream
#: ``protenix_base_constraint_v0.5.0``) ships weights for (`input.py:144-147`,
#: `backends/protenix.py` checkpoint gate). Intersected with ``target.features``
#: in `profile_gated_features`, so renaming or dropping a common feature
#: cannot leave a stale entry here.
_PROFILE_GATED_FEATURES: dict[str, frozenset[str]] = {
    "protenix": frozenset({"pocket_constraints", "contact_constraints"}),
}

#: Common-schema features a model's dialect carries and its released weight
#: profile reads, but whose release-default *option* turns off: Protenix and
#: OpenDDE both write a common ``templates`` field into their native input,
#: and both drop it with an ``ignored_templates`` record unless the caller
#: passes ``use_template=true`` (`backends/protenix.py:1112`,
#: `backends/opendde.py:382`, both released ``False``).
_OPTION_GATED_FEATURES: dict[str, frozenset[str]] = {
    "protenix": frozenset({"templates"}),
    "opendde": frozenset({"templates"}),
}


def profile_gated_features(model: str) -> tuple[str, ...]:
    """Common-schema features ``model``'s released profile cannot reach.

    A subset of `common_schema_features`: the dialect carries the field, but
    only a named weight profile has the weights to read it. Reported beside
    `common_schema_features` so a reader does not take every common feature
    as something today's default run actually uses.
    """
    target = _TARGETS.get(model)
    features = target.features if target is not None else frozenset()
    return tuple(sorted(_PROFILE_GATED_FEATURES.get(model, frozenset()) & features))


def option_gated_features(model: str) -> tuple[str, ...]:
    """Common-schema features ``model``'s released *default* turns off.

    A subset of `common_schema_features`, distinct from
    `profile_gated_features`: the released profile has the weights to read
    the field, but the released default option leaves it off, as
    `use_template=false` does.
    """
    target = _TARGETS.get(model)
    features = target.features if target is not None else frozenset()
    return tuple(sorted(_OPTION_GATED_FEATURES.get(model, frozenset()) & features))


#: Scientific inputs a model's *native* dialect carries, through FoldJAX's own
#: port, that the common schema has no field for. Only what the port consumes
#: is listed; a native field the port refuses (OpenFold3's ``covalent_bonds``,
#: `models/openfold3/data/featurize.py`) is not a feature of that model here.
#:
#: Ligands of several CCD components (glycans) are the common ``ccd`` list
#: (``multi_residue_ligand`` in ``_TARGETS``) for every model but OpenFold3,
#: whose v0.5.0 declares ``ccd_codes`` lists and ``sdf_file_path`` but raises
#: NotImplementedError for more than one code and for SDF ligands
#: (core/data/primitives/structure/query.py), upstream and here, so neither
#: is listed for it.
#:
#: Pocket and contact restraints are the common ``constraints`` field too
#: (``pocket_constraints`` and ``contact_constraints`` in ``_TARGETS``):
#: pockets for Boltz-2, Protenix and OpenFold3, contacts for Boltz-2 and
#: Protenix, whose ``constraint`` is embedded by trunk_blocks/embedders.py
#: ``constraint_embedder`` (a checkpoint without constraint weights refuses
#: it). OpenDDE shares the Protenix featurizer but not the embedder, and its
#: upstream inference build ignores ``constraint``
#: (``_IGNORED_CONSTRAINT_MODELS``), so the port drops either as upstream
#: does. A contact on a ligand atom and Protenix atom contacts have no
#: common spelling and stay native input. A pocket or contact's ``force``
#: (Boltz-2's own steering field) is now a common spelling too
#: (`_pocket_constraints`, `_contact_constraints`); every other model
#: refuses it rather than silently dropping it.
#:
#: - ``user_ccd``: a caller-defined chemical component (AlphaFold 3
#:   ``userCCD``/``userCCDPath``).
#: - ``ligand_file``: a ligand read from a structure file (Protenix/OpenDDE
#:   ``FILE_`` ligands).
#: - ``cyclic_polymer``: Boltz-2 and OpenFold3 ``cyclic`` chains.
#:
#: ESMFold2 has no native dialect: it reads the common document itself.
_NATIVE_ONLY: dict[str, frozenset[str]] = {
    "alphafold3": frozenset({"user_ccd"}),
    "boltz2": frozenset({"cyclic_polymer"}),
    "esmfold2": frozenset(),
    "opendde": frozenset({"ligand_file"}),
    "openfold3": frozenset({"cyclic_polymer"}),
    "protenix": frozenset({"ligand_file"}),
}


def native_only_features(
    model: str, capabilities: ModelCapabilities
) -> tuple[str, ...]:
    """Scientific inputs this model takes that the common schema cannot express.

    ``supports_templates`` and ``supports_affinity`` describe the *model*, not
    whether a common job can safely reach that machinery. A dialect may lack a
    field even though the runnable backend supports the feature. Naming that
    gap is the honest half of the answer; a backend whose released runtime
    discards a field must instead report the model-level capability as false.
    The rest come from `_NATIVE_ONLY`: native fields the port consumes and the
    common schema has no field for. Reachable only through native input.
    """
    target = _TARGETS.get(model)
    features = target.features if target is not None else frozenset()
    unreachable = []
    if capabilities.supports_templates and not (
        {"templates", "templates_unmapped"} & features
    ):
        unreachable.append("templates")
    if capabilities.supports_affinity and "affinity" not in features:
        unreachable.append("affinity")
    unreachable.extend(sorted(_NATIVE_ONLY.get(model, frozenset()) - features))
    return tuple(unreachable)


def compatibility(
    document: Any,
    model: str,
    *,
    msa: str | None = None,
    base: Path | None = None,
) -> str | None:
    """Why ``model`` cannot run this common-schema job, or None if it can.

    The answer comes from `_validate` itself rather than from a second table:
    "can this backend express this document" already has exactly one
    implementation, and a discovery command that reimplemented it would
    eventually disagree with the one that decides. ``msa``, given, applies
    the same alignment policy `predict` applies (`_refuse_bare_proteins`), and
    ``base``, the job file's directory, checks that the files it names exist.
    """
    from copy import deepcopy

    from foldjax.registry import get_backend

    backend = get_backend(model)
    target = _TARGETS.get(backend.name)
    if target is None:
        return f"{backend.name} has no common-schema dialect"
    if is_jobs_document(document):
        return (
            "a multi-job file holds several jobs; ask about one job at a time"
        )
    if not isinstance(document, dict):
        return "a FoldJAX job must be a JSON or YAML mapping"
    try:
        job = deepcopy(document)
        _validate(
            job,
            backend.name,
            target,
            backend.capabilities().entity_types,
            base=base,
        )
        if msa is not None:
            _apply_msa_policy(job, backend.name, target, msa)
    except (ValueError, FileNotFoundError) as error:
        return str(error).splitlines()[0]
    return None


def read_utf8_text(path: Path, *, what: str) -> str:
    """A user's input file as text, with a UTF-8 byte-order mark removed.

    Windows editors write the mark, and `json` refuses it as an unexpected
    character at line 1. A file in another encoding is refused naming the file:
    the bare codec message gives a byte offset but not which input it was in.
    """
    path = Path(path)
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"{path} is not UTF-8 text ({error.reason} at byte {error.start}); "
            f"save the {what} file as UTF-8"
        ) from error


def read_job_document(path: Path) -> Any:
    """Load a job file as JSON or YAML.

    The common schema is the same either way; YAML is accepted because it is
    what people already write for Boltz, and JSON is a subset of it. YAML is
    parsed in safe mode, so a job file can never construct Python objects.
    """
    path = Path(path)
    text = read_utf8_text(path, what="job")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as error:
            # One line: the parser's own message spans several, with a caret
            # drawing of the document that reads as a crash in a terminal.
            problem = getattr(error, "problem", None) or str(error).splitlines()[0]
            mark = getattr(error, "problem_mark", None)
            where = (
                f" (line {mark.line + 1}, column {mark.column + 1})"
                if mark is not None
                else ""
            )
            raise ValueError(
                f"{path} is not readable as YAML: {problem}{where}"
            ) from None
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not readable as JSON: {error}") from None


#: The one top-level key of a multi-job common-schema file:
#: ``{"jobs": [{job}, {job}, ...]}``. A mapping, not a top-level list, because
#: a list is already a native shape -- AlphaFold Server's job list and the
#: Protenix/OpenDDE list of jobs -- and `foldjax.api.detect_input_format` and
#: `_job_model_seeds` treat every top-level list as native. No native dialect
#: has a top-level ``jobs`` key: AlphaFold 3 has ``name``/``sequences``/
#: ``modelSeeds``, a Boltz YAML ``version``/``sequences``, OpenFold3's query set
#: ``seeds``/``queries``, and ESMFold2 reads the single-job common schema.
JOBS_KEY = "jobs"

#: Common-schema fields holding a path, resolved against the job file's own
#: directory. A job split out of a multi-job file is written elsewhere, so
#: these are made absolute against the source's directory first.
_PATH_FIELDS = ("unpaired_msa", "paired_msa")


def is_jobs_document(document: Any) -> bool:
    """Whether ``document`` is a multi-job file rather than one job."""
    return (
        isinstance(document, Mapping)
        and JOBS_KEY in document
        and "entities" not in document
    )


def is_jobs_file(path: Path) -> bool:
    """Whether ``path`` holds a multi-job common-schema document."""
    from foldjax.schema import JOB_DOCUMENT_SUFFIXES

    path = Path(path)
    if path.suffix.lower() not in JOB_DOCUMENT_SUFFIXES or not path.is_file():
        return False
    try:
        return is_jobs_document(read_job_document(path))
    except (OSError, ValueError):
        return False


def read_jobs_file(path: Path) -> list[tuple[str, dict[str, Any]]]:
    """Return ``(name, job)`` for each job of a multi-job file, in order.

    The container is checked here -- one ``jobs`` key, a non-empty list of
    mappings, each with a name of its own that no other job in the file
    shares, before or after it is made safe for a directory name. Each job's
    content is checked later, per run and per model, exactly as a file of its
    own would be.
    """
    from foldjax.output import safe_job_name

    path = Path(path)
    document = read_job_document(path)
    if not is_jobs_document(document):
        raise ValueError(f"{path} is not a multi-job file ({{{JOBS_KEY!r}: [...]}})")
    allowed = frozenset({JOBS_KEY})
    _reject_unknown(
        set(document) - allowed, allowed, f"top-level fields of multi-job file {path}"
    )
    jobs = document[JOBS_KEY]
    if not isinstance(jobs, list) or not jobs:
        raise ValueError(f"{path}: {JOBS_KEY} must be a non-empty list of jobs")
    named: list[tuple[str, dict[str, Any]]] = []
    names: dict[str, int] = {}
    directories: dict[str, tuple[int, str]] = {}
    for index, job in enumerate(jobs):
        where = f"{path} {JOBS_KEY}[{index}]"
        if not isinstance(job, dict):
            raise ValueError(f"{where} must be a job mapping")
        name = job.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"{where} needs a non-empty name: in a multi-job file each "
                "job's name is its output directory"
            )
        name = name.strip()
        where = f"{where} ({name!r})"
        if name in names:
            raise ValueError(
                f"{where} repeats the name of {JOBS_KEY}[{names[name]}]; job "
                "names must be unique within the file"
            )
        names[name] = index
        directory = safe_job_name(name)
        if directory in directories:
            other, other_name = directories[directory]
            raise ValueError(
                f"{where} and {JOBS_KEY}[{other}] ({other_name!r}) would both "
                f"write to the output directory {directory!r}; rename one"
            )
        directories[directory] = (index, name)
        named.append((name, job))
    return named


def _absolute_job_paths(job: dict[str, Any], base: Path) -> dict[str, Any]:
    """A copy of ``job`` whose relative path fields name the same files from anywhere.

    Not resolved: ``_path`` resolves when the native document is written, and
    leaving that to it is what makes a split job's native input identical to
    the one its own file would produce. A field that is not a usable path is
    left alone, so it is refused later with the same message as in a file of
    its own.
    """
    from copy import deepcopy

    def absolute(value: Any) -> Any:
        if not isinstance(value, str) or not value.strip():
            return value
        path = Path(value.strip())
        return value if path.is_absolute() else str(base / path)

    job = deepcopy(job)
    entities = job.get("entities")
    if not isinstance(entities, list):
        return job
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        for key in _PATH_FIELDS:
            if key in entity:
                entity[key] = absolute(entity[key])
        templates = entity.get("templates")
        if isinstance(templates, list):
            for template in templates:
                if isinstance(template, dict) and "mmcif" in template:
                    template["mmcif"] = absolute(template["mmcif"])
    return job


def expand_jobs_file(
    path: Path, *, root: Path | None = None
) -> tuple[tuple[Path, Any], ...]:
    """Write each job of a multi-job file as its own document; return them.

    Each entry is ``(generated path, JobSource)``. The generated file is named
    after the job, so a batch puts it in ``<out>/<model>/<job name>`` exactly
    as it would a file of that name, and it lives in a directory keyed by its
    own content: an unchanged job keeps its path, and therefore its resume
    identity, when another job in the same file is edited or reordered.
    ``root`` replaces the store's ``runtime/jobs/split`` (`foldjax plan`
    writes into a scratch directory, never the store).
    """
    import hashlib

    from foldjax import paths
    from foldjax.output import safe_job_name
    from foldjax.schema import JobSource

    path = Path(path)
    base = path.parent.absolute()
    root = Path(root) if root is not None else paths.runtime_dir("jobs") / "split"
    expanded: list[tuple[Path, Any]] = []
    for index, (name, job) in enumerate(read_jobs_file(path)):
        try:
            text = json.dumps(_absolute_job_paths(job, base), indent=2)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{path} {JOBS_KEY}[{index}] ({name!r}) cannot be written as "
                f"JSON: {error}"
            ) from error
        digest = hashlib.sha256(text.encode()).hexdigest()[:16]
        target = root / digest / f"{safe_job_name(name)}.json"
        if not (target.is_file() and target.read_text(encoding="utf-8") == text):
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_text_atomic(target, text)
        expanded.append((target, JobSource(path=path, index=index, name=name)))
    return tuple(expanded)


#: Models whose upstream folds a protein chain with no alignment from its
#: sequence alone by default. ESMFold2's has no search path: ``forward`` on
#: ``infer_protein`` builds a depth-1 MSA (transformers-esmfold2
#: ``protein_utils.py:452-453``). Every other upstream refuses such a chain
#: (Boltz-2, ``boltz/main.py:581-583``) or searches for an alignment
#: (Protenix, OpenDDE, OpenFold3, and AlphaFold 3's data pipeline), so under
#: ``msa="none"`` the chain is refused rather than silently folded alone.
_SINGLE_SEQUENCE_UPSTREAM = frozenset({"esmfold2"})


def _refuse_bare_proteins(job: dict[str, Any], model: str) -> None:
    """Refuse protein chains with no alignment unless the caller opted in."""
    bare = [
        _ids(entity)[0]
        for entity in job["entities"]
        if entity.get("type") == "protein" and not entity.get("unpaired_msa")
    ]
    if not bare:
        return
    raise ValueError(
        f"{model}: protein chain(s) {', '.join(bare)} have no alignment, and "
        f"upstream {model} does not fold a protein from its sequence alone by "
        "default (it searches for an alignment or refuses the job). Pass one "
        "of: --msa auto (msa='auto') to search the ColabFold MMseqs2 server, "
        "which SENDS THE SEQUENCE off this machine (FOLDJAX_MSA_SERVER_URL "
        "points at your own); an unpaired_msa path on the entity; or --msa "
        "single (msa='single') to fold from the single sequence on purpose"
    )


def _apply_msa_policy(
    job: dict[str, Any],
    model: str,
    target: _Target,
    msa: str,
    options: Mapping[str, Any] | None = None,
) -> None:
    """Refuse what the alignment policy refuses, before anything searches.

    The same conditions `_search_alignments` raises on, asked of a validated
    job without contacting a server.
    """
    if msa not in MSA_POLICIES:
        raise ValueError(f"msa must be one of {MSA_POLICIES}; got {msa!r}")
    if msa == "none" and model not in _SINGLE_SEQUENCE_UPSTREAM:
        _refuse_bare_proteins(job, model)
    if msa not in ("auto", "required"):
        return
    if "unpaired_msa" not in target.features:
        raise ValueError(
            f"{model} cannot take a searched alignment; run it with msa='none'"
        )
    read = _nucleic_msa_read(
        model, use_rna_msa=(options or {}).get("use_rna_msa") is True
    )
    if msa == "required" and (read is None or "rna" in read):
        bare_rna = [
            entity
            for entity in job["entities"]
            if entity.get("type") == "rna" and not entity.get("unpaired_msa")
        ]
        if bare_rna and _rna_msa_pipeline() is None:
            from foldjax.msa_search import _RNA_MSA_COMMAND_ENV

            raise ValueError(
                f"RNA entity {_ids(bare_rna[0])[0]!r} has no alignment and no RNA "
                f"search is configured; set {_RNA_MSA_COMMAND_ENV} to a local "
                "nhmmer workflow, or supply unpaired_msa for it"
            )


def _checked_common_job(
    source: Path,
    capabilities: ModelCapabilities,
    *,
    msa: str,
    options: Mapping[str, Any] | None,
    templates: str,
    ignored: list[dict[str, Any]] | None = None,
    ignored_templates: list[dict[str, Any]] | None = None,
    ignored_constraints: list[dict[str, Any]] | None = None,
    check_files: bool = False,
    msa_pairing: str = "model",
    template_dir: Path | None = None,
    pocket_conditioning: bool | None = None,
    selection_pockets: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Read and validate a common job exactly as translation will, writing nothing.

    ``check_files`` also requires every alignment and template the job names
    to exist (`preflight` asks; translation leaves a missing one to the
    backend, which reports it in its own words). ``pocket_conditioning`` and
    ``selection_pockets`` are `_validate_pocket_constraints`'s.
    """
    from foldjax.msa_search import refuse_msa_pairing
    from foldjax.template_search import refuse_template_search

    model = capabilities.model
    target = _TARGETS.get(model)
    if target is None:
        raise ValueError(f"unsupported model: {model}")
    if msa not in MSA_POLICIES:
        raise ValueError(f"msa must be one of {MSA_POLICIES}; got {msa!r}")
    if templates not in TEMPLATE_POLICIES:
        raise ValueError(
            f"templates must be one of {TEMPLATE_POLICIES}; got {templates!r}"
        )
    refuse_template_search(model, templates, options, template_dir=template_dir)
    if msa_pairing != "model" and msa not in ("auto", "required"):
        raise ValueError(
            f"msa_pairing={msa_pairing!r} chooses how a searched alignment is "
            "paired; set msa='auto' or 'required' (--msa auto) or drop it"
        )
    refuse_msa_pairing(model, msa_pairing)
    source = Path(source)
    job = read_job_document(source)
    if is_jobs_document(job):
        raise ValueError(
            f"{source} is a multi-job file; pass it as one of several inputs "
            "(inputs=...) to run every job in it"
        )
    if not isinstance(job, dict):
        raise ValueError("a FoldJAX job must be a JSON or YAML mapping")
    _validate(
        job,
        model,
        target,
        capabilities.entity_types,
        options=options,
        ignored=ignored,
        ignored_templates=ignored_templates,
        ignored_constraints=ignored_constraints,
        base=source.parent if check_files else None,
        pocket_conditioning=pocket_conditioning,
        selection_pockets=selection_pockets,
    )
    # Before anything is written: the refusal is about the job, not the run.
    _apply_msa_policy(job, model, target, msa, options)
    return job


def validate_common_input(
    source: Path,
    capabilities: ModelCapabilities,
    *,
    msa: str = "none",
    options: Mapping[str, Any] | None = None,
    templates: str = "none",
    msa_pairing: str = "model",
    template_dir: Path | None = None,
    pocket_conditioning: bool | None = None,
) -> None:
    """Raise what `materialize_native_input` would raise about this job.

    And that every alignment and template it names exists, which a backend
    otherwise discovers after loading. Read-only: nothing is searched,
    fetched or written, so `foldjax plan` can refuse exactly what `foldjax
    predict` refuses. What only featurization
    can know -- the MSA rows a model stores, which ``padding.msa`` must not
    undercut -- is still checked at run time. ``pocket_conditioning`` is
    `materialize_native_input`'s.
    """
    _checked_common_job(
        source,
        capabilities,
        msa=msa,
        options=options,
        templates=templates,
        check_files=True,
        msa_pairing=msa_pairing,
        template_dir=template_dir,
        pocket_conditioning=pocket_conditioning,
    )


def materialize_native_input(
    source: Path,
    capabilities: ModelCapabilities,
    output_dir: Path,
    *,
    seed: int,
    msa: str = "none",
    options: Mapping[str, Any] | None = None,
    ignored: list[dict[str, Any]] | None = None,
    ignored_templates: list[dict[str, Any]] | None = None,
    templates: str = "none",
    template_max_date: str | None = None,
    template_search: list[dict[str, Any]] | None = None,
    ignored_constraints: list[dict[str, Any]] | None = None,
    constraints: list[dict[str, Any]] | None = None,
    msa_search: list[dict[str, Any]] | None = None,
    msa_pairing: str = "model",
    template_dir: Path | None = None,
    msa_stats: list[dict[str, Any]] | None = None,
    pocket_conditioning: bool | None = None,
) -> Path:
    """Translate a FoldJAX JSON document to one backend-native input file.

    ``msa_pairing`` chooses how a searched alignment pairs a complex
    (`foldjax.msa_search.resolve_pairing`), ``template_dir`` searches a private
    folder of mmCIFs for templates, and ``msa_stats`` receives each chain's
    alignment depth and Neff (`foldjax.msa_stats`) as the native input reads
    it.

    ``ignored``, when given, receives one record per alignment the document
    named but the native input leaves out (see ``IGNORE_NUCLEIC_MSA``), and
    ``ignored_templates`` one per template (see ``IGNORE_TEMPLATES``).
    ``templates="auto"`` searches templates for protein chains that name none
    (`foldjax.template_search`); ``template_search``, when given, receives
    one record per searched chain; ``msa_search`` does the same for
    ``msa="auto"``/``"required"``, a failed chain's record carrying ``error``.
    ``ignored_constraints`` receives pocket and contact restraints dropped
    as upstream drops them (see ``IGNORE_CONSTRAINTS``), and ``constraints``
    one record per pocket and per contact written into the native input, with
    the ``max_distance`` it runs at and whether that came from the job or
    from the upstream default. Under ``pocket_sampling=select`` each pocket
    record also carries ``route``: ``native`` when the native input conditions
    on it as well, ``selection`` when FoldJAX's selection is the only thing
    that reads it; ``pocket_conditioning`` (None: the translation table) says
    which, for a model whose checkpoint decides.
    """
    from foldjax.pocket_selection import requested

    model = capabilities.model
    target = _TARGETS.get(model)
    if target is None:
        raise ValueError(f"unsupported model: {model}")
    source = Path(source)
    dropped: list[dict[str, Any]] = []
    dropped_templates: list[dict[str, Any]] = []
    dropped_constraints: list[dict[str, Any]] = []
    selection_pockets: list[dict[str, Any]] = []
    job = _checked_common_job(
        source,
        capabilities,
        msa=msa,
        options=options,
        templates=templates,
        ignored=dropped,
        ignored_templates=dropped_templates,
        ignored_constraints=dropped_constraints,
        msa_pairing=msa_pairing,
        template_dir=template_dir,
        pocket_conditioning=pocket_conditioning,
        selection_pockets=selection_pockets,
    )
    # Only under ``select``: an ``off`` run's records keep their shape.
    pocket_route = {} if requested(options) == "off" else {"route": "native"}
    if dropped_constraints:
        import warnings

        for record in dropped_constraints:
            described = []
            if record["binders"]:
                described.append(
                    f"pocket constraint (binder {', '.join(record['binders'])})"
                )
            if record.get("contacts"):
                described.append(f"{len(record['contacts'])} contact constraint(s)")
            warnings.warn(
                f"{model}: the job's {' and '.join(described)}: "
                f"{record['reason']}; the run manifest records it",
                UserWarning,
                stacklevel=2,
            )
    if ignored_constraints is not None:
        ignored_constraints.extend(dropped_constraints)
    if constraints is not None:
        constraints.extend(
            _with_force(
                {
                    "kind": "pocket",
                    "binder": pocket["binder"],
                    "contacts": [list(contact) for contact in pocket["contacts"]],
                    "max_distance": _pocket_max_distance(model, pocket),
                    "max_distance_source": (
                        "job" if pocket["max_distance"] is not None else "upstream"
                    ),
                    **pocket_route,
                },
                pocket["force"],
            )
            for pocket in _pocket_constraints(job)
        )
        # Validation left these an explicit distance (no upstream default to
        # inherit), so the value is the job's.
        constraints.extend(
            _with_force(
                {
                    "kind": "pocket",
                    "binder": pocket["binder"],
                    "contacts": [list(contact) for contact in pocket["contacts"]],
                    "max_distance": float(pocket["max_distance"]),
                    "max_distance_source": "job",
                    "route": "selection",
                },
                pocket["force"],
            )
            for pocket in selection_pockets
        )
        constraints.extend(
            _with_force(
                {
                    "kind": "contact",
                    "token1": list(contact["token1"]),
                    "token2": list(contact["token2"]),
                    "max_distance": _contact_max_distance(model, contact),
                    "max_distance_source": (
                        "job" if contact["max_distance"] is not None else "upstream"
                    ),
                },
                contact["force"],
            )
            for contact in _contact_constraints(job)
        )

    base = source.parent
    for record in (*dropped, *dropped_templates):
        record["resolved_path"] = _path(record["path"], base)
    if dropped or dropped_templates:
        import warnings

        for record in (*dropped, *dropped_templates):
            warnings.warn(
                f"{model}: chain(s) {', '.join(record['chains'])} name "
                f"{record['field']} {record['path']!r}: {record['reason']}; "
                "the run manifest records it",
                UserWarning,
                stacklevel=2,
            )
    if ignored is not None:
        ignored.extend(dropped)
    if ignored_templates is not None:
        ignored_templates.extend(dropped_templates)

    # Created before the dialects are built: OpenFold3 writes alongside its
    # document rather than only into it, and a searched alignment is recorded
    # beside both.
    output_dir = Path(output_dir)
    if output_dir.is_symlink():
        raise ValueError(f"generated input directory is a symlink: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Validation ran before the search, so a searched RNA alignment is only
    # attached where the backend reads one; elsewhere it would be discarded
    # after the check above.
    read = _nucleic_msa_read(
        model, use_rna_msa=(options or {}).get("use_rna_msa") is True
    )
    searched = _search_alignments(
        job,
        target,
        # `single` searches exactly as much as `none` does: nothing.
        policy="none" if msa == "single" else msa,
        model=model,
        search_rna=read is None or "rna" in read,
        pairing=msa_pairing,
    )
    if msa_search is not None:
        msa_search.extend(searched)
    if searched:
        _write_text_atomic(
            output_dir / "msa_search.json", json.dumps(searched, indent=2)
        )
    elif msa in ("none", "single"):
        _warn_single_sequence(job, model, asked=msa == "single")
    if templates != "none":
        from foldjax.template_search import search_templates

        # After validation, like the alignment search: what is attached here
        # is born in the backend's form, so the writer below translates it
        # exactly as it translates a template the caller wrote.
        records = search_templates(
            job,
            model,
            max_date=template_max_date,
            destination=output_dir,
            required=templates == "required",
            template_dir=template_dir,
        )
        _write_text_atomic(
            output_dir / "template_search.json", json.dumps(records, indent=2)
        )
        if template_search is not None:
            template_search.extend(records)
    if msa_stats is not None:
        from foldjax.msa_stats import job_msa_stats

        msa_stats.extend(
            job_msa_stats(
                job, base, skip={str(record["resolved_path"]) for record in dropped}
            )
        )

    # Which writer, and its suffix, are the port table's; OpenDDE reads the
    # Protenix dialect, so the two entries name one writer rather than a
    # membership test here.
    writer = provider(PORTS[model].input_dialect)
    document: Any = writer(job, base, seed=seed, destination=output_dir)
    path = output_dir / f"{model}_input{target.suffix}"
    text = document if isinstance(document, str) else json.dumps(document, indent=2)
    _write_text_atomic(path, text)
    return path
