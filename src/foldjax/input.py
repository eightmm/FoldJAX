"""FoldJAX common-input to model-native input conversion.

The common schema is deliberately model-neutral: modifications and covalent
bonds are expressed once here and translated into each backend's own key names.
A field a backend cannot express is rejected rather than dropped, because
silently discarding an MSA or a modification changes the science without
changing the exit code.
"""

from __future__ import annotations

import difflib
import json
import operator
import os
import shutil
import string
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from foldjax.models.boltz2.data.identifiers import validate_ccd_identifier
from foldjax.schema import MSA_POLICIES, ModelCapabilities

_ENTITY_TYPES = ("protein", "dna", "rna", "ligand")
_JOB_KEYS = frozenset({"name", "entities", "bonds", "properties"})
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
        # A structural template, in the two forms the dialects actually take.
        # AlphaFold 3 and Protenix require an explicit query-residue ->
        # template-residue map and refuse a bare file (`folding_input.py:378`,
        # `template_features.py:329`); Boltz-2 takes a path and aligns it itself
        # (`parse/schema.py:986`). OpenDDE's shipped inference configuration
        # disables its template path, so the common route refuses both forms.
        # These are different inputs, not one input in two spellings, so they
        # are separate features and a document is refused by whichever model
        # cannot use its form.
        "templates",
        "templates_unmapped",
        # Binding affinity. Only Boltz-2 has the head, and its schema addresses
        # the binder by chain id (`parse/schema.py:988`).
        "affinity",
    }
)

_NO_TEMPLATES = {"templates", "templates_unmapped", "affinity"}

# Boltz derives pairing from a single per-chain a3m, so a separate paired MSA has
# nowhere to go.
_TARGETS = {
    "alphafold3": _Target(
        ".json", _ALL_FEATURES - {"templates_unmapped", "affinity"}
    ),
    # foldjax.models.boltz2 dispatches its parser on the file suffix and rejects .json
    # outright; JSON is a YAML subset, so the document is written as .yaml.
    "boltz2": _Target(".yaml", _ALL_FEATURES - {"paired_msa", "templates"}),
    # OpenDDE consumes most of the Protenix list-of-jobs dialect, including its
    # modified-polymer and entity/copy-addressed covalent-bond representations.
    # Its released inference defaults set ``use_template=False``: a generated
    # ``templatesPath`` is deliberately discarded by the native compatibility
    # path.  A common job must therefore reject templates instead of advertising
    # a field whose scientific input will not reach the model.
    "opendde": _Target(".json", _ALL_FEATURES - _NO_TEMPLATES),
    "protenix": _Target(".json", _ALL_FEATURES - {"templates_unmapped", "affinity"}),
    # OpenFold3 expresses everything, but with two constraints its own layer
    # enforces and this writer therefore has to: alignment files are selected by
    # *stem* and only database names are parsed, and a paired MSA that cannot
    # actually be paired collapses the MSA to the query sequence alone.
    # The released OpenFold3 query schema declares covalent bonds but its
    # featurizer never applies them. Advertising the field would silently drop
    # chemistry, so reject it until the upstream pipeline consumes the contract.
    # OpenFold3's template features come from its own cache/search pipeline;
    # its query document has no field for a caller-supplied structure, so a
    # template here would be dropped rather than used.
    "openfold3": _Target(".json", _ALL_FEATURES - {"bonds"} - _NO_TEMPLATES),
    # ESMFold2 has no second native dialect: its NumPy adapter reads the common
    # document and implements Biohub's all-biomolecule tokenizer directly.
    # Taxonomy-paired MSAs and structural templates are not part of this route;
    # every chemistry feature below is represented without loss.
    "esmfold2": _Target(
        ".json",
        {"unpaired_msa", "modifications", "ligand_ccd", "ligand_smiles", "bonds"},
    ),
}


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
                item.strip()
                for item in value
                if isinstance(item, str) and item.strip()
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
            "letters only"
            if allowed is None
            else "IUPAC " + "".join(sorted(allowed))
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
        position = _strict_position(
            item["position"], name="modification position"
        )
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


_TEMPLATE_KEYS = frozenset(
    {"mmcif", "query_indices", "template_indices", "chain_id"}
)


def _templates(entity: dict[str, Any]) -> list[dict[str, Any]]:
    """Return validated structural templates for one chain.

    ``mmcif`` is a path. ``query_indices``/``template_indices`` are the
    residue map three of the five dialects require; supplying one without the
    other, or lists of different lengths, is refused here rather than producing
    a silently truncated mapping inside a featurizer.
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


_Endpoint = tuple[str, int, str]


def _bonds(job: dict[str, Any], chains: set[str]) -> list[tuple[_Endpoint, _Endpoint]]:
    """Return validated ``[chain, residue, atom]`` endpoint pairs.

    ``chains`` is the set of known chain ids, or empty to skip that check when
    the caller does not need chain resolution.
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
            endpoints.append((chain, residue, atom))
        pairs.append((endpoints[0], endpoints[1]))
    return pairs


def _validate(
    job: dict[str, Any],
    model: str,
    target: _Target,
    entity_types: tuple[str, ...],
) -> None:
    """Check the common document against what ``model`` can express."""
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
            for field_name in ("ccd", "smiles"):
                field_value = entity.get(field_name)
                if field_value is not None and (
                    not isinstance(field_value, str) or not field_value.strip()
                ):
                    raise ValueError(
                        f"ligand {field_name} must be a non-empty string"
                    )
                if field_value is not None:
                    entity[field_name] = field_value.strip()
            if not (entity.get("ccd") or entity.get("smiles")):
                raise ValueError("ligand entity requires ccd or smiles")
            if entity.get("ccd") and entity.get("smiles"):
                raise ValueError("ligand entity accepts either ccd or smiles, not both")
            if model == "boltz2" and entity.get("ccd"):
                entity["ccd"] = validate_ccd_identifier(
                    entity["ccd"], field="ligand CCD code"
                )
            feature = "ligand_ccd" if entity.get("ccd") else "ligand_smiles"
            if feature not in target.features:
                _reject(model, feature, "supply the other ligand representation")
            continue

        entity["sequence"] = _normalize_sequence(
            entity.get("sequence"), kind=kind, chain=_ids(entity)[0]
        )
        for feature in ("unpaired_msa", "paired_msa"):
            value = entity.get(feature)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{feature} must be a non-empty path string")
            if value is not None and feature not in target.features:
                _reject(model, feature, f"remove it from entity {_ids(entity)[0]!r}")
            if value is not None and model == "opendde" and kind == "rna":
                _reject(
                    model,
                    f"RNA {feature}",
                    "its shipped inference default is use_rna_msa=False and "
                    "would discard the alignment; protein MSAs remain supported",
                )
            if value is not None:
                entity[feature] = value.strip()
        if entity.get("modifications") and "modifications" not in target.features:
            _reject(
                model,
                "modifications",
                f"remove it from entity {_ids(entity)[0]!r}",
            )
        modifications = _modifications(entity)
        if model == "boltz2":
            for ccd, _ in modifications:
                validate_ccd_identifier(ccd, field="modification CCD code")

        for template in _templates(entity):
            feature = "templates" if template["mapping"] else "templates_unmapped"
            if feature in target.features:
                continue
            if model == "opendde":
                _reject(
                    model,
                    "templates in the common schema",
                    "its shipped inference default is use_template=False and "
                    "would discard the generated templatesPath",
                )
            if feature == "templates_unmapped" and "templates" in target.features:
                _reject(
                    model,
                    "a template without a residue map",
                    "it requires query_indices and template_indices; Boltz-2 is "
                    "the one backend that aligns a bare mmCIF itself",
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

    if job.get("bonds") and "bonds" not in target.features:
        _reject(model, "bonds", "use that model's native input format instead")
    _bonds(job, chains)
    if job.get("properties") and "affinity" not in target.features:
        _reject(
            model,
            "binding affinity",
            "only Boltz-2 carries an affinity head",
        )
    _affinity_binder(job, chains)


def _alphafold3(job: dict[str, Any], base: Path, seed: int) -> dict[str, Any]:
    sequences = []
    for entity in job["entities"]:
        kind = entity["type"]
        body: dict[str, Any] = {"id": _ids(entity)}
        if kind == "ligand":
            if entity.get("ccd"):
                body["ccdCodes"] = [str(entity["ccd"])]
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
                        "mmcifPath": _path(template["mmcif"], base),
                        "queryIndices": [pair[0] for pair in template["mapping"]],
                        "templateIndices": [pair[1] for pair in template["mapping"]],
                    }
                    for template in templates
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


def _boltz(job: dict[str, Any], base: Path) -> dict[str, Any]:
    sequences = []
    for entity in job["entities"]:
        kind = entity["type"]
        body: dict[str, Any] = {"id": _ids(entity)}
        if kind == "ligand":
            if entity.get("ccd"):
                body["ccd"] = str(entity["ccd"])
            else:
                body["smiles"] = str(entity["smiles"])
        else:
            body["sequence"] = str(entity["sequence"])
            if entity.get("unpaired_msa"):
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
    # Boltz keeps templates at the top level and aligns each one itself
    # (`parse/schema.py:1633`), so there is no residue map to carry across.
    templates = [
        {"cif": _path(template["mmcif"], base)}
        for entity in job["entities"]
        for template in _templates(entity)
    ]
    if templates:
        native["templates"] = templates
    binder = _affinity_binder(job, set())
    if binder is not None:
        native["properties"] = [{"affinity": {"binder": binder}}]
    return native


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
        entry: dict[str, Any] = {
            "mmcif": path.read_text(encoding="utf-8"),
            "queryIndices": [pair[0] for pair in template["mapping"]],
            "templateIndices": [pair[1] for pair in template["mapping"]],
        }
        if template["chain_id"] is not None:
            entry["chainId"] = template["chain_id"]
        payload.append(entry)
    directory = destination / "templates"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"entity_{entity_index:04d}.json"
    _write_text_atomic(target, json.dumps(payload))
    return str(target)


def _protenix(
    job: dict[str, Any], base: Path, seed: int, destination: Path
) -> list[dict[str, Any]]:
    sequences = []
    # Protenix addresses covalent bonds by 1-based entity number and copy index
    # rather than by chain id, and derives both from this sequences list.
    endpoints: dict[str, tuple[int, int]] = {}
    for entity_number, entity in enumerate(job["entities"], start=1):
        kind = entity["type"]
        ids = _ids(entity)
        for copy_id, chain_id in enumerate(ids, start=1):
            endpoints[chain_id] = (entity_number, copy_id)
        body: dict[str, Any] = {"id": ids, "count": len(ids)}
        if kind == "ligand":
            body["ligand"] = (
                f"CCD_{entity['ccd']}" if entity.get("ccd") else str(entity["smiles"])
            )
        else:
            body["sequence"] = str(entity["sequence"])
            if entity.get("unpaired_msa"):
                body["unpairedMsaPath"] = _path(entity["unpaired_msa"], base)
            if entity.get("paired_msa"):
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
        searched.extend(
            _run_search(rna_pipeline, rna, policy=policy, paired=False)
        )
    return searched


def _run_search(
    pipeline: Any,
    entities: list[dict[str, Any]],
    *,
    policy: str,
    paired: bool,
) -> list[dict[str, str]]:
    """Search for these chains and attach what came back."""
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


def _write_text_atomic(path: Path, text: str) -> None:
    """Replace a generated input file without following an existing symlink."""
    with tempfile.TemporaryDirectory(
        prefix=".foldjax-input-", dir=path.parent
    ) as scratch:
        staged = Path(scratch) / path.name
        staged.write_text(text, encoding="utf-8")
        os.replace(staged, path)


def _openfold3(job: dict[str, Any], base: Path, destination: Path) -> dict[str, Any]:
    """Build OpenFold3's inference query document."""
    chains: list[dict[str, Any]] = []
    for entity_index, entity in enumerate(job["entities"]):
        kind = entity["type"]
        ids = _ids(entity)
        body: dict[str, Any] = {"molecule_type": kind, "chain_ids": ids}
        if kind == "ligand":
            if entity.get("ccd"):
                body["ccd_codes"] = [str(entity["ccd"])]
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
        chains.append(body)

    query: dict[str, Any] = {"chains": chains}
    return {"queries": {str(job.get("name", "query")): query}}


def common_schema_features(model: str) -> tuple[str, ...]:
    """Which common-schema fields this backend's native dialect can carry."""
    target = _TARGETS.get(model)
    if target is None:
        return ()
    return tuple(sorted(target.features))


def native_only_features(
    model: str, capabilities: ModelCapabilities
) -> tuple[str, ...]:
    """Scientific inputs this model takes that the common schema cannot express.

    ``supports_templates`` and ``supports_affinity`` describe the *model*, not
    whether a common job can safely reach that machinery. A dialect may lack a
    field even though the runnable backend supports the feature. Naming that
    gap is the honest half of the answer; a backend whose released runtime
    discards a field must instead report the model-level capability as false.
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
    return tuple(unreachable)


def compatibility(document: Any, model: str) -> str | None:
    """Why ``model`` cannot run this common-schema job, or None if it can.

    The answer comes from `_validate` itself rather than from a second table:
    "can this backend express this document" already has exactly one
    implementation, and a discovery command that reimplemented it would
    eventually disagree with the one that decides.
    """
    from copy import deepcopy

    from foldjax.registry import get_backend

    backend = get_backend(model)
    target = _TARGETS.get(backend.name)
    if target is None:
        return f"{backend.name} has no common-schema dialect"
    if not isinstance(document, dict):
        return "a FoldJAX job must be a JSON or YAML mapping"
    try:
        _validate(
            deepcopy(document),
            backend.name,
            target,
            backend.capabilities().entity_types,
        )
    except (ValueError, FileNotFoundError) as error:
        return str(error).splitlines()[0]
    return None


def read_job_document(path: Path) -> Any:
    """Load a job file as JSON or YAML.

    The common schema is the same either way; YAML is accepted because it is
    what people already write for Boltz, and JSON is a subset of it. YAML is
    parsed in safe mode, so a job file can never construct Python objects.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as error:
            raise ValueError(f"{path} is not readable as YAML: {error}") from error
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not readable as JSON: {error}") from error


def materialize_native_input(
    source: Path,
    capabilities: ModelCapabilities,
    output_dir: Path,
    *,
    seed: int,
    msa: str = "none",
) -> Path:
    """Translate a FoldJAX JSON document to one backend-native input file."""
    model = capabilities.model
    target = _TARGETS.get(model)
    if target is None:
        raise ValueError(f"unsupported model: {model}")
    if msa not in MSA_POLICIES:
        raise ValueError(f"msa must be one of {MSA_POLICIES}; got {msa!r}")
    source = Path(source)
    job = read_job_document(source)
    if not isinstance(job, dict):
        raise ValueError("a FoldJAX job must be a JSON or YAML mapping")
    _validate(job, model, target, capabilities.entity_types)

    base = source.parent
    # Created before the dialects are built: OpenFold3 writes alongside its
    # document rather than only into it, and a searched alignment is recorded
    # beside both.
    output_dir = Path(output_dir)
    if output_dir.is_symlink():
        raise ValueError(f"generated input directory is a symlink: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    searched = _search_alignments(job, target, policy=msa, model=model)
    if searched:
        _write_text_atomic(
            output_dir / "msa_search.json", json.dumps(searched, indent=2)
        )
    elif msa == "none":
        _warn_single_sequence(job, model)

    if model == "esmfold2":
        # No dialect to translate into: the adapter consumes this schema.
        # Unlike every generated native dialect, this is otherwise a direct
        # copy; resolve MSA paths now so relocating the generated document does
        # not change which alignment it names.
        for entity in job["entities"]:
            if entity.get("unpaired_msa"):
                entity["unpaired_msa"] = _path(entity["unpaired_msa"], base)
        document: Any = job
    elif model == "alphafold3":
        document = _alphafold3(job, base, seed)
    elif model == "boltz2":
        document = _boltz(job, base)
    elif model in {"opendde", "protenix"}:
        document = _protenix(job, base, seed, output_dir)
    else:
        document = _openfold3(job, base, output_dir)
    path = output_dir / f"{model}_input{target.suffix}"
    text = document if isinstance(document, str) else json.dumps(document, indent=2)
    _write_text_atomic(path, text)
    return path
