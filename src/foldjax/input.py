"""FoldJAX common-input to model-native input conversion.

The common schema is deliberately model-neutral: modifications and covalent
bonds are expressed once here and translated into each backend's own key names.
A field a backend cannot express is rejected rather than dropped, because
silently discarding an MSA or a modification changes the science without
changing the exit code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from foldjax.schema import ModelCapabilities

_ENTITY_TYPES = ("protein", "dna", "rna", "ligand")
_JOB_KEYS = frozenset({"name", "entities", "bonds"})
_POLYMER_KEYS = frozenset(
    {"type", "id", "sequence", "unpaired_msa", "paired_msa", "modifications"}
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
    }
)

# Boltz derives pairing from a single per-chain a3m, so a separate paired MSA has
# nowhere to go. Chai takes a job-level MSA directory rather than per-chain
# paths, addresses ligands by SMILES only, and expresses modifications through
# bracketed residues inside the sequence itself.
_TARGETS = {
    "alphafold3": _Target(".json", _ALL_FEATURES),
    # foldjax.models.boltz2 dispatches its parser on the file suffix and rejects .json
    # outright; JSON is a YAML subset, so the document is written as .yaml.
    "boltz2": _Target(".yaml", _ALL_FEATURES - {"paired_msa"}),
    # OpenDDE consumes the Protenix list-of-jobs dialect, including its modified
    # polymer and entity/copy-addressed covalent-bond representations.
    "opendde": _Target(".json", _ALL_FEATURES),
    "protenix": _Target(".json", _ALL_FEATURES),
    # OpenFold3 expresses everything, but with two constraints its own layer
    # enforces and this writer therefore has to: alignment files are selected by
    # *stem* and only database names are parsed, and a paired MSA that cannot
    # actually be paired collapses the MSA to the query sequence alone.
    "openfold3": _Target(".json", _ALL_FEATURES),
    # Chai's FASTA carries no alignment path, so `unpaired_msa` is expressed by
    # writing Chai's own `.aligned.pqt` files beside it. See `_chai`.
    "chai": _Target(".fasta", frozenset({"ligand_smiles", "unpaired_msa"})),
}


def _ids(entity: dict[str, Any]) -> list[str]:
    value = entity.get("id")
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and value:
        return [str(item) for item in value]
    raise ValueError("every entity requires a non-empty id")


def _path(value: Any, base: Path) -> str:
    path = Path(str(value))
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
        ccd = str(item["ccd"]).strip()
        try:
            position = int(item["position"])
        except (TypeError, ValueError) as error:
            raise ValueError("modification position must be an integer") from error
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
            chain, atom = str(chain), str(atom)
            try:
                residue = int(residue)
            except (TypeError, ValueError) as error:
                raise ValueError("bond residue index must be an integer") from error
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
    unknown_job = set(job) - _JOB_KEYS
    if unknown_job:
        raise ValueError(f"unsupported top-level fields: {sorted(unknown_job)}")
    entities = job.get("entities")
    if not isinstance(entities, list) or not entities:
        raise ValueError("FoldJAX input requires a non-empty entities list")

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
        unknown = set(entity) - allowed
        if unknown:
            raise ValueError(f"unsupported {kind} entity fields: {sorted(unknown)}")
        for chain_id in _ids(entity):
            if chain_id in chains:
                raise ValueError(f"duplicate chain id: {chain_id!r}")
            chains.add(chain_id)

        if kind == "ligand":
            if not (entity.get("ccd") or entity.get("smiles")):
                raise ValueError("ligand entity requires ccd or smiles")
            if entity.get("ccd") and entity.get("smiles"):
                raise ValueError("ligand entity accepts either ccd or smiles, not both")
            feature = "ligand_ccd" if entity.get("ccd") else "ligand_smiles"
            if feature not in target.features:
                _reject(model, feature, "supply the other ligand representation")
            continue

        if not entity.get("sequence"):
            raise ValueError(f"{kind} entity requires sequence")
        for feature in ("unpaired_msa", "paired_msa", "modifications"):
            if entity.get(feature) and feature not in target.features:
                _reject(model, feature, f"remove it from entity {_ids(entity)[0]!r}")
        _modifications(entity)

    if job.get("bonds") and "bonds" not in target.features:
        _reject(model, "bonds", "use that model's native input format instead")
    _bonds(job, chains)


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
            # failing a job the other four models accept.
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
    return native


def _protenix(job: dict[str, Any], base: Path, seed: int) -> list[dict[str, Any]]:
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


#: Chai reads user-supplied alignments from a directory of ``.aligned.pqt``
#: files rather than from its input file, so the materialized job writes them
#: beside the FASTA under this name and the backend looks for it there.
CHAI_MSA_DIRNAME = "chai_msa"


def _chai(job: dict[str, Any], base: Path, output_dir: Path) -> str:
    """Render Chai's public FASTA contract, one record per chain.

    Chai's native input is a FASTA, which has nowhere to put an alignment
    path. Any ``unpaired_msa`` in the job is therefore converted to the
    ``.aligned.pqt`` form Chai reads and written to a sibling directory, so a
    job that pins an alignment pins it for Chai too instead of quietly leaving
    Chai on a different one.
    """
    records = []
    alignments: list[tuple[str, Path]] = []
    for entity in job["entities"]:
        kind = entity["type"]
        sequence = (
            str(entity["smiles"]) if kind == "ligand" else str(entity["sequence"])
        )
        if kind != "ligand" and entity.get("unpaired_msa"):
            alignments.append((sequence, Path(_path(entity["unpaired_msa"], base))))
        for chain_id in _ids(entity):
            records.append(f">{kind}|name={chain_id}\n{sequence}")

    if alignments:
        from foldjax.models.chai.data.msa import write_aligned_pqt_from_a3m

        destination = Path(output_dir) / CHAI_MSA_DIRNAME
        for _sequence, alignment in alignments:
            write_aligned_pqt_from_a3m(
                alignment.read_text(encoding="utf-8"), destination
            )
    return "\n".join(records) + "\n"


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


def _openfold3_msa(value: Any, base: Path, field: str) -> list[str]:
    path = Path(_path(value, base))
    if path.suffix and path.stem not in _OPENFOLD3_MSA_STEMS:
        raise ValueError(
            f"{field} entry {path.name!r} would be ignored: OpenFold3 selects "
            f"alignment files by stem and accepts only {sorted(_OPENFOLD3_MSA_STEMS)}"
        )
    return [str(path)]


def _openfold3(job: dict[str, Any], base: Path) -> dict[str, Any]:
    """Build OpenFold3's inference query document."""
    chains: list[dict[str, Any]] = []
    known: set[str] = set()
    for entity in job["entities"]:
        kind = entity["type"]
        ids = _ids(entity)
        known.update(ids)
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
                    entity["unpaired_msa"], base, "unpaired_msa"
                )
            if entity.get("paired_msa"):
                body["paired_msa_file_paths"] = _openfold3_msa(
                    entity["paired_msa"], base, "paired_msa"
                )
            modifications = _modifications(entity)
            if modifications:
                # OpenFold3 names these by residue rather than by PTM type.
                body["non_canonical_residues"] = {
                    position: ccd for ccd, position in modifications
                }
        chains.append(body)

    query: dict[str, Any] = {"chains": chains}
    bonds = _bonds(job, known)
    if bonds:
        query["bonds"] = [[list(first), list(second)] for first, second in bonds]
    return {"queries": {str(job.get("name", "query")): query}}


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
) -> Path:
    """Translate a FoldJAX JSON document to one backend-native input file."""
    model = capabilities.model
    target = _TARGETS.get(model)
    if target is None:
        raise ValueError(f"unsupported model: {model}")
    source = Path(source)
    job = read_job_document(source)
    if not isinstance(job, dict):
        raise ValueError("a FoldJAX job must be a JSON or YAML mapping")
    _validate(job, model, target, capabilities.entity_types)

    base = source.parent
    if model == "alphafold3":
        document: Any = _alphafold3(job, base, seed)
    elif model == "boltz2":
        document = _boltz(job, base)
    elif model in {"opendde", "protenix"}:
        document = _protenix(job, base, seed)
    elif model == "openfold3":
        document = _openfold3(job, base)
    else:
        document = None  # written after output_dir exists; Chai needs it

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if document is None:
        document = _chai(job, base, output_dir)
    path = output_dir / f"{model}_input{target.suffix}"
    text = document if isinstance(document, str) else json.dumps(document, indent=2)
    path.write_text(text, encoding="utf-8")
    return path
