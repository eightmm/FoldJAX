"""Original-style structure and confidence output for Protenix JAX."""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import gemmi
import numpy as np

from foldjax.models._output_validation import require_finite_coordinates
from foldjax.models.protenix.data.static_io import save_output_npz

_RESTYPE_NAMES = (
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",
    "A",
    "G",
    "C",
    "U",
    "UNK",
    "DA",
    "DG",
    "DC",
    "DT",
    "UNK",
    "UNK",
)
_ELEMENTS = (
    "H HE LI BE B C N O F NE NA MG AL SI P S CL AR K CA SC TI V CR MN FE CO "
    "NI CU ZN GA GE AS SE BR KR RB SR Y ZR NB MO TC RU RH PD AG CD IN SN SB "
    "TE I XE CS BA LA CE PR ND PM SM EU GD TB DY HO ER TM YB LU HF TA W RE OS "
    "IR PT AU HG TL PB BI PO AT RN FR RA AC TH PA U NP PU AM CM BK CF ES FM "
    "MD NO LR RF DB SG BH HS MT DS RG CN NH FL MC LV TS OG"
).split()
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def write_protenix_outputs(
    root: str | Path,
    *,
    job_name: str,
    seed: int,
    output: dict[str, Any],
    features: dict[str, Any],
    include_raw: bool = False,
    include_trunk: bool = False,
) -> list[Path]:
    """Write ranked CIF and summary JSON files in the upstream directory layout.

    Featurizers can preserve exact chemical labels by supplying the optional
    per-atom arrays ``output_atom_name``, ``output_atom_element``,
    ``output_atom_res_name``, ``output_atom_chain_id``, and
    ``output_atom_res_id``. Existing static features are supported through a
    deterministic fallback reconstructed from encoded model features.
    """

    coordinates = np.asarray(output.get("coordinate"))
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError("coordinate must have shape (num_samples, n_atom, 3)")
    require_finite_coordinates(coordinates, model="Protenix/OpenDDE")
    if "atom_plddt" not in output:
        raise ValueError("original-style output requires atom_plddt confidence")
    atom_plddt = np.asarray(output["atom_plddt"])
    if atom_plddt.shape != coordinates.shape[:2]:
        raise ValueError("atom_plddt must have shape (num_samples, n_atom)")

    safe_name = sanitize_job_name(job_name)
    prediction_dir = Path(root) / safe_name / f"seed_{int(seed)}" / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    raw_covalent_bonds = _raw_covalent_bonds(features)
    metadata = _atom_metadata(
        features,
        coordinates.shape[1],
        require_provenance=len(raw_covalent_bonds) > 0,
    )
    covalent_bonds = _validated_covalent_bonds(
        raw_covalent_bonds,
        metadata,
        coordinates.shape[1],
    )
    ranks = _sample_ranks(output, coordinates.shape[0])
    paths: list[Path] = []
    for sample_index, rank in enumerate(ranks):
        cif_path = prediction_dir / f"{safe_name}_sample_{rank}.cif"
        _write_cif(
            cif_path,
            name=safe_name,
            coordinates=coordinates[sample_index],
            b_factors=atom_plddt[sample_index] * 100.0,
            metadata=metadata,
            covalent_bonds=covalent_bonds,
        )
        confidence_path = (
            prediction_dir / f"{safe_name}_summary_confidence_sample_{rank}.json"
        )
        confidence_path.write_text(
            json.dumps(
                _sample_summary(output, sample_index, coordinates.shape[0]),
                indent=4,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        paths.extend((cif_path, confidence_path))
    if include_raw:
        raw_path = prediction_dir / "raw_output.npz"
        omitted = save_output_npz(raw_path, output, include_trunk=include_trunk)
        for name, nbytes in omitted:
            # Warned rather than dropped in silence: the array is genuinely absent
            # from the file, and a caller reading it back would otherwise see a
            # missing key with no explanation.
            warnings.warn(
                f"{raw_path.name} omits {name} ({nbytes / 2**30:.1f} GiB); "
                "confidence outputs are quadratic in token count",
                RuntimeWarning,
                stacklevel=2,
            )
        paths.append(raw_path)
    return paths


def sanitize_job_name(name: str) -> str:
    """Return the filesystem-safe name used by original-style outputs."""

    safe = _SAFE_NAME.sub("_", str(name).strip()).strip("._")
    return safe or "prediction"


def _sample_ranks(output: dict[str, Any], num_samples: int) -> np.ndarray:
    if "summary_ranking_score" in output:
        score = np.asarray(output["summary_ranking_score"])
        if score.shape == (num_samples,):
            order = np.argsort(-score, kind="stable")
            ranks = np.empty(num_samples, dtype=np.int64)
            ranks[order] = np.arange(num_samples)
            return ranks
    return np.arange(num_samples, dtype=np.int64)


def _sample_summary(
    output: dict[str, Any], sample_index: int, num_samples: int
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in output.items():
        if not (
            key.startswith("summary_")
            or key.startswith("chain_")
            or key.startswith("has_")
            or key.startswith("pb_")
            or key in {"disorder", "ranking_score", "num_recycles"}
        ):
            continue
        array = np.asarray(value)
        json_key = key.removeprefix("summary_")
        if key == "num_recycles" and array.ndim == 0:
            summary[json_key] = _json_value(array)
            continue
        if array.ndim == 0 or array.shape[0] != num_samples:
            continue
        summary[json_key] = _json_value(array[sample_index])
    return summary


def _json_value(value: Any) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        item = array.item()
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    return _json_safe_list(array.tolist())


def _json_safe_list(value: Any) -> Any:
    if isinstance(value, list):
        return [_json_safe_list(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atom_metadata(
    features: dict[str, Any],
    n_atom: int,
    *,
    require_provenance: bool = False,
) -> dict[str, np.ndarray]:
    explicit_keys = {
        "name": "output_atom_name",
        "element": "output_atom_element",
        "res_name": "output_atom_res_name",
        "chain_id": "output_atom_chain_id",
        "res_id": "output_atom_res_id",
    }
    if any(key in features for key in explicit_keys.values()):
        missing = [key for key in explicit_keys.values() if key not in features]
        if missing:
            raise ValueError(
                "output atom metadata contract is incomplete; missing "
                + ", ".join(missing)
            )
        metadata = {
            name: np.asarray(features[key]) for name, key in explicit_keys.items()
        }
        _validate_metadata_lengths(metadata, n_atom)
    else:
        atom_to_token = np.asarray(features["atom_to_token_idx"], dtype=np.int64)
        if atom_to_token.shape != (n_atom,):
            raise ValueError("atom_to_token_idx must have shape (n_atom,)")
        token_restype = np.argmax(np.asarray(features["restype"]), axis=-1)
        token_asym = np.asarray(features["asym_id"], dtype=np.int64)
        token_residue = np.asarray(features["residue_index"], dtype=np.int64)
        metadata = {
            "name": _decode_atom_names(np.asarray(features["ref_atom_name_chars"])),
            "element": _decode_elements(np.asarray(features["ref_element"])),
            "res_name": np.asarray(
                [_RESTYPE_NAMES[int(i)] for i in token_restype[atom_to_token]],
                dtype="U3",
            ),
            "chain_id": np.asarray(
                [_chain_id(int(i)) for i in token_asym[atom_to_token]], dtype="U8"
            ),
            "res_id": token_residue[atom_to_token],
        }
        _validate_metadata_lengths(metadata, n_atom)

    provenance_keys = {
        "entity_id": "atom_entity_id",
        "polymer_type": "output_atom_polymer_type",
    }
    missing_provenance = [
        key for key in provenance_keys.values() if key not in features
    ]
    if require_provenance and missing_provenance:
        raise ValueError(
            "covalent output metadata is incomplete; missing "
            + ", ".join(missing_provenance)
        )
    if not missing_provenance:
        metadata.update(
            {name: np.asarray(features[key]) for name, key in provenance_keys.items()}
        )
        _validate_metadata_lengths(metadata, n_atom)
    return metadata


def _raw_covalent_bonds(features: dict[str, Any]) -> np.ndarray:
    raw = np.asarray(
        features.get("covalent_atom_indices", np.empty((0, 2), dtype=np.int64))
    )
    if raw.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError("covalent_atom_indices must have shape (n_bond, 2)")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("covalent_atom_indices must contain integer atom indices")
    return raw.astype(np.int64, copy=False)


def _validated_covalent_bonds(
    bonds: np.ndarray,
    metadata: dict[str, np.ndarray],
    n_atom: int,
) -> np.ndarray:
    if len(bonds) == 0:
        return bonds
    if np.any((bonds < 0) | (bonds >= n_atom)):
        raise ValueError("covalent_atom_indices contains an out-of-range atom index")
    if "entity_id" not in metadata or "polymer_type" not in metadata:
        raise ValueError("covalent output metadata is incomplete")

    atom_addresses = [
        (
            str(metadata["chain_id"][index]),
            int(metadata["res_id"][index]),
            str(metadata["res_name"][index]),
            str(metadata["name"][index]),
        )
        for index in range(n_atom)
    ]
    address_counts: dict[tuple[str, int, str, str], int] = {}
    for address in atom_addresses:
        address_counts[address] = address_counts.get(address, 0) + 1

    kept: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for atom1, atom2 in bonds:
        index1, index2 = int(atom1), int(atom2)
        residue1 = atom_addresses[index1][:3]
        residue2 = atom_addresses[index2][:3]
        if residue1 == residue2:
            # Upstream suppresses intra-residue bonds when include_bonds=False.
            continue
        for index in (index1, index2):
            if address_counts[atom_addresses[index]] != 1:
                raise ValueError(
                    "covalent bond endpoint does not have a unique CIF atom address"
                )
        canonical = (min(index1, index2), max(index1, index2))
        if canonical not in seen:
            kept.append((index1, index2))
            seen.add(canonical)
    return np.asarray(kept, dtype=np.int64).reshape((-1, 2))


def _validate_metadata_lengths(metadata: dict[str, np.ndarray], n_atom: int) -> None:
    for key, value in metadata.items():
        if value.shape != (n_atom,):
            raise ValueError(f"atom metadata {key} must have shape (n_atom,)")


def _decode_atom_names(encoded: np.ndarray) -> np.ndarray:
    if encoded.ndim != 3 or encoded.shape[1:] != (4, 64):
        raise ValueError("ref_atom_name_chars must have shape (n_atom, 4, 64)")
    return np.asarray(
        [
            "".join(chr(int(index) + 32) for index in np.argmax(atom, axis=-1)).strip()
            for atom in encoded
        ],
        dtype="U4",
    )


def _decode_elements(encoded: np.ndarray) -> np.ndarray:
    if encoded.ndim != 2 or encoded.shape[1] != 128:
        raise ValueError("ref_element must have shape (n_atom, 128)")
    return np.asarray(
        [
            _ELEMENTS[index] if index < len(_ELEMENTS) else "X"
            for index in np.argmax(encoded, axis=-1)
        ],
        dtype="U2",
    )


def _chain_id(index: int) -> str:
    index += 1
    chars = []
    while index:
        index, remainder = divmod(index - 1, 26)
        chars.append(chr(ord("A") + remainder))
    return "".join(reversed(chars))


def _write_cif(
    path: Path,
    *,
    name: str,
    coordinates: np.ndarray,
    b_factors: np.ndarray,
    metadata: dict[str, np.ndarray],
    covalent_bonds: np.ndarray,
) -> None:
    structure = gemmi.Structure()
    structure.name = name
    _add_cif_entities(structure, metadata)
    model = gemmi.Model("1")
    chain_order = list(dict.fromkeys(str(x) for x in metadata["chain_id"]))
    for chain_id in chain_order:
        chain = gemmi.Chain(chain_id)
        atom_indices = np.flatnonzero(metadata["chain_id"].astype(str) == chain_id)
        current_key: tuple[int, str] | None = None
        residue: gemmi.Residue | None = None
        for atom_index in atom_indices:
            key = (
                int(metadata["res_id"][atom_index]),
                str(metadata["res_name"][atom_index]),
            )
            if key != current_key:
                if residue is not None:
                    chain.add_residue(residue)
                residue = gemmi.Residue()
                residue.name = key[1]
                residue.seqid = gemmi.SeqId(key[0], " ")
                residue.subchain = chain_id
                if "entity_id" in metadata:
                    entity_ids = np.unique(metadata["entity_id"][atom_indices])
                    polymer_types = np.unique(metadata["polymer_type"][atom_indices])
                    if len(entity_ids) != 1 or len(polymer_types) != 1:
                        raise ValueError(
                            f"chain {chain_id!r} has inconsistent entity provenance"
                        )
                    residue.entity_id = str(entity_ids[0])
                    if str(polymer_types[0]) == "non-polymer":
                        residue.het_flag = "H"
                    else:
                        residue.het_flag = "A"
                        residue.label_seq = key[0]
                current_key = key
            atom = gemmi.Atom()
            atom.name = str(metadata["name"][atom_index]) or "X"
            element = str(metadata["element"][atom_index])
            atom.element = gemmi.Element(element if element != "X" else "C")
            atom.pos = gemmi.Position(*map(float, coordinates[atom_index]))
            atom.occ = 1.0
            atom.b_iso = round(float(b_factors[atom_index]), 2)
            assert residue is not None
            residue.add_atom(atom)
        if residue is not None:
            chain.add_residue(residue)
        model.add_chain(chain)
    structure.add_model(model)
    document = structure.make_mmcif_document()
    _set_covalent_struct_conn(document, metadata, covalent_bonds)
    document.write_file(str(path))


def _add_cif_entities(
    structure: gemmi.Structure, metadata: dict[str, np.ndarray]
) -> None:
    if "entity_id" not in metadata:
        return
    entity_ids = [str(value) for value in metadata["entity_id"]]
    chain_ids = [str(value) for value in metadata["chain_id"]]
    polymer_types = [str(value) for value in metadata["polymer_type"]]
    polymer_type_mapping = {
        "polypeptide(L)": gemmi.PolymerType.PeptideL,
        "polydeoxyribonucleotide": gemmi.PolymerType.Dna,
        "polyribonucleotide": gemmi.PolymerType.Rna,
    }
    for entity_id in dict.fromkeys(entity_ids):
        entity_mask = np.asarray(entity_ids) == entity_id
        entity_polymer_types = set(np.asarray(polymer_types)[entity_mask])
        if len(entity_polymer_types) != 1:
            raise ValueError(f"entity {entity_id!r} has inconsistent polymer types")
        polymer_type = entity_polymer_types.pop()
        entity = gemmi.Entity(entity_id)
        entity.subchains = list(dict.fromkeys(np.asarray(chain_ids)[entity_mask]))
        if polymer_type == "non-polymer":
            entity.entity_type = gemmi.EntityType.NonPolymer
        else:
            try:
                entity.polymer_type = polymer_type_mapping[polymer_type]
            except KeyError as exc:
                raise ValueError(
                    f"unsupported output polymer type: {polymer_type!r}"
                ) from exc
            entity.entity_type = gemmi.EntityType.Polymer
            sequences = []
            for chain_id in entity.subchains:
                atom_indices = np.flatnonzero(
                    entity_mask & (np.asarray(chain_ids) == chain_id)
                )
                residues = list(
                    dict.fromkeys(
                        (
                            int(metadata["res_id"][index]),
                            str(metadata["res_name"][index]),
                        )
                        for index in atom_indices
                    )
                )
                sequences.append([res_name for _res_id, res_name in residues])
            if any(sequence != sequences[0] for sequence in sequences[1:]):
                raise ValueError(
                    f"copies of entity {entity_id!r} have inconsistent sequences"
                )
            entity.full_sequence = sequences[0]
        structure.entities.append(entity)


def _set_covalent_struct_conn(
    document: gemmi.cif.Document,
    metadata: dict[str, np.ndarray],
    bonds: np.ndarray,
) -> None:
    if len(bonds) == 0:
        return
    category: dict[str, list[str | bool]] = {
        "id": [],
        "conn_type_id": [],
        "pdbx_value_order": [],
        "ptnr1_label_asym_id": [],
        "ptnr2_label_asym_id": [],
        "ptnr1_label_comp_id": [],
        "ptnr2_label_comp_id": [],
        "ptnr1_label_seq_id": [],
        "ptnr2_label_seq_id": [],
        "ptnr1_label_atom_id": [],
        "ptnr2_label_atom_id": [],
        "pdbx_ptnr1_PDB_ins_code": [],
        "pdbx_ptnr2_PDB_ins_code": [],
    }
    for connection_id, (atom1, atom2) in enumerate(bonds, start=1):
        category["id"].append(str(connection_id))
        category["conn_type_id"].append("covale")
        category["pdbx_value_order"].append("sing")
        for side, atom_index in ((1, int(atom1)), (2, int(atom2))):
            category[f"ptnr{side}_label_asym_id"].append(
                str(metadata["chain_id"][atom_index])
            )
            category[f"ptnr{side}_label_comp_id"].append(
                str(metadata["res_name"][atom_index])
            )
            polymer_type = str(metadata["polymer_type"][atom_index])
            category[f"ptnr{side}_label_seq_id"].append(
                str(int(metadata["res_id"][atom_index]))
                if polymer_type != "non-polymer"
                else False
            )
            category[f"ptnr{side}_label_atom_id"].append(
                str(metadata["name"][atom_index])
            )
            category[f"pdbx_ptnr{side}_PDB_ins_code"].append(False)
    document.sole_block().set_mmcif_category("_struct_conn.", category)
