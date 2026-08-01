"""Native ModelCIF serialization matching Chai's public structure output."""

from __future__ import annotations

import string
from collections import defaultdict
from collections.abc import Mapping
from io import StringIO
from typing import TYPE_CHECKING

import gemmi
import modelcif
import numpy as np
from ihm import (
    DNAChemComp,
    LPeptideChemComp,
    NonPolymerChemComp,
    RNAChemComp,
    SaccharideChemComp,
)
from modelcif import Assembly, AsymUnit, Entity, dumper, model

from foldjax.models.chai.data.input import EntityType

if TYPE_CHECKING:
    from foldjax.models.chai.data.structure import StructureContext


class _LocalPLDDT(modelcif.qa_metric.Local, modelcif.qa_metric.PLDDT):
    name = "pLDDT"
    software = None
    description = "Predicted lddt"


_CHAIN_VOCAB = [*string.ascii_uppercase, *string.ascii_lowercase]
_CHAIN_VOCAB += [x + y for x in _CHAIN_VOCAB for y in _CHAIN_VOCAB]
_AA_3_TO_1 = dict(
    zip(
        (
            "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER "
            "THR TRP TYR VAL"
        ).split(),
        "ARNDCQEGHILKMFPSTWYV",
        strict=True,
    )
)


def chain_id(asym_id: int) -> str:
    """Return Chai's one- then two-letter chain identifier."""

    if asym_id <= 0 or asym_id > len(_CHAIN_VOCAB):
        raise ValueError(f"asym_id is outside the Chai chain vocabulary: {asym_id}")
    return _CHAIN_VOCAB[asym_id - 1]


def _decode_tensorcode(value: np.ndarray) -> str:
    return "".join(chr(int(code)) for code in value if int(code) != 255).rstrip()


def _chem_component(residue_name: str, entity_type: EntityType, asym_id: int):
    if entity_type is EntityType.LIGAND:
        return NonPolymerChemComp(id=f"{residue_name}{asym_id}")
    if entity_type is EntityType.MANUAL_GLYCAN:
        return SaccharideChemComp(id=residue_name, name=residue_name)
    if entity_type is EntityType.PROTEIN:
        code = _AA_3_TO_1.get(residue_name, residue_name)
        canonical = gemmi.find_tabulated_residue(residue_name).one_letter_code
        return LPeptideChemComp(residue_name, code, code_canonical=canonical)
    if entity_type is EntityType.DNA:
        return DNAChemComp(residue_name, residue_name, code_canonical=residue_name[-1])
    if entity_type is EntityType.RNA:
        return RNAChemComp(residue_name, residue_name, code_canonical=residue_name)
    raise NotImplementedError(f"Cannot write entity type: {entity_type.name}")


def _chain_metadata(
    context: StructureContext,
    asym_entity_names: Mapping[int, str] | None,
) -> dict[int, AsymUnit]:
    valid_tokens = np.asarray(context.token_exists_mask, bool)
    asym_ids = sorted(
        int(value)
        for value in np.unique(context.token_asym_id[valid_tokens])
        if int(value) != 0
    )
    names = (
        {asym_id: chain_id(asym_id) for asym_id in asym_ids}
        if asym_entity_names is None
        else dict(asym_entity_names)
    )
    if set(names) != set(asym_ids):
        raise ValueError("asym_entity_names keys must match non-padding asym IDs")
    if len(set(names.values())) != len(names):
        raise ValueError("ModelCIF chain names must be unique")

    entities: dict[tuple[object, ...], Entity] = {}
    result: dict[int, AsymUnit] = {}
    for asym_id in asym_ids:
        token_indices = np.flatnonzero(
            valid_tokens & (context.token_asym_id == asym_id)
        )
        entity_values = np.unique(context.token_entity_type[token_indices])
        if entity_values.size != 1:
            raise ValueError(f"asym_id {asym_id} contains multiple entity types")
        entity_type = EntityType(int(entity_values[0]))
        residue_indices = context.token_residue_index[token_indices].astype(int)
        if residue_indices.min(initial=0) < 0:
            raise ValueError("residue indices must be non-negative")
        representatives: dict[int, int] = {}
        for token_index, residue_index in zip(
            token_indices, residue_indices, strict=True
        ):
            representatives.setdefault(int(residue_index), int(token_index))
        if sorted(representatives) != list(range(max(representatives) + 1)):
            raise ValueError(f"asym_id {asym_id} has missing residue indices")
        sequence = [
            _decode_tensorcode(context.token_residue_name[representatives[index]])
            for index in range(len(representatives))
        ]
        key: tuple[object, ...] = (
            (entity_type.value, asym_id)
            if entity_type is EntityType.LIGAND
            else (entity_type.value, *sequence)
        )
        entity = entities.get(key)
        if entity is None:
            entity = Entity(
                sequence=[
                    _chem_component(residue, entity_type, asym_id)
                    for residue in sequence
                ],
                description=f"Entity {names[asym_id]}",
            )
            entities[key] = entity
        result[asym_id] = AsymUnit(
            entity=entity,
            details=f"Chain {names[asym_id]}",
            id=names[asym_id],
        )
    return result


def modelcif_text(
    coords: np.ndarray,
    plddts: np.ndarray | None,
    context: StructureContext,
    *,
    atom_indices: np.ndarray | None = None,
    asym_entity_names: Mapping[int, str] | None = None,
) -> str:
    """Serialize one unpadded prediction to Chai-compatible ModelCIF text."""

    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (atom, 3)")
    if atom_indices is None:
        atom_indices = np.arange(context.num_atoms, dtype=np.int64)
    else:
        atom_indices = np.asarray(atom_indices, dtype=np.int64)
    if atom_indices.shape != (coords.shape[0],):
        raise ValueError("atom_indices must identify every coordinate")
    if np.any(atom_indices < 0) or np.any(atom_indices >= context.num_atoms):
        raise ValueError("atom_indices are outside the structure context")
    if plddts is not None:
        plddts = np.asarray(plddts)
        if plddts.shape != (coords.shape[0],):
            raise ValueError("plddts must have shape (atom,)")

    asym_units = _chain_metadata(context, asym_entity_names)
    assembly = Assembly(elements=asym_units.values(), name="Assembly 1")
    output_model = model.AbInitioModel(assembly, name="pred_model_1")
    selected_index = {
        int(original): index for index, original in enumerate(atom_indices)
    }
    ligand_name_counts: dict[tuple[int, str], int] = defaultdict(int)

    for local_index, atom_index_value in enumerate(atom_indices):
        atom_index = int(atom_index_value)
        if not bool(context.atom_exists_mask[atom_index]):
            continue
        token_index = int(context.atom_token_index[atom_index])
        asym_id = int(context.token_asym_id[token_index])
        entity_type = EntityType(int(context.token_entity_type[token_index]))
        atom_name = context.atom_ref_name[atom_index]
        atom_id = atom_name
        is_ligand = entity_type is EntityType.LIGAND
        if is_ligand:
            ligand_name_counts[asym_id, atom_name] += 1
            atom_id = f"{atom_name}_{ligand_name_counts[asym_id, atom_name]}"
        x, y, z = (float(value) for value in coords[local_index])
        output_model.add_atom(
            model.Atom(
                asym_unit=asym_units[asym_id],
                type_symbol=gemmi.Element(int(context.atom_ref_element[atom_index])).name,
                seq_id=int(context.token_residue_index[token_index]) + 1,
                atom_id=atom_id,
                x=x,
                y=y,
                z=z,
                het=is_ligand,
                biso=None if plddts is None else float(plddts[local_index]),
                occupancy=1.0,
            )
        )

    if plddts is not None:
        valid_tokens = np.asarray(context.token_exists_mask, bool)
        for asym_id, asym_unit in asym_units.items():
            token_indices = np.flatnonzero(
                valid_tokens & (context.token_asym_id == asym_id)
            )
            entity_type = EntityType(int(context.token_entity_type[token_indices[0]]))
            if entity_type is EntityType.LIGAND:
                continue
            for token_index in token_indices:
                centre = int(context.token_centre_atom_index[token_index])
                local_index = selected_index.get(centre)
                if local_index is None or not bool(context.atom_exists_mask[centre]):
                    continue
                residue = int(context.token_residue_index[token_index]) + 1
                output_model.qa_metrics.append(
                    _LocalPLDDT(asym_unit.residue(residue), float(plddts[local_index]))
                )

    system = modelcif.System(title="Chai-1 predicted structure")
    system.authors = ["Chai Discovery team"]
    system.model_groups.append(model.ModelGroup([output_model], name="pred"))
    stream = StringIO()
    dumper.write(stream, systems=[system])
    return stream.getvalue()
