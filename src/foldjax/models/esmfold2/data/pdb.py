"""Writing an ESMFold2 prediction out as PDB, without torch or OpenFold.

Upstream renders through OpenFold's `OFProtein`, which carries no chain field
and indexes a fixed 37-atom protein alphabet. That is fine for the single
chain its `infer_protein` covers and lossy for anything else: a complex comes
out as one chain, and any atom outside the 37 is dropped silently. This writer
takes the features the model was given -- `atom_to_token`, `asym_id`,
`ref_atom_name_chars` -- so chains keep their identity and every atom that was
folded is written.

One deliberate difference from upstream: the b-factor column carries pLDDT on
the **0-100** scale that every viewer and every other predictor assumes.
ESMFold2's head produces 0-1 and upstream forwards it unscaled, so a file
written by upstream colours as though nothing were confident. Pass
`plddt_scale=1.0` to reproduce that.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from foldjax.models.esmfold2.data import chemistry
from foldjax.models.esmfold2.data.all_atom_constants import (
    ELEMENT_NUMBER_TO_SYMBOL,
    MOL_TYPE_NONPOLYMER,
)

#: Chain identifiers in the order PDB files conventionally use them. Past 62
#: chains a PDB file cannot name them at all, and the writer says so.
CHAIN_ALPHABET = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
)

ATOMIC_NUMBER_TO_ELEMENT = {6: "C", 7: "N", 8: "O", 16: "S", 1: "H", 15: "P"}


def decode_atom_name(codes: np.ndarray) -> str:
    """Undo the four-character `ord - 32` encoding, zeros meaning blank."""
    return "".join(
        chr(int(code) + 32) if int(code) != 0 else " " for code in codes
    ).strip()


def _decode_text(codes: np.ndarray) -> str:
    """Decode zero-padded ASCII writer metadata."""

    return bytes(int(value) for value in codes if int(value)).decode("ascii")


def _format_atom_name(name: str) -> str:
    """PDB columns 13-16: two-character element field, then the name.

    A name shorter than four characters starts in column 14, which is what
    keeps `CA` (alpha carbon) distinct from `CA` (calcium) for every reader
    that follows the format.
    """
    return name if len(name) >= 4 else f" {name:<3}"


def to_pdb(
    coords: np.ndarray,
    features: Mapping[str, np.ndarray],
    plddt_per_atom: np.ndarray | None = None,
    *,
    plddt_scale: float = 100.0,
    model_number: int | None = None,
) -> str:
    """One sample's coordinates as a PDB string.

    `coords` is `[n_atoms, 3]` for a single structure; `features` is the
    dictionary the model was given, with or without its batch axis.
    """
    take = lambda name: _drop_batch(np.asarray(features[name]))  # noqa: E731
    atom_to_token = take("atom_to_token")
    atom_mask = take("atom_attention_mask").astype(bool)
    token_mask = take("token_attention_mask").astype(bool)
    residue_index = take("residue_index")
    res_type = take("res_type")
    asym_id = take("asym_id")
    name_codes = take("ref_atom_name_chars")
    elements = take("ref_element")
    coords = _drop_batch(np.asarray(coords))

    if plddt_per_atom is None:
        b_factors = np.zeros(coords.shape[0], dtype=np.float32)
    else:
        b_factors = _drop_batch(np.asarray(plddt_per_atom)) * plddt_scale

    n_chains = int(asym_id.max()) + 1 if asym_id.size else 0
    if n_chains > len(CHAIN_ALPHABET):
        raise ValueError(
            f"{n_chains} chains cannot be named in a PDB file; write mmCIF"
        )

    lines: list[str] = []
    if model_number is not None:
        lines.append(f"MODEL     {model_number:>4}")

    serial = 0
    previous_chain: int | None = None
    for atom in range(atom_to_token.shape[0]):
        if not atom_mask[atom]:
            continue
        token = int(atom_to_token[atom])
        if not token_mask[token]:
            continue
        chain = int(asym_id[token])
        if previous_chain is not None and chain != previous_chain:
            serial += 1
            lines.append(f"TER   {serial:>5}")
        previous_chain = chain

        serial += 1
        name = decode_atom_name(name_codes[atom])
        residue = chemistry.RES_TYPE_TO_3LETTER.get(int(res_type[token]), "UNK")
        element = ATOMIC_NUMBER_TO_ELEMENT.get(int(elements[atom]), name[:1])
        x, y, z = (float(value) for value in coords[atom])
        lines.append(
            f"ATOM  {serial:>5} {_format_atom_name(name)} {residue:>3}"
            f" {CHAIN_ALPHABET[chain]}{int(residue_index[token]) + 1:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
            f"{1.00:>6.2f}{float(b_factors[atom]):>6.2f}"
            f"          {element:>2}"
        )

    serial += 1
    lines.append(f"TER   {serial:>5}")
    if model_number is not None:
        lines.append("ENDMDL")
    lines.append("END")
    return "\n".join(lines) + "\n"


def to_mmcif(
    coords: np.ndarray,
    features: Mapping[str, np.ndarray],
    plddt_per_atom: np.ndarray | None = None,
    *,
    plddt_scale: float = 100.0,
    name: str = "prediction",
) -> str:
    """Write mmCIF, retaining arbitrary chain and CCD residue identifiers.

    Legacy protein feature dictionaries do not carry writer metadata and keep
    the established PDB-through-Gemmi path. The all-biomolecule builder does:
    it is written directly because PDB cannot represent multi-character chain
    IDs and truncates the component namespace that ligands/modifications use.
    """
    import gemmi

    if {
        "token_chain_id_chars",
        "token_residue_name_chars",
    } <= features.keys():
        take = lambda key: _drop_batch(np.asarray(features[key]))  # noqa: E731
        atom_to_token = take("atom_to_token")
        atom_mask = take("atom_attention_mask").astype(bool)
        token_mask = take("token_attention_mask").astype(bool)
        residue_index = take("residue_index")
        mol_type = take("mol_type")
        atom_names = take("ref_atom_name_chars")
        elements = take("ref_element")
        chain_names = take("token_chain_id_chars")
        residue_names = take("token_residue_name_chars")
        coordinates = _drop_batch(np.asarray(coords))
        confidence = (
            np.zeros(coordinates.shape[0], dtype=np.float32)
            if plddt_per_atom is None
            else _drop_batch(np.asarray(plddt_per_atom)) * plddt_scale
        )

        structure = gemmi.Structure()
        structure.name = name
        model = gemmi.Model("1")
        chain_objects: dict[str, gemmi.Chain] = {}
        residue_objects: dict[tuple[str, int, str], gemmi.Residue] = {}
        for atom_index, owner in enumerate(atom_to_token):
            if not atom_mask[atom_index]:
                continue
            token = int(owner)
            if not token_mask[token]:
                continue
            chain_name = _decode_text(chain_names[token])
            residue_name = _decode_text(residue_names[token])
            chain = chain_objects.get(chain_name)
            if chain is None:
                model.add_chain(gemmi.Chain(chain_name))
                chain = model[-1]
                chain_objects[chain_name] = chain
            residue_key = (chain_name, int(residue_index[token]), residue_name)
            residue = residue_objects.get(residue_key)
            if residue is None:
                residue = gemmi.Residue()
                residue.name = residue_name
                residue.seqid = gemmi.SeqId(int(residue_index[token]) + 1, " ")
                residue.het_flag = (
                    "H" if int(mol_type[token]) == MOL_TYPE_NONPOLYMER else "A"
                )
                chain.add_residue(residue)
                residue = chain[-1]
                residue_objects[residue_key] = residue
            atom = gemmi.Atom()
            atom.name = decode_atom_name(atom_names[atom_index])
            symbol = ELEMENT_NUMBER_TO_SYMBOL.get(int(elements[atom_index]), "C")
            atom.element = gemmi.Element(symbol.title())
            atom.pos = gemmi.Position(
                *(float(value) for value in coordinates[atom_index])
            )
            atom.occ = 1.0
            atom.b_iso = float(confidence[atom_index])
            residue.add_atom(atom)

        # Gemmi copies the model when it is inserted. Populate the local model
        # first so the structure receives the chains and atoms rather than the
        # empty shell.
        structure.add_model(model)
        structure.setup_entities()
        document = structure.make_mmcif_document()
        document.sole_block().name = name
        return document.as_string()

    structure = gemmi.read_pdb_string(
        to_pdb(coords, features, plddt_per_atom, plddt_scale=plddt_scale)
    )
    structure.setup_entities()
    structure.name = name
    document = structure.make_mmcif_document()
    document.sole_block().name = name
    return document.as_string()


def to_pdb_models(
    coords: np.ndarray,
    features: Mapping[str, np.ndarray],
    plddt_per_atom: np.ndarray | None = None,
    *,
    plddt_scale: float = 100.0,
) -> str:
    """Several diffusion samples as numbered MODEL records in one file."""
    coords = np.asarray(coords)
    if coords.ndim == 2:
        coords = coords[None]
    plddt = None if plddt_per_atom is None else np.asarray(plddt_per_atom)
    blocks = []
    for index in range(coords.shape[0]):
        block = to_pdb(
            coords[index],
            features,
            None if plddt is None else plddt[index],
            plddt_scale=plddt_scale,
            model_number=index + 1,
        )
        blocks.append(block.removesuffix("END\n"))
    return "".join(blocks) + "END\n"


def _drop_batch(array: np.ndarray) -> np.ndarray:
    """Features arrive with a leading batch axis of one; coordinates may not."""
    return array[0] if array.ndim > 1 and array.shape[0] == 1 else array


__all__ = [
    "ATOMIC_NUMBER_TO_ELEMENT",
    "CHAIN_ALPHABET",
    "decode_atom_name",
    "to_pdb",
    "to_pdb_models",
]
