"""`foldjax check DIR`: PoseBusters plausibility checks on predicted ligands.

PoseBusters (Buttenschoen, Morris and Deane, Chem. Sci. 15:3130, 2024) is an
optional extra (``pip install 'foldjax[posebusters]'``). Each sample's mmCIF is
split into its ligands and its polymer; every ligand with at least two heavy
atoms is checked with PoseBusters' ``dock`` configuration (the ligand alone,
and against the predicted protein) and the result is one row per ligand with
``pb_valid`` -- every check passed -- and the individual ``pb.<check>``
columns.

Bond orders are not in an mmCIF's coordinates, so they are perceived from the
geometry (RDKit's ``DetermineBonds``, neutral total charge). A ligand whose
bonds cannot be perceived that way is reported with the reason rather than
checked against a guessed chemistry. These are FoldJAX-run checks on the
written structure, not scores the model reported.
"""

from __future__ import annotations

import csv
import io
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_WATER = frozenset({"HOH", "DOD", "WAT", "H2O"})


def _posebusters_missing() -> ModuleNotFoundError:
    """The refusal, with the install line `foldjax doctor` prints for an extra.

    One command, for this installation: a checkout adds the extra with
    ``uv sync --inexact`` (a bare ``uv sync --extra`` removes the others), a
    pip install with ``pip install``.
    """
    from foldjax.doctor import install_command

    return ModuleNotFoundError(
        "PoseBusters is not installed; install the optional extra with "
        f"`{install_command('posebusters')}`",
        name="posebusters",
    )


_CCD_CACHE: dict[str, str | None] = {}
_BOND_ORDER = {"SING": 1.0, "DOUB": 2.0, "TRIP": 3.0, "AROM": 1.5}


def _ccd_block(code: str) -> str | None:
    """The ``data_<code>`` block of FoldJAX's managed CCD, if it is installed."""
    if code in _CCD_CACHE:
        return _CCD_CACHE[code]
    from foldjax.paths import assets_dir

    path = assets_dir() / "components.cif"
    if not path.is_file():
        _CCD_CACHE[code] = None
        return None
    header = f"data_{code}"
    lines: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if lines and line.startswith("data_"):
                break
            if lines or line.strip() == header:
                lines.append(line)
    _CCD_CACHE[code] = "".join(lines) or None
    return _CCD_CACHE[code]


def _from_ccd(code: str, atoms: dict[str, Any]):
    """The ligand with its CCD bonds, charges and orders, at predicted positions."""
    import gemmi
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    text = _ccd_block(code)
    if text is None:
        return None
    block = gemmi.cif.read_string(text).sole_block()
    table = block.find("_chem_comp_atom.", ["atom_id", "type_symbol", "charge"])
    names, charges = [], {}
    for row in table:
        name = gemmi.cif.as_string(row[0])
        if name in atoms:
            names.append(name)
            try:
                charges[name] = int(float(gemmi.cif.as_string(row[2]) or 0))
            except ValueError:
                charges[name] = 0
    if set(names) != set(atoms):
        return None
    mol = Chem.RWMol()
    index = {}
    conformer = Chem.Conformer(len(names))
    for k, name in enumerate(names):
        atom = Chem.Atom(atoms[name].element.name.capitalize())
        atom.SetFormalCharge(charges[name])
        index[name] = mol.AddAtom(atom)
        position = atoms[name].pos
        conformer.SetAtomPosition(k, Point3D(position.x, position.y, position.z))
    for row in block.find(
        "_chem_comp_bond.", ["atom_id_1", "atom_id_2", "value_order"]
    ):
        a, b = gemmi.cif.as_string(row[0]), gemmi.cif.as_string(row[1])
        if a in index and b in index:
            order = _BOND_ORDER.get(gemmi.cif.as_string(row[2]).upper(), 1.0)
            kind = {
                1.0: Chem.BondType.SINGLE,
                2.0: Chem.BondType.DOUBLE,
                3.0: Chem.BondType.TRIPLE,
                1.5: Chem.BondType.AROMATIC,
            }[order]
            mol.AddBond(index[a], index[b], kind)
    mol.AddConformer(conformer, assignId=True)
    mol = mol.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def _ligand_mol(residue: Any, smiles: str | None = None):
    """An RDKit molecule for one ligand residue, and how its bonds were found.

    In order: the residue's CCD definition (FoldJAX's managed
    ``components.cif``), a SMILES the file itself records for the component,
    and bond perception from the coordinates.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDetermineBonds
    from rdkit.Geometry import Point3D

    atoms = {atom.name: atom for atom in residue if atom.element.name not in ("H", "D")}
    templated = _from_ccd(residue.name, atoms)
    if templated is not None:
        return templated, "ccd"
    mol = Chem.RWMol()
    conformer = Chem.Conformer(len(atoms))
    for k, atom in enumerate(atoms.values()):
        mol.AddAtom(Chem.Atom(atom.element.name.capitalize()))
        conformer.SetAtomPosition(k, Point3D(atom.pos.x, atom.pos.y, atom.pos.z))
    mol.AddConformer(conformer, assignId=True)
    mol = mol.GetMol()
    if smiles:
        template = Chem.MolFromSmiles(smiles)
        if template is not None:
            rdDetermineBonds.DetermineConnectivity(mol)
            try:
                assigned = AllChem.AssignBondOrdersFromTemplate(template, mol)
                Chem.SanitizeMol(assigned)
                return assigned, "file smiles"
            except (ValueError, RuntimeError):
                mol = Chem.RWMol(mol)
                for bond in list(mol.GetBonds()):
                    mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                mol = mol.GetMol()
    rdDetermineBonds.DetermineBonds(mol, charge=0)
    Chem.SanitizeMol(mol)
    return mol, "geometry"


def _file_smiles(structure: Any) -> dict[str, str]:
    """``_chem_comp`` SMILES a writer recorded (AlphaFold 3 does), by component."""
    import gemmi

    found: dict[str, str] = {}
    try:
        document = gemmi.cif.read(str(structure))
    except Exception:  # noqa: BLE001 - optional enrichment only
        return found
    block = document.sole_block()
    for tag in ("_chem_comp.pdbx_smiles", "_chem_comp.smiles"):
        table = block.find(["_chem_comp.id", tag])
        for row in table:
            code, smiles = gemmi.cif.as_string(row[0]), gemmi.cif.as_string(row[1])
            if smiles and smiles not in ("?", "."):
                found.setdefault(code, smiles)
    return found


def _split(structure_path: Path, scratch: Path):
    """Ligand residues and a PDB of the polymer part, from one mmCIF."""
    import gemmi

    from foldjax.structure_format import pdb_violations

    structure = gemmi.read_structure(str(structure_path))
    structure.setup_entities()
    structure.remove_alternative_conformations()
    ligands = []
    for chain in structure[0]:
        for residue in chain:
            if (
                residue.entity_type != gemmi.EntityType.Polymer
                and residue.name not in _WATER
                and sum(a.element.name not in ("H", "D") for a in residue) >= 2
            ):
                ligands.append(
                    (chain.name, residue.name, int(residue.seqid.num), residue)
                )
    protein = structure.clone()
    for chain in protein[0]:
        for k in reversed(range(len(chain))):
            if chain[k].entity_type != gemmi.EntityType.Polymer:
                del chain[k]
    protein.remove_empty_chains()
    protein_path = None
    if not pdb_violations(protein):
        protein_path = scratch / "protein.pdb"
        protein_path.write_text(protein.make_pdb_string(), encoding="utf-8")
    return ligands, protein_path


def check_structure(
    structure_path: str | os.PathLike[str], *, buster: Any = None
) -> list[dict[str, Any]]:
    """PoseBusters rows for every ligand of one structure."""
    try:
        from posebusters import PoseBusters
    except ImportError as error:
        raise _posebusters_missing() from error
    structure_path = Path(structure_path)
    rows: list[dict[str, Any]] = []
    smiles = _file_smiles(structure_path)
    with tempfile.TemporaryDirectory(prefix="foldjax-check-") as scratch:
        ligands, protein = _split(structure_path, Path(scratch))
        for chain, name, number, residue in ligands:
            row: dict[str, Any] = {"chain": chain, "ligand": name, "residue": number}
            try:
                mol, row["pb_bonds_from"] = _ligand_mol(residue, smiles.get(name))
            except Exception as error:  # noqa: BLE001 - RDKit raises several types
                row["pb_valid"] = None
                row["pb_error"] = (
                    "no bond orders: not in the managed CCD, no SMILES in the "
                    f"file, and perception from coordinates failed ({error})"
                )
                rows.append(row)
                continue
            config = "dock" if protein is not None else "mol"
            if protein is None:
                row["pb_note"] = (
                    "the polymer exceeds the PDB format, so only the ligand-only "
                    "('mol') checks ran"
                )
            tool = buster or PoseBusters(config=config)
            frame = tool.bust(mol_pred=mol, mol_cond=protein if protein else None)
            record = frame.iloc[0].to_dict()
            checks: dict[str, bool | None] = {}
            for key, value in record.items():
                if isinstance(value, bool) or type(value).__name__ == "bool_":
                    checks[str(key)] = bool(value)
                elif value is None or (isinstance(value, float) and value != value):
                    # A check PoseBusters could not compute (it logs why).
                    checks[str(key)] = None
            # Not computed is not passed: PB-valid means every check passed.
            row["pb_valid"] = bool(checks) and all(v is True for v in checks.values())
            row["pb_failed"] = sorted(k for k, v in checks.items() if v is False)
            row["pb_not_computed"] = sorted(k for k, v in checks.items() if v is None)
            row["pb_config"] = config
            for key, value in checks.items():
                row[f"pb.{key}"] = value
            rows.append(row)
    return rows


def check_directory(root: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """One row per ligand per sample of a finished output directory."""
    from foldjax.results import load_results

    try:
        import posebusters  # noqa: F401
    except ImportError as error:
        raise _posebusters_missing() from error
    report = load_results(root)
    rows: list[dict[str, Any]] = []
    for run, sample in report.samples():
        base = {
            "model": run.model,
            "input": run.input,
            "configuration": run.configuration,
            "job": sample.job,
            "seed": sample.seed,
            "sample": sample.sample,
            "structure_path": str(sample.structure_path)
            if sample.structure_path
            else None,
        }
        if sample.structure_path is None or not sample.structure_verified:
            rows.append({**base, "pb_error": "structure missing or changed"})
            continue
        found = check_structure(sample.structure_path)
        if not found:
            rows.append({**base, "pb_error": "no ligand in the structure"})
        rows.extend({**base, **row} for row in found)
    return rows


def to_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    from foldjax.results import _cell, _columns

    buffer = io.StringIO()
    columns = _columns(rows)
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: _cell(row.get(name)) for name in columns})
    return buffer.getvalue()


def render_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        f"{'model':<11s}{'seed':>5s}{'smp':>4s}  {'ligand':<8s}{'valid':<7s}failed"
    ]
    for row in rows:
        head = (
            f"{str(row.get('model')):<11s}{str(row.get('seed')):>5s}"
            f"{str(row.get('sample')):>4s}  "
        )
        if row.get("pb_error"):
            lines.append(
                f"{head}{str(row.get('ligand') or '-'):<8s}-      {row['pb_error']}"
            )
            continue
        valid = row.get("pb_valid")
        lines.append(
            f"{head}{str(row.get('ligand')):<8s}{('yes' if valid else 'no'):<7s}"
            + ", ".join(row.get("pb_failed") or [])
        )
    return "\n".join(lines)
