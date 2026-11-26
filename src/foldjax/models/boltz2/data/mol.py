import pickle
from typing import Any, BinaryIO

from rdkit.Chem import Mol

from foldjax.models.boltz2.data import const
from foldjax.models.boltz2.data.identifiers import resolve_molecule_pickle


class _MolUnpickler(pickle.Unpickler):
    """Admit only RDKit's ``Mol`` global, as ESMFold2's CCD reader does.

    A molecule pickle (the CCD directory beside the weights, or the
    ``extra_mols`` file preprocessing writes) holds ``Mol`` objects and
    nothing else; RDKit keeps their properties inside its own binary. A plain
    ``pickle.load`` would resolve whatever global the file named.
    """

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == ("rdkit.Chem.rdchem", "Mol"):
            return Mol
        raise pickle.UnpicklingError(
            f"molecule pickle names an unexpected global {module}.{name}"
        )


def load_mol_pickle(handle: BinaryIO) -> Any:
    """Read one molecule pickle with only ``rdkit.Chem.rdchem.Mol`` allowed."""
    return _MolUnpickler(handle).load()


def load_molecules(moldir: str, molecules: list[str]) -> dict[str, Mol]:
    """Load the given input data.

    Parameters
    ----------
    moldir : str
        The path to the molecules directory.
    molecules : list[str]
        The molecules to load.

    Returns
    -------
    dict[str, Mol]
        The loaded molecules.
    """
    loaded_mols = {}
    for molecule in molecules:
        path = resolve_molecule_pickle(moldir, molecule)
        with path.open("rb") as f:
            loaded_mols[molecule] = load_mol_pickle(f)
    return loaded_mols


def load_canonicals(moldir: str) -> dict[str, Mol]:
    """Load the given input data.

    Parameters
    ----------
    moldir : str
        The molecules to load.

    Returns
    -------
    dict[str, Mol]
        The loaded molecules.

    """
    return load_molecules(moldir, const.canonical_tokens)
