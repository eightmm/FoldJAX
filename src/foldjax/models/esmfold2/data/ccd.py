"""Trusted Biohub CCD chemistry for the ESMFold2 NumPy featurizer.

Biohub publishes ``ccd.pkl`` beside the ESMFold2 structure checkpoint.  The
file is a dictionary of RDKit molecules with atom names, leaving-atom flags,
formal charges, bonds and three conformers.  FoldJAX downloads it with a pinned
size and SHA-256 before this reader sees it; arbitrary job files never choose a
pickle path.

The store is lazy because loading roughly fifty thousand components takes a
few seconds and protein-only jobs retain the historical in-package chemistry
path without touching the CCD at all.
"""

from __future__ import annotations

import hashlib
import pickle
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from foldjax.models.esmfold2.data.all_atom_constants import RES_TYPE_TO_CCD

BIOHUB_CCD_SHA256 = (
    "9ff44b1927c6b9198e38ffe0928706827a09a350c15530beeeabebfa88038fc5"
)
BIOHUB_CCD_SIZE = 417_306_584
_HASH_CHUNK = 1 << 20
_DERIVED_COMPONENT_CACHE_LIMIT = 256


class CCDStore:
    """Lazy, path-bound view of Biohub's verified RDKit molecule dictionary."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._molecules: dict[str, Any] | None = None
        self._molecule_cache: OrderedDict[str, Any] = OrderedDict()
        self._conformer_cache: OrderedDict[str, Any] = OrderedDict()
        self._atom_cache: OrderedDict[str, Any] = OrderedDict()
        self._bond_cache: OrderedDict[str, Any] = OrderedDict()
        self._leaving_cache: OrderedDict[str, Any] = OrderedDict()

    @staticmethod
    def _cached(cache: OrderedDict[str, Any], component: str) -> tuple[bool, Any]:
        """Read and refresh one entry, including a cached ``None``."""

        try:
            value = cache.pop(component)
        except KeyError:
            return False, None
        cache[component] = value
        return True, value

    @staticmethod
    def _remember(cache: OrderedDict[str, Any], component: str, value: Any) -> Any:
        """Retain one derived component value without growing for every job."""

        cache[component] = value
        cache.move_to_end(component)
        while len(cache) > _DERIVED_COMPONENT_CACHE_LIMIT:
            cache.popitem(last=False)
        return value

    def _load(self) -> dict[str, Any]:
        if self._molecules is not None:
            return self._molecules
        if not self.path.is_file():
            raise FileNotFoundError(
                "ESMFold2 all-biomolecule featurization requires Biohub's "
                f"ccd.pkl beside the structure checkpoint; missing: {self.path}. "
                "Run `foldjax weights fetch --model esmfold2` to fetch the "
                "complete verified bundle."
            )
        # Pickle is executable input. Although the backend resolves this path
        # beside a managed checkpoint rather than from the job document, a
        # caller may supply an external weights directory. Verify the exact
        # publisher artifact here as well as in the asset store before opening
        # it with pickle.
        if self.path.stat().st_size != BIOHUB_CCD_SIZE:
            raise ValueError(
                "ESMFold2 ccd.pkl does not match the registered Biohub asset "
                f"(unexpected size): {self.path}"
            )
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            while chunk := handle.read(_HASH_CHUNK):
                digest.update(chunk)
        if digest.hexdigest() != BIOHUB_CCD_SHA256:
            raise ValueError(
                "ESMFold2 ccd.pkl does not match the registered Biohub asset "
                f"(SHA-256 mismatch): {self.path}"
            )
        with self.path.open("rb") as handle:
            value = pickle.load(handle)  # noqa: S301
        if not isinstance(value, dict):
            raise ValueError(f"ESMFold2 CCD is not a component dictionary: {self.path}")
        self._molecules = value
        return value

    def _molecule(self, component: str) -> Any | None:
        component = component.upper()
        found, cached = self._cached(self._molecule_cache, component)
        if found:
            return cached
        from rdkit import Chem

        source = self._load().get(component)
        if source is None or source.GetNumConformers() == 0:
            molecule = None
        else:
            molecule = Chem.RemoveHs(source, sanitize=False)
            if molecule.GetNumConformers() == 0:
                molecule = None
        return self._remember(self._molecule_cache, component, molecule)

    def conformer(self, component: str) -> dict[str, np.ndarray] | None:
        """Computed conformer, then ideal, then the first available conformer."""

        component = component.upper()
        found, cached = self._cached(self._conformer_cache, component)
        if found:
            return cached
        molecule = self._molecule(component)
        if molecule is None:
            result = None
        else:
            conformers = list(molecule.GetConformers())
            selected = next(
                (
                    conformer
                    for conformer in conformers
                    if conformer.GetPropsAsDict().get("name") == "Computed"
                ),
                None,
            )
            if selected is None:
                selected = next(
                    (
                        conformer
                        for conformer in conformers
                        if conformer.GetPropsAsDict().get("name") == "Ideal"
                    ),
                    conformers[0],
                )
            result = {}
            for atom in molecule.GetAtoms():
                name = atom.GetPropsAsDict().get("name")
                if not isinstance(name, str) or not name:
                    continue
                position = selected.GetAtomPosition(atom.GetIdx())
                result[name] = np.asarray(
                    (position.x, position.y, position.z), dtype=np.float32
                )
            if not result:
                result = None
        return self._remember(self._conformer_cache, component, result)

    def idealized_position(
        self, component: str, atom_name: str
    ) -> np.ndarray | None:
        conformer = self.conformer(component)
        return None if conformer is None else conformer.get(atom_name)

    def residue_position(self, res_type: int, atom_name: str) -> np.ndarray | None:
        component = RES_TYPE_TO_CCD.get(int(res_type))
        return (
            None
            if component is None
            else self.idealized_position(component, atom_name)
        )

    def atoms(self, component: str) -> list[tuple[str, str, int]] | None:
        """Included ``(name, element, formal_charge)`` atoms in CCD order."""

        component = component.upper()
        found, cached = self._cached(self._atom_cache, component)
        if found:
            return cached
        molecule = self._molecule(component)
        result = None
        if molecule is not None:
            values = []
            for atom in molecule.GetAtoms():
                name = atom.GetPropsAsDict().get("name")
                if isinstance(name, str) and name:
                    values.append((name, atom.GetSymbol(), atom.GetFormalCharge()))
            result = values or None
        return self._remember(self._atom_cache, component, result)

    def bonds(self, component: str) -> list[tuple[str, str]] | None:
        """Heavy/significant-hydrogen CCD bonds in included atom-name space."""

        component = component.upper()
        found, cached = self._cached(self._bond_cache, component)
        if found:
            return cached
        molecule = self._molecule(component)
        result = None
        if molecule is not None:
            values = []
            for bond in molecule.GetBonds():
                left = bond.GetBeginAtom().GetPropsAsDict().get("name")
                right = bond.GetEndAtom().GetPropsAsDict().get("name")
                if isinstance(left, str) and left and isinstance(right, str) and right:
                    values.append((left, right))
            result = values or None
        return self._remember(self._bond_cache, component, result)

    def leaving_atoms(self, component: str) -> set[str]:
        component = component.upper()
        found, cached = self._cached(self._leaving_cache, component)
        if found:
            return cached
        molecule = self._load().get(component)
        values = set()
        if molecule is not None:
            for atom in molecule.GetAtoms():
                if atom.HasProp("leaving_atom") and atom.GetProp("leaving_atom") == "1":
                    if atom.HasProp("name") and atom.GetProp("name"):
                        values.add(atom.GetProp("name"))
        return self._remember(self._leaving_cache, component, values)


@lru_cache(maxsize=2)
def _cached_store(
    path: str,
    identity: tuple[int, int, int, int, int],
) -> CCDStore:
    del identity
    return CCDStore(path)


def get_ccd_store(path: str | Path) -> CCDStore:
    """Reuse one verified 417 MB dictionary until its file identity changes."""

    resolved = Path(path).resolve()
    try:
        stat = resolved.stat()
    except OSError:
        # Preserve CCDStore's actionable missing-file message.
        return CCDStore(resolved)
    identity = (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )
    return _cached_store(str(resolved), identity)


__all__ = [
    "BIOHUB_CCD_SHA256",
    "BIOHUB_CCD_SIZE",
    "CCDStore",
    "get_ccd_store",
]
