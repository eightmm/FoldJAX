"""Rigidly compare predicted biomolecular structures without copying them.

Alignment is deliberately a post-processing operation.  Prediction artifacts,
run manifests, and resume identities stay untouched; the durable result is only
a small transform report.  Materialized coordinates are opt-in because writing
one transformed mmCIF per prediction would otherwise nearly double a batch's
structure storage.

Coordinates use the row-vector convention ``aligned = xyz @ rotation +
translation``.  Fits are ordinary, reflection-free Kabsch fits.  They never
discard outliers or move chains independently: either would make disagreements
between models look smaller than they are.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

if TYPE_CHECKING:
    from foldjax.schema import PredictionResult

AtomSelection = Literal["representative", "backbone", "heavy"]
_SELECTIONS = frozenset({"representative", "backbone", "heavy"})
_PROTEIN_BACKBONE = ("N", "CA", "C", "O")
_NUCLEIC_BACKBONE = ("P", "O5'", "C5'", "C4'", "C3'", "O3'")


class StructureAlignmentError(ValueError):
    """A requested structural correspondence is absent or ambiguous."""


@dataclass(frozen=True, slots=True)
class RigidTransform:
    """One reflection-free transform, applied as ``xyz @ rotation + translation``."""

    rotation: tuple[tuple[float, float, float], ...]
    translation: tuple[float, float, float]

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError(
                "a rigid transform needs a [3, 3] rotation and [3] translation"
            )
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("a rigid transform must be finite")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("a rigid transform rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise ValueError("a rigid transform rotation must not reflect coordinates")

    @classmethod
    def identity(cls) -> RigidTransform:
        return cls(
            rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            translation=(0.0, 0.0, 0.0),
        )

    def apply(self, coordinates: Any) -> np.ndarray:
        value = np.asarray(coordinates, dtype=np.float64)
        if value.shape[-1:] != (3,):
            raise ValueError(f"coordinates must end in an xyz axis, got {value.shape}")
        return value @ np.asarray(self.rotation) + np.asarray(self.translation)

    def summary(self) -> dict[str, object]:
        return {
            "rotation": [list(row) for row in self.rotation],
            "translation": list(self.translation),
            "convention": "aligned_xyz = source_xyz @ rotation + translation",
        }


@dataclass(frozen=True, slots=True)
class StructureAlignment:
    key: str
    source: Path
    source_sha256: str
    transform: RigidTransform
    chain_map: tuple[tuple[str, str], ...]
    selection: AtomSelection
    matched_atoms: int
    matched_residues: int
    reference_atoms: int
    rmsd: float

    @property
    def coverage(self) -> float:
        return (
            self.matched_atoms / self.reference_atoms if self.reference_atoms else 0.0
        )

    def summary(self) -> dict[str, object]:
        return {
            "key": self.key,
            "source": str(self.source),
            "source_sha256": self.source_sha256,
            "selection": self.selection,
            "chain_map": dict(self.chain_map),
            "matched_atoms": self.matched_atoms,
            "matched_residues": self.matched_residues,
            "reference_atoms": self.reference_atoms,
            "coverage": self.coverage,
            "rmsd_angstrom": self.rmsd,
            "transform": self.transform.summary(),
        }


@dataclass(frozen=True, slots=True)
class AlignmentReport:
    """Lazy transforms for a group of structures aligned to one reference."""

    reference_key: str | None
    reference: Path
    reference_sha256: str
    alignments: tuple[StructureAlignment, ...]

    def keys(self) -> tuple[str, ...]:
        return tuple(item.key for item in self.alignments)

    def __getitem__(self, key: str) -> StructureAlignment:
        for item in self.alignments:
            if item.key == key:
                return item
        raise KeyError(key)

    def summary(self) -> dict[str, object]:
        return {
            "schema": "foldjax-structure-alignment-v1",
            "reference_key": self.reference_key,
            "reference": str(self.reference),
            "reference_sha256": self.reference_sha256,
            "alignments": [item.summary() for item in self.alignments],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.summary(), indent=indent, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, contents: str) -> AlignmentReport:
        """Restore a report while retaining source-digest checks on use."""
        try:
            document = json.loads(contents)
            if not isinstance(document, dict):
                raise ValueError("alignment report root must be an object")
            if document.get("schema") != "foldjax-structure-alignment-v1":
                raise ValueError("unsupported alignment report schema")
            records = []
            for item in document["alignments"]:
                transform = item["transform"]
                selection = str(item["selection"])
                if selection not in _SELECTIONS:
                    raise ValueError(f"unsupported atom selection: {selection!r}")
                records.append(
                    StructureAlignment(
                        key=str(item["key"]),
                        source=Path(item["source"]),
                        source_sha256=str(item["source_sha256"]),
                        transform=RigidTransform(
                            rotation=tuple(tuple(row) for row in transform["rotation"]),
                            translation=tuple(transform["translation"]),
                        ),
                        chain_map=tuple(
                            (str(left), str(right))
                            for left, right in item["chain_map"].items()
                        ),
                        selection=cast(AtomSelection, selection),
                        matched_atoms=int(item["matched_atoms"]),
                        matched_residues=int(item["matched_residues"]),
                        reference_atoms=int(item["reference_atoms"]),
                        rmsd=float(item["rmsd_angstrom"]),
                    )
                )
            return cls(
                reference_key=(
                    str(document["reference_key"])
                    if document.get("reference_key") is not None
                    else None
                ),
                reference=Path(document["reference"]),
                reference_sha256=str(document["reference_sha256"]),
                alignments=tuple(records),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise StructureAlignmentError(
                f"invalid alignment report: {error}"
            ) from error

    @classmethod
    def read_json(cls, path: str | os.PathLike[str]) -> AlignmentReport:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def write_json(
        self, path: str | os.PathLike[str], *, overwrite: bool = False
    ) -> Path:
        """Explicitly persist the small transform report, never coordinate copies."""
        target = Path(path)
        _prepare_target(target, overwrite=overwrite)
        _write_atomic(target, self.to_json())
        return target

    def aligned_cif(self, key: str) -> str:
        """Return a transformed mmCIF string without writing it to disk."""
        item = self[key]
        _verify_source(item.source, item.source_sha256)
        return _transform_to_cif(item.source, item.transform)

    def write_aligned(
        self,
        key: str,
        path: str | os.PathLike[str],
        *,
        overwrite: bool = False,
    ) -> Path:
        """Materialize one aligned mmCIF only when a caller explicitly asks."""
        target = Path(path)
        _prepare_target(target, overwrite=overwrite)
        _write_atomic(target, self.aligned_cif(key))
        return target


@dataclass(slots=True)
class _Chain:
    name: str
    kind: str
    residues: list[Any]
    sequence: str
    fingerprint: tuple[object, ...]


@dataclass(slots=True)
class _Parsed:
    path: Path
    chains: dict[str, _Chain]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source(path: Path, expected: str) -> None:
    try:
        actual = _sha256(path)
    except OSError as error:
        raise StructureAlignmentError(
            f"cannot verify aligned structure source {path}: {error}"
        ) from error
    if actual != expected:
        raise StructureAlignmentError(
            f"structure changed after alignment: {path} "
            f"({expected[:12]} != {actual[:12]})"
        )


def _polymer_kind(polymer_type: Any) -> str:
    name = str(polymer_type).rsplit(".", 1)[-1]
    if name in {"PeptideL", "PeptideD", "CyclicPseudoPeptide"}:
        return "protein"
    if name in {"Dna", "Rna", "DnaRnaHybrid", "Pna"}:
        return "nucleic"
    return "other"


def _residue_letter(residue: Any) -> str:
    import gemmi

    code = gemmi.find_tabulated_residue(residue.name).one_letter_code.strip()
    return (code or "X").upper()


def _atom_names(residue: Any) -> tuple[str, ...]:
    return tuple(
        sorted(
            atom.name.strip().replace("*", "'")
            for atom in residue
            if str(atom.element.name).upper() not in {"H", "D"}
        )
    )


def _parse(path: Path) -> _Parsed:
    import gemmi

    path = path.resolve()
    if not path.is_file():
        raise StructureAlignmentError(f"structure does not exist: {path}")
    try:
        structure = gemmi.read_structure(str(path))
    except Exception as error:  # noqa: BLE001 - normalized as a public domain error
        raise StructureAlignmentError(
            f"cannot read structure {path}: {error}"
        ) from error
    if not structure or not structure[0]:
        raise StructureAlignmentError(f"structure has no coordinate model: {path}")
    structure.setup_entities()
    chains: dict[str, _Chain] = {}
    for chain in structure[0]:
        name = str(chain.name)
        if name in chains:
            raise StructureAlignmentError(f"duplicate chain id {name!r} in {path}")
        polymer = list(chain.get_polymer())
        if polymer:
            kind = _polymer_kind(chain.get_polymer().check_polymer_type())
            residues = polymer
            if kind in {"protein", "nucleic"}:
                sequence = "".join(_residue_letter(residue) for residue in polymer)
                fingerprint: tuple[object, ...] = (kind, sequence)
            else:
                sequence = ""
                fingerprint = (
                    kind,
                    tuple((residue.name, _atom_names(residue)) for residue in residues),
                )
        else:
            residues = list(chain)
            kind = "ligand"
            sequence = ""
            fingerprint = (
                kind,
                tuple((residue.name, _atom_names(residue)) for residue in residues),
            )
        if residues:
            chains[name] = _Chain(name, kind, residues, sequence, fingerprint)
    if not chains:
        raise StructureAlignmentError(f"structure has no residues: {path}")
    return _Parsed(path=path, chains=chains)


def _compatible(source: _Chain, reference: _Chain) -> bool:
    return source.kind == reference.kind


def _chain_similarity(source: _Chain, reference: _Chain) -> float:
    if not _compatible(source, reference):
        return 0.0
    if source.kind not in {"protein", "nucleic"}:
        return float(source.fingerprint == reference.fingerprint)
    pairs = _align_sequence(source.sequence, reference.sequence)
    if not pairs:
        return 0.0
    matches = sum(source.sequence[i] == reference.sequence[j] for i, j in pairs)
    return matches / max(len(source.sequence), len(reference.sequence))


def _chain_map(
    source: _Parsed,
    reference: _Parsed,
    explicit: Mapping[str, str] | None,
) -> dict[str, str]:
    if explicit is not None:
        result = {str(left): str(right) for left, right in explicit.items()}
        if len(set(result.values())) != len(result):
            raise StructureAlignmentError(
                "chain_map cannot map two source chains to one reference chain"
            )
        for left, right in result.items():
            if left not in source.chains or right not in reference.chains:
                raise StructureAlignmentError(
                    f"unknown chain mapping {left!r} -> {right!r}"
                )
            if not _compatible(source.chains[left], reference.chains[right]):
                raise StructureAlignmentError(
                    f"incompatible chain mapping {left!r} -> {right!r}"
                )
        return result

    result: dict[str, str] = {}
    used: set[str] = set()
    # The common FoldJAX input normally preserves chain ids.  Prefer that
    # provenance over a geometry-minimizing homomer permutation.
    for name, chain in source.chains.items():
        candidate = reference.chains.get(name)
        if candidate is not None and _chain_similarity(chain, candidate) >= 0.5:
            result[name] = name
            used.add(name)

    for name, chain in source.chains.items():
        if name in result:
            continue
        candidates = [
            other.name
            for other in reference.chains.values()
            if other.name not in used and other.fingerprint == chain.fingerprint
        ]
        if len(candidates) == 1:
            result[name] = candidates[0]
            used.add(candidates[0])
        elif len(candidates) > 1:
            raise StructureAlignmentError(
                f"chain {name!r} has ambiguous matches {candidates}; pass chain_map"
            )

    remaining_source = [
        chain for name, chain in source.chains.items() if name not in result
    ]
    remaining_reference = [
        chain for name, chain in reference.chains.items() if name not in used
    ]
    # A single partial/mutated chain is unambiguous even when its sequence is
    # not exact; residue alignment below establishes the actual correspondence.
    for kind in ("protein", "nucleic", "ligand", "other"):
        left = [chain for chain in remaining_source if chain.kind == kind]
        right = [chain for chain in remaining_reference if chain.kind == kind]
        if len(left) == len(right) == 1 and _chain_similarity(left[0], right[0]) >= 0.5:
            result[left[0].name] = right[0].name
            used.add(right[0].name)
    if not result:
        raise StructureAlignmentError(
            "no compatible chains could be matched; pass an explicit chain_map"
        )
    return result


def _align_sequence(left: str, right: str) -> list[tuple[int, int]]:
    """Deterministic global alignment, returning residue index pairs."""
    rows, cols = len(left) + 1, len(right) + 1
    score = np.empty((rows, cols), dtype=np.int32)
    trace = np.zeros((rows, cols), dtype=np.int8)
    score[:, 0] = -2 * np.arange(rows)
    score[0, :] = -2 * np.arange(cols)
    trace[1:, 0] = 1
    trace[0, 1:] = 2
    for i in range(1, rows):
        for j in range(1, cols):
            diagonal = score[i - 1, j - 1] + (2 if left[i - 1] == right[j - 1] else -1)
            up = score[i - 1, j] - 2
            across = score[i, j - 1] - 2
            # Diagonal-first tie breaking makes identical inputs map by index.
            best = max(diagonal, up, across)
            score[i, j] = best
            trace[i, j] = 0 if diagonal == best else (1 if up == best else 2)
    pairs: list[tuple[int, int]] = []
    i, j = len(left), len(right)
    while i or j:
        direction = trace[i, j]
        if i and j and direction == 0:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i and (not j or direction == 1):
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def _best_atom(residue: Any, name: str) -> Any | None:
    normalized = name.replace("*", "'")
    candidates = [
        atom for atom in residue if atom.name.strip().replace("*", "'") == normalized
    ]
    if not candidates:
        return None
    return max(
        candidates, key=lambda atom: (float(atom.occ), atom.altloc in {"\0", " ", "A"})
    )


def _finite_position(atom: Any | None) -> np.ndarray | None:
    if atom is None:
        return None
    value = np.asarray((atom.pos.x, atom.pos.y, atom.pos.z), dtype=np.float64)
    return value if np.isfinite(value).all() else None


def _residue_pairs(source: _Chain, reference: _Chain) -> list[tuple[Any, Any]]:
    if source.kind in {"protein", "nucleic"}:
        return [
            (source.residues[i], reference.residues[j])
            for i, j in _align_sequence(source.sequence, reference.sequence)
        ]
    if len(source.residues) == len(reference.residues) == 1:
        if source.residues[0].name == reference.residues[0].name:
            return [(source.residues[0], reference.residues[0])]
    by_key: dict[tuple[str, int, str], Any] = {}
    for residue in reference.residues:
        by_key[(residue.name, int(residue.seqid.num), str(residue.seqid.icode))] = (
            residue
        )
    return [
        (residue, by_key[key])
        for residue in source.residues
        if (key := (residue.name, int(residue.seqid.num), str(residue.seqid.icode)))
        in by_key
    ]


def _selected_names(kind: str, selection: AtomSelection) -> tuple[str, ...] | None:
    if selection == "heavy":
        return None
    if kind == "protein":
        return ("CA",) if selection == "representative" else _PROTEIN_BACKBONE
    if kind == "nucleic":
        return ("C4'",) if selection == "representative" else _NUCLEIC_BACKBONE
    return None


def _reference_atom_count(
    chain: _Chain,
    selection: AtomSelection,
    *,
    ligand_fallback: bool,
) -> int:
    if chain.kind not in {"protein", "nucleic"} and selection == "representative":
        if not ligand_fallback:
            return 0
        names = None
    else:
        names = _selected_names(chain.kind, selection)
    count = 0
    for residue in chain.residues:
        chosen = _atom_names(residue) if names is None else names
        count += sum(
            _finite_position(_best_atom(residue, name)) is not None for name in chosen
        )
    return count


def _points(
    source: _Parsed,
    reference: _Parsed,
    mapping: Mapping[str, str],
    selection: AtomSelection,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    mobile: list[np.ndarray] = []
    target: list[np.ndarray] = []
    matched_residues = 0
    mapped = []
    for left, right in mapping.items():
        source_chain = source.chains[left]
        reference_chain = reference.chains[right]
        pairs = _residue_pairs(source_chain, reference_chain)
        mapped.append((source_chain, reference_chain, pairs))

    # A ligand-only target still has a well-defined rigid fit when atom names
    # agree.  Mixed complexes deliberately fit their polymers by default and
    # carry ligands along without letting a large ligand dominate the frame.
    ligand_fallback = selection == "representative" and not any(
        chain.kind in {"protein", "nucleic"} and pairs for chain, _, pairs in mapped
    )
    reference_atoms = sum(
        _reference_atom_count(
            reference_chain,
            selection,
            ligand_fallback=ligand_fallback,
        )
        for _, reference_chain, _ in mapped
    )

    for source_chain, _, pairs in mapped:
        if (
            source_chain.kind not in {"protein", "nucleic"}
            and selection == "representative"
            and not ligand_fallback
        ):
            continue
        names = (
            None
            if source_chain.kind not in {"protein", "nucleic"} and ligand_fallback
            else _selected_names(source_chain.kind, selection)
        )
        for source_residue, reference_residue in pairs:
            if names is None:
                source_names = set(_atom_names(source_residue))
                reference_names = set(_atom_names(reference_residue))
                chosen = tuple(sorted(source_names & reference_names))
            else:
                chosen = names
            residue_matched = False
            for name in chosen:
                reference_position = _finite_position(
                    _best_atom(reference_residue, name)
                )
                source_position = _finite_position(_best_atom(source_residue, name))
                if source_position is None or reference_position is None:
                    continue
                mobile.append(source_position)
                target.append(reference_position)
                residue_matched = True
            matched_residues += int(residue_matched)

    return (
        np.asarray(mobile, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
        matched_residues,
        reference_atoms,
    )


def _fit(mobile: np.ndarray, target: np.ndarray) -> tuple[RigidTransform, float]:
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[1:] != (3,):
        raise StructureAlignmentError("alignment points must be matching [n, 3] arrays")
    if len(mobile) < 3:
        raise StructureAlignmentError(
            f"alignment needs at least 3 matched atoms, found {len(mobile)}"
        )
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    mobile_zero = mobile - mobile_center
    target_zero = target - target_center
    if np.linalg.matrix_rank(mobile_zero) < 2 or np.linalg.matrix_rank(target_zero) < 2:
        raise StructureAlignmentError("alignment points are collinear")
    left, _, right = np.linalg.svd(mobile_zero.T @ target_zero)
    correction = np.eye(3)
    correction[-1, -1] = math.copysign(1.0, np.linalg.det(left @ right))
    rotation = left @ correction @ right
    translation = target_center - mobile_center @ rotation
    aligned = mobile @ rotation + translation
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    transform = RigidTransform(
        rotation=tuple(tuple(float(value) for value in row) for row in rotation),
        translation=tuple(float(value) for value in translation),
    )
    return transform, rmsd


def align_structures(
    structures: Mapping[str, str | os.PathLike[str]],
    *,
    reference: str | os.PathLike[str],
    selection: AtomSelection = "representative",
    chain_map: Mapping[str, str] | None = None,
) -> AlignmentReport:
    """Align named structures to a named member or an external PDB/mmCIF.

    The function is read-only.  It returns transforms and metrics; callers opt
    into ``write_json`` or ``write_aligned`` separately.
    """
    if selection not in _SELECTIONS:
        raise ValueError(
            f"selection must be one of {sorted(_SELECTIONS)}, got {selection!r}"
        )
    if not structures:
        raise StructureAlignmentError("at least one source structure is required")
    normalized = {str(key): Path(path).resolve() for key, path in structures.items()}
    if len(normalized) != len(structures) or any(not key for key in normalized):
        raise StructureAlignmentError("structure keys must be unique non-empty strings")

    reference_key: str | None
    if isinstance(reference, str) and reference in normalized:
        reference_key = reference
        reference_path = normalized[reference]
    else:
        reference_key = None
        reference_path = Path(reference).resolve()
    parsed_reference = _parse(reference_path)
    reference_digest = _sha256(reference_path)
    records: list[StructureAlignment] = []
    for key, source_path in normalized.items():
        parsed_source = _parse(source_path)
        mapping = _chain_map(parsed_source, parsed_reference, chain_map)
        mobile, target, residues, reference_atoms = _points(
            parsed_source, parsed_reference, mapping, selection
        )
        transform, rmsd = _fit(mobile, target)
        records.append(
            StructureAlignment(
                key=key,
                source=source_path,
                source_sha256=_sha256(source_path),
                transform=transform,
                chain_map=tuple(sorted(mapping.items())),
                selection=selection,
                matched_atoms=len(mobile),
                matched_residues=residues,
                reference_atoms=reference_atoms,
                rmsd=rmsd,
            )
        )
    return AlignmentReport(
        reference_key=reference_key,
        reference=reference_path,
        reference_sha256=reference_digest,
        alignments=tuple(records),
    )


def align_predictions(
    results: Sequence[PredictionResult],
    *,
    reference: str | os.PathLike[str],
    selection: AtomSelection = "representative",
    chain_map: Mapping[str, str] | None = None,
) -> AlignmentReport:
    """Align every materialized sample in common ``PredictionResult`` objects.

    Keys have the stable form ``model/seed-N/sample-II`` and may be passed back
    as ``reference``.  An external PDB/mmCIF path is accepted instead.
    """
    structures: dict[str, Path] = {}
    for result in results:
        for position, sample in enumerate(result.samples):
            if sample.structure_path is None:
                continue
            try:
                index = int((sample.metadata or {}).get("sample", position))
            except (TypeError, ValueError):
                index = position
            key = f"{result.model}/seed-{int(sample.seed)}/sample-{index:02d}"
            if key in structures:
                raise StructureAlignmentError(
                    f"duplicate prediction structure key: {key}"
                )
            structures[key] = Path(sample.structure_path)
    return align_structures(
        structures,
        reference=reference,
        selection=selection,
        chain_map=chain_map,
    )


def _prepare_target(path: Path, *, overwrite: bool) -> None:
    if (path.exists() or path.is_symlink()) and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_atomic(path: Path, contents: str) -> None:
    with tempfile.TemporaryDirectory(
        prefix=".foldjax-align-", dir=path.parent
    ) as scratch:
        staged = Path(scratch) / path.name
        staged.write_text(contents, encoding="utf-8")
        os.replace(staged, path)


def _transform_to_cif(path: Path, transform: RigidTransform) -> str:
    import gemmi

    suffixes = tuple(suffix.lower() for suffix in path.suffixes)
    if ".cif" in suffixes or ".mmcif" in suffixes:
        document = gemmi.cif.read_file(str(path))
        found = False
        for block in document:
            coordinates = block.find(
                ["_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z"]
            )
            if not coordinates:
                continue
            found = True
            values = np.asarray(
                [[float(row[0]), float(row[1]), float(row[2])] for row in coordinates],
                dtype=np.float64,
            )
            aligned = transform.apply(values)
            for row, xyz in zip(coordinates, aligned, strict=True):
                row[0], row[1], row[2] = (f"{float(value):.3f}" for value in xyz)
        if not found:
            raise StructureAlignmentError(f"mmCIF has no atom_site coordinates: {path}")
        return document.as_string()

    structure = gemmi.read_structure(str(path))
    matrix = np.asarray(transform.rotation)
    offset = np.asarray(transform.translation)
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    xyz = (
                        np.asarray((atom.pos.x, atom.pos.y, atom.pos.z)) @ matrix
                        + offset
                    )
                    atom.pos = gemmi.Position(*map(float, xyz))
    return structure.make_mmcif_document().as_string()
