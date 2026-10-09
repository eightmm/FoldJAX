"""Pocket-aware sample selection: the ``pocket_sampling=select`` route.

A FoldJAX-only route that every backend takes. It changes nothing about how
a model samples: after prediction each sample is scored against the job's
pocket constraints with Boltz-2's own pocket rule, and the run's ``best``
prefers a sample that satisfies every pocket. On a model whose native input
conditions on the pocket (Boltz-2, OpenFold3, the Protenix constraint
checkpoint) the conditioning runs as it does today and the selection is
added on top; on one that has no pocket input (AlphaFold 3, ESMFold2), or
whose inference build ignores it (OpenDDE), or whose checkpoint has no
embedder for it (the released Protenix default), the pocket reaches only
this selection, and the job runs instead of being refused or dropped.

The rule is the one Boltz-2's featurizer uses to label a pocket residue when
it builds the training feature (`foldjax.models.boltz2.data.feature.
featurizerv2`, ``token_dist < binder_pocket_cutoff``): a residue is satisfied
when the smallest distance between any of its heavy atoms and any heavy atom
of the binder chain is below ``max_distance``; a pocket is satisfied when
every listed residue is; a sample when every pocket is. Hydrogens are
removed before measuring, as Boltz-2's structures carry none.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from foldjax.schema import PredictionResult, PredictionSample

#: The option, and its two values. ``off`` is the default and leaves every
#: backend exactly as it is; ``select`` scores and ranks as described above.
POCKET_SAMPLING = "pocket_sampling"
POCKET_SAMPLING_VALUES = ("off", "select")

#: The name the capability record lists this route under
#: (`ModelCapabilities.foldjax_only_features`).
FEATURE = "pocket_selection"

#: Sample metadata key the scores live under, and the confidence.json and
#: manifest field that repeats them.
METADATA_KEY = "pocket_sampling"

#: What ``best.selection`` says when the pocket decided among the samples.
SELECTION = "pocket-satisfied samples, within-model confidence ranking"


def requested(options: Mapping[str, Any] | None) -> str:
    """The route ``options`` ask for, validated: ``off`` or ``select``."""
    value = (options or {}).get(POCKET_SAMPLING, "off")
    if value in POCKET_SAMPLING_VALUES:
        return str(value)
    raise ValueError(
        f"{POCKET_SAMPLING} must be one of {POCKET_SAMPLING_VALUES}, got {value!r}"
    )


@dataclasses.dataclass(frozen=True, slots=True)
class JobChain:
    """One chain of the common job, in the job's own order."""

    id: str
    kind: str
    residues: int
    #: A polymer's one-letter sequence; None for a ligand.
    sequence: str | None = None


def job_chains(job: Any) -> tuple[JobChain, ...]:
    """Every chain of a `foldjax.job.Job`, entities in order, copies in order."""
    from foldjax.job import Ligand

    chains: list[JobChain] = []
    for entity in job.entities:
        ids = entity.id if isinstance(entity.id, tuple) else (entity.id,)
        for chain in ids:
            if isinstance(entity, Ligand):
                codes = entity.ccd
                count = len(codes) if isinstance(codes, tuple) else 1
                chains.append(JobChain(str(chain), "ligand", count))
            else:
                sequence = str(entity.sequence).upper()
                chains.append(
                    JobChain(str(chain), entity.kind, len(sequence), sequence)
                )
    return tuple(chains)


def pocket_records(constraints: Iterable[Mapping[str, Any]] | None) -> list[dict]:
    """The pocket entries of a manifest-style ``constraints`` list."""
    return [
        dict(record) for record in constraints or () if record.get("kind") == "pocket"
    ]


@dataclasses.dataclass(slots=True)
class _Chain:
    name: str
    #: Residue key (label_seq where numeric, else the author number) ->
    #: heavy-atom coordinates.
    residues: dict[int, np.ndarray]
    #: The residues' one-letter codes in key order; "X" where gemmi has none.
    letters: str
    #: Every heavy atom of the chain, in file order.
    atoms: np.ndarray


def _read_chains(path: Path) -> list[_Chain]:
    import gemmi

    structure = gemmi.read_structure(str(path))
    if len(structure) == 0:
        raise ValueError(f"{path} has no model")
    structure.remove_alternative_conformations()
    structure.remove_hydrogens()
    chains: list[_Chain] = []
    for chain in structure[0]:
        residues: dict[int, list[list[float]]] = {}
        letters: dict[int, str] = {}
        atoms: list[list[float]] = []
        for residue in chain:
            key = residue.label_seq
            if key is None:
                key = residue.seqid.num
            rows = [atom.pos.tolist() for atom in residue]
            if not rows:
                continue
            residues.setdefault(int(key), []).extend(rows)
            info = gemmi.find_tabulated_residue(residue.name)
            letters.setdefault(
                int(key),
                (info.one_letter_code.strip().upper() if info else "") or "X",
            )
            atoms.extend(rows)
        if not atoms:
            continue
        chains.append(
            _Chain(
                chain.name,
                {key: np.asarray(rows, dtype=float) for key, rows in residues.items()},
                "".join(letters[key] for key in sorted(letters)),
                np.asarray(atoms, dtype=float),
            )
        )
    return chains


def _is(chain: _Chain, described: JobChain) -> bool:
    """Whether ``chain`` can be the job's ``described`` chain.

    A polymer must have the job's length and spell its sequence wherever
    gemmi knows the residue ("X", a modified or unknown residue, matches
    anything); a ligand must be a chain of residues gemmi knows no polymer
    code for, with the job's residue count (one per CCD code, one for a
    SMILES).
    """
    if len(chain.residues) != described.residues:
        return False
    if described.kind == "ligand":
        return set(chain.letters) <= {"X"}
    sequence = described.sequence or ""
    return all(
        letter == "X" or expected == "X" or letter == expected
        for letter, expected in zip(chain.letters, sequence, strict=False)
    )


#: Ports whose writer labels chains A, B, ... by position rather than by the
#: job's ids (`foldjax.models.openfold3.output._chain_label`), so for them the
#: job's order is consulted before the names: two ligands of one residue each
#: are told apart only by it.
_POSITIONAL_WRITERS = frozenset({"openfold3"})


def _by_position(
    chains: Sequence[_Chain], job: Sequence[JobChain]
) -> dict[str, _Chain] | str:
    """The structure's chain order against the job's, every chain validated."""
    if len(chains) != len(job):
        return (
            f"the structure has {len(chains)} chain(s) named "
            f"{', '.join(chain.name for chain in chains)} and the job {len(job)}"
        )
    for chain, described in zip(chains, job, strict=True):
        if not _is(chain, described):
            return (
                f"structure chain {chain.name!r} ({len(chain.residues)} residues) "
                f"is not job chain {described.id!r} ({described.residues} residues "
                f"of {described.kind})"
            )
    return {described.id: chain for chain, described in zip(chains, job, strict=True)}


def _by_name(
    chains: Sequence[_Chain], job: Sequence[JobChain], needed: set[str]
) -> dict[str, _Chain] | str:
    """The structure's chain names, every needed chain validated."""
    by_id = {described.id: described for described in job}
    by_name = {chain.name: chain for chain in chains}
    missing = sorted(needed - set(by_name))
    if missing:
        return f"the structure names no chain {', '.join(map(repr, missing))}"
    for chain_id in sorted(needed):
        if chain_id in by_id and not _is(by_name[chain_id], by_id[chain_id]):
            return (
                f"structure chain {chain_id!r} ({len(by_name[chain_id].residues)} "
                f"residues) is not job chain {chain_id!r} "
                f"({by_id[chain_id].residues} residues of {by_id[chain_id].kind})"
            )
    return by_name


def _chain_map(
    chains: Sequence[_Chain],
    job: Sequence[JobChain],
    needed: set[str],
    *,
    positional_first: bool = False,
) -> dict[str, _Chain] | str:
    """Job chain id -> structure chain, or why no mapping is trustworthy.

    By name when the structure names every chain the pockets need (AlphaFold
    3, Boltz-2, ESMFold2, Protenix and OpenDDE keep the job's ids); else by
    position, the structure's chain order against the job's entity order
    with copies in order (OpenFold3 labels chains A, B, ... in that order,
    and ``positional_first`` puts that route first). Either way every chain
    the mapping uses must have the job's length, kind and sequence, so a
    label that merely coincides with another chain's id cannot stand in
    for it.
    """
    routes = (
        (_by_position(chains, job), _by_name(chains, job, needed))
        if positional_first
        else (_by_name(chains, job, needed), _by_position(chains, job))
    )
    for mapping in routes:
        if not isinstance(mapping, str):
            return mapping
    return (
        "neither the names nor the order identify the pocket chains: "
        + "; ".join(reason for reason in routes if isinstance(reason, str))
    )


def _min_distance(residue: np.ndarray, binder: np.ndarray) -> float:
    deltas = residue[:, None, :] - binder[None, :, :]
    return float(np.sqrt(np.einsum("ijk,ijk->ij", deltas, deltas)).min())


def score_structure(
    path: Path,
    pockets: Sequence[Mapping[str, Any]],
    job: Sequence[JobChain],
    *,
    model: str | None = None,
) -> dict[str, Any]:
    """Score one structure file against ``pockets`` (manifest pocket records).

    Returns the block written under the sample's ``pocket_sampling``:
    ``satisfied`` is true when every pocket is, false when one is not, and
    null with a ``reason`` when the file cannot be put against the job.
    ``model`` names the port that wrote ``path``, which decides how its
    chains are matched to the job's (`_chain_map`).
    """
    try:
        chains = _read_chains(Path(path))
    except Exception as error:  # noqa: BLE001 - reported, never fatal to a run
        return {
            "mode": "select",
            "satisfied": None,
            "reason": f"cannot read {path}: {error}",
        }
    needed = {str(pocket["binder"]) for pocket in pockets}
    needed.update(str(chain) for pocket in pockets for chain, _ in pocket["contacts"])
    mapping = _chain_map(
        chains, job, needed, positional_first=model in _POSITIONAL_WRITERS
    )
    if isinstance(mapping, str):
        return {"mode": "select", "satisfied": None, "reason": mapping}
    scored: list[dict[str, Any]] = []
    for pocket in pockets:
        binder = mapping[str(pocket["binder"])].atoms
        threshold = float(pocket["max_distance"])
        residues: list[dict[str, Any]] = []
        for chain_id, index in pocket["contacts"]:
            coordinates = mapping[str(chain_id)].residues.get(int(index))
            if coordinates is None:
                return {
                    "mode": "select",
                    "satisfied": None,
                    "reason": (
                        f"residue {int(index)} of chain {chain_id!r} has no heavy "
                        f"atom in {path}"
                    ),
                }
            distance = _min_distance(coordinates, binder)
            residues.append(
                {
                    "chain": str(chain_id),
                    "residue": int(index),
                    "min_distance": round(distance, 3),
                    "satisfied": bool(distance < threshold),
                }
            )
        scored.append(
            {
                "binder": str(pocket["binder"]),
                "max_distance": threshold,
                "satisfied": all(residue["satisfied"] for residue in residues),
                "residues": residues,
            }
        )
    return {
        "mode": "select",
        "satisfied": all(pocket["satisfied"] for pocket in scored),
        "pockets": scored,
    }


def annotate(
    result: PredictionResult,
    *,
    pockets: Sequence[Mapping[str, Any]],
    job: Sequence[JobChain],
) -> PredictionResult:
    """Score every sample that has a structure file; coordinates stay as they are."""
    samples = []
    for sample in result.samples:
        path = sample.structure_path
        if path is None or not Path(path).is_file():
            block = {
                "mode": "select",
                "satisfied": None,
                "reason": "the sample has no structure file to measure",
            }
        else:
            block = score_structure(Path(path), pockets, job, model=result.model)
        samples.append(
            dataclasses.replace(
                sample, metadata={**(sample.metadata or {}), METADATA_KEY: block}
            )
        )
    return dataclasses.replace(result, samples=tuple(samples))


def satisfied(sample: PredictionSample) -> bool | None:
    """Whether ``sample`` satisfied every pocket; None when not scored."""
    block = (sample.metadata or {}).get(METADATA_KEY)
    if not isinstance(block, Mapping):
        return None
    value = block.get("satisfied")
    return value if isinstance(value, bool) else None


def manifest_block(
    result: PredictionResult,
    *,
    pockets: Sequence[Mapping[str, Any]] | None = None,
    native_conditioning: bool | None = None,
) -> dict[str, Any] | None:
    """The manifest's ``pocket_sampling`` block; None when the route was off.

    ``pockets`` and ``native_conditioning`` are known to the run that
    translated the job; a manifest combining several seeds has only the
    samples' own blocks and records the counts alone.
    """
    scored = [satisfied(sample) for sample in result.samples]
    if not any(
        isinstance((sample.metadata or {}).get(METADATA_KEY), Mapping)
        for sample in result.samples
    ):
        return None
    block: dict[str, Any] = {
        "mode": "select",
        # The route is FoldJAX's own: no upstream model selects on a pocket.
        "route": "foldjax",
        "samples": len(scored),
        "satisfied": sum(1 for value in scored if value is True),
        "unscored": sum(1 for value in scored if value is None),
    }
    if native_conditioning is not None:
        block["native_conditioning"] = bool(native_conditioning)
    if pockets is not None:
        block["pockets"] = [dict(pocket) for pocket in pockets]
    return block
