"""Write multi-job files for screens: one target against a ligand library, or
bait proteins against candidate proteins.

Both produce an ordinary ``{"jobs": [...]}`` document that ``foldjax predict
--input`` runs as a batch (see docs/input.md for the format), so a screen is
resumable (``--resume``), shardable (``--shard``) and lands in
``<out>/<model>/<job name>`` like any batch.

**Names are stable.** A job's name is its output directory and its resume
identity, so it must not depend on the order of the library: a ligand keeps
the name its SMILES/SDF record gives it, and a record without one is named
from a digest of its canonical SMILES. Reordering or extending the library
leaves every existing job's name -- and its finished output -- in place.

**Alignments are reused, not searched again.** Relative ``unpaired_msa``,
``paired_msa`` and template paths of the target are made absolute, so the jobs
file can live anywhere and every job reads the same alignment files.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _clean(name: str) -> str:
    return _NAME.sub("_", name.strip()).strip("._-")


def _absolute(document: dict[str, Any], base: Path) -> dict[str, Any]:
    from foldjax.input import _absolute_job_paths

    return _absolute_job_paths(document, base)


def _chain_ids(document: Mapping[str, Any]) -> set[str]:
    taken: set[str] = set()
    for entity in document.get("entities") or []:
        ids = entity.get("id") if isinstance(entity, Mapping) else None
        if isinstance(ids, list):
            taken.update(str(item) for item in ids)
        elif ids is not None:
            taken.add(str(ids))
    return taken


def _next_chain(taken: set[str]) -> str:
    from foldjax.input import chain_labels

    for label in chain_labels():
        if label not in taken:
            return label
    raise AssertionError("unreachable")


def read_ligands(
    path: str | os.PathLike[str], *, skip_invalid: bool = False
) -> tuple[list[tuple[str | None, str]], list[str]]:
    """``(name or None, canonical SMILES)`` per record, and the records skipped.

    ``.smi``/``.smiles``/``.txt``: one ``SMILES [name]`` per line (``#``
    comments and blank lines ignored). ``.sdf``/``.sd``: the record's title
    line names it; coordinates are not carried -- every carried model builds
    its own conformer from the chemistry.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    path = Path(path)
    records: list[tuple[str | None, str]] = []
    bad: list[str] = []
    suffix = path.suffix.lower()
    if suffix in {".sdf", ".sd"}:
        supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=True)
        for index, molecule in enumerate(supplier):
            if molecule is None:
                bad.append(f"record {index + 1}: unreadable molecule")
                continue
            title = (
                molecule.GetProp("_Name").strip() if molecule.HasProp("_Name") else ""
            )
            records.append((title or None, Chem.MolToSmiles(molecule)))
    elif suffix in {".smi", ".smiles", ".txt", ".csv", ".tsv"}:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            parts = re.split(r"[\s,]+", text, maxsplit=1)
            molecule = Chem.MolFromSmiles(parts[0])
            if molecule is None:
                if number == 1 and parts[0].lower() in {"smiles", "smi"}:
                    continue  # a header row
                bad.append(f"line {number}: unreadable SMILES {parts[0]!r}")
                continue
            name = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
            records.append((name, Chem.MolToSmiles(molecule)))
    else:
        raise ValueError(
            f"{path}: ligand library must be .smi/.smiles/.txt (SMILES [name] per "
            "line) or .sdf"
        )
    if bad and not skip_invalid:
        raise ValueError(
            f"{path}: {len(bad)} unreadable record(s), first: {bad[0]}; fix them "
            "or pass --skip-invalid"
        )
    if not records:
        raise ValueError(f"{path}: no ligand records")
    return records, bad


def _ligand_names(records: Sequence[tuple[str | None, str]]) -> list[str]:
    """Stable, unique, directory-safe names."""
    names: list[str] = []
    counts: dict[str, int] = {}
    for given, smiles in records:
        base = _clean(given) if given else ""
        if not base:
            base = "lig-" + hashlib.sha256(smiles.encode()).hexdigest()[:10]
        counts[base] = counts.get(base, 0) + 1
        names.append(base)
    out = []
    for (given, smiles), base in zip(records, names, strict=True):
        if counts[base] > 1:
            # Two records share a name: the SMILES digest keeps each one's
            # directory fixed whatever order they come in.
            base = f"{base}-{hashlib.sha256(smiles.encode()).hexdigest()[:8]}"
        out.append(base)
    seen: set[str] = set()
    for name in out:
        if name in seen:
            raise ValueError(
                f"ligand {name!r} appears twice with the same SMILES; remove the "
                "duplicate record"
            )
        seen.add(name)
    return out


def expand_ligands(
    target: str | os.PathLike[str],
    library: str | os.PathLike[str],
    *,
    affinity: bool = False,
    skip_invalid: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The ``{jobs: [...]}`` document, and a summary of what was written."""
    from foldjax.input import read_job_document

    target = Path(target)
    document = read_job_document(target)
    if not isinstance(document, Mapping) or "entities" not in document:
        raise ValueError(f"{target} must be one common-schema job (with entities)")
    base = _absolute(copy.deepcopy(dict(document)), target.parent.absolute())
    if affinity and base.get("properties"):
        raise ValueError(
            f"{target} already sets properties; --affinity would add a second "
            "affinity binder"
        )
    records, skipped = read_ligands(library, skip_invalid=skip_invalid)
    names = _ligand_names(records)
    chain = _next_chain(_chain_ids(base))
    stem = _clean(str(base.get("name") or target.stem)) or "target"
    jobs = []
    for name, (_given, smiles) in zip(names, records, strict=True):
        job = copy.deepcopy(base)
        job["name"] = f"{stem}__{name}"
        job["entities"] = list(job["entities"]) + [
            {"type": "ligand", "id": chain, "smiles": smiles}
        ]
        if affinity:
            job["properties"] = [{"affinity": {"binder": chain}}]
        jobs.append(job)
    summary = {
        "target": str(target),
        "library": str(library),
        "jobs": len(jobs),
        "skipped": skipped,
        "ligand_chain": chain,
        "affinity": affinity,
    }
    return {"jobs": jobs}, summary


def _proteins(path: str | os.PathLike[str]) -> list[tuple[str, str]]:
    from foldjax.job import parse_fasta

    path = Path(path)
    records = parse_fasta(path.read_text(encoding="utf-8"))
    if not records:
        raise ValueError(f"no FASTA records in {path}")
    out = []
    seen: set[str] = set()
    for header, sequence in records:
        name = _clean(header.split()[0] if header.split() else "") or "protein"
        if name in seen:
            raise ValueError(
                f"{path}: two records are named {name!r}; FASTA ids name the jobs"
            )
        seen.add(name)
        out.append((name, "".join(sequence.split()).upper()))
    return out


def pulldown(
    baits: str | os.PathLike[str] | None,
    candidates: str | os.PathLike[str] | None,
    *,
    all_vs_all: bool = False,
    msa_dir: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pairwise protein-protein jobs: bait x candidate, or every unordered pair.

    Each job is chain A (the first protein) and chain B (the second), named
    ``<first>__<second>`` from the FASTA ids. With ``msa_dir``, a file
    ``<id>.a3m`` there becomes that chain's ``unpaired_msa``; otherwise
    alignments are left to ``--msa auto`` (cached per sequence, so each
    protein is searched once across the whole screen).
    """
    bait_list = _proteins(baits) if baits else []
    candidate_list = _proteins(candidates) if candidates else []
    if all_vs_all:
        pool: dict[str, str] = {}
        for name, sequence in [*bait_list, *candidate_list]:
            if name in pool and pool[name] != sequence:
                raise ValueError(f"{name!r} names two different sequences")
            pool[name] = sequence
        names = list(pool)
        if len(names) < 2:
            raise ValueError("--all-vs-all needs at least two proteins")
        pairs = [
            ((names[i], pool[names[i]]), (names[j], pool[names[j]]))
            for i in range(len(names))
            for j in range(i + 1, len(names))
        ]
    else:
        if not bait_list or not candidate_list:
            raise ValueError("give --baits and --candidates (or --all-vs-all)")
        pairs = [(bait, other) for bait in bait_list for other in candidate_list]
    msa = Path(msa_dir).absolute() if msa_dir is not None else None
    jobs = []
    missing: set[str] = set()
    for (first, first_seq), (second, second_seq) in pairs:
        entities = []
        for chain, (name, sequence) in zip(
            ("A", "B"), ((first, first_seq), (second, second_seq)), strict=True
        ):
            entity: dict[str, Any] = {
                "type": "protein",
                "id": chain,
                "sequence": sequence,
            }
            if msa is not None:
                candidate = msa / f"{name}.a3m"
                if candidate.is_file():
                    entity["unpaired_msa"] = str(candidate)
                else:
                    missing.add(name)
            entities.append(entity)
        jobs.append({"name": f"{first}__{second}", "entities": entities})
    summary = {
        "mode": "all-vs-all" if all_vs_all else "bait-x-candidate",
        "jobs": len(jobs),
        "missing_msas": sorted(missing),
    }
    return {"jobs": jobs}, summary


def write_jobs(document: Mapping[str, Any], out: str | os.PathLike[str]) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    staged = out.with_name(f".{out.name}.tmp")
    staged.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    os.replace(staged, out)
    return out


def screen_table(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per-job affinity/score rows, ranked within each model only.

    One row per (model, configuration, job): the model's own ranking score
    and, where the model has an affinity head (Boltz-2), its affinity outputs,
    taken from the sample the model itself ranks first. ``rank_within_model``
    orders jobs by the affinity prediction when present (lower
    ``affinity_pred_value`` = stronger binding, upstream's convention) and by
    the ranking score otherwise. Models are never ranked against each other.
    """
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        job = row.get("job") or row.get("input_name")
        key = (str(row.get("model")), str(row.get("configuration")), str(job))
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (model, configuration, job), members in groups.items():
        ranked = [
            row for row in members if isinstance(row.get("ranking_value"), (int, float))
        ]
        top = (
            max(ranked, key=lambda row: row["ranking_value"]) if ranked else members[0]
        )
        record: dict[str, Any] = {
            "model": model,
            "configuration": configuration,
            "job": job,
            "samples": len(members),
            "ranking_key": top.get("ranking_key"),
            "ranking_value": top.get("ranking_value"),
            "seed": top.get("seed"),
            "sample": top.get("sample"),
            "structure_path": top.get("structure_path"),
        }
        for name in sorted(top):
            if name.startswith("score.affinity"):
                record[name] = top[name]
        out.append(record)
    by_model: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in out:
        by_model.setdefault((record["model"], record["configuration"]), []).append(
            record
        )
    for members in by_model.values():
        affinity = [
            r
            for r in members
            if isinstance(r.get("score.affinity_pred_value"), (int, float))
        ]
        if affinity:
            order = sorted(affinity, key=lambda r: r["score.affinity_pred_value"])
            basis = "score.affinity_pred_value (lower binds tighter)"
        else:
            order = sorted(
                (
                    r
                    for r in members
                    if isinstance(r.get("ranking_value"), (int, float))
                ),
                key=lambda r: -r["ranking_value"],
            )
            basis = "ranking_value (the model's own ranking score)"
        for position, record in enumerate(order, start=1):
            record["rank_within_model"] = position
            record["rank_basis"] = basis
        for record in members:
            record.setdefault("rank_within_model", None)
            record.setdefault("rank_basis", basis)
    out.sort(
        key=lambda r: (
            r["model"],
            r["configuration"],
            r["rank_within_model"] is None,
            r["rank_within_model"] or 0,
            r["job"],
        )
    )
    return out
