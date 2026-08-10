"""Compare the structures FoldJAX and each upstream actually produced.

The confidence columns in `report.py` say what each implementation thinks of
its own prediction. This says how far apart the predictions themselves are --
and, more importantly, how far apart they *should* be expected to be.

That second part is the whole difficulty. The two sides do not share a random
tape: torch and JAX draw different diffusion noise from the same seed, so even
a bit-exact port would return different samples. A cross-implementation
TM-score is therefore uninterpretable on its own. What makes it interpretable
is the same number computed *within* each implementation, across its own
samples. If FoldJAX's five samples agree with each other no better than they
agree with upstream's, the implementations are as close as the model's own
sampling allows, and no tighter comparison exists to make.

For an unconverged target that spread is enormous -- Boltz-2 at 132 tokens
produces samples of its own that share TM 0.06 -- and reading a low
cross-implementation score there as a porting defect would be a mistake.

Usage:

    python -m bench.structures --work /path/to/bench-work --out summary.json

where `--work` is the directory `drive.py` was given, so that each
`<model>-<impl>-<case>` subdirectory still holds its written structures.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

import numpy as np


def ca_coords(path: Path) -> tuple[np.ndarray, list[str]]:
    """CA coordinates and their residue keys, from an mmCIF atom_site loop.

    The keys matter. Chai omits residues it did not resolve, and omits
    different ones per sample -- at 1,531 residues its five structures carry
    1,425 to 1,531 CA atoms. Comparing by position would then align residue i
    of one structure against residue j of another, and requiring equal counts
    (which this did at first) silently drops the pair instead, so a run that
    compared 1 of 25 pairs reported a median over that 1 as though it were the
    answer.

    A key also has to survive a writer that fills a column with a placeholder
    rather than omitting it. OpenFold3 writes `label_seq_id` as `.` and names
    every chain `Axp`, so a `.get(..., fallback)` -- which only fires when the
    column is *absent* -- gave all 1,003 residues the key `Axp:.`. They collapsed
    into one bucket and every OpenFold3 CA RMSD came out as exactly 0.000,
    between structures whose backbones demonstrably differ. Read the value, then
    fall back on the placeholder.
    """
    rows: list[tuple[float, float, float]] = []
    keys: list[str] = []
    header: list[str] = []
    in_loop = False
    for line in path.read_text().splitlines():
        if line.startswith("_atom_site."):
            header.append(line.strip().split(".", 1)[1])
            in_loop = True
            continue
        if not in_loop:
            continue
        if line.startswith("#") or not line.strip():
            if rows:
                break
            continue
        parts = line.split()
        if len(parts) < len(header):
            continue
        record = dict(zip(header, parts))
        if record.get("label_atom_id") != "CA":
            continue
        rows.append(
            (
                float(record["Cartn_x"]),
                float(record["Cartn_y"]),
                float(record["Cartn_z"]),
            )
        )
        chain = _first_real(record, ("label_asym_id", "auth_asym_id"), "?")
        number = _first_real(record, ("label_seq_id", "auth_seq_id"), str(len(keys)))
        keys.append(f"{chain}:{number}")
    return np.asarray(rows, dtype=np.float64), keys


def _first_real(record: dict[str, str], names: tuple[str, ...], default: str) -> str:
    """First field that is present and is not an mmCIF placeholder."""
    for name in names:
        value = record.get(name)
        if value not in (None, ".", "?"):
            return value
    return default


def common_residues(
    left: tuple[np.ndarray, list[str]], right: tuple[np.ndarray, list[str]]
) -> tuple[np.ndarray, np.ndarray]:
    """The two coordinate sets restricted to the residues they share."""
    left_index = {key: i for i, key in enumerate(left[1])}
    shared = [key for key in right[1] if key in left_index]
    if not shared:
        return np.empty((0, 3)), np.empty((0, 3))
    right_index = {key: i for i, key in enumerate(right[1])}
    li = np.asarray([left_index[k] for k in shared])
    ri = np.asarray([right_index[k] for k in shared])
    return left[0][li], right[0][ri]


def superpose(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kabsch-align `p` onto `q`, returning both centred."""
    pc = p - p.mean(axis=0)
    qc = q - q.mean(axis=0)
    u, _, vt = np.linalg.svd(pc.T @ qc)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return pc @ rotation.T, qc


def _tm_from_alignment(p: np.ndarray, q: np.ndarray, subset, d0: float, n: int):
    """TM of the full pair under the superposition fitted on `subset` only."""
    a, b = superpose(p[subset], q[subset])
    # Refit the whole structure with the rotation the subset chose.
    pc = p - p[subset].mean(axis=0)
    qc = q - q[subset].mean(axis=0)
    u, _, vt = np.linalg.svd((p[subset] - p[subset].mean(axis=0)).T @ b)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    distances = np.linalg.norm(pc @ rotation.T - qc, axis=-1)
    return float(np.sum(1.0 / (1.0 + (distances / d0) ** 2)) / n), distances


def tm_and_rmsd(p: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    """TM-score with the published iterative search, plus global-fit RMSD.

    A single global Kabsch fit understates TM badly on anything flexible or
    multi-domain -- one poorly-placed segment drags the whole superposition and
    two structures that share a fold can score below 0.2. TM-score is defined
    as the *maximum* over superpositions, and the standard heuristic finds it by
    seeding on fragments of decreasing length and iterating: superpose on the
    current subset, keep the residues that land within a cutoff, superpose
    again. Without that, the absolute numbers here mean nothing and only the
    ratio between `cross` and `within` is readable.

    RMSD stays on the global fit, where it is the conventional quantity.
    """
    n = len(p)
    d0 = 1.24 * (max(n, 19) - 15) ** (1 / 3) - 1.8
    best = -1.0

    seeds: list[np.ndarray] = []
    length = n
    while length >= 4:
        for start in range(0, n - length + 1, max(length, 1)):
            seeds.append(np.arange(start, start + length))
        length //= 2
    if not seeds:
        seeds = [np.arange(n)]

    for seed in seeds:
        subset = seed
        for _ in range(20):
            score, distances = _tm_from_alignment(p, q, subset, d0, n)
            best = max(best, score)
            # Grow the cutoff until enough residues survive to define a fit.
            cutoff = d0
            while True:
                keep = np.flatnonzero(distances < cutoff)
                if keep.size >= 4 or cutoff > 100.0:
                    break
                cutoff += 0.5
            if keep.size < 4 or (
                keep.size == subset.size and np.array_equal(keep, subset)
            ):
                break
            subset = keep

    a, b = superpose(p, q)
    rmsd = float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1))))
    return float(best), rmsd


def structures(directory: Path) -> list[Path]:
    """Every written .cif under one run directory, in a stable order."""
    found = sorted(
        path
        for path in directory.rglob("*.cif")
        if "input" not in path.parent.name.lower() or path.name.endswith(".cif")
    )
    return [path for path in found if path.stat().st_size > 0]


def _pairs(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def compare(left: list[np.ndarray], right: list[np.ndarray] | None) -> dict | None:
    """Pairwise TM over one set, or between two sets."""
    tms: list[float] = []
    rmsds: list[float] = []
    combos = (
        itertools.combinations(range(len(left)), 2)
        if right is None
        else itertools.product(range(len(left)), range(len(right)))
    )
    other = left if right is None else right
    dropped = 0
    for i, j in combos:
        a, b = common_residues(left[i], other[j])
        if len(a) < 4:
            dropped += 1
            continue
        tm, rmsd = tm_and_rmsd(a, b)
        tms.append(tm)
        rmsds.append(rmsd)
    if not tms:
        return None
    return {"tm": _pairs(tms), "rmsd": _pairs(rmsds), "dropped": dropped}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    runs: dict[tuple[str, str, str], list[np.ndarray]] = {}
    for directory in sorted(args.work.iterdir()):
        if not directory.is_dir() or directory.name.endswith("-warmup"):
            continue
        parts = directory.name.split("-")
        if len(parts) < 3:
            continue
        model, impl, case = parts[0], parts[1], parts[-1]
        coords = [ca_coords(path) for path in structures(directory)]
        coords = [c for c in coords if c[0].size]
        if coords:
            runs[(model, impl, case)] = coords

    rows = []
    keys = sorted({(model, case) for model, _impl, case in runs})
    for model, case in keys:
        fj = runs.get((model, "foldjax", case))
        up = runs.get((model, "upstream", case))
        rows.append(
            {
                "model": model,
                "case": case,
                "foldjax_samples": len(fj or []),
                "upstream_samples": len(up or []),
                "within_foldjax": compare(fj, None) if fj else None,
                "within_upstream": compare(up, None) if up else None,
                "cross": compare(fj, up) if fj and up else None,
            }
        )

    if args.results is not None:
        for row in rows:
            for impl in ("foldjax", "upstream"):
                path = args.results / f"{row['model']}-{impl}-{row['case']}.json"
                if not path.exists():
                    continue
                document = json.loads(path.read_text())
                scores = [
                    sample.get("scores", {}) for sample in document.get("samples") or []
                ]
                row[f"{impl}_scores"] = scores

    print(
        "| model | case | cross TM | within FoldJAX TM | within upstream TM "
        "| cross RMSD A |"
    )
    print("|" + "---|" * 6)
    for row in rows:

        def cell(block, field="tm"):
            if not block or not block[field]:
                return "-"
            values = block[field]
            unit = "" if field == "tm" else ""
            return (
                f"{values['median']:.3f}{unit} "
                f"({values['min']:.3f}-{values['max']:.3f})"
            )

        print(
            f"| {row['model']} | {row['case']} | {cell(row['cross'])} "
            f"| {cell(row['within_foldjax'])} | {cell(row['within_upstream'])} "
            f"| {cell(row['cross'], 'rmsd')} |"
        )
    print(
        "\nTM-score, iterative search, median over all pairs (min-max in "
        "brackets). Read `cross` against the two `within` columns: torch and "
        "JAX draw different diffusion noise from the same seed, so `within` -- "
        "how well an implementation agrees with *itself* across samples -- is "
        "the closest any correct port could come. `cross` at or above `within` "
        "means the two are as close as this model's own sampling allows."
    )

    if args.out is not None:
        args.out.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
