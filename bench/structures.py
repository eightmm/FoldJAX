"""Compare the structures FoldJAX and each upstream actually produced.

The confidence columns in `report.py` say what each implementation thinks of
its own prediction. This says how far apart the predictions themselves are --
and, more importantly, how far apart they *should* be expected to be.

That second part is the whole difficulty. The two sides do not share a random
tape: torch and JAX draw different diffusion noise even from the same seed, so
even a bit-exact port would return different samples. Historical OpenFold3
upstream artifacts recorded before the 2026-08-31 harness fix additionally use
a generated seed rather than the requested 101. A cross-implementation
TM-score is therefore uninterpretable on its own. What makes it interpretable
is the same number computed *within* each implementation, across its own
samples. If FoldJAX's five samples agree with each other no better than they
agree with upstream's, the implementations are as close as the model's own
sampling allows, and no tighter comparison exists to make.

For an unconverged target that spread is enormous -- Boltz-2 at 132 tokens
produces samples of its own that share TM 0.06 -- and reading a low
cross-implementation score there as a porting defect would be a mistake.

Homomers need one more step before any of that is readable. Residues are paired
by (chain id, residue id), and nothing makes two implementations -- or two
samples of one implementation -- assign the same label to the same copy of a
repeated chain. On 2026-09-10 that put OpenFold3's homotetramer at 37 A and
TM 0.569 while each side agreed with itself to TM 1.000, and it split Boltz-2's
own 4k samples across TM 0.59-1.00. Interchangeable chains are therefore
matched by minimum RMSD before the score is computed; see `best_assignment`.

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
from typing import NamedTuple

import numpy as np


class CAStructure(NamedTuple):
    """One prediction's CA trace: coordinates, residue keys, chains, residues.

    `coords` and `keys` are what the comparison has always used, in that order,
    so index access still reads. `chains` and `comps` are what grouping
    interchangeable chains needs: which chain each CA belongs to, and the
    residue it is part of.
    """

    coords: np.ndarray
    keys: list[str]
    chains: list[str]
    comps: list[str]


MAX_ASSIGNMENTS = 720
"""Chain assignments to score exhaustively before falling back to centroids.

720 is 6! -- six interchangeable copies. Real complexes here are tetramers (24),
so the fallback exists for correctness at scale rather than for any measured
case, and its own test lowers this to reach it.
"""

_ONE_LETTER = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def ca_coords(path: Path) -> CAStructure:
    """CA coordinates and their residue keys, from an mmCIF atom_site loop.

    The keys matter. Historical Chai benchmark artifacts omit residues they did
    not resolve, and omit different ones per sample -- at 1,531 residues their
    five structures carry 1,425 to 1,531 CA atoms. Comparing by position would
    then align residue i of one structure against residue j of another, and
    requiring equal counts (which this did at first) silently drops the pair
    instead, so a run that compared 1 of 25 pairs reported a median over that 1
    as though it were the answer.

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
    chains: list[str] = []
    comps: list[str] = []
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
        chains.append(chain)
        comps.append(_first_real(record, ("label_comp_id", "auth_comp_id"), "UNK"))
    return CAStructure(np.asarray(rows, dtype=np.float64), keys, chains, comps)


def _first_real(record: dict[str, str], names: tuple[str, ...], default: str) -> str:
    """First field that is present and is not an mmCIF placeholder."""
    for name in names:
        value = record.get(name)
        if value not in (None, ".", "?"):
            return value
    return default


def common_residues(
    left: CAStructure, right: CAStructure
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


def kabsch(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The rotation taking centred `p` onto centred `q`, and the two centroids.

    Split out of `superpose` so the chain-assignment fallback can apply the
    same transform to atoms that were not part of the fit.
    """
    pm = p.mean(axis=0)
    qm = q.mean(axis=0)
    u, _, vt = np.linalg.svd((p - pm).T @ (q - qm))
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, sign]) @ u.T, pm, qm


def superpose(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kabsch-align `p` onto `q`, returning both centred."""
    rotation, pm, qm = kabsch(p, q)
    return (p - pm) @ rotation.T, q - qm


def rmsd(p: np.ndarray, q: np.ndarray) -> float:
    """RMSD after a single global fit -- the conventional quantity."""
    a, b = superpose(p, q)
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1))))


def chain_sequences(structure: CAStructure) -> dict[str, str]:
    """One-letter sequence per chain, over the residues that carry a CA.

    A residue outside the standard twenty keeps its component id rather than
    collapsing to `X`: two different ligands must not read as the same entity
    and so become interchangeable.
    """
    letters: dict[str, list[str]] = {}
    for chain, comp in zip(structure.chains, structure.comps, strict=True):
        letters.setdefault(chain, []).append(_ONE_LETTER.get(comp, f"({comp})"))
    return {chain: "".join(seq) for chain, seq in letters.items()}


def _by_sequence(structure: CAStructure) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for chain, sequence in chain_sequences(structure).items():
        grouped.setdefault(sequence, []).append(chain)
    return grouped


def interchangeable_groups(
    left: CAStructure, right: CAStructure
) -> list[tuple[list[str], list[str]]]:
    """Chains of one sequence, per side, for every sequence both sides carry.

    Chains keep the order they appear in the file, which is what makes the
    identity assignment the first candidate below.
    """
    left_groups = _by_sequence(left)
    right_groups = _by_sequence(right)
    return [
        (left_groups[sequence], right_groups[sequence])
        for sequence in left_groups
        if sequence in right_groups
    ]


def _is_identity(mapping: dict[str, str]) -> bool:
    return all(ours == theirs for ours, theirs in mapping.items())


def _group_candidates(lefts: list[str], rights: list[str]) -> list[dict[str, str]]:
    """Every injective pairing of one group, the identity-by-name one first."""
    if len(lefts) <= len(rights):
        pairings = [
            dict(zip(lefts, order, strict=True))
            for order in itertools.permutations(rights, len(lefts))
        ]
    else:
        pairings = [
            dict(zip(order, rights, strict=True))
            for order in itertools.permutations(lefts, len(rights))
        ]
    pairings.sort(key=lambda mapping: not _is_identity(mapping))
    return pairings


def relabel(right: CAStructure, mapping: dict[str, str]) -> CAStructure | None:
    """`right` with each assigned chain renamed to the left chain it answers.

    None when the renaming would give two chains the same label, which would
    make the residue keys ambiguous. The identity mapping returns `right`
    itself, so a structure that needs no permutation is compared through the
    exact arrays it always was.
    """
    rename = {theirs: ours for ours, theirs in mapping.items()}
    labels = [rename.get(chain, chain) for chain in right.chains]
    if labels == right.chains:
        return right
    if len(set(labels)) != len(set(right.chains)):
        return None
    keys = [
        f"{label}:{key.split(':', 1)[1]}"
        for label, key in zip(labels, right.keys, strict=True)
    ]
    return CAStructure(right.coords, keys, labels, right.comps)


def _seed_mapping(groups: list[tuple[list[str], list[str]]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for lefts, rights in groups:
        mapping.update(dict(zip(lefts, rights)))
    return mapping


def _search_assignment(
    left: CAStructure,
    right: CAStructure,
    groups: list[tuple[list[str], list[str]]],
) -> dict[str, str]:
    """The assignment with the lowest globally-superposed RMSD.

    Ties keep the earlier candidate, and the identity pairing is generated
    first, so two predictions that already correspond are never reshuffled.
    """
    best: dict[str, str] | None = None
    best_score: float | None = None
    for combination in itertools.product(
        *(_group_candidates(lefts, rights) for lefts, rights in groups)
    ):
        mapping = {k: v for part in combination for k, v in part.items()}
        renamed = relabel(right, mapping)
        if renamed is None:
            continue
        a, b = common_residues(left, renamed)
        if len(a) < 4:
            continue
        score = rmsd(a, b)
        if best_score is None or score < best_score:
            best, best_score = mapping, score
    return best if best is not None else _seed_mapping(groups)


def _chain_rows(structure: CAStructure) -> dict[str, np.ndarray]:
    rows: dict[str, list[int]] = {}
    for position, chain in enumerate(structure.chains):
        rows.setdefault(chain, []).append(position)
    return {chain: np.asarray(index) for chain, index in rows.items()}


def _centroid_assignment(
    left: CAStructure,
    right: CAStructure,
    groups: list[tuple[list[str], list[str]]],
) -> dict[str, str]:
    """Hungarian matching on chain centroids, for groups too large to enumerate.

    Centroids only separate the copies once the two structures are in a common
    frame, and fitting that frame needs an assignment -- the circularity the
    exhaustive search avoids by trying all of them. Seeding the fit on the
    written chain order is not enough: with two copies of a tetramer swapped the
    fit splits the difference, the centroids move with it, and the matching
    returns the labelling it started from. So the frame is also seeded on single
    chain pairs, one at a time, which no labelling of the rest can disturb. Each
    seed produces one assignment; the one with the lowest RMSD wins.

    This is still a heuristic. It searches a few candidate frames rather than
    every assignment, and a complex whose copies are near-coincident in every
    one of them can be matched wrongly.
    """
    from scipy.optimize import linear_sum_assignment

    left_rows = _chain_rows(left)
    right_rows = _chain_rows(right)

    def assign(fit: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict[str, str]:
        rotation, pm, qm = fit
        moved = (left.coords - pm) @ rotation.T
        target = right.coords - qm
        mapping: dict[str, str] = {}
        for lefts, rights in groups:
            ours = np.stack([moved[left_rows[name]].mean(axis=0) for name in lefts])
            theirs = np.stack(
                [target[right_rows[name]].mean(axis=0) for name in rights]
            )
            cost = np.linalg.norm(ours[:, None, :] - theirs[None, :, :], axis=-1)
            rows, columns = linear_sum_assignment(cost)
            for row, column in zip(rows, columns, strict=True):
                mapping[lefts[row]] = rights[column]
        return mapping

    seeds = []
    written_order = _seed_mapping(groups)
    renamed = relabel(right, written_order)
    a, b = common_residues(left, renamed if renamed is not None else right)
    if len(a) >= 4:
        seeds.append(kabsch(a, b))
    for lefts, rights in groups:
        anchor = left.coords[left_rows[lefts[0]]]
        for name in rights:
            partner = right.coords[right_rows[name]]
            if len(anchor) == len(partner) and len(anchor) >= 4:
                seeds.append(kabsch(anchor, partner))

    best: dict[str, str] | None = None
    best_score: float | None = None
    for seed in seeds:
        mapping = assign(seed)
        candidate = relabel(right, mapping)
        if candidate is None:
            continue
        p, q = common_residues(left, candidate)
        if len(p) < 4:
            continue
        score = rmsd(p, q)
        if best_score is None or score < best_score:
            best, best_score = mapping, score
    return best if best is not None else written_order


def best_assignment(
    left: CAStructure, right: CAStructure
) -> tuple[dict[str, str], bool]:
    """Which chain of `right` answers which chain of `left`, and whether it moved.

    Only chains whose sequence appears on both sides are candidates for
    reassignment; everything else keeps the label it was written with.
    """
    groups = interchangeable_groups(left, right)
    if not groups:
        return {}, False
    total = 1
    for lefts, rights in groups:
        larger, smaller = max(len(lefts), len(rights)), min(len(lefts), len(rights))
        for taken in range(smaller):
            total *= larger - taken
    if total == 1:
        mapping = _seed_mapping(groups)
    elif total > MAX_ASSIGNMENTS:
        mapping = _centroid_assignment(left, right, groups)
    else:
        mapping = _search_assignment(left, right, groups)
    return mapping, not _is_identity(mapping)


def aligned_residues(
    left: CAStructure, right: CAStructure
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Shared residues under the best chain assignment, and whether it permuted."""
    mapping, permuted = best_assignment(left, right)
    if not permuted:
        return (*common_residues(left, right), False)
    renamed = relabel(right, mapping)
    if renamed is None:
        return (*common_residues(left, right), False)
    return (*common_residues(left, renamed), True)


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

    return float(best), rmsd(p, q)


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


def compare(
    left: list[CAStructure], right: list[CAStructure] | None
) -> dict | None:
    """Pairwise TM over one set, or between two sets.

    `permuted` counts the scored pairs whose interchangeable chains had to be
    reassigned. It is the difference between a homomer that disagrees and one
    that was merely labelled in another order.
    """
    tms: list[float] = []
    rmsds: list[float] = []
    permuted = 0
    combos = (
        itertools.combinations(range(len(left)), 2)
        if right is None
        else itertools.product(range(len(left)), range(len(right)))
    )
    other = left if right is None else right
    dropped = 0
    for i, j in combos:
        a, b, moved = aligned_residues(left[i], other[j])
        if len(a) < 4:
            dropped += 1
            continue
        tm, distance = tm_and_rmsd(a, b)
        tms.append(tm)
        rmsds.append(distance)
        permuted += int(moved)
    if not tms:
        return None
    return {
        "tm": _pairs(tms),
        "rmsd": _pairs(rmsds),
        "dropped": dropped,
        "permuted": permuted,
    }


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
        # Split on the implementation, not on every hyphen. A model whose name
        # contains one -- `protenix-v2` -- parsed as model "protenix" with
        # implementation "v2", which matches neither side, so every one of its
        # runs was dropped and the table simply had one fewer model in it than
        # the matrix had measured.
        for impl in ("foldjax", "upstream"):
            model, marker, case = directory.name.partition(f"-{impl}-")
            if marker:
                break
        else:
            continue
        if not case:
            continue
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
        "| cross RMSD A | chain perm |"
    )
    print("|" + "---|" * 7)
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

        def permutations(row):
            parts = [
                f"{label} {block['permuted']}/{block['tm']['n']}"
                for label, block in (
                    ("cross", row["cross"]),
                    ("fj", row["within_foldjax"]),
                    ("up", row["within_upstream"]),
                )
                if block and block.get("permuted")
            ]
            return "; ".join(parts) if parts else "-"

        print(
            f"| {row['model']} | {row['case']} | {cell(row['cross'])} "
            f"| {cell(row['within_foldjax'])} | {cell(row['within_upstream'])} "
            f"| {cell(row['cross'], 'rmsd')} | {permutations(row)} |"
        )
    print(
        "\nTM-score, iterative search, median over all pairs (min-max in "
        "brackets). Read `cross` against the two `within` columns: torch and "
        "JAX do not share a diffusion random tape even from the same seed; "
        "historical OpenFold3 upstream artifacts also used a different "
        "effective seed. Therefore `within` -- how well an implementation "
        "agrees with *itself* across samples -- is the closest any correct port "
        "could come. `cross` at or above `within` means the two are as close as "
        "this model's own sampling allows. `chain perm` counts the scored "
        "pairs -- cross, within FoldJAX, within upstream -- whose "
        "interchangeable chains had to be reassigned before scoring, because "
        "the two structures labelled the identical copies in a different "
        "order. It is a labelling difference, not a structural one."
    )

    if args.out is not None:
        args.out.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
