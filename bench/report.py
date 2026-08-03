"""Render the collected rows as the comparison table.

Reads whatever `drive.py` has written so far, so an interrupted matrix still
reports every row that finished. A row that failed is printed as a failure
rather than omitted -- a model that could not run at a size is a result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(results: Path) -> list[dict]:
    rows = []
    for path in sorted(results.glob("*.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return rows


#: Preference order for the one confidence number shown per row. Each is a
#: whole-prediction score, best-first among the samples.
_CONFIDENCE_KEYS = (
    "ranking_score",
    "aggregate_score",
    "confidence_score",
    "complex_plddt",
    "ptm",
    "plddt",
)


def _available(row: dict | None) -> set[str]:
    if not row:
        return set()
    return {
        key
        for sample in row.get("samples") or []
        if isinstance(sample.get("scores"), dict)
        for key in sample["scores"]
    }


def _shared_key(left: dict | None, right: dict | None) -> str | None:
    """The best confidence field *both* sides reported, or None.

    Falling back independently on each side put `ptm` in one column and
    `confidence_score` in the other and printed them as though they were a
    comparison. These scores are only comparable between two implementations of
    the same model, and only when they are the same field.
    """
    common = _available(left) & _available(right)
    return next((key for key in _CONFIDENCE_KEYS if key in common), None)


def _confidence(row: dict | None, key: str | None) -> str:
    """Best sample's value of `key`, or '-' if this run has no such number."""
    if row is None or key is None:
        return "-"
    values = [
        sample["scores"][key]
        for sample in row.get("samples") or []
        if isinstance(sample.get("scores"), dict) and key in sample["scores"]
    ]
    return f"{max(values):.4f}" if values else "-"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    rows = load(args.results)
    if not rows:
        print("no results yet")
        return 0

    schedule = next(
        (row["schedule"] for row in rows if isinstance(row.get("schedule"), dict)), {}
    )
    if schedule:
        print(
            f"schedule: {schedule['num_samples']} samples, "
            f"{schedule['num_steps']} steps, {schedule['num_recycles']} recycles, "
            "seed 101, one alignment shared by both sides\n"
        )

    by_key: dict[tuple, dict] = {
        (row["model"], row["case"], row["impl"]): row for row in rows
    }
    cases = sorted(
        {(row["length"], row["case"]) for row in rows}, key=lambda item: item[0]
    )
    models = sorted({row["model"] for row in rows})

    header = (
        "| tokens | model | FoldJAX s | upstream s | FoldJAX MiB | upstream MiB "
        "| speed | memory | confidence | FoldJAX | upstream |"
    )
    print(header)
    print("|" + "---|" * 11)
    for length, case in cases:
        for model in models:
            fj = by_key.get((model, case, "foldjax"))
            up = by_key.get((model, case, "upstream"))
            if fj is None and up is None:
                continue

            def broken(row) -> bool:
                """A run that produced no structure did not run, whatever it exited."""
                return bool(
                    row.get("failed")
                    or row.get("returncode", 0) != 0
                    or not row.get("samples")
                )

            def cell(row, field, suffix=""):
                if row is None:
                    return "-"
                if broken(row):
                    return "failed"
                return f"{row[field]:,.0f}{suffix}"

            ratio_time = ratio_mem = "-"
            usable = fj is not None and up is not None and not (
                broken(fj) or broken(up)
            )
            if usable and up["wall_s"]:
                ratio_time = f"{up['wall_s'] / fj['wall_s']:.2f}x"
            if usable and fj["peak_mib"]:
                ratio_mem = f"{up['peak_mib'] / fj['peak_mib']:.2f}x"

            key = _shared_key(fj, up)
            print(
                f"| {length} | {model} | {cell(fj, 'wall_s')} | {cell(up, 'wall_s')} "
                f"| {cell(fj, 'peak_mib')} | {cell(up, 'peak_mib')} "
                f"| {ratio_time} | {ratio_mem} | {key or '-'} "
                f"| {_confidence(fj, key)} | {_confidence(up, key)} |"
            )
    print(
        "\nspeed/memory are upstream divided by FoldJAX: above 1.00x means "
        "FoldJAX is faster / uses less. `failed` means that side did not "
        "produce a structure at that size."
    )
    if args.markdown:
        print(_method_notes())
    return 0


def _method_notes() -> str:
    return """
Both columns run the same job. The upstream side's input is generated by
FoldJAX's own translator, so neither side is reading a hand-written
approximation of the other's job file, and both name the same alignment.

Both columns are measured the same way. Peak memory is the live-bytes
high-water mark -- `peak_bytes_in_use` under JAX, `max_memory_allocated` under
torch -- never `nvidia-smi`, which reports the caching allocator's reserved
pool and therefore tracks its doubling schedule rather than the model. Each
measurement is its own process, because a peak is a process-lifetime high-water
mark and two runs in one process report only the larger.

FoldJAX is timed warm: each case is run once to fill the XLA compile cache and
that run is discarded. A cold JAX run is mostly compilation, which the torch
side has no equivalent of -- at 132 tokens that was 174 s against 29 s, almost
all of it compiling. Compilation is paid once per shape and replayed from disk
afterwards, which is what a second prediction costs.

The confidence column names the best field *both* implementations report, so
the two numbers are the same statistic. They are still different samples: the
torch and JAX PRNG streams differ, so a seed does not put the two on the same
random tape. Same-seed parity was established per port against a matched tape
and is a separate exercise from this table.

A row is `failed` when that side produced no structure, which is not the same
as a non-zero exit. Upstream OpenDDE runs out of memory at 976 tokens -- it
asks for 40.8 GiB of triangle-attention softmax on top of 58.5 GiB already
held, on a 95 GiB card -- then catches the error, writes it to an `ERR/` file
and exits 0. Taken at its word that was a 12-second run that beat everything
else in the table.

Every upstream runs on its own fast path. Two of them did not at first, and
both differences were large enough to change the conclusions rather than shade
them -- see `bench/upstream-environments.md`:

* OpenDDE had no `cuequivariance`, so its `auto` triangle kernels fell back to
  plain torch. Installing the cu13 build took its 490-token peak from 91,191
  MiB to 18,580 and its time from 117 s to 71 s, and made 970 tokens run at all
  where it had been an out-of-memory failure.
* Protenix's fused layer norm is a CUDA extension built on first use, and this
  host has the CUDA runtime but no compiler. Completing the toolkit inside its
  own virtualenv, and adding sm_120 to an architecture list that stopped at
  compute_100, took its 970-token time from 235 s to 94 s.

Boltz-2 and Chai needed nothing: Boltz-2 already had cuEquivariance, and Chai
does not use it.
""".rstrip()


if __name__ == "__main__":
    raise SystemExit(main())
