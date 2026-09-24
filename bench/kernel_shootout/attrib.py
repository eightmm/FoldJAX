"""Name the buffers behind each arm's memory from XLA GPU dumps (x48).

Each arm's run dumps XLA with `--xla_dump_to=<run>/dump --xla_dump_hlo_as_text`,
one family per process. XLA writes, per module, the buffer assignment, its
values, a memory-usage report and an HloLiveRange with a `Live ranges at N (peak)`
section: every buffer live at the program's peak step, in bytes. For each run
this script:

* picks the family's module, the one whose buffer assignment has the largest
  preallocated temp allocation;
* prints the buffers live at the peak step of at least `--min-mib`, each with
  its shape and dtype (from the buffer assignment) and its producing
  instruction's opcode (custom-call target, fusion kind, `(aliased)` when the
  instruction carries output-to-operand aliasing), plus the instruction the
  peak step is at;
* prints the memory-usage report's top rows, and a census by opcode of
  instructions whose result is at least `--min-mib`;
* with `--diff BASE`, diffs the peak sets against BASE by (shape, producer
  opcode). That answers what one arm holds at its peak that the other does not.

    python attrib.py RUNS_DIR --diff released-msa_layer [--min-mib 32]
"""

# ruff: noqa: E501

import argparse
import re
from collections import Counter
from pathlib import Path

DTYPE_BYTES = {"f32": 4, "bf16": 2, "f16": 2, "s32": 4, "u32": 4, "pred": 1, "s8": 1, "u8": 1, "f64": 8, "s64": 8}
OWNERS = {"copy", "transpose", "custom-call", "fusion", "dot", "concatenate", "dynamic-update-slice", "while", "convolution"}
SHAPE = re.compile(r"\b(f32|bf16|f16|s32|u32|pred|s8|u8|f64|s64)\[([0-9,]*)\]")


def nbytes(dtype: str, dims: str) -> int:
    size = DTYPE_BYTES.get(dtype, 4)
    for d in filter(None, dims.split(",")):
        size *= int(d)
    return size


def temp_size(path: Path) -> int:
    best = 0
    for line in path.read_text(errors="replace").splitlines():
        m = re.match(r"allocation \d+: size (\d+),.*preallocated-temp", line.strip())
        if m:
            best = max(best, int(m.group(1)))
    return best


def value_shapes(path: Path) -> dict[str, str]:
    shapes = {}
    for line in path.read_text(errors="replace").splitlines():
        v = re.match(r"\s*value: <\d+ (\S+) @\d+> \(size=\d+,offset=\d+\): (\S+)", line)
        if v:
            shapes.setdefault(v.group(1), v.group(2).split("{")[0])
    return shapes


def hlo_index(path: Path) -> dict[str, str]:
    index = {}
    for line in path.read_text(errors="replace").splitlines():
        m = re.match(r"\s*(?:ROOT )?%?([\w.\-]+) = (\S+) ([\w\-]+)\((.*)", line)
        if not m:
            continue
        name, _, opcode, rest = m.groups()
        detail = opcode
        t = re.search(r'custom_call_target="([^"]+)"', rest)
        if t:
            detail += f" {t.group(1)}"
        k = re.search(r"kind=(\w+)", rest)
        if k:
            detail += f" kind={k.group(1)}"
        if "output_to_operand_aliasing" in rest:
            detail += " (aliased)"
        # What a copy, a custom call or a fusion reads: the suspects are a
        # layout copy of the transition's operand and a copy of `m` kept for a
        # second consumer, and both are named by their operand.
        operands = re.findall(r"%([\w.\-]+)", rest.split("),")[0])
        if opcode in ("copy", "custom-call", "fusion", "transpose") and operands:
            detail += " <- " + ", ".join(operands[:3])
        op_name = re.search(r'op_name="([^"]+)"', rest)
        if op_name:
            detail += f"  [{op_name.group(1)[-60:]}]"
        index.setdefault(name, detail)
    return index


def peak_entries(path: Path):
    lines = path.read_text(errors="replace").splitlines()
    step, entries = None, []
    sequence = {}
    in_seq = False
    for line in lines:
        if line.strip().startswith("InstructionSequence:"):
            in_seq = True
            continue
        if in_seq:
            m = re.match(r"\s*(\d+):(\S+)", line)
            if m:
                sequence[int(m.group(1))] = m.group(2)
                continue
            in_seq = False
    for i, line in enumerate(lines):
        m = re.match(r"\s*Live ranges at (\d+) \(peak\):", line)
        if m:
            step = int(m.group(1))
            for entry in lines[i + 1 :]:
                e = re.match(r"\s*([\w.\-]+)(\{[^}]*\})?: (\d+) bytes", entry)
                if not e:
                    break
                entries.append((e.group(1), e.group(2) or "{}", int(e.group(3))))
            break
    return step, sequence.get(step), entries


def large_instructions(path: Path, min_bytes: int) -> Counter:
    counts = Counter()
    for line in path.read_text(errors="replace").splitlines():
        m = re.match(r"\s*(?:ROOT )?%?([\w.\-]+) = (\S+?) ([\w\-]+)\(", line)
        if not m:
            continue
        s = SHAPE.search(m.group(2))
        # Entry-level opcodes only: the ones that own a buffer. Instructions
        # inside a fusion body are named here too and would drown the census.
        if m.group(3) not in OWNERS:
            continue
        if s and nbytes(*s.groups()) >= min_bytes:
            counts[(m.group(3), f"{s.group(1)}[{s.group(2)}]")] += 1
    return counts


def analyse(run: Path, min_bytes: int):
    dump = run / "dump"
    best = None
    for ba in dump.glob("*buffer-assignment.txt"):
        size = temp_size(ba)
        if best is None or size > best[0]:
            best = (size, ba)
    if best is None:
        return None
    size, ba = best
    stem = ba.name[: -len("-buffer-assignment.txt")]
    hlo = dump / f"{stem}.txt"
    live = dump / f"{stem}-live-range.txt"
    report = dump / f"{stem}-memory-usage-report.txt"
    shapes = value_shapes(ba)
    index = hlo_index(hlo) if hlo.exists() else {}
    step, at, entries = peak_entries(live) if live.exists() else (None, None, [])
    rows = [(b, n, idx, shapes.get(n, "?"), index.get(n, "parameter?")) for n, idx, b in entries]
    return {"stem": stem, "temp": size, "step": step, "at": at, "rows": rows, "report": report, "hlo": hlo}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--diff", default=None)
    ap.add_argument("--min-mib", type=float, default=32.0)
    args = ap.parse_args()
    min_bytes = int(args.min_mib * 2**20)
    results = {}
    for run in sorted(p for p in Path(args.runs).iterdir() if (p / "dump").is_dir()):
        r = analyse(run, min_bytes)
        if r is None:
            print(f"== {run.name}: no buffer-assignment dump")
            continue
        results[run.name] = r
        total = sum(b for b, *_ in r["rows"])
        print(f"\n== {run.name}: {r['stem']}")
        print(f"   preallocated temp {r['temp'] / 2**20:.1f} MiB; peak step {r['step']} at {r['at']} ({index_of(r)}); live at peak {total / 2**20:.1f} MiB")
        for b, name, idx, shape, what in sorted(r["rows"], reverse=True):
            if b >= min_bytes:
                print(f"   {b / 2**20:9.1f}  {name + idx:44s} {shape:32s} {what}")
        if r["report"].exists():
            print("   -- memory-usage report (top) --")
            for line in r["report"].read_text().splitlines()[:14]:
                print("   " + line[:190])
        if r["hlo"].exists():
            print(f"   -- instructions >= {args.min_mib:g} MiB by opcode --")
            for (op, shape), n in sorted(large_instructions(r["hlo"], min_bytes).items()):
                print(f"   {n:4d} x {op:24s} {shape}")
    if args.diff and args.diff in results:
        family = args.diff.split("-", 1)[1]
        base = Counter()
        for b, name, idx, shape, what in results[args.diff]["rows"]:
            base[(shape, producer_class(what))] += b
        for name, r in results.items():
            if name == args.diff or not name.endswith(family):
                continue
            mine = Counter()
            for b, n, idx, shape, what in r["rows"]:
                mine[(shape, producer_class(what))] += b
            print(f"\n== live at peak in {name} minus {args.diff}:")
            for key in sorted(set(mine) | set(base), key=lambda k: -(mine.get(k, 0) - base.get(k, 0))):
                delta = mine.get(key, 0) - base.get(key, 0)
                if abs(delta) >= min_bytes:
                    print(f"   {delta / 2**20:+10.1f} MiB  {key[0]:36s} {key[1]}")


def producer_class(what: str) -> str:
    """Opcode, plus the custom-call target: the diff's grouping key."""
    words = what.split()
    if not words:
        return "?"
    if words[0] == "custom-call" and len(words) > 1:
        return f"custom-call {words[1]}"
    if words[0] == "fusion" and len(words) > 1 and words[1].startswith("kind="):
        return f"fusion {words[1]}"
    return words[0]


def index_of(r):
    return "" if r["at"] is None else r["at"]


if __name__ == "__main__":
    main()
