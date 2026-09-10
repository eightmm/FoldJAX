"""Copy CPU-parity fixtures out of a stored capture, by digest.

    python -m tests.parity.fetch --from <capture directory>

Matching is by content, not by name: the capture directories keep their own
layout (``native-A/trunk.npz``, ``torch/tape.npz``), and a file that hashes to
what the manifest records is that file whatever it is called there. Nothing is
written under ``--from``; the capture roots are read-only evidence.

Sizes are compared before anything is hashed. A capture directory holds
checkpoints and 490 MB chemistry assets, and hashing all of it to find 14 MB of
tape would make the helper unusable.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ._fixtures import ENV_VAR, fixtures_root, sha256_of
from ._manifest import FixtureFile, ManifestEntry, all_entries


@dataclass(frozen=True)
class Want:
    entry: ManifestEntry
    spec: FixtureFile

    @property
    def destination_dir(self) -> Path:
        return Path(self.entry.port) / self.entry.case

    def __str__(self) -> str:
        return f"{self.entry.port}/{self.entry.case}/{self.spec.name}"


def _wanted(manifest_dir: Path | None, ports: Sequence[str]) -> list[Want]:
    wants = [
        Want(entry=entry, spec=spec)
        for entry in all_entries(manifest_dir)
        if not ports or entry.port in ports
        for spec in entry.files.values()
    ]
    return sorted(wants, key=str)


def _already_there(want: Want, dest: Path) -> bool:
    path = dest / want.destination_dir / want.spec.name
    if not path.is_file() or path.stat().st_size != want.spec.size_bytes:
        return False
    return sha256_of(path) == want.spec.sha256


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(f"{target.name}.partial")
    shutil.copyfile(source, staged)
    os.replace(staged, target)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.parity.fetch")
    parser.add_argument(
        "--from",
        dest="sources",
        action="append",
        required=True,
        metavar="DIR",
        help="capture directory to search (repeatable)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help=f"fixture root to fill (default: ${ENV_VAR} or {fixtures_root()})",
    )
    parser.add_argument(
        "--port", action="append", default=[], help="only this port (repeatable)"
    )
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    dest = args.dest if args.dest is not None else fixtures_root()
    wants = _wanted(args.manifest_dir, args.port)
    if not wants:
        print("no manifest entries selected: nothing to fetch", file=sys.stderr)
        return 1

    outstanding: dict[str, list[Want]] = {}
    for want in wants:
        if _already_there(want, dest):
            print(f"present  {want}")
            continue
        outstanding.setdefault(want.spec.sha256, []).append(want)
    if not outstanding:
        return 0

    sizes = {want.spec.size_bytes for group in outstanding.values() for want in group}
    for source in args.sources:
        root = Path(source).expanduser()
        if not root.is_dir():
            print(f"not a directory: {root}", file=sys.stderr)
            return 1
        for candidate in sorted(root.rglob("*")):
            if not outstanding:
                break
            if not candidate.is_file() or candidate.stat().st_size not in sizes:
                continue
            group = outstanding.pop(sha256_of(candidate), None)
            if group is None:
                continue
            for want in group:
                target = dest / want.destination_dir / want.spec.name
                if args.dry_run:
                    print(f"would copy {candidate} -> {target}")
                    continue
                _copy(candidate, target)
                if not _already_there(want, dest):
                    print(f"copy of {candidate} did not verify", file=sys.stderr)
                    return 1
                print(f"fetched  {want}  <- {candidate}")

    if outstanding:
        for group in outstanding.values():
            for want in group:
                print(
                    f"NOT FOUND {want} (sha256 {want.spec.sha256}, "
                    f"{want.spec.size_bytes} bytes); recorded capture: "
                    f"{want.entry.capture_provenance}",
                    file=sys.stderr,
                )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
