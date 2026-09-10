"""Loader and validator for the per-port CPU-parity fixture manifests.

The manifests are the only part of a fixture that lives in git: the ``.npz``
files themselves are 7-16 MB per case against a 22 MB pack with no LFS, so the
repository carries their digests and the files are fetched by digest into a
directory outside the tree (``tests/parity/_fixtures.py``).

Validation is deliberately strict -- unknown keys are rejected, not ignored --
because a manifest that silently drops a key would certify a fixture nobody
checked. ``docs/parity-cpu.md`` and ``tests/parity/manifest/README.md`` carry
the schema in prose; this module is what enforces it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Version of the *manifest file format* below. Bumped when this module stops
#: accepting what it accepts today; unrelated to the per-entry `tripwire`,
#: which versions the capture that produced the fixture.
SCHEMA_VERSION = 1

MANIFEST_DIR = Path(__file__).resolve().parent / "manifest"

#: Tier A is a trunk-boundary comparison against a stored native trunk capture
#: (minutes); tier B replays the full stored tape to coordinates (nightly).
TIERS = ("A", "B")

_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")

_TOP_LEVEL_KEYS = frozenset({"schema_version", "port", "entries"})
_ENTRY_REQUIRED = (
    "case",
    "tier",
    "tokens",
    "capture",
    "tripwire",
    "files",
    "gpu_residual_angstrom",
    "cpu_calibration",
    "tolerance",
    "test_node_ids",
)
_ENTRY_OPTIONAL = ("excluded_samples", "notes")
_CAPTURE_REQUIRED = ("provenance", "commit")
_FILE_KEYS = frozenset({"sha256", "bytes"})
_GPU_REQUIRED = ("max",)
_GPU_OPTIONAL = ("per_sample",)
_CALIBRATION_REQUIRED = ("residual_angstrom", "wall_seconds", "source_commit")
_CALIBRATION_OPTIONAL = ("host", "recorded")
_TOLERANCE_REQUIRED = ("metric", "value", "set_from")

#: Node ids in a manifest entry name tests in this package, and `test_gate.py`
#: asserts they are actually collected under `-m cpu_parity`.
NODE_ID_PREFIX = "tests/parity/"


class ManifestError(ValueError):
    """A manifest does not satisfy the schema."""


@dataclass(frozen=True)
class FixtureFile:
    name: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ManifestEntry:
    port: str
    case: str
    tier: str
    tokens: int
    capture_provenance: str
    capture_commit: str
    tripwire: Mapping[str, str]
    files: Mapping[str, FixtureFile]
    gpu_residual_max: float
    gpu_residual_per_sample: tuple[float, ...] | None
    cpu_residual: float
    cpu_wall_seconds: float
    cpu_source_commit: str
    tolerance_metric: str
    tolerance_value: float
    tolerance_set_from: str
    test_node_ids: tuple[str, ...]
    excluded_samples: tuple[int, ...]
    notes: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.port, self.case, self.tier)

    @property
    def capture_dir(self) -> Path:
        """Directory to hand `python -m tests.parity.fetch --from`."""
        return Path(self.capture_provenance).parent


@dataclass(frozen=True)
class PortManifest:
    port: str
    source: str
    entries: tuple[ManifestEntry, ...]


def _fail(source: str, where: str, problem: str) -> ManifestError:
    return ManifestError(f"{source}: {where} {problem}")


def _mapping(source: str, where: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(source, where, f"must be an object, got {type(value).__name__}")
    return value


def _check_keys(
    source: str,
    where: str,
    value: Mapping[str, Any],
    required: Sequence[str],
    optional: Sequence[str] = (),
) -> None:
    missing = [key for key in required if key not in value]
    if missing:
        raise _fail(source, where, f"is missing {', '.join(sorted(missing))}")
    unknown = set(value) - set(required) - set(optional)
    if unknown:
        raise _fail(source, where, f"has unknown key(s) {', '.join(sorted(unknown))}")


def _text(source: str, where: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(source, where, "must be a non-empty string")
    return value


def _number(source: str, where: str, value: Any, *, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _fail(source, where, "must be a number")
    if value < minimum:
        raise _fail(source, where, f"must be >= {minimum}, got {value}")
    return float(value)


def _positive_int(source: str, where: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(source, where, "must be an integer")
    if value <= 0:
        raise _fail(source, where, f"must be > 0, got {value}")
    return value


def _parse_files(source: str, where: str, value: Any) -> dict[str, FixtureFile]:
    files = _mapping(source, where, value)
    if not files:
        raise _fail(source, where, "must list at least one file")
    parsed: dict[str, FixtureFile] = {}
    for name, spec in files.items():
        at = f"{where}[{name!r}]"
        if not name or "/" in name or "\\" in name or name in {".", ".."}:
            raise _fail(source, at, "must be a plain file name")
        entry = _mapping(source, at, spec)
        unknown = set(entry) - _FILE_KEYS
        if unknown:
            raise _fail(source, at, f"has unknown key(s) {', '.join(sorted(unknown))}")
        missing = _FILE_KEYS - set(entry)
        if missing:
            raise _fail(source, at, f"is missing {', '.join(sorted(missing))}")
        digest = entry["sha256"]
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise _fail(source, f"{at}.sha256", "must be 64 lowercase hex digits")
        parsed[name] = FixtureFile(
            name=name,
            sha256=digest,
            size_bytes=_positive_int(source, f"{at}.bytes", entry["bytes"]),
        )
    return parsed


def _parse_entry(source: str, port: str, index: int, value: Any) -> ManifestEntry:
    where = f"entries[{index}]"
    entry = _mapping(source, where, value)
    _check_keys(source, where, entry, _ENTRY_REQUIRED, _ENTRY_OPTIONAL)

    case = _text(source, f"{where}.case", entry["case"])
    tier = entry["tier"]
    if tier not in TIERS:
        raise _fail(source, f"{where}.tier", f"must be one of {', '.join(TIERS)}")

    capture = _mapping(source, f"{where}.capture", entry["capture"])
    _check_keys(source, f"{where}.capture", capture, _CAPTURE_REQUIRED)
    provenance = _text(source, f"{where}.capture.provenance", capture["provenance"])
    if not Path(provenance).is_absolute():
        raise _fail(source, f"{where}.capture.provenance", "must be an absolute path")

    tripwire_raw = _mapping(source, f"{where}.tripwire", entry["tripwire"])
    if not tripwire_raw:
        # A fixture with no recorded schema identity cannot fail loudly when the
        # featurizer or the tape layout moves under it; it just silently stops
        # meaning what it meant.
        raise _fail(source, f"{where}.tripwire", "must record at least one key")
    tripwire = {
        name: _text(source, f"{where}.tripwire[{name!r}]", item)
        for name, item in tripwire_raw.items()
    }

    gpu = _mapping(
        source, f"{where}.gpu_residual_angstrom", entry["gpu_residual_angstrom"]
    )
    _check_keys(
        source, f"{where}.gpu_residual_angstrom", gpu, _GPU_REQUIRED, _GPU_OPTIONAL
    )
    gpu_max = _number(
        source, f"{where}.gpu_residual_angstrom.max", gpu["max"], minimum=0.0
    )
    per_sample_raw = gpu.get("per_sample")
    per_sample: tuple[float, ...] | None = None
    if per_sample_raw is not None:
        if not isinstance(per_sample_raw, list) or not per_sample_raw:
            raise _fail(
                source,
                f"{where}.gpu_residual_angstrom.per_sample",
                "must be a non-empty list or null",
            )
        per_sample = tuple(
            _number(
                source,
                f"{where}.gpu_residual_angstrom.per_sample[{i}]",
                item,
                minimum=0.0,
            )
            for i, item in enumerate(per_sample_raw)
        )

    calibration = _mapping(source, f"{where}.cpu_calibration", entry["cpu_calibration"])
    _check_keys(
        source,
        f"{where}.cpu_calibration",
        calibration,
        _CALIBRATION_REQUIRED,
        _CALIBRATION_OPTIONAL,
    )
    cpu_residual = _number(
        source,
        f"{where}.cpu_calibration.residual_angstrom",
        calibration["residual_angstrom"],
        minimum=0.0,
    )
    cpu_wall = _number(
        source,
        f"{where}.cpu_calibration.wall_seconds",
        calibration["wall_seconds"],
        minimum=0.0,
    )
    cpu_commit = _text(
        source, f"{where}.cpu_calibration.source_commit", calibration["source_commit"]
    )

    tolerance = _mapping(source, f"{where}.tolerance", entry["tolerance"])
    _check_keys(source, f"{where}.tolerance", tolerance, _TOLERANCE_REQUIRED)
    # Closed tolerances only: CPU XLA picks different chunk/scan shapes than the
    # GPU run that produced the capture, so a zero tolerance is a promise the
    # arithmetic never made (memory: chunking-a-free-axis-is-not-bitwise).
    tolerance_value = _number(
        source, f"{where}.tolerance.value", tolerance["value"], minimum=0.0
    )
    if tolerance_value <= 0.0:
        raise _fail(source, f"{where}.tolerance.value", "must be > 0")
    if tolerance_value < cpu_residual:
        raise _fail(
            source,
            f"{where}.tolerance.value",
            f"({tolerance_value}) is below its own CPU calibration residual "
            f"({cpu_residual}); the test could never have passed",
        )

    node_ids_raw = entry["test_node_ids"]
    if not isinstance(node_ids_raw, list) or not node_ids_raw:
        raise _fail(source, f"{where}.test_node_ids", "must be a non-empty list")
    node_ids: list[str] = []
    for i, item in enumerate(node_ids_raw):
        at = f"{where}.test_node_ids[{i}]"
        node_id = _text(source, at, item)
        if not node_id.startswith(NODE_ID_PREFIX) or "::" not in node_id:
            raise _fail(source, at, f"must be a '{NODE_ID_PREFIX}...::test' node id")
        node_ids.append(node_id)

    excluded_raw = entry.get("excluded_samples", [])
    if not isinstance(excluded_raw, list):
        raise _fail(source, f"{where}.excluded_samples", "must be a list")
    excluded: list[int] = []
    for i, item in enumerate(excluded_raw):
        at = f"{where}.excluded_samples[{i}]"
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise _fail(source, at, "must be a sample index >= 0")
        excluded.append(item)

    return ManifestEntry(
        port=port,
        case=case,
        tier=tier,
        tokens=_positive_int(source, f"{where}.tokens", entry["tokens"]),
        capture_provenance=provenance,
        capture_commit=_text(source, f"{where}.capture.commit", capture["commit"]),
        tripwire=tripwire,
        files=_parse_files(source, f"{where}.files", entry["files"]),
        gpu_residual_max=gpu_max,
        gpu_residual_per_sample=per_sample,
        cpu_residual=cpu_residual,
        cpu_wall_seconds=cpu_wall,
        cpu_source_commit=cpu_commit,
        tolerance_metric=_text(
            source, f"{where}.tolerance.metric", tolerance["metric"]
        ),
        tolerance_value=tolerance_value,
        tolerance_set_from=_text(
            source, f"{where}.tolerance.set_from", tolerance["set_from"]
        ),
        test_node_ids=tuple(node_ids),
        excluded_samples=tuple(excluded),
        notes=str(entry.get("notes", "")),
    )


def parse_manifest(
    payload: Any, *, source: str, expected_port: str | None = None
) -> PortManifest:
    """Validate one port manifest already read from JSON."""
    document = _mapping(source, "manifest", payload)
    _check_keys(source, "manifest", document, tuple(_TOP_LEVEL_KEYS))
    version = document["schema_version"]
    if version != SCHEMA_VERSION:
        raise _fail(
            source,
            "schema_version",
            f"is {version!r}, this loader speaks {SCHEMA_VERSION}",
        )
    port = _text(source, "port", document["port"])
    if expected_port is not None and port != expected_port:
        raise _fail(source, "port", f"is {port!r} but the file is {expected_port}.json")
    entries_raw = document["entries"]
    if not isinstance(entries_raw, list) or not entries_raw:
        raise _fail(source, "entries", "must be a non-empty list")
    entries = tuple(
        _parse_entry(source, port, index, item)
        for index, item in enumerate(entries_raw)
    )
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        if (entry.case, entry.tier) in seen:
            raise _fail(
                source, "entries", f"repeats case {entry.case!r} tier {entry.tier}"
            )
        seen.add((entry.case, entry.tier))
    return PortManifest(port=port, source=source, entries=entries)


def manifest_paths(manifest_dir: Path | None = None) -> list[Path]:
    """Every port manifest, sorted. Non-JSON files (the README) are not one."""
    root = MANIFEST_DIR if manifest_dir is None else manifest_dir
    return sorted(root.glob("*.json"))


def load_manifest(port: str, manifest_dir: Path | None = None) -> PortManifest:
    root = MANIFEST_DIR if manifest_dir is None else manifest_dir
    path = root / f"{port}.json"
    if not path.is_file():
        raise ManifestError(f"no CPU-parity manifest for port {port!r} at {path}")
    return _load_path(path)


def _load_path(path: Path) -> PortManifest:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ManifestError(f"{path.name}: is not valid JSON ({error})") from error
    return parse_manifest(payload, source=path.name, expected_port=path.stem)


def load_all_manifests(manifest_dir: Path | None = None) -> list[PortManifest]:
    return [_load_path(path) for path in manifest_paths(manifest_dir)]


def all_entries(manifest_dir: Path | None = None) -> Iterator[ManifestEntry]:
    for manifest in load_all_manifests(manifest_dir):
        yield from manifest.entries


def entry_for(
    port: str, case: str, tier: str, manifest_dir: Path | None = None
) -> ManifestEntry:
    manifest = load_manifest(port, manifest_dir)
    for entry in manifest.entries:
        if entry.case == case and entry.tier == tier:
            return entry
    known = ", ".join(f"{e.case}/tier-{e.tier}" for e in manifest.entries)
    raise ManifestError(
        f"{manifest.source}: no entry for case {case!r} tier {tier!r} (has {known})"
    )
