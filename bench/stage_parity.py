"""Stage-level deterministic parity of one port against one stored native capture.

One JSON per model, six stages, each measured on CPU from the stored native
capture with every random draw replayed from its tape:

=====  ===================================================================
S1     input features: port featurizer (re-run in this checkout) vs native
S2     weights: converted parameters vs the native checkpoint
S3     trunk: single/pair from the stored inputs vs the native trunk
S4     diffusion: coordinates from the *native* trunk + the native noise tape
S5     confidence: heads from the *native* trunk + *native* coordinates
S6     final: end-to-end tape replay, CA and all-atom RMSD
=====  ===================================================================

The replays are not re-implemented here. Every stage that runs a model goes
through the CPU parity subset's own loaders and replay helpers
(``tests/parity/test_<port>.py``), resolving the digest-verified fixture copies
through ``tests.parity._fixtures.FixtureStore``; extra native arrays the
fixtures do not carry (confidence inputs, head outputs) are read from the
capture directory named by ``--capture``. Where an upstream tensor is injected
in place of the port's own (S4, S5), the seam is a module-global patch that is
counted, and the compiled-program pool is cleared on both sides of it so a
cached trace can neither serve nor leak the patch.

A stage the capture cannot support is written as ``not_captured`` with the
reason; nothing is synthesized to fill it. ``bench/stage_parity_results/
MISSING.md`` names the upstream hook a new capture would need.

Usage::

    JAX_PLATFORMS=cpu FOLDJAX_HOME=<store> PYTHONPATH=<tree>/src:<tree> \\
        python bench/stage_parity.py --model protenix \\
        --capture <bench>/protenix-master-native-20260909-9yET4f/protein_1ubq/native-A \\
        --out bench/stage_parity_results/protenix.json
    python bench/stage_parity.py --table bench/stage_parity_results

The metric functions at the top import NumPy only, so their unit tests
(``tests/test_bench_stage_parity.py``) need neither JAX nor weights.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import subprocess
import tempfile
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]

STAGES = (
    ("S1", "features"),
    ("S2", "weights"),
    ("S3", "trunk"),
    ("S4", "diffusion"),
    ("S5", "confidence"),
    ("S6", "final"),
)

# --------------------------------------------------------------------------
# Metrics (NumPy only)
# --------------------------------------------------------------------------


def _same_shape(port: np.ndarray, native: np.ndarray) -> None:
    if port.shape != native.shape:
        raise ValueError(f"shape mismatch: port {port.shape}, native {native.shape}")


def exact_match_fraction(port: Any, native: Any) -> float:
    """Fraction of elements that are exactly equal (integers, bools, strings)."""
    left, right = np.asarray(port), np.asarray(native)
    _same_shape(left, right)
    if left.size == 0:
        return 1.0
    return float(np.count_nonzero(left == right) / left.size)


def max_abs(port: Any, native: Any) -> float:
    """Largest absolute element difference, computed in float64."""
    left = np.asarray(port, dtype=np.float64)
    right = np.asarray(native, dtype=np.float64)
    _same_shape(left, right)
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left - right)))


def relative_rms(port: Any, native: Any) -> float:
    """``rms(port - native) / rms(native)``; 0 for two all-zero arrays."""
    left = np.asarray(port, dtype=np.float64)
    right = np.asarray(native, dtype=np.float64)
    _same_shape(left, right)
    if left.size == 0:
        return 0.0
    error = float(np.sqrt(np.mean((left - right) ** 2)))
    scale = float(np.sqrt(np.mean(right**2)))
    if scale == 0.0:
        return 0.0 if error == 0.0 else float("inf")
    return error / scale


def _is_exact_kind(array: np.ndarray) -> bool:
    return array.dtype.kind in "biuUSO"


def array_parity(port: Any, native: Any) -> dict[str, Any]:
    """Shape/dtype record plus the metric that suits the array's kind.

    Integer, boolean and string arrays are categorical, so they get an exact
    match fraction; anything else gets max-abs and relative RMS. A pair where
    one side is categorical and the other float (a mask stored as int64 by one
    implementation and float32 by the other) is compared numerically *and*
    exactly, since a cast is the only difference that should exist.
    """
    left, right = np.asarray(port), np.asarray(native)
    record: dict[str, Any] = {
        "port_shape": list(left.shape),
        "native_shape": list(right.shape),
        "port_dtype": str(left.dtype),
        "native_dtype": str(right.dtype),
    }
    if left.shape != right.shape:
        record["shape_equal"] = False
        return record
    record["shape_equal"] = True
    exact_left, exact_right = _is_exact_kind(left), _is_exact_kind(right)
    if exact_left and exact_right:
        record["exact_match_fraction"] = exact_match_fraction(left, right)
        return record
    if left.dtype.kind in "US" or right.dtype.kind in "US":
        record["exact_match_fraction"] = exact_match_fraction(
            left.astype(str), right.astype(str)
        )
        return record
    record["max_abs"] = max_abs(left, right)
    record["relative_rms"] = relative_rms(left, right)
    if exact_left or exact_right:
        record["exact_match_fraction"] = exact_match_fraction(
            left.astype(np.float64), right.astype(np.float64)
        )
    return record


def summarize_features(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Collapse per-array parity records into the S1 headline numbers."""
    exact = [r for r in records.values() if r.get("shape_equal") and "max_abs" not in r]
    floats = [r for r in records.values() if r.get("shape_equal") and "max_abs" in r]
    shape_mismatch = sorted(n for n, r in records.items() if not r.get("shape_equal"))
    int_elements_mismatched = [
        n for n, r in records.items() if r.get("exact_match_fraction", 1.0) < 1.0
    ]
    return {
        "arrays_compared": len(records),
        "categorical_arrays": len(exact),
        "float_arrays": len(floats),
        "shape_mismatches": shape_mismatch,
        "min_exact_match_fraction": min(
            (r["exact_match_fraction"] for r in exact), default=None
        ),
        "arrays_not_exact": sorted(int_elements_mismatched),
        "max_float_max_abs": max((r["max_abs"] for r in floats), default=None),
        "max_float_relative_rms": max(
            (r["relative_rms"] for r in floats), default=None
        ),
    }


def kabsch_rmsd(
    mobile: Any, reference: Any, mask: Any | None = None
) -> float:
    """RMSD after one proper-rotation least-squares fit of ``mobile`` onto ``reference``.

    The fit and the RMSD use the same atoms (``mask``), which is the
    conventional all-atom or CA RMSD. Reflections are excluded.
    """
    from bench.structures import rmsd

    p = np.asarray(mobile, dtype=np.float64)
    q = np.asarray(reference, dtype=np.float64)
    _same_shape(p, q)
    if p.ndim != 2 or p.shape[-1] != 3:
        raise ValueError(f"expected (atoms, 3) coordinates, got {p.shape}")
    if mask is not None:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != p.shape[:1]:
            raise ValueError(f"mask {selected.shape} does not match atoms {p.shape[:1]}")
        p, q = p[selected], q[selected]
    if len(p) < 3:
        raise ValueError(f"need at least three atoms for a fit, got {len(p)}")
    return rmsd(p, q)


def per_sample_rmsd(
    port: Any, native: Any, mask: Any | None = None
) -> list[float]:
    """:func:`kabsch_rmsd` for every sample of ``(samples, atoms, 3)`` arrays."""
    left = np.asarray(port, dtype=np.float64)
    right = np.asarray(native, dtype=np.float64)
    _same_shape(left, right)
    if left.ndim != 3:
        raise ValueError(f"expected (samples, atoms, 3), got {left.shape}")
    return [kabsch_rmsd(a, b, mask) for a, b in zip(left, right, strict=True)]


def max_unaligned_displacement(port: Any, native: Any, mask: Any | None = None) -> float:
    """Largest per-atom distance without any fit (the replay shares a frame)."""
    left = np.asarray(port, dtype=np.float64)
    right = np.asarray(native, dtype=np.float64)
    _same_shape(left, right)
    distance = np.linalg.norm(left - right, axis=-1)
    if mask is not None:
        distance = distance[..., np.asarray(mask, dtype=bool)]
    return float(distance.max()) if distance.size else 0.0


def multiset_delta(native_values: Sequence[Any], port_values: Sequence[Any]) -> dict:
    """Compare two bags of numbers independently of how they are arranged.

    A checkpoint conversion that only renames, transposes, stacks, splits or
    concatenates tensors leaves the sorted list of all values unchanged; any
    arithmetic on a value (a fold, a rescale, a lossy cast) moves it. This is a
    mapping-independent check that the converted parameters are the
    checkpoint's numbers and nothing else.
    """
    left = np.sort(
        np.concatenate([np.asarray(v, np.float64).ravel() for v in native_values])
        if native_values
        else np.zeros(0)
    )
    right = np.sort(
        np.concatenate([np.asarray(v, np.float64).ravel() for v in port_values])
        if port_values
        else np.zeros(0)
    )
    record: dict[str, Any] = {
        "native_elements": int(left.size),
        "port_elements": int(right.size),
    }
    if left.size != right.size:
        record["comparable"] = False
        return record
    record["comparable"] = True
    if left.size == 0:
        record["sorted_max_abs"] = 0.0
        return record
    finite = np.isfinite(left) & np.isfinite(right)
    record["nonfinite_mismatch"] = int(np.count_nonzero(~finite & (left != right)))
    record["sorted_max_abs"] = float(np.max(np.abs(left[finite] - right[finite])))
    return record


def ca_mask_from_names(atom_names: Any, elements: Any | None = None) -> np.ndarray:
    """Protein alpha carbons: atom name ``CA`` and, when given, element carbon."""
    names = np.char.strip(np.asarray(atom_names).astype(str))
    mask = names == "CA"
    if elements is not None:
        element = np.char.upper(np.char.strip(np.asarray(elements).astype(str)))
        mask &= element == "C"
    return mask


def coordinate_metrics(
    port: Any,
    native: Any,
    *,
    atom_mask: Any | None = None,
    ca_mask: Any | None = None,
) -> dict[str, Any]:
    """Per-sample all-atom and CA RMSD (one Kabsch fit each) plus max |dx|."""
    port = np.asarray(port, np.float64)
    native = np.asarray(native, np.float64)
    atoms = None if atom_mask is None else np.asarray(atom_mask, bool)
    record: dict[str, Any] = {
        "samples": int(port.shape[0]),
        "atoms": int(port.shape[1] if atoms is None else atoms.sum()),
        "all_atom_rmsd_angstrom": per_sample_rmsd(port, native, atoms),
        "max_unaligned_displacement_angstrom": max_unaligned_displacement(
            port, native, atoms
        ),
    }
    if ca_mask is not None:
        ca = np.asarray(ca_mask, bool)
        if atoms is not None:
            ca = ca & atoms
        record["ca_atoms"] = int(ca.sum())
        if ca.sum() >= 3:
            record["ca_rmsd_angstrom"] = per_sample_rmsd(port, native, ca)
    return record


# --------------------------------------------------------------------------
# Stage records
# --------------------------------------------------------------------------


def stage_record(
    status: str,
    *,
    condition: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    headline: Mapping[str, Any] | None = None,
    runtime_s: float | None = None,
    notes: str = "",
) -> dict[str, Any]:
    if status not in ("measured", "not_captured", "error"):
        raise ValueError(f"unknown stage status {status!r}")
    return {
        "status": status,
        "condition": dict(condition or {}),
        "headline": dict(headline or {}),
        "metrics": dict(metrics or {}),
        "runtime_s": runtime_s,
        "notes": notes,
    }


def not_captured(reason: str, *, needs: str = "") -> dict[str, Any]:
    return stage_record(
        "not_captured",
        notes=reason + (f" Needs: {needs}" if needs else ""),
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def run_stage(name: str, function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run one stage; a failure is recorded as an ``error`` stage, not raised.

    ``pytest.fail`` (which the parity helpers use for a missing fixture or a
    stale tripwire) raises a ``BaseException``, so that is caught too.
    """
    started = time.perf_counter()
    print(f"[stage_parity] {name} ...", flush=True)
    try:
        record = function()
    except KeyboardInterrupt:
        raise
    except BaseException as error:  # noqa: BLE001 -- recorded, not swallowed
        record = stage_record(
            "error",
            notes=f"{type(error).__name__}: {error}"[:4000],
            metrics={"traceback": traceback.format_exc()[-6000:]},
        )
    if record.get("runtime_s") is None and record["status"] != "not_captured":
        record["runtime_s"] = round(time.perf_counter() - started, 1)
    print(
        f"[stage_parity] {name}: {record['status']} "
        f"{json.dumps(_jsonable(record['headline']))} ({record['runtime_s']} s)",
        flush=True,
    )
    return record


# --------------------------------------------------------------------------
# Shared helpers for the model runners (JAX imported lazily)
# --------------------------------------------------------------------------

CPU_CONDITION = {"backend": "cpu", "matmul_precision": "highest"}


@contextlib.contextmanager
def highest_precision() -> Iterator[None]:
    """Ask every port for float32 matmuls at ``highest`` (CPU has no TF32)."""
    from foldjax.execution import matmul_precision_scope

    with matmul_precision_scope("highest"):
        yield


def require_cpu() -> None:
    import jax

    if jax.default_backend() != "cpu":
        raise RuntimeError(
            f"stage parity is a CPU measurement; this process is on "
            f"{jax.default_backend()!r} (set JAX_PLATFORMS=cpu)"
        )


def npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def fixture_case(port: str, case: str, tier: str):
    from tests.parity._fixtures import FixtureStore, fixtures_root

    return FixtureStore(root=fixtures_root()).case(port, case, tier=tier)


class TrackingState(dict):
    """A state dict that records which keys a mapper actually read."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.used: set[str] = set()

    def __getitem__(self, key: Any) -> Any:
        value = super().__getitem__(key)
        self.used.add(key)
        return value

    def get(self, key: Any, default: Any = None) -> Any:
        if super().__contains__(key):
            self.used.add(key)
        return super().get(key, default)


def tree_leaves_with_paths(tree: Any) -> list[tuple[str, Any]]:
    import jax

    flat, _ = jax.tree_util.tree_flatten_with_path(
        tree, is_leaf=lambda leaf: leaf is None
    )
    return [
        (jax.tree_util.keystr(path), leaf) for path, leaf in flat if leaf is not None
    ]


def _group_of(path: str, depth: int = 1) -> str:
    import re

    parts = [p for p in re.split(r"[.\[\]'\"]+", path) if p]
    return ".".join(parts[:depth]) if parts else "<root>"


def compare_parameter_trees(
    stored: Any, mapped: Any, *, group_depth: int = 1
) -> dict[str, Any]:
    """Leaf-by-leaf ``stored`` vs ``mapped``, max |delta| in the stored dtype.

    ``mapped`` is cast to each stored leaf's dtype before the difference, so a
    stored bf16 leaf is judged at bf16 and a float32 leaf at float32.
    """
    left = tree_leaves_with_paths(stored)
    right = dict(tree_leaves_with_paths(mapped))
    if [p for p, _ in left] != list(right):
        missing = sorted(set(right) - {p for p, _ in left})
        extra = sorted({p for p, _ in left} - set(right))
        raise ValueError(
            f"parameter trees differ: {len(missing)} only in mapped "
            f"({missing[:5]}), {len(extra)} only in stored ({extra[:5]})"
        )
    groups: dict[str, dict[str, Any]] = {}
    for path, leaf in left:
        stored_leaf = np.asarray(leaf)
        mapped_leaf = np.asarray(right[path])
        if stored_leaf.shape != mapped_leaf.shape:
            raise ValueError(f"{path}: stored {stored_leaf.shape} mapped {mapped_leaf.shape}")
        group = groups.setdefault(
            _group_of(path, group_depth),
            {"leaves": 0, "elements": 0, "max_abs": 0.0, "dtypes": set()},
        )
        group["leaves"] += 1
        group["elements"] += int(stored_leaf.size)
        group["dtypes"].add(str(stored_leaf.dtype))
        if stored_leaf.size:
            if stored_leaf.dtype.kind in "biu":
                delta = float(np.max(np.abs(
                    stored_leaf.astype(np.int64) - mapped_leaf.astype(np.int64)
                )))
            else:
                cast = mapped_leaf.astype(stored_leaf.dtype)
                delta = float(np.max(np.abs(
                    stored_leaf.astype(np.float64) - cast.astype(np.float64)
                )))
            group["max_abs"] = max(group["max_abs"], delta)
    for group in groups.values():
        group["dtypes"] = sorted(group["dtypes"])
    return {
        "groups": groups,
        "leaves": len(left),
        "elements": sum(g["elements"] for g in groups.values()),
        "max_abs": max((g["max_abs"] for g in groups.values()), default=0.0),
    }


def checkpoint_coverage(
    state: TrackingState, *, ignored: Callable[[str], bool] = lambda key: False
) -> dict[str, Any]:
    """Which checkpoint tensors the mapper read, by key and by element."""

    def size(value: Any) -> int:
        return int(np.asarray(value).size)

    keys = [k for k in dict.keys(state) if not ignored(k)]
    used = [k for k in keys if k in state.used]
    unused = sorted(set(keys) - set(used))
    return {
        "checkpoint_tensors": len(keys),
        "tensors_read": len(used),
        "checkpoint_elements": sum(size(dict.__getitem__(state, k)) for k in keys),
        "elements_read": sum(size(dict.__getitem__(state, k)) for k in used),
        "unread_tensors": unused[:50],
        "unread_tensor_count": len(unused),
    }


def sha256(path: Path) -> str:
    from tests.parity._fixtures import sha256_of

    return sha256_of(path)


def clear_jit_pools(*pools: Any) -> None:
    for pool in pools:
        pool.clear_cache()


@contextlib.contextmanager
def counted_patch(module: Any, name: str, replacement: Callable[..., Any]):
    """Replace ``module.name`` and count the calls; restore on exit."""
    original = getattr(module, name)
    calls = {"n": 0}

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return replacement(*args, **kwargs)

    setattr(module, name, wrapper)
    try:
        yield calls
    finally:
        setattr(module, name, original)


# --------------------------------------------------------------------------
# Protenix
# --------------------------------------------------------------------------


def run_protenix(capture: Path, stages: set[str]) -> dict[str, dict[str, Any]]:
    import jax
    import jax.numpy as jnp
    import pytest

    import tests.parity.test_protenix as parity
    from foldjax.models.protenix.models import model as protenix_model
    from foldjax.models.protenix.models import predict as prediction

    case_dir = capture.parent
    results: dict[str, dict[str, Any]] = {}
    pools = (
        protenix_model._compiled_protenix_infer,
        protenix_model._compiled_protenix_infer_deterministic,
    )
    base = {
        **CPU_CONDITION,
        "input": "port-featurized foldjax-input.npz (port-A of the same capture; "
        "its input gate certified it equal to native-input.npz)",
        "msa": "native per-cycle MSA rows injected (msa-tape.npz)",
        "trunk_dtype": "bfloat16 (native AMP match: upstream autocast bf16)",
        "triangle_attention": "cueq_jit (port default, as in the parity subset)",
        "glu_backend": "xla",
    }

    def loaded() -> tuple[dict, Any, dict]:
        case_a = fixture_case("protenix", "protein_1ubq", "A")
        stored = parity._arrays(case_a.path("foldjax-input.npz"))
        features = parity._nest(stored)
        cycles = parity._msa_cycles(features, parity._arrays(case_a.path("msa-tape.npz")))
        return stored, features, cycles

    # -- S1 ---------------------------------------------------------------
    def s1() -> dict[str, Any]:
        from bench.protenix_closure_report import (
            COMMON_FIELDS,
            check_inputs,
            flat_features,
        )
        from bench.protenix_foldjax_capture import featurize

        argv = json.loads((case_dir / "port-A" / "foldjax-argv.json").read_text())
        input_json = Path(argv[argv.index("--input-json") + 1])
        with tempfile.TemporaryDirectory() as scratch:
            features = featurize(input_json, Path(scratch))
        port = flat_features(features)
        native: dict[str, np.ndarray] = {}
        for name in ("native-input.npz", "native-derived.npz", "native-identity.npz"):
            native.update(npz(capture / name))
        records = {
            name: array_parity(port[name], native[name]) for name in sorted(COMMON_FIELDS)
        }
        gate = check_inputs(capture, features)
        stored = npz(case_dir / "port-A" / "foldjax-input.npz")
        from bench.entity_parity import compare_feature_dicts

        drift = compare_feature_dicts(port, stored)
        summary = summarize_features(records)
        return stage_record(
            "measured",
            condition={
                **CPU_CONDITION,
                "port_featurizer": "foldjax.models.protenix.data.featurize_json."
                "featurize_protein_json, re-run in this checkout "
                "(bench.protenix_foldjax_capture.featurize)",
                "input_document": str(input_json),
                "native": "native-input.npz + native-derived.npz + native-identity.npz",
                "name_mapping": "bench.protenix_closure_report.COMMON_FIELDS "
                "(the 41 fields both sides spell the same)",
            },
            headline={
                "metric": "min exact-match fraction (categorical) / max |d| (float)",
                "value": f"{summary['min_exact_match_fraction']} / "
                f"{summary['max_float_max_abs']}",
            },
            metrics={
                "summary": summary,
                "arrays": records,
                "semantic_input_gate": {
                    "passed": gate.get("passed"),
                    "failures": gate.get("failures"),
                    "derived_mappings": gate.get("derived_mappings"),
                },
                "rerun_vs_capture_time_port_features_equal": drift["equal"],
                "rerun_vs_capture_time_port_features_value_mismatches": drift[
                    "value_mismatches"
                ],
            },
            notes="Derived fields (relp, local-atom padding, polymer flags) are "
            "checked by the capture's own semantic gate (check_inputs), whose "
            "verdict is recorded, rather than by name.",
        )

    # -- S2 ---------------------------------------------------------------
    def s2() -> dict[str, Any]:
        from foldjax import torch_archive
        from foldjax.models._weights_io import _load_native_numpy_tree
        from foldjax.models.protenix.bridge.torch_mapping import (
            map_protenix_inference_state_dict,
        )
        from foldjax.paths import weights_dir

        native_path = capture / "checkpoint" / "protenix_base_default_v1.0.0.pt"
        stored_path = weights_dir("protenix") / parity.CHECKPOINT
        obj = torch_archive.load(native_path)
        state = obj["model"] if "model" in obj else obj
        state = {
            (k[len("module.") :] if str(k).startswith("module.") else str(k)): v
            for k, v in state.items()
        }
        tracked = TrackingState(state)
        mapped = map_protenix_inference_state_dict(tracked)
        mapped = jax.tree.map(np.asarray, mapped)
        stored = _load_native_numpy_tree(stored_path, prestack=False)
        tree = compare_parameter_trees(stored, mapped)
        coverage = checkpoint_coverage(tracked)
        bag = multiset_delta(
            [dict.__getitem__(tracked, k) for k in sorted(tracked.used)],
            [leaf for _, leaf in tree_leaves_with_paths(stored)],
        )
        return stage_record(
            "measured",
            condition={
                "backend": "cpu (host NumPy; no model run)",
                "native_checkpoint": str(native_path),
                "native_sha256": sha256(native_path),
                "converted": str(stored_path),
                "mapper": "foldjax.models.protenix.bridge.torch_mapping."
                "map_protenix_inference_state_dict",
                "reader": "foldjax.torch_archive.load (no torch)",
            },
            headline={
                "metric": "max |stored - map(native)| over groups (storage dtype)",
                "value": tree["max_abs"],
            },
            metrics={
                "stored_vs_mapped": tree,
                "coverage": coverage,
                "value_multiset_read_vs_stored": bag,
            },
        )

    # -- S3 ---------------------------------------------------------------
    def s3() -> dict[str, Any]:
        _, features, cycles = loaded()
        native = npz(fixture_case("protenix", "protein_1ubq", "A").path("trunk.npz"))
        started = time.perf_counter()
        with highest_precision():
            port = parity._run_trunk(features, cycles)
        runtime = time.perf_counter() - started
        arrays = {
            name: {
                "relative_rms": relative_rms(port[name], native[stored]),
                "max_abs": max_abs(port[name], native[stored]),
                "native_max_abs_value": float(np.abs(native[stored]).max()),
            }
            for name, stored in parity.TRUNK_ARRAYS.items()
        }
        return stage_record(
            "measured",
            condition={**base, "recycles": parity.RECYCLES},
            headline={
                "metric": "relative RMS single / pair",
                "value": f"{arrays['single']['relative_rms']:.3e} / "
                f"{arrays['pair']['relative_rms']:.3e}",
            },
            metrics={"arrays": arrays},
            runtime_s=round(runtime, 1),
        )

    def entity(native: np.ndarray, port: np.ndarray, stored: dict) -> dict:
        per_chain = parity._entity_rmsd(
            native, port, stored["output_atom_chain_id"].tolist()
        )
        return {str(k): v for k, v in per_chain.items()}

    def coordinate_record(port: np.ndarray, native: np.ndarray, stored: dict) -> dict:
        record = coordinate_metrics(
            port,
            native,
            ca_mask=ca_mask_from_names(
                stored["output_atom_name"], stored["output_atom_element"]
            ),
        )
        record["entity_rmsd_angstrom_panel_metric"] = entity(native, port, stored)
        return record

    def sampler_tape() -> dict[str, np.ndarray]:
        return parity._arrays(fixture_case("protenix", "protein_1ubq", "B").path(
            "sampler-tape.npz"
        ))

    def native_coordinates() -> np.ndarray:
        return parity._array(
            fixture_case("protenix", "protein_1ubq", "B").path("prediction.npz"),
            "coordinate",
        ).astype(np.float64)

    # -- S6 ---------------------------------------------------------------
    def s6() -> dict[str, Any]:
        stored, features, cycles = loaded()
        tape = sampler_tape()
        clear_jit_pools(*pools)
        started = time.perf_counter()
        with highest_precision(), pytest.MonkeyPatch.context() as mp:
            port = parity._replay_coordinates(features, cycles, tape, mp)
        runtime = time.perf_counter() - started
        record = coordinate_record(port, native_coordinates(), stored)
        return stage_record(
            "measured",
            condition={
                **base,
                "tape": "native sampler-tape.npz (init/step noise, rotations, "
                "translations, noise schedule) + MSA cycle rows",
                "schedule": "native, pinned",
                "samples_x_steps": "5 x 200",
                "diffusion_attention": "xla_jit (as calibrated)",
                "confidence": "off",
            },
            headline={
                "metric": "per-sample all-atom / CA RMSD (A), worst sample",
                "value": f"{max(record['all_atom_rmsd_angstrom']):.4f} / "
                f"{max(record['ca_rmsd_angstrom']):.4f}",
            },
            metrics=record,
            runtime_s=round(runtime, 1),
        )

    def native_trunk_patch(s_inputs, s_trunk, z_trunk):
        values = tuple(
            jnp.asarray(np.asarray(v, np.float32)) for v in (s_inputs, s_trunk, z_trunk)
        )

        def injected(*args: Any, **kwargs: Any):
            return values

        return injected

    # -- S4 ---------------------------------------------------------------
    def s4() -> dict[str, Any]:
        stored, features, cycles = loaded()
        tape = sampler_tape()
        trunk = npz(fixture_case("protenix", "protein_1ubq", "A").path("trunk.npz"))
        clear_jit_pools(*pools)
        started = time.perf_counter()
        try:
            with (
                highest_precision(),
                pytest.MonkeyPatch.context() as mp,
                counted_patch(
                    protenix_model,
                    "pairformer_output_from_s_inputs",
                    native_trunk_patch(trunk["s_inputs"], trunk["s"], trunk["z"]),
                ) as calls,
            ):
                port = parity._replay_coordinates(features, cycles, tape, mp)
        finally:
            clear_jit_pools(*pools)
        runtime = time.perf_counter() - started
        if calls["n"] < 1:
            raise RuntimeError("the native-trunk injection never fired")
        record = coordinate_record(port, native_coordinates(), stored)
        record["injection_calls"] = calls["n"]
        return stage_record(
            "measured",
            condition={
                **base,
                "injected_native": "trunk.npz s_inputs/s/z (float32) replace "
                "pairformer_output_from_s_inputs; sampler-tape.npz; noise schedule",
                "trunk_dtype": "n/a (trunk not run; native trunk injected)",
                "samples_x_steps": "5 x 200",
                "diffusion_attention": "xla_jit",
                "diffusion_dtype": "float32 (native diffusion_skip_amp=true)",
            },
            headline={
                "metric": "per-sample all-atom RMSD (A), worst sample",
                "value": f"{max(record['all_atom_rmsd_angstrom']):.4f}",
            },
            metrics=record,
            runtime_s=round(runtime, 1),
        )

    # -- S5 ---------------------------------------------------------------
    def s5() -> dict[str, Any]:
        _, features, cycles = loaded()
        conf_in = npz(capture / "confidence-input.npz")
        native_pred = npz(capture / "prediction.npz")
        native_disto = npz(capture / "distogram.npz")
        coords = jnp.asarray(conf_in["x_pred_coords"].astype(np.float32))

        def native_coords(*args: Any, **kwargs: Any):
            return coords

        clear_jit_pools(*pools)
        started = time.perf_counter()
        try:
            with (
                highest_precision(),
                counted_patch(
                    protenix_model,
                    "pairformer_output_from_s_inputs",
                    native_trunk_patch(
                        conf_in["s_inputs"], conf_in["s_trunk"], conf_in["z_trunk"]
                    ),
                ) as trunk_calls,
                counted_patch(
                    protenix_model, "sample_diffusion_with_module", native_coords
                ) as sampler_calls,
            ):
                output = prediction.protenix_predict_static(
                    parity._params(),
                    dict(features),
                    None,
                    num_samples=parity.SAMPLES,
                    num_sampling_steps=parity.SAMPLING_STEPS,
                    recycling_steps=parity.RECYCLES,
                    trunk_dtype=jnp.bfloat16,
                    cycle_msa_index_tape=jax.tree.map(jnp.asarray, cycles),
                    use_diffusion_efficient_fusion=True,
                    run_confidence=True,
                    return_confidence_logits=True,
                    return_confidence_details=True,
                    return_trunk=False,
                    glu_backend="xla",
                    diffusion_attention_backend="xla_jit",
                )
                output = jax.device_get(jax.block_until_ready(output))
        finally:
            clear_jit_pools(*pools)
        runtime = time.perf_counter() - started
        if trunk_calls["n"] < 1 or sampler_calls["n"] < 1:
            raise RuntimeError(
                f"injection did not fire: trunk {trunk_calls}, sampler {sampler_calls}"
            )
        output = {k: np.asarray(v) for k, v in output.items()}
        samples = parity.SAMPLES
        native_atom_plddt = np.stack(
            [native_pred[f"full_data.{i}.atom_plddt"] for i in range(samples)]
        )
        native_pae = np.stack(
            [native_pred[f"full_data.{i}.token_pair_pae"] for i in range(samples)]
        )
        native_pde = np.stack(
            [native_pred[f"full_data.{i}.token_pair_pde"] for i in range(samples)]
        )
        native_ptm = np.array(
            [native_pred[f"summary_confidence.{i}.ptm"] for i in range(samples)]
        )
        native_plddt = np.array(
            [native_pred[f"summary_confidence.{i}.plddt"] for i in range(samples)]
        )
        port_keys = sorted(output)
        metrics: dict[str, Any] = {"port_output_keys": port_keys}
        scale = 100.0 if float(np.abs(native_atom_plddt).max()) > 1.5 else 1.0
        pairs = {
            "plddt_logits": (output.get("plddt"), native_pred["plddt"], 1.0),
            "pae_logits": (output.get("pae"), native_pred["pae"], 1.0),
            "pde_logits": (output.get("pde"), native_pred["pde"], 1.0),
            "atom_plddt": (output.get("atom_plddt"), native_atom_plddt, scale),
            "token_pair_pae_angstrom": (
                output.get("token_pair_pae"), native_pae, 1.0,
            ),
            "token_pair_pde_angstrom": (
                output.get("token_pair_pde"), native_pde, 1.0,
            ),
            "ptm": (output.get("summary_ptm"), native_ptm, 1.0),
            "summary_plddt": (output.get("summary_plddt"), native_plddt, 1.0),
            "distogram_logits_from_native_z": (
                output.get("distogram_logits"), native_disto["logits"], 1.0,
            ),
        }
        for name, (port, native, factor) in pairs.items():
            if port is None:
                metrics[name] = {"missing_from_port_output": True}
                continue
            port = np.asarray(port, np.float64).reshape(np.shape(native)) * factor
            metrics[name] = {
                "max_abs": max_abs(port, native),
                "relative_rms": relative_rms(port, native),
                "native_range": [float(np.min(native)), float(np.max(native))],
            }
        metrics["atom_plddt_scale_factor_applied_to_port"] = scale
        return stage_record(
            "measured",
            condition={
                **base,
                "injected_native": "confidence-input.npz s_inputs/s_trunk/z_trunk "
                "replace the trunk; x_pred_coords replace the sampler",
                "trunk_dtype": "n/a (native trunk injected)",
                "confidence_dtype": "float32 (native confidence_skip_amp=true)",
                "confidence_triangle_attention": "follows trunk (cueq_jit)",
            },
            headline={
                "metric": "max |d| pLDDT (0-100) / PAE (A) / pTM",
                "value": "{:.3g} / {:.3g} / {:.3g}".format(
                    metrics["atom_plddt"].get("max_abs", float("nan")),
                    metrics["token_pair_pae_angstrom"].get("max_abs", float("nan")),
                    metrics["ptm"].get("max_abs", float("nan")),
                ),
            },
            metrics=metrics,
            runtime_s=round(runtime, 1),
        )

    runners = {"S1": s1, "S2": s2, "S3": s3, "S6": s6, "S4": s4, "S5": s5}
    for name, function in runners.items():
        if name in stages:
            results[name] = run_stage(f"protenix {name}", function)
    return results


# --------------------------------------------------------------------------
# CLI and table
# --------------------------------------------------------------------------

MODELS: dict[str, Callable[[Path, set[str]], dict[str, dict[str, Any]]]] = {
    "protenix": run_protenix,
}


def environment() -> dict[str, Any]:
    record: dict[str, Any] = {
        "host": platform.node(),
        "python": platform.python_version(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "foldjax_home": os.environ.get("FOLDJAX_HOME"),
    }
    try:
        record["cpu_affinity"] = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        record["cpu_affinity"] = None
    try:
        record["git_commit"] = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        record["git_commit"] = None
    try:
        import jax
        import jaxlib

        record["jax"] = jax.__version__
        record["jaxlib"] = jaxlib.__version__
    except ImportError:  # pragma: no cover
        pass
    return record


def run_model(model: str, capture: Path, stages: set[str]) -> dict[str, Any]:
    require_cpu()
    started = time.perf_counter()
    results = MODELS[model](capture, stages)
    ordered = {
        f"{key}_{label}": results.get(key, stage_record("not_captured", notes="not run"))
        for key, label in STAGES
        if key in stages
    }
    return {
        "model": model,
        "capture": str(capture),
        "environment": environment(),
        "wall_s": round(time.perf_counter() - started, 1),
        "stages": ordered,
    }


def _short_condition(condition: Mapping[str, Any]) -> str:
    keys = (
        "backend",
        "trunk_dtype",
        "diffusion_dtype",
        "confidence_dtype",
        "matmul_precision",
        "injected_native",
    )
    parts = [f"{k}={condition[k]}" for k in keys if k in condition]
    return "; ".join(parts)


def render_table(results_dir: Path) -> str:
    rows = [
        "| model | stage | status | metric | value | condition | runtime (s) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for path in sorted(results_dir.glob("*.json")):
        report = json.loads(path.read_text())
        for stage, record in report["stages"].items():
            headline = record.get("headline", {})
            if record["status"] == "measured":
                metric, value = headline.get("metric", ""), headline.get("value", "")
                condition = _short_condition(record.get("condition", {}))
            else:
                metric, value = "", ""
                condition = record.get("notes", "").split(". ")[0]
            if isinstance(value, float):
                value = f"{value:.4g}"
            runtime = record.get("runtime_s")
            rows.append(
                f"| {report['model']} | {stage} | {record['status']} | {metric} | "
                f"{value} | {condition} | {'' if runtime is None else runtime} |"
            )
    return "\n".join(rows) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(MODELS))
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--stages",
        default=",".join(key for key, _ in STAGES),
        help="comma-separated subset, e.g. S1,S3",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="keep stages already in --out that this run does not measure",
    )
    parser.add_argument("--table", type=Path, help="render TABLE.md rows for a directory")
    args = parser.parse_args(argv)
    if args.table is not None:
        print(render_table(args.table), end="")
        return 0
    if not (args.model and args.capture and args.out):
        parser.error("--model, --capture and --out are required")
    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    unknown = stages - {key for key, _ in STAGES}
    if unknown:
        parser.error(f"unknown stages {sorted(unknown)}")
    report = run_model(args.model, args.capture.resolve(), stages)
    if args.merge and args.out.is_file():
        previous = json.loads(args.out.read_text())
        merged = dict(previous.get("stages", {}))
        merged.update(report["stages"])
        report["stages"] = {
            f"{key}_{label}": merged[f"{key}_{label}"]
            for key, label in STAGES
            if f"{key}_{label}" in merged
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=False) + "\n")
    print(f"[stage_parity] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
