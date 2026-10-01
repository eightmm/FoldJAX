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
        --capture <protenix master capture>/protein_1ubq/native-A \\
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
    mobile: Any,
    reference: Any,
    mask: Any | None = None,
    measure_mask: Any | None = None,
) -> float:
    """RMSD after one proper-rotation fit of ``mobile`` onto ``reference``.

    The fit uses the atoms in ``mask`` (all atoms when omitted); the RMSD is
    taken over ``measure_mask`` (the fitted atoms when omitted) *without
    refitting*, which is the project's convention: one Kabsch transform on all
    valid system atoms per sample, then RMSD on whichever subset is reported.
    Reflections are excluded.
    """
    from bench.structures import kabsch

    p = np.asarray(mobile, dtype=np.float64)
    q = np.asarray(reference, dtype=np.float64)
    _same_shape(p, q)
    if p.ndim != 2 or p.shape[-1] != 3:
        raise ValueError(f"expected (atoms, 3) coordinates, got {p.shape}")

    def selection(value: Any | None) -> np.ndarray:
        if value is None:
            return np.ones(p.shape[0], dtype=bool)
        selected = np.asarray(value, dtype=bool)
        if selected.shape != p.shape[:1]:
            raise ValueError(
                f"mask {selected.shape} does not match atoms {p.shape[:1]}"
            )
        return selected

    fit = selection(mask)
    measure = fit if measure_mask is None else selection(measure_mask)
    if fit.sum() < 3:
        raise ValueError(f"need at least three atoms for a fit, got {int(fit.sum())}")
    if not measure.any():
        raise ValueError("the measured atom set is empty")
    rotation, p_centre, q_centre = kabsch(p[fit], q[fit])
    aligned = (p - p_centre) @ rotation.T
    target = q - q_centre
    squared = np.sum((aligned[measure] - target[measure]) ** 2, axis=-1)
    return float(np.sqrt(np.mean(squared)))


def per_sample_rmsd(
    port: Any,
    native: Any,
    mask: Any | None = None,
    measure_mask: Any | None = None,
) -> list[float]:
    """:func:`kabsch_rmsd` for every sample of ``(samples, atoms, 3)`` arrays."""
    left = np.asarray(port, dtype=np.float64)
    right = np.asarray(native, dtype=np.float64)
    _same_shape(left, right)
    if left.ndim != 3:
        raise ValueError(f"expected (samples, atoms, 3), got {left.shape}")
    return [
        kabsch_rmsd(a, b, mask, measure_mask) for a, b in zip(left, right, strict=True)
    ]


def max_unaligned_displacement(
    port: Any, native: Any, mask: Any | None = None
) -> float:
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

    Only floating-point arrays enter the bags; integer and boolean leaves
    (layer flags, AMP markers, index buffers) are counted separately, since
    a converter may legitimately add them.
    """

    left, native_other = _float_values(native_values)
    right, port_other = _float_values(port_values)
    left.sort()
    right.sort()
    record: dict[str, Any] = {
        "native_elements": int(left.size),
        "port_elements": int(right.size),
        "native_non_float_elements": native_other,
        "port_non_float_elements": port_other,
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


def _float_values(values: Sequence[Any]) -> tuple[np.ndarray, int]:
    kept, other = [], 0
    for value in values:
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.floating) or array.dtype.name in (
            "bfloat16",
            "float8_e4m3fn",
            "float8_e5m2",
        ):
            kept.append(array.astype(np.float64).ravel())
        else:
            other += int(array.size)
    return (np.concatenate(kept) if kept else np.zeros(0)), other


def multiset_containment(
    checkpoint_values: Sequence[Any], port_values: Sequence[Any]
) -> dict[str, Any]:
    """Is every converted float a checkpoint float, independent of the mapping?

    Unlike :func:`multiset_delta` this does not need to know which checkpoint
    tensors the mapper read: the checkpoint may hold tensors inference never
    uses (training heads, aliases), so the converted bag is only expected to
    be *contained* in the checkpoint bag. Reported:

    * ``port_values_absent_from_checkpoint`` -- converted elements whose exact
      value occurs nowhere in the checkpoint (0 for a pure rearrangement);
    * ``max_distance_to_nearest_checkpoint_value`` -- for those, how far the
      nearest checkpoint value is (a lossy cast or a fold shows up here);
    * ``multiplicity_excess`` -- converted elements beyond the number of times
      their value occurs in the checkpoint (a tensor used twice shows up here,
      which is legitimate for shared weights).
    """
    checkpoint, checkpoint_other = _float_values(checkpoint_values)
    port, port_other = _float_values(port_values)
    record: dict[str, Any] = {
        "checkpoint_float_elements": int(checkpoint.size),
        "port_float_elements": int(port.size),
        "checkpoint_non_float_elements": checkpoint_other,
        "port_non_float_elements": port_other,
    }
    if port.size == 0:
        record.update(
            port_values_absent_from_checkpoint=0,
            max_distance_to_nearest_checkpoint_value=0.0,
            multiplicity_excess=0,
        )
        return record
    if checkpoint.size == 0:
        record.update(
            port_values_absent_from_checkpoint=int(port.size),
            max_distance_to_nearest_checkpoint_value=float("inf"),
            multiplicity_excess=int(port.size),
        )
        return record
    reference, reference_counts = np.unique(checkpoint, return_counts=True)
    del checkpoint
    values, counts = np.unique(port, return_counts=True)
    del port
    position = np.searchsorted(reference, values)
    clipped = np.clip(position, 0, reference.size - 1)
    present = reference[clipped] == values
    # NaN never compares equal; count it as present when both sides hold NaN.
    nan = np.isnan(values)
    if nan.any() and np.isnan(reference).any():
        present |= nan
    absent = ~present
    record["port_values_absent_from_checkpoint"] = int(counts[absent].sum())
    if absent.any():
        missing = values[absent]
        index = np.searchsorted(reference, missing)
        below = reference[np.clip(index - 1, 0, reference.size - 1)]
        above = reference[np.clip(index, 0, reference.size - 1)]
        distance = np.minimum(np.abs(missing - below), np.abs(missing - above))
        record["max_distance_to_nearest_checkpoint_value"] = float(np.nanmax(distance))
    else:
        record["max_distance_to_nearest_checkpoint_value"] = 0.0
    available = np.where(present, reference_counts[clipped], 0)
    record["multiplicity_excess"] = int(
        np.maximum(counts[present] - available[present], 0).sum()
    )
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
    """Per-sample all-atom and CA RMSD under one all-atom fit, plus max |dx|.

    Both RMSDs share the sample's single Kabsch fit on all valid atoms; the CA
    value is that fit measured on the alpha carbons, not a CA-only refit.
    """
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
        if ca.any():
            record["ca_rmsd_angstrom"] = per_sample_rmsd(port, native, atoms, ca)
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
            raise ValueError(
                f"{path}: stored {stored_leaf.shape} mapped {mapped_leaf.shape}"
            )
        group = groups.setdefault(
            _group_of(path, group_depth),
            {"leaves": 0, "elements": 0, "max_abs": 0.0, "dtypes": set()},
        )
        group["leaves"] += 1
        group["elements"] += int(stored_leaf.size)
        group["dtypes"].add(str(stored_leaf.dtype))
        if stored_leaf.size:
            if stored_leaf.dtype.kind in "biu":
                delta = float(
                    np.max(
                        np.abs(
                            stored_leaf.astype(np.int64) - mapped_leaf.astype(np.int64)
                        )
                    )
                )
            else:
                cast = mapped_leaf.astype(stored_leaf.dtype)
                delta = float(
                    np.max(
                        np.abs(stored_leaf.astype(np.float64) - cast.astype(np.float64))
                    )
                )
            group["max_abs"] = max(group["max_abs"], delta)
    for group in groups.values():
        group["dtypes"] = sorted(group["dtypes"])
    return {
        "groups": groups,
        "leaves": len(left),
        "elements": sum(g["elements"] for g in groups.values()),
        "max_abs": max((g["max_abs"] for g in groups.values()), default=0.0),
    }


def weight_value_checks(
    state: TrackingState,
    port_leaves: Sequence[Any],
    *,
    ignored: Callable[[str], bool] = lambda key: False,
) -> dict[str, Any]:
    """Mapping-independent checks of converted parameters against a checkpoint.

    ``coverage_direct_reads`` counts the tensors the mapper read through the
    state dict. It is a lower bound when a mapper slices the state with
    ``items()`` into a plain sub-dict (Boltz-2's confidence Pairformer does),
    because those reads bypass the tracker; ``value_containment`` does not
    depend on it.
    """
    coverage = checkpoint_coverage(state, ignored=ignored)
    keys = [k for k in dict.keys(state) if not ignored(k)]
    checks: dict[str, Any] = {
        "coverage_direct_reads": coverage,
        "value_containment_port_in_checkpoint": multiset_containment(
            [dict.__getitem__(state, k) for k in keys], port_leaves
        ),
    }
    if coverage["tensors_read"] == coverage["checkpoint_tensors"]:
        checks["value_multiset_checkpoint_vs_port"] = multiset_delta(
            [dict.__getitem__(state, k) for k in keys], port_leaves
        )
    return checks


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


#: Native re-captures that add the trunk, sampler and confidence boundaries
#: the 2026-09-09 master captures lack (``--capture-stages`` in
#: ``bench/esmfold2_tape.py`` and ``bench/opendde_closure_capture.py``). Same
#: command, input, seed and environment as the stored capture; S3-S5 read
#: their native side from here, S1, S2 and S6 stay on ``--capture``.
STAGE_CAPTURE_ROOT = Path(
    "/home/jaemin/non-project/optimizing/foldjax-bench/stage-captures-20261001"
)
STAGE_CAPTURES = {
    "esmfold2": STAGE_CAPTURE_ROOT / "esmfold2" / "protein_1ubq" / "native-A",
    "opendde": STAGE_CAPTURE_ROOT / "opendde" / "protein_1ubq" / "native-A",
}


def native_at_port_width(native: Any, port_dtype: Any) -> tuple[np.ndarray, str]:
    """A native float tensor at the width the port carries at that seam.

    Narrowed only when that is lossless (upstream's autocast leaves
    bf16-representable values in float32 storage); otherwise it stays float32
    and the record says so, rather than rounding the injected value.
    """
    native = np.asarray(native, np.float32)
    dtype = np.dtype(port_dtype)
    if dtype == np.float32:
        return native, "float32"
    narrowed = native.astype(dtype)
    if np.array_equal(narrowed.astype(np.float32), native):
        return narrowed, f"{dtype} (exact)"
    return native, f"float32 (not exactly representable in {dtype}; kept wide)"


def bitwise_equal_files(left: Path, right: Path) -> dict[str, Any]:
    """Per-array bitwise equality of two npz files (names, dtypes, bytes)."""
    a, b = npz(left), npz(right)
    arrays = {
        name: bool(
            name in b
            and a[name].dtype == b[name].dtype
            and a[name].shape == b[name].shape
            and a[name].tobytes() == b[name].tobytes()
        )
        for name in sorted(set(a) | set(b))
    }
    return {"all_equal": all(arrays.values()), "arrays": arrays}


def rerun_agreement(
    stored: Path,
    new: Path,
    *,
    bitwise: Sequence[str],
    coordinates: tuple[str, str],
    values: Mapping[str, Sequence[str]],
    atom_mask: Any | None = None,
    ca_mask: Any | None = None,
) -> dict[str, Any]:
    """How a native re-capture compares with the stored capture it repeats.

    The tape and inputs must be bitwise equal (the same draws); coordinates and
    head outputs are GPU reruns and are reported against the stored
    ``native-A-vs-native-B.json`` rerun floor of the same campaign.
    """
    record: dict[str, Any] = {
        "stored": str(stored),
        "new": str(new),
        "bitwise": {
            name: bitwise_equal_files(stored / name, new / name) for name in bitwise
        },
    }
    file, key = coordinates
    record["coordinates"] = coordinate_metrics(
        npz(new / file)[key],
        npz(stored / file)[key],
        atom_mask=atom_mask,
        ca_mask=ca_mask,
    )
    record["values_max_abs"] = {}
    for file, keys in values.items():
        left, right = npz(new / file), npz(stored / file)
        for name in keys:
            record["values_max_abs"][f"{file}:{name}"] = max_abs(
                left[name], right[name]
            )
    floor = stored.parent / "native-A-vs-native-B.json"
    if floor.is_file():
        data = json.loads(floor.read_text())
        coords = data.get("coordinates", {})
        record["stored_rerun_floor_native_A_vs_B"] = {
            "file": str(floor),
            "entity_rmsd": coords.get("entity_rmsd"),
            "global_rmsd": coords.get("global_rmsd"),
        }
    return record


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
        cycles = parity._msa_cycles(
            features, parity._arrays(case_a.path("msa-tape.npz"))
        )
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
            name: array_parity(port[name], native[name])
            for name in sorted(COMMON_FIELDS)
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
        checks = weight_value_checks(
            tracked,
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
                **checks,
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
        return parity._arrays(
            fixture_case("protenix", "protein_1ubq", "B").path("sampler-tape.npz")
        )

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
                output.get("token_pair_pae"),
                native_pae,
                1.0,
            ),
            "token_pair_pde_angstrom": (
                output.get("token_pair_pde"),
                native_pde,
                1.0,
            ),
            "ptm": (output.get("summary_ptm"), native_ptm, 1.0),
            "summary_plddt": (output.get("summary_plddt"), native_plddt, 1.0),
            "distogram_logits_from_native_z": (
                output.get("distogram_logits"),
                native_disto["logits"],
                1.0,
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
                "metric": "max |d| atom pLDDT (0-1) / PAE (A) / pTM",
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
# Boltz-2
# --------------------------------------------------------------------------


def atom_names_from_onehot(chars: Any) -> np.ndarray:
    """Decode a ``(..., 4, 64)`` one-hot atom-name array (``chr(i + 32)``)."""
    index = np.asarray(chars).argmax(-1)
    flat = index.reshape(-1, index.shape[-1])
    names = ["".join(chr(int(c) + 32) for c in row).strip() for row in flat]
    return np.asarray(names).reshape(index.shape[:-1])


def rigid_group_max_abs(port: Any, native: Any, groups: Any, mask: Any) -> float:
    """Max |d| after a separate Kabsch fit per group (e.g. per reference conformer).

    Reference conformers are randomly rotated and translated by the
    featurizer; when that draw is not replayed, this is the part of the
    coordinate difference that is not a rigid motion.
    """
    p = np.asarray(port, np.float64)
    q = np.asarray(native, np.float64)
    labels = np.asarray(groups)
    valid = np.asarray(mask, bool)
    worst = 0.0
    for label in np.unique(labels[valid]):
        members = valid & (labels == label)
        a, b = p[members], q[members]
        if len(a) >= 3:
            from bench.structures import superpose

            a, b = superpose(a, b)
        else:
            a, b = a - a.mean(0), b - b.mean(0)
        worst = max(worst, float(np.max(np.abs(a - b))) if a.size else 0.0)
    return worst


def run_boltz2(capture: Path, stages: set[str]) -> dict[str, dict[str, Any]]:
    import jax
    import jax.numpy as jnp

    import tests.parity.test_boltz2 as parity

    # Read at call time by the trunk; the parity subset pins it the same way.
    os.environ[parity.TRIANGLE_MULTIPLICATION_ENV] = "xla"

    from bench.boltz_amp_report import compare_coordinates
    from foldjax.models.boltz2.bridge.native import load_params
    from foldjax.models.boltz2.models.trunk_blocks.trunk import (
        _cast_trunk_params,
        boltz2_sample_forward,
    )
    from tests.models.boltz2.scripts.parity_matched_tape import (
        captured_sampler_trunk,
        load_features,
        load_tape,
    )

    results: dict[str, dict[str, Any]] = {}
    cache: dict[str, Any] = {}

    def case(tier: str):
        return fixture_case("boltz2", parity.CASE, tier)

    def params() -> Any:
        if "params" not in cache:
            cache["params"] = load_params(parity._weights())
        return cache["params"]

    def meta_effective() -> tuple[dict, dict]:
        return parity._capture_metadata(case("B"))

    raw_features = npz(capture / "features.npz")
    atom_valid = raw_features["atom_pad_mask"][0].astype(bool)
    atom_names = atom_names_from_onehot(raw_features["ref_atom_name_chars"][0])
    atom_element = raw_features["ref_element"][0].argmax(-1)
    token_of_atom = raw_features["atom_to_token"][0].argmax(-1)
    protein_atom = raw_features["mol_type"][0][token_of_atom] == 0
    ca_mask = (atom_names == "CA") & (atom_element == 6) & protein_atom & atom_valid

    base = {
        **CPU_CONDITION,
        "input": "native features.npz (upstream featurizer output; core-only)",
        "msa": "all 8,192 native rows (capture ran subsample_msa=false)",
        "trunk_dtype": "bfloat16 parameters, float32 pair residual "
        "(native bf16-mixed AMP match; parity-subset pin)",
        "kernels": "attention/triangle/GLU = xla; triangle multiplication = xla",
        "recycles": 3,
    }

    def coordinate_record(port: np.ndarray, native: np.ndarray) -> dict[str, Any]:
        record = coordinate_metrics(port, native, atom_mask=atom_valid, ca_mask=ca_mask)
        report = compare_coordinates(port, native, raw_features)
        record["entity_rmsd_angstrom_panel_metric"] = report["entity_rmsd"]
        return record

    def sample(trunk: Mapping[str, Any]) -> np.ndarray:
        """The parity subset's tier-B sampler call, with ``trunk`` supplied."""
        meta, _ = meta_effective()
        tape = load_tape(case("B").path("tape.npz"), meta)
        parity._assert_schedule(meta, tape["sigmas"])
        features = load_features(case("B").path("features.npz"))
        with highest_precision(), jax.default_matmul_precision("highest"):
            output = boltz2_sample_forward(
                params(),
                features,
                jax.random.PRNGKey(int(meta["seed"])),
                trunk=trunk,
                recycling_steps=int(meta["num_recycles"]),
                num_sampling_steps=int(meta["num_steps"]),
                multiplicity=int(meta["num_samples"]),
                step_scale=float(meta["step_scale"]),
                gamma_0=float(meta["gamma_0"]),
                gamma_min=float(meta["gamma_min"]),
                noise_scale=float(meta["noise_scale"]),
                sigma_data=float(meta["sigma_data"]),
                init_noise=jnp.asarray(tape["init_noise"]),
                step_noises=jnp.asarray(tape["step_noises"]),
                aug_transforms=(
                    jnp.asarray(tape["rotations"]),
                    jnp.asarray(tape["translations"]),
                ),
                use_scan=True,
                compute_dtype=jnp.bfloat16,
                **parity.SHARED_OPTIONS,
                **parity.DENOISER_OPTIONS,
            )
            return np.asarray(jax.device_get(output["sample_atom_coords"]), np.float64)

    def native_coordinates() -> np.ndarray:
        return npz(case("B").path("coordinate.npz"))["coordinate"].astype(np.float64)

    def port_trunk() -> dict[str, Any]:
        if "trunk" not in cache:
            meta, effective = meta_effective()
            features = load_features(case("A").path("features.npz"))
            started = time.perf_counter()
            with highest_precision(), jax.default_matmul_precision("highest"):
                trunk = parity._port_trunk(params(), features, meta, effective)
                jax.block_until_ready(trunk["s"])
            cache["trunk_seconds"] = time.perf_counter() - started
            cache["trunk"] = trunk
        return cache["trunk"]

    # -- S1 ---------------------------------------------------------------
    def s1() -> dict[str, Any]:
        from foldjax.models.boltz2.data.featurize import featurize_yaml
        from foldjax.paths import weights_dir

        provenance = json.loads((capture / "provenance.json").read_text())
        yaml_path = Path(provenance["requested_settings"]["input"])
        seed = int(provenance["requested_settings"]["seed"])
        with tempfile.TemporaryDirectory() as scratch:
            port, _, _ = featurize_yaml(
                yaml_path, Path(scratch), weights_dir("boltz2") / "mols", seed=seed
            )
        native = raw_features
        common = sorted(set(port) & set(native))
        records = {name: array_parity(port[name], native[name]) for name in common}
        summary = summarize_features(records)
        rigid = None
        if records.get("ref_pos", {}).get("shape_equal"):
            rigid = rigid_group_max_abs(
                port["ref_pos"][0],
                native["ref_pos"][0],
                native["ref_space_uid"][0],
                native["atom_pad_mask"][0] > 0,
            )
        return stage_record(
            "measured",
            condition={
                **CPU_CONDITION,
                "port_featurizer": "foldjax.models.boltz2.data.featurize."
                "featurize_yaml "
                "(torch-free), re-run in this checkout",
                "input_document": str(yaml_path),
                "seed": seed,
                "native": "features.npz (upstream boltz b1ebfc4 featurizer)",
                "name_mapping": "identical names; every key both sides emit",
                "random_draws": "NOT replayed: upstream's reference-conformer "
                "augmentation draws (preprocessing-tape.npz) have no port "
                "injection hook, so ref_pos differs by a rigid motion per conformer",
            },
            headline={
                "metric": "min exact-match fraction (categorical) / max |d| (float)",
                "value": f"{summary['min_exact_match_fraction']} / "
                f"{summary['max_float_max_abs']:.3g} (ref_pos after per-conformer "
                f"Kabsch: {rigid:.2g})",
            },
            metrics={
                "summary": summary,
                "only_in_port": sorted(set(port) - set(native)),
                "only_in_native": sorted(set(native) - set(port)),
                "ref_pos_max_abs_after_per_conformer_kabsch": rigid,
                "arrays": records,
            },
        )

    # -- S2 ---------------------------------------------------------------
    def s2() -> dict[str, Any]:
        from foldjax.models.boltz2.bridge import export_weights
        from foldjax.models.boltz2.bridge.checkpoint import load_checkpoint_state_dict
        from foldjax.paths import foldjax_home

        native_path = foldjax_home() / "downloads" / "boltz2" / "boltz2_conf.ckpt"
        provenance = json.loads((capture / "provenance.json").read_text())
        state = TrackingState(load_checkpoint_state_dict(native_path))
        captured: dict[str, Any] = {}

        def capture_params(tree: Any, path: Any, dtype: Any = None) -> dict:
            captured["params"] = tree
            captured["dtype"] = dtype
            return {"weights_path": str(path)}

        original = (
            export_weights.load_checkpoint_state_dict,
            export_weights.save_params,
        )
        export_weights.load_checkpoint_state_dict = lambda _path: state
        export_weights.save_params = capture_params
        try:
            with tempfile.TemporaryDirectory() as scratch:
                export_weights.export_confidence(native_path, Path(scratch))
        finally:
            export_weights.load_checkpoint_state_dict, export_weights.save_params = (
                original
            )
        mapped = jax.tree.map(np.asarray, captured["params"])
        stored = jax.tree.map(
            np.asarray, load_params(parity._weights(), prestack=False)
        )
        tree = compare_parameter_trees(stored, mapped)
        checks = weight_value_checks(
            state,
            [leaf for _, leaf in tree_leaves_with_paths(stored)],
        )
        return stage_record(
            "measured",
            condition={
                "backend": "cpu (host NumPy; no model run)",
                "native_checkpoint": str(native_path),
                "native_sha256": sha256(native_path),
                "capture_checkpoint_sha256": provenance["checkpoint_sha256"],
                "converted": str(parity._weights()) + ".safetensors",
                "mapper": "foldjax.models.boltz2.bridge.export_weights."
                "export_confidence (writer intercepted, nothing written)",
                "reader": "foldjax.torch_archive.load (no torch)",
            },
            headline={
                "metric": "max |stored - map(native)| over groups (storage dtype)",
                "value": tree["max_abs"],
            },
            metrics={
                "stored_vs_mapped": tree,
                **checks,
            },
        )

    # -- S3 ---------------------------------------------------------------
    def s3() -> dict[str, Any]:
        trunk = port_trunk()
        native = npz(case("A").path("trunk.npz"))
        arrays = {
            name: {
                "relative_rms": relative_rms(trunk[name], native[name]),
                "max_abs": max_abs(trunk[name], native[name]),
                "native_max_abs_value": float(np.abs(native[name]).max()),
            }
            for name in parity.TRUNK_ARRAYS
        }
        return stage_record(
            "measured",
            condition=base,
            headline={
                "metric": "relative RMS single (s) / pair (z)",
                "value": f"{arrays['s']['relative_rms']:.3e} / "
                f"{arrays['z']['relative_rms']:.3e}",
            },
            metrics={"arrays": arrays},
            runtime_s=round(cache["trunk_seconds"], 1),
        )

    # -- S4 ---------------------------------------------------------------
    def s4() -> dict[str, Any]:
        trunk = captured_sampler_trunk(npz(case("B").path("trunk.npz")))
        started = time.perf_counter()
        port = sample(trunk)
        runtime = time.perf_counter() - started
        record = coordinate_record(port, native_coordinates())
        return stage_record(
            "measured",
            condition={
                **base,
                "trunk_dtype": "n/a (native trunk injected)",
                "injected_native": "trunk.npz s/z/s_inputs/relative_position_encoding "
                "(trunk= argument); tape.npz init/step noise + rotations/translations",
                "diffusion_dtype": "float32 score model (released island)",
                "samples_x_steps": "5 x 200",
            },
            headline={
                "metric": "per-sample all-atom / CA RMSD (A), worst sample",
                "value": f"{max(record['all_atom_rmsd_angstrom']):.4f} / "
                f"{max(record['ca_rmsd_angstrom']):.4f}",
            },
            metrics=record,
            runtime_s=round(runtime, 1),
        )

    # -- S5 ---------------------------------------------------------------
    def s5() -> dict[str, Any]:
        from foldjax.models.boltz2.models.heads.confidence import (
            confidence_module_forward,
        )
        from foldjax.models.boltz2.models.heads.distogram import distogram_forward

        native_out = npz(capture / "forward-output.npz")
        trunk = npz(case("B").path("trunk.npz"))
        features = load_features(case("B").path("features.npz"))
        coords = native_out["sample_atom_coords"].astype(np.float32)
        coordinate_file = native_coordinates()
        compute = jnp.bfloat16
        head_params = _cast_trunk_params(params()["confidence"], compute)
        disto_params = {"distogram": _cast_trunk_params(params()["distogram"], compute)}
        s_inputs = jnp.asarray(trunk["s_inputs"], jnp.float32)
        s = jnp.asarray(trunk["s"], jnp.float32)
        z = jnp.asarray(trunk["z"], jnp.float32)
        native_disto = jnp.asarray(native_out["pdistogram"][:, :, :, 0], jnp.float32)
        options = {
            key: parity.SHARED_OPTIONS[key]
            for key in (
                "chunk_size",
                "matmul_precision",
                "attention_backend",
                "triangle_backend",
                "glu_backend",
            )
        }

        def one(x_pred: jnp.ndarray) -> dict[str, jnp.ndarray]:
            return confidence_module_forward(
                head_params,
                s_inputs=s_inputs,
                s=s,
                z=z,
                x_pred=x_pred,
                feats=features,
                pred_distogram_logits=native_disto,
                multiplicity=1,
                use_scan=True,
                return_pair_chains_iptm=True,
                **options,
            )

        started = time.perf_counter()
        with highest_precision(), jax.default_matmul_precision("highest"):
            port_disto = np.asarray(
                jax.device_get(distogram_forward(disto_params, z)), np.float64
            )
            program = jax.jit(one)
            per_sample = [
                jax.device_get(program(jnp.asarray(coords[i : i + 1])))
                for i in range(coords.shape[0])
            ]
        runtime = time.perf_counter() - started
        keys = sorted(set(per_sample[0]) & set(native_out))
        metrics: dict[str, Any] = {
            "native_coordinates": "forward-output.npz sample_atom_coords",
            "forward_output_vs_coordinate_npz_max_abs": max_abs(
                coords, coordinate_file
            ),
            "distogram_logits_from_native_z": {
                "max_abs": max_abs(port_disto, native_out["pdistogram"]),
                "relative_rms": relative_rms(port_disto, native_out["pdistogram"]),
            },
        }
        for key in keys:
            port_value = np.concatenate(
                [
                    np.asarray(out[key], np.float64).reshape((1, -1))
                    for out in per_sample
                ]
            ).reshape(native_out[key].shape)
            native_value = native_out[key].astype(np.float64)
            metrics[key] = {
                "max_abs": max_abs(port_value, native_value),
                "relative_rms": relative_rms(port_value, native_value),
                "native_range": [float(native_value.min()), float(native_value.max())],
            }
        return stage_record(
            "measured",
            condition={
                **base,
                "trunk_dtype": "n/a (native trunk injected)",
                "confidence_dtype": "bfloat16 parameters (native bf16-mixed AMP; "
                "confidence runs under autocast upstream)",
                "injected_native": "trunk.npz s_inputs/s/z; forward-output.npz "
                "sample_atom_coords and pdistogram[...,0] (pred_distogram_logits)",
                "confidence_schedule": "one sample at a time "
                "(run_confidence_sequentially=true upstream)",
            },
            headline={
                "metric": "max |d| pLDDT (0-1) / PAE (A) / pTM",
                "value": "{:.3g} / {:.3g} / {:.3g}".format(
                    metrics.get("plddt", {}).get("max_abs", float("nan")),
                    metrics.get("pae", {}).get("max_abs", float("nan")),
                    metrics.get("ptm", {}).get("max_abs", float("nan")),
                ),
            },
            metrics=metrics,
            runtime_s=round(runtime, 1),
        )

    # -- S6 ---------------------------------------------------------------
    def s6() -> dict[str, Any]:
        trunk = port_trunk()
        started = time.perf_counter()
        port = sample(trunk)
        runtime = time.perf_counter() - started + cache["trunk_seconds"]
        record = coordinate_record(port, native_coordinates())
        return stage_record(
            "measured",
            condition={
                **base,
                "injected_native": "tape.npz only (init/step noise, rotations, "
                "translations); trunk is the port's own (S3)",
                "samples_x_steps": "5 x 200",
            },
            headline={
                "metric": "per-sample all-atom / CA RMSD (A), worst sample",
                "value": f"{max(record['all_atom_rmsd_angstrom']):.4f} / "
                f"{max(record['ca_rmsd_angstrom']):.4f}",
            },
            metrics=record,
            runtime_s=round(runtime, 1),
            notes="Runtime includes the trunk (S3).",
        )

    runners = {"S1": s1, "S2": s2, "S3": s3, "S4": s4, "S5": s5, "S6": s6}
    for name, function in runners.items():
        if name in stages:
            if name == "S6" and "trunk" not in cache:
                port_trunk()
            results[name] = run_stage(f"boltz2 {name}", function)
    return results


# --------------------------------------------------------------------------
# OpenFold3 (OpenBind checkpoint)
# --------------------------------------------------------------------------


def run_openfold3(capture: Path, stages: set[str]) -> dict[str, dict[str, Any]]:
    import jax
    import jax.numpy as jnp

    import tests.parity.test_openfold3 as parity
    from bench.openbind_core_replay import native_trunk_arrays
    from foldjax.models.openfold3 import inference
    from foldjax.models.openfold3.bridge.chemistry import representative_atom_table

    results: dict[str, dict[str, Any]] = {}
    scratch = tempfile.TemporaryDirectory()
    case_a = fixture_case("openfold3", parity.CASE, "A")
    case_b = fixture_case("openfold3", parity.CASE, "B")
    capture_a = parity.capture_shaped_dir(case_a.files, Path(scratch.name) / "a")
    capture_b = parity.capture_shaped_dir(case_b.files, Path(scratch.name) / "b")
    checkpoint = parity.resolve_checkpoint()
    raw_input = npz(capture / "input.npz")
    atom_valid = raw_input["atom_mask"][0].astype(bool)
    ca_mask = ca_mask_from_names(
        raw_input["atom_array.0.annotation.atom_name"],
        raw_input["atom_array.0.annotation.element"],
    )
    base = {
        **CPU_CONDITION,
        "input": "native input.npz (upstream featurizer output; core-only)",
        "msa": "native per-cycle MSA row draws (tape randint/randperm) replayed",
        "trunk_dtype": "float32 (native 32-true)",
        "triangle_kernel": "xla (native ran cuEq; no CPU cuEq)",
        "recycles": 4,
    }

    def clear() -> None:
        inference._compiled_predict.clear_cache()

    def run(capture_dir: Path, *, stop_after_trunk: bool, **overrides: Any):
        """``tests.parity.test_openfold3.run_replay`` with config overrides."""
        tape, effective, features = parity.native_batch(capture_dir)
        config = parity.replay_config(
            features, tape, effective, stop_after_trunk=stop_after_trunk
        )
        if overrides:
            config = config._replace(**overrides)
        program = inference.compile_predict(
            config, representative_atom_table(), triangle_kernel="xla"
        )
        started = time.perf_counter()
        with highest_precision():
            result = jax.device_get(
                program(
                    jax.random.key(101),
                    features,
                    parity.inference_params(str(checkpoint)),
                    noise_tape=tape.noise,
                    augmentation_tape=tape.augmentation(),
                )
            )
        return result, time.perf_counter() - started

    def coordinate_record(port: np.ndarray) -> dict[str, Any]:
        native = npz(capture_b / "coordinate.npz")["coordinate"].astype(np.float64)
        record = coordinate_metrics(port, native, atom_mask=atom_valid, ca_mask=ca_mask)
        record["entity_rmsd_angstrom_panel_metric"] = parity.entity_rmsd(
            capture_b, port
        )["entity_rmsd"]
        return record

    def trunk_patch():
        native, _ = native_trunk_arrays(capture_a, raw_input["token_mask"].shape[-1])
        values = tuple(jnp.asarray(x) for x in native)
        return lambda *args, **kwargs: values

    # -- S1 ---------------------------------------------------------------
    def s1() -> dict[str, Any]:
        from foldjax.models.openfold3.data.featurize import featurize_query

        # The capture's predictions/inference_query_set.json is upstream's
        # rewritten copy (its MSA paths point at upstream's pre-parsed cache),
        # so the port featurizes the document the native run was given; the
        # sequences and protein MSA path are checked against the rewritten copy.
        query = Path(
            "/home/jaemin/non-project/optimizing/foldjax-bench/"
            "upstream-default-multimodal-n5-20260904/work/protein_rna_1urn/foldjax/"
            "openfold3/inputs/openfold3_input.json"
        )
        rewritten = json.loads(
            (capture / "predictions" / "inference_query_set.json").read_text()
        )
        original = json.loads(query.read_text())
        for name, spec in original["queries"].items():
            mine = [c["sequence"] for c in spec["chains"]]
            theirs = [c["sequence"] for c in rewritten["queries"][name]["chains"]]
            if mine != theirs:
                raise RuntimeError(f"{query} is not the document the capture ran")
        port = featurize_query(query, seed=101)
        native = {k: v for k, v in raw_input.items() if not k.startswith("atom_array.")}
        common = sorted(set(port) & set(native))
        records = {name: array_parity(port[name], native[name]) for name in common}
        summary = summarize_features(records)
        rigid = None
        if records.get("ref_pos", {}).get("shape_equal"):
            rigid = rigid_group_max_abs(
                port["ref_pos"][0],
                native["ref_pos"][0],
                native["ref_space_uid"][0],
                native["atom_mask"][0] > 0,
            )
        return stage_record(
            "measured",
            condition={
                **CPU_CONDITION,
                "port_featurizer": "foldjax.models.openfold3.data.featurize."
                "featurize_query (vendored NumPy featurizer), re-run in this checkout",
                "input_document": str(query),
                "seed": 101,
                "native": "input.npz (upstream OpenFold3 v0.5.0 batch)",
                "name_mapping": "identical names; every key both sides emit",
                "random_draws": "NOT replayed: reference conformers come from "
                "RDKit ETKDGv3 embedding (upstream core/data/primitives/structure/"
                "conformer.py:_compute_conformer), a preprocessing draw this capture "
                "did not tape; ref_pos is therefore expected to differ beyond a "
                "rigid motion",
            },
            headline={
                "metric": "min exact-match fraction (categorical) / max |d| (float)",
                "value": f"{summary['min_exact_match_fraction']} / "
                f"{summary['max_float_max_abs']:.3g} (ref_pos after per-conformer "
                f"Kabsch: {rigid:.2g})",
            },
            metrics={
                "summary": summary,
                "only_in_port": sorted(set(port) - set(native)),
                "only_in_native": sorted(set(native) - set(port)),
                "ref_pos_max_abs_after_per_conformer_kabsch": rigid,
                "arrays": records,
            },
        )

    # -- S2 ---------------------------------------------------------------
    def s2() -> dict[str, Any]:
        from foldjax.models.openfold3.bridge.checkpoint import load_checkpoint
        from foldjax.models.openfold3.bridge.torch_mapping import (
            map_inference_params,
            prune_sample_diffusion_aliases,
            resolve_model_prefix,
        )

        state = load_checkpoint(checkpoint)
        prefix = resolve_model_prefix(state, None)
        prune_sample_diffusion_aliases(state, prefix=prefix)
        tracked = TrackingState(state)
        mapped = jax.tree.map(np.asarray, map_inference_params(tracked, prefix))
        mapped_leaves = [leaf for _, leaf in tree_leaves_with_paths(mapped)]
        checks = weight_value_checks(
            tracked,
            mapped_leaves,
            ignored=lambda key: key.endswith("version_tensor"),
        )
        containment = checks["value_containment_port_in_checkpoint"]
        return stage_record(
            "measured",
            condition={
                "backend": "cpu (host NumPy; no model run)",
                "native_checkpoint": str(checkpoint),
                "native_sha256": sha256(checkpoint),
                "converted": "none stored: the runtime maps the .pt in-process "
                "(.foldjax-conversion.json names the .pt itself), so stored == mapped",
                "mapper": "foldjax.models.openfold3.bridge.torch_mapping."
                "map_inference_params (after prune_sample_diffusion_aliases)",
                "reader": "foldjax.torch_archive via bridge.checkpoint.load_checkpoint",
            },
            headline={
                "metric": "mapped floats absent from checkpoint / max distance "
                "(no stored file)",
                "value": f"{containment['port_values_absent_from_checkpoint']} / "
                f"{containment['max_distance_to_nearest_checkpoint_value']}",
            },
            metrics={
                "mapped_leaves": len(mapped_leaves),
                **checks,
            },
            notes="No converted artifact exists for OpenFold3; the stage reports "
            "coverage of the checkpoint and that the mapped parameters are a "
            "rearrangement of the checkpoint values.",
        )

    # -- S3 ---------------------------------------------------------------
    def s3() -> dict[str, Any]:
        clear()
        prediction, seconds = run(capture_a, stop_after_trunk=True)
        native, _ = native_trunk_arrays(capture_a, raw_input["token_mask"].shape[-1])
        port = (prediction.single_inputs, prediction.single, prediction.pair)
        arrays = {}
        for name, left, right in zip(
            ("single_inputs", "single", "pair"), port, native, strict=True
        ):
            left = np.asarray(left)
            arrays[name] = {
                "relative_rms": relative_rms(left.reshape(right.shape), right),
                "max_abs": max_abs(left.reshape(right.shape), right),
                "native_max_abs_value": float(np.abs(right).max()),
            }
        arrays_scaled = parity.trunk_residuals(port, native)
        return stage_record(
            "measured",
            condition=base,
            headline={
                "metric": "relative RMS single / pair",
                "value": f"{arrays['single']['relative_rms']:.3e} / "
                f"{arrays['pair']['relative_rms']:.3e}",
            },
            metrics={
                "arrays": arrays,
                "parity_subset_metric_max_abs_over_max_native": arrays_scaled,
            },
            runtime_s=round(seconds, 1),
        )

    # -- S6 ---------------------------------------------------------------
    def s6() -> dict[str, Any]:
        clear()
        prediction, seconds = run(capture_b, stop_after_trunk=False)
        record = coordinate_record(np.asarray(prediction.coordinates, np.float64))
        return stage_record(
            "measured",
            condition={
                **base,
                "injected_native": "tape.npz (MSA draws, initial/churn noise, "
                "augmentation quaternions/translations)",
                "samples_x_steps": "5 x 200",
            },
            headline={
                "metric": "per-sample all-atom / CA RMSD (A), worst sample",
                "value": f"{max(record['all_atom_rmsd_angstrom']):.4f} / "
                f"{max(record['ca_rmsd_angstrom']):.4f}",
            },
            metrics=record,
            runtime_s=round(seconds, 1),
        )

    # -- S4 ---------------------------------------------------------------
    def s4() -> dict[str, Any]:
        clear()
        try:
            with counted_patch(inference, "trunk", trunk_patch()) as calls:
                prediction, seconds = run(capture_b, stop_after_trunk=False)
        finally:
            clear()
        if calls["n"] < 1:
            raise RuntimeError("the native-trunk injection never fired")
        record = coordinate_record(np.asarray(prediction.coordinates, np.float64))
        record["injection_calls"] = calls["n"]
        return stage_record(
            "measured",
            condition={
                **base,
                "trunk_dtype": "n/a (native trunk injected)",
                "injected_native": "trunk-00.npz s_inputs/s/z replace "
                "inference.trunk; tape.npz noise + augmentation",
                "diffusion_dtype": "float32",
                "samples_x_steps": "5 x 200",
            },
            headline={
                "metric": "per-sample all-atom / CA RMSD (A), worst sample",
                "value": f"{max(record['all_atom_rmsd_angstrom']):.4f} / "
                f"{max(record['ca_rmsd_angstrom']):.4f}",
            },
            metrics=record,
            runtime_s=round(seconds, 1),
        )

    # -- S5 ---------------------------------------------------------------
    def s5() -> dict[str, Any]:
        raw = npz(capture / "raw-output.npz")
        native_coords = jnp.asarray(raw["atom_positions_predicted"][0])
        clear()
        try:
            with (
                counted_patch(inference, "trunk", trunk_patch()) as trunk_calls,
                counted_patch(
                    inference, "sample_diffusion", lambda *a, **k: native_coords
                ) as sampler_calls,
            ):
                prediction, seconds = run(
                    capture_b,
                    stop_after_trunk=False,
                    return_plddt_logits=True,
                    returned_pair_logits=(
                        "pae_logits",
                        "pde_logits",
                        "distogram_logits",
                    ),
                )
        finally:
            clear()
        if trunk_calls["n"] < 1 or sampler_calls["n"] < 1:
            raise RuntimeError(
                f"injection did not fire: trunk {trunk_calls}, sampler {sampler_calls}"
            )
        metrics: dict[str, Any] = {}
        pairs = {
            "plddt_logits": (prediction.plddt_logits, raw["plddt_logits"][0]),
            "pae_logits": (prediction.pae_logits, raw["pae_logits"][0]),
            "pde_logits": (prediction.pde_logits, raw["pde_logits"][0]),
            "distogram_logits_from_native_z": (
                prediction.distogram_logits,
                raw["distogram_logits"][0, 0],
            ),
        }
        for name, (port, native) in pairs.items():
            if port is None:
                metrics[name] = {"missing_from_port_output": True}
                continue
            port = np.asarray(port, np.float64).reshape(native.shape)
            metrics[name] = {
                "max_abs": max_abs(port, native),
                "relative_rms": relative_rms(port, native),
            }
        # Upstream's own written scores, one JSON per sample (sample_1..5 is
        # the original sample index; rounded by the writer).
        written = capture / "predictions" / "protein_rna_1urn" / "seed_101"
        native_plddt, native_pae, native_ptm, native_iptm = [], [], [], []
        for index in range(1, 6):
            stem = written / f"protein_rna_1urn_seed_101_sample_{index}"
            full = json.loads(Path(f"{stem}_confidences.json").read_text())
            summary = json.loads(
                Path(f"{stem}_confidences_aggregated.json").read_text()
            )
            native_plddt.append(full["plddt"])
            native_pae.append(full["pae"])
            native_ptm.append(summary["ptm"])
            native_iptm.append(summary["iptm"])
        port_plddt = np.asarray(prediction.plddt, np.float64).reshape(5, -1)
        native_plddt = np.asarray(native_plddt, np.float64)
        scale = 100.0 if native_plddt.max() > 1.5 and port_plddt.max() <= 1.5 else 1.0
        port_plddt = port_plddt[:, : native_plddt.shape[1]] * scale
        metrics["plddt_written"] = {
            "max_abs": max_abs(port_plddt, native_plddt),
            "scale_factor_applied_to_port": scale,
        }
        native_pae = np.asarray(native_pae, np.float64)
        if prediction.pae_logits is not None:
            from foldjax.models.openfold3.models.confidence import (
                probs_to_expected_error,
            )

            # The bins the port's own pTM path is configured with.
            tape, effective, features = parity.native_batch(capture_b)
            config = parity.replay_config(
                features, tape, effective, stop_after_trunk=False
            )
            port_pae = np.asarray(
                probs_to_expected_error(
                    jax.nn.softmax(jnp.asarray(prediction.pae_logits), axis=-1),
                    bin_min=0.0,
                    bin_max=config.pae_bin_max,
                    no_bins=config.pae_bins,
                )
            )
            metrics["pae_written_angstrom"] = {
                "max_abs": max_abs(port_pae.reshape(native_pae.shape), native_pae)
            }
        metrics["ptm_written"] = {
            "max_abs": max_abs(np.asarray(prediction.ptm).reshape(5), native_ptm)
        }
        metrics["iptm_written"] = {
            "max_abs": max_abs(np.asarray(prediction.iptm).reshape(5), native_iptm)
        }
        metrics["injection_calls"] = {
            "trunk": trunk_calls["n"],
            "sampler": sampler_calls["n"],
        }
        return stage_record(
            "measured",
            condition={
                **base,
                "trunk_dtype": "n/a (native trunk injected)",
                "confidence_dtype": "float32",
                "injected_native": "trunk-00.npz replaces inference.trunk; "
                "raw-output.npz atom_positions_predicted replaces sample_diffusion",
            },
            headline={
                "metric": "max |d| pLDDT (0-100) / PAE (A) / pTM, vs written JSON",
                "value": "{:.3g} / {:.3g} / {:.3g}".format(
                    metrics["plddt_written"]["max_abs"],
                    metrics.get("pae_written_angstrom", {}).get(
                        "max_abs", float("nan")
                    ),
                    metrics["ptm_written"]["max_abs"],
                ),
            },
            metrics=metrics,
            runtime_s=round(seconds, 1),
            notes="Written JSON scores are rounded by upstream's writer; the "
            "logit rows are the unrounded comparison.",
        )

    runners = {"S1": s1, "S2": s2, "S3": s3, "S6": s6, "S4": s4, "S5": s5}
    try:
        for name, function in runners.items():
            if name in stages:
                results[name] = run_stage(f"openfold3 {name}", function)
    finally:
        scratch.cleanup()
    return results


# --------------------------------------------------------------------------
# ESMFold2
# --------------------------------------------------------------------------


def confidence_comparison(
    port: Mapping[str, Any], native: Mapping[str, Any], names: Mapping[str, str]
) -> dict[str, Any]:
    """max-abs / relative RMS for each ``native name -> port name`` present on both."""
    metrics: dict[str, Any] = {}
    for native_name, port_name in names.items():
        if native_name not in native or port_name not in port:
            metrics[native_name] = {"missing": True}
            continue
        expected = np.asarray(native[native_name], np.float64)
        value = np.asarray(port[port_name], np.float64)
        if value.size != expected.size:
            metrics[native_name] = {
                "shape_mismatch": [list(value.shape), list(expected.shape)]
            }
            continue
        value = value.reshape(expected.shape)
        metrics[native_name] = {
            "port_name": port_name,
            "max_abs": max_abs(value, expected),
            "relative_rms": relative_rms(value, expected),
            "native_range": [float(expected.min()), float(expected.max())],
        }
    return metrics


def run_esmfold2(
    capture: Path, stages: set[str], stage_dir: Path | None = None
) -> dict[str, dict[str, Any]]:
    import jax
    import jax.numpy as jnp

    import tests.parity.test_esmfold2 as parity

    results: dict[str, dict[str, Any]] = {}
    features = npz(capture / "features.npz")
    atom_valid = features["atom_attention_mask"][0].astype(bool)
    names = atom_names_from_onehot(
        np.eye(64, dtype=np.int8)[np.clip(features["ref_atom_name_chars"][0], 0, 63)]
    )
    ca_mask = (names == "CA") & (features["ref_element"][0] == 6) & atom_valid

    # -- S1 ---------------------------------------------------------------
    def s1() -> dict[str, Any]:
        from foldjax.models.esmfold2 import inference
        from foldjax.paths import weights_dir

        document_path = Path(
            "/home/jaemin/non-project/optimizing/foldjax-bench/jctc-matrix-20260904/"
            "work/foldjax/esmfold2-protein_1ubq-seed101-cold/inputs/esmfold2_input.json"
        )
        document = json.loads(document_path.read_text())
        # That job's MSA directory is gone; the bench's 1UBQ alignment with the
        # sha256 OpenDDE's capture recorded for this sequence stands in for it.
        msa = Path(
            "/home/jaemin/non-project/optimizing/foldjax-bench/"
            "upstream-default-multimodal-n5-20260904/data/msa/1ubq_unpaired.a3m"
        )
        for entity in document["entities"]:
            if entity.get("unpaired_msa"):
                entity["unpaired_msa"] = str(msa)
        port = inference.build_common_job_features(
            document,
            base_dir=document_path.parent,
            ccd_path=weights_dir("esmfold2") / "ccd.pkl",
            seed=101,
        )
        port = {k: np.asarray(v) for k, v in port.items()}
        common = sorted(set(port) & set(features))
        records = {name: array_parity(port[name], features[name]) for name in common}
        summary = summarize_features(records)
        return stage_record(
            "measured",
            condition={
                **CPU_CONDITION,
                "port_featurizer": "foldjax.models.esmfold2.inference."
                "build_common_job_features (the backend's all-atom path), re-run "
                "in this checkout",
                "input_document": str(document_path),
                "msa": f"{msa} (sha256 {sha256(msa)})",
                "input_provenance": "INFERRED: the capture records only the "
                "feature archive's sha256 (core_only_shared_features=true), not "
                "the document it was built from; this is the JCTC-matrix 1UBQ job "
                "with the same sequence, its MSA path replaced as noted",
                "seed": 101,
                "native": "features.npz (the shared archive the capture ran)",
            },
            headline={
                "metric": "min exact-match fraction (categorical) / max |d| (float)",
                "value": f"{summary['min_exact_match_fraction']} / "
                f"{summary['max_float_max_abs']}",
            },
            metrics={
                "summary": summary,
                "only_in_port": sorted(set(port) - set(features)),
                "only_in_native": sorted(set(features) - set(port)),
                "arrays": records,
            },
            notes="The native side is the archive the capture ran, but the "
            "capture does not record which featurizer or document produced it, "
            "so a mismatch here does not by itself locate a port defect.",
        )

    # -- S2 ---------------------------------------------------------------
    def s2() -> dict[str, Any]:
        from safetensors import safe_open

        weights = parity._weights_directory()
        loaded = parity.load_structure_model(weights)
        params = {k: np.asarray(v) for k, v in loaded.parameters.items()}
        file = weights / "model.safetensors"
        groups: dict[str, dict[str, Any]] = {}
        file_keys: set[str] = set()
        with safe_open(file, framework="numpy") as handle:
            for name in handle.keys():  # noqa: SIM118
                file_keys.add(name)
                if name not in params:
                    continue
                native = handle.get_tensor(name)
                stored = params[name]
                group = groups.setdefault(
                    name.split(".", 1)[0],
                    {"tensors": 0, "elements": 0, "max_abs": 0.0, "dtypes": set()},
                )
                group["tensors"] += 1
                group["elements"] += int(native.size)
                group["dtypes"].add(f"{native.dtype}->{stored.dtype}")
                delta = (
                    max_abs(stored.astype(native.dtype), native) if native.size else 0.0
                )
                group["max_abs"] = max(group["max_abs"], delta)
        for group in groups.values():
            group["dtypes"] = sorted(group["dtypes"])
        binding = json.loads((capture / "metadata.json").read_text())["binding"]
        return stage_record(
            "measured",
            condition={
                "backend": "cpu (host; no model run)",
                "native_checkpoint": str(file),
                "native_sha256": sha256(file),
                "capture_checkpoint_sha256": binding["checkpoint"]["model.safetensors"],
                "converted": "parameters as the replay loads them: "
                "foldjax.models.esmfold2.inference.load(dtype='float32', "
                "language_model=False)",
                "mapping": "none by design: the port spells parameters as upstream's "
                "state_dict, so the checkpoint loads by being read",
            },
            headline={
                "metric": "max |loaded - checkpoint| over groups (checkpoint dtype)",
                "value": max((g["max_abs"] for g in groups.values()), default=0.0),
            },
            metrics={
                "groups": groups,
                "checkpoint_tensors": len(file_keys),
                "loaded_tensors": len(params),
                "checkpoint_tensors_not_loaded": sorted(file_keys - set(params))[:50],
                "loaded_not_in_checkpoint": sorted(set(params) - file_keys)[:50],
            },
            notes="ESM-C (the 25.4 GB language model) is not loaded or compared: "
            "S6 injects the native LM hidden states.",
        )

    # -- S6 ---------------------------------------------------------------
    def s6() -> dict[str, Any]:
        case = fixture_case("esmfold2", parity.CASE, parity.TIER)
        weights = parity._weights_directory()
        loaded = parity.load_structure_model(weights)
        case.assert_tripwire(parity.observed_schema(weights, loaded.settings))
        captured: dict[str, Any] = {}
        original = parity._compiled_predict

        def recording() -> Any:
            program = original()

            def call(*args: Any, **kwargs: Any) -> Any:
                output = program(*args, **kwargs)
                captured["output"] = output
                return output

            return call

        parity._compiled_predict = recording
        try:
            with highest_precision():
                coords, replay_features, seconds = parity.replay_to_coordinates(
                    case, loaded
                )
        finally:
            parity._compiled_predict = original
        native = npz(case.path(parity.FIXTURE_COORDS))["coords"].astype(np.float64)
        port = np.asarray(coords, np.float64)
        record = coordinate_metrics(port, native, atom_mask=atom_valid, ca_mask=ca_mask)
        record["entity_rmsd_angstrom_panel_metric"] = parity.per_sample_rmsd(
            native, port, replay_features
        )
        output = {
            k: np.asarray(v) for k, v in jax.device_get(captured["output"]).items()
        }
        native_conf = npz(capture / "upstream_confidence.npz")
        record["port_output_keys"] = sorted(output)
        record["end_to_end_confidence"] = confidence_comparison(
            output,
            native_conf,
            {k: k for k in sorted(native_conf) if k in output},
        )
        return stage_record(
            "measured",
            condition={
                **CPU_CONDITION,
                "input": "native features.npz (shared archive; core-only)",
                "trunk_dtype": "bfloat16 (native CUDA bf16 autocast; replay settings)",
                "injected_native": "upstream_lm.npz ESM-C hidden states (LM not run); "
                "tape.npz: initial_pair_state, LM-encoder dropout masks, MSA "
                "column/row draws, diffusion initial/churn normals, rotations, "
                "translations",
                "samples_x_steps": "5 x released schedule",
            },
            headline={
                "metric": "per-sample all-atom / CA RMSD (A), worst sample",
                "value": f"{max(record['all_atom_rmsd_angstrom']):.4f} / "
                f"{max(record['ca_rmsd_angstrom']):.4f}",
            },
            metrics=record,
            runtime_s=round(seconds, 1),
            notes="end_to_end_confidence compares the replay's own confidence "
            "outputs (port trunk, port coordinates) with upstream_confidence.npz; "
            "it is not S5, which needs the native trunk and coordinates injected.",
        )

    # -- S3-S5: the stage re-capture ----------------------------------------
    stage_capture = STAGE_CAPTURES["esmfold2"] if stage_dir is None else stage_dir
    cache: dict[str, Any] = {}

    def staged() -> dict[str, Any]:
        """The re-capture's own inputs, tape, LM output and stage tensors."""
        if "staged" not in cache:
            if not (stage_capture / "stages.npz").is_file():
                raise FileNotFoundError(f"no stages.npz in {stage_capture}")
            weights = parity._weights_directory()
            cache["staged"] = {
                "features": npz(stage_capture / "features.npz"),
                "tape": npz(stage_capture / "tape.npz"),
                "lm": npz(stage_capture / "upstream_lm.npz")["lm_hidden_states"],
                "stages": npz(stage_capture / "stages.npz"),
                "metadata": json.loads((stage_capture / "metadata.json").read_text()),
                "loaded": parity.load_structure_model(weights),
                "rerun": rerun_agreement(
                    capture,
                    stage_capture,
                    bitwise=("features.npz", "tape.npz"),
                    coordinates=("upstream_coords.npz", "coords"),
                    values={
                        "upstream_lm.npz": ("lm_hidden_states",),
                        "upstream_confidence.npz": (
                            "plddt",
                            "pae",
                            "ptm",
                            "complex_plddt",
                        ),
                    },
                    atom_mask=atom_valid,
                    ca_mask=ca_mask,
                ),
            }
        return cache["staged"]

    def replay_stage(
        *, stop_after_trunk: bool = False, representations: tuple[str, ...] = ()
    ) -> tuple[dict[str, np.ndarray], float]:
        """One jitted predict on the re-capture's tape, traced afresh.

        A new closure per call so a patched module global is read at this
        trace and never served from (or left in) another stage's cache.
        """
        from bench.esmfold2_lm_encoder_candidate import compiler_control
        from bench.esmfold2_tape import SEED, _model_features, _replay_settings
        from foldjax.models.esmfold2.models import model as structure_model

        data = staged()
        loaded = data["loaded"]
        settings = _replay_settings(loaded.settings)
        features = data["features"]
        n_chains = int(features["asym_id"].max()) + 1

        def program(key, arrays, params, dynamic):
            output = structure_model.predict(
                key,
                arrays,
                params,
                settings=settings,
                n_chains=n_chains,
                stop_after_trunk=stop_after_trunk,
                return_representations=representations,
                **dynamic,
            )
            if stop_after_trunk:
                output = dict(output)
                output["distogram_logits"] = structure_model._distogram_logits(
                    output["pair"], params
                )
            return output

        arrays = {k: jnp.asarray(v) for k, v in _model_features(features).items()}
        dynamic = {
            "lm_hidden_states": jnp.asarray(data["lm"]),
            **{k: jnp.asarray(v) for k, v in data["tape"].items()},
        }
        jitted = jax.jit(program, compiler_options=compiler_control("default"))
        jax.clear_caches()
        previous = jax.config.jax_default_matmul_precision
        jax.config.update("jax_default_matmul_precision", "highest")
        try:
            started = time.perf_counter()
            output = jitted(jax.random.key(SEED), arrays, loaded.parameters, dynamic)
            output = jax.device_get(jax.block_until_ready(output))
            seconds = time.perf_counter() - started
        finally:
            jax.config.update("jax_default_matmul_precision", previous)
            jax.clear_caches()
        return {k: np.asarray(v) for k, v in output.items()}, seconds

    def injections(names: Mapping[str, str]) -> Any:
        """Counted patches that hand the port the native tensors at its seams.

        ``names`` maps a seam to the stages.npz key it receives. Each value is
        handed over at the port's own width there when that is lossless.
        """
        from foldjax.models.esmfold2.models import model as structure_model

        native = staged()["stages"]
        widths: dict[str, str] = {}
        stack = contextlib.ExitStack()
        counts: dict[str, dict[str, int]] = {}

        def at_width(seam: str, value: Any) -> Any:
            array, width = native_at_port_width(native[names[seam]], value.dtype)
            widths[seam] = width
            if tuple(array.shape) != tuple(value.shape):
                raise ValueError(
                    f"{seam}: native {array.shape} vs port {tuple(value.shape)}"
                )
            return jnp.asarray(array)

        if "single_inputs" in names:
            original_inputs = structure_model.inputs_embedding

            def inputs(*args: Any, **kwargs: Any) -> Any:
                return at_width("single_inputs", original_inputs(*args, **kwargs))

            counts["single_inputs"] = stack.enter_context(
                counted_patch(structure_model, "inputs_embedding", inputs)
            )
        if "relative_position_encoding" in names:
            original_relpos = structure_model.relative_position_encoding

            def relpos(*args: Any, **kwargs: Any) -> Any:
                return at_width(
                    "relative_position_encoding", original_relpos(*args, **kwargs)
                )

            counts["relative_position_encoding"] = stack.enter_context(
                counted_patch(structure_model, "relative_position_encoding", relpos)
            )
        if "token_bonds_encoding" in names:
            original_bonds = structure_model._token_bonds_encoding

            def bonds(*args: Any, **kwargs: Any) -> Any:
                return at_width("token_bonds_encoding", original_bonds(*args, **kwargs))

            counts["token_bonds_encoding"] = stack.enter_context(
                counted_patch(structure_model, "_token_bonds_encoding", bonds)
            )
        if "pair" in names:
            original_trunk = structure_model.folding_trunk
            coda = {"n": 0}

            def trunk(x: Any, params: Any, prefix: str, *args: Any, **kwargs: Any):
                result = original_trunk(x, params, prefix, *args, **kwargs)
                if prefix != "parcae_coda":
                    return result
                coda["n"] += 1
                return at_width("pair", result)

            stack.enter_context(counted_patch(structure_model, "folding_trunk", trunk))
            counts["pair"] = coda
        if "coordinates" in names:
            coords = np.asarray(native[names["coordinates"]], np.float32)
            coords = jnp.asarray(coords.reshape(-1, *coords.shape[-2:]))
            sampler = {"n": 0}

            def sample(*args: Any, **kwargs: Any):
                sampler["n"] += 1
                widths["coordinates"] = "float32"
                return coords, None

            stack.enter_context(
                counted_patch(structure_model.diffusion, "sample", sample)
            )
            counts["coordinates"] = sampler
        return stack, counts, widths

    def require_fired(counts: Mapping[str, Mapping[str, int]]) -> None:
        silent = sorted(name for name, count in counts.items() if count["n"] < 1)
        if silent:
            raise RuntimeError(f"native injection never fired at {silent}")

    def stage_condition(**extra: Any) -> dict[str, Any]:
        data = staged()
        return {
            **CPU_CONDITION,
            "stage_capture": str(stage_capture),
            "input": "re-capture features.npz (bitwise equal to the stored "
            "capture's: "
            f"{data['rerun']['bitwise']['features.npz']['all_equal']})",
            "tape": "re-capture tape.npz (bitwise equal to the stored capture's: "
            f"{data['rerun']['bitwise']['tape.npz']['all_equal']}) and its own "
            "upstream_lm.npz (LM not run)",
            "samples_x_steps": "5 x released schedule",
            **extra,
        }

    def s3() -> dict[str, Any]:
        data = staged()
        native = data["stages"]
        port, seconds = replay_stage(
            stop_after_trunk=True, representations=("single_inputs", "pair")
        )
        pairs = {
            "single_inputs": ("single_inputs", "trunk.s_inputs"),
            "pair": ("pair", "trunk.z_trunk"),
            "distogram_logits": ("distogram_logits", "trunk.distogram_logits"),
        }
        arrays = {
            name: {
                "relative_rms": relative_rms(port[p], native[n]),
                "max_abs": max_abs(port[p], native[n]),
                "native_max_abs_value": float(np.abs(native[n]).max()),
                "native_key": n,
            }
            for name, (p, n) in pairs.items()
        }
        return stage_record(
            "measured",
            condition=stage_condition(
                trunk_dtype="bfloat16 (native CUDA bf16 autocast; replay settings)",
                compared_at="the parcae_coda output (upstream's z after "
                "z.float(), the z_trunk sample() receives), the input embedding "
                "x_inputs (sample()'s s_inputs) and the distogram logits",
            ),
            headline={
                "metric": "relative RMS s_inputs / pair (z_trunk)",
                "value": f"{arrays['single_inputs']['relative_rms']:.3e} / "
                f"{arrays['pair']['relative_rms']:.3e}",
            },
            metrics={
                "arrays": arrays,
                "native_rerun_vs_stored_capture": data["rerun"],
            },
            runtime_s=round(seconds, 1),
            notes="ESMFold2 has no single trunk output: x_inputs is the input "
            "embedding the sampler and confidence head read as their single "
            "input, so its row checks the input stage the trunk starts from.",
        )

    def s4() -> dict[str, Any]:
        data = staged()
        stack, counts, widths = injections(
            {
                "single_inputs": "trunk.s_inputs",
                "pair": "trunk.z_trunk",
                "relative_position_encoding": "trunk.relative_position_encoding",
            }
        )
        with stack:
            port, seconds = replay_stage()
        require_fired(counts)
        native = data["stages"]["diffusion.sample_atom_coords"].astype(np.float64)
        record = coordinate_metrics(
            np.asarray(port["sample_atom_coords"], np.float64).reshape(native.shape),
            native,
            atom_mask=atom_valid,
            ca_mask=ca_mask,
        )
        record["injection_calls"] = {k: v["n"] for k, v in counts.items()}
        record["injected_width"] = widths
        record["native_rerun_vs_stored_capture"] = data["rerun"]
        return stage_record(
            "measured",
            condition=stage_condition(
                trunk_dtype="n/a (native trunk injected)",
                diffusion_dtype="float32 (port default; native sampler runs "
                "outside the trunk autocast)",
                injected_native="stages.npz trunk.z_trunk at the parcae_coda "
                "output, trunk.s_inputs at inputs_embedding, "
                "trunk.relative_position_encoding at relative_position_encoding; "
                "tape.npz diffusion initial/churn normals, rotations, translations",
            ),
            headline={
                "metric": "per-sample all-atom / CA RMSD (A), worst sample",
                "value": f"{max(record['all_atom_rmsd_angstrom']):.4f} / "
                f"{max(record['ca_rmsd_angstrom']):.4f}",
            },
            metrics=record,
            runtime_s=round(seconds, 1),
            notes="Reference: the re-capture's own sample_atom_coords (the "
            "coordinates its sampler returned from the trunk injected here).",
        )

    def s5() -> dict[str, Any]:
        data = staged()
        stack, counts, widths = injections(
            {
                "single_inputs": "confidence_in.s_inputs",
                "pair": "confidence_in.z",
                "relative_position_encoding": (
                    "confidence_in.relative_position_encoding"
                ),
                "token_bonds_encoding": "confidence_in.token_bonds_encoding",
                "coordinates": "confidence_in.x_pred",
            }
        )
        with stack:
            port, seconds = replay_stage()
        require_fired(counts)
        native = {
            name.removeprefix("confidence_out."): value
            for name, value in data["stages"].items()
            if name.startswith("confidence_out.")
        }
        metrics: dict[str, Any] = confidence_comparison(
            port, native, {k: k for k in sorted(native)}
        )
        metrics["injection_calls"] = {k: v["n"] for k, v in counts.items()}
        metrics["injected_width"] = widths
        metrics["native_rerun_vs_stored_capture"] = data["rerun"]

        def value(name: str) -> float:
            return metrics.get(name, {}).get("max_abs", float("nan"))

        return stage_record(
            "measured",
            condition=stage_condition(
                trunk_dtype="n/a (native trunk injected)",
                confidence_dtype=str(staged()["loaded"].settings.confidence_dtype)
                + " (port setting; the native head's own folding trunk runs "
                "under bf16 autocast)",
                injected_native="stages.npz confidence_in.{s_inputs, z, "
                "relative_position_encoding, token_bonds_encoding} at their "
                "seams; confidence_in.x_pred replaces diffusion.sample",
            ),
            headline={
                "metric": "max |d| pLDDT (0-1) / PAE (A) / pTM",
                "value": "{:.3g} / {:.3g} / {:.3g}".format(
                    value("plddt"), value("pae"), value("ptm")
                ),
            },
            metrics=metrics,
            runtime_s=round(seconds, 1),
            notes="Reference: the re-capture's confidence_head outputs "
            "(forward hook), from the inputs injected here.",
        )

    runners = {"S1": s1, "S2": s2, "S3": s3, "S6": s6, "S4": s4, "S5": s5}
    for name, function in runners.items():
        if name in stages:
            results[name] = run_stage(f"esmfold2 {name}", function)
    return results


# --------------------------------------------------------------------------
# OpenDDE
# --------------------------------------------------------------------------


def run_opendde(
    capture: Path, stages: set[str], stage_dir: Path | None = None
) -> dict[str, dict[str, Any]]:
    import jax
    import jax.numpy as jnp

    import tests.parity.test_opendde as parity

    results: dict[str, dict[str, Any]] = {}
    identity = npz(capture / "native-identity.npz")
    ca_mask = ca_mask_from_names(
        identity["output_atom_name"], identity["output_atom_element"]
    )

    # -- S1 ---------------------------------------------------------------
    def s1() -> dict[str, Any]:
        from bench.protenix_closure_report import flat_features
        from foldjax.models.opendde.data.featurize_json import (
            featurize_opendde_json,
            load_jobs,
        )

        provenance = json.loads((capture / "provenance.json").read_text())
        document = Path(
            "/home/jaemin/non-project/optimizing/foldjax-bench/"
            "upstream-default-multimodal-n5-20260904/work/protein_1ubq/foldjax/"
            "opendde/inputs/opendde_input.json"
        )
        if sha256(document) != provenance["input_sha256"]:
            raise RuntimeError("the input document is not the one the capture ran")
        stored_port = npz(capture.parent / "fj-highest" / "foldjax-input.npz")
        use_template = any(k.startswith("template_") for k in stored_port)
        (job,) = load_jobs(document)
        port = flat_features(
            featurize_opendde_json(
                job,
                base_dir=document.parent,
                n_queries=32,
                n_keys=128,
                max_msa_depth=16384,
                seed=int(provenance.get("seed", 101)),
                use_template=use_template,
            )
        )
        native: dict[str, np.ndarray] = {}
        for name in ("native-input.npz", "native-derived.npz", "native-identity.npz"):
            native.update(npz(capture / name))
        common = sorted(set(port) & set(native))
        records = {name: array_parity(port[name], native[name]) for name in common}
        summary = summarize_features(records)
        from bench.entity_parity import compare_feature_dicts

        drift = compare_feature_dicts(port, stored_port)
        return stage_record(
            "measured",
            condition={
                **CPU_CONDITION,
                "port_featurizer": "foldjax.models.opendde.data.featurize_json."
                "featurize_opendde_json, re-run in this checkout "
                "(n_queries 32, n_keys 128, max_msa_depth 16384)",
                "input_document": str(document),
                "input_sha256_matches_capture": True,
                "use_template": use_template,
                "native": "native-input.npz + native-derived.npz + native-identity.npz",
                "name_mapping": "fields both sides spell the same; derived fields "
                "(relp, MSA mask, counts) are listed as only-in-one-side",
            },
            headline={
                "metric": "min exact-match fraction (categorical) / max |d| (float)",
                "value": f"{summary['min_exact_match_fraction']} / "
                f"{summary['max_float_max_abs']}",
            },
            metrics={
                "summary": summary,
                "only_in_port": sorted(set(port) - set(native)),
                "only_in_native": sorted(set(native) - set(port)),
                "arrays": records,
                "capture_time_input_audit_passed": json.loads(
                    (capture.parent / "fj-highest" / "input-audit.json").read_text()
                ).get("gate_passed"),
                "rerun_vs_capture_time_port_features_equal": drift["equal"],
                "rerun_vs_capture_time_port_features_value_mismatches": drift[
                    "value_mismatches"
                ],
            },
        )

    # -- S2 ---------------------------------------------------------------
    def s2() -> dict[str, Any]:
        from foldjax import torch_archive
        from foldjax.models._weights_io import _load_native_numpy_tree
        from foldjax.models.opendde.bridge.checkpoint import unwrap_state_dict
        from foldjax.models.opendde.bridge.torch_mapping import (
            map_opendde_inference_state_dict,
        )
        from foldjax.paths import foldjax_home

        native_path = foldjax_home() / "downloads" / "opendde" / "opendde.pt"
        state = TrackingState(unwrap_state_dict(torch_archive.load(native_path)))
        mapped = jax.tree.map(np.asarray, map_opendde_inference_state_dict(state))
        stored = _load_native_numpy_tree(parity.weights_path(), prestack=False)
        tree = compare_parameter_trees(stored, mapped)
        checks = weight_value_checks(
            state,
            [leaf for _, leaf in tree_leaves_with_paths(stored)],
        )
        provenance = json.loads((capture / "provenance.json").read_text())
        return stage_record(
            "measured",
            condition={
                "backend": "cpu (host NumPy; no model run)",
                "native_checkpoint": str(native_path),
                "native_sha256": sha256(native_path),
                "capture_checkpoint_sha256": provenance.get("checkpoint_sha256"),
                "converted": str(parity.weights_path()),
                "mapper": "foldjax.models.opendde.bridge.torch_mapping."
                "map_opendde_inference_state_dict",
                "reader": "foldjax.torch_archive.load (no torch)",
            },
            headline={
                "metric": "max |stored - map(native)| over groups (storage dtype)",
                "value": tree["max_abs"],
            },
            metrics={
                "stored_vs_mapped": tree,
                **checks,
            },
        )

    # -- S6 ---------------------------------------------------------------
    def s6() -> dict[str, Any]:
        from foldjax.models.opendde.bridge.weights_io import load_native_weights
        from foldjax.models.opendde.models.model import opendde_infer_static

        case = fixture_case("opendde", parity.CASE, parity.TIER)
        case.assert_tripwire(parity.observed_tripwire())
        features = parity.load_features(case)
        cycle_msa = parity.load_cycle_msa(case)
        with np.load(case.path("tape.npz"), allow_pickle=False) as archive:
            tape = {
                name: np.asarray(archive[name], np.float32) for name in archive.files
            }
        steps, samples = tape["step_noises"].shape[:2]
        params = load_native_weights(parity.weights_path())
        started = time.perf_counter()
        with highest_precision(), jax.default_matmul_precision(parity.MATMUL_PRECISION):
            output = opendde_infer_static(
                features,
                params,
                jnp.asarray(tape["noise_schedule"]),
                key=None,
                num_samples=samples,
                num_recycles=len(cycle_msa),
                run_confidence=True,
                cycle_msa_features=cycle_msa,
                diffusion_attention_backend="xla",
                trunk_single_attention_backend="xla",
                trunk_triangle_attention_backend="xla",
                structural_single_attention_backend="xla",
                structural_triangle_attention_backend="xla",
                init_noise=jnp.asarray(tape["init_noise"]),
                step_noises=tuple(
                    jnp.asarray(tape["step_noises"][i]) for i in range(steps)
                ),
                rotations=jnp.asarray(tape["rotations"]),
                translations=jnp.asarray(tape["translations"]),
            )
            output = jax.device_get(output)
        seconds = time.perf_counter() - started
        # The released summaries (pLDDT, PAE, pTM), from the port's logits by
        # the port's own postprocess, as the CLI writes them.
        from foldjax.models.opendde.postprocess import opendde_confidence_scores

        scores = opendde_confidence_scores(
            dict(output),
            features,
            num_recycles=len(cycle_msa),
            include_shape_complementarity=False,
        )
        output = {**output, **jax.device_get(dict(scores))}
        output = {k: np.asarray(v) for k, v in output.items() if hasattr(v, "shape")}
        port = output["coordinate"].astype(np.float64)
        while port.ndim > 3 and port.shape[0] == 1:
            port = port[0]
        native = parity.native_coordinates(case)
        record = coordinate_metrics(port, native, ca_mask=ca_mask)
        record["entity_rmsd_angstrom_panel_metric"] = [
            {
                str(k): v
                for k, v in parity.entity_rmsds(case, port[i], native[i]).items()
            }
            for i in range(samples)
        ]
        raw = npz(capture / "raw.npz")
        native_conf = {
            "plddt_logits": raw["plddt"],
            "pae_logits": raw["pae"],
            "pde_logits": raw["pde"],
            "atom_plddt": np.stack(
                [raw[f"full_data.{i}.atom_plddt"] for i in range(samples)]
            ),
            "token_pair_pae": np.stack(
                [raw[f"full_data.{i}.token_pair_pae"] for i in range(samples)]
            ),
            "ptm": np.array(
                [raw[f"summary_confidence.{i}.ptm"] for i in range(samples)]
            ),
        }
        record["port_output_keys"] = sorted(output)
        record["end_to_end_confidence"] = confidence_comparison(
            output,
            native_conf,
            {
                "plddt_logits": "plddt",
                "pae_logits": "pae",
                "pde_logits": "pde",
                "atom_plddt": "atom_plddt",
                "token_pair_pae": "token_pair_pae",
                "ptm": "summary_ptm",
            },
        )
        return stage_record(
            "measured",
            condition={
                **CPU_CONDITION,
                "input": "native-input.npz + native-derived.npz (upstream "
                "featurizer output; core-only)",
                "trunk_dtype": "float32 (port default; native TF32 trunk -> CPU "
                "highest, see docs/parity-cpu.md)",
                "kernels": "all attention backends xla",
                "injected_native": "tape.npz (noise schedule, initial/churn noise, "
                "rotations, translations) + msa.npz per-recycle MSA rows",
                "samples_x_steps": f"{samples} x {steps} (all samples; the parity "
                "subset replays sample 0 only)",
                "confidence": "on",
            },
            headline={
                "metric": "per-sample all-atom / CA RMSD (A), worst sample",
                "value": f"{max(record['all_atom_rmsd_angstrom']):.4f} / "
                f"{max(record['ca_rmsd_angstrom']):.4f}",
            },
            metrics=record,
            runtime_s=round(seconds, 1),
            notes="end_to_end_confidence compares the replay's own confidence "
            "with raw.npz; it is not S5, which needs the native trunk injected.",
        )

    # -- S3-S5: the stage re-capture ----------------------------------------
    stage_capture = STAGE_CAPTURES["opendde"] if stage_dir is None else stage_dir
    cache: dict[str, Any] = {}

    class CaptureCase:
        """``case.path`` of the parity loaders over a capture's own layout."""

        def __init__(self, root: Path) -> None:
            self.root = root
            self.entry = type(
                "Entry",
                (),
                {
                    "port": "opendde",
                    "case": "protein_1ubq",
                    "tier": "stage re-capture",
                    "capture_provenance": str(root),
                },
            )()

        def path(self, name: str) -> Path:
            nested = name in ("tape.npz", "msa.npz", "coordinate.npz")
            return self.root / "torch" / name if nested else self.root / name

    def staged() -> dict[str, Any]:
        if "staged" not in cache:
            from foldjax.models.opendde.bridge.weights_io import load_native_weights

            if not (stage_capture / "stages.npz").is_file():
                raise FileNotFoundError(f"no stages.npz in {stage_capture}")
            case = CaptureCase(stage_capture)
            with np.load(case.path("tape.npz"), allow_pickle=False) as archive:
                tape = {k: np.asarray(archive[k], np.float32) for k in archive.files}
            stages_npz = npz(stage_capture / "stages.npz")
            # The same tensors seen at two boundaries must be one value.
            boundary_identity = {
                f"{left} == {right}": bool(
                    np.array_equal(stages_npz[left], stages_npz[right])
                )
                for left, right in (
                    ("structural_trunk.out.s_inputs", "diffusion.in.s_inputs"),
                    ("structural_trunk.out.s", "diffusion.in.s"),
                    ("structural_trunk.out.z", "diffusion.in.z"),
                    ("residue_trunk.out.s_inputs", "confidence.in.s_inputs"),
                    ("residue_trunk.out.s", "confidence.in.s_trunk"),
                    ("residue_trunk.out.z", "confidence.in.z_trunk"),
                    ("diffusion.out.coordinate", "confidence.in.x_pred_coords"),
                )
            }
            cache["staged"] = {
                "case": case,
                "features": parity.load_features(case),
                "cycle_msa": parity.load_cycle_msa(case),
                "tape": tape,
                "stages": stages_npz,
                "boundary_identity": boundary_identity,
                "params": load_native_weights(parity.weights_path()),
                "rerun": rerun_agreement(
                    capture,
                    stage_capture,
                    bitwise=(
                        "native-input.npz",
                        "native-derived.npz",
                        "native-identity.npz",
                        "torch/tape.npz",
                        "torch/msa.npz",
                    ),
                    coordinates=("raw.npz", "coordinate"),
                    values={
                        "raw.npz": (
                            "plddt",
                            "pae",
                            "pde",
                            *(f"summary_confidence.{i}.ptm" for i in range(5)),
                            *(f"summary_confidence.{i}.plddt" for i in range(5)),
                        )
                    },
                    ca_mask=ca_mask,
                ),
            }
        return cache["staged"]

    def replay_stage(
        *,
        stop_after_trunk: bool = False,
        run_confidence: bool = False,
        capture_names: tuple[str, ...] = (),
    ) -> tuple[dict[str, np.ndarray], float]:
        """The S6 call on the re-capture's inputs and tape (eager, all xla)."""
        from foldjax.models import _capture
        from foldjax.models.opendde.models.model import opendde_infer_static

        data = staged()
        tape = data["tape"]
        steps, samples = tape["step_noises"].shape[:2]
        cycle_msa = data["cycle_msa"]
        started = time.perf_counter()
        with (
            highest_precision(),
            jax.default_matmul_precision(parity.MATMUL_PRECISION),
            _capture.capturing(capture_names),
        ):
            output = opendde_infer_static(
                data["features"],
                data["params"],
                jnp.asarray(tape["noise_schedule"]),
                key=None,
                num_samples=samples,
                num_recycles=len(cycle_msa),
                run_confidence=run_confidence,
                stop_after_trunk=stop_after_trunk,
                cycle_msa_features=cycle_msa,
                diffusion_attention_backend="xla",
                trunk_single_attention_backend="xla",
                trunk_triangle_attention_backend="xla",
                structural_single_attention_backend="xla",
                structural_triangle_attention_backend="xla",
                init_noise=jnp.asarray(tape["init_noise"]),
                step_noises=tuple(
                    jnp.asarray(tape["step_noises"][i]) for i in range(steps)
                ),
                rotations=jnp.asarray(tape["rotations"]),
                translations=jnp.asarray(tape["translations"]),
            )
            output = jax.device_get(output)
        seconds = time.perf_counter() - started
        if run_confidence:
            from foldjax.models.opendde.postprocess import opendde_confidence_scores

            scores = opendde_confidence_scores(
                dict(output),
                data["features"],
                num_recycles=len(cycle_msa),
                include_shape_complementarity=False,
            )
            output = {**output, **jax.device_get(dict(scores))}
        return {
            k: np.asarray(v) for k, v in output.items() if hasattr(v, "shape")
        }, seconds

    def injections(
        residue: tuple[str, str, str] | None,
        structural: tuple[str, str, str] | None,
        coordinates: str | None,
    ) -> Any:
        """Counted module-global patches at the port's trunk/sampler seams.

        ``residue`` replaces ``pairformer_output_from_s_inputs`` (the residue
        trunk is then not run); ``structural`` replaces the expanded
        ``s_inputs`` and the refiner's ``s``/``z`` (the refiner is not run;
        the expander still builds the structural pair features); ``coordinates``
        replaces ``sample_diffusion``.
        """
        from foldjax.models.opendde.models import model as opendde_model

        native = staged()["stages"]
        stack = contextlib.ExitStack()
        counts: dict[str, dict[str, int]] = {}
        widths: dict[str, str] = {}

        def at_width(key: str, like: Any) -> Any:
            array, width = native_at_port_width(native[key], like.dtype)
            widths[key] = width
            if tuple(array.shape) != tuple(like.shape):
                raise ValueError(f"{key}: native {array.shape} vs port {like.shape}")
            return jnp.asarray(array)

        if residue is not None:

            def trunk(features: Any, s_inputs: Any, *args: Any, **kwargs: Any):
                s_key, si_key, z_key = residue[1], residue[0], residue[2]
                return (
                    at_width(si_key, s_inputs),
                    jnp.asarray(np.asarray(native[s_key], np.float32)),
                    jnp.asarray(np.asarray(native[z_key], np.float32)),
                )

            counts["residue_trunk"] = stack.enter_context(
                counted_patch(opendde_model, "pairformer_output_from_s_inputs", trunk)
            )
            widths[residue[1]] = widths[residue[2]] = "float32"
        if structural is not None:
            original_expand = opendde_model.structural_token_expand

            def expand(*args: Any, **kwargs: Any):
                s_inputs, s, z, pair_features = original_expand(*args, **kwargs)
                return at_width(structural[0], s_inputs), s, z, pair_features

            def refine(s: Any, z: Any, *args: Any, **kwargs: Any):
                return at_width(structural[1], s), at_width(structural[2], z)

            counts["structural_expand"] = stack.enter_context(
                counted_patch(opendde_model, "structural_token_expand", expand)
            )
            counts["structural_refiner"] = stack.enter_context(
                counted_patch(opendde_model, "structural_refiner_stack", refine)
            )
        if coordinates is not None:
            coords = np.asarray(native[coordinates], np.float32)
            coords = jnp.asarray(coords.reshape(-1, *coords.shape[-2:]))
            widths[coordinates] = "float32"

            def sample(*args: Any, **kwargs: Any):
                return coords

            counts["sample_diffusion"] = stack.enter_context(
                counted_patch(opendde_model, "sample_diffusion", sample)
            )
        return stack, counts, widths

    def require_fired(counts: Mapping[str, Mapping[str, int]]) -> None:
        silent = sorted(name for name, count in counts.items() if count["n"] < 1)
        if silent:
            raise RuntimeError(f"native injection never fired at {silent}")

    def stage_condition(**extra: Any) -> dict[str, Any]:
        data = staged()
        bitwise = data["rerun"]["bitwise"]
        return {
            **CPU_CONDITION,
            "stage_capture": str(stage_capture),
            "input": "re-capture native-input.npz + native-derived.npz (bitwise "
            "equal to the stored capture's: "
            f"{bitwise['native-input.npz']['all_equal']} / "
            f"{bitwise['native-derived.npz']['all_equal']})",
            "tape": "re-capture torch/tape.npz + torch/msa.npz (bitwise equal to "
            "the stored capture's: "
            f"{bitwise['torch/tape.npz']['all_equal']} / "
            f"{bitwise['torch/msa.npz']['all_equal']})",
            "kernels": "all attention backends xla",
            **extra,
        }

    def s3() -> dict[str, Any]:
        data = staged()
        native = data["stages"]
        port, seconds = replay_stage(
            stop_after_trunk=True,
            capture_names=(
                "single_inputs",
                "single",
                "pair",
                "structural_single_inputs",
                "structural_single",
                "structural_pair",
            ),
        )
        pairs = {
            "single_inputs": ("single_inputs", "residue_trunk.out.s_inputs"),
            "single": ("single", "residue_trunk.out.s"),
            "pair": ("pair", "residue_trunk.out.z"),
            "structural_single_inputs": (
                "structural_single_inputs",
                "structural_trunk.out.s_inputs",
            ),
            "structural_single": ("structural_single", "structural_trunk.out.s"),
            "structural_pair": ("structural_pair", "structural_trunk.out.z"),
        }
        arrays = {
            name: {
                "relative_rms": relative_rms(port[p], native[n]),
                "max_abs": max_abs(port[p], native[n]),
                "native_max_abs_value": float(np.abs(native[n]).max()),
                "native_key": n,
            }
            for name, (p, n) in pairs.items()
        }
        return stage_record(
            "measured",
            condition=stage_condition(
                trunk_dtype="float32 (port default; native TF32 trunk -> CPU "
                "highest, see docs/parity-cpu.md)",
                recycles=len(data["cycle_msa"]),
            ),
            headline={
                "metric": "relative RMS single (s) / pair (z); residue trunk, "
                "then structural refiner",
                "value": "{:.3e} / {:.3e}; {:.3e} / {:.3e}".format(
                    arrays["single"]["relative_rms"],
                    arrays["pair"]["relative_rms"],
                    arrays["structural_single"]["relative_rms"],
                    arrays["structural_pair"]["relative_rms"],
                ),
            },
            metrics={
                "arrays": arrays,
                "native_boundary_identity": data["boundary_identity"],
                "native_rerun_vs_stored_capture": data["rerun"],
            },
            runtime_s=round(seconds, 1),
        )

    def s4() -> dict[str, Any]:
        data = staged()
        stack, counts, widths = injections(
            (
                "residue_trunk.out.s_inputs",
                "residue_trunk.out.s",
                "residue_trunk.out.z",
            ),
            ("diffusion.in.s_inputs", "diffusion.in.s", "diffusion.in.z"),
            None,
        )
        with stack:
            port, seconds = replay_stage()
        require_fired(counts)
        native = data["stages"]["diffusion.out.coordinate"].astype(np.float64)
        coords = port["coordinate"].astype(np.float64)
        while coords.ndim > 3 and coords.shape[0] == 1:
            coords = coords[0]
        record = coordinate_metrics(coords, native, ca_mask=ca_mask)
        record["injection_calls"] = {k: v["n"] for k, v in counts.items()}
        record["injected_width"] = widths
        record["native_boundary_identity"] = data["boundary_identity"]
        record["native_rerun_vs_stored_capture"] = data["rerun"]
        tape = data["tape"]
        return stage_record(
            "measured",
            condition=stage_condition(
                trunk_dtype="n/a (native trunk injected)",
                diffusion_dtype="float32 (native skip_amp.sample_diffusion=true)",
                injected_native="stages.npz diffusion.in.{s_inputs, s, z} (the "
                "structural tensors run_sample_diffusion_stage received) at "
                "structural_token_expand / structural_refiner_stack; residue "
                "trunk from residue_trunk.out; torch/tape.npz noise schedule, "
                "initial/step noise, rotations, translations",
                samples_x_steps=f"{tape['step_noises'].shape[1]} x "
                f"{tape['step_noises'].shape[0]}",
            ),
            headline={
                "metric": "per-sample all-atom / CA RMSD (A), worst sample",
                "value": f"{max(record['all_atom_rmsd_angstrom']):.4f} / "
                f"{max(record['ca_rmsd_angstrom']):.4f}",
            },
            metrics=record,
            runtime_s=round(seconds, 1),
            notes="Reference: the re-capture's own sampler output "
            "(diffusion.out.coordinate), from the tensors injected here.",
        )

    def s5() -> dict[str, Any]:
        data = staged()
        stack, counts, widths = injections(
            (
                "confidence.in.s_inputs",
                "confidence.in.s_trunk",
                "confidence.in.z_trunk",
            ),
            ("diffusion.in.s_inputs", "diffusion.in.s", "diffusion.in.z"),
            "confidence.in.x_pred_coords",
        )
        with stack:
            port, seconds = replay_stage(run_confidence=True)
        require_fired(counts)
        native_stage = data["stages"]
        raw = npz(stage_capture / "raw.npz")
        samples = int(data["tape"]["step_noises"].shape[1])
        native = {
            "plddt_logits": native_stage["confidence.out.plddt"],
            "pae_logits": native_stage["confidence.out.pae"],
            "pde_logits": native_stage["confidence.out.pde"],
            "resolved_logits": native_stage["confidence.out.resolved"],
            "atom_plddt": np.stack(
                [raw[f"full_data.{i}.atom_plddt"] for i in range(samples)]
            ),
            "token_pair_pae": np.stack(
                [raw[f"full_data.{i}.token_pair_pae"] for i in range(samples)]
            ),
            "ptm": np.array(
                [raw[f"summary_confidence.{i}.ptm"] for i in range(samples)]
            ),
            "summary_plddt": np.array(
                [raw[f"summary_confidence.{i}.plddt"] for i in range(samples)]
            ),
        }
        metrics: dict[str, Any] = confidence_comparison(
            port,
            native,
            {
                "plddt_logits": "plddt",
                "pae_logits": "pae",
                "pde_logits": "pde",
                "resolved_logits": "resolved",
                "atom_plddt": "atom_plddt",
                "token_pair_pae": "token_pair_pae",
                "ptm": "summary_ptm",
                "summary_plddt": "summary_plddt",
            },
        )
        metrics["port_output_keys"] = sorted(port)
        metrics["injection_calls"] = {k: v["n"] for k, v in counts.items()}
        metrics["injected_width"] = widths
        metrics["native_boundary_identity"] = data["boundary_identity"]
        metrics["native_rerun_vs_stored_capture"] = data["rerun"]

        def value(name: str) -> float:
            return metrics.get(name, {}).get("max_abs", float("nan"))

        return stage_record(
            "measured",
            condition=stage_condition(
                trunk_dtype="n/a (native trunk injected)",
                confidence_dtype="float32 (native skip_amp.confidence_head=true)",
                injected_native="stages.npz confidence.in.{s_inputs, s_trunk, "
                "z_trunk} replace pairformer_output_from_s_inputs; "
                "confidence.in.x_pred_coords replace sample_diffusion; "
                "summaries vs the re-capture's raw.npz",
            ),
            headline={
                "metric": "max |d| atom pLDDT (0-100) / PAE (A) / pTM",
                "value": "{:.3g} / {:.3g} / {:.3g}".format(
                    value("atom_plddt"), value("token_pair_pae"), value("ptm")
                ),
            },
            metrics=metrics,
            runtime_s=round(seconds, 1),
            notes="Logits are compared with the re-capture's confidence-head "
            "outputs; the released summaries with its raw.npz (the port's "
            "postprocess on the port's logits).",
        )

    runners = {"S1": s1, "S2": s2, "S3": s3, "S6": s6, "S4": s4, "S5": s5}
    for name, function in runners.items():
        if name in stages:
            results[name] = run_stage(f"opendde {name}", function)
    return results


# --------------------------------------------------------------------------
# AlphaFold 3: FoldJAX's vendored source against DeepMind's run_alphafold.py
# --------------------------------------------------------------------------

AF3_NO_STAGE = (
    "Not a stage-injection row: AlphaFold 3 is compared as two complete CPU "
    "runs (DeepMind's run_alphafold.py and FoldJAX's vendored AF3) on the same "
    "input and seed; no intermediate trunk or sampler boundary is taped or "
    "injected, so S3/S4 have no separate measurement."
)


def cif_coordinates(path: Path) -> tuple[list[tuple[str, int, str]], np.ndarray]:
    """Atom keys (chain, residue number, atom name) and coordinates of an mmCIF."""
    import gemmi

    structure = gemmi.read_structure(str(path))
    keys, xyz = [], []
    for chain in structure[0]:
        for residue in chain:
            for atom in residue:
                keys.append((chain.name, residue.seqid.num, atom.name))
                xyz.append(atom.pos.tolist())
    return keys, np.asarray(xyz, np.float64)


def cli_route_comparison(native_dir: Path, port_dir: Path) -> dict[str, Any]:
    """Per-sample coordinates of two CLI output trees, matched by atom key."""
    record: dict[str, Any] = {"native": str(native_dir), "port": str(port_dir)}

    def sample_files(root: Path) -> dict[int, Path]:
        found: dict[int, Path] = {}
        for path in sorted(root.rglob("*_model.cif")):
            for part in path.parts:
                if part.startswith("seed-") and "_sample-" in part:
                    found[int(part.rsplit("_sample-", 1)[1])] = path
        return found

    native, port = sample_files(native_dir), sample_files(port_dir)
    record["samples"] = sorted(native)
    if not native or sorted(native) != sorted(port):
        record["error"] = f"sample sets differ: {sorted(native)} vs {sorted(port)}"
        return record
    rows = []
    for index in sorted(native):
        nkeys, nxyz = cif_coordinates(native[index])
        pkeys, pxyz = cif_coordinates(port[index])
        if nkeys != pkeys:
            common = sorted(set(nkeys) & set(pkeys))
            nidx = {k: i for i, k in enumerate(nkeys)}
            pidx = {k: i for i, k in enumerate(pkeys)}
            nxyz = nxyz[[nidx[k] for k in common]]
            pxyz = pxyz[[pidx[k] for k in common]]
        rows.append(
            {
                "sample": index,
                "atoms": int(len(nxyz)),
                "atom_keys_identical": nkeys == pkeys,
                "max_abs_dxyz_angstrom": max_abs(pxyz, nxyz),
                "all_atom_rmsd_angstrom": kabsch_rmsd(pxyz, nxyz),
                "coordinates_identical_as_written": bool(np.array_equal(pxyz, nxyz)),
            }
        )
    record["per_sample"] = rows
    return record


def run_alphafold3(capture: Path, stages: set[str]) -> dict[str, dict[str, Any]]:
    """``capture`` is the DeepMind arm of ``bench/af3_closure_capture.py --cpu``;
    the FoldJAX arm is its sibling ``harness-foldjax``, and the CLI route's
    output trees (``cli-deepmind``, ``cli-foldjax``) sit beside both."""
    root = capture.parent
    port_dir = root / "harness-foldjax"
    results: dict[str, dict[str, Any]] = {}
    provenance = {
        arm: json.loads((path / "provenance.json").read_text())
        for arm, path in (("deepmind", capture), ("foldjax", port_dir))
    }
    condition = {
        "backend": "cpu (JAX_PLATFORMS=cpu, both arms)",
        "matmul_precision": "AF3 default (bfloat16: 'all'; not pinned)",
        "native": "DeepMind run_alphafold.py v3.0.4 (archived checkout "
        "deepmind-af3-85c4d20): ModelRunner + predict_structure",
        "port": "FoldJAX vendored AF3 through foldjax.predict, "
        "options buckets=DeepMind's list",
        "environment": "the common FoldJAX JAX environment for both arms",
        "samples_x_steps": "{} x 200; recycles {}".format(
            provenance["deepmind"]["samples"], provenance["deepmind"]["native_recycles"]
        ),
        "attention": provenance["deepmind"]["attention"],
        "buckets": provenance["deepmind"]["buckets"],
        "seed": 101,
    }

    def s1() -> dict[str, Any]:
        native, port = npz(capture / "input.npz"), npz(port_dir / "input.npz")
        common = sorted(set(native) & set(port))
        records = {name: array_parity(port[name], native[name]) for name in common}
        bitwise = {
            name: bool(
                native[name].dtype == port[name].dtype
                and native[name].shape == port[name].shape
                and native[name].tobytes() == port[name].tobytes()
            )
            for name in common
        }
        metadata_equal = json.loads(
            (capture / "input-metadata.json").read_text()
        ) == json.loads((port_dir / "input-metadata.json").read_text())
        summary = summarize_features(records)
        identical = (
            all(bitwise.values()) and set(native) == set(port) and metadata_equal
        )
        return stage_record(
            "measured",
            condition={
                **condition,
                "compared": "the featurised batch each arm handed "
                "ModelRunner.run_inference (input.npz + input-metadata.json)",
            },
            headline={
                "metric": "arrays bitwise identical (count) / max |d| (float)",
                "value": f"{sum(bitwise.values())}/{len(bitwise)} identical"
                f"{' (all, metadata equal)' if identical else ''} / "
                f"{summary['max_float_max_abs']}",
            },
            metrics={
                "summary": summary,
                "bitwise_identical": identical,
                "bitwise": bitwise,
                "metadata_equal": metadata_equal,
                "only_in_port": sorted(set(port) - set(native)),
                "only_in_native": sorted(set(native) - set(port)),
                "arrays": records,
                "port_input_audit": json.loads(
                    (port_dir / "input-audit.json").read_text()
                ),
            },
        )

    def s2() -> dict[str, Any]:
        native = json.loads((capture / "parameters.json").read_text())
        port = json.loads((port_dir / "parameters.json").read_text())
        differing = sorted(
            name
            for name in set(native) | set(port)
            if native.get(name) != port.get(name)
        )
        same_file = (
            provenance["deepmind"]["weights_sha256"]
            == provenance["foldjax"]["weights_sha256"]
        )
        return stage_record(
            "measured",
            condition={
                "backend": "cpu (host hashes; no model run)",
                "parameter_file": "af3.bin, sha256 "
                + provenance["deepmind"]["weights_sha256"],
                "compared": "per-parameter sha256 of the arrays each arm's "
                "ModelRunner.model_params held at inference",
            },
            headline={
                "metric": "same parameter file / parameters with differing hash",
                "value": f"{same_file} / {len(differing)} of {len(native)}",
            },
            metrics={
                "weights_sha256": {
                    arm: record["weights_sha256"] for arm, record in provenance.items()
                },
                "same_parameter_file": same_file,
                "parameters": len(native),
                "differing": differing[:50],
            },
        )

    def s5() -> dict[str, Any]:
        native, port = npz(capture / "confidence.npz"), npz(port_dir / "confidence.npz")
        numeric = sorted(
            name
            for name in set(native) & set(port)
            if native[name].dtype.kind in "fiub"
        )
        values = {
            name: {
                "max_abs": max_abs(port[name], native[name]),
                "bitwise_identical": bool(
                    native[name].tobytes() == port[name].tobytes()
                ),
            }
            for name in numeric
        }
        identical = sum(v["bitwise_identical"] for v in values.values())

        def worst(suffix: str) -> float:
            hits = [v["max_abs"] for k, v in values.items() if k.endswith(suffix)]
            return max(hits) if hits else float("nan")

        raw_native, raw_port = npz(capture / "raw.npz"), npz(port_dir / "raw.npz")
        raw = {
            name: {
                "max_abs": max_abs(raw_port[name], raw_native[name]),
                "bitwise_identical": bool(
                    raw_native[name].tobytes() == raw_port[name].tobytes()
                ),
            }
            for name in sorted(set(raw_native) & set(raw_port))
            if raw_native[name].dtype.kind in "fiub"
        }

        return stage_record(
            "measured",
            condition={**condition, "compared": "every numeric confidence leaf"},
            headline={
                "metric": "max |d| atom pLDDT / PAE / pTM; leaves bitwise identical",
                "value": "{:.3g} / {:.3g} / {:.3g}; {}/{}".format(
                    worst(".atom_plddt"),
                    worst(".numerical.full_pae"),
                    worst(".metadata.ptm"),
                    identical,
                    len(values),
                ),
            },
            metrics={"leaves": values, "raw_model_outputs": raw},
            notes="raw_model_outputs compares run_inference's padded outputs "
            "(predicted lDDT, PAE/PDE, distogram contact probabilities, the "
            "diffusion samples) before extraction.",
        )

    def s6() -> dict[str, Any]:
        native = npz(capture / "coordinate.npz")
        port = npz(port_dir / "coordinate.npz")
        mask = native["mask"][0].astype(bool)
        record = coordinate_metrics(
            port["coordinate"], native["coordinate"], atom_mask=mask
        )
        record["max_abs_dxyz_angstrom"] = max_abs(
            port["coordinate"][:, mask], native["coordinate"][:, mask]
        )
        record["bitwise_identical"] = bool(
            np.array_equal(port["coordinate"][:, mask], native["coordinate"][:, mask])
        )
        record["masks_equal"] = bool(np.array_equal(native["mask"], port["mask"]))
        if (root / "cli-deepmind").is_dir() and (root / "cli-foldjax").is_dir():
            record["cli_route"] = cli_route_comparison(
                root / "cli-deepmind", root / "cli-foldjax"
            )
        return stage_record(
            "measured",
            condition={
                **condition,
                "cli_route": "jctc-v2/e9/run_af3_deepmind.py (AF3CMP_CPU_SMOKE=1) "
                "vs foldjax.cli predict --option buckets=..., CIF coordinates",
            },
            headline={
                "metric": "max |dxyz| (A) over all samples / all-atom RMSD worst "
                "sample",
                "value": f"{record['max_abs_dxyz_angstrom']:.4g} / "
                f"{max(record['all_atom_rmsd_angstrom']):.4g}"
                + (" (bitwise identical)" if record["bitwise_identical"] else ""),
            },
            metrics=record,
        )

    for name in ("S3", "S4"):
        if name in stages:
            results[name] = not_captured(AF3_NO_STAGE)
    runners = {"S1": s1, "S2": s2, "S5": s5, "S6": s6}
    for name, function in runners.items():
        if name in stages:
            results[name] = run_stage(f"alphafold3 {name}", function)
    return results


# --------------------------------------------------------------------------
# CLI and table
# --------------------------------------------------------------------------

MODELS: dict[str, Callable[..., dict[str, dict[str, Any]]]] = {
    "protenix": run_protenix,
    "boltz2": run_boltz2,
    "openfold3": run_openfold3,
    "esmfold2": run_esmfold2,
    "opendde": run_opendde,
    "alphafold3": run_alphafold3,
}

#: Models compared against a second implementation rather than a CPU parity
#: manifest; their table context comes from the record itself.
NO_PARITY_MANIFEST = frozenset({"alphafold3"})


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
            ["git", "-C", str(REPO), "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            check=True,
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


def run_model(
    model: str, capture: Path, stages: set[str], stage_capture: Path | None = None
) -> dict[str, Any]:
    require_cpu()
    started = time.perf_counter()
    if model in STAGE_CAPTURES:
        results = MODELS[model](capture, stages, stage_capture)
    elif stage_capture is not None:
        raise ValueError(f"{model} takes no --stage-capture")
    else:
        results = MODELS[model](capture, stages)
    ordered = {
        f"{key}_{label}": results.get(
            key, stage_record("not_captured", notes="not run")
        )
        for key, label in STAGES
        if key in stages
    }
    env = environment()
    for record in ordered.values():
        # Per stage, because --merge combines stages from separate runs.
        record["source"] = {
            "git": env.get("git_commit"),
            "cpu_affinity": env.get("cpu_affinity"),
            "recorded": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    return {
        "model": model,
        "capture": str(capture),
        "environment": env,
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


TABLE_PREAMBLE = """\
# Stage parity: FoldJAX ports against stored native captures (CPU)

Generated by `python bench/stage_parity.py --table bench/stage_parity_results`
from the per-model JSON files beside this one; edit those, not this table.
Every model stage runs on CPU XLA with float32 matmuls at `highest`, against a
native capture taken on GPU. The trunk dtype follows upstream's own AMP policy
(bf16 trunk for Protenix, Boltz-2 and ESMFold2) and is recorded per row. S3-S6
replay every draw the capture taped (MSA rows, dropout, initial and churn
noise, augmentation); the S1 featurizer draws that were *not* replayed are
named in `MISSING.md`. The JSON files carry the per-array, per-sample and
per-group numbers, the full condition of each stage and the source commit.

* S1 categorical arrays: exact-match fraction; float arrays: max |d|.
* S2: max |d| between the stored converted parameters and the port's own
  mapping of the native checkpoint, in the stored dtype; the JSON adds a
  mapping-independent check that every converted float occurs in the
  checkpoint.
* S3: relative RMS = rms(port - native) / rms(native), per array.
* S4/S6: one proper Kabsch fit on all atoms per sample; all-atom RMSD and CA
  RMSD (CA measured under that fit, not refitted); worst of all five samples,
  *including* samples the CPU parity manifests exclude (listed per model
  below, with the manifests' own CPU calibration).
* S5: max |d| on the native scale of each score.
* `not_captured`: the capture holds nothing to compare; see `MISSING.md`.

Runtimes are wall seconds on 16 pinned cores of a shared 144-core host whose
load average was 80-90 during the recorded runs; they are not comparable to
the manifests' 8-core `wall_seconds`. The Boltz-2, ESMFold2 and OpenFold3
records name commit `9b7aeab`, which landed while they ran; the code they
executed was `e70017b`, which differs from it only by formatting and an
OpenDDE-only change. The OpenDDE record names `4efa1b2`, whose code is
`9b7aeab` (the commits between them add result files only).
"""


def _manifest_context(model: str, report: Mapping[str, Any]) -> list[str]:
    """The parity manifests' own calibration and exclusions, beside our numbers."""
    lines: list[str] = []
    final = report["stages"].get("S6_final", {})
    per_sample = final.get("metrics", {}).get("all_atom_rmsd_angstrom")
    if per_sample:
        values = ", ".join(f"{v:.4f}" for v in per_sample)
        lines.append(f"  * S6 all-atom RMSD per sample (A): {values}")
    if model in NO_PARITY_MANIFEST:
        cli = final.get("metrics", {}).get("cli_route", {}).get("per_sample")
        if cli:
            values = ", ".join(f"{row['max_abs_dxyz_angstrom']:.4g}" for row in cli)
            lines.append(
                f"  * CLI route (run_alphafold.py vs foldjax.cli), max |dxyz| per "
                f"sample (A): {values}"
            )
        lines.append(
            "  * no CPU parity manifest: the reference is DeepMind's "
            "run_alphafold.py run on CPU beside FoldJAX, not a GPU capture"
        )
        return lines
    try:
        from tests.parity._manifest import load_manifest

        manifest = load_manifest(model)
    except Exception as error:  # noqa: BLE001 -- context only
        lines.append(f"  * parity manifest unavailable: {error}")
        return lines
    for entry in manifest.entries:
        excluded = list(entry.excluded_samples)
        lines.append(
            f"  * manifest tier {entry.tier} ({entry.tolerance_metric}): CPU "
            f"calibration {entry.cpu_residual:.4g}, tolerance "
            f"{entry.tolerance_value:.4g}"
            + (f", excluded samples {excluded}" if excluded else "")
        )
    return lines


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_table(results_dir: Path) -> str:
    rows = [
        "| model | stage | status | metric | value | condition | runtime (s) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    cases = []
    for path in sorted(results_dir.glob("*.json")):
        report = json.loads(path.read_text())
        capture = Path(report.get("capture", ""))
        cases.append(f"* {report['model']}: `{capture.parent.name}` (`{capture}`)")
        cases.extend(_manifest_context(report["model"], report))
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
                "| "
                + " | ".join(
                    _cell(cell)
                    for cell in (
                        report["model"],
                        stage,
                        record["status"],
                        metric,
                        value,
                        condition,
                        "" if runtime is None else runtime,
                    )
                )
                + " |"
            )
    return (
        TABLE_PREAMBLE
        + "\nCases:\n\n"
        + "\n".join(cases)
        + "\n\n"
        + "\n".join(rows)
        + "\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(MODELS))
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--stage-capture",
        type=Path,
        help="native re-capture with stages.npz for S3-S5 (ESMFold2, OpenDDE); "
        "defaults to STAGE_CAPTURES",
    )
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
    parser.add_argument(
        "--table", type=Path, help="render TABLE.md rows for a directory"
    )
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
    report = run_model(
        args.model,
        args.capture.resolve(),
        stages,
        None if args.stage_capture is None else args.stage_capture.resolve(),
    )
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
