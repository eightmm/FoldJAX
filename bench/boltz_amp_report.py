"""Portable, sample-index-paired Boltz native/AMP diagnostic reports.

Strict confidence tolerances and the 0.05-Angstrom coordinate diagnostic are
retained, not calibrated or relaxed here. Captured-feature replay is core-only
evidence; successful reports are not model admission or preprocessing proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from bench.af3_closure_capture import sha
from bench.boltz_closure_capture import save_new
from bench.boltz_foldjax_capture import array_identity
from bench.entity_parity import compare_entity_parity, compare_feature_dicts
from bench.precision_policy import coordinate_gate

_STAGES = (
    "input_embedder",
    "rel_pos",
    *(
        f"cycle-{cycle:02d}/{stage}"
        for cycle in (0, 3)
        for stage in ("msa_module", "pairformer_module")
    ),
)
_RAW_EXCLUDE = {"s", "z", "sample_atom_coords", "single", "pair", "single_inputs"}
_PUBLIC_EXCLUDE = {"s", "z", "coords", "masks", "token_masks", "exception"}


def _json(path):
    return json.loads(path.read_text())


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _relative(root, name):
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes capture root")
    return path


def _arrays(path, names=None):
    with np.load(path, allow_pickle=False) as archive:
        return {
            name: archive[name]
            for name in archive.files
            if names is None or name in names
        }


def _bundle(root, label):
    arrays = _arrays(_relative(root, f"{label}.npz"))
    metadata = _json(_relative(root, f"{label}.tree.json"))
    if set(arrays) != {
        name for name, value in metadata.items() if value.get("kind") != "none"
    }:
        raise ValueError(f"output archive and metadata leaf sets differ: {label}")
    if any(
        list(value.shape) != metadata[name]["shape"] for name, value in arrays.items()
    ):
        raise ValueError(f"output archive and metadata shapes differ: {label}")
    return arrays, metadata


def _dtype(metadata, name, value):
    info = metadata.get(name, {})
    return str(info.get("native_dtype", info.get("dtype", value.dtype))).removeprefix(
        "torch."
    )


def compare_arrays(left, right, left_metadata=None, right_metadata=None):
    """Report each leaf, preserving undefined positions and exact discrete fields."""
    left_metadata, right_metadata = left_metadata or {}, right_metadata or {}
    leaves = {}
    for name in sorted(left.keys() & right.keys()):
        a, b = np.asarray(left[name]), np.asarray(right[name])
        if a.dtype.kind not in "fbiuUS" or b.dtype.kind not in "fbiuUS":
            raise ValueError(f"unsupported numeric/output dtype at {name}")
        item = {
            "native_shape": list(a.shape),
            "candidate_shape": list(b.shape),
            "native_storage_dtype": str(a.dtype),
            "candidate_storage_dtype": str(b.dtype),
            "native_original_dtype": _dtype(left_metadata, name, a),
            "candidate_original_dtype": _dtype(right_metadata, name, b),
            "strict_pass": False,
        }
        item["original_dtype_equal"] = (
            item["native_original_dtype"] == item["candidate_original_dtype"]
        )
        item["shape_equal"] = a.shape == b.shape
        if item["shape_equal"]:
            item["storage_bytes_equal"] = (
                a.dtype == b.dtype and a.tobytes() == b.tobytes()
            )
            if a.dtype.kind != "f" or b.dtype.kind != "f":
                item["strict_pass"] = a.dtype == b.dtype and bool(np.array_equal(a, b))
                item["comparison"] = "exact discrete values and storage dtype"
            else:
                same_undefined = all(
                    np.array_equal(fn(a), fn(b))
                    for fn in (np.isnan, np.isposinf, np.isneginf)
                )
                valid = np.isfinite(a) & np.isfinite(b)
                x, y = a[valid].astype(np.float64), b[valid].astype(np.float64)
                with np.errstate(over="ignore", invalid="ignore"):
                    delta = np.abs(y - x)
                overflow = not np.isfinite(delta).all()
                maximum = float(delta.max(initial=0.0)) if not overflow else None
                allowance = 1e-4 + 1e-4 * np.abs(x)
                item.update(
                    {
                        "undefined_positions_equal": same_undefined,
                        "finite_entries": int(x.size),
                        "max_abs": maximum,
                        "rmse": None
                        if overflow
                        else (
                            maximum * float(np.sqrt(np.mean((delta / maximum) ** 2)))
                            if maximum
                            else 0.0
                        ),
                        "numeric_overflow": overflow,
                        "strict_failures": int(np.count_nonzero(delta > allowance)),
                        "strict_pass": same_undefined
                        and not overflow
                        and bool(np.all(delta <= allowance)),
                    }
                )
        leaves[name] = item
    missing, extra = (
        sorted(left.keys() - right.keys()),
        sorted(right.keys() - left.keys()),
    )
    return {
        "missing_from_candidate": missing,
        "extra_in_candidate": extra,
        "leaves": leaves,
        "strict_atol": 1e-4,
        "strict_rtol": 1e-4,
        "strict_pass": bool(left)
        and not missing
        and not extra
        and all(x["strict_pass"] for x in leaves.values()),
        "original_dtypes_equal": not missing
        and not extra
        and all(x["original_dtype_equal"] for x in leaves.values()),
    }


def _archive_parity(left, right):
    a, b = _arrays(left), _arrays(right)
    report = compare_feature_dicts(a, b)
    report["all_bytes_equal"] = report["equal"] and all(
        a[name].tobytes() == b[name].tobytes() for name in a
    )
    return report


def verify_capture(root, *, foldjax):
    complete = _json(root / "capture-complete.json")
    if complete.get("passed") is not True:
        raise ValueError("capture did not complete")
    bindings = complete["stage_artifacts"] if foldjax else complete["artifacts"]
    if not foldjax and not {
        "forward-output",
        "predict-step-output",
        "preprocessing-tape",
    }.issubset(bindings):
        raise ValueError(
            "native capture does not bind all required output/tape artifacts"
        )
    checked = {}
    for label, hashes in bindings.items():
        for suffix, key in ((".npz", "arrays_sha256"), (".tree.json", "tree_sha256")):
            name = label + suffix
            actual = sha(_relative(root, name))
            if actual != hashes[key]:
                raise ValueError(f"capture artifact hash mismatch: {name}")
            checked[name] = actual
    if foldjax:
        output_bindings = [
            ("prediction.npz", "prediction_sha256"),
            ("prediction.tree.json", "prediction_tree_sha256"),
        ]
        if "effective_options_sha256" in complete:
            output_bindings.append(
                ("effective-options.json", "effective_options_sha256")
            )
        for name, key in output_bindings:
            actual = sha(root / name)
            if actual != complete[key]:
                raise ValueError(f"capture output hash mismatch: {name}")
            checked[name] = actual
    return complete, checked


def verify_reference(native, candidate, *, foldjax):
    a, b = _json(native / "provenance.json"), _json(candidate / "provenance.json")
    if foldjax:
        required = {
            "capture-complete.json",
            "provenance.json",
            "features.npz",
            "tape.npz",
            "tape.json",
            "effective-model-settings.json",
            "forward-output.tree.json",
        }
        if not required.issubset(b["native_artifacts"]):
            raise ValueError(
                "candidate does not bind required native reference artifacts"
            )
        for name, expected in b["native_artifacts"].items():
            if name not in required or sha(native / name) != expected:
                raise ValueError(f"candidate native reference hash mismatch: {name}")
        features = _arrays(native / "features.npz")
        for name, value in features.items():
            if (
                value.dtype == np.int64
                and value.size
                and (
                    value.min() < np.iinfo(np.int32).min
                    or value.max() > np.iinfo(np.int32).max
                )
            ):
                raise ValueError(
                    f"native feature integer narrowing would overflow: {name}"
                )
        converted = {
            name: value.astype(np.int32)
            if value.dtype == np.int64
            else value.astype(np.float32)
            if value.dtype == np.float64
            else value
            for name, value in features.items()
        }
        options = _json(candidate / "effective-options.json")
        actual = options["features"]
        if actual != {name: array_identity(value) for name, value in converted.items()}:
            raise ValueError(
                "candidate feature-entry hashes differ from declared "
                "native-feature cast"
            )
        if options["sampler_tape"] != {
            name: array_identity(value)
            for name, value in _arrays(native / "tape.npz").items()
        }:
            raise ValueError("candidate sampler-entry hashes differ from native tape")
        return {
            "native_artifact_hashes_verified": True,
            "feature_entry_hashes_verified": True,
            "sampler_entry_hashes_verified": True,
            "feature_mapping": (
                "captured int64 -> int32 and float64 -> float32; otherwise unchanged"
            ),
            "independent_preprocessing": False,
        }
    pairs = {}
    for name in ("features.npz", "tape.npz", "preprocessing-tape.npz"):
        pairs[name] = _archive_parity(native / name, candidate / name)
        if not pairs[name]["all_bytes_equal"]:
            raise ValueError(f"uncontrolled native repeat: {name} differs")
    for name in (
        "input_sha256",
        "checkpoint_sha256",
        "upstream_python_source",
        "wrapper_sha256",
        "legacy_capture_sha256",
        "helper_source_sha256",
        "torch_version",
        "cuda_version",
    ):
        if a[name] != b[name]:
            raise ValueError(f"uncontrolled native repeat provenance: {name}")
    if _json(native / "effective-model-settings.json") != _json(
        candidate / "effective-model-settings.json"
    ):
        raise ValueError("uncontrolled native repeat: effective settings differ")
    return {
        "controlled_native_repeat": True,
        "exact_input_tape_archives": pairs,
        "not_native_variability_calibration": True,
    }


def entity_inputs(features):
    """Map each valid atom to one native token and a distinct entity instance."""
    atom_mask = np.asarray(features["atom_pad_mask"])
    token_mask = np.asarray(features["token_pad_mask"])
    mapping = np.asarray(features["atom_to_token"])
    if (
        atom_mask.ndim != 2
        or atom_mask.shape[0] != 1
        or token_mask.ndim != 2
        or token_mask.shape[0] != 1
    ):
        raise ValueError("native masks require a singleton feature batch")
    if not np.isin(atom_mask, [0, 1]).all() or not np.isin(token_mask, [0, 1]).all():
        raise ValueError("native masks must be binary")
    if mapping.shape != (1, atom_mask.shape[1], token_mask.shape[1]):
        raise ValueError("atom_to_token shape disagrees with native masks")
    valid = atom_mask[0].astype(bool)
    selected = mapping[0, valid]
    if (
        not valid.any()
        or not np.isin(selected, [0, 1]).all()
        or not (selected.sum(-1) == 1).all()
    ):
        raise ValueError("each valid atom requires exactly one one-hot native token")
    indices = selected.argmax(-1)
    if not token_mask[0, indices].all():
        raise ValueError("valid atom maps to a padded native token")
    values = {}
    for name in ("entity_id", "mol_type", "asym_id"):
        array = np.asarray(features[name])
        if array.shape != token_mask.shape or array.dtype.kind not in "iu":
            raise ValueError(f"native token identity {name} has invalid shape or dtype")
        if (array[0, indices] < 0).any():
            raise ValueError(f"native token identity {name} must be nonnegative")
        values[name] = array[0, indices]
    labels = [
        f"entity={e}|mol_type={m}|asym={c}"
        for e, m, c in zip(
            values["entity_id"], values["mol_type"], values["asym_id"], strict=True
        )
    ]
    keys = [f"native_atom_index={i}" for i in np.flatnonzero(valid)]
    return valid, keys, labels


def compare_coordinates(left, right, features):
    valid, keys, labels = entity_inputs(features)
    a, b = np.asarray(left), np.asarray(right)
    expected = (5, len(valid), 3)
    if a.shape != expected or b.shape != expected:
        raise ValueError(
            f"coordinates require five original-index samples of shape {expected}"
        )
    if any(
        value.dtype.kind not in "fiu" or not np.isfinite(value).all()
        for value in (a, b)
    ):
        raise ValueError("coordinates must be finite real numeric arrays")
    mask = np.ones((5, int(valid.sum())), dtype=bool)
    report = compare_entity_parity(
        a[:, valid], b[:, valid], keys, keys, labels, labels, mask, mask
    )
    report["coordinate_diagnostic"] = coordinate_gate(report["entity_rmsd"])
    report["alignment"] = (
        "one proper unweighted whole-system Kabsch per sample; no entity refit"
    )
    return report


def build_report(native, candidate):
    candidate_provenance = _json(candidate / "provenance.json")
    foldjax = candidate_provenance.get("arm") == "foldjax-core-only"
    _, native_bindings = verify_capture(native, foldjax=False)
    complete, candidate_bindings = verify_capture(candidate, foldjax=foldjax)
    reference = verify_reference(native, candidate, foldjax=foldjax)
    trunk_only = foldjax and complete["trunk_only"]
    report = {
        "schema_version": 1,
        "not_parity_admission": True,
        "candidate_kind": "foldjax" if foldjax else "native-repeat",
        "trunk_only": trunk_only,
        "reference_control": reference,
        "source_ids": {
            "native_provenance_sha256": sha(native / "provenance.json"),
            "candidate_provenance_sha256": sha(candidate / "provenance.json"),
            "native_source_id": _digest(
                _json(native / "provenance.json")["upstream_python_source"]
            ),
            "candidate_source_id": _digest(
                candidate_provenance.get(
                    "source_files", candidate_provenance.get("upstream_python_source")
                )
            ),
        },
        "verified_artifacts": {
            "native": native_bindings,
            "candidate": candidate_bindings,
        },
        "scope": (
            "captured-native-feature core diagnostic; not independent preprocessing "
            "or full admission"
        ),
        "tape_scope": (
            "entry/archive identity, not independent per-step device consumer proof"
        ),
        "acceptance": (
            "historical strict diagnostics only; no new allowance "
            "or native-repeat calibration"
        ),
        "sample_order": "original sampler index 0..4; no ranking permutation",
        "stage_comparisons": {},
    }
    if foldjax:
        options = _json(candidate / "effective-options.json")
        report["candidate_compilation"] = {
            "compiler_options": options.get("compiler_options"),
            "confidence_chain_ids": options.get("confidence_chain_ids"),
            "effective_options_hash_bound": "effective_options_sha256"
            in _json(candidate / "capture-complete.json"),
        }
    for stage in _STAGES:
        label = f"trunk-boundaries/{stage}"
        missing = [
            side
            for side, root in (("native", native), ("candidate", candidate))
            if not (root / f"{label}.npz").exists()
            or not (root / f"{label}.tree.json").exists()
        ]
        if missing:
            report["stage_comparisons"][stage] = {"missing_stage_from": missing}
        else:
            a, am = _bundle(native, label)
            b, bm = _bundle(candidate, label)
            report["stage_comparisons"][stage] = compare_arrays(a, b, am, bm)
    native_output, native_metadata = _bundle(native, "forward-output")
    candidate_output, candidate_metadata = _bundle(
        candidate, "prediction" if foldjax else "forward-output"
    )
    report["trunk_representations"] = compare_arrays(
        {name: native_output[name] for name in ("s", "z")},
        {
            name: candidate_output[field]
            for name, field in (
                ("s", "single" if foldjax else "s"),
                ("z", "pair" if foldjax else "z"),
            )
            if field in candidate_output
        },
        native_metadata,
        {
            name: candidate_metadata.get(field, {})
            for name, field in (
                ("s", "single" if foldjax else "s"),
                ("z", "pair" if foldjax else "z"),
            )
        },
    )
    if trunk_only:
        report["not_evaluated"] = ["coordinates", "raw confidence", "public confidence"]
        return report
    report["coordinates"] = compare_coordinates(
        native_output["sample_atom_coords"],
        candidate_output["sample_atom_coords"],
        _arrays(native / "features.npz"),
    )
    report["raw_confidence"] = compare_arrays(
        {
            name: value
            for name, value in native_output.items()
            if name not in _RAW_EXCLUDE
        },
        {
            name: value
            for name, value in candidate_output.items()
            if name not in _RAW_EXCLUDE
        },
        native_metadata,
        candidate_metadata,
    )
    public, public_meta = _bundle(native, "predict-step-output")
    public = {
        name: value for name, value in public.items() if name not in _PUBLIC_EXCLUDE
    }
    if foldjax:
        candidate_public = {
            name: candidate_output[name] for name in public if name in candidate_output
        }
        candidate_public_meta = candidate_metadata
    else:
        candidate_public, candidate_public_meta = _bundle(
            candidate, "predict-step-output"
        )
        candidate_public = {
            name: value
            for name, value in candidate_public.items()
            if name not in _PUBLIC_EXCLUDE
        }
    report["public_confidence"] = compare_arrays(
        public, candidate_public, public_meta, candidate_public_meta
    )
    report["native_nonarray_forward_fields"] = sorted(
        name for name, value in native_metadata.items() if value.get("kind") == "none"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite report {args.out.name}")
    report = build_report(
        args.native.resolve(strict=True), args.candidate.resolve(strict=True)
    )
    save_new(args.out, report)


if __name__ == "__main__":
    main()
