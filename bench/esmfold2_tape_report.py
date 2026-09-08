"""Hash-bound shared-feature ESMFold2 diagnostics; never model admission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench.boltz_amp_report import compare_arrays
from bench.entity_parity import compare_entity_parity, compare_feature_dicts
from bench.esmfold2_tape import (
    _npz,
    _sha256,
    _verify_reference_outputs,
    validate_saved_input_control,
)
from bench.precision_policy import coordinate_gate


def entity_inputs(features):
    atom = np.asarray(features["atom_attention_mask"])
    token = np.asarray(features["token_attention_mask"])
    mapping = np.asarray(features["atom_to_token"])
    if atom.ndim != 2 or atom.shape[0] != 1 or token.ndim != 2 or token.shape[0] != 1:
        raise ValueError("singleton native feature batch required")
    if not np.isin(atom, [0, 1]).all() or not np.isin(token, [0, 1]).all():
        raise ValueError("native masks must be binary")
    if mapping.shape != atom.shape or mapping.dtype.kind not in "iu":
        raise ValueError("native atom_to_token must be integer [batch, atoms]")
    valid = atom[0].astype(bool)
    indices = mapping[0, valid]
    if (
        not valid.any()
        or np.any(indices < 0)
        or np.any(indices >= token.shape[1])
        or not token[0, indices].all()
    ):
        raise ValueError("valid atom maps outside valid tokens")
    identities = []
    for name in ("entity_id", "mol_type", "asym_id"):
        array = np.asarray(features[name])
        if (
            array.shape != token.shape
            or array.dtype.kind not in "iu"
            or np.any(array[0, indices] < 0)
        ):
            raise ValueError(f"invalid native token identity {name}")
        identities.append(array[0, indices])
    labels = [
        f"entity={e}|mol_type={m}|asym={a}" for e, m, a in zip(*identities, strict=True)
    ]
    return valid, [f"native_atom_index={i}" for i in np.flatnonzero(valid)], labels


def compare_coordinates(left, right, features):
    valid, keys, labels = entity_inputs(features)
    if left.shape != (5, len(valid), 3) or right.shape != left.shape:
        raise ValueError("five original-order coordinate samples required")
    mask = np.ones((5, int(valid.sum())), bool)
    report = compare_entity_parity(
        left[:, valid], right[:, valid], keys, keys, labels, labels, mask, mask
    )
    report["coordinate_diagnostic"] = coordinate_gate(report["entity_rmsd"])
    return report


def _lm_artifact(root, document):
    schema = document.get("lm_schema")
    if schema is None:
        return None
    filename = schema["filename"]
    if Path(filename).name != filename or _sha256(root / filename) != schema["sha256"]:
        raise ValueError("LM artifact identity differs")
    arrays = _npz(root / filename)
    if set(arrays) != {"lm_hidden_states"}:
        raise ValueError("unexpected LM archive fields")
    value = arrays["lm_hidden_states"]
    original = schema["original_dtype"]
    expected_storage = "float32" if original == "bfloat16" else original
    if (
        original not in ("float16", "float32", "bfloat16")
        or str(value.dtype) != expected_storage
        or schema["storage_dtype"] != expected_storage
        or list(value.shape) != schema["shape"]
        or value.ndim != 4
        or not np.isfinite(value).all()
    ):
        raise ValueError("LM shape/dtype/value contract differs")
    if original == "bfloat16" and np.any(value.view(np.uint32) & 0xFFFF):
        raise ValueError("LM archive is not lossless BF16 storage")
    return value


def compare_lm(native, candidate, a, b):
    native_arm = a.get("lm_arm", "legacy_unrecorded_lm")
    candidate_arm = b.get("lm_arm", "legacy_unrecorded_lm")
    if native_arm not in (
        "native_independent_lm",
        "legacy_unrecorded_lm",
    ) or candidate_arm not in (
        "jax_independent_lm",
        "native_lm_interchange_downstream_core_only",
        "legacy_unrecorded_lm",
    ):
        raise ValueError("unknown LM execution arm")
    left, right = _lm_artifact(native, a), _lm_artifact(candidate, b)
    if (
        native_arm != "legacy_unrecorded_lm"
        and left is None
        or candidate_arm != "legacy_unrecorded_lm"
        and right is None
    ):
        raise ValueError("declared LM execution arm lacks its artifact")
    interchange = candidate_arm == "native_lm_interchange_downstream_core_only"
    if interchange:
        if (
            left is None
            or right is None
            or a["lm_schema"]["original_dtype"] != b["lm_schema"]["original_dtype"]
            or left.shape != right.shape
            or left.dtype != right.dtype
            or left.tobytes() != right.tobytes()
        ):
            raise ValueError(
                "native-LM interchange does not match its bound native artifact"
            )
    result = {
        "native_arm": native_arm,
        "candidate_arm": candidate_arm,
        "downstream_only": interchange,
        "independent_lm_comparison": native_arm == "native_independent_lm"
        and candidate_arm == "jax_independent_lm"
        and left is not None
        and right is not None,
        "native_schema": a.get("lm_schema"),
        "candidate_schema": b.get("lm_schema"),
        "statistics": None,
        "scope": "LM diagnostic only; not a coordinate or confidence admission gate",
    }
    if left is not None and right is not None:
        result["statistics"] = compare_arrays(
            {"lm_hidden_states": left},
            {"lm_hidden_states": right},
            {"lm_hidden_states": {"dtype": a["lm_schema"]["original_dtype"]}},
            {"lm_hidden_states": {"dtype": b["lm_schema"]["original_dtype"]}},
        )
    return result


def validate_shim_control(candidate, shim, *, downstream_only, expected_shape):
    filename = shim["filename"]
    if (
        not downstream_only
        or Path(filename).name != filename
        or _sha256(candidate / filename) != shim["sha256"]
    ):
        raise ValueError("native shim control artifact identity differs")
    values = _npz(candidate / filename)
    if (
        set(values) != {"pair"}
        or values["pair"].shape != tuple(expected_shape)
        or list(expected_shape) != shim["shape"]
        or values["pair"].dtype != np.float32
        or shim["dtype"] != "float32"
        or not np.isfinite(values["pair"]).all()
    ):
        raise ValueError("native shim control pair contract differs")


def build_report(native, candidate, features_path):
    paths = [native / "metadata.json", candidate / "metadata.json"]
    hashes = [_sha256(path) for path in paths]
    a, b = [json.loads(path.read_text()) for path in paths]
    for root, document in ((native, a), (candidate, b)):
        _verify_reference_outputs(root, document)
        binding = document.get("binding")
        if (
            not binding
            or not binding.get("source")
            or not binding.get("checkpoint")
            or not binding.get("runner")
        ):
            raise ValueError("missing source/checkpoint/runner identity")
    if b["binding"].get("reference_manifest_sha256") != hashes[0]:
        raise ValueError("candidate belongs to another native capture")
    if a["binding"]["checkpoint"] != b["binding"]["checkpoint"]:
        raise ValueError("checkpoint identities differ")
    if not a["input_sha256"] == b["input_sha256"] == _sha256(features_path):
        raise ValueError("shared native feature identity differs")
    if not a["tape_sha256"] == b["tape_sha256"] == _sha256(native / "tape.npz"):
        raise ValueError("tape identity differs")
    features = _npz(features_path)
    feature_check = compare_feature_dicts(features, _npz(native / "features.npz"))
    if not feature_check["equal"]:
        raise ValueError("saved shared features differ from bound original input")
    lm = compare_lm(native, candidate, a, b)
    shim = b.get("native_shim_control")
    if shim is not None:
        shape = (
            *features["token_attention_mask"].shape,
            features["token_attention_mask"].shape[1],
            a["config"]["d_pair"],
        )
        validate_shim_control(
            candidate, shim, downstream_only=lm["downstream_only"], expected_shape=shape
        )
    embedding = b.get("native_input_control")
    if embedding is not None:
        if not shim or not lm["downstream_only"]:
            raise ValueError(
                "native input control requires native LM and shim controls"
            )
        validate_saved_input_control(
            candidate,
            embedding,
            (
                *features["token_attention_mask"].shape,
                a["config"]["inputs"]["d_inputs"],
            ),
        )
    left = _npz(native / "upstream_confidence.npz")
    right = _npz(candidate / "jax_confidence.npz")
    confidence = compare_arrays(left, right)
    per_sample = {
        name: [
            compare_arrays({name: left[name][i]}, {name: right[name][i]})
            for i in range(5)
        ]
        for name in left.keys() & right.keys()
        if left[name].ndim > 0
        and left[name].shape[0] == 5
        and right[name].shape == left[name].shape
    }
    report = {
        "schema_version": 1,
        "full_model_admission": None,
        "language_model": lm,
        "native_shim_control": shim,
        "native_input_control": embedding,
        "scope": (
            (
                "shared-native-feature/native-LM downstream-only diagnostic; "
                if lm["downstream_only"]
                else "shared-native-feature core diagnostic; "
            )
            + ("native shim pair substituted; " if shim else "")
            + ("native input embedding substituted; " if embedding else "")
            + "independent preprocessing not proved; no crystal"
        ),
        "alignment": (
            "one whole-system Kabsch per original-order sample; "
            "entity measurement without refit"
        ),
        "manifest_sha256": {"native": hashes[0], "candidate": hashes[1]},
        "shared_feature_equality": feature_check,
        "coordinates": compare_coordinates(
            _npz(native / "upstream_coords.npz")["coords"],
            _npz(candidate / "jax_coords.npz")["coords"],
            features,
        ),
        "confidence": confidence,
        "confidence_per_sample": per_sample,
        "missing_candidate_raw_heads": sorted(left.keys() - right.keys()),
        "raw_head_retention_complete": left.keys() <= right.keys(),
    }
    for root, document in ((native, a), (candidate, b)):
        _verify_reference_outputs(root, document)
        _lm_artifact(root, document)
    if (
        hashes != [_sha256(path) for path in paths]
        or _sha256(features_path) != a["input_sha256"]
        or _sha256(native / "tape.npz") != a["tape_sha256"]
        or (
            shim is not None and _sha256(candidate / shim["filename"]) != shim["sha256"]
        )
    ):
        raise ValueError("report input changed while measuring")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native", "candidate", "features", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = build_report(args.native, args.candidate, args.features)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
