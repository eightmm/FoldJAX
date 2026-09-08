"""Fail-closed comparisons of AF3 closure artifacts, not performance admission."""

import argparse
import hashlib
import json
import re
import shlex
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from bench.entity_parity import compare_entity_parity, compare_feature_dicts
from bench.precision_policy import coordinate_gate

TAPE_COVERAGE = {
    "initial": 1,
    "churn": 1000,
    "rotation": 1000,
    "translation": 1000,
    "padding_gumbel": 11,
}
SHARED_PROVENANCE = (
    "weights_sha256",
    "input_sha256",
    "cpp_sha256",
    "shared_generated_assets",
    "buckets",
    "attention",
    "kernel_selection",
    "kernel_deviation",
    "jax_version",
    "runtime_versions",
    "neutral_padding",
    "matmul_precision",
    "native_recycles",
    "steps",
    "samples",
    "xla_autotune_sha256",
    "separate_executable_cache",
)

RAW_HEADS = frozenset(
    {
        "predicted_lddt",
        "predicted_experimentally_resolved",
        "full_pde",
        "average_pde",
        "full_pae",
        "tmscore_adjusted_pae_global",
        "tmscore_adjusted_pae_interface",
        "distogram.contact_probs",
        "__identifier__",
        "diffusion_samples.mask",
    }
)


def compiler_policy(flags):
    """Compare inherited flags, excluding explicitly role-specific AOT paths."""
    controls = (
        "--xla_gpu_dump_autotune_results_to=",
        "--xla_gpu_load_autotune_results_from=",
        "--xla_gpu_require_complete_aot_autotune_results=",
    )
    # Preserve order: repeated flags may use last-value-wins semantics.
    return [token for token in shlex.split(flags) if not token.startswith(controls)]


def _xla_cache_entries(path):
    """Read canonical top-level text records, not braces inside HLO strings."""
    header, *entries = re.split(r"(?m)^results \{\n", path.read_text())
    if (
        header != "version: 3\n"
        or not entries
        or any(not entry.endswith("}\n") for entry in entries)
    ):
        raise ValueError("unrecognized XLA autotune text dump")
    return set(entries)


def _is_recorded_compiler_extension(roots, provenance):
    for parent, child in ((0, 1), (1, 0)):
        a, b = provenance[parent], provenance[child]
        if {a.get("mode"), b.get("mode")} != {"audit", "performance"}:
            continue
        paths = [root / "xla-autotune.textproto" for root in roots]
        digests = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
        if (
            a.get("xla_autotune_sha256") == digests[parent]
            and b.get("xla_autotune_load_sha256") in (digests[parent], digests[child])
            and b.get("xla_autotune_sha256") == digests[child]
            and _xla_cache_entries(paths[parent]) <= _xla_cache_entries(paths[child])
        ):
            return True
    return False


def _arrays(value):
    if isinstance(value, Mapping):
        return {key: np.asarray(array) for key, array in value.items()}
    with np.load(value, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def compare_confidence(left, right):
    """Compare flat mappings or NPZs with fixed tolerances and explicit NaNs."""
    left, right = _arrays(left), _arrays(right)
    report = {
        "missing_from_left": sorted(right.keys() - left.keys()),
        "missing_from_right": sorted(left.keys() - right.keys()),
        "leaves": {},
        "atol": 1e-4,
        "rtol": 1e-4,
    }
    for key in sorted(left.keys() & right.keys()):
        a, b = left[key], right[key]
        leaf = {"shape_equal": a.shape == b.shape, "dtype_equal": a.dtype == b.dtype}
        leaf["passed"] = False
        if leaf["shape_equal"] and leaf["dtype_equal"]:
            leaf["bitwise_equal"] = a.tobytes() == b.tobytes()
            if a.dtype.kind == "f":
                nonfinite_equal = all(
                    np.array_equal(fn(a), fn(b))
                    for fn in (np.isnan, np.isposinf, np.isneginf)
                )
                finite = np.isfinite(a) & np.isfinite(b)
                av, bv = a[finite].astype(np.float64), b[finite].astype(np.float64)
                error = np.abs(av - bv)
                leaf.update(
                    nonfinite_equal=nonfinite_equal,
                    max_absolute_error=float(error.max(initial=0)),
                    max_scaled_error=float(
                        (error / (1e-4 + 1e-4 * np.abs(bv))).max(initial=0)
                    ),
                    passed=bool(
                        nonfinite_equal and np.all(error <= 1e-4 + 1e-4 * np.abs(bv))
                    ),
                )
            elif a.dtype.kind in "biuSU":
                leaf["passed"] = bool(np.array_equal(a, b))
            else:
                leaf["error"] = f"unsupported dtype {a.dtype}"
        report["leaves"][key] = leaf
    report["passed"] = (
        bool(left)
        and not (report["missing_from_left"] or report["missing_from_right"])
        and all(leaf["passed"] for leaf in report["leaves"].values())
    )
    return report


def _json(path):
    def invalid(value):
        raise ValueError(f"nonstandard JSON constant {value}")

    return json.loads(path.read_text(), parse_constant=invalid)


def compare_first_warm(root):
    """Separate same-mode repeat diagnostic, including direct raw coordinates."""
    root = Path(root)
    report = {"passed": False, "scope": "uninstrumented first/warm raw repeat"}
    try:
        finished = _json(root / "finished.json")
        if finished.get("instrumented") is not False:
            raise ValueError("repeat requires an uninstrumented completed arm")
        first, warm = _arrays(root / "raw-first.npz"), _arrays(root / "raw.npz")
        for values in (first, warm):
            if not (RAW_HEADS | {"diffusion_samples.atom_positions"}).issubset(values):
                raise ValueError("repeat evidence is missing raw heads or coordinates")
        report["raw_repeat"] = compare_confidence(first, warm)
        report["passed"] = report["raw_repeat"]["passed"]
        if "warm_inference_repeats_seconds" in finished:
            times = finished["warm_inference_repeats_seconds"]
            equality = finished.get("warm_raw_bitwise_equal_to_first", [])
            report["all_warm_repeats"] = (
                bool(times)
                and len(times) == len(equality)
                and all(
                    type(t) in (float, int) and np.isfinite(t) and t > 0 for t in times
                )
                and all(value is True for value in equality)
            )
            report["passed"] &= report["all_warm_repeats"]
    except (OSError, ValueError, TypeError, KeyError) as error:
        report["error"] = str(error)
    return report


def compare_arms(left, right, *, require_tape=True):
    """Compare two completed arms; disabling tape is only an output bridge.

    Use the bridge for uninstrumented performance versus audited output, never
    as proof of independently matched actual RNG. Timing admission is separate.
    """
    left, right = Path(left), Path(right)
    report = {
        "passed": False,
        "scope": "AF3 provisional finite-case diagnostic; not all-model closure",
        "rdkit_rng_scope": "seed/conformer boundary only; internal PRNG unobserved",
        "checks": {},
    }
    checks = report["checks"]
    try:
        for root in (left, right):
            for name in ("finished.json", "provenance.json"):
                if not _json(root / name):
                    raise ValueError(f"empty evidence {root / name}")
        for name in (
            "input-metadata.json",
            "config.json",
            "effective-config.json",
            "parameters.json",
        ):
            a, b = _json(left / name), _json(right / name)
            checks[name] = bool(a) and a == b
            if (
                name in ("config.json", "effective-config.json")
                and isinstance(a, dict)
                and isinstance(b, dict)
            ):
                report.setdefault("config_differences", {})[name] = {
                    "left_only": {k: a[k] for k in sorted(a.keys() - b.keys())},
                    "right_only": {k: b[k] for k in sorted(b.keys() - a.keys())},
                    "changed": {
                        k: {"left": a[k], "right": b[k]}
                        for k in sorted(a.keys() & b.keys())
                        if a[k] != b[k]
                    },
                    "scope": "top-level explanation only; exact config gate unchanged",
                }
        provenance = [_json(root / "provenance.json") for root in (left, right)]
        checks["xla_cache_artifacts"] = all(
            hashlib.sha256((root / "xla-autotune.textproto").read_bytes()).hexdigest()
            == value.get("xla_autotune_sha256")
            for root, value in zip((left, right), provenance, strict=True)
        )
        checks["separate_executable_cache"] = all(
            value.get("separate_executable_cache") is True for value in provenance
        )
        checks["compiler_policy"] = all(
            "xla_flags" in value for value in provenance
        ) and compiler_policy(provenance[0]["xla_flags"]) == compiler_policy(
            provenance[1]["xla_flags"]
        )
        for name in SHARED_PROVENANCE:
            checks[f"provenance.{name}"] = (
                all(name in value for value in provenance)
                and provenance[0][name] == provenance[1][name]
            )
        if not require_tape and not checks["provenance.xla_autotune_sha256"]:
            checks["provenance.xla_autotune_sha256"] = _is_recorded_compiler_extension(
                (left, right), provenance
            )
            report["compiler_extension_bridge"] = checks[
                "provenance.xla_autotune_sha256"
            ]
        if any(value.get("fixed_kernel_control_only") for value in provenance):
            for name in (
                "fixed_kernel_control_only",
                "fixed_kernel_manifest_sha256",
                "fixed_kernel_result_sha256",
                "fixed_kernel_entries",
            ):
                checks[f"provenance.{name}"] = (
                    all(name in value for value in provenance)
                    and provenance[0][name] == provenance[1][name]
                )
        for name in ("input.npz", "identity.npz"):
            a, b = _arrays(left / name), _arrays(right / name)
            result = compare_feature_dicts(a, b)
            report[name] = result
            checks[name] = bool(a) and result["equal"]
        if require_tape:
            coverage = [_json(root / "tape-coverage.json") for root in (left, right)]
            checks["tape_coverage"] = all(
                value == TAPE_COVERAGE
                and all(type(count) is int for count in value.values())
                for value in coverage
            )
            preprocessing = [
                _json(root / "preprocessing-tape.json") for root in (left, right)
            ]
            checks["preprocessing_tape"] = preprocessing[0] == preprocessing[1] and all(
                isinstance(value, dict)
                and value.get("rdkit_internal_rng_observed") is False
                and all(
                    isinstance(value.get(name), list) and bool(value[name])
                    for name in ("numpy_draws", "rdkit_seed_and_conformer_boundary")
                )
                for value in preprocessing
            )
            tapes = [_json(root / "tape.json") for root in (left, right)]
            checks["tape"] = (
                all(isinstance(tape, dict) for tape in tapes)
                and bool(tapes[0])
                and tapes[0] == tapes[1]
                and all(type(count) is int and count > 0 for count in tapes[0].values())
                and sum(tapes[0].values()) == sum(TAPE_COVERAGE.values())
            )
        else:
            report["scope"] = "output-only audit/performance bridge; no RNG admission"
        a, b = [_arrays(root / "coordinate.npz") for root in (left, right)]
        if a.keys() != {"coordinate", "mask"} or b.keys() != a.keys():
            raise ValueError("coordinate archive requires coordinate and mask only")
        for value in (a, b):
            xyz, mask = value["coordinate"], value["mask"]
            if xyz.ndim != 3 or xyz.shape[0] != 5 or xyz.shape[-1] != 3:
                raise ValueError("coordinates require five samples of atom xyz")
            if mask.dtype != np.bool_ or mask.shape != xyz.shape[:2]:
                raise ValueError("gather mask requires boolean sample/atom shape")
        checks["gather_mask"] = np.array_equal(a["mask"], b["mask"])
        identities = [_arrays(root / "identity.npz") for root in (left, right)]
        fields = (
            "chain_id",
            "chain_type",
            "res_id",
            "res_name",
            "atom_name",
            "atom_element",
        )
        for identity, value in zip(identities, (a, b), strict=True):
            if not set(fields).issubset(identity):
                raise ValueError("incomplete atom identities")
            if any(
                identity[name].shape != (value["coordinate"].shape[1],)
                for name in fields
            ):
                raise ValueError("atom identities do not match coordinate cardinality")
        keys = [
            list(zip(*(identity[name].tolist() for name in fields), strict=True))
            for identity in identities
        ]
        parity = compare_entity_parity(
            a["coordinate"],
            b["coordinate"],
            keys[0],
            keys[1],
            identities[0]["chain_id"].tolist(),
            identities[1]["chain_id"].tolist(),
            a["mask"],
            b["mask"],
        )
        report["entity_parity"] = parity
        report["coordinate_gate"] = coordinate_gate(parity["entity_rmsd"])
        checks["coordinates"] = report["coordinate_gate"]["coordinate_gate_passed"]
        raw = [_arrays(root / "raw.npz") for root in (left, right)]
        for values in raw:
            if not RAW_HEADS.issubset(values):
                missing = sorted(RAW_HEADS - values.keys())
                raise ValueError(f"raw confidence evidence missing: {missing}")
            if "diffusion_samples.atom_positions" not in values:
                raise ValueError("raw coordinate evidence missing")
            del values["diffusion_samples.atom_positions"]
        checks["raw_masks"] = compare_feature_dicts(
            {key: value for key, value in raw[0].items() if "mask" in key},
            {key: value for key, value in raw[1].items() if "mask" in key},
        )["equal"]
        extracted = [_arrays(root / "confidence.npz") for root in (left, right)]
        for values in extracted:
            for sample in range(5):
                for field in (
                    "atom_plddt",
                    "numerical.full_pae",
                    "numerical.full_pde",
                    "metadata.ptm",
                    "metadata.iptm",
                    "metadata.ranking_score",
                ):
                    if f"{sample}.{field}" not in values:
                        raise ValueError(
                            f"extracted confidence evidence missing: {sample}.{field}"
                        )
        for name, values in (
            ("raw_confidence", raw),
            ("confidence", extracted),
        ):
            result = compare_confidence(*values)
            report[name] = result
            checks[name] = result["passed"]
        report["passed"] = all(checks.values())
    except (OSError, ValueError, TypeError, KeyError) as error:
        report["error"] = str(error)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--output-only", action="store_true")
    args = parser.parse_args()
    report = compare_arms(args.left, args.right, require_tape=not args.output_only)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": report["passed"], "scope": report["scope"]}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
