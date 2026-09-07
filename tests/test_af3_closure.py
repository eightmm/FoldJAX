import hashlib
import json

import numpy as np
import pytest

from bench.af3_closure import (
    RAW_HEADS,
    SHARED_PROVENANCE,
    TAPE_COVERAGE,
    compare_arms,
    compare_confidence,
)


def test_confidence_nonfinite_and_exact_types():
    values = {
        "f": np.array([np.nan, np.inf, -np.inf, 1.0]),
        "b": np.array(True),
        "s": np.array(b"id"),
        "i": np.array(3),
    }
    assert compare_confidence(values, values)["passed"]
    candidate = dict(values, f=np.array([np.nan, np.inf, -np.inf, 1.0001]))
    assert compare_confidence(values, candidate)["passed"]
    assert not compare_confidence({}, {})["passed"]


def test_first_warm_repeat_requires_completed_ordinary_outputs(tmp_path):
    from bench.af3_closure import compare_first_warm

    root = tmp_path / "arm"
    _arm(root)
    assert not compare_first_warm(root)["passed"]
    (root / "finished.json").write_text(json.dumps({"instrumented": False}))
    assert not compare_first_warm(root)["passed"]
    with np.load(root / "raw.npz") as archive:
        first = dict(archive)
    np.savez(root / "raw-first.npz", **first)
    assert compare_first_warm(root)["passed"]
    (root / "finished.json").write_text(
        json.dumps(
            {
                "instrumented": False,
                "warm_inference_repeats_seconds": [1.0, 1.1],
                "warm_raw_bitwise_equal_to_first": [True, False],
            }
        )
    )
    assert not compare_first_warm(root)["passed"]
    (root / "finished.json").write_text(json.dumps({"instrumented": False}))
    first["full_pae"] = first["full_pae"] + 0.02
    np.savez(root / "raw-first.npz", **first)
    assert not compare_first_warm(root)["passed"]


@pytest.mark.parametrize(
    "value",
    [
        np.array([0.0, np.inf, -np.inf, 1.0]),
        np.array([np.nan, -np.inf, -np.inf, 1.0]),
        np.array([np.nan, np.inf, -np.inf, 1.01]),
        np.array([1.0]),
        np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32),
    ],
)
def test_confidence_rejects_errors(value):
    assert not compare_confidence(
        {"f": np.array([np.nan, np.inf, -np.inf, 1.0])}, {"f": value}
    )["passed"]


@pytest.mark.parametrize(
    "left,right",
    [
        ({"a": np.array(1)}, {"b": np.array(1)}),
        ({"a": np.array(True)}, {"a": np.array(False)}),
        ({"a": np.array("A")}, {"a": np.array("B")}),
        ({"a": np.array(1)}, {"a": np.array(2)}),
        ({"a": np.array(object())}, {"a": np.array(object())}),
    ],
)
def test_confidence_exact_contract(left, right):
    assert not compare_confidence(left, right)["passed"]


def _arm(path):
    path.mkdir()
    xla_cache = 'version: 3\nresults {\n  hlo: "graph"\n  version: 50\n}\n'
    (path / "xla-autotune.textproto").write_text(xla_cache)
    for name in (
        "finished",
        "provenance",
        "input-metadata",
        "config",
        "effective-config",
        "parameters",
    ):
        (path / f"{name}.json").write_text(json.dumps({"ok": True}))
    (path / "tape.json").write_text(json.dumps({"draw": sum(TAPE_COVERAGE.values())}))
    (path / "tape-coverage.json").write_text(json.dumps(TAPE_COVERAGE))
    (path / "preprocessing-tape.json").write_text(
        json.dumps(
            {
                "numpy_draws": ["draw1", "draw2"],
                "rdkit_seed_and_conformer_boundary": ["conformer"],
                "rdkit_internal_rng_observed": False,
            }
        )
    )
    (path / "provenance.json").write_text(
        json.dumps(
            {
                **dict.fromkeys(SHARED_PROVENANCE, "same"),
                "separate_executable_cache": True,
                "xla_flags": "",
                "xla_autotune_sha256": hashlib.sha256(xla_cache.encode()).hexdigest(),
            }
        )
    )
    np.savez(path / "input.npz", x=np.array([1.0]))
    np.savez(
        path / "identity.npz",
        chain_id=np.array(["A", "A", "B", "B"]),
        chain_type=np.array(["protein"] * 4),
        res_id=np.arange(4),
        res_name=np.array(["ALA"] * 4),
        atom_name=np.array(["CA"] * 4),
        atom_element=np.array(["C"] * 4),
    )
    xyz = np.broadcast_to(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        (5, 4, 3),
    )
    np.savez(path / "coordinate.npz", coordinate=xyz, mask=np.ones((5, 4), bool))
    np.savez(
        path / "raw.npz",
        **{
            **{name: np.ones((5, 4)) for name in RAW_HEADS},
            "diffusion_samples.atom_positions": xyz,
            "predicted_lddt": np.ones((5, 4)),
            "__identifier__": np.array(b"test"),
        },
    )
    np.savez(
        path / "confidence.npz",
        **{
            f"{sample}.{field}": np.ones(1)
            for sample in range(5)
            for field in (
                "atom_plddt",
                "numerical.full_pae",
                "numerical.full_pde",
                "metadata.ptm",
                "metadata.iptm",
                "metadata.ranking_score",
            )
        },
    )


def test_arm_success_and_missing_evidence(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    _arm(left)
    _arm(right)
    assert compare_arms(left, right)["passed"]
    (right / "finished.json").unlink()
    assert not compare_arms(left, right)["passed"]


@pytest.mark.parametrize(
    "failure",
    [
        "tape",
        "config",
        "confidence",
        "mask",
        "samples",
        "coordinate",
        "identity",
        "input",
        "parameters",
    ],
)
def test_arm_gates(tmp_path, failure):
    left, right = tmp_path / "left", tmp_path / "right"
    _arm(left)
    _arm(right)
    if failure in ("tape", "config", "parameters"):
        (right / f"{failure}.json").write_text(json.dumps({"different": 2}))
    elif failure == "confidence":
        np.savez(right / "confidence.npz", score=np.zeros(5))
    elif failure == "identity":
        np.savez(right / "identity.npz", chain_id=np.array(["A"]))
    elif failure == "input":
        np.savez(right / "input.npz", x=np.array([2.0]))
    else:
        with np.load(right / "coordinate.npz") as archive:
            xyz, mask = archive["coordinate"].copy(), archive["mask"].copy()
        if failure == "mask":
            mask[0, 0] = False
        elif failure == "samples":
            xyz, mask = xyz[:1], mask[:1]
        else:
            xyz[4, 3, 2] += 3
        np.savez(right / "coordinate.npz", coordinate=xyz, mask=mask)
    assert not compare_arms(left, right)["passed"]


def test_output_bridge_is_not_tape_admission(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    _arm(left)
    _arm(right)
    (right / "tape.json").write_text("{}")
    assert not compare_arms(left, right)["passed"]
    bridge = compare_arms(left, right, require_tape=False)
    assert bridge["passed"]
    assert "no RNG admission" in bridge["scope"]


@pytest.mark.parametrize(
    "failure", ["raw_missing", "raw_mask", "empty_tape", "missing_both_heads"]
)
def test_incomplete_or_float_mask_evidence_fails(tmp_path, failure):
    left, right = tmp_path / "left", tmp_path / "right"
    _arm(left)
    _arm(right)
    if failure == "empty_tape":
        for root in (left, right):
            (root / "tape.json").write_text("{}")
    else:
        for root in (left, right) if failure == "missing_both_heads" else (right,):
            with np.load(root / "raw.npz") as archive:
                values = dict(archive)
            if failure == "raw_mask":
                values["diffusion_samples.mask"][0, 0] += 1e-5
            else:
                del values["full_pae"]
            np.savez(root / "raw.npz", **values)
    assert not compare_arms(left, right)["passed"]


@pytest.mark.parametrize(
    "failure",
    [
        "coverage",
        "coverage_bool",
        "preprocess_empty",
        "preprocess_order",
        "internal_rng",
        "provenance_missing",
        "provenance_different",
    ],
)
def test_tape_and_shared_provenance_prerequisites(tmp_path, failure):
    left, right = tmp_path / "left", tmp_path / "right"
    _arm(left)
    _arm(right)
    if failure.startswith("coverage"):
        path = right / "tape-coverage.json"
        value = json.loads(path.read_text())
        value["initial"] = True if failure == "coverage_bool" else 0
    elif failure.startswith("provenance"):
        path = right / "provenance.json"
        value = json.loads(path.read_text())
        if failure.endswith("missing"):
            del value["weights_sha256"]
        else:
            value["weights_sha256"] = "other"
    else:
        path = right / "preprocessing-tape.json"
        value = json.loads(path.read_text())
        if failure == "internal_rng":
            value["rdkit_internal_rng_observed"] = True
        else:
            value["numpy_draws"] = (
                [] if failure.endswith("empty") else ["draw2", "draw1"]
            )
    path.write_text(json.dumps(value))
    assert not compare_arms(left, right)["passed"]


def test_independent_source_hashes_may_differ(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    for root in (left, right):
        _arm(root)
        path = root / "provenance.json"
        value = json.loads(path.read_text())
        value["source_files"] = {"source.py": root.name}
        path.write_text(json.dumps(value))
    assert compare_arms(left, right)["passed"]


def test_stale_compiler_digest_does_not_prove_matched_decisions(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    _arm(left)
    _arm(right)
    (right / "xla-autotune.textproto").write_text("changed")
    assert not compare_arms(left, right)["passed"]


def test_fixed_kernel_profiles_cannot_silently_differ(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    for root in (left, right):
        _arm(root)
        path = root / "provenance.json"
        value = json.loads(path.read_text())
        value.update(
            fixed_kernel_control_only=True,
            fixed_kernel_manifest_sha256="same",
            fixed_kernel_result_sha256="same",
            fixed_kernel_entries=19,
        )
        path.write_text(json.dumps(value))
    assert compare_arms(left, right)["passed"]
    value["fixed_kernel_result_sha256"] = "other"
    path.write_text(json.dumps(value))
    assert not compare_arms(left, right)["passed"]


def test_compiler_policy_ignores_only_recorded_aot_control_flags():
    from bench.af3_closure import compiler_policy

    assert compiler_policy(
        "--xla_gpu_dump_autotune_results_to=/a --xla_gpu_load_autotune_results_from=/b "
        "--xla_gpu_require_complete_aot_autotune_results=true --other=true"
    ) == ["--other=true"]
    assert compiler_policy("--other=true") != compiler_policy("--other=false")
    assert compiler_policy("--other=true --other=false") != compiler_policy(
        "--other=false --other=true"
    )


@pytest.mark.parametrize("mutate_parent_entry", [False, True])
def test_output_bridge_requires_preserved_compiler_decisions(
    tmp_path, mutate_parent_entry
):
    left, right = tmp_path / "left", tmp_path / "right"
    _arm(left)
    _arm(right)
    parent = 'version: 3\nresults {\n  hlo: "graph"\n  version: 50\n}\n'
    child = parent + 'results {\n  hlo: "extra"\n  version: 50\n}\n'
    if mutate_parent_entry:
        child = child.replace('"graph"', '"changed"')

    def digest(value):
        return hashlib.sha256(value.encode()).hexdigest()

    for root, mode, data in ((left, "audit", parent), (right, "performance", child)):
        (root / "xla-autotune.textproto").write_text(data)
        path = root / "provenance.json"
        value = json.loads(path.read_text())
        value.update(
            mode=mode,
            xla_autotune_sha256=digest(data),
            xla_autotune_load_sha256=digest(parent),
        )
        path.write_text(json.dumps(value))
    assert (
        compare_arms(left, right, require_tape=False)["passed"]
        is not mutate_parent_entry
    )
    assert not compare_arms(left, right)["passed"]


def test_report_requires_all_audit_performance_and_repeat_gates(tmp_path):
    from bench.af3_closure_report import summarize_case

    audit, performance = tmp_path / "audit", tmp_path / "performance"
    for mode in (audit, performance):
        mode.mkdir()
        for arm in ("native", "foldjax"):
            root = mode / arm
            _arm(root)
            path = root / "provenance.json"
            value = json.loads(path.read_text())
            value.update(mode=mode.name, source_files={"model.py": "hash"})
            path.write_text(json.dumps(value))
            if mode == performance:
                (root / "finished.json").write_text(json.dumps({"instrumented": False}))
                (root / "raw-first.npz").write_bytes((root / "raw.npz").read_bytes())
    summary, _ = summarize_case("test", audit, performance, artifact_base=tmp_path)
    assert summary["passed"] and len(summary["gates"]) == 6
    assert "source_tree_sha256" in summary["provenance"]["audit/native"]
    assert "source_files" not in summary["provenance"]["audit/native"]
    (performance / "foldjax/raw-first.npz").unlink()
    summary, _ = summarize_case("test", audit, performance, artifact_base=tmp_path)
    assert not summary["passed"]
    assert not summary["gates"]["foldjax_first_warm"]
