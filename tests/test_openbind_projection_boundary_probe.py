import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend import core

from bench import openbind_projection_boundary_probe as probe
from foldjax.models.openfold3.models import primitives


def case():
    return probe.full.Case("pairformer", "tri_att_start", 16)


def boundaries(c):
    return {
        f"{key}_{r}": np.full(probe.expected_shape(c, key), 0.25, np.float32)
        for key in probe.FIELDS
        for r in (0, 1)
    }


def observed_inputs(norm, c):
    return {
        key: probe.value_identity(
            norm.reshape(c.shape if key == "bias" else c.shape[1:])
        )
        for key in probe.PROJECTIONS
    }


def test_three_fixed_real_weight_cases_and_four_precision_controls():
    cases = probe.case_specs()
    assert [c.length for c in cases] == [16, 17, 437]
    assert all(
        c.family == "pairformer" and c.operator == "tri_att_start" and c.width == 128
        for c in cases
    )
    assert probe.CONTROLS == (
        "high",
        "TF32_TF32_F32",
        "TF32_TF32_F32_X3",
        "F32_F32_F32",
    )
    assert "never\nthe candidate LN output" in probe.__doc__


@pytest.mark.parametrize("damage", ["missing", "dtype", "shape", "nan", "repeat"])
def test_boundary_archive_rejects_malformed_or_unstable_arrays(damage):
    c = case()
    arrays = boundaries(c)
    assert all(m["bitwise_equal"] for m in probe.validate_boundary(arrays, c).values())
    if damage == "missing":
        del arrays["q_0"]
    if damage == "dtype":
        arrays["q_0"] = arrays["q_0"].astype(np.float64)
    if damage == "shape":
        arrays["q_0"] = arrays["q_0"][:1]
    if damage == "nan":
        arrays["q_0"][0, 0, 0] = np.nan
    if damage == "repeat":
        arrays["q_1"][0, 0, 0] += 1
    with pytest.raises(ValueError):
        probe.validate_boundary(arrays, c)


def test_teacher_must_be_exact_actual_native_projection_input():
    c = case()
    norm = boundaries(c)["norm_0"]
    observed = observed_inputs(norm, c)
    probe.validate_inputs(observed, norm, c)
    changed = norm.copy()
    changed[0, 0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="differs from captured LN"):
        probe.validate_inputs(observed, changed, c)


def test_missing_projection_input_capture_rejected():
    c = case()
    norm = boundaries(c)["norm_0"]
    observed = observed_inputs(norm, c)
    del observed["bias"]
    with pytest.raises(ValueError, match="capture incomplete"):
        probe.validate_inputs(observed, norm, c)


def test_candidate_high_uses_teacher_directly_not_recomputed_norm(monkeypatch):
    c = case()
    norm = jnp.asarray(
        np.random.default_rng(11).normal(size=c.shape).astype(np.float32)
    )
    weights = {
        name + ".weight": jnp.ones((4 if key == "bias" else 128, 128), jnp.float32)
        * 0.01
        for key, name in probe.PROJECTIONS.items()
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("teacher projection must not normalize")

    monkeypatch.setattr(primitives, "layer_norm", forbidden)
    actual = probe.projection_group(primitives, norm, weights, "high")
    for key, name in probe.PROJECTIONS.items():
        x = norm if key == "bias" else norm.reshape(norm.shape[1:])
        np.testing.assert_allclose(
            actual[key],
            np.asarray(x) @ np.asarray(weights[name + ".weight"]).T,
            rtol=1e-5,
            atol=2e-6,
        )
        assert actual[key].shape == probe.expected_shape(c, key)


@pytest.mark.parametrize("control", probe.CONTROLS)
def test_explicit_precision_is_in_actual_dot_jaxpr_not_metadata_only(control):
    c = case()
    norm = jnp.ones(c.shape)
    weights = {
        name + ".weight": jnp.ones((4 if key == "bias" else 128, 128))
        for key, name in probe.PROJECTIONS.items()
    }
    graph = jax.make_jaxpr(
        lambda x, w: probe.projection_group(primitives, x, w, control)
    )(norm, weights).jaxpr

    def collect(g):
        dots = []
        for eq in g.eqns:
            if eq.primitive.name == "dot_general":
                dots.append(eq)
            for value in eq.params.values():
                if isinstance(value, core.ClosedJaxpr):
                    dots += collect(value.jaxpr)
                elif isinstance(value, core.Jaxpr):
                    dots += collect(value)
        return dots

    dots = collect(graph)
    assert len(dots) == 5
    for dot in dots:
        expected = (
            (jax.lax.Precision.HIGH,) * 2
            if control == "high"
            else getattr(jax.lax.DotAlgorithmPreset, control)
        )
        assert dot.params["precision"] == expected


def reference(tmp_path, monkeypatch):
    c = case()
    arrays = boundaries(c)
    monkeypatch.setattr(probe, "case_specs", lambda: [c])
    record = {
        "spec": c.identity(),
        "status": "ok",
        "native_repeat": probe.validate_boundary(arrays, c),
        "projection_inputs": [observed_inputs(arrays["norm_0"], c)] * 2,
        "baseline_comparison": {"bitwise_equal": True},
        "module_repeat": probe.metrics(arrays["norm_0"], arrays["norm_1"]),
        "module_archive_sha256": probe.save_arrays(
            tmp_path / f"{c.name}-module.npz",
            first=arrays["norm_0"],
            second=arrays["norm_1"],
        ),
        "archive_sha256": probe.save_arrays(tmp_path / f"{c.name}.npz", **arrays),
    }
    manifest = {
        "panel": probe.PANEL,
        "mode": "native",
        "completed": True,
        "controls": list(probe.CONTROLS),
        "upstream_reference_sha256": "a" * 64,
        "source": {
            "commit": probe.UPSTREAM_COMMIT,
            "operator_path": probe.NATIVE_OPERATOR,
        },
        "policy": probe.full.POLICY,
        "cases": {c.name: record},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return c, manifest


def test_reference_hash_binds_consumed_boundary_archive(tmp_path, monkeypatch):
    c, m = reference(tmp_path, monkeypatch)
    assert probe.verify_reference(tmp_path, "a" * 64) == m
    path = tmp_path / f"{c.name}.npz"
    path.write_bytes(path.read_bytes() + b"change")
    with pytest.raises(ValueError, match="archive changed"):
        probe.verify_reference(tmp_path, "a" * 64)


@pytest.mark.parametrize(
    "damage", ["source", "baseline", "teacher", "original", "manifest"]
)
def test_reference_provenance_and_teacher_corruption_rejected(
    tmp_path, monkeypatch, damage
):
    c, m = reference(tmp_path, monkeypatch)
    sha = probe.digest(tmp_path / "manifest.json")
    if damage == "source":
        m["source"]["commit"] = "wrong"
    if damage == "baseline":
        m["cases"][c.name]["baseline_comparison"]["bitwise_equal"] = False
    if damage == "teacher":
        m["cases"][c.name]["projection_inputs"][0]["q"]["sha256"] = "wrong"
    if damage == "original":
        m["upstream_reference_sha256"] = "b" * 64
    if damage == "manifest":
        m["extra"] = True
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError):
        probe.verify_reference(
            tmp_path, "a" * 64, sha if damage == "manifest" else None
        )


def test_candidate_source_identity_names_actual_ordinary_primitives():
    root = probe.Path(__file__).resolve().parents[1]
    identity = probe.source_identity(root, False)
    assert identity["operator_path"] == probe.CANDIDATE_OPERATOR
    assert identity["operator_sha256"] == probe.digest(root / probe.CANDIDATE_OPERATOR)
    assert "openbind_projection_boundary_probe.py" in probe.helper_identities()


def test_shared_cuda_norm_control_binds_its_runtime_source():
    root = probe.Path(__file__).resolve().parents[1]
    identity = probe.source_identity(root, False, "cuda-welford")
    assert identity["norm_control_source"] == {
        probe.SHARED_NORM: probe.digest(root / probe.SHARED_NORM)
    }
    with pytest.raises(ValueError, match="native baseline"):
        probe.source_identity(root, True, "cuda-welford")


@pytest.mark.parametrize("implementation", ["generic", "cuda-welford"])
def test_norm_control_is_separate_from_teacher_projection(monkeypatch, implementation):
    from types import SimpleNamespace

    import jax.numpy as jnp

    from foldjax.models.boltz2.models.primitives import native_amp_norm

    x = jnp.ones((1, 2, 2, 128), jnp.float32)
    params = SimpleNamespace(weight=jnp.ones(128), bias=jnp.zeros(128))
    calls = []

    def generic(value, p):
        assert value is x and p is params
        calls.append("generic")
        return value * 2

    def cuda(value, weight, bias, eps):
        assert value is x and weight is params.weight and bias is params.bias
        assert eps == 1e-5
        calls.append("cuda-welford")
        return value * 3, None, None

    monkeypatch.setattr(native_amp_norm, "_cuda_layer_norm", cuda)
    result = probe.candidate_norm(
        SimpleNamespace(layer_norm=generic), x, params, implementation
    )
    assert calls == [implementation]
    np.testing.assert_array_equal(result, x * (2 if implementation == "generic" else 3))


def test_host_rne_exact_positive_negative_ties_and_tf32_grid_idempotence():
    bits = np.array(
        [
            0x3F801000,
            0x3F803000,
            0xBF801000,
            0xBF803000,
            0x3F800FFF,
            0x3F801001,
            0xBF800FFF,
            0xBF801001,
            0x3F802000,
            0xBF802000,
            0x3FFFF000,
            0xBFFFF000,
            0x00001000,
            0x00003000,
            0x80001000,
            0x80003000,
            0x00000000,
            0x80000000,
        ],
        np.uint32,
    )
    expected = np.array(
        [
            0x3F800000,
            0x3F804000,
            0xBF800000,
            0xBF804000,
            0x3F800000,
            0x3F802000,
            0xBF800000,
            0xBF802000,
            0x3F802000,
            0xBF802000,
            0x40000000,
            0xC0000000,
            0x00000000,
            0x00004000,
            0x80000000,
            0x80004000,
            0x00000000,
            0x80000000,
        ],
        np.uint32,
    )
    result = probe.tf32_rne(bits.view(np.float32))
    np.testing.assert_array_equal(result.view(np.uint32), expected)
    np.testing.assert_array_equal(probe.tf32_rne(result).view(np.uint32), expected)
    assert not np.array_equal(
        expected[:4], (bits[:4] + np.uint32(0x1000)) & np.uint32(0xFFFFE000)
    )


def test_host_rne_preserves_nan_payloads_infinities_and_input_bytes():
    bits = np.array(
        [0x7F800000, 0xFF800000, 0x7F800001, 0x7FC01234, 0xFFC01234, 0xFF801FFF],
        np.uint32,
    )
    before = bits.copy()
    output = probe.tf32_rne(bits.view(np.float32))
    np.testing.assert_array_equal(output.view(np.uint32), before)
    np.testing.assert_array_equal(bits, before)
    assert not np.shares_memory(output, bits)


def test_rne_preserves_scalar_and_noncontiguous_shapes():
    scalar = np.array(0x3F801000, np.uint32).view(np.float32)
    assert probe.tf32_rne(scalar).shape == ()
    values = np.full((2, 4), scalar, np.float32)[:, ::2]
    assert not values.flags.c_contiguous
    result = probe.tf32_rne(values)
    assert result.shape == values.shape
    np.testing.assert_array_equal(
        result.view(np.uint32), np.full(values.shape, 0x3F800000, np.uint32)
    )


@pytest.mark.parametrize("dtype", [np.float16, np.float64, np.int32])
def test_rne_rejects_non_fp32_storage(dtype):
    with pytest.raises(TypeError, match="requires FP32"):
        probe.tf32_rne(np.ones((2, 2), dtype))


def control_inputs():
    norm = np.full(
        (1, 2, 2, 128), np.array(0x3F801000, np.uint32).view(np.float32), np.float32
    )
    weights = {
        name + ".weight": np.full(
            (4 if key == "bias" else 128, 128),
            np.array(0xBF803000, np.uint32).view(np.float32),
            np.float32,
        )
        for key, name in probe.PROJECTIONS.items()
    }
    return norm, weights


def test_rne_control_hashes_original_and_effective_teacher_and_all_five_weights():
    norm, weights = control_inputs()
    original = {
        "native_norm_0": probe.value_identity(norm),
        **{k: probe.value_identity(v) for k, v in weights.items()},
    }
    effective_norm, effective_weights, identities = probe.candidate_operands(
        norm, weights, "rne"
    )
    assert identities["original"] == original
    assert probe.value_identity(norm) == original["native_norm_0"]
    assert identities["effective"]["native_norm_0"] == probe.value_identity(
        effective_norm
    )
    for key, value in effective_weights.items():
        assert probe.value_identity(weights[key]) == original[key]
        assert identities["effective"][key] == probe.value_identity(value)
        assert identities["effective"][key]["sha256"] != original[key]["sha256"]
    assert (
        identities["effective"]["native_norm_0"]["sha256"]
        != original["native_norm_0"]["sha256"]
    )
    assert probe.selected_controls("rne") == ("high",)
    metadata = probe.operand_control_metadata("rne")
    assert metadata["stage"] == "host_before_device_transfer"
    assert (
        metadata["production_modified"] is False and metadata["teacher_forced"] is True
    )


def test_default_control_preserves_original_arrays_and_algorithm_panel():
    norm, weights = control_inputs()
    actual_norm, actual_weights, identities = probe.candidate_operands(
        norm, weights, "none"
    )
    assert actual_norm is norm
    assert all(actual_weights[k] is v for k, v in weights.items())
    assert identities["original"] == identities["effective"]
    assert probe.selected_controls("none") == probe.CONTROLS


@pytest.mark.parametrize("damage", ["missing", "extra", "nan", "dtype", "overflow"])
def test_effective_operand_preflight_is_fail_closed(damage):
    norm, weights = control_inputs()
    if damage == "missing":
        del weights["mha.linear_q.weight"]
    if damage == "extra":
        weights["layer_norm.weight"] = np.ones(128, np.float32)
    if damage == "nan":
        norm[0, 0, 0, 0] = np.nan
    if damage == "dtype":
        norm = norm.astype(np.float64)
    if damage == "overflow":
        norm[0, 0, 0, 0] = np.finfo(np.float32).max
    with pytest.raises(ValueError):
        probe.candidate_operands(norm, weights, "rne")


def test_native_cli_rejects_control_before_reading_source_or_writing(tmp_path):
    with pytest.raises(ValueError, match="native baseline must not"):
        probe.main(
            [
                "--mode",
                "native",
                "--source-root",
                str(tmp_path),
                "--upstream-reference",
                str(tmp_path),
                "--out-dir",
                str(tmp_path / "new"),
                "--operand-rounding",
                "rne",
            ]
        )
    assert not (tmp_path / "new").exists()


def test_controlled_native_reference_cannot_replace_original_baseline(
    tmp_path, monkeypatch
):
    _, manifest = reference(tmp_path, monkeypatch)
    manifest["operand_control"] = {"mode": "rne"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source/policy"):
        probe.verify_reference(tmp_path, "a" * 64)
