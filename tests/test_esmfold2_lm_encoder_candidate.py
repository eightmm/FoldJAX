import json

import numpy as np
import pytest

from bench.esmfold2_lm_encoder_candidate import (
    compare_boundaries,
    first_difference_indices,
    validate_capture,
    validate_first_norm,
)
from bench.esmfold2_tape import _save_npz, _sha256


@pytest.mark.parametrize(
    "bad", [None, "capture", "instrumentation", "dtype", "shape", "nan", "lossy"]
)
def test_transition_projection_requires_full_lossless_unchanged_capture(bad):
    from bench.esmfold2_lm_encoder_candidate import validate_transition_projection

    arrays = {
        "transition_projection.input": np.zeros((1, 3, 3, 16), np.float32),
        "transition_projection.output": np.zeros((1, 3, 3, 4), np.float32),
    }
    report = {
        "capture_transition_projection": True,
        "instrumentation_output_bytes_equal": True,
        "boundaries": {"block.input": {"shape": [1, 3, 3, 4]}},
    }
    for key, value in arrays.items():
        report["boundaries"][key] = {
            "scope": "full",
            "original_dtype": "torch.bfloat16",
            "shape": list(value.shape),
            "stored_shape": list(value.shape),
        }
    key = "transition_projection.input"
    if bad == "capture":
        report["capture_transition_projection"] = False
    elif bad == "instrumentation":
        report["instrumentation_output_bytes_equal"] = False
    elif bad == "dtype":
        report["boundaries"][key]["original_dtype"] = "torch.float32"
    elif bad == "shape":
        arrays[key] = arrays[key][:, :1]
    elif bad == "nan":
        arrays[key].flat[0] = np.nan
    elif bad == "lossy":
        arrays[key].flat[0] = np.float32(1.00001)
    if bad:
        with pytest.raises(ValueError):
            validate_transition_projection(report, arrays)
    else:
        validate_transition_projection(report, arrays)


def test_native_linear_output_is_bf16_and_rejects_unverified_bias():
    import jax
    import jax.numpy as jnp

    from bench.esmfold2_lm_encoder_candidate import native_output_linear

    x = jnp.ones((2, 4), jnp.float32)
    params = {"p.weight": jnp.ones((3, 4), jnp.float32)}
    result = native_output_linear(x, params, "p")
    assert result.shape == (2, 3) and result.dtype == jnp.bfloat16
    np.testing.assert_array_equal(result, jnp.full((2, 3), 4))
    traced = jax.make_jaxpr(
        lambda x, w: native_output_linear(
            x, {"p.weight": w}, "p", preserve_boundary=True
        )
    )(x, params["p.weight"])
    assert any(
        eqn.primitive.name == "optimization_barrier" for eqn in traced.jaxpr.eqns
    )
    with pytest.raises(ValueError, match="bias-free"):
        native_output_linear(x, {**params, "p.bias": jnp.zeros(3)}, "p")


def test_native_ffi_block_control_rejects_unverified_bias():
    from bench.esmfold2_lm_encoder_candidate import ffi_output_linear

    with pytest.raises(ValueError, match="bias-free"):
        ffi_output_linear(
            np.ones((2, 4)), {"p.bias": np.zeros(3)}, "p", target="unused"
        )


def test_compiler_controls_are_explicit_and_default_is_unchanged():
    from bench.esmfold2_lm_encoder_candidate import compiler_control

    assert compiler_control("default") == {}
    assert compiler_control("native-chunks-strict-rounding") == {
        **compiler_control("no-triton-native-chunks"),
        "xla_allow_excess_precision": False,
    }
    assert compiler_control("no-triton-native-chunks") == {
        "xla_gpu_enable_triton_gemm": False,
        "xla_disable_hlo_passes": (
            "cublas-pad-for-gemms,dynamic-slice-fusion-rewriter-v2,dot-merger"
        ),
    }
    assert compiler_control("no-triton-no-padding-no-slice-fusion") == {
        "xla_gpu_enable_triton_gemm": False,
        "xla_disable_hlo_passes": (
            "cublas-pad-for-gemms,dynamic-slice-fusion-rewriter-v2"
        ),
    }
    assert compiler_control("no-triton-gemm") == {"xla_gpu_enable_triton_gemm": False}
    assert compiler_control("no-triton-no-padding") == {
        "xla_gpu_enable_triton_gemm": False,
        "xla_disable_hlo_passes": "cublas-pad-for-gemms",
    }
    with pytest.raises(ValueError):
        compiler_control("unknown")


@pytest.mark.parametrize("bad", [None, "archive", "checkpoint", "dtype", "shape"])
def test_native_block_binding(tmp_path, bad):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"weights")
    pair = np.zeros((1, 2, 2, 4), np.float32)
    arrays = {"block.input": pair, "block.output": pair, "pair_mask": pair[..., 0]}
    _save_npz(tmp_path / "native.npz", arrays)
    report = {
        "scope": "teacher_forced_first_lm_encoder_block_dropout_disabled",
        "entry_cast": "float32_to_bfloat16",
        "chunk_size": 64,
        "kernel_backend": None,
        "archive_sha256": _sha256(tmp_path / "native.npz"),
        "bindings": {str(checkpoint): _sha256(checkpoint)},
        "input_shim_policy": {"checkpoint_sha256": _sha256(checkpoint)},
        "boundaries": {
            k: {
                "scope": "full",
                "shape": list(v.shape),
                "original_dtype": "torch.bfloat16",
            }
            for k, v in arrays.items()
        },
    }
    if bad == "archive":
        report["archive_sha256"] = "wrong"
    elif bad == "checkpoint":
        checkpoint.write_bytes(b"changed")
    elif bad == "dtype":
        report["boundaries"]["block.input"]["original_dtype"] = "torch.float32"
    elif bad == "shape":
        report["boundaries"]["block.input"]["shape"] = [3]
    (tmp_path / "report.json").write_text(json.dumps(report))
    if bad:
        with pytest.raises(ValueError):
            validate_capture(tmp_path, tmp_path)
    else:
        validate_capture(tmp_path, tmp_path)


def test_original_dtype_is_not_float32_archive_storage():
    arrays = {"norm": np.ones(3, np.float32)}
    result = compare_boundaries(
        arrays,
        arrays,
        {"norm": {"original_dtype": "torch.float32"}},
        {"norm": {"dtype": "bfloat16"}},
    )["leaves"]["norm"]
    assert (
        result["native_storage_dtype"] == result["candidate_storage_dtype"] == "float32"
    )
    assert result["native_original_dtype"] == "float32"
    assert result["candidate_original_dtype"] == "bfloat16"
    assert not result["original_dtype_equal"]
    assert result["strict_pass"]


def test_missing_original_dtype_fails_closed():
    arrays = {"output": np.ones(3, np.float32)}
    with pytest.raises(KeyError):
        compare_boundaries(
            arrays, arrays, {"output": {}}, {"output": {"dtype": "bfloat16"}}
        )


def test_complete_operand_first_indices_are_not_slice_limited():
    lhs = np.zeros((1, 64, 65, 4), np.float32)
    rhs = lhs.copy()
    rhs[0, 63, 64, 3] = 1
    report = first_difference_indices(lhs, rhs)
    assert report == {"different_entries": 1, "first_indices": [[0, 63, 64, 3]]}
    assert first_difference_indices(lhs, lhs)["different_entries"] == 0


def test_operand_shape_change_rejected():
    with pytest.raises(ValueError):
        first_difference_indices(np.zeros(2), np.zeros(3))


@pytest.mark.parametrize("bad", [None, "slice", "dtype", "nonfinite"])
def test_full_norm_capture_contract(bad):
    value = np.ones((1, 3, 3, 4), np.float32)
    shape = list(value.shape)
    schema = {
        "scope": "full",
        "original_dtype": "torch.float32",
        "shape": shape,
        "stored_shape": shape,
    }
    report = {
        "capture_first_norm": True,
        "boundaries": {
            "block.input": {"shape": shape},
            "first_norm.output": schema,
        },
    }
    if bad == "slice":
        schema["scope"] = "slice"
    elif bad == "dtype":
        schema["original_dtype"] = "torch.bfloat16"
    elif bad == "nonfinite":
        value.flat[0] = np.nan
    if bad:
        with pytest.raises(ValueError):
            validate_first_norm(report, {"first_norm.output": value})
    else:
        validate_first_norm(report, {"first_norm.output": value})


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "missing",
        "instrumentation",
        "shape",
        "dtype",
        "slice",
        "nan",
        "lossy",
        "input",
    ],
)
def test_full_projection_capture_requires_actual_norm_input_and_bf16_output(bad):
    from bench.esmfold2_lm_encoder_candidate import validate_first_projection

    shape = [1, 3, 3, 4]
    arrays = {
        "first_norm.output": np.ones(shape, np.float32),
        "first_projection.input": np.ones(shape, np.float32),
        "first_projection.output": np.ones([1, 3, 3, 16], np.float32),
    }
    boundaries = {"block.input": {"shape": shape}}
    for key, value in arrays.items():
        boundaries[key] = {
            "shape": list(value.shape),
            "stored_shape": list(value.shape),
            "scope": "full",
            "original_dtype": (
                "torch.bfloat16"
                if key == "first_projection.output"
                else "torch.float32"
            ),
        }
    report = {
        "capture_first_norm": True,
        "capture_first_projection": True,
        "instrumentation_output_bytes_equal": True,
        "boundaries": boundaries,
    }
    output_schema = boundaries["first_projection.output"]
    if bad == "missing":
        report["capture_first_projection"] = False
    elif bad == "instrumentation":
        report["instrumentation_output_bytes_equal"] = False
    elif bad == "shape":
        output_schema["stored_shape"] = [1, 1, 1, 16]
    elif bad == "dtype":
        output_schema["original_dtype"] = "torch.float32"
    elif bad == "slice":
        output_schema["scope"] = "slice"
    elif bad in ("nan", "lossy"):
        arrays["first_projection.output"].flat[0] = np.nan if bad == "nan" else 1.001
    elif bad == "input":
        arrays["first_projection.input"].flat[0] = 2
    if bad:
        with pytest.raises(ValueError):
            validate_first_projection(report, arrays)
    else:
        validate_first_projection(report, arrays)


@pytest.mark.parametrize(
    "bad", [None, "flag", "instrumentation", "shape", "dtype", "nan", "lossy"]
)
def test_full_incoming_boundary_contract(bad):
    from bench.esmfold2_lm_encoder_candidate import validate_incoming_boundary

    shape = [1, 3, 3, 4]
    arrays = {
        "incoming." + k: np.ones(shape, np.float32)
        for k in ("input", "left", "right", "output")
    }
    boundaries = {"block.input": {"shape": shape}}
    for key in arrays:
        boundaries[key] = {
            "shape": shape,
            "stored_shape": shape,
            "scope": "full",
            "original_dtype": "torch.float32"
            if key in ("incoming.left", "incoming.right")
            else "torch.bfloat16",
        }
    report = {
        "capture_incoming_boundary": True,
        "instrumentation_output_bytes_equal": True,
        "boundaries": boundaries,
    }
    if bad == "flag":
        report["capture_incoming_boundary"] = False
    elif bad == "instrumentation":
        report["instrumentation_output_bytes_equal"] = False
    elif bad == "shape":
        boundaries["incoming.left"]["stored_shape"] = [1, 1, 1, 4]
    elif bad == "dtype":
        boundaries["incoming.left"]["original_dtype"] = "torch.bfloat16"
    elif bad in ("nan", "lossy"):
        arrays["incoming.output"].flat[0] = np.nan if bad == "nan" else 1.001
    if bad:
        with pytest.raises(ValueError):
            validate_incoming_boundary(report, arrays)
    else:
        validate_incoming_boundary(report, arrays)


@pytest.mark.parametrize("tokens", [16, 64, 65, 130])
def test_incoming_chunk_assembly_preserves_axes_and_tail(tokens):
    import jax.numpy as jnp

    from bench.esmfold2_lm_encoder_candidate import assemble_incoming_chunks

    left = (
        jnp.arange(tokens * tokens * 2, dtype=jnp.float32)
        .reshape(1, tokens, tokens, 2)
        .astype(jnp.bfloat16)
    )
    output = left + jnp.bfloat16(1)
    lhs = [left[:, :, i : i + 64] for i in range(0, tokens, 64)]
    values = [output[:, i : i + 64] for i in range(0, tokens, 64)]
    actual_left, actual_output = assemble_incoming_chunks(lhs, values, tokens)
    np.testing.assert_array_equal(actual_left, left)
    np.testing.assert_array_equal(actual_output, output)
    with pytest.raises(ValueError):
        assemble_incoming_chunks(lhs[:-1], values, tokens)
    with pytest.raises(ValueError):
        assemble_incoming_chunks([x.astype(jnp.float32) for x in lhs], values, tokens)


@pytest.mark.parametrize("tokens", [64, 65, 130])
def test_outgoing_chunk_assembly_uses_first_pair_axis(tokens):
    import jax.numpy as jnp

    from bench.esmfold2_lm_encoder_candidate import assemble_triangle_chunks

    left = (
        jnp.arange(tokens * tokens, dtype=jnp.float32)
        .reshape(1, tokens, tokens, 1)
        .astype(jnp.bfloat16)
    )
    chunks = [left[:, start : start + 64] for start in range(0, tokens, 64)]
    actual, output = assemble_triangle_chunks(chunks, chunks, tokens, left_axis=1)
    np.testing.assert_array_equal(actual, left)
    np.testing.assert_array_equal(output, left)
    with pytest.raises(ValueError):
        assemble_triangle_chunks(chunks, chunks, tokens, left_axis=0)


@pytest.mark.parametrize(
    "bad",
    [None, "flag", "shape", "norm_dtype", "projection_dtype", "nonfinite", "lossy"],
)
def test_complete_outgoing_boundary_contract(bad):
    from bench.esmfold2_lm_encoder_candidate import validate_outgoing_boundary

    shape = [1, 3, 3, 4]
    arrays, boundaries = {}, {"block.input": {"shape": shape}}
    dtypes = {
        "first_norm.output": "torch.float32",
        "first_projection.input": "torch.float32",
        "first_projection.output": "torch.bfloat16",
    }
    for direction in ("incoming", "outgoing"):
        leaves = (
            ("input", "left", "right", "output")
            if direction == "incoming"
            else ("left", "right", "output", "norm_mix", "proj_emit", "proj_gate")
        )
        for leaf in leaves:
            dtypes[f"{direction}.{leaf}"] = (
                "torch.float32"
                if leaf in ("left", "right", "norm_mix")
                else "torch.bfloat16"
            )
    for key, dtype in dtypes.items():
        target = [1, 3, 3, 16] if key == "first_projection.output" else shape
        arrays[key] = np.ones(target, np.float32)
        boundaries[key] = {
            "shape": target,
            "stored_shape": target,
            "scope": "full",
            "original_dtype": dtype,
        }
    report = {
        "boundaries": boundaries,
        "capture_first_norm": True,
        "capture_first_projection": True,
        "capture_incoming_boundary": True,
        "capture_outgoing_boundary": True,
        "instrumentation_output_bytes_equal": True,
    }
    if bad == "flag":
        report["capture_outgoing_boundary"] = False
    elif bad == "shape":
        boundaries["outgoing.left"]["stored_shape"] = [1, 1, 1, 4]
    elif bad == "norm_dtype":
        boundaries["outgoing.norm_mix"]["original_dtype"] = "torch.bfloat16"
    elif bad == "projection_dtype":
        boundaries["outgoing.proj_emit"]["original_dtype"] = "torch.float32"
    elif bad == "nonfinite":
        arrays["outgoing.right"].flat[0] = np.nan
    elif bad == "lossy":
        arrays["outgoing.proj_gate"].flat[0] = 1.001
    if bad:
        with pytest.raises(ValueError):
            validate_outgoing_boundary(report, arrays)
    else:
        validate_outgoing_boundary(report, arrays)
