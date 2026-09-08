import numpy as np
import pytest

from bench.esmfold2_contraction_probe import select_contraction, validate_operands


@pytest.mark.parametrize("transposed", [False, True])
def test_outgoing_storage_controls_preserve_logical_contraction(transposed):
    from bench.esmfold2_contraction_probe import outgoing_layout_operands

    rng = np.random.default_rng(91)
    operands = {
        "lhs": rng.integers(-4, 4, (1, 3, 5, 2)).astype(np.float32),
        "rhs": rng.integers(-4, 4, (1, 7, 5, 2)).astype(np.float32),
    }
    left, right = outgoing_layout_operands(operands, transposed)
    equation = "dkj,dki->dji" if transposed else "djk,dki->dji"
    actual = np.einsum(equation, right, left).transpose(2, 1, 0)[None]
    expected = np.einsum("bikd,bjkd->bijd", operands["lhs"], operands["rhs"])
    np.testing.assert_array_equal(actual, expected)
    assert left.flags.c_contiguous and right.flags.c_contiguous


def test_outgoing_storage_control_rejects_multiple_batches():
    from bench.esmfold2_contraction_probe import outgoing_layout_operands

    with pytest.raises(ValueError, match="batch-one"):
        outgoing_layout_operands(
            {"lhs": np.zeros((2, 3, 5, 2)), "rhs": np.zeros((2, 5, 5, 2))}, False
        )


@pytest.mark.parametrize(
    "bad", [None, "partial", "dtype", "nan", "scope", "control", "absent"]
)
def test_full_native_contraction_contract(bad):
    shapes = {"lhs": [1, 64, 65, 4], "rhs": [1, 65, 65, 4], "output": [1, 64, 65, 4]}
    arrays = {
        "first_contraction." + k: np.zeros(v, np.float32) for k, v in shapes.items()
    }
    report = {
        "capture_first_contraction": True,
        "precision_control": False,
        "boundaries": {"block.input": {"shape": [1, 65, 65, 4]}},
    }
    for name, shape in shapes.items():
        report["boundaries"]["first_contraction." + name] = {
            "shape": shape,
            "stored_shape": shape,
            "scope": "full",
            "native_stride": [
                shape[1] * shape[2] * shape[3],
                shape[2] * shape[3],
                shape[3],
                1,
            ],
            "native_storage_offset": 0,
            "original_dtype": "torch.bfloat16" if name == "output" else "torch.float32",
        }
    key = "first_contraction.lhs"
    if bad == "partial":
        arrays[key] = arrays[key][:, :, :3]
    elif bad == "dtype":
        report["boundaries"][key]["original_dtype"] = "torch.bfloat16"
    elif bad == "nan":
        arrays[key].flat[0] = np.nan
    elif bad == "scope":
        report["boundaries"][key]["scope"] = "slice"
    elif bad == "control":
        report["precision_control"] = True
    elif bad == "absent":
        report["capture_first_contraction"] = False
    if bad:
        with pytest.raises(ValueError):
            validate_operands(report, arrays)
    else:
        assert validate_operands(report, arrays)["lhs"].shape == (1, 64, 65, 4)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "negative_offset",
        "zero_stride",
        "huge_stride",
        "policy",
        "instrumentation",
    ],
)
def test_incoming_selection_keeps_native_axes_stride_and_right_offset(bad):
    shape = [1, 65, 65, 4]
    arrays, boundaries = {}, {"block.input": {"shape": shape}}
    for leaf in ("input", "left", "right", "output"):
        key = "incoming." + leaf
        arrays[key] = np.ones(shape, np.float32)
        operand = leaf in ("left", "right")
        boundaries[key] = {
            "shape": shape,
            "stored_shape": shape,
            "scope": "full",
            "original_dtype": "torch.float32" if operand else "torch.bfloat16",
            "native_stride": [33800, 520, 8, 1] if operand else [16900, 260, 4, 1],
            "native_storage_offset": 4 if leaf == "right" else 0,
        }
    report = {
        "boundaries": boundaries,
        "capture_incoming_boundary": True,
        "instrumentation_output_bytes_equal": True,
        "precision_control": False,
    }
    if bad == "negative_offset":
        boundaries["incoming.right"]["native_storage_offset"] = -1
    elif bad == "zero_stride":
        boundaries["incoming.left"]["native_stride"][2] = 0
    elif bad == "huge_stride":
        boundaries["incoming.left"]["native_stride"][1] = 1 << 30
    elif bad == "policy":
        report["precision_control"] = True
    elif bad == "instrumentation":
        report["instrumentation_output_bytes_equal"] = False
    if bad:
        with pytest.raises(ValueError):
            select_contraction(report, arrays, "incoming")
    else:
        operands, schemas = select_contraction(report, arrays, "incoming")
        assert operands["lhs"].shape == (1, 65, 64, 4)
        assert operands["rhs"].shape == (1, 65, 65, 4)
        assert operands["output"].shape == (1, 64, 65, 4)
        assert schemas["rhs"]["native_storage_offset"] == 4
        assert schemas["lhs"]["native_stride"] == [33800, 520, 8, 1]
        assert boundaries["incoming.left"]["shape"] == shape
