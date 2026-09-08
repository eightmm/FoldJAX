import numpy as np
import pytest

from bench.esmfold2_linear_shapes import compiler_options, operands


def test_shape_control_operands_are_repeatable_and_finite():
    first, second = list(operands()), list(operands())
    assert len(first) == len(second) == 8
    assert len({name for name, _, _ in first}) == 8
    for (name, x, w), (other, xx, ww) in zip(first, second, strict=True):
        assert name == other and x.shape[-1] == w.shape[-1]
        assert x.dtype == w.dtype == np.float32
        assert np.isfinite(x).all() and np.isfinite(w).all()
        np.testing.assert_array_equal(x, xx)
        np.testing.assert_array_equal(w, ww)


def test_shape_compiler_controls_are_explicit():
    assert compiler_options("default") == {}
    assert compiler_options("split-k-1") == {"xla_gpu_experimental_force_split_k": 1}
    assert compiler_options("no-triton") == {"xla_gpu_enable_triton_gemm": False}
    with pytest.raises(ValueError):
        compiler_options("unknown")
