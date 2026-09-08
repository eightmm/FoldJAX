import numpy as np
import pytest

from bench.boltz_triangle_boundary import native_vector4_mean


def test_native_mean_uses_sequential_vector_four_sum():
    x = np.zeros((2, 128), np.float32)
    x[:, :4] = [2**25, 1, -(2**25), 3]
    np.testing.assert_array_equal(native_vector4_mean(x), np.full(2, 3 / 128))


@pytest.mark.parametrize(
    "shape,dtype", [((2, 127), np.float32), ((2, 128), np.float64)]
)
def test_native_mean_rejects_other_profiles(shape, dtype):
    with pytest.raises(ValueError):
        native_vector4_mean(np.zeros(shape, dtype))
