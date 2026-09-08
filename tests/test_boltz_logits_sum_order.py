import numpy as np
import pytest

from bench.boltz_logits_sum_order import ordered_sum


@pytest.mark.parametrize("lanes", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("contiguous", [False, True])
def test_lane_layout_and_reduction(lanes, contiguous):
    x = np.zeros((2, 128), np.float32)
    x[:, :4] = [2**25, 1, -(2**25), 3]
    expected = []
    for row in x:
        totals = []
        for lane in range(lanes):
            indices = (
                range(lane * (128 // lanes), (lane + 1) * (128 // lanes))
                if contiguous else range(lane, 128, lanes)
            )
            value = np.float32(0)
            for i in indices:
                value = np.float32(value + row[i])
            totals.append(value)
        offset = lanes // 2
        while offset:
            totals = [np.float32(v + totals[i ^ offset]) for i, v in enumerate(totals)]
            offset //= 2
        expected.append(totals[0])
    np.testing.assert_array_equal(
        ordered_sum(x, lanes, contiguous=contiguous), expected
    )


@pytest.mark.parametrize(
    "width,dtype,lanes",
    [(127, np.float32, 4), (128, np.float64, 4), (128, np.float32, 3)],
)
def test_rejects_unobserved_boundary(width, dtype, lanes):
    with pytest.raises(ValueError):
        ordered_sum(np.zeros((2, width), dtype), lanes)
