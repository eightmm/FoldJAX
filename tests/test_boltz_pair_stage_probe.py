import numpy as np
import pytest

from bench.boltz_pair_stage_probe import STAGES, reconstruct_inputs


def test_teacher_forcing_uses_ordered_fp32_residuals():
    initial = np.array([1e8, 1], np.float32)
    opm = np.array([-1e8, 2], np.float32)
    updates = {
        key: np.array([i + 1, i + 2], np.float32) for i, key in enumerate(STAGES)
    }
    expected = initial + opm
    stages = {}
    for key in STAGES:
        stages[key] = expected
        expected = expected + updates[key]
    actual = reconstruct_inputs(initial, opm, updates, expected)
    for key in STAGES:
        np.testing.assert_array_equal(actual[key], stages[key])


@pytest.mark.parametrize("change", ["missing", "dtype", "shape", "residual"])
def test_teacher_forcing_fails_closed(change):
    initial = np.zeros((2,), np.float32)
    updates = {key: np.ones_like(initial) for key in STAGES}
    final = np.full_like(initial, 5)
    if change == "missing":
        updates.pop(STAGES[0])
    elif change == "dtype":
        updates[STAGES[0]] = updates[STAGES[0]].astype(np.float16)
    elif change == "shape":
        updates[STAGES[0]] = np.ones((1,), np.float32)
    else:
        final += 1
    with pytest.raises(ValueError):
        reconstruct_inputs(initial, initial, updates, final)
