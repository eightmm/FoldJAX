import numpy as np
import pytest

from bench.protenix_dropout_tape import load_dropout_tape


def test_legacy_disabled_branch(tmp_path):
    assert load_dropout_tape(tmp_path, {"mc_dropout_applied": False}, {}) == (
        None,
        None,
    )


def test_disabled_branch_rejects_stale_rate_without_a_tape(tmp_path):
    with pytest.raises(ValueError, match="disabled branch"):
        load_dropout_tape(
            tmp_path, {"mc_dropout_applied": False, "mc_dropout_rate": 0.4}, {}
        )


@pytest.mark.parametrize("defect", [None, "dtype", "count", "rate", "branch"])
def test_dropout_tape_contract(tmp_path, defect):
    masks = np.ones((10, 2, 2, 3), bool)
    completion = {
        "mc_dropout_applied": True,
        "mc_dropout_rate": 0.4,
        "mc_dropout_mask_calls": 10,
        "mc_dropout_random_draws": [0.1],
    }
    if defect == "dtype":
        masks = masks.astype(np.float32)
    elif defect == "count":
        masks = masks[:-1]
    elif defect == "rate":
        completion["mc_dropout_rate"] = 0.5
    elif defect == "branch":
        completion["mc_dropout_applied"] = False
    np.savez(tmp_path / "dropout-tape.npz", keep_masks=masks)
    if defect:
        with pytest.raises(ValueError, match="dropout"):
            load_dropout_tape(
                tmp_path,
                completion,
                {"mc_dropout_rate": 0.4, "mc_dropout_apply_rate": 0.4},
            )
    else:
        actual, rate = load_dropout_tape(
            tmp_path, completion, {"mc_dropout_rate": 0.4, "mc_dropout_apply_rate": 0.4}
        )
        np.testing.assert_array_equal(actual, masks)
        assert rate == 0.4


@pytest.mark.parametrize("draw", [0.7, float("nan"), -0.1, 1.0, True])
def test_rejects_inconsistent_or_invalid_decision_before_mask_load(tmp_path, draw):
    with pytest.raises(ValueError, match="decision"):
        load_dropout_tape(
            tmp_path,
            {"mc_dropout_applied": True, "mc_dropout_random_draws": [draw]},
            {"mc_dropout_apply_rate": 0.4},
        )
