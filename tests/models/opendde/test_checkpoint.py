from __future__ import annotations

import numpy as np
import pytest

from foldjax.models.opendde.bridge.checkpoint import unwrap_state_dict


def test_unwrap_state_dict_reads_model_payload_and_strips_ddp_prefix() -> None:
    weight = np.arange(6, dtype=np.float32).reshape(2, 3)
    checkpoint = {
        "model": {
            "module.distogram_head.linear.weight": weight,
            "distogram_head.linear.bias": np.zeros(2, dtype=np.float32),
        },
        "optimizer": {"ignored": True},
    }

    state = unwrap_state_dict(checkpoint)

    assert set(state) == {
        "distogram_head.linear.weight",
        "distogram_head.linear.bias",
    }
    np.testing.assert_array_equal(
        state["distogram_head.linear.weight"],
        weight,
    )
    assert "module.distogram_head.linear.weight" in checkpoint["model"]


def test_unwrap_state_dict_accepts_raw_state_dict() -> None:
    state = unwrap_state_dict({"layer.weight": np.ones((2, 2), dtype=np.float32)})

    assert set(state) == {"layer.weight"}


def test_unwrap_state_dict_rejects_non_mapping_payload() -> None:
    with pytest.raises(TypeError, match="state dict"):
        unwrap_state_dict({"model": [1, 2, 3]})
