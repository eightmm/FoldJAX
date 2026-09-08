import numpy as np
import pytest

from bench.esmfold2_lm_stack import validate_stack_input


@pytest.mark.parametrize(
    "bad", [None, "layers", "disabled", "nonfinite", "lossy", "mask", "width", "empty"]
)
def test_stack_requires_full_native_bf16_input_and_native_config(bad):
    config = {"d_pair": 4, "lm_encoder": {"enabled": True, "n_layers": 4}}
    value = np.ones((1, 3, 3, 4), np.float32)
    mask = np.ones((1, 3, 3), np.float32)
    if bad == "layers":
        config["lm_encoder"]["n_layers"] = True
    elif bad == "disabled":
        config["lm_encoder"]["enabled"] = False
    elif bad == "nonfinite":
        value.flat[0] = np.nan
    elif bad == "lossy":
        value.flat[0] = 1.00001
    elif bad == "mask":
        mask.flat[0] = 0.5
    elif bad == "width":
        config["d_pair"] = 5
    elif bad == "empty":
        value, mask = value[:, :0, :0], mask[:, :0, :0]
    if bad:
        with pytest.raises(ValueError):
            validate_stack_input(value, mask, config)
    else:
        assert validate_stack_input(value, mask, config) == 4
