import numpy as np
import pytest

from bench.esmfold2_lm_encoder_probe import block_state, validate_inputs


@pytest.mark.parametrize("bad", [None, "dtype", "shape", "nan", "mask", "layers"])
def test_input_contract(bad):
    pair = np.zeros((1, 3, 3, 4), np.float32)
    mask = np.ones((1, 3), np.float32)
    config = {"d_pair": 4, "lm_encoder": {"n_layers": 4}}
    if bad == "dtype":
        pair = pair.astype(np.float64)
    elif bad == "shape":
        pair = pair[:, :2]
    elif bad == "nan":
        pair.flat[0] = np.nan
    elif bad == "mask":
        mask[0, 0] = 0.5
    elif bad == "layers":
        config["lm_encoder"]["n_layers"] = 0
    if bad:
        with pytest.raises(ValueError):
            validate_inputs(pair, mask, config)
    else:
        validate_inputs(pair, mask, config)


def test_exact_first_block_weight_prefix_only():
    class Handle:
        def keys(self):
            return [
                "lm_encoder.blocks.0.a", "lm_encoder.blocks.1.a",
                "folding_trunk.blocks.0.a", "lm_encoder.blocks.01.a",
            ]

        def get_tensor(self, name):
            assert name == "lm_encoder.blocks.0.a"
            return 7

    assert block_state(Handle()) == {"a": 7}


def test_missing_block_weights_rejected():
    class Empty:
        def keys(self):
            return []

    with pytest.raises(ValueError, match="missing native"):
        block_state(Empty())
