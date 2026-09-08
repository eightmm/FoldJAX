from types import SimpleNamespace

import numpy as np
import pytest

from bench.esmfold2_atom_encoder_candidate import capture


def fixtures():
    def identity(x, *args, **kwargs):
        return x

    atom = SimpleNamespace(
        **{
            name: identity
            for name in (
                "_atom_linear",
                "swa_attention",
                "swiglu_ffn",
                "swa_block",
                "layer_norm",
            )
        }
    )
    model = SimpleNamespace(atom_encoder=lambda x: (x, x, x, (x, x)))
    return atom, model


@pytest.mark.parametrize("failure", [None, "duplicate", "exception"])
def test_capture_restores_functions_and_rejects_duplicate_boundaries(failure):
    atom, model = fixtures()
    originals = dict(vars(atom)), dict(vars(model))
    value = np.ones((1, 2), np.float32)

    def run():
        atom._atom_linear(value, {}, "inputs_embedder.probe", True)
        if failure == "duplicate":
            atom._atom_linear(value, {}, "inputs_embedder.probe", True)
        if failure == "exception":
            raise RuntimeError("failed execution")
        model.atom_encoder(value)
        return value

    if failure:
        with pytest.raises(ValueError if failure == "duplicate" else RuntimeError):
            capture(atom, model, run)
    else:
        actual, stored = capture(atom, model, run)
        np.testing.assert_array_equal(actual, value)
        assert set(stored) == {
            "probe.input",
            "probe.output",
            "atom_attention_encoder.tokens",
            "atom_attention_encoder.queries",
            "atom_attention_encoder.conditioning",
            "atom_attention_encoder.rope_cos",
            "atom_attention_encoder.rope_sin",
        }
    assert vars(atom) == originals[0]
    assert vars(model) == originals[1]
