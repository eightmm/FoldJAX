import numpy as np
import pytest

from bench import boltz_pwa_logits_probe as probe


@pytest.fixture
def operands():
    return (
        np.zeros((1, 437, 437, 128), np.float32),
        np.zeros((8, 128), np.float32),
        np.zeros((1, 437, 437, 8), np.float32),
    )


def test_complete_shapes_and_native_storage(operands):
    probe.validate_arrays(*operands)


@pytest.mark.parametrize("index", range(3))
def test_shape_mismatch_fails(operands, index):
    values = list(operands)
    values[index] = values[index][..., :-1]
    with pytest.raises(ValueError, match="shape"):
        probe.validate_arrays(*values)


@pytest.mark.parametrize("index", range(3))
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_rejected(operands, index, value):
    operands[index].flat[0] = value
    with pytest.raises(ValueError, match="nonfinite"):
        probe.validate_arrays(*operands)


def test_output_must_be_lossless_bf16(operands):
    operands[2].flat[0] = 1.001
    with pytest.raises(ValueError, match="losslessly"):
        probe.validate_arrays(*operands)


def test_candidate_preserves_head_order_and_native_single_column(monkeypatch):
    import jax.numpy as jnp

    from foldjax.models.boltz2.models.primitives import _common

    x = jnp.zeros((1, 2, 2, 128), jnp.float32)
    kernel = jnp.broadcast_to(jnp.arange(8, dtype=jnp.bfloat16), (128, 8))
    calls = []

    def linear(a, w):
        assert a is x
        assert w.shape == (128, 1)
        assert w.dtype == jnp.bfloat16
        calls.append(int(w[0, 0]))
        return jnp.broadcast_to(w[0, 0], (*a.shape[:-1], 1))

    monkeypatch.setattr(_common, "linear", linear)
    result = probe.candidate_forward(x, kernel)
    assert calls == list(range(8))
    assert result.shape == (1, 2, 2, 8)
    np.testing.assert_array_equal(np.asarray(result[0, 0, 0]), np.arange(8))


def test_reference_rejects_unreproduced_native(tmp_path):
    (tmp_path / "report.json").write_text('{"arm": "native", "passed": false}')
    with pytest.raises(ValueError, match="reproduced full"):
        probe.load_reference(tmp_path)


def test_bound_file_mutation_rejected(tmp_path):
    path = tmp_path / "source.py"
    path.write_text("before")
    bindings = {path: probe.sha(path)}
    probe.verify_bindings(bindings)
    path.write_text("after")
    with pytest.raises(ValueError):
        probe.verify_bindings(bindings)
