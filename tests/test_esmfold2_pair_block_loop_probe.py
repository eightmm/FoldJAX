"""CPU contracts for the synthetic composed ESMFold2 pair-block probe."""

from __future__ import annotations

import contextlib
import os
from types import SimpleNamespace

import numpy as np
import pytest

from bench import esmfold2_pair_block_loop_probe as probe


def _weights() -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    for direction in probe._TRIANGLE_NAMES:
        engine = f"{probe.BLOCK_PREFIX}.{direction}._engine"
        tensors.update(
            {
                f"{engine}.norm_start.weight": np.ones(256, np.float32),
                f"{engine}.norm_start.bias": np.zeros(256, np.float32),
                f"{engine}.proj_bundle.weight": np.ones((1024, 256), np.float32),
                f"{engine}.norm_mix.weight": np.ones(256, np.float32),
                f"{engine}.norm_mix.bias": np.zeros(256, np.float32),
                f"{engine}.proj_emit.weight": np.ones((256, 256), np.float32),
                f"{engine}.proj_gate.weight": np.ones((256, 256), np.float32),
            }
        )
    tensors.update(
        {
            f"{probe.TRANSITION_PREFIX}.norm.weight": np.ones(256, np.float32),
            f"{probe.TRANSITION_PREFIX}.norm.bias": np.zeros(256, np.float32),
            f"{probe.TRANSITION_PREFIX}.ffn.w12.weight": np.ones(
                (2048, 256), np.float32
            ),
            f"{probe.TRANSITION_PREFIX}.ffn.w3.weight": np.ones(
                (256, 1024), np.float32
            ),
        }
    )
    return tensors


@pytest.mark.parametrize("fail", [False, True])
def test_transition_patch_restores_original_on_success_and_failure(fail):
    def original(*args):
        return ("original", args[2])

    module = SimpleNamespace(_autocast_transition=original)
    expected = pytest.raises(RuntimeError) if fail else contextlib.nullcontext()
    with expected:
        with probe.patched_mapped_transition(module):
            if fail:
                raise RuntimeError("compile failed")
            assert module._autocast_transition(None, None, "other", True, 1e-5) == (
                "original",
                "other",
            )
    assert module._autocast_transition is original


def test_actual_pair_boundary_routes_only_candidate_through_mapped(monkeypatch):
    calls = []

    def original(value, _params, prefix, _residual, _eps):
        calls.append(("original", prefix))
        return value + "-original"

    def pair_update_block(value, params, prefix, *, mask, native_autocast):
        assert mask == "mask" and native_autocast is True
        return module._autocast_transition(
            value, params, f"{prefix}.pair_transition", True, 1e-5
        )

    module = SimpleNamespace(
        _autocast_transition=original, pair_update_block=pair_update_block
    )

    def mapped(value, params, prefix, *, residual, eps, original):
        calls.append(("mapped", prefix))
        assert residual is True and eps == 1e-5
        return value + "-mapped"

    monkeypatch.setattr(probe, "mapped_transition", mapped)
    assert probe.pair_block_call(module, "pair", {}, "mask") == "pair-original"
    with probe.patched_mapped_transition(module):
        assert probe.pair_block_call(module, "pair", {}, "mask") == "pair-mapped"
    assert calls == [
        ("original", probe.TRANSITION_PREFIX),
        ("mapped", probe.TRANSITION_PREFIX),
    ]


def test_candidate_compile_does_not_reuse_original_jax_trace(monkeypatch):
    import jax.numpy as jnp

    def original_transition(value, _params, _prefix, _residual, _eps):
        return value + 1

    def pair_update_block(value, params, prefix, *, mask, native_autocast):
        assert mask.shape == value.shape and native_autocast is True
        return module._autocast_transition(
            value, params, f"{prefix}.pair_transition", True, 1e-5
        )

    module = SimpleNamespace(
        _autocast_transition=original_transition,
        pair_update_block=pair_update_block,
    )

    def mapped(value, _params, _prefix, *, residual, eps, original):
        assert residual is True and eps == 1e-5
        return original(value, {}, probe.TRANSITION_PREFIX, residual, eps) + 1

    monkeypatch.setattr(probe, "mapped_transition", mapped)
    value = jnp.array([1.0])
    mask = jnp.ones((1,), jnp.float32)

    def original(value, params, mask):
        return probe.pair_block_call(module, value, params, mask)

    original_compiled, _ = probe._compile(original, value, {}, mask)
    with probe.patched_mapped_transition(module):

        def candidate(value, params, mask):
            return probe.pair_block_call(module, value, params, mask)

        candidate_compiled, _ = probe._compile(candidate, value, {}, mask)

    original_value, _ = probe._execute(original_compiled, value, {}, mask)
    candidate_value, _ = probe._execute(candidate_compiled, value, {}, mask)
    np.testing.assert_array_equal(original_value, np.array([2.0]))
    np.testing.assert_array_equal(candidate_value, np.array([3.0]))


def test_weight_loader_rejects_missing_required_leaf(tmp_path):
    from safetensors.numpy import save_file

    tensors = _weights()
    tensors.pop(f"{probe.TRANSITION_PREFIX}.ffn.w3.weight")
    path = tmp_path / "weights.safetensors"
    save_file(tensors, path)

    with pytest.raises(ValueError, match="missing"):
        probe.load_pair_block_weights(path)


def test_weight_loader_rejects_invalid_shape(tmp_path):
    from safetensors.numpy import save_file

    tensors = _weights()
    tensors[f"{probe.TRANSITION_PREFIX}.ffn.w3.weight"] = np.ones(
        (255, 1024), np.float32
    )
    path = tmp_path / "weights.safetensors"
    save_file(tensors, path)

    with pytest.raises(ValueError, match="transition w3"):
        probe.load_pair_block_weights(path)


@pytest.mark.parametrize(
    "rows,scale",
    [(0, 1.0), (128, 1.0), (129, 0.0), (2096, float("nan"))],
)
def test_input_argument_validation(rows, scale):
    with pytest.raises(ValueError):
        probe.validate_input_args(rows, scale)


@pytest.mark.parametrize("alias", ["same", "hardlink", "symlink"])
def test_output_path_rejects_checkpoint_alias(tmp_path, alias):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")
    output = weights if alias == "same" else tmp_path / "report.json"
    if alias == "hardlink":
        os.link(weights, output)
    elif alias == "symlink":
        output.symlink_to(weights)

    with pytest.raises(ValueError, match="--output"):
        probe.validate_output_path(weights, output)
