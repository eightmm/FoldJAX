import importlib
import json

import numpy as np
import pytest

from bench.boltz_historical_replay import digest
from bench.openbind_real_pair_replay import load_pair_capture, private_pair_operators


def capture(root, mask):
    arrays = {"z": np.zeros((1, 2, 2, 128), np.float32)}
    record = {"module": "pairformer_stack.blocks.0.pair_stack"}
    for side in ("input", "output"):
        path = root / f"first-pair-{side}.npz"
        np.savez(path, **arrays, **({"pair_mask": mask} if side == "input" else {}))
        record[side] = {"sha256": digest(path), "bytes": path.stat().st_size}
    (root / "trace.json").write_text(json.dumps({"first_pair_blocks": [record]}))


def test_real_pair_capture_is_bound_and_mask_checked(tmp_path):
    capture(tmp_path, np.ones((1, 2, 2), np.float32))
    z, mask, expected = load_pair_capture(tmp_path)
    assert z.shape == expected.shape == (1, 2, 2, 128)
    assert mask.shape == (1, 2, 2)
    np.savez(tmp_path / "first-pair-input.npz", z=np.zeros(1))
    with pytest.raises(ValueError, match="identity"):
        load_pair_capture(tmp_path)


@pytest.mark.parametrize("value", [0.5, np.nan])
def test_invalid_pair_mask_is_rejected_even_with_valid_hashes(tmp_path, value):
    capture(tmp_path, np.full((1, 2, 2), value, np.float32))
    with pytest.raises(ValueError, match="invalid real pair"):
        load_pair_capture(tmp_path)


def test_private_block_dispatch_restores_after_exception(monkeypatch):
    pytest.importorskip("jax")
    from types import SimpleNamespace

    from foldjax.models.openfold3.models import native_triangle_ops as ops

    block = importlib.import_module("foldjax.models.openfold3.models.pair_block")
    original = block.tri_mul_out_in, block.triangle_attention
    calls = []

    def residual(z, params, **kwargs):
        calls.append((z, params, kwargs["outgoing"]))
        return z + 10

    monkeypatch.setattr(ops, "native_triangle_multiplication_residual", residual)
    monkeypatch.setattr("bench.openbind_real_pair_replay.map_private_samples",
                        lambda fn, z, mask: fn(z, mask))
    with pytest.raises(RuntimeError, match="trace failure"):
        with private_pair_operators():
            params = SimpleNamespace(tri_mul_out="out", tri_mul_in="in")
            assert block.tri_mul_out_in(1, params, pair_mask=None) == 21
            assert calls == [(1, "out", True), (11, "in", False)]
            with pytest.raises(ValueError, match="four heads"):
                block.triangle_attention(None, None, no_heads=8, mask=None)
            raise RuntimeError("trace failure")
    assert (block.tri_mul_out_in, block.triangle_attention) == original


@pytest.mark.parametrize("batch", [1, 5])
def test_private_sample_mapping_preserves_samples_and_masks(batch):
    jax = pytest.importorskip("jax")
    from bench.openbind_real_pair_replay import map_private_samples

    z = np.arange(batch * 8, dtype=np.float32).reshape(batch, 2, 2, 2)
    mask = (z[..., 0] % 3 != 0).astype(np.float32)

    def one(x, m):
        assert x.shape == (1, 2, 2, 2)
        return x * m[..., None] + 7

    actual = jax.jit(lambda x, m: map_private_samples(one, x, m))(z, mask)
    np.testing.assert_array_equal(actual, z * mask[..., None] + 7)
