"""Checkpoint loading and inspection.

These run without any real weights: a synthetic checkpoint exercises wrapper
stripping, nesting search, and fused/unfused layout inspection. Separate mapper
gates cover both triangular-multiplication layouts.
"""

from __future__ import annotations

import builtins
from pathlib import Path

import numpy as np
import pytest

from foldjax.models.openfold3.bridge.checkpoint import (
    count_blocks,
    describe,
    detect_fused_tri_mul,
    iter_shapes,
    load_checkpoint,
    strip_wrapper_prefixes,
    unwrap_state_dict,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("module.trunk.linear.weight", "trunk.linear.weight"),
        ("model.module.a.b", "a.b"),
        ("_orig_mod.ema.x.y", "x.y"),
        ("trunk.linear.weight", "trunk.linear.weight"),
        # "model_something" is not a prefix and must survive.
        ("model_head.weight", "model_head.weight"),
    ],
)
def test_strip_wrapper_prefixes(raw: str, expected: str) -> None:
    assert strip_wrapper_prefixes(raw) == expected


def test_unwrap_finds_a_flat_mapping() -> None:
    flat = {"a.b": 1, "c.d": 2}
    assert unwrap_state_dict(flat) is flat


@pytest.mark.parametrize("wrapper", ["state_dict", "model", "module", "params"])
def test_unwrap_looks_inside_common_wrappers(wrapper: str) -> None:
    inner = {"a.b": 1}
    assert unwrap_state_dict({wrapper: inner, "epoch": 3}) is inner


def test_unwrap_refuses_to_guess() -> None:
    with pytest.raises(KeyError, match="could not find a parameter mapping"):
        unwrap_state_dict({"epoch": 3, "optimizer": {}})


def test_unwrap_rejects_a_non_mapping() -> None:
    with pytest.raises(TypeError, match="not a mapping"):
        unwrap_state_dict([1, 2, 3])


def _synthetic() -> dict[str, np.ndarray]:
    return {
        "trunk.pairformer.blocks.0.pair_stack.tri_mul_out.linear_a_p.weight": np.zeros(
            (4, 4), dtype=np.float32
        ),
        "trunk.pairformer.blocks.0.pair_stack.tri_mul_out.linear_b_p.weight": np.zeros(
            (4, 4), dtype=np.float32
        ),
        "trunk.msa_module.blocks.0.outer_product_mean.linear_1.weight": np.zeros(
            (3, 4), dtype=np.float32
        ),
        "diffusion_module.atom_attn_enc.linear_q.0.weight": np.zeros(
            (2, 4), dtype=np.float32
        ),
    }


def test_describe_summarizes_top_level_structure() -> None:
    counts = describe(_synthetic(), depth=1)
    assert counts == {"diffusion_module": 1, "trunk": 3}
    deeper = describe(_synthetic(), depth=2)
    assert deeper["trunk.pairformer"] == 2
    assert deeper["trunk.msa_module"] == 1


def test_iter_shapes_filters_and_sorts() -> None:
    pairs = list(iter_shapes(_synthetic(), "tri_mul_out"))
    assert [key.split(".")[-2] for key in (k for k, _ in pairs)] == [
        "linear_a_p",
        "linear_b_p",
    ]
    assert all(shape == (4, 4) for _key, shape in pairs)


def test_counts_blocks_per_stack() -> None:
    """The depth a checkpoint was produced at, read off its keys.

    Distinct indices rather than matching keys: every block contributes many
    tensors, so counting keys would report the tensor count per stack.
    """
    state = {
        "trunk.pairformer_stack.blocks.0.tri_mul_out.linear_a_p.weight": np.zeros(1),
        "trunk.pairformer_stack.blocks.0.tri_mul_out.linear_b_p.weight": np.zeros(1),
        "trunk.pairformer_stack.blocks.1.tri_mul_out.linear_a_p.weight": np.zeros(1),
        "trunk.msa_module.blocks.0.outer_product_mean.linear_1.weight": np.zeros(1),
    }

    assert count_blocks(state, "pairformer_stack.blocks") == 2
    assert count_blocks(state, "msa_module.blocks") == 1
    assert count_blocks(state, "template_pair_stack.blocks") is None


def test_detects_the_unfused_layout() -> None:
    assert detect_fused_tri_mul(_synthetic()) is False


def test_detects_the_fused_layout() -> None:
    fused = {"trunk.pairformer.blocks.0.tri_mul_out.linear_ab_p.weight": np.zeros(1)}
    assert detect_fused_tri_mul(fused) is True


def test_reports_unknown_when_neither_layout_appears() -> None:
    assert detect_fused_tri_mul({"trunk.linear.weight": np.zeros(1)}) is None


def test_load_safetensors_round_trip(tmp_path: Path) -> None:
    from safetensors.numpy import save_file

    path = tmp_path / "weights.safetensors"
    save_file({"module.trunk.linear.weight": np.ones((2, 3), dtype=np.float32)}, path)
    loaded = load_checkpoint(path)
    # The wrapper prefix is stripped on load.
    assert set(loaded) == {"trunk.linear.weight"}
    np.testing.assert_allclose(loaded["trunk.linear.weight"], np.ones((2, 3)))


def test_load_refuses_keys_that_collide_after_prefix_stripping(
    tmp_path: Path,
) -> None:
    from safetensors.numpy import save_file

    path = tmp_path / "ambiguous.safetensors"
    save_file(
        {
            "trunk.linear.weight": np.zeros((2, 3), dtype=np.float32),
            "model.module.trunk.linear.weight": np.ones((2, 3), dtype=np.float32),
        },
        path,
    )

    with pytest.raises(ValueError, match="collide.*trunk.linear.weight"):
        load_checkpoint(path)


def test_load_torch_checkpoint_round_trip(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "weights.pt"
    torch.save({"state_dict": {"module.a.weight": torch.ones(2, 2), "epoch": 1}}, path)
    loaded = load_checkpoint(path)
    assert "a.weight" in loaded
    np.testing.assert_allclose(loaded["a.weight"], np.ones((2, 2)))


def test_unwrap_and_load_prefer_ema_params_like_upstream(tmp_path: Path) -> None:
    """Upstream ``get_state_dict_from_checkpoint(init_from_ema_weights=True)``."""
    raw_state = {"model.trunk.linear.weight": np.zeros((2, 2), dtype=np.float32)}
    ema_state = {"trunk.linear.weight": np.full((2, 2), 3.0, dtype=np.float32)}
    payload = {
        "state_dict": raw_state,
        "module": raw_state,
        "ema": {"decay": 0.999, "params": ema_state},
    }
    assert unwrap_state_dict(payload) is ema_state
    assert unwrap_state_dict(payload, use_ema=False) is raw_state

    # Empty or malformed EMA records fall through to ordinary wrapper keys,
    # and raise KeyError when no usable parameter mapping exists anywhere.
    assert (
        unwrap_state_dict({"state_dict": raw_state, "ema": {"params": {}}}) is raw_state
    )
    assert (
        unwrap_state_dict({"state_dict": raw_state, "ema": {"params": "bad"}})
        is raw_state
    )
    assert unwrap_state_dict({"state_dict": raw_state, "ema": None}) is raw_state
    with pytest.raises(KeyError, match="could not find a parameter mapping"):
        unwrap_state_dict({"ema": {"params": {}}})

    torch = pytest.importorskip("torch")
    path = tmp_path / "ema_training.ckpt"
    torch.save(
        {
            "state_dict": {"model.a.weight": torch.zeros(2, 2)},
            "ema": {"decay": 0.999, "params": {"a.weight": torch.full((2, 2), 5.0)}},
        },
        path,
    )
    np.testing.assert_allclose(load_checkpoint(path)["a.weight"], np.full((2, 2), 5.0))
    np.testing.assert_allclose(
        load_checkpoint(path, use_ema=False)["a.weight"], np.zeros((2, 2))
    )


def test_load_torch_checkpoint_never_imports_torch(monkeypatch) -> None:
    fixture = Path(__file__).parents[2] / "data" / "torch_archive_fixture.pt"
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("prediction checkpoint loading imported torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    loaded = load_checkpoint(fixture)

    assert set(loaded) == {"w", "h", "b", "i", "flag", "scalar", "view"}
    assert loaded["w"].shape == (3, 5)
