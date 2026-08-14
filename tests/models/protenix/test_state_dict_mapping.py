from __future__ import annotations

import numpy as np
import pytest

from foldjax import torch_archive
from foldjax.models.protenix.bridge import torch_mapping
from foldjax.models.protenix.bridge.torch_mapping import (
    map_adaptive_layer_norm_state_dict,
    map_constraint_embedder_state_dict,
    map_distogram_state_dict,
    map_fourier_state_dict,
    map_layer_norm_state_dict,
    map_linear_state_dict,
    map_transition_state_dict,
    require_key,
)


def test_map_linear_state_dict_keeps_torch_weight_layout() -> None:
    state = {
        "linear.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "linear.bias": np.array([0.5, -0.5], dtype=np.float32),
    }

    params = map_linear_state_dict(state, "linear")

    np.testing.assert_array_equal(np.asarray(params.weight), state["linear.weight"])
    np.testing.assert_array_equal(np.asarray(params.bias), state["linear.bias"])


def test_map_distogram_state_dict_supports_prefix() -> None:
    state = {
        "distogram_head.linear.weight": np.ones((4, 3), dtype=np.float32),
        "distogram_head.linear.bias": np.zeros((4,), dtype=np.float32),
    }

    params = map_distogram_state_dict(state, "distogram_head")

    assert tuple(params.linear.weight.shape) == (4, 3)
    assert tuple(params.linear.bias.shape) == (4,)


def test_map_layer_norm_state_dict_supports_optional_params() -> None:
    state = {
        "norm.weight": np.ones((3,), dtype=np.float32),
        "norm.bias": np.zeros((3,), dtype=np.float32),
    }

    params = map_layer_norm_state_dict(state, "norm")
    no_params = map_layer_norm_state_dict(state, "norm", scale=False, offset=False)

    assert tuple(params.weight.shape) == (3,)
    assert tuple(params.bias.shape) == (3,)
    assert no_params.weight is None
    assert no_params.bias is None


def test_map_transition_state_dict_uses_protenix_key_names() -> None:
    state = {
        "transition.layernorm1.weight": np.ones((4,), dtype=np.float32),
        "transition.layernorm1.bias": np.zeros((4,), dtype=np.float32),
        "transition.linear_no_bias_a.weight": np.ones((8, 4), dtype=np.float32),
        "transition.linear_no_bias_b.weight": np.ones((8, 4), dtype=np.float32),
        "transition.linear_no_bias.weight": np.ones((4, 8), dtype=np.float32),
    }

    params = map_transition_state_dict(state, "transition")

    assert tuple(params.layer_norm.weight.shape) == (4,)
    assert params.linear_a.bias is None
    assert tuple(params.linear_a.weight.shape) == (8, 4)
    assert tuple(params.linear_b.weight.shape) == (8, 4)
    assert tuple(params.linear_out.weight.shape) == (4, 8)


def test_map_adaptive_layer_norm_state_dict_uses_protenix_key_names() -> None:
    state = {
        "adln.layernorm_s.weight": np.ones((4,), dtype=np.float32),
        "adln.linear_s.weight": np.ones((5, 4), dtype=np.float32),
        "adln.linear_s.bias": np.zeros((5,), dtype=np.float32),
        "adln.linear_nobias_s.weight": np.ones((5, 4), dtype=np.float32),
    }

    params = map_adaptive_layer_norm_state_dict(state, "adln")

    assert params.layernorm_a.weight is None
    assert params.layernorm_a.bias is None
    assert tuple(params.layernorm_s.weight.shape) == (4,)
    assert params.layernorm_s.bias is None
    assert tuple(params.linear_s.weight.shape) == (5, 4)
    assert tuple(params.linear_s.bias.shape) == (5,)
    assert params.linear_no_bias_s.bias is None


def test_map_fourier_state_dict_uses_protenix_key_names() -> None:
    state = {
        "fourier.w": np.ones((6,), dtype=np.float32),
        "fourier.b": np.zeros((6,), dtype=np.float32),
    }

    params = map_fourier_state_dict(state, "fourier")

    assert tuple(params.w.shape) == (6,)
    assert tuple(params.b.shape) == (6,)


def test_map_constraint_embedder_state_dict_supports_optional_paths() -> None:
    state = {
        "constraint_embedder.pocket_z_embedder.weight": np.ones(
            (4, 1),
            dtype=np.float32,
        ),
        "constraint_embedder.contact_z_embedder.weight": np.ones(
            (4, 2),
            dtype=np.float32,
        ),
        "constraint_embedder.substructure_z_embedder.network.0.weight": np.ones(
            (8, 3),
            dtype=np.float32,
        ),
        "constraint_embedder.substructure_z_embedder.network.3.weight": np.ones(
            (4, 8),
            dtype=np.float32,
        ),
    }

    params = map_constraint_embedder_state_dict(state, "constraint_embedder")
    absent = map_constraint_embedder_state_dict({}, "constraint_embedder")

    assert tuple(params.pocket_z.weight.shape) == (4, 1)
    assert tuple(params.contact_z.weight.shape) == (4, 2)
    assert params.contact_atom_z is None
    assert len(params.substructure_z.layers) == 2
    assert tuple(params.substructure_z.layers[0].weight.shape) == (8, 3)
    assert absent.pocket_z is None
    assert absent.contact_z is None
    assert absent.contact_atom_z is None
    assert absent.substructure_z is None


def test_require_key_reports_missing_checkpoint_key() -> None:
    with pytest.raises(KeyError, match="missing checkpoint key: absent.weight"):
        require_key({}, "absent.weight")


def test_checkpoint_conversion_uses_the_restricted_archive_reader(
    tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "protenix.pt"
    checkpoint.write_bytes(b"trusted fixture")
    loaded = {"model": {"module.weight": np.asarray([1.0])}}
    seen = []
    sentinel = object()

    def fake_load(path):
        seen.append(path)
        return loaded

    monkeypatch.setattr(torch_archive, "load", fake_load)
    monkeypatch.setattr(
        torch_mapping,
        "map_protenix_inference_state_dict",
        lambda state: sentinel if set(state) == {"weight"} else None,
    )

    assert torch_mapping.load_torch_checkpoint(checkpoint) is sentinel
    assert seen == [checkpoint]
