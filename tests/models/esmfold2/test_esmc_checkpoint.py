"""The ESMC loader, against a miniature checkpoint in the published layout.

The real file is 25.4 GB across six shards; what has to be right about reading
it is the layout, not the size. This builds the same layout small: an index,
two shards, the `esmc.` prefix the masked-LM wrapper adds, Transformer
Engine's `_extra_state` entries, and a head the structure model never reads.
"""

from __future__ import annotations

import json
from typing import Any

import jax
import numpy as np
import pytest
from safetensors.numpy import save_file

from foldjax.models.esmfold2.bridge import esmc


@pytest.fixture
def checkpoint(tmp_path):
    first = {
        "esmc.embed.weight": (
            np.arange(24, dtype=np.float32).reshape(6, 4) / np.float32(7.0)
        ),
        "esmc.transformer.blocks.0.attn.q_ln.weight": np.ones(4, dtype=np.float32),
        # Transformer Engine's bookkeeping: a pickled object, not a tensor.
        "esmc.transformer.blocks.0.attn.layernorm_qkv._extra_state": np.zeros(
            2, dtype=np.uint8
        ),
    }
    second = {
        "esmc.transformer.norm.weight": np.full(4, 2.0, dtype=np.float32),
        # The masked-LM head, which the structure model never reads.
        "lm_head.weight": np.zeros((6, 4), dtype=np.float32),
    }
    save_file(first, tmp_path / "model-00001-of-00002.safetensors")
    save_file(second, tmp_path / "model-00002-of-00002.safetensors")
    (tmp_path / esmc.INDEX_NAME).write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {
                    **dict.fromkeys(first, "model-00001-of-00002.safetensors"),
                    **dict.fromkeys(second, "model-00002-of-00002.safetensors"),
                },
            }
        )
    )
    (tmp_path / esmc.CONFIG_NAME).write_text(
        json.dumps({"d_model": 4, "n_heads": 2, "n_layers": 1, "vocab_size": 6})
    )
    return tmp_path


def test_every_shard_of_the_index_is_read(checkpoint) -> None:
    assert len(esmc.shard_paths(checkpoint)) == 2
    parameters = esmc.load_parameters(checkpoint, dtype=None)
    assert "embed.weight" in parameters
    assert "transformer.norm.weight" in parameters


def test_the_wrapper_prefix_and_its_head_are_dropped(checkpoint) -> None:
    """The encoder spells its parameters without `esmc.`, and has no LM head."""
    parameters = esmc.load_parameters(checkpoint, dtype=None)
    assert not any(name.startswith("esmc.") for name in parameters)
    assert not any(name.startswith("lm_head") for name in parameters)


def test_transformer_engines_extra_state_is_skipped(checkpoint) -> None:
    """It is a pickled object rather than a tensor; loading it would raise."""
    parameters = esmc.load_parameters(checkpoint, dtype=None)
    assert not any(name.endswith("_extra_state") for name in parameters)


def test_the_dtype_cast_happens_on_load(checkpoint) -> None:
    parameters = esmc.load_parameters(checkpoint, dtype="bfloat16")
    assert str(parameters["embed.weight"].dtype) == "bfloat16"
    historical = jax.numpy.asarray(
        np.arange(24, dtype=np.float32).reshape(6, 4) / np.float32(7.0)
    ).astype(jax.numpy.bfloat16)
    np.testing.assert_array_equal(parameters["embed.weight"], historical)


def test_device_transfer_casts_on_host_without_staging(
    checkpoint, monkeypatch: Any
) -> None:
    original_device_put = esmc.jax.device_put
    transferred: list[np.ndarray] = []

    def spy_device_put(value: np.ndarray) -> jax.Array:
        transferred.append(value)
        return original_device_put(value)

    def forbid_asarray(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("ESMC parameter transfer used a staged jnp.asarray")

    monkeypatch.setattr(esmc.jax, "device_put", spy_device_put)
    monkeypatch.setattr(esmc.jnp, "asarray", forbid_asarray)

    parameters = esmc.load_parameters(checkpoint, dtype="bfloat16")

    assert len(transferred) == 3
    assert all(isinstance(value, np.ndarray) for value in transferred)
    assert all(str(value.dtype) == "bfloat16" for value in transferred)
    assert all(str(value.dtype) == "bfloat16" for value in parameters.values())


def test_host_only_load_never_transfers_a_parameter(
    checkpoint, monkeypatch: Any
) -> None:
    def forbid_device_put(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("host-only ESMC load transferred a parameter")

    monkeypatch.setattr(esmc.jax, "device_put", forbid_device_put)
    parameters = esmc.load_parameters(checkpoint, dtype="bfloat16", to_device=False)

    assert all(isinstance(value, np.ndarray) for value in parameters.values())
    assert all(str(value.dtype) == "bfloat16" for value in parameters.values())


def test_the_settings_come_from_the_checkpoints_config(checkpoint) -> None:
    settings = esmc.load_settings(checkpoint)
    assert (settings.d_model, settings.n_heads, settings.n_layers) == (4, 2, 1)


def test_a_directory_without_a_checkpoint_says_so(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="no safetensors"):
        esmc.shard_paths(tmp_path)
