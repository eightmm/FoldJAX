"""The ESMC loader, against a miniature checkpoint in the published layout.

The real file is 25.4 GB across six shards; what has to be right about reading
it is the layout, not the size. This builds the same layout small: an index,
two shards, the `esmc.` prefix the masked-LM wrapper adds, Transformer
Engine's `_extra_state` entries, and a head the structure model never reads.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from foldjax.models.esmfold2.bridge import esmc


@pytest.fixture
def checkpoint(tmp_path):
    first = {
        "esmc.embed.weight": np.ones((6, 4), dtype=np.float32),
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


def test_the_settings_come_from_the_checkpoints_config(checkpoint) -> None:
    settings = esmc.load_settings(checkpoint)
    assert (settings.d_model, settings.n_heads, settings.n_layers) == (4, 2, 1)


def test_a_directory_without_a_checkpoint_says_so(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="no safetensors"):
        esmc.shard_paths(tmp_path)
