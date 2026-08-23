"""Reading the ESMC-6B checkpoint, which is sharded, prefixed and TE-flavoured.

Three differences from the structure weights, all of them in the file rather
than the model:

* it ships as six safetensors behind a `model.safetensors.index.json`;
* it is published as `ESMCForMaskedLM`, so every key carries an `esmc.`
  prefix that the encoder itself does not use. Stripped on load, because the
  alternative is spelling the prefix at 1,048 call sites;
* because upstream builds it out of Transformer Engine modules when they are
  available, it carries `_extra_state` entries that are pickled Python objects
  rather than tensors. Upstream drops them with a `_load_state_dict_pre_hook`;
  so does this.

The published file is float32 and 25.4 GB. `dtype="bfloat16"` is the default
here and halves what it occupies: the output is fed through a softmax over 81
layers and then a layer norm, which is not a place where the last eight bits
of mantissa decide anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors import safe_open

from foldjax.models.esmfold2.models.esmc import ESMCSettings, settings_from_config

INDEX_NAME = "model.safetensors.index.json"
WEIGHTS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"
#: The masked-LM wrapper's submodule name, which the encoder does not use.
CHECKPOINT_PREFIX = "esmc."


def shard_paths(directory: str | Path) -> list[Path]:
    """Every safetensors file of the checkpoint, index or single file."""
    directory = Path(directory)
    index = directory / INDEX_NAME
    if index.exists():
        with index.open() as handle:
            mapping = json.load(handle)["weight_map"]
        return [directory / name for name in sorted(set(mapping.values()))]
    single = directory / WEIGHTS_NAME
    if single.exists():
        return [single]
    shards = sorted(directory.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors checkpoint under {directory}")
    return shards


def load_settings(directory: str | Path) -> ESMCSettings:
    """The checkpoint's own width, head count and depth."""
    with Path(directory, CONFIG_NAME).open() as handle:
        return settings_from_config(json.load(handle))


def load_parameters(
    directory: str | Path, *, dtype: str | None = "bfloat16", to_device: bool = True
) -> dict[str, jnp.ndarray]:
    """Every tensor in the checkpoint, keyed the way `esmc.encode` asks.

    The `esmc.` prefix of the masked-LM wrapper is stripped, and the masked-LM
    head itself -- which the structure model never reads -- is dropped with it.
    `_extra_state` entries go too: Transformer Engine writes its quantisation
    bookkeeping there, it is not a tensor, and nothing here reads it.
    """
    parameters: dict[str, jnp.ndarray] = {}
    for shard in shard_paths(directory):
        with safe_open(shard, framework="numpy") as handle:
            for name in handle.keys():  # noqa: SIM118 -- safetensors has no __iter__
                if name.endswith("_extra_state"):
                    continue
                key = name.removeprefix(CHECKPOINT_PREFIX)
                if not (key.startswith("transformer.") or key == "embed.weight"):
                    continue
                array = handle.get_tensor(name)
                if dtype is not None:
                    # Cast while this is still a host array. The published ESMC
                    # checkpoint is 25.4 GB of float32 and normally loads as
                    # bfloat16; transferring first briefly stages the full-width
                    # leaf on device and builds a conversion executable for each
                    # distinct shape.
                    array = array.astype(jnp.dtype(dtype))
                value = jax.device_put(array) if to_device else array
                parameters[key] = value
    return parameters


__all__ = [
    "CONFIG_NAME",
    "INDEX_NAME",
    "WEIGHTS_NAME",
    "load_parameters",
    "load_settings",
    "shard_paths",
]
