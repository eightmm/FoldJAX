"""Reading the ESMC-6B checkpoint, which is sharded and carries TE leftovers.

Two differences from the structure weights. It ships as several safetensors
files behind a `model.safetensors.index.json`, and -- because upstream builds
it out of Transformer Engine modules when they are available -- it carries
`_extra_state` entries that are serialised Python objects rather than tensors.
Upstream drops them on load with a `_load_state_dict_pre_hook`; so does this.

At 6B parameters in bfloat16 the file is about 12 GB and in float32 about 24.
`dtype="bfloat16"` is the sensible default for a model whose output is fed
through a softmax over 81 layers and then a layer norm.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
from safetensors import safe_open

from foldjax.models.esmfold2.models.esmc import ESMCSettings, settings_from_config

INDEX_NAME = "model.safetensors.index.json"
WEIGHTS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"


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

    `_extra_state` entries are skipped: Transformer Engine writes its
    quantisation bookkeeping there, it is not a tensor, and nothing in this
    port reads it.
    """
    parameters: dict[str, jnp.ndarray] = {}
    for shard in shard_paths(directory):
        with safe_open(shard, framework="numpy") as handle:
            for name in handle.keys():  # noqa: SIM118 -- safetensors has no __iter__
                if name.endswith("_extra_state"):
                    continue
                array = handle.get_tensor(name)
                value = jnp.asarray(array) if to_device else array
                if dtype is not None:
                    value = value.astype(jnp.dtype(dtype))
                parameters[name] = value
    return parameters


__all__ = [
    "CONFIG_NAME",
    "INDEX_NAME",
    "WEIGHTS_NAME",
    "load_parameters",
    "load_settings",
    "shard_paths",
]
