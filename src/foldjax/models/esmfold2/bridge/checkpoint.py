"""Reading upstream's released ESMFold2 checkpoint, without torch.

There is no name mapping in this file, and that is deliberate: every function
in `models/` spells its parameters exactly as upstream's `state_dict` does, so
the checkpoint loads by being read. The alternative -- a table of 1,594
renames -- is a second place for the port to be wrong, and one that no parity
test can see.

What the loader does have to do is read the *configuration*. The released
`config.json` differs from upstream's dataclass defaults in most of the fields
that matter -- 48 trunk layers against 24, three loops against twenty, fourteen
sampling steps against sixty-eight, and a diffusion churn of 1.003 where the
dataclass says zero -- so the settings always come from the file.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from safetensors import safe_open

from foldjax.models.esmfold2.models.model import ModelSettings, settings_from_config

#: What `assets.py` stages into the weight store, and what upstream publishes.
WEIGHTS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"


def load_config(directory: str | Path) -> dict[str, object]:
    """Upstream's `config.json`, as a plain mapping."""
    with Path(directory, CONFIG_NAME).open() as handle:
        return json.load(handle)


def load_settings(directory: str | Path) -> ModelSettings:
    """The released checkpoint's own settings, never this port's defaults."""
    return settings_from_config(load_config(directory))


def load_parameters(
    directory: str | Path, *, dtype: str | None = None, to_device: bool = True
) -> dict[str, jnp.ndarray]:
    """Every tensor in the checkpoint, keyed the way the model asks for it.

    `dtype` casts on load -- `"bfloat16"` halves the 940 MB the structure
    weights occupy -- and leaves the buffers alone, since `FourierEmbedding`'s
    frozen draws and the confidence head's distance boundaries are values
    rather than weights and rounding them changes the model's answer rather
    than its footprint.
    """
    buffers = {"boundaries", "w", "b"}
    parameters: dict[str, jnp.ndarray] = {}
    with safe_open(Path(directory, WEIGHTS_NAME), framework="numpy") as handle:
        for name in handle.keys():  # noqa: SIM118 -- safetensors has no __iter__
            array = jnp.asarray(handle.get_tensor(name)) if to_device else (
                handle.get_tensor(name)
            )
            if dtype is not None and name.rsplit(".", 1)[-1] not in buffers:
                array = array.astype(jnp.dtype(dtype))
            parameters[name] = array
    return parameters


def missing_parameters(
    parameters: Mapping[str, object], required: Mapping[str, object]
) -> list[str]:
    """Keys a run would ask for and the checkpoint does not have.

    Used by the loader's own test rather than at run time: a missing key raises
    where it is read, which names the module, and that is the more useful
    failure. This exists so the test can report all of them at once.
    """
    return sorted(name for name in required if name not in parameters)


def parameter_report(directory: str | Path) -> dict[str, object]:
    """A summary of what is in the file, for `foldjax weights` and for humans."""
    counts: dict[str, int] = {}
    total = 0
    with safe_open(Path(directory, WEIGHTS_NAME), framework="numpy") as handle:
        for name in handle.keys():  # noqa: SIM118
            slice_ = handle.get_slice(name)
            size = int(np.prod(slice_.get_shape())) if slice_.get_shape() else 1
            total += size
            counts[name.split(".", 1)[0]] = counts.get(name.split(".", 1)[0], 0) + size
    return {"parameters": total, "by_module": dict(sorted(counts.items()))}


__all__ = [
    "CONFIG_NAME",
    "WEIGHTS_NAME",
    "load_config",
    "load_parameters",
    "load_settings",
    "missing_parameters",
    "parameter_report",
]
