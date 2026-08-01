"""JAX/XLA inference port of Boltz-2.

Public API (lazily imported so importing this package stays cheap and torch-free):

- ``predict`` / ``featurize`` — high-level end-to-end inference (api.py).
- ``build_job_yaml`` — build a job YAML from bare sequences/ligands.
- ``boltz2_predict`` — low-level pure-JAX model fn (compose with other JAX code).
- ``load_params`` — load native (safetensors) weights into a JAX pytree.
"""

from typing import TYPE_CHECKING

__all__ = [
    "__version__",
    "predict",
    "featurize",
    "build_job_yaml",
    "boltz2_predict",
    "load_params",
]

__version__ = "0.1.0"

_LAZY = {
    "predict": ("foldjax.models.boltz2.api", "predict"),
    "featurize": ("foldjax.models.boltz2.api", "featurize"),
    "build_job_yaml": ("foldjax.models.boltz2.data.job_yaml", "build_job_yaml"),
    "boltz2_predict": ("foldjax.models.boltz2.models.predict", "boltz2_predict"),
    "load_params": ("foldjax.models.boltz2.bridge.native", "load_params"),
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module 'foldjax.models.boltz2' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # static-analysis hints only; not executed at runtime
    from foldjax.models.boltz2.api import featurize, predict
    from foldjax.models.boltz2.bridge.native import load_params
    from foldjax.models.boltz2.data.job_yaml import build_job_yaml
    from foldjax.models.boltz2.models.predict import boltz2_predict
