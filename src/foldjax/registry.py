"""Lazy backend registry."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib import import_module

from foldjax.backends.base import Backend
from foldjax.schema import ModelCapabilities

BackendFactory = Callable[[], Backend]

_ALIASES = {
    "af3": "alphafold3",
    "alphafold-3": "alphafold3",
    "alphafold3": "alphafold3",
    "boltz": "boltz2",
    "boltz2": "boltz2",
    "boltz-jax": "boltz2",
    "chai": "chai",
    "chai1": "chai",
    "chai-1": "chai",
    "chai-jax": "chai",
    "open-dde": "opendde",
    "opendde": "opendde",
    "opendde-jax": "opendde",
    "protenix": "protenix",
    "protenix-jax": "protenix",
    "openfold3": "openfold3",
    "of3": "openfold3",
    "openfold-3": "openfold3",
    "openfold3-jax": "openfold3",
}
_IMPORTS = {
    "alphafold3": ("foldjax.backends.alphafold3", "AlphaFold3Backend"),
    "boltz2": ("foldjax.backends.boltz2", "Boltz2Backend"),
    "chai": ("foldjax.backends.chai", "ChaiBackend"),
    "opendde": ("foldjax.backends.opendde", "OpenDDEBackend"),
    "protenix": ("foldjax.backends.protenix", "ProtenixBackend"),
    "openfold3": ("foldjax.backends.openfold3", "OpenFold3Backend"),
}
_OVERRIDES: dict[str, BackendFactory] = {}


def available_models() -> tuple[str, ...]:
    return tuple(sorted(_IMPORTS))


def normalize_model_name(name: str) -> str:
    normalized = _ALIASES.get(name.strip().lower())
    if normalized is None:
        raise ValueError(
            f"unknown model {name!r}; choose one of {', '.join(available_models())}"
        )
    return normalized


def get_backend(name: str) -> Backend:
    normalized = normalize_model_name(name)
    if normalized in _OVERRIDES:
        return _OVERRIDES[normalized]()
    module_name, class_name = _IMPORTS[normalized]
    return getattr(import_module(module_name), class_name)()


def capabilities(name: str) -> ModelCapabilities:
    return get_backend(name).capabilities()


@contextmanager
def backend_override(name: str, factory: BackendFactory) -> Iterator[None]:
    normalized = normalize_model_name(name)
    previous = _OVERRIDES.get(normalized)
    _OVERRIDES[normalized] = factory
    try:
        yield
    finally:
        if previous is None:
            _OVERRIDES.pop(normalized, None)
        else:
            _OVERRIDES[normalized] = previous
