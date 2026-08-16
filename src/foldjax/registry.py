"""Lazy backend registry."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib import import_module

from foldjax.backends.base import Backend
from foldjax.schema import ModelCapabilities, ModelInfo, RuntimeInfo

BackendFactory = Callable[[], Backend]

_ALIASES = {
    "af3": "alphafold3",
    "alphafold-3": "alphafold3",
    "alphafold3": "alphafold3",
    "boltz": "boltz2",
    "boltz2": "boltz2",
    "boltz-jax": "boltz2",
    "open-dde": "opendde",
    "opendde": "opendde",
    "opendde-jax": "opendde",
    "protenix": "protenix",
    "protenix-jax": "protenix",
    "esmfold2": "esmfold2",
    "esm-fold2": "esmfold2",
    "esmfold-2": "esmfold2",
    "openfold3": "openfold3",
    "of3": "openfold3",
    "openfold-3": "openfold3",
    "openfold3-jax": "openfold3",
}
_IMPORTS = {
    "alphafold3": ("foldjax.backends.alphafold3", "AlphaFold3Backend"),
    "boltz2": ("foldjax.backends.boltz2", "Boltz2Backend"),
    "opendde": ("foldjax.backends.opendde", "OpenDDEBackend"),
    "protenix": ("foldjax.backends.protenix", "ProtenixBackend"),
    "openfold3": ("foldjax.backends.openfold3", "OpenFold3Backend"),
    "esmfold2": ("foldjax.backends.esmfold2", "ESMFold2Backend"),
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
    """One backend's capabilities, with the common-schema reach filled in.

    A backend describes itself; only the input layer knows which of those
    abilities a common job document can actually reach. Joining the two here
    keeps every backend's `capabilities()` unchanged and keeps the public answer
    from claiming a field the schema does not have.
    """
    import dataclasses

    from foldjax.input import common_schema_features, native_only_features

    described = get_backend(name).capabilities()
    return dataclasses.replace(
        described,
        common_schema_features=common_schema_features(described.model),
        native_only_features=native_only_features(described.model, described),
    )


def _runtime_info(name: str) -> RuntimeInfo:
    """Return non-mutating native-runtime readiness for one backend."""
    if name != "alphafold3":
        return RuntimeInfo(
            ready=True,
            setup=None,
            requires_network=False,
            notes="No generated runtime preparation is required.",
        )

    from foldjax.models.alphafold3 import build

    ready = build.is_ready()
    blocker = None if ready else build.runtime_blocker()
    if blocker is not None:
        return RuntimeInfo(
            ready=False,
            setup=blocker,
            requires_network=False,
            notes=(
                "An external alphafold3 native package is already imported in "
                "this process. FoldJAX will not replace a loaded extension; "
                "restart to select the managed runtime."
            ),
        )
    return RuntimeInfo(
        ready=ready,
        setup=None if ready else "foldjax runtime prepare --model alphafold3",
        requires_network=not ready,
        notes=(
            "FoldJAX always selects its vendored, ABI-specific C++ extension "
            "and CCD tables from the writable runtime store. Independently "
            "installed packages are not adopted implicitly; preparation "
            "fetches pinned build sources."
        ),
    )


def model_info(name: str) -> ModelInfo:
    """Return enough readiness detail to choose a backend before predicting."""
    from foldjax.assets import assets_for, profile_status

    backend = get_backend(name)
    spec = assets_for(backend.name)
    sizes = [item.size for item in spec.downloads]
    download_bytes = (
        sum(sizes) if sizes and all(size is not None for size in sizes) else None
    )
    ready = spec.ready()
    if ready:
        setup = None
    elif spec.downloads:
        setup = f"foldjax weights fetch --model {backend.name}"
    else:
        setup = spec.notes
    return ModelInfo(
        model=backend.name,
        capabilities=capabilities(backend.name),
        execution={
            knob: tuple(values)
            for knob, (_native, values) in backend.execution_options.items()
        },
        weights_ready=ready,
        weights_path=spec.native_path(),
        weights_source=spec.source,
        weights_licence=spec.licence,
        weights_fetchable=bool(spec.downloads),
        download_bytes=download_bytes,
        runtime=_runtime_info(backend.name),
        setup=setup,
        notes=spec.notes,
        weight_profiles=profile_status(backend.name),
    )


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
