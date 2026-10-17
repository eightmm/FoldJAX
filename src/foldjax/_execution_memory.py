"""Release managed execution caches at completed-input boundaries."""

from __future__ import annotations

import sys
import warnings

from foldjax.models import _managed_memory

# These pools own JIT wrappers outside JAX's own caches. Inspect loaded modules
# only: cleanup must never initialize a model, CUDA, or an external AF3 runtime.
_CACHES = {
    "foldjax.models.openfold3.inference": ("_compiled_predict",),
    "foldjax.models.protenix.models.model": (
        "_compiled_protenix_infer",
        "_compiled_protenix_infer_deterministic",
    ),
    "foldjax.models.opendde.models.model": (
        "_compiled_opendde_infer",
        "_compiled_opendde_infer_deterministic",
    ),
    "foldjax.models.opendde.models.shape_complementarity": (
        "_compiled_shape_complementarity",
        "_compiled_shape_complementarity_deterministic",
    ),
    "foldjax.models.esmfold2.inference": (
        "compiled_predict",
        "_compiled_language_model_embedding",
    ),
    "alphafold3.model.pipeline.inter_chain_bonds": (
        "get_polymer_ligand_and_ligand_ligand_bonds",
        "get_ligand_ligand_bonds",
        "get_polymer_ligand_bonds",
    ),
    "alphafold3.constants.chemical_components": ("_load_ccd_pickle_cached",),
}


def warn_release_failure(subject: str, error: Exception) -> None:
    try:
        warnings.warn(
            f"releasing {subject} failed ({type(error).__name__}: {error}); "
            "a fresh process may be needed to reclaim memory",
            RuntimeWarning,
            stacklevel=3,
        )
    except Exception:  # noqa: BLE001 - warning filters must not replace a result
        pass


def release_execution_memory() -> None:
    """Drop compiler/tuning references, collect cycles, and trim freed host pages.

    Backend sessions invalidate their weight/runner trees first. Caller-owned
    result arrays remain valid. The JAX cache is process-wide, so unrelated
    concurrent workloads require another process. CUDA's preallocated pool and
    context belong to the process; only exiting a warm worker releases those.
    """
    from foldjax.cache import _CONFIG_LOCK

    with _CONFIG_LOCK:
        for module_name, names in _CACHES.items():
            module = sys.modules.get(module_name)
            if module is None:
                continue
            for name in names:
                owner = getattr(module, name, None)
                clear = getattr(owner, "cache_clear", None) or getattr(
                    owner, "clear_cache", None
                )
                if clear is not None:
                    try:
                        clear()
                    except Exception as error:  # noqa: BLE001 - best-effort cleanup
                        warn_release_failure(f"{module_name}.{name}", error)
        module = sys.modules.get("tokamax._src.ops.op")
        if module is not None:
            try:
                overlay = getattr(module, "AUTOTUNING_CACHE_OVERLAY_STACK", None)
                cache = getattr(module, "_AUTOTUNING_CACHE", None)
                # An embedding caller may own an outer tuning overlay. Never
                # replace its ContextVar or the associated JAX trace context.
                if cache is not None and (overlay is None or not overlay.get()):
                    cache.clear()
            except Exception as error:  # noqa: BLE001 - best-effort cleanup
                warn_release_failure("Tokamax tuning cache", error)
        jax = sys.modules.get("jax")
        if jax is not None:
            try:
                jax.clear_caches()
            except Exception as error:  # noqa: BLE001 - preserve prediction outcome
                warn_release_failure("JAX execution caches", error)
        for subject, reclaim in (
            ("unreachable model data", _managed_memory.gc.collect),
            ("freed host pages", _managed_memory._malloc_trim),
        ):
            try:
                reclaim()
            except Exception as error:  # noqa: BLE001 - preserve prediction outcome
                warn_release_failure(subject, error)
