"""Match prediction lowerings to trusted cache entries without loading them."""

from __future__ import annotations

import dataclasses
import os
import tempfile
from pathlib import Path
from typing import Any

from foldjax import cache, memory_policy
from foldjax._cache_tiers import open_directory
from foldjax.api import resolve_cache_dir, resolve_requests
from foldjax.cache_verify import _entry_trusted
from foldjax.execution import clear_chunking_record, recorded_chunking
from foldjax.registry import get_backend
from foldjax.schema import PredictionRequest


class UnsupportedCacheVerificationError(ValueError):
    """The prediction cannot expose its exact persistent-cache key safely."""


_LIMITATION = (
    "Exact lowered graph keys only: matching files are not deserialized or "
    "validated. This does not verify language-model staging graphs, kernel "
    "autotuning, all prediction executables, or successful execution. Input "
    "preprocessing and checkpoint reads may be required; model graphs are "
    "neither compiled nor executed."
)


def lowered_cache_key(lowered: Any) -> str:
    """Use the installed JAX compiler's HLO/options/device cache-key pipeline.

    JAX exposes lowering publicly, but persistent-key preparation remains
    internal. Refuse unsupported lowering/profiling variants instead of
    approximating a key from text or guessing compilation options.
    """
    import numpy as np
    from jax._src import compilation_cache, config
    from jax._src.interpreters import pxla

    computation = getattr(lowered, "_lowering", None)
    if not isinstance(computation, pxla.MeshComputation):
        raise UnsupportedCacheVerificationError(
            "unsupported JAX lowering implementation"
        )
    if config.enable_pgle.value or config.compilation_cache_expect_pgle.value:
        raise UnsupportedCacheVerificationError(
            "PGLE cache verification is unsupported"
        )
    arguments = computation.compile_args
    if arguments["host_callbacks"]:
        raise UnsupportedCacheVerificationError("host-callback graphs are unsupported")
    shape_poly = arguments.get("shape_poly_state")
    if shape_poly is not None and shape_poly.uses_dim_vars:
        raise UnsupportedCacheVerificationError(
            "shape-polymorphic graphs are unsupported"
        )
    devices = computation._device_list
    if devices is None or len(devices) != arguments["num_devices"]:
        raise UnsupportedCacheVerificationError(
            "lowering has no concrete device assignment"
        )
    # These are the same transformations used by UnloadedMeshExecutable.from_hlo
    # and _cached_compilation before compiler.compile_or_get_cached is called.
    inputs = tuple(
        pxla.maybe_concretize_mesh(sharding, devices)
        for sharding in arguments["in_shardings"]
    )
    outputs = tuple(
        pxla.maybe_concretize_mesh(sharding, devices)
        for sharding in arguments["out_shardings"]
    )
    allow_inputs, allow_outputs = pxla.get_prop_to_input_output(
        inputs, outputs, len(arguments["ordered_effects"])
    )
    device_array = np.asarray(tuple(devices), dtype=object)
    options = pxla.create_compile_options(
        computation._hlo,
        arguments["tuple_args"],
        allow_inputs,
        allow_outputs,
        arguments["backend"],
        device_array,
        dict(computation._compiler_options_kvs),
    )
    key = compilation_cache.get_cache_key(
        computation._hlo,
        device_array,
        options,
        arguments["backend"],
        config.remove_custom_partitioning_ptr_from_cache_key.value,
    )
    if not key or Path(key).name != key or "\\" in key or "\0" in key:
        raise UnsupportedCacheVerificationError("JAX returned an unsafe cache key")
    return key


def _inspect_key(path: Path, key: str, *, private: bool) -> dict[str, Any]:
    from jax._src.lru_cache import _CACHE_SUFFIX

    result: dict[str, Any] = {"path": str(path), "status": "missing"}
    if not private:
        reason = cache.compile_cache_untrusted_reason(path)
        if reason is not None:
            return {**result, "status": "untrusted", "reason": reason}
    try:
        with open_directory(path, private=private) as directory:
            info = os.stat(
                f"{key}{_CACHE_SUFFIX}", dir_fd=directory, follow_symlinks=False
            )
            result["status"] = (
                "key_present" if _entry_trusted(info, private=private) else "untrusted"
            )
    except FileNotFoundError:
        pass
    except OSError as error:
        result.update(status="unavailable", reason=str(error))
    return result


def _prepare_lowering_request(
    request: PredictionRequest, backend: Any, scratch: Path
) -> PredictionRequest:
    from foldjax import _pallas_gpu
    from foldjax.input import (
        IGNORE_CONSTRAINTS,
        IGNORE_NUCLEIC_MSA,
        IGNORE_TEMPLATES,
        materialize_native_input,
    )
    from foldjax.pocket_selection import POCKET_SAMPLING
    from foldjax.pocket_selection import requested as pocket_selection_requested

    _pallas_gpu.install()
    backend.validate_request(request)
    capabilities = backend.capabilities()
    request = dataclasses.replace(request, output_dir=scratch / "outputs")
    if request.input_format == "foldjax":
        pocket_conditioning = (
            backend.pocket_conditioning(request)
            if pocket_selection_requested(request.options) == "select"
            else None
        )
        native_input = materialize_native_input(
            request.input,
            capabilities,
            scratch / "inputs",
            seed=request.seed,
            msa=request.msa,
            options=backend.apply_sampling(request),
            templates=request.templates,
            template_max_date=request.template_max_date,
            msa_pairing=request.msa_pairing,
            template_dir=request.template_dir,
            pocket_conditioning=pocket_conditioning,
        )
        request = dataclasses.replace(
            request,
            input=native_input,
            input_format=(
                "native" if "native" in capabilities.input_formats else "foldjax"
            ),
        )
    if request.input_format not in capabilities.input_formats:
        raise ValueError(
            f"{backend.name} does not support input format {request.input_format!r}"
        )
    consumed = {
        IGNORE_NUCLEIC_MSA,
        IGNORE_TEMPLATES,
        IGNORE_CONSTRAINTS,
        POCKET_SAMPLING,
    }
    if consumed & set(request.options):
        request = dataclasses.replace(
            request,
            options={
                key: value
                for key, value in request.options.items()
                if key not in consumed
            },
        )
    return request


def _verify_resolved(request: PredictionRequest, *, scratch: Path) -> dict[str, Any]:
    import jax
    from jax._src import config

    from foldjax._gpu import default_device, require_supported_gpu
    from foldjax._tokamax_device import selected_tokamax_device
    from foldjax.search.msa import CachedMsaUnavailableError, cached_msa_only

    clear_chunking_record()
    backend = get_backend(request.model)
    row: dict[str, Any] = {"model": backend.name, "input": str(request.input)}
    lower = getattr(backend, "lower_cache_graphs", None)
    if not callable(lower):
        return {
            **row,
            "status": "unsupported",
            "reason": f"{backend.name} has no prediction lowering-only adapter",
        }
    device = default_device(request)
    require_supported_gpu(device)
    graphs = []
    try:
        with (
            jax.default_device(device),
            selected_tokamax_device(device),
            memory_policy.selected_device_scope(device),
            # Prediction's layered cache disables XLA's auxiliary caches too.
            # Keep option preparation identical without creating cache paths or
            # initializing/deserializing JAX's persistent CacheInterface.
            config.persistent_cache_enable_xla_caches("none"),
            cached_msa_only(),
            backend.session((request,)) as active_backend,
        ):
            prepared = _prepare_lowering_request(request, active_backend, scratch)
            namespace = resolve_cache_dir(prepared, active_backend)
            private = resolve_cache_dir(
                dataclasses.replace(
                    prepared, cache_dir=cache.private_compile_cache_root()
                ),
                active_backend,
            )
            row.update(
                cache_dir=str(namespace),
                graph_revision=cache.cache_revision(active_backend.name),
            )
            prepared = dataclasses.replace(prepared, cache_dir=namespace)
            for name, lowered in active_backend.lower_cache_graphs(prepared):
                key = lowered_cache_key(lowered)
                tiers = {"shared": _inspect_key(namespace, key, private=False)}
                if os.path.abspath(private) != os.path.abspath(namespace):
                    tiers["private"] = _inspect_key(private, key, private=True)
                found = next(
                    (
                        tier
                        for tier, info in tiers.items()
                        if info["status"] == "key_present"
                    ),
                    None,
                )
                graphs.append(
                    {
                        "graph": name,
                        "cache_key": key,
                        "status": "key_present" if found else "missing",
                        "source": found,
                        "tiers": tiers,
                    }
                )
            chunking = recorded_chunking()
            if chunking is not None:
                row["resolved_chunking"] = chunking
    except UnsupportedCacheVerificationError as error:
        return {**row, "status": "unsupported", "reason": str(error), "graphs": graphs}
    except CachedMsaUnavailableError as error:
        raise ValueError(
            "lowering-only verification requires existing valid MSA cache entries; "
            f"it never searches or repairs alignments: {error}. Run predict or "
            "MSA prefetch first, or supply local alignments"
        ) from error
    return {
        **row,
        "status": (
            "key_present"
            if graphs and all(graph["status"] == "key_present" for graph in graphs)
            else "missing"
        ),
        "graphs": graphs,
    }


def verify_prediction_cache(request: PredictionRequest) -> dict[str, Any]:
    """Lower supported prediction graphs for a real input and inspect exact keys."""
    if not request.use_compile_cache:
        raise ValueError("cache verification requires an enabled persistent cache")
    if request.options.get("use_msa_server") is True:
        raise ValueError(
            "lowering-only verification does not run a native MSA server search; "
            "use --msa auto with existing cached alignments, or supply local "
            "alignments and omit use_msa_server"
        )
    if request.templates in {"auto", "required"} and request.template_dir is None:
        raise ValueError(
            "lowering-only verification does not search public templates; "
            "supply local templates or use --templates none"
        )
    with tempfile.TemporaryDirectory(prefix="foldjax-prediction-verify-") as temporary:
        resolved = resolve_requests(
            request, draw_seeds=False, jobs_root=Path(temporary)
        )
        # An unspecified random seed cannot affect a graph key, but preprocessing
        # still needs a concrete seed. Report the chosen probe seed explicitly.
        rows = []
        for index, item in enumerate(resolved):
            seed = item.seeds[0] if item.seeds else item.seed
            probe = dataclasses.replace(
                item, seed=0 if seed is None else seed, seeds=None, num_seeds=None
            )
            rows.append(
                {
                    **_verify_resolved(probe, scratch=Path(temporary) / f"job-{index}"),
                    "seed": probe.seed,
                }
            )
    return {
        "strategy": "prediction_lowering",
        "status": (
            "key_present"
            if rows and all(row["status"] == "key_present" for row in rows)
            else "unsupported"
            if rows and all(row["status"] == "unsupported" for row in rows)
            else "missing"
        ),
        "limitation": _LIMITATION,
        "results": rows,
    }
