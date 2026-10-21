"""Inspect expected cache namespaces without lowering or deserializing code."""

from __future__ import annotations

import dataclasses
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from foldjax import cache
from foldjax._cache_tiers import open_directory
from foldjax.api import resolve_cache_dir, resolve_requests
from foldjax.registry import get_backend, normalize_model_name
from foldjax.schema import PredictionRequest

_LIMITATION = (
    "Namespace presence only: entries are not deserialized or matched against "
    "current HLO. This does not prove complete profile coverage, cache hits, "
    "or kernel-tuning coverage."
)


def _entry_trusted(info: os.stat_result, *, private: bool) -> bool:
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size < 1:
        return False
    if private:
        return info.st_uid == os.geteuid() and not info.st_mode & 0o077
    if cache.shared_compile_cache_trusted():
        return True
    return info.st_uid in (os.geteuid(), 0) and not (
        info.st_mode & 0o002
        or info.st_mode & 0o020
        and not cache._group_is_only_this_user(info.st_gid)
    )


def _inspect_tier(path: Path, *, private: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "status": "missing",
        "entries": 0,
        "rejected_entries": 0,
    }
    if not private:
        reason = cache.compile_cache_untrusted_reason(path)
        if reason is not None:
            return {**result, "status": "untrusted", "reason": reason}
    try:
        # The prediction cache uses this same fd walk. Never create a missing
        # tier, follow an entry symlink, or open executable contents here.
        with open_directory(path, private=private) as directory:
            for name in os.listdir(directory):
                if name == "-cache" or not name.endswith("-cache"):
                    continue
                try:
                    info = os.stat(name, dir_fd=directory, follow_symlinks=False)
                except FileNotFoundError:
                    # Concurrent eviction does not hide other extant entries.
                    continue
                if _entry_trusted(info, private=private):
                    result["entries"] += 1
                else:
                    result["rejected_entries"] += 1
            result["status"] = (
                "entries_present" if result["entries"] else "missing_entries"
            )
    except FileNotFoundError:
        pass
    except OSError as error:
        result.update(status="unavailable", reason=str(error))
    return result


def _verify_resolved(request: PredictionRequest) -> dict[str, Any]:
    backend = get_backend(request.model)
    namespace = resolve_cache_dir(request, backend)
    private = resolve_cache_dir(
        dataclasses.replace(request, cache_dir=cache.private_compile_cache_root()),
        backend,
    )
    tiers = {"shared": _inspect_tier(namespace, private=False)}
    if os.path.abspath(private) != os.path.abspath(namespace):
        tiers["private"] = _inspect_tier(private, private=True)
    present = [
        tier
        for tier, details in tiers.items()
        if details["status"] == "entries_present"
    ]
    return {
        "model": backend.name,
        "input": str(request.input),
        "cache_dir": str(namespace),
        "graph_revision": cache.cache_revision(backend.name),
        "status": "entries_present" if present else "missing",
        "present_tiers": present,
        "tiers": tiers,
    }


def verify_cache(
    request: PredictionRequest, *, token_buckets: str | None = None
) -> dict[str, Any]:
    """Check trusted namespace inventories; no weights load, search or prediction.

    Request resolution still checks installed weight locations and queries the
    selected runtime/device, exactly as namespace selection for prediction does.
    """
    if not request.use_compile_cache:
        raise ValueError("cache verification requires an enabled persistent cache")
    with tempfile.TemporaryDirectory(prefix="foldjax-cache-verify-") as temporary:
        jobs = Path(temporary)
        if token_buckets is None:
            resolved = resolve_requests(request, draw_seeds=False, jobs_root=jobs)
            rows = [_verify_resolved(item) for item in resolved]
        else:
            rows = _verify_buckets(request, token_buckets, jobs=jobs)
    return {
        "strategy": "namespace_presence",
        "status": (
            "entries_present"
            if rows and all(row["status"] == "entries_present" for row in rows)
            else "missing"
        ),
        "limitation": _LIMITATION,
        "results": rows,
    }


def _verify_buckets(
    request: PredictionRequest, spec: str, *, jobs: Path
) -> list[dict[str, Any]]:
    from foldjax.warm_buckets import (
        AF3_TOKEN_BUCKETS,
        TOKEN_BUCKETS,
        bucket_request,
        parse_token_buckets,
        representative_lengths,
    )

    models = tuple(normalize_model_name(name) for name in request.resolved_models)
    if len(set(models)) != len(models):
        raise ValueError("models resolve to the same backend more than once")
    if len(models) > 1 and request.weights is not None:
        raise ValueError("one explicit weights path cannot be shared by several models")
    if request.msa not in {"none", "single"}:
        raise ValueError("synthetic bucket verification uses --msa single")
    if (
        "alphafold3" in models
        and "buckets" in request.options
        and request.padding is not None
    ):
        raise ValueError("native AF3 buckets and neutral padding cannot both be set")
    rows = []
    for model in models:
        grid = parse_token_buckets(
            spec, model=model, options=dict(request.options), padding=request.padding
        )
        default_grid = (
            tuple(request.options["buckets"])
            if model == "alphafold3" and "buckets" in request.options
            else AF3_TOKEN_BUCKETS
            if model == "alphafold3"
            else TOKEN_BUCKETS
        )
        for bucket in grid:
            previous = max((size for size in default_grid if size < bucket), default=0)
            for index, length in enumerate(
                representative_lengths(bucket, previous=previous)
            ):
                scalar = bucket_request(
                    request,
                    model=model,
                    bucket=bucket,
                    length=length,
                    index=index,
                    jobs=jobs,
                )
                resolved = resolve_requests(scalar, draw_seeds=False, jobs_root=jobs)
                rows.extend(
                    {**_verify_resolved(item), "bucket": bucket, "real_tokens": length}
                    for item in resolved
                )
    return rows
