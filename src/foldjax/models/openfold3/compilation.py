"""Persistent compilation cache.

Compiling the released architecture is expensive and grows with sequence length:
measured on an RTX PRO 6000, ~290 s at 32 tokens, ~340 s at 128, and ~650 s at 384
-- where XLA itself prints a slow-compile warning. Without a persistent cache that
cost is paid again in every new process, which is most of the wall clock of a
one-off prediction.

Enabling it is opt-in rather than automatic. Importing the port already sets
allocator defaults, but those are process-local; writing
hundreds of megabytes to a directory the caller never named is a different kind of
side effect.
"""

from __future__ import annotations

import os
from pathlib import Path


def default_cache_dir() -> Path:
    """``$OPENFOLD3_JAX_CACHE``, else the FoldJAX home, else a pre-FoldJAX cache.

    The root comes from `foldjax_home()`, so `FOLDJAX_HOME` and `XDG_CACHE_HOME`
    move it the way they move everything else FoldJAX writes. Compiled
    executables for this architecture run to hundreds of megabytes, so a
    checkout that already populated ``~/.cache/openfold3_jax/jit`` from the
    standalone package keeps using it rather than silently recompiling into a
    new location.

    Only the native ``openfold3-jax-predict`` path reaches this. Predictions
    driven through `foldjax predict` are given a cache directory that
    `api.predict` has already namespaced per model, weight identity, and
    compile-relevant options.
    """
    explicit = os.environ.get("OPENFOLD3_JAX_CACHE")
    if explicit:
        return Path(explicit)
    from foldjax.paths import foldjax_home

    current = foldjax_home() / "openfold3" / "jit"
    if current.exists():
        return current
    legacy = Path.home() / ".cache" / "openfold3_jax" / "jit"
    return legacy if legacy.exists() else current


def enable_compilation_cache(
    directory: str | os.PathLike[str] | None = None,
    *,
    min_compile_time_secs: float = 1.0,
) -> Path:
    """Cache compiled executables on disk and return the directory used.

    Args:
        directory: where to store them; :func:`default_cache_dir` when omitted.
        min_compile_time_secs: skip caching anything that compiled faster than
            this. The default keeps the cache to the handful of large executables
            that matter instead of every small kernel.

    Returns:
        The directory, created if necessary.

    A cache entry is keyed on the computation *and* the accelerator and jaxlib
    version, so a driver or version change silently misses rather than returning a
    stale executable. Entries are never evicted -- delete the directory to reclaim
    the space.
    """
    import jax

    path = Path(directory) if directory is not None else default_cache_dir()
    path.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(path))
    jax.config.update(
        "jax_persistent_cache_min_compile_time_secs", min_compile_time_secs
    )
    # Without this, only executables above a size threshold are written, which
    # excludes exactly the small-but-slow-to-compile graphs.
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    return path
