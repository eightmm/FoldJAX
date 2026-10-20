"""Keep embedded kernel source locations stable across installation roots."""

from __future__ import annotations

import os
import sys

SOURCE_FILE_CANONICALIZATION_REGEX = r".*/(site-packages|dist-packages|src)/"
_ENVIRONMENT = "JAX_HLO_SOURCE_FILE_CANONICALIZATION_REGEX"
_CONFIG = "jax_hlo_source_file_canonicalization_regex"


def _configure() -> None:
    config = getattr(sys.modules.get("jax"), "config", None)
    if config is None:
        # Do not import JAX here: CLI device/allocator selection must precede it.
        os.environ.setdefault(_ENVIRONMENT, SOURCE_FILE_CANONICALIZATION_REGEX)
    elif getattr(config, _CONFIG) is None:
        # None is JAX's unset default; an explicit empty string disables rewriting.
        config.update(
            _CONFIG, os.environ.get(_ENVIRONMENT, SOURCE_FILE_CANONICALIZATION_REGEX)
        )


_configure()
