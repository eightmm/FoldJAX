"""Shared isolation for the FoldJAX suite and the vendored port suites.

The ports used to be separate repositories, so each ran pytest in its own
process and could not observe another's environment. They now share one
session: OpenDDE's CLI hands asset paths to the Protenix featurizer through
``os.environ``, and once that ran, later Protenix tests picked up a stale
``components.cif`` from a deleted tmp directory instead of skipping.

The product-side fix lives in ``foldjax.backends.opendde``, which restores these
around its in-process call. This fixture covers tests that invoke the native
CLIs directly, so no suite can leak into the next regardless of ordering.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

_LEAKY_ENVIRONMENT = (
    "JAX_PLATFORMS",
    "PROTENIX_CCD_COMPONENTS_FILE",
    "PROTENIX_CCD_RDKIT_MOL_FILE",
    "PROTENIX_KALIGN_BINARY",
    "PROTENIX_TEMPLATE_MMCIF_DIR",
    "PROTENIX_TEMPLATE_OBSOLETE_FILE",
    "PROTENIX_TEMPLATE_RELEASE_DATES_FILE",
)


@pytest.fixture(autouse=True)
def _isolate_native_asset_environment() -> Iterator[None]:
    saved = {name: os.environ.get(name) for name in _LEAKY_ENVIRONMENT}
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
