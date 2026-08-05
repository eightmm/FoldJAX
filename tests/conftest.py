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
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the opt-in parity flag where pytest will actually parse it.

    Only conftests on the rootdir chain (and in `test*` directories named as
    args) are consulted for options. This file is one; a `tests/models/<name>/`
    conftest is not, because it is reached through `models/`. Registering it in
    one of those meant `pytest --run-official-parity` from the repository root --
    the documented command -- failed with "unrecognized arguments", so the flag
    worked only when that one directory was named explicitly.
    """
    parser.addoption(
        "--run-official-parity",
        action="store_true",
        default=False,
        help="run tests that require official model assets or an upstream checkout",
    )


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


@pytest.fixture
def ccd_components() -> Path:
    """The released ``components.cif``, or skip the test.

    Anything outside the vendored CCD subset -- an arbitrary ligand, most
    modified residues -- is read out of this file, which is a 490 MB managed
    download rather than a repository fixture. A clean checkout and every CI
    runner legitimately does not have it, and reporting that as a failure
    blames the featurizer for a missing optional asset.

    This used to pass anywhere only by accident: the featurizer guessed a
    sibling checkout six directories up, which held on exactly one machine.
    It now resolves the managed store, so this fixture and the product agree
    on where the file is.
    """
    from foldjax.paths import assets_dir

    configured = os.environ.get("PROTENIX_CCD_COMPONENTS_FILE")
    path = Path(configured) if configured else assets_dir() / "components.cif"
    if not path.is_file():
        pytest.skip(
            "needs the released components.cif "
            "(`foldjax weights fetch --model protenix`)"
        )
    return path
