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
    args) are consulted for options. `tests/parity/conftest.py` is reached
    through `tests/`, so registering it there made `pytest --run-cpu-parity`
    from the repository root -- the documented command -- fail to parse.

    There is no `--run-official-parity` beside it. It was registered here and
    read nowhere: the sole `official_parity` test gates itself on the assets it
    needs (`pytest.skip` when the publisher CCD files are absent), which is the
    condition that actually decides whether it can run. A flag would have
    skipped it on machines that do have them. Select it by marker --
    `-m official_parity` or `-m 'not official_parity'` -- which needs no option.
    """
    parser.addoption(
        "--run-cpu-parity",
        action="store_true",
        default=False,
        help="run the CPU parity regression subset (tests/parity, marker cpu_parity)",
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
