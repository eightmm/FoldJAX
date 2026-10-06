"""Shared isolation for the FoldJAX suite and the vendored port suites.

The ports used to be separate repositories, so each ran pytest in its own
process and could not observe another's environment. They now share one
session: OpenDDE's CLI used to hand asset paths to the Protenix featurizer
through ``os.environ``, and once that ran, later Protenix tests picked up a
stale ``components.cif`` from a deleted tmp directory instead of skipping.

The CLI now passes those paths explicitly, and ``foldjax.backends.opendde``
still restores what the native CLI exports (``JAX_PLATFORMS``). This fixture
covers tests that set the ``PROTENIX_*`` variables themselves, so no suite can
leak into the next regardless of ordering.
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


def pytest_report_header(config: pytest.Config) -> list[str]:
    """Say out loud which vendored suites the optional-import gate left out.

    The gate itself is `collect_ignore` in `tests/models/conftest.py`; this hook
    has to live here, because pytest only calls it from the rootdir conftest or
    a plugin. Placed beside the gate it never ran, and some 80 Boltz-2 parity
    tests disappeared from every run on a torch-free machine without a line.
    """
    from .models.conftest import _OPTIONAL_SUITES, _skipped

    return [
        f"vendored parity suites not collected: {len(_OPTIONAL_SUITES[module][1])} "
        f"modules need {module} ({environment})"
        for module, environment in sorted(_skipped.items())
    ]


_LEAKY_ENVIRONMENT = (
    "JAX_PLATFORMS",
    "PROTENIX_CCD_COMPONENTS_FILE",
    "PROTENIX_CCD_RDKIT_MOL_FILE",
    "PROTENIX_KALIGN_BINARY",
    "PROTENIX_TEMPLATE_MMCIF_DIR",
    "PROTENIX_TEMPLATE_OBSOLETE_FILE",
    "PROTENIX_TEMPLATE_RELEASE_DATES_FILE",
)


@pytest.fixture
def trust_ancestors_above_tmp_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Judge a compile cache only from ``tmp_path`` down.

    The trust policy walks every ancestor of a cache, so a run whose basetemp
    sits in a group-writable tree -- a checkout under a directory whose group
    has a second member -- refuses every cache a test builds, whatever the
    test is about. Tests of the cache-on path take this fixture;
    the tests of the ancestor rule itself do not, and still walk to `/`.
    """
    from foldjax import cache

    boundary = Path(os.path.realpath(tmp_path))

    def reason(path: Path) -> str | None:
        leaf = Path(os.path.realpath(path))
        for directory in (leaf, *leaf.parents):
            if not directory.is_relative_to(boundary):
                return None
            found = cache._untrusted_directory_reason(
                directory, leaf=directory == leaf
            )
            if found is not None:
                return found
        return None

    monkeypatch.setattr(cache, "_untrusted_cache_reason", reason)


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


@pytest.fixture(autouse=True)
def _restore_release_reclaim() -> Iterator[None]:
    """`foldjax predict` turns the reclaim off for its process; undo that here."""
    from foldjax.models import _managed_memory

    saved = _managed_memory._RECLAIM_AT_RELEASE
    try:
        yield
    finally:
        _managed_memory._RECLAIM_AT_RELEASE = saved


@pytest.fixture(autouse=True)
def _restore_progress() -> Iterator[None]:
    """`foldjax predict` enables progress lines for its process; undo that here.

    Left on, every later test in the session takes the CLI's channel: a search
    failure under `auto` becomes a progress line instead of the `UserWarning`
    a library caller gets (`msa_search.report_search_failure`), so the tests
    that assert that warning failed whenever a CLI test ran first.
    """
    from foldjax import progress

    saved = (progress._enabled, progress._stream)
    try:
        yield
    finally:
        progress._enabled, progress._stream = saved


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
