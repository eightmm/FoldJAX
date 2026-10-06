"""Shared isolation for the FoldJAX suite and the vendored port suites.

The ports used to be separate repositories, so each ran pytest in its own
process and could not observe another's environment. They now share one
session: OpenDDE's CLI hands asset paths to the Protenix featurizer through
``os.environ``, and once that ran, later Protenix tests picked up a stale
``components.cif`` from a deleted tmp directory instead of skipping.

The product-side fix lives in ``foldjax.backends.opendde``, which restores these
around its in-process call. `_restore_process_state` covers tests that invoke
the native CLIs directly, so no suite can leak into the next regardless of
ordering.
"""

from __future__ import annotations

import os
import sys
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


#: The JAX persistent-cache settings `foldjax.cache.compilation_cache_scope`
#: and the AlphaFold 3 session move. JAX holds them process-wide.
_JAX_CACHE_CONFIG = (
    "jax_compilation_cache_dir",
    "jax_enable_compilation_cache",
    "jax_persistent_cache_min_compile_time_secs",
    "jax_persistent_cache_min_entry_size_bytes",
)


def _jax_cache_config() -> dict[str, object] | None:
    # Read only once something else imported JAX: several tests assert that a
    # code path does not import it, which an import here would decide for them.
    jax = sys.modules.get("jax")
    if jax is None:
        return None
    return {name: getattr(jax.config, name) for name in _JAX_CACHE_CONFIG}


@pytest.fixture(autouse=True)
def _restore_process_state() -> Iterator[None]:
    """Put back the process-wide state a test can change without monkeypatch.

    One session runs every suite, so whatever a test leaves behind is the
    next test's starting point, and its result then depends on the order the
    two happened to run in. The audit that added this found tests that passed
    only because an earlier one had not run: an in-process ``foldjax predict``
    left progress lines on, a native CLI left asset paths in ``os.environ``,
    and ``monkeypatch.delenv`` of an unset name, which records nothing to undo,
    let the code under test keep the value it set.

    The whole environment is restored, not a list of known names: the list
    is what the next leak is missing from. Progress and the memory policy's
    one-time warnings and recorded decision are module state with a known
    fresh-process value, so each test also *starts* from it: a test asserting
    a ``UserWarning`` is asserting the library channel, which a progress line
    replaces once progress is on. The JAX persistent-cache settings are
    process config, and JAX's file-cache object is rebuilt when they move so
    it does not keep the test's directory.
    """
    from foldjax import memory_policy, progress

    environment = dict(os.environ)
    progress_state = (progress._enabled, progress._stream)
    warned = set(memory_policy._WARNED)
    recorded = memory_policy._RECORDED.get()
    jax_config = _jax_cache_config()
    progress.disable()
    memory_policy.reset_warnings()
    memory_policy.clear_record()
    try:
        yield
    finally:
        if dict(os.environ) != environment:
            os.environ.clear()
            os.environ.update(environment)
        progress._enabled, progress._stream = progress_state
        memory_policy._WARNED.clear()
        memory_policy._WARNED.update(warned)
        memory_policy._RECORDED.set(recorded)
        if jax_config is not None and _jax_cache_config() != jax_config:
            import jax
            from jax.experimental.compilation_cache import compilation_cache

            compilation_cache.reset_cache()
            for name, value in jax_config.items():
                jax.config.update(name, value)


@pytest.fixture(autouse=True)
def _restore_release_reclaim() -> Iterator[None]:
    """`foldjax predict` turns the reclaim off for its process; undo that here."""
    from foldjax.models import _managed_memory

    saved = _managed_memory._RECLAIM_AT_RELEASE
    try:
        yield
    finally:
        _managed_memory._RECLAIM_AT_RELEASE = saved


@pytest.fixture(scope="session")
def _alphafold3_runtime_error() -> str | None:
    """Prepare AlphaFold 3's native runtime once; why it failed, or ``None``."""
    from foldjax.models.alphafold3 import build

    try:
        build.ensure_ready()
    except Exception as error:  # noqa: BLE001 - the reason is the skip message
        return f"{type(error).__name__}: {error}"[:500]
    return None


@pytest.fixture
def alphafold3_runtime(_alphafold3_runtime_error: str | None) -> None:
    """Skip when FoldJAX's AlphaFold 3 runtime cannot be built on this host.

    The runtime is a CMake build of AlphaFold 3's C++ extension. Each test used
    to reach it through ``register_runtime()``, so on a host that cannot build
    it -- no zlib headers is enough -- every one of them retried the whole
    build before failing. It is now attempted once per session.

    CI builds the runtime in a step of its own and sets
    ``FOLDJAX_REQUIRE_AF3_RUNTIME=1`` on that shard, where a runtime that
    still cannot be prepared is a failure rather than a skip.
    """
    if _alphafold3_runtime_error is None:
        return
    message = f"AlphaFold 3 runtime could not be prepared: {_alphafold3_runtime_error}"
    if os.environ.get("FOLDJAX_REQUIRE_AF3_RUNTIME") == "1":
        pytest.fail(message)
    pytest.skip(message)


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
