from __future__ import annotations

import os
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path
from typing import NoReturn

import pytest


def _unavailable(config: pytest.Config, message: str) -> NoReturn:
    if config.getoption("--run-official-parity"):
        pytest.fail(message, pytrace=False)
    pytest.skip(message)


# `--run-official-parity` is registered in tests/conftest.py, not here. pytest
# only parses options from *initial* conftests, and the initial set is the
# rootdir chain plus `test*` subdirectories of the args; `tests/models/chai` is
# reached through `models/`, which does not match, so registering it here made
# the flag exist only when someone ran this directory by name. From the
# documented root invocation it was an unrecognized argument, which is why
# these 83 tests had no way to run in CI.


def _require_official_runtime(config: pytest.Config) -> None:
    if config.getoption("--run-official-parity") and find_spec("torch") is None:
        raise pytest.UsageError(
            "--run-official-parity requires the torch-bridge dependencies"
        )


def pytest_configure(config: pytest.Config) -> None:
    _require_official_runtime(config)


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    if config.getoption("--run-official-parity"):
        return
    skip = pytest.mark.skip(reason="use --run-official-parity to enable this test")
    for item in items:
        if "official_parity" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def official_asset_path(request: pytest.FixtureRequest) -> Callable[[str], Path]:
    """Resolve an opt-in official component without assuming a sibling checkout."""
    configured = os.environ.get("CHAI_JAX_OFFICIAL_ASSET_DIR")
    if not configured:
        _unavailable(
            request.config,
            "set CHAI_JAX_OFFICIAL_ASSET_DIR to run official parity tests",
        )
    asset_dir = Path(configured).expanduser()
    if not asset_dir.is_dir():
        _unavailable(
            request.config,
            f"official Chai asset directory is unavailable: {asset_dir}",
        )

    def resolve(filename: str) -> Path:
        path = asset_dir / filename
        if not path.is_file():
            _unavailable(
                request.config, f"official Chai component is unavailable: {path}"
            )
        return path

    return resolve


@pytest.fixture(scope="session")
def upstream_chai_dir(request: pytest.FixtureRequest) -> Path:
    """Resolve an opt-in upstream source checkout used by cross-project tests."""
    configured = os.environ.get("CHAI_JAX_UPSTREAM_DIR")
    if not configured:
        _unavailable(
            request.config, "set CHAI_JAX_UPSTREAM_DIR to run upstream parity tests"
        )
    path = Path(configured).expanduser()
    if not (path / "chai_lab").is_dir():
        _unavailable(request.config, f"upstream Chai checkout is unavailable: {path}")
    return path


@pytest.fixture(scope="session")
def upstream_chai_python(
    request: pytest.FixtureRequest, upstream_chai_dir: Path
) -> Path:
    configured = os.environ.get("CHAI_JAX_UPSTREAM_PYTHON")
    path = (
        Path(configured).expanduser()
        if configured
        else upstream_chai_dir / ".venv" / "bin" / "python"
    )
    if not path.is_file():
        _unavailable(request.config, f"upstream Chai Python is unavailable: {path}")
    return path


@pytest.fixture(scope="session")
def native_conformer_path(request: pytest.FixtureRequest) -> Path:
    configured = os.environ.get("CHAI_JAX_CONFORMER_PATH")
    if not configured:
        _unavailable(
            request.config,
            "set CHAI_JAX_CONFORMER_PATH to run conformer parity tests",
        )
    path = Path(configured).expanduser()
    if not path.is_file():
        _unavailable(request.config, f"native conformer archive is unavailable: {path}")
    return path


@pytest.fixture(scope="session")
def chai_trunk_module(official_asset_path):
    torch = pytest.importorskip("torch")
    path = official_asset_path("trunk.pt")
    return torch.jit.load(str(path), map_location="cpu").eval()
