"""Portable Protenix data-root selection without importing publisher code."""

from __future__ import annotations

from pathlib import Path

import pytest

from bench import run_upstream
from bench.provenance import execution_identity


@pytest.mark.parametrize("root_value", [None, ""])
def test_protenix_missing_or_empty_root_uses_historical_home_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, root_value: str | None
) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(run_upstream.Path, "home", staticmethod(lambda: home))
    if root_value is None:
        monkeypatch.delenv("PROTENIX_ROOT_DIR", raising=False)
    else:
        monkeypatch.setenv("PROTENIX_ROOT_DIR", root_value)

    paths = run_upstream.upstream_checkpoint_paths("protenix")

    assert paths == {
        "model": home / "protenix" / "checkpoint" / "protenix_base_default_v1.0.0.pt"
    }


@pytest.mark.parametrize(
    ("root_value", "expected"),
    [
        ("data", "data"),
        ("~/protenix-data", "home/protenix-data"),
    ],
)
def test_protenix_root_env_resolves_relative_and_tilde_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_value: str,
    expected: str,
) -> None:
    home = tmp_path / "home"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PROTENIX_ROOT_DIR", root_value)

    checkpoint = run_upstream.upstream_checkpoint_paths("protenix-v2")["model"]

    assert (
        checkpoint == (tmp_path / expected).resolve() / "checkpoint" / "protenix-v2.pt"
    )


def test_protenix_root_env_drives_checkpoint_command_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "protenix-data"
    checkpoint = root / "checkpoint" / "protenix_base_default_v1.0.0.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"weights")
    monkeypatch.setenv("PROTENIX_ROOT_DIR", str(root))
    monkeypatch.setattr(run_upstream, "upstream_checkout", lambda _model: tmp_path)
    monkeypatch.setattr(run_upstream, "_toolkit_env", lambda _repo: {})

    paths = run_upstream.upstream_checkpoint_paths("protenix")
    argv, _cwd, _environment = run_upstream.command(
        "protenix",
        tmp_path / "job.json",
        tmp_path / "output",
        {"num_samples": 1, "num_steps": 2, "num_recycles": 3},
        101,
    )
    execution = execution_identity(
        {"PROTENIX_ROOT_DIR": str(root)},
        timing_state="cold-or-unspecified",
        traced=False,
    )

    assert paths == {"model": checkpoint.resolve()}
    assert argv[argv.index("--load_checkpoint_dir") + 1] == str(root / "checkpoint")
    assert run_upstream.upstream_implicit_asset_paths("protenix") == {
        "ccd.components": run_upstream.COMPONENTS,
        "ccd.rdkit_cache": run_upstream.RDKIT_CACHE,
    }
    assert execution["environment"]["PROTENIX_ROOT_DIR"] != str(root)
