"""Portable upstream Boltz cache selection without importing publisher code."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from bench import run_upstream
from bench.provenance import execution_identity


@pytest.mark.parametrize("cache_value", [None, ""])
def test_boltz_cache_missing_or_empty_uses_home_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cache_value: str | None
) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(run_upstream.Path, "home", staticmethod(lambda: home))
    if cache_value is None:
        monkeypatch.delenv("BOLTZ_CACHE", raising=False)
    else:
        monkeypatch.setenv("BOLTZ_CACHE", cache_value)

    paths = run_upstream.upstream_checkpoint_paths("boltz2")

    assert paths == {"model": home / ".boltz" / "boltz2_conf.ckpt"}


@pytest.mark.parametrize(
    ("cache_value", "expected"),
    [("relative-cache", "relative-cache"), ("~/cache", "home/cache")],
)
def test_boltz_cache_env_matches_native_relative_and_tilde_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache_value: str,
    expected: str,
) -> None:
    home = tmp_path / "home"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("BOLTZ_CACHE", cache_value)

    checkpoint = run_upstream.upstream_checkpoint_paths("boltz2")["model"]

    assert checkpoint == (tmp_path / expected).resolve() / "boltz2_conf.ckpt"


def test_boltz_cache_env_drives_checkpoint_and_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    checkpoint = cache / "boltz2_conf.ckpt"
    checkpoint.write_bytes(b"weights")
    monkeypatch.setenv("BOLTZ_CACHE", str(cache))
    monkeypatch.setattr(run_upstream, "upstream_checkout", lambda _model: tmp_path)

    paths = run_upstream.upstream_checkpoint_paths("boltz2")
    argv, _cwd, _environment = run_upstream.command(
        "boltz2",
        tmp_path / "job.json",
        tmp_path / "output",
        {"num_samples": 1, "num_steps": 2, "num_recycles": 3},
        101,
    )

    assert paths == {"model": cache.resolve() / "boltz2_conf.ckpt"}
    assert argv[argv.index("--cache") + 1] == str(cache.resolve())


def test_boltz_assets_and_execution_provenance_use_the_env_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    mols = cache / "mols"
    mols.mkdir(parents=True)
    for name in ("mols.tar", "boltz2_aff.ckpt"):
        (cache / name).write_bytes(name.encode())
    (mols / "ALA.pkl").write_bytes(b"molecule")
    job = tmp_path / "job.json"
    job.write_text(
        json.dumps(
            {
                "version": 1,
                "sequences": [
                    {"protein": {"id": "A", "sequence": "AC", "msa": "a.a3m"}}
                ],
            }
        )
    )
    monkeypatch.setenv("BOLTZ_CACHE", str(cache))
    monkeypatch.setattr(
        run_upstream, "_upstream_boltz_canonical_tokens", lambda: ("ALA",)
    )

    assets = run_upstream.upstream_implicit_asset_paths("boltz2", native_input=job)
    environment = execution_identity(
        {"BOLTZ_CACHE": str(cache)},
        timing_state="cold-or-unspecified",
        traced=False,
    )

    assert assets == {"boltz.canonical-protein.ALA": mols / "ALA.pkl"}
    assert environment["environment"]["BOLTZ_CACHE"] != str(cache)
    assert "import boltz" not in inspect.getsource(run_upstream)
