from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest

from foldjax import cache_verify
from foldjax.api import resolve_cache_dir, resolve_requests
from foldjax.cli import main
from foldjax.registry import get_backend
from foldjax.schema import PaddingConfig, PredictionRequest


@pytest.fixture
def request_and_namespaces(tmp_path, monkeypatch):
    source = tmp_path / "job.json"
    source.write_text('{"version": 1}')
    weights = tmp_path / "weights.jax"
    weights.write_bytes(b"weights")
    private = tmp_path / "private"
    monkeypatch.setenv("FOLDJAX_PRIVATE_COMPILE_CACHE", str(private))
    monkeypatch.delenv("FOLDJAX_TRUST_SHARED_COMPILE_CACHE", raising=False)
    request = PredictionRequest(
        model="boltz2",
        input=source,
        input_format="native",
        weights=weights,
        cache_dir=tmp_path / "shared",
        padding=PaddingConfig(tokens=128),
    )
    resolved = resolve_requests(request, draw_seeds=False)[0]
    backend = get_backend(request.model)
    shared_namespace = resolve_cache_dir(resolved, backend)
    private_namespace = resolve_cache_dir(
        dataclasses.replace(resolved, cache_dir=private), backend
    )
    return request, shared_namespace, private_namespace


def _entry(namespace: Path, *, private=False, name="jit_model-cache"):
    missing = []
    parent = namespace
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    for directory in reversed(missing):
        directory.mkdir(mode=0o700)
    namespace.chmod(0o700 if private else 0o755)
    entry = namespace / name
    entry.write_bytes(b"persistent entry; never deserialize in verification")
    entry.chmod(0o600 if private else 0o644)
    return entry


def test_shared_namespace_presence_is_readonly_and_does_not_predict(
    request_and_namespaces, monkeypatch
):
    from foldjax import api, warmup

    request, shared, private = request_and_namespaces
    entry = _entry(shared)
    shared.chmod(0o555)
    before = entry.stat()

    def refuse(*_args, **_kwargs):
        pytest.fail("verification must not execute prediction or warming")

    monkeypatch.setattr(api, "_predict_resolved", refuse)
    monkeypatch.setattr(warmup, "warm_cache", refuse)
    monkeypatch.setattr(get_backend(request.model), "predict", refuse)
    result = cache_verify.verify_cache(request)
    assert result["strategy"] == "namespace_presence"
    assert result["status"] == "entries_present"
    row = result["results"][0]
    assert row["present_tiers"] == ["shared"]
    assert row["graph_revision"] is None
    assert row["tiers"]["shared"]["entries"] == 1
    assert row["tiers"]["private"]["status"] == "missing"
    assert not private.exists()
    assert entry.stat() == before
    assert list(shared.iterdir()) == [entry]
    assert "current HLO" in result["limitation"]


def test_private_namespace_is_accepted_without_creating_missing_shared(
    request_and_namespaces,
):
    request, shared, private = request_and_namespaces
    _entry(private, private=True)
    result = cache_verify.verify_cache(request)
    assert result["status"] == "entries_present"
    assert result["results"][0]["present_tiers"] == ["private"]
    assert not shared.exists()


def test_empty_and_tuning_only_namespaces_do_not_pass(request_and_namespaces):
    request, shared, _ = request_and_namespaces
    _entry(shared, name="jit_model-atime")
    (shared / "empty-cache").touch()
    tuning = shared / ".tokamax-autotuning-v2"
    tuning.mkdir()
    (tuning / "result.json").write_text("{}")
    result = cache_verify.verify_cache(request)
    assert result["status"] == "missing"
    assert result["results"][0]["tiers"]["shared"]["entries"] == 0


def test_unsafe_shared_tier_does_not_hide_safe_private(request_and_namespaces):
    request, shared, private = request_and_namespaces
    _entry(shared)
    shared.chmod(0o777)
    _entry(private, private=True)
    row = cache_verify.verify_cache(request)["results"][0]
    assert row["present_tiers"] == ["private"]
    assert row["tiers"]["shared"]["status"] == "untrusted"


def test_private_tier_does_not_inherit_shared_trust(
    request_and_namespaces, monkeypatch
):
    request, _, private = request_and_namespaces
    _entry(private, private=True)
    private.chmod(0o755)
    monkeypatch.setenv("FOLDJAX_TRUST_SHARED_COMPILE_CACHE", "1")
    result = cache_verify.verify_cache(request)
    assert result["status"] == "missing"
    assert result["results"][0]["tiers"]["private"]["status"] == "unavailable"


def test_symlink_hardlink_and_unsafe_mode_entries_are_not_counted(
    request_and_namespaces, tmp_path
):
    request, shared, _ = request_and_namespaces
    entry = _entry(shared, name="unsafe-cache")
    entry.chmod(0o666)
    external = tmp_path / "outside"
    external.write_bytes(b"compiled")
    (shared / "symlink-cache").symlink_to(external)
    os.link(external, shared / "hardlink-cache")
    result = cache_verify.verify_cache(request)
    tier = result["results"][0]["tiers"]["shared"]
    assert tier["entries"] == 0
    assert tier["rejected_entries"] == 3
    assert result["status"] == "missing"


def test_namespace_symlink_is_not_followed(request_and_namespaces, tmp_path):
    request, shared, _ = request_and_namespaces
    target = tmp_path / "target"
    _entry(target)
    shared.parent.mkdir(parents=True)
    shared.symlink_to(target, target_is_directory=True)
    result = cache_verify.verify_cache(request)
    assert result["status"] == "missing"
    assert result["results"][0]["tiers"]["shared"]["status"] == "unavailable"


def test_concurrent_eviction_does_not_hide_remaining_entries(
    request_and_namespaces, monkeypatch
):
    request, shared, _ = request_and_namespaces
    _entry(shared)
    _entry(shared, name="evicted-cache")
    actual_stat = os.stat

    def evict(path, *args, **kwargs):
        if path == "evicted-cache" and "dir_fd" in kwargs:
            raise FileNotFoundError(path)
        return actual_stat(path, *args, **kwargs)

    monkeypatch.setattr(cache_verify.os, "stat", evict)
    result = cache_verify.verify_cache(request)
    assert result["status"] == "entries_present"
    assert result["results"][0]["tiers"]["shared"]["entries"] == 1


@pytest.mark.parametrize("present", [False, True])
def test_verify_cli_returns_json_and_missing_exit_status(
    request_and_namespaces, capsys, present
):
    request, shared, _ = request_and_namespaces
    if present:
        _entry(shared)
    status = main(
        [
            "cache",
            "warm",
            "--verify-only",
            "--model",
            request.model,
            "--input",
            str(request.input),
            "--input-format",
            "native",
            "--weights",
            str(request.weights),
            "--cache-dir",
            str(request.cache_dir),
            "--pad-tokens",
            "128",
        ]
    )
    output = capsys.readouterr()
    payload = json.loads(output.out)
    assert status == (0 if present else 3)
    assert payload["status"] == ("entries_present" if present else "missing")
    assert "current HLO" in output.err
    assert "execute_once" not in output.err
    assert "Prediction command template" not in output.err


def test_bucket_verification_checks_both_lengths_and_every_bucket(
    request_and_namespaces, monkeypatch
):
    request, _, _ = request_and_namespaces
    seen = []

    def inspect(resolved):
        document = json.loads(resolved.input.read_text())
        length = len(document["entities"][0]["sequence"])
        seen.append((resolved.padding.tokens, length, resolved.msa))
        return {
            "model": resolved.model,
            "status": (
                "entries_present" if resolved.padding.tokens == 128 else "missing"
            ),
        }

    monkeypatch.setattr(cache_verify, "_verify_resolved", inspect)
    result = cache_verify.verify_cache(request, token_buckets="256,128")
    assert result["status"] == "missing"
    assert [bucket for bucket, _, _ in seen] == [128, 128, 256, 256]
    assert seen[0][1] != seen[1][1]
    assert all(msa == "single" for _, _, msa in seen)
    assert [row["bucket"] for row in result["results"]] == [128, 128, 256, 256]


def test_bucket_cli_uses_the_warm_namespace_and_retains_fixed_msa(
    request_and_namespaces, tmp_path, capsys
):
    from foldjax.warm_buckets import bucket_request

    request, _, _ = request_and_namespaces
    request = dataclasses.replace(
        request, padding=PaddingConfig(msa=2048), max_msa_depth=2048
    )
    scalar = bucket_request(
        request,
        model="boltz2",
        bucket=128,
        length=120,
        index=0,
        jobs=tmp_path / "jobs",
    )
    resolved = resolve_requests(scalar, draw_seeds=False)[0]
    namespace = resolve_cache_dir(resolved, get_backend("boltz2"))
    _entry(namespace)
    status = main(
        [
            "cache",
            "warm",
            "--verify-only",
            "--model",
            "boltz2",
            "--token-buckets",
            "128",
            "--weights",
            str(request.weights),
            "--cache-dir",
            str(request.cache_dir),
            "--fixed-msa",
            "2048",
            "--msa",
            "single",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert status == 0
    assert len(result["results"]) == 2
    assert all(row["cache_dir"] == str(namespace) for row in result["results"])
    assert not request.output_dir


def test_bucket_verification_preserves_native_af3_profile(
    request_and_namespaces, monkeypatch
):
    request, _, _ = request_and_namespaces
    request = dataclasses.replace(
        request, model="alphafold3", padding=None, options={"buckets": [256, 512]}
    )
    seen = []

    def inspect(resolved):
        seen.append(resolved)
        return {"status": "missing"}

    monkeypatch.setattr(cache_verify, "_verify_resolved", inspect)
    result = cache_verify.verify_cache(request, token_buckets="256")
    assert result["status"] == "missing"
    assert len(seen) == 2
    assert all(item.padding is None for item in seen)
    assert all(item.options["buckets"] == [256, 512] for item in seen)
