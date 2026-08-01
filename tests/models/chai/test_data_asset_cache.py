from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from foldjax.models.chai.data.asset_cache import AssetCache, AssetError, HttpResponse


def test_local_asset_is_content_addressed_and_verified(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    source.write_bytes(b"asset")
    digest = hashlib.sha256(b"asset").hexdigest()
    cache = AssetCache(tmp_path / "cache")

    first = cache.resolve(source, expected_sha256=digest)
    second = cache.resolve(source, expected_sha256=digest)
    assert first.path == second.path
    assert first.sha256 == digest
    assert first.path.read_bytes() == b"asset"
    assert first.provenance["source_kind"] == "local"


def test_https_asset_download_is_cached_without_second_request(tmp_path: Path) -> None:
    calls: list[str] = []

    def transport(url: str, timeout: float) -> HttpResponse:
        calls.append(url)
        return HttpResponse(status=200, body=b"remote")

    digest = hashlib.sha256(b"remote").hexdigest()
    cache = AssetCache(tmp_path / "cache", transport=transport)
    first = cache.resolve("https://example.test/templates.npz", expected_sha256=digest)
    second = cache.resolve("https://example.test/templates.npz", expected_sha256=digest)
    assert first.path == second.path
    assert calls == ["https://example.test/templates.npz"]


def test_asset_cache_rejects_insecure_url_and_digest_mismatch(tmp_path: Path) -> None:
    cache = AssetCache(tmp_path / "cache")
    with pytest.raises(ValueError, match="HTTPS"):
        cache.resolve("http://example.test/a.npz", expected_sha256="a" * 64)
    source = tmp_path / "source.npz"
    source.write_bytes(b"asset")
    with pytest.raises(AssetError, match="SHA-256"):
        cache.resolve(source, expected_sha256="a" * 64)
