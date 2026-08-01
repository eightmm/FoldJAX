"""Small content-addressed cache for standalone preprocessing assets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

_MAX_ASSET_BYTES = 2 * 1024 * 1024 * 1024


class AssetError(RuntimeError):
    """An asset source or cached object violated the integrity contract."""


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes


@dataclass(frozen=True)
class ResolvedAsset:
    path: Path
    sha256: str
    provenance: Mapping[str, object]


HttpTransport = Callable[[str, float], HttpResponse]


def _transport(url: str, timeout: float) -> HttpResponse:
    request = urllib.request.Request(url, headers={"User-Agent": "chai-jax/0.1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpResponse(response.status, response.read(_MAX_ASSET_BYTES + 1))
    except urllib.error.HTTPError as error:
        return HttpResponse(error.code, error.read())


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _valid_digest(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


class AssetCache:
    """Materialize local or HTTPS assets by SHA-256 with strict provenance."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        transport: HttpTransport = _transport,
        timeout: float = 60.0,
        max_bytes: int = _MAX_ASSET_BYTES,
    ) -> None:
        if timeout <= 0 or max_bytes <= 0:
            raise ValueError("asset timeout and size limit must be positive")
        self.cache_dir = Path(cache_dir)
        self.transport = transport
        self.timeout = timeout
        self.max_bytes = max_bytes

    def _read_source(self, source: str | Path) -> tuple[bytes, dict[str, object]]:
        text = str(source)
        parsed = urllib.parse.urlsplit(text)
        if parsed.scheme:
            if parsed.scheme != "https" or not parsed.netloc:
                raise ValueError("remote preprocessing assets require an HTTPS URL")
            if parsed.username or parsed.password:
                raise ValueError(
                    "asset URL credentials must not be embedded in the URL"
                )
            try:
                response = self.transport(text, self.timeout)
            except OSError as error:
                raise AssetError(
                    f"failed to download preprocessing asset: {text}"
                ) from error
            if not 200 <= response.status < 300:
                raise AssetError(
                    f"preprocessing asset download failed with HTTP {response.status}"
                )
            data = response.body
            provenance: dict[str, object] = {
                "source_kind": "https",
                "source": text,
            }
        else:
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"preprocessing asset is not a file: {path}")
            try:
                data = path.read_bytes()
            except OSError as error:
                raise AssetError(
                    f"failed to read preprocessing asset: {path}"
                ) from error
            provenance = {
                "source_kind": "local",
                "source": str(path.resolve()),
            }
        if len(data) > self.max_bytes:
            raise AssetError("preprocessing asset exceeds the configured size limit")
        return data, provenance

    def _cached(self, digest: str) -> ResolvedAsset | None:
        directory = self.cache_dir / digest
        if not directory.exists():
            return None
        asset = directory / "asset.npz"
        provenance_path = directory / "provenance.json"
        if (
            not directory.is_dir()
            or not asset.is_file()
            or not provenance_path.is_file()
        ):
            raise AssetError(f"preprocessing asset cache is incomplete: {directory}")
        data = asset.read_bytes()
        if _digest(data) != digest:
            raise AssetError(f"preprocessing asset cache hash mismatch: {asset}")
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AssetError(
                f"invalid preprocessing asset provenance: {directory}"
            ) from error
        if provenance.get("sha256") != digest or provenance.get("bytes") != len(data):
            raise AssetError(f"preprocessing asset provenance mismatch: {directory}")
        return ResolvedAsset(asset.resolve(), digest, provenance)

    def resolve(
        self,
        source: str | Path,
        *,
        expected_sha256: str | None = None,
    ) -> ResolvedAsset:
        if expected_sha256 is not None and not _valid_digest(expected_sha256):
            raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
        if expected_sha256 is not None:
            cached = self._cached(expected_sha256)
            if cached is not None:
                return cached

        data, source_provenance = self._read_source(source)
        digest = _digest(data)
        if expected_sha256 is not None and digest != expected_sha256:
            raise AssetError(
                "preprocessing asset SHA-256 mismatch: "
                f"expected {expected_sha256}, got {digest}"
            )
        cached = self._cached(digest)
        if cached is not None:
            return cached

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{digest}.", dir=self.cache_dir))
        destination = self.cache_dir / digest
        provenance = {
            **source_provenance,
            "sha256": digest,
            "bytes": len(data),
        }
        try:
            (temporary / "asset.npz").write_bytes(data)
            (temporary / "provenance.json").write_text(
                json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            try:
                os.replace(temporary, destination)
            except OSError:
                if not destination.exists():
                    raise
                shutil.rmtree(temporary)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        resolved = self._cached(digest)
        if resolved is None:
            raise AssertionError("asset cache disappeared after materialization")
        return resolved


__all__ = ["AssetCache", "AssetError", "HttpResponse", "ResolvedAsset"]
