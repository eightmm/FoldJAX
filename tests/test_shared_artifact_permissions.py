"""Published weights and generated runtime data follow the store's umask."""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from foldjax import assets
from foldjax.models import _weights_io
from foldjax.models.alphafold3 import build
from foldjax.models.boltz2.bridge import native


@contextmanager
def _umask(mask: int):
    previous = os.umask(mask)
    try:
        yield
    finally:
        os.umask(previous)


@pytest.mark.parametrize("mask, mode", [(0o022, 0o644), (0o002, 0o664), (0o077, 0o600)])
@pytest.mark.parametrize("safetensors", [True, False])
def test_boltz_native_weights_and_sidecar_follow_umask(
    tmp_path: Path, monkeypatch, mask: int, mode: int, safetensors: bool
) -> None:
    if safetensors:
        pytest.importorskip("safetensors.numpy")
    monkeypatch.setattr(native, "_have_safetensors", lambda: safetensors)
    params = {"weight": np.arange(4, dtype=np.float32), "heads": 2}
    with _umask(mask):
        saved = native.save_params(params, tmp_path / "weights")
    for name in ("weights_path", "sidecar"):
        assert Path(saved[name]).stat().st_mode & 0o777 == mode
    np.testing.assert_array_equal(
        native.load_params(saved["weights_path"])["weight"], params["weight"]
    )


@pytest.mark.parametrize("compress", [True, False])
def test_shared_native_writer_repairs_a_private_stage(
    tmp_path: Path, compress: bool
) -> None:
    target = tmp_path / "weights.jax"
    target.touch(mode=0o600)
    params = {"weight": np.arange(4, dtype=np.float32)}
    with _umask(0o022):
        _weights_io.save_native_weights(target, params, compress=compress)
    assert target.stat().st_mode & 0o777 == 0o644
    np.testing.assert_array_equal(
        _weights_io.load_native_weights(target)["weight"], params["weight"]
    )


@pytest.mark.parametrize("source_mode", [0o600, 0o644, 0o444])
def test_staged_weights_are_shareable_without_chmodding_the_source(
    tmp_path: Path, source_mode: int
) -> None:
    source = tmp_path / "published"
    source.write_bytes(b"checkpoint")
    source.chmod(source_mode)
    staged = tmp_path / "staged"
    with _umask(0o022):
        assets._stage_weight_file(source, staged)
    assert staged.read_bytes() == source.read_bytes()
    assert staged.stat().st_mode & 0o777 == 0o644
    assert source.stat().st_mode & 0o777 == source_mode
    assert staged.samefile(source) is (source_mode == 0o644)


@pytest.mark.parametrize("resume", [False, True])
def test_download_publication_normalizes_a_private_partial(
    tmp_path: Path, monkeypatch, resume: bool
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "store"))
    source = tmp_path / "source"
    payload = b"verified weights"
    source.write_bytes(payload)
    item = assets.Download(
        name="weights.bin", url=source.as_uri(),
        sha256=hashlib.sha256(payload).hexdigest(), size=len(payload),
    )
    target = item.target("boltz2")
    target.parent.mkdir(parents=True)
    partial = target.with_suffix(target.suffix + ".part")
    partial.write_bytes(payload if resume else payload[:3])
    partial.chmod(0o600)
    with _umask(0o022):
        downloaded = assets.download(item, "boltz2")
    assert downloaded.read_bytes() == payload
    assert downloaded.stat().st_mode & 0o777 == 0o644


@pytest.mark.parametrize(
    "created_mode, published_mode", [(0o600, 0o644), (0o700, 0o744)]
)
def test_generated_runtime_data_normalizes_a_private_temporary_file(
    tmp_path: Path, created_mode: int, published_mode: int
) -> None:
    target = tmp_path / "runtime" / "ccd.pickle"

    def generate(path: Path) -> None:
        descriptor = os.open(path, os.O_CREAT | os.O_WRONLY, created_mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(b"generated runtime data")

    with _umask(0o022):
        build._generate_atomic(target, generate)
    assert target.read_bytes() == b"generated runtime data"
    assert target.stat().st_mode & 0o777 == published_mode


def test_runtime_marker_normalizes_an_existing_private_stage(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / ".foldjax-complete"
    staged = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    staged.touch(mode=0o600)
    monkeypatch.setattr(build, "runtime_key", lambda: "test-runtime")
    with _umask(0o022):
        build._write_marker(target)
    assert target.read_text() == "test-runtime\n"
    assert target.stat().st_mode & 0o777 == 0o644


def test_extracted_molecule_bundle_does_not_publish_archive_private_modes(
    tmp_path: Path, monkeypatch
) -> None:
    names = assets._BOLTZ_CANONICAL_MOLECULES
    monkeypatch.setattr(assets, "_BOLTZ_MOLECULE_COUNT", len(names))
    archive = tmp_path / "mols.tar"
    with tarfile.open(archive, "w") as bundle:
        directory = tarfile.TarInfo("mols")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o700
        bundle.addfile(directory)
        for name in names:
            member = tarfile.TarInfo(f"mols/{name}")
            member.size = len(b"molecule")
            member.mode = 0o600
            bundle.addfile(member, io.BytesIO(b"molecule"))
    target = tmp_path / "weights" / "mols"
    with _umask(0o022):
        assets._unpack_boltz_mols(archive, target)
    assert target.stat().st_mode & 0o777 == 0o755
    assert all((target / name).stat().st_mode & 0o777 == 0o644 for name in names)
    assert assets._boltz_molecules_complete(target)
