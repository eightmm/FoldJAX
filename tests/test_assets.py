"""The weight store: fetch, verify, convert, resolve.

The network and the multi-GB conversions are not exercised here — the download
loop is driven against a local file:// URL so verification, resume, and failure
behaviour are all real, and the registry itself is checked for the mistakes that
would only surface after someone waited on a 2.6 GB download.
"""

from __future__ import annotations

import dataclasses
import hashlib
import http.client
import json
import tarfile
from pathlib import Path

import pytest

from foldjax import assets, paths
from foldjax.registry import model_info


def _write_boltz_native_bundle(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    header = json.dumps(
        {"d:weight": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}
    ).encode()
    tensor = len(header).to_bytes(8, "little") + header + b"x"
    sidecar = json.dumps(
        {"backend": "safetensors", "keys": ["d:weight"], "scalars": {}}
    ).encode()
    for name in assets._BOLTZ_NATIVE_FILES:
        (root / name).write_bytes(sidecar if name.endswith(".json") else tensor)


def test_home_follows_its_environment(tmp_path: Path, monkeypatch) -> None:
    # This checkout has a `.foldjax/` store, which would otherwise outrank both
    # fallbacks below; the case where it does is the next test.
    monkeypatch.setattr(paths, "_repository_store", lambda: None)

    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "explicit"))
    assert paths.foldjax_home() == tmp_path / "explicit"

    monkeypatch.delenv("FOLDJAX_HOME")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert paths.foldjax_home() == tmp_path / "xdg" / "foldjax"

    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    assert paths.foldjax_home() == tmp_path / "home" / ".cache" / "foldjax"


def test_a_checkout_store_outranks_the_cache(tmp_path: Path, monkeypatch) -> None:
    """`.foldjax/` in the working tree wins, and only when it is really there.

    Keeping weights beside the source is what makes a checkout self-contained,
    so it has to beat `~/.cache`; an explicit `FOLDJAX_HOME` still beats it,
    which is how the bench harness points at its own store.
    """
    store = tmp_path / "checkout" / ".foldjax"
    store.mkdir(parents=True)
    monkeypatch.delenv("FOLDJAX_HOME", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))

    monkeypatch.setattr(paths, "_repository_store", lambda: store)
    assert paths.foldjax_home() == store

    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "explicit"))
    assert paths.foldjax_home() == tmp_path / "explicit"

    monkeypatch.delenv("FOLDJAX_HOME")
    monkeypatch.setattr(paths, "_repository_store", lambda: None)
    assert paths.foldjax_home() == tmp_path / "xdg" / "foldjax"


def test_home_description_includes_generated_runtime_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))

    assert paths.describe()["runtime"] == str(tmp_path / "runtime")


def test_the_checkout_store_is_found_next_to_the_source(monkeypatch) -> None:
    """`_repository_store` resolves from this module, not the process's cwd."""
    root = Path(paths.__file__).resolve().parents[2]
    expected = root / ".foldjax"
    monkeypatch.chdir(Path(__file__).parent)
    assert paths._repository_store() == (expected if expected.is_dir() else None)


def test_every_registered_model_is_a_real_backend() -> None:
    for name in assets.available():
        assert name in foldjax_models(), name


def foldjax_models() -> tuple[str, ...]:
    from foldjax.registry import available_models

    return available_models()


def test_registry_declares_what_each_model_needs() -> None:
    for name in assets.available():
        spec = assets.REGISTRY[name]
        # AlphaFold 3 still has nothing public to download, but needs a registry
        # entry so the store knows where manually supplied parameters belong.
        # OpenFold3 p1 is public through its publisher's unsigned S3 bucket.
        if name != "alphafold3":
            assert spec.downloads, name
        assert spec.requires, name
        assert spec.licence and spec.source, name
        # `native` is what --weights points at, so it has to be inside the
        # model's weight directory rather than an absolute path.
        assert not Path(spec.native).is_absolute(), name
        for url in (item.url for item in spec.downloads):
            assert url.startswith("https://"), (name, url)


def test_published_hashes_are_well_formed() -> None:
    for name in assets.available():
        for item in assets.REGISTRY[name].downloads:
            if item.sha256 is None:
                continue
            assert len(item.sha256) == 64, (name, item.name)
            assert set(item.sha256) <= set("0123456789abcdef"), (name, item.name)


def test_every_automatic_download_has_a_published_identity() -> None:
    for name in assets.available():
        for item in assets.REGISTRY[name].downloads:
            assert item.sha256 is not None or item.size is not None, (
                name,
                item.name,
            )


def test_every_conversion_source_has_a_hash_and_schema() -> None:
    for spec in assets.REGISTRY.values():
        if not spec.conversion_sources:
            continue
        assert spec.conversion_schema
        published = {item.name: item for item in spec.downloads}
        for name in spec.conversion_sources:
            assert published[name].sha256 is not None, (spec.model, name)


def test_a_directory_model_resolves_to_its_asset_root_not_a_single_file(
    tmp_path: Path, monkeypatch
) -> None:
    """`native = "."` means the model reads a directory.

    AlphaFold 3 is the one that does: it loads its own parameter file from a
    directory it is handed, so returning a single path from `resolve_weights`
    would fail only at inference.
    """
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    root = paths.weights_dir("alphafold3")
    root.mkdir(parents=True)
    (root / "af3.bin").write_bytes(b"parameters")

    assert assets.resolve_weights("alphafold3") == root
    assert assets.assets_for("alphafold3").ready()


def test_a_partially_converted_model_is_not_reported_ready(
    tmp_path: Path, monkeypatch
) -> None:
    """One of several required artifacts is not a usable installation.

    The next test covers what `ready()` reports; what this one adds is that
    `resolve_weights` refuses rather than handing back a path to a half-built
    directory, and names the command that finishes the job.
    """
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    root = paths.weights_dir("boltz2")
    root.mkdir(parents=True)
    (root / "boltz2_conf.safetensors").write_bytes(b"weights")  # mols/ missing

    assert not assets.assets_for("boltz2").ready()
    with pytest.raises(FileNotFoundError, match="foldjax weights fetch"):
        assets.resolve_weights("boltz2")


def test_boltz_is_not_ready_without_a_populated_molecule_directory(
    tmp_path: Path, monkeypatch
) -> None:
    """Boltz reads CCD pickles; an absent or empty directory cannot predict."""
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    root = paths.weights_dir("boltz2")
    _write_boltz_native_bundle(root)

    assert not assets.assets_for("boltz2").ready()
    (root / "mols").mkdir()
    assert not assets.assets_for("boltz2").ready()
    for name in dict(
        assets.assets_for("boltz2").directory_requires
    )["mols"]:
        (root / "mols" / name).write_bytes(b"molecule")
    # All canonical residues can be present while extraction is still hundreds
    # of files short; the full pinned archive count is part of readiness.
    assert not assets.assets_for("boltz2").ready()
    monkeypatch.setattr(
        assets, "_BOLTZ_MOLECULE_COUNT", len(assets._BOLTZ_CANONICAL_MOLECULES)
    )
    assets._write_text_atomic(
        root / "mols" / ".foldjax-complete", assets._boltz_molecule_marker()
    )
    assert assets.assets_for("boltz2").ready()


def test_a_random_file_does_not_make_boltz_molecules_ready(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    root = paths.weights_dir("boltz2")
    (root / "mols").mkdir(parents=True)
    _write_boltz_native_bundle(root)
    (root / "mols" / "README").write_text("interrupted extraction")

    assert not assets.assets_for("boltz2").ready()


def test_a_complete_legacy_boltz_molecule_store_needs_no_marker(
    tmp_path: Path, monkeypatch
) -> None:
    names = (*assets._BOLTZ_CANONICAL_MOLECULES, "EXTRA.pkl")
    monkeypatch.setattr(assets, "_BOLTZ_MOLECULE_COUNT", len(names))
    destination = tmp_path / "mols"
    destination.mkdir()
    for name in names:
        (destination / name).write_bytes(b"molecule")

    assert not assets._boltz_marker_complete(destination)
    assert assets._boltz_molecules_complete(destination)


def test_a_marked_boltz_store_missing_a_noncanonical_file_is_incomplete(
    tmp_path: Path, monkeypatch
) -> None:
    names = (*assets._BOLTZ_CANONICAL_MOLECULES, "EXTRA.pkl")
    monkeypatch.setattr(assets, "_BOLTZ_MOLECULE_COUNT", len(names))
    destination = tmp_path / "mols"
    destination.mkdir()
    for name in names:
        (destination / name).write_bytes(b"molecule")
    assets._write_text_atomic(
        destination / ".foldjax-complete", assets._boltz_molecule_marker()
    )
    (destination / "EXTRA.pkl").unlink()

    assert assets._boltz_marker_complete(destination)
    assert not assets._boltz_molecules_complete(destination)


def test_boltz_completion_index_detects_in_place_damage_without_count_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    names = (*assets._BOLTZ_CANONICAL_MOLECULES, "EXTRA.pkl")
    monkeypatch.setattr(assets, "_BOLTZ_MOLECULE_COUNT", len(names))
    destination = tmp_path / "mols"
    destination.mkdir()
    for name in names:
        (destination / name).write_bytes(b"molecule")
    assets._write_boltz_molecule_completion(destination)
    count = assets._boltz_molecule_count
    monkeypatch.setattr(
        assets,
        "_boltz_molecule_count",
        lambda *_: pytest.fail("current completion index should avoid a full scan"),
    )

    assert assets._boltz_molecules_complete(destination)

    (destination / "EXTRA.pkl").write_bytes(b"")
    assert not assets._boltz_molecules_complete(destination)
    monkeypatch.setattr(assets, "_boltz_molecule_count", count)


def _write_molecule_archive(path: Path, names: tuple[str, ...]) -> None:
    source = path.parent / "archive-source"
    source.mkdir()
    for name in names:
        (source / name).write_bytes(b"molecule")
    with tarfile.open(path, "w") as archive:
        archive.add(source, arcname="mols")


def test_boltz_molecule_marker_is_written_only_for_a_complete_archive(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "mols.tar"
    _write_molecule_archive(archive, ("ALA.pkl",))
    destination = tmp_path / "weights" / "mols"

    with pytest.raises(RuntimeError, match="complete published set"):
        assets._unpack_boltz_mols(archive, destination)

    assert not (destination / ".foldjax-complete").exists()


def test_boltz_stale_marker_does_not_prevent_repair(
    tmp_path: Path, monkeypatch
) -> None:
    archive = tmp_path / "mols.tar"
    names = tuple(assets._BOLTZ_CANONICAL_MOLECULES)
    monkeypatch.setattr(assets, "_BOLTZ_MOLECULE_COUNT", len(names))
    _write_molecule_archive(archive, names)
    destination = tmp_path / "weights" / "mols"
    destination.mkdir(parents=True)
    (destination / ".foldjax-complete").write_text("stale\n")

    assets._unpack_boltz_mols(archive, destination)

    assert assets._boltz_molecules_complete(destination)
    assert assets._boltz_marker_complete(destination)


def test_boltz_repair_replaces_extra_molecule_files(
    tmp_path: Path, monkeypatch
) -> None:
    archive = tmp_path / "mols.tar"
    names = tuple(assets._BOLTZ_CANONICAL_MOLECULES)
    monkeypatch.setattr(assets, "_BOLTZ_MOLECULE_COUNT", len(names))
    _write_molecule_archive(archive, names)
    destination = tmp_path / "weights" / "mols"
    destination.mkdir(parents=True)
    (destination / "stale.pkl").write_bytes(b"not published")

    assets._unpack_boltz_mols(archive, destination)

    assert not (destination / "stale.pkl").exists()
    assert assets._boltz_molecules_complete(destination)


@pytest.mark.parametrize("missing", assets._BOLTZ_NATIVE_FILES)
def test_boltz_requires_both_native_bundles_and_their_sidecars(
    tmp_path: Path, monkeypatch, missing: str
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    monkeypatch.setattr(
        assets, "_BOLTZ_MOLECULE_COUNT", len(assets._BOLTZ_CANONICAL_MOLECULES)
    )
    root = paths.weights_dir("boltz2")
    _write_boltz_native_bundle(root)
    (root / missing).unlink()
    molecules = root / "mols"
    molecules.mkdir()
    for name in assets._BOLTZ_CANONICAL_MOLECULES:
        (molecules / name).write_bytes(b"molecule")

    assert not assets.assets_for("boltz2").ready()


def test_boltz_in_progress_marker_blocks_a_nominally_complete_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    monkeypatch.setattr(
        assets, "_BOLTZ_MOLECULE_COUNT", len(assets._BOLTZ_CANONICAL_MOLECULES)
    )
    root = paths.weights_dir("boltz2")
    _write_boltz_native_bundle(root)
    molecules = root / "mols"
    molecules.mkdir()
    for name in assets._BOLTZ_CANONICAL_MOLECULES:
        (molecules / name).write_bytes(b"molecule")
    (root / assets._BOLTZ_CONVERSION_MARKER).write_text("interrupted\n")

    assert not assets.assets_for("boltz2").ready()


@pytest.mark.parametrize(
    "corrupt",
    ["boltz2_conf.safetensors", "boltz2_aff.safetensors.json"],
)
def test_boltz_rejects_truncated_weights_and_malformed_sidecars(
    tmp_path: Path, monkeypatch, corrupt: str
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    monkeypatch.setattr(
        assets, "_BOLTZ_MOLECULE_COUNT", len(assets._BOLTZ_CANONICAL_MOLECULES)
    )
    root = paths.weights_dir("boltz2")
    _write_boltz_native_bundle(root)
    molecules = root / "mols"
    molecules.mkdir()
    for name in assets._BOLTZ_CANONICAL_MOLECULES:
        (molecules / name).write_bytes(b"molecule")
    (root / corrupt).write_bytes(b"truncated")

    assert not assets.assets_for("boltz2").ready()


def test_boltz_native_manifest_detects_post_conversion_changes(
    tmp_path: Path,
) -> None:
    _write_boltz_native_bundle(tmp_path)
    assets._write_boltz_native_manifest(tmp_path)
    assert assets._boltz_native_complete(tmp_path)

    sidecar = tmp_path / "boltz2_conf.safetensors.json"
    sidecar.write_text(sidecar.read_text() + "\n")

    assert not assets._boltz_native_complete(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("keys", ["different"]),
        ("scalars", []),
        ("scalars", {"bad-path": True}),
        ("scalars", {"d:flag": "not a native scalar"}),
        ("scalars", {"d:weight": True}),
    ],
)
def test_boltz_sidecars_must_describe_the_exact_tensor_bundle(
    tmp_path: Path, field: str, value: object
) -> None:
    _write_boltz_native_bundle(tmp_path)
    sidecar = tmp_path / "boltz2_conf.safetensors.json"
    payload = json.loads(sidecar.read_text())
    payload[field] = value
    sidecar.write_text(json.dumps(payload))

    assert not assets._boltz_native_files_valid(tmp_path)


@pytest.mark.parametrize(
    "entry",
    [
        {"shape": [1], "data_offsets": [0, 1]},
        {"dtype": "NOT_REAL", "shape": [1], "data_offsets": [0, 1]},
        {"dtype": "U8", "shape": [2], "data_offsets": [0, 1]},
        {"dtype": "U8", "shape": [-1], "data_offsets": [0, 1]},
    ],
)
def test_boltz_rejects_unloadable_safetensors_entries(
    tmp_path: Path, entry: dict[str, object]
) -> None:
    _write_boltz_native_bundle(tmp_path)
    header = json.dumps({"d:weight": entry}).encode()
    (tmp_path / "boltz2_conf.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header + b"x"
    )

    assert not assets._boltz_native_files_valid(tmp_path)


def test_boltz_rejects_tensor_keys_the_native_loader_cannot_decode(
    tmp_path: Path,
) -> None:
    _write_boltz_native_bundle(tmp_path)
    header = json.dumps(
        {"weight": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}
    ).encode()
    (tmp_path / "boltz2_conf.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header + b"x"
    )
    sidecar = tmp_path / "boltz2_conf.safetensors.json"
    sidecar.write_text(
        json.dumps({"backend": "safetensors", "keys": ["weight"], "scalars": {}})
    )

    assert not assets._boltz_native_files_valid(tmp_path)


def test_boltz_native_manifest_tracks_registry_source_identity(
    tmp_path: Path, monkeypatch
) -> None:
    _write_boltz_native_bundle(tmp_path)
    assets._write_boltz_native_manifest(tmp_path)
    assert assets._boltz_native_complete(tmp_path)

    spec = assets.REGISTRY["boltz2"]
    downloads = list(spec.downloads)
    downloads[0] = dataclasses.replace(downloads[0], sha256="0" * 64)
    monkeypatch.setattr(
        assets,
        "REGISTRY",
        {
            **assets.REGISTRY,
            "boltz2": dataclasses.replace(spec, downloads=tuple(downloads)),
        },
    )

    assert not assets._boltz_native_complete(tmp_path)


def test_generic_native_manifest_tracks_source_and_converter_identity(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    spec = assets.REGISTRY["opendde"]
    native = spec.native_path()
    native.parent.mkdir(parents=True)
    native.write_bytes(b"native weights")
    assets._write_native_manifest(spec)
    assert spec.ready()

    downloads = tuple(
        dataclasses.replace(item, sha256="0" * 64)
        if item.name == "opendde.pt"
        else item
        for item in spec.downloads
    )
    changed = dataclasses.replace(spec, downloads=downloads)
    monkeypatch.setattr(
        assets, "REGISTRY", {**assets.REGISTRY, "opendde": changed}
    )
    assert not changed.ready()

    changed = dataclasses.replace(spec, conversion_schema="opendde-native-v2")
    monkeypatch.setattr(
        assets, "REGISTRY", {**assets.REGISTRY, "opendde": changed}
    )
    assert not changed.ready()


@pytest.mark.parametrize("legacy_schema", [None, "test-native-v1"])
def test_fetch_rebuilds_unproven_or_stale_native_weights(
    tmp_path: Path, monkeypatch, legacy_schema: str | None
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    payload = b"published checkpoint"
    source = assets.Download(
        name="opendde.pt",
        url="https://example.invalid/opendde.pt",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )
    conversions = []

    def convert(model: str, _source: Path) -> Path:
        conversions.append(model)
        native = paths.weights_dir(model) / "opendde.jax"
        native.write_bytes(b"rebuilt native weights")
        return native

    current = dataclasses.replace(
        assets.REGISTRY["opendde"],
        downloads=(source,),
        convert=convert,
        conversion_sources=(source.name,),
        conversion_schema="test-native-v2",
    )
    native = current.native_path()
    native.parent.mkdir(parents=True)
    native.write_bytes(b"legacy native weights")
    if legacy_schema is not None:
        assets._write_native_manifest(
            dataclasses.replace(current, conversion_schema=legacy_schema)
        )
    monkeypatch.setattr(
        assets, "REGISTRY", {**assets.REGISTRY, "opendde": current}
    )

    def fake_download(item, model, *, on_progress=None):
        target = item.target(model)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return target

    monkeypatch.setattr(assets, "download", fake_download)

    assert assets.fetch("opendde") == native
    assert conversions == ["opendde"]
    assert native.read_bytes() == b"rebuilt native weights"
    assert current.ready()


def test_fast_fetch_repairs_same_size_corruption_in_direct_assets(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    checkpoint_payload = b"published checkpoint"
    metadata_payload = b"published metadata"
    checkpoint = assets.Download(
        name="opendde.pt",
        url="https://example.invalid/opendde.pt",
        sha256=hashlib.sha256(checkpoint_payload).hexdigest(),
        size=len(checkpoint_payload),
    )
    metadata = assets.Download(
        name="foldjax-test-metadata.json",
        url="https://example.invalid/metadata.json",
        sha256=hashlib.sha256(metadata_payload).hexdigest(),
        size=len(metadata_payload),
        shared=True,
    )
    current = dataclasses.replace(
        assets.REGISTRY["opendde"],
        downloads=(checkpoint, metadata),
        convert=lambda *_args, **_kwargs: pytest.fail(
            "a current native conversion must not be rebuilt"
        ),
        conversion_sources=(checkpoint.name,),
        conversion_schema="test-native-v2",
    )
    native = current.native_path()
    native.parent.mkdir(parents=True)
    native.write_bytes(b"current native weights")
    checkpoint.target("opendde").parent.mkdir(parents=True, exist_ok=True)
    checkpoint.target("opendde").write_bytes(checkpoint_payload)
    metadata.target("opendde").parent.mkdir(parents=True, exist_ok=True)
    metadata.target("opendde").write_bytes(b"x" * len(metadata_payload))
    assets._write_native_manifest(current)
    monkeypatch.setattr(
        assets, "REGISTRY", {**assets.REGISTRY, "opendde": current}
    )
    downloaded = []

    def fake_download(item, model, *, on_progress=None):
        downloaded.append(item.name)
        target = item.target(model)
        target.write_bytes(metadata_payload)
        return target

    monkeypatch.setattr(assets, "download", fake_download)

    assert assets.fetch("opendde") == native
    assert downloaded == [metadata.name]
    assert metadata.target("opendde").read_bytes() == metadata_payload


def test_download_only_fully_verifies_conversion_sources(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    payload = b"published checkpoint"
    checkpoint = assets.Download(
        name="opendde.pt",
        url="https://example.invalid/opendde.pt",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )
    current = dataclasses.replace(
        assets.REGISTRY["opendde"],
        downloads=(checkpoint,),
        conversion_sources=(checkpoint.name,),
        conversion_schema="test-native-v2",
    )
    native = current.native_path()
    native.parent.mkdir(parents=True)
    native.write_bytes(b"current native weights")
    target = checkpoint.target("opendde")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"x" * len(payload))
    assets._write_native_manifest(current)
    monkeypatch.setattr(
        assets, "REGISTRY", {**assets.REGISTRY, "opendde": current}
    )
    downloaded = []

    def fake_download(item, model, *, on_progress=None):
        downloaded.append(item.name)
        item.target(model).write_bytes(payload)
        return item.target(model)

    monkeypatch.setattr(assets, "download", fake_download)

    assert assets.fetch("opendde", convert=False) == paths.downloads_dir(
        "opendde"
    )
    assert downloaded == [checkpoint.name]
    assert target.read_bytes() == payload


def test_download_only_refuses_models_with_no_downloadable_parameters(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))

    with pytest.raises(RuntimeError, match="no downloadable parameter files"):
        assets.fetch("alphafold3", convert=False)


@pytest.mark.parametrize("fail", [False, True])
def test_opendde_conversion_publishes_atomically(
    tmp_path: Path, monkeypatch, fail: bool
) -> None:
    from foldjax.models.opendde.bridge import export_weights

    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    source = paths.downloads_dir("opendde")
    source.mkdir(parents=True)
    (source / "opendde.pt").write_bytes(b"checkpoint")
    final = paths.weights_dir("opendde") / "opendde.jax"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"previous complete weights")
    monkeypatch.setattr(export_weights, "load_torch_checkpoint", lambda _path: {})

    def save(path, _params, *, compress):
        assert compress is False
        assert Path(path) != final
        assert final.read_bytes() == b"previous complete weights"
        Path(path).write_bytes(b"new native weights")
        if fail:
            raise RuntimeError("conversion interrupted")

    monkeypatch.setattr(export_weights, "save_native_weights", save)

    if fail:
        with pytest.raises(RuntimeError, match="conversion interrupted"):
            assets._convert_opendde("opendde", source)
        assert final.read_bytes() == b"previous complete weights"
    else:
        assert assets._convert_opendde("opendde", source) == final
        assert final.read_bytes() == b"new native weights"
    assert not list(final.parent.glob(".foldjax-opendde-native-*"))


def test_safetensors_probe_rejects_an_oversized_header_before_reading_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "hostile.safetensors"
    header_size = assets._SAFETENSORS_MAX_HEADER + 1
    with path.open("wb") as handle:
        handle.write(header_size.to_bytes(8, "little"))
        handle.truncate(header_size + 9)

    assert not assets._valid_safetensors(path)


def test_a_zero_byte_required_weight_is_not_ready(
    tmp_path: Path, monkeypatch
) -> None:
    """An interrupted conversion must not become a permanent ready install."""
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    root = paths.weights_dir("protenix")
    root.mkdir(parents=True)
    (root / "protenix_base_default_v1.0.0.jax").touch()

    assert not assets.assets_for("protenix").ready()


def test_esmfold2_requires_only_the_supported_inference_bundle() -> None:
    spec = assets.assets_for("esmfold2")
    required = set(spec.requires)
    downloads = {item.name for item in spec.downloads}
    assert {"model.safetensors", "config.json"} <= required
    assert "ccd.pkl" not in required
    assert "ccd.pkl" not in downloads
    assert "esmc/config.json" in required
    assert "esmc/model.safetensors.index.json" in required
    assert {
        f"esmc/model-{index:05d}-of-00006.safetensors" for index in range(1, 7)
    } <= required
    assert model_info("esmfold2").download_bytes == 26_347_754_143


def test_esmfold2_profiles_publish_their_exact_transfer_contract() -> None:
    assert assets.available_profiles("esmfold2") == (
        "released",
        "structure-only",
    )
    released = assets.assets_for("esmfold2", profile="released")
    structure = assets.assets_for("esmfold2", profile="structure-only")
    assert released is assets.assets_for("esmfold2")
    assert {item.name for item in structure.downloads} == {
        "model.safetensors",
        "config.json",
    }
    assert structure.requires == ("model.safetensors", "config.json")
    assert sum(item.size or 0 for item in structure.downloads) == 939_507_565
    assert "no_language_model=true" in structure.notes

    with pytest.raises(ValueError, match="unsupported asset profile.*esmfold2"):
        assets.assets_for("esmfold2", profile="tiny")
    with pytest.raises(ValueError, match="choose one of released"):
        assets.assets_for("opendde", profile="structure-only")


def test_protenix_profiles_publish_isolated_structure_and_encoder_bundles() -> None:
    assert assets.available_profiles("protenix") == (
        "released",
        "mini-esm-v0.5.0",
        "mini-ism-v0.5.0",
    )
    expected = {
        "mini-esm-v0.5.0": (
            "protenix-mini-esm",
            "protenix_mini_esm_v0.5.0.jax",
            "protenix_mini_esm_v0.5.0.pt",
            "esm2_t36_3B_UR50D.pt",
            541_640_990,
            5_678_116_398,
        ),
        "mini-ism-v0.5.0": (
            "protenix-mini-ism",
            "protenix_mini_ism_v0.5.0.jax",
            "protenix_mini_ism_v0.5.0.pt",
            "esm2_t36_3B_UR50D_ism.pt",
            541_640_990,
            11_356_246_722,
        ),
    }
    for profile, (
        internal_model,
        native,
        mini_name,
        embedding_name,
        mini_size,
        embedding_size,
    ) in expected.items():
        spec = assets.assets_for("protenix", profile=profile)
        published = {item.name: item for item in spec.downloads}
        assert spec.model == internal_model
        assert spec.native == native
        assert spec.requires == (native, embedding_name)
        assert spec.conversion_sources == (mini_name, embedding_name)
        assert spec.staged_sources == ((embedding_name, embedding_name),)
        assert published[mini_name].size == mini_size
        assert published[embedding_name].size == embedding_size
        assert published[mini_name].url.endswith(mini_name)
        assert published[embedding_name].url.endswith(embedding_name)
        # The CLI passes this internal root back through fetch after resolving
        # the public profile; that round trip must select the same contract.
        assert assets.assets_for(internal_model, profile=profile) == spec


def test_protenix_variant_fetch_converts_and_stages_in_one_isolated_root(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    profile = "mini-esm-v0.5.0"
    published = {
        "protenix_mini_esm_v0.5.0.pt": b"mini checkpoint",
        "esm2_t36_3B_UR50D.pt": b"esm checkpoint",
    }
    downloads = tuple(
        assets.Download(
            name=name,
            url=f"https://example.invalid/{name}",
            sha256=hashlib.sha256(payload).hexdigest(),
            size=len(payload),
        )
        for name, payload in published.items()
    )
    spec = dataclasses.replace(
        assets.assets_for("protenix", profile=profile),
        downloads=downloads,
    )
    original_assets_for = assets.assets_for

    def fake_assets_for(model: str, *, profile: str | None = None):
        if model in {"protenix", spec.model} and profile in {
            None,
            "mini-esm-v0.5.0",
        }:
            return spec
        return original_assets_for(model, profile=profile)

    def fake_download(item, model, *, on_progress=None):
        target = item.target(model)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(published[item.name])
        return target

    from foldjax.models.protenix.bridge import torch_mapping, weights_io

    monkeypatch.setattr(assets, "assets_for", fake_assets_for)
    monkeypatch.setattr(assets, "download", fake_download)
    monkeypatch.setattr(
        torch_mapping,
        "load_torch_checkpoint",
        lambda path: {"source": path.name},
    )

    def fake_save(path, state, *, compress):
        assert state == {"source": "protenix_mini_esm_v0.5.0.pt"}
        assert compress is False
        Path(path).write_bytes(b"native jax weights")

    monkeypatch.setattr(weights_io, "save_native_weights", fake_save)

    native = assets.fetch("protenix", profile=profile)
    root = paths.weights_dir("protenix-mini-esm")
    embedding = root / "esm2_t36_3B_UR50D.pt"
    assert native == root / "protenix_mini_esm_v0.5.0.jax"
    assert native.read_bytes() == b"native jax weights"
    assert embedding.read_bytes() == published[embedding.name]
    assert (root / assets._NATIVE_MANIFEST).is_file()
    assert spec.ready()
    assert not paths.weights_dir("protenix").exists()

    embedding.write_bytes(b"corrupted staged checkpoint")
    assert not spec.ready()


def test_protenix_variant_adopts_verified_legacy_download_without_copying(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    payload = b"official esm checkpoint"
    item = assets.Download(
        name="esm2_t36_3B_UR50D.pt",
        url="https://example.invalid/esm.pt",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )
    legacy = paths.downloads_dir("protenix") / item.name
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(payload)

    assets._adopt_legacy_protenix_variant_download(item, "protenix-mini-esm")

    adopted = item.target("protenix-mini-esm")
    assert adopted.read_bytes() == payload
    assert adopted.samefile(legacy)


def test_esmfold2_fetch_adopts_verified_staged_bundle_without_downloading(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    payloads = {
        "model.safetensors": b"structure weights",
        "config.json": b"structure config",
        "esmc/model-00001-of-00001.safetensors": b"language model shard",
    }
    downloads = tuple(
        assets.Download(
            name=name,
            url=f"https://example.invalid/{name}",
            sha256=hashlib.sha256(payload).hexdigest(),
            size=len(payload),
        )
        for name, payload in payloads.items()
    )
    spec = dataclasses.replace(
        assets.REGISTRY["esmfold2"],
        downloads=downloads,
        requires=tuple(payloads),
        ready_check=assets._esmfold2_ready,
    )
    monkeypatch.setattr(
        assets, "REGISTRY", {**assets.REGISTRY, "esmfold2": spec}
    )
    staged_root = paths.weights_dir("esmfold2")
    for name, payload in payloads.items():
        staged = staged_root / name
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(payload)

    def unexpected_download(item, model, *, on_progress=None):
        pytest.fail(f"unexpected download of {item.name} for {model}")

    monkeypatch.setattr(assets, "download", unexpected_download)

    result = assets.fetch("esmfold2")

    assert result == staged_root / "model.safetensors"
    for name in payloads:
        adopted = paths.downloads_dir("esmfold2") / name
        assert adopted.samefile(staged_root / name)
    manifest = json.loads((staged_root / assets._ESMFOLD2_MARKER).read_text())
    assert set(manifest["files"]) == set(payloads)
    assert spec.ready()


def test_esmfold2_staged_adoption_falls_back_to_an_atomic_copy(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    payload = b"published structure weights"
    item = assets.Download(
        name="model.safetensors",
        url="https://example.invalid/model.safetensors",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )
    staged = paths.weights_dir("esmfold2") / item.name
    staged.parent.mkdir(parents=True)
    staged.write_bytes(payload)
    monkeypatch.setattr(
        assets.os,
        "link",
        lambda source, target: (_ for _ in ()).throw(OSError("cross-device")),
    )

    assert assets._adopt_staged_esmfold2_download(item, "esmfold2")

    adopted = item.target("esmfold2")
    assert adopted.read_bytes() == payload
    assert not adopted.samefile(staged)
    assert not list(adopted.parent.glob(f".{adopted.name}.adopt-*"))


@pytest.mark.parametrize("legacy_kind", ["corrupt", "symlink", "partial"])
def test_esmfold2_fetch_rejects_unverified_staged_sources(
    tmp_path: Path, monkeypatch, legacy_kind: str
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    payload = b"published structure weights"
    item = assets.Download(
        name="model.safetensors",
        url="https://example.invalid/model.safetensors",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )
    spec = dataclasses.replace(
        assets.REGISTRY["esmfold2"],
        downloads=(item,),
        requires=(item.name,),
        ready_check=assets._esmfold2_ready,
    )
    monkeypatch.setattr(
        assets, "REGISTRY", {**assets.REGISTRY, "esmfold2": spec}
    )
    staged = paths.weights_dir("esmfold2") / item.name
    staged.parent.mkdir(parents=True)
    if legacy_kind == "corrupt":
        staged.write_bytes(b"x" * len(payload))
    elif legacy_kind == "symlink":
        external = tmp_path / "external.safetensors"
        external.write_bytes(payload)
        staged.symlink_to(external)
    else:
        staged.with_suffix(staged.suffix + ".part").write_bytes(payload)

    downloaded = []

    def fake_download(download_item, model, *, on_progress=None):
        downloaded.append(download_item.name)
        target = download_item.target(model)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return target

    monkeypatch.setattr(assets, "download", fake_download)

    assets.fetch("esmfold2")

    assert downloaded == [item.name]
    assert not staged.is_symlink()
    assert staged.read_bytes() == payload
    assert spec.ready()


def test_esmfold2_structure_only_adopts_only_its_selected_files(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    payloads = {
        "model.safetensors": b"structure weights",
        "config.json": b"structure config",
        "esmc/model.safetensors": b"language model weights",
    }
    downloads = tuple(
        assets.Download(
            name=name,
            url=f"https://example.invalid/{name}",
            sha256=hashlib.sha256(payload).hexdigest(),
            size=len(payload),
        )
        for name, payload in payloads.items()
    )
    spec = dataclasses.replace(
        assets.REGISTRY["esmfold2"],
        downloads=downloads,
        requires=tuple(payloads),
        ready_check=assets._esmfold2_ready,
    )
    monkeypatch.setattr(
        assets, "REGISTRY", {**assets.REGISTRY, "esmfold2": spec}
    )
    staged_root = paths.weights_dir("esmfold2")
    for name, payload in payloads.items():
        staged = staged_root / name
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(payload)

    def unexpected_download(item, model, *, on_progress=None):
        pytest.fail(f"unexpected download of {item.name} for {model}")

    monkeypatch.setattr(assets, "download", unexpected_download)

    assets.fetch("esmfold2", profile="structure-only")

    download_root = paths.downloads_dir("esmfold2")
    assert (download_root / "model.safetensors").is_file()
    assert (download_root / "config.json").is_file()
    assert not (download_root / "esmc").exists()
    manifest = json.loads((staged_root / assets._ESMFOLD2_MARKER).read_text())
    assert set(manifest["files"]) == {"model.safetensors", "config.json"}
    assert assets.assets_for("esmfold2", profile="structure-only").ready()
    assert not assets.assets_for("esmfold2", profile="released").ready()


def test_missing_protenix_variant_points_to_the_public_profile_command(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))

    with pytest.raises(FileNotFoundError) as error:
        assets.resolve_weights("protenix", profile="mini-ism-v0.5.0")

    message = str(error.value)
    assert (
        "foldjax weights fetch --model protenix --profile mini-ism-v0.5.0"
        in message
    )
    assert "--model protenix-mini-ism" not in message


def test_esmfold2_structure_only_fetch_never_downloads_esmc(
    tmp_path: Path, monkeypatch
) -> None:
    """The smaller variant is independently stageable and never implies ESMC."""
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    payloads = {
        "model.safetensors": b"structure",
        "config.json": b"config",
        "esmc/config.json": b"esmc",
    }
    downloads = tuple(
        assets.Download(
            name=name,
            url=f"https://example.invalid/{name}",
            sha256=hashlib.sha256(payload).hexdigest(),
            size=len(payload),
        )
        for name, payload in payloads.items()
    )
    base = dataclasses.replace(
        assets.REGISTRY["esmfold2"],
        downloads=downloads,
        requires=tuple(payloads),
        ready_check=assets._esmfold2_ready,
    )
    monkeypatch.setattr(
        assets, "REGISTRY", {**assets.REGISTRY, "esmfold2": base}
    )
    fetched = []

    def fake_download(item, model, *, on_progress=None):
        fetched.append(item.name)
        target = item.target(model)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payloads[item.name])
        return target

    monkeypatch.setattr(assets, "download", fake_download)

    result = assets.fetch("esmfold2", profile="structure-only")
    assert result == paths.weights_dir("esmfold2") / "model.safetensors"
    assert fetched == ["model.safetensors", "config.json"]
    assert not (paths.downloads_dir("esmfold2") / "esmc/config.json").exists()
    assert assets.assets_for("esmfold2", profile="structure-only").ready()
    assert not assets.assets_for("esmfold2", profile="released").ready()
    assert assets.resolve_weights("esmfold2", profile="structure-only") == result
    with pytest.raises(
        FileNotFoundError,
        match=r"weights fetch --model esmfold2`",
    ):
        assets.resolve_weights("esmfold2")

    profiles = {
        row["profile"]: row for row in assets.profile_status("esmfold2")
    }
    assert profiles["structure-only"]["ready"] is True
    assert profiles["structure-only"]["download_bytes"] == len(
        payloads["model.safetensors"]
    ) + len(payloads["config.json"])
    assert profiles["released"]["ready"] is False


def test_esmfold2_released_bundle_satisfies_both_profiles(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    payloads = {
        "model.safetensors": b"structure",
        "config.json": b"config",
        "esmc/config.json": b"esmc",
    }
    downloads = tuple(
        assets.Download(
            name=name,
            url=f"https://example.invalid/{name}",
            sha256=hashlib.sha256(payload).hexdigest(),
            size=len(payload),
        )
        for name, payload in payloads.items()
    )
    base = dataclasses.replace(
        assets.REGISTRY["esmfold2"],
        downloads=downloads,
        requires=tuple(payloads),
        ready_check=assets._esmfold2_ready,
    )
    monkeypatch.setattr(
        assets, "REGISTRY", {**assets.REGISTRY, "esmfold2": base}
    )

    def fake_download(item, model, *, on_progress=None):
        target = item.target(model)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payloads[item.name])
        return target

    monkeypatch.setattr(assets, "download", fake_download)
    assets.fetch("esmfold2")

    assert assets.assets_for("esmfold2", profile="released").ready()
    assert assets.assets_for("esmfold2", profile="structure-only").ready()


def test_esmfold2_staging_atomically_repairs_a_partial_target(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    source = paths.downloads_dir("esmfold2")
    source.mkdir(parents=True)
    published = source / "model.safetensors"
    payload = b"complete published checkpoint"
    published.write_bytes(payload)
    target = paths.weights_dir("esmfold2") / "model.safetensors"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"")
    monkeypatch.setattr(
        assets.os,
        "link",
        lambda source, target: (_ for _ in ()).throw(OSError("cross-device")),
    )
    item = assets.Download(
        name="model.safetensors",
        url="https://example.invalid/model.safetensors",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )
    spec = dataclasses.replace(
        assets.REGISTRY["esmfold2"], downloads=(item,), requires=(item.name,)
    )
    monkeypatch.setattr(
        assets, "REGISTRY", {**assets.REGISTRY, "esmfold2": spec}
    )

    assets._stage_esmfold2("esmfold2", source)

    assert target.read_bytes() == published.read_bytes()
    assert not list(target.parent.glob(".foldjax-stage-*"))


def test_openfold3_public_checkpoint_is_staged_atomically(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    payload = b"public-openfold3-p1"
    item = assets.Download(
        name="of3_ft3_v1.pt",
        url="https://example.invalid/of3_ft3_v1.pt",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )
    spec = dataclasses.replace(
        assets.REGISTRY["openfold3"],
        downloads=(item,),
        conversion_sources=(item.name,),
    )
    source = paths.downloads_dir("openfold3") / item.name
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)

    target = assets._stage_single_published_file(
        "openfold3", paths.downloads_dir("openfold3"), spec=spec
    )

    assert target == paths.weights_dir("openfold3") / item.name
    assert target.read_bytes() == payload
    assert not list(target.parent.glob(".foldjax-stage-*"))


@pytest.mark.parametrize(
    "model", ["boltz2", "esmfold2", "opendde", "openfold3", "protenix"]
)
def test_conversion_lock_cleans_only_known_abandoned_staging(
    tmp_path: Path, monkeypatch, model: str
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    root = paths.weights_dir(model)
    root.mkdir(parents=True)
    if model == "boltz2":
        abandoned = root / ".foldjax-mols-dead"
    elif model in {"esmfold2", "openfold3"}:
        abandoned = root / ".foldjax-stage-dead"
    else:
        abandoned = root / f".foldjax-{model}-native-dead"
    abandoned.mkdir()
    (abandoned / "partial").write_bytes(b"partial")
    unrelated = root / ".user-staging"
    unrelated.mkdir()

    with assets._conversion_lock(model):
        assert not abandoned.exists()
        assert unrelated.is_dir()


def test_esmfold2_ready_rejects_same_size_corruption(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    payload = b"published"
    item = assets.Download(
        name="model.safetensors",
        url="https://example.invalid/model.safetensors",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )
    spec = dataclasses.replace(
        assets.REGISTRY["esmfold2"], downloads=(item,), requires=(item.name,)
    )
    monkeypatch.setattr(
        assets, "REGISTRY", {**assets.REGISTRY, "esmfold2": spec}
    )
    source = paths.downloads_dir("esmfold2") / item.name
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    target = paths.weights_dir("esmfold2") / item.name
    target.parent.mkdir(parents=True)
    target.write_bytes(b"corrupted")

    assert not spec.ready()
    assets._stage_esmfold2("esmfold2", paths.downloads_dir("esmfold2"))
    assert target.read_bytes() == payload
    assert spec.ready()


def test_managed_weight_conversion_is_torch_free() -> None:
    assert all(not spec.needs_torch for spec in assets.REGISTRY.values())
    assert "needs no PyTorch installation" in assets.assets_for("openfold3").notes
    assert "torch-free" in assets.assets_for("boltz2").notes


def test_missing_managed_weights_do_not_request_a_removed_torch_extra(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    with pytest.raises(FileNotFoundError) as error:
        assets.resolve_weights("protenix")

    message = str(error.value)
    assert "foldjax weights fetch --model protenix" in message
    assert "torch-bridge" not in message


def test_manual_alphafold3_resolution_gives_the_real_setup_step(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))

    with pytest.raises(FileNotFoundError) as error:
        assets.resolve_weights("alphafold3")

    message = str(error.value)
    assert "Request the parameters from DeepMind" in message
    assert "weights fetch" not in message
    assert str(paths.weights_dir("alphafold3")) in message


def test_shared_assets_are_not_duplicated_per_model(
    tmp_path: Path, monkeypatch
) -> None:
    """Protenix and OpenDDE read the same CCD files; one copy, not two."""
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    shared = [
        item
        for item in assets.REGISTRY["protenix"].downloads
        if item.name == "components.cif"
    ]
    assert shared, "components.cif is no longer registered"
    item = shared[0]
    assert item.shared
    shared_path = paths.assets_dir() / item.name
    assert item.target("protenix") == shared_path
    assert item.target("opendde") == shared_path


def _serve(tmp_path: Path, payload: bytes) -> str:
    source = tmp_path / "published.bin"
    source.write_bytes(payload)
    return source.resolve().as_uri()


def test_download_verifies_the_published_hash(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    payload = b"native weights would go here" * 64
    item = assets.Download(
        name="thing.bin",
        url=_serve(tmp_path, payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )
    path = assets.download(item, "protenix")
    assert path.read_bytes() == payload

    # A second call is a no-op rather than a re-download.
    stamp = path.stat().st_mtime_ns
    assert assets.download(item, "protenix").stat().st_mtime_ns == stamp


def test_openfold3_fetch_downloads_stages_and_records_public_source(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    payload = b"small stand-in for public p1 weights"
    item = assets.Download(
        name="of3_ft3_v1.pt",
        url=_serve(tmp_path, payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )
    spec = dataclasses.replace(
        assets.REGISTRY["openfold3"],
        downloads=(item,),
        conversion_sources=(item.name,),
    )
    monkeypatch.setattr(
        assets, "REGISTRY", {**assets.REGISTRY, "openfold3": spec}
    )

    result = assets.fetch("openfold3")

    assert result == paths.weights_dir("openfold3") / item.name
    assert result.read_bytes() == payload
    assert assets._native_manifest_complete(spec)
    assert assets.fetch("openfold3") == result


def test_fresh_download_hashes_a_large_payload_only_once(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    payload = b"published bytes" * 64
    item = assets.Download(
        name="thing.bin",
        url=_serve(tmp_path, payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )
    digest = assets._digest
    calls: list[Path] = []

    def counted(path: Path) -> str:
        calls.append(path)
        return digest(path)

    monkeypatch.setattr(assets, "_digest", counted)
    assert assets.download(item, "protenix").read_bytes() == payload
    assert len(calls) == 1


def test_a_corrupted_download_fails_loudly_and_leaves_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    item = assets.Download(
        name="thing.bin",
        url=_serve(tmp_path, b"not what was published"),
        sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="failed verification"):
        assets.download(item, "protenix")
    assert not (paths.downloads_dir("protenix") / "thing.bin").exists()
    assert not list(paths.downloads_dir("protenix").glob("*.part"))


def test_hash_mismatch_is_retried_before_failing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    payload = b"published"
    item = assets.Download(
        name="thing.bin",
        url="https://example.invalid/thing.bin",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )
    attempts = []

    def fake_stream(item, partial, *, on_progress=None):
        attempts.append(1)
        partial.write_bytes(b"corrupted" if len(attempts) == 1 else payload)

    monkeypatch.setattr(assets, "_stream", fake_stream)

    assert assets.download(item, "protenix").read_bytes() == payload
    assert len(attempts) == 2


def test_a_truncated_file_is_re_downloaded(tmp_path: Path, monkeypatch) -> None:
    """A size mismatch means the last attempt died mid-write."""
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    payload = b"complete payload"
    item = assets.Download(
        name="thing.bin", url=_serve(tmp_path, payload), size=len(payload)
    )
    target = paths.downloads_dir("protenix") / "thing.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(payload[:4])

    assert assets.download(item, "protenix").read_bytes() == payload


def test_status_reports_progress_per_model(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    rows = {row["model"]: row for row in assets.status()}
    assert set(rows) == set(assets.available())
    assert all(row["converted"] is False for row in rows.values())
    assert all(row["downloaded"].startswith("0/") for row in rows.values())


def test_every_registered_backend_has_managed_assets() -> None:
    """`openfold3` used to be the counter-example, until it was vendored.

    With an entry for every backend, "has no managed assets" is no longer
    reachable through any real model name, so the invariant is what is worth
    asserting -- a new backend without a weight-store entry is the regression.
    """
    assert set(assets.available()) >= set(foldjax_models())


def test_an_unmanaged_model_says_so(monkeypatch) -> None:
    """The two failure modes must stay distinguishable.

    A name that is not a backend at all is an "unknown model"; a backend with no
    weight-store entry is "has no managed assets", and only the second means
    `--weights` would help. Since no real model now takes the second branch, it
    is reached by removing an entry rather than by naming a model that lacks
    one -- otherwise this test would have quietly stopped covering it.
    """
    with pytest.raises(ValueError, match="unknown model"):
        assets.assets_for("not-a-model")

    registry = dict(assets.REGISTRY)
    registry.pop("openfold3")
    monkeypatch.setattr(assets, "REGISTRY", registry)
    with pytest.raises(ValueError, match="has no managed assets"):
        assets.assets_for("openfold3")


def test_alphafold3_is_managed_but_never_downloaded() -> None:
    """Its parameters are not redistributable, so the store only locates them.

    The entry still has to declare what a ready installation looks like, or
    `--model alphafold3` would fall back to an unhelpful "no managed assets".
    """
    spec = assets.assets_for("alphafold3")
    assert spec.downloads == ()
    assert spec.requires == ("af3.bin",)
    assert spec.native == "."
    assert "redistributable" in spec.licence


@pytest.mark.parametrize(
    "names",
    [
        ("af3.bin.zst",),
        ("af3.0.bin", "af3.1.bin"),
        ("af3.bin].0", "af3.bin].1"),
    ],
    ids=["compressed", "split", "upstream-bracketed-split"],
)
def test_alphafold3_readiness_accepts_upstream_parameter_layouts(
    tmp_path: Path, monkeypatch, names: tuple[str, ...]
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    root = paths.weights_dir("alphafold3")
    root.mkdir(parents=True)
    for name in names:
        (root / name).write_bytes(b"parameters")

    assert assets.assets_for("alphafold3").ready()
    assert assets.resolve_weights("alphafold3") == root


def test_alphafold3_readiness_rejects_a_gapped_parameter_layout(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    root = paths.weights_dir("alphafold3")
    root.mkdir(parents=True)
    (root / "af3.0.bin").write_bytes(b"parameters")
    (root / "af3.2.bin").write_bytes(b"parameters")

    assert not assets.assets_for("alphafold3").ready()


def test_fetching_a_non_redistributable_model_explains_itself(
    tmp_path: Path, monkeypatch
) -> None:
    """AlphaFold 3's parameters cannot be fetched, and the error must say so.

    The entry has no downloads and nothing to convert, so `fetch` fell through
    to its post-conversion check and reported "conversion did not produce
    af3.bin" -- describing a step that never runs and omitting the one thing
    the user has to do. It also reached the CLI as an unhandled traceback.
    """
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    with pytest.raises(RuntimeError, match="releases them only on request") as error:
        assets.fetch("alphafold3")
    message = str(error.value)
    assert "Request the parameters from DeepMind" in message
    assert str(tmp_path) in message, "must say where to put the file"


def test_openfold3_registry_uses_the_publishers_public_p1_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    spec = assets.assets_for("openfold3")

    assert not spec.in_default_setup
    assert spec.conversion_sources == ("of3_ft3_v1.pt",)
    assert len(spec.downloads) == 1
    checkpoint = spec.downloads[0]
    assert checkpoint.name == "of3_ft3_v1.pt"
    assert checkpoint.url == (
        "https://openfold.s3.amazonaws.com/"
        "openfold3_params/of3_ft3_v1.pt"
    )
    assert checkpoint.size == 2_288_027_095
    assert checkpoint.sha256 == (
        "aedd8f3eb814e3926c8974ef34c9499d"
        "f224443f173b7e396c97684da6e3eeb6"
    )
    info = model_info("openfold3")
    assert info.weights_fetchable
    assert info.download_bytes == checkpoint.size
    assert "OpenFold3 p1" in spec.notes
    assert "incompatible" in spec.notes


def test_openfold3_missing_weights_point_to_the_managed_fetch_command(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))

    with pytest.raises(FileNotFoundError) as error:
        assets.resolve_weights("openfold3")

    message = str(error.value)
    assert "foldjax weights fetch --model openfold3" in message
    assert "of3_ft3_v1.pt" in message
    assert "Request access" not in message


def test_the_cli_reports_an_unfetchable_model_without_a_traceback(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from foldjax.cli import main

    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    assert main(["weights", "fetch", "--model", "alphafold3"]) == 1
    assert "releases them only on request" in capsys.readouterr().err


def test_a_converted_model_still_fetches_a_file_added_later(
    tmp_path: Path, monkeypatch
) -> None:
    """A registry that gains a file must reach machines that already fetched.

    `fetch` returned early on `ready()` alone, so when the shared template
    metadata was added, every existing installation reported success and never
    downloaded it. Converting again is still skipped because rebuilding what is
    already on disk costs minutes.
    """
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    spec = assets.REGISTRY["opendde"]
    weights = paths.weights_dir("opendde")
    weights.mkdir(parents=True)
    (weights / "opendde.jax").write_bytes(b"weights")
    assert spec.ready()
    assets._write_native_manifest(spec)

    payload = b"published later"
    added = assets.Download(
        name="added.json", url="unused://", size=len(payload), shared=True
    )
    added.target("opendde").parent.mkdir(parents=True, exist_ok=True)
    added.target("opendde").write_bytes(b"x")
    patched = dataclasses.replace(
        spec,
        downloads=(*spec.downloads, added),
        convert=lambda *a, **k: pytest.fail(
            "must not re-convert an existing installation"
        ),
    )
    monkeypatch.setattr(assets, "REGISTRY", {**assets.REGISTRY, "opendde": patched})
    fetched = []

    def fake_download(item, model, *, on_progress=None):
        fetched.append(item.name)
        target = item.target(model)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return target

    monkeypatch.setattr(assets, "download", fake_download)

    result = assets.fetch("opendde")
    assert result == weights / "opendde.jax"
    assert "added.json" in fetched
    assert (paths.assets_dir() / "added.json").read_bytes() == payload
    assert (weights / assets._NATIVE_MANIFEST).is_file()


def test_a_truncated_download_is_not_accepted(tmp_path: Path, monkeypatch) -> None:
    """A short read is a truncation, not a finished download.

    Even a server-provided response length must be honoured: without it a
    half-written payload can be renamed into place as if the request finished.
    """
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    item = assets.Download(name="w.bin", url="https://example.invalid/w.bin")

    class _Short:
        headers = {"Content-Length": "100"}

        def read(self, _size):
            payload, self._sent = (
                (b"x" * 40, True) if not getattr(self, "_sent", False) else (b"", True)
            )
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda *a, **k: _Short())
    with pytest.raises(OSError, match="arrived incomplete"):
        assets.download(item, "protenix")
    # Nothing partial is left behind to be mistaken for the real file.
    assert not list(tmp_path.rglob("*.part"))
    assert not list(tmp_path.rglob("w.bin"))


def test_a_complete_download_is_kept(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    item = assets.Download(name="w.bin", url="https://example.invalid/w.bin")

    class _Whole:
        headers = {"Content-Length": "40"}

        def read(self, _size):
            if getattr(self, "_sent", False):
                return b""
            self._sent = True
            return b"x" * 40

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda *a, **k: _Whole())
    written = assets.download(item, "protenix")
    assert written.read_bytes() == b"x" * 40


def test_a_dropped_connection_is_retried(tmp_path: Path, monkeypatch) -> None:
    """Several GB of progress should not be lost to one transient failure."""
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    item = assets.Download(name="w.bin", url="https://example.invalid/w.bin")
    attempts = []

    class _Flaky:
        def __init__(self, attempt):
            self.attempt = attempt
            self.headers = {"Content-Length": "40"}
            self._sent = False

        def read(self, _size):
            if self.attempt == 0:
                raise OSError("connection reset")
            if self._sent:
                return b""
            self._sent = True
            return b"x" * 40

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(*_a, **_k):
        attempts.append(1)
        return _Flaky(len(attempts) - 1)

    monkeypatch.setattr(assets.urllib.request, "urlopen", urlopen)
    assert assets.download(item, "protenix").read_bytes() == b"x" * 40
    assert len(attempts) == 2


def test_http_incomplete_read_is_retried(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    payload = b"whole payload"
    item = assets.Download(
        name="w.bin", url="https://example.invalid/w.bin", size=len(payload)
    )
    attempts = 0

    class _Response:
        status = 200
        headers = {"Content-Length": str(len(payload))}

        def __init__(self, fail: bool):
            self.fail = fail
            self.sent = False

        def read(self, _size):
            if self.fail:
                self.fail = False
                raise http.client.IncompleteRead(b"partial", len(payload))
            if self.sent:
                return b""
            self.sent = True
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        return _Response(attempts == 1)

    monkeypatch.setattr(assets.urllib.request, "urlopen", urlopen)
    assert assets.download(item, "protenix").read_bytes() == payload
    assert attempts == 2


def test_a_dropped_download_resumes_from_the_saved_prefix(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    payload = b"abcdefgh"
    item = assets.Download(
        name="w.bin", url="https://example.invalid/w.bin", size=len(payload)
    )
    requests = []

    class _Response:
        def __init__(
            self,
            body: bytes,
            *,
            status: int,
            total: int,
            content_range: str | None = None,
        ):
            self.body = body
            self.status = status
            self.headers = {"Content-Length": str(total)}
            if content_range is not None:
                self.headers["Content-Range"] = content_range

        def read(self, _size):
            body, self.body = self.body, b""
            return body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(request, **_kwargs):
        requests.append(request.get_header("Range"))
        if len(requests) == 1:
            return _Response(payload[:4], status=200, total=len(payload))
        return _Response(
            payload[4:], status=206, total=4, content_range="bytes 4-7/8"
        )

    monkeypatch.setattr(assets.urllib.request, "urlopen", urlopen)

    assert assets.download(item, "protenix").read_bytes() == payload
    assert requests == [None, "bytes=4-"]


def test_resume_rejects_a_206_for_the_wrong_offset(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    payload = b"abcdefgh"
    item = assets.Download(
        name="w.bin",
        url="https://example.invalid/w.bin",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )
    partial = item.target("protenix").with_suffix(".bin.part")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(payload[:4])

    class _WrongRange:
        status = 206
        headers = {"Content-Length": "4", "Content-Range": "bytes 2-5/8"}

        def read(self, _size):
            return b"efgh"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    calls = 0

    def urlopen(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _WrongRange()
        raise ValueError("invalid Content-Range")

    monkeypatch.setattr(assets.urllib.request, "urlopen", urlopen)

    with pytest.raises(ValueError, match="invalid Content-Range"):
        assets.download(item, "protenix")
    assert not partial.exists()


def test_unpinned_crash_left_partial_is_not_promoted(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    item = assets.Download(name="w.bin", url="https://example.invalid/w.bin")
    partial = item.target("protenix").with_suffix(".bin.part")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"crash-prefix")
    seen_ranges: list[str | None] = []

    class _Whole:
        status = 200
        headers = {"Content-Length": "5"}

        def __init__(self):
            self.body = b"whole"

        def read(self, _size):
            body, self.body = self.body, b""
            return body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(request, **_kwargs):
        seen_ranges.append(request.get_header("Range"))
        return _Whole()

    monkeypatch.setattr(assets.urllib.request, "urlopen", urlopen)

    assert assets.download(item, "protenix").read_bytes() == b"whole"
    assert seen_ranges == [None]


def test_fetch_reports_conversion_and_cached_lifecycle_events(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    payload = b"published checkpoint"
    item = assets.Download(
        name="opendde.pt",
        url="https://example.invalid/opendde.pt",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )

    def convert(model: str, _source: Path) -> Path:
        target = paths.weights_dir(model) / "opendde.jax"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"native")
        return target

    spec = dataclasses.replace(
        assets.REGISTRY["opendde"],
        downloads=(item,),
        convert=convert,
        conversion_sources=(item.name,),
        conversion_schema="event-test-v1",
    )
    monkeypatch.setattr(assets, "REGISTRY", {**assets.REGISTRY, "opendde": spec})

    def fake_download(download, model, *, on_progress=None):
        target = download.target(model)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        if on_progress is not None:
            on_progress(download.name, len(payload), len(payload))
        return target

    monkeypatch.setattr(assets, "download", fake_download)
    events: list[assets.AssetEvent] = []

    assert assets.fetch("opendde", on_event=events.append) == spec.native_path()

    assert [(event.action, event.status) for event in events] == [
        ("resolve", "start"),
        ("download", "start"),
        ("download", "done"),
        ("convert", "waiting"),
        ("convert", "start"),
        ("convert", "done"),
        ("validate", "start"),
        ("validate", "done"),
        ("ready", "done"),
    ]
    assert events[-1].summary()["path"] == str(spec.native_path())

    cached: list[assets.AssetEvent] = []
    assert assets.fetch("opendde", on_event=cached.append) == spec.native_path()
    assert [(event.action, event.status) for event in cached] == [
        ("resolve", "start"),
        ("ready", "skip"),
    ]

    download_only: list[assets.AssetEvent] = []
    assert assets.fetch(
        "opendde", convert=False, on_event=download_only.append
    ) == paths.downloads_dir("opendde")
    assert [(event.action, event.status) for event in download_only] == [
        ("resolve", "start"),
        ("download", "skip"),
        ("ready", "done"),
    ]
    assert "already present" in download_only[1].message


@pytest.mark.parametrize(
    "profile",
    [assets.PROTENIX_MINI_ESM_PROFILE, assets.PROTENIX_MINI_ISM_PROFILE],
)
def test_protenix_variant_events_use_the_public_model_name(profile: str) -> None:
    spec = assets.assets_for("protenix", profile=profile)
    events: list[assets.AssetEvent] = []

    assets._asset_event(
        events.append,
        spec,
        profile,
        "resolve",
        "start",
        "checking profile",
    )

    assert events[0].model == "protenix"
    assert events[0].profile == profile


def test_full_size_hash_invalid_partial_restarts_from_zero(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    payload = b"published"
    item = assets.Download(
        name="w.bin",
        url="https://example.invalid/w.bin",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )
    partial = item.target("protenix").with_suffix(".bin.part")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"x" * len(payload))
    seen_ranges: list[str | None] = []

    class _Whole:
        status = 200
        headers = {"Content-Length": str(len(payload))}

        def __init__(self):
            self.body = payload

        def read(self, _size):
            body, self.body = self.body, b""
            return body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(request, **_kwargs):
        seen_ranges.append(request.get_header("Range"))
        return _Whole()

    monkeypatch.setattr(assets.urllib.request, "urlopen", urlopen)

    assert assets.download(item, "protenix").read_bytes() == payload
    assert seen_ranges == [None]
