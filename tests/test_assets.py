"""The weight store: fetch, verify, convert, resolve.

The network and the multi-GB conversions are not exercised here — the download
loop is driven against a local file:// URL so verification, resume, and failure
behaviour are all real, and the registry itself is checked for the mistakes that
would only surface after someone waited on a 2.6 GB download.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from foldjax import assets, paths


def test_home_follows_its_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "explicit"))
    assert paths.foldjax_home() == tmp_path / "explicit"

    monkeypatch.delenv("FOLDJAX_HOME")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert paths.foldjax_home() == tmp_path / "xdg" / "foldjax"

    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    assert paths.foldjax_home() == tmp_path / "home" / ".cache" / "foldjax"


def test_every_registered_model_is_a_real_backend() -> None:
    for name in assets.available():
        assert name in foldjax_models(), name


def foldjax_models() -> tuple[str, ...]:
    from foldjax.registry import available_models

    return available_models()


def test_registry_declares_what_each_model_needs() -> None:
    for name in assets.available():
        spec = assets.REGISTRY[name]
        # Two models have nothing to download, for two different reasons, and
        # both still need a registry entry so the store knows where the files
        # belong. AlphaFold 3's parameters go only to applicants who accept
        # DeepMind's terms. OpenFold3's are Apache-2.0 and redistributable, but
        # the HuggingFace repository is access-gated, so an unauthenticated
        # fetch would 401 -- which reads like a broken URL rather than an
        # access request nobody has made.
        if name not in ("alphafold3", "openfold3"):
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
    (root / "af3.bin").touch()

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
    (root / "boltz2_conf.safetensors").touch()  # mols/ still missing

    assert not assets.assets_for("boltz2").ready()
    with pytest.raises(FileNotFoundError, match="foldjax weights fetch"):
        assets.resolve_weights("boltz2")


def test_boltz_is_not_ready_without_its_molecule_directory(
    tmp_path: Path, monkeypatch
) -> None:
    """Boltz reads CCD pickles from a directory; weights alone cannot predict."""
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    root = paths.weights_dir("boltz2")
    root.mkdir(parents=True)
    (root / "boltz2_conf.safetensors").touch()

    assert not assets.assets_for("boltz2").ready()
    (root / "mols").mkdir()
    assert assets.assets_for("boltz2").ready()


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


def test_fetching_a_gated_model_does_not_call_it_non_redistributable(
    tmp_path: Path, monkeypatch
) -> None:
    """The same code path serves two models whose reasons are opposite.

    `fetch` used to assert "parameters are not redistributable" for anything
    with no downloads. That is true of AlphaFold 3 and false of OpenFold3, whose
    code *and* weights are Apache-2.0 with published training data -- what stops
    an automatic download there is a gated repository, not a licence. The shared
    sentence now says only that the publisher releases them on request, and the
    model's own `notes` carry the reason.
    """
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    with pytest.raises(RuntimeError) as error:
        assets.fetch("openfold3")
    message = str(error.value)
    assert "not redistributable" not in message
    assert "huggingface.co/OpenFold/OpenFold3" in message
    assert "of3_ft3_v1.pt" in message


def test_the_cli_reports_an_unfetchable_model_without_a_traceback(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from foldjax.cli import main

    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    assert main(["weights", "fetch", "--model", "alphafold3"]) == 1
    assert "releases them only on request" in capsys.readouterr().err


def test_a_truncated_download_is_not_accepted(tmp_path: Path, monkeypatch) -> None:
    """A short read is a truncation, not a finished download.

    Most published files here carry no sha256, so without a length check a
    half-written multi-GB checkpoint gets renamed into place and passes every
    later "is it there?" test permanently.
    """
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    item = assets.Download(name="w.bin", url="https://example.invalid/w.bin")

    class _Short:
        headers = {"Content-Length": "100"}

        def read(self, _size):
            payload, self._sent = (b"x" * 40, True) if not getattr(
                self, "_sent", False
            ) else (b"", True)
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
