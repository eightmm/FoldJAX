"""Fetch, verify, stage, and resolve model assets.

Upstreams publish several different formats. FoldJAX converts the torch
archives used by Boltz-2, OpenDDE, and Protenix, stages ESMFold2's native
safetensors bundle, and locates manually supplied AlphaFold 3/OpenFold3
parameters. Managed checkpoint loading remains PyTorch-free.

Nothing here runs during prediction, and no file is redistributed — each is
downloaded from its own publisher under that project's terms.
"""

from __future__ import annotations

import dataclasses
import hashlib
import http.client
import json
import math
import os
import re
import shutil
import tempfile
import time
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from foldjax.paths import assets_dir, downloads_dir, weights_dir

_CHUNK = 1 << 20
#: Seconds a socket may stall before the attempt is abandoned. Generous, since
#: a large file over a slow link is not the same thing as a stalled one.
_TIMEOUT = 120
#: A dropped connection part-way through several GB should not mean starting
#: the whole fetch again by hand.
_ATTEMPTS = 3
_CONTENT_RANGE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", re.IGNORECASE)

RELEASED_PROFILE = "released"
STRUCTURE_ONLY_PROFILE = "structure-only"
PROTENIX_MINI_ESM_PROFILE = "mini-esm-v0.5.0"
PROTENIX_MINI_ISM_PROFILE = "mini-ism-v0.5.0"

# Publisher SHA-256 identities for both Protenix mini profiles and their exact
# ESM/ISM encoders. Conversion manifests bind native outputs to these values.
_PROTENIX_MINI_ESM_SHA256: str | None = (
    "1301bba9ad322518eace60fd244ded7904439f55317e90d402cc7a0c06026664"
)
_PROTENIX_ESM2_3B_SHA256: str | None = (
    "7de8b4082ba15891959ab368b77ce3886697af1efb16d3c9e9e7b0c5d3f07500"
)
_PROTENIX_MINI_ISM_SHA256 = (
    "a90d3040cdb2c84430878ea724ad817698ee68cc8b12f802a91dac3b081521a2"
)
_PROTENIX_ESM2_3B_ISM_SHA256 = (
    "6dfa8d55fff3d25662b7bf4cbe536aa9cb6c9dd09c4d6ea8a4e8a11c8fb8779a"
)


@dataclass(frozen=True, slots=True)
class Download:
    """One published file."""

    name: str
    url: str
    sha256: str | None = None
    size: int | None = None
    # Shared chemistry data lands in assets/ rather than downloads/<model>/,
    # because several models read the same file.
    shared: bool = False

    def target(self, model: str) -> Path:
        return (assets_dir() if self.shared else downloads_dir(model)) / self.name


@dataclass(frozen=True, slots=True)
class AssetEvent:
    """One structured weight-store lifecycle event.

    Library callers receive events instead of FoldJAX printing behind their
    backs.  The CLI renders them on stderr while preserving stdout for paths and
    JSON.  Byte-level download progress remains the separate ``on_progress``
    callback so existing integrations do not need to change.
    """

    model: str
    profile: str
    action: str
    status: str
    message: str
    item: str | None = None
    path: Path | None = None
    bytes: int | None = None
    elapsed_seconds: float | None = None
    phase: str = "weights"

    def summary(self) -> dict[str, object]:
        """Return a JSON-friendly representation for structured loggers."""

        return {
            "phase": self.phase,
            "model": self.model,
            "profile": self.profile,
            "action": self.action,
            "status": self.status,
            "message": self.message,
            "item": self.item,
            "path": None if self.path is None else str(self.path),
            "bytes": self.bytes,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True, slots=True)
class ModelAssets:
    """Everything one backend needs before it can predict."""

    model: str
    source: str
    licence: str
    downloads: tuple[Download, ...]
    # What --weights should point at, relative to weights/<model>/. "." means
    # the model takes an asset directory rather than a single file.
    native: str
    # Every relative path that must exist before the model can predict. A
    # single-file model repeats `native`; AlphaFold 3 takes a directory.
    requires: tuple[str, ...]
    convert: Callable[[str, Path], Path]
    # FoldJAX's checkpoint reader converts the released torch archives without
    # importing PyTorch. Keep this flag for a future converter that genuinely
    # needs an optional runtime, but make the torch-free path the default.
    needs_torch: bool = False
    notes: str = ""
    #: Whether `foldjax setup` fetches this without being asked. False for a
    #: model whose runtime is not in the default install: downloading gigabytes
    #: for a backend the environment cannot run is not a service to anyone.
    #: `foldjax weights fetch --model <name>` still gets it.
    in_default_setup: bool = True
    extras: tuple[str, ...] = field(default_factory=tuple)
    directory_requires: tuple[tuple[str, tuple[str, ...]], ...] = ()
    ready_check: Callable[[Path], bool] | None = None
    # Conversion outputs are bound to these exact published inputs and this
    # deliberately bumped schema. A converter change without a schema bump is
    # a review error; a registry checkpoint change invalidates old output
    # automatically.
    conversion_sources: tuple[str, ...] = ()
    conversion_schema: str | None = None
    # Published files copied/hard-linked beside ``native`` for runtime use.
    # Their target stat and source identity join the native conversion
    # manifest, so a truncated or replaced language-model checkpoint cannot
    # leave a profile reporting ready.
    staged_sources: tuple[tuple[str, str], ...] = ()
    # Legacy base conversions predate manifests and stay prediction-compatible.
    # New multi-file profiles have no such legacy form and require provenance.
    requires_manifest: bool = False

    def native_path(self) -> Path:
        root = weights_dir(self.model)
        return root if self.native == "." else root / self.native

    def ready(self) -> bool:
        root = weights_dir(self.model)
        if self.ready_check is not None:
            return self.ready_check(root)
        directory_requires = dict(self.directory_requires)
        for item in self.requires:
            path = root / item
            try:
                if self.requires_manifest and path.is_symlink():
                    return False
                if path.is_file():
                    if path.stat().st_size == 0:
                        return False
                elif path.is_dir():
                    required = directory_requires.get(item, ())
                    if required and any(
                        not (path / name).is_file()
                        or (path / name).stat().st_size == 0
                        for name in required
                    ):
                        return False
                    if not required and next(path.iterdir(), None) is None:
                        return False
                else:
                    return False
            except OSError:
                return False
        # An unmarked result predates conversion manifests and remains usable
        # for prediction compatibility. The next explicit fetch rebuilds it
        # from verified sources; provenance is never guessed retroactively.
        marker = root / _NATIVE_MANIFEST
        if self.conversion_sources:
            if marker.exists():
                return _native_manifest_complete(self)
            if self.requires_manifest:
                return False
        return True


# --------------------------------------------------------------------------
# Conversions. Each takes the model name and the directory the downloads landed
# in, and returns the converted artifact prediction will load.
# --------------------------------------------------------------------------


def _convert_protenix(model: str, source: Path) -> Path:
    from foldjax.models.protenix.bridge.torch_mapping import load_torch_checkpoint
    from foldjax.models.protenix.bridge.weights_io import save_native_weights

    out = weights_dir(model) / "protenix_base_default_v1.0.0.jax"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Written uncompressed: these are float32 weights, so gzip saves about 6%
    # of disk but costs 6x on every load (8.5 s vs 1.3 s for OpenDDE's 2.6 GB).
    # Stage beside the final file so a crash never exposes a partial archive.
    with tempfile.TemporaryDirectory(
        prefix=".foldjax-protenix-native-", dir=out.parent
    ) as scratch:
        staged = Path(scratch) / out.name
        save_native_weights(
            staged, load_torch_checkpoint(source / "protenix.pt"), compress=False
        )
        if not _nonempty_file(staged):
            raise RuntimeError("Protenix conversion produced empty native weights")
        os.replace(staged, out)
    return out


def _convert_protenix_variant(
    model: str,
    source: Path,
    *,
    sources_already_verified: bool = False,
    spec: ModelAssets | None = None,
) -> Path:
    """Convert one mini checkpoint and atomically stage its ESM/ISM encoder."""
    from foldjax.models.protenix.bridge.torch_mapping import load_torch_checkpoint
    from foldjax.models.protenix.bridge.weights_io import save_native_weights

    spec = spec or _protenix_variant_assets_for_internal_model(model)
    if spec.model != model or len(spec.staged_sources) != 1:
        raise RuntimeError(f"invalid Protenix variant asset contract for {model}")
    mini_name = spec.conversion_sources[0]
    embedding_source, embedding_target = spec.staged_sources[0]
    if Path(embedding_target).name != embedding_target:
        raise RuntimeError(
            f"invalid Protenix variant staging target: {embedding_target!r}"
        )
    published = {item.name: item for item in spec.downloads}
    try:
        mini_item = published[mini_name]
        embedding_item = published[embedding_source]
    except KeyError as error:
        raise RuntimeError(
            f"{model} variant source is not registered: {error.args[0]}"
        ) from error

    for item in (mini_item, embedding_item):
        origin = item.target(model)
        if not sources_already_verified and not _verified(origin, item):
            raise RuntimeError(f"Protenix variant source asset is incomplete: {origin}")

    root = weights_dir(model)
    root.mkdir(parents=True, exist_ok=True)
    native = root / spec.native
    embedding = root / embedding_target
    with tempfile.TemporaryDirectory(
        prefix=".foldjax-protenix-variant-", dir=root
    ) as scratch:
        stage = Path(scratch)
        staged_native = stage / spec.native
        try:
            params = load_torch_checkpoint(mini_item.target(model))
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"Protenix mini checkpoint conversion failed for {mini_name}: "
                f"{error}"
            ) from error
        save_native_weights(staged_native, params, compress=False)
        if not _nonempty_file(staged_native):
            raise RuntimeError("Protenix mini conversion produced empty native weights")

        staged_embedding = stage / Path(embedding_target).name
        origin = embedding_item.target(model)
        try:
            os.link(origin, staged_embedding)
        except OSError:
            shutil.copy2(origin, staged_embedding)
        if not _same_file_contents(staged_embedding, origin, embedding_item):
            raise RuntimeError(
                f"Protenix ESM/ISM checkpoint staging was incomplete: {origin}"
            )

        embedding.parent.mkdir(parents=True, exist_ok=True)
        # Publish the auxiliary checkpoint first and the model last. A killed
        # publication is never ready because the manifest is written only after
        # both replacements and binds the stats of both files.
        os.replace(staged_embedding, embedding)
        os.replace(staged_native, native)
    return native


def _convert_opendde(model: str, source: Path) -> Path:
    from foldjax.models.opendde.bridge.export_weights import (
        load_torch_checkpoint,
        save_native_weights,
    )

    out = weights_dir(model) / "opendde.jax"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Written uncompressed and published atomically for the same reason as the
    # Protenix archive above.
    with tempfile.TemporaryDirectory(
        prefix=".foldjax-opendde-native-", dir=out.parent
    ) as scratch:
        staged = Path(scratch) / out.name
        save_native_weights(
            staged, load_torch_checkpoint(source / "opendde.pt"), compress=False
        )
        if not _nonempty_file(staged):
            raise RuntimeError("OpenDDE conversion produced empty native weights")
        os.replace(staged, out)
    return out


_BOLTZ_CANONICAL_MOLECULES = (
    "ALA.pkl",
    "ARG.pkl",
    "ASN.pkl",
    "ASP.pkl",
    "CYS.pkl",
    "GLN.pkl",
    "GLU.pkl",
    "GLY.pkl",
    "HIS.pkl",
    "ILE.pkl",
    "LEU.pkl",
    "LYS.pkl",
    "MET.pkl",
    "PHE.pkl",
    "PRO.pkl",
    "SER.pkl",
    "THR.pkl",
    "TRP.pkl",
    "TYR.pkl",
    "VAL.pkl",
    "UNK.pkl",
    "A.pkl",
    "G.pkl",
    "C.pkl",
    "U.pkl",
    "N.pkl",
    "DA.pkl",
    "DG.pkl",
    "DC.pkl",
    "DT.pkl",
    "DN.pkl",
)
_BOLTZ_MOLECULE_COUNT = 45_227
_BOLTZ_MOLECULE_ARCHIVE_SHA256 = (
    "39e076d96dbec6b4e86982bbda16f3a53a2a60c9bdc17828d88f6f9a0c7d1fd7"
)
_BOLTZ_CONVERSION_MARKER = ".foldjax-conversion-in-progress"
_BOLTZ_NATIVE_MARKER = ".foldjax-native.json"
_BOLTZ_MOLECULE_INDEX = ".foldjax-index.json"
_ESMFOLD2_MARKER = ".foldjax-assets.json"
_NATIVE_MANIFEST = ".foldjax-conversion.json"
_SAFETENSORS_MAX_HEADER = 100_000_000


def _boltz_molecule_archive_sha256() -> str:
    """Read molecule identity from the registry, including monkeypatched updates."""
    try:
        return next(
            item.sha256
            for item in REGISTRY["boltz2"].downloads
            if item.name == "mols.tar" and item.sha256 is not None
        )
    except (KeyError, StopIteration):
        return _BOLTZ_MOLECULE_ARCHIVE_SHA256


def _boltz_molecule_marker() -> str:
    return (
        "foldjax-boltz-molecules-v2 "
        f"archive={_boltz_molecule_archive_sha256()} "
        f"count={_BOLTZ_MOLECULE_COUNT}\n"
    )


def _boltz_molecule_count(destination: Path) -> int:
    try:
        return sum(
            1
            for path in destination.iterdir()
            if path.suffix == ".pkl" and _nonempty_file(path)
        )
    except OSError:
        return 0


def _boltz_molecule_inventory(destination: Path) -> dict[str, list[int]]:
    """Return cheap per-file identity for the extracted CCD archive.

    A containing directory's mtime changes for create/delete/rename, but not
    when an existing molecule is truncated in place. Recording each file's
    size and mtime catches both without re-reading 1.86 GB of pickle payloads.
    """
    inventory: dict[str, list[int]] = {}
    for path in destination.iterdir():
        if path.suffix != ".pkl" or path.is_symlink() or not path.is_file():
            continue
        stat = path.stat()
        inventory[path.name] = [stat.st_size, stat.st_mtime_ns]
    return inventory


def _boltz_molecules_complete(destination: Path) -> bool:
    """Whether the pinned Boltz molecule archive is present in full.

    The canonical 31 residues appear before the end of the 45k-file archive,
    so checking only those can label an interrupted extraction as complete.
    The exact count also accepts legacy complete stores made before FoldJAX
    wrote a completion marker.
    """
    marker = destination / ".foldjax-complete"
    if marker.exists() and not _boltz_marker_complete(destination):
        return False
    if not all(
        _nonempty_file(destination / name)
        for name in _BOLTZ_CANONICAL_MOLECULES
    ):
        return False
    index_state = _boltz_molecule_index_state(destination)
    if index_state is not None:
        return index_state
    # Count even when the marker is current. The marker proves provenance, but
    # the directory is user-writable and losing one of the other 45k CCD files
    # must make a later fetch repair it. The same count deliberately accepts a
    # complete pre-marker store without forcing a 1.86 GB re-extraction.
    return _boltz_molecule_count(destination) == _BOLTZ_MOLECULE_COUNT


def _boltz_molecule_index_state(destination: Path) -> bool | None:
    """Validate a current inventory, or return ``None`` for a legacy store."""
    index = destination / _BOLTZ_MOLECULE_INDEX
    if not index.exists():
        return None
    try:
        payload = json.loads(index.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return False
        if payload.get("schema") == 1:
            return None
        if payload.get("schema") != 2:
            return False
        return payload == {
            "schema": 2,
            "archive_sha256": _boltz_molecule_archive_sha256(),
            "count": _BOLTZ_MOLECULE_COUNT,
            "files": _boltz_molecule_inventory(destination),
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _boltz_molecule_index_complete(destination: Path) -> bool:
    return _boltz_molecule_index_state(destination) is True


def _write_boltz_molecule_completion(destination: Path) -> None:
    """Bind a complete extraction to every molecule's cheap file identity."""
    inventory = _boltz_molecule_inventory(destination)
    if (
        len(inventory) != _BOLTZ_MOLECULE_COUNT
        or any(
            inventory.get(name, [0])[0] <= 0
            for name in _BOLTZ_CANONICAL_MOLECULES
        )
        or any(value[0] <= 0 for value in inventory.values())
    ):
        raise RuntimeError("cannot mark an incomplete Boltz molecule archive")
    _write_text_atomic(
        destination / ".foldjax-complete", _boltz_molecule_marker()
    )
    payload = {
        "schema": 2,
        "archive_sha256": _boltz_molecule_archive_sha256(),
        "count": _BOLTZ_MOLECULE_COUNT,
        "files": inventory,
    }
    _write_text_atomic(
        destination / _BOLTZ_MOLECULE_INDEX,
        json.dumps(payload, sort_keys=True) + "\n",
    )


def _boltz_marker_complete(destination: Path) -> bool:
    try:
        return (
            destination.joinpath(".foldjax-complete").read_text(encoding="utf-8")
            == _boltz_molecule_marker()
        )
    except OSError:
        return False


def _write_text_atomic(path: Path, contents: str) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            handle.write(contents)
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _convert_boltz2(model: str, source: Path) -> Path:
    from foldjax.models.boltz2.bridge.export_weights import (
        export_affinity,
        export_confidence,
    )

    out_dir = weights_dir(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    in_progress = out_dir / _BOLTZ_CONVERSION_MARKER
    _write_text_atomic(in_progress, "conversion in progress\n")
    complete = False
    try:
        if not _boltz_native_provenance_complete(out_dir):
            # Both native bundles are one readiness unit. Build them away from
            # the public weight root and replace each complete file while the
            # in-progress marker keeps concurrent predictors from accepting a
            # half-published mixture.
            with tempfile.TemporaryDirectory(
                prefix=".foldjax-boltz-native-", dir=out_dir.parent
            ) as scratch:
                staged = Path(scratch)
                export_confidence(source / "boltz2_conf.ckpt", staged)
                affinity = export_affinity(source / "boltz2_aff.ckpt", staged)
                if affinity is None or not all(
                    _nonempty_file(staged / name) for name in _BOLTZ_NATIVE_FILES
                ):
                    raise RuntimeError(
                        "Boltz conversion did not produce complete confidence and "
                        "affinity bundles"
                    )
                for name in _BOLTZ_NATIVE_FILES:
                    os.replace(staged / name, out_dir / name)
        _write_boltz_native_manifest(out_dir)
        _unpack_boltz_mols(
            source / "mols.tar",
            out_dir / "mols",
            force=not _boltz_molecule_index_complete(out_dir / "mols"),
        )
        if not _boltz_native_complete(
            out_dir
        ) or not _boltz_molecules_complete(out_dir / "mols"):
            raise RuntimeError("Boltz conversion left an incomplete native bundle")
        complete = True
        return out_dir / "boltz2_conf.safetensors"
    finally:
        if complete:
            in_progress.unlink(missing_ok=True)


def _unpack_boltz_mols(
    archive: Path, destination: Path, *, force: bool = False
) -> None:
    """Boltz needs its CCD molecule directory beside the weights."""
    import tarfile

    if not force and _boltz_molecules_complete(destination):
        if not _boltz_molecule_index_complete(destination):
            _write_boltz_molecule_completion(destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".foldjax-mols-", dir=destination.parent
    ) as scratch:
        unpacked = Path(scratch) / "unpacked"
        unpacked.mkdir()
        with tarfile.open(archive) as tar:
            # Modern Python warns without an explicit policy; refuse anything
            # that would escape the staging directory.
            tar.extractall(unpacked, filter="data")
        entries = [path for path in unpacked.iterdir() if path.name != "__MACOSX"]
        staged = (
            entries[0] if len(entries) == 1 and entries[0].is_dir() else unpacked
        )
        if (
            _boltz_molecule_count(staged) != _BOLTZ_MOLECULE_COUNT
            or not all(
                _nonempty_file(staged / name)
                for name in _BOLTZ_CANONICAL_MOLECULES
            )
        ):
            raise RuntimeError(
                "Boltz molecule archive did not contain the complete published set"
            )
        _write_boltz_molecule_completion(staged)
        replaced = Path(scratch) / "replaced"
        if destination.exists() or destination.is_symlink():
            os.replace(destination, replaced)
        try:
            os.replace(staged, destination)
        except BaseException:
            if replaced.exists() or replaced.is_symlink():
                os.replace(replaced, destination)
            raise


# Shared Protenix-family chemistry assets. Protenix and OpenDDE both read them,
# so they live in assets/ rather than under one model.
_CCD_COMPONENTS = Download(
    name="components.cif",
    url="https://protenix.tos-cn-beijing.volces.com/common/components.cif",
    sha256="bb31ae5cf6c8bc669924313077cb4231ee5ffefd3a20118cd14f3ec89f8bb6a5",
    size=490777362,
    shared=True,
)
_CCD_RDKIT = Download(
    name="components.cif.rdkit_mol.pkl",
    url="https://protenix.tos-cn-beijing.volces.com/common/components.cif.rdkit_mol.pkl",
    sha256="d1cfb71f5993a3ebea7c47877022d7f597bbfbaf86e28a4770e957da6c50cd35",
    size=142498117,
    shared=True,
)
# Template metadata. `resolve_template_search_hits` needs both to decide which
# hits a query may see: when each candidate was released, and which obsolete
# entries were superseded by what. Same publisher and terms as the CCD above,
# and shared for the same reason -- Protenix and OpenDDE read one copy.
_TEMPLATE_RELEASE_DATES = Download(
    name="release_date_cache.json",
    url="https://protenix.tos-cn-beijing.volces.com/common/release_date_cache.json",
    sha256="8b1ef12ddc01a0d5eb2d388c77ded91aa906eebce7440726c57b6f8d1a3ec142",
    size=12754898,
    shared=True,
)
_TEMPLATE_OBSOLETE = Download(
    name="obsolete_to_successor.json",
    url="https://protenix.tos-cn-beijing.volces.com/common/obsolete_to_successor.json",
    sha256="2bc08348d0efba438c109bb27be6fa25b611d371c60b8a8da3de387a4a0698ad",
    size=86882,
    shared=True,
)


@dataclass(frozen=True, slots=True)
class _ProtenixVariant:
    profile: str
    model: str
    mini: Download
    embedding: Download
    native: str


_PROTENIX_VARIANTS = {
    PROTENIX_MINI_ESM_PROFILE: _ProtenixVariant(
        profile=PROTENIX_MINI_ESM_PROFILE,
        model="protenix-mini-esm",
        mini=Download(
            name="protenix_mini_esm_v0.5.0.pt",
            url=(
                "https://protenix.tos-cn-beijing.volces.com/checkpoint/"
                "protenix_mini_esm_v0.5.0.pt"
            ),
            sha256=_PROTENIX_MINI_ESM_SHA256,
            size=541_640_990,
        ),
        embedding=Download(
            name="esm2_t36_3B_UR50D.pt",
            url=(
                "https://protenix.tos-cn-beijing.volces.com/checkpoint/"
                "esm2_t36_3B_UR50D.pt"
            ),
            sha256=_PROTENIX_ESM2_3B_SHA256,
            size=5_678_116_398,
        ),
        native="protenix_mini_esm_v0.5.0.jax",
    ),
    PROTENIX_MINI_ISM_PROFILE: _ProtenixVariant(
        profile=PROTENIX_MINI_ISM_PROFILE,
        model="protenix-mini-ism",
        mini=Download(
            name="protenix_mini_ism_v0.5.0.pt",
            url=(
                "https://protenix.tos-cn-beijing.volces.com/checkpoint/"
                "protenix_mini_ism_v0.5.0.pt"
            ),
            sha256=_PROTENIX_MINI_ISM_SHA256,
            size=541_640_990,
        ),
        embedding=Download(
            name="esm2_t36_3B_UR50D_ism.pt",
            url=(
                "https://protenix.tos-cn-beijing.volces.com/checkpoint/"
                "esm2_t36_3B_UR50D_ism.pt"
            ),
            sha256=_PROTENIX_ESM2_3B_ISM_SHA256,
            size=11_356_246_722,
        ),
        native="protenix_mini_ism_v0.5.0.jax",
    ),
}
_PROTENIX_PROFILE_BY_INTERNAL_MODEL = {
    variant.model: profile for profile, variant in _PROTENIX_VARIANTS.items()
}


def _public_model_name(model: str) -> str:
    """Hide profile-specific storage roots from user-facing model labels."""

    return "protenix" if model in _PROTENIX_PROFILE_BY_INTERNAL_MODEL else model


def _stage_esmfold2(
    model: str,
    source: Path,
    *,
    sources_already_verified: bool = False,
    spec: ModelAssets | None = None,
) -> Path:
    """Put the published files where the loader expects a model.

    Nothing is converted: the port names its parameters the way upstream's
    `state_dict` does, so the published safetensors loads as it ships. What the
    loader does need is a *directory* holding `config.json` beside the weights
    -- the released configuration differs from upstream's dataclass defaults in
    most of the fields that matter -- and the download step keeps files under
    `downloads/`. Files are hard-linked where the filesystem allows it, so the
    roughly 1.3 GB structure checkpoint and, for the released profile, the
    additional 25.4 GB language model are not stored twice.

    ESMC-6B, the language model the trunk folds representations from, is a
    separate 25.4 GB checkpoint that upstream distributes apart from this one.
    The released profile publishes it under `<weights>/esmfold2/esmc`; the
    structure-only profile deliberately stages only the two structure files.
    """
    out = weights_dir(model)
    out.mkdir(parents=True, exist_ok=True)
    spec = spec or assets_for(model)
    for item in spec.downloads:
        origin = item.target(model)
        try:
            relative = origin.relative_to(downloads_dir(model))
        except ValueError:
            continue
        target = out / relative
        if not sources_already_verified and not _verified(origin, item):
            raise RuntimeError(f"ESMFold2 source asset is incomplete: {origin}")
        try:
            if (
                not target.is_symlink()
                and _same_file_contents(target, origin, item)
            ):
                continue
        except OSError:
            pass
        target.parent.mkdir(parents=True, exist_ok=True)
        # Publish only after a complete hard-link/copy exists. A killed fallback
        # copy therefore leaves the old target (or no target) intact and the
        # next ``weights fetch`` can repair it.
        with tempfile.TemporaryDirectory(
            prefix=".foldjax-stage-", dir=target.parent
        ) as scratch:
            staged = Path(scratch) / target.name
            try:
                os.link(origin, staged)
            except OSError:
                shutil.copy2(origin, staged)
            if (
                not _same_file_contents(staged, origin, item)
            ):
                raise RuntimeError(f"ESMFold2 asset staging was incomplete: {origin}")
            os.replace(staged, target)
    manifest: dict[str, object] = {"schema": 1, "files": {}}
    files = manifest["files"]
    assert isinstance(files, dict)
    for item in spec.downloads:
        origin = item.target(model)
        try:
            relative = origin.relative_to(downloads_dir(model))
        except ValueError:
            continue
        target = out / relative
        stat = target.stat()
        files[relative.as_posix()] = {
            "sha256": item.sha256,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    _write_text_atomic(
        out / _ESMFOLD2_MARKER,
        json.dumps(manifest, sort_keys=True) + "\n",
    )
    return out


def _bring_your_own(model: str, source: Path) -> Path:
    """No conversion: the published file is already what the model loads."""
    return weights_dir(model)


_AF3_PARAMETER_PATTERNS = (
    re.compile(r"(?P<model>.*)\.(?P<index>\d+)\.bin\.zst$"),
    re.compile(r"(?P<model>.*)\.bin\.zst\.(?P<index>\d+)$"),
    re.compile(r"(?P<model>.*)\.(?P<index>\d+)\.bin$"),
    # Kept for byte-for-byte compatibility with the unusual spelling accepted
    # by upstream's own parameter selector.
    re.compile(r"(?P<model>.*)\.bin\]\.(?P<index>\d+)$"),
)


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _alphafold3_ready(root: Path) -> bool:
    """Mirror upstream's plain, compressed, and split parameter filenames."""
    try:
        files = [path for path in root.iterdir() if path.is_file()]
    except OSError:
        return False
    plain = [
        path
        for path in files
        if _nonempty_file(path)
        and (path.name.endswith(".bin") or path.name.endswith(".bin.zst"))
        and not any(pattern.fullmatch(path.name) for pattern in _AF3_PARAMETER_PATTERNS)
    ]
    if plain:
        return True
    for pattern in _AF3_PARAMETER_PATTERNS:
        groups: dict[str, list[int]] = {}
        for path in files:
            match = pattern.fullmatch(path.name)
            if match is not None and _nonempty_file(path):
                groups.setdefault(match.group("model"), []).append(
                    int(match.group("index"))
                )
        if len(groups) == 1:
            indices = sorted(next(iter(groups.values())))
            if len(indices) >= 2 and indices == list(range(indices[-1] + 1)):
                return True
    return False


_BOLTZ_NATIVE_FILES = (
    "boltz2_conf.safetensors",
    "boltz2_conf.safetensors.json",
    "boltz2_aff.safetensors",
    "boltz2_aff.safetensors.json",
)

# Dtypes that ``safetensors.numpy.load_file`` can materialize in the runtime
# version FoldJAX declares. Byte widths make the cheap header probe reject a
# bundle that advertises a shape its data range cannot actually hold.
_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
    "C64": 8,
}


def _safetensors_entry_valid(entry: dict[str, object]) -> bool:
    """Whether one header entry is loadable and matches its byte range."""
    if set(entry) != {"dtype", "shape", "data_offsets"}:
        return False
    dtype = entry["dtype"]
    shape = entry["shape"]
    offsets = entry["data_offsets"]
    if not isinstance(dtype, str) or dtype not in _SAFETENSORS_DTYPE_BYTES:
        return False
    if not isinstance(shape, list) or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 0
        for size in shape
    ):
        return False
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(
            isinstance(offset, bool) or not isinstance(offset, int)
            for offset in offsets
        )
        or offsets[0] < 0
        or offsets[1] < offsets[0]
    ):
        return False
    elements = 1
    for size in shape:
        elements *= size
    return offsets[1] - offsets[0] == elements * _SAFETENSORS_DTYPE_BYTES[dtype]


def _native_tree_key_valid(key: object) -> bool:
    """Validate the path grammar consumed by Boltz's native pytree loader."""
    if not isinstance(key, str) or not key:
        return False
    for component in key.split("/"):
        tag, separator, value = component.partition(":")
        if not separator or not value or tag not in {"d", "a", "i"}:
            return False
        if tag == "i" and (not value.isdecimal() or value != str(int(value))):
            return False
    return True


def _native_scalar_valid(value: object) -> bool:
    """Match the scalar leaves emitted and accepted by ``native.py``."""
    if value is None or isinstance(value, (bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return isinstance(value, dict) and value == {"__none__": True}


def _safetensors_keys(path: Path) -> set[str] | None:
    """Validate a safetensors envelope and return its tensor keys."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            header_size = int.from_bytes(handle.read(8), "little")
            if (
                header_size <= 0
                or header_size > _SAFETENSORS_MAX_HEADER
                or header_size > size - 8
            ):
                return None
            header = json.loads(handle.read(header_size))
        if not isinstance(header, dict):
            return None
        tensors = {
            key: value for key, value in header.items() if key != "__metadata__"
        }
        metadata = header.get("__metadata__")
        if metadata is not None and (
            not isinstance(metadata, dict)
            or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in metadata.items()
            )
        ):
            return None
        if not tensors or not all(
            isinstance(value, dict) and _safetensors_entry_valid(value)
            for value in tensors.values()
        ):
            return None
        offsets = [value["data_offsets"] for value in tensors.values()]
        data_size = size - 8 - header_size
        ordered = sorted((value[0], value[1]) for value in offsets)
        if ordered[0][0] != 0 or ordered[-1][1] != data_size:
            return None
        if any(
            current[1] != following[0]
            for current, following in zip(ordered, ordered[1:])
        ):
            return None
        return set(tensors)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _valid_safetensors(path: Path) -> bool:
    """Validate the safetensors envelope without reading tensor data."""
    return _safetensors_keys(path) is not None


def _valid_boltz_sidecar(path: Path, tensor_keys: set[str]) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        keys = payload.get("keys") if isinstance(payload, dict) else None
        scalars = payload.get("scalars") if isinstance(payload, dict) else None
        return (
            isinstance(payload, dict)
            and payload.get("backend") == "safetensors"
            and isinstance(keys, list)
            and bool(keys)
            and all(_native_tree_key_valid(key) for key in keys)
            and len(keys) == len(set(keys))
            and set(keys) == tensor_keys
            and isinstance(scalars, dict)
            and all(
                _native_tree_key_valid(key) and _native_scalar_valid(value)
                for key, value in scalars.items()
            )
            and not tensor_keys.intersection(scalars)
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _boltz_native_files_valid(root: Path) -> bool:
    for weights_name in (
        "boltz2_conf.safetensors",
        "boltz2_aff.safetensors",
    ):
        keys = _safetensors_keys(root / weights_name)
        if keys is None or not _valid_boltz_sidecar(
            root / f"{weights_name}.json", keys
        ):
            return False
    return True


def _conversion_identity(spec: ModelAssets) -> dict[str, object]:
    """Return the registry identity a native conversion was produced from."""
    if not spec.conversion_sources or spec.conversion_schema is None:
        raise RuntimeError(f"{spec.model} has no conversion identity")
    published = {item.name: item for item in spec.downloads}
    sources: dict[str, dict[str, object]] = {}
    for name in spec.conversion_sources:
        try:
            item = published[name]
        except KeyError as error:
            raise RuntimeError(
                f"{spec.model} conversion source is not registered: {name}"
            ) from error
        if item.sha256 is None:
            if spec.model not in _PROTENIX_PROFILE_BY_INTERNAL_MODEL:
                raise RuntimeError(
                    f"{spec.model} conversion source has no published SHA-256: {name}"
                )
            # The exact Protenix release sizes are pinned while the four
            # publisher-file hashes are being recorded. Once a SHA constant is
            # filled, this identity changes and invalidates the interim
            # conversion without accepting it as the hash-bound release.
        sources[name] = {"sha256": item.sha256, "size": item.size}
    return {"converter": spec.conversion_schema, "sources": sources}


def _native_manifest(spec: ModelAssets) -> dict[str, object]:
    native = spec.native_path()
    stat = native.stat()
    manifest: dict[str, object] = {
        "schema": 1,
        "conversion": _conversion_identity(spec),
        "native": {
            "path": spec.native,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
    }
    if spec.staged_sources:
        published = {item.name: item for item in spec.downloads}
        staged: dict[str, dict[str, object]] = {}
        for source_name, target_name in spec.staged_sources:
            item = published[source_name]
            target_stat = (weights_dir(spec.model) / target_name).stat()
            staged[target_name] = {
                "source": {
                    "name": source_name,
                    "sha256": item.sha256,
                    "size": item.size,
                },
                "size": target_stat.st_size,
                "mtime_ns": target_stat.st_mtime_ns,
            }
        manifest["staged"] = staged
    return manifest


def _write_native_manifest(spec: ModelAssets) -> None:
    if not _nonempty_file(spec.native_path()):
        raise RuntimeError(f"{spec.model} conversion produced no native weights")
    _write_text_atomic(
        weights_dir(spec.model) / _NATIVE_MANIFEST,
        json.dumps(_native_manifest(spec), sort_keys=True) + "\n",
    )


def _native_manifest_complete(spec: ModelAssets) -> bool:
    try:
        recorded = json.loads(
            (weights_dir(spec.model) / _NATIVE_MANIFEST).read_text(
                encoding="utf-8"
            )
        )
        return recorded == _native_manifest(spec)
    except (OSError, TypeError, ValueError, RuntimeError, json.JSONDecodeError):
        return False


def _boltz_native_manifest(root: Path) -> dict[str, object]:
    return {
        "schema": 2,
        "conversion": _conversion_identity(REGISTRY["boltz2"]),
        "files": {
            name: {
                "size": (root / name).stat().st_size,
                "mtime_ns": (root / name).stat().st_mtime_ns,
            }
            for name in _BOLTZ_NATIVE_FILES
        },
    }


def _boltz_legacy_native_manifest(root: Path) -> dict[str, object]:
    """The output-stat manifest written before source identity was recorded."""
    current = _boltz_native_manifest(root)
    return {"schema": 1, "files": current["files"]}


def _write_boltz_native_manifest(root: Path) -> None:
    if not _boltz_native_files_valid(root):
        raise RuntimeError("Boltz conversion produced an invalid native bundle")
    _write_text_atomic(
        root / _BOLTZ_NATIVE_MARKER,
        json.dumps(_boltz_native_manifest(root), sort_keys=True) + "\n",
    )


def _boltz_native_provenance_complete(root: Path) -> bool:
    if not _boltz_native_files_valid(root):
        return False
    try:
        return json.loads(
            (root / _BOLTZ_NATIVE_MARKER).read_text(encoding="utf-8")
        ) == _boltz_native_manifest(root)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _boltz_native_complete(root: Path) -> bool:
    """Accept validated legacy files; enforce stat identity once marked."""
    if not _boltz_native_files_valid(root):
        return False
    marker = root / _BOLTZ_NATIVE_MARKER
    if not marker.exists():
        return True
    try:
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        expected = (
            _boltz_legacy_native_manifest(root)
            if isinstance(recorded, dict) and recorded.get("schema") == 1
            else _boltz_native_manifest(root)
        )
        return recorded == expected
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _boltz_ready(root: Path) -> bool:
    """Match the files the confidence and affinity loaders actually open."""
    return (
        not (root / _BOLTZ_CONVERSION_MARKER).exists()
        and _boltz_native_complete(root)
        and _boltz_molecules_complete(root / "mols")
    )


def _esmfold2_manifest_ready(
    root: Path, downloads: tuple[Download, ...], *, allow_superset: bool
) -> bool:
    """Validate one ESMFold2 profile against its atomic staging manifest."""
    try:
        manifest = json.loads(
            (root / _ESMFOLD2_MARKER).read_text(encoding="utf-8")
        )
        if not isinstance(manifest, dict):
            return False
        recorded = manifest["files"]
        if manifest.get("schema") != 1 or not isinstance(recorded, dict):
            return False
        expected: set[str] = set()
        for item in downloads:
            origin = item.target("esmfold2")
            try:
                relative = origin.relative_to(downloads_dir("esmfold2"))
            except ValueError:
                continue
            name = relative.as_posix()
            expected.add(name)
            entry = recorded.get(name)
            target = root / relative
            if (
                not isinstance(entry, dict)
                or target.is_symlink()
                or not target.is_file()
            ):
                return False
            stat = target.stat()
            if (
                entry.get("sha256") != item.sha256
                or entry.get("size") != stat.st_size
                or entry.get("mtime_ns") != stat.st_mtime_ns
                or (item.size is not None and stat.st_size != item.size)
            ):
                return False
        recorded_names = set(recorded)
        return (
            expected <= recorded_names
            if allow_superset
            else recorded_names == expected
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _esmfold2_ready(root: Path) -> bool:
    """Validate the complete released bundle used by the default profile."""
    spec = REGISTRY.get("esmfold2")
    return spec is not None and _esmfold2_manifest_ready(
        root, spec.downloads, allow_superset=False
    )


def _esmfold2_structure_only_ready(root: Path) -> bool:
    """The full release also satisfies the smaller structure-only profile."""
    spec = REGISTRY.get("esmfold2")
    if spec is None:
        return False
    downloads = tuple(
        item for item in spec.downloads if not item.name.startswith("esmc/")
    )
    return _esmfold2_manifest_ready(root, downloads, allow_superset=True)


REGISTRY: dict[str, ModelAssets] = {
    "alphafold3": ModelAssets(
        model="alphafold3",
        source="https://github.com/google-deepmind/alphafold3",
        licence="AlphaFold 3 model parameters terms of use (not redistributable)",
        # Empty on purpose. DeepMind releases the parameters only to applicants
        # who accept their terms, so FoldJAX never downloads or redistributes
        # them; it only knows where to look once you have them.
        downloads=(),
        native=".",
        requires=("af3.bin",),
        convert=_bring_your_own,
        ready_check=_alphafold3_ready,
        notes="Request the parameters from DeepMind, then put af3.bin (or the "
        "af3.bin.zst you were sent) in this directory. AlphaFold 3 reads its "
        "own format, so nothing is converted.",
    ),
    "protenix": ModelAssets(
        model="protenix",
        source="https://github.com/bytedance/Protenix",
        licence="Apache-2.0",
        downloads=(
            Download(
                name="protenix.pt",
                url=(
                    "https://protenix.tos-cn-beijing.volces.com/checkpoint/"
                    "protenix_base_default_v1.0.0.pt"
                ),
                sha256=(
                    "2b7d5a8b30494514fc47fd2271a16260528cdba170ba09cc112fdecd8f85ec04"
                ),
                size=1475950125,
            ),
            _CCD_COMPONENTS,
            _CCD_RDKIT,
            _TEMPLATE_RELEASE_DATES,
            _TEMPLATE_OBSOLETE,
        ),
        native="protenix_base_default_v1.0.0.jax",
        requires=("protenix_base_default_v1.0.0.jax",),
        convert=_convert_protenix,
        conversion_sources=("protenix.pt",),
        conversion_schema="protenix-native-v1",
        notes="CCD assets are shared with OpenDDE and only needed for ligands "
        "and modified residues; the two template JSONs likewise, and only when "
        "predicting with templates.",
    ),
    "opendde": ModelAssets(
        model="opendde",
        source="https://huggingface.co/aurekaresearch/OpenDDE",
        licence="see the OpenDDE model card",
        downloads=(
            Download(
                name="opendde.pt",
                url=(
                    "https://huggingface.co/aurekaresearch/OpenDDE/resolve/"
                    "eddd563ce96571f784012edd8f045181c8f8627d/opendde.pt"
                ),
                sha256=(
                    "7b826620390afad877ee2babc6a4d0df81b94d3a0be030959853d6a7da0807cc"
                ),
                size=2625249069,
            ),
            _CCD_COMPONENTS,
            _CCD_RDKIT,
            _TEMPLATE_RELEASE_DATES,
            _TEMPLATE_OBSOLETE,
        ),
        native="opendde.jax",
        requires=("opendde.jax",),
        convert=_convert_opendde,
        conversion_sources=("opendde.pt",),
        conversion_schema="opendde-native-v1",
    ),
    "esmfold2": ModelAssets(
        model="esmfold2",
        source="https://huggingface.co/biohub/ESMFold2",
        licence="MIT for the structure weights; ESMC-6B is MIT plus its own terms",
        downloads=(
            Download(
                name="model.safetensors",
                url=(
                    "https://huggingface.co/biohub/ESMFold2/resolve/"
                    "8fc3ff471022fdce52c77030685eb775de0c00a3/model.safetensors"
                ),
                sha256=(
                    "138fd4350d6892b81ce6be7ff9bf5a93ae9d4d3751f46a27438a3f9f0dcefa0e"
                ),
                size=939505228,
            ),
            Download(
                name="config.json",
                url=(
                    "https://huggingface.co/biohub/ESMFold2/resolve/"
                    "8fc3ff471022fdce52c77030685eb775de0c00a3/config.json"
                ),
                sha256=(
                    "e9ec2496ec433a1dce18627ed4bf3785b4ce0c1d69e4bb4663dad1ab895da012"
                ),
                size=2337,
            ),
            # ESMC-6B, the language model the trunk folds representations from.
            # Upstream distributes it as its own repository and the store
            # keeps it in the `esmc/` subdirectory the loader looks in. It is
            # 25.4 GB and it is not optional -- without it the trunk runs with
            # its language-model branch absent, which is a different model.
            Download(
                name="esmc/config.json",
                url=(
                    "https://huggingface.co/biohub/ESMC-6B/resolve/"
                    "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a/config.json"
                ),
                sha256=(
                    "c5566fab6a17fd674141331fe75de917b7904d99fb7a410d2b1593c21e576913"
                ),
                size=341,
            ),
            Download(
                name="esmc/model.safetensors.index.json",
                url=(
                    "https://huggingface.co/biohub/ESMC-6B/resolve/"
                    "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a/model.safetensors.index.json"
                ),
                sha256=(
                    "6846456e20e6ee2c37461f7bfc21d316d69bdaf165b925691afcb39e583244da"
                ),
                size=97349,
            ),
            Download(
                name="esmc/model-00001-of-00006.safetensors",
                url=(
                    "https://huggingface.co/biohub/ESMC-6B/resolve/"
                    "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a/model-00001-of-00006.safetensors"
                ),
                sha256=(
                    "bd90149ff223e6ac1a0cac6147a5ae0df20d3a21df4f65356a1f19cd14f4aa8a"
                ),
                size=4864457920,
            ),
            Download(
                name="esmc/model-00002-of-00006.safetensors",
                url=(
                    "https://huggingface.co/biohub/ESMC-6B/resolve/"
                    "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a/model-00002-of-00006.safetensors"
                ),
                sha256=(
                    "f75e2144d8269fe2eb4b3e0823fb089b94f176d8024153e85b8fb573a42294fa"
                ),
                size=4971211344,
            ),
            Download(
                name="esmc/model-00003-of-00006.safetensors",
                url=(
                    "https://huggingface.co/biohub/ESMC-6B/resolve/"
                    "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a/model-00003-of-00006.safetensors"
                ),
                sha256=(
                    "f699f01ecc9691d9c6470492765fe54b8b5d2e9f277c139e89427433ffdfe0b2"
                ),
                size=4863752992,
            ),
            Download(
                name="esmc/model-00004-of-00006.safetensors",
                url=(
                    "https://huggingface.co/biohub/ESMC-6B/resolve/"
                    "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a/model-00004-of-00006.safetensors"
                ),
                sha256=(
                    "46add1b7be098bbfdc3073884851ba3057f1b33ea23a158b650a37007dabd13d"
                ),
                size=4971211344,
            ),
            Download(
                name="esmc/model-00005-of-00006.safetensors",
                url=(
                    "https://huggingface.co/biohub/ESMC-6B/resolve/"
                    "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a/model-00005-of-00006.safetensors"
                ),
                sha256=(
                    "1e1cb62f060a34e18f54a31a76683ef888b8cec59e73315f5b31d25d45a1f88c"
                ),
                size=4863752992,
            ),
            Download(
                name="esmc/model-00006-of-00006.safetensors",
                url=(
                    "https://huggingface.co/biohub/ESMC-6B/resolve/"
                    "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a/model-00006-of-00006.safetensors"
                ),
                sha256=(
                    "56c73e13ae96e777ce65eee99364056069ef93b646470f352f83c5f1037b1b18"
                ),
                size=873762296,
            ),
        ),
        native="model.safetensors",
        requires=(
            "model.safetensors",
            "config.json",
            "esmc/config.json",
            "esmc/model.safetensors.index.json",
            "esmc/model-00001-of-00006.safetensors",
            "esmc/model-00002-of-00006.safetensors",
            "esmc/model-00003-of-00006.safetensors",
            "esmc/model-00004-of-00006.safetensors",
            "esmc/model-00005-of-00006.safetensors",
            "esmc/model-00006-of-00006.safetensors",
        ),
        convert=_stage_esmfold2,
        ready_check=_esmfold2_ready,
        in_default_setup=False,
        notes="Loaded as published -- the JAX port names its parameters the "
        "way upstream's state_dict does, so nothing is converted. Two "
        "checkpoints: the 940 MB structure network, and ESMC-6B, the 6B "
        "language model it folds representations from, which upstream "
        "distributes as its own repository and which this fetches into "
        "<weights>/esmc. The current protein-only feature path uses in-package "
        "canonical chemistry and never reads the publisher's ccd.pkl, so that "
        "417 MB file is not fetched. Budget 26.4 GB when hard links are "
        "available, or up to 52.7 GB when downloads and staged weights must "
        "be copied, plus an hour. "
        "Folding without the "
        "language model is possible and is a different model, so it has to be "
        "asked for with no_language_model=true.",
    ),
    "openfold3": ModelAssets(
        model="openfold3",
        source="https://github.com/aqlaboratory/openfold-3",
        licence="Apache-2.0 for code and weights; the weight repository is gated",
        # Empty on purpose, and for a different reason than AlphaFold 3's. These
        # weights *are* redistributable -- Apache-2.0, with the training data
        # published -- but the repository is access-gated: even its example
        # config returns HTTP 401 unauthenticated. A download FoldJAX cannot
        # authenticate would fail with a bare 401, which reads like a broken URL
        # rather than an access request nobody has made yet.
        downloads=(),
        native="of3_ft3_v1.pt",
        requires=("of3_ft3_v1.pt",),
        convert=_bring_your_own,
        notes="Request access at https://huggingface.co/OpenFold/OpenFold3, then "
        "put of3_ft3_v1.pt in this directory. The port reads the released "
        "checkpoint directly with FoldJAX's built-in torch-archive reader, so "
        "loading it needs no PyTorch installation. A "
        "safetensors export can also be given to --weights.",
    ),
    "boltz2": ModelAssets(
        model="boltz2",
        source="https://github.com/jwohlwend/boltz",
        licence="MIT",
        downloads=(
            Download(
                name="boltz2_conf.ckpt",
                url=(
                    "https://huggingface.co/boltz-community/boltz-2/resolve/"
                    "6fdef46d763fee7fbb83ca5501ccceff43b85607/boltz2_conf.ckpt"
                ),
                sha256=(
                    "090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1"
                ),
                size=2286561469,
            ),
            Download(
                name="boltz2_aff.ckpt",
                url=(
                    "https://huggingface.co/boltz-community/boltz-2/resolve/"
                    "6fdef46d763fee7fbb83ca5501ccceff43b85607/boltz2_aff.ckpt"
                ),
                sha256=(
                    "dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e"
                ),
                size=2062139170,
            ),
            Download(
                name="mols.tar",
                url=(
                    "https://huggingface.co/boltz-community/boltz-2/resolve/"
                    "6fdef46d763fee7fbb83ca5501ccceff43b85607/mols.tar"
                ),
                sha256=(
                    "39e076d96dbec6b4e86982bbda16f3a53a2a60c9bdc17828d88f6f9a0c7d1fd7"
                ),
                size=1855662080,
            ),
        ),
        native="boltz2_conf.safetensors",
        requires=(*_BOLTZ_NATIVE_FILES, "mols"),
        convert=_convert_boltz2,
        directory_requires=(("mols", _BOLTZ_CANONICAL_MOLECULES),),
        ready_check=_boltz_ready,
        conversion_sources=(
            "boltz2_conf.ckpt",
            "boltz2_aff.ckpt",
            "mols.tar",
        ),
        conversion_schema="boltz2-safetensors-v1",
        notes="mols.tar is unpacked to weights/boltz2/mols and is passed as the "
        "`mols` option automatically. Conversion and prediction use FoldJAX's "
        "torch-free checkpoint reader and NumPy featurizer.",
    ),
}


def _protenix_variant_assets(profile: str) -> ModelAssets:
    variant = _PROTENIX_VARIANTS[profile]
    base = REGISTRY["protenix"]
    shared = tuple(item for item in base.downloads if item.shared)
    return dataclasses.replace(
        base,
        model=variant.model,
        downloads=(variant.mini, variant.embedding, *shared),
        native=variant.native,
        requires=(variant.native, variant.embedding.name),
        convert=_convert_protenix_variant,
        in_default_setup=False,
        conversion_sources=(variant.mini.name, variant.embedding.name),
        conversion_schema=f"{variant.model}-native-v1",
        staged_sources=((variant.embedding.name, variant.embedding.name),),
        requires_manifest=True,
        notes=(
            f"Protenix {profile} managed bundle. The mini structure checkpoint "
            "is converted with FoldJAX's restricted torch-archive reader and "
            f"{variant.embedding.name} is staged beside the native .jax file "
            "for the JAX ESM/ISM provider."
        ),
    )


def _protenix_variant_assets_for_internal_model(model: str) -> ModelAssets:
    try:
        profile = _PROTENIX_PROFILE_BY_INTERNAL_MODEL[model]
    except KeyError as error:
        raise ValueError(f"unknown internal Protenix asset root: {model}") from error
    return _protenix_variant_assets(profile)


def available() -> tuple[str, ...]:
    return tuple(sorted(REGISTRY))


def available_profiles(model: str) -> tuple[str, ...]:
    """Return the managed asset profiles accepted for one model.

    Every model keeps the compatible ``released`` default. ESMFold2 additionally
    exposes ``structure-only`` for the explicit no-language-model variant, which
    avoids downloading or requiring ESMC-6B.
    """
    from foldjax.registry import normalize_model_name

    normalized = normalize_model_name(model)
    if normalized not in REGISTRY:
        raise ValueError(
            f"{normalized} has no managed assets; supply --weights explicitly "
            f"(managed: {', '.join(available())})"
        )
    if normalized == "esmfold2":
        return (RELEASED_PROFILE, STRUCTURE_ONLY_PROFILE)
    if normalized == "protenix":
        return (
            RELEASED_PROFILE,
            PROTENIX_MINI_ESM_PROFILE,
            PROTENIX_MINI_ISM_PROFILE,
        )
    return (RELEASED_PROFILE,)


def assets_for(model: str, *, profile: str | None = None) -> ModelAssets:
    """Return the managed files for ``profile`` (the release by default)."""
    from foldjax.registry import normalize_model_name

    internal_profile = _PROTENIX_PROFILE_BY_INTERNAL_MODEL.get(model)
    if internal_profile is not None:
        if profile is not None and profile != internal_profile:
            raise ValueError(
                f"internal asset root {model!r} belongs to profile "
                f"{internal_profile!r}, not {profile!r}"
            )
        normalized = "protenix"
        profile = internal_profile
    else:
        normalized = normalize_model_name(model)
    if normalized not in REGISTRY:
        raise ValueError(
            f"{normalized} has no managed assets; supply --weights explicitly "
            f"(managed: {', '.join(available())})"
        )
    selected = RELEASED_PROFILE if profile is None else profile
    if not isinstance(selected, str) or selected not in available_profiles(normalized):
        choices = ", ".join(available_profiles(normalized))
        raise ValueError(
            f"unsupported asset profile {selected!r} for {normalized}; "
            f"choose one of {choices}"
        )
    spec = REGISTRY[normalized]
    if selected == RELEASED_PROFILE:
        return spec

    if normalized == "protenix":
        return _protenix_variant_assets(selected)

    downloads = tuple(
        item for item in spec.downloads if not item.name.startswith("esmc/")
    )
    requires = tuple(
        item for item in spec.requires if not item.startswith("esmc/")
    )
    return dataclasses.replace(
        spec,
        downloads=downloads,
        requires=requires,
        ready_check=_esmfold2_structure_only_ready,
        notes=(
            "ESMFold2 structure-only profile: the 940 MB structure checkpoint "
            "and config, without ESMC-6B. This profile is valid only with "
            "no_language_model=true and predicts a deliberately different model "
            "from the released structure+ESMC bundle."
        ),
    )


def profile_status(model: str) -> tuple[dict[str, object], ...]:
    """Report readiness and transfer size for every managed profile."""
    rows = []
    for profile in available_profiles(model):
        spec = assets_for(model, profile=profile)
        sizes = [item.size for item in spec.downloads]
        download_bytes = (
            sum(sizes) if sizes and all(size is not None for size in sizes) else None
        )
        downloaded = sum(
            1
            for item in spec.downloads
            if _present_download(item.target(spec.model), item)
        )
        rows.append(
            {
                "profile": profile,
                "ready": spec.ready(),
                "downloaded": f"{downloaded}/{len(spec.downloads)}",
                "download_bytes": download_bytes,
                "path": str(spec.native_path()),
                "notes": spec.notes,
            }
        )
    return tuple(rows)


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _same_file_contents(first: Path, second: Path, item: Download) -> bool:
    """Compare a staged asset with its verified download.

    Published SHA-256 values are authoritative. For small files without one,
    hashing both sides is cheap and avoids accepting equal-size corruption. For
    a large hashless file there is no identity stronger than its published size.
    """
    try:
        if not first.is_file() or not second.is_file():
            return False
        if first.samefile(second):
            return True
        if first.stat().st_size != second.stat().st_size:
            return False
    except OSError:
        return False
    if item.sha256 is not None:
        return _digest(first) == item.sha256
    return _digest(first) == _digest(second)


def _verified(
    path: Path, item: Download, *, known_sha256: str | None = None
) -> bool:
    """True when the file on disk is already the published one."""
    if not path.is_file():
        return False
    if item.size is not None and path.stat().st_size != item.size:
        return False
    if item.sha256 is not None:
        actual = known_sha256 if known_sha256 is not None else _digest(path)
        return actual == item.sha256
    # No published hash: a non-empty file of the right size is as far as this
    # can go, so re-downloading it would be guesswork either way.
    return path.stat().st_size > 0


def _present_download(path: Path, item: Download) -> bool:
    """Cheap status check; downloads still receive full verification on use."""
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        return item.size is None or path.stat().st_size == item.size
    except OSError:
        return False


@contextmanager
def _download_lock(path: Path) -> Iterator[None]:
    """Serialize writers of one published file across local processes."""
    lock = path.with_name(f".{path.name}.lock")
    with lock.open("a+b") as handle:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - FoldJAX targets Linux/CUDA
            yield
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _conversion_lock(model: str) -> Iterator[None]:
    """Serialize publication of one model's native bundle."""
    root = weights_dir(model)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".conversion.lock"
    with lock.open("a+b") as handle:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - FoldJAX targets Linux/CUDA
            yield
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                _cleanup_abandoned_staging(model)
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _cleanup_abandoned_staging(model: str) -> None:
    """Remove only FoldJAX-owned temp trees after exclusive model locking."""
    root = weights_dir(model)
    candidates: list[Path] = []
    if model == "boltz2":
        candidates.extend(root.parent.glob(".foldjax-boltz-native-*"))
        candidates.extend(root.glob(".foldjax-mols-*"))
    elif model in {"opendde", "protenix"}:
        candidates.extend(root.glob(f".foldjax-{model}-native-*"))
    elif model in _PROTENIX_PROFILE_BY_INTERNAL_MODEL:
        candidates.extend(root.glob(".foldjax-protenix-variant-*"))
    elif model == "esmfold2":
        candidates.extend(root.glob(".foldjax-stage-*"))
        candidates.extend((root / "esmc").glob(".foldjax-stage-*"))
    for candidate in candidates:
        if candidate.is_symlink() or candidate.is_file():
            candidate.unlink(missing_ok=True)
        elif candidate.is_dir():
            shutil.rmtree(candidate)


def download(item: Download, model: str, *, on_progress=None) -> Path:
    """Download one file unless a verified copy is already present.

    These are multi-gigabyte model weights over the public internet, so a
    dropped or stalled connection is an ordinary event rather than an
    exceptional one. Three things follow from that:

    * a read that ends short of ``Content-Length`` is a truncation, not a
      finished download. Without this check a half-written checkpoint can be
      renamed into place before its identity is verified;
    * a socket with no timeout can wait forever on a server that accepted the
      connection and then stopped sending;
    * one transient failure should not throw away several GB of progress, so
      the attempt is retried.
    """
    path = item.target(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    with _download_lock(path):
        if _verified(path, item):
            return path
        has_published_identity = item.sha256 is not None or item.size is not None
        if has_published_identity and _verified(partial, item):
            os.replace(partial, path)
            return path
        # With no publisher hash or size, a partial inherited from a killed
        # process has no trustworthy completion boundary. A prefix produced by
        # this invocation may still be resumed on its next retry below.
        if not has_published_identity and partial.is_file():
            partial.unlink()
        last_error: Exception | None = None
        for attempt in range(_ATTEMPTS):
            try:
                _stream(item, partial, on_progress=on_progress)
                actual: str | None = None
                if item.sha256 is not None:
                    actual = _digest(partial)
                    if actual != item.sha256:
                        raise ValueError(
                            f"{item.name} failed verification: expected sha256 "
                            f"{item.sha256}, got {actual}"
                        )
                if not _verified(partial, item, known_sha256=actual):
                    raise ValueError(
                        f"{item.name} failed verification after download"
                    )
                os.replace(partial, path)
                break
            except (OSError, ValueError, http.client.IncompleteRead) as error:
                last_error = error
                # A bad complete payload cannot be resumed. Short/network
                # failures retain the prefix for the next Range request.
                if isinstance(error, ValueError):
                    partial.unlink(missing_ok=True)
                if attempt == _ATTEMPTS - 1:
                    partial.unlink(missing_ok=True)
                    raise
        else:  # pragma: no cover - the loop always breaks or raises
            raise last_error  # type: ignore[misc]
    return path


def _adopt_legacy_protenix_variant_download(item: Download, model: str) -> bool:
    """Hard-link a verified pre-profile download into its isolated namespace.

    Early FoldJAX ESM experiments downloaded the official files under
    ``downloads/protenix``. Profiles intentionally use separate roots to keep
    conversion manifests independent, but re-downloading 5--11 GB solely to
    change namespaces would be wasteful.
    """
    if model not in _PROTENIX_PROFILE_BY_INTERNAL_MODEL or item.shared:
        return False
    legacy = downloads_dir("protenix") / item.name
    target = item.target(model)
    if not _verified(legacy, item):
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    with _download_lock(target):
        if _verified(target, item):
            return True
        with tempfile.TemporaryDirectory(
            prefix=f".{item.name}.adopt-", dir=target.parent
        ) as scratch:
            staged = Path(scratch) / item.name
            try:
                os.link(legacy, staged)
            except OSError:
                shutil.copy2(legacy, staged)
            if not _same_file_contents(staged, legacy, item):
                raise RuntimeError(
                    f"Protenix profile adoption was incomplete: {legacy}"
                )
            os.replace(staged, target)
    return True


def _adopt_staged_esmfold2_download(item: Download, model: str) -> bool:
    """Move a verified legacy ESMFold2 stage into the download namespace.

    Before ESMFold2 gained an atomic staging manifest, users could place its
    published files directly under ``weights/esmfold2``.  Keep those files as
    the runtime copy, but first hard-link (or copy) each verified regular file
    into ``downloads/esmfold2`` so the normal staging path can record current
    provenance without transferring the multi-GB bundle again.
    """
    if model != "esmfold2" or item.shared:
        return False
    target = item.target(model)
    try:
        relative = target.relative_to(downloads_dir(model))
    except ValueError:
        return False
    staged_source = weights_dir(model) / relative
    try:
        if staged_source.is_symlink() or not _verified(staged_source, item):
            return False
        source_stat = staged_source.stat()
        source_identity = (
            source_stat.st_dev,
            source_stat.st_ino,
            source_stat.st_size,
            source_stat.st_mtime_ns,
            source_stat.st_ctime_ns,
        )
    except OSError:
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    with _download_lock(target):
        if not target.is_symlink() and _verified(target, item):
            return True
        # Recheck identity after waiting for the download lock without hashing
        # a 25.4 GB bundle twice. A concurrent/manual writer that replaced or
        # changed the source after verification is not adopted.
        try:
            current = staged_source.stat()
            current_identity = (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            )
            if staged_source.is_symlink() or current_identity != source_identity:
                return False
        except OSError:
            return False
        with tempfile.TemporaryDirectory(
            prefix=f".{target.name}.adopt-", dir=target.parent
        ) as scratch:
            adopted = Path(scratch) / target.name
            try:
                os.link(staged_source, adopted)
            except OSError:
                shutil.copy2(staged_source, adopted)
            if not _same_file_contents(adopted, staged_source, item):
                raise RuntimeError(
                    f"ESMFold2 staged-file adoption was incomplete: {staged_source}"
                )
            os.replace(adopted, target)
    return True


def _stream(item: Download, partial: Path, *, on_progress=None) -> None:
    """Fetch ``item`` into ``partial``, or raise if it arrives incomplete."""
    offset = partial.stat().st_size if partial.is_file() else 0
    if item.size is not None and offset >= item.size:
        partial.unlink()
        offset = 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = urllib.request.Request(  # noqa: S310 - registry pins https
        item.url, headers=headers
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:  # noqa: S310
        status = getattr(response, "status", None)
        if status is None:
            getcode = getattr(response, "getcode", None)
            status = getcode() if getcode is not None else None
        if not offset and status == 206:
            raise ValueError(
                f"{item.name} server returned a partial response without a "
                "Range request"
            )
        accepted_range = bool(offset and status == 206)
        range_total: int | None = None
        if accepted_range:
            content_range = response.headers.get("Content-Range")
            match = (
                _CONTENT_RANGE.fullmatch(content_range.strip())
                if content_range
                else None
            )
            if match is None or int(match.group(1)) != offset:
                raise ValueError(
                    f"{item.name} server returned an invalid Content-Range for "
                    f"the requested offset {offset}: {content_range!r}"
                )
            end = int(match.group(2))
            if end < offset:
                raise ValueError(
                    f"{item.name} server returned a reversed Content-Range: "
                    f"{content_range!r}"
                )
            if match.group(3) != "*":
                range_total = int(match.group(3))
                if range_total <= end:
                    raise ValueError(
                        f"{item.name} server returned an invalid total in "
                        f"Content-Range: {content_range!r}"
                    )
                if item.size is not None and range_total != item.size:
                    raise ValueError(
                        f"{item.name} server reports {range_total} bytes, expected "
                        f"{item.size}"
                    )
        if offset and not accepted_range:
            # Servers may ignore Range. Restart deliberately rather than append
            # a complete response to an existing prefix.
            offset = 0
        response_length = int(response.headers.get("Content-Length") or 0)
        if accepted_range and response_length and end - offset + 1 != response_length:
            raise ValueError(
                f"{item.name} Content-Length disagrees with Content-Range"
            )
        total = (
            item.size
            or range_total
            or (offset + response_length if response_length else None)
        )
        mode = "ab" if accepted_range else "wb"
        done = offset
        with open(partial, mode) as handle:
            while chunk := response.read(_CHUNK):
                handle.write(chunk)
                done += len(chunk)
                if on_progress is not None:
                    on_progress(item.name, done, total)
    if total and done != total:
        raise OSError(
            f"{item.name} arrived incomplete: {done} of {total} bytes. "
            "The connection dropped; the verified prefix was kept for retry"
        )


def _conversion_provenance_recorded(spec: ModelAssets) -> bool:
    """Whether a ready legacy/native result already carries current metadata."""
    root = weights_dir(spec.model)
    if spec.model == "boltz2":
        return _boltz_native_provenance_complete(
            root
        ) and _boltz_molecule_index_complete(root / "mols")
    if spec.conversion_sources:
        return _native_manifest_complete(spec)
    return True


def _write_conversion_provenance(spec: ModelAssets) -> None:
    """Adopt a verified legacy result or mark a newly converted one."""
    if spec.model == "boltz2":
        molecules = weights_dir(spec.model) / "mols"
        _write_boltz_molecule_completion(molecules)
        _write_boltz_native_manifest(weights_dir(spec.model))
    elif spec.conversion_sources:
        _write_native_manifest(spec)


def _downloads_ready_for_fast_path(spec: ModelAssets) -> bool:
    """Check direct-use assets fully while avoiding needless model re-hashes."""
    provenance = _conversion_provenance_recorded(spec)
    for item in spec.downloads:
        target = item.target(spec.model)
        covered = provenance and item.name in spec.conversion_sources
        if spec.model == "esmfold2" and spec.ready():
            covered = True
        if not (
            _present_download(target, item) if covered else _verified(target, item)
        ):
            return False
    return True


def _asset_event(
    callback: Callable[[AssetEvent], None] | None,
    spec: ModelAssets,
    profile: str | None,
    action: str,
    status: str,
    message: str,
    *,
    item: str | None = None,
    path: Path | None = None,
    bytes: int | None = None,
    elapsed_seconds: float | None = None,
) -> None:
    """Emit one optional event without coupling the asset store to a logger."""

    if callback is None:
        return
    callback(
        AssetEvent(
            model=_public_model_name(spec.model),
            profile=profile or RELEASED_PROFILE,
            action=action,
            status=status,
            message=message,
            item=item,
            path=path,
            bytes=bytes,
            elapsed_seconds=elapsed_seconds,
        )
    )


def _conversion_action(spec: ModelAssets) -> tuple[str, str]:
    """Return the honest terminal verb and format description for ``spec``."""

    if spec.convert is _stage_esmfold2:
        return "stage", "published safetensors for direct JAX loading"
    if spec.convert is _bring_your_own:
        return "direct", "publisher-native parameters for direct JAX loading"
    if spec.convert is _convert_boltz2:
        return "convert", "publisher checkpoints to native JAX safetensors"
    return "convert", "publisher checkpoint to native JAX parameters"


def _check_free_space(
    spec: ModelAssets,
    *,
    convert: bool,
    on_event: Callable[[AssetEvent], None] | None,
    profile: str | None,
) -> None:
    """Refuse a download the disk cannot hold, and say when it will be tight.

    The registry pins every published file's size, so the answer is known
    before the first byte moves -- and the failure it prevents is expensive:
    ESMFold2's complete bundle is about 26 GB, and running out at the end
    wastes the whole transfer. Conversion writes a second copy of what it
    reads, so the comfortable figure is roughly twice the download; that one is
    a warning rather than a refusal because the peak depends on the model and a
    wrong refusal is worse than a tight success.
    """
    sizes = [item.size for item in spec.downloads]
    if not sizes or any(size is None for size in sizes):
        return
    needed = sum(int(size) for size in sizes)
    root = downloads_dir(spec.model)
    probe = root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        return
    if free < needed:
        raise RuntimeError(
            f"{spec.model} needs {needed / 1e9:.1f} GB of published files and "
            f"{free / 1e9:.1f} GB is free at {root}. Free some space, or point "
            "FOLDJAX_HOME at a larger filesystem."
        )
    if convert and free < needed * 2:
        _asset_event(
            on_event,
            spec,
            profile,
            "resolve",
            "warn",
            f"{free / 1e9:.1f} GB free for a {needed / 1e9:.1f} GB download; "
            "conversion writes a second copy and may not fit",
            path=root,
        )


def fetch(
    model: str,
    *,
    profile: str | None = None,
    on_progress=None,
    on_event: Callable[[AssetEvent], None] | None = None,
    convert: bool = True,
) -> Path:
    """Download and convert one model's selected managed profile.

    ``None`` keeps the compatible complete released profile. Re-running is
    cheap: verified downloads and an existing conversion are both skipped.
    """
    spec = assets_for(model, profile=profile)
    total_started = time.perf_counter()
    _asset_event(
        on_event,
        spec,
        profile,
        "resolve",
        "start",
        f"checking {len(spec.downloads)} published file(s)",
        path=spec.native_path(),
    )
    if not convert and not spec.downloads:
        raise RuntimeError(
            f"FoldJAX has no downloadable parameter files for {spec.model}.\n"
            f"{spec.notes}\nExpected here: {weights_dir(spec.model)}"
        )
    # Converted already *and* holding every published file: nothing to do. The
    # second half matters when the registry gains a file a machine fetched
    # before it existed -- the shared template metadata did exactly that, and a
    # `ready()`-only check would have returned success while never fetching it.
    # Cheap size checks are enough here because a current conversion manifest
    # binds the native result to the registry's exact source hashes. Download
    # performs full verification whenever that provenance is absent or stale.
    if (
        convert
        and spec.ready()
        and _conversion_provenance_recorded(spec)
        and _downloads_ready_for_fast_path(spec)
    ):
        _asset_event(
            on_event,
            spec,
            profile,
            "ready",
            "skip",
            "verified native weights are already ready",
            path=spec.native_path(),
            elapsed_seconds=time.perf_counter() - total_started,
        )
        return spec.native_path()

    _check_free_space(spec, convert=convert, on_event=on_event, profile=profile)
    provenance = _conversion_provenance_recorded(spec)
    for item in spec.downloads:
        if _adopt_legacy_protenix_variant_download(item, spec.model):
            _asset_event(
                on_event,
                spec,
                profile,
                "download",
                "skip",
                "adopted an existing verified file",
                item=item.name,
                path=item.target(spec.model),
                bytes=item.size,
            )
            continue
        if _adopt_staged_esmfold2_download(item, spec.model):
            _asset_event(
                on_event,
                spec,
                profile,
                "download",
                "skip",
                "adopted an existing verified file",
                item=item.name,
                path=item.target(spec.model),
                bytes=item.size,
            )
            continue
        if (
            convert
            and provenance
            and item.name in spec.conversion_sources
            and _present_download(item.target(spec.model), item)
        ):
            _asset_event(
                on_event,
                spec,
                profile,
                "download",
                "skip",
                "verified source is already present",
                item=item.name,
                path=item.target(spec.model),
                bytes=item.size,
            )
            continue
        if _verified(item.target(spec.model), item):
            _asset_event(
                on_event,
                spec,
                profile,
                "download",
                "skip",
                "verified published file is already present",
                item=item.name,
                path=item.target(spec.model),
                bytes=item.size,
            )
            continue
        item_started = time.perf_counter()
        _asset_event(
            on_event,
            spec,
            profile,
            "download",
            "start",
            "fetching and verifying published file",
            item=item.name,
            path=item.target(spec.model),
            bytes=item.size,
        )
        try:
            downloaded = download(item, spec.model, on_progress=on_progress)
        except Exception:
            _asset_event(
                on_event,
                spec,
                profile,
                "download",
                "error",
                "download or verification failed",
                item=item.name,
                path=item.target(spec.model),
                elapsed_seconds=time.perf_counter() - item_started,
            )
            raise
        _asset_event(
            on_event,
            spec,
            profile,
            "download",
            "done",
            "published file is verified",
            item=item.name,
            path=downloaded,
            # Telemetry must never turn a successful verified download into a
            # failure because the filesystem changed between verification and
            # logging. The registry's pinned size is the useful value here.
            bytes=item.size,
            elapsed_seconds=time.perf_counter() - item_started,
        )
    if not convert:
        _asset_event(
            on_event,
            spec,
            profile,
            "ready",
            "done",
            "downloads are verified; conversion was not requested",
            path=downloads_dir(spec.model),
            elapsed_seconds=time.perf_counter() - total_started,
        )
        return downloads_dir(spec.model)
    action, description = _conversion_action(spec)
    _asset_event(
        on_event,
        spec,
        profile,
        action,
        "waiting",
        "waiting for the model conversion lock",
        path=spec.native_path(),
    )
    with _conversion_lock(spec.model):
        if spec.ready() and _conversion_provenance_recorded(spec):
            # Another process may have converted while this one downloaded.
            _asset_event(
                on_event,
                spec,
                profile,
                action,
                "skip",
                "another process prepared the native weights",
                path=spec.native_path(),
            )
            _asset_event(
                on_event,
                spec,
                profile,
                "ready",
                "done",
                "native weights are verified",
                path=spec.native_path(),
                elapsed_seconds=time.perf_counter() - total_started,
            )
            return spec.native_path()

        conversion_started = time.perf_counter()
        _asset_event(
            on_event,
            spec,
            profile,
            action,
            "start",
            description,
            path=spec.native_path(),
            bytes=sum(item.size or 0 for item in spec.downloads) or None,
        )
        try:
            if spec.convert is _stage_esmfold2:
                _stage_esmfold2(
                    spec.model,
                    downloads_dir(spec.model),
                    sources_already_verified=True,
                    spec=spec,
                )
            elif spec.convert is _convert_protenix_variant:
                _convert_protenix_variant(
                    spec.model,
                    downloads_dir(spec.model),
                    sources_already_verified=True,
                    spec=spec,
                )
            else:
                spec.convert(spec.model, downloads_dir(spec.model))
        except Exception:
            _asset_event(
                on_event,
                spec,
                profile,
                action,
                "error",
                f"{description} failed",
                path=spec.native_path(),
                elapsed_seconds=time.perf_counter() - conversion_started,
            )
            raise
        _asset_event(
            on_event,
            spec,
            profile,
            action,
            "done",
            description,
            path=spec.native_path(),
            elapsed_seconds=time.perf_counter() - conversion_started,
        )
        validation_started = time.perf_counter()
        _asset_event(
            on_event,
            spec,
            profile,
            "validate",
            "start",
            "checking the native bundle and source provenance",
            path=spec.native_path(),
        )
        try:
            if not _conversion_provenance_recorded(spec):
                _write_conversion_provenance(spec)
            if not spec.ready():
                missing = [
                    item
                    for item in spec.requires
                    if not (weights_dir(spec.model) / item).exists()
                ]
                if not spec.downloads:
                    # Nothing was downloaded because nothing can be: these are the
                    # models whose parameters their publisher hands out only on
                    # request. Reporting it as a failed conversion describes a step
                    # that never runs and hides the one thing the user has to do.
                    #
                    # The reason differs per model and only `notes` knows it --
                    # AlphaFold 3's parameters may not be redistributed at all,
                    # while OpenFold3's are Apache-2.0 behind a gated repository.
                    # Asserting either one here would be wrong for the other.
                    raise RuntimeError(
                        f"FoldJAX cannot fetch {spec.model}'s parameters; its "
                        f"publisher releases them only on request.\n{spec.notes}\n"
                        f"Expected here: {weights_dir(spec.model)}\n"
                        f"Still missing: {', '.join(missing)}"
                    )
                raise RuntimeError(
                    f"{spec.model} conversion did not produce: {', '.join(missing)}"
                )
        except Exception:
            _asset_event(
                on_event,
                spec,
                profile,
                "validate",
                "error",
                "native bundle or source provenance is incomplete",
                path=spec.native_path(),
                elapsed_seconds=time.perf_counter() - validation_started,
            )
            raise
        _asset_event(
            on_event,
            spec,
            profile,
            "validate",
            "done",
            "native bundle and conversion provenance are complete",
            path=spec.native_path(),
            elapsed_seconds=time.perf_counter() - validation_started,
        )
        _asset_event(
            on_event,
            spec,
            profile,
            "ready",
            "done",
            "weights are ready for prediction",
            path=spec.native_path(),
            elapsed_seconds=time.perf_counter() - total_started,
        )
        return spec.native_path()


def resolve_weights(model: str, *, profile: str | None = None) -> Path:
    """Return weights ready for the selected managed profile.

    Omitting ``profile`` keeps the complete released-model requirement. A
    structure-only ESMFold2 request must ask for that profile explicitly; it is
    never inferred here from unrelated request state.
    """
    spec = assets_for(model, profile=profile)
    if spec.ready():
        return spec.native_path()
    native = spec.native_path()
    if not spec.downloads:
        raise FileNotFoundError(
            f"no usable {spec.model} weights at {native}.\n"
            f"{spec.notes}\n"
            f"Expected under: {weights_dir(spec.model)}\n"
            "For prediction, supply PredictionRequest.weights or pass "
            "`foldjax predict --weights PATH`."
        )
    profile_flag = (
        ""
        if profile is None or profile == RELEASED_PROFILE
        else f" --profile {profile}"
    )
    public_model = _public_model_name(spec.model)
    raise FileNotFoundError(
        f"no converted {spec.model} weights at {native}. Run "
        f"`foldjax weights fetch --model {public_model}{profile_flag}`, or "
        "supply PredictionRequest.weights / `foldjax predict --weights PATH`."
    )


def status() -> Iterator[dict[str, object]]:
    """Report, per managed model, what is downloaded and what is converted."""
    for name in available():
        spec = REGISTRY[name]
        downloaded = sum(
            1
            for item in spec.downloads
            if _present_download(item.target(spec.model), item)
        )
        yield {
            "model": name,
            "licence": spec.licence,
            "source": spec.source,
            "downloaded": f"{downloaded}/{len(spec.downloads)}",
            "converted": spec.ready(),
            "path": str(spec.native_path()),
        }
