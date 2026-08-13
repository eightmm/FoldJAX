"""Fetch released model files and convert them into native JAX weights.

FoldJAX predicts from torch-free JAX weights, but every upstream project
publishes torch checkpoints. This module owns that gap: it records where the
official files live, downloads them, verifies what can be verified, converts
them once, and resolves the converted artifact at prediction time.

Nothing here runs during prediction, and no file is redistributed — each is
downloaded from its own publisher under that project's terms.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from collections.abc import Callable, Iterator
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
    needs_torch: bool = True
    notes: str = ""
    #: Whether `foldjax setup` fetches this without being asked. False for a
    #: model whose runtime is not in the default install: downloading gigabytes
    #: for a backend the environment cannot run is not a service to anyone.
    #: `foldjax weights fetch --model <name>` still gets it.
    in_default_setup: bool = True
    extras: tuple[str, ...] = field(default_factory=tuple)

    def native_path(self) -> Path:
        root = weights_dir(self.model)
        return root if self.native == "." else root / self.native

    def ready(self) -> bool:
        root = weights_dir(self.model)
        return all((root / item).exists() for item in self.requires)


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
    # The loader sniffs for gzip, so archives written before this still work.
    save_native_weights(
        out, load_torch_checkpoint(source / "protenix.pt"), compress=False
    )
    return out


def _convert_opendde(model: str, source: Path) -> Path:
    from foldjax.models.opendde.bridge.export_weights import (
        load_torch_checkpoint,
        save_native_weights,
    )

    out = weights_dir(model) / "opendde.jax"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Written uncompressed: these are float32 weights, so gzip saves about 6%
    # of disk but costs 6x on every load (8.5 s vs 1.3 s for OpenDDE's 2.6 GB).
    # The loader sniffs for gzip, so archives written before this still work.
    save_native_weights(
        out, load_torch_checkpoint(source / "opendde.pt"), compress=False
    )
    return out


def _convert_boltz2(model: str, source: Path) -> Path:
    from foldjax.models.boltz2.bridge.export_weights import (
        export_affinity,
        export_confidence,
    )

    out_dir = weights_dir(model)
    weights = export_confidence(source / "boltz2_conf.ckpt", out_dir)
    affinity = source / "boltz2_aff.ckpt"
    if affinity.is_file():
        export_affinity(affinity, out_dir)
    _unpack_boltz_mols(source / "mols.tar", out_dir / "mols")
    return weights


def _unpack_boltz_mols(archive: Path, destination: Path) -> None:
    """Boltz needs its CCD molecule directory beside the weights."""
    import tarfile

    if destination.is_dir() and any(destination.iterdir()):
        return
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        # Python 3.12 warns without an explicit policy; refuse anything that
        # would escape the destination.
        tar.extractall(destination, filter="data")
    # The tarball may or may not nest everything under one directory.
    entries = [path for path in destination.iterdir() if path.name != "__MACOSX"]
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        for item in list(inner.iterdir()):
            shutil.move(str(item), destination / item.name)
        inner.rmdir()


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


def _stage_esmfold2(model: str, source: Path) -> Path:
    """Put the published files where `from_pretrained` expects a model.

    Nothing is converted -- this backend drives upstream's torch model and
    loads its safetensors as they ship -- but the loader takes a *directory*
    holding `config.json` beside the weights, and the download step keeps
    files under `downloads/`. Hard-linked where the filesystem allows it, so
    the 1.3 GB is not stored twice.
    """
    out = weights_dir(model)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("model.safetensors", "config.json", "ccd.pkl"):
        origin = source / name
        target = out / name
        if not origin.is_file() or target.exists():
            continue
        try:
            os.link(origin, target)
        except OSError:
            shutil.copy2(origin, target)
    return out


def _bring_your_own(model: str, source: Path) -> Path:
    """No conversion: the published file is already what the model loads."""
    return weights_dir(model)


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
        needs_torch=False,
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
            ),
            _CCD_COMPONENTS,
            _CCD_RDKIT,
            _TEMPLATE_RELEASE_DATES,
            _TEMPLATE_OBSOLETE,
        ),
        native="protenix_base_default_v1.0.0.jax",
        requires=("protenix_base_default_v1.0.0.jax",),
        convert=_convert_protenix,
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
                sha256=None,
                size=None,
            ),
            Download(
                name="ccd.pkl",
                url=(
                    "https://huggingface.co/biohub/ESMFold2/resolve/"
                    "8fc3ff471022fdce52c77030685eb775de0c00a3/ccd.pkl"
                ),
                sha256=(
                    "9ff44b1927c6b9198e38ffe0928706827a09a350c15530beeeabebfa88038fc5"
                ),
                size=417306584,
            ),
        ),
        native="model.safetensors",
        requires=("model.safetensors", "config.json"),
        convert=_stage_esmfold2,
        in_default_setup=False,
        notes="Torch weights, loaded as they ship -- nothing is converted, "
        "because this backend drives upstream's model rather than porting it. "
        "Loading also pulls ESMC-6B (~24 GB) from Hugging Face on first use: "
        "it is the language model ESMFold2 reads its representations from, and "
        "upstream distributes it separately. Needs the esmfold2 extra.",
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
        "checkpoint directly, so nothing is converted -- but a .pt is a torch "
        "file, so loading it needs the torch-bridge extra. A safetensors export "
        "loads without torch and can be given to --weights instead.",
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
                    "main/boltz2_conf.ckpt"
                ),
            ),
            Download(
                name="boltz2_aff.ckpt",
                url=(
                    "https://huggingface.co/boltz-community/boltz-2/resolve/"
                    "main/boltz2_aff.ckpt"
                ),
            ),
            Download(
                name="mols.tar",
                url=(
                    "https://huggingface.co/boltz-community/boltz-2/resolve/"
                    "main/mols.tar"
                ),
            ),
        ),
        native="boltz2_conf.safetensors",
        requires=("boltz2_conf.safetensors", "mols"),
        convert=_convert_boltz2,
        extras=("boltz-preprocess",),
        notes="mols.tar is unpacked to weights/boltz2/mols and is passed as the "
        "`mols` option automatically.",
    ),
}


def available() -> tuple[str, ...]:
    return tuple(sorted(REGISTRY))


def assets_for(model: str) -> ModelAssets:
    from foldjax.registry import normalize_model_name

    normalized = normalize_model_name(model)
    if normalized not in REGISTRY:
        raise ValueError(
            f"{normalized} has no managed assets; supply --weights explicitly "
            f"(managed: {', '.join(available())})"
        )
    return REGISTRY[normalized]


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _verified(path: Path, item: Download) -> bool:
    """True when the file on disk is already the published one."""
    if not path.is_file():
        return False
    if item.size is not None and path.stat().st_size != item.size:
        return False
    if item.sha256 is not None:
        return _digest(path) == item.sha256
    # No published hash: a non-empty file of the right size is as far as this
    # can go, so re-downloading it would be guesswork either way.
    return path.stat().st_size > 0


def download(item: Download, model: str, *, on_progress=None) -> Path:
    """Download one file unless a verified copy is already present.

    These are multi-gigabyte model weights over the public internet, so a
    dropped or stalled connection is an ordinary event rather than an
    exceptional one. Three things follow from that:

    * a read that ends short of ``Content-Length`` is a truncation, not a
      finished download. Most published files here carry no sha256, so without
      this check a half-written checkpoint is renamed into place and passes
      every later "is it there?" test permanently;
    * a socket with no timeout can wait forever on a server that accepted the
      connection and then stopped sending;
    * one transient failure should not throw away several GB of progress, so
      the attempt is retried.
    """
    path = item.target(model)
    if _verified(path, item):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")

    last_error: Exception | None = None
    for attempt in range(_ATTEMPTS):
        try:
            _stream(item, partial, on_progress=on_progress)
            break
        except (OSError, ValueError) as error:
            last_error = error
            partial.unlink(missing_ok=True)
            if attempt == _ATTEMPTS - 1:
                raise
    else:  # pragma: no cover - the loop always breaks or raises
        raise last_error  # type: ignore[misc]

    if item.sha256 is not None:
        actual = _digest(partial)
        if actual != item.sha256:
            partial.unlink(missing_ok=True)
            raise ValueError(
                f"{item.name} failed verification: expected sha256 {item.sha256}, "
                f"got {actual}"
            )
    partial.replace(path)
    return path


def _stream(item: Download, partial: Path, *, on_progress=None) -> None:
    """Fetch ``item`` into ``partial``, or raise if it arrives incomplete."""
    request = urllib.request.Request(item.url)  # noqa: S310 - pinned https
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:  # noqa: S310
        declared = int(response.headers.get("Content-Length") or 0) or item.size
        done = 0
        with open(partial, "wb") as handle:
            while chunk := response.read(_CHUNK):
                handle.write(chunk)
                done += len(chunk)
                if on_progress is not None:
                    on_progress(item.name, done, declared)
    if declared and done != declared:
        raise OSError(
            f"{item.name} arrived incomplete: {done} of {declared} bytes. "
            "The connection dropped; nothing was kept"
        )


def fetch(model: str, *, on_progress=None, convert: bool = True) -> Path:
    """Download and convert one model's released files.

    Returns the artifact prediction loads. Re-running is cheap: verified
    downloads and an existing conversion are both skipped.
    """
    spec = assets_for(model)
    # Converted already *and* holding every published file: nothing to do. The
    # second half matters when the registry gains a file a machine fetched
    # before it existed -- the shared template metadata did exactly that, and a
    # `ready()`-only check would have returned success while never fetching it.
    # Existence, not verification, because re-hashing several GB to decide
    # whether to skip is the cost this early return exists to avoid.
    if convert and spec.ready():
        if all(item.target(spec.model).is_file() for item in spec.downloads):
            return spec.native_path()

    for item in spec.downloads:
        download(item, spec.model, on_progress=on_progress)
    if not convert:
        return downloads_dir(spec.model)
    if spec.ready():
        # Here only because a published file was missing above; converting again
        # would spend minutes and the torch-bridge extra to rebuild what is
        # already on disk.
        return spec.native_path()

    spec.convert(spec.model, downloads_dir(spec.model))
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
            # AlphaFold 3's parameters may not be redistributed at all, while
            # OpenFold3's are Apache-2.0 behind a gated repository. Asserting
            # either one here would be wrong for the other.
            raise RuntimeError(
                f"FoldJAX cannot fetch {spec.model}'s parameters; its publisher "
                f"releases them only on request.\n{spec.notes}\n"
                f"Expected here: {weights_dir(spec.model)}\n"
                f"Still missing: {', '.join(missing)}"
            )
        raise RuntimeError(
            f"{spec.model} conversion did not produce: {', '.join(missing)}"
        )
    return spec.native_path()


def resolve_weights(model: str) -> Path:
    """Return converted weights for ``model``, or say exactly how to get them."""
    spec = assets_for(model)
    if spec.ready():
        return spec.native_path()
    native = spec.native_path()
    extras = "".join(f" --extra {name}" for name in spec.extras)
    raise FileNotFoundError(
        f"no converted {spec.model} weights at {native}. Run "
        f"`foldjax weights fetch --model {spec.model}`"
        + (
            f" after `uv sync --extra cuda13 --extra torch-bridge{extras}`"
            if spec.needs_torch
            else ""
        )
        + ", or pass --weights explicitly."
    )


def status() -> Iterator[dict[str, object]]:
    """Report, per managed model, what is downloaded and what is converted."""
    for name in available():
        spec = REGISTRY[name]
        downloaded = sum(
            1 for item in spec.downloads if _verified(item.target(spec.model), item)
        )
        yield {
            "model": name,
            "licence": spec.licence,
            "source": spec.source,
            "downloaded": f"{downloaded}/{len(spec.downloads)}",
            "converted": spec.ready(),
            "path": str(spec.native_path()),
        }
