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
import shutil
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from foldjax.paths import assets_dir, downloads_dir, weights_dir

_CHUNK = 1 << 20


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
    # single-file model repeats `native`; Chai needs a bundle and an archive.
    requires: tuple[str, ...]
    convert: Callable[[str, Path], Path]
    needs_torch: bool = True
    notes: str = ""
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


def _convert_chai(model: str, source: Path) -> Path:
    from foldjax.models.chai.bridge.bundle_io import (
        COMPONENT_TENSOR_COUNTS,
        export_native_bundle,
    )
    from foldjax.models.chai.bridge.conformer_export import export_native_conformers

    out_dir = weights_dir(model)
    bundle = out_dir / "models" / "chai1"
    if not bundle.exists():
        export_native_bundle(
            {
                component: source / "models_v2" / f"{component}.pt"
                for component in COMPONENT_TENSOR_COUNTS
            },
            bundle,
            chai_source=CHAI_COMPONENT_SOURCE,
            chai_release=CHAI_RELEASE,
        )
    conformers = out_dir / "conformers.npz"
    if not conformers.exists():
        export_native_conformers(source / "conformers_v1.apkl", conformers)
    return conformers


CHAI_COMPONENT_SOURCE = "https://github.com/chaidiscovery/chai-lab"
CHAI_RELEASE = "v0.6.1"
_CHAI_COMPONENTS = (
    "bond_loss_input_proj.pt",
    "confidence_head.pt",
    "diffusion_module.pt",
    "feature_embedding.pt",
    "token_embedder.pt",
    "trunk.pt",
)

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
        ),
        native="protenix_base_default_v1.0.0.jax",
        requires=("protenix_base_default_v1.0.0.jax",),
        convert=_convert_protenix,
        notes="CCD assets are shared with OpenDDE and only needed for ligands "
        "and modified residues.",
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
        ),
        native="opendde.jax",
        requires=("opendde.jax",),
        convert=_convert_opendde,
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
    "chai": ModelAssets(
        model="chai",
        source=CHAI_COMPONENT_SOURCE,
        licence="Apache-2.0",
        downloads=(
            Download(
                name="conformers_v1.apkl",
                url=(
                    "https://chaiassets.com/chai1-inference-depencencies/"
                    "conformers_v1.apkl"
                ),
            ),
            *(
                Download(
                    name=f"models_v2/{component}",
                    url=(
                        "https://chaiassets.com/chai1-inference-depencencies/"
                        f"models_v2/{component}"
                    ),
                )
                for component in _CHAI_COMPONENTS
            ),
        ),
        # Chai takes an asset root: a native model bundle plus a conformer
        # archive, which is why this one is a directory.
        native=".",
        requires=("models/chai1", "conformers.npz"),
        convert=_convert_chai,
        notes="The conformer archive is decoded directly, so this needs no "
        "chai-lab checkout.",
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
    """Download one file unless a verified copy is already present."""
    path = item.target(model)
    if _verified(path, item):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")

    with urllib.request.urlopen(item.url) as response:  # noqa: S310 - pinned https
        total = int(response.headers.get("Content-Length") or 0) or item.size
        done = 0
        with open(partial, "wb") as handle:
            while chunk := response.read(_CHUNK):
                handle.write(chunk)
                done += len(chunk)
                if on_progress is not None:
                    on_progress(item.name, done, total)

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


def fetch(model: str, *, on_progress=None, convert: bool = True) -> Path:
    """Download and convert one model's released files.

    Returns the artifact prediction loads. Re-running is cheap: verified
    downloads and an existing conversion are both skipped.
    """
    spec = assets_for(model)
    if convert and spec.ready():
        return spec.native_path()

    for item in spec.downloads:
        download(item, spec.model, on_progress=on_progress)
    if not convert:
        return downloads_dir(spec.model)

    spec.convert(spec.model, downloads_dir(spec.model))
    if not spec.ready():
        missing = [
            item
            for item in spec.requires
            if not (weights_dir(spec.model) / item).exists()
        ]
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
