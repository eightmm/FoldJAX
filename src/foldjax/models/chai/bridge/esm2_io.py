"""Native sharded export/loading for Chai's 5.68 GB ESM2 TorchScript asset."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from foldjax.models.chai.bridge.component_io import (
    load_native_component_state_dict,
    save_native_component_state_dict,
)

ESM2_FORMAT = "chai-jax-esm2-bundle"
ESM2_SCHEMA_VERSION = 1
ESM2_SOURCE_SHA256 = "074673b97e1c1ff9c3cf949294749dd446ad1c83cee50a466b4841bb4d89c27a"
ESM2_SOURCE_FILENAME = "traced_sdpa_esm2_t36_3B_UR50D_fp16.pt"
ESM2_NUM_LAYERS = 36
ESM2_HIDDEN_DIM = 2560
ESM2_FFN_DIM = 10240
ESM2_NUM_HEADS = 40
ESM2_HEAD_DIM = 64
_MANIFEST = "manifest.json"

_LAYER_NAME_MAP = {
    "self_attn.q_proj.weight": "q_weight",
    "self_attn.q_proj.bias": "q_bias",
    "self_attn.k_proj.weight": "k_weight",
    "self_attn.k_proj.bias": "k_bias",
    "self_attn.v_proj.weight": "v_weight",
    "self_attn.v_proj.bias": "v_bias",
    "self_attn.out_proj.weight": "out_weight",
    "self_attn.out_proj.bias": "out_bias",
    "self_attn.rot_emb.inv_freq": "rotary_inv_freq",
    "self_attn_layer_norm.weight": "attention_norm_weight",
    "self_attn_layer_norm.bias": "attention_norm_bias",
    "fc1.weight": "fc1_weight",
    "fc1.bias": "fc1_bias",
    "fc2.weight": "fc2_weight",
    "fc2.bias": "fc2_bias",
    "final_layer_norm.weight": "final_norm_weight",
    "final_layer_norm.bias": "final_norm_bias",
}


@dataclass(frozen=True)
class NativeEsm2Bundle:
    manifest: dict[str, Any]
    global_state: dict[str, np.ndarray]
    layers: tuple[dict[str, np.ndarray], ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_shape(name: str) -> tuple[int, ...]:
    shapes = {
        "q_weight": (ESM2_HIDDEN_DIM, ESM2_HIDDEN_DIM),
        "q_bias": (ESM2_HIDDEN_DIM,),
        "k_weight": (ESM2_HIDDEN_DIM, ESM2_HIDDEN_DIM),
        "k_bias": (ESM2_HIDDEN_DIM,),
        "v_weight": (ESM2_HIDDEN_DIM, ESM2_HIDDEN_DIM),
        "v_bias": (ESM2_HIDDEN_DIM,),
        "out_weight": (ESM2_HIDDEN_DIM, ESM2_HIDDEN_DIM),
        "out_bias": (ESM2_HIDDEN_DIM,),
        "rotary_inv_freq": (ESM2_HEAD_DIM // 2,),
        "attention_norm_weight": (ESM2_HIDDEN_DIM,),
        "attention_norm_bias": (ESM2_HIDDEN_DIM,),
        "fc1_weight": (ESM2_FFN_DIM, ESM2_HIDDEN_DIM),
        "fc1_bias": (ESM2_FFN_DIM,),
        "fc2_weight": (ESM2_HIDDEN_DIM, ESM2_FFN_DIM),
        "fc2_bias": (ESM2_HIDDEN_DIM,),
        "final_norm_weight": (ESM2_HIDDEN_DIM,),
        "final_norm_bias": (ESM2_HIDDEN_DIM,),
    }
    return shapes[name]


def _validate_layer(layer: dict[str, np.ndarray], index: int) -> None:
    if set(layer) != set(_LAYER_NAME_MAP.values()):
        raise ValueError(f"ESM2 layer {index} tensor names mismatch")
    for name, value in layer.items():
        if value.shape != _expected_shape(name) or value.dtype != np.float16:
            raise ValueError(f"ESM2 layer {index} tensor mismatch: {name}")


def _validate_global(state: dict[str, np.ndarray]) -> None:
    expected = {
        "embed_tokens_weight": (33, ESM2_HIDDEN_DIM),
        "final_norm_weight": (ESM2_HIDDEN_DIM,),
        "final_norm_bias": (ESM2_HIDDEN_DIM,),
    }
    if set(state) != set(expected):
        raise ValueError("ESM2 global tensor names mismatch")
    if any(
        state[name].shape != shape or state[name].dtype != np.float16
        for name, shape in expected.items()
    ):
        raise ValueError("ESM2 global tensor mismatch")


def load_esm2_torchscript_state(
    path: str | Path,
) -> tuple[dict[str, np.ndarray], tuple[dict[str, np.ndarray], ...]]:
    """Extract only tensors reached by the traced encoder forward."""

    import torch  # lazy bridge dependency

    source = Path(path)
    module = torch.jit.load(str(source), map_location="cpu").eval()
    state = module.state_dict()
    global_state = {
        "embed_tokens_weight": state["embed_tokens.weight"].detach().numpy(),
        "final_norm_weight": state["emb_layer_norm_after.weight"].detach().numpy(),
        "final_norm_bias": state["emb_layer_norm_after.bias"].detach().numpy(),
    }
    layers = []
    for index in range(ESM2_NUM_LAYERS):
        prefix = f"layers.{index}."
        layer = {
            native_name: state[prefix + source_name].detach().numpy()
            for source_name, native_name in _LAYER_NAME_MAP.items()
        }
        _validate_layer(layer, index)
        layers.append(layer)
    _validate_global(global_state)
    return global_state, tuple(layers)


def export_native_esm2(source: str | Path, destination: str | Path) -> dict[str, Any]:
    """Export the pinned official artifact into 37 checksummed native shards."""

    source_path = Path(source)
    if source_path.name != ESM2_SOURCE_FILENAME or not source_path.is_file():
        raise ValueError("source must be the official Chai ESM2 TorchScript file")
    source_sha256 = _sha256_file(source_path)
    if source_sha256 != ESM2_SOURCE_SHA256:
        raise ValueError("official Chai ESM2 source SHA-256 mismatch")
    destination_path = Path(destination)
    if destination_path.exists() or destination_path.is_symlink():
        raise FileExistsError(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            dir=destination_path.parent, prefix=f".{destination_path.name}."
        )
    )
    try:
        global_state, layers = load_esm2_torchscript_state(source_path)
        archives: dict[str, str] = {}
        global_path = temporary / "global.npz"
        save_native_component_state_dict(
            global_state, global_path, component="esm2.global"
        )
        archives[global_path.name] = _sha256_file(global_path)
        for index, layer in enumerate(layers):
            layer_path = temporary / f"layer_{index:02d}.npz"
            save_native_component_state_dict(
                layer, layer_path, component=f"esm2.layer.{index:02d}"
            )
            archives[layer_path.name] = _sha256_file(layer_path)
        manifest: dict[str, Any] = {
            "format": ESM2_FORMAT,
            "schema_version": ESM2_SCHEMA_VERSION,
            "source_filename": ESM2_SOURCE_FILENAME,
            "source_sha256": source_sha256,
            "model": {
                "layers": ESM2_NUM_LAYERS,
                "hidden_dim": ESM2_HIDDEN_DIM,
                "ffn_dim": ESM2_FFN_DIM,
                "heads": ESM2_NUM_HEADS,
                "head_dim": ESM2_HEAD_DIM,
                "dtype": "float16",
            },
            "archives": archives,
        }
        (temporary / _MANIFEST).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination_path)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_native_esm2(path: str | Path) -> NativeEsm2Bundle:
    """Load and validate native ESM2 shards without importing Torch."""

    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("native ESM2 bundle must be a regular directory")
    manifest = read_native_esm2_manifest(root)
    expected_names = {
        "global.npz",
        *(f"layer_{index:02d}.npz" for index in range(ESM2_NUM_LAYERS)),
    }
    archives = manifest["archives"]
    if {entry.name for entry in root.iterdir()} != expected_names | {_MANIFEST}:
        raise ValueError("native ESM2 bundle files mismatch")
    for name, expected_sha256 in archives.items():
        if _sha256_file(root / name) != expected_sha256:
            raise ValueError(f"native ESM2 archive checksum mismatch: {name}")
    global_state = load_native_component_state_dict(
        root / "global.npz", expected_component="esm2.global"
    )
    layers = tuple(
        load_native_component_state_dict(
            root / f"layer_{index:02d}.npz",
            expected_component=f"esm2.layer.{index:02d}",
        )
        for index in range(ESM2_NUM_LAYERS)
    )
    for index, layer in enumerate(layers):
        _validate_layer(layer, index)
    _validate_global(global_state)
    return NativeEsm2Bundle(manifest, global_state, layers)


def read_native_esm2_manifest(path: str | Path) -> dict[str, Any]:
    """Validate lightweight ESM2 provenance without loading weight shards."""

    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("native ESM2 bundle must be a regular directory")
    try:
        manifest = json.loads((root / _MANIFEST).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid native ESM2 manifest") from error
    if (
        manifest.get("format") != ESM2_FORMAT
        or manifest.get("schema_version") != ESM2_SCHEMA_VERSION
        or manifest.get("source_filename") != ESM2_SOURCE_FILENAME
        or manifest.get("source_sha256") != ESM2_SOURCE_SHA256
    ):
        raise ValueError("native ESM2 provenance mismatch")
    expected_model = {
        "layers": ESM2_NUM_LAYERS,
        "hidden_dim": ESM2_HIDDEN_DIM,
        "ffn_dim": ESM2_FFN_DIM,
        "heads": ESM2_NUM_HEADS,
        "head_dim": ESM2_HEAD_DIM,
        "dtype": "float16",
    }
    if manifest.get("model") != expected_model:
        raise ValueError("native ESM2 model configuration mismatch")
    expected_names = {"global.npz", *(
        f"layer_{index:02d}.npz" for index in range(ESM2_NUM_LAYERS)
    )}
    archives = manifest.get("archives")
    if not isinstance(archives, dict) or set(archives) != expected_names:
        raise ValueError("native ESM2 archive manifest mismatch")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in archives.values()
    ):
        raise ValueError("native ESM2 archive digest mismatch")
    return manifest


__all__ = [
    "ESM2_SOURCE_SHA256",
    "NativeEsm2Bundle",
    "export_native_esm2",
    "load_esm2_torchscript_state",
    "load_native_esm2",
    "read_native_esm2_manifest",
]
