"""Load Chai-1 TorchScript components and extract named float tensors.

Chai-1 ships every model component (``feature_embedding.pt``,
``bond_loss_input_proj.pt``, ``token_embedder.pt``, ``trunk.pt``,
``diffusion_module.pt``, ``confidence_head.pt``) as a TorchScript archive loaded
with ``torch.jit.load``. Despite being scripted, calling ``state_dict()`` on the
loaded ``RecursiveScriptModule`` yields a flat dict of named float tensors with
stable parameter names and shapes -- this is what makes a weight-compatible JAX
port feasible (verified Phase 1).

This module is the ONLY entry point that needs torch; it imports it lazily so
the native JAX runtime never pulls torch in.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

CHAI_COMPONENTS = (
    "feature_embedding.pt",
    "bond_loss_input_proj.pt",
    "token_embedder.pt",
    "trunk.pt",
    "diffusion_module.pt",
    "confidence_head.pt",
)

NATIVE_FORMAT = "chai-jax-component"
NATIVE_FORMAT_VERSION = 1
NATIVE_MANIFEST_KEY = "__chai_jax_manifest__"


def load_component_state_dict(path: str | Path) -> dict[str, np.ndarray]:
    """Load a Chai TorchScript component and return its state_dict as numpy.

    Returns a ``{name: np.ndarray}`` dict. Integer buffers (e.g. embedding
    offsets) are preserved with their original dtype.
    """
    import torch  # lazy, torch-bridge extra only

    module = torch.jit.load(str(path), map_location="cpu").eval()
    return {k: v.detach().cpu().numpy() for k, v in module.state_dict().items()}


def summarize_state_dict(state: dict[str, np.ndarray]) -> list[tuple[str, tuple, str]]:
    """Return ``(name, shape, dtype)`` rows for inspection/reporting."""
    return [(k, tuple(v.shape), str(v.dtype)) for k, v in state.items()]


def _tensor_metadata(value: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(value)
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "sha256": hashlib.sha256(memoryview(contiguous)).hexdigest(),
    }


def save_native_component_state_dict(
    state: Mapping[str, np.ndarray],
    path: str | Path,
    *,
    component: str,
) -> None:
    """Save a Torch-free, checksummed Chai-JAX component archive."""
    if NATIVE_MANIFEST_KEY in state:
        raise ValueError(f"reserved tensor name: {NATIVE_MANIFEST_KEY}")
    arrays = {name: np.asarray(value) for name, value in state.items()}
    manifest = {
        "format": NATIVE_FORMAT,
        "version": NATIVE_FORMAT_VERSION,
        "component": component,
        "tensors": {name: _tensor_metadata(value) for name, value in arrays.items()},
    }
    manifest_bytes = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as output:
        temporary = Path(output.name)
        np.savez(
            output,
            **{
                NATIVE_MANIFEST_KEY: np.frombuffer(manifest_bytes, dtype=np.uint8),
                **arrays,
            },
        )
    try:
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_native_component_state_dict(
    path: str | Path,
    *,
    expected_component: str | None = None,
) -> dict[str, np.ndarray]:
    """Load and validate a native component archive without importing torch."""
    with np.load(path, allow_pickle=False) as archive:
        if NATIVE_MANIFEST_KEY not in archive.files:
            raise ValueError("native component manifest is missing")
        manifest = json.loads(archive[NATIVE_MANIFEST_KEY].tobytes().decode("utf-8"))
        if manifest.get("format") != NATIVE_FORMAT:
            raise ValueError("invalid native component format")
        if manifest.get("version") != NATIVE_FORMAT_VERSION:
            raise ValueError(
                f"unsupported native component version: {manifest.get('version')}"
            )
        component = manifest.get("component")
        if expected_component is not None and component != expected_component:
            raise ValueError(
                "component mismatch: "
                f"expected {expected_component!r}, got {component!r}"
            )
        tensor_metadata = manifest.get("tensors")
        if not isinstance(tensor_metadata, dict):
            raise ValueError("invalid native component tensor manifest")
        actual_names = set(archive.files) - {NATIVE_MANIFEST_KEY}
        if actual_names != set(tensor_metadata):
            raise ValueError("native component tensor names mismatch")

        state: dict[str, np.ndarray] = {}
        for name, metadata in tensor_metadata.items():
            value = np.array(archive[name], copy=True)
            expected_shape = tuple(metadata["shape"])
            expected_dtype = metadata["dtype"]
            if value.shape != expected_shape or value.dtype.str != expected_dtype:
                raise ValueError(f"tensor metadata mismatch: {name}")
            if _tensor_metadata(value)["sha256"] != metadata["sha256"]:
                raise ValueError(f"tensor checksum mismatch: {name}")
            state[name] = value
    return state


def convert_component_to_native(
    source: str | Path, destination: str | Path
) -> int:
    """Convert one Chai TorchScript component to the native JAX archive."""
    source_path = Path(source)
    state = load_component_state_dict(source_path)
    save_native_component_state_dict(
        state, destination, component=source_path.name
    )
    return len(state)
