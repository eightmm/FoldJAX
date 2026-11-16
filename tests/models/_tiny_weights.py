"""Random weights in each port's released layout, one block deep.

The padding perturbation tests (``tests/models/*/test_padding_perturbation.py``)
need a whole model -- trunk, diffusion and confidence -- whose weights are not
zero and whose repeated stacks are not empty, because a zero weight multiplies
padded garbage into nothing and an empty stack skips exactly the operators the
tests are about. They must also run without torch and without the released
checkpoint, which rules out both the parity fixtures and the store.

What they use instead is a *manifest*: the released checkpoint's tensor names
and shapes, read once with the torch-free archive reader and cut down to block
0 of every ``.blocks.N`` / ``.layers.N`` stack. Each port's state-dict mapper
infers stack depth from the keys it is handed, so the manifest maps to a
one-block model at the released widths. The tests fill it with seeded normal
noise and hand it to the mapper the released weights go through.

Buffers that are not weights -- bin boundaries, the version tensor -- keep
their released values in the manifest, because random boundaries would not be
sorted and the heads that bin on them assume they are.

Regenerate a manifest from a checkpoint (a torch archive, or ESMFold2's
``model.safetensors``) with::

    python -m tests.models._tiny_weights protenix /path/to/protenix.pt
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

#: Tensors whose released values the manifest keeps instead of a shape.
_VALUE_BUFFERS = frozenset({"lower_bins", "upper_bins", "boundaries", "version_tensor"})

#: One stack entry per repeated block. Boltz-2's diffusion conditioning also
#: keeps one `LayerNorm + Linear` pair-bias projection per transformer layer
#: in a parallel list (`proj_z.N.{0,1}.*`), which must be cut with the layers
#: or the bias stacks to `layers x heads` heads. ESMFold2's diffusion token
#: transformer interleaves `attn_blocks.N` and `transition_blocks.N`.
_STACK = re.compile(
    r"\.(layers|blocks|attn_blocks|transition_blocks)\.(\d+)\."
    r"|\.(atom_enc_proj_z|atom_dec_proj_z|token_trans_proj_z)\.(\d+)\.\d+\."
)

MANIFEST_NAMES = {
    "boltz2": "boltz2_tiny_manifest.json",
    "protenix": "protenix_tiny_manifest.json",
    "opendde": "opendde_tiny_manifest.json",
    "openfold3": "openfold3_tiny_manifest.json",
    "esmfold2": "esmfold2_tiny_manifest.json",
}

_SAFETENSORS_DTYPES = {"F32": np.float32, "F16": np.float16}


def manifest_path(model: str) -> Path:
    return Path(__file__).resolve().parent / model / "fixtures" / MANIFEST_NAMES[model]


def _unwrap(checkpoint: Any) -> Mapping[str, Any]:
    for key in ("state_dict", "model"):
        if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get(key), Mapping):
            return checkpoint[key]
    return checkpoint


def _load_safetensors_shapes(path: Path) -> dict[str, np.ndarray]:
    """Shape-only arrays from a safetensors header; kept buffers hold values.

    The header is a JSON map of name to dtype, shape and byte offsets, so the
    weights themselves are never read: a broadcast zero of the right shape
    stands in for each one, and only the value buffers come off the file.
    """

    with path.open("rb") as handle:
        header_size = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_size))
        header.pop("__metadata__", None)
        state: dict[str, np.ndarray] = {}
        for name, entry in header.items():
            shape = tuple(int(size) for size in entry["shape"])
            if name.rsplit(".", 1)[-1] in _VALUE_BUFFERS:
                start, end = entry["data_offsets"]
                handle.seek(8 + header_size + start)
                dtype = _SAFETENSORS_DTYPES[entry["dtype"]]
                raw = handle.read(end - start)
                state[name] = np.frombuffer(raw, dtype).reshape(shape)
            else:
                state[name] = np.broadcast_to(np.float32(0), shape)
    return state


def build_manifest(
    checkpoint_path: str | Path,
    *,
    keep_blocks: int = 1,
    strip_prefix: str = "",
) -> dict[str, Any]:
    """Names, shapes and the few kept buffer values of one checkpoint."""

    if Path(checkpoint_path).suffix == ".safetensors":
        state = _load_safetensors_shapes(Path(checkpoint_path))
    else:
        from foldjax import torch_archive

        state = _unwrap(torch_archive.load(checkpoint_path))
    tensors: dict[str, list[int]] = {}
    values: dict[str, list[float]] = {}
    stacks: dict[str, int] = {}
    for key, value in state.items():
        name = str(key)
        if strip_prefix and name.startswith(strip_prefix):
            name = name[len(strip_prefix) :]
        if not isinstance(value, np.ndarray):
            continue
        kept = True
        for match in _STACK.finditer(name):
            group = 1 if match.group(1) else 3
            stack = name[: match.end(group)]
            index = int(match.group(group + 1))
            stacks[stack] = max(stacks.get(stack, 0), index + 1)
            if index >= keep_blocks:
                kept = False
        if not kept:
            continue
        tensors[name] = [int(size) for size in value.shape]
        if name.rsplit(".", 1)[-1] in _VALUE_BUFFERS:
            values[name] = np.asarray(value, dtype=np.float64).reshape(-1).tolist()
    return {
        "source": Path(checkpoint_path).name,
        "keep_blocks": keep_blocks,
        "released_depth": dict(sorted(stacks.items())),
        "tensors": dict(sorted(tensors.items())),
        "values": values,
    }


def write_manifest(manifest: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["{"]
    for section in ("source", "keep_blocks", "released_depth"):
        lines.append(f'"{section}": {json.dumps(manifest[section])},')
    lines.append('"tensors": {')
    tensors = manifest["tensors"]
    for index, (name, shape) in enumerate(tensors.items()):
        comma = "," if index + 1 < len(tensors) else ""
        lines.append(f'"{name}": {json.dumps(shape)}{comma}')
    lines.append("},")
    lines.append(f'"values": {json.dumps(manifest["values"])}')
    lines.append("}")
    path.write_text("\n".join(lines) + "\n")


def load_manifest(model: str) -> dict[str, Any]:
    with manifest_path(model).open() as handle:
        return json.load(handle)


def random_state_dict(
    manifest: Mapping[str, Any],
    *,
    seed: int = 0,
    scale: float = 1.0,
    dtype: Any = np.float32,
) -> dict[str, np.ndarray]:
    """Seeded normal noise in the manifest's layout.

    Linear weights are scaled by ``scale / sqrt(fan_in)`` so a one-block model
    keeps activations of order one; one-dimensional weights (layer norms) sit
    around one, biases around zero. The distribution is irrelevant to what the
    perturbation tests check as long as nothing is zero.
    """

    rng = np.random.default_rng(seed)
    values = manifest.get("values", {})
    state: dict[str, np.ndarray] = {}
    for name, shape in manifest["tensors"].items():
        shape = tuple(int(size) for size in shape)
        if name in values:
            array = np.asarray(values[name], dtype=np.float64).reshape(shape)
        elif name.endswith(".bias"):
            array = 0.1 * rng.standard_normal(shape)
        elif len(shape) <= 1:
            array = 1.0 + 0.1 * rng.standard_normal(shape)
        else:
            fan_in = shape[-1] if len(shape) == 2 else int(np.prod(shape[1:]))
            array = rng.standard_normal(shape) * (scale / np.sqrt(max(fan_in, 1)))
        state[name] = np.asarray(array, dtype=dtype)
    return state


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] not in MANIFEST_NAMES:
        models = "|".join(MANIFEST_NAMES)
        raise SystemExit(
            f"usage: python -m tests.models._tiny_weights {{{models}}} CHECKPOINT"
        )
    model, checkpoint = args
    strip = "module." if model in {"protenix", "opendde"} else ""
    manifest = build_manifest(checkpoint, strip_prefix=strip)
    write_manifest(manifest, manifest_path(model))
    print(
        f"{model}: {len(manifest['tensors'])} tensors from {manifest['source']} "
        f"-> {manifest_path(model)}"
    )


if __name__ == "__main__":
    main()
