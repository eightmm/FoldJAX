"""Checkpoint loading and inspection.

Reads an OpenFold3 checkpoint into a flat ``{key: array}`` mapping that the
mappers in ``torch_mapping`` consume. Loading is deliberately separated from
mapping so a checkpoint can be *inspected* before any parameter layout is
assumed — which is the open risk in this port: the released weights' top-level
prefixes, real dimensions, and whether the triangular multiplication is stored
fused or unfused are all unverified.

``safetensors`` is preferred when available because it needs no torch. A
``.pt``/``.ckpt`` file falls back to ``torch.load``, which means the torch-bridge
extra; production inference should convert once to safetensors rather than
depending on that path.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

# Nesting keys that upstream checkpoints commonly wrap the parameters in.
_STATE_KEYS = ("state_dict", "model", "module", "params", "ema_state_dict")
# Prefixes added by wrappers rather than by the model itself.
_STRIP_PREFIXES = ("module.", "model.", "_orig_mod.", "ema.")


def strip_wrapper_prefixes(
    key: str, prefixes: tuple[str, ...] = _STRIP_PREFIXES
) -> str:
    """Remove DDP/compile/EMA wrapper prefixes, repeatedly."""
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def unwrap_state_dict(payload: Any) -> Mapping[str, Any]:
    """Find the parameter mapping inside a loaded checkpoint payload.

    Checkpoints nest the weights under varying keys (``state_dict``, ``model``,
    ...). This walks one level of those before giving up, and raises rather than
    guessing if nothing looks like a parameter mapping.
    """
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint payload is not a mapping: {type(payload)!r}")

    if any(isinstance(key, str) and "." in key for key in payload):
        return payload

    for candidate in _STATE_KEYS:
        if candidate in payload and isinstance(payload[candidate], Mapping):
            return payload[candidate]

    raise KeyError(
        "could not find a parameter mapping in the checkpoint; "
        f"top-level keys: {sorted(str(k) for k in payload)[:20]}"
    )


def load_checkpoint(path: str | Path) -> dict[str, np.ndarray]:
    """Load a checkpoint into a flat ``{key: numpy array}`` mapping.

    Wrapper prefixes are stripped so the keys match what the mappers expect.
    """
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.numpy import load_file

        raw: Mapping[str, Any] = load_file(str(path))
    else:
        try:
            import torch
        except ImportError:
            from foldjax import torch_archive

            loaded = torch_archive.load(path)
        else:
            loaded = torch.load(str(path), map_location="cpu", weights_only=False)
        raw = unwrap_state_dict(loaded)

    flat: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        name = strip_wrapper_prefixes(str(key))
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        flat[name] = np.asarray(value)
    return flat


def describe(state: Mapping[str, np.ndarray], *, depth: int = 1) -> dict[str, int]:
    """Return a ``{prefix: tensor count}`` summary at the given nesting depth.

    Use this to discover a checkpoint's top-level structure without assuming it.
    """
    counts: dict[str, int] = {}
    for key in state:
        prefix = ".".join(key.split(".")[:depth])
        counts[prefix] = counts.get(prefix, 0) + 1
    return dict(sorted(counts.items()))


def iter_shapes(
    state: Mapping[str, np.ndarray], pattern: str = ""
) -> Iterator[tuple[str, tuple[int, ...]]]:
    """Yield ``(key, shape)`` for keys containing ``pattern``, in sorted order."""
    for key in sorted(state):
        if pattern in key:
            yield key, tuple(np.shape(state[key]))


def detect_fused_tri_mul(state: Mapping[str, np.ndarray]) -> bool | None:
    """Report whether triangular multiplication is stored fused.

    Returns ``True`` for the fused layout (``linear_ab_p``/``linear_ab_g``),
    ``False`` for the unfused one (``linear_a_p``/``linear_b_p``), and ``None`` if
    neither appears. This is the port's single largest unverified assumption: the
    mappers implement the unfused layout only.
    """
    fused = any("linear_ab_p" in key or "linear_ab_g" in key for key in state)
    unfused = any("linear_a_p" in key or "linear_b_p" in key for key in state)
    if fused and not unfused:
        return True
    if unfused and not fused:
        return False
    return None
