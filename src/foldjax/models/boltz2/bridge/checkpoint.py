"""Inspect published Boltz checkpoints through FoldJAX's NumPy reader."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def _load_checkpoint(path: Path) -> dict[str, Any]:
    from foldjax import torch_archive

    checkpoint = torch_archive.load(path)
    if not isinstance(checkpoint, dict):
        msg = f"Expected checkpoint dict, got {type(checkpoint).__name__}"
        raise SystemExit(msg)
    return checkpoint


def checkpoint_state_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Return the model state dict from a Lightning or plain checkpoint."""

    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        msg = f"Expected state dict, got {type(state).__name__}"
        raise SystemExit(msg)
    return state


def load_checkpoint_state_dict(path: Path) -> dict[str, Any]:
    """Load a checkpoint state dict as NumPy-backed values."""

    return checkpoint_state_dict(_load_checkpoint(path))


def describe_checkpoint(path: Path, limit: int) -> list[str]:
    """Return compact checkpoint key/shape lines."""

    checkpoint = _load_checkpoint(path)
    state = checkpoint_state_dict(checkpoint)
    lines = [
        f"checkpoint: {path}",
        f"top_level_keys: {sorted(checkpoint.keys())}",
        f"state_tensors: {len(state)}",
    ]
    for index, (key, value) in enumerate(state.items()):
        if index >= limit:
            lines.append(f"... {len(state) - limit} more tensors")
            break
        shape = tuple(value.shape) if hasattr(value, "shape") else "-"
        dtype = getattr(value, "dtype", "-")
        lines.append(f"{key}: shape={shape} dtype={dtype}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    for line in describe_checkpoint(args.checkpoint, args.limit):
        print(line)


if __name__ == "__main__":
    main()
