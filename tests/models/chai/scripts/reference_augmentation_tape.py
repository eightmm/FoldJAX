"""Replay deterministic non-standard-residue rigid augmentations in parity runs.

This is a benchmark-only control. Production Torch and JAX inference retain their
native random-number generators; the tape removes that deliberate RNG boundary
when comparing framework-native prepared features and trunk outputs.
"""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np


def make_tape(seed: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``count`` proper rotations and translations from one NumPy stream."""
    if count < 1:
        raise ValueError("reference augmentation count must be positive")
    rng = np.random.default_rng(seed)
    quaternion = rng.normal(size=(count, 4)).astype(np.float32)
    quaternion /= np.maximum(
        np.linalg.norm(quaternion, axis=-1, keepdims=True), np.float32(1e-8)
    )
    quaternion *= np.where(quaternion[:, :1] < 0, -1.0, 1.0).astype(np.float32)
    real, i, j, k = np.moveaxis(quaternion, -1, 0)
    rotations = (
        np.stack(
            (
                1 - 2 * (j * j + k * k),
                2 * (i * j - k * real),
                2 * (i * k + j * real),
                2 * (i * j + k * real),
                1 - 2 * (i * i + k * k),
                2 * (j * k - i * real),
                2 * (i * k - j * real),
                2 * (j * k + i * real),
                1 - 2 * (i * i + j * j),
            ),
            axis=-1,
        )
        .reshape(count, 3, 3)
        .astype(np.float32)
    )
    translations = rng.normal(size=(count, 3)).astype(np.float32)
    return rotations, translations


def save_tape(path: Path, *, seed: int, count: int) -> None:
    rotations, translations = make_tape(seed, count)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, rotations=rotations, translations=translations)


class ReferenceAugmentationTape:
    def __init__(self, path: str | Path) -> None:
        with np.load(path, allow_pickle=False) as archive:
            self.rotations = np.asarray(archive["rotations"], np.float32)
            self.translations = np.asarray(archive["translations"], np.float32)
        if self.rotations.shape != (len(self.translations), 3, 3):
            raise ValueError("invalid reference augmentation rotation shape")
        if self.translations.shape != (len(self.rotations), 3):
            raise ValueError("invalid reference augmentation translation shape")
        self.index = 0

    def transform(self, position: np.ndarray) -> np.ndarray:
        if self.index >= len(self.rotations):
            raise ValueError("reference augmentation tape is exhausted")
        value = np.asarray(position, np.float32)
        centered = value - value.mean(axis=0, keepdims=True)
        result = (
            centered @ self.rotations[self.index].T
            + self.translations[self.index][None]
        ).astype(np.float32)
        self.index += 1
        return result

    def assert_exhausted(self) -> None:
        if self.index != len(self.rotations):
            raise ValueError(
                "reference augmentation tape consumption mismatch: "
                f"used {self.index}, available {len(self.rotations)}"
            )


@contextlib.contextmanager
def replay_jax(path: str | Path | None) -> Iterator[None]:
    if path is None:
        yield
        return
    import foldjax.models.chai.data.structure as structure

    tape = ReferenceAugmentationTape(path)
    original = structure._random_rigid_augment

    def augment(conformer: object, rng: object) -> object:
        del rng
        return conformer._replace(position=tape.transform(conformer.position))

    structure._random_rigid_augment = augment
    try:
        yield
        tape.assert_exhausted()
    finally:
        structure._random_rigid_augment = original


@contextlib.contextmanager
def replay_torch(path: str | Path | None) -> Iterator[None]:
    if path is None:
        yield
        return
    import chai_lab.data.parsing.structure.residue as residue
    import torch

    tape = ReferenceAugmentationTape(path)
    original = residue.center_random_augmentation

    def augment(
        atom_coords: object, atom_single_mask: object, **kwargs: object
    ) -> object:
        del atom_single_mask, kwargs
        value = atom_coords.detach().cpu().numpy()
        if value.shape[0] != 1:
            raise ValueError("reference augmentation replay supports batch size 1")
        transformed = tape.transform(value[0])
        return torch.from_numpy(transformed[None]).to(atom_coords.device)

    residue.center_random_augmentation = augment
    try:
        yield
        tape.assert_exhausted()
    finally:
        residue.center_random_augmentation = original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    save_tape(args.output, seed=args.seed, count=args.count)


if __name__ == "__main__":
    main()
