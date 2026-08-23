"""JAX-free resolution of one native Boltz-2 checkpoint bundle."""

from __future__ import annotations

from pathlib import Path


def native_weight_candidates(path: str | Path) -> tuple[Path, ...]:
    """Return candidates in the exact order used by the native loader."""

    path = Path(path)
    if path.suffix in {".safetensors", ".npz"}:
        return (path,)
    return (path.with_suffix(".safetensors"), path.with_suffix(".npz"))


def resolve_native_weight_bundle(
    path: str | Path,
) -> tuple[Path, Path] | None:
    """Return the selected weight file and its scalar sidecar path.

    The sidecar is returned even when absent because absence is an observable
    loader state: creating it later changes the parameter tree.
    """

    weights = next(
        (
            candidate
            for candidate in native_weight_candidates(path)
            if candidate.is_file()
        ),
        None,
    )
    if weights is None:
        return None
    return weights, weights.with_suffix(weights.suffix + ".json")
