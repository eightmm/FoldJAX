"""Static feature and inference output IO helpers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


def load_static_feature_npz(path: str | Path) -> dict[str, Any]:
    """Load a static Protenix feature dictionary from an ``.npz`` file."""

    features: dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            value = data[key]
            parts = key.split(".")
            target = features
            for part in parts[:-1]:
                child = target.setdefault(part, {})
                if not isinstance(child, dict):
                    raise ValueError(f"conflicting static feature key: {key}")
                target = child
            target[parts[-1]] = value
    return features


def save_static_feature_npz(
    path: str | Path,
    features: dict[str, Any],
) -> None:
    """Save a static Protenix feature dictionary as an ``.npz`` file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat: dict[str, np.ndarray] = {}

    def add_values(prefix: str, values: dict[str, Any]) -> None:
        for key, value in values.items():
            if "." in key:
                raise ValueError("static feature keys cannot contain '.'")
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                add_values(full_key, value)
            else:
                flat[full_key] = np.asarray(value)

    add_values("", features)
    np.savez_compressed(path, **flat)


#: Size above which :func:`flatten_output_dict` leaves an array out. Confidence
#: outputs are quadratic in token count and several carry a bin axis on top, so at
#: 2030 tokens one of them is 37 GiB -- larger than the model's own peak, and enough
#: to kill a prediction that had already finished. Copying it to host also costs a
#: device-side staging buffer, so the failure lands *after* all the compute.
DEFAULT_OUTPUT_ARRAY_BUDGET_BYTES = 4 * 2**30


def _output_nbytes(value: Any) -> int | None:
    """Bytes ``value`` would occupy once materialized, or ``None`` if unknown.

    Sequences are summed rather than skipped, because a *list* of per-sample arrays
    has no ``nbytes`` of its own and would otherwise pass an attribute-only check;
    ``np.asarray`` on a list of JAX arrays stacks them on the device first.

    This is necessary and **not sufficient**. A size that fits does not mean the
    transfer fits: moving a JAX array to host may run a deferred computation and
    needs a device-side staging buffer. A 2030-token target OOM'd here asking for
    37 GiB on an array whose own size was under the budget, so
    :func:`flatten_output_dict` also catches the failure rather than relying on this
    number to predict it.
    """
    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    if isinstance(value, (list, tuple)):
        total = 0
        for item in value:
            item_bytes = _output_nbytes(item)
            if item_bytes is None:
                return None
            total += item_bytes
        return total
    return None


def flatten_output_dict(
    output: dict[str, Any],
    *,
    include_trunk: bool = False,
    max_bytes: int | None = DEFAULT_OUTPUT_ARRAY_BUDGET_BYTES,
    report: Callable[[str], None] | None = None,
) -> tuple[dict[str, np.ndarray], tuple[tuple[str, int], ...]]:
    """Convert JAX/NumPy output values into a flat NumPy mapping.

    Values over ``max_bytes`` are omitted rather than transferred; ``None`` keeps
    everything. The size comes from array metadata, so an oversized output is never
    copied to host just to find out how big it is.

    Returns:
        ``(arrays, omitted)`` where ``omitted`` is ``(name, nbytes)`` pairs.
    """

    flat: dict[str, np.ndarray] = {}
    omitted: list[tuple[str, int]] = []
    skip = set() if include_trunk else {"s_inputs", "s_trunk", "z_trunk"}

    def add(name: str, value: Any) -> None:
        nbytes = _output_nbytes(value)
        if report is not None:
            shape = getattr(value, "shape", None)
            size = "?" if nbytes is None else f"{nbytes / 2**30:.2f} GiB"
            report(f"    {name}: {type(value).__name__} shape={shape} {size}")
        if max_bytes is not None and nbytes is not None and nbytes > max_bytes:
            omitted.append((name, int(nbytes)))
            return
        try:
            flat[name] = np.asarray(value)
        except Exception as error:  # noqa: BLE001 - see below
            # A size check on the value is not sufficient, and finding that out the
            # hard way is why this exists. Moving a JAX array to host can run a
            # deferred computation and needs a device-side staging buffer, so the
            # transfer can exhaust memory even when the *result* is well under
            # ``max_bytes``. Losing one array is bad; losing a finished prediction
            # that took ten minutes because of the file it was being written to is
            # worse, so the failure is recorded and the rest of the outputs survive.
            if "RESOURCE_EXHAUSTED" not in str(error) and not isinstance(
                error, MemoryError
            ):
                raise
            omitted.append((name, int(nbytes) if nbytes is not None else -1))

    for key, value in output.items():
        if key in skip:
            continue
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                add(f"{key}.{sub_key}", sub_value)
        else:
            add(key, value)
    return flat, tuple(omitted)


def save_output_npz(
    path: str | Path,
    output: dict[str, Any],
    *,
    include_trunk: bool = False,
    max_bytes: int | None = DEFAULT_OUTPUT_ARRAY_BUDGET_BYTES,
    report: Callable[[str], None] | None = None,
) -> tuple[tuple[str, int], ...]:
    """Save inference output arrays to compressed ``.npz``.

    Returns the ``(name, nbytes)`` pairs left out by ``max_bytes`` so the caller can
    report them; silently dropping an output would be worse than the crash it avoids.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_output, omitted = flatten_output_dict(
        output, include_trunk=include_trunk, max_bytes=max_bytes, report=report
    )
    np.savez_compressed(path, **flat_output)
    return omitted
