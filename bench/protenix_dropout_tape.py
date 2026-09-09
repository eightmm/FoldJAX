"""Observe native CUDA dropout masks without replacing its RNG or arithmetic."""

from contextlib import contextmanager


def load_dropout_tape(root, completion, config):
    """Validate the optional native branch before forwarding stochastic inputs."""
    import numpy as np

    applied = completion.get("mc_dropout_applied")
    path = root / "dropout-tape.npz"
    if type(applied) is not bool:
        raise ValueError("missing MC dropout decision")
    if applied or "mc_dropout_apply_rate" in config:
        draws = completion.get("mc_dropout_random_draws")
        probability = config.get("mc_dropout_apply_rate")
        if (
            not isinstance(draws, list)
            or len(draws) != 1
            or type(draws[0]) not in (int, float)
            or not 0 <= draws[0] < 1
            or type(probability) not in (int, float)
            or not 0 <= probability <= 1
            or (draws[0] < probability) != applied
        ):
            raise ValueError("native dropout decision contradicts draw/config")
    if not applied:
        if (
            path.exists()
            or completion.get("mc_dropout_mask_calls", 0) != 0
            or completion.get("mc_dropout_rate") is not None
        ):
            raise ValueError("unexpected dropout tape on disabled branch")
        return None, None
    rate = completion.get("mc_dropout_rate")
    if (
        completion.get("mc_dropout_mask_calls") != 10
        or not isinstance(rate, (float, int))
        or isinstance(rate, bool)
        or not 0 < rate < 1
        or rate != config.get("mc_dropout_rate")
    ):
        raise ValueError("invalid native dropout rate/count")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"keep_masks"}:
            raise ValueError("unexpected dropout tape keys")
        masks = archive["keep_masks"]
    if (
        masks.dtype != np.bool_
        or masks.ndim != 4
        or masks.shape[0] != 10
        or masks.shape[1] != masks.shape[2]
        or 0 in masks.shape
    ):
        raise ValueError("invalid native dropout mask shape/dtype")
    return masks, float(rate)


@contextmanager
def capture_native_dropout(*, expected_calls, rate):
    """Yield actual masks; fail closed on another operator/schema or call count.

    Scope this around the native recycling branch, not the entire model.
    Torch is a benchmark-only lazy dependency, never an installable dependency.
    CPU functional dropout decomposes differently and is intentionally rejected.
    """
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    if expected_calls < 1 or not 0 < rate < 1:
        raise ValueError("dropout tape requires positive calls and rate in (0, 1)")
    masks = []

    class Observer(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if func == torch.ops.aten.native_dropout.default:
                if len(args) != 3 or args[1] != rate or args[2] is not True:
                    raise ValueError("unexpected native dropout probability/training")
                output, mask = result
                if (
                    mask.dtype != torch.bool
                    or mask.shape != args[0].shape
                    or output.shape != mask.shape
                    or not mask.is_cuda
                ):
                    raise ValueError("unsupported native dropout mask schema")
                masks.append(mask.detach().cpu().numpy().copy())
                if len(masks) > expected_calls:
                    raise ValueError("extra native dropout event")
            return result

    with Observer():
        yield masks
    if len(masks) != expected_calls:
        raise ValueError("missing native CUDA dropout events")
