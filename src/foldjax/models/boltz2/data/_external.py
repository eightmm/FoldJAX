"""Self-contained copies of the few functions the featurization core needs
from outside ``boltz.data``.

These are copied from the original Boltz source so the
``foldjax.models.boltz2.data`` package has no ``import boltz`` dependency:

- ``center_random_augmentation`` / ``randomly_rotate`` (+ rotation helpers)
                               from ``boltz.model.modules.utils``

One deliberate change: the random draws come from a caller-supplied NumPy
generator instead of torch's global RNG. Upstream's global RNG is reproducible
only because its predict path calls ``seed_everything(seed)``; the NumPy tensor
layer had no such seed, so every featurization drew a different ``ref_pos``.
An explicit generator makes the augmentation a function of the job seed, and
both tensor backends consume the same normals.
"""

from __future__ import annotations

import numpy as np

from foldjax.models.boltz2.data._torch import from_numpy, torch

Device = torch.types.Device


# ---------------------------------------------------------------------------
# from boltz.model.modules.utils (rotation helpers + augmentation)
# ---------------------------------------------------------------------------
def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _standard_normal(
    rng: np.random.Generator, shape: tuple[int, ...], dtype: torch.dtype | None
) -> torch.Tensor:
    draw = from_numpy(rng.standard_normal(shape))
    return draw if dtype is None else draw.to(dtype=dtype)


def random_quaternions(
    n: int,
    rng: np.random.Generator,
    dtype: torch.dtype | None = None,
    device: Device | None = None,
) -> torch.Tensor:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        rng: Generator the normals are drawn from.
        dtype: Type to return.
        device: Unused; kept for upstream's signature. Draws are host-side.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    o = _standard_normal(rng, (n, 4), dtype)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o


def random_rotations(
    n: int,
    rng: np.random.Generator,
    dtype: torch.dtype | None = None,
    device: Device | None = None,
) -> torch.Tensor:
    """
    Generate random rotations as 3x3 rotation matrices.

    Args:
        n: Number of rotation matrices in a batch to return.
        rng: Generator the normals are drawn from.
        dtype: Type to return.
        device: Unused; kept for upstream's signature.

    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """
    quaternions = random_quaternions(n, rng, dtype=dtype, device=device)
    return quaternion_to_matrix(quaternions)


def randomly_rotate(coords, rng, return_second_coords=False, second_coords=None):
    R = random_rotations(len(coords), rng, coords.dtype, coords.device)

    if return_second_coords:
        return torch.einsum("bmd,bds->bms", coords, R), torch.einsum(
            "bmd,bds->bms", second_coords, R
        ) if second_coords is not None else None

    return torch.einsum("bmd,bds->bms", coords, R)


def center_random_augmentation(
    atom_coords,
    atom_mask,
    *,
    rng: np.random.Generator,
    s_trans=1.0,
    augmentation=True,
    centering=True,
    return_second_coords=False,
    second_coords=None,
):
    """Algorithm 19"""
    if centering:
        atom_mean = torch.sum(
            atom_coords * atom_mask[:, :, None], dim=1, keepdim=True
        ) / torch.sum(atom_mask[:, :, None], dim=1, keepdim=True)
        atom_coords = atom_coords - atom_mean

        if second_coords is not None:
            # apply same transformation also to this input
            second_coords = second_coords - atom_mean

    if augmentation:
        atom_coords, second_coords = randomly_rotate(
            atom_coords, rng, return_second_coords=True, second_coords=second_coords
        )
        trans_shape = tuple(atom_coords[:, 0:1, :].shape)
        random_trans = _standard_normal(rng, trans_shape, atom_coords.dtype) * s_trans
        atom_coords = atom_coords + random_trans

        if second_coords is not None:
            second_coords = second_coords + random_trans

    if return_second_coords:
        return atom_coords, second_coords

    return atom_coords
