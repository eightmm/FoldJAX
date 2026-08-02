"""Pure JAX Transition forward pass."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from foldjax.models.boltz2.models.primitives._common import layer_norm as _layer_norm
from foldjax.models.boltz2.models.primitives.glu_backend import gated_linear_unit

TransitionParams = Mapping[str, Mapping[str, jnp.ndarray]]


# The transition widens its input before narrowing it again, and the widened
# form is the largest buffer in Boltz-2's graph: the MSA transition alone is
# f32[1, 1024, 490, 512] -- 980 MiB of a 2,072 MiB arena on a 490-residue job,
# because that one call site was the only transition not given a row chunk.
#
# Deriving the block from the tensor rather than from the call site fixes every
# caller at once, and it is exact: each row of axis 1 is independent, so no
# reduction is split.
_WIDE_BUDGET_BYTES = 256 * 1024**2


def _auto_row_chunk(x: jnp.ndarray, params: TransitionParams) -> int | None:
    """Rows of axis 1 whose widened form fits the budget, or None to do it whole."""
    if x.ndim != 4 or x.shape[1] < 2:
        return None
    wide = params["fc1"]["kernel"].shape[-1] + params["fc2"]["kernel"].shape[-1]
    per_row = x.shape[2] * wide * x.dtype.itemsize
    if per_row <= 0 or per_row * x.shape[1] <= _WIDE_BUDGET_BYTES:
        return None
    return max(1, _WIDE_BUDGET_BYTES // per_row)


def transition_forward(
    params: TransitionParams,
    x: jnp.ndarray,
    chunk_size: int | None = None,
    eps: float = 1e-5,
    row_chunk_size: int | None = None,
    glu_backend: str = "xla",
) -> jnp.ndarray:
    """Run a Boltz Transition block using mapped PyTorch parameters.

    ``chunk_size`` chunks the hidden (SwiGLU) dimension; the fc2 accumulation
    keeps it bit-exact. ``row_chunk_size`` is an orthogonal outer-row chunk for
    any rank-4 input ``[B, N, ..., C]`` -- the pair tensor and, just as
    importantly, the MSA: it splits axis 1 into independent blocks and
    concatenates. No reduction is split, so it is mathematically exact, though
    not bit-identical under the default matmul precision, where XLA tiles the
    smaller GEMM differently: measured at 1.2e-4 relative, and 2.6e-7 under
    ``float32`` precision. Left as ``None`` it is chosen from the size of the
    widened form; pass ``0`` to force the single-op / hidden-chunk path.

    ``glu_backend="tokamax"`` runs the swish GLU through the fused Triton kernel
    (GPU, low precision); ``"xla"`` (default) keeps the bit-exact split-matmul.
    """

    if row_chunk_size is None:
        row_chunk_size = _auto_row_chunk(x, params)
    if (
        row_chunk_size is not None
        and row_chunk_size > 0
        and x.ndim == 4
        and x.shape[1] > row_chunk_size
    ):
        n = x.shape[1]
        blocks = []
        for start in range(0, n, row_chunk_size):
            stop = min(start + row_chunk_size, n)
            blocks.append(
                transition_forward(
                    params,
                    x[:, start:stop],
                    chunk_size=chunk_size,
                    eps=eps,
                    # Already split; 0 stops the slice from splitting again.
                    row_chunk_size=0,
                    glu_backend=glu_backend,
                )
            )
        return jnp.concatenate(blocks, axis=1)

    x = _layer_norm(x, params["norm"]["scale"], params["norm"]["bias"], eps)
    fc1_kernel = params["fc1"]["kernel"]
    fc2_kernel = params["fc2"]["kernel"]
    fc3_kernel = params["fc3"]["kernel"]

    if glu_backend != "xla":
        hidden = gated_linear_unit(
            x, fc1_kernel, fc2_kernel, jax.nn.silu, backend=glu_backend
        )
        return hidden @ fc3_kernel

    if chunk_size is None:
        fc12 = x @ jnp.concatenate((fc1_kernel, fc2_kernel), axis=-1)
        fc1, fc2 = jnp.split(fc12, 2, axis=-1)
        hidden = jax.nn.silu(fc1) * fc2
        return hidden @ fc3_kernel

    if chunk_size <= 0:
        msg = f"chunk_size must be positive, got {chunk_size}"
        raise ValueError(msg)

    out = jnp.zeros((*x.shape[:-1], fc3_kernel.shape[-1]), dtype=x.dtype)
    hidden_dim = fc3_kernel.shape[0]
    for start in range(0, hidden_dim, chunk_size):
        stop = min(start + chunk_size, hidden_dim)
        hidden = jax.nn.silu(x @ fc1_kernel[:, start:stop]) * (
            x @ fc2_kernel[:, start:stop]
        )
        out = out + hidden @ fc3_kernel[start:stop, :]
    return out
