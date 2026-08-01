"""Low-level ops used by Chai-1 components, ported to JAX.

Math is recovered exactly from the TorchScript ``.code`` of the exported
components (the upstream Chai repo ships no nn.Module source -- the only
authoritative reference is the scripted graph and the chai_lab feature
generators).

Notes on precision: the exported ``input_projs.*.0`` linears cast input/weight
to bfloat16 (``torch.to(x, 15)``) before ``torch.linear``. We expose an fp32
``linear`` (precision-matched parity reference) and a ``linear_bf16`` variant
that reproduces the runtime cast.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def linear(
    x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray | None = None
) -> jnp.ndarray:
    """torch.linear: y = x @ weight.T (+ bias). weight is (out, in)."""
    y = jnp.matmul(x, weight.T, precision=jax.lax.Precision.HIGHEST)
    if bias is not None:
        y = y + bias
    return y


def linear_bf16(
    x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray | None = None
) -> jnp.ndarray:
    """Reproduce the exported runtime: cast to bf16, linear, keep bf16 out."""
    xb = x.astype(jnp.bfloat16)
    wb = weight.astype(jnp.bfloat16)
    # torch.linear accumulates the bf16 matmul in fp32 then rounds; match that
    # by accumulating in fp32 and casting the result back to bf16.
    y = jnp.matmul(xb, wb.T, preferred_element_type=jnp.float32)
    if bias is not None:
        y = y + bias.astype(jnp.float32)
    return y.astype(jnp.bfloat16)


def layer_norm(
    x: jnp.ndarray,
    weight: jnp.ndarray | None = None,
    bias: jnp.ndarray | None = None,
    *,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """LayerNorm over the last axis (torch.nn.functional.layer_norm semantics)."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    y = (x - mean) / jnp.sqrt(var + eps)
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y


def embedding(weight: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    """torch.embedding(weight, idx): row lookup. weight is (num, dim)."""
    return weight[idx]


def one_hot(idx: jnp.ndarray, num_classes: int) -> jnp.ndarray:
    """torch.one_hot(idx, num_classes) -> float (matches the .to(float) in graph)."""
    return jnp.asarray(idx[..., None] == jnp.arange(num_classes), dtype=jnp.float32)


def contiguous_segment_mean(
    values: jnp.ndarray,
    mask: jnp.ndarray,
    segment_ids: jnp.ndarray,
    *,
    num_segments: int,
) -> jnp.ndarray:
    """Mean-pool contiguous masked rows without atomic scatter reductions.

    Chai stores real atoms contiguously by token, followed by masked padding.
    Prefix sums therefore give a fixed reduction order on GPU, unlike
    ``scatter_add`` whose BF16 accumulation order can vary between launches.
    """
    if values.ndim != 3 or mask.shape != values.shape[:2]:
        raise ValueError(
            "values must be [batch, rows, channels] with a [batch, rows] mask"
        )
    if segment_ids.shape != mask.shape:
        raise ValueError("segment_ids must match the mask shape")

    def pool_one(
        batch_values: jnp.ndarray,
        batch_mask: jnp.ndarray,
        batch_segments: jnp.ndarray,
    ) -> jnp.ndarray:
        valid_values = jnp.where(batch_mask[:, None], batch_values, 0)
        searchable = jnp.where(batch_mask, batch_segments, num_segments)
        targets = jnp.arange(num_segments, dtype=searchable.dtype)

        def sorted_prefix_pool(_: None) -> jnp.ndarray:
            prefix = jnp.concatenate(
                [
                    jnp.zeros((1, batch_values.shape[-1]), batch_values.dtype),
                    jnp.cumsum(valid_values, axis=0),
                ],
                axis=0,
            )
            starts = jnp.searchsorted(searchable, targets, side="left")
            ends = jnp.searchsorted(searchable, targets, side="right")
            sums = prefix[ends] - prefix[starts]
            counts = (ends - starts).astype(batch_values.dtype)
            return sums / jnp.maximum(counts[:, None], 1)

        def masked_hole_pool(_: None) -> jnp.ndarray:
            membership = (
                (batch_segments[:, None] == targets[None])
                & batch_mask[:, None]
            )
            prefix = jnp.concatenate(
                [
                    jnp.zeros((1, batch_values.shape[-1]), batch_values.dtype),
                    jnp.cumsum(valid_values, axis=0),
                ],
                axis=0,
            )
            row = jnp.arange(batch_values.shape[0])[:, None]
            starts = jnp.min(
                jnp.where(membership, row, batch_values.shape[0]), axis=0
            )
            ends = jnp.max(jnp.where(membership, row + 1, 0), axis=0)
            sums = prefix[ends] - prefix[starts]
            counts = membership.sum(axis=0).astype(batch_values.dtype)
            return jnp.where(
                counts[:, None] > 0,
                sums / jnp.maximum(counts[:, None], 1),
                0,
            )

        is_sorted = jnp.all(searchable[1:] >= searchable[:-1])
        return jax.lax.cond(is_sorted, sorted_prefix_pool, masked_hole_pool, None)

    return jax.vmap(pool_one)(values, mask, segment_ids)


def rbf_restraint_encoding(
    raw_data: jnp.ndarray,
    radii: jnp.ndarray,
    *,
    width: float,
    clamp_max: float = 16.0,
) -> jnp.ndarray:
    """Restraint RBF encoding, exactly per exported forward graph.

    Given ``raw_data`` (..., 1 after unsqueeze) and ``radii`` (num_radii,):

        e = ((radii - raw_data) / width) ** 2
        e = clamp_max(e, clamp_max)
        enc = exp(-e); enc[e == clamp_max] = 0
        mask = (raw_data == -1)            # missing restraint
        enc = enc * (1 - mask)
        out = cat([enc, mask], -1)         # last dim = num_radii + 1
    """
    raw = raw_data[..., None]
    r = radii.reshape((1,) * (raw.ndim - 1) + (radii.shape[-1],))
    e = ((r - raw) / width) ** 2
    e = jnp.minimum(e, clamp_max)
    enc = jnp.exp(-e)
    enc = jnp.where(e == clamp_max, 0.0, enc)
    mask = (raw == -1.0).astype(jnp.float32)
    enc = enc * (1.0 - mask)
    return jnp.concatenate([enc, mask], axis=-1)
