"""Which parts of OpenFold3 a narrow ``dtype`` reaches, and which it does not.

Contract: **the token/pair representation track narrows; anything atom- or
coordinate-shaped stays float32.** The default is ``bfloat16``, with
``dtype="float32"`` restoring upstream's ``32-true`` inference precision
(benchmark/panel evidence is recorded in ``docs/cli.md``,
``docs/engineering-notes.md``, and
``docs/archive/openfold3-bf16-default-evidence-2026-09-12.md``).

Parameter split and boundary casts live in
:func:`~foldjax.models.openfold3.inference.cast_narrow_params`,
``models/trunk.py``, ``models/input_embedders.py``, and ``models/heads.py``:

* **Narrowed (``dtype``)**: MSA embedder/stack, Pairformer stack, trunk
  recycling norms/projections, template tower body (via narrowed inputs),
  diffusion conditioning pair branch, and the confidence head's Pairformer
  stack (controlled by ``confidence_dtype``, which follows ``dtype`` when unset
  for memory reasons and can be pinned to ``float32`` independently without
  affecting coordinates).
* **Always float32**: the entire ``input_embedder`` subtree (slightly wider
  than upstream's ``atom_attn_enc`` float32 island so future edits cannot
  narrow it accidentally; narrowing it collapses pLDDT), the denoiser (atom
  encoder/decoder and diffusion token transformer), diffusion conditioning
  single branch, confidence geometry re-embedding, and all output logit heads
  (distogram, PAE, PDE, pLDDT, experimentally-resolved).

Arithmetic and bitwise caveats:

* Narrowed attention softmax runs bfloat16 by default (matching upstream's
  ``softmax_no_cast`` in triangle attention and MSA pair-weighted averaging);
  ``high_precision_attention=True`` restores upstream's float32
  ``PairFormerBlock.attn_pair_bias`` attention core.
* :func:`~foldjax.models.openfold3.models.primitives.layer_norm` upcasts
  normalization to float32 and rounds once on exit, matching upstream's
  ``primitives/normalization.py``, and guards output width on ``x.dtype``
  rather than the promoted dtype. Under ``jax.jit`` on CPU, XLA may ignore
  declared intermediate widths at large shapes (e.g. ``[20000, 128]`` vs
  ``[64, 128]``), so per-op width checks must be verified eager or against
  compiled HLO. Whether the GPU compiler does the same at trunk shapes is not
  established; a jit-against-eager byte comparison of one norm on a
  trunk-shaped slab settles it.
* :func:`narrow_floats` never casts ``LayerNormParams`` (0.381 MiB across
  1,163 affine tensors), matching upstream's unrounded ``weight.float()`` and
  keeping denoiser stack-level pair norms narrow under the ``x.dtype`` guard
  (validated across six seeds on 6ZTX: core 0.58-1.02 A vs 0.56-0.93 A wide,
  with both arms reproducing the same seed-404 arm basin). Under
  ``triangle_kernel=cueq-full``, cuEquivariance receives float32 norm weights
  against bfloat16 ``x`` (upstream's autocast signature); whether its internal
  fused accumulation matches pure JAX remains unverified.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from foldjax.models.openfold3.models.primitives import LayerNormParams

#: Accepted spellings in `foldjax.execution`'s vocabulary.
DTYPES: tuple[str, ...] = ("float32", "bfloat16")

#: Single default read by config, `released_config`, and cache-namespace
#: stripping; unset `confidence_dtype` follows this value.
DEFAULT_DTYPE = "bfloat16"


def narrow_dtype(name: str) -> Any:
    """Return the narrow JAX dtype for ``name``, or ``None`` for ``"float32"``.

    Returning ``None`` lets every narrowing site skip casting altogether so the
    float32 profile emits no convert ops.
    """

    if name not in DTYPES:
        raise ValueError(
            f"OpenFold3 dtype must be one of {', '.join(DTYPES)}; got {name!r}"
        )
    return jnp.bfloat16 if name == "bfloat16" else None


def narrow_floats[T](tree: T, dtype: Any) -> T:
    """Cast floating leaves of ``tree`` to ``dtype``, leaving other leaves intact.

    Integer and boolean leaves stay uncast: JAX's promotion rules keep integer
    one-hots narrow when multiplied by bfloat16 weights, and casting masks or
    indices would alter the program.

    :class:`~foldjax.models.openfold3.models.primitives.LayerNormParams` nodes
    (including those nested inside ``AdaLNParams``) are deliberately excluded
    so normalization multiplies by the unrounded float32 affine (matching
    upstream's ``weight.float()``) while ``primitives.layer_norm``'s
    ``x.dtype`` guard keeps trunk and denoiser stack-level pair norms narrow.
    """

    if dtype is None:
        return tree

    def cast(value):
        if isinstance(value, LayerNormParams):
            return value
        value_dtype = getattr(value, "dtype", None)
        if value_dtype is not None and jnp.issubdtype(value_dtype, jnp.floating):
            return value.astype(dtype)
        return value

    return jax.tree.map(
        cast, tree, is_leaf=lambda node: isinstance(node, LayerNormParams)
    )
