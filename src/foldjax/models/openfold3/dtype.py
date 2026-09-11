"""Which parts of OpenFold3 a narrow ``dtype`` reaches, and which it does not.

The one-sentence shape: **the token/pair representation track narrows; anything
atom- or coordinate-shaped stays float32.** That is AlphaFold 3's released
inference shape, read out of its source rather than out of its config name, and
it is close to the regime this checkpoint was trained under.

Reading AlphaFold 3's shape correctly takes two facts, not one.
``diffusion_head.py:221`` opens ``utils.bfloat16_context()`` around the whole
diffusion head, which looks like "the diffusion head is bfloat16". It is not.
``components/haiku_modules.py:175`` requests every parameter with
``hk.get_parameter(..., inputs.dtype, ...)`` and ``components/utils.py:48``'s
getter narrows only when the requested dtype is *already* bfloat16, so the
context narrows nothing on its own: a module's dtype is decided by the dtype of
the activation that enters it. Following that rule through the diffusion head:

* ``atom_cross_attention.py:49-53`` builds the per-atom conditioning from
  ``ref_structure.positions``, which is float32, so ``queries_single_cond``,
  ``keys_single_cond`` and ``pair_act`` are float32; the noisy positions that
  become ``queries_act`` are float32 too. **The atom cross-attention encoder and
  decoder run in float32**, inside the bfloat16 context. ``model.py:176``'s
  explicit ``.astype(dtype)`` on the concatenated encoder output is the tell --
  it would be redundant if the encoder returned bfloat16.
* ``diffusion_head.py:265-268`` then casts ``act``, ``trunk_single_cond``,
  ``trunk_pair_cond`` and ``sequence_mask`` to float32 before the token
  transformer. Only two of those four change anything: ``act`` arrived float32
  from the atom encoder, and ``single_cond`` was promoted to float32 back at
  ``diffusion_head.py:192-196`` where the float32 noise embedding is added.
  The load-bearing cast is ``trunk_pair_cond``.
* ``confidence_head.py:129-131`` casts ``pair``, ``single`` and ``target_feat``
  *down* to bfloat16, so the confidence head's re-embedding Pairformer does
  narrow. It comes back up the moment that stack returns -- ``:163`` for the
  pair track, ``:244`` for the single track -- and both casts land *before*
  the logit heads' own layer norms, which is what makes them load-bearing
  rather than cosmetic. Only the pTM reduction is pulled out of the context
  entirely (``:230``).

Upstream OpenFold3 is the second source, and it is about this checkpoint rather
than a sibling's. Inference runs float32 (``entry_points/validator.py:127``
``precision: "32-true"``), but all four released training configs run
``bf16-mixed`` -- ``examples/training_yamls/initial_training.yml:23`` and
``finetune_1/2/3.yml`` -- so these weights were *trained* with bfloat16
activations. Under that regime upstream pins exactly two islands to float32
inside the model, and both agree with AlphaFold 3's shape:

* ``feature_embedders/input_embedders.py:129-131`` wraps the input embedder's
  ``atom_attn_enc`` in ``autocast(dtype=torch.float32)``;
* ``heads/prediction_heads.py:88-89`` wraps the confidence head's ``embed_zij``
  in the same, restoring the incoming dtype at ``:118``.

The port measured the first island independently: a whole-trunk bfloat16 cast
takes pLDDT from 0.858 to 0.466 and the damage starts in the input embedder,
while keeping the embedder float32 recovers it (pLDDT -0.001, CA RMSD 0.040 A
against a 0.005 A rerun floor, at 1,003 tokens on 2026-08-10).

The confidence head is therefore its own narrowing group, with its own knob
(``confidence_dtype``, following ``dtype`` unless set). It is separable for a
reason: it consumes predicted coordinates and emits scores, never coordinates,
so narrowing it cannot move a structure, and it is the region with no
measurement on this port. Protenix measured the same change at 3,012 tokens --
coordinates bitwise unchanged, atom pLDDT at most 0.0099, chain pTM/ipTM at
most 1.9e-4, PAE means at most 0.005 -- which is evidence from a sibling, not
from here.

What the option narrows in *this* port, which is the same shape stated as a
parameter split, is :func:`~foldjax.models.openfold3.inference.cast_narrow_params`
and the boundary casts it needs are in ``models/trunk.py``,
``models/input_embedders.py`` and ``models/heads.py``. Two deliberate
departures from the readings above:

* the input embedder's *five projections* stay float32 here, where upstream's
  autocast would run them at bfloat16. Only its atom encoder has to be wide,
  but keeping the whole subtree wide costs two ``[N, C]`` and one ``[N, N, C]``
  projection and buys an embedder that no future edit can narrow by accident.
  The cast lands on ``s`` and ``z`` immediately after.
* the template tower narrows its *body*, which upstream's autocast does too,
  but it gets there by being handed a narrowed feature mapping rather than by
  a cast inside the tower.

A narrowed region's attention softmax runs bfloat16, pair bias included. Both
upstreams do that on purpose -- ``core/model/primitives/attention.py:107-122``
(``softmax_no_cast``) and ``primitives/normalization.py:66-78`` disable
autocast specifically so bfloat16 softmax and layer norm stay bfloat16,
overriding torch's float32 default for both, and AlphaFold 3's evoformer is the
same shape. What is *not* narrowed anywhere is an output logit: the heads keep
float32 parameters, so distogram, PAE, PDE, pLDDT and experimentally-resolved
logits are float32 in both profiles. The one place a bfloat16 pair bias into a
softmax is known to be fatal on a port is the diffusion token transformer,
which is float32 here.

What this module does **not** claim: that upstream OpenFold3 validates this at
inference. It does not -- it ships ``32-true``. Nothing here has a GPU accuracy
row yet.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

#: Accepted spellings, in `foldjax.execution`'s neutral vocabulary. The default
#: is named here rather than repeated at each caller so the released profile has
#: one definition.
DTYPES: tuple[str, ...] = ("float32", "bfloat16")

#: What a request that says nothing gets: upstream's inference precision.
DEFAULT_DTYPE = "float32"


def narrow_dtype(name: str) -> Any:
    """Return the dtype ``name`` narrows to, or ``None`` when it narrows nothing.

    ``None`` rather than ``jnp.float32`` on purpose: every narrowing site below
    is guarded on it, so the released profile emits no casts at all rather than
    a tree of float32-to-float32 converts that a reader would have to prove are
    free.
    """

    if name not in DTYPES:
        raise ValueError(
            f"OpenFold3 dtype must be one of {', '.join(DTYPES)}; got {name!r}"
        )
    return jnp.bfloat16 if name == "bfloat16" else None


def narrow_floats[T](tree: T, dtype: Any) -> T:
    """Cast every floating leaf of ``tree`` to ``dtype``, leaving the rest alone.

    Integer and boolean leaves stay as they are: JAX's promotion lattice puts
    every integer below every float, so an integer one-hot multiplied by a
    bfloat16 weight is already bfloat16, and casting a mask's index arithmetic
    to a float would be a different program.
    """

    if dtype is None:
        return tree

    def cast(value):
        value_dtype = getattr(value, "dtype", None)
        if value_dtype is not None and jnp.issubdtype(value_dtype, jnp.floating):
            return value.astype(dtype)
        return value

    return jax.tree.map(cast, tree)
