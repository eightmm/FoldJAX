"""Native mixed precision realised per stage, separate from trunk-wide narrowing.

``trunk_dtype`` narrows one root parameter field and casts the trunk's feature
dict, which is enough for the trunk because every activation inside it is then
BF16 and an ordinary BF16 weight gives a BF16 matmul. The three stages here do
not have that property: their inputs are a mix of FP32 geometry and BF16 trunk
representations, and upstream exempts named projections from autocast
individually. Each function below rebuilds one parameter subtree so that the
realised dtype of every matmul matches what torch autocast would have done.

The exemptions are not inferred from the port. They are the projections
upstream constructs with ``precision=torch.float32``, which run FP32 and cast
their result back to the input's dtype whatever the ambient autocast says --
``protenix/model/modules/transformer.py`` for the atom encoder and decoder,
``protenix/model/modules/diffusion.py`` for the conditioner and the diffusion
module, and nowhere else in ``protenix/`` (a census of ``precision=``).
"""

import jax
import jax.numpy as jnp

from foldjax.models.protenix.models.primitives.primitives import (
    AutocastLinearParams,
    Fp32PrecisionLinearParams,
    LayerNormParams,
    LinearParams,
)


def _require_original_fp32(params) -> None:
    """Refuse already-narrowed parameters: widening cannot undo a rounding."""
    leaves = jax.tree.leaves(params)
    if any(
        hasattr(x, "dtype")
        and jnp.issubdtype(x.dtype, jnp.floating)
        and x.dtype != jnp.float32
        for x in leaves
    ):
        raise ValueError("native autocast requires original FP32 parameters")


def _autocast_linear(value: LinearParams) -> AutocastLinearParams:
    return AutocastLinearParams(
        value.weight.astype(jnp.bfloat16),
        None if value.bias is None else value.bias.astype(jnp.bfloat16),
    )


def _autocast_linears(params):
    """Every ``LinearParams`` in the subtree becomes a BF16 autocast projection.

    ``LayerNormParams`` is deliberately a leaf and deliberately untouched:
    upstream's layer norm quantizes its own affine operands to the input dtype
    and accumulates in FP32 (``OpenFoldLayerNorm.forward``), which is what
    :func:`~foldjax.models.protenix.models.primitives.primitives.layer_norm`
    reproduces from the *input* dtype. Narrowing the stored affine would be the
    same arithmetic; leaving it FP32 keeps one representation of it.
    """
    return jax.tree.map(
        lambda value: (
            _autocast_linear(value) if isinstance(value, LinearParams) else value
        ),
        params,
        is_leaf=lambda x: isinstance(x, (LinearParams, LayerNormParams)),
    )


def _narrow_arrays(params, dtype=jnp.bfloat16):
    """Round every float leaf, keeping the parameter node types as they are.

    This is the trunk's realisation: with BF16 activations an ordinary BF16
    weight already gives a BF16 matmul, so no node needs to change type.
    """
    return jax.tree.map(
        lambda value: (
            value.astype(dtype)
            if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating)
            else value
        ),
        params,
    )


def native_input_autocast_params(params):
    """Prepare FP32 publisher parameters for the observed BF16 encoder route.

    Call before any blanket narrowing: widening rounded operands cannot recover
    the FP32 geometry projections or normalization affine values.
    """
    _require_original_fp32(params)
    result = _autocast_linears(params)
    original = params.atom_encoder.cache
    cache = result.atom_encoder.cache._replace(
        linear_ref_pos=original.linear_ref_pos,
        linear_d=original.linear_d,
    )
    return result._replace(atom_encoder=result.atom_encoder._replace(cache=cache))


def native_confidence_autocast_params(params):
    """Realise ``skip_amp.confidence_head = False`` on the confidence head.

    Three groups, because what reaches them differs:

    * The pairformer blocks and ``input_strunk_ln`` are reached with BF16 --
      the trunk's own single and pair representations -- so they take the
      trunk's realisation: narrow the weights and leave the node types alone.
    * Four projections are reached with FP32 and have to narrow their own
      operands. Two are the distance projections, because upstream computes
      ``cdist`` inside ``autocast(enabled=False)`` and leaves the projections
      outside it (``confidence.py:276-306``). The other two are ``linear_s1``
      and ``linear_s2``: their input is ``s_inputs``, which leaves the input
      embedder FP32 (it concatenates raw reference features that native
      autocast does not narrow), so a merely narrowed weight would promote the
      outer-sum initialiser back to an FP32 matmul and take the pair tensor,
      and the whole stack after it, with it. Upstream's ``Linear`` has no such
      escape: under autocast ``F.linear`` narrows both operands whatever the
      input dtype was.
    * ``output`` stays FP32 in full. Upstream upcasts ``s_single`` and
      ``z_pair`` and runs the four output projections inside
      ``autocast(enabled=False)`` (``confidence.py:317-340``), which the port
      already mirrors with its ``astype(float32)`` boundary. Rounding those
      weights would change a stage that is FP32 under every policy.

    ``lower_bins``/``upper_bins`` stay FP32 with ``output``: they are compared
    against the FP32 distances, not multiplied.
    """
    _require_original_fp32(params)
    embedding = params.distance_embedding
    narrowed = _narrow_arrays(params._replace(distance_embedding=None, output=None))
    return narrowed._replace(
        linear_s1=_autocast_linear(params.linear_s1),
        linear_s2=_autocast_linear(params.linear_s2),
        distance_embedding=embedding._replace(
            linear_d=_autocast_linear(embedding.linear_d),
            linear_d_wo_onehot=_autocast_linear(embedding.linear_d_wo_onehot),
        ),
        output=params.output,
    )


def native_diffusion_autocast_params(params):
    """Realise ``skip_amp.sample_diffusion = False`` on the denoising network.

    Unlike the confidence head this subtree mixes dtypes: the atom encoder
    reads FP32 reference geometry beside BF16 trunk conditioning, so every
    projection has to narrow its own operands rather than inherit a dtype from
    upstream activations. Hence autocast projections throughout, and the ten
    ``precision=torch.float32`` exemptions rebuilt as FP32 projections that
    narrow their result back -- dropping that final cast would widen the whole
    denoiser downstream of the first geometry projection.
    """
    _require_original_fp32(params)
    result = _autocast_linears(params)

    def exempt(source: LinearParams | None) -> Fp32PrecisionLinearParams | None:
        # The three ``has_coords`` projections are absent from an encoder built
        # without coordinates; the diffusion module always has them.
        if source is None:
            return None
        return Fp32PrecisionLinearParams(source.weight, source.bias)

    encoder = params.atom_encoder
    cache = result.atom_encoder.cache._replace(
        # transformer.py:646-658 -- reference position and the pair distance.
        linear_ref_pos=exempt(encoder.cache.linear_ref_pos),
        linear_d=exempt(encoder.cache.linear_d),
    )
    atom_encoder = result.atom_encoder._replace(
        cache=cache,
        # transformer.py:668-690 -- the has_coords conditioning projections.
        linear_s=exempt(encoder.linear_s),
        linear_z=exempt(encoder.linear_z),
        linear_r=exempt(encoder.linear_r),
    )
    conditioning = result.conditioning._replace(
        # diffusion.py:59-79 -- pair, single and noise conditioners.
        linear_z=exempt(params.conditioning.linear_z),
        linear_s=exempt(params.conditioning.linear_s),
        linear_n=exempt(params.conditioning.linear_n),
    )
    return result._replace(
        conditioning=conditioning,
        atom_encoder=atom_encoder,
        # diffusion.py:302-307 -- Algorithm 20 line 4.
        linear_s=exempt(params.linear_s),
        atom_decoder=result.atom_decoder._replace(
            # transformer.py:988-990 -- the coordinate update.
            linear_out=exempt(params.atom_decoder.linear_out),
        ),
    )
