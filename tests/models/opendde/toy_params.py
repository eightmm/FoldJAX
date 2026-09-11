"""A parameter tree small enough to stub with and real enough to prepare.

Since 2026-09-11 OpenDDE's CLI narrows the confidence head on every default
run, so a stub that hands ``main`` a sentinel object no longer reaches the
model: ``cast_confidence_params`` rebuilds ``params.confidence`` and needs a
real :class:`ConfidenceHeadParams` to rebuild. Tests that stub the whole
prediction still want a cheap tree, so this is the cheap tree -- toy shapes
everywhere, a genuine confidence head, and opaque markers for the stages no
CLI-level preparation touches.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from foldjax.models.opendde.models.model import OpenDDEInferenceParams
from foldjax.models.protenix.models.heads.confidence import (
    ConfidenceDistanceEmbeddingParams,
    ConfidenceHeadParams,
    ConfidenceOutputParams,
)
from foldjax.models.protenix.models.primitives.primitives import (
    LayerNormParams,
    LinearParams,
)
from foldjax.models.protenix.models.trunk_blocks.pairformer import PairformerStackParams

N_TOKEN, N_ATOM, C_S_INPUTS, C_S, C_Z, N_BINS, N_OUT = 2, 3, 5, 4, 3, 8, 2


def array(*shape: int) -> jnp.ndarray:
    """A deterministic FP32 array, seeded by its own shape."""

    rng = np.random.default_rng(sum(shape) * 7 + len(shape))
    return jnp.asarray(rng.normal(size=shape), dtype=jnp.float32)


def confidence_head_params() -> ConfidenceHeadParams:
    """A released-shaped head: adjacent finite bins, so compact binning is exact."""

    lower = jnp.asarray(np.linspace(2.0, 18.0, N_BINS), dtype=jnp.float32)
    upper = jnp.concatenate([lower[1:], jnp.asarray([22.0], dtype=jnp.float32)])
    return ConfidenceHeadParams(
        input_strunk_ln=LayerNormParams(array(C_S), array(C_S)),
        linear_s1=LinearParams(array(C_Z, C_S_INPUTS)),
        linear_s2=LinearParams(array(C_Z, C_S_INPUTS)),
        distance_embedding=ConfidenceDistanceEmbeddingParams(
            lower_bins=lower,
            upper_bins=upper,
            linear_d=LinearParams(array(C_Z, N_BINS)),
            linear_d_wo_onehot=LinearParams(array(C_Z, 1)),
        ),
        pairformer_stack=PairformerStackParams(blocks=()),
        output=ConfidenceOutputParams(
            pae_ln=LayerNormParams(array(C_Z), array(C_Z)),
            pde_ln=LayerNormParams(array(C_Z), array(C_Z)),
            plddt_ln=LayerNormParams(array(C_S), array(C_S)),
            resolved_ln=LayerNormParams(array(C_S), array(C_S)),
            linear_pae=LinearParams(array(N_OUT, C_Z)),
            linear_pde=LinearParams(array(N_OUT, C_Z)),
            plddt_weight=array(2, C_S, N_OUT),
            resolved_weight=array(2, C_S, N_OUT),
        ),
    )


def inference_params() -> OpenDDEInferenceParams:
    """The whole tree: a real confidence head, markers everywhere else.

    The markers are deliberately not arrays. A preparation that reached a
    stage it has no business in would have to raise rather than silently
    return a rebuilt copy, which is what makes this safe to stub with.
    """

    return OpenDDEInferenceParams(
        input_embedder=object(),
        pairformer_output=object(),
        structural_expander=object(),
        structural_refiner=object(),
        diffusion=SimpleNamespace(
            conditioning=SimpleNamespace(relpe=object()),
            atom_encoder=object(),
        ),
        distogram=object(),
        confidence=confidence_head_params(),
    )
