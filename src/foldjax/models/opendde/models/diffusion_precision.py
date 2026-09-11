"""OpenDDE's denoiser autocast boundary: Protenix's, plus one exemption.

``OpenDDEInferenceParams.diffusion`` is the same
:class:`~foldjax.models.protenix.models.diffusion.diffusion.DiffusionModuleParams`
NamedTuple Protenix uses, so the boundary Protenix already realises applies
here: autocast projections throughout, with the projections upstream
constructs ``precision=torch.float32`` rebuilt as FP32 islands that narrow
their result back, and every per-head pair bias delivered in FP32.

One field is OpenDDE's own and Protenix's function gets it wrong. OpenDDE
compresses the 384-channel trunk pair representation to 128 channels before
conditioning on it (``c_z`` 384 against ``c_z_pair_diffusion`` 128,
``opendde/config/model_base.py:20,190``), and it builds that projection
``precision=torch.float32`` like the rest of the conditioner
(``opendde/model/modules/diffusion.py:170-174``). Protenix has no such field,
so Protenix's ``native_diffusion_autocast_params`` narrows it along with
the ordinary projections. It is exempted again here.

A census of ``precision=`` across ``opendde/`` returns eleven projections and
nothing else: five in the atom encoder (``transformer.py:2338,2347,2363,
2373,2377``), the decoder's coordinate update (``transformer.py:3705``), four
in the conditioner (``diffusion.py:173,179,190,198``), and Algorithm 20 line 4
(``diffusion.py:1203``). Ten of the eleven are Protenix's ten. Upstream's
``Linear.forward`` (``opendde/model/modules/primitives.py:75-91``) has the
same three steps Protenix's does -- widen the input, multiply in FP32, cast
the result back to the input's dtype -- so the port's FP32 island node type
reproduces it unchanged.

Nothing here is a *measured* OpenDDE choice. The FP32 pair-bias output in
particular is inherited from a Protenix measurement on 5DEI and has no
OpenDDE accuracy row behind it; see ``docs/cli.md``.
"""

from __future__ import annotations

from foldjax.models.protenix.models.diffusion.diffusion import DiffusionModuleParams
from foldjax.models.protenix.models.input_precision import (
    native_diffusion_autocast_params as _protenix_diffusion_autocast_params,
)
from foldjax.models.protenix.models.primitives.primitives import (
    Fp32PrecisionLinearParams,
)

#: The conditioner field OpenDDE adds to Protenix's shared parameter tree.
_COMPRESSION_FIELD = "linear_z_trunk"


def native_diffusion_autocast_params(
    params: DiffusionModuleParams,
) -> DiffusionModuleParams:
    """Realise ``skip_amp.sample_diffusion = False`` on OpenDDE's denoiser.

    Refuses parameters that are already narrowed -- widening a rounded
    operand cannot recover it -- by way of the shared Protenix function.
    """
    conditioning = params.conditioning
    if not hasattr(conditioning, _COMPRESSION_FIELD):
        raise ValueError(
            "OpenDDE diffusion conditioning must carry "
            f"{_COMPRESSION_FIELD!r}; got a "
            f"{type(conditioning).__name__} without it"
        )
    result = _protenix_diffusion_autocast_params(params)
    source = getattr(conditioning, _COMPRESSION_FIELD)
    if source is None:  # pragma: no cover - the released tree always has it
        return result
    return result._replace(
        conditioning=result.conditioning._replace(
            linear_z_trunk=Fp32PrecisionLinearParams(source.weight, source.bias)
        )
    )
