"""OpenDDE reaches Protenix's gated transitions and is not offered the kernel.

OpenDDE's diffusion conditioner, its structural refiner and its trunk all call
`foldjax.models.protenix.models.primitives.primitives.transition`, so the fused
route is in its process and one keyword away. It is not offered, because a
kernel belongs to the port whose numbers were measured and nobody has measured
a GLU on either of these two -- the same rule the shared attention-flag builder
states for its `extra_backends` parameter.

Nothing downstream enforces that. There is no guard in the OpenDDE model that
would reject the value; what keeps it out is three absences -- from the parser,
from the adapter's option set, and from every OpenDDE call site. These tests
are those absences written down, because an absence nobody asserts is an
absence the next shared-builder edit deletes by accident.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from foldjax.backends.opendde import OpenDDEBackend
from foldjax.schema import PredictionRequest


def _request(tmp_path: Path, **options) -> PredictionRequest:
    job = tmp_path / "job.json"
    job.write_text("{}")
    weights = tmp_path / "opendde.jax"
    weights.touch()
    return PredictionRequest(
        model="opendde",
        input=job,
        weights=weights,
        output_dir=tmp_path / "out",
        seed=5,
        cache_dir=tmp_path / "cache",
        options=options,
    )


def test_the_adapter_has_no_glu_backend_option() -> None:
    assert "glu_backend" not in OpenDDEBackend.native_options
    assert "glu_backend" not in OpenDDEBackend.compile_options


def test_asking_opendde_for_the_fused_glu_is_an_error(tmp_path: Path) -> None:
    """Rejected by name, through the ordinary unknown-option path.

    Not a bespoke refusal: the value is simply not one of this port's
    options, which is the strongest form the guarantee can take -- there is
    no branch to get wrong.
    """
    with pytest.raises(ValueError, match="unsupported OpenDDE options"):
        OpenDDEBackend().predict(_request(tmp_path, glu_backend="tokamax"))
    with pytest.raises(ValueError, match="unsupported OpenDDE options"):
        OpenDDEBackend().predict(_request(tmp_path, glu_backend="xla"))


def test_opendde_call_sites_leave_the_transition_on_its_released_route(
    monkeypatch,
) -> None:
    """The shared primitive is reachable; no OpenDDE caller reaches for it.

    Patching the one seam to the Triton kernel and running the two OpenDDE
    modules that own gated transitions proves the default they inherit is the
    split-matmul one, rather than proving it by reading their call sites.
    """
    from importlib import import_module

    import jax.numpy as jnp
    import numpy as np

    from foldjax.models import _glu
    from foldjax.models.protenix.models.primitives.primitives import (
        LayerNormParams,
        LinearParams,
        TransitionParams,
    )

    def explode(*args, **kwargs):
        raise AssertionError("an OpenDDE call site reached the fused GLU")

    monkeypatch.setattr(_glu, "_fused", explode)

    rng = np.random.default_rng(0)

    def array(*shape):
        return jnp.asarray(rng.standard_normal(shape) * 0.5, dtype=jnp.float32)

    params = TransitionParams(
        layer_norm=LayerNormParams(weight=array(8) + 1.0, bias=array(8)),
        linear_a=LinearParams(weight=array(16, 8)),
        linear_b=LinearParams(weight=array(16, 8)),
        linear_out=LinearParams(weight=array(8, 16)),
    )
    # The symbol each module imported, not a fresh one: this is the callable
    # its own `transition(...)` lines resolve to.
    for name in (
        "foldjax.models.opendde.models.diffusion_conditioning",
        "foldjax.models.opendde.models.structural_refiner",
    ):
        call = import_module(name).transition
        assert call(array(6, 8), params).shape == (6, 8)
