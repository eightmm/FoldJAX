"""Where the released fused gated linear unit is resolved, and where refused.

``glu_backend`` ships as ``tokamax``. That makes the released value and an
explicit request the same string in ``api.predict``, so the two policies this
module pins live at different layers on purpose:

* ``api.predict`` cannot tell them apart and therefore *resolves* -- under
  context parallelism the fused default becomes the blocked XLA path, exactly
  as the ``cueq`` triangle default already does, and the resolved value is
  what reaches the model and the retained-runner identity.
* ``Boltz2Backend.validate_native_options`` sees an option dict, where an
  absent key still means "unset", so that is where a caller who explicitly
  names the fused kernel under context parallelism is refused.

The end-to-end proof that the default does not raise on a real multi-device
mesh is ``tests/test_boltz2_session.py`` -- its forced four-device probe now
omits the knob and asserts the model receives ``xla``.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import foldjax.backends.boltz2 as backend_module
import foldjax.models.boltz2.api as api
import foldjax.models.boltz2.bridge.native as native
import foldjax.models.boltz2.models.predict as predict_module
from foldjax.backends.boltz2 import Boltz2Backend
from foldjax.models._glu import GLU_BACKENDS
from tests.models.cp_probe_env import inherited_environment


def _fake_features() -> dict[str, np.ndarray]:
    return {
        "atom_pad_mask": np.ones((1, 3), dtype=np.float32),
        "token_pad_mask": np.ones((1, 2), dtype=np.float32),
        "mol_type": np.asarray([[0, 0]], dtype=np.int32),
        "token_to_rep_atom": np.asarray([[[1, 0, 0], [0, 0, 1]]], dtype=np.int64),
        "atom_to_token": np.asarray([[[1, 0], [1, 0], [0, 1]]], dtype=np.int64),
        "affinity_token_mask": np.zeros((1, 2), dtype=np.float32),
    }


def test_released_default_is_the_fused_kernel_on_both_surfaces() -> None:
    signature = inspect.signature(api.predict)

    assert signature.parameters["glu_backend"].default == "tokamax"
    assert backend_module._RELEASED_COMPILE_DEFAULTS["glu_backend"] == "tokamax"
    assert "tokamax" in GLU_BACKENDS


def test_serial_default_reaches_the_model_as_the_fused_kernel(
    tmp_path: Path, monkeypatch
) -> None:
    seen: list[str] = []

    monkeypatch.setattr(
        api, "featurize", lambda **kwargs: (_fake_features(), "job", tmp_path)
    )
    monkeypatch.setattr(native, "load_params", lambda path: {"trunk": {}})

    def fake_predict(params, feats, key, **kwargs):
        seen.append(kwargs["glu_backend"])
        return {
            "sample_atom_coords": jnp.zeros((1, 3, 3)),
            "plddt": jnp.ones((1, 3)),
            "iptm": jnp.asarray([0.5]),
        }

    monkeypatch.setattr(predict_module, "boltz2_predict", fake_predict)

    api.predict(
        seq=["ACD"],
        weights=tmp_path / "boltz2_conf",
        mols=tmp_path,
        out_dir=tmp_path,
        write_fmt=None,
    )

    assert seen == ["tokamax"]


def test_explicit_xla_still_reaches_the_model(tmp_path: Path, monkeypatch) -> None:
    seen: list[str] = []

    monkeypatch.setattr(
        api, "featurize", lambda **kwargs: (_fake_features(), "job", tmp_path)
    )
    monkeypatch.setattr(native, "load_params", lambda path: {"trunk": {}})

    def fake_predict(params, feats, key, **kwargs):
        seen.append(kwargs["glu_backend"])
        return {
            "sample_atom_coords": jnp.zeros((1, 3, 3)),
            "plddt": jnp.ones((1, 3)),
            "iptm": jnp.asarray([0.5]),
        }

    monkeypatch.setattr(predict_module, "boltz2_predict", fake_predict)

    api.predict(
        seq=["ACD"],
        weights=tmp_path / "boltz2_conf",
        mols=tmp_path,
        out_dir=tmp_path,
        write_fmt=None,
        glu_backend="xla",
    )

    assert seen == ["xla"]


def test_backend_refuses_an_explicit_fused_glu_under_context_parallelism() -> None:
    with pytest.raises(ValueError, match="context parallelism requires"):
        Boltz2Backend().validate_native_options(
            {"glu_backend": "tokamax", "cp_devices": 2}
        )


@pytest.mark.parametrize(
    "options",
    [
        # Unset: the caller accepts whatever the run can partition.
        {"cp_devices": 2},
        # Explicitly the path the resolution lands on anyway.
        {"glu_backend": "xla", "cp_devices": 2},
        # The fused kernel is only refused *with* context parallelism.
        {"glu_backend": "tokamax"},
        {"glu_backend": "tokamax", "cp_devices": 1},
    ],
)
def test_backend_accepts_a_partitionable_glu_request(
    options: dict[str, object],
) -> None:
    Boltz2Backend().validate_native_options(options)


#: Four devices give both a 4-shard 1-D mesh and the smallest 2x2 grid, which
#: is the pair of layouts the transition takes different routes under.
_DEVICES = 4

#: A direct library caller is the door neither layer above can close. ``api``
#: and the adapter both judge the knob as a string before the run starts; a
#: caller who opens a mesh with ``context_parallel`` and calls a transition --
#: which is the shape the MSA module's own ``shard_map`` calls it in -- reaches
#: the kernel with neither of them consulted, and a Pallas kernel declares its
#: outputs with no ``manual_axis_type`` for a checked ``shard_map`` to accept.
#: Protenix and OpenFold3 already refuse at their GLU site
#: (``protenix/models/primitives/primitives.py``,
#: ``openfold3/models/primitives.py``) and this port's triangle multiplication
#: refuses at the top of its own forward; the transition was the one fused GLU
#: here with no site guard, and it is reached both from
#: ``primitives/transition.py``'s pair ``shard_map`` and from
#: ``trunk_blocks/msa.py``'s depth one.
#:
#: A forced device count has to be set before JAX initialises, so this runs in
#: a subprocess. The probe uses ``#`` comments rather than docstrings because it
#: lives inside a triple-quoted literal.
_POINT_OF_USE_PROBE = textwrap.dedent(
    """
    import jax
    import jax.numpy as jnp

    from foldjax.models._cp import context_parallel
    from foldjax.models.boltz2.models.primitives.transition import (
        transition_forward,
    )

    assert jax.device_count() == 4, jax.devices()

    N, C, WIDE = 4, 4, 8
    params = {
        "norm": {
            "scale": jnp.ones((C,), jnp.float32),
            "bias": jnp.zeros((C,), jnp.float32),
        },
        "fc1": {"kernel": jnp.zeros((C, WIDE), jnp.float32)},
        "fc2": {"kernel": jnp.zeros((C, WIDE), jnp.float32)},
        "fc3": {"kernel": jnp.zeros((WIDE, C), jnp.float32)},
    }
    x = jnp.zeros((1, N, N, C), jnp.float32)

    # `cp_pair` takes the pair `shard_map`; `cp_msa` takes the local-tile path
    # the MSA module calls from inside its own. Both reach the fused GLU.
    routes = ({"cp_pair": True}, {"cp_msa": True})

    for layout in ("1d", "2d"):
        with context_parallel(4, layout=layout):
            for route in routes:
                # The positive control: this configuration computes, so the
                # refusal below is attributable to the knob and not to the
                # shapes, the parameters or the mesh.
                value = transition_forward(params, x, glu_backend="xla", **route)
                assert value.shape == (1, N, N, C), (layout, route, value.shape)

                try:
                    transition_forward(params, x, glu_backend="tokamax", **route)
                except ValueError as error:
                    assert "cannot be partitioned" in str(error), (
                        layout,
                        route,
                        str(error),
                    )
                else:
                    raise AssertionError(("no refusal", layout, route))

    # Off the mesh the guard is silent. On a CPU the fused kernel then fails in
    # its own frames -- which is the failure the guard replaces, and the reason
    # this arm reads the message rather than asserting success.
    try:
        transition_forward(params, x, glu_backend="tokamax", cp_msa=True)
    except BaseException as error:
        assert "cannot be partitioned" not in str(error), str(error)

    print("BOLTZ2_GLU_POINT_OF_USE_OK")
    """
)


def test_a_direct_transition_refuses_the_fused_glu_under_a_mesh() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _POINT_OF_USE_PROBE],
        capture_output=True,
        text=True,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": f"--xla_force_host_platform_device_count={_DEVICES}",
            **inherited_environment(),
        },
        timeout=600,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "BOLTZ2_GLU_POINT_OF_USE_OK" in completed.stdout, completed.stdout
