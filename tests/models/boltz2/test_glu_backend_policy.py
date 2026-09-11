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
