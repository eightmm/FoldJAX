"""Export cropping must not compile each real length as a device slice."""

from __future__ import annotations

from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import compiler

from foldjax.models.boltz2.data.bucket import crop_prediction_outputs
from foldjax.models.openfold3.output import crop_prediction
from foldjax.models.protenix.data.padding import crop_protenix_outputs
from foldjax.padding import PaddingPlan


def _case(model):
    if model == "protenix":
        shapes = {
            "coordinate": (2, 5, 3),
            "plddt": (2, 5, 2),
            "atom_plddt": (2, 5),
            "contact_probs": (2, 4, 4),
            "pae": (2, 4, 4, 3),
            "s_trunk": (4, 3),
            "z_trunk": (4, 4, 3),
        }
        indices = {
            "coordinate": np.s_[..., :3, :],
            "plddt": np.s_[..., :3, :],
            "atom_plddt": np.s_[..., :3],
            "contact_probs": np.s_[..., :2, :2],
            "pae": np.s_[..., :2, :2, :],
            "s_trunk": np.s_[..., :2, :],
            "z_trunk": np.s_[..., :2, :2, :],
        }

        def crop(values):
            return crop_protenix_outputs(
                values,
                PaddingPlan(
                    actual={"tokens": 2, "atoms": 3},
                    target={"tokens": 4, "atoms": 5},
                ),
            )

    elif model == "boltz2":
        shapes = {
            "sample_atom_coords": (2, 5, 3),
            "single": (1, 4, 3),
            "plddt": (1, 4),
            "pair": (1, 4, 4, 3),
            "pae": (1, 4, 4),
        }
        indices = {
            "sample_atom_coords": np.s_[..., :3, :],
            "single": np.s_[:, :2, :],
            "plddt": np.s_[:, :2],
            "pair": np.s_[:, :2, :2, :],
            "pae": np.s_[:, :2, :2],
        }

        def crop(values):
            return crop_prediction_outputs(values, original_tokens=2, original_atoms=3)

    else:
        shapes = {
            "coordinates": (2, 5, 3),
            "plddt": (2, 5),
            "plddt_logits": (2, 5, 3),
            "pae_logits": (2, 4, 4, 3),
            "pae": (2, 4, 4),
        }
        indices = {
            "coordinates": np.s_[..., :3, :],
            "plddt": np.s_[..., :3],
            "plddt_logits": np.s_[..., :3, :],
            "pae_logits": np.s_[..., :2, :2, :],
            "pae": np.s_[..., :2, :2],
        }

        def crop(values):
            prediction = namedtuple("ExportPrediction", values)(**values)
            return crop_prediction(
                prediction,
                {"token_mask": [[1, 1, 0, 0]], "atom_mask": [[1, 1, 1, 0, 0]]},
            )._asdict()

    return crop, shapes, indices


class DeviceExport:
    def __init__(self, value):
        self.value = value
        self.transfers = 0

    def __array__(self, dtype=None, copy=None):
        self.transfers += 1
        return np.asarray(self.value, dtype=dtype)

    def __getitem__(self, key):
        raise AssertionError("export cropping must transfer before device indexing")


@pytest.mark.parametrize("model", ("protenix", "boltz2", "openfold3"))
@pytest.mark.parametrize("dtype", (np.float32, jnp.bfloat16))
def test_export_crops_transfer_each_recognized_array_once(model, dtype) -> None:
    crop, shapes, indices = _case(model)
    values = {
        name: np.arange(np.prod(shape)).reshape(shape).astype(dtype)
        for name, shape in shapes.items()
    }
    device_values = {name: DeviceExport(value) for name, value in values.items()}
    untouched = DeviceExport(np.zeros((4, 4), dtype=np.float32))
    result = crop({**device_values, "unrecognized": untouched})

    assert result["unrecognized"] is untouched
    assert untouched.transfers == 0
    for name, value in values.items():
        np.testing.assert_array_equal(result[name], value[indices[name]])
        assert result[name].dtype == value.dtype
        assert device_values[name].transfers == 1
        assert result[name].base is None
        assert not np.shares_memory(result[name], value)


@pytest.mark.parametrize("model", ("protenix", "boltz2", "openfold3"))
def test_already_unpadded_host_exports_keep_their_storage(model) -> None:
    crop, shapes, indices = _case(model)
    values = {
        name: np.arange(np.prod(shape)).reshape(shape)[indices[name]].copy()
        for name, shape in shapes.items()
    }
    device_values = {name: DeviceExport(value) for name, value in values.items()}

    result = crop(device_values)

    for name, value in values.items():
        np.testing.assert_array_equal(result[name], value)
        assert np.shares_memory(result[name], value)
        assert device_values[name].transfers == 1


@pytest.mark.parametrize("model", ("protenix", "boltz2", "openfold3"))
def test_real_jax_export_crops_do_not_compile_slice_programs(
    monkeypatch, model
) -> None:
    crop, shapes, indices = _case(model)
    values = {
        name: np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        for name, shape in shapes.items()
    }
    device_values = {name: jax.device_put(value) for name, value in values.items()}

    def forbid_compile(*args, **kwargs):
        raise AssertionError("host output cropping must not compile")

    monkeypatch.setattr(compiler, "compile_or_get_cached", forbid_compile)
    result = crop(device_values)

    for name, value in values.items():
        assert isinstance(result[name], np.ndarray)
        np.testing.assert_array_equal(result[name], value[indices[name]])
        assert result[name].base is None
