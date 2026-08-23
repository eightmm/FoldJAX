"""OpenFold3's complete token/atom/MSA/template serving profile."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.backends.openfold3 import _padding_plan, _sampler_noise_mask
from foldjax.models.openfold3.data import (
    collapse_identical_templates,
    compact_zero_template_pair_features,
    pad_features,
)
from foldjax.models.openfold3.data.featurize import _ZERO_TEMPLATE_PAIR_MARKER
from foldjax.models.openfold3.data.validation import validate_features
from foldjax.models.openfold3.inference import Prediction
from foldjax.models.openfold3.models import template_module
from foldjax.models.openfold3.models.sampler import (
    padded_noise_tape,
    sample_diffusion,
)
from foldjax.models.openfold3.output import crop_prediction
from foldjax.schema import PaddingConfig
from tests.models.openfold3.feature_fixture import minimal_features


def test_padding_covers_all_four_dynamic_axes() -> None:
    features = minimal_features(tokens=4, atoms=7, msa_rows=2, templates=1)
    padded = pad_features(
        features, n_token=8, n_atom=13, n_msa=5, n_templates=3
    )

    assert padded["token_mask"].shape == (1, 8)
    assert padded["atom_mask"].shape == (1, 13)
    assert padded["msa"].shape == (1, 5, 8, 32)
    assert padded["msa_mask"].shape == (1, 5, 8)
    assert padded["template_restype"].shape == (1, 3, 8, 32)
    assert padded["template_distogram"].shape == (1, 3, 8, 8, 39)
    assert padded["template_unit_vector"].shape == (1, 3, 8, 8, 3)
    assert float(padded["token_mask"].sum()) == 4
    assert float(padded["atom_mask"].sum()) == 7
    assert float(padded["msa_mask"].sum()) == 8
    assert float(padded["template_backbone_frame_mask"].sum()) == 4
    np.testing.assert_array_equal(
        padded["template_padding_mask"], np.asarray([[1.0, 0.0, 0.0]])
    )


def test_zero_template_compaction_runs_after_final_padding_shape() -> None:
    features = minimal_features(tokens=4, atoms=7, msa_rows=2, templates=4)
    features["template_pseudo_beta_mask"].fill(0.0)
    features["template_backbone_frame_mask"].fill(0.0)
    collapsed = collapse_identical_templates(features)
    assert collapsed["template_restype"].shape == (1, 1, 4, 32)
    padded = pad_features(
        collapsed, n_token=8, n_atom=13, n_msa=5, n_templates=3
    )

    compact = compact_zero_template_pair_features(padded)

    assert compact["template_restype"].shape == (1, 3, 8, 32)
    np.testing.assert_array_equal(
        compact["template_padding_mask"], np.asarray([[1.0, 0.0, 0.0]])
    )
    assert compact[_ZERO_TEMPLATE_PAIR_MARKER].shape == ()
    for name in (
        "template_distogram",
        "template_pseudo_beta_mask",
        "template_unit_vector",
        "template_backbone_frame_mask",
    ):
        assert name not in compact
    validate_features(padded)

    repadded = pad_features(
        padded, n_token=8, n_atom=13, n_msa=5, n_templates=4
    )
    np.testing.assert_array_equal(
        repadded["template_padding_mask"],
        np.asarray([[1.0, 0.0, 0.0, 0.0]]),
    )


def test_padding_plan_reports_real_storage_and_standard_targets() -> None:
    features = minimal_features(tokens=4, atoms=7, msa_rows=2, templates=1)
    plan = _padding_plan(features, PaddingConfig())

    assert plan.actual == {"tokens": 4, "atoms": 7, "msa": 2, "templates": 1}
    assert plan.storage == plan.actual
    assert plan.target == {
        "tokens": 256,
        "atoms": 256,
        "msa": 64,
        "templates": 1,
    }


def test_existing_archive_padding_is_never_shrunk() -> None:
    features = minimal_features(tokens=4, atoms=7, msa_rows=2, templates=1)
    padded = pad_features(
        features, n_token=8, n_atom=13, n_msa=5, n_templates=3
    )
    plan = _padding_plan(padded, PaddingConfig())

    assert plan.actual == {"tokens": 4, "atoms": 7, "msa": 2, "templates": 1}
    assert plan.storage == {"tokens": 8, "atoms": 13, "msa": 5, "templates": 3}
    assert all(plan.target[name] >= size for name, size in plan.storage.items())


def test_public_crop_removes_atom_and_pair_padding() -> None:
    features = pad_features(
        minimal_features(tokens=4, atoms=7),
        n_token=8,
        n_atom=13,
        n_msa=2,
        n_templates=1,
    )
    prediction = Prediction(
        coordinates=np.zeros((2, 13, 3), dtype=np.float32),
        plddt=np.zeros((2, 13), dtype=np.float32),
        ptm=np.zeros(2, dtype=np.float32),
        iptm=np.zeros(2, dtype=np.float32),
        chain_pair_iptm=None,
        pae_logits=np.zeros((2, 8, 8, 64), dtype=np.float32),
        pde_logits=np.zeros((2, 8, 8, 64), dtype=np.float32),
        distogram_logits=np.zeros((1, 8, 8, 64), dtype=np.float32),
        experimentally_resolved_logits=np.zeros((2, 13, 2), dtype=np.float32),
    )

    cropped = crop_prediction(prediction, features)

    assert cropped.coordinates.shape == (2, 7, 3)
    assert cropped.plddt.shape == (2, 7)
    assert cropped.pae_logits.shape == (2, 4, 4, 64)
    assert cropped.pde_logits.shape == (2, 4, 4, 64)
    assert cropped.distogram_logits.shape == (1, 4, 4, 64)
    assert cropped.experimentally_resolved_logits.shape == (2, 7, 2)


def test_crop_rejects_interleaved_padding() -> None:
    features = minimal_features(tokens=4, atoms=7)
    features["token_mask"] = np.asarray([[1, 0, 1, 1]], dtype=np.float32)
    prediction = Prediction(
        coordinates=np.zeros((1, 7, 3), dtype=np.float32),
        plddt=np.zeros((1, 7), dtype=np.float32),
        ptm=np.zeros(1, dtype=np.float32),
        iptm=np.zeros(1, dtype=np.float32),
        chain_pair_iptm=None,
        pae_logits=None,
        pde_logits=None,
        distogram_logits=None,
    )

    with pytest.raises(ValueError, match="token padding must be a suffix"):
        crop_prediction(prediction, features)


def test_plan_rejects_interleaved_archive_padding_before_execution() -> None:
    features = minimal_features(tokens=4, atoms=7)
    features["msa_mask"] = np.asarray(
        [[[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]]],
        dtype=np.float32,
    )
    for name in ("msa", "has_deletion", "deletion_value"):
        value = features[name]
        features[name] = np.concatenate(
            [value[:, :1], np.zeros_like(value[:, :1]), value[:, :1]], axis=1
        )

    with pytest.raises(ValueError, match="MSA-row padding"):
        _padding_plan(features, PaddingConfig())


def test_padded_sampler_tape_keeps_the_exact_real_atom_prefix() -> None:
    import jax

    exact = padded_noise_tape(
        jax.random.key(17),
        n_steps=3,
        n_sample=2,
        actual_atoms=5,
        target_atoms=5,
    )
    padded = padded_noise_tape(
        jax.random.key(17),
        n_steps=3,
        n_sample=2,
        actual_atoms=5,
        target_atoms=11,
    )

    np.testing.assert_array_equal(np.asarray(padded[:, :, :5]), np.asarray(exact))
    np.testing.assert_array_equal(
        np.asarray(padded[:, :, 5:]), np.zeros((4, 2, 6, 3), np.float32)
    )


def test_masked_sampler_keeps_unpadded_coordinates_and_zeros_suffix() -> None:
    import jax

    schedule = jnp.asarray([2.0, 1.0], dtype=jnp.float32)
    kwargs = {
        "noise_schedule": schedule,
        "denoise_fn": lambda coordinates, _t: coordinates,
        "gamma_0": 0.0,
        "gamma_min": 1.0,
        "noise_scale": 1.0,
        "step_scale": 1.0,
        "augment_fn": None,
    }
    exact = sample_diffusion(jax.random.key(19), shape=(2, 5, 3), **kwargs)
    mask = jnp.asarray([[1] * 5 + [0] * 6] * 2, dtype=bool)
    padded = sample_diffusion(
        jax.random.key(19),
        shape=(2, 11, 3),
        noise_mask=mask,
        **kwargs,
    )

    np.testing.assert_array_equal(np.asarray(padded[:, :5]), np.asarray(exact))
    np.testing.assert_array_equal(
        np.asarray(padded[:, 5:]), np.zeros((2, 6, 3), np.float32)
    )


def test_sampler_mask_preserves_existing_storage_stride_across_samples() -> None:
    import jax

    features = pad_features(
        minimal_features(tokens=4, atoms=7),
        n_token=8,
        n_atom=13,
        n_msa=2,
        n_templates=1,
    )
    plan = _padding_plan(features, PaddingConfig(atoms=32))
    noise_mask = _sampler_noise_mask(plan, num_samples=2)

    assert np.all(noise_mask[:, :13])
    assert not np.any(noise_mask[:, 13:])

    schedule = jnp.asarray([2.0, 1.0], dtype=jnp.float32)
    kwargs = {
        "noise_schedule": schedule,
        "denoise_fn": lambda coordinates, _t: coordinates,
        "gamma_0": 0.0,
        "gamma_min": 1.0,
        "noise_scale": 1.0,
        "step_scale": 1.0,
        "augment_fn": None,
    }
    exact = sample_diffusion(jax.random.key(29), shape=(2, 13, 3), **kwargs)
    padded = sample_diffusion(
        jax.random.key(29),
        shape=(2, 32, 3),
        noise_mask=jnp.asarray(noise_mask),
        **kwargs,
    )

    np.testing.assert_array_equal(np.asarray(padded[:, :13]), np.asarray(exact))
    np.testing.assert_array_equal(
        np.asarray(padded[:, 13:]), np.zeros((2, 19, 3), np.float32)
    )


@pytest.mark.parametrize("scan_templates", [True, False])
def test_template_padding_rows_are_neutral_in_both_reduction_paths(
    monkeypatch, scan_templates: bool
) -> None:
    values = jnp.asarray([2.0, 4.0, 100.0], dtype=jnp.float32).reshape(
        1, 3, 1, 1, 1
    )
    batch = {
        "template_values": values,
        "template_padding_mask": jnp.asarray([[1.0, 1.0, 0.0]]),
    }
    params = SimpleNamespace(
        template_pair_embedder=object(),
        template_pair_stack=object(),
        linear_t=object(),
    )
    monkeypatch.setattr(
        template_module,
        "template_pair_embedder",
        lambda batch, _z, _params, **_kwargs: batch["template_values"],
    )
    monkeypatch.setattr(
        template_module,
        "template_pair_stack",
        lambda value, _params, **_kwargs: value,
    )
    monkeypatch.setattr(
        template_module, "linear", lambda value, _params: value
    )

    result = template_module.template_embedder(
        batch,
        jnp.zeros((1, 1, 1, 1), dtype=jnp.float32),
        params,
        pair_mask=jnp.ones((1, 1, 1), dtype=jnp.float32),
        no_heads=1,
        scan_templates=scan_templates,
    )

    np.testing.assert_array_equal(np.asarray(result), np.asarray([[[[3.0]]]]))


@pytest.mark.parametrize("scan_templates", [True, False])
def test_template_reduction_without_padding_mask_keeps_legacy_average(
    monkeypatch, scan_templates: bool
) -> None:
    values = jnp.asarray([2.0, 4.0, 99.0], dtype=jnp.float32).reshape(
        1, 3, 1, 1, 1
    )
    batch = {"template_values": values}
    params = SimpleNamespace(
        template_pair_embedder=object(),
        template_pair_stack=object(),
        linear_t=object(),
    )
    monkeypatch.setattr(
        template_module,
        "template_pair_embedder",
        lambda batch, _z, _params, **_kwargs: batch["template_values"],
    )
    monkeypatch.setattr(
        template_module,
        "template_pair_stack",
        lambda value, _params, **_kwargs: value,
    )
    monkeypatch.setattr(
        template_module, "linear", lambda value, _params: value
    )

    result = template_module.template_embedder(
        batch,
        jnp.zeros((1, 1, 1, 1), dtype=jnp.float32),
        params,
        pair_mask=jnp.ones((1, 1, 1), dtype=jnp.float32),
        no_heads=1,
        scan_templates=scan_templates,
    )

    np.testing.assert_array_equal(np.asarray(result), np.asarray([[[[35.0]]]]))
