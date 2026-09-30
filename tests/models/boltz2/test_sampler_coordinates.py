"""The Boltz-2 sampler's coordinate path: re-centred every step, full FP32.

Upstream `AtomDiffusion.sample` centres, rotates and translates the
coordinates on every step (`diffusionv2.py:351-357`), and runs those products
and the Kabsch align at full FP32. The released port once switched the
augmentation off in `api.predict` -- its structures drifted to ~46 A off the
origin -- and ran both products under the TF32 scope, which doubled the
backbone bond-length noise. These tests pin the properties, not the spellings.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import foldjax.models.boltz2.api as api
import foldjax.models.boltz2.bridge.native as native
import foldjax.models.boltz2.models.predict as predict_module
import foldjax.models.boltz2.models.trunk_blocks.trunk as trunk_impl
from foldjax.models.boltz2.models.trunk_blocks.trunk import (
    _native_storage_mean,
    _rotate,
    _weighted_rigid_align,
    boltz2_sample_forward,
)


def _dot_precisions(function, *args) -> list[object]:
    with jax.default_matmul_precision("tensorfloat32"):
        jaxpr = jax.make_jaxpr(function)(*args)
    found = []

    def walk(inner):
        for eqn in inner.eqns:
            if eqn.primitive.name == "dot_general":
                found.append(eqn.params["precision"])
            for value in eqn.params.values():
                if isinstance(value, jax.extend.core.ClosedJaxpr):
                    walk(value.jaxpr)
                elif isinstance(value, jax.extend.core.Jaxpr):
                    walk(value)

    walk(jaxpr.jaxpr)
    return found


def _is_highest(precision) -> bool:
    highest = jax.lax.Precision.HIGHEST
    return precision == highest or precision == (highest, highest)


def test_coordinate_products_run_at_full_fp32_under_a_tf32_scope() -> None:
    coords = jax.random.normal(jax.random.PRNGKey(0), (2, 64, 3)) * 40.0
    target = coords + 0.1
    mask = jnp.ones((2, 64), dtype=jnp.float32)
    rotation = jnp.broadcast_to(jnp.eye(3), (2, 3, 3))

    align = _dot_precisions(_weighted_rigid_align, coords, target, mask, mask)
    rotate = _dot_precisions(_rotate, coords, rotation)

    assert align and rotate
    assert all(_is_highest(p) for p in align + rotate), align + rotate


def test_native_storage_mean_is_upstream_mean_and_ignores_a_serving_suffix() -> None:
    real, native, target = 40, 64, 96
    coords = jax.random.normal(jax.random.PRNGKey(1), (2, native, 3)) * 30.0
    mask = jnp.asarray([[1.0] * real + [0.0] * (native - real)] * 2)

    # Unpadded: upstream's `atom_coords.mean(dim=-2)` over the native storage,
    # window-padding atoms included.
    np.testing.assert_array_equal(
        np.asarray(_native_storage_mean(coords, mask)),
        np.asarray(coords.mean(axis=-2, keepdims=True)),
    )

    # A serving suffix, however far away, must not move the centre.
    suffix = jnp.full((2, target - native, 3), 1.0e4, dtype=coords.dtype)
    padded = jnp.concatenate([coords, suffix], axis=1)
    padded_mask = jnp.concatenate([mask, jnp.zeros((2, target - native))], axis=1)
    np.testing.assert_allclose(
        np.asarray(_native_storage_mean(padded, padded_mask)),
        np.asarray(coords.mean(axis=-2, keepdims=True)),
        rtol=0,
        atol=1e-5,
    )


def _drifting_sampler_inputs(atoms: int):
    params = {
        "trunk": {},
        "conditioned_diffusion": {"diffusion_conditioning": {}, "score_model": {}},
    }
    feats = {
        "atom_pad_mask": jnp.ones((1, atoms), dtype=jnp.float32),
        "token_pad_mask": jnp.ones((1, 2), dtype=jnp.float32),
    }
    trunk = {
        "s_inputs": jnp.zeros((1, 2, 1), dtype=jnp.float32),
        "s": jnp.zeros((1, 2, 1), dtype=jnp.float32),
        "z": jnp.zeros((1, 2, 2, 1), dtype=jnp.float32),
        "relative_position_encoding": jnp.zeros((1, 2, 2, 1), dtype=jnp.float32),
    }
    return params, feats, trunk


@pytest.mark.parametrize("augmentation", [True, False])
def test_the_denoiser_always_sees_centred_coordinates(
    monkeypatch: pytest.MonkeyPatch, augmentation: bool
) -> None:
    """A denoiser with a constant pull drifts an un-centred sampler away.

    The late steps carry almost no noise, so what the denoiser is handed there
    is the carried state: centred (to within the ~1 A random translation) when
    the sampler re-centres, 10+ A off when it does not.
    """
    seen: list[np.ndarray] = []
    monkeypatch.setattr(
        trunk_impl, "diffusion_conditioning_forward", lambda *_a, **_k: {}
    )

    def denoise(*_args, r_noisy, **_kwargs):
        seen.append(np.asarray(r_noisy))
        return r_noisy + jnp.asarray([4.0, 0.0, 0.0], r_noisy.dtype)

    monkeypatch.setattr(trunk_impl, "_preconditioned_score_forward", denoise)
    params, feats, trunk = _drifting_sampler_inputs(atoms=64)

    boltz2_sample_forward(
        params,
        feats,
        jax.random.PRNGKey(5),
        num_sampling_steps=12,
        multiplicity=2,
        augmentation=augmentation,
        use_scan=False,
        trunk=trunk,
    )

    late = np.stack(seen[-4:])  # [steps, samples, atoms, 3]
    offset = np.linalg.norm(late.mean(axis=-2), axis=-1).max()
    if augmentation:
        assert offset < 6.0, offset
    else:
        assert offset > 10.0, offset


def test_predict_forwards_the_per_step_augmentation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[bool] = []
    features = {
        "atom_pad_mask": np.ones((1, 3), dtype=np.float32),
        "token_pad_mask": np.ones((1, 2), dtype=np.float32),
        "mol_type": np.asarray([[0, 0]], dtype=np.int32),
        "token_to_rep_atom": np.asarray([[[1, 0, 0], [0, 0, 1]]], dtype=np.int64),
        "atom_to_token": np.asarray([[[1, 0], [1, 0], [0, 1]]], dtype=np.int64),
        "affinity_token_mask": np.zeros((1, 2), dtype=np.float32),
    }
    monkeypatch.setattr(api, "featurize", lambda **_k: (features, "job", tmp_path))
    monkeypatch.setattr(native, "load_params", lambda path: {"trunk": {}})

    def fake_predict(params, feats, key, **kwargs):
        seen.append(kwargs["augmentation"])
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

    assert seen == [True]
