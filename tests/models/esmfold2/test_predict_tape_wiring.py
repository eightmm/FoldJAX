"""Torch-free public predict tape wiring through real recurrence and sampler."""

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2.models import model as m
from tests.models.esmfold2.test_distogram_output_flag import _cheap_features


def _fixture(monkeypatch):
    batch, samples, atoms, tokens, width = 2, 3, 4, 2, 2
    features = {
        name: jnp.repeat(value, batch, axis=0)
        for name, value in _cheap_features().items()
    }
    for name in (
        "ref_pos",
        "ref_element",
        "ref_charge",
        "ref_atom_name_chars",
        "ref_space_uid",
        "atom_attention_mask",
        "atom_to_token",
    ):
        features[name] = jnp.repeat(features[name], 2, axis=1)
    features["msa"] = jnp.arange(batch * 4 * tokens).reshape(batch, 4, tokens) % 20
    features["msa_attention_mask"] = jnp.ones((batch, 4, tokens), bool)
    features["has_deletion"] = (
        jnp.arange(batch * 4 * tokens).reshape(batch, 4, tokens).astype(jnp.float32)
    )
    features["deletion_value"] = features["has_deletion"] + 10
    settings = replace(
        m.ModelSettings(),
        d_pair=width,
        d_inputs=width,
        trunk_n_layers=0,
        lm_encoder_n_layers=None,
        coda_n_layers=0,
        msa_n_layers=0,
        max_msa_depth=2,
        msa_column_mask_rate=0.2,
        num_recycles=1,
        num_samples=samples,
        trunk_dtype="float32",
        confidence_sample_sequential=False,
        diffusion=replace(
            m.ModelSettings().diffusion, num_steps=4, c_token=2, noise_scale=1.003
        ),
    )
    steps = len(m.diffusion.noise_schedule(settings.diffusion)) - 1
    shapes = {
        "initial_pair_state": (batch, tokens, tokens, width),
        "lm_dropout_masks": (2, batch, tokens, tokens, width),
        "msa_column_keep": (batch, tokens),
        "msa_row_choices": (2, 2),
        "diffusion_initial_normal": (batch * samples, atoms, 3),
        "diffusion_rotation_quaternions": (steps, batch * samples, 4),
        "diffusion_translations": (steps, batch * samples, 1, 3),
        "diffusion_churn_normals": (steps, batch * samples, atoms, 3),
    }
    rng = np.random.default_rng(7)
    tape = {
        name: jnp.asarray(rng.normal(size=shape), jnp.float32)
        for name, shape in shapes.items()
    }
    tape["lm_dropout_masks"] = tape["lm_dropout_masks"] > 0
    tape["msa_column_keep"] = jnp.array([[False, True], [True, False]])
    tape["msa_row_choices"] = jnp.array([[0, 1], [0, 3]], jnp.int32)
    params = {
        "parcae_log_delta": jnp.zeros(width),
        "parcae_log_a": jnp.zeros(width),
        "parcae_b_cont": jnp.eye(width),
        "parcae_input_norm.weight": jnp.ones(width),
        "parcae_input_norm.bias": jnp.zeros(width),
    }
    seen = {"loop": [], "msa": [], "dropout": [], "sample": []}
    monkeypatch.setattr(
        m, "inputs_embedding", lambda *a, **kw: jnp.ones((batch, tokens, width))
    )
    monkeypatch.setattr(
        m,
        "relative_position_encoding",
        lambda *a, **kw: jnp.zeros((batch, tokens, tokens, width)),
    )
    monkeypatch.setattr(
        m,
        "_token_bonds_encoding",
        lambda *a, **kw: jnp.zeros((batch, tokens, tokens, width)),
    )
    monkeypatch.setattr(m, "language_model_pair", lambda hidden, *a, **kw: hidden)
    monkeypatch.setattr(m, "linear", lambda value, *a: value)
    monkeypatch.setattr(m, "folding_trunk", lambda value, *a, **kw: value)

    def encoder(pair, single, one_hot, has_deletion, deletion, mask, *a, **kw):
        jax.debug.callback(
            lambda *values: seen["msa"].append(tuple(np.asarray(v) for v in values)),
            one_hot,
            has_deletion,
            deletion,
            mask,
            ordered=True,
        )
        return (
            pair
            + (one_hot.sum(axis=(2, 3)) + has_deletion.sum(axis=2))[:, :, None, None]
        )

    monkeypatch.setattr(m, "msa_encoder", encoder)
    original_loop, original_dropout, original_sample = (
        m.run_loops,
        m._dropout,
        m.diffusion.sample,
    )

    def loop(key, z, *a, **kw):
        jax.debug.callback(
            lambda *values: seen["loop"].append(tuple(np.asarray(v) for v in values)),
            z,
            kw["lm_dropout_masks"],
            kw["msa_row_choices"],
            ordered=True,
        )
        return original_loop(key, z, *a, **kw)

    def dropout(key, x, rate, **kw):
        jax.debug.callback(
            lambda value: seen["dropout"].append(np.asarray(value)),
            kw["keep_mask"],
            ordered=True,
        )
        return original_dropout(key, x, rate, **kw)

    def sample(*a, **kw):
        names = [name for name in shapes if name.startswith("diffusion_")]
        jax.debug.callback(
            lambda *values: seen["sample"].append(
                dict(zip(names, map(np.asarray, values), strict=True))
            ),
            *(kw[name] for name in names),
            ordered=True,
        )
        return original_sample(*a, **kw)

    monkeypatch.setattr(m, "run_loops", loop)
    monkeypatch.setattr(m, "_dropout", dropout)
    monkeypatch.setattr(m.diffusion, "sample", sample)
    monkeypatch.setattr(
        m.diffusion,
        "build_cache",
        lambda *a, **kw: SimpleNamespace(
            atom_mask=jnp.repeat(features["atom_attention_mask"], samples, axis=0),
            n_tokens=tokens,
        ),
    )
    monkeypatch.setattr(
        m.diffusion,
        "diffusion_module",
        lambda x, sigma, *a, **kw: (
            x / (1 + sigma[:, None, None]),
            jnp.zeros((batch * samples, tokens, 2)),
        ),
    )
    monkeypatch.setattr(m, "confidence_head", lambda *a, **kw: {})

    def run(key, draws, feats=None, **kwargs):
        return m.predict(
            key,
            features if feats is None else feats,
            params,
            settings=settings,
            lm_hidden_states=jnp.ones((batch, tokens, tokens, width)),
            n_chains=1,
            return_distogram_logits=False,
            **draws,
            **kwargs,
        )

    return run, tape, features, seen, settings


@pytest.mark.parametrize("compiled", [False, True])
def test_public_predict_all_eight_tapes_reach_real_consumers(monkeypatch, compiled):
    run, tape, features, seen, settings = _fixture(monkeypatch)
    fn = jax.jit(run) if compiled else run
    output = fn(jax.random.key(0), tape)
    jax.block_until_ready(output)
    assert output["sample_atom_coords"].shape == (6, 4, 3)
    np.testing.assert_array_equal(seen["loop"][0][0], tape["initial_pair_state"])
    np.testing.assert_array_equal(seen["loop"][0][1], tape["lm_dropout_masks"])
    np.testing.assert_array_equal(seen["loop"][0][2], tape["msa_row_choices"])
    np.testing.assert_array_equal(np.stack(seen["dropout"]), tape["lm_dropout_masks"])
    for name, value in seen["sample"][0].items():
        np.testing.assert_array_equal(value, tape[name])
    assert (
        tape["diffusion_churn_normals"].shape[0]
        == len(m.diffusion.noise_schedule(settings.diffusion)) - 1
    )
    keep = np.broadcast_to(
        np.asarray(tape["msa_column_keep"])[:, None, :], (2, 4, 2)
    ).copy()
    keep[:, 0, :] = True
    assert len(seen["msa"]) == 2
    for index, (one_hot, has_deletion, deletion, mask) in enumerate(seen["msa"]):
        rows = np.asarray(tape["msa_row_choices"])[index]
        np.testing.assert_array_equal(mask, np.swapaxes(keep[:, rows], 1, 2))
        expected_hot = np.eye(one_hot.shape[-1], dtype=np.float32)[
            np.asarray(features["msa"])[:, rows]
        ] * keep[:, rows, :, None]
        np.testing.assert_array_equal(one_hot, np.swapaxes(expected_hot, 1, 2))
        np.testing.assert_array_equal(
            has_deletion,
            np.swapaxes(np.asarray(features["has_deletion"])[:, rows], 1, 2),
        )
        np.testing.assert_array_equal(
            deletion, np.swapaxes(np.asarray(features["deletion_value"])[:, rows], 1, 2)
        )
    other = fn(jax.random.key(99), tape)
    np.testing.assert_array_equal(
        output["sample_atom_coords"], other["sample_atom_coords"]
    )


@pytest.mark.parametrize(
    "bad", ["loop_conflict", "missing_mask", "sample_axis", "schedule_axis"]
)
@pytest.mark.parametrize("compiled", [False, True])
def test_public_predict_rejects_bad_tape_wiring(monkeypatch, bad, compiled):
    run, tape, features, _, _ = _fixture(monkeypatch)
    if bad == "loop_conflict":
        features["msa_loop_tape"] = jnp.zeros((2, 2, 2, 2), jnp.int32)
    if bad == "missing_mask":
        features.pop("msa_attention_mask")
    if bad == "sample_axis":
        tape["diffusion_initial_normal"] = tape["diffusion_initial_normal"][:3]
    if bad == "schedule_axis":
        tape["diffusion_churn_normals"] = tape["diffusion_churn_normals"][:1]
    fn = jax.jit(run) if compiled else run
    with pytest.raises(ValueError):
        fn(jax.random.key(0), tape, features)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("bad", ["shape", "dtype"])
def test_public_predict_validates_initial_pair_before_network(
    monkeypatch, compiled, bad
):
    run, tape, _, _, _ = _fixture(monkeypatch)
    if bad == "shape":
        tape["initial_pair_state"] = tape["initial_pair_state"][:1]
    else:
        tape["initial_pair_state"] = tape["initial_pair_state"].astype(jnp.int32)
    fn = jax.jit(run) if compiled else run
    with pytest.raises(ValueError, match="initial pair state"):
        fn(jax.random.key(0), tape)


def test_public_eager_predict_rejects_nonfinite_initial_pair(monkeypatch):
    run, tape, _, _, _ = _fixture(monkeypatch)
    tape["initial_pair_state"] = tape["initial_pair_state"].at[0, 0, 0, 0].set(jnp.nan)
    with pytest.raises(ValueError, match="initial pair state must be finite"):
        run(jax.random.key(0), tape)
