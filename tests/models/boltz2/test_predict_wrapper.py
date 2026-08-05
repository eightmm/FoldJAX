"""Parity test: boltz2_predict equals calling the head functions directly.

Loads the torch-free native conf bundle (which now includes the confidence
head) and the real 1UBQ_A features, then asserts the wrapper's per-head outputs
exactly equal the individual head functions called on the same trunk/sample.
This proves the wrapper does not alter numerics. Marked slow (loads the full
~1.9GiB checkpoint bundle).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import foldjax.models.boltz2.models.predict as predict_module
from foldjax.models.boltz2.bridge.native import load_features_npz, load_params
from foldjax.models.boltz2.models.heads.bfactor import bfactor_forward
from foldjax.models.boltz2.models.heads.confidence import confidence_module_forward
from foldjax.models.boltz2.models.heads.distogram import distogram_forward
from foldjax.models.boltz2.models.predict import boltz2_predict
from foldjax.models.boltz2.models.trunk_blocks.trunk import (
    boltz2_sample_forward,
    boltz2_trunk_forward,
)
from foldjax.paths import weights_dir

# Both used to be read out of `../boltz_jax/outputs/`, a sibling checkout's
# output directory, which meant this test could only ever run on the machine
# that produced them. The weights now come from the FoldJAX weight store, which
# is where `foldjax weights fetch --model boltz2` puts exactly this file, and
# the features are a 5 MB fixture carried here.
WEIGHTS = weights_dir("boltz2") / "boltz2_conf"
FEATURES = Path(__file__).parent / "fixtures/1UBQ_A.npz"

RECYCLING = 0
STEPS = 4
SEED = 0


def test_predict_applies_compute_dtype_to_precomputed_trunk(monkeypatch) -> None:
    seen = {}

    def fake_trunk(params, feats, **kwargs):
        seen["param_dtype"] = params["kernel"].dtype
        seen["float_feat_dtype"] = feats["float_feat"].dtype
        seen["int_feat_dtype"] = feats["int_feat"].dtype
        return {
            "s_inputs": jnp.zeros((1, 1, 1), dtype=jnp.bfloat16),
            "s": jnp.zeros((1, 1, 1), dtype=jnp.bfloat16),
            "z": jnp.zeros((1, 1, 1, 1), dtype=jnp.bfloat16),
        }

    def fake_sample(params, feats, key, *, trunk, **kwargs):
        seen["sample_trunk_dtype"] = trunk["s"].dtype
        return {"sample_atom_coords": jnp.zeros((1, 1, 3))}

    monkeypatch.setattr(predict_module, "boltz2_trunk_forward", fake_trunk)
    monkeypatch.setattr(predict_module, "boltz2_sample_forward", fake_sample)

    predict_module.boltz2_predict(
        {"trunk": {"kernel": jnp.ones((1,), dtype=jnp.float32)}},
        {
            "float_feat": jnp.ones((1,), dtype=jnp.float32),
            "int_feat": jnp.ones((1,), dtype=jnp.int32),
        },
        jax.random.PRNGKey(0),
        run_confidence=False,
        run_distogram=False,
        compute_dtype=jnp.bfloat16,
    )

    assert seen == {
        "param_dtype": jnp.dtype(jnp.bfloat16),
        "float_feat_dtype": jnp.dtype(jnp.bfloat16),
        "int_feat_dtype": jnp.dtype(jnp.int32),
        "sample_trunk_dtype": jnp.dtype(jnp.bfloat16),
    }


def test_predict_can_map_confidence_sequentially(monkeypatch) -> None:
    trunk = {
        "s_inputs": jnp.zeros((1, 1, 1)),
        "s": jnp.zeros((1, 1, 1)),
        "z": jnp.zeros((1, 1, 1, 1)),
    }
    coords = jnp.arange(9, dtype=jnp.float32).reshape(3, 1, 3)

    monkeypatch.setattr(
        predict_module, "boltz2_trunk_forward", lambda *args, **kwargs: trunk
    )
    monkeypatch.setattr(
        predict_module,
        "boltz2_sample_forward",
        lambda *args, **kwargs: {"sample_atom_coords": coords},
    )
    monkeypatch.setattr(
        predict_module,
        "distogram_forward",
        lambda *args, **kwargs: jnp.zeros((1, 1, 1, 1, 1)),
    )
    monkeypatch.setattr(
        predict_module,
        "confidence_module_forward",
        lambda *args, x_pred, **kwargs: {"plddt": x_pred[:, 0, 0]},
    )

    out = predict_module.boltz2_predict(
        {"trunk": {}, "confidence": {}},
        {},
        jax.random.PRNGKey(0),
        multiplicity=3,
        run_distogram=False,
        confidence_sequentially=True,
        return_pair_chains_iptm=False,
    )

    np.testing.assert_array_equal(np.asarray(out["plddt"]), np.asarray(coords[:, 0, 0]))


def test_predict_runs_affinity_when_params_are_supplied(monkeypatch) -> None:
    trunk = {
        "s_inputs": jnp.zeros((1, 2, 3)),
        "s": jnp.zeros((1, 2, 3)),
        "z": jnp.zeros((1, 2, 2, 4)),
    }
    coords = jnp.zeros((1, 3, 3))
    seen = {}

    monkeypatch.setattr(
        predict_module, "boltz2_trunk_forward", lambda *args, **kwargs: trunk
    )
    monkeypatch.setattr(
        predict_module,
        "boltz2_sample_forward",
        lambda *args, **kwargs: {"sample_atom_coords": coords},
    )

    def fake_affinity(params, **kwargs):
        seen["params"] = params
        seen["multiplicity"] = kwargs["multiplicity"]
        return {"affinity_pred_value": jnp.asarray([3.25])}

    monkeypatch.setattr(predict_module, "affinity_module_forward", fake_affinity)

    affinity_params = {"head": {"kernel": jnp.ones((1,))}}
    out = predict_module.boltz2_predict(
        {"trunk": {}},
        {},
        jax.random.PRNGKey(0),
        run_confidence=False,
        run_distogram=False,
        affinity_params=affinity_params,
    )

    assert seen == {"params": affinity_params, "multiplicity": 1}
    np.testing.assert_array_equal(np.asarray(out["affinity_pred_value"]), [3.25])


def test_predict_runs_complete_affinity_ensemble_on_best_sample(monkeypatch) -> None:
    trunk = {
        "s_inputs": jnp.zeros((1, 3, 2)),
        "s": jnp.zeros((1, 3, 2)),
        "z": jnp.arange(18, dtype=jnp.float32).reshape(1, 3, 3, 2),
    }
    coords = jnp.arange(18, dtype=jnp.float32).reshape(2, 3, 3)
    feats = {
        "token_pad_mask": jnp.ones((1, 3)),
        "mol_type": jnp.asarray([[0, 3, 2]]),
        "affinity_token_mask": jnp.asarray([[0, 1, 0]]),
    }
    seen = []

    monkeypatch.setattr(
        predict_module, "boltz2_trunk_forward", lambda *args, **kwargs: trunk
    )
    monkeypatch.setattr(
        predict_module,
        "boltz2_sample_forward",
        lambda *args, **kwargs: {"sample_atom_coords": coords},
    )
    monkeypatch.setattr(
        predict_module,
        "distogram_forward",
        lambda *args, **kwargs: jnp.zeros((1, 3, 3, 1, 2)),
    )
    monkeypatch.setattr(
        predict_module,
        "confidence_module_forward",
        lambda *args, **kwargs: {"iptm": jnp.asarray([0.1, 0.9])},
    )

    def fake_input_embedder(params, feats, *, affinity, **kwargs):
        assert affinity is True
        return jnp.full((1, 3, 2), 7.0)

    monkeypatch.setattr(predict_module, "input_embedder_forward", fake_input_embedder)

    def fake_affinity(params, *, s_inputs, z, x_pred, **kwargs):
        seen.append((s_inputs, z, x_pred))
        marker = params["marker"]
        logit = jnp.log(3.0) if marker == 3.0 else jnp.asarray(0.0)
        return {
            "affinity_pred_value": jnp.asarray([[marker]]),
            "affinity_logits_binary": jnp.asarray([[logit]]),
        }

    monkeypatch.setattr(predict_module, "affinity_module_forward", fake_affinity)

    out = predict_module.boltz2_predict(
        {"trunk": {"input_embedder": {}}, "confidence": {}},
        feats,
        jax.random.PRNGKey(0),
        run_confidence=True,
        run_distogram=False,
        multiplicity=2,
        affinity_params={
            "ensemble": True,
            "mw_correction": False,
            "modules": [{"marker": 1.0}, {"marker": 3.0}],
        },
    )

    np.testing.assert_allclose(np.asarray(out["affinity_pred_value"]), [[2.0]])
    np.testing.assert_allclose(
        np.asarray(out["affinity_probability_binary"]), [[0.625]]
    )
    np.testing.assert_allclose(np.asarray(out["affinity_pred_value1"]), [[1.0]])
    np.testing.assert_allclose(np.asarray(out["affinity_pred_value2"]), [[3.0]])
    assert len(seen) == 2
    for s_inputs, z, x_pred in seen:
        np.testing.assert_array_equal(np.asarray(s_inputs), np.full((1, 3, 2), 7.0))
        np.testing.assert_array_equal(np.asarray(x_pred), np.asarray(coords[1:2, None]))
        assert np.all(np.asarray(z)[:, 2] == 0)
        assert np.all(np.asarray(z)[:, :, 2] == 0)


@pytest.fixture(scope="module")
def params():
    if not (WEIGHTS.with_suffix(".safetensors").exists() or
            WEIGHTS.with_suffix(".npz").exists()):
        pytest.skip(f"native weights not found: {WEIGHTS}")
    return load_params(WEIGHTS)


@pytest.fixture(scope="module")
def feats():
    if not FEATURES.exists():
        pytest.skip(f"features not found: {FEATURES}")
    return load_features_npz(FEATURES)


@pytest.mark.slow
def test_predict_matches_direct_head_calls(params, feats) -> None:
    key = jax.random.PRNGKey(SEED)

    out = boltz2_predict(
        params,
        feats,
        key,
        recycling_steps=RECYCLING,
        num_sampling_steps=STEPS,
        augmentation=False,
        run_confidence=True,
        run_distogram=True,
        run_bfactor=True,
    )

    # Reproduce the structure sampler with the identical key/args.
    sample = boltz2_sample_forward(
        params,
        feats,
        key,
        recycling_steps=RECYCLING,
        num_sampling_steps=STEPS,
        augmentation=False,
    )["sample_atom_coords"]
    np.testing.assert_array_equal(
        np.asarray(out["sample_atom_coords"]), np.asarray(sample)
    )

    trunk = boltz2_trunk_forward(params["trunk"], feats, recycling_steps=RECYCLING)
    pdistogram = distogram_forward(params, trunk["z"])
    np.testing.assert_array_equal(
        np.asarray(out["pdistogram"]), np.asarray(pdistogram)
    )

    pbfactor = bfactor_forward(params, trunk["s"])
    np.testing.assert_array_equal(
        np.asarray(out["pbfactor"]), np.asarray(pbfactor)
    )

    conf = confidence_module_forward(
        params["confidence"],
        s_inputs=trunk["s_inputs"],
        s=trunk["s"],
        z=trunk["z"],
        x_pred=sample,
        feats=feats,
        pred_distogram_logits=pdistogram[:, :, :, 0],
        multiplicity=1,
    )
    for k in ("plddt", "pae", "pde", "ptm", "iptm", "complex_plddt"):
        np.testing.assert_array_equal(
            np.asarray(out[k]), np.asarray(conf[k]), err_msg=f"mismatch {k}"
        )

    coords = np.asarray(out["sample_atom_coords"])
    assert np.isfinite(coords).all()
    assert jnp.all(jnp.isfinite(out["ptm"]))
