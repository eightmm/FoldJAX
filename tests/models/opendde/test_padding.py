from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.backends.opendde import OpenDDEBackend, _shape_profile
from foldjax.models.opendde import postprocess as postprocess_impl
from foldjax.models.opendde.cli import predict as predict_impl
from foldjax.models.opendde.data.padding import (
    crop_opendde_outputs,
    pad_opendde_features,
    select_opendde_model_features,
)
from foldjax.models.opendde.models import model as model_impl
from foldjax.models.opendde.models.geometry import uniform_random_rotations
from foldjax.models.opendde.models.msa_sampling import (
    pad_opendde_msa_cycle_features,
    sample_opendde_msa_cycle_features,
)
from foldjax.models.opendde.models.sampling import make_padded_random_tapes
from foldjax.models.protenix.models.primitives.primitives import (
    LayerNormParams,
    LinearParams,
)
from foldjax.models.protenix.models.trunk_blocks.msa import (
    OuterProductMeanParams,
    outer_product_mean,
)
from foldjax.schema import PaddingConfig, PredictionRequest


def _params_double():
    """A weights stand-in shaped like the real parameter tree.

    `object()` was enough while OpenDDE shipped float32, because that path
    casts nothing and never touches the tree. The shipped trunk is bfloat16
    since 2026-08-28, so `cast_trunk_params` runs on every default prediction
    -- these flows had simply never exercised it. Empty subtrees keep the
    double cheap: `jax.tree.map` over `{}` is a no-op, and `_replace` is what
    the cast actually needs.
    """
    from foldjax.models.opendde.models.model import OpenDDEInferenceParams

    return OpenDDEInferenceParams(
        **{name: {} for name in OpenDDEInferenceParams._fields}
    )


def _msa_features(depth: int, tokens: int = 3) -> dict[str, np.ndarray]:
    msa = np.arange(depth * tokens, dtype=np.int64).reshape(depth, tokens) % 31
    return {
        "msa": msa,
        "has_deletion": msa.astype(np.float32) + 100.0,
        "deletion_value": msa.astype(np.float32) + 200.0,
    }


def _sampled_cycles(
    depth: int,
    *,
    tokens: int = 3,
    cycles: int = 2,
) -> tuple[dict[str, np.ndarray], ...]:
    return sample_opendde_msa_cycle_features(
        _msa_features(depth, tokens),
        num_recycles=cycles,
        seed=17,
    )


def test_opendde_msa_padding_preserves_sampled_prefix_and_masks_suffix() -> None:
    sampled = _sampled_cycles(5)

    padded, plan = pad_opendde_msa_cycle_features(sampled, PaddingConfig())

    assert plan.summary() == {
        "actual": {"msa": 5},
        "storage": {"msa": 5},
        "target": {"msa": 64},
        "changed": True,
    }
    for before, after in zip(sampled, padded, strict=True):
        for name in ("msa", "has_deletion", "deletion_value", "msa_mask"):
            np.testing.assert_array_equal(after[name][:5], before[name])
        assert after["msa"].shape == (64, 3)
        np.testing.assert_array_equal(after["msa"][5:], 31)
        np.testing.assert_array_equal(after["has_deletion"][5:], 0)
        np.testing.assert_array_equal(after["deletion_value"][5:], 0)
        np.testing.assert_array_equal(after["msa_mask"][5:], 0)


def test_opendde_msa_padding_reports_real_storage_and_target_depths() -> None:
    cycle = _sampled_cycles(4, cycles=1)[0]
    cycle = {name: value.copy() for name, value in cycle.items()}
    cycle["msa_mask"][2:] = 0

    (padded,), plan = pad_opendde_msa_cycle_features(
        (cycle,),
        PaddingConfig(msa=8),
    )

    assert plan.actual == {"msa": 2}
    assert plan.storage == {"msa": 4}
    assert plan.target == {"msa": 8}
    np.testing.assert_array_equal(padded["msa_mask"][:2], 1)
    np.testing.assert_array_equal(padded["msa_mask"][2:], 0)


def test_opendde_padded_msa_rows_cannot_change_the_pair_update() -> None:
    params = OuterProductMeanParams(
        layer_norm=LayerNormParams(),
        linear_1=LinearParams(jnp.asarray([[1.0, -0.5]])),
        linear_2=LinearParams(jnp.asarray([[0.25, 2.0]])),
        linear_out=LinearParams(jnp.asarray([[1.5]])),
    )
    real = jnp.asarray(
        [
            [[1.0, 3.0], [2.0, -1.0]],
            [[-2.0, 0.5], [4.0, 2.0]],
            [[0.25, -3.0], [1.5, 5.0]],
        ],
        dtype=jnp.float32,
    )
    real_mask = jnp.ones(real.shape[:2], dtype=jnp.float32)
    adversarial = jnp.full((5, 2, 2), 1.0e6, dtype=jnp.float32)
    padded = jnp.concatenate((real, adversarial), axis=0)
    padded_mask = jnp.pad(real_mask, ((0, 5), (0, 0)))

    expected = outer_product_mean(real, real_mask, params)
    actual = outer_product_mean(padded, padded_mask, params)

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=1e-6,
        atol=1e-6,
    )


def test_opendde_msa_padding_refuses_shrink_nonprefix_and_grid_overflow() -> None:
    sampled = _sampled_cycles(3, cycles=1)
    with pytest.raises(ValueError, match="smaller than the input size 3"):
        pad_opendde_msa_cycle_features(sampled, PaddingConfig(msa=2))

    broken = {name: value.copy() for name, value in sampled[0].items()}
    broken["msa_mask"][1] = 0
    with pytest.raises(ValueError, match="contiguous prefix"):
        pad_opendde_msa_cycle_features((broken,), PaddingConfig(msa=4))

    over_grid = sample_opendde_msa_cycle_features(
        _msa_features(16385, tokens=1),
        num_recycles=1,
        seed=17,
        msa_depth=20000,
    )
    with pytest.raises(ValueError, match="largest standard bucket 16384"):
        pad_opendde_msa_cycle_features(over_grid, PaddingConfig())
    _padded, plan = pad_opendde_msa_cycle_features(
        over_grid,
        PaddingConfig(overflow="exact"),
    )
    assert plan.target == {"msa": 16385}


def test_opendde_advertises_all_masked_axes_and_rejects_templates(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "job.json"
    input_path.write_text("{}", encoding="utf-8")
    backend = OpenDDEBackend()

    assert backend.capabilities().padding_axes == (
        "tokens",
        "atoms",
        "msa",
        "structural_tokens",
    )
    request = PredictionRequest(
        model="opendde",
        input=input_path,
        input_format="opendde",
        padding=PaddingConfig(templates=4),
    )
    with pytest.raises(ValueError, match="explicit padding axes: templates"):
        backend.validate_request(request)


def test_opendde_backend_forwards_padding_and_returns_concrete_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "job.json"
    weights_path = tmp_path / "weights.npz"
    input_path.write_text("{}", encoding="utf-8")
    weights_path.write_bytes(b"native")
    seen: dict[str, object] = {}
    expected = {
        "actual": {"msa": 5},
        "storage": {"msa": 5},
        "target": {"msa": 64},
        "changed": True,
    }

    def native_main(argv, **kwargs):
        seen["argv"] = argv
        seen.update(kwargs)
        kwargs["padding_profiles"].append(expected)
        return []

    monkeypatch.setattr(
        "foldjax.backends.opendde.import_module",
        lambda _name: SimpleNamespace(main=native_main),
    )
    result = OpenDDEBackend().predict(
        PredictionRequest(
            model="opendde",
            input=input_path,
            weights=weights_path,
            output_dir=tmp_path / "out",
            padding=True,
        )
    )

    assert seen["padding"] == PaddingConfig()
    assert result.shape_profile == expected
    assert result.raw["padding"] == expected


def test_native_cli_pads_after_sampling_and_reports_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "job.json"
    weights_path = tmp_path / "weights.npz"
    input_path.write_text(json.dumps([{"name": "job"}]), encoding="utf-8")
    weights_path.write_bytes(b"native")
    features = _opendde_features(msa_depth=5)
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        predict_impl,
        "_load_jobs",
        lambda _path: [{"name": "job"}],
    )
    monkeypatch.setattr(
        predict_impl, "_load_weights", lambda _path: _params_double()
    )
    monkeypatch.setattr(predict_impl, "_featurize", lambda _job, **_kwargs: features)

    def fake_predict(_features, _params, **kwargs):
        seen.update(kwargs)
        return {"coordinate": np.zeros((1, 3, 3), dtype=np.float32)}

    monkeypatch.setattr(predict_impl, "_predict", fake_predict)
    monkeypatch.setattr(
        predict_impl,
        "_score",
        lambda output, _features, *, num_recycles: output,
    )
    output_path = tmp_path / "out" / "job.cif"
    monkeypatch.setattr(
        predict_impl,
        "_write",
        lambda _root, **_kwargs: [output_path],
    )
    profiles: list[dict[str, object]] = []

    predict_impl.main(
        [
            "--input-json",
            str(input_path),
            "--weights",
            str(weights_path),
            "--out",
            str(tmp_path / "out"),
            "--n-sample",
            "1",
            "--n-step",
            "1",
            "--n-cycle",
            "2",
        ],
        padding=PaddingConfig(msa=64),
        padding_profiles=profiles,
    )

    cycles = seen["cycle_msa_features"]
    assert len(cycles) == 2
    assert all(cycle["msa"].shape == (64, 256) for cycle in cycles)
    assert seen["preserve_prefix_rng"] is True
    for name in ("init_noise", "step_noises", "rotations", "translations"):
        assert name not in seen
    assert profiles == [
        {
            "actual": {
                "tokens": 3,
                "atoms": 3,
                "msa": 5,
                "structural_tokens": 3,
            },
            "storage": {
                "tokens": 3,
                "atoms": 3,
                "msa": 5,
                "structural_tokens": 3,
            },
            "target": {
                "tokens": 256,
                "atoms": 256,
                "msa": 64,
                "structural_tokens": 256,
            },
            "changed": True,
            "static": {"chains": 1},
        }
    ]


def test_native_cli_falls_back_to_materialized_tapes_for_other_prngs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "job.json"
    weights_path = tmp_path / "weights.npz"
    input_path.write_text(json.dumps([{"name": "job"}]), encoding="utf-8")
    weights_path.write_bytes(b"native")
    features = _opendde_features(msa_depth=5)
    seen: dict[str, object] = {}

    monkeypatch.setattr(predict_impl, "_load_jobs", lambda _path: [{"name": "job"}])
    monkeypatch.setattr(
        predict_impl, "_load_weights", lambda _path: _params_double()
    )
    monkeypatch.setattr(predict_impl, "_featurize", lambda _job, **_kwargs: features)
    monkeypatch.setattr(predict_impl, "_prefix_rng_is_supported", lambda: False)

    sentinels = tuple(object() for _ in range(4))
    tape_calls: list[dict[str, object]] = []

    def fake_tapes(**kwargs):
        tape_calls.append(kwargs)
        return sentinels

    monkeypatch.setattr(
        "foldjax.models.opendde.models.sampling.make_padded_random_tapes",
        fake_tapes,
    )

    def fake_predict(_features, _params, **kwargs):
        seen.update(kwargs)
        return {"coordinate": np.zeros((1, 3, 3), dtype=np.float32)}

    monkeypatch.setattr(predict_impl, "_predict", fake_predict)
    monkeypatch.setattr(
        predict_impl,
        "_score",
        lambda output, _features, *, num_recycles: output,
    )
    output_path = tmp_path / "out" / "job.cif"
    monkeypatch.setattr(
        predict_impl,
        "_write",
        lambda _root, **_kwargs: [output_path],
    )

    predict_impl.main(
        [
            "--input-json",
            str(input_path),
            "--weights",
            str(weights_path),
            "--out",
            str(tmp_path / "out"),
            "--n-sample",
            "1",
            "--n-step",
            "1",
            "--n-cycle",
            "2",
        ],
        padding=PaddingConfig(msa=64),
    )

    assert seen["preserve_prefix_rng"] is False
    assert tuple(
        seen[name]
        for name in ("init_noise", "step_noises", "rotations", "translations")
    ) == sentinels
    assert len(tape_calls) == 1
    np.testing.assert_array_equal(
        tape_calls[0].pop("key"),
        jax.random.PRNGKey(101),
    )
    assert tape_calls[0] == {
        "num_samples": 1,
        "num_steps": 1,
        "actual_atom": 3,
        "target_atom": 256,
    }


def test_full_opendde_padding_masks_every_axis_and_crops_outputs() -> None:
    features = _opendde_features(msa_depth=5)
    sampled = sample_opendde_msa_cycle_features(features, num_recycles=2, seed=7)

    padded, cycles, plan = pad_opendde_features(
        features,
        sampled,
        PaddingConfig(tokens=4, atoms=5, msa=8, structural_tokens=6),
        n_queries=2,
        n_keys=4,
    )

    assert plan.actual == {
        "tokens": 3,
        "atoms": 3,
        "msa": 5,
        "structural_tokens": 3,
    }
    np.testing.assert_array_equal(padded["token_padding_mask"], [1, 1, 1, 0])
    np.testing.assert_array_equal(padded["atom_padding_mask"], [1, 1, 1, 0, 0])
    np.testing.assert_array_equal(
        padded["structural_token_padding_mask"], [1, 1, 1, 0, 0, 0]
    )
    np.testing.assert_array_equal(padded["parent_residue_idx"][3:], 0)
    np.testing.assert_array_equal(padded["atom_to_structural_token_idx"][3:], 0)
    assert all(cycle["msa"].shape == (8, 4) for cycle in cycles)
    assert all(np.all(cycle["msa_mask"][:, 3:] == 0) for cycle in cycles)

    model_features = select_opendde_model_features(padded)
    assert "chemical_bond_atom_indices" not in model_features
    assert "msa" not in model_features
    assert "token_padding_mask" in model_features

    cropped = crop_opendde_outputs(
        {
            "coordinate": np.zeros((1, 5, 3), dtype=np.float32),
            "distogram_logits": np.zeros((4, 4, 96), dtype=np.float32),
            "structural_s_trunk": np.zeros((6, 8), dtype=np.float32),
            "structural_z_trunk": np.zeros((6, 6, 8), dtype=np.float32),
        },
        plan,
    )
    assert cropped["coordinate"].shape == (1, 3, 3)
    assert cropped["distogram_logits"].shape == (3, 3, 96)
    assert cropped["structural_s_trunk"].shape == (3, 8)
    assert cropped["structural_z_trunk"].shape == (3, 3, 8)


def test_padded_random_tapes_preserve_the_unpadded_real_prefix() -> None:
    key = jax.random.PRNGKey(13)
    padded = make_padded_random_tapes(
        key=key,
        num_samples=2,
        num_steps=3,
        actual_atom=3,
        target_atom=5,
    )
    init_key, step_key, rotation_key, translation_key = jax.random.split(key, 4)
    expected_init = jax.random.normal(init_key, (2, 3, 3), dtype=jnp.float32)
    expected_steps = jnp.stack(
        tuple(
            jax.random.normal(step_key_i, (2, 3, 3), dtype=jnp.float32)
            for step_key_i in jax.random.split(step_key, 3)
        ),
        axis=0,
    )
    expected_rotations = uniform_random_rotations(rotation_key, (3, 2))
    expected_translations = jax.random.normal(
        translation_key, (3, 2, 3), dtype=jnp.float32
    )

    assert padded[1].shape == (3, 2, 5, 3)
    np.testing.assert_array_equal(np.asarray(padded[0][:, :3]), expected_init)
    np.testing.assert_array_equal(np.asarray(padded[1][..., :3, :]), expected_steps)
    np.testing.assert_array_equal(np.asarray(padded[0][:, 3:]), 0)
    np.testing.assert_array_equal(np.asarray(padded[1][..., 3:, :]), 0)
    np.testing.assert_array_equal(np.asarray(padded[2]), expected_rotations)
    np.testing.assert_array_equal(np.asarray(padded[3]), expected_translations)


def test_atom_only_padding_cannot_mark_token_zero_as_ligand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = _opendde_features(msa_depth=1)
    sampled = sample_opendde_msa_cycle_features(features, num_recycles=1, seed=7)
    padded, _cycles, plan = pad_opendde_features(
        features,
        sampled,
        PaddingConfig(tokens=3, atoms=5, msa=1, structural_tokens=3),
        n_queries=2,
        n_keys=4,
    )
    assert plan.actual["tokens"] == plan.target["tokens"] == 3
    np.testing.assert_array_equal(padded["atom_to_token_idx"][3:], 0)

    seen: dict[str, np.ndarray] = {}

    def fake_confidence_scores_from_logits(**kwargs):
        seen["token_is_ligand"] = np.asarray(kwargs["token_is_ligand"])
        return {
            "token_pair_pde": jnp.zeros((1, 3, 3), dtype=jnp.float32),
        }

    monkeypatch.setattr(
        postprocess_impl,
        "confidence_scores_from_logits",
        fake_confidence_scores_from_logits,
    )
    monkeypatch.setattr(
        postprocess_impl,
        "calculate_chain_based_gpde",
        lambda *_args, **_kwargs: {},
    )
    postprocess_impl.opendde_confidence_scores(
        {
            "coordinate": jnp.zeros((1, 5, 3), dtype=jnp.float32),
            "plddt": jnp.zeros((1, 5, 50), dtype=jnp.float32),
            "pae": jnp.zeros((1, 3, 3, 64), dtype=jnp.float32),
            "pde": jnp.zeros((1, 3, 3, 64), dtype=jnp.float32),
            "distogram_logits": jnp.zeros((1, 3, 3, 96), dtype=jnp.float32),
        },
        padded,
        num_recycles=1,
        include_shape_complementarity=False,
    )

    np.testing.assert_array_equal(seen["token_is_ligand"], [0, 0, 0])


def test_opendde_graph_routes_residue_structural_and_atom_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = _opendde_features(msa_depth=1)
    features["token_padding_mask"] = np.asarray([1, 1, 0], dtype=np.float32)
    features["atom_padding_mask"] = np.asarray([1, 1, 0], dtype=np.float32)
    features["structural_token_padding_mask"] = np.asarray(
        [1, 1, 0], dtype=np.float32
    )
    seen: dict[str, np.ndarray] = {}

    def fake_pairformer(
        _features,
        s_inputs,
        _params,
        *,
        pair_mask,
        **_kwargs,
    ):
        seen["residue_pair"] = np.asarray(pair_mask)
        return s_inputs, jnp.zeros((3, 2)), jnp.zeros((3, 3, 2))

    def fake_refiner(s, z, pair_mask, _bias, _params, **_kwargs):
        seen["structural_pair"] = np.asarray(pair_mask)
        return s, z

    def fake_diffusion(*_args, token_mask, atom_mask, **_kwargs):
        seen["diffusion_token"] = np.asarray(token_mask)
        seen["diffusion_atom"] = np.asarray(atom_mask)
        return jnp.zeros((1, 3, 3), dtype=jnp.float32)

    def fake_sample(denoise_fn, _schedule, *, atom_mask, **_kwargs):
        seen["sampler_atom"] = np.asarray(atom_mask)
        return denoise_fn(jnp.zeros((1, 3, 3)), jnp.ones((1,)))

    monkeypatch.setattr(
        model_impl,
        "input_feature_embedder",
        lambda *_args, **_kwargs: jnp.zeros((3, 2)),
    )
    monkeypatch.setattr(model_impl, "pairformer_output_from_s_inputs", fake_pairformer)
    monkeypatch.setattr(
        model_impl,
        "structural_token_expand",
        lambda *_args, **_kwargs: (
            jnp.ones((3, 2)),
            jnp.ones((3, 2)),
            jnp.ones((3, 3, 2)),
            {},
        ),
    )
    monkeypatch.setattr(
        model_impl,
        "prepare_structural_features",
        lambda residue, _pair: {
            **residue,
            "relp": jnp.zeros((3, 3, 139)),
            "d_lm": jnp.zeros((1,)),
            "v_lm": jnp.zeros((1,)),
            "pad_info": {},
            "structural_pair_attn_bias": jnp.zeros((3, 3)),
        },
    )
    monkeypatch.setattr(model_impl, "structural_refiner_stack", fake_refiner)
    monkeypatch.setattr(
        model_impl,
        "diffusion_conditioning_prepare_cache",
        lambda *_args, **_kwargs: jnp.ones((3, 3, 2)),
    )
    monkeypatch.setattr(
        model_impl,
        "atom_attention_encoder_prepare_diffusion_cache",
        lambda *_args, **_kwargs: (jnp.zeros((1,)), jnp.zeros((1,))),
    )
    monkeypatch.setattr(model_impl, "diffusion_module_forward", fake_diffusion)
    monkeypatch.setattr(model_impl, "sample_diffusion", fake_sample)
    monkeypatch.setattr(
        model_impl,
        "distogram_head",
        lambda *_args, **_kwargs: jnp.zeros((3, 3, 96)),
    )

    params = SimpleNamespace(
        input_embedder=object(),
        pairformer_output=object(),
        structural_expander=object(),
        structural_refiner=object(),
        diffusion=SimpleNamespace(conditioning=object(), atom_encoder=object()),
        distogram=object(),
        confidence=object(),
    )
    output = model_impl.opendde_infer_static(
        features,
        params,
        jnp.asarray([1.0, 0.5]),
        key=None,
        num_samples=1,
        num_recycles=1,
        run_confidence=False,
    )

    expected_pair = np.asarray(
        [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
        dtype=bool,
    )
    np.testing.assert_array_equal(seen["residue_pair"], expected_pair)
    np.testing.assert_array_equal(seen["structural_pair"], expected_pair)
    np.testing.assert_array_equal(seen["diffusion_token"], [1, 1, 0])
    np.testing.assert_array_equal(seen["diffusion_atom"], [1, 1, 0])
    np.testing.assert_array_equal(seen["sampler_atom"], [1, 1, 0])
    assert output["coordinate"].shape == (1, 3, 3)


def test_opendde_profile_aggregation_keeps_heterogeneous_runs_visible() -> None:
    first = {"target": {"msa": 64}, "static": {"chains": 1}}
    second = {"target": {"msa": 64}, "static": {"chains": 2}}
    legacy = {"target": {"msa": 64}}

    assert _shape_profile([first, first], padded=True) == first
    assert _shape_profile([first, second], padded=True) == {
        "per_run": [first, second]
    }
    # Older callback/test doubles did not report static metadata. Keep accepting
    # that dictionary unchanged while real CLI runs now include chain identity.
    assert _shape_profile([legacy], padded=True) == legacy
    assert _shape_profile([], padded=False) is None


def _opendde_features(*, msa_depth: int) -> dict[str, np.ndarray]:
    n_token = 3
    n_atom = 3
    atom_to_token = np.arange(n_atom, dtype=np.int64)
    template_pair_mask = np.zeros((4, n_token, n_token), dtype=np.float32)
    template_pair_mask[0] = 1.0
    return {
        **_msa_features(msa_depth, n_token),
        "restype": np.eye(32, dtype=np.float32)[:n_token],
        "profile": np.zeros((n_token, 32), dtype=np.float32),
        "deletion_mean": np.zeros((n_token,), dtype=np.float32),
        "residue_index": np.arange(n_token, dtype=np.int64),
        "token_index": np.arange(n_token, dtype=np.int64),
        "asym_id": np.zeros((n_token,), dtype=np.int64),
        "entity_id": np.zeros((n_token,), dtype=np.int64),
        "sym_id": np.zeros((n_token,), dtype=np.int64),
        "has_frame": np.ones((n_token,), dtype=np.int64),
        "frame_atom_index": np.zeros((n_token, 3), dtype=np.int64),
        "atom_to_token_idx": atom_to_token,
        "atom_to_tokatom_idx": np.zeros((n_atom,), dtype=np.int64),
        "ref_pos": np.arange(n_atom * 3, dtype=np.float32).reshape(n_atom, 3),
        "ref_space_uid": atom_to_token.copy(),
        "ref_charge": np.zeros((n_atom,), dtype=np.float32),
        "ref_mask": np.ones((n_atom,), dtype=np.float32),
        "ref_atom_name_chars": np.zeros((n_atom, 4, 64), dtype=np.float32),
        "ref_element": np.zeros((n_atom, 128), dtype=np.float32),
        "distogram_rep_atom_mask": np.ones((n_atom,), dtype=np.float32),
        "pae_rep_atom_mask": np.ones((n_atom,), dtype=np.int64),
        "plddt_m_rep_atom_mask": np.ones((n_atom,), dtype=np.int64),
        "modified_res_mask": np.zeros((n_atom,), dtype=np.int64),
        "is_protein": np.ones((n_atom,), dtype=np.int64),
        "is_ligand": np.zeros((n_atom,), dtype=np.int64),
        "is_dna": np.zeros((n_atom,), dtype=np.int64),
        "is_rna": np.zeros((n_atom,), dtype=np.int64),
        "relp": np.zeros((n_token, n_token, 139), dtype=np.float32),
        "token_bonds": np.zeros((n_token, n_token), dtype=np.float32),
        "template_aatype": np.zeros((4, n_token), dtype=np.int32),
        "template_atom_positions": np.zeros((4, n_token, 24, 3), np.float32),
        "template_atom_mask": np.zeros((4, n_token, 24), dtype=bool),
        "template_pseudo_beta_mask": template_pair_mask,
        "template_distogram": np.zeros((4, n_token, n_token, 39), np.float32),
        "template_unit_vector": np.zeros((4, n_token, n_token, 3), np.float32),
        "template_backbone_frame_mask": template_pair_mask.copy(),
        "structural_token_index": np.arange(n_token, dtype=np.int64),
        "residue_token_group_id": np.arange(n_token, dtype=np.int64),
        "subtoken_role": np.ones((n_token,), dtype=np.int64),
        "subtoken_role_id": np.ones((n_token,), dtype=np.int64),
        "twin_token_idx": np.full((n_token,), -1, dtype=np.int64),
        "parent_residue_idx": np.arange(n_token, dtype=np.int64),
        "atom_to_structural_token_idx": atom_to_token.copy(),
        "atom_to_structural_tokatom_idx": np.zeros((n_atom,), dtype=np.int64),
        "structural_distogram_rep_atom_mask": np.ones((n_atom,), dtype=np.int64),
        "structural_pae_rep_atom_mask": np.ones((n_atom,), dtype=np.int64),
        "structural_has_frame": np.ones((n_token,), dtype=np.int64),
        "structural_frame_atom_index": np.zeros((n_token, 3), dtype=np.int64),
        "prev_parent_residue_idx": np.full((n_token,), -1, dtype=np.int64),
        "next_parent_residue_idx": np.full((n_token,), -1, dtype=np.int64),
        "structural_is_polymer": np.ones((n_token,), dtype=np.int64),
        "structural_polymer_type": np.ones((n_token,), dtype=np.int64),
        "structural_seq_pos": np.arange(n_token, dtype=np.int64),
        "chemical_bond_atom_indices": np.zeros((0, 2), dtype=np.int64),
    }


def test_the_model_feature_filter_keeps_template_multiplicity() -> None:
    """The dedup's multiplicity must reach the graph, or the divisor is wrong.

    `select_opendde_model_features` keeps a name when it is in `_MODEL_FEATURES`
    *or* starts with `template_`, and it is the second clause that carries
    `template_multiplicity` through. That is incidental: Protenix's equivalent
    filter is an explicit set, and there the key had to be added to
    `_PADDED_MODEL_FEATURES` by hand. Tightening this filter to a list would
    reintroduce the trap it currently avoids by accident -- the key would be
    dropped with no error, `template_embedder` would fall back to
    `num_templates = shape[0]`, and a four-row query deduplicated to two would
    be divided by two instead of four, on the padded path only.
    """
    from foldjax.models.opendde.data.padding import select_opendde_model_features

    kept = select_opendde_model_features(
        {
            "template_multiplicity": np.asarray([1.0, 3.0], dtype=np.float32),
            "template_distogram": np.zeros((2, 3, 3, 39), dtype=np.float32),
            "not_a_model_feature": np.zeros((3,), dtype=np.float32),
        }
    )

    assert "template_multiplicity" in kept
    assert "template_distogram" in kept
    assert "not_a_model_feature" not in kept


def test_the_multiplication_backend_default_is_keyed_on_the_trunk_dtype(
    monkeypatch,
) -> None:
    """Blocked for a narrow trunk, fused for float32, explicit env still wins.

    The blocked path is measured faster *and* smaller on a bfloat16 trunk --
    13.4% and 2.59 GiB at 1,003 residues, 7.9% and 6.35 GiB at 1,531 -- and the
    reason is that its accumulators follow the trunk's width. On a float32
    trunk they do not narrow, none of that was measured, and OpenDDE ships
    float32, so the shipped default must stay fused. A single default for both
    dtypes is the bug this pins.
    """
    import jax.numpy as jnp

    from foldjax.models.opendde.models import model as model_impl

    seen: list[str | None] = []

    @model_impl._with_cueq_triangle_defaults
    def spy(**_kwargs) -> None:
        seen.append(os.environ.get("PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND"))

    monkeypatch.delenv("PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND", raising=False)
    spy(trunk_dtype=jnp.bfloat16)
    spy(trunk_dtype=jnp.float32)
    spy(trunk_dtype=None)
    assert seen == ["xla", "cueq", "cueq"], seen

    # Nothing leaks into the process for the next model to inherit.
    assert "PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND" not in os.environ

    # An explicit environment still wins, on either dtype: `setdefault`, not a
    # pin, is what lets a job too large for one path ask for the other.
    seen.clear()
    monkeypatch.setenv("PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND", "cueq")
    spy(trunk_dtype=jnp.bfloat16)
    assert seen == ["cueq"], seen
