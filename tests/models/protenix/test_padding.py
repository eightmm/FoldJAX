from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.backends.protenix import ProtenixBackend
from foldjax.models.protenix.cli.predict import _padded_noise_tapes
from foldjax.models.protenix.data.padding import (
    crop_protenix_outputs,
    pad_protenix_features,
)
from foldjax.models.protenix.models import model as model_impl
from foldjax.models.protenix.models.diffusion.diffusion import (
    _prefix_atom_normal,
    sample_diffusion,
)
from foldjax.models.protenix.models.heads.confidence import (
    confidence_scores_from_logits,
)
from foldjax.models.protenix.models.primitives.attention import (
    AttentionParams,
    local_attention,
)
from foldjax.models.protenix.models.primitives.primitives import (
    LayerNormParams,
    LinearParams,
)
from foldjax.models.protenix.models.trunk_blocks.embedders import (
    compact_relative_position_features,
)
from foldjax.models.protenix.models.trunk_blocks.msa import (
    MSAPairWeightedAveragingParams,
    msa_pair_weighted_averaging,
)
from foldjax.models.protenix.relative_position import (
    COMPACT_RELP_FIELDS,
    compact_relative_position_storage,
)
from foldjax.padding import PaddingPlan
from foldjax.schema import PaddingConfig, PredictionRequest


def test_feature_padding_resolves_every_axis_and_crop_restores_public_shapes() -> None:
    features = _features()
    features["is_ligand"] = np.zeros(features["ref_pos"].shape[0], np.int64)
    features["token_is_ligand"] = np.zeros(features["restype"].shape[0], bool)

    padded, plan = pad_protenix_features(
        features,
        PaddingConfig(tokens=8, atoms=8, msa=4, templates=4),
        n_queries=2,
        n_keys=4,
    )

    assert plan.actual == {"tokens": 3, "atoms": 5, "msa": 2, "templates": 1}
    assert plan.target == {"tokens": 8, "atoms": 8, "msa": 4, "templates": 4}
    assert padded["restype"].shape == (8, 32)
    assert padded["ref_pos"].shape == (8, 3)
    assert padded["msa"].shape == (4, 8)
    assert padded["relp"].shape == (8, 8, 139)
    assert padded["template_distogram"].shape == (4, 8, 8, 39)
    np.testing.assert_array_equal(padded["relp"][:3, :3], features["relp"])
    np.testing.assert_array_equal(
        padded["token_padding_mask"], [1, 1, 1, 0, 0, 0, 0, 0]
    )
    np.testing.assert_array_equal(padded["atom_padding_mask"], [1, 1, 1, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(padded["msa_mask"][:2, :3], 1.0)
    np.testing.assert_array_equal(padded["msa_mask"][:, 3:], 0.0)
    np.testing.assert_array_equal(padded["msa_mask"][2:], 0.0)
    np.testing.assert_array_equal(padded["atom_to_token_idx"][5:], 3)
    assert not np.any(padded["pad_info"]["mask_trunked"][-1, -1])

    output = {
        "coordinate": np.zeros((2, 8, 3), dtype=np.float32),
        "plddt": np.zeros((2, 8, 50), dtype=np.float32),
        "pae": np.zeros((2, 8, 8, 64), dtype=np.float32),
        "atom_plddt": np.zeros((2, 8), dtype=np.float32),
        "contact_probs": np.zeros((2, 8, 8), dtype=np.float32),
        "s_trunk": np.zeros((8, 4), dtype=np.float32),
        "z_trunk": np.zeros((8, 8, 4), dtype=np.float32),
    }
    cropped = crop_protenix_outputs(output, plan)
    assert cropped["coordinate"].shape == (2, 5, 3)
    assert cropped["plddt"].shape == (2, 5, 50)
    assert cropped["pae"].shape == (2, 3, 3, 64)
    assert cropped["atom_plddt"].shape == (2, 5)
    assert cropped["contact_probs"].shape == (2, 3, 3)
    assert cropped["s_trunk"].shape == (3, 4)
    assert cropped["z_trunk"].shape == (3, 3, 4)


def test_generated_reference_frames_pad_without_creating_valid_dummy_frames():
    from foldjax.models.protenix.data.featurize_json import featurize_protein_json

    features = featurize_protein_json(
        {"sequences": [{"proteinChain": {"sequence": "AG"}}]}, n_queries=2, n_keys=4
    )
    original = features["frame_atom_index"].copy()
    padded, _ = pad_protenix_features(
        features, PaddingConfig(tokens=4), n_queries=2, n_keys=4
    )
    assert padded["frame_atom_index"].shape == (4, 3)
    np.testing.assert_array_equal(padded["frame_atom_index"][:2], original)
    np.testing.assert_array_equal(padded["frame_atom_index"][2:], -1)
    np.testing.assert_array_equal(padded["has_frame"][2:], 0)
    np.testing.assert_array_equal(features["frame_atom_index"], original)


def test_feature_padding_keeps_compact_relp_and_zeroes_dummy_pairs() -> None:
    features = _features()
    del features["relp"]
    features.update(
        compact_relative_position_storage(
            asym_id=features["asym_id"],
            residue_index=features["residue_index"],
            entity_id=features["entity_id"],
            sym_id=features["sym_id"],
            token_index=features["token_index"],
        )
    )
    original_dense = np.asarray(compact_relative_position_features(features))

    padded, _ = pad_protenix_features(
        features,
        PaddingConfig(tokens=8, atoms=8, msa=4, templates=4),
        n_queries=2,
        n_keys=4,
    )

    assert "relp" not in padded
    assert set(COMPACT_RELP_FIELDS) <= padded.keys()
    rebuilt = np.asarray(compact_relative_position_features(padded))
    np.testing.assert_array_equal(
        rebuilt,
        np.pad(original_dense, ((0, 5), (0, 5), (0, 0))),
    )


def test_compiled_padding_allowlist_keeps_required_model_features(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_compiled(input_feature_dict, *_args, **_kwargs):
        captured.update(input_feature_dict)
        return {}

    monkeypatch.setattr(model_impl, "_compiled_protenix_infer", fake_compiled)
    required = {
        "entity_id",
        "residue_index",
        "sym_id",
        "token_index",
        "v_lm",
    }
    features = {
        "asym_id": jnp.asarray([0, 0], dtype=jnp.int32),
        "token_padding_mask": jnp.asarray([1, 1], dtype=jnp.float32),
        "entity_id": jnp.asarray([0, 0], dtype=jnp.int32),
        "residue_index": jnp.asarray([1, 2], dtype=jnp.int32),
        "sym_id": jnp.asarray([0, 0], dtype=jnp.int32),
        "token_index": jnp.asarray([0, 1], dtype=jnp.int32),
        "v_lm": jnp.zeros((1, 2, 4, 1), dtype=jnp.float32),
        "chemical_bond_order": jnp.zeros((7,), dtype=jnp.float32),
    }

    model_impl.protenix_infer_compiled(
        features,
        (),
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        padded_generated_schema=True,
    )

    assert required <= captured.keys()
    assert "chemical_bond_order" not in captured


def test_compiled_static_mask_does_not_claim_generated_padding_provenance(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_compiled(input_feature_dict, *_args, **_kwargs):
        captured.update(input_feature_dict)
        return {}

    monkeypatch.setattr(model_impl, "_compiled_protenix_infer", fake_compiled)
    features = {
        "asym_id": jnp.asarray([0, 0], dtype=jnp.int32),
        "token_padding_mask": jnp.asarray([1, 1], dtype=jnp.float32),
        "constraint_feature": jnp.ones((2, 2, 1), dtype=jnp.float32),
        "chemical_bond_order": jnp.ones((7,), dtype=jnp.float32),
    }

    model_impl.protenix_infer_compiled(
        features,
        (),
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
    )

    assert "constraint_feature" in captured
    assert "chemical_bond_order" in captured


def test_feature_padding_rejects_non_native_template_depth_and_constraints() -> None:
    features = _features()
    with pytest.raises(ValueError, match="target must be exactly 4"):
        pad_protenix_features(
            features,
            PaddingConfig(tokens=8, atoms=8, msa=4, templates=5),
            n_queries=2,
            n_keys=4,
        )

    features["constraint_feature"] = {"contact": np.zeros((3, 3, 1))}
    with pytest.raises(ValueError, match="constraint features"):
        pad_protenix_features(
            features,
            PaddingConfig(tokens=8, atoms=8, msa=4, templates=4),
            n_queries=2,
            n_keys=4,
        )


def test_feature_padding_fails_early_on_incomplete_generated_schema() -> None:
    features = _features()
    del features["profile"]

    with pytest.raises(ValueError, match=r"complete generated.*profile"):
        pad_protenix_features(
            features,
            PaddingConfig(tokens=8, atoms=8, msa=4, templates=4),
            n_queries=2,
            n_keys=4,
        )


def test_local_atom_mask_blocks_adversarial_dummy_keys() -> None:
    params = AttentionParams(
        linear_q=LinearParams(jnp.ones((1, 1)), jnp.zeros((1,))),
        linear_k=LinearParams(jnp.ones((1, 1))),
        linear_v=LinearParams(jnp.ones((1, 1))),
        linear_o=LinearParams(jnp.ones((1, 1))),
        linear_g=None,
    )
    real = jnp.asarray([[1.0], [2.0], [3.0]])
    padded = jnp.concatenate([real, jnp.asarray([[1.0e6]])], axis=0)
    bias = jnp.zeros((1, 2, 2, 4), dtype=jnp.float32)

    expected = local_attention(
        real,
        real,
        params,
        1,
        trunked_attn_bias=bias,
        n_queries=2,
        n_keys=4,
    )
    actual = local_attention(
        padded,
        padded,
        params,
        1,
        trunked_attn_bias=bias,
        n_queries=2,
        n_keys=4,
        sequence_mask=jnp.asarray([1, 1, 1, 0]),
    )

    np.testing.assert_allclose(np.asarray(actual[:3]), np.asarray(expected), atol=1e-6)
    np.testing.assert_array_equal(np.asarray(actual[3]), np.zeros((1,), np.float32))


def test_msa_pair_mask_blocks_adversarial_dummy_token() -> None:
    params = MSAPairWeightedAveragingParams(
        layernorm_m=LayerNormParams(),
        linear_mv=LinearParams(jnp.eye(2)),
        layernorm_z=LayerNormParams(),
        linear_z=LinearParams(jnp.ones((1, 2))),
        linear_mg=LinearParams(jnp.zeros((2, 2))),
        linear_out=LinearParams(jnp.eye(2)),
    )
    real_m = jnp.asarray([[[1.0, -1.0], [2.0, 0.0], [-1.0, 3.0]]])
    real_z = jnp.arange(18, dtype=jnp.float32).reshape(3, 3, 2) / 10.0
    padded_m = jnp.pad(real_m, ((0, 0), (0, 1), (0, 0))).at[:, 3].set(1.0e6)
    padded_z = jnp.pad(real_z, ((0, 1), (0, 1), (0, 0))).at[3].set(1.0e6)
    padded_z = padded_z.at[:, 3].set(-1.0e6)
    mask = jnp.asarray([1, 1, 1, 0], dtype=bool)

    expected = msa_pair_weighted_averaging(real_m, real_z, params)
    actual = msa_pair_weighted_averaging(
        padded_m,
        padded_z,
        params,
        mask[:, None] & mask[None, :],
    )

    np.testing.assert_allclose(
        np.asarray(actual[:, :3]), np.asarray(expected), rtol=1e-5, atol=1e-5
    )


def test_confidence_masks_exclude_adversarial_dummy_tokens_and_atoms() -> None:
    real_pae = jnp.arange(18, dtype=jnp.float32).reshape(1, 3, 3, 2) / 10.0
    real_pde = jnp.flip(real_pae, axis=-1)
    real_dist = jnp.arange(18, dtype=jnp.float32).reshape(3, 3, 2) / 20.0
    real_plddt = jnp.arange(6, dtype=jnp.float32).reshape(1, 3, 2) / 10.0
    real_has_frame = jnp.ones((3,), dtype=bool)
    real_asym_id = jnp.asarray([0, 0, 1], dtype=jnp.int32)
    real_ligand = jnp.asarray([0, 0, 1], dtype=bool)

    expected = confidence_scores_from_logits(
        plddt_logits=real_plddt,
        pae_logits=real_pae,
        pde_logits=real_pde,
        distogram_logits=real_dist,
        token_has_frame=real_has_frame,
        token_asym_id=real_asym_id,
        token_is_ligand=real_ligand,
        n_chain=2,
    )

    padded_pae = jnp.full((1, 5, 5, 2), 1.0e4, dtype=jnp.float32)
    padded_pde = jnp.full((1, 5, 5, 2), -1.0e4, dtype=jnp.float32)
    padded_dist = jnp.full((5, 5, 2), 1.0e4, dtype=jnp.float32)
    padded_plddt = jnp.full((1, 5, 2), -1.0e4, dtype=jnp.float32)
    padded_pae = padded_pae.at[:, :3, :3].set(real_pae)
    padded_pde = padded_pde.at[:, :3, :3].set(real_pde)
    padded_dist = padded_dist.at[:3, :3].set(real_dist)
    padded_plddt = padded_plddt.at[:, :3].set(real_plddt)
    token_mask = jnp.asarray([1, 1, 1, 0, 0], dtype=bool)
    atom_mask = jnp.asarray([1, 1, 1, 0, 0], dtype=bool)
    actual = confidence_scores_from_logits(
        plddt_logits=padded_plddt,
        pae_logits=padded_pae,
        pde_logits=padded_pde,
        distogram_logits=padded_dist,
        token_has_frame=jnp.ones((5,), dtype=bool),
        token_asym_id=jnp.asarray([0, 0, 1, 0, 1], dtype=jnp.int32),
        token_is_ligand=jnp.asarray([0, 0, 1, 1, 1], dtype=bool),
        token_mask=token_mask,
        atom_mask=atom_mask,
        n_chain=2,
    )

    for name in (
        "summary_plddt",
        "summary_gpde",
        "summary_ptm",
        "summary_iptm",
        "summary_ranking_score",
        "chain_ptm",
        "chain_iptm",
        "chain_pair_iptm",
        "chain_pair_iptm_global",
        "chain_gpde",
        "chain_pair_gpde",
        "chain_pair_pae_mean",
        "chain_pair_pae_min",
    ):
        np.testing.assert_allclose(
            np.asarray(actual[name]),
            np.asarray(expected[name]),
            rtol=1e-5,
            atol=1e-5,
        )


def test_backend_padding_flags_and_shape_profile_are_opt_in(
    tmp_path, monkeypatch
) -> None:
    input_path = tmp_path / "job.json"
    weight_path = tmp_path / "weights.npz"
    input_path.write_text("[]", encoding="utf-8")
    weight_path.write_bytes(b"weights")
    seen: list[tuple[str, ...]] = []

    def fake_main(argv, *, on_padding_plan=None):
        seen.append(tuple(argv))
        if "--padding" in argv:
            on_padding_plan(
                PaddingPlan(
                    actual={"tokens": 3, "atoms": 5, "msa": 2, "templates": 1},
                    storage={"tokens": 3, "atoms": 5, "msa": 2, "templates": 4},
                    target={"tokens": 8, "atoms": 8, "msa": 4, "templates": 4},
                ),
                {"chains": 2},
            )
        return []

    monkeypatch.setattr(
        "foldjax.backends.protenix.import_module",
        lambda _name: SimpleNamespace(main=fake_main),
    )
    backend = ProtenixBackend()
    common = dict(
        model="protenix",
        input=input_path,
        weights=weight_path,
        output_dir=tmp_path / "out",
        use_compile_cache=False,
    )

    exact = backend.predict(PredictionRequest(**common))
    assert "--padding" not in seen[-1]
    assert exact.shape_profile is None

    padded = backend.predict(
        PredictionRequest(
            **common,
            padding=PaddingConfig(tokens=8, atoms=8, msa=4, templates=4),
        )
    )
    assert "--padding" in seen[-1]
    assert ("--pad-tokens", "8") == seen[-1][
        seen[-1].index("--pad-tokens") : seen[-1].index("--pad-tokens") + 2
    ]
    assert padded.shape_profile == padded.raw["padding_plans"][0]
    assert padded.shape_profile["static"] == {"chains": 2}


@pytest.mark.parametrize("chunk_size", [None, 1])
def test_padded_noise_keeps_every_sample_real_prefix(chunk_size) -> None:
    padded_init, padded_steps = _padded_noise_tapes(
        seed=17,
        num_samples=2,
        num_steps=3,
        actual_atom=3,
        target_atom=7,
        diffusion_chunk_size=chunk_size,
    )

    root_key = jax.random.PRNGKey(17)
    chunk_sizes = (2,) if chunk_size is None else (1, 1)
    chunk_keys = (
        (root_key,)
        if chunk_size is None
        else tuple(jax.random.split(root_key, len(chunk_sizes)))
    )
    expected_init_chunks = []
    expected_step_chunks = [[] for _ in range(3)]
    for expected_chunk_size, chunk_key in zip(chunk_sizes, chunk_keys, strict=True):
        chunk_key, init_key = jax.random.split(chunk_key)
        expected_init_chunks.append(
            jax.random.normal(init_key, (expected_chunk_size, 3, 3), dtype=jnp.float32)
        )
        for step_index, step_key in enumerate(jax.random.split(chunk_key, 3)):
            expected_step_chunks[step_index].append(
                jax.random.normal(
                    step_key, (expected_chunk_size, 3, 3), dtype=jnp.float32
                )
            )
    expected_init = jnp.concatenate(expected_init_chunks, axis=0)
    expected_steps = jnp.stack(
        tuple(jnp.concatenate(chunks, axis=0) for chunks in expected_step_chunks),
        axis=0,
    )

    assert padded_steps.shape == (3, 2, 7, 3)
    np.testing.assert_array_equal(np.asarray(padded_init[:, :3]), expected_init)
    np.testing.assert_array_equal(np.asarray(padded_steps[..., :3, :]), expected_steps)
    np.testing.assert_array_equal(np.asarray(padded_init[:, 3:]), 0)
    np.testing.assert_array_equal(np.asarray(padded_steps[..., 3:, :]), 0)


@pytest.mark.parametrize("chunk_size", [None, 1, 2, 4])
def test_masked_atom_draws_match_the_materialized_noise_tape(chunk_size) -> None:
    num_samples, num_steps, actual_atom, target_atom = 5, 7, 11, 19
    seed = 23
    expected_init, expected_steps = _padded_noise_tapes(
        seed=seed,
        num_samples=num_samples,
        num_steps=num_steps,
        actual_atom=actual_atom,
        target_atom=target_atom,
        diffusion_chunk_size=chunk_size,
    )
    root_key = jax.random.PRNGKey(seed)
    chunk_sizes = (
        (num_samples,)
        if chunk_size is None
        else tuple(
            min(chunk_size, num_samples - start)
            for start in range(0, num_samples, chunk_size)
        )
    )
    chunk_keys = (
        (root_key,)
        if chunk_size is None
        else tuple(jax.random.split(root_key, len(chunk_sizes)))
    )
    atom_mask = jnp.arange(target_atom) < actual_atom
    actual_init_chunks = []
    actual_step_chunks = [[] for _ in range(num_steps)]
    for current_size, chunk_key in zip(chunk_sizes, chunk_keys, strict=True):
        chunk_key, init_key = jax.random.split(chunk_key)
        actual_init_chunks.append(
            _prefix_atom_normal(
                init_key,
                atom_mask,
                num_samples=current_size,
                dtype=jnp.float32,
            )
        )
        for step_index, step_key in enumerate(jax.random.split(chunk_key, num_steps)):
            actual_step_chunks[step_index].append(
                _prefix_atom_normal(
                    step_key,
                    atom_mask,
                    num_samples=current_size,
                    dtype=jnp.float32,
                )
            )
    actual_init = jnp.concatenate(actual_init_chunks, axis=0)
    actual_steps = jnp.stack(
        tuple(jnp.concatenate(chunks, axis=0) for chunks in actual_step_chunks),
        axis=0,
    )

    np.testing.assert_array_equal(
        np.asarray(actual_init).reshape(-1).view(np.uint8),
        np.asarray(expected_init).reshape(-1).view(np.uint8),
    )
    np.testing.assert_array_equal(
        np.asarray(actual_steps).reshape(-1).view(np.uint8),
        np.asarray(expected_steps).reshape(-1).view(np.uint8),
    )


@pytest.mark.parametrize("chunk_size", [None, 1, 2, 4])
@pytest.mark.parametrize("use_scan", [False, True])
def test_masked_padding_noise_matches_the_materialized_tape_bitwise(
    chunk_size, use_scan
) -> None:
    num_samples, num_steps, actual_atom, target_atom = 5, 3, 5, 8
    seed = 19
    init_noise, step_noises = _padded_noise_tapes(
        seed=seed,
        num_samples=num_samples,
        num_steps=num_steps,
        actual_atom=actual_atom,
        target_atom=target_atom,
        diffusion_chunk_size=chunk_size,
    )
    schedule = jnp.asarray([2.0, 1.0, 0.4, 0.0], dtype=jnp.float32)
    atom_mask = jnp.arange(target_atom) < actual_atom

    def denoise(x, t):
        return x * jnp.asarray(0.83, x.dtype) + t[..., None, None] * 0.01

    common = {
        "denoise_fn": denoise,
        "noise_schedule": schedule,
        "num_samples": num_samples,
        "n_atom": target_atom,
        "key": jax.random.PRNGKey(seed),
        "diffusion_chunk_size": chunk_size,
        "use_scan": use_scan,
        "atom_mask": atom_mask,
    }
    expected = sample_diffusion(
        init_noise=init_noise,
        step_noises=step_noises,
        **common,
    )
    actual = sample_diffusion(preserve_prefix_rng=True, **common)

    np.testing.assert_array_equal(
        np.asarray(actual).reshape(-1).view(np.uint8),
        np.asarray(expected).reshape(-1).view(np.uint8),
    )


def test_masked_padding_noise_is_lazy_and_requires_partitionable_threefry() -> None:
    num_samples, num_steps, actual_atom, target_atom = 2, 8, 7, 32
    schedule = jnp.linspace(2.0, 0.0, num_steps + 1, dtype=jnp.float32)
    atom_mask = jnp.arange(target_atom) < actual_atom
    key = jax.random.PRNGKey(11)
    init_noise, step_noises = _padded_noise_tapes(
        seed=11,
        num_samples=num_samples,
        num_steps=num_steps,
        actual_atom=actual_atom,
        target_atom=target_atom,
        diffusion_chunk_size=None,
    )

    def denoise(x, _t):
        return x * jnp.asarray(0.9, x.dtype)

    old = jax.jit(
        lambda init, steps, mask: sample_diffusion(
            denoise,
            schedule,
            num_samples=num_samples,
            n_atom=target_atom,
            key=key,
            init_noise=init,
            step_noises=steps,
            atom_mask=mask,
            use_scan=True,
        )
    )
    compact = jax.jit(
        lambda random_key, mask: sample_diffusion(
            denoise,
            schedule,
            num_samples=num_samples,
            n_atom=target_atom,
            key=random_key,
            atom_mask=mask,
            use_scan=True,
            preserve_prefix_rng=True,
        )
    )
    old_executable = old.lower(init_noise, step_noises, atom_mask).compile()
    compact_executable = compact.lower(key, atom_mask).compile()
    old_memory = old_executable.memory_analysis()
    compact_memory = compact_executable.memory_analysis()

    assert (
        old_memory.argument_size_in_bytes > 25 * compact_memory.argument_size_in_bytes
    )
    compact_hlo = compact.lower(key, atom_mask).compiler_ir(dialect="stablehlo")
    assert "stablehlo.while" in str(compact_hlo)

    with jax.threefry_partitionable(False):
        with pytest.raises(ValueError, match="jax_threefry_partitionable"):
            sample_diffusion(
                denoise,
                schedule,
                num_samples=num_samples,
                n_atom=target_atom,
                key=key,
                atom_mask=atom_mask,
                use_scan=True,
                preserve_prefix_rng=True,
            )
    with jax.default_prng_impl("rbg"):
        with pytest.raises(ValueError, match="jax_default_prng_impl"):
            sample_diffusion(
                denoise,
                schedule,
                num_samples=num_samples,
                n_atom=target_atom,
                key=key,
                atom_mask=atom_mask,
                use_scan=True,
                preserve_prefix_rng=True,
            )


def _features() -> dict[str, np.ndarray]:
    n_token = 3
    n_atom = 5
    n_msa = 2
    atom_to_token = np.asarray([0, 0, 1, 2, 2], dtype=np.int64)
    template_mask = np.zeros((4, n_token, n_token), dtype=np.float32)
    template_mask[0] = 1.0
    return {
        "restype": np.eye(32, dtype=np.float32)[:n_token],
        "profile": np.zeros((n_token, 32), dtype=np.float32),
        "deletion_mean": np.zeros((n_token,), dtype=np.float32),
        "residue_index": np.arange(n_token, dtype=np.int64),
        "token_index": np.arange(n_token, dtype=np.int64),
        "asym_id": np.zeros((n_token,), dtype=np.int64),
        "entity_id": np.zeros((n_token,), dtype=np.int64),
        "sym_id": np.zeros((n_token,), dtype=np.int64),
        "has_frame": np.ones((n_token,), dtype=np.int64),
        "atom_to_token_idx": atom_to_token,
        "atom_to_tokatom_idx": np.asarray([0, 1, 0, 0, 1], dtype=np.int64),
        "ref_pos": np.arange(n_atom * 3, dtype=np.float32).reshape(n_atom, 3),
        "ref_space_uid": atom_to_token.copy(),
        "ref_charge": np.zeros((n_atom,), dtype=np.float32),
        "ref_mask": np.ones((n_atom,), dtype=np.float32),
        "ref_atom_name_chars": np.zeros((n_atom, 4, 64), dtype=np.float32),
        "ref_element": np.zeros((n_atom, 128), dtype=np.float32),
        "distogram_rep_atom_mask": np.asarray([1, 0, 1, 1, 0], np.float32),
        "msa": np.zeros((n_msa, n_token), dtype=np.int64),
        "has_deletion": np.zeros((n_msa, n_token), dtype=np.float32),
        "deletion_value": np.zeros((n_msa, n_token), dtype=np.float32),
        "relp": np.zeros((n_token, n_token, 139), dtype=np.float32),
        "token_bonds": np.zeros((n_token, n_token), dtype=np.float32),
        "template_aatype": np.zeros((4, n_token), dtype=np.int32),
        "template_atom_positions": np.zeros((4, n_token, 24, 3), np.float32),
        "template_atom_mask": np.zeros((4, n_token, 24), dtype=bool),
        "template_pseudo_beta_mask": template_mask,
        "template_distogram": np.zeros((4, n_token, n_token, 39), np.float32),
        "template_unit_vector": np.zeros((4, n_token, n_token, 3), np.float32),
        "template_backbone_frame_mask": template_mask.copy(),
    }


def test_padding_pads_exactly_the_fields_that_carry_a_template_axis() -> None:
    """The list has one owner, so the two modules cannot drift apart.

    `_pad_templates` iterates `_TEMPLATE_FIELDS`; the axis itself is created by
    `_as_protenix_dict` in `template_features`. A field added there and not here
    would silently go unpadded, and the resulting shape error would surface
    downstream with nothing about templates in it.
    """
    from foldjax.models.protenix.data.padding import _TEMPLATE_FIELDS
    from foldjax.models.protenix.data.template_features import (
        TEMPLATE_FIELDS,
        _as_protenix_dict,
    )

    # Comparing the two names would prove nothing: padding builds its set *from*
    # TEMPLATE_FIELDS, so they agree by construction and cannot drift apart in
    # the direction this test exists to catch. The drift that can happen is a
    # field appearing in the producer and not in the list, so ask the producer.
    num_templates, num_residues, num_atoms = 2, 3, 24
    produced = _as_protenix_dict(
        np.zeros((num_templates, num_residues), dtype=np.int32),
        np.zeros((num_templates, num_residues, num_atoms, 3), dtype=np.float32),
        np.zeros((num_templates, num_residues, num_atoms), dtype=np.float32),
    )

    assert set(produced) == set(TEMPLATE_FIELDS)
    assert set(_TEMPLATE_FIELDS) == set(TEMPLATE_FIELDS)
    for name, value in produced.items():
        assert value.shape[0] == num_templates, name
