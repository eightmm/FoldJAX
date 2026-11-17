"""The released checkpoint, loaded and run end to end.

The port names its parameters the way upstream's `state_dict` does, so there
is no rename table to test -- but that only helps if every name the model asks
for is a name the file has. This runs the real weights on a toy complex with
every optional branch on (MSA encoder, LM encoder, language-model shim), which
is the one check that reaches all 1,594 of them.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from safetensors.numpy import save_file

from foldjax.models.esmfold2.bridge import checkpoint
from foldjax.models.esmfold2.models import model as jax_model
from foldjax.paths import weights_dir

N_TOKENS, N_ATOMS, MSA_DEPTH = 8, 24, 3


def _weights():
    directory = weights_dir("esmfold2")
    if not (directory / checkpoint.WEIGHTS_NAME).exists():
        pytest.skip("esmfold2 weights are not in the store")
    return directory


def _features():
    rng = np.random.default_rng(0)
    token_index = np.arange(N_TOKENS, dtype=np.int64)[None]
    msa = rng.integers(0, 20, (1, MSA_DEPTH, N_TOKENS)).astype(np.int64)
    return {
        "token_index": token_index,
        "residue_index": token_index.copy(),
        # Two chains, so the inter-chain confidence terms are real.
        "asym_id": (np.arange(N_TOKENS, dtype=np.int64) // 4)[None],
        "sym_id": np.zeros((1, N_TOKENS), dtype=np.int64),
        "entity_id": (np.arange(N_TOKENS, dtype=np.int64) // 4)[None],
        "mol_type": np.zeros((1, N_TOKENS), dtype=np.int64),
        "res_type": rng.integers(0, 20, (1, N_TOKENS)).astype(np.int64),
        "token_bonds": np.zeros((1, N_TOKENS, N_TOKENS, 1), dtype=np.float32),
        "token_attention_mask": np.ones((1, N_TOKENS), dtype=np.float32),
        "ref_pos": rng.standard_normal((1, N_ATOMS, 3)).astype(np.float32),
        "ref_element": np.full((1, N_ATOMS), 6, dtype=np.int64),
        "ref_charge": np.zeros((1, N_ATOMS), dtype=np.float32),
        "ref_atom_name_chars": np.zeros((1, N_ATOMS, 4), dtype=np.int64),
        "ref_space_uid": (np.arange(N_ATOMS, dtype=np.int64) // 3)[None],
        "atom_attention_mask": np.ones((1, N_ATOMS), dtype=np.float32),
        "atom_to_token": (np.arange(N_ATOMS, dtype=np.int64) // 3)[None],
        "distogram_atom_idx": (np.arange(N_TOKENS, dtype=np.int64) * 3)[None],
        "msa": msa,
        "msa_attention_mask": np.ones((1, MSA_DEPTH, N_TOKENS), dtype=np.float32),
        "has_deletion": np.zeros((1, MSA_DEPTH, N_TOKENS), dtype=np.float32),
        "deletion_value": np.zeros((1, MSA_DEPTH, N_TOKENS), dtype=np.float32),
    }


def test_parameter_cast_precedes_direct_device_transfer(
    tmp_path, monkeypatch: Any
) -> None:
    source = {
        "trunk.weight": np.asarray([1.25, -2.5], dtype=np.float32),
        # Buffers describe exact boundaries and retain their published dtype.
        "confidence.boundaries": np.asarray([0.0, 1.0], dtype=np.float32),
    }
    save_file(source, tmp_path / checkpoint.WEIGHTS_NAME)
    original_device_put = checkpoint.jax.device_put
    transferred: list[np.ndarray] = []

    def spy_device_put(value: np.ndarray) -> jax.Array:
        transferred.append(value)
        return original_device_put(value)

    def forbid_asarray(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("structure parameter transfer used a staged jnp.asarray")

    monkeypatch.setattr(checkpoint.jax, "device_put", spy_device_put)
    monkeypatch.setattr(checkpoint.jnp, "asarray", forbid_asarray)

    parameters = checkpoint.load_parameters(tmp_path, dtype="bfloat16")

    assert sorted(str(value.dtype) for value in transferred) == ["bfloat16", "float32"]
    assert str(parameters["trunk.weight"].dtype) == "bfloat16"
    assert parameters["confidence.boundaries"].dtype == jnp.float32
    np.testing.assert_array_equal(
        np.asarray(parameters["trunk.weight"]), source["trunk.weight"]
    )


def test_host_only_parameter_load_keeps_arrays_off_device(
    tmp_path, monkeypatch: Any
) -> None:
    save_file(
        {"trunk.weight": np.asarray([1.25, -2.5], dtype=np.float32)},
        tmp_path / checkpoint.WEIGHTS_NAME,
    )

    def forbid_device_put(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("host-only structure load transferred a parameter")

    monkeypatch.setattr(checkpoint.jax, "device_put", forbid_device_put)
    parameters = checkpoint.load_parameters(tmp_path, dtype="bfloat16", to_device=False)

    assert isinstance(parameters["trunk.weight"], np.ndarray)
    assert str(parameters["trunk.weight"].dtype) == "bfloat16"


def test_parameter_subset_does_not_materialize_unselected_leaves(
    tmp_path, monkeypatch: Any
) -> None:
    source = {
        "language_model.base_z_combine": np.arange(5, dtype=np.float32),
        "large.unused.weight": np.ones((32, 32), dtype=np.float32),
    }
    save_file(source, tmp_path / checkpoint.WEIGHTS_NAME)
    transferred: list[np.ndarray] = []
    original_device_put = checkpoint.jax.device_put

    def spy_device_put(value: np.ndarray) -> jax.Array:
        transferred.append(value)
        return original_device_put(value)

    monkeypatch.setattr(checkpoint.jax, "device_put", spy_device_put)
    parameters = checkpoint.load_parameters(
        tmp_path,
        names=("language_model.base_z_combine",),
    )

    assert set(parameters) == {"language_model.base_z_combine"}
    assert len(transferred) == 1
    np.testing.assert_array_equal(
        parameters["language_model.base_z_combine"],
        source["language_model.base_z_combine"],
    )


@pytest.mark.real_store
def test_the_released_settings_are_read_from_the_file() -> None:
    """Not the dataclass defaults, which differ in almost every field."""
    settings = checkpoint.load_settings(_weights())
    assert settings.trunk_n_layers == 48
    assert settings.num_recycles == 3
    assert settings.msa_n_layers == 4
    assert settings.diffusion.num_steps == 14
    # The churn the dataclass calls zero. A sampler that took the default
    # would run a noiseless ODE and never say so.
    assert settings.diffusion.noise_scale == pytest.approx(1.003)


def test_settings_keep_the_published_num_loops_checkpoint_key() -> None:
    settings = jax_model.settings_from_config({"num_loops": 3})

    assert settings.num_recycles == 3


def test_checkpoint_sampling_does_not_override_the_fixed_msa_cap() -> None:
    settings = jax_model.settings_from_config(
        {
            "num_loops": 7,
            "num_diffusion_samples": 9,
            "max_msa_depth": 17,
            "structure_head": {"inference_num_steps": 11},
        }
    )

    assert settings.num_recycles == 7
    assert settings.num_samples == 9
    assert settings.diffusion.num_steps == 11
    assert settings.max_msa_depth == 1024


@pytest.mark.slow
@pytest.mark.real_store
def test_the_whole_model_runs_on_the_released_weights() -> None:
    directory = _weights()
    parameters = checkpoint.load_parameters(directory)
    settings = jax_model.with_overrides(
        checkpoint.load_settings(directory), num_recycles=1, num_samples=2, num_steps=2
    )
    features = _features()
    hidden = np.zeros((1, N_TOKENS, 81, 2560), dtype=np.float32)

    output = jax_model.predict(
        jax.random.key(0),
        features,
        parameters,
        settings=settings,
        lm_hidden_states=hidden,
    )

    assert output["sample_atom_coords"].shape == (2, N_ATOMS, 3)
    assert output["plddt"].shape == (2, N_TOKENS)
    assert output["pair_chains_iptm"].shape == (2, 2, 2)
    for name, value in output.items():
        assert np.all(np.isfinite(np.asarray(value))), name
    # pLDDT is a probability-weighted mean over bins spanning 0 to 1.
    plddt = np.asarray(output["plddt"])
    assert plddt.min() >= 0.0 and plddt.max() <= 1.0


@pytest.mark.real_store
def test_the_weights_file_resolves_to_its_directory() -> None:
    """The store's `native=` entry names the file; the loader needs the folder.

    A caller has no reason to know the difference, and `foldjax predict` hands
    over whichever the store resolved.
    """
    from foldjax.models.esmfold2 import inference

    directory = _weights()
    model = inference.load(
        directory / checkpoint.WEIGHTS_NAME, language_model=False, dtype="bfloat16"
    )
    assert model.settings.trunk_n_layers == 48


@pytest.mark.slow
@pytest.mark.real_store
def test_a_job_folds_from_sequence_to_pdb(tmp_path) -> None:
    """Featurise, fold, write -- the whole backend path, without ESMC.

    The structure alone is not the released model; what this checks is that the
    path holds together from a sequence to a file, not that the fold is good.
    """
    from foldjax.models.esmfold2 import inference, output

    model = inference.load(_weights(), language_model=False)
    assert not model.has_language_model

    peptide = "MQIFVKTLTGKTITLE"
    prediction, features = inference.predict_job(
        inference.seed_key(0),
        [(peptide, "A", 0, 0)],
        None,
        model,
        num_recycles=0,
        num_samples=1,
        num_steps=2,
    )
    written = output.write_prediction_outputs(
        prediction, features, tmp_path, name="job"
    )

    import gemmi

    assert len(written["structures"]) == 1
    assert written["structures"][0].suffix == ".cif"
    structure = gemmi.read_structure(str(written["structures"][0]))
    atoms = [atom for chain in structure[0] for res in chain for atom in res]
    assert len(atoms) == int(features["atom_attention_mask"].sum())
    assert {chain.name for chain in structure[0]} == {"A"}
    assert all(np.isfinite([a.pos.x, a.pos.y, a.pos.z]).all() for a in atoms)
    assert 0.0 <= written["summary"][0]["plddt"] <= 1.0


def _realized_trunk_dtypes(monkeypatch, settings, parameters, features, hidden):
    """The dtype of every tensor the trunk and the MSA encoder actually run on.

    Spies rather than inspects, because the thing worth asserting is what the
    traced graph builds, and that is only visible from inside the call.
    """
    pair: list = []
    msa: list = []
    real_loops, real_msa = jax_model.run_loops, jax_model.msa_encoder

    def loops_spy(key, z, z_init, *args, **kwargs):
        pair.extend((z.dtype, z_init.dtype))
        out = real_loops(key, z, z_init, *args, **kwargs)
        pair.append(out.dtype)
        return out

    def msa_spy(injected, one_hot, mask, has_deletion, deletion_value, *a, **k):
        msa.extend(
            (injected.dtype, one_hot.dtype, has_deletion.dtype, deletion_value.dtype)
        )
        return real_msa(injected, one_hot, mask, has_deletion, deletion_value, *a, **k)

    monkeypatch.setattr(jax_model, "run_loops", loops_spy)
    monkeypatch.setattr(jax_model, "msa_encoder", msa_spy)
    jax_model.predict(
        jax.random.key(0),
        features,
        parameters,
        settings=settings,
        lm_hidden_states=hidden,
    )
    return pair, msa


@pytest.mark.slow
@pytest.mark.real_store
def test_the_trunk_runs_at_the_dtype_it_was_configured_for(monkeypatch) -> None:
    """`trunk_dtype` is a claim about tensors, so assert the tensors.

    `test_the_released_trunk_is_bfloat16` asserts the *setting*, and a setting
    cannot notice being ignored: the pair path is `z_init + rel_pos +
    token_bonds`, and one float32 term among those three widens `z_init`,
    which `z` follows, which is then the width of all forty-eight trunk
    layers. That is a bfloat16 trunk on paper running float32 in fact, and no
    shape, key or score check can see it -- only the realized dtype can.

    Both dtypes are exercised on purpose. Pinning bfloat16 alone would pass
    for a port that hard-coded it and ignored the setting just as thoroughly.
    """
    directory = _weights()
    parameters = checkpoint.load_parameters(directory)
    features = _features()
    hidden = np.zeros((1, N_TOKENS, 81, 2560), dtype=np.float32)

    for configured in ("bfloat16", "float32"):
        settings = dataclasses.replace(
            jax_model.with_overrides(
                checkpoint.load_settings(directory),
                num_recycles=1,
                num_samples=1,
                num_steps=1,
            ),
            trunk_dtype=configured,
        )
        pair, msa = _realized_trunk_dtypes(
            monkeypatch, settings, parameters, features, hidden
        )
        expected = jnp.dtype(configured)
        assert pair, "the spy never fired -- run_loops was not reached"
        assert all(seen == expected for seen in pair), (
            f"configured {configured}, trunk ran {sorted({str(d) for d in pair})}"
        )
        assert msa, "the spy never fired -- msa_encoder was not reached"
        assert all(seen == expected for seen in msa), (
            f"configured {configured}, MSA encoder ran {sorted({str(d) for d in msa})}"
        )


def test_experimental_and_malformed_checkpoint_types_are_rejected_before_weights(
    tmp_path, monkeypatch: Any
) -> None:
    import json

    from foldjax.models.esmfold2 import inference

    assert jax_model.settings_from_config({}).d_pair == 256
    assert jax_model.settings_from_config({"type": None}).d_pair == 256
    assert jax_model.settings_from_config({"type": "release"}).d_pair == 256

    with pytest.raises(ValueError, match="experimental"):
        jax_model.settings_from_config({"type": "experimental"})

    for bad_type in ("", "v2", 123, True):
        with pytest.raises(ValueError, match="unsupported ESMFold2 checkpoint type"):
            jax_model.settings_from_config({"type": bad_type})

    (tmp_path / checkpoint.CONFIG_NAME).write_text(
        json.dumps({"type": "experimental"}), encoding="utf-8"
    )
    save_file(
        {"trunk.weight": np.asarray([1.0], dtype=np.float32)},
        tmp_path / checkpoint.WEIGHTS_NAME,
    )

    def forbid_load_parameters(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("unsupported checkpoint config reached parameter loading")

    monkeypatch.setattr(checkpoint, "load_parameters", forbid_load_parameters)

    with pytest.raises(ValueError, match="experimental"):
        inference.load(tmp_path, language_model=False)
    with pytest.raises(ValueError, match="experimental"):
        inference.load_language_model_stage(tmp_path)


def test_rope_config_fields_reach_inputs_and_diffusion_and_validate_bounds() -> None:
    from foldjax.models.esmfold2.models import atom, diffusion

    custom_cfg = {
        "inputs": {
            "atom_encoder": {
                "d_atom": 64,
                "n_heads": 2,
                "n_blocks": 1,
                "spatial_rope_base_frequency": 50.0,
                "n_spatial_rope_pairs_per_axis": 1,
                "n_uid_rope_pairs": 5,
                "uid_rope_base_frequency": 500.0,
            }
        },
        "structure_head": {
            "diffusion_module": {
                "c_atom": 64,
                "atom_num_heads": 2,
                "atom_num_blocks": 1,
            }
        },
    }
    settings = jax_model.settings_from_config(custom_cfg)
    assert settings.inputs_spatial_rope_base_frequency == 50.0
    assert settings.inputs_n_spatial_rope_pairs_per_axis == 1
    assert settings.inputs_n_uid_rope_pairs == 5
    assert settings.inputs_uid_rope_base_frequency == 500.0
    assert settings.diffusion.spatial_rope_base_frequency == 50.0
    assert settings.diffusion.n_spatial_rope_pairs_per_axis == 1
    assert settings.diffusion.n_uid_rope_pairs == 5
    assert settings.diffusion.uid_rope_base_frequency == 500.0

    rng = np.random.default_rng(11)
    ref_pos = jnp.asarray(rng.normal(size=(1, 8, 3)).astype(np.float32) * 3.0)
    ref_uid = jnp.arange(8, dtype=jnp.int32)[None]
    atom_mask = jnp.ones((1, 8), dtype=jnp.float32)
    ref_charge = jnp.zeros((1, 8), dtype=jnp.float32)
    elem_oh = jnp.zeros((1, 8, 128), dtype=jnp.float32).at[..., 6].set(1.0)
    chars_oh = jnp.zeros((1, 8, 4, 64), dtype=jnp.float32).at[..., 0].set(1.0)
    atom_to_tok = (jnp.arange(8, dtype=jnp.int32) // 2)[None]

    def zero_atom_params(prefix: str) -> dict[str, jnp.ndarray]:
        dot = f"{prefix}."
        p = {
            f"{dot}atom_linear.weight": jnp.zeros((64, 389), jnp.float32),
            f"{dot}atom_norm.weight": jnp.ones((64,), jnp.float32),
            f"{dot}atom_norm.bias": jnp.zeros((64,), jnp.float32),
            f"{dot}atom_to_token_linear.weight": jnp.zeros((384, 64), jnp.float32),
            f"{dot}atom_transformer.blocks.0.adaln_modulation.1.weight": jnp.zeros(
                (6 * 64, 64), jnp.float32
            ),
            f"{dot}atom_transformer.blocks.0.attn.Wqkv.weight": jnp.zeros(
                (3 * 64, 64), jnp.float32
            ),
            f"{dot}atom_transformer.blocks.0.attn.gate_proj.weight": jnp.zeros(
                (64, 64), jnp.float32
            ),
            f"{dot}atom_transformer.blocks.0.attn.out_proj.weight": jnp.zeros(
                (64, 64), jnp.float32
            ),
            f"{dot}atom_transformer.blocks.0.ffn.w_up.weight": jnp.zeros(
                (256, 64), jnp.float32
            ),
            f"{dot}atom_transformer.blocks.0.ffn.w_down.weight": jnp.zeros(
                (64, 128), jnp.float32
            ),
        }
        return p

    expected_cos, expected_sin = atom.build_3d_rope(
        ref_pos,
        ref_uid,
        head_dim=32,
        n_spatial_per_axis=1,
        n_uid_pairs=5,
        spatial_base_frequency=50.0,
        uid_base_frequency=500.0,
    )
    default_cos, _ = atom.build_3d_rope(ref_pos, ref_uid, head_dim=32)
    assert not np.array_equal(np.asarray(expected_cos), np.asarray(default_cos))

    seen_rope: list[tuple[jnp.ndarray, jnp.ndarray]] = []
    real_transformer = atom.atom_transformer

    def capture_transformer(*args: Any, **kwargs: Any) -> jnp.ndarray:
        seen_rope.append((kwargs["cos"], kwargs["sin"]))
        return real_transformer(*args, **kwargs)

    inp_params = zero_atom_params("inputs_embedder.atom_attention_encoder")
    orig_transformer = atom.atom_transformer
    try:
        atom.atom_transformer = capture_transformer  # type: ignore[assignment]
        jax_model.inputs_embedding(
            jnp.zeros((1, 4, 33), jnp.float32),
            jnp.zeros((1, 4, 33), jnp.float32),
            jnp.zeros((1, 4), jnp.float32),
            ref_pos,
            atom_mask,
            ref_uid,
            ref_charge,
            elem_oh,
            chars_oh,
            atom_to_tok,
            inp_params,
            settings=settings,
            n_tokens=4,
        )
    finally:
        atom.atom_transformer = orig_transformer  # type: ignore[assignment]
    np.testing.assert_array_equal(seen_rope[0][0], expected_cos)
    np.testing.assert_array_equal(seen_rope[0][1], expected_sin)

    diff_params = zero_atom_params("atom_encoder")
    diff_params.update(
        {
            "conditioning.z_input_norm.weight": jnp.ones((512,), jnp.float32),
            "conditioning.z_input_norm.bias": jnp.zeros((512,), jnp.float32),
            "conditioning.z_proj.weight": jnp.zeros((256, 512), jnp.float32),
        }
    )
    for idx in range(2):
        diff_params[f"conditioning.z_transitions.{idx}.norm.weight"] = jnp.ones(
            (256,), jnp.float32
        )
        diff_params[f"conditioning.z_transitions.{idx}.norm.bias"] = jnp.zeros(
            (256,), jnp.float32
        )
        for proj, shape in (
            ("a_proj", (512, 256)),
            ("b_proj", (512, 256)),
            ("out_proj", (256, 512)),
        ):
            diff_params[f"conditioning.z_transitions.{idx}.{proj}.weight"] = jnp.zeros(
                shape, jnp.float32
            )
    cache = diffusion.build_cache(
        ref_pos,
        ref_charge,
        atom_mask,
        elem_oh,
        chars_oh,
        ref_uid,
        atom_to_tok,
        jnp.zeros((1, 4, 4, 256), jnp.float32),
        jnp.zeros((1, 4, 4, 256), jnp.float32),
        diff_params,
        settings=settings.diffusion,
        n_tokens=4,
    )
    np.testing.assert_array_equal(cache.cos, expected_cos)
    np.testing.assert_array_equal(cache.sin, expected_sin)

    # Edge cases: zero pairs (identity) and exact half-head limit (3*4 + 4 == 16).
    zero_cos, zero_sin = atom.build_3d_rope(
        ref_pos, ref_uid, head_dim=32, n_spatial_per_axis=0, n_uid_pairs=0
    )
    np.testing.assert_array_equal(zero_cos, jnp.ones((1, 8, 16), jnp.bfloat16))
    np.testing.assert_array_equal(zero_sin, jnp.zeros((1, 8, 16), jnp.bfloat16))

    limit_cfg = {
        "inputs": {
            "atom_encoder": {
                "n_spatial_rope_pairs_per_axis": 4,
                "n_uid_rope_pairs": 4,
            }
        }
    }
    assert (
        jax_model.settings_from_config(limit_cfg).inputs_n_spatial_rope_pairs_per_axis
        == 4
    )

    for bad_atom_cfg in (
        {"n_spatial_rope_pairs_per_axis": 4, "n_uid_rope_pairs": 5},  # 17 > 16
        {"n_spatial_rope_pairs_per_axis": -1},
        {"n_uid_rope_pairs": True},
        {"spatial_rope_base_frequency": 0.0},
        {"uid_rope_base_frequency": -10.0},
        {"spatial_rope_base_frequency": float("nan")},
        {"uid_rope_base_frequency": False},
    ):
        with pytest.raises(ValueError):
            jax_model.settings_from_config({"inputs": {"atom_encoder": bad_atom_cfg}})

    for bad_head_dim in (0, -2, 3, True):
        with pytest.raises(ValueError, match="head_dim"):
            atom.build_3d_rope(ref_pos, ref_uid, head_dim=bad_head_dim)  # type: ignore[arg-type]
