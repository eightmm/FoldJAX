from __future__ import annotations

import os
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import foldjax.models.opendde.bridge.torch_mapping as mapping_impl
import foldjax.models.opendde.models.model as model_impl
from foldjax.models.opendde.models.model import (
    OpenDDEInferenceParams,
    prepare_structural_features,
)


def _static_validation_features() -> dict[str, object]:
    return {
        "restype": jnp.zeros((2, 32), dtype=jnp.float32),
        "ref_pos": jnp.zeros((3, 3), dtype=jnp.float32),
        "token_index": jnp.arange(2, dtype=jnp.int32),
        "asym_id": jnp.asarray([0, 0], dtype=jnp.int32),
        "residue_index": jnp.asarray([10, 11], dtype=jnp.int32),
        "entity_id": jnp.asarray([0, 0], dtype=jnp.int32),
        "sym_id": jnp.asarray([0, 0], dtype=jnp.int32),
        "atom_to_token_idx": jnp.asarray([0, 1, 1], dtype=jnp.int32),
        "atom_to_tokatom_idx": jnp.asarray([0, 0, 1], dtype=jnp.int32),
        "has_frame": jnp.asarray([True, True]),
        "frame_atom_index": jnp.zeros((2, 3), dtype=jnp.int32),
        "pae_rep_atom_mask": jnp.asarray([1, 1, 0], dtype=jnp.int32),
        "distogram_rep_atom_mask": jnp.asarray([1, 1, 0], dtype=jnp.int32),
        "parent_residue_idx": jnp.asarray([0, 0, 1], dtype=jnp.int32),
        "subtoken_role_id": jnp.asarray([1, 2, 1], dtype=jnp.int32),
        "structural_token_index": jnp.arange(3, dtype=jnp.int32),
        "atom_to_structural_token_idx": jnp.asarray(
            [0, 2, 2],
            dtype=jnp.int32,
        ),
        "atom_to_structural_tokatom_idx": jnp.asarray(
            [0, 0, 1],
            dtype=jnp.int32,
        ),
        "structural_distogram_rep_atom_mask": jnp.asarray(
            [1, 1, 0],
            dtype=jnp.int32,
        ),
        "structural_pae_rep_atom_mask": jnp.asarray(
            [1, 0, 1],
            dtype=jnp.int32,
        ),
        "structural_has_frame": jnp.asarray([True, False, True]),
        "structural_frame_atom_index": jnp.zeros((3, 3), dtype=jnp.int32),
    }


def test_static_structural_validation_rejects_batched_metadata() -> None:
    features = _static_validation_features()
    features["parent_residue_idx"] = jnp.zeros((1, 3), dtype=jnp.int32)
    features["subtoken_role_id"] = jnp.zeros((1, 3), dtype=jnp.int32)

    with pytest.raises(ValueError, match="unbatched"):
        model_impl._validate_static_structural_features(
            features,
            require_residue_confidence_mask=True,
        )


def test_static_structural_validation_rejects_duplicate_representative_atoms() -> None:
    features = _static_validation_features()
    features["atom_to_token_idx"] = jnp.asarray([0, 0, 1], dtype=jnp.int32)

    with pytest.raises(ValueError, match="one representative atom per residue token"):
        model_impl._validate_static_structural_features(
            features,
            require_residue_confidence_mask=True,
        )


def test_static_validation_requires_residue_aliases_without_confidence() -> None:
    features = _static_validation_features()
    del features["pae_rep_atom_mask"]

    with pytest.raises(KeyError, match="pae_rep_atom_mask"):
        model_impl._validate_static_structural_features(
            features,
            require_residue_confidence_mask=False,
        )


def test_prepare_structural_features_replaces_aliases_and_recomputes_relp() -> None:
    features = {
        "token_index": jnp.asarray([0, 1], dtype=jnp.int32),
        "asym_id": jnp.asarray([3, 4], dtype=jnp.int32),
        "residue_index": jnp.asarray([10, 20], dtype=jnp.int32),
        "entity_id": jnp.asarray([5, 6], dtype=jnp.int32),
        "sym_id": jnp.asarray([0, 1], dtype=jnp.int32),
        "atom_to_token_idx": jnp.asarray([0, 1, 1], dtype=jnp.int32),
        "atom_to_tokatom_idx": jnp.asarray([0, 0, 1], dtype=jnp.int32),
        "has_frame": jnp.asarray([True, True]),
        "frame_atom_index": jnp.zeros((2, 3), dtype=jnp.int32),
        "pae_rep_atom_mask": jnp.asarray([1, 1, 0], dtype=jnp.int32),
        "distogram_rep_atom_mask": jnp.asarray([1, 0, 1], dtype=jnp.int32),
        "parent_residue_idx": jnp.asarray([0, 0, 1], dtype=jnp.int32),
        "structural_token_index": jnp.asarray([0, 1, 0], dtype=jnp.int32),
        "atom_to_structural_token_idx": jnp.asarray([0, 2, 2], dtype=jnp.int32),
        "atom_to_structural_tokatom_idx": jnp.asarray(
            [0, 0, 1],
            dtype=jnp.int32,
        ),
        "structural_has_frame": jnp.asarray([True, False, True]),
        "structural_frame_atom_index": jnp.zeros((3, 3), dtype=jnp.int32),
        "structural_pae_rep_atom_mask": jnp.asarray([1, 0, 1], dtype=jnp.int32),
        "structural_distogram_rep_atom_mask": jnp.asarray(
            [1, 1, 0],
            dtype=jnp.int32,
        ),
        "msa": jnp.ones((1, 2, 32), dtype=jnp.float32),
        "template_restype": jnp.ones((1, 2, 32), dtype=jnp.float32),
    }
    pair = {
        "structural_pair_attn_bias": jnp.arange(9, dtype=jnp.float32).reshape(
            3,
            3,
        )
    }

    actual = prepare_structural_features(features, pair)

    np.testing.assert_array_equal(actual["asym_id"], np.asarray([3, 3, 4]))
    np.testing.assert_array_equal(actual["residue_index"], np.asarray([10, 10, 20]))
    np.testing.assert_array_equal(actual["token_index"], np.asarray([0, 1, 0]))
    np.testing.assert_array_equal(
        actual["atom_to_token_idx"],
        np.asarray([0, 2, 2]),
    )
    assert actual["relp"].shape == (3, 3, 139)
    assert "msa" not in actual
    assert "template_restype" not in actual
    np.testing.assert_array_equal(
        actual["residue_level_token_index"],
        features["token_index"],
    )
    np.testing.assert_array_equal(
        actual["structural_pair_attn_bias"],
        pair["structural_pair_attn_bias"],
    )

    compact = prepare_structural_features(features, pair, include_relp=False)
    assert "relp" not in compact


def test_full_inference_mapper_composes_open_dde_specific_modules(
    monkeypatch,
) -> None:
    sentinels = {
        "input_embedder": object(),
        "pairformer_output": object(),
        "structural_expander": object(),
        "structural_refiner": object(),
        "diffusion": object(),
        "distogram": object(),
        "confidence": object(),
    }
    monkeypatch.setattr(
        mapping_impl,
        "map_input_feature_embedder_state_dict",
        lambda state, prefix: sentinels["input_embedder"],
    )
    monkeypatch.setattr(
        mapping_impl,
        "map_pairformer_output_state_dict",
        lambda state: sentinels["pairformer_output"],
    )
    monkeypatch.setattr(
        mapping_impl,
        "map_structural_token_expander_state_dict",
        lambda state: sentinels["structural_expander"],
    )
    monkeypatch.setattr(
        mapping_impl,
        "map_released_structural_refiner_state_dict",
        lambda state: sentinels["structural_refiner"],
    )
    monkeypatch.setattr(
        mapping_impl,
        "map_released_diffusion_module_state_dict",
        lambda state: sentinels["diffusion"],
    )
    monkeypatch.setattr(
        mapping_impl,
        "map_released_distogram_state_dict",
        lambda state: sentinels["distogram"],
    )
    monkeypatch.setattr(
        mapping_impl,
        "map_confidence_head_state_dict",
        lambda state, prefix: sentinels["confidence"],
    )

    actual = mapping_impl.map_opendde_inference_state_dict({})

    for name, value in sentinels.items():
        assert getattr(actual, name) is value


@pytest.mark.parametrize("return_representations", [True, False])
def test_static_inference_routes_residue_heads_and_structural_diffusion(
    monkeypatch, return_representations: bool
) -> None:
    """Both output contracts, including the default.

    The default stopped returning the six representation tensors -- the
    structural pair one alone was 1,311 MiB of a 1,869 MiB output, and
    returning it held the refiner's working buffer live to the end of the
    program. Only the opted-in contract was covered, so nothing checked that
    the default still returns everything the CIF writer and the confidence
    summaries read.
    """
    monkeypatch.delenv("PROTENIX_TRIANGLE_BACKEND", raising=False)
    monkeypatch.delenv("PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND", raising=False)
    residue_s_inputs = jnp.full((2, 5), 1.0)
    residue_s = jnp.full((2, 4), 2.0)
    residue_z = jnp.full((2, 2, 3), 3.0)
    structural_s_inputs = jnp.full((3, 5), 4.0)
    structural_s = jnp.full((3, 4), 5.0)
    structural_z = jnp.full((3, 3, 3), 6.0)
    refined_s = structural_s + 1.0
    refined_z = structural_z + 1.0
    pair_bias = jnp.arange(9, dtype=jnp.float32).reshape(3, 3)
    pair_cache = jnp.full((3, 3, 2), 8.0)
    coordinates = jnp.full((1, 3, 3), 9.0)
    structural_atom_map = jnp.asarray([0, 2, 2], dtype=jnp.int32)
    residue_pair_mask = jnp.ones((2, 2), dtype=jnp.float32)
    features = _static_validation_features()
    features["atom_to_structural_token_idx"] = structural_atom_map
    features.update(
        {
            "ref_charge": jnp.zeros((3,), dtype=jnp.float32),
            "ref_mask": jnp.ones((3,), dtype=jnp.float32),
            "ref_element": jnp.zeros((3, 128), dtype=jnp.float32),
            "ref_atom_name_chars": jnp.zeros((3, 4, 64), dtype=jnp.float32),
            "d_lm": jnp.zeros((1, 1, 3), dtype=jnp.float32),
            "v_lm": jnp.ones((1, 1), dtype=jnp.float32),
            "pad_info": {},
        }
    )
    structural_features = {
        **features,
        "atom_to_token_idx": structural_atom_map,
        "relp": jnp.zeros((3, 3, 139), dtype=jnp.float32),
        "structural_pair_attn_bias": pair_bias,
    }
    params = OpenDDEInferenceParams(
        input_embedder=object(),
        pairformer_output=object(),
        structural_expander=object(),
        structural_refiner=object(),
        diffusion=SimpleNamespace(
            conditioning=SimpleNamespace(relpe=object()),
            atom_encoder=object(),
        ),
        distogram=object(),
        confidence=object(),
    )

    monkeypatch.setattr(
        model_impl,
        "input_feature_embedder",
        lambda *args, **kwargs: residue_s_inputs,
    )

    def fake_pairformer(*args, **kwargs):
        assert kwargs["pair_mask"] is residue_pair_mask
        # Both fused, which is what upstream runs: `auto` resolves to
        # cuequivariance in config/model_base.py:31-32 and OpenDDE's
        # c_hidden == c_z == 384 clears the kernel's guard. These were pinned to
        # XLA on a 1,531-token OOM, which was one size generalised to every
        # size -- the same job peaks at 10,552 MiB at 490 tokens and 34,408 at
        # 970, with 60+ GiB spare. An explicit environment still wins, which is
        # how the largest jobs ask for the blocked path.
        assert os.environ["PROTENIX_TRIANGLE_BACKEND"] == "cueq_jit"
        assert os.environ["PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND"] == "cueq"
        return residue_s_inputs, residue_s, residue_z

    monkeypatch.setattr(
        model_impl,
        "pairformer_output_from_s_inputs",
        fake_pairformer,
    )
    monkeypatch.setattr(
        model_impl,
        "structural_token_expand",
        lambda *args, **kwargs: (
            structural_s_inputs,
            structural_s,
            structural_z,
            {"structural_pair_attn_bias": pair_bias},
        ),
    )
    monkeypatch.setattr(
        model_impl,
        "prepare_structural_features",
        lambda *args, **kwargs: structural_features,
    )

    def fake_refiner(s, z, pair_mask, extra_attn_bias, params, **kwargs):
        assert s is structural_s
        assert z is structural_z
        assert pair_mask is None
        assert extra_attn_bias is pair_bias
        return refined_s, refined_z

    monkeypatch.setattr(model_impl, "structural_refiner_stack", fake_refiner)

    structural_relp_encoding = jnp.full((3, 3, 2), 7.0)
    monkeypatch.setattr(
        model_impl,
        "relative_position_encoding_from_features",
        lambda actual, _params: (
            structural_relp_encoding
            if actual is structural_features
            else pytest.fail("unexpected structural features")
        ),
    )

    def fake_pair_cache(relp, z, params, *, relp_encoding):
        assert relp is None
        assert relp_encoding is structural_relp_encoding
        assert z is refined_z
        return pair_cache

    monkeypatch.setattr(
        model_impl,
        "diffusion_conditioning_prepare_cache",
        fake_pair_cache,
    )

    def fake_atom_cache(atom_to_token_idx, *args, z, **kwargs):
        np.testing.assert_array_equal(atom_to_token_idx, structural_atom_map)
        np.testing.assert_array_equal(z, pair_cache[None, ...])
        return object(), object()

    monkeypatch.setattr(
        model_impl,
        "atom_attention_encoder_prepare_diffusion_cache",
        fake_atom_cache,
    )

    def fake_denoiser(
        atom_to_token_idx,
        *args,
        s_inputs,
        s_trunk,
        z_trunk,
        extra_attn_bias,
        **kwargs,
    ):
        np.testing.assert_array_equal(atom_to_token_idx, structural_atom_map)
        assert s_inputs is structural_s_inputs
        assert s_trunk is refined_s
        assert z_trunk is refined_z
        assert extra_attn_bias is pair_bias
        return jnp.zeros((1, 3, 3), dtype=jnp.float32)

    monkeypatch.setattr(model_impl, "diffusion_module_forward", fake_denoiser)

    def fake_sampler(denoise_fn, *args, **kwargs):
        denoise_fn(
            jnp.zeros((1, 3, 3), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
        )
        return coordinates

    monkeypatch.setattr(model_impl, "sample_diffusion", fake_sampler)

    def fake_distogram(z, params):
        assert z is residue_z
        return jnp.full((2, 2, 2), 10.0)

    monkeypatch.setattr(model_impl, "distogram_head", fake_distogram)

    def fake_confidence(
        input_features,
        s_inputs,
        s_trunk,
        z_trunk,
        pair_mask,
        x_pred_coords,
        params,
        **kwargs,
    ):
        assert input_features is features
        assert s_inputs is residue_s_inputs
        assert s_trunk is residue_s
        assert z_trunk is residue_z
        assert pair_mask is None
        assert x_pred_coords is coordinates
        return {"plddt": jnp.full((1, 3, 2), 11.0)}

    monkeypatch.setattr(model_impl, "confidence_head", fake_confidence)

    actual = model_impl.opendde_infer_static(
        features,
        params,
        jnp.asarray([1.0, 0.0]),
        key=None,
        num_samples=1,
        pair_mask=residue_pair_mask,
        return_representations=return_representations,
    )

    representations = (
        "s_inputs",
        "s_trunk",
        "z_trunk",
        "structural_s_inputs",
        "structural_s_trunk",
        "structural_z_trunk",
    )
    if return_representations:
        assert actual["s_inputs"] is residue_s_inputs
        assert actual["s_trunk"] is residue_s
        assert actual["z_trunk"] is residue_z
        assert actual["structural_s_inputs"] is structural_s_inputs
        assert actual["structural_s_trunk"] is refined_s
        assert actual["structural_z_trunk"] is refined_z
    else:
        assert not [name for name in representations if name in actual]
    # Whatever the contract, everything the outputs are built from stays.
    assert actual["coordinate"] is coordinates
    assert actual["distogram_logits"].shape == (2, 2, 2)
    assert actual["plddt"].shape == (1, 3, 2)
    assert "PROTENIX_TRIANGLE_BACKEND" not in os.environ
    assert "PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND" not in os.environ


def test_opendde_trunk_requests_its_own_msa_ordering() -> None:
    """OpenDDE must ask the shared trunk for the MSA-stack-first order.

    The trunk function comes from the Protenix port, whose model runs the two
    MSA sub-updates the other way round. Nothing about the shapes distinguishes
    them, so this is the only place the difference is stated, and getting it
    wrong is silent: it cost OpenDDE a pair representation correlating 0.47
    with upstream's and 31.5 A of coordinate error on a matched random tape,
    while every confidence score stayed within 1% of upstream's.
    """

    import inspect

    from foldjax.models.opendde.models import model as opendde_model

    source = inspect.getsource(opendde_model)
    trunk_call = source.split("pairformer_output_from_s_inputs(", 1)[1].split(
        "\n    )", 1
    )[0]
    assert "msa_stack_first=True" in trunk_call
