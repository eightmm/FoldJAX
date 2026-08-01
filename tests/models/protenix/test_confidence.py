from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.protenix.models.heads.confidence import (
    RDKIT_VDWS,
    ConfidenceDistanceEmbeddingParams,
    ConfidenceHeadParams,
    ConfidenceOutputParams,
    _contract_bins,
    calculate_chain_based_gpde,
    calculate_chain_based_plddt,
    calculate_chain_based_ptm,
    calculate_chain_pair_pae,
    calculate_clash,
    calculate_iptm,
    calculate_normalization,
    calculate_ptm,
    calculate_vdw_clash,
    compute_contact_prob,
    confidence_distance_embedding,
    confidence_head,
    confidence_one_hot,
    confidence_output_logits,
    confidence_scores_from_logits,
    get_bin_centers,
    logits_to_score,
)
from foldjax.models.protenix.models.primitives.primitives import (
    LayerNormParams,
    LinearParams,
)
from foldjax.models.protenix.models.trunk_blocks.pairformer import PairformerStackParams


def test_confidence_one_hot_uses_open_bin_edges() -> None:
    x = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    lower = jnp.asarray([1.0, 2.0])
    upper = jnp.asarray([2.0, 4.0])

    result = confidence_one_hot(x, lower, upper)

    np.testing.assert_array_equal(
        np.asarray(result),
        np.asarray(
            [
                [False, False],
                [False, False],
                [False, True],
                [False, False],
            ],
        ),
    )


def test_confidence_distance_embedding_combines_binned_and_scalar_distance() -> None:
    params = ConfidenceDistanceEmbeddingParams(
        lower_bins=jnp.asarray([0.0, 1.5]),
        upper_bins=jnp.asarray([1.5, 10.0]),
        linear_d=LinearParams(
            weight=jnp.asarray([[10.0, 1.0], [100.0, 2.0]]),
            bias=None,
        ),
        linear_d_wo_onehot=LinearParams(
            weight=jnp.asarray([[0.5], [2.0]]),
            bias=None,
        ),
    )
    coords = jnp.asarray([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])

    result = confidence_distance_embedding(coords, params)

    expected = np.asarray(
        [
            [[0.0, 0.0], [1.0 + 2.5, 2.0 + 10.0]],
            [[1.0 + 2.5, 2.0 + 10.0], [0.0, 0.0]],
        ],
    )
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-6, atol=1e-6)


def test_confidence_output_logits_project_pair_and_atom_outputs() -> None:
    params = ConfidenceOutputParams(
        pae_ln=LayerNormParams(weight=jnp.ones((2,)), bias=jnp.zeros((2,))),
        pde_ln=LayerNormParams(weight=jnp.ones((2,)), bias=jnp.zeros((2,))),
        plddt_ln=LayerNormParams(weight=jnp.ones((2,)), bias=jnp.zeros((2,))),
        resolved_ln=LayerNormParams(weight=jnp.ones((2,)), bias=jnp.zeros((2,))),
        linear_pae=LinearParams(weight=jnp.asarray([[1.0, -1.0]]), bias=None),
        linear_pde=LinearParams(weight=jnp.asarray([[2.0, 1.0]]), bias=None),
        plddt_weight=jnp.asarray(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[2.0, 0.0], [0.0, 3.0]],
            ],
        ),
        resolved_weight=jnp.asarray(
            [
                [[0.5], [1.5]],
                [[1.0], [2.0]],
            ],
        ),
    )
    s_single = jnp.asarray([[1.0, 3.0], [2.0, 6.0]])
    z_pair = jnp.asarray(
        [
            [[1.0, 3.0], [2.0, 4.0]],
            [[5.0, 7.0], [8.0, 10.0]],
        ],
    )
    atom_to_token_idx = jnp.asarray([0, 1])
    atom_to_tokatom_idx = jnp.asarray([0, 1])

    output = confidence_output_logits(
        s_single,
        z_pair,
        atom_to_token_idx,
        atom_to_tokatom_idx,
        params,
    )

    assert set(output) == {"plddt", "pae", "pde", "resolved"}
    assert output["pae"].shape == (2, 2, 1)
    assert output["pde"].shape == (2, 2, 1)
    assert output["plddt"].shape == (2, 2)
    assert output["resolved"].shape == (2, 1)
    np.testing.assert_allclose(np.asarray(output["plddt"][0]), [-1.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(np.asarray(output["resolved"][1]), [1.0], atol=1e-5)


def test_get_bin_centers_matches_protenix_formula() -> None:
    centers = get_bin_centers(min_bin=2.0, max_bin=10.0, no_bins=4)

    np.testing.assert_allclose(
        np.asarray(centers),
        np.asarray([3.0, 5.0, 7.0, 9.0]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_logits_to_score_uses_softmax_weighted_bin_centers() -> None:
    logits = jnp.asarray([[0.0, 0.0], [-20.0, 20.0]], dtype=jnp.float32)

    score, prob = logits_to_score(
        logits,
        min_bin=0.0,
        max_bin=2.0,
        no_bins=2,
        return_prob=True,
    )

    np.testing.assert_allclose(np.asarray(score), [1.0, 1.5], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(prob[0]),
        [0.5, 0.5],
        rtol=1e-6,
        atol=1e-6,
    )


def test_compute_contact_prob_sums_bins_below_threshold() -> None:
    logits = jnp.zeros((1, 1, 4), dtype=jnp.float32)

    contact = compute_contact_prob(
        logits,
        min_bin=0.0,
        max_bin=4.0,
        no_bins=4,
        thres=2.0,
    )

    np.testing.assert_allclose(
        np.asarray(contact),
        np.asarray([[0.5]]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_confidence_scores_from_logits_returns_basic_full_data() -> None:
    plddt = jnp.asarray([[[0.0, 0.0], [-20.0, 20.0]]], dtype=jnp.float32)
    pae = jnp.zeros((1, 2, 2, 2), dtype=jnp.float32)
    pde = jnp.ones((1, 2, 2, 2), dtype=jnp.float32)
    distogram = jnp.asarray(
        [
            [[20.0, -20.0], [20.0, -20.0]],
            [[20.0, -20.0], [20.0, -20.0]],
        ],
        dtype=jnp.float32,
    )

    scores = confidence_scores_from_logits(
        plddt_logits=plddt,
        pae_logits=pae,
        pde_logits=pde,
        distogram_logits=distogram,
        plddt_max_bin=2.0,
        pae_max_bin=2.0,
        pde_max_bin=2.0,
        distogram_min_bin=0.0,
        distogram_max_bin=2.0,
        contact_threshold=1.0,
    )

    assert scores["atom_plddt"].shape == (1, 2)
    assert scores["token_pair_pae"].shape == (1, 2, 2)
    assert scores["token_pair_pde"].shape == (1, 2, 2)
    assert scores["contact_probs"].shape == (2, 2)
    assert scores["summary_plddt"].shape == (1,)
    assert scores["summary_gpde"].shape == (1,)
    np.testing.assert_allclose(
        np.asarray(scores["summary_plddt"]),
        [125.0],
        rtol=1e-6,
        atol=1e-6,
    )


def test_calculate_ptm_matches_protenix_row_mean_then_frame_max() -> None:
    pae_prob = jnp.asarray(
        [
            [
                [[1.0, 0.0], [1.0, 0.0]],
                [[0.0, 1.0], [0.0, 1.0]],
            ]
        ],
        dtype=jnp.float32,
    )
    has_frame = jnp.asarray([True, True])
    norm = calculate_normalization(2)
    weights = 1.0 / (1.0 + (np.asarray([0.5, 1.5]) / norm) ** 2)
    expected = max(weights[0], weights[1])

    ptm = calculate_ptm(
        pae_prob,
        has_frame,
        min_bin=0.0,
        max_bin=2.0,
        no_bins=2,
    )

    np.testing.assert_allclose(np.asarray(ptm), [expected], rtol=1e-6, atol=1e-6)


def test_calculate_iptm_uses_inter_chain_columns_only() -> None:
    pae_prob = jnp.asarray(
        [
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ]
        ],
        dtype=jnp.float32,
    )
    has_frame = jnp.asarray([True, True])
    asym_id = jnp.asarray([0, 1])
    norm = calculate_normalization(2)
    weights = 1.0 / (1.0 + (np.asarray([0.5, 1.5]) / norm) ** 2)
    expected = max(weights[1], weights[0])

    iptm = calculate_iptm(
        pae_prob,
        has_frame,
        asym_id,
        min_bin=0.0,
        max_bin=2.0,
        no_bins=2,
    )

    np.testing.assert_allclose(np.asarray(iptm), [expected], rtol=1e-6, atol=1e-6)


def test_calculate_chain_based_ptm_matches_upstream_shapes_and_ligand_global() -> None:
    logits = jnp.asarray(
        np.random.default_rng(7).normal(size=(2, 4, 4, 3)), dtype=jnp.float32
    )
    pae_prob = jax.nn.softmax(logits, axis=-1)
    out = calculate_chain_based_ptm(
        pae_prob,
        has_frame=jnp.asarray([True, True, True, True]),
        asym_id=jnp.asarray([0, 0, 1, 2]),
        token_is_ligand=jnp.asarray([False, False, True, False]),
        min_bin=0.0,
        max_bin=3.0,
        no_bins=3,
    )

    assert out["chain_ptm"].shape == (2, 3)
    assert out["chain_iptm"].shape == (2, 3)
    assert out["chain_pair_iptm"].shape == (2, 3, 3)
    assert out["chain_pair_iptm_global"].shape == (2, 3, 3)
    np.testing.assert_allclose(
        np.asarray(out["chain_pair_iptm"]),
        np.swapaxes(np.asarray(out["chain_pair_iptm"]), -1, -2),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(out["chain_pair_iptm_global"][:, 0, 1]),
        np.asarray(out["chain_iptm"][:, 1]),
        atol=1e-6,
    )


def test_confidence_scores_exposes_complete_upstream_summary_contract() -> None:
    logits = jnp.zeros((1, 3, 3, 2), dtype=jnp.float32)
    scores = confidence_scores_from_logits(
        plddt_logits=jnp.zeros((1, 3, 2), dtype=jnp.float32),
        pae_logits=logits,
        pde_logits=logits,
        distogram_logits=jnp.zeros((3, 3, 2), dtype=jnp.float32),
        token_has_frame=jnp.asarray([True, True, True]),
        token_asym_id=jnp.asarray([0, 1, 1]),
        token_is_ligand=jnp.asarray([False, True, True]),
        num_recycles=10,
    )

    for key in (
        "chain_ptm",
        "chain_iptm",
        "chain_pair_iptm",
        "chain_pair_iptm_global",
        "disorder",
        "num_recycles",
        "ranking_score",
    ):
        assert key in scores
    np.testing.assert_array_equal(np.asarray(scores["num_recycles"]), 10)
    np.testing.assert_array_equal(np.asarray(scores["disorder"]), [0.0])


def test_confidence_scores_adds_ptm_iptm_when_masks_are_given() -> None:
    logits = jnp.zeros((1, 2, 2, 2), dtype=jnp.float32)
    scores = confidence_scores_from_logits(
        plddt_logits=jnp.zeros((1, 2, 2), dtype=jnp.float32),
        pae_logits=logits,
        pde_logits=logits,
        distogram_logits=jnp.zeros((2, 2, 2), dtype=jnp.float32),
        plddt_max_bin=2.0,
        pae_max_bin=2.0,
        pde_max_bin=2.0,
        distogram_min_bin=0.0,
        distogram_max_bin=2.0,
        token_has_frame=jnp.asarray([True, True]),
        token_asym_id=jnp.asarray([0, 1]),
    )

    assert scores["summary_ptm"].shape == (1,)
    assert scores["summary_iptm"].shape == (1,)
    assert scores["summary_ranking_score"].shape == (1,)


def test_calculate_chain_based_plddt_matches_token_chain_masks() -> None:
    atom_plddt = jnp.asarray([[10.0, 20.0, 30.0]], dtype=jnp.float32)
    asym_id = jnp.asarray([0, 0, 1])
    atom_to_token_idx = jnp.asarray([0, 1, 2])

    out = calculate_chain_based_plddt(atom_plddt, asym_id, atom_to_token_idx)

    np.testing.assert_allclose(np.asarray(out["chain_plddt"]), [[15.0, 30.0]])
    np.testing.assert_allclose(
        np.asarray(out["chain_pair_plddt"]),
        [[[0.0, 20.0], [20.0, 0.0]]],
    )


def test_calculate_chain_pair_pae_returns_weighted_mean_and_min() -> None:
    token_pair_pae = jnp.asarray(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]],
        dtype=jnp.float32,
    )
    asym_id = jnp.asarray([0, 0, 1])
    token_has_frame = jnp.asarray([True, True, True])

    out = calculate_chain_pair_pae(token_pair_pae, asym_id, token_has_frame)

    np.testing.assert_allclose(
        np.asarray(out["chain_pair_pae_mean"]),
        [[[3.0, 4.5], [7.5, 9.0]]],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(out["chain_pair_pae_min"]),
        [[[1.0, 3.0], [7.0, 9.0]]],
        rtol=1e-6,
        atol=1e-6,
    )


def test_calculate_chain_based_gpde_returns_intra_and_interface_values() -> None:
    token_pair_pde = jnp.asarray(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]],
        dtype=jnp.float32,
    )
    contact_probs = jnp.ones((3, 3), dtype=jnp.float32)
    asym_id = jnp.asarray([0, 0, 1])

    out = calculate_chain_based_gpde(token_pair_pde, contact_probs, asym_id)

    np.testing.assert_allclose(np.asarray(out["chain_gpde"]), [[3.0, 9.0]])
    np.testing.assert_allclose(
        np.asarray(out["chain_pair_gpde"]),
        [[[0.0, 4.5], [4.5, 0.0]]],
        rtol=1e-6,
        atol=1e-6,
    )


def test_confidence_scores_adds_chain_metrics_when_chain_inputs_are_given() -> None:
    logits = jnp.zeros((1, 3, 3, 2), dtype=jnp.float32)
    scores = confidence_scores_from_logits(
        plddt_logits=jnp.zeros((1, 3, 2), dtype=jnp.float32),
        pae_logits=logits,
        pde_logits=logits,
        distogram_logits=jnp.zeros((3, 3, 2), dtype=jnp.float32),
        plddt_max_bin=2.0,
        pae_max_bin=2.0,
        pde_max_bin=2.0,
        distogram_min_bin=0.0,
        distogram_max_bin=2.0,
        token_has_frame=jnp.asarray([True, True, True]),
        token_asym_id=jnp.asarray([0, 0, 1]),
        atom_to_token_idx=jnp.asarray([0, 1, 2]),
    )

    assert scores["chain_plddt"].shape == (1, 2)
    assert scores["chain_pair_plddt"].shape == (1, 2, 2)
    assert scores["chain_pair_pae_mean"].shape == (1, 2, 2)
    assert scores["chain_pair_pae_min"].shape == (1, 2, 2)
    assert scores["chain_gpde"].shape == (1, 2)
    assert scores["chain_pair_gpde"].shape == (1, 2, 2)


def test_calculate_clash_flags_dense_inter_chain_contacts() -> None:
    coords = jnp.asarray(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.2],
                [0.0, 0.0, 0.4],
                [0.0, 0.0, 0.6],
            ],
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [30.0, 0.0, 0.0],
            ],
        ],
        dtype=jnp.float32,
    )
    asym_id = jnp.asarray([0, 0, 1, 1])
    atom_to_token_idx = jnp.asarray([0, 1, 2, 3])

    has_clash = calculate_clash(
        coords,
        asym_id,
        atom_to_token_idx,
        threshold=1.1,
    )

    np.testing.assert_array_equal(np.asarray(has_clash), np.asarray([True, False]))


def test_calculate_clash_excludes_ligand_chain_pairs() -> None:
    coords = jnp.asarray(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]]],
        dtype=jnp.float32,
    )
    asym_id = jnp.asarray([0, 1])
    atom_to_token_idx = jnp.asarray([0, 1])

    has_clash = calculate_clash(
        coords,
        asym_id,
        atom_to_token_idx,
        atom_is_polymer=jnp.asarray([True, False]),
        threshold=1.1,
    )

    np.testing.assert_array_equal(np.asarray(has_clash), np.asarray([False]))


def test_confidence_scores_adds_clash_penalized_ranking() -> None:
    logits = jnp.zeros((2, 2, 2, 2), dtype=jnp.float32)
    coords = jnp.asarray(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]],
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )

    scores = confidence_scores_from_logits(
        plddt_logits=jnp.zeros((2, 2, 2), dtype=jnp.float32),
        pae_logits=logits,
        pde_logits=logits,
        distogram_logits=jnp.zeros((2, 2, 2), dtype=jnp.float32),
        plddt_max_bin=2.0,
        pae_max_bin=2.0,
        pde_max_bin=2.0,
        distogram_min_bin=0.0,
        distogram_max_bin=2.0,
        token_has_frame=jnp.asarray([True, True]),
        token_asym_id=jnp.asarray([0, 1]),
        atom_to_token_idx=jnp.asarray([0, 1]),
        atom_coordinate=coords,
    )

    assert scores["has_clash"].shape == (2,)
    assert scores["summary_ranking_score"].shape == (2,)
    assert scores["summary_ranking_score"][0] < scores["summary_ranking_score"][1]


def test_calculate_vdw_clash_uses_element_radii() -> None:
    coords = jnp.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    asym_id = jnp.asarray([0, 1])
    atom_to_token_idx = jnp.asarray([0, 1])
    elements_one_hot = jnp.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)

    has_vdw = calculate_vdw_clash(
        coords,
        asym_id,
        atom_to_token_idx,
        elements_one_hot,
        threshold=0.75,
    )

    np.testing.assert_array_equal(np.asarray(has_vdw), np.asarray([True, False]))


def test_calculate_vdw_clash_skips_same_molecule_pairs() -> None:
    coords = jnp.asarray([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=jnp.float32)
    asym_id = jnp.asarray([0, 1])
    atom_to_token_idx = jnp.asarray([0, 1])
    elements_one_hot = jnp.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)

    has_vdw = calculate_vdw_clash(
        coords,
        asym_id,
        atom_to_token_idx,
        elements_one_hot,
        mol_id=jnp.asarray([5, 5]),
        threshold=0.75,
    )

    np.testing.assert_array_equal(np.asarray(has_vdw), np.asarray([False]))


def test_confidence_scores_adds_vdw_penalized_ranking_when_inputs_are_given() -> None:
    logits = jnp.zeros((2, 2, 2, 2), dtype=jnp.float32)
    coords = jnp.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )

    scores = confidence_scores_from_logits(
        plddt_logits=jnp.zeros((2, 2, 2), dtype=jnp.float32),
        pae_logits=logits,
        pde_logits=logits,
        distogram_logits=jnp.zeros((2, 2, 2), dtype=jnp.float32),
        plddt_max_bin=2.0,
        pae_max_bin=2.0,
        pde_max_bin=2.0,
        distogram_min_bin=0.0,
        distogram_max_bin=2.0,
        token_has_frame=jnp.asarray([True, True]),
        token_asym_id=jnp.asarray([0, 1]),
        atom_to_token_idx=jnp.asarray([0, 1]),
        atom_coordinate=coords,
        elements_one_hot=jnp.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=jnp.float32),
    )

    assert scores["has_vdw_clash"].shape == (2,)
    assert scores["summary_ranking_score_vdw_penalized"].shape == (2,)
    assert (
        scores["summary_ranking_score_vdw_penalized"][0]
        < scores["summary_ranking_score_vdw_penalized"][1]
    )


def test_confidence_head_stacks_sample_axis_like_protenix() -> None:
    params = _empty_confidence_params(c_s_inputs=3, c_s=2, c_z=2)
    features = {
        "distogram_rep_atom_mask": jnp.asarray([1, 1, 0]),
        "atom_to_token_idx": jnp.asarray([0, 1, 1]),
        "atom_to_tokatom_idx": jnp.asarray([0, 0, 1]),
    }
    coordinates = jnp.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [6.0, 8.0, 0.0]],
        ],
        dtype=jnp.float32,
    )

    output = confidence_head(
        features,
        s_inputs=jnp.zeros((2, 3), dtype=jnp.float32),
        s_trunk=jnp.zeros((2, 2), dtype=jnp.float32),
        z_trunk=jnp.zeros((2, 2, 2), dtype=jnp.float32),
        pair_mask=None,
        x_pred_coords=coordinates,
        params=params,
    )

    assert output["plddt"].shape == (2, 3, 2)
    assert output["pae"].shape == (2, 2, 2, 1)
    assert output["pde"].shape == (2, 2, 2, 1)
    assert output["resolved"].shape == (2, 3, 1)


def _empty_confidence_params(
    *,
    c_s_inputs: int,
    c_s: int,
    c_z: int,
) -> ConfidenceHeadParams:
    return ConfidenceHeadParams(
        input_strunk_ln=LayerNormParams(
            weight=jnp.ones((c_s,)),
            bias=jnp.zeros((c_s,)),
        ),
        linear_s1=LinearParams(weight=jnp.zeros((c_z, c_s_inputs)), bias=None),
        linear_s2=LinearParams(weight=jnp.zeros((c_z, c_s_inputs)), bias=None),
        distance_embedding=ConfidenceDistanceEmbeddingParams(
            lower_bins=jnp.asarray([0.0]),
            upper_bins=jnp.asarray([10.0]),
            linear_d=LinearParams(weight=jnp.zeros((c_z, 1)), bias=None),
            linear_d_wo_onehot=LinearParams(weight=jnp.zeros((c_z, 1)), bias=None),
        ),
        pairformer_stack=PairformerStackParams(blocks=()),
        output=ConfidenceOutputParams(
            pae_ln=LayerNormParams(weight=jnp.ones((c_z,)), bias=jnp.zeros((c_z,))),
            pde_ln=LayerNormParams(weight=jnp.ones((c_z,)), bias=jnp.zeros((c_z,))),
            plddt_ln=LayerNormParams(weight=jnp.ones((c_s,)), bias=jnp.zeros((c_s,))),
            resolved_ln=LayerNormParams(
                weight=jnp.ones((c_s,)),
                bias=jnp.zeros((c_s,)),
            ),
            linear_pae=LinearParams(weight=jnp.zeros((1, c_z)), bias=None),
            linear_pde=LinearParams(weight=jnp.zeros((1, c_z)), bias=None),
            plddt_weight=jnp.zeros((2, c_s, 2)),
            resolved_weight=jnp.zeros((2, c_s, 1)),
        ),
    )


def _clash_case(n_atom: int, n_chain: int, seed: int = 0):
    """Random coordinates in a box small enough that clashes actually occur.

    One atom of a *non-zero* chain is pinned to the origin. That is where padded
    rows land -- ``jnp.pad`` fills coordinates with zero -- so if the padding
    sentinel chain id were ever dropped, the pad rows would register as chain 0
    atoms sitting on top of it and invent an inter-chain clash. Without an atom
    there the test passes whether the sentinel works or not, which was true of the
    first version of it.
    """
    key = jax.random.split(jax.random.key(seed), 2)
    coords = jax.random.uniform(
        key[0], (3, n_atom, 3), dtype=jnp.float32, minval=2.0, maxval=8.0
    )
    coords = coords.at[:, 1, :].set(0.0)
    asym_id = jnp.asarray(np.arange(n_atom) % n_chain, dtype=jnp.int32)
    assert int(asym_id[1]) != 0, "the pinned atom must not be in chain 0"
    atom_to_token_idx = jnp.arange(n_atom, dtype=jnp.int32)
    return coords, asym_id, atom_to_token_idx


def test_padded_rows_cannot_invent_a_clash() -> None:
    """The padding sentinel is load-bearing, so its failure must be detectable.

    ``jnp.pad`` gives pad rows coordinate zero. If they also carried a real chain
    id they would clash with anything near the origin, and the count would depend on
    the chunk size. This asserts the fixture is arranged so that would show.
    """
    coords, asym_id, atom_to_token_idx = _clash_case(n_atom=17, n_chain=3)
    origin_atoms = np.flatnonzero(
        np.all(np.asarray(coords[0]) == 0.0, axis=-1)
    )
    assert origin_atoms.size >= 1, "no atom at the origin; padding damage would hide"
    assert int(asym_id[origin_atoms[0]]) != 0


@pytest.mark.parametrize("row_chunk_size", [1, 3, 7, 8, 16, 1000])
def test_calculate_clash_is_independent_of_the_row_chunk(row_chunk_size) -> None:
    """Chunking the atom rows must not change the flag.

    The unchunked form builds the whole ``[samples, n_atom, n_atom]`` distance
    matrix, which at 5 samples and the 16134 atoms of a 2030-token target is 5.2 TB;
    it died during autotuning on an ``f32[5, 16134, 16134]`` fusion. Counting
    partitions over rows, so a block at a time is the same sum -- but only if the
    block boundaries and the padding are handled right, which is what this checks.
    Sizes 7 and 3 do not divide 17.
    """
    coords, asym_id, atom_to_token_idx = _clash_case(n_atom=17, n_chain=3)

    reference = calculate_clash(
        coords, asym_id, atom_to_token_idx, threshold=2.0, row_chunk_size=17
    )
    actual = calculate_clash(
        coords, asym_id, atom_to_token_idx, threshold=2.0,
        row_chunk_size=row_chunk_size,
    )
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))


def test_the_clash_chunk_test_sees_both_outcomes() -> None:
    """Guard the parametrization above from comparing all-False against all-False."""
    coords, asym_id, atom_to_token_idx = _clash_case(n_atom=17, n_chain=3)
    dense = calculate_clash(
        coords, asym_id, atom_to_token_idx, threshold=2.0, row_chunk_size=4
    )
    sparse = calculate_clash(
        coords * 100.0, asym_id, atom_to_token_idx, threshold=2.0, row_chunk_size=4
    )
    assert bool(jnp.any(dense)), "no clash detected, so the comparison proves nothing"
    assert not bool(jnp.any(sparse)), "everything clashes even when spread out"


@pytest.mark.parametrize("row_chunk_size", [1, 5, 9, 16, 1000])
def test_calculate_vdw_clash_is_independent_of_the_row_chunk(row_chunk_size) -> None:
    """``any`` is associative over the row partition, so the flags must not move."""
    coords, asym_id, atom_to_token_idx = _clash_case(n_atom=17, n_chain=3, seed=1)
    elements = jnp.zeros((17, len(RDKIT_VDWS)), dtype=jnp.float32).at[:, 5].set(1.0)

    reference = calculate_vdw_clash(
        coords, asym_id, atom_to_token_idx, elements, threshold=0.75, row_chunk_size=17
    )
    actual = calculate_vdw_clash(
        coords, asym_id, atom_to_token_idx, elements, threshold=0.75,
        row_chunk_size=row_chunk_size,
    )
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))


def test_contract_bins_matches_broadcast_reduction():
    """The bin contraction must agree with the broadcast-multiply-then-sum form."""

    rng = np.random.default_rng(11)
    pae_prob = jnp.asarray(rng.random((3, 12, 12, 16), dtype=np.float32))
    per_bin_weight = jnp.asarray(rng.random(16, dtype=np.float32))

    reference = jnp.sum(pae_prob.astype(jnp.float32) * per_bin_weight, axis=-1)
    actual = _contract_bins(pae_prob, per_bin_weight)

    assert actual.shape == reference.shape
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(reference), rtol=1e-6, atol=1e-5
    )


def test_contract_bins_does_not_materialize_the_product():
    """Guard the memory fix, not just the numbers.

    ``sum(pae_prob * per_bin_weight, -1)`` is numerically fine and was the original
    code; the problem is that it builds a temporary the size of ``pae_prob``. This
    runs eagerly (this port jits primitives, not the graph), so nothing fuses it
    away, and ``calculate_chain_based_ptm`` reaches it once per chain and per chain
    pair -- at 2030 tokens that was ~12 x 4.91 GiB and it OOM'd a finished
    prediction. A revert to the broadcast form stays numerically correct and would
    pass the test above, so assert on the lowered graph instead.
    """

    pae_prob = jnp.zeros((2, 5, 5, 8), jnp.float32)
    per_bin_weight = jnp.zeros((8,), jnp.float32)
    jaxpr = jax.make_jaxpr(_contract_bins)(pae_prob, per_bin_weight)
    equations = jaxpr.jaxpr.eqns

    assert [eq.primitive.name for eq in equations] == ["dot_general"]
    # TF32 would silently drop ~10 bits: callers run with
    # jax_default_matmul_precision="default", so the precision must be explicit.
    assert equations[0].params["precision"] == (
        jax.lax.Precision.HIGHEST,
        jax.lax.Precision.HIGHEST,
    )
