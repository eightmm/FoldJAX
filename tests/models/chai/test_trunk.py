from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import foldjax.models.chai.models.trunk as trunk_lib

pytestmark = pytest.mark.official_parity


@pytest.fixture(scope="module")
def official_key_state(chai_trunk_module):
    return {
        key: np.zeros((), dtype=np.float32)
        for key in chai_trunk_module.state_dict()
    }


def test_map_trunk_consumes_all_official_tensors_once(
    official_key_state,
) -> None:
    assert len(official_key_state) == 1398
    params = trunk_lib.map_trunk(official_key_state)
    assert len(jax.tree.leaves(params)) == 1398
    assert len(params.pairformer_blocks) == 48
    assert len(params.template.blocks) == 2
    assert len(params.msa.blocks) == 4


def test_map_trunk_strict_rejects_unconsumed_tensor(
    official_key_state,
) -> None:
    state = dict(official_key_state)
    state["unexpected.weight"] = np.zeros((), dtype=np.float32)
    with pytest.raises(ValueError, match="unconsumed=.*unexpected.weight"):
        trunk_lib.map_trunk(state)


def _sentinel_params() -> trunk_lib.TrunkParams:
    return trunk_lib.TrunkParams(
        single_recycling="single_recycling",
        pair_recycling="pair_recycling",
        template="template",
        msa="msa",
        pairformer_blocks=tuple(range(48)),
    )


def _inputs():
    return dict(
        token_single_trunk_initial_repr=jnp.ones((1, 2, 1)),
        token_pair_trunk_initial_repr=jnp.full((1, 2, 2, 1), 3.0),
        token_single_trunk_repr=jnp.full((1, 2, 1), 2.0),
        token_pair_trunk_repr=jnp.full((1, 2, 2, 1), 4.0),
        msa_input_feats=jnp.zeros((1, 2, 2, 1)),
        msa_mask=jnp.asarray([[[True, True], [True, False]]]),
        template_input_feats=jnp.zeros((1, 4, 2, 2, 1)),
        template_input_masks=jnp.ones((1, 4, 2, 2), dtype=bool),
        token_single_mask=jnp.asarray([[True, False]]),
        token_pair_mask=jnp.asarray([[[True, False], [False, False]]]),
    )


def test_forward_matches_exact_leaf_orchestration(monkeypatch) -> None:
    events: list[object] = []

    def fake_recycling(value, params):
        events.append(params)
        multiplier = 10 if params == "single_recycling" else 100
        return value * multiplier

    def fake_template(pair, features, masks, pair_mask, params):
        del features, masks, pair_mask
        events.append(params)
        return pair + 5

    def fake_msa(
        features,
        mask,
        single,
        pair,
        pair_mask,
        params,
        **kwargs,
    ):
        del features, mask, pair_mask, kwargs
        events.append((params, float(single[0, 0, 0])))
        return pair + 7

    def fake_pairformer(single, pair, token_mask, pair_mask, params):
        del token_mask, pair_mask
        events.append(params)
        increment = params + 1
        return single + increment, pair + 2 * increment

    monkeypatch.setattr(trunk_lib, "recycling_projection", fake_recycling)
    monkeypatch.setattr(trunk_lib, "template_embedding", fake_template)
    monkeypatch.setattr(trunk_lib, "msa_module", fake_msa)
    monkeypatch.setattr(trunk_lib, "pairformer_block", fake_pairformer)

    single, pair = trunk_lib.trunk_forward(
        **_inputs(), params=_sentinel_params(), use_scan=False
    )

    assert events[:4] == [
        "pair_recycling",
        "single_recycling",
        "template",
        ("msa", 21.0),
    ]
    assert events[4:] == list(range(48))
    np.testing.assert_array_equal(np.asarray(single), np.full(single.shape, 1197))
    # The exported trunk adds the pre-MSA pair state to the complete MSA module
    # output before entering the 48-block Pairformer stack.
    np.testing.assert_array_equal(np.asarray(pair), np.full(pair.shape, 3175))


def test_forward_is_sensitive_only_to_unmasked_feature_branches(
    monkeypatch,
) -> None:
    def fake_recycling(value, params):
        del params
        return value

    def fake_template(pair, features, masks, pair_mask, params):
        del pair_mask, params
        return pair + jnp.sum(jnp.where(masks[..., None], features, 0))

    def fake_msa(
        features,
        mask,
        single,
        pair,
        pair_mask,
        params,
        **kwargs,
    ):
        del single, pair_mask, params, kwargs
        return pair + jnp.sum(jnp.where(mask[..., None], features, 0))

    def fake_pairformer(single, pair, token_mask, pair_mask, params):
        del token_mask, pair_mask, params
        return single, pair

    monkeypatch.setattr(trunk_lib, "recycling_projection", fake_recycling)
    monkeypatch.setattr(trunk_lib, "template_embedding", fake_template)
    monkeypatch.setattr(trunk_lib, "msa_module", fake_msa)
    monkeypatch.setattr(trunk_lib, "pairformer_block", fake_pairformer)
    inputs = _inputs()

    baseline = trunk_lib.trunk_forward(
        **inputs, params=_sentinel_params(), use_scan=False
    )[1]
    active_template = dict(inputs)
    active_template["template_input_feats"] = inputs[
        "template_input_feats"
    ].at[0, 0, 0, 0, 0].set(2)
    masked_template = dict(inputs)
    masked_template["template_input_masks"] = inputs[
        "template_input_masks"
    ].at[0, 0, 0, 0].set(False)
    masked_template["template_input_feats"] = inputs[
        "template_input_feats"
    ].at[0, 0, 0, 0, 0].set(2)
    active_msa = dict(inputs)
    active_msa["msa_input_feats"] = inputs["msa_input_feats"].at[
        0, 0, 0, 0
    ].set(3)
    masked_msa = dict(inputs)
    masked_msa["msa_input_feats"] = inputs["msa_input_feats"].at[
        0, 1, 1, 0
    ].set(3)

    active_template_out = trunk_lib.trunk_forward(
        **active_template, params=_sentinel_params(), use_scan=False
    )[1]
    masked_template_out = trunk_lib.trunk_forward(
        **masked_template, params=_sentinel_params(), use_scan=False
    )[1]
    active_msa_out = trunk_lib.trunk_forward(
        **active_msa, params=_sentinel_params(), use_scan=False
    )[1]
    masked_msa_out = trunk_lib.trunk_forward(
        **masked_msa, params=_sentinel_params(), use_scan=False
    )[1]

    assert not np.array_equal(np.asarray(active_template_out), np.asarray(baseline))
    np.testing.assert_array_equal(np.asarray(masked_template_out), np.asarray(baseline))
    assert not np.array_equal(np.asarray(active_msa_out), np.asarray(baseline))
    np.testing.assert_array_equal(np.asarray(masked_msa_out), np.asarray(baseline))


def test_forward_requires_exact_48_block_stack(monkeypatch) -> None:
    params = _sentinel_params()._replace(pairformer_blocks=tuple(range(47)))
    monkeypatch.setattr(trunk_lib, "recycling_projection", lambda value, params: value)
    monkeypatch.setattr(
        trunk_lib,
        "template_embedding",
        lambda pair, features, masks, pair_mask, params: pair,
    )
    monkeypatch.setattr(
        trunk_lib,
        "msa_module",
        lambda features, mask, single, pair, pair_mask, params, **kwargs: pair,
    )
    with pytest.raises(ValueError, match="exactly 48"):
        trunk_lib.trunk_forward(**_inputs(), params=params, use_scan=False)


def test_scan_pairformer_stack_matches_python_loop(monkeypatch) -> None:
    def fake_pairformer(single, pair, token_mask, pair_mask, params):
        del token_mask, pair_mask
        return single + params, pair + 2 * params

    monkeypatch.setattr(trunk_lib, "pairformer_block", fake_pairformer)
    single = jnp.ones((1, 2, 1))
    pair = jnp.ones((1, 2, 2, 1))
    token_mask = jnp.ones((1, 2), dtype=bool)
    pair_mask = jnp.ones((1, 2, 2), dtype=bool)
    blocks = tuple(jnp.asarray(index, dtype=jnp.float32) for index in range(48))

    expected = trunk_lib._pairformer_stack(
        single, pair, token_mask, pair_mask, blocks, use_scan=False
    )
    actual = trunk_lib._pairformer_stack(
        single, pair, token_mask, pair_mask, blocks, use_scan=True
    )

    np.testing.assert_array_equal(np.asarray(actual[0]), np.asarray(expected[0]))
    np.testing.assert_array_equal(np.asarray(actual[1]), np.asarray(expected[1]))
