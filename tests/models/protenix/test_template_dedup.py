"""Deduplicating a template axis that carries the same template more than once.

A query with fewer than ``MAX_TEMPLATES`` hits is padded up to four, and
``template_embedder`` runs the whole pairformer stack once per row. The padded
rows are identical to each other but *not* to the gap row, so the interesting
case is two distinct templates out of four -- not one, and not four.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.protenix.bridge.torch_mapping import (
    map_template_embedder_state_dict,
)
from foldjax.models.protenix.data.template_features import (
    MAX_TEMPLATES,
    TEMPLATE_FIELDS,
    dedup_templates,
)
from foldjax.models.protenix.models.trunk_blocks.template import template_embedder
from tests.models.protenix.test_template import _template_state

N_TOKEN, BINS, N_ATOM = 3, 39, 24


def _templates(n_template: int = MAX_TEMPLATES) -> dict[str, np.ndarray]:
    return {
        "template_aatype": np.zeros((n_template, N_TOKEN), np.int32),
        "template_atom_positions": np.zeros(
            (n_template, N_TOKEN, N_ATOM, 3), np.float32
        ),
        "template_atom_mask": np.zeros((n_template, N_TOKEN, N_ATOM), bool),
        "template_pseudo_beta_mask": np.zeros(
            (n_template, N_TOKEN, N_TOKEN), np.float32
        ),
        "template_distogram": np.zeros(
            (n_template, N_TOKEN, N_TOKEN, BINS), np.float32
        ),
        "template_unit_vector": np.zeros(
            (n_template, N_TOKEN, N_TOKEN, 3), np.float32
        ),
        "template_backbone_frame_mask": np.zeros(
            (n_template, N_TOKEN, N_TOKEN), np.float32
        ),
    }


def _as_featurized() -> dict[str, np.ndarray]:
    """What the featurizer really produces for a query with no template hits.

    Measured, not assumed: row 0 carries the all-gap restype 31 and rows 1-3 are
    zero-padded, because ``_pad_templates`` zero-pads rather than gap-pads.
    """
    features = _templates()
    features["template_aatype"][0] = 31
    return features


def _rows(features: dict[str, np.ndarray]) -> int:
    return np.asarray(features["template_distogram"]).shape[0]


def test_a_template_free_query_has_two_distinct_rows_not_one() -> None:
    """The case the optimisation exists for, and the one a collapse gets wrong."""
    deduped = dedup_templates(_as_featurized())

    assert _rows(deduped) == 2
    np.testing.assert_array_equal(deduped["template_multiplicity"], [1.0, 3.0])
    # The survivors keep first-occurrence order, so the gap row stays first.
    assert deduped["template_aatype"][0][0] == 31
    assert deduped["template_aatype"][1][0] == 0


def test_all_identical_rows_reduce_to_one() -> None:
    """The collapse is just the case where the distinct set has size one."""
    deduped = dedup_templates(_templates())

    assert _rows(deduped) == 1
    np.testing.assert_array_equal(deduped["template_multiplicity"], [4.0])
    for name in TEMPLATE_FIELDS:
        assert np.asarray(deduped[name]).shape[0] == 1, name


def test_four_distinct_templates_are_left_alone() -> None:
    features = _templates()
    for index in range(MAX_TEMPLATES):
        features["template_distogram"][index, 0, 0, index] = 1.0

    deduped = dedup_templates(features)

    assert _rows(deduped) == MAX_TEMPLATES
    assert "template_multiplicity" not in deduped
    for name, array in features.items():
        np.testing.assert_array_equal(deduped[name], array)


@pytest.mark.parametrize("name", sorted(TEMPLATE_FIELDS))
def test_any_single_field_can_make_a_row_distinct(name: str) -> None:
    """Every field is part of the identity, not just the distogram."""
    features = _templates()
    array = features[name]
    array[2] = True if array.dtype == bool else 1

    deduped = dedup_templates(features)

    assert _rows(deduped) == 2
    np.testing.assert_array_equal(deduped["template_multiplicity"], [3.0, 1.0])


def test_a_single_template_is_returned_unchanged() -> None:
    deduped = dedup_templates(_templates(n_template=1))
    assert _rows(deduped) == 1
    assert "template_multiplicity" not in deduped


def test_features_without_templates_pass_through() -> None:
    assert list(dedup_templates({"restype": np.zeros(3)})) == ["restype"]


def test_non_template_features_are_preserved() -> None:
    features = _as_featurized()
    features["restype"] = np.zeros((N_TOKEN, 32), np.float32)

    assert dedup_templates(features)["restype"].shape == (N_TOKEN, 32)


def test_disagreeing_template_row_counts_are_rejected() -> None:
    features = _templates()
    features["template_aatype"] = features["template_aatype"][:2]
    with pytest.raises(ValueError, match="row count"):
        dedup_templates(features)


def test_padding_still_requires_the_native_depth() -> None:
    """Why the call site runs the dedup *after* padding, not before."""
    from foldjax.models.protenix.data.padding import _pad_templates

    deduped = dedup_templates(_as_featurized())
    with pytest.raises(ValueError, match="native template depth 4"):
        _pad_templates({}, deduped, N_TOKEN, N_TOKEN + 5)


def test_multiplicity_survives_the_padded_path_feature_filter() -> None:
    """`model.py` drops unlisted features silently, which would divide by 2.

    On the padded path the model keeps only `_PADDED_MODEL_FEATURES`. If
    `template_multiplicity` were missing from that allow-list it would be
    discarded with no error, `template_embedder` would fall back to
    `num_templates = shape[0]`, and a query whose four rows deduplicated to two
    would be divided by two instead of four -- a wrong value, on the padded path
    only, with nothing in the logs.
    """
    from foldjax.models.protenix.models.model import _PADDED_MODEL_FEATURES

    assert "template_multiplicity" in _PADDED_MODEL_FEATURES


def test_the_model_computes_the_same_update_from_the_deduplicated_axis() -> None:
    """Four rows summed, against two rows weighted by what they stand for.

    DO NOT TIGHTEN THIS INTO A BIT-EQUALITY CHECK. The OpenFold3 equivalent
    (``4ad6f38``) is gated on bit-identity and its outputs md5-match; copying
    that gate here would be wrong for two independent reasons, and it would
    probably still pass on some inputs, which is the trap.

    First, ``template_embedder`` divides by ``1e-7 + n`` and the epsilon does
    not scale with ``n``, so ``4x / (4 + 1e-7)`` is not ``x / (1 + 1e-7)``.

    Second -- and NOT for the reason it first appears. ``m * x`` against adding
    ``x`` to itself ``m`` times is bit-identical for the multiplicities that
    occur here: ``2x`` and ``4x`` are exact, and ``(x + x) + x`` is a single
    rounding of the same exact value as ``3 * x``. Measured over 300,000 random
    float32 values, zero disagree. What actually diverges is the *association*
    of the running sum once there is more than one distinct row: the loop
    computes ``((f0 + f1) + f1) + f1`` and the deduplicated form computes
    ``f0 + 3 * f1``, which are different orderings of the same exact sum. Those
    disagree on about 45% of random pairs. See the test below for both facts.
    Gate against the run-to-run floor.
    """
    params = map_template_embedder_state_dict(_template_state(np.random.default_rng(6)))
    rng = np.random.default_rng(7)
    z = jnp.asarray(rng.normal(size=(N_TOKEN, N_TOKEN, 4)).astype(np.float32))
    base = {"asym_id": jnp.asarray([1, 1, 1])}

    features = _as_featurized()
    deduped = dedup_templates(features)
    assert _rows(deduped) == 2

    def run(source: dict[str, np.ndarray]) -> np.ndarray:
        batch = {**base, **{k: jnp.asarray(v) for k, v in source.items()}}
        return np.asarray(template_embedder(batch, z, None, params))

    np.testing.assert_allclose(run(features), run(deduped), rtol=1e-6, atol=1e-6)


def test_what_actually_diverges_is_the_association_not_the_multiply() -> None:
    """Pins both halves, because the obvious explanation is the wrong one.

    "``3 * f`` is one rounding where ``f + f + f`` is two" sounds right and is
    false: ``f + f`` is exact, so ``(f + f) + f`` rounds the same exact value
    that ``3 * f`` does. Anyone tightening the gate above will reach for that
    argument, find it does not hold, and conclude the tolerance is unnecessary.
    It is necessary for a different reason -- the running sum is re-associated.
    """
    rng = np.random.default_rng(0)
    f0 = rng.normal(size=20000).astype(np.float32)
    f1 = rng.normal(size=20000).astype(np.float32)

    # The multiply itself is exact at every multiplicity that occurs here.
    for m in (2.0, 3.0, 4.0):
        repeated = f1.copy()
        for _ in range(int(m) - 1):
            repeated = (repeated + f1).astype(np.float32)
        np.testing.assert_array_equal(repeated, (np.float32(m) * f1).astype(np.float32))

    # The accumulation order is what changes, once a second row is in the sum.
    loop = (((f0 + f1).astype(np.float32) + f1).astype(np.float32) + f1).astype(
        np.float32
    )
    deduped = (f0 + (np.float32(3.0) * f1).astype(np.float32)).astype(np.float32)
    assert (loop != deduped).mean() > 0.1
