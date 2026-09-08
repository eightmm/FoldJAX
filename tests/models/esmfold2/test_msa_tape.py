from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2.models import model as m
from tests.models.esmfold2.test_lm_dropout_tape import _loop_inputs


def _settings():
    return replace(
        m.ModelSettings(), msa_n_layers=0, max_msa_depth=2, msa_column_mask_rate=0.2
    )


@pytest.mark.parametrize(
    "bad", ["dtype", "shape", "repeat", "range", "query", "inactive", "column_dtype"]
)
def test_msa_tape_validation(bad):
    rows = np.array([[0, 2], [0, 3]], np.int64)
    column = np.ones((1, 2), bool)
    settings = _settings()
    if bad == "dtype":
        rows = rows.astype(np.float32)
    if bad == "shape":
        rows = rows[:1]
    if bad == "repeat":
        rows[0] = [0, 0]
    if bad == "range":
        rows[0] = [0, 4]
    if bad == "query":
        rows[0] = [1, 2]
    if bad == "inactive":
        settings = replace(settings, msa_n_layers=None)
    if bad == "column_dtype":
        column = column.astype(np.float32)
    with pytest.raises(ValueError, match="MSA tape"):
        m.validate_msa_tape(
            column, rows, batch=1, tokens=2, depth=4, loops=2, settings=settings
        )


def test_native_empty_tapes_only_allowed_for_inactive_draws():
    assert m.validate_msa_tape(
        np.empty(0, bool),
        np.empty(0, np.int64),
        batch=1,
        tokens=2,
        depth=None,
        loops=2,
        settings=_settings(),
    ) == (None, None)
    with pytest.raises(ValueError):
        m.validate_msa_tape(
            np.empty(0, bool),
            None,
            batch=1,
            tokens=2,
            depth=4,
            loops=2,
            settings=_settings(),
        )


@pytest.mark.parametrize("compiled", [False, True])
def test_rows_and_lm_tapes_drive_each_loop_all_msa_features(monkeypatch, compiled):
    z, params, settings = _loop_inputs()
    settings = replace(settings, msa_n_layers=0, max_msa_depth=2)
    values = jnp.arange(8, dtype=jnp.float32).reshape(1, 2, 4)
    inputs = dict(
        msa_one_hot=values[..., None],
        msa_mask=values + 1,
        has_deletion=values + 2,
        deletion_value=values + 3,
        x_inputs=z,
    )

    def encoder(pair, single, one_hot, has_deletion, deletion, mask, *args, **kwargs):
        combined = one_hot[..., 0] + 3 * has_deletion + 7 * deletion + 11 * mask
        return pair + combined.sum(axis=2)[:, :, None, None]

    monkeypatch.setattr(m, "msa_encoder", encoder)
    rows = jnp.array([[0, 1], [0, 3]], jnp.int32)
    masks = jnp.ones((2, *z.shape), bool)

    def run(key, indices):
        return m.run_loops(
            key,
            z,
            z,
            z,
            inputs,
            jnp.ones(z.shape[:-1]),
            params,
            settings=settings,
            total_steps=2,
            lm_dropout_masks=masks,
            msa_row_choices=indices,
        )

    fn = jax.jit(run) if compiled else run
    first = fn(jax.random.key(0), rows)
    np.testing.assert_array_equal(first, fn(jax.random.key(99), rows))
    assert not np.array_equal(first, fn(jax.random.key(0), rows[::-1]))


def test_column_tape_preserves_native_query_override():
    column = jnp.array([[False, True]])
    keep = m._msa_column_keep(
        jax.random.key(0),
        jnp.ones((1, 2)),
        0.2,
        preserve_prefix_rng=False,
        keep_tape=column,
    )
    np.testing.assert_array_equal(keep, column)
    native_mask = jnp.broadcast_to(keep[:, None, :], (1, 4, 2)).at[:, 0, :].set(True)
    np.testing.assert_array_equal(native_mask[:, 0], True)
    np.testing.assert_array_equal(native_mask[:, 1], column)
