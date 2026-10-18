"""Host feature transfers must not compile a JAX identity for each shape."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.opendde import postprocess
from foldjax.models.opendde.models import shape_complementarity as shape_comp


def _refuse_staging(monkeypatch):
    from jax._src import dispatch

    original = dispatch.apply_primitive

    def apply(primitive, *args, **kwargs):
        if primitive.name == "stage":
            pytest.fail("a concrete host transfer compiled a stage primitive")
        return original(primitive, *args, **kwargs)

    monkeypatch.setattr(dispatch, "apply_primitive", apply)


@pytest.mark.parametrize("n_token", [8, 13])
@pytest.mark.parametrize("x64", [False, True])
def test_shape_features_preserve_values_and_lowering_without_staging(
    monkeypatch, n_token, x64
):
    # Views exercise the transfer's copy/contiguity handling; the wider integer
    # leaves exercise the same x64 canonicalization as the former asarray.
    resolved = shape_comp.ShapeCompTokenFeatures(
        atom_to_token_idx=np.repeat(np.arange(n_token, dtype=np.int64), 4)[::2],
        rep_atom_indices=np.arange(2 * n_token, dtype=np.int64)[::2],
        rep_atom_valid=np.ones(2 * n_token, dtype=bool)[::2],
        token_asym_id=np.repeat(np.arange(n_token, dtype=np.int64) % 2, 2)[::2],
        token_role_id=np.full(2 * n_token, -1, dtype=np.int64)[::2],
        is_structural=False,
        is_protein_token=np.ones(2 * n_token, dtype=bool)[::2],
    )
    device = jax.devices("cpu")[0]
    pool = jax.jit(lambda graph: graph)
    with jax.enable_x64(x64), jax.default_device(device):
        expected = shape_comp.ShapeCompGraphFeatures(
            *(
                jnp.asarray(getattr(resolved, name))
                for name in shape_comp.ShapeCompGraphFeatures._fields
            )
        )
        old_hlo = pool.lower(expected).as_text()
        _refuse_staging(monkeypatch)
        actual = shape_comp._graph_features(resolved)
        for old, new in zip(expected, actual, strict=True):
            assert old.shape == new.shape
            assert old.dtype == new.dtype
            assert old.weak_type == new.weak_type
            assert new.devices() == old.devices() == {device}
            np.testing.assert_array_equal(new, old)
        assert pool.lower(actual).as_text() == old_hlo
    pool.clear_cache()


@pytest.mark.parametrize("on_device", [False, True])
def test_postprocess_reads_token_count_without_staging_the_chain_array(
    monkeypatch, on_device
):
    asym_id = np.asarray([0, 0, 1, 1], dtype=np.int64)
    if on_device:
        asym_id = jax.device_put(asym_id)
    output = {"summary_ranking_score": jax.device_put(np.asarray([0.5], np.float32))}
    seen = []

    def shape_scores(output, features, *, n_token, deterministic):
        seen.append(n_token)
        return {}

    monkeypatch.setattr(postprocess, "_shape_complementarity_scores", shape_scores)
    _refuse_staging(monkeypatch)
    actual = postprocess.opendde_confidence_scores(
        output, {"asym_id": asym_id}, num_recycles=1
    )
    np.testing.assert_array_equal(actual["summary_ranking_score"], [0.5])
    assert seen == [4]
