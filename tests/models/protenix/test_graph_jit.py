"""Tracing the whole Protenix graph instead of dispatching it op by op.

``protenix_infer_static`` is written eagerly and calls individually jitted
primitives, so one 35-residue prediction dispatched 666 separate executables.
``protenix_infer_compiled`` traces it once. Three things have to hold for that
to be safe, and none of them needs the 1.4 GB checkpoint to check:

* the parameter tree survives the split into traced arrays and static flags,
* features the graph cannot consume are dropped rather than coerced, and
* the chain count reaches the confidence scores as a Python integer, because
  it sizes their per-chain loops and reading it off ``asym_id`` inside ``jit``
  raises.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models._graph import (
    is_traceable_feature,
    merge_static_flags,
    split_static_flags,
)
from foldjax.models.protenix.models import model as model_impl
from foldjax.models.protenix.models.heads.confidence import (
    ConfidenceDistanceEmbeddingParams,
)
from foldjax.models.protenix.models.model import (
    GRAPH_STATIC_ARGNAMES,
    protenix_infer_static,
)
from foldjax.models.protenix.models.primitives.primitives import LinearParams


class _Inner(NamedTuple):
    kernel: jnp.ndarray
    has_s: bool


class _Params(NamedTuple):
    weight: jnp.ndarray
    inner: _Inner
    enabled: bool


def _params() -> _Params:
    return _Params(
        weight=jnp.ones((2, 3)),
        inner=_Inner(kernel=jnp.zeros((3, 3)), has_s=True),
        enabled=False,
    )


def test_boolean_configuration_leaves_travel_as_static_arguments():
    arrays, _, flags = split_static_flags(_params())
    assert flags
    assert all(isinstance(value, bool) for _, value in flags)
    assert all(jnp.asarray(array).ndim > 0 for array in arrays)


def test_the_parameter_tree_round_trips():
    original = _params()
    restored = merge_static_flags(*split_static_flags(original))
    assert restored.inner.has_s is True
    assert restored.enabled is False
    np.testing.assert_array_equal(
        np.asarray(restored.weight), np.asarray(original.weight)
    )


def test_the_flags_stay_python_bools_inside_a_trace():
    """The whole reason for the split: a Python ``if`` on a tracer raises."""
    arrays, treedef, flags = split_static_flags(_params())

    @jax.jit
    def run(param_arrays):
        params = merge_static_flags(param_arrays, treedef, flags)
        # Would raise TracerBoolConversionError if `has_s` were traced.
        scale = 2.0 if params.inner.has_s else 0.5
        return params.weight.sum() * scale

    assert float(run(arrays)) == pytest.approx(12.0)


def test_writer_only_features_are_dropped_before_tracing():
    assert is_traceable_feature(np.zeros(3, np.float32))
    assert is_traceable_feature(np.zeros(3, np.int64))
    assert is_traceable_feature(np.zeros(3, bool))
    assert is_traceable_feature({"nested": np.zeros(3)})
    # The featurizer emits atom names for the CIF writer; a string dtype has no
    # traceable representation.
    assert not is_traceable_feature(np.array(["CA", "CB"]))
    assert not is_traceable_feature(object())


def test_the_static_argnames_are_real_parameters():
    """A typo here would silently trace something that must stay static."""
    import inspect

    signature = inspect.signature(protenix_infer_static)
    unknown = sorted(set(GRAPH_STATIC_ARGNAMES) - set(signature.parameters))
    assert not unknown, f"not parameters of protenix_infer_static: {unknown}"


def test_the_chain_count_is_static():
    assert "n_chain" in GRAPH_STATIC_ARGNAMES
    assert "preserve_prefix_rng" in GRAPH_STATIC_ARGNAMES


class _ConfidenceOnly(NamedTuple):
    distance_embedding: ConfidenceDistanceEmbeddingParams


class _ModelOnly(NamedTuple):
    confidence: _ConfidenceOnly


def _confidence_only_params(*, contiguous: bool = True) -> _ModelOnly:
    lower = jnp.asarray([0.0, 1.0, 2.0], dtype=jnp.float32)
    upper = jnp.asarray(
        [1.0, 2.0 if contiguous else 2.5, 3.0], dtype=jnp.float32
    )
    distance = ConfidenceDistanceEmbeddingParams(
        lower_bins=lower,
        upper_bins=upper,
        linear_d=LinearParams(weight=jnp.ones((4, 3)), bias=jnp.zeros((4,))),
        linear_d_wo_onehot=LinearParams(weight=jnp.ones((4, 1)), bias=None),
    )
    return _ModelOnly(confidence=_ConfidenceOnly(distance_embedding=distance))


def test_compiled_wrapper_host_gates_a_static_compact_distance_flag(
    monkeypatch,
) -> None:
    captured: list[bool] = []

    def fake_compiled(*_args, **kwargs):
        captured.append(kwargs["compact_confidence_distance_bins"])
        return {}

    monkeypatch.setattr(model_impl, "_compiled_protenix_infer", fake_compiled)
    features = {"asym_id": jnp.asarray([0], dtype=jnp.int32)}
    noise_schedule = jnp.asarray([1.0, 0.0], dtype=jnp.float32)

    model_impl.protenix_infer_compiled(
        features, _confidence_only_params(), noise_schedule
    )
    model_impl.protenix_infer_compiled(
        features, _confidence_only_params(contiguous=False), noise_schedule
    )
    model_impl.protenix_infer_compiled(
        features,
        _confidence_only_params(),
        noise_schedule,
        compact_confidence_distance_bins=False,
    )

    assert "compact_confidence_distance_bins" in GRAPH_STATIC_ARGNAMES
    assert captured == [True, False, False]


def test_guidance_stays_on_the_eager_path():
    """TFG configuration is a plain mapping, so it cannot be a static argument.

    ``protenix_predict_static`` has to notice that and fall back rather than
    raise an unhashable-argument error deep inside ``jit``.
    """
    import inspect

    from foldjax.models.protenix.models import predict as predict_module

    source = inspect.getsource(predict_module.protenix_predict_static)
    assert "guidance_config" in source
    assert "protenix_infer_static" in source
