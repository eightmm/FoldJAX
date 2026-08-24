"""Tracing the whole OpenDDE graph instead of dispatching it op by op.

`opendde_infer_static` is written eagerly and calls individually jitted
primitives, so one prediction dispatched over a thousand small programs.
`opendde_infer_compiled` traces it once. Three things have to hold for that to
be safe, and none of them needs the 2.6 GB checkpoint to check.
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
from foldjax.models.opendde.models import model as model_impl
from foldjax.models.opendde.models.model import (
    GRAPH_STATIC_ARGNAMES,
    opendde_infer_static,
)
from foldjax.models.protenix.models.heads.confidence import (
    ConfidenceDistanceEmbeddingParams,
)
from foldjax.models.protenix.models.primitives.primitives import LinearParams

# The graph helpers moved to `foldjax.models._graph` when Protenix needed the
# same three answers; these names keep the test reading as it was written.
_is_model_input = is_traceable_feature
_merge_params = merge_static_flags
_split_params = split_static_flags


class _Inner(NamedTuple):
    kernel: jnp.ndarray
    has_s: bool
    cross_attention_mode: bool


class _Params(NamedTuple):
    weight: jnp.ndarray
    inner: _Inner


def _params() -> _Params:
    return _Params(
        weight=jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
        inner=_Inner(
            kernel=jnp.ones((2, 2), jnp.float32),
            has_s=jnp.asarray(True),
            cross_attention_mode=jnp.asarray(False),
        ),
    )


def test_boolean_config_flags_are_separated_from_the_weights() -> None:
    """Released weights carry scalar bool flags the modules branch on.

    Left in the traced pytree they become tracers and every `if flag:` raises,
    so they travel as static arguments instead.
    """
    arrays, treedef, flags = _split_params(_params())

    assert [bool(value) for _, value in flags] == [True, False]
    assert all(jnp.asarray(item).dtype != jnp.bool_ for item in arrays)
    assert len(arrays) == 2


def test_the_parameter_tree_round_trips_exactly() -> None:
    original = _params()
    restored = _merge_params(*_split_params(original))

    assert isinstance(restored, _Params)
    assert isinstance(restored.inner, _Inner)
    np.testing.assert_array_equal(restored.weight, original.weight)
    np.testing.assert_array_equal(restored.inner.kernel, original.inner.kernel)
    # Flags come back as Python bools, which is what the `if` branches want.
    assert restored.inner.has_s is True
    assert restored.inner.cross_attention_mode is False


def test_restored_flags_drive_python_branches_under_jit() -> None:
    """The point of the split: a flag must still be usable in a Python `if`."""
    arrays, treedef, flags = _split_params(_params())

    @jax.jit
    def run(param_arrays):
        params = _merge_params(param_arrays, treedef, flags)
        # Would raise TracerBoolConversionError if the flag were traced.
        return params.weight * (2.0 if params.inner.has_s else 0.0)

    np.testing.assert_array_equal(run(arrays), np.arange(6).reshape(2, 3) * 2.0)


def test_string_metadata_is_kept_out_of_the_traced_features() -> None:
    """The featurizer emits `output_atom_*` string arrays for the CIF writer.

    A string dtype cannot be traced, so they are dropped rather than coerced.
    """
    assert _is_model_input(np.zeros(3, np.float32))
    assert _is_model_input(np.zeros(3, np.int64))
    assert _is_model_input(np.zeros(3, bool))
    assert _is_model_input({"pad_info_entry": np.zeros(2)})
    assert not _is_model_input(np.asarray(["CA", "CB"]))
    assert not _is_model_input("a plain string")


def test_every_static_argname_is_a_real_keyword() -> None:
    """A typo here would silently trace a value that must stay static."""
    import inspect

    accepted = set(inspect.signature(opendde_infer_static).parameters)
    unknown = sorted(set(GRAPH_STATIC_ARGNAMES) - accepted)
    assert unknown == []


def test_no_array_shaped_argument_is_marked_static() -> None:
    """Marking an array static would embed it in the graph and blow up compile."""
    import inspect

    signature = inspect.signature(opendde_infer_static)
    array_like = {
        name
        for name, parameter in signature.parameters.items()
        if "ndarray" in str(parameter.annotation)
        or "Sequence[jnp" in str(parameter.annotation)
    }
    assert not (array_like & set(GRAPH_STATIC_ARGNAMES)), array_like & set(
        GRAPH_STATIC_ARGNAMES
    )


def test_value_checks_can_be_skipped_for_tracing() -> None:
    """The index-range checks read values, so the compiled path hoists them out."""
    import inspect

    signature = inspect.signature(opendde_infer_static)
    assert "validate_feature_values" in signature.parameters
    assert signature.parameters["validate_feature_values"].default is True
    assert "validate_feature_values" in GRAPH_STATIC_ARGNAMES


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


def test_compiled_wrapper_reuses_the_host_gated_static_compact_flag(
    monkeypatch,
) -> None:
    captured: list[bool] = []

    monkeypatch.setattr(
        model_impl,
        "_validate_static_structural_features",
        lambda *_args, **_kwargs: (1, 1, 1),
    )

    def fake_compiled(*_args, **kwargs):
        captured.append(kwargs["compact_confidence_distance_bins"])
        return {}

    monkeypatch.setattr(model_impl, "_compiled_opendde_infer", fake_compiled)
    noise_schedule = jnp.asarray([1.0, 0.0], dtype=jnp.float32)

    model_impl.opendde_infer_compiled({}, _confidence_only_params(), noise_schedule)
    model_impl.opendde_infer_compiled(
        {}, _confidence_only_params(contiguous=False), noise_schedule
    )
    model_impl.opendde_infer_compiled(
        {},
        _confidence_only_params(),
        noise_schedule,
        compact_confidence_distance_bins=False,
    )

    assert "compact_confidence_distance_bins" in GRAPH_STATIC_ARGNAMES
    assert captured == [True, False, False]


@pytest.mark.parametrize(
    "flag",
    ["use_diffusion_scan", "use_pairformer_scan", "use_structural_refiner_scan"],
)
def test_the_scan_flags_the_compiled_path_turns_on_exist(flag: str) -> None:
    """Unrolled, 20 diffusion steps become 20 copies in one executable."""
    import inspect

    assert flag in inspect.signature(opendde_infer_static).parameters
