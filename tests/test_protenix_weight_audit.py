from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bench.protenix_weight_audit import (
    compare_trees,
    map_with_source_keys,
    normalize_state_keys,
    stable_file_identity,
)
from foldjax.models.protenix.bridge.torch_mapping import map_distogram_state_dict


def _state():
    return {
        "linear.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "linear.bias": np.asarray([0.125, -0.25], np.float32),
    }


def test_explicit_native_keys_follow_the_real_mapping():
    mapped, reads, sources = map_with_source_keys(_state(), map_distogram_state_dict)
    result = compare_trees(mapped, jax.tree.map(lambda x: x.copy(), mapped), sources)
    assert result["passed"]
    assert result["counts"]["array_leaves"] == 2
    assert reads == {"linear.weight": 1, "linear.bias": 1}
    assert {leaf["source_key"] for leaf in result["leaf_mapping"]} == set(_state())
    assert (
        result["canonical_contents_sha256"][0] == result["canonical_contents_sha256"][1]
    )


@pytest.mark.parametrize("change", ["bytes", "dtype", "shape", "unmapped"])
def test_parameter_comparison_fails_closed(change):
    mapped, _, sources = map_with_source_keys(_state(), map_distogram_state_dict)
    managed = mapped
    if change == "bytes":
        managed = mapped._replace(
            linear=mapped.linear._replace(bias=mapped.linear.bias.at[0].add(1e-6))
        )
    elif change == "dtype":
        managed = mapped._replace(
            linear=mapped.linear._replace(bias=mapped.linear.bias.astype(jnp.bfloat16))
        )
    elif change == "shape":
        managed = mapped._replace(
            linear=mapped.linear._replace(bias=mapped.linear.bias[:, None])
        )
    else:
        sources = {}
    assert not compare_trees(mapped, managed, sources)["passed"]


@pytest.mark.parametrize("replacement", [False, 1, None])
def test_static_boolean_values_and_types_are_compared_and_hashed(replacement):
    Weights = namedtuple("Weights", "array enabled optional")
    value = jnp.asarray([1], jnp.float32)
    left = Weights(value, True, None)
    same = compare_trees(left, left, {id(value): ("native", value)})
    assert same["passed"]
    assert same["counts"]["static_leaves"] == 1
    assert same["counts"]["none_leaves"] == 1
    result = compare_trees(
        left, left._replace(enabled=replacement), {id(value): ("native", value)}
    )
    assert not result["passed"]
    assert (
        result["canonical_contents_sha256"][0] != result["canonical_contents_sha256"][1]
    )


def test_tree_structure_mismatch_is_not_zipped_away():
    with pytest.raises(ValueError, match="tree definitions"):
        compare_trees({"a": None}, {"b": None}, {})


def test_missing_and_extra_native_keys_are_observable():
    state = _state()
    state["unmapped.weight"] = np.ones(1, np.float32)
    _, reads, _ = map_with_source_keys(state, map_distogram_state_dict)
    assert set(state) - set(reads) == {"unmapped.weight"}
    del state["linear.bias"]
    with pytest.raises(KeyError, match="missing checkpoint key"):
        map_with_source_keys(state, map_distogram_state_dict)


def test_mapping_never_silently_narrows_native_storage():
    state = _state()
    state["linear.bias"] = state["linear.bias"].astype(np.float64)
    with jax.enable_x64(False):
        with pytest.raises(ValueError, match="changed native parameter bytes"):
            map_with_source_keys(state, map_distogram_state_dict)


def test_module_prefix_normalization_detects_collisions():
    values, origins = normalize_state_keys({"module.a": 1, "b": 2})
    assert values == {"a": 1, "b": 2}
    assert origins == {"a": "module.a", "b": "b"}
    with pytest.raises(ValueError, match="collision"):
        normalize_state_keys({"module.a": 1, "a": 2})


def test_read_access_time_is_not_a_checkpoint_mutation():
    fields = {
        name: 1
        for name in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_atime_ns",
        )
    }
    with patch("pathlib.Path.stat", return_value=SimpleNamespace(**fields)):
        before = stable_file_identity("checkpoint")
    with patch(
        "pathlib.Path.stat",
        return_value=SimpleNamespace(**{**fields, "st_atime_ns": 2}),
    ):
        assert stable_file_identity("checkpoint") == before
    with patch(
        "pathlib.Path.stat",
        return_value=SimpleNamespace(**{**fields, "st_mtime_ns": 2}),
    ):
        assert stable_file_identity("checkpoint") != before
