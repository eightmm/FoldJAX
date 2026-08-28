from __future__ import annotations

import re
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.opendde.data.featurize_json import featurize_opendde_json
from foldjax.models.opendde.data.padding import select_opendde_model_features
from foldjax.models.protenix.data.featurize_json import featurize_protein_json
from foldjax.models.protenix.models.primitives.primitives import LinearParams
from foldjax.models.protenix.models.trunk_blocks.embedders import (
    RelativePositionParams,
    compact_relative_position_features,
    relative_position_encoding,
    resolve_relative_position_features,
)
from foldjax.models.protenix.relative_position import (
    COMPACT_RELP_COMPONENTS,
    COMPACT_RELP_FIELDS,
    COMPACT_RELP_MARKER,
    COMPACT_RELP_RESIDUE_BIN,
    compact_relative_position_storage,
    normalize_relative_position_storage,
)
from tests.models.cp_probe_env import inherited_environment


def _compact(n_token: int = 8) -> dict[str, np.ndarray]:
    return compact_relative_position_storage(
        asym_id=np.repeat(np.arange(2), n_token // 2),
        residue_index=np.arange(n_token),
        entity_id=np.tile(np.arange(2), n_token // 2),
        sym_id=np.arange(n_token) % 3,
        token_index=np.arange(n_token),
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_compact_projection_is_byte_identical_under_strict_promotion(dtype) -> None:
    compact = _compact()
    dense = np.asarray(compact_relative_position_features(compact))
    rng = np.random.default_rng(20260828)
    weight = rng.normal(size=(19, 139)).astype(np.float32).astype(dtype)
    params = RelativePositionParams(
        linear_no_bias=LinearParams(weight=jnp.asarray(weight), bias=None)
    )

    def project(features):
        return relative_position_encoding(
            resolve_relative_position_features(features),
            params,
        )

    with jax.numpy_dtype_promotion("strict"):
        historical = jax.jit(project)({"relp": jnp.asarray(dense)})
        actual = jax.jit(project)(jax.tree.map(jnp.asarray, compact))
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(historical))


def test_dense_relp_wins_over_malformed_private_marker() -> None:
    dense = np.arange(4 * 4 * 5, dtype=np.float32).reshape(4, 4, 5)
    for marker in (
        np.asarray(1, dtype=np.uint8),
        np.asarray([99], dtype=np.int32),
    ):
        features = {**_compact(4), "relp": dense, COMPACT_RELP_MARKER: marker}
        normalized = normalize_relative_position_storage(features, n_token=4)
        assert normalized["relp"] is dense
        assert not (set(COMPACT_RELP_FIELDS) & normalized.keys())
        assert resolve_relative_position_features(features) is dense


def test_incomplete_or_stale_compact_marker_fails_explicitly() -> None:
    compact = _compact(4)
    incomplete = dict(compact)
    incomplete.pop(COMPACT_RELP_COMPONENTS[-1])
    with pytest.raises(ValueError, match="incomplete"):
        normalize_relative_position_storage(incomplete, n_token=4)

    stale = dict(compact)
    stale[COMPACT_RELP_MARKER] = np.asarray(2, dtype=np.uint8)
    with pytest.raises(ValueError, match="version 1"):
        normalize_relative_position_storage(stale, n_token=4)

    malformed_component = dict(compact)
    malformed_component[COMPACT_RELP_RESIDUE_BIN] = np.zeros((4, 4), np.int16)
    with pytest.raises(ValueError, match="dtype uint8"):
        normalize_relative_position_storage(malformed_component, n_token=4)

    markerless = dict(compact)
    markerless.pop(COMPACT_RELP_MARKER)
    with pytest.raises(ValueError, match="no provenance marker"):
        normalize_relative_position_storage(markerless, n_token=4, require=True)


def test_compact_padding_sentinel_rebuilds_historical_zero_pairs() -> None:
    from foldjax.models.protenix.relative_position import (
        pad_compact_relative_position_storage,
    )

    compact = _compact(4)
    dense = np.asarray(compact_relative_position_features(compact))
    padded = pad_compact_relative_position_storage(
        compact,
        storage_token=4,
        target_token=7,
    )
    padded["token_padding_mask"] = np.asarray([1, 1, 1, 1, 0, 0, 0], np.float32)
    normalize_relative_position_storage(
        padded,
        n_token=7,
        token_padding_mask=padded["token_padding_mask"],
        require=True,
    )
    rebuilt = np.asarray(compact_relative_position_features(padded))
    expected = np.pad(dense, ((0, 3), (0, 3), (0, 0)))
    np.testing.assert_array_equal(rebuilt, expected)


def test_compact_and_dense_mappings_have_distinct_jit_cache_identity() -> None:
    compact = jax.tree.map(jnp.asarray, _compact())
    dense = {"relp": compact_relative_position_features(compact)}
    params = RelativePositionParams(
        linear_no_bias=LinearParams(
            weight=jnp.arange(7 * 139, dtype=jnp.float32).reshape(7, 139),
            bias=None,
        )
    )

    @jax.jit
    def project(features):
        return relative_position_encoding(
            resolve_relative_position_features(features), params
        )

    dense_out = project(dense)
    compact_out = project(compact)
    np.testing.assert_array_equal(np.asarray(compact_out), np.asarray(dense_out))
    assert project._cache_size() == 2
    assert jax.tree.structure(compact) != jax.tree.structure(dense)


def test_compact_storage_scales_quadratically_at_four_bytes_per_pair() -> None:
    small = _compact(8)
    large = _compact(16)

    def component_bytes(features):
        return sum(
            np.asarray(features[name]).nbytes for name in COMPACT_RELP_COMPONENTS
        )

    assert component_bytes(small) == 4 * 8**2
    assert component_bytes(large) == 4 * 16**2
    assert component_bytes(large) == 4 * component_bytes(small)
    assert 139 * 16**2 > 34 * component_bytes(large)


def test_compact_lowering_removes_the_139_channel_entry_argument() -> None:
    n_token = 8
    compact = jax.tree.map(jnp.asarray, _compact(n_token))
    dense = {"relp": compact_relative_position_features(compact)}
    params = RelativePositionParams(
        linear_no_bias=LinearParams(
            weight=jnp.ones((5, 139), dtype=jnp.float32),
            bias=None,
        )
    )

    def project(features):
        return relative_position_encoding(
            resolve_relative_position_features(features), params
        )

    dense_hlo = str(jax.jit(project).lower(dense).compiler_ir("stablehlo"))
    compact_hlo = str(jax.jit(project).lower(compact).compiler_ir("stablehlo"))
    signature = re.compile(r"func.func public @main\((.*?)\) ->", re.DOTALL)
    dense_signature = signature.search(dense_hlo)
    compact_signature = signature.search(compact_hlo)
    assert dense_signature is not None and compact_signature is not None
    assert f"tensor<{n_token}x{n_token}x139xi8>" in dense_signature.group(1)
    assert f"tensor<{n_token}x{n_token}x139xi8>" not in compact_signature.group(1)
    assert compact_signature.group(1).count(f"tensor<{n_token}x{n_token}xui8>") == 4


def test_managed_protenix_and_opendde_featurizers_use_private_compact_form() -> None:
    job = {"sequences": [{"proteinChain": {"sequence": "AG"}}]}
    protenix = featurize_protein_json(job, n_queries=2, n_keys=4)
    opendde = featurize_opendde_json(
        job,
        n_queries=2,
        n_keys=4,
        seed=101,
        augment_reference=False,
    )
    for features in (protenix, opendde):
        assert "relp" not in features
        assert set(COMPACT_RELP_FIELDS) <= features.keys()
    selected = select_opendde_model_features(opendde)
    assert set(COMPACT_RELP_FIELDS) <= selected.keys()


_CP_PROBE = textwrap.dedent(
    """
    import os
    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel, shard_pair_rows
    from foldjax.models.protenix.models.primitives.primitives import LinearParams
    from foldjax.models.protenix.models.trunk_blocks.embedders import (
        RelativePositionParams,
        relative_position_encoding,
        resolve_relative_position_features,
    )
    from foldjax.models.protenix.relative_position import (
        compact_relative_position_storage,
    )

    assert jax.device_count() == 4, jax.devices()
    layout = os.environ["FOLDJAX_CP_PROBE_LAYOUT"]
    n = 16
    compact = compact_relative_position_storage(
        asym_id=np.repeat(np.arange(2), n // 2),
        residue_index=np.arange(n),
        entity_id=np.zeros(n, dtype=np.int32),
        sym_id=np.arange(n) % 2,
        token_index=np.arange(n),
    )
    compact = jax.tree.map(jnp.asarray, compact)
    rng = np.random.default_rng(7)

    def run(features, weight):
        relp = shard_pair_rows(resolve_relative_position_features(features))
        params = RelativePositionParams(
            linear_no_bias=LinearParams(weight=weight, bias=None)
        )
        return relative_position_encoding(relp, params)

    for dtype in (jnp.bfloat16, jnp.float32):
        weight = jnp.asarray(rng.normal(size=(9, 139)), dtype=dtype)
        reference = np.asarray(jax.jit(run)(compact, weight))
        jax.clear_caches()
        with context_parallel(4, layout=layout):
            actual = np.asarray(jax.jit(run)(compact, weight))
        np.testing.assert_array_equal(actual, reference)
    print("COMPACT_RELP_CP_OK")
    """
)


@pytest.mark.parametrize("layout", ["1d", "2d"])
def test_compact_protenix_opendde_projections_match_on_four_cpu_devices(
    layout: str,
) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _CP_PROBE],
        capture_output=True,
        text=True,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
            "FOLDJAX_CP_PROBE_LAYOUT": layout,
            **inherited_environment(),
        },
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "COMPACT_RELP_CP_OK" in completed.stdout
