"""Torch-compatible recycle-MSA subsampling tests."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.chai.data.msa import subsample_and_reorder_msa_features
from foldjax.models.chai.inference import (
    _embed_msa_for_recycle,
    _subsample_msa_for_recycle,
)


def test_subsample_is_noop_when_real_depth_is_at_most_limit() -> None:
    features = np.arange(1 * 4 * 2 * 1, dtype=np.float32).reshape(1, 4, 2, 1)
    mask = np.asarray([[[True, True], [True, False], [False, False], [False, False]]])

    sampled_features, sampled_mask = subsample_and_reorder_msa_features(
        features,
        mask,
        select_n_rows=2,
        random_values=np.ones(4, dtype=np.float16),
    )

    assert sampled_features is features
    assert sampled_mask is mask


@pytest.mark.official_parity
def test_two_recycle_subsamples_match_upstream_given_identical_random_draws(
    tmp_path: Path,
    upstream_chai_dir: Path,
    upstream_chai_python: Path,
) -> None:
    output = tmp_path / "upstream.npz"
    script = r"""
import numpy as np
import torch
from chai_lab.data.dataset.msas.utils import subsample_and_reorder_msa_feats_n_mask

features = torch.arange(1 * 8 * 3 * 1, dtype=torch.float32).reshape(1, 8, 3, 1)
mask = torch.tensor([[
    [1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1],
    [0, 0, 0], [1, 1, 0], [0, 0, 0], [1, 1, 1],
]], dtype=torch.bool)
draw_generator = torch.Generator().manual_seed(1234)
random_values = [
    torch.rand((8,), dtype=torch.float16, generator=draw_generator)
    for _ in range(2)
]
call_generator = torch.Generator().manual_seed(1234)
sampled = [
    subsample_and_reorder_msa_feats_n_mask(
        features, mask, select_n_rows=3, generator=call_generator
    )
    for _ in range(2)
]
np.savez(
    __OUTPUT__,
    features=features.numpy(),
    mask=mask.numpy(),
    random_values_0=random_values[0].numpy(),
    random_values_1=random_values[1].numpy(),
    sampled_features_0=sampled[0][0].numpy(),
    sampled_mask_0=sampled[0][1].numpy(),
    sampled_features_1=sampled[1][0].numpy(),
    sampled_mask_1=sampled[1][1].numpy(),
)
""".replace("__OUTPUT__", repr(str(output)))
    subprocess.run(
        [upstream_chai_python, "-c", script],
        cwd=upstream_chai_dir,
        check=True,
    )
    with np.load(output) as reference:
        actual = []
        for recycle in range(2):
            sampled_features, sampled_mask = subsample_and_reorder_msa_features(
                reference["features"],
                reference["mask"],
                select_n_rows=3,
                random_values=reference[f"random_values_{recycle}"],
            )
            np.testing.assert_array_equal(
                sampled_features, reference[f"sampled_features_{recycle}"]
            )
            np.testing.assert_array_equal(
                sampled_mask, reference[f"sampled_mask_{recycle}"]
            )
            actual.append(sampled_features)
        assert not np.array_equal(actual[0][:, :3], actual[1][:, :3])


def test_runtime_subsample_selects_4096_rows_deterministically() -> None:
    depth = 4102
    features = jnp.arange(depth, dtype=jnp.float32).reshape(1, depth, 1, 1)
    mask = np.ones((1, depth, 1), dtype=np.bool_)
    key = jax.random.PRNGKey(17)

    first_features, first_mask = _subsample_msa_for_recycle(features, mask, key)
    second_features, second_mask = _subsample_msa_for_recycle(features, mask, key)

    assert first_features.shape[1] == 4096
    assert first_mask.shape[1] == 4096
    assert int(np.asarray(first_mask).sum()) == 4096
    np.testing.assert_array_equal(first_features, second_features)
    np.testing.assert_array_equal(first_mask, second_mask)


def test_runtime_buckets_trailing_masked_rows_without_subsampling() -> None:
    features = jnp.arange(8, dtype=jnp.float32).reshape(1, 8, 1, 1)
    mask = np.zeros((1, 8, 1), dtype=np.bool_)
    mask[:, :3] = True

    sampled_features, sampled_mask = _subsample_msa_for_recycle(
        features, mask, jax.random.PRNGKey(19)
    )

    assert sampled_features.shape == (1, 128, 1, 1)
    assert sampled_mask.shape == (1, 128, 1)
    np.testing.assert_array_equal(
        np.asarray(sampled_features)[:, :3], np.arange(3).reshape(1, 3, 1, 1)
    )
    assert not np.asarray(sampled_mask)[:, 3:].any()


def test_raw_msa_is_cropped_before_compiled_embedding(monkeypatch) -> None:
    mask = np.zeros((1, 8, 2), dtype=np.bool_)
    mask[:, :3] = True
    features = {
        name: np.zeros((1, 8, 2), dtype=np.float32)
        for name in (
            "IsPairedMSA",
            "MSADataSource",
            "MSADeletionValue",
            "MSAHasDeletion",
            "MSAOneHot",
        )
    }
    seen_depths: list[int] = []

    def fake_embed(raw_features, weight, bias):
        del weight, bias
        seen_depths.extend(value.shape[1] for value in raw_features.values())
        return jnp.zeros((1, 128, 2, 64), dtype=jnp.bfloat16)

    monkeypatch.setattr(
        "foldjax.models.chai.inference._compiled_embed_msa", fake_embed
    )
    prepared = SimpleNamespace(
        padded_inputs={"msa_mask": mask},
        features=features,
    )
    params = {
        "input_projs.MSA.0.weight": np.zeros((64, 1), dtype=np.float32),
        "input_projs.MSA.0.bias": np.zeros((64,), dtype=np.float32),
    }

    embedded, embedded_mask = _embed_msa_for_recycle(
        prepared, params, key=None
    )

    assert seen_depths == [128] * 5
    assert embedded.shape[1] == 128
    assert embedded_mask.shape[1] == 128
    assert int(np.asarray(embedded_mask).sum()) == 6


def test_default_no_subsample_preserves_more_than_4096_active_rows() -> None:
    depth = 5000
    features = jnp.arange(depth, dtype=jnp.float32).reshape(1, depth, 1, 1)
    mask = np.ones((1, depth, 1), dtype=np.bool_)

    selected_features, selected_mask = _subsample_msa_for_recycle(
        features, mask, key=None
    )

    assert selected_features.shape == (1, 8192, 1, 1)
    assert selected_mask.shape == (1, 8192, 1)
    np.testing.assert_array_equal(
        np.asarray(selected_features)[0, :depth, 0, 0], np.arange(depth)
    )
    assert int(np.asarray(selected_mask).sum()) == depth


def test_runtime_uses_distinct_deterministic_keys_for_each_recycle() -> None:
    depth = 4102
    features = jnp.arange(depth, dtype=jnp.float32).reshape(1, depth, 1, 1)
    mask = np.ones((1, depth, 1), dtype=np.bool_)

    def recycle_orders(seed: int) -> tuple[np.ndarray, np.ndarray]:
        key = jax.random.PRNGKey(seed)
        orders = []
        for _ in range(2):
            key, recycle_key = jax.random.split(key)
            sampled, sampled_mask = _subsample_msa_for_recycle(
                features, mask, recycle_key
            )
            assert int(np.asarray(sampled_mask).sum()) == 4096
            orders.append(np.asarray(sampled)[0, :4096, 0, 0])
        return orders[0], orders[1]

    first = recycle_orders(17)
    second = recycle_orders(17)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert not np.array_equal(first[0], first[1])
