from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import foldjax.models.chai.models.msa as msa_lib
from foldjax.models.chai.bridge.component_io import load_component_state_dict

pytestmark = pytest.mark.official_parity

@pytest.fixture(scope="module")
def trunk_path(official_asset_path):
    return official_asset_path("trunk.pt")


@pytest.fixture(scope="module")
def official_state(trunk_path):
    return load_component_state_dict(trunk_path)


@pytest.fixture(scope="module")
def official_params(official_state):
    return msa_lib.map_msa_module(official_state)


def test_official_mapping_consumes_all_110_msa_tensors(
    official_state, official_params
) -> None:
    official_keys = {
        key for key in official_state if key.startswith("msa_module.")
    }
    assert len(official_keys) == 110
    assert len(jax.tree.leaves(official_params)) == 110
    assert len(official_params.blocks) == 4
    assert all(
        block.weighted_averaging is not None
        and block.msa_transition is not None
        for block in official_params.blocks[:3]
    )
    assert official_params.blocks[3].weighted_averaging is None
    assert official_params.blocks[3].msa_transition is None


def test_linear_s2m_matches_official_callable(official_state, trunk_path) -> None:
    torch = pytest.importorskip("torch")
    module = torch.jit.load(str(trunk_path), map_location="cpu")
    torch.manual_seed(5)
    value = torch.randn(1, 5, 384).bfloat16()
    with torch.no_grad():
        expected = module.msa_module.linear_s2m(value).float().numpy()
    actual = msa_lib.linear_bf16(
        jnp.asarray(value.float().numpy(), dtype=jnp.bfloat16),
        jnp.asarray(official_state["msa_module.linear_s2m.weight"]),
    )
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32), expected, rtol=2e-5, atol=2e-5
    )


def test_msa_transition_real_weights_against_torch_formula(official_state) -> None:
    torch = pytest.importorskip("torch")
    params = msa_lib.map_pairformer_transition(
        official_state, "msa_module.msa_transition.0"
    )
    torch.manual_seed(7)
    value = torch.randn(1, 4, 5, 64)
    with torch.no_grad():
        normalized = torch.nn.functional.layer_norm(
            value.float(),
            (64,),
            torch.from_numpy(
                official_state["msa_module.msa_transition.0.layer_norm.weight"]
            ),
            torch.from_numpy(
                official_state["msa_module.msa_transition.0.layer_norm.bias"]
            ),
        )
        merged = torch.nn.functional.linear(
            normalized.bfloat16(),
            torch.from_numpy(
                official_state[
                    "msa_module.msa_transition.0.linear_no_bias_ab.weight"
                ]
            ).bfloat16(),
        )
        first, second = merged.chunk(2, dim=-1)
        expected = torch.nn.functional.linear(
            torch.nn.functional.silu(first) * second,
            torch.from_numpy(
                official_state["msa_module.msa_transition.0.linear_out.weight"]
            ).bfloat16(),
        ).float().numpy()
    actual = msa_lib.pairformer_transition(jnp.asarray(value.numpy()), params)
    actual_array = np.asarray(actual, dtype=np.float32)
    difference = actual_array - expected
    relative_l2 = np.linalg.norm(difference) / np.linalg.norm(expected)
    # CPU Torch and XLA use different BF16 GEMM reduction orders.  Both sides
    # round to BF16; compare the aggregate drift and the largest observed bin.
    assert relative_l2 < 5e-3
    assert np.max(np.abs(difference)) <= 0.5


def test_outer_product_mean_real_weights_against_torch_formula(
    official_state,
) -> None:
    torch = pytest.importorskip("torch")
    functional = torch.nn.functional
    params = msa_lib.map_outer_product_mean(
        official_state, "msa_module.outer_product_mean.0"
    )
    generator = torch.Generator().manual_seed(11)
    value = torch.randn(1, 3, 4, 64, generator=generator)
    mask = torch.tensor(
        [[[True, True, False, True], [True] * 4, [False, True, True, True]]]
    )
    state = {
        key.removeprefix("msa_module.outer_product_mean.0."): torch.from_numpy(
            np.asarray(array)
        )
        for key, array in official_state.items()
        if key.startswith("msa_module.outer_product_mean.0.")
    }
    normalized = functional.layer_norm(value.float(), (64,))
    normalized = normalized.masked_fill(~mask[..., None], 0).bfloat16()
    weight_a, weight_b = state["weight_ab"].bfloat16().unbind()
    first = torch.einsum("abc,defc->abdef", weight_a, normalized)
    second = torch.einsum("abc,defc->abdef", weight_b, normalized)
    output = torch.einsum("abcde,afcdg->cegabf", first, second)
    output = output.reshape(1, 4, 4, 512)
    output = functional.layer_norm(
        output.float(),
        (512,),
        state["ln_out.weight"],
        state["ln_out.bias"],
        0.1,
    )
    expected = functional.linear(
        output.bfloat16(),
        state["linear_out.weight"].bfloat16(),
        state["linear_out.bias"].bfloat16(),
    )
    actual = msa_lib.outer_product_mean(
        jnp.asarray(value.numpy()),
        jnp.asarray(mask.numpy()),
        params,
        chunk_size=4096,
    )
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32),
        expected.float().numpy(),
        rtol=2e-2,
        atol=2e-2,
    )


def test_weighted_averaging_real_weights_against_torch_formula(
    official_state,
) -> None:
    torch = pytest.importorskip("torch")
    functional = torch.nn.functional
    prefix = "msa_module.msa_pair_weighted_averaging.0."
    params = msa_lib.map_msa_pair_weighted_averaging(
        official_state, prefix.removesuffix(".")
    )
    state = {
        key.removeprefix(prefix): torch.from_numpy(np.asarray(array))
        for key, array in official_state.items()
        if key.startswith(prefix)
    }
    generator = torch.Generator().manual_seed(13)
    msa = torch.randn(1, 3, 4, 64, generator=generator)
    pair = torch.randn(1, 4, 4, 256, generator=generator)
    msa_mask = torch.tensor(
        [[[True, True, False, True], [True] * 4, [False, True, True, True]]]
    )
    pair_mask = torch.ones(1, 4, 4, dtype=torch.bool)
    pair_mask[:, :, -1] = False
    pair_normalized = functional.layer_norm(
        pair.float(),
        (256,),
        state["layernorm_pair.weight"],
        state["layernorm_pair.bias"],
    )
    logits = functional.linear(
        pair_normalized.bfloat16(), state["linear_pair.weight"].bfloat16()
    ).permute(0, 3, 1, 2)
    weights = logits.masked_fill(~pair_mask[:, None], -10000).softmax(
        dim=-1, dtype=torch.float32
    )
    msa_normalized = functional.layer_norm(
        msa.float(),
        (64,),
        state["layernorm_msa.weight"],
        state["layernorm_msa.bias"],
    )
    value_gate = functional.linear(
        msa_normalized.bfloat16(),
        state["linear_msa2vg.weight"].bfloat16(),
    ).reshape(1, 3, 4, 2, 8, 32)
    value, gate = value_gate.permute(3, 0, 1, 2, 4, 5).unbind()
    value = value.masked_fill(~msa_mask[..., None, None], 0)
    attended = torch.einsum(
        "abcd,aedbf->aecbf", weights.bfloat16(), value
    )
    attended = (gate.sigmoid() * attended).reshape(1, 3, 4, 256)
    expected = functional.linear(
        attended, state["linear_out_no_bias.weight"].bfloat16()
    )
    actual = msa_lib.msa_pair_weighted_averaging(
        jnp.asarray(msa.numpy()),
        jnp.asarray(pair.numpy()),
        jnp.asarray(msa_mask.numpy()),
        jnp.asarray(pair_mask.numpy()),
        params,
    )
    actual_array = np.asarray(actual, dtype=np.float32)
    expected_array = expected.float().numpy()
    relative_l2 = np.linalg.norm(actual_array - expected_array) / np.linalg.norm(
        expected_array
    )
    assert relative_l2 < 1e-2
    assert np.max(np.abs(actual_array - expected_array)) <= 0.5


@pytest.mark.parametrize("valid_rows", [0, 4])
def test_masked_padding_rows_do_not_change_msa_pair_output(
    official_params, valid_rows
) -> None:
    rng = np.random.default_rng(17)
    msa = jnp.asarray(rng.normal(size=(1, 8, 4, 64)).astype(np.float32))
    msa_mask = np.zeros((1, 8, 4), dtype=np.bool_)
    msa_mask[:, :valid_rows] = True
    single = jnp.asarray(rng.normal(size=(1, 4, 384)).astype(np.float32))
    pair = jnp.asarray(rng.normal(size=(1, 4, 4, 256)).astype(np.float32))
    pair_mask = jnp.ones((1, 4, 4), dtype=jnp.bool_)

    full = msa_lib.msa_module(
        msa,
        jnp.asarray(msa_mask),
        single,
        pair,
        pair_mask,
        official_params,
        outer_product_chunk_size=8,
        weighted_averaging_chunk_size=8,
        transition_chunk_size=8,
    )
    cropped_rows = max(1, valid_rows)
    cropped = msa_lib.msa_module(
        msa[:, :cropped_rows],
        jnp.asarray(msa_mask[:, :cropped_rows]),
        single,
        pair,
        pair_mask,
        official_params,
        outer_product_chunk_size=cropped_rows,
        weighted_averaging_chunk_size=cropped_rows,
        transition_chunk_size=cropped_rows,
    )

    np.testing.assert_array_equal(np.asarray(cropped), np.asarray(full))


def test_weighted_depth_chunk_128_matches_1024_exactly(official_params) -> None:
    params = official_params.blocks[0].weighted_averaging
    assert params is not None
    rng = np.random.default_rng(18)
    msa = jnp.asarray(rng.normal(size=(1, 1024, 4, 64)).astype(np.float32))
    pair = jnp.asarray(rng.normal(size=(1, 4, 4, 256)).astype(np.float32))
    msa_mask = jnp.asarray(rng.random((1, 1024, 4)) > 0.1)
    pair_mask = jnp.ones((1, 4, 4), dtype=jnp.bool_)

    chunk_1024 = msa_lib.msa_pair_weighted_averaging(
        msa, msa_mask=msa_mask, pair=pair, pair_mask=pair_mask,
        params=params, chunk_size=1024
    )
    chunk_128 = msa_lib.msa_pair_weighted_averaging(
        msa, msa_mask=msa_mask, pair=pair, pair_mask=pair_mask,
        params=params, chunk_size=128
    )

    chunk_128_array = np.asarray(chunk_128, dtype=np.float32)
    chunk_1024_array = np.asarray(chunk_1024, dtype=np.float32)
    difference = chunk_128_array - chunk_1024_array
    relative_l2 = np.linalg.norm(difference) / np.linalg.norm(chunk_1024_array)
    assert relative_l2 < 1e-4
    assert np.max(np.abs(difference)) <= 0.125


def test_four_block_orchestration(monkeypatch, official_params) -> None:
    events: list[str] = []
    opm_ids = {
        id(block.outer_product_mean): index
        for index, block in enumerate(official_params.blocks)
    }
    weighted_ids = {
        id(block.weighted_averaging): index
        for index, block in enumerate(official_params.blocks[:3])
    }
    pair_ids = {
        id(block.pair): index for index, block in enumerate(official_params.blocks)
    }

    def fake_linear(value, weight, bias=None):
        del weight, bias
        return jnp.zeros(value.shape[:-1] + (64,), dtype=value.dtype)

    def fake_opm(msa, mask, params, *, chunk_size):
        del mask, chunk_size
        index = opm_ids[id(params)]
        events.append(f"opm{index}")
        return jnp.full(
            (msa.shape[0], msa.shape[2], msa.shape[2], 256),
            index + 1,
            dtype=msa.dtype,
        )

    def fake_transition(value, params):
        del params
        events.append("transition")
        return jnp.ones_like(value)

    def fake_weighted(msa, pair, msa_mask, pair_mask, params, *, chunk_size):
        del pair, msa_mask, pair_mask, chunk_size
        index = weighted_ids[id(params)]
        events.append(f"weighted{index}")
        return jnp.ones_like(msa)

    def fake_pair(pair, pair_mask, params):
        del pair_mask
        index = pair_ids[id(params)]
        events.append(f"pair{index}")
        return pair + 10

    monkeypatch.setattr(msa_lib, "linear_bf16", fake_linear)
    monkeypatch.setattr(msa_lib, "outer_product_mean", fake_opm)
    monkeypatch.setattr(msa_lib, "pairformer_transition", fake_transition)
    monkeypatch.setattr(
        msa_lib, "msa_pair_weighted_averaging", fake_weighted
    )
    monkeypatch.setattr(msa_lib, "_msa_pair_block", fake_pair)
    output = msa_lib.msa_module(
        jnp.zeros((1, 2, 3, 64)),
        jnp.ones((1, 2, 3), dtype=bool),
        jnp.zeros((1, 3, 384)),
        jnp.zeros((1, 3, 3, 256)),
        jnp.ones((1, 3, 3), dtype=bool),
        official_params,
    )
    assert events == [
        "opm0",
        "transition",
        "weighted0",
        "pair0",
        "opm1",
        "transition",
        "weighted1",
        "pair1",
        "opm2",
        "transition",
        "weighted2",
        "pair2",
        "opm3",
        "pair3",
    ]
    np.testing.assert_array_equal(np.asarray(output), np.full(output.shape, 50))
