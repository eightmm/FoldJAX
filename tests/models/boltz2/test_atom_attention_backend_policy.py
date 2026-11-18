"""Stage-scoped backend policy for the Boltz-2 BF16 atom attention pilot."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import foldjax.backends.boltz2 as backend_module
import foldjax.models.boltz2.models.diffusion.diffusion_transformer as dt_module
import foldjax.models.boltz2.models.trunk_blocks.trunk as trunk_module
from foldjax.backends.boltz2 import Boltz2Backend
from foldjax.models.boltz2 import api
from foldjax.models.boltz2.models.primitives.attention_backend import (
    tokamax_dot_product_attention,
)


def test_tokamax_auto_and_forced_triton_are_distinct(monkeypatch) -> None:
    import tokamax

    implementations = []

    def fake_attention(q, k, v, **kwargs):
        implementations.append(kwargs["implementation"])
        return jnp.zeros_like(q)

    monkeypatch.setattr(tokamax, "dot_product_attention", fake_attention)
    q = jnp.ones((1, 2, 1, 4), dtype=jnp.bfloat16)
    k = jnp.ones((1, 3, 1, 4), dtype=jnp.bfloat16)
    v = jnp.ones((1, 3, 1, 4), dtype=jnp.bfloat16)
    bias = jnp.zeros((1, 1, 2, 3), dtype=jnp.bfloat16)
    mask = jnp.ones((1, 3), dtype=bool)

    for backend in ("tokamax", "triton"):
        out = tokamax_dot_product_attention(
            q, k, v, bias, mask, scale=0.5, backend=backend
        )
        np.testing.assert_array_equal(np.asarray(out), np.zeros((1, 2, 1, 4)))

    assert implementations == [None, "triton"]


def test_forced_triton_preserves_empty_key_zero_contract(monkeypatch) -> None:
    import tokamax

    seen_mask = []

    def fake_attention(q, k, v, **kwargs):
        seen_mask.append(np.asarray(kwargs["mask"]))
        return jnp.ones_like(q)

    monkeypatch.setattr(tokamax, "dot_product_attention", fake_attention)
    q = jnp.ones((1, 2, 1, 4), dtype=jnp.bfloat16)
    k = jnp.ones((1, 3, 1, 4), dtype=jnp.bfloat16)
    v = jnp.ones((1, 3, 1, 4), dtype=jnp.bfloat16)
    bias = jnp.zeros((1, 1, 2, 3), dtype=jnp.bfloat16)
    out = tokamax_dot_product_attention(
        q,
        k,
        v,
        bias,
        jnp.zeros((1, 3), dtype=bool),
        scale=0.5,
        backend="triton",
    )

    assert seen_mask[0][0, 0, 0, 0]
    np.testing.assert_array_equal(np.asarray(out), np.zeros((1, 2, 1, 4)))


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32])
def test_forced_triton_rejects_non_bfloat16_inputs(dtype) -> None:
    q = jnp.ones((1, 2, 1, 4), dtype=dtype)
    k = jnp.ones((1, 3, 1, 4), dtype=dtype)
    v = jnp.ones((1, 3, 1, 4), dtype=dtype)
    bias = jnp.zeros((1, 1, 2, 3), dtype=dtype)
    mask = jnp.ones((1, 3), dtype=bool)

    with pytest.raises(ValueError, match="requires bfloat16"):
        tokamax_dot_product_attention(
            q, k, v, bias, mask, scale=0.5, backend="triton"
        )


def test_atom_transformer_routes_triton_to_tokamax_wrapper(monkeypatch) -> None:
    seen = []

    def fake_attention(q, k, v, bias, mask, **kwargs):
        seen.append((kwargs["backend"], q.dtype, k.dtype, v.dtype))
        return jnp.zeros_like(q)

    monkeypatch.setattr(dt_module, "tokamax_dot_product_attention", fake_attention)
    c_s = 4
    params = {
        "proj_q": {
            "kernel": jnp.eye(c_s, dtype=jnp.bfloat16),
            "bias": jnp.zeros((c_s,), dtype=jnp.bfloat16),
        },
        "proj_g": {"kernel": jnp.zeros((c_s, c_s), dtype=jnp.bfloat16)},
        "proj_k": {"kernel": jnp.eye(c_s, dtype=jnp.bfloat16)},
        "proj_v": {"kernel": jnp.eye(c_s, dtype=jnp.bfloat16)},
        "proj_o": {"kernel": jnp.eye(c_s, dtype=jnp.bfloat16)},
    }
    out = dt_module._attention_pair_bias_no_proj_z_forward(
        params,
        s=jnp.ones((1, 2, c_s), dtype=jnp.bfloat16),
        bias=jnp.zeros((1, 2, 3, 2), dtype=jnp.bfloat16),
        mask=jnp.ones((1, 3), dtype=bool),
        k_in=jnp.ones((1, 3, c_s), dtype=jnp.bfloat16),
        multiplicity=1,
        inf=1e6,
        attention_backend="triton",
        distributed_pair_bias=False,
    )

    assert seen == [("triton", jnp.bfloat16, jnp.bfloat16, jnp.bfloat16)]
    assert out.dtype == jnp.bfloat16


@pytest.mark.parametrize(
    ("global_backend", "atom_backend", "expected"),
    [("xla", None, "xla"), ("tokamax", None, "tokamax"), ("xla", "triton", "triton")],
)
def test_trunk_resolves_only_input_embedder_backend(
    monkeypatch, global_backend: str, atom_backend: str | None, expected: str
) -> None:
    class ReachedInputEmbedderError(Exception):
        pass

    def fake_input_embedder(params, feats, *, attention_backend, **kwargs):
        assert attention_backend == expected
        raise ReachedInputEmbedderError

    monkeypatch.setattr(trunk_module, "input_embedder_forward", fake_input_embedder)
    with pytest.raises(ReachedInputEmbedderError):
        trunk_module.boltz2_trunk_forward(
            {"input_embedder": {}},
            {"token_pad_mask": jnp.ones((1, 2), dtype=jnp.float32)},
            attention_backend=global_backend,
            atom_attention_backend=atom_backend,
        )


def test_sharded_trunk_rejects_fused_atom_override_before_model_work() -> None:
    with pytest.raises(ValueError, match="sharded trunk"):
        trunk_module.boltz2_trunk_forward(
            {},
            {"token_pad_mask": jnp.ones((1, 2), dtype=jnp.float32)},
            atom_attention_backend="triton",
            mesh=object(),
        )


def test_sharded_trunk_accepts_explicit_inherited_fused_backend(monkeypatch) -> None:
    class ReachedInputEmbedderError(Exception):
        pass

    def fake_input_embedder(params, feats, *, attention_backend, **kwargs):
        assert attention_backend == "tokamax"
        raise ReachedInputEmbedderError

    monkeypatch.setattr(trunk_module, "input_embedder_forward", fake_input_embedder)
    with pytest.raises(ReachedInputEmbedderError):
        trunk_module.boltz2_trunk_forward(
            {"input_embedder": {}},
            {"token_pad_mask": jnp.ones((1, 2), dtype=jnp.float32)},
            attention_backend="tokamax",
            atom_attention_backend="tokamax",
            mesh=object(),
        )


@pytest.mark.parametrize("entrypoint", ["graph_score", "sample"])
def test_standalone_graphs_forward_atom_attention_override(
    monkeypatch, entrypoint: str
) -> None:
    class ReachedTrunkError(Exception):
        pass

    def fake_trunk(params, feats, **kwargs):
        assert kwargs["attention_backend"] == "xla"
        assert kwargs["atom_attention_backend"] == "triton"
        raise ReachedTrunkError

    monkeypatch.setattr(trunk_module, "boltz2_trunk_forward", fake_trunk)
    feats = {"token_pad_mask": jnp.ones((1, 2), dtype=jnp.float32)}
    with pytest.raises(ReachedTrunkError):
        if entrypoint == "graph_score":
            trunk_module.boltz2_graph_score_forward(
                {"trunk": {}},
                feats,
                jnp.zeros((1, 1, 3), dtype=jnp.float32),
                jnp.zeros((1,), dtype=jnp.float32),
                attention_backend="xla",
                trunk_atom_attention_backend="triton",
            )
        else:
            trunk_module.boltz2_sample_forward(
                {"trunk": {}},
                feats,
                jnp.zeros((2,), dtype=jnp.uint32),
                compute_dtype=jnp.bfloat16,
                attention_backend="xla",
                trunk_atom_attention_backend="triton",
            )


@pytest.mark.parametrize(
    ("device", "match"),
    [
        (SimpleNamespace(platform="cpu"), "NVIDIA GPU"),
        (
            SimpleNamespace(
                platform="gpu",
                device_kind="NVIDIA T4",
                compute_capability=7.5,
            ),
            "compute capability >= 8.0",
        ),
        (
            SimpleNamespace(
                platform="gpu",
                device_kind="AMD Radeon",
                compute_capability=9.0,
            ),
            "NVIDIA GPU",
        ),
        (
            SimpleNamespace(platform="gpu", device_kind="NVIDIA unknown"),
            "could not verify",
        ),
    ],
)
def test_forced_triton_device_preflight_rejects_unsupported_runtime(
    device, match: str
) -> None:
    with pytest.raises(RuntimeError, match=match):
        api._require_triton_attention_device(
            SimpleNamespace(devices=lambda: [device])
        )


def test_forced_triton_device_preflight_accepts_ampere_or_newer() -> None:
    api._require_triton_attention_device(
        SimpleNamespace(
            devices=lambda: [
                SimpleNamespace(
                    platform="gpu",
                    device_kind="NVIDIA A100",
                    compute_capability=(8, 0),
                )
            ]
        )
    )


def test_high_level_forced_triton_preflights_before_featurization(
    monkeypatch,
    tmp_path,
) -> None:
    class PreflightError(Exception):
        pass

    def fail_preflight(_jax: object) -> None:
        raise PreflightError

    monkeypatch.setattr(
        api,
        "_require_triton_attention_device",
        fail_preflight,
    )
    monkeypatch.setattr(
        api,
        "featurize",
        lambda **kwargs: pytest.fail("featurization ran before Triton preflight"),
    )

    with pytest.raises(PreflightError):
        api.predict(
            seq=["ACD"],
            weights=tmp_path / "unused",
            mols=tmp_path,
            trunk_atom_attention_backend="triton",
        )


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"trunk_atom_attention_backend": "unknown"}, "must be one of"),
        (
            {
                "trunk_atom_attention_backend": "triton",
                "compute_dtype": "float32",
            },
            "requires compute_dtype='bfloat16'",
        ),
        (
            {"trunk_atom_attention_backend": "triton", "cp_devices": 2},
            "context parallelism requires",
        ),
    ],
)
def test_backend_rejects_unsupported_atom_attention_policy(
    options: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        Boltz2Backend().validate_native_options(options)


def test_backend_refuses_the_fused_base_attention_under_cp() -> None:
    """The base knob is refused, and it is the base knob that is named.

    This used to assert the same option dict was *accepted*: the scoped check
    reads the global and collapses an equal spelling to ``None``, so naming
    the fused kernel globally passed every mesh refusal the adapter had and
    left the atom-window ``shard_map`` to raise out of Pallas on a card.

    The scoped canonicalization is still what happens first -- which is why
    the message has to be matched, not just the raise: a scoped refusal firing
    here would mean the collapse broke, and the base one firing means the gap
    is closed. ``attention_backend`` ships as ``xla``, so unlike
    ``glu_backend`` there is no default to resolve away.
    """

    with pytest.raises(ValueError) as raised:
        Boltz2Backend().validate_native_options(
            {
                "attention_backend": "tokamax",
                "trunk_atom_attention_backend": "tokamax",
                "cp_devices": 2,
            }
        )

    assert "requires attention_backend='xla'" in str(raised.value)
    assert "trunk_atom_attention_backend" not in str(raised.value)


def test_backend_accepts_the_partitionable_base_attention_with_cp() -> None:
    Boltz2Backend().validate_native_options(
        {
            "attention_backend": "xla",
            "trunk_atom_attention_backend": "xla",
            "cp_devices": 2,
        }
    )


def test_backend_accepts_the_fused_base_attention_without_cp() -> None:
    # The refusal is the mesh's, not the kernel's: serial runs keep the knob.
    Boltz2Backend().validate_native_options(
        {"attention_backend": "tokamax", "cp_devices": 1}
    )
    Boltz2Backend().validate_native_options({"attention_backend": "tokamax"})


def test_high_level_refuses_the_fused_base_attention_before_featurization(
    monkeypatch, tmp_path
) -> None:
    """Plan time, not run time: the refusal precedes the featurizer.

    ``api.predict`` is the surface the CLI and the adapter both go through,
    and it cannot distinguish an explicit ``tokamax`` from a default one --
    but it does not have to, because this knob's default is ``xla``.
    """

    monkeypatch.setattr(
        api,
        "featurize",
        lambda **kwargs: pytest.fail("featurization ran before the mesh refusal"),
    )

    with pytest.raises(ValueError, match="requires attention_backend='xla'"):
        api.predict(
            seq=["ACD"],
            weights=tmp_path / "unused",
            mols=tmp_path,
            cp_devices=4,
            attention_backend="tokamax",
        )


def test_the_base_attention_ships_partitionable_so_nothing_is_resolved() -> None:
    """Why the fix above is a refusal and not a resolution.

    ``glu_backend``, ``triangle_backend`` and ``diffusion_attention_backend``
    ship fused, so a mesh has to resolve them or refuse the released
    configuration; this knob ships ``xla`` on both surfaces, so an omitted
    request is already the partitionable spelling and reaches the model
    untouched. That is the fact a resolution added here would contradict, and
    it has to be pinned on both surfaces, because a flip on either would
    silently reopen the door the refusal closes.

    The end-to-end proof on a real four-device mesh -- that the omitted knob
    is what the model receives -- is the forced probe in
    ``tests/test_boltz2_session.py``, beside the same assertion for
    ``glu_backend``.
    """

    assert (
        inspect.signature(api.predict).parameters["attention_backend"].default == "xla"
    )
    assert backend_module._RELEASED_COMPILE_DEFAULTS["attention_backend"] == "xla"
    # Omitting the knob under a mesh is accepted, which is the other half of
    # the refusal: only a named fused kernel is a request the run cannot honour.
    Boltz2Backend().validate_native_options({"cp_devices": 4})


def test_tokamax_attention_block_limits_and_invalid_inputs() -> None:
    import jax

    from foldjax.execution import matmul_precision_scope
    from foldjax.models._tokamax_attention import (
        pad_attention_to_block,
        should_pad_attention_to_block,
        tokamax_attention_block,
    )

    expected_blocks = {
        1: 16,
        15: 16,
        16: 16,
        17: 32,
        31: 32,
        32: 32,
        33: 64,
        37: 64,
        64: 64,
        65: 64,
        70: 64,
        128: 64,
        1003: 64,
        1024: 64,
    }
    for length, expected in expected_blocks.items():
        assert tokamax_attention_block(length) == expected

    for bad_length in (0, -1, -64, False, True):
        with pytest.raises(ValueError, match="positive"):
            tokamax_attention_block(bad_length)  # type: ignore[arg-type]

    q_ok = jnp.ones((1, 16, 2, 4), dtype=jnp.float32)
    q_bf16 = jnp.ones((1, 16, 2, 4), dtype=jnp.bfloat16)
    q_f16 = jnp.ones((1, 16, 2, 4), dtype=jnp.float16)

    assert not should_pad_attention_to_block(q_ok)
    for non_highest in (
        "default",
        "bfloat16",
        "high",
        "tensorfloat32",
        "TF32_TF32_F32",
        "TF32_TF32_F32_X3",
    ):
        with jax.default_matmul_precision(non_highest):
            assert not should_pad_attention_to_block(q_ok)
            assert not should_pad_attention_to_block(q_bf16)

    for highest_policy in ("highest", "float32", "F32_F32_F32"):
        with jax.default_matmul_precision(highest_policy):
            assert should_pad_attention_to_block(q_ok)
            assert not should_pad_attention_to_block(q_bf16)
            assert not should_pad_attention_to_block(q_f16)

    with matmul_precision_scope("highest"):
        assert should_pad_attention_to_block(q_ok)
        assert not should_pad_attention_to_block(q_bf16)
        with matmul_precision_scope("high"):
            assert not should_pad_attention_to_block(q_ok)

    for bad_block in (0, -8, False, True):
        with pytest.raises(ValueError, match="positive"):
            pad_attention_to_block(q_ok, q_ok, q_ok, block=bad_block)  # type: ignore[arg-type]

    # Rank below 3, mismatched ranks, mismatched k/v lengths, and empty axes fail.
    with pytest.raises(ValueError, match="rank"):
        pad_attention_to_block(jnp.ones((4, 4)), jnp.ones((4, 4)), jnp.ones((4, 4)))
    with pytest.raises(ValueError, match="rank"):
        pad_attention_to_block(q_ok, q_ok[0], q_ok[0])
    with pytest.raises(ValueError, match="sequence lengths must match"):
        pad_attention_to_block(q_ok, q_ok, jnp.ones((1, 8, 2, 4), dtype=jnp.float32))
    empty_seq = jnp.zeros((1, 0, 2, 4), dtype=jnp.float32)
    with pytest.raises(ValueError, match="positive"):
        pad_attention_to_block(empty_seq, q_ok, q_ok)
    with pytest.raises(ValueError, match="positive"):
        pad_attention_to_block(q_ok, empty_seq, empty_seq)

    # Malformed bias and mask ranks or sequence dimensions fail even when aligned.
    with pytest.raises(ValueError, match="bias"):
        pad_attention_to_block(q_ok, q_ok, q_ok, bias=jnp.zeros((1, 1, 2, 16, 16)))
    with pytest.raises(ValueError, match="bias"):
        pad_attention_to_block(q_ok, q_ok, q_ok, bias=jnp.zeros((1, 2, 5, 16)))
    with pytest.raises(ValueError, match="bias"):
        pad_attention_to_block(q_ok, q_ok, q_ok, bias=jnp.zeros((1, 2, 16, 5)))
    with pytest.raises(ValueError, match="mask"):
        pad_attention_to_block(
            q_ok, q_ok, q_ok, mask=jnp.ones((1, 1, 2, 16, 16), dtype=bool)
        )
    with pytest.raises(ValueError, match="mask"):
        pad_attention_to_block(
            q_ok, q_ok, q_ok, mask=jnp.ones((1, 2, 5, 16), dtype=bool)
        )
    with pytest.raises(ValueError, match="mask"):
        pad_attention_to_block(
            q_ok, q_ok, q_ok, mask=jnp.ones((1, 2, 16, 5), dtype=bool)
        )

    # Equal to the block limit returns the exact input objects with zero ops.
    qp, kp, vp, bp, mp, orig_q = pad_attention_to_block(
        q_ok, q_ok, q_ok, None, None, block=16
    )
    assert qp is q_ok and kp is q_ok and vp is q_ok
    assert bp is None and mp is None and orig_q == 16
    traced = jax.make_jaxpr(
        lambda x: pad_attention_to_block(x, x, x, block=16)[0]
    )(q_ok)
    assert traced.jaxpr.eqns == []

    # Scalar, 1-D, and broadcast-axis biases/masks pad without expanding unit axes.
    q5 = jnp.ones((2, 5, 3, 4), dtype=jnp.float32)
    b1d = jnp.ones((5,), dtype=jnp.float32)
    m_bcast = jnp.ones((2, 1, 5, 1), dtype=bool)
    _, _, _, bp1, mp1, n5 = pad_attention_to_block(
        q5, q5, q5, bias=b1d, mask=m_bcast, block=16
    )
    assert n5 == 5
    assert bp1 is not None and bp1.shape == (16,)
    assert mp1 is not None and mp1.shape == (2, 1, 16, 16)
    assert bool(jnp.all(mp1[:, :, :5, :5]))
    assert not bool(jnp.any(mp1[:, :, :5, 5:]))
    assert not bool(jnp.any(mp1[:, :, 5:, :]))

    _, _, _, bp_scalar, mp_scalar, _ = pad_attention_to_block(
        q5,
        q5,
        q5,
        bias=jnp.asarray(0.5, dtype=jnp.float32),
        mask=jnp.asarray(True),
        block=16,
    )
    assert bp_scalar is not None and bp_scalar.shape == ()
    assert mp_scalar is not None and mp_scalar.shape == (1, 1, 1, 16)
    assert bool(jnp.all(mp_scalar[..., :5]))
    assert not bool(jnp.any(mp_scalar[..., 5:]))


@pytest.mark.parametrize("seq_len", [1, 37, 70])
def test_boltz2_tokamax_ragged_padding_matches_unpadded_and_preserves_broadcast_bias(
    monkeypatch, seq_len: int
) -> None:
    import jax
    import tokamax

    rng = np.random.default_rng(100 + seq_len)
    batch, heads, dim = 2, 4, 16
    q_np = rng.normal(size=(batch, seq_len, heads, dim))
    k_np = rng.normal(size=(batch, seq_len, heads, dim))
    v_np = rng.normal(size=(batch, seq_len, heads, dim))
    # Batch axis is 1 (broadcast across multiplicity), head axis is 1 (broadcast).
    bias_np = rng.normal(size=(1, 1, seq_len, seq_len)) * 0.25
    raw_mask = rng.random((batch, seq_len)) > 0.35
    raw_mask[0, 0] = True
    # Second batch element has all cols masked -> empty-batch zero contract.
    raw_mask[1, :] = False
    mask = jnp.asarray(raw_mask, dtype=bool)
    scale = float(dim) ** -0.5

    real_dpa = tokamax.dot_product_attention
    recorded: list[
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], np.ndarray]
    ] = []

    def xla_spy(qa, ka, va, *, bias=None, mask=None, scale=None, implementation=None):
        del implementation
        recorded.append(
            (tuple(qa.shape), tuple(ka.shape), tuple(bias.shape), np.asarray(mask))
        )
        return real_dpa(
            qa, ka, va, bias=bias, mask=mask, scale=scale, implementation="xla"
        )

    def unpadded_expected(qa, ka, va, ba):
        ref_mask = mask[:, None, None, :]
        ref_empty = ~jnp.any(ref_mask, axis=-1, keepdims=True)
        ref_safe = ref_mask.at[..., 0].set(ref_mask[..., 0] | ref_empty[..., 0])
        res = real_dpa(
            qa, ka, va, bias=ba, mask=ref_safe, scale=scale, implementation="xla"
        ).astype(va.dtype)
        return jnp.where(ref_empty, jnp.zeros_like(res), res)

    q_f32 = jnp.asarray(q_np, dtype=jnp.float32)
    k_f32 = jnp.asarray(k_np, dtype=jnp.float32)
    v_f32 = jnp.asarray(v_np, dtype=jnp.float32)
    bias_f32 = jnp.asarray(bias_np, dtype=jnp.float32)
    q_bf16 = jnp.asarray(q_np, dtype=jnp.bfloat16)
    k_bf16 = jnp.asarray(k_np, dtype=jnp.bfloat16)
    v_bf16 = jnp.asarray(v_np, dtype=jnp.bfloat16)
    bias_bf16 = jnp.asarray(bias_np, dtype=jnp.bfloat16)

    with jax.default_matmul_precision("highest"):
        expected_f32_highest = unpadded_expected(q_f32, k_f32, v_f32, bias_f32)
    expected_f32_default = unpadded_expected(q_f32, k_f32, v_f32, bias_f32)
    expected_bf16 = unpadded_expected(q_bf16, k_bf16, v_bf16, bias_bf16)

    monkeypatch.setattr(tokamax, "dot_product_attention", xla_spy)

    # (a) bf16 (any precision) and float32 at default/high precision do NOT pad.
    no_pad_cases = [
        (q_bf16, k_bf16, v_bf16, bias_bf16, "triton", None, expected_bf16),
        (q_bf16, k_bf16, v_bf16, bias_bf16, "tokamax", "highest", expected_bf16),
        (q_f32, k_f32, v_f32, bias_f32, "tokamax", None, expected_f32_default),
        (q_f32, k_f32, v_f32, bias_f32, "tokamax", "default", expected_f32_default),
        (q_f32, k_f32, v_f32, bias_f32, "tokamax", "high", expected_f32_default),
    ]
    for qa, ka, va, ba, backend, policy, exp in no_pad_cases:
        recorded.clear()
        if policy is None:
            out_unpadded = tokamax_dot_product_attention(
                qa, ka, va, ba, mask, scale=scale, backend=backend
            )
        else:
            with jax.default_matmul_precision(policy):
                out_unpadded = tokamax_dot_product_attention(
                    qa, ka, va, ba, mask, scale=scale, backend=backend
                )
        assert len(recorded) == 1
        q_seen, k_seen, bias_seen, mask_seen = recorded[0]
        assert q_seen == (batch, seq_len, heads, dim)
        assert k_seen == (batch, seq_len, heads, dim)
        assert bias_seen == (1, 1, seq_len, seq_len)
        assert mask_seen.shape == (batch, 1, 1, seq_len)
        assert mask_seen[1, 0, 0, 0]
        assert out_unpadded.shape == (batch, seq_len, heads, dim)
        assert out_unpadded.dtype == qa.dtype
        np.testing.assert_array_equal(
            np.asarray(out_unpadded, dtype=np.float32),
            np.asarray(exp, dtype=np.float32),
        )

    # (b) float32 at highest precision pads ragged lengths and matches XLA ref.
    recorded.clear()
    with jax.default_matmul_precision("highest"):
        actual = tokamax_dot_product_attention(
            q_f32, k_f32, v_f32, bias_f32, mask, scale=scale, backend="tokamax"
        )

    expected_padded = 16 if seq_len == 1 else (64 if seq_len == 37 else 128)
    expected_bias_seq = 1 if seq_len == 1 else expected_padded
    assert len(recorded) == 1
    q_seen, k_seen, bias_seen, mask_seen = recorded[0]
    assert q_seen == (batch, expected_padded, heads, dim)
    assert k_seen == (batch, expected_padded, heads, dim)
    # Broadcast batch/head axes (and size-1 sequence axes at N=1) stay size 1.
    assert bias_seen == (1, 1, expected_bias_seq, expected_bias_seq)
    # Padded positions are strictly False, including on the empty batch row.
    assert not mask_seen[..., seq_len:].any()
    assert mask_seen[1, 0, 0, 0]
    assert actual.shape == (batch, seq_len, heads, dim)
    assert actual.dtype == jnp.float32
    np.testing.assert_array_equal(
        np.asarray(actual[1], dtype=np.float32),
        np.zeros((seq_len, heads, dim), dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(actual[0], dtype=np.float32),
        np.asarray(expected_f32_highest[0], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_boltz2_tokamax_aligned_is_noop_and_padded_queries_never_leak(
    monkeypatch,
) -> None:
    import jax
    import tokamax

    real_dpa = tokamax.dot_product_attention
    seen_shapes: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []

    def spy_aligned(qa, ka, va, *, bias=None, mask=None, scale=None, **kwargs):
        del kwargs
        seen_shapes.append((tuple(qa.shape), tuple(ka.shape), tuple(bias.shape)))
        return real_dpa(
            qa, ka, va, bias=bias, mask=mask, scale=scale, implementation="xla"
        )

    monkeypatch.setattr(tokamax, "dot_product_attention", spy_aligned)
    with jax.default_matmul_precision("highest"):
        for n_q, n_k in [(16, 16), (32, 128), (64, 64), (1024, 1024)]:
            q = jnp.ones((1, n_q, 2, 4), dtype=jnp.float32)
            k = jnp.ones((1, n_k, 2, 4), dtype=jnp.float32)
            bias = jnp.zeros((1, 2, n_q, n_k), dtype=jnp.float32)
            mask = jnp.ones((1, n_k), dtype=bool)
            seen_shapes.clear()
            out = tokamax_dot_product_attention(
                q, k, v=k, bias=bias, mask=mask, scale=0.5, backend="tokamax"
            )
            assert seen_shapes == [((1, n_q, 2, 4), (1, n_k, 2, 4), (1, 2, n_q, n_k))]
            assert out.shape == (1, n_q, 2, 4)

        # Aligned query (16) with ragged cols (37): pads cols to 64, no query slice.
        seen_shapes.clear()
        q16 = jnp.ones((1, 16, 2, 4), dtype=jnp.float32)
        k37 = jnp.ones((1, 37, 2, 4), dtype=jnp.float32)
        b_rect = jnp.zeros((1, 2, 16, 37), dtype=jnp.float32)
        m37_cols = jnp.ones((1, 37), dtype=bool)
        out_rect = tokamax_dot_product_attention(
            q16, k37, v=k37, bias=b_rect, mask=m37_cols, scale=0.5, backend="tokamax"
        )
        assert seen_shapes == [((1, 16, 2, 4), (1, 64, 2, 4), (1, 2, 16, 64))]
        assert out_rect.shape == (1, 16, 2, 4)

    def unpadded_ref(qa, ka, va, ba, ma):
        cm = ma[:, None, None, :].astype(bool)
        emp = ~jnp.any(cm, axis=-1, keepdims=True)
        sm = cm.at[..., 0].set(cm[..., 0] | emp[..., 0])
        res = real_dpa(qa, ka, va, bias=ba, mask=sm, scale=0.5, implementation="xla")
        return jnp.where(emp, jnp.zeros_like(res), res.astype(va.dtype))

    # Aligned highest-precision and ragged bf16/TF32 jaxprs emit zero pad/slice.
    q64 = jnp.ones((1, 64, 2, 4), dtype=jnp.float32)
    b64 = jnp.zeros((1, 2, 64, 64), dtype=jnp.float32)
    m64 = jnp.ones((1, 64), dtype=bool)
    with jax.default_matmul_precision("highest"):
        wrapped_jaxpr = jax.make_jaxpr(
            lambda qa, ka, va, ba, ma: tokamax_dot_product_attention(
                qa, ka, va, ba, ma, scale=0.5, backend="tokamax"
            )
        )(q64, q64, q64, b64, m64)
        ref_jaxpr = jax.make_jaxpr(unpadded_ref)(q64, q64, q64, b64, m64)
    primitives = [eqn.primitive.name for eqn in wrapped_jaxpr.jaxpr.eqns]
    ref_primitives = [eqn.primitive.name for eqn in ref_jaxpr.jaxpr.eqns]
    assert "pad" not in primitives
    assert primitives == ref_primitives

    q37_bf16 = jnp.ones((1, 37, 2, 4), dtype=jnp.bfloat16)
    b37_bf16 = jnp.zeros((1, 2, 37, 37), dtype=jnp.bfloat16)
    m37_1 = jnp.ones((1, 37), dtype=bool)
    with jax.default_matmul_precision("high"):
        bf16_jaxpr = jax.make_jaxpr(
            lambda qa, ka, va, ba, ma: tokamax_dot_product_attention(
                qa, ka, va, ba, ma, scale=0.5, backend="triton"
            )
        )(q37_bf16, q37_bf16, q37_bf16, b37_bf16, m37_1)
        bf16_ref_jaxpr = jax.make_jaxpr(unpadded_ref)(
            q37_bf16, q37_bf16, q37_bf16, b37_bf16, m37_1
        )
    assert [e.primitive.name for e in bf16_jaxpr.jaxpr.eqns] == [
        e.primitive.name for e in bf16_ref_jaxpr.jaxpr.eqns
    ]

    # Poison padded query rows inside tokamax and verify they never leak out.
    def poison_padded_queries(qa, ka, va, **kwargs):
        del ka, kwargs
        assert qa.shape[1] == 64
        clean = jnp.full((qa.shape[0], 37, qa.shape[2], va.shape[3]), 3.0, va.dtype)
        poison = jnp.full(
            (qa.shape[0], 27, qa.shape[2], va.shape[3]), jnp.nan, va.dtype
        )
        return jnp.concatenate([clean, poison], axis=1)

    monkeypatch.setattr(tokamax, "dot_product_attention", poison_padded_queries)
    q37 = jnp.ones((2, 37, 2, 4), dtype=jnp.float32)
    b37 = jnp.zeros((1, 2, 37, 37), dtype=jnp.float32)
    m37 = jnp.ones((2, 37), dtype=bool)
    with jax.default_matmul_precision("highest"):
        out37 = tokamax_dot_product_attention(
            q37, q37, q37, b37, m37, scale=0.5, backend="tokamax"
        )
    assert out37.shape == (2, 37, 2, 4)
    assert bool(jnp.isfinite(out37).all())
    np.testing.assert_array_equal(
        np.asarray(out37), np.full((2, 37, 2, 4), 3.0, dtype=np.float32)
    )

    # Trace-time reading inside jax.jit re-traces between high and highest.
    traced_q_lens: list[int] = []

    def trace_spy(qa, ka, va, *, bias=None, mask=None, scale=None, **kwargs):
        del kwargs
        traced_q_lens.append(qa.shape[1])
        return real_dpa(
            qa, ka, va, bias=bias, mask=mask, scale=scale, implementation="xla"
        )

    monkeypatch.setattr(tokamax, "dot_product_attention", trace_spy)
    jitted = jax.jit(
        lambda qa, ka, va, ba, ma: tokamax_dot_product_attention(
            qa, ka, va, ba, ma, scale=0.5, backend="tokamax"
        )
    )
    with jax.default_matmul_precision("high"):
        assert jitted(q37, q37, q37, b37, m37).shape == (2, 37, 2, 4)
    with jax.default_matmul_precision("highest"):
        assert jitted(q37, q37, q37, b37, m37).shape == (2, 37, 2, 4)
    assert traced_q_lens == [37, 64]
