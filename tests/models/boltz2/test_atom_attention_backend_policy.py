"""Stage-scoped backend policy for the Boltz-2 BF16 atom attention pilot."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

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


def test_backend_accepts_explicit_inherited_fused_backend_with_cp() -> None:
    Boltz2Backend().validate_native_options(
        {
            "attention_backend": "tokamax",
            "trunk_atom_attention_backend": "tokamax",
            "cp_devices": 2,
        }
    )
