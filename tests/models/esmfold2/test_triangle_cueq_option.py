"""Opt-in cuEquivariance triangle multiplication for ESMFold2's pair stacks."""

from __future__ import annotations

import dataclasses
import json
import sys
import types
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.backends import esmfold2 as backend_module
from foldjax.backends.esmfold2 import ESMFold2Backend
from foldjax.models._jit_pool import BoundedJitPool
from foldjax.models.esmfold2 import inference
from foldjax.models.esmfold2.data.features import build_features
from foldjax.models.esmfold2.models import model as structure_model
from foldjax.models.esmfold2.models import trunk
from foldjax.registry import get_backend
from foldjax.schema import PredictionRequest

ATOL = 1e-5


class _FakeCueq:
    class TriMulPrecision:
        DEFAULT = "default"
        TF32 = "tf32"
        IEEE = "ieee"
        TF32x3 = "tf32x3"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def triangle_multiplicative_update(self, **kwargs: Any) -> jax.Array:
        self.calls.append(kwargs)
        x = kwargs["x"]
        return jnp.full((*x.shape[:-1], kwargs["p_out_weight"].shape[0]), 7.0, x.dtype)


@pytest.fixture
def fake_cueq(monkeypatch: pytest.MonkeyPatch) -> _FakeCueq:
    fake = _FakeCueq()
    monkeypatch.setattr("foldjax.models._cueq.load_cueq", lambda: fake, raising=True)
    return fake


def _load_cpu_cuex(monkeypatch: pytest.MonkeyPatch) -> Any:
    if (
        "cuequivariance_jax" not in sys.modules
        and "cuequivariance_ops_jax" not in sys.modules
    ):
        monkeypatch.setitem(sys.modules, "cuequivariance_ops_jax", None)
    return pytest.importorskip("cuequivariance_jax")


@pytest.fixture
def cueq_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    cuex = _load_cpu_cuex(monkeypatch)
    seen: list[dict[str, Any]] = []

    def update(**kwargs: Any) -> jax.Array:
        assert kwargs.get("fallback") is False
        seen.append(kwargs)
        return cuex.triangle_multiplicative_update(**{**kwargs, "fallback": True})

    proxy = types.SimpleNamespace(
        TriMulPrecision=cuex.TriMulPrecision, triangle_multiplicative_update=update
    )
    monkeypatch.setattr("foldjax.models._cueq.load_cueq", lambda: proxy)
    return seen


def _arr(rng: np.random.Generator, *shape: int, scale: float = 1.0) -> jnp.ndarray:
    raw = rng.standard_normal(shape).astype(np.float32) * np.float32(scale)
    return jnp.asarray(raw)


def _tri_params(
    rng: np.random.Generator, prefix: str, width: int, *, with_biases: bool = False
) -> dict[str, jnp.ndarray]:
    e = f"{prefix}._engine" if prefix else "_engine"
    s = float(width**-0.5)
    out = {
        f"{e}.norm_start.weight": _arr(rng, width),
        f"{e}.norm_start.bias": _arr(rng, width, scale=0.1),
        f"{e}.proj_bundle.weight": _arr(rng, 4 * width, width, scale=s),
        f"{e}.norm_mix.weight": _arr(rng, width),
        f"{e}.norm_mix.bias": _arr(rng, width, scale=0.1),
        f"{e}.proj_emit.weight": _arr(rng, width, width, scale=s),
        f"{e}.proj_gate.weight": _arr(rng, width, width, scale=s),
    }
    if with_biases:
        out[f"{e}.proj_bundle.bias"] = _arr(rng, 4 * width, scale=0.1)
        out[f"{e}.proj_emit.bias"] = _arr(rng, width, scale=0.1)
        out[f"{e}.proj_gate.bias"] = _arr(rng, width, scale=0.1)
    return out


def _trunk_params(rng: np.random.Generator, width: int = 32) -> dict[str, jnp.ndarray]:
    s = float(width**-0.5)
    return {
        **_tri_params(rng, "blocks.0.tri_mul_out", width),
        **_tri_params(rng, "blocks.0.tri_mul_in", width),
        "blocks.0.pair_transition.norm.weight": _arr(rng, width),
        "blocks.0.pair_transition.norm.bias": _arr(rng, width, scale=0.1),
        "blocks.0.pair_transition.ffn.w12.weight": _arr(rng, 4 * width, width, scale=s),
        "blocks.0.pair_transition.ffn.w3.weight": _arr(rng, width, 2 * width, scale=s),
    }


def _mask(extent: int, *, batch: int = 1) -> jnp.ndarray:
    valid = np.ones((batch, extent), dtype=np.float32)
    if extent >= 3:
        valid[:, -1] = 0.0
    pair_mask = valid[:, :, None] * valid[:, None, :]
    if extent >= 4:
        pair_mask[:, 0, 1] = 0.0
    return jnp.asarray(pair_mask)


def _request(tmp_path: Any, **options: Any) -> PredictionRequest:
    job = tmp_path / "job.json"
    if not job.exists():
        doc = {"entities": [{"type": "protein", "id": "A", "sequence": "ACDE"}]}
        job.write_text(json.dumps(doc), encoding="utf-8")
    weights = tmp_path / "weights"
    weights.mkdir(exist_ok=True)
    return PredictionRequest(
        model="esmfold2",
        input=job,
        input_format="foldjax",
        weights=weights,
        output_dir=tmp_path / "out",
        options=options,
    )


@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize("with_biases", [False, True])
@pytest.mark.parametrize(("width", "extent"), [(32, 11), (256, 6)])
def test_cueq_packing_matches_dense_float32_in_both_directions(
    cueq_calls: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    outgoing: bool,
    with_biases: bool,
    width: int,
    extent: int,
) -> None:
    cuex = _load_cpu_cuex(monkeypatch)
    rng = np.random.default_rng(100 + width + extent + int(outgoing) + 2 * with_biases)
    pair, mask = _arr(rng, 1, extent, extent, width), _mask(extent)
    params = _tri_params(rng, "tri_mul", width, with_biases=with_biases)
    kw: dict[str, Any] = {"outgoing": outgoing, "mask": mask}

    with jax.default_matmul_precision("float32"):
        expected = trunk.triangle_multiplicative(
            pair, params, "tri_mul", triangle_backend="xla", **kw
        )
        assert cueq_calls == []
        actual = trunk.triangle_multiplicative(
            pair, params, "tri_mul", triangle_backend="cueq", **kw
        )

    (call,) = cueq_calls
    assert call["direction"] == ("outgoing" if outgoing else "incoming")
    assert call["precision"] == cuex.TriMulPrecision.IEEE
    assert call["p_in_weight"].shape == call["g_in_weight"].shape == (2 * width, width)
    assert call["p_out_weight"].shape == call["g_out_weight"].shape == (width, width)
    bias_shapes = tuple(
        None if call[k] is None else call[k].shape
        for k in ("p_in_bias", "g_in_bias", "p_out_bias", "g_out_bias")
    )
    wanted = ((2 * width,), (2 * width,), (width,), (width,))
    assert bias_shapes == (wanted if with_biases else (None,) * 4)
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=ATOL)


def test_swapping_packed_halves_and_4d_mask_broadcast(
    cueq_calls: list[dict[str, Any]],
) -> None:
    rng = np.random.default_rng(7)
    width, extent = 32, 6
    pair, bcast_mask = _arr(rng, 2, extent, extent, width), _mask(extent, batch=1)
    params = _tri_params(rng, "", width)
    bundle = params["_engine.proj_bundle.weight"]
    h, q = bundle.shape[0] // 2, bundle.shape[0] // 4

    with jax.default_matmul_precision("float32"):
        exp_nomask = trunk.triangle_multiplicative(pair, params, "", outgoing=True)
        act_nomask = trunk.triangle_multiplicative(
            pair, params, "", outgoing=True, triangle_backend="cueq"
        )
        exp_bcast = trunk.triangle_multiplicative(
            pair, params, "", outgoing=False, mask=bcast_mask
        )
        act_bcast = trunk.triangle_multiplicative(
            pair, params, "", outgoing=False, mask=bcast_mask, triangle_backend="cueq"
        )
        swapped = {
            **params,
            "_engine.proj_bundle.weight": jnp.concatenate(
                [bundle[q:h], bundle[:q], bundle[h + q :], bundle[h : h + q]]
            ),
        }
        wrong = trunk.triangle_multiplicative(
            pair, swapped, "", outgoing=True, triangle_backend="cueq"
        )

    assert [c["mask"].shape for c in cueq_calls[:2]] == [(2, extent, extent)] * 2
    np.testing.assert_allclose(act_nomask, exp_nomask, atol=ATOL, rtol=ATOL)
    np.testing.assert_allclose(act_bcast, exp_bcast, atol=ATOL, rtol=ATOL)
    assert float(np.max(np.abs(np.asarray(wrong) - np.asarray(exp_nomask)))) > 0.1


@pytest.mark.parametrize("extent", [12, 64, 65, 128])
def test_native_autocast_bfloat16_replaces_both_dense_and_streamed_routes(
    fake_cueq: _FakeCueq, monkeypatch: pytest.MonkeyPatch, extent: int
) -> None:
    rng = np.random.default_rng(200 + extent)
    pair_bf16 = _arr(rng, 1, extent, extent, 32).astype(jnp.bfloat16)
    params = _trunk_params(rng)
    monkeypatch.setattr(
        trunk,
        "_autocast_triangle_streamed",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("streamed reached")),
    )
    sentinel = (jnp.ones((1,), jnp.bfloat16), jnp.ones((1,), jnp.bfloat16))
    slot = trunk.LentBuffers(sentinel)
    ws = lambda b: trunk.trunk_workspace(  # noqa: E731
        pair_bf16, params, native_autocast=True, triangle_backend=b
    )
    assert ws("cueq") is None
    if extent == 128:
        assert len(ws("xla")) == 2

    with jax.default_matmul_precision("highest"):
        actual = trunk.folding_trunk(
            pair_bf16,
            params,
            n_layers=1,
            mask=_mask(extent),
            native_autocast=True,
            workspace=slot,
            triangle_backend="cueq",
        )

    assert slot.buffers is sentinel
    assert actual.dtype == jnp.bfloat16 and actual.shape == pair_bf16.shape
    assert [c["direction"] for c in fake_cueq.calls] == ["outgoing", "incoming"]
    norm_names = ("norm_in_weight", "norm_in_bias", "norm_out_weight", "norm_out_bias")
    proj_names = ("p_in_weight", "g_in_weight", "p_out_weight", "g_out_weight")
    for call in fake_cueq.calls:
        assert call["fallback"] is False
        assert call["x"].dtype == jnp.bfloat16 and call["mask"].dtype == jnp.bfloat16
        assert call["precision"] == fake_cueq.TriMulPrecision.DEFAULT
        assert all(call[n].dtype == jnp.float32 for n in norm_names)
        assert all(call[w].dtype == jnp.bfloat16 for w in proj_names)


def test_float32_pair_autocast_keeps_xla_route(fake_cueq: _FakeCueq) -> None:
    rng = np.random.default_rng(33)
    pair_f32, mask, params = _arr(rng, 1, 10, 10, 32), _mask(10), _trunk_params(rng)

    expected = trunk.folding_trunk(
        pair_f32, params, n_layers=1, mask=mask, native_autocast=True
    )
    with trunk._triangle_backend_scope("cueq"):
        actual = trunk.folding_trunk(
            pair_f32, params, n_layers=1, mask=mask, native_autocast=True
        )
    assert fake_cueq.calls == []
    np.testing.assert_array_equal(actual, expected)


def test_cueq_triangle_minimal_guards_and_scope(
    fake_cueq: _FakeCueq, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(77)
    pair = jnp.zeros((1, 4, 4, 32), dtype=jnp.float32)
    params = _tri_params(rng, "tri_mul", 32)
    run = lambda z=pair, p=params, b="cueq": trunk.triangle_multiplicative(  # noqa: E731
        z, p, "tri_mul", outgoing=True, triangle_backend=b
    )

    with pytest.raises(ValueError, match="requires bfloat16 or float32"):
        run(pair.astype(jnp.float16))
    missing = {k: v for k, v in params.items() if not k.endswith("norm_mix.bias")}
    with pytest.raises(ValueError, match="requires affine layer norms"):
        run(p=missing)
    for non_4d in ((4, 4, 32), (1, 2, 4, 4, 32)):
        with pytest.raises(ValueError):
            run(jnp.zeros(non_4d))
    for invalid in ("", "pallas", 0, True):
        with pytest.raises(ValueError, match="triangle_backend must be one of"):
            run(b=invalid)

    with pytest.raises(RuntimeError, match="boom"):
        with trunk._triangle_backend_scope("cueq"):
            assert trunk._resolve_triangle_backend(None) == "cueq"
            run(b="xla")
            assert fake_cueq.calls == []
            raise RuntimeError("boom")
    assert trunk._resolve_triangle_backend(None) == "xla"

    monkeypatch.setattr(trunk, "cp_mesh", lambda: types.SimpleNamespace())
    with pytest.raises(ValueError, match="cannot be partitioned"):
        run()


def test_adapter_options_neutral_mapping_and_refusals(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_esmfold2_backend import _stub_prediction

    assert (
        backend_module._TRIANGLE_BACKENDS == trunk.TRIANGLE_BACKENDS == ("xla", "cueq")
    )
    assert "triangle_backend" in ESMFold2Backend.native_options
    assert "triangle_backend" in ESMFold2Backend.compile_options
    assert backend_module._FIXED_COMPILE_DEFAULTS["triangle_backend"] == "xla"

    backend = get_backend("esmfold2")
    req = lambda **opts: _request(tmp_path, **opts)  # noqa: E731
    pred = lambda **kw: inference.predict(  # noqa: E731
        jax.random.key(0),
        {},
        dataclasses.replace,
        **kw,  # type: ignore[arg-type]
    )

    for neutral, expected in (("auto", "xla"), ("xla", "xla"), ("cueq", "cueq")):
        assert (
            backend.apply_sampling(req(triangle_kernel=neutral))["triangle_backend"]
            == expected
        )
    with pytest.raises(
        ValueError, match="does not support triangle_kernel='cueq-full'"
    ):
        backend.apply_sampling(req(triangle_kernel="cueq-full"))
    for bad in ("", "auto", 0, True):
        with pytest.raises(ValueError, match="triangle_backend must be one of"):
            backend.validate_native_options({"triangle_backend": bad})
        with pytest.raises(ValueError, match="triangle_backend must be one of"):
            pred(triangle_backend=bad)

    monkeypatch.setattr("foldjax.models._pallas_pair.gpu_process", lambda: False)
    with pytest.raises(ValueError, match="needs a GPU") as excinfo:
        backend.validate_request(req(triangle_kernel="cueq"))
    assert "platform 'cpu'" in str(excinfo.value)
    with pytest.raises(ValueError, match="needs a GPU"):
        pred(triangle_backend="cueq")

    monkeypatch.setattr(
        backend_module,
        "_gpu_process",
        lambda: (_ for _ in ()).throw(AssertionError("device touched")),
    )
    for safe_opts in ({}, {"triangle_backend": "xla"}, {"triangle_kernel": "auto"}):
        backend.validate_request(req(**safe_opts))

    monkeypatch.setattr(backend_module, "_gpu_process", lambda: True)
    monkeypatch.setattr("foldjax.models._pallas_pair.gpu_process", lambda: True)
    assert backend_module._strict_cp_devices(True) == 1
    assert backend_module._strict_cp_devices("bad") == 1
    assert backend_module._strict_cp_devices("2") == 2
    backend.validate_native_options({"triangle_backend": "cueq", "cp_devices": 1})
    with pytest.raises(ValueError, match="cp_devices must be positive"):
        backend.predict(req(triangle_backend="cueq", cp_devices=0))
    with pytest.raises(ValueError, match="cannot be partitioned"):
        backend.validate_request(req(triangle_kernel="cueq", cp_devices=2))
    with pytest.raises(ValueError, match="cannot be partitioned"):
        pred(cp_shards=2, triangle_backend="cueq")

    seen_d, res_d = _stub_prediction(tmp_path / "d", monkeypatch, {})
    assert "triangle_backend" not in seen_d
    assert "triangle_backend" not in res_d.raw["overrides"]
    seen_c, res_c = _stub_prediction(
        tmp_path / "c", monkeypatch, {"triangle_kernel": "cueq"}
    )
    assert (
        seen_c["triangle_backend"]
        == res_c.raw["overrides"]["triangle_backend"]
        == "cueq"
    )


def test_cache_profile_and_compiled_predict_namespace_identity(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = ESMFold2Backend()
    prof = lambda **opts: backend.cache_profile(_request(tmp_path, **opts))  # noqa: E731
    omitted_prof, cueq_prof = prof(), prof(triangle_backend="cueq")
    assert "triangle_backend" not in omitted_prof
    assert omitted_prof == prof(triangle_backend="xla") == prof(triangle_kernel="auto")
    assert cueq_prof == prof(triangle_kernel="cueq") != omitted_prof
    assert cueq_prof["triangle_backend"] == "cueq"

    monkeypatch.setattr("foldjax.models._pallas_pair.gpu_process", lambda: True)
    features = build_features([("AG", "A", 0, 0)])
    settings = dataclasses.replace(
        structure_model.ModelSettings(), trunk_n_layers=0, coda_n_layers=0
    )
    bond_name = "to" + "ken_bonds.weight"
    loaded = inference.LoadedModel(
        {bond_name: jnp.ones((settings.d_pair, 1), jnp.bfloat16)}, settings
    )
    lm_embed = jnp.zeros((1, 2, 8), dtype=jnp.float32)
    traces: list[str] = []

    def tiny_predict(rng_arg: Any, model_features: Any, parameters: Any, **kwargs: Any):
        del rng_arg, parameters, kwargs
        resolved = trunk._resolve_triangle_backend(None)
        traces.append(resolved)
        return {
            "probe": model_features["asym_id"].astype(jnp.int32)
            + int(resolved == "cueq")
        }

    monkeypatch.setattr(structure_model, "predict", tiny_predict)
    pool = BoundedJitPool(
        inference._run, static_argnames=inference._COMPILED_PREDICT_STATIC_ARGNAMES
    )
    monkeypatch.setattr(inference, "_compiled_predict_pool", pool)
    inference._compiled_predict_factory.cache_clear()

    def run_pred(**kw: Any) -> dict[str, jnp.ndarray]:
        return inference.predict(
            jax.random.key(0),
            features,
            loaded,
            precomputed_lm_embedding=lm_embed,
            **kw,
        )

    try:
        out_omitted = run_pred()
        out_xla = run_pred(triangle_backend="xla")
        jax.block_until_ready((out_omitted, out_xla))
        assert traces == ["xla"] and pool._entry_count() == 1  # noqa: SLF001
        np.testing.assert_array_equal(out_omitted["probe"], out_xla["probe"])

        out_cueq = run_pred(triangle_backend="cueq")
        jax.block_until_ready(out_cueq)
        assert traces == ["xla", "cueq"] and pool._entry_count() == 2  # noqa: SLF001
        assert int(out_cueq["probe"][0, 0]) == 1
    finally:
        pool.clear_cache()
        inference._compiled_predict_factory.cache_clear()


@pytest.mark.parametrize("extent", [12, 65])
def test_default_omitted_never_calls_cueq_and_matches_explicit_xla(
    monkeypatch: pytest.MonkeyPatch, extent: int
) -> None:
    monkeypatch.setattr(
        "foldjax.models._cueq.fused_triangle_multiplication",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("cueq called")),
        raising=True,
    )
    rng = np.random.default_rng(90 + extent)
    pair = _arr(rng, 1, extent, extent, 32).astype(jnp.bfloat16)
    mask, params = _mask(extent), _trunk_params(rng)
    fn = lambda **kw: jax.jit(  # noqa: E731
        lambda z: trunk.folding_trunk(
            z, params, n_layers=1, mask=mask, native_autocast=True, **kw
        )
    )
    omitted_fn, explicit_fn = fn(), fn(triangle_backend="xla")
    assert str(omitted_fn.lower(pair).compiler_ir(dialect="stablehlo")) == str(
        explicit_fn.lower(pair).compiler_ir(dialect="stablehlo")
    )
    assert (
        np.asarray(omitted_fn(pair)).tobytes()
        == np.asarray(explicit_fn(pair)).tobytes()
    )
