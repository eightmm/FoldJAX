"""cuEquivariance and Pallas triangle multiplication for ESMFold2's pair stacks."""

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
from foldjax.models import _pallas_pair
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


@pytest.fixture
def fake_pallas(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    def update(x: jax.Array, **kwargs: Any) -> jax.Array:
        seen.append({"x": x, **kwargs})
        return jnp.full((*x.shape[:-1], kwargs["p_out"][0].shape[0]), 9.0, x.dtype)

    monkeypatch.setattr(
        _pallas_pair, "triangle_multiplication", update, raising=True
    )
    return seen


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


@pytest.mark.parametrize("outgoing", [True, False])
def test_pallas_dispatch_and_interpret_parity(
    fake_cueq: _FakeCueq,
    monkeypatch: pytest.MonkeyPatch,
    outgoing: bool,
) -> None:
    rng = np.random.default_rng(40 + int(outgoing))
    width, extent = 32, 11
    pair = _arr(rng, 1, extent, extent, width).astype(jnp.bfloat16)
    mask = _mask(extent)
    params = _tri_params(rng, "tri_mul", width)

    monkeypatch.setattr(_pallas_pair, "INTERPRET", True)
    seen_pallas: list[str] = []
    real_pallas = _pallas_pair.triangle_multiplication

    def counted_pallas(*args: Any, **kwargs: Any) -> jax.Array:
        seen_pallas.append(kwargs["direction"])
        return real_pallas(*args, **kwargs)

    monkeypatch.setattr(_pallas_pair, "triangle_multiplication", counted_pallas)
    actual = trunk.triangle_multiplicative(
        pair,
        params,
        "tri_mul",
        outgoing=outgoing,
        mask=mask,
        native_autocast=True,
        triangle_backend="pallas",
    )
    expected = trunk.triangle_multiplicative(
        pair,
        params,
        "tri_mul",
        outgoing=outgoing,
        mask=mask,
        native_autocast=True,
        triangle_backend="xla",
    )
    assert seen_pallas == ["outgoing" if outgoing else "incoming"]
    assert fake_cueq.calls == []
    assert actual.shape == expected.shape and actual.dtype == jnp.bfloat16
    act_f32, exp_f32 = np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    rel = float(np.max(np.abs(act_f32 - exp_f32)) / np.max(np.abs(exp_f32)))
    assert rel < 3e-2

    cueq_out = trunk.triangle_multiplicative(
        pair,
        params,
        "tri_mul",
        outgoing=outgoing,
        mask=mask,
        native_autocast=True,
        triangle_backend="cueq",
    )
    assert len(fake_cueq.calls) == 1
    assert seen_pallas == ["outgoing" if outgoing else "incoming"]
    assert cueq_out.dtype == jnp.bfloat16

    biased = _tri_params(rng, "tri_mul", width, with_biases=True)
    with pytest.raises(ValueError, match="bias-free"):
        trunk.triangle_multiplicative(
            pair,
            biased,
            "tri_mul",
            outgoing=outgoing,
            mask=mask,
            native_autocast=True,
            triangle_backend="pallas",
        )


@pytest.mark.parametrize("backend", ["cueq", "pallas"])
@pytest.mark.parametrize("extent", [12, 64, 65, 128])
def test_native_autocast_bfloat16_replaces_both_dense_and_streamed_routes(
    fake_cueq: _FakeCueq,
    fake_pallas: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    extent: int,
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
    assert ws(backend) is None
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
            triangle_backend=backend,
        )

    assert slot.buffers is sentinel
    assert actual.dtype == jnp.bfloat16 and actual.shape == pair_bf16.shape
    if backend == "cueq":
        assert fake_pallas == []
        assert [c["direction"] for c in fake_cueq.calls] == ["outgoing", "incoming"]
        norm_names = (
            "norm_in_weight",
            "norm_in_bias",
            "norm_out_weight",
            "norm_out_bias",
        )
        proj_names = ("p_in_weight", "g_in_weight", "p_out_weight", "g_out_weight")
        for call in fake_cueq.calls:
            assert call["fallback"] is False
            assert call["x"].dtype == call["mask"].dtype == jnp.bfloat16
            assert call["precision"] == fake_cueq.TriMulPrecision.DEFAULT
            assert all(call[n].dtype == jnp.float32 for n in norm_names)
            assert all(call[w].dtype == jnp.bfloat16 for w in proj_names)
    else:
        assert fake_cueq.calls == []
        assert [c["direction"] for c in fake_pallas] == ["outgoing", "incoming"]
        for call in fake_pallas:
            assert call["x"].dtype == call["mask"].dtype == jnp.bfloat16
            assert all(
                t.dtype == jnp.float32
                for n in ("norm_in", "norm_out")
                for t in call[n]
            )
            assert all(
                call[w][0].dtype == jnp.bfloat16 and call[w][1] is None
                for w in ("p_in", "g_in", "p_out", "g_out")
            )


def test_float32_pair_autocast_keeps_xla_route(
    fake_cueq: _FakeCueq, fake_pallas: list[dict[str, Any]]
) -> None:
    rng = np.random.default_rng(33)
    pair_f32, mask, params = _arr(rng, 1, 10, 10, 32), _mask(10), _trunk_params(rng)

    expected = trunk.folding_trunk(
        pair_f32, params, n_layers=1, mask=mask, native_autocast=True
    )
    for backend in ("cueq", "pallas"):
        with trunk._triangle_backend_scope(backend):
            actual = trunk.folding_trunk(
                pair_f32, params, n_layers=1, mask=mask, native_autocast=True
            )
        assert fake_cueq.calls == [] and fake_pallas == []
        np.testing.assert_array_equal(actual, expected)


def test_cueq_triangle_minimal_guards_and_scope(
    fake_cueq: _FakeCueq,
    fake_pallas: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(77)
    pair = jnp.zeros((1, 4, 4, 32), dtype=jnp.float32)
    params = _tri_params(rng, "tri_mul", 32)
    run = lambda z=pair, p=params, b="cueq": trunk.triangle_multiplicative(  # noqa: E731
        z, p, "tri_mul", outgoing=True, triangle_backend=b
    )

    for bad_kernel in ("", "xla"):
        with pytest.raises(ValueError, match="unsupported fused triangle kernel"):
            trunk._cueq_triangle(
                pair,
                params,
                "tri_mul",
                outgoing=True,
                mask=None,
                eps=1e-5,
                kernel=bad_kernel,
            )

    for fused_backend in ("cueq", "pallas"):
        with pytest.raises(ValueError, match="requires bfloat16 or float32"):
            run(pair.astype(jnp.float16), b=fused_backend)
        missing = {k: v for k, v in params.items() if not k.endswith("norm_mix.bias")}
        with pytest.raises(
            ValueError,
            match=f"triangle_backend={fused_backend!r} requires affine layer norms",
        ):
            run(p=missing, b=fused_backend)
        for non_4d in ((4, 4, 32), (1, 2, 4, 4, 32)):
            with pytest.raises(ValueError):
                run(jnp.zeros(non_4d), b=fused_backend)

    for invalid in ("", "triton", "cueq-pallas", 0, True):
        with pytest.raises(ValueError, match="triangle_backend must be one of"):
            run(b=invalid)

    for fused_backend in ("cueq", "pallas"):
        with pytest.raises(RuntimeError, match="boom"):
            with trunk._triangle_backend_scope(fused_backend):
                assert trunk._resolve_triangle_backend(None) == fused_backend
                run(b="xla")
                assert fake_cueq.calls == [] and fake_pallas == []
                raise RuntimeError("boom")
        assert trunk._resolve_triangle_backend(None) == "xla"

    monkeypatch.setattr(trunk, "cp_mesh", lambda: types.SimpleNamespace())
    for fused_backend, label in (("cueq", "cuEquivariance"), ("pallas", "Pallas")):
        with pytest.raises(ValueError, match=f"{label}.*cannot be partitioned"):
            run(b=fused_backend)


def test_adapter_options_neutral_mapping_and_refusals(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_esmfold2_backend import _stub_prediction

    assert (
        backend_module._TRIANGLE_BACKENDS
        == trunk.TRIANGLE_BACKENDS
        == ("xla", "cueq", "pallas")
    )
    assert "triangle_backend" in ESMFold2Backend.native_options
    assert "triangle_backend" in ESMFold2Backend.compile_options
    assert backend_module._FIXED_COMPILE_DEFAULTS["triangle_backend"] == "xla"
    # The refusal helpers take unvalidated values: anything not a fused name
    # is a ValueError under a mesh and passes the GPU check to validation.
    for bad in ("triton", [], 3):
        with pytest.raises(ValueError, match="requires triangle_backend='xla'"):
            backend_module._refuse_cp_triangle_backend(bad, 2)
        backend_module._refuse_gpu_only_kernels({"triangle_backend": bad})
    backend_module._refuse_cp_triangle_backend("xla", 2)

    backend = get_backend("esmfold2")
    req = lambda **opts: _request(tmp_path, **opts)  # noqa: E731
    pred = lambda **kw: inference.predict(  # noqa: E731
        jax.random.key(0),
        {},
        dataclasses.replace,
        **kw,  # type: ignore[arg-type]
    )

    # `auto` passes nothing: it realises what an omitted option does.
    assert "triangle_backend" not in backend.apply_sampling(req(triangle_kernel="auto"))
    for neutral, expected in (("xla", "xla"), ("cueq", "cueq")):
        assert (
            backend.apply_sampling(req(triangle_kernel=neutral))["triangle_backend"]
            == expected
        )
    with pytest.raises(
        ValueError, match="does not support triangle_kernel='cueq-full'"
    ):
        backend.apply_sampling(req(triangle_kernel="cueq-full"))
    with pytest.raises(ValueError, match="triangle_kernel must be one of"):
        backend.apply_sampling(req(triangle_kernel="pallas"))
    for bad in ("", "auto", 0, True, None):
        with pytest.raises(ValueError, match="triangle_backend must be one of"):
            backend.validate_native_options({"triangle_backend": bad})
        if bad is not None:
            with pytest.raises(ValueError, match="triangle_backend must be one of"):
                pred(triangle_backend=bad)
        with pytest.raises(ValueError, match="triangle_backend must be one of"):
            inference._run(
                jax.random.key(0),
                {},
                {},
                None,
                structure_model.ModelSettings(),
                1,
                False,
                triangle_backend=bad,  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="triangle_backend must be one of"):
            inference.compiled_predict(
                structure_model.ModelSettings(),
                1,
                triangle_backend=bad,  # type: ignore[arg-type]
            )

    monkeypatch.setattr("foldjax.models._pallas_pair.gpu_process", lambda: False)
    for refused_opts in (
        {"triangle_kernel": "cueq"},
        {"triangle_backend": "cueq"},
        {"triangle_backend": "pallas"},
    ):
        with pytest.raises(ValueError, match="needs a GPU") as excinfo:
            backend.validate_request(req(**refused_opts))
        assert "platform 'cpu'" in str(excinfo.value)
        assert "--option triangle_backend=xla" in str(excinfo.value)
    with pytest.raises(ValueError, match="needs a GPU"):
        backend.predict(req(triangle_backend="pallas"))
    for fused_backend in ("cueq", "pallas"):
        with pytest.raises(ValueError, match="needs a GPU") as excinfo:
            pred(triangle_backend=fused_backend)
        assert "--option triangle_backend=xla" in str(excinfo.value)

    seen_cpu, res_cpu = _stub_prediction(tmp_path / "cpu", monkeypatch, {})
    assert "triangle_backend" not in seen_cpu
    assert "triangle_backend" not in res_cpu.raw["overrides"]

    monkeypatch.setattr(
        backend_module,
        "_gpu_process",
        lambda: (_ for _ in ()).throw(AssertionError("device touched")),
    )
    for safe_opts in (
        {},
        {"cp_devices": 2},
        {"triangle_backend": "xla"},
        {"triangle_kernel": "auto"},
    ):
        backend.validate_request(req(**safe_opts))

    monkeypatch.setattr(backend_module, "_gpu_process", lambda: True)
    monkeypatch.setattr("foldjax.models._pallas_pair.gpu_process", lambda: True)
    assert backend_module._strict_cp_devices(True) == 1
    assert backend_module._strict_cp_devices("bad") == 1
    assert backend_module._strict_cp_devices("2") == 2
    assert backend_module.realised_triangle_backend(None) == "pallas"
    assert backend_module.realised_triangle_backend(None, cp_devices=1) == "pallas"
    assert backend_module.realised_triangle_backend(None, cp_devices=0) == "pallas"
    assert backend_module.realised_triangle_backend(None, cp_devices=2) == "xla"
    # A float32 trunk keeps XLA: the fused kernel would run float32 GEMMs there.
    assert (
        backend_module.realised_triangle_backend(None, trunk_dtype="float32") == "xla"
    )
    assert backend_module.realised_triangle_backend(None, trunk_dtype="") == "xla"
    assert backend_module.realised_triangle_backend("xla", cp_devices=1) == "xla"
    assert backend_module.realised_triangle_backend("cueq", cp_devices=2) == "cueq"
    assert backend_module.realised_triangle_backend("pallas", cp_devices=2) == "pallas"
    for fused_backend in ("cueq", "pallas"):
        backend.validate_native_options(
            {"triangle_backend": fused_backend, "cp_devices": 1}
        )
        with pytest.raises(ValueError, match="cp_devices must be positive"):
            backend.predict(req(triangle_backend=fused_backend, cp_devices=0))
        with pytest.raises(ValueError, match="cannot be partitioned"):
            backend.validate_request(req(triangle_backend=fused_backend, cp_devices=2))
        with pytest.raises(ValueError, match="cannot be partitioned"):
            backend.predict(req(triangle_backend=fused_backend, cp_devices=2))
        with pytest.raises(ValueError, match="cannot be partitioned"):
            pred(cp_shards=2, triangle_backend=fused_backend)
        with pytest.raises(ValueError, match="cannot be partitioned"):
            inference._run(
                jax.random.key(0),
                {},
                {},
                None,
                structure_model.ModelSettings(),
                1,
                False,
                cp_shards=2,
                triangle_backend=fused_backend,
            )
    backend.validate_request(req(cp_devices=2))
    with pytest.raises(ValueError, match="cannot be partitioned"):
        backend.validate_request(req(triangle_kernel="cueq", cp_devices=2))

    seen_d, res_d = _stub_prediction(tmp_path / "d", monkeypatch, {})
    assert (
        seen_d["triangle_backend"]
        == res_d.raw["overrides"]["triangle_backend"]
        == "pallas"
    )
    seen_s, res_s = _stub_prediction(tmp_path / "s", monkeypatch, {"cp_devices": 1})
    assert (
        seen_s["triangle_backend"]
        == res_s.raw["overrides"]["triangle_backend"]
        == "pallas"
    )
    seen_a, res_a = _stub_prediction(
        tmp_path / "a", monkeypatch, {"triangle_kernel": "auto"}
    )
    assert (
        seen_a["triangle_backend"]
        == res_a.raw["overrides"]["triangle_backend"]
        == "pallas"
    )
    seen_p, res_p = _stub_prediction(
        tmp_path / "p", monkeypatch, {"triangle_backend": "pallas"}
    )
    assert (
        seen_p["triangle_backend"]
        == res_p.raw["overrides"]["triangle_backend"]
        == "pallas"
    )
    seen_cp, res_cp = _stub_prediction(
        tmp_path / "cp", monkeypatch, {"cp_devices": 2}
    )
    assert "triangle_backend" not in seen_cp
    assert "triangle_backend" not in res_cp.raw["overrides"]
    seen_x, res_x = _stub_prediction(
        tmp_path / "x", monkeypatch, {"triangle_backend": "xla"}
    )
    assert (
        seen_x["triangle_backend"]
        == res_x.raw["overrides"]["triangle_backend"]
        == "xla"
    )
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

    # CPU / non-GPU: omitted realises xla and aliases explicit xla.
    monkeypatch.setattr("foldjax.models._pallas_pair.gpu_process", lambda: False)
    assert backend_module.realised_triangle_backend(None) == "xla"
    omitted_cpu = prof()
    cueq_cpu = prof(triangle_backend="cueq")
    pallas_cpu = prof(triangle_backend="pallas")
    assert "triangle_backend" not in omitted_cpu
    assert omitted_cpu == prof(triangle_backend="xla") == prof(triangle_kernel="auto")
    assert cueq_cpu == prof(triangle_kernel="cueq") != omitted_cpu
    assert cueq_cpu["triangle_backend"] == "cueq"
    assert pallas_cpu["triangle_backend"] == "pallas"
    assert pallas_cpu != omitted_cpu and pallas_cpu != cueq_cpu

    # Serial GPU: omitted realises pallas; cueq and xla stay separate.
    monkeypatch.setattr("foldjax.models._pallas_pair.gpu_process", lambda: True)
    omitted_gpu = prof()
    pallas_gpu = prof(triangle_backend="pallas")
    cueq_gpu = prof(triangle_backend="cueq")
    xla_gpu = prof(triangle_backend="xla")
    assert omitted_gpu["triangle_backend"] == "pallas"
    assert (
        omitted_gpu
        == pallas_gpu
        == prof(triangle_kernel="auto")
        == prof(cp_devices=1)
    )
    assert cueq_gpu == prof(triangle_kernel="cueq") != omitted_gpu
    assert cueq_gpu["triangle_backend"] == "cueq"
    assert xla_gpu != omitted_gpu and xla_gpu != cueq_gpu
    assert "triangle_backend" not in xla_gpu
    # Context-parallel GPU: omitted realises xla and aliases explicit xla.
    cp_omitted_gpu = prof(cp_devices=2)
    assert "triangle_backend" not in cp_omitted_gpu
    assert cp_omitted_gpu == prof(cp_devices=2, triangle_backend="xla")
    assert prof(cp_devices=2, triangle_backend="cueq")["triangle_backend"] == "cueq"
    assert prof(cp_devices=2, triangle_backend="pallas")["triangle_backend"] == "pallas"

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
    codes = {"xla": 0, "cueq": 1, "pallas": 2}

    def tiny_predict(rng_arg: Any, model_features: Any, parameters: Any, **kwargs: Any):
        del rng_arg, parameters, kwargs
        resolved = trunk._resolve_triangle_backend(None)
        traces.append(resolved)
        return {
            "probe": model_features["asym_id"].astype(jnp.int32) + codes[resolved]
        }

    monkeypatch.setattr(structure_model, "predict", tiny_predict)
    pool = BoundedJitPool(
        inference._run, static_argnames=inference._COMPILED_PREDICT_STATIC_ARGNAMES
    )
    monkeypatch.setattr(inference, "_compiled_predict_pool", pool)
    inference._compiled_predict_factory.cache_clear()

    def run_pred(model_arg: Any = loaded, **kw: Any) -> dict[str, jnp.ndarray]:
        return inference.predict(
            jax.random.key(0),
            features,
            model_arg,
            precomputed_lm_embedding=lm_embed,
            **kw,
        )

    try:
        # GPU serial: omitted and explicit pallas share one executable; cueq/xla split.
        assert backend_module.realised_glu_backend(None) == "pallas"
        out_omitted_gpu = run_pred()
        out_pallas_gpu = run_pred(triangle_backend="pallas")
        jax.block_until_ready((out_omitted_gpu, out_pallas_gpu))
        assert traces == ["pallas"] and pool._entry_count() == 1  # noqa: SLF001
        assert int(out_omitted_gpu["probe"][0, 0]) == 2
        np.testing.assert_array_equal(out_omitted_gpu["probe"], out_pallas_gpu["probe"])

        out_cueq_gpu = run_pred(triangle_backend="cueq")
        jax.block_until_ready(out_cueq_gpu)
        assert traces == ["pallas", "cueq"] and pool._entry_count() == 2  # noqa: SLF001
        assert int(out_cueq_gpu["probe"][0, 0]) == 1

        out_xla_gpu = run_pred(triangle_backend="xla", glu_backend="xla")
        jax.block_until_ready(out_xla_gpu)
        assert (
            traces == ["pallas", "cueq", "xla"] and pool._entry_count() == 3  # noqa: SLF001
        )
        assert int(out_xla_gpu["probe"][0, 0]) == 0

        # CPU: omitted shares the xla executable without recompiling.
        monkeypatch.setattr("foldjax.models._pallas_pair.gpu_process", lambda: False)
        out_omitted_cpu = run_pred()
        jax.block_until_ready(out_omitted_cpu)
        assert (
            traces == ["pallas", "cueq", "xla"] and pool._entry_count() == 3  # noqa: SLF001
        )
        np.testing.assert_array_equal(out_omitted_cpu["probe"], out_xla_gpu["probe"])

        # GPU + float32 trunk: omitted resolves to xla.
        monkeypatch.setattr("foldjax.models._pallas_pair.gpu_process", lambda: True)
        f32_loaded = inference.LoadedModel(
            {bond_name: jnp.ones((settings.d_pair, 1), jnp.float32)},
            dataclasses.replace(settings, trunk_dtype=jnp.dtype(jnp.float32)),
        )
        out_f32_gpu = run_pred(f32_loaded)
        jax.block_until_ready(out_f32_gpu)
        assert (
            traces == ["pallas", "cueq", "xla", "xla"]
            and pool._entry_count() == 4  # noqa: SLF001
        )
        assert int(out_f32_gpu["probe"][0, 0]) == 0

        # GPU + CP: omitted resolves to xla without refusal.
        seen_cp_backend: list[str] = []

        def fake_compiled_predict(*_args: Any, **kwargs: Any):
            seen_cp_backend.append(kwargs.get("triangle_backend", "xla"))
            raise RuntimeError("reached compiled_predict")

        monkeypatch.setattr(inference, "compiled_predict", fake_compiled_predict)
        with pytest.raises(RuntimeError, match="reached compiled_predict"):
            run_pred(cp_shards=2)
        assert seen_cp_backend == ["xla"]
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
    monkeypatch.setattr(
        _pallas_pair,
        "triangle_multiplication",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("pallas called")),
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
