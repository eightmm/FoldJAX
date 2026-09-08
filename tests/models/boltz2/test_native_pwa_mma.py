"""CPU contract/IR checks, not CUDA numeric or other-shape parity admission."""

import itertools
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend import core as jax_core

from foldjax.models.boltz2.models.primitives import native_pwa_mma as native
from foldjax.models.boltz2.models.trunk_blocks import msa


def test_private_lowering_returns_four_actual_ir_results_without_global_patch():
    from jax._src.lib.mlir import ir
    from jax._src.lib.mlir.dialects import func
    from jax._src.pallas.triton import primitives

    registry = native.triton_lowering.triton_lowering_rules
    original = registry[primitives.elementwise_inline_asm_p]
    with native.triton_lowering._new_ir_context(), ir.Location.unknown():
        module = ir.Module.create()
        u32 = ir.RankedTensorType.get((32,), ir.IntegerType.get_signless(32))
        f32 = ir.RankedTensorType.get((32,), ir.F32Type.get())
        with ir.InsertionPoint(module.body):
            fn = func.FuncOp(
                "mma", ir.FunctionType.get([u32] * 3 + [f32] * 4, [f32] * 4)
            )
            block = fn.add_entry_block()
            with ir.InsertionPoint(block):
                context = SimpleNamespace(platform="cuda", compute_capability=120)
                outputs = native._mma_lowering(
                    SimpleNamespace(context=context), *block.arguments
                )
                assert len(outputs) == 4 and all(
                    isinstance(x, ir.Value) for x in outputs
                )
                func.ReturnOp(outputs)
        assert module.operation.verify()
        op = outputs[0].owner
        assert op.name == "tt.elementwise_inline_asm"
        assert list(op.operands) == list(block.arguments)
        assert all(x.owner == op for x in outputs)
        assert native._ASM in str(op) and native._CONSTRAINTS in str(op)
        assert "arith.addf" not in str(module) and "tt.dot" not in str(module)
    assert registry[primitives.elementwise_inline_asm_p] is original
    assert original is primitives._elementwise_inline_asm_lowering


def test_full_kernel_lowers_guarded_stores_without_compiler_monkeypatch():
    from jax._src import sharding_impls
    from jax._src.interpreters import mlir
    from jax._src.pallas.triton import primitives

    registry = native.triton_lowering.triton_lowering_rules
    original = registry[primitives.elementwise_inline_asm_p]
    graph = jax.make_jaxpr(native._pwa_outputs)(
        jax.ShapeDtypeStruct((437, 437), jnp.bfloat16),
        jax.ShapeDtypeStruct((2, 437, 32), jnp.bfloat16),
        jax.ShapeDtypeStruct((1,), jnp.float32),
    )
    call = next(e for e in graph.jaxpr.eqns if e.primitive.name == "pallas_call")
    context = mlir.ModuleContext(
        platforms=("cuda",),
        backend=None,
        axis_context=sharding_impls.ShardingContext(num_devices=1),
        keepalives=[],
        channel_iterator=itertools.count(1),
        host_callbacks=[],
        lowering_parameters=mlir.LoweringParameters(),
    )
    # Generate actual Triton IR, not CUDA machine code or a GPU executable.
    result = native.triton_lowering.lower_jaxpr_to_triton_module(
        call.params["jaxpr"], call.params["grid_mapping"], "cuda", 120, context
    )
    assert result.module.operation.verify()
    assert result.grid == [4, 55]
    fn = next(iter(result.module.body.operations))
    ops = list(fn.regions[0].blocks[0].operations)
    reduction = next(op for op in ops if op.name == "tt.reduce")
    stores = [op for op in ops if op.name == "tt.store"]
    assert len(stores) == 5
    assert len([op for op in ops if op.name == "tt.reduce"]) == 1
    for store in stores[:4]:
        select = store.operands[1].owner
        assert select.name == "arith.select"
        assert select.operands[1].owner.name == "scf.for"
        nan_splat = select.operands[2].owner
        assert nan_splat.name == "tt.splat"
        assert np.isnan(float(nan_splat.operands[0].owner.attributes["value"].value))
        comparison = select.operands[0].owner.operands[0].owner
        assert comparison.name == "arith.cmpi"
        assert comparison.operands[0] == reduction.results[0]
    text = str(result.module)
    assert native._ASM in text and "mov.u32 $0, %laneid;" in text
    assert "tt.dot" not in text and "callback" not in text
    assert not context.host_callbacks
    assert registry[primitives.elementwise_inline_asm_p] is original
    assert original is primitives._elementwise_inline_asm_lowering


def test_valid_lane_guard_preserves_all_register_bits():
    words = np.tile(
        np.array(
            [0, 0x80000000, 1, 0x80000001, 0x3F800000, 0xBF800000, 0x7F7FFFFF, 7],
            np.uint32,
        ),
        (4, 4),
    )
    carry = tuple(words.view(np.float32))
    output = jax.jit(native._guard_lane_identity)(carry, jnp.arange(32))
    np.testing.assert_array_equal(np.asarray(output).view(np.uint32), words)


@pytest.mark.parametrize("bad_lane", range(32))
def test_each_invalid_physical_lane_poisons_all_output_registers(bad_lane):
    from bench import boltz_pwa_full_mma_probe as probe

    physical = np.arange(32, dtype=np.int32)
    physical[bad_lane] = (bad_lane + 1) % 32
    carry = tuple(np.arange(128, dtype=np.float32).reshape(4, 32))
    output = np.asarray(jax.jit(native._guard_lane_identity)(carry, physical))
    assert output.shape == (4, 32) and np.isnan(output).all()
    # The frozen benchmark must reject the runtime signal, not admit matching
    # nonfinite payloads or silently round them into a passing BF16 comparison.
    stats = probe.empty_stats()
    probe.add_stats(stats, output, np.asarray(carry))
    result = probe.finish_stats(stats)
    assert result["nonfinite"] == 128 and not result["bitwise_equal"]


@pytest.mark.parametrize(
    "platform,capability", [("rocm", 120), ("cuda", None), ("cuda", 75)]
)
def test_lowering_rejects_unsupported_instruction_backend(platform, capability):
    context = SimpleNamespace(platform=platform, compute_capability=capability)
    with pytest.raises(NotImplementedError):
        native._mma_lowering(SimpleNamespace(context=context))


@pytest.mark.parametrize("problem", ["arity", "lane_shape", "words", "carry"])
def test_private_primitive_rejects_wrong_register_contract(problem):
    registers = [jax.core.ShapedArray((32,), jnp.uint32)] * 3
    registers += [jax.core.ShapedArray((32,), jnp.float32)] * 4
    if problem == "arity":
        registers.pop()
    elif problem == "lane_shape":
        registers[0] = jax.core.ShapedArray((16,), jnp.uint32)
    elif problem == "words":
        registers[0] = jax.core.ShapedArray((32,), jnp.bfloat16)
    else:
        registers[3] = jax.core.ShapedArray((32,), jnp.bfloat16)
    with pytest.raises((TypeError, ValueError)):
        native._mma_abstract(*registers)


def test_no_eager_arithmetic_fallback():
    with pytest.raises(RuntimeError, match="Triton-only"):
        native._mma_eager()


def test_runtime_register_schedule_and_carried_c_match_validated_probe():
    from bench.boltz_pwa_mma_probe import mma_assembly

    np.testing.assert_array_equal(
        native._residue_k(jnp.arange(448)),
        np.concatenate((np.arange(21), np.full(11, -1), np.arange(21, 437))),
    )
    assert (native._ASM, native._CONSTRAINTS) == mma_assembly(8)
    graph = jax.make_jaxpr(native._pwa_outputs)(
        jax.ShapeDtypeStruct((437, 437), jnp.bfloat16),
        jax.ShapeDtypeStruct((2, 437, 32), jnp.bfloat16),
        jax.ShapeDtypeStruct((1,), jnp.float32),
    )
    casts = [e for e in graph.jaxpr.eqns if e.primitive.name == "bitcast_convert_type"]
    assert len(casts) == 2
    assert all(e.outvars[0].aval.dtype == jnp.uint16 for e in casts)
    call = next(e for e in graph.jaxpr.eqns if e.primitive.name == "pallas_call")
    assert call.params["grid_mapping"].grid == (4, 55)
    assert call.params["compiler_params"].num_warps == 1
    assert not call.params["interpret"]
    kernel = call.params["jaxpr"]
    loop = next(e for e in kernel.eqns if e.primitive.name == "scan")
    assert loop.params["length"] == 56
    body = loop.params["jaxpr"].jaxpr
    mma = next(e for e in body.eqns if e.primitive is native._mma1688_p)
    assert mma.invars[-4:] == body.invars[-4:]
    assert body.outvars[-4:] == mma.outvars
    assert len([e for e in body.eqns if e.primitive.name == "masked_load"]) == 6
    for eq in body.eqns:
        assert "dot" not in eq.primitive.name
        if eq.primitive.name == "add":
            assert all(v.aval.dtype == jnp.int32 for v in eq.invars)
    assert graph.jaxpr.outvars[0].aval.dtype == jnp.float32
    assert graph.jaxpr.outvars[1].aval.dtype == jnp.bfloat16
    assert graph.jaxpr.outvars[2].aval.dtype == jnp.int32


@pytest.mark.parametrize("rows", [1, 1199, 839, 4435, 4437])
def test_private_positive_rows_do_not_widen_production_dispatch(rows):
    graph = jax.make_jaxpr(native._pwa_raw_fp32)(
        jax.ShapeDtypeStruct((437, 437), jnp.bfloat16),
        jax.ShapeDtypeStruct((rows, 437, 32), jnp.bfloat16),
        jax.ShapeDtypeStruct((1,), jnp.float32),
    )
    call = next(e for e in graph.jaxpr.eqns if e.primitive.name == "pallas_call")
    assert call.params["grid_mapping"].grid == (rows * 2, 55)
    assert graph.jaxpr.outvars[0].aval.shape == (rows, 437, 32)
    operands = (
        jax.ShapeDtypeStruct((1, 1, 437, 437), jnp.bfloat16),
        jax.ShapeDtypeStruct((1, 1, rows, 437, 32), jnp.bfloat16),
    )
    assert not native._observed_shape(*operands)
    with pytest.raises(ValueError, match="S4436"):
        native.native_pwa_contraction(*operands)


@pytest.mark.parametrize(
    "shape", [(0, 437, 32), (1199, 436, 32), (1199, 437, 16), (437, 32)]
)
def test_private_row_kernel_still_rejects_nonpositive_s_and_changed_n_d(shape):
    with pytest.raises(ValueError, match="positive S, fixed N437/D32"):
        native._pwa_raw_fp32(
            jax.ShapeDtypeStruct((437, 437), jnp.bfloat16),
            jax.ShapeDtypeStruct(shape, jnp.bfloat16),
            jax.ShapeDtypeStruct((1,), jnp.float32),
        )


@pytest.mark.parametrize("operand", ["weights", "values"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16, jnp.uint16])
def test_private_kernel_rejects_non_bf16_raw_operands(operand, dtype):
    w_dtype = dtype if operand == "weights" else jnp.bfloat16
    v_dtype = dtype if operand == "values" else jnp.bfloat16
    with pytest.raises(TypeError, match="already-BF16"):
        native._pwa_raw_fp32(
            jax.ShapeDtypeStruct((437, 437), w_dtype),
            jax.ShapeDtypeStruct((2, 437, 32), v_dtype),
            jax.ShapeDtypeStruct((1,), jnp.float32),
        )


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((), jnp.float32),
        ((2,), jnp.float32),
        ((1, 1), jnp.float32),
        ((1,), jnp.bfloat16),
        ((1,), jnp.float16),
        ((1,), jnp.int32),
    ],
)
def test_private_kernel_rejects_non_scalar_fp32_initial_register(shape, dtype):
    with pytest.raises(ValueError, match="positive S, fixed N437/D32"):
        native._pwa_raw_fp32(
            jax.ShapeDtypeStruct((437, 437), jnp.bfloat16),
            jax.ShapeDtypeStruct((2, 437, 32), jnp.bfloat16),
            jax.ShapeDtypeStruct(shape, dtype),
        )


def test_private_kernel_fails_explicitly_when_pallas_unavailable(monkeypatch):
    monkeypatch.setattr(native, "pl", None)
    with pytest.raises(ImportError, match="requires Pallas Triton"):
        native._pwa_raw_fp32(
            jax.ShapeDtypeStruct((437, 437), jnp.bfloat16),
            jax.ShapeDtypeStruct((2, 437, 32), jnp.bfloat16),
            jax.ShapeDtypeStruct((1,), jnp.float32),
        )


def observed_operands(rows=4436):
    return (
        jax.ShapeDtypeStruct((1, 1, 437, 437), jnp.bfloat16),
        jax.ShapeDtypeStruct((1, 1, rows, 437, 32), jnp.bfloat16),
    )


@pytest.mark.parametrize("rows", [4436, 1199, 839])
def test_observed_dispatch_traces_both_branches_and_lowers_on_cpu_unpatched(rows):
    from jax._src.pallas.triton import primitives

    registry = native.triton_lowering.triton_lowering_rules
    original = registry[primitives.elementwise_inline_asm_p]
    operands = observed_operands(rows)
    expected = operands[1]

    def forced(w, v):
        return native.native_pwa_contraction(w, v, original_msa_rows=4436)

    def dispatch(w, v):
        return native.pair_weighted_contraction(w, v, original_msa_rows=4436)

    assert jax.eval_shape(forced, *operands) == expected
    assert jax.eval_shape(dispatch, *operands) == expected
    with jax.default_device(jax.devices("cpu")[0]):
        text = jax.jit(dispatch).lower(*operands).as_text()
    assert "stablehlo.dot_general" in text
    assert "triton" not in text.lower() and "custom_call" not in text
    assert registry[primitives.elementwise_inline_asm_p] is original
    assert original is primitives._elementwise_inline_asm_lowering


@pytest.mark.parametrize("rows", [4436, 1199, 839])
def test_forced_native_wrapper_rejects_cp_before_tracing_kernel(monkeypatch, rows):
    monkeypatch.setattr(native, "cp_mesh", lambda: object())
    with pytest.raises(ValueError, match="unsharded"):
        native.native_pwa_contraction(*observed_operands(rows), original_msa_rows=4436)


@pytest.mark.parametrize("rows", [4436, 1199, 839])
def test_dispatch_is_explicitly_cuda_with_existing_other_platform_branch(
    monkeypatch, rows
):
    operands = observed_operands(rows)
    seen = []

    def platform(*args, **branches):
        seen.append((args, branches))
        return "selected"

    monkeypatch.setattr(native.jax.lax, "platform_dependent", platform)
    assert (
        native.pair_weighted_contraction(*operands, original_msa_rows=4436)
        == "selected"
    )
    args, branches = seen.pop()
    assert args == operands
    assert set(branches) == {"cuda", "default"}
    assert branches["cuda"].func is native.native_pwa_contraction
    assert branches["cuda"].keywords == {"original_msa_rows": 4436}
    w = jnp.arange(9, dtype=jnp.float32).reshape(1, 1, 3, 3)
    v = jnp.arange(24, dtype=jnp.float32).reshape(1, 1, 2, 3, 4)
    np.testing.assert_array_equal(
        branches["default"](w, v), jnp.einsum("bhij,bhsjd->bhsid", w, v)
    )


@pytest.mark.parametrize(
    "problem",
    [
        "rows",
        "batch",
        "heads",
        "tokens",
        "channels",
        "fp32",
        "fp16",
        "cp",
        "unavailable",
    ],
)
def test_every_unobserved_operand_preserves_existing_einsum(monkeypatch, problem):
    w, v = observed_operands()
    shape, dtype = list(v.shape), v.dtype
    if problem == "rows":
        shape[2] = 1199
    elif problem == "batch":
        shape[0] = 2
    elif problem == "heads":
        shape[1] = 8
    elif problem == "tokens":
        shape[3] = 436
    elif problem == "channels":
        shape[4] = 16
    elif problem in ("fp32", "fp16"):
        dtype = jnp.float32 if problem == "fp32" else jnp.float16
    elif problem == "cp":
        monkeypatch.setattr(native, "cp_mesh", lambda: object())
    else:
        monkeypatch.setattr(native, "_mma1688_p", None)
    v = jax.ShapeDtypeStruct(tuple(shape), dtype)
    seen = []
    monkeypatch.setattr(native.jnp, "einsum", lambda *args: seen.append(args) or "old")
    monkeypatch.setattr(
        native.jax.lax,
        "platform_dependent",
        lambda *a, **k: pytest.fail("unexpected native dispatch"),
    )
    assert native.pair_weighted_contraction(w, v) == "old"
    assert seen == [("bhij,bhsjd->bhsid", w, v)]


@pytest.mark.parametrize("rows", [0, None, 4436, 9000])
def test_actual_msa_amp_caller_keeps_full_rows_or_existing_auto_chunks(
    monkeypatch, rows
):
    def shape(dims):
        return jax.ShapeDtypeStruct(dims, jnp.bfloat16)

    params = {
        "norm_m": {"scale": shape((64,)), "bias": shape((64,))},
        "norm_z": {"scale": shape((128,)), "bias": shape((128,))},
        "proj_m": {"kernel": shape((64, 256))},
        "proj_z": {"kernel": shape((128, 8))},
        "proj_g": {"kernel": shape((64, 256))},
        "proj_o": {"kernel": shape((256, 64))},
    }
    seen = []

    def contraction(w, v, *, original_msa_rows):
        seen.append((w.shape, v.shape, w.dtype, v.dtype, original_msa_rows))
        return jnp.zeros(v.shape, v.dtype)

    monkeypatch.setattr(msa, "pair_weighted_contraction", contraction)
    jax.make_jaxpr(
        lambda p, m, z, mask: msa.pair_weighted_averaging_forward(
            p, m, z, mask, row_chunk_size=rows
        )
    )(
        params,
        jax.ShapeDtypeStruct((1, 4436, 437, 64), jnp.float32),
        jax.ShapeDtypeStruct((1, 437, 437, 128), jnp.float32),
        jax.ShapeDtypeStruct((1, 437, 437), jnp.float32),
    )
    expected_rows = [1199, 1199, 1199, 839] if rows is None else [4436]
    assert [entry[1][2] for entry in seen] == [
        s for s in expected_rows for _ in range(8)
    ]
    assert all(
        w == (1, 1, 437, 437) and wd == vd == jnp.bfloat16 and original == 4436
        for w, _, wd, vd, original in seen
    )


@pytest.mark.parametrize("rows", [839, 1199, 1200, 4435, 4436, 4437])
@pytest.mark.parametrize("original", [None, 839, 1199, 4436, 7000])
def test_row_dispatch_requires_the_verified_original_msa_context(rows, original):
    operands = observed_operands(rows)
    expected = (rows == 4436 and original is None) or (
        original == 4436 and rows in (839, 1199, 4436)
    )
    assert native._observed_shape(*operands, original_msa_rows=original) is expected
    if not expected:
        with pytest.raises(ValueError, match="original S4436"):
            native.native_pwa_contraction(*operands, original_msa_rows=original)


def _shape_pwa_case(rows=4436, dtype=jnp.bfloat16):
    def shape(dims, kind=dtype):
        return jax.ShapeDtypeStruct(dims, kind)

    params = {
        "norm_m": {"scale": shape((64,)), "bias": shape((64,))},
        "norm_z": {"scale": shape((128,)), "bias": shape((128,))},
        "proj_m": {"kernel": shape((64, 256))},
        "proj_z": {"kernel": shape((128, 8))},
        "proj_g": {"kernel": shape((64, 256))},
        "proj_o": {"kernel": shape((256, 64))},
    }
    return (
        params,
        shape((1, rows, 437, 64), jnp.float32),
        shape((1, 437, 437, 128), jnp.float32),
        shape((1, 437, 437), jnp.float32),
    )


def _pallas_calls(graph):
    if isinstance(graph, jax_core.ClosedJaxpr):
        graph = graph.jaxpr
    if isinstance(graph, jax_core.Jaxpr):
        for eq in graph.eqns:
            if (
                eq.primitive.name == "pallas_call"
                and eq.params.get("name") == "boltz_native_pwa_mma1688"
            ):
                yield eq
            else:
                yield from _pallas_calls(eq.params)
    elif isinstance(graph, dict):
        for value in graph.values():
            yield from _pallas_calls(value)
    elif isinstance(graph, (tuple, list)):
        for value in graph:
            yield from _pallas_calls(value)


def test_actual_default_pwa_traces_all_verified_cuda_row_chunks_without_mock():
    graph = jax.make_jaxpr(msa.pair_weighted_averaging_forward)(*_shape_pwa_case())
    calls = list(_pallas_calls(graph))
    assert [call.outvars[0].aval.shape for call in calls] == [
        (rows, 437, 32) for rows in (1199, 1199, 1199, 839) for _ in range(8)
    ]
    assert all(
        call.params["grid_mapping"].grid == (call.outvars[0].aval.shape[0] * 2, 55)
        and call.params["compiler_params"].num_warps == 1
        for call in calls
    )
    assert graph.jaxpr.outvars[0].aval.shape == (1, 4436, 437, 64)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16, jnp.float32])
def test_native_pwa_norm_only_selected_for_bf16(monkeypatch, dtype):
    calls = []

    def norm(x, scale, bias, eps):
        assert x.dtype == jnp.float32
        calls.append(x.shape)
        return x

    monkeypatch.setattr(msa, "amp_layer_norm", norm)
    jax.make_jaxpr(lambda *args: msa.pair_weighted_averaging_forward(*args))(
        *_shape_pwa_case(dtype=dtype)
    )
    expected = [(1, 437, 437, 128)] + [
        (1, rows, 437, 64) for rows in (1199, 1199, 1199, 839)
    ]
    assert calls == (expected if dtype == jnp.bfloat16 else [])


def test_bf16_msa_residual_keeps_its_different_auto_chunk_profile():
    params, m, z, mask = _shape_pwa_case()
    m = jax.ShapeDtypeStruct(m.shape, jnp.bfloat16)
    assert msa._auto_pair_averaging_chunk(m, params) == 2399
    graph = jax.make_jaxpr(msa.pair_weighted_averaging_forward)(params, m, z, mask)
    assert list(_pallas_calls(graph)) == []


@pytest.mark.parametrize(
    "rows,chunk,dtype,cp",
    [
        (4436, None, jnp.bfloat16, True),
        (4436, None, jnp.float32, False),
        (4436, None, jnp.float16, False),
        (1199, None, jnp.bfloat16, False),
        (839, None, jnp.bfloat16, False),
        (7000, None, jnp.bfloat16, False),
        (4436, 1199, jnp.bfloat16, False),
        (4436, 839, jnp.bfloat16, False),
        (4436, 3237, jnp.bfloat16, False),
        (7000, 4436, jnp.bfloat16, False),
    ],
)
def test_other_actual_pwa_profiles_do_not_trace_native_mma(
    monkeypatch, rows, chunk, dtype, cp
):
    if cp:
        monkeypatch.setattr(native, "cp_mesh", lambda: object())
    graph = jax.make_jaxpr(
        lambda *operands: msa.pair_weighted_averaging_forward(
            *operands, row_chunk_size=chunk
        )
    )(*_shape_pwa_case(rows, dtype))
    assert list(_pallas_calls(graph)) == []


def test_runtime_native_wrapper_does_not_change_operand_bits_or_output_layout(
    monkeypatch,
):
    w = jnp.zeros((1, 1, 437, 437), jnp.bfloat16)
    v = jax.ShapeDtypeStruct((1, 1, 4436, 437, 32), jnp.bfloat16)
    # Shape-only tracing avoids allocating the full MSA on CPU.
    calls = []

    def outputs(weight, value, initial):
        calls.append((weight.shape, value.shape, initial.shape))
        return None, jnp.zeros(value.shape, jnp.bfloat16), None

    monkeypatch.setattr(native, "_pwa_outputs", outputs)
    result = jax.eval_shape(lambda a, b: native.native_pwa_contraction(a, b), w, v)
    assert calls == [((437, 437), (4436, 437, 32), (1,))]
    assert result.shape == v.shape and result.dtype == jnp.bfloat16
