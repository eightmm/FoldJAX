"""``pair_residual_dtype`` stores the pair carry narrow, and says so.

Three failure modes are in scope and none is visible from a config field.

The first is a default that drifts. `compute_dtype="bfloat16"` narrows every
trunk GEMM; this option is the separate question of what the pair residual is
*stored* in between them, and the released answer is now bfloat16.
``test_released_default_realises_a_bfloat16_pair_carry`` reads the dtype off
arrays the trunk actually produced with the option omitted entirely, so a
default that silently reverts fails here rather than in a memory table.

The second is the float32 arm losing its identity. It is still exactly
reachable, and it must still be the program it was while it was the default:
``test_the_float32_arm_emits_nothing`` lowers it twice -- once as spelled,
once with the option's own helper replaced by a strict identity, i.e. with
the feature physically absent -- and compares the programs as text.

The third is an arm that never fires. A patched arm measuring dead code is
how a knob gets reported as free, so every test here reads ``.dtype`` off an
array the trunk produced, or reads the lowered program. The precision checks
that used to run "off vs opt-in" now run the *omitted* arm too, because what
they guard is the released program and that is no longer the wide one.

The parameters are synthetic and tiny on purpose: the checkpoint-parity
fixtures need a torch install and an upstream checkout, and a gate that only
runs where those exist is a gate that runs nowhere.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.boltz2.models.primitives import _common
from foldjax.models.boltz2.models.triangle import triangle as tri_module
from foldjax.models.boltz2.models.trunk_blocks import msa as msa_module
from foldjax.models.boltz2.models.trunk_blocks import pairformer as pf_module
from foldjax.models.boltz2.models.trunk_blocks import trunk as trunk_module

N_TOKEN = 6
N_MSA = 3
C_S_IN = 6
C_S = 8
C_Z = 4
C_M = 8
C_HID = 2
HEADS = 2
R_MAX, S_MAX = 32, 2
MSA_CATEGORIES = 33
PAIR = f"tensor<1x{N_TOKEN}x{N_TOKEN}x{C_Z}x"


def _build():
    rng = np.random.default_rng(20260911)

    def arr(*shape, scale=0.3):
        return jnp.asarray(rng.normal(scale=scale, size=shape), jnp.float32)

    def norm(c):
        return {"scale": arr(c) * 0.1 + 1.0, "bias": arr(c) * 0.1}

    def kern(i, o):
        return {"kernel": arr(i, o, scale=1.0 / np.sqrt(i))}

    def kern_b(i, o):
        return {"kernel": arr(i, o, scale=1.0 / np.sqrt(i)), "bias": arr(o, scale=0.05)}

    def transition(c):
        return {
            "norm": norm(c),
            "fc1": kern(c, 2 * c),
            "fc2": kern(c, 2 * c),
            "fc3": kern(2 * c, c),
        }

    def tri_mult(c):
        return {
            "norm_in": norm(c),
            "norm_out": norm(c),
            "g_in": kern(c, 2 * c),
            "p_in": kern(c, 2 * c),
            "p_out": kern(c, c),
            "g_out": kern(c, c),
        }

    def tri_att(c):
        return {
            "layer_norm": norm(c),
            "linear": kern(c, HEADS),
            "mha": {
                "linear_q": kern(c, c),
                "linear_k": kern(c, c),
                "linear_v": kern(c, c),
                "linear_g": kern(c, c),
                "linear_o": kern(c, c),
            },
        }

    def pair_block():
        return {
            "tri_mul_out": tri_mult(C_Z),
            "tri_mul_in": tri_mult(C_Z),
            "tri_att_start": tri_att(C_Z),
            "tri_att_end": tri_att(C_Z),
            "transition_z": transition(C_Z),
        }

    def pairformer_layer():
        block = pair_block()
        block["pre_norm_s"] = norm(C_S)
        block["attention"] = {
            "proj_q": kern_b(C_S, C_S),
            "proj_g": kern(C_S, C_S),
            "proj_k": kern(C_S, C_S),
            "proj_v": kern(C_S, C_S),
            "proj_z_norm": norm(C_Z),
            "proj_z": kern(C_Z, HEADS),
            "proj_o": kern(C_S, C_S),
        }
        block["transition_s"] = transition(C_S)
        return block

    def msa_layer():
        return {
            "pair_weighted_averaging": {
                "norm_m": norm(C_M),
                "norm_z": norm(C_Z),
                "proj_m": kern(C_M, HEADS * C_HID),
                "proj_z": kern(C_Z, HEADS),
                "proj_g": kern(C_M, HEADS * C_HID),
                "proj_o": kern(HEADS * C_HID, C_M),
            },
            "msa_transition": transition(C_M),
            "outer_product_mean": {
                "norm": norm(C_M),
                "proj_a": kern(C_M, C_HID),
                "proj_b": kern(C_M, C_HID),
                "proj_o": kern_b(C_HID * C_HID, C_Z),
            },
            "pairformer_layer": pair_block(),
        }

    n_rel = 2 * R_MAX + 2
    n_chain = 2 * S_MAX + 2
    params = {
        # Stubbed below: this file is about the pair residual, not the embedder.
        "input_embedder": {},
        "s_init": kern(C_S_IN, C_S),
        "z_init_1": kern(C_S_IN, C_Z),
        "z_init_2": kern(C_S_IN, C_Z),
        "rel_pos": {"linear_layer": kern(2 * n_rel + 1 + n_chain, C_Z)},
        "token_bonds": kern(1, C_Z),
        "token_bonds_type": arr(4, C_Z),
        "contact_conditioning": {
            "encoder": kern_b(6 + 1 + 4, C_Z),
            "fourier_embedding": {"proj": kern_b(1, 4)},
            "encoding_unspecified": arr(C_Z),
            "encoding_unselected": arr(C_Z),
        },
        "s_norm": norm(C_S),
        "z_norm": norm(C_Z),
        "s_recycle": kern(C_S, C_S),
        "z_recycle": kern(C_Z, C_Z),
        "msa_module": {
            "msa_proj": kern(MSA_CATEGORIES + 3, C_M),
            "s_proj": kern(C_S_IN, C_M),
            "layers": [msa_layer(), msa_layer()],
        },
        "pairformer_module": {"layers": [pairformer_layer(), pairformer_layer()]},
    }

    def ints(*shape):
        return jnp.asarray(rng.integers(0, 3, size=shape), jnp.int32)

    feats = {
        "token_pad_mask": jnp.ones((1, N_TOKEN), jnp.float32),
        "token_bonds": jnp.asarray(
            rng.integers(0, 2, (1, N_TOKEN, N_TOKEN, 1)), jnp.float32
        ),
        "type_bonds": ints(1, N_TOKEN, N_TOKEN),
        "contact_conditioning": jnp.asarray(
            rng.random((1, N_TOKEN, N_TOKEN, 8)), jnp.float32
        ),
        "contact_threshold": jnp.asarray(
            rng.random((1, N_TOKEN, N_TOKEN)) * 10, jnp.float32
        ),
        "asym_id": ints(1, N_TOKEN),
        "residue_index": ints(1, N_TOKEN),
        "token_index": ints(1, N_TOKEN),
        "entity_id": ints(1, N_TOKEN),
        "sym_id": ints(1, N_TOKEN),
        "mol_type": ints(1, N_TOKEN),
        "cyclic_period": jnp.zeros((1, N_TOKEN), jnp.float32),
        "msa": jnp.asarray(
            rng.integers(0, MSA_CATEGORIES, (1, N_MSA, N_TOKEN)), jnp.int32
        ),
        "has_deletion": jnp.asarray(
            rng.integers(0, 2, (1, N_MSA, N_TOKEN)), jnp.float32
        ),
        "deletion_value": jnp.asarray(rng.random((1, N_MSA, N_TOKEN)), jnp.float32),
        "msa_paired": jnp.asarray(rng.integers(0, 2, (1, N_MSA, N_TOKEN)), jnp.float32),
        "msa_mask": jnp.ones((1, N_MSA, N_TOKEN), jnp.float32),
    }
    return params, feats, arr(1, N_TOKEN, C_S_IN)


#: The two arms every precision check runs: the wide one, spelled, and the
#: released one, omitted. Omission is what a released run actually does, and
#: a check that only ever ran the explicit spelling would not notice the
#: default moving out from under it.
_ARMS = (("float32", False), (None, True))


def _run(
    monkeypatch,
    pair_residual_dtype,
    *,
    lower_only=False,
    omit=False,
    trunk_dtype=jnp.bfloat16,
):
    params, feats, s_inputs = _build()
    params = trunk_module._cast_trunk_params(params, trunk_dtype)
    fired = []

    def stub(_params, _feats, **_kwargs):
        fired.append(True)
        return s_inputs

    monkeypatch.setattr(trunk_module, "input_embedder_forward", stub)
    extra = {} if omit else {"pair_residual_dtype": pair_residual_dtype}

    def call(p, f):
        return trunk_module.boltz2_trunk_forward(
            p,
            f,
            recycling_steps=1,
            use_scan=True,
            triangle_backend="xla",
            glu_backend="xla",
            **extra,
        )

    compiled = jax.jit(call)
    result = (
        compiled.lower(params, feats).as_text()
        if lower_only
        else jax.device_get(compiled(params, feats))
    )
    assert fired, "the input-embedder stub never ran; the probe measured nothing"
    return result


def test_the_released_default_is_the_narrow_arm(monkeypatch) -> None:
    """Omitting the option must name the bfloat16 program, not the wide one."""

    omitted = _run(monkeypatch, None, lower_only=True, omit=True)
    assert omitted == _run(monkeypatch, "bfloat16", lower_only=True)
    assert omitted == _run(monkeypatch, "auto", lower_only=True)
    assert omitted != _run(monkeypatch, "float32", lower_only=True)


def test_the_float32_arm_emits_nothing(monkeypatch) -> None:
    """The wide arm is still the program it was while it was the default.

    Not a numerically-equal cast: no operation at all. `"float32"` and the
    bare `None` sentinel both resolve to "emit nothing", so replacing the
    helper with a strict identity -- the feature physically absent from the
    trace -- has to leave the text alone.
    """

    spelled = _run(monkeypatch, "float32", lower_only=True)
    assert spelled == _run(monkeypatch, None, lower_only=True)

    for module in (trunk_module, msa_module, pf_module):
        monkeypatch.setattr(module, "_residual_cast", lambda x, dtype: x)
    assert _run(monkeypatch, "float32", lower_only=True) == spelled


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16, jnp.int32])
def test_residual_cast_is_the_identity_object_when_unset(dtype) -> None:
    x = jnp.zeros((2, 2), dtype)
    assert _common.residual_cast(x, None) is x
    assert _common.residual_cast(x, dtype) is x


def _spy(monkeypatch, pair_residual_dtype, *, omit=False, **kwargs):
    seen: dict[str, list[str]] = {
        "msa_layer_in": [],
        "pairformer_layer_in": [],
        "pairformer_layer_out": [],
        "pairformer_single_out": [],
    }
    msa_original = msa_module.msa_layer_forward
    pf_original = pf_module.pairformer_layer_forward

    def msa_spy(params, z, m, *args, **kwargs):
        seen["msa_layer_in"].append(z.dtype)
        return msa_original(params, z, m, *args, **kwargs)

    def pf_spy(params, s, z, *args, **kwargs):
        seen["pairformer_layer_in"].append(z.dtype)
        out_s, out_z = pf_original(params, s, z, *args, **kwargs)
        seen["pairformer_layer_out"].append(out_z.dtype)
        seen["pairformer_single_out"].append(out_s.dtype)
        return out_s, out_z

    monkeypatch.setattr(msa_module, "msa_layer_forward", msa_spy)
    monkeypatch.setattr(pf_module, "pairformer_layer_forward", pf_spy)
    result = _run(monkeypatch, pair_residual_dtype, omit=omit, **kwargs)
    assert all(seen.values()), "no layer ran; the spy recorded nothing"
    return seen, result


@pytest.mark.parametrize("arm", ["float32", None])
def test_the_float32_arm_realises_a_float32_pair_carry(monkeypatch, arm) -> None:
    seen, result = _spy(monkeypatch, arm)
    assert set(seen["msa_layer_in"]) == {jnp.dtype(jnp.float32)}
    assert set(seen["pairformer_layer_in"]) == {jnp.dtype(jnp.float32)}
    assert set(seen["pairformer_layer_out"]) == {jnp.dtype(jnp.float32)}
    assert result["z"].dtype == jnp.float32


@pytest.mark.parametrize("omit", [True, False])
def test_the_released_default_realises_a_bfloat16_pair_carry(
    monkeypatch, omit
) -> None:
    seen, result = _spy(monkeypatch, jnp.bfloat16, omit=omit)
    assert set(seen["msa_layer_in"]) == {jnp.dtype(jnp.bfloat16)}
    assert set(seen["pairformer_layer_in"]) == {jnp.dtype(jnp.bfloat16)}
    assert set(seen["pairformer_layer_out"]) == {jnp.dtype(jnp.bfloat16)}
    # Upstream runs the single branch inside `torch.autocast(enabled=False)`
    # (boltz/model/layers/pairformer.py:105), and it is ~0.1% of the pair
    # tensor's bytes, so it does not move with the pair carry.
    assert set(seen["pairformer_single_out"]) == {jnp.dtype(jnp.float32)}
    # The knob's blast radius stops at the trunk: the conditioner, the
    # confidence module and the affinity head see the dtype they always saw.
    assert result["z"].dtype == jnp.float32
    assert result["s"].dtype == jnp.float32


def test_an_fp32_trunk_keeps_its_fp32_carry_under_the_released_default(
    monkeypatch,
) -> None:
    """`compute_dtype="float32"` must not have to opt out of the new default.

    The narrow storage is only coherent between the narrowed GEMMs it sits
    in, so the released spelling follows the trunk rather than demanding one.
    An FP32 run that never named this option must still be the FP32 program.
    """

    seen, result = _spy(monkeypatch, None, omit=True, trunk_dtype=jnp.float32)
    assert set(seen["msa_layer_in"]) == {jnp.dtype(jnp.float32)}
    assert set(seen["pairformer_layer_out"]) == {jnp.dtype(jnp.float32)}
    assert result["z"].dtype == jnp.float32
    omitted = _run(monkeypatch, None, lower_only=True, trunk_dtype=jnp.float32)
    assert omitted == _run(
        monkeypatch, "float32", lower_only=True, trunk_dtype=jnp.float32
    )


def test_the_default_narrows_pair_shaped_buffers_in_the_lowered_program(
    monkeypatch,
) -> None:
    wide = _run(monkeypatch, "float32", lower_only=True)
    narrowed = _run(monkeypatch, None, lower_only=True, omit=True)
    assert wide != narrowed
    assert narrowed.count(PAIR + "bf16>") > wide.count(PAIR + "bf16>")
    assert narrowed.count(PAIR + "f32>") < wide.count(PAIR + "f32>")
    # Nothing the caller passes in or gets back changes width: every byte this
    # option saves is a temp, and temps are repacked.
    def signature(text):
        head, _, tail = text.partition(") -> (")
        head = head.split("func.func public @main(", 1)[1]
        return (
            head.count("xf32>"),
            head.count("xbf16>"),
            tail.split(")", 1)[0].count("xf32>"),
            tail.split(")", 1)[0].count("xbf16>"),
        )

    assert signature(wide) == signature(narrowed)


def test_the_two_arms_move_the_numbers(monkeypatch) -> None:
    # An arm that computes the same answer is an arm that never fired.
    wide = _run(monkeypatch, "float32")
    narrowed = _run(monkeypatch, None, omit=True)
    assert not np.array_equal(wide["z"], narrowed["z"])
    assert not np.array_equal(wide["s"], narrowed["s"])


def test_auto_follows_the_trunk_width() -> None:
    resolve = trunk_module._resolve_pair_residual_dtype
    default = trunk_module.PAIR_RESIDUAL_DTYPE_DEFAULT
    assert resolve(default, jnp.bfloat16) == jnp.dtype(jnp.bfloat16)
    # An FP32 trunk keeps the FP32 stream rather than failing a request the
    # caller never made: the narrow storage is only coherent between the
    # narrowed GEMMs it sits in.
    assert resolve(default, jnp.float32) is None


def test_the_resolver_is_idempotent() -> None:
    # `predict.py` resolves and `boltz2_trunk_forward` resolves again.
    resolve = trunk_module._resolve_pair_residual_dtype
    for kernel in (jnp.bfloat16, jnp.float32):
        once = resolve(trunk_module.PAIR_RESIDUAL_DTYPE_DEFAULT, kernel)
        assert resolve(once, kernel) == once


def test_float32_resolves_to_the_emit_nothing_sentinel() -> None:
    resolve = trunk_module._resolve_pair_residual_dtype
    for kernel in (jnp.bfloat16, jnp.float32):
        assert resolve("float32", kernel) is None
        assert resolve(jnp.float32, kernel) is None
        # `None` keeps the meaning it had while it was the default spelling.
        # Only *omission* changed what it resolves to, which is why the user
        # boundary refuses `None` while the resolver still reads it.
        assert resolve(None, kernel) is None


def test_narrow_pair_residual_requires_a_bfloat16_trunk() -> None:
    with pytest.raises(ValueError, match="requires a bfloat16 trunk"):
        trunk_module._resolve_pair_residual_dtype(jnp.bfloat16, jnp.float32)


def test_an_unknown_width_is_refused() -> None:
    with pytest.raises(ValueError, match="'auto', 'bfloat16' or 'float32'"):
        trunk_module._resolve_pair_residual_dtype(jnp.float16, jnp.bfloat16)


def test_every_layer_spells_the_same_released_default() -> None:
    """The four places that have to agree, pinned to each other.

    The adapter repeats the value as a literal on purpose -- cache-directory
    selection must not import the model runtime -- so nothing but a test
    stops the two from drifting apart.
    """

    import inspect

    from foldjax.backends import boltz2 as backend_module
    from foldjax.models.boltz2 import api as native_api

    default = trunk_module.PAIR_RESIDUAL_DTYPE_DEFAULT
    signatures = (
        native_api.predict,
        trunk_module.boltz2_trunk_forward,
        trunk_module.boltz2_sample_forward,
    )
    for function in signatures:
        parameter = inspect.signature(function).parameters["pair_residual_dtype"]
        assert parameter.default == default, function.__name__
    released = backend_module._RELEASED_COMPILE_DEFAULTS["pair_residual_dtype"]
    assert released == default
    assert type(released) is type(default)


def test_the_user_boundary_refuses_the_inverted_spelling() -> None:
    """`None` meant float32 and would now read as "the default"."""

    from foldjax.backends.boltz2 import Boltz2Backend

    backend = Boltz2Backend()
    with pytest.raises(ValueError, match="spell the arm you want"):
        backend.validate_native_options({"pair_residual_dtype": None})
    with pytest.raises(ValueError, match="requires"):
        backend.validate_native_options(
            {"pair_residual_dtype": "bfloat16", "compute_dtype": "float32"}
        )
    # Both widths, and the released spelling, are sayable outright.
    for value in ("auto", "bfloat16", "float32"):
        backend.validate_native_options({"pair_residual_dtype": value})
    # The wide arm needs no permission from `compute_dtype`, and an FP32
    # trunk that never names the option is not asked to opt out of it.
    backend.validate_native_options(
        {"pair_residual_dtype": "float32", "compute_dtype": "float32"}
    )
    backend.validate_native_options({"compute_dtype": "float32"})


def test_the_native_api_refuses_the_inverted_spelling(tmp_path) -> None:
    from foldjax.models.boltz2 import api as native_api

    with pytest.raises(ValueError, match="spell the arm you want"):
        native_api.predict(
            seq=["ACD"],
            weights=tmp_path / "unused",
            mols=tmp_path,
            pair_residual_dtype=None,
        )


def test_an_fp32_request_clears_the_native_validation(monkeypatch, tmp_path) -> None:
    """The coupling rule must not fire on a default the caller never set."""

    from foldjax.models.boltz2 import api as native_api

    class ReachedError(RuntimeError):
        pass

    def reached(**kwargs):
        raise ReachedError

    monkeypatch.setattr(native_api, "featurize", reached)
    for options in ({"compute_dtype": "float32"}, {"pair_residual_dtype": "float32"}):
        with pytest.raises(ReachedError):
            native_api.predict(
                seq=["ACD"], weights=tmp_path / "unused", mols=tmp_path, **options
            )


def test_both_arms_keep_the_bfloat16_triangle_contraction(monkeypatch) -> None:
    """The narrowed carry must not be mistaken for a different AMP policy.

    `triangle_multiplication_forward` used to decide "is this autocast?" by
    reading the activation width. A BF16 pair residual is still autocast, but
    that reading called it FP32-model and ran the contraction in float32 --
    a *wider* program than either arm intends. The assertion is the
    contraction operand's dtype, not the carry's, and it runs on the omitted
    arm because that is now the released one.
    """

    monkeypatch.setenv("BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND", "xla")
    seen: dict[str, list] = {}
    original = tri_module._chunked_triangle_einsum

    def spy(a, b, direction, chunk_size):
        seen.setdefault(direction, []).append(a.dtype)
        return original(a, b, direction, chunk_size)

    monkeypatch.setattr(tri_module, "_chunked_triangle_einsum", spy)
    for arm, omit in _ARMS:
        seen.clear()
        _run(monkeypatch, arm, omit=omit)
        assert seen, "triangle multiplication never ran"
        for direction, dtypes in seen.items():
            assert set(dtypes) == {jnp.dtype(jnp.bfloat16)}, (arm, omit, direction)


def test_the_default_keeps_the_fused_cueq_native_amp_branch(monkeypatch) -> None:
    """Same question on the released backend, which is cuEq, not XLA."""

    cueq = pytest.importorskip(
        "foldjax.models.boltz2.models.triangle.triangle_cueq"
    )
    try:
        cueq._load_cueq()
    except RuntimeError as error:  # no cuEquivariance in this environment
        pytest.skip(str(error))
    monkeypatch.delenv("BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND", raising=False)
    seen: list = []
    original = cueq._cueq_triangle_native_amp

    def spy(cuex, params, x, mask, direction, *, eps):
        seen.append(x.dtype)
        return original(cuex, params, x, mask, direction, eps=eps)

    monkeypatch.setattr(cueq, "_cueq_triangle_native_amp", spy)

    # The fused norm returns the width it is given, unlike `nn.LayerNorm`.
    # On the released backend that makes the pair normalisation inside
    # triangle multiplication bfloat16 by default, where plain XLA triangle
    # multiplication keeps float32 -- the two backends diverge further under
    # the default than they did while it was opt-in, and the docs say so, so
    # it is asserted rather than assumed.
    normed: list = []
    load_original = cueq._load_cueq_amp_primitives

    def load_spy():
        norm, gemm, gemm_dual = load_original()

        def norm_spy(x, *args, **kwargs):
            out = norm(x, *args, **kwargs)
            normed.append((x.dtype, out.dtype))
            return out

        return norm_spy, gemm, gemm_dual

    monkeypatch.setattr(cueq, "_load_cueq_amp_primitives", load_spy)

    seen.clear()
    normed.clear()
    _run(monkeypatch, "float32")
    wide_calls = len(seen)
    assert wide_calls, "the fused native-AMP branch never ran"
    assert set(seen) == {jnp.dtype(jnp.float32)}
    assert (jnp.dtype(jnp.float32), jnp.dtype(jnp.float32)) in normed

    seen.clear()
    normed.clear()
    _run(monkeypatch, None, omit=True)
    # Same branch, same number of entries -- only the stored width differs.
    assert len(seen) == wide_calls
    assert set(seen) == {jnp.dtype(jnp.bfloat16)}
    assert (jnp.dtype(jnp.float32), jnp.dtype(jnp.float32)) not in normed
    assert set(normed) == {(jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.bfloat16))}


@pytest.mark.parametrize(
    ("x_dtype", "kernel_dtype", "expected"),
    [
        (jnp.float32, jnp.bfloat16, True),
        (jnp.bfloat16, jnp.bfloat16, False),
        (jnp.float32, jnp.float32, False),
    ],
)
def test_native_amp_inference_is_unchanged_when_unset(
    x_dtype, kernel_dtype, expected
) -> None:
    x = jnp.zeros((2, 2), x_dtype)
    kernel = jnp.zeros((2, 2), kernel_dtype)
    assert tri_module.resolve_native_amp(x, kernel, None) is expected
    assert tri_module.resolve_native_amp(x, kernel, True) is True
    assert tri_module.resolve_native_amp(x, kernel, False) is False


def test_the_knob_rounds_no_pair_bias_that_was_not_already_rounded(
    monkeypatch,
) -> None:
    """The rule that cost a whole chain on Protenix, checked on both biases.

    A bias re-projected every block and added under an exponential must not
    pick up a rounding it did not have. Both of Boltz-2's do their arithmetic
    at the same width on the narrow arm as on the wide one: the single
    track's because its projection is inside the autocast-disabled parameter
    island and `pairformer_layer_forward` hands it an explicitly widened
    pair, and triangle attention's because its projection kernel is BF16 in
    the released run already, which is upstream's autocast Linear.

    The narrow arm is now the omitted one, so that is the arm measured here.
    """

    from foldjax.models.boltz2.models.primitives import attention as attn_module
    from foldjax.models.boltz2.models.triangle import (
        triangle_attention as tri_att_module,
    )

    seen: dict[str, list] = {"single": [], "triangle": []}
    single_original = attn_module._attention_qblock
    triangle_original = tri_att_module._attention_block

    def single_spy(q, k, v, bias, key_mask, scale):
        seen["single"].append(bias.dtype)
        return single_original(q, k, v, bias, key_mask, scale)

    def triangle_spy(q_blk, k, v, tri_bias, mask_bias_blk, *args, **kwargs):
        seen["triangle"].append(tri_bias.dtype)
        return triangle_original(
            q_blk, k, v, tri_bias, mask_bias_blk, *args, **kwargs
        )

    monkeypatch.setattr(attn_module, "_attention_qblock", single_spy)
    monkeypatch.setattr(tri_att_module, "_attention_block", triangle_spy)
    widths = []
    for arm, omit in _ARMS:
        seen["single"].clear()
        seen["triangle"].clear()
        _run(monkeypatch, arm, omit=omit)
        assert seen["single"] and seen["triangle"], "no softmax ran"
        widths.append((set(seen["single"]), set(seen["triangle"])))
    wide, released = widths
    assert wide == released
    assert released == ({jnp.dtype(jnp.float32)}, {jnp.dtype(jnp.bfloat16)})
