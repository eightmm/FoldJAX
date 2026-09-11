"""``pair_residual_dtype`` must narrow the pair carry, and only when asked.

Two failure modes are in scope and neither is visible from a config field.

The first is a default that drifts. `compute_dtype="bfloat16"` already narrows
every trunk GEMM; what this option changes is the *storage* of the pair
residual, which the released trunk keeps in float32 -- `z_init` inherits it
from ContactConditioning's `encoding_unspecified` (an nn.Parameter, not a
Linear kernel, so `_cast_trunk_params` leaves it alone) and OuterProductMean
re-promotes it above 384 tokens. So ``test_released_default_emits_nothing``
lowers the shipped trunk twice in one process -- once as shipped, once with
the option's own helper replaced by a strict identity, i.e. with the feature
physically absent -- and compares the programs as text.

The second is an arm that never fires. A patched arm measuring dead code is
how a knob gets reported as free, so every other test here reads ``.dtype``
off an array the trunk actually produced, or reads the lowered program.

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


def _run(monkeypatch, pair_residual_dtype, *, lower_only=False, omit=False):
    params, feats, s_inputs = _build()
    params = trunk_module._cast_trunk_params(params, jnp.bfloat16)
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


def test_released_default_emits_nothing(monkeypatch) -> None:
    shipped = _run(monkeypatch, None, lower_only=True)
    omitted = _run(monkeypatch, None, lower_only=True, omit=True)
    assert shipped == omitted

    # Replace the option's own helper with a strict identity in every module
    # that calls it: the feature is then physically absent from the trace. If
    # the released default emitted even one convert, this text would differ.
    for module in (trunk_module, msa_module, pf_module):
        monkeypatch.setattr(module, "_residual_cast", lambda x, dtype: x)
    assert _run(monkeypatch, None, lower_only=True) == shipped


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16, jnp.int32])
def test_residual_cast_is_the_identity_object_when_unset(dtype) -> None:
    x = jnp.zeros((2, 2), dtype)
    assert _common.residual_cast(x, None) is x
    assert _common.residual_cast(x, dtype) is x


def _spy(monkeypatch, pair_residual_dtype):
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
    result = _run(monkeypatch, pair_residual_dtype)
    assert all(seen.values()), "no layer ran; the spy recorded nothing"
    return seen, result


def test_default_realises_a_float32_pair_carry(monkeypatch) -> None:
    seen, result = _spy(monkeypatch, None)
    assert set(seen["msa_layer_in"]) == {jnp.dtype(jnp.float32)}
    assert set(seen["pairformer_layer_in"]) == {jnp.dtype(jnp.float32)}
    assert set(seen["pairformer_layer_out"]) == {jnp.dtype(jnp.float32)}
    assert result["z"].dtype == jnp.float32


def test_opt_in_realises_a_bfloat16_pair_carry(monkeypatch) -> None:
    seen, result = _spy(monkeypatch, jnp.bfloat16)
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


def test_opt_in_narrows_pair_shaped_buffers_in_the_lowered_program(
    monkeypatch,
) -> None:
    default = _run(monkeypatch, None, lower_only=True)
    narrowed = _run(monkeypatch, jnp.bfloat16, lower_only=True)
    assert default != narrowed
    assert narrowed.count(PAIR + "bf16>") > default.count(PAIR + "bf16>")
    assert narrowed.count(PAIR + "f32>") < default.count(PAIR + "f32>")
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

    assert signature(default) == signature(narrowed)


def test_opt_in_moves_the_numbers(monkeypatch) -> None:
    # An arm that computes the same answer is an arm that never fired.
    default = _run(monkeypatch, None)
    narrowed = _run(monkeypatch, jnp.bfloat16)
    assert not np.array_equal(default["z"], narrowed["z"])
    assert not np.array_equal(default["s"], narrowed["s"])


def test_float32_is_not_spellable() -> None:
    with pytest.raises(ValueError, match="null is its spelling"):
        trunk_module._resolve_pair_residual_dtype(jnp.float32, jnp.bfloat16)


def test_narrow_pair_residual_requires_a_bfloat16_trunk() -> None:
    with pytest.raises(ValueError, match="requires a bfloat16 trunk"):
        trunk_module._resolve_pair_residual_dtype(jnp.bfloat16, jnp.float32)


def test_unset_resolves_to_none() -> None:
    assert trunk_module._resolve_pair_residual_dtype(None, jnp.float32) is None
    assert trunk_module._resolve_pair_residual_dtype(None, jnp.bfloat16) is None


def test_backend_refuses_float32_and_an_fp32_trunk() -> None:
    from foldjax.backends.boltz2 import Boltz2Backend

    backend = Boltz2Backend()
    with pytest.raises(ValueError, match="pair_residual_dtype"):
        backend.validate_native_options({"pair_residual_dtype": "float32"})
    with pytest.raises(ValueError, match="requires"):
        backend.validate_native_options(
            {"pair_residual_dtype": "bfloat16", "compute_dtype": "float32"}
        )
    backend.validate_native_options({"pair_residual_dtype": "bfloat16"})
    backend.validate_native_options({"pair_residual_dtype": None})


def test_opt_in_keeps_the_bfloat16_triangle_contraction(monkeypatch) -> None:
    """The narrowed carry must not be mistaken for a different AMP policy.

    `triangle_multiplication_forward` used to decide "is this autocast?" by
    reading the activation width. A BF16 pair residual is still autocast, but
    that reading called it FP32-model and ran the contraction in float32 --
    an arm that fires and measures a *wider* program than the default. The
    assertion is the contraction operand's dtype, not the carry's.
    """

    monkeypatch.setenv("BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND", "xla")
    seen: dict[str, list] = {}
    original = tri_module._chunked_triangle_einsum

    def spy(a, b, direction, chunk_size):
        seen.setdefault(direction, []).append(a.dtype)
        return original(a, b, direction, chunk_size)

    monkeypatch.setattr(tri_module, "_chunked_triangle_einsum", spy)
    for arm in (None, jnp.bfloat16):
        seen.clear()
        _run(monkeypatch, arm)
        assert seen, "triangle multiplication never ran"
        for direction, dtypes in seen.items():
            assert set(dtypes) == {jnp.dtype(jnp.bfloat16)}, (arm, direction)


def test_opt_in_keeps_the_fused_cueq_native_amp_branch(monkeypatch) -> None:
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
    # That is a second precision change the pin makes on the released
    # backend, and the docs say so, so it is asserted rather than assumed.
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
    _run(monkeypatch, None)
    default_calls = len(seen)
    assert default_calls, "the fused native-AMP branch never ran"
    assert set(seen) == {jnp.dtype(jnp.float32)}
    assert (jnp.dtype(jnp.float32), jnp.dtype(jnp.float32)) in normed

    seen.clear()
    normed.clear()
    _run(monkeypatch, jnp.bfloat16)
    # Same branch, same number of entries -- only the stored width differs.
    assert len(seen) == default_calls
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
    at the same width with the pin on as with it off: the single track's
    because its projection is inside the autocast-disabled parameter island
    and `pairformer_layer_forward` hands it an explicitly widened pair, and
    triangle attention's because its projection kernel is BF16 in the
    released run already, which is upstream's autocast Linear.
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
    widths = {}
    for arm in (None, jnp.bfloat16):
        seen["single"].clear()
        seen["triangle"].clear()
        _run(monkeypatch, arm)
        assert seen["single"] and seen["triangle"], "no softmax ran"
        widths[arm] = (set(seen["single"]), set(seen["triangle"]))
    assert widths[None] == widths[jnp.bfloat16]
    assert widths[None] == ({jnp.dtype(jnp.float32)}, {jnp.dtype(jnp.bfloat16)})
