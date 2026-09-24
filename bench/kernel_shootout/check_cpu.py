"""CPU checks for the prototype kernels: interpret-mode numerics + Triton lowering.

1. Interpret mode (``fjk.INTERPRET = True``) against plain-JAX float32
   references, at sizes that do not divide the blocks (masked edges).
2. Lowering to the Triton dialect for ``cuda`` with the GPU info pinned to the
   RTX PRO 6000 (sm_120): catches Pallas->Triton lowering errors without a
   card.  PTX generation is not exercised (that happens in XLA's compile).

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= python check_cpu.py
"""

# ruff: noqa: E501, N806 -- bench tooling: long table f-strings, A/B array names

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fjk  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")


def ln(x, w, b, eps):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + eps) * w + b


def ref_attention(q, k, v, bias, key_mask, scale):
    q, k, v = (t.astype(jnp.float32) for t in (q, k, v))
    s = (
        jnp.einsum("rqhd,rkhd->rhqk", q, k) * scale
        + bias[None]
        + key_mask[:, None, None, :]
    )
    p = jax.nn.softmax(s, axis=-1)
    return jnp.einsum("rhqk,rkhd->rqhd", p, v)


def ref_trimul(x, mask, prm, direction, eps):
    f = lambda t: t.astype(jnp.float32)  # noqa: E731
    x = f(x)
    xn = ln(x, f(prm["norm_in"]["scale"]), f(prm["norm_in"]["bias"]), eps)
    ab = jax.nn.sigmoid(xn @ f(prm["g_in"]["kernel"])) * (xn @ f(prm["p_in"]["kernel"]))
    ab = ab * mask[..., None]
    h = ab.shape[-1] // 2
    a, b = ab[..., :h], ab[..., h:]
    if direction == "outgoing":
        e = jnp.einsum("ikc,jkc->ijc", a, b)
    else:
        e = jnp.einsum("kic,kjc->ijc", a, b)
    en = ln(e, f(prm["norm_out"]["scale"]), f(prm["norm_out"]["bias"]), eps)
    return (en @ f(prm["p_out"]["kernel"])) * jax.nn.sigmoid(
        xn @ f(prm["g_out"]["kernel"])
    )


def ref_transition(x, w, b, w1, w2, w3, eps):
    f = lambda t: t.astype(jnp.float32)  # noqa: E731
    xn = ln(f(x), f(w), f(b), eps)
    return (jax.nn.silu(xn @ f(w1)) * (xn @ f(w2))) @ f(w3)


def rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))


def interpret_checks() -> list[str]:
    fjk.INTERPRET = True
    rng = np.random.default_rng(0)
    failures = []
    bf = jnp.bfloat16

    # Attention: R=3 rows, N=37 (not a multiple of 16), H=2, D=32.
    r, n, h, d = 5, 37, 2, 32
    q, k, v = (jnp.asarray(rng.normal(size=(r, n, h, d)), bf) for _ in range(3))
    bias = jnp.asarray(rng.normal(size=(h, n, n)) * 2, jnp.float32)
    km = np.zeros((r, n), np.float32)
    km[:, n - 3 :] = -1e9
    km[1, 5] = -1e9
    km = jnp.asarray(km)
    expect = ref_attention(q, k, v, bias, km, d**-0.5)
    for rb in (1, 2, 3):
        out = fjk.flash_triangle_attention(
            q, k, v, bias, km, scale=d**-0.5, rb=rb, bq=16, bk=16
        )
        err = rel(out, expect)
        print(f"attention rb={rb} interpret rel err {err:.2e}")
        if not err < 2e-2:
            failures.append(f"attention rb={rb} {err}")

    for c in (32, 256):  # one K chunk, and two (the OpenDDE-width code path)
        failures += _pair_checks(rng, c)
    fjk.INTERPRET = False
    return failures


def _pair_checks(rng, c) -> list[str]:
    failures = []
    bf = jnp.bfloat16
    # Triangle multiplication: N=21 -> P=441, bm=64 overhangs.
    n, hid = 21, c
    x = jnp.asarray(rng.normal(size=(n, n, c)), bf)
    mask = np.ones((n, n), np.float32)
    mask[n - 2 :, :] = 0
    mask[:, n - 2 :] = 0
    mask = jnp.asarray(mask)
    w = lambda *s: jnp.asarray(rng.normal(size=s) / np.sqrt(s[0]), bf)  # noqa: E731
    prm = {
        "norm_in": {
            "scale": jnp.asarray(1 + 0.1 * rng.normal(size=c), jnp.float32),
            "bias": jnp.asarray(0.1 * rng.normal(size=c), jnp.float32),
        },
        "norm_out": {
            "scale": jnp.asarray(1 + 0.1 * rng.normal(size=hid), jnp.float32),
            "bias": jnp.asarray(0.1 * rng.normal(size=hid), jnp.float32),
        },
        "p_in": {"kernel": w(c, 2 * hid)},
        "g_in": {"kernel": w(c, 2 * hid)},
        "p_out": {"kernel": w(hid, c)},
        "g_out": {"kernel": w(c, c)},
    }
    for direction in ("outgoing", "incoming"):
        out = fjk.triangle_multiplication_k1k2(
            x, mask, prm, direction=direction, eps=1e-5
        )
        err = rel(out, ref_trimul(x, mask, prm, direction, 1e-5))
        print(f"trimul c={c} {direction} interpret rel err {err:.2e}")
        if not err < 3e-2:
            failures.append(f"trimul {direction} {err}")

    # Transition: P=441, C=32, F=128 (fc=64 -> 2 chunks).
    lw = jnp.asarray(1 + 0.1 * rng.normal(size=c), jnp.float32)
    lb = jnp.asarray(0.1 * rng.normal(size=c), jnp.float32)
    w1, w2, w3 = w(c, 4 * c), w(c, 4 * c), w(4 * c, c)
    flat = x.reshape(n * n, c)
    out = fjk.fused_transition(flat, lw, lb, w1, w2, w3, eps=1e-5, bm=64, fc=64)
    err = rel(out, ref_transition(flat, lw, lb, w1, w2, w3, 1e-5))
    print(f"transition c={c} interpret rel err {err:.2e}")
    if not err < 3e-2:
        failures.append(f"transition c={c} {err}")
    return failures


def lowering_checks() -> list[str]:
    """Lower each kernel at production-like block sizes to the Triton dialect."""
    from jax._src.pallas.triton import gpu_info, pallas_call_registration

    info = gpu_info.get_gpu_info_from_version(gpu_info.GpuVersion.RTX_PRO_6000)
    pallas_call_registration.gpu_info_lib.get_gpu_info = lambda: info
    print("pinned gpu info:", info)
    failures = []
    bf = jnp.bfloat16
    S = jax.ShapeDtypeStruct
    n = 100
    cases = {
        "attention": (
            lambda q, k, v, b, m: fjk.flash_triangle_attention(
                q, k, v, b, m, scale=0.17, rb=2, bq=64, bk=64
            ),
            (S((8, n, 4, 32), bf),) * 3
            + (S((4, n, n), jnp.float32), S((8, n), jnp.float32)),
        ),
        "trimul_k1": (
            lambda x, m, w, b, wp, wg: fjk.trimul_k1(
                x, m, w, b, wp, wg, eps=1e-5, bm=64
            ),
            (
                S((n * n, 128), bf),
                S((n * n,), jnp.float32),
                S((128,), jnp.float32),
                S((128,), jnp.float32),
                S((128, 256), bf),
                S((128, 256), bf),
            ),
        ),
        "trimul_k2": (
            lambda e, x, a, b, c, d, wp, wg: fjk.trimul_k2(
                e, x, a, b, c, d, wp, wg, eps=1e-5, bm=64
            ),
            (S((128, n * n), bf), S((n * n, 128), bf))
            + (S((128,), jnp.float32),) * 4
            + (S((128, 128), bf), S((128, 128), bf)),
        ),
        "transition": (
            lambda x, w, b, w1, w2, w3: fjk.fused_transition(
                x, w, b, w1, w2, w3, eps=1e-5, bm=64, fc=128
            ),
            (
                S((n * n, 128), bf),
                S((128,), jnp.float32),
                S((128,), jnp.float32),
                S((128, 512), bf),
                S((128, 512), bf),
                S((512, 128), bf),
            ),
        ),
        "attention_h12": (
            lambda q, k, v, b, m: fjk.flash_triangle_attention(
                q, k, v, b, m, scale=0.17, rb=2, bq=64, bk=64
            ),
            (S((8, n, 12, 32), bf),) * 3 + (S((12, n, n), bf), S((8, n), jnp.float32)),
        ),
        "trimul_k1_c384": (
            lambda x, m, w, b, wp, wg: fjk.trimul_k1(
                x, m, w, b, wp, wg, eps=1e-5, bm=32
            ),
            (
                S((n * n, 384), bf),
                S((n * n,), jnp.float32),
                S((384,), jnp.float32),
                S((384,), jnp.float32),
                S((384, 768), bf),
                S((384, 768), bf),
            ),
        ),
        "trimul_k2_c384": (
            lambda e, x, a, b, c, d, wp, wg: fjk.trimul_k2(
                e, x, a, b, c, d, wp, wg, eps=1e-5, bm=32
            ),
            (S((384, n * n), bf), S((n * n, 384), bf))
            + (S((384,), jnp.float32),) * 4
            + (S((384, 384), bf), S((384, 384), bf)),
        ),
        "transition_c384": (
            lambda x, w, b, w1, w2, w3: fjk.fused_transition(
                x, w, b, w1, w2, w3, eps=1e-5, bm=32, fc=128
            ),
            (
                S((n * n, 384), bf),
                S((384,), jnp.float32),
                S((384,), jnp.float32),
                S((384, 1536), bf),
                S((384, 1536), bf),
                S((1536, 384), bf),
            ),
        ),
    }
    for name, (fn, args) in cases.items():
        try:
            text = (
                jax.jit(fn).trace(*args).lower(lowering_platforms=("cuda",)).as_text()
            )
            ok = "triton" in text.lower()
            print(
                f"lowering {name}: ok ({len(text)} chars, triton call {'found' if ok else 'MISSING'})"
            )
            if not ok:
                failures.append(f"lowering {name}: no triton call")
        except Exception as error:  # noqa: BLE001
            print(f"lowering {name}: FAILED {error!r}"[:1500])
            failures.append(f"lowering {name}")
    return failures


if __name__ == "__main__":
    fails = interpret_checks() + lowering_checks()
    print("FAILURES:", fails if fails else "none")
    sys.exit(1 if fails else 0)
