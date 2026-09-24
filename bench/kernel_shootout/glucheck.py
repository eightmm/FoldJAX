"""The GLU call sites `glu_backend=pallas` reaches beyond the pair transition.

The shootout measured the pair transition only. The one option also routes
the MSA and single transitions and the diffusion conditioned transitions. Those
run the unit-only kernel, or the whole-transition kernel at widths it was not
tuned for. This times each shape under xla / tokamax / pallas, per call, so an
x43 row can be split into the pair gain and these. One process per cell.

    python glucheck.py OUT_DIR           # driver
"""

# ruff: noqa: E501

import json
import subprocess
import sys
import time
from pathlib import Path

#: (label, rows, K, F, dtype, activation, transition?)
SHAPES = [
    ("protenix-diffusion-f32", 5 * 2096, 768, 1536, "float32", "silu", False),
    ("protenix-diffusion-bf16", 5 * 2096, 768, 1536, "bfloat16", "silu", False),
    ("single-transition-384", 2096, 384, 1536, "bfloat16", "silu", True),
    ("boltz2-msa-transition-64", 8192 * 2096 // 16, 64, 256, "bfloat16", "silu", True),
]
BACKENDS = ("xla", "tokamax", "pallas")


def cell(label: str, backend: str) -> dict:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models import _glu, _pallas_pair

    jax.config.update("jax_default_matmul_precision", "high")
    _, rows, k, f, dtype, act, transition = next(s for s in SHAPES if s[0] == label)
    dt = jnp.dtype(dtype)
    activation = getattr(jax.nn, act)
    key = jax.random.split(jax.random.key(0), 4)
    x = jax.random.normal(key[0], (rows, k), jnp.float32).astype(dt)
    w1 = (jax.random.normal(key[1], (k, f)) / k**0.5).astype(dt)
    w2 = (jax.random.normal(key[2], (k, f)) / k**0.5).astype(dt)
    w3 = (jax.random.normal(key[3], (f, k)) / f**0.5).astype(dt)
    norm = (jnp.ones(k, jnp.float32), jnp.zeros(k, jnp.float32))
    if transition and backend == "pallas":

        def fn(x, w1, w2, w3):
            return _pallas_pair.transition(x, norm, w1, w2, w3, eps=1e-5)
    elif transition:

        def fn(x, w1, w2, w3):
            m = x.astype(jnp.float32)
            m = (m - m.mean(-1, keepdims=True)) * jax.lax.rsqrt(m.var(-1, keepdims=True) + 1e-5)
            h = _glu.gated_linear_unit(m.astype(dt), w1, w2, activation, backend=backend)
            return h @ w3
    else:

        def fn(x, w1, w2, w3):
            return _glu.gated_linear_unit(x, w1, w2, activation, backend=backend)

    compiled = jax.jit(fn).lower(x, w1, w2, w3).compile()
    compiled(x, w1, w2, w3).block_until_ready()
    times = []
    for _ in range(7):
        t0 = time.perf_counter()
        compiled(x, w1, w2, w3).block_until_ready()
        times.append(time.perf_counter() - t0)
    ma = compiled.memory_analysis()
    return {"label": label, "backend": backend, "ms": round(float(np.median(times)) * 1e3, 3),
            "temp_mib": round(ma.temp_size_in_bytes / 2**20, 1) if ma else None}


if __name__ == "__main__":
    if sys.argv[1] == "--cell":
        try:
            rec = cell(sys.argv[2], sys.argv[3])
        except BaseException as error:  # noqa: BLE001
            rec = {"label": sys.argv[2], "backend": sys.argv[3], "error": repr(error)[:600]}
        print(json.dumps(rec), flush=True)
        sys.exit(0)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "glu.jsonl", "a") as log:
        for label, *_ in SHAPES:
            for backend in BACKENDS:
                proc = subprocess.run([sys.executable, __file__, "--cell", label, backend],
                                      capture_output=True, text=True, timeout=1200)
                line = (proc.stdout.strip().splitlines() or [json.dumps(
                    {"label": label, "backend": backend, "error": proc.stderr[-600:]})])[-1]
                print(line, flush=True)
                log.write(line + "\n")
