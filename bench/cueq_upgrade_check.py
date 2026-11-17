"""GPU upgrade check comparing cuEquivariance 0.11.1 and 0.13.0 on FoldJAX kernels.

Run once in each environment on the same GPU, then compare the two JSON artifacts on
CPU::

    python bench/cueq_upgrade_check.py --out results/cueq_0111.json
    python bench/cueq_upgrade_check.py --out results/cueq_0130.json
    python bench/cueq_upgrade_check.py --compare \\
        results/cueq_0111.json results/cueq_0130.json

In ``--out`` mode, JAX, Triton, and cuEquivariance write compile caches under the
user cache directories (``XDG_CACHE_HOME``, ``TRITON_CACHE_DIR``) rather than
under ``--out``. Set ``XDG_CACHE_HOME`` and ``TRITON_CACHE_DIR`` to a private
directory per run when the default cache is shared.

Importing this module and running ``--compare`` require only NumPy and the Python
standard library. ``--out`` imports JAX, FoldJAX, and cuEquivariance lazily and
requires an active GPU backend.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import io
import json
import math
import os
import statistics
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

ATT_BATCH = 1
ATT_HEADS = 4
ATT_HEAD_DIM = 32
ATT_LENGTHS = (256, 1003, 1024, 2100, 3012)
ATT_DTYPES = ("bf16", "fp32")
ATT_MASKS = ("all_valid", "last5pct_invalid")

MUL_BATCH = 1
MUL_CHANNELS = 128
MUL_LENGTHS = (256, 1024, 2100)
MUL_DIRECTIONS = ("outgoing", "incoming")
MUL_DTYPE_MODES = ("bf16", "bf16_fp32norm", "fp32")

LN_LAYOUTS = (
    ("bijd_to_bijd", "bijd->bijd"),
    ("dbij_to_bijd", "dbij->bijd"),
)
LN_DTYPES = ("bf16", "fp32")

DEFAULT_WARMUP = 3
DEFAULT_REPEATS = 10
DEFAULT_MAX_SAMPLE_ELEMENTS = 1_000_000
MAX_REPORTED_ITEMS = 5
NORM_EPS = 1e-5
MASK_BIAS_INF = 1e9


@dataclass(frozen=True)
class CaseSpec:
    """One benchmark and numerical parity case."""

    name: str
    family: str
    length: int
    dtype: str
    variant: str


def default_case_specs() -> list[CaseSpec]:
    """Return the ordered panel of cuEquivariance cases FoldJAX depends on."""
    specs: list[CaseSpec] = []
    for n in ATT_LENGTHS:
        for dtype in ATT_DTYPES:
            for mask_kind in ATT_MASKS:
                specs.append(
                    CaseSpec(
                        name=f"a_raw_tri_att_n{n}_{dtype}_{mask_kind}",
                        family="raw_triangle_attention",
                        length=n,
                        dtype=dtype,
                        variant=mask_kind,
                    )
                )
    for n in ATT_LENGTHS:
        for mask_kind in ATT_MASKS:
            specs.append(
                CaseSpec(
                    name=f"b_cueq_att_core_n{n}_bf16_{mask_kind}",
                    family="cueq_attention_core",
                    length=n,
                    dtype="bf16",
                    variant=mask_kind,
                )
            )
    for n in MUL_LENGTHS:
        for direction in MUL_DIRECTIONS:
            for dtype_mode in MUL_DTYPE_MODES:
                specs.append(
                    CaseSpec(
                        name=f"c_fused_tri_mul_n{n}_{direction}_{dtype_mode}",
                        family="fused_triangle_multiplication",
                        length=n,
                        dtype=dtype_mode,
                        variant=direction,
                    )
                )
            specs.append(
                CaseSpec(
                    name=f"c_boltz2_amp_tri_mul_n{n}_{direction}",
                    family="boltz2_amp_triangle_multiplication",
                    length=n,
                    dtype="fp32_bf16amp",
                    variant=direction,
                )
            )
    for n in MUL_LENGTHS:
        for layout_tag, layout in LN_LAYOUTS:
            for dtype in LN_DTYPES:
                specs.append(
                    CaseSpec(
                        name=f"d_ln_transpose_{layout_tag}_n{n}_{dtype}",
                        family="layer_norm_transpose",
                        length=n,
                        dtype=dtype,
                        variant=layout,
                    )
                )
    return specs


def select_cases(
    specs: Sequence[CaseSpec], case_filter: str | None = None
) -> list[CaseSpec]:
    """Filter cases by substring; raise if nothing matches."""
    if not case_filter:
        selected = list(specs)
    else:
        selected = [spec for spec in specs if case_filter in spec.name]
    if not selected:
        raise ValueError(f"no benchmark cases matched filter {case_filter!r}")
    return selected


def validate_run_config(
    *, warmup: int, repeats: int, max_sample_elements: int
) -> None:
    """Validate timing and sampling bounds before running any case."""
    if isinstance(warmup, bool) or warmup < 0:
        raise ValueError(f"warmup must be >= 0; got {warmup!r}")
    if isinstance(repeats, bool) or repeats < 1:
        raise ValueError(f"repeats must be >= 1; got {repeats!r}")
    if isinstance(max_sample_elements, bool) or max_sample_elements < 1:
        raise ValueError(
            f"max_sample_elements must be >= 1; got {max_sample_elements!r}"
        )


def extract_sample(
    array: np.ndarray, max_elements: int = DEFAULT_MAX_SAMPLE_ELEMENTS
) -> tuple[np.ndarray, int]:
    """Return a float32 fixed-stride 1D sample of at most ``max_elements``."""
    if isinstance(max_elements, bool) or max_elements < 1:
        raise ValueError(f"max_elements must be >= 1; got {max_elements!r}")
    flat = np.ascontiguousarray(array).reshape(-1)
    total = int(flat.size)
    if total == 0:
        return np.empty((0,), dtype=np.float32), 1
    if total <= max_elements:
        stride = 1
        sampled = flat
    else:
        stride = (total + max_elements - 1) // max_elements
        sampled = flat[::stride]
    return np.ascontiguousarray(sampled, dtype=np.float32), stride


def summarize_array(
    array: np.ndarray,
    *,
    dtype_name: str,
    max_sample_elements: int = DEFAULT_MAX_SAMPLE_ELEMENTS,
) -> tuple[dict[str, Any], np.ndarray]:
    """Summarize raw array bytes, magnitude, finiteness, and fixed-stride sample."""
    contig = np.ascontiguousarray(array)
    raw_bytes = contig.view(np.uint8).tobytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    values_f32 = contig.astype(np.float32)
    all_finite = bool(np.isfinite(values_f32).all())
    if values_f32.size == 0:
        max_abs: float | None = 0.0
    elif all_finite:
        max_abs = float(np.max(np.abs(values_f32)))
    else:
        max_abs = None
    sample, stride = extract_sample(contig, max_elements=max_sample_elements)
    summary = {
        "shape": [int(dim) for dim in contig.shape],
        "dtype": dtype_name,
        "sha256": sha256,
        "max_abs": max_abs,
        "all_finite": all_finite,
        "sample_elements": int(sample.size),
        "sample_stride": int(stride),
    }
    return summary, sample


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write ``payload`` to ``path`` via a temp file and ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_raw = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_raw)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def write_sample_npy(path: Path, sample: np.ndarray) -> str:
    """Serialize ``sample`` as ``.npy`` atomically and return its SHA-256 hex digest."""
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(sample, dtype=np.float32), allow_pickle=False)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    _atomic_write_bytes(path, payload)
    return digest


def sample_filename(out_path: Path, case_name: str) -> str:
    """Return the sidecar ``.npy`` filename next to ``out_path`` for ``case_name``."""
    return f"{out_path.stem}.{case_name}.npy"


def _existing_outputs(out_path: Path) -> list[Path]:
    """Return existing report or sidecar paths next to ``out_path``."""
    existing: list[Path] = []
    if out_path.exists() or out_path.is_symlink():
        existing.append(out_path)
    parent = out_path.parent
    if parent.is_dir() and out_path.stem:
        prefix = f"{out_path.stem}."
        existing.extend(
            sorted(
                entry
                for entry in parent.iterdir()
                if entry.name != out_path.name
                and entry.name.startswith(prefix)
                and entry.name.endswith(".npy")
                and len(entry.name) >= len(prefix) + 4
            )
        )
    return existing


def _dist_version(*names: str) -> str | None:
    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def _collect_versions(jax_mod: Any, jaxlib_mod: Any, cuex_mod: Any) -> dict[str, Any]:
    cue_ver: str | None = None
    try:
        import cuequivariance as cue_mod

        cue_ver = getattr(cue_mod, "__version__", None)
    except Exception:  # noqa: BLE001
        cue_ver = None
    cue_ops_ver: str | None = None
    try:
        import cuequivariance_ops_jax as cue_ops_mod

        cue_ops_ver = getattr(cue_ops_mod, "__version__", None)
    except Exception:  # noqa: BLE001
        cue_ops_ver = None
    return {
        "jax": getattr(jax_mod, "__version__", None) or _dist_version("jax"),
        "jaxlib": getattr(jaxlib_mod, "__version__", None) or _dist_version("jaxlib"),
        "cuequivariance": cue_ver or _dist_version("cuequivariance"),
        "cuequivariance_jax": getattr(cuex_mod, "__version__", None)
        or _dist_version("cuequivariance-jax", "cuequivariance_jax"),
        "cuequivariance_ops_jax": cue_ops_ver
        or _dist_version(
            "cuequivariance-ops-jax",
            "cuequivariance-ops-jax-cu13",
            "cuequivariance-ops-jax-cu12",
            "cuequivariance_ops_jax",
        ),
    }


def _prng(jax_mod: Any, seed: int) -> Any:
    return jax_mod.random.key(seed)


def _attention_invalid_cols(length: int) -> int:
    return max(1, round(0.05 * length))


def _build_case_execution(
    spec: CaseSpec,
    *,
    jax_mod: Any,
    jnp_mod: Any,
    cuex_mod: Any,
    cueq_mod: Any,
) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
    """Construct the jitted callable, device arguments, and metadata for ``spec``."""
    n = spec.length
    if spec.family in ("raw_triangle_attention", "cueq_attention_core"):
        jnp_dtype = jnp_mod.bfloat16 if spec.dtype == "bf16" else jnp_mod.float32
        seed = 10_000 + n * 4 + (0 if spec.dtype == "bf16" else 1)
        r_q, r_k, r_v, r_b = jax_mod.random.split(_prng(jax_mod, seed), 4)
        qkv_shape = (ATT_BATCH, n, ATT_HEADS, n, ATT_HEAD_DIM)
        bias_shape = (ATT_BATCH, 1, ATT_HEADS, n, n)
        q = jax_mod.random.normal(r_q, qkv_shape, dtype=jnp_dtype)
        k = jax_mod.random.normal(r_k, qkv_shape, dtype=jnp_dtype)
        v = jax_mod.random.normal(r_v, qkv_shape, dtype=jnp_dtype)
        bias = jax_mod.random.normal(r_b, bias_shape, dtype=jnp_dtype)
        if spec.variant == "all_valid":
            invalid_cols = 0
            mask = jnp_mod.ones((ATT_BATCH, n, 1, 1, n), dtype=jnp_mod.bool_)
        elif spec.variant == "last5pct_invalid":
            invalid_cols = _attention_invalid_cols(n)
            valid_cols = n - invalid_cols
            col_valid = jnp_mod.arange(n) < valid_cols
            mask = jnp_mod.broadcast_to(
                col_valid[None, None, None, None, :], (ATT_BATCH, n, 1, 1, n)
            )
        else:
            raise ValueError(f"unknown attention mask variant: {spec.variant!r}")
        scale = float(ATT_HEAD_DIM**-0.5)
        params_meta = {
            "batch": ATT_BATCH,
            "heads": ATT_HEADS,
            "head_dim": ATT_HEAD_DIM,
            "length": n,
            "dtype": spec.dtype,
            "mask": spec.variant,
            "invalid_cols": invalid_cols,
            "scale": scale,
        }
        if spec.family == "raw_triangle_attention":
            precision = cueq_mod.triangle_attention_precision()

            def raw_att(q_in, k_in, v_in, b_in, m_in):
                out, _, _ = cuex_mod.triangle_attention(
                    q=q_in,
                    k=k_in,
                    v=v_in,
                    bias=b_in,
                    mask=m_in,
                    scale=scale,
                    precision=precision,
                )
                return out

            return jax_mod.jit(raw_att), (q, k, v, bias, mask), params_meta

        mask_bias = jnp_mod.where(mask, 0.0, -MASK_BIAS_INF).astype(jnp_mod.float32)

        def core_att(q_in, k_in, v_in, b_in, mb_in):
            return cueq_mod.cueq_attention_core(
                q_in,
                k_in,
                v_in,
                b_in,
                mb_in,
                scale=scale,
                precision=None,
            )

        return jax_mod.jit(core_att), (q, k, v, bias, mask_bias), params_meta

    if spec.family == "fused_triangle_multiplication":
        direction = spec.variant
        x_dtype = jnp_mod.float32 if spec.dtype == "fp32" else jnp_mod.bfloat16
        norm_dtype = (
            jnp_mod.bfloat16 if spec.dtype == "bf16" else jnp_mod.float32
        )
        seed = 20_000 + n * 8 + (0 if direction == "outgoing" else 1)
        rngs = jax_mod.random.split(_prng(jax_mod, seed), 9)
        c = MUL_CHANNELS
        inv_sqrt_c = 1.0 / math.sqrt(c)
        x = jax_mod.random.normal(
            rngs[0], (MUL_BATCH, n, n, c), dtype=jnp_mod.float32
        ).astype(x_dtype)
        invalid_cols = _attention_invalid_cols(n)
        valid = jnp_mod.arange(n) < (n - invalid_cols)
        pair_mask = (valid[:, None] & valid[None, :])[None].astype(x_dtype)
        norm_in_w = (
            1.0 + 0.1 * jax_mod.random.normal(rngs[1], (c,), dtype=jnp_mod.float32)
        ).astype(norm_dtype)
        norm_in_b = (
            0.1 * jax_mod.random.normal(rngs[2], (c,), dtype=jnp_mod.float32)
        ).astype(norm_dtype)
        p_in_w = (
            jax_mod.random.normal(rngs[3], (2 * c, c), dtype=jnp_mod.float32)
            * inv_sqrt_c
        ).astype(x_dtype)
        g_in_w = (
            jax_mod.random.normal(rngs[4], (2 * c, c), dtype=jnp_mod.float32)
            * inv_sqrt_c
        ).astype(x_dtype)
        norm_out_w = (
            1.0 + 0.1 * jax_mod.random.normal(rngs[5], (c,), dtype=jnp_mod.float32)
        ).astype(norm_dtype)
        norm_out_b = (
            0.1 * jax_mod.random.normal(rngs[6], (c,), dtype=jnp_mod.float32)
        ).astype(norm_dtype)
        p_out_w = (
            jax_mod.random.normal(rngs[7], (c, c), dtype=jnp_mod.float32) * inv_sqrt_c
        ).astype(x_dtype)
        g_out_w = (
            jax_mod.random.normal(rngs[8], (c, c), dtype=jnp_mod.float32) * inv_sqrt_c
        ).astype(x_dtype)

        def fused_mul(
            x_in,
            m_in,
            ni_w,
            ni_b,
            pi_w,
            gi_w,
            no_w,
            no_b,
            po_w,
            go_w,
        ):
            return cueq_mod.fused_triangle_multiplication(
                x_in,
                direction=direction,
                mask=m_in,
                norm_in=(ni_w, ni_b),
                p_in=(pi_w, None),
                g_in=(gi_w, None),
                norm_out=(no_w, no_b),
                p_out=(po_w, None),
                g_out=(go_w, None),
                eps=NORM_EPS,
            )

        args = (
            x,
            pair_mask,
            norm_in_w,
            norm_in_b,
            p_in_w,
            g_in_w,
            norm_out_w,
            norm_out_b,
            p_out_w,
            g_out_w,
        )
        params_meta = {
            "batch": MUL_BATCH,
            "channels": c,
            "length": n,
            "direction": direction,
            "dtype_mode": spec.dtype,
            "invalid_cols": invalid_cols,
            "eps": NORM_EPS,
        }
        return jax_mod.jit(fused_mul), args, params_meta

    if spec.family == "boltz2_amp_triangle_multiplication":
        from foldjax.models.boltz2.models.triangle.triangle_cueq import (
            cueq_triangle_multiplication_forward,
        )

        direction = spec.variant
        seed = 25_000 + n * 8 + (0 if direction == "outgoing" else 1)
        rngs = jax_mod.random.split(_prng(jax_mod, seed), 9)
        c = MUL_CHANNELS
        inv_sqrt_c = 1.0 / math.sqrt(c)
        x = jax_mod.random.normal(rngs[0], (MUL_BATCH, n, n, c), dtype=jnp_mod.float32)
        invalid_cols = _attention_invalid_cols(n)
        valid = jnp_mod.arange(n) < (n - invalid_cols)
        pair_mask = (valid[:, None] & valid[None, :])[None].astype(jnp_mod.float32)
        tree = {
            "norm_in": {
                "scale": 1.0
                + 0.1 * jax_mod.random.normal(rngs[1], (c,), dtype=jnp_mod.float32),
                "bias": 0.1
                * jax_mod.random.normal(rngs[2], (c,), dtype=jnp_mod.float32),
            },
            "p_in": {
                "kernel": (
                    jax_mod.random.normal(rngs[3], (c, 2 * c), dtype=jnp_mod.float32)
                    * inv_sqrt_c
                ).astype(jnp_mod.bfloat16)
            },
            "g_in": {
                "kernel": (
                    jax_mod.random.normal(rngs[4], (c, 2 * c), dtype=jnp_mod.float32)
                    * inv_sqrt_c
                ).astype(jnp_mod.bfloat16)
            },
            "norm_out": {
                "scale": 1.0
                + 0.1 * jax_mod.random.normal(rngs[5], (c,), dtype=jnp_mod.float32),
                "bias": 0.1
                * jax_mod.random.normal(rngs[6], (c,), dtype=jnp_mod.float32),
            },
            "p_out": {
                "kernel": (
                    jax_mod.random.normal(rngs[7], (c, c), dtype=jnp_mod.float32)
                    * inv_sqrt_c
                ).astype(jnp_mod.bfloat16)
            },
            "g_out": {
                "kernel": (
                    jax_mod.random.normal(rngs[8], (c, c), dtype=jnp_mod.float32)
                    * inv_sqrt_c
                ).astype(jnp_mod.bfloat16)
            },
        }

        def boltz2_mul(p_tree, x_in, m_in):
            return cueq_triangle_multiplication_forward(
                p_tree,
                x_in,
                m_in,
                direction,
                eps=NORM_EPS,
                native_amp=True,
            )

        params_meta = {
            "batch": MUL_BATCH,
            "channels": c,
            "length": n,
            "direction": direction,
            "dtype_mode": spec.dtype,
            "invalid_cols": invalid_cols,
            "eps": NORM_EPS,
        }
        return jax_mod.jit(boltz2_mul), (tree, x, pair_mask), params_meta

    if spec.family == "layer_norm_transpose":
        from cuequivariance_jax.triangle._layer_norm_transpose import (
            layer_norm_transpose,
        )

        layout = spec.variant
        x_dtype = jnp_mod.bfloat16 if spec.dtype == "bf16" else jnp_mod.float32
        c = MUL_CHANNELS
        seed = (
            30_000
            + n * 8
            + (0 if layout == "bijd->bijd" else 2)
            + (0 if spec.dtype == "bf16" else 1)
        )
        r_x, r_w, r_b = jax_mod.random.split(_prng(jax_mod, seed), 3)
        in_shape = (
            (MUL_BATCH, n, n, c)
            if layout == "bijd->bijd"
            else (c, MUL_BATCH, n, n)
        )
        x = jax_mod.random.normal(r_x, in_shape, dtype=jnp_mod.float32).astype(x_dtype)
        scale_w = 1.0 + 0.1 * jax_mod.random.normal(r_w, (c,), dtype=jnp_mod.float32)
        bias_b = 0.1 * jax_mod.random.normal(r_b, (c,), dtype=jnp_mod.float32)

        def ln_fn(x_in, w_in, b_in):
            return layer_norm_transpose(
                x_in,
                w_in,
                b_in,
                eps=NORM_EPS,
                layout=layout,
                fallback=False,
            )

        params_meta = {
            "batch": MUL_BATCH,
            "channels": c,
            "length": n,
            "layout": layout,
            "dtype": spec.dtype,
            "affine_dtype": "float32",
            "eps": NORM_EPS,
        }
        return jax_mod.jit(ln_fn), (x, scale_w, bias_b), params_meta

    raise ValueError(f"unsupported case family: {spec.family!r}")


def _ensure_foldjax_on_path() -> None:
    if importlib.util.find_spec("foldjax") is not None:
        return
    src_dir = Path(__file__).resolve().parents[1] / "src"
    if src_dir.is_dir():
        sys.path.insert(0, str(src_dir))


def run_benchmark(
    out_path: Path,
    *,
    specs: Sequence[CaseSpec] | None = None,
    case_filter: str | None = None,
    warmup: int = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
    max_sample_elements: int = DEFAULT_MAX_SAMPLE_ELEMENTS,
    _allow_non_gpu: bool = False,
) -> dict[str, Any]:
    """Execute the cuEquivariance benchmark panel and atomically write ``out_path``.

    Raises ``ValueError`` if ``out_path`` or any matching ``<stem>.*.npy`` sidecar
    already exists.
    """
    out_path = Path(out_path)
    validate_run_config(
        warmup=warmup, repeats=repeats, max_sample_elements=max_sample_elements
    )
    chosen_specs = select_cases(
        default_case_specs() if specs is None else specs, case_filter=case_filter
    )
    existing = _existing_outputs(out_path)
    if existing:
        preview = ", ".join(str(path) for path in existing[:MAX_REPORTED_ITEMS])
        if len(existing) > MAX_REPORTED_ITEMS:
            preview = f"{preview}, +{len(existing) - MAX_REPORTED_ITEMS} more"
        raise ValueError(
            f"refusing to overwrite existing output ({preview}); "
            "pass a fresh --out path"
        )
    _ensure_foldjax_on_path()

    import jax
    import jax.numpy as jnp
    import jaxlib

    backend = jax.default_backend()
    if backend != "gpu" and not _allow_non_gpu:
        raise RuntimeError(
            f"GPU backend is required for --out run mode; JAX initialized {backend!r}"
        )
    devices = jax.devices("gpu") if backend == "gpu" else jax.devices()
    device = devices[0]
    gpu_name = str(getattr(device, "device_kind", None) or device)

    from foldjax.models import _cueq as cueq_mod

    cuex = cueq_mod.load_cueq()
    versions = _collect_versions(jax, jaxlib, cuex)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cases_payload: dict[str, dict[str, Any]] = {}
    for spec in chosen_specs:
        jitted = args = compiled = out = host_out = sample = None
        try:
            jitted, args, params_meta = _build_case_execution(
                spec,
                jax_mod=jax,
                jnp_mod=jnp,
                cuex_mod=cuex,
                cueq_mod=cueq_mod,
            )
            t_compile = time.perf_counter()
            compiled = jitted.lower(*args).compile()
            compile_ms = (time.perf_counter() - t_compile) * 1e3
            for _ in range(warmup):
                out = jax.block_until_ready(compiled(*args))
            times_ms: list[float] = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                out = jax.block_until_ready(compiled(*args))
                times_ms.append((time.perf_counter() - t0) * 1e3)
            median_ms = float(statistics.median(times_ms))
            dtype_name = str(out.dtype)
            host_out = np.asarray(out)
            summary, sample = summarize_array(
                host_out,
                dtype_name=dtype_name,
                max_sample_elements=max_sample_elements,
            )
            sample_rel = sample_filename(out_path, spec.name)
            sample_sha256 = write_sample_npy(out_path.parent / sample_rel, sample)
            cases_payload[spec.name] = {
                "status": "ok",
                "spec": asdict(spec),
                "params": params_meta,
                "compile_ms": round(compile_ms, 4),
                "median_ms": round(median_ms, 6),
                "times_ms": [round(val, 6) for val in times_ms],
                **summary,
                "sample_file": sample_rel,
                "sample_sha256": sample_sha256,
            }
        except Exception as exc:  # noqa: BLE001
            cases_payload[spec.name] = {
                "status": "error",
                "spec": asdict(spec),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        finally:
            del jitted, args, compiled, out, host_out, sample
            gc.collect()

    report = {
        "versions": versions,
        "gpu_name": gpu_name,
        "config": {
            "warmup": warmup,
            "repeats": repeats,
            "max_sample_elements": max_sample_elements,
            "matmul_precision": str(jax.config.jax_default_matmul_precision),
        },
        "cases": cases_payload,
    }
    encoded = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _atomic_write_bytes(out_path, encoded)
    return report


def _validate_relative_sample_file(sample_file: Any, report_path: Path) -> Path:
    if not isinstance(sample_file, str) or not sample_file:
        raise ValueError(f"invalid sample_file in {report_path}: {sample_file!r}")
    rel = Path(sample_file)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(
            f"sample_file must stay relative to {report_path.parent}: {sample_file!r}"
        )
    return report_path.parent / rel


def _load_sample_array(sample_path: Path, expected_sha256: Any) -> np.ndarray:
    if not sample_path.is_file():
        raise ValueError(f"missing sample file: {sample_path}")
    raw = sample_path.read_bytes()
    if expected_sha256 is not None:
        if not isinstance(expected_sha256, str) or not expected_sha256:
            raise ValueError(f"invalid sample_sha256 for {sample_path}")
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"sample SHA-256 mismatch for {sample_path}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
    try:
        loaded = np.load(io.BytesIO(raw), allow_pickle=False)
    except (EOFError, OSError, ValueError) as exc:
        raise ValueError(f"corrupt sample file {sample_path}: {exc}") from exc
    if not isinstance(loaded, np.ndarray):
        close = getattr(loaded, "close", None)
        if callable(close):
            close()
        raise ValueError(f"sample {sample_path} must be a .npy array")
    arr = np.asarray(loaded)
    if arr.ndim != 1 or not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(
            f"sample {sample_path} must be a 1D floating array; "
            f"got shape {arr.shape}, dtype {arr.dtype}"
        )
    return arr


def load_report(path: Path) -> dict[str, Any]:
    """Load and validate a JSON benchmark report for ``--compare``."""
    if not path.is_file():
        raise ValueError(f"report file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report root must be a JSON object: {path}")
    cases = payload.get("cases")
    if not isinstance(cases, dict) or not cases:
        raise ValueError(f"report {path} must contain a non-empty 'cases' object")
    for case_name, entry in cases.items():
        if (
            not isinstance(case_name, str)
            or not case_name
            or not isinstance(entry, dict)
        ):
            raise ValueError(f"invalid case entry {case_name!r} in {path}")
        status = entry.get("status", "ok")
        if status not in ("ok", "error"):
            raise ValueError(f"invalid status {status!r} for {case_name} in {path}")
        if status == "error":
            continue
        median_ms = entry.get("median_ms")
        if (
            isinstance(median_ms, bool)
            or not isinstance(median_ms, (int, float))
            or not math.isfinite(median_ms)
            or median_ms < 0
        ):
            raise ValueError(
                f"invalid median_ms {median_ms!r} for {case_name} in {path}"
            )
        shape = entry.get("shape")
        if not isinstance(shape, list) or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim < 0
            for dim in shape
        ):
            raise ValueError(f"invalid shape {shape!r} for {case_name} in {path}")
        dtype_name = entry.get("dtype")
        if not isinstance(dtype_name, str) or not dtype_name:
            raise ValueError(f"invalid dtype {dtype_name!r} for {case_name} in {path}")
        sha256 = entry.get("sha256")
        if not isinstance(sha256, str) or not sha256:
            raise ValueError(f"invalid sha256 {sha256!r} for {case_name} in {path}")
        _validate_relative_sample_file(entry.get("sample_file"), path)
    return payload


def compute_sample_diffs(
    sample_a: np.ndarray, sample_b: np.ndarray
) -> tuple[float, float]:
    """Compute ``(max_abs_diff, max_rel_diff)`` from two 1D sample arrays.

    Relative difference uses ``|b - a| / max(|a|, |b|)`` elementwise with ``0 / 0 = 0``.
    """
    if sample_a.shape != sample_b.shape:
        raise ValueError(
            f"sample shape mismatch: {sample_a.shape} vs {sample_b.shape}"
        )
    if sample_a.size == 0:
        return 0.0, 0.0
    a = sample_a.astype(np.float64)
    b = sample_b.astype(np.float64)
    if np.isnan(a).any() or np.isnan(b).any():
        return float("nan"), float("nan")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        if np.array_equal(a, b):
            return 0.0, 0.0
        return float("inf"), float("inf")
    abs_diff = np.abs(b - a)
    max_abs = float(np.max(abs_diff))
    denom = np.maximum(np.abs(a), np.abs(b))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = np.where(abs_diff == 0.0, 0.0, abs_diff / denom)
    max_rel = float(np.max(rel_diff))
    return max_abs, max_rel


def _format_ratio(time_a: float, time_b: float) -> str:
    if time_a == 0.0:
        return "1.000x" if time_b == 0.0 else "inf"
    return f"{time_b / time_a:.3f}x"


def compare_reports(path_a: Path, path_b: Path) -> str:
    """Compare two benchmark JSON reports and format the summary table."""
    report_a = load_report(path_a)
    report_b = load_report(path_b)
    cases_a: dict[str, dict[str, Any]] = report_a["cases"]
    cases_b: dict[str, dict[str, Any]] = report_b["cases"]
    ordered_names = list(cases_a) + [name for name in cases_b if name not in cases_a]

    ver_a = report_a.get("versions") or {}
    ver_b = report_b.get("versions") or {}
    header_lines = [
        f"A: {path_a.name}  gpu={report_a.get('gpu_name', 'unknown')}  "
        f"cueq_jax={ver_a.get('cuequivariance_jax', 'unknown')}  "
        f"cueq_ops_jax={ver_a.get('cuequivariance_ops_jax', 'unknown')}  "
        f"jax={ver_a.get('jax', 'unknown')}",
        f"B: {path_b.name}  gpu={report_b.get('gpu_name', 'unknown')}  "
        f"cueq_jax={ver_b.get('cuequivariance_jax', 'unknown')}  "
        f"cueq_ops_jax={ver_b.get('cuequivariance_ops_jax', 'unknown')}  "
        f"jax={ver_b.get('jax', 'unknown')}",
        "",
    ]

    columns = (
        "case",
        "time A (ms)",
        "time B (ms)",
        "ratio B/A",
        "bit-identical",
        "max abs diff",
        "max rel diff",
    )
    rows: list[tuple[str, ...]] = []
    for name in ordered_names:
        rec_a = cases_a.get(name)
        rec_b = cases_b.get(name)
        if rec_a is None or rec_b is None:
            t_a = (
                "MISSING"
                if rec_a is None
                else (
                    "ERROR"
                    if rec_a.get("status", "ok") != "ok"
                    else f"{float(rec_a['median_ms']):.3f}"
                )
            )
            t_b = (
                "MISSING"
                if rec_b is None
                else (
                    "ERROR"
                    if rec_b.get("status", "ok") != "ok"
                    else f"{float(rec_b['median_ms']):.3f}"
                )
            )
            rows.append((name, t_a, t_b, "-", "-", "-", "-"))
            continue
        ok_a = rec_a.get("status", "ok") == "ok"
        ok_b = rec_b.get("status", "ok") == "ok"
        if not (ok_a and ok_b):
            t_a = f"{float(rec_a['median_ms']):.3f}" if ok_a else "ERROR"
            t_b = f"{float(rec_b['median_ms']):.3f}" if ok_b else "ERROR"
            rows.append((name, t_a, t_b, "-", "-", "-", "-"))
            continue

        time_a = float(rec_a["median_ms"])
        time_b = float(rec_b["median_ms"])
        identical = (
            rec_a["sha256"] == rec_b["sha256"]
            and rec_a["shape"] == rec_b["shape"]
            and rec_a["dtype"] == rec_b["dtype"]
        )
        sample_a = _load_sample_array(
            _validate_relative_sample_file(rec_a["sample_file"], path_a),
            rec_a.get("sample_sha256"),
        )
        sample_b = _load_sample_array(
            _validate_relative_sample_file(rec_b["sample_file"], path_b),
            rec_b.get("sample_sha256"),
        )
        try:
            max_abs, max_rel = compute_sample_diffs(sample_a, sample_b)
        except ValueError as exc:
            raise ValueError(f"case {name}: {exc}") from exc

        rows.append(
            (
                name,
                f"{time_a:.3f}",
                f"{time_b:.3f}",
                _format_ratio(time_a, time_b),
                "yes" if identical else "no",
                f"{max_abs:.3e}",
                f"{max_rel:.3e}",
            )
        )

    widths = [
        max(len(col), *(len(row[idx]) for row in rows))
        for idx, col in enumerate(columns)
    ]

    def fmt_row(values: Sequence[str]) -> str:
        cells = [
            values[0].ljust(widths[0]),
            *(values[idx].rjust(widths[idx]) for idx in range(1, len(columns))),
        ]
        return "| " + " | ".join(cells) + " |"

    sep = "|-" + "-|-".join("-" * width for width in widths) + "-|"
    table_lines = [fmt_row(columns), sep, *(fmt_row(row) for row in rows)]
    return "\n".join(header_lines + table_lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run or compare cuEquivariance GPU checks on FoldJAX kernel call shapes."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--out",
        type=Path,
        help=(
            "Run the GPU benchmark panel and write JSON results to a fresh PATH "
            "(exits 2 if any case fails)."
        ),
    )
    mode.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("A_JSON", "B_JSON"),
        help="Compare two JSON result files on CPU and print a summary table.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Warm-up iterations per case in --out mode (default: {DEFAULT_WARMUP}).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Timed iterations per case in --out mode (default: {DEFAULT_REPEATS}).",
    )
    parser.add_argument(
        "--max-sample-elements",
        type=int,
        default=DEFAULT_MAX_SAMPLE_ELEMENTS,
        help=(
            "Maximum elements saved per .npy sample in --out mode "
            f"(default: {DEFAULT_MAX_SAMPLE_ELEMENTS})."
        ),
    )
    parser.add_argument(
        "--case-filter",
        default=None,
        help="Optional substring filter on case names in --out mode.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Returns 0 on success, 1 on runtime or validation error, and 2 when ``--out``
    writes a report containing one or more failed cases.
    """
    args = build_parser().parse_args(argv)
    try:
        if args.compare is not None:
            table = compare_reports(args.compare[0], args.compare[1])
            print(table)
            return 0
        report = run_benchmark(
            args.out,
            case_filter=args.case_filter,
            warmup=args.warmup,
            repeats=args.repeats,
            max_sample_elements=args.max_sample_elements,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    cases = report["cases"]
    failed = [name for name, entry in cases.items() if entry.get("status") != "ok"]
    if failed:
        shown = ", ".join(failed[:MAX_REPORTED_ITEMS])
        if len(failed) > MAX_REPORTED_ITEMS:
            shown = f"{shown} (+{len(failed) - MAX_REPORTED_ITEMS} more)"
        print(
            f"error: {len(failed)} of {len(cases)} cases failed: {shown}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
