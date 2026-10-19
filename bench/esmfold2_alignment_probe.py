"""CPU diagnostics for the experimental ESMFold2 quaternion alignment solve.

The instrumented Jacobi arithmetic mirrors the production helper, but exposes
its diagonalization state and sweep count. The production helper is also run
unchanged. FP64 oracles consume the same FP32-rounded covariance, separating
solver error from unavailable input precision. None of these diagnostics admits
a numerical default or explains an uncaptured full-model residual by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from functools import partial
from pathlib import Path

import numpy as np

SWEEPS = (1, 2, 3, 5, 10, 20, 40)
WEAK_GAP = 32 * np.finfo(np.float32).eps


def quaternion_matrix(correlation, xp=np):
    scale = xp.max(xp.abs(correlation), axis=(-2, -1), keepdims=True)
    matrix = correlation / xp.where(scale > 0, scale, 1.0)
    xx, xy, xz = matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 0, 2]
    yx, yy, yz = matrix[:, 1, 0], matrix[:, 1, 1], matrix[:, 1, 2]
    zx, zy, zz = matrix[:, 2, 0], matrix[:, 2, 1], matrix[:, 2, 2]
    return xp.stack(
        (
            xx + yy + zz,
            yz - zy,
            zx - xz,
            xy - yx,
            yz - zy,
            xx - yy - zz,
            xy + yx,
            xz + zx,
            zx - xz,
            xy + yx,
            yy - xx - zz,
            yz + zy,
            xy - yx,
            xz + zx,
            yz + zy,
            zz - xx - yy,
        ),
        axis=-1,
    ).reshape((-1, 4, 4))


def jacobi_state(correlation, sweeps):
    """Bench-only copy of the fixed-size algorithm, with observable state."""
    import jax
    import jax.numpy as jnp

    symmetric = quaternion_matrix(correlation, jnp)
    vectors = jnp.broadcast_to(jnp.eye(4, dtype=correlation.dtype), symmetric.shape)

    def sweep(_, state):
        diagonal, vectors = state
        for p, q in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            off = diagonal[:, p, q]
            delta = 0.5 * (diagonal[:, q, q] - diagonal[:, p, p])
            denominator = jnp.abs(delta) + jnp.sqrt(delta * delta + off * off)
            tangent = (
                jnp.where(delta < 0, -1.0, 1.0)
                * off
                / jnp.where(denominator > 0, denominator, 1.0)
            )
            cosine = jax.lax.rsqrt(1.0 + tangent * tangent)
            sine = tangent * cosine
            col_p, col_q = diagonal[:, :, p], diagonal[:, :, q]
            next_p = cosine[:, None] * col_p - sine[:, None] * col_q
            next_q = sine[:, None] * col_p + cosine[:, None] * col_q
            updated = diagonal.at[:, :, p].set(next_p)
            updated = updated.at[:, p, :].set(next_p)
            updated = updated.at[:, :, q].set(next_q)
            updated = updated.at[:, q, :].set(next_q)
            updated = updated.at[:, p, p].set(diagonal[:, p, p] - tangent * off)
            updated = updated.at[:, q, q].set(diagonal[:, q, q] + tangent * off)
            updated = updated.at[:, p, q].set(0.0)
            diagonal = updated.at[:, q, p].set(0.0)
            vec_p, vec_q = vectors[:, :, p], vectors[:, :, q]
            vectors = vectors.at[:, :, p].set(
                cosine[:, None] * vec_p - sine[:, None] * vec_q
            )
            vectors = vectors.at[:, :, q].set(
                sine[:, None] * vec_p + cosine[:, None] * vec_q
            )
        return diagonal, vectors

    diagonal, vectors = jax.lax.fori_loop(0, sweeps, sweep, (symmetric, vectors))
    largest = jnp.argmax(jnp.diagonal(diagonal, axis1=-2, axis2=-1), axis=-1)
    quaternion = jnp.take_along_axis(vectors, largest[:, None, None], axis=-1)[:, :, 0]
    from foldjax.models.esmfold2.models.diffusion import quaternion_to_rotation

    rotation = jnp.swapaxes(quaternion_to_rotation(quaternion), -1, -2)
    return rotation, quaternion, diagonal


def synthetic_panel(samples=32, seed=17):
    if isinstance(samples, bool) or samples <= 0:
        raise ValueError("samples per family must be positive")
    rng = np.random.default_rng(seed)
    gaussian = rng.normal(size=(samples, 3, 3))
    blocks, labels = [gaussian], ["random"] * samples

    def proper_basis():
        basis = np.linalg.qr(rng.normal(size=(samples, 3, 3)))[0]
        basis[:, :, -1] *= np.linalg.det(basis)[:, None]
        return basis

    for name, singular, reflection in (
        ("proper", (3, 2, 1), False),
        ("ill_conditioned", (1, 1e-3, 1e-6), False),
        ("rank_two", (3, 1, 0), False),
        ("rank_one", (3, 0, 0), False),
        ("zero", (0, 0, 0), False),
        ("reflection_unique", (3, 2, 1), True),
        ("reflection_weak_gap", (3, 1, 1 - 1e-6), True),
        ("reflection_equal", (1, 1, 1), True),
    ):
        left, right = proper_basis(), proper_basis()
        if reflection:
            left[:, :, 0] *= -1
        blocks.append(
            (left * np.asarray(singular)[None, None, :]) @ right.transpose(0, 2, 1)
        )
        labels.extend([name] * samples)
    for scale in (1e-25, 1e-12, 1e12, 1e25):
        blocks.append(gaussian * scale)
        labels.extend([f"scale_{scale:g}"] * samples)
    return np.concatenate(blocks).astype(np.float32), np.asarray(labels)


def load_covariances(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        if "correlation" not in archive:
            raise ValueError("capture must contain correlation[..., 3, 3]")
        source = np.asarray(archive["correlation"])
        if source.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("correlation must use float32 or float64")
        if source.ndim < 2 or source.shape[-2:] != (3, 3) or source.size == 0:
            raise ValueError("correlation must end in nonempty 3-by-3 matrices")
        if not np.isfinite(source).all():
            raise ValueError("captured correlation contains nonfinite values")
        rounded = source.astype(np.float32).reshape((-1, 3, 3))
        if not np.isfinite(rounded).all():
            raise ValueError("captured correlation exceeds float32 range")
        labels = (
            np.asarray(archive["labels"]).reshape(-1)
            if "labels" in archive
            else np.asarray([f"capture_{index}" for index in range(len(rounded))])
        )
    if labels.shape != (len(rounded),) or labels.dtype.kind not in "US":
        raise ValueError("labels must contain one string per captured matrix")
    return rounded, labels, tuple(source.shape)


def svd_rotation64(correlation):
    u, _, vh = np.linalg.svd(np.asarray(correlation, np.float64))
    u[:, :, -1] *= np.linalg.det(u @ vh)[:, None]
    return u @ vh


def rotation_metrics(correlation, rotation, oracle):
    h = np.asarray(correlation, np.float64)
    r = np.asarray(rotation, np.float64)
    scale = np.maximum(np.linalg.norm(h, axis=(-2, -1)), np.finfo(np.float64).tiny)
    optimum = np.sum(oracle * h, axis=(-2, -1))
    objective = np.sum(r * h, axis=(-2, -1))
    return {
        "rotation_max_abs_error": np.max(np.abs(r - oracle), axis=(-2, -1)),
        "objective": objective,
        "objective_normalized_deficit": (optimum - objective) / scale,
        "orthogonality_max_abs_error": np.max(
            np.abs(r @ r.transpose(0, 2, 1) - np.eye(3)), axis=(-2, -1)
        ),
        "proper_determinant_error": np.abs(np.linalg.det(r) - 1),
    }


def eigen_metrics(correlation, quaternion, diagonal):
    k = quaternion_matrix(np.asarray(correlation, np.float64))
    q = np.asarray(quaternion, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    eigenvalues = np.linalg.eigvalsh(k)
    norm = np.maximum(np.max(np.abs(eigenvalues), axis=-1), np.finfo(np.float64).tiny)
    kq = (k @ q[..., None])[..., 0]
    rayleigh = np.sum(q * kq, axis=-1)
    off_diagonal = np.asarray(diagonal, np.float64).copy()
    off_diagonal[:, np.arange(4), np.arange(4)] = 0
    return {
        "eigen_residual_normalized": np.linalg.norm(kq - rayleigh[:, None] * q, axis=-1)
        / norm,
        "top_eigenvalue_normalized_deficit": (eigenvalues[:, -1] - rayleigh) / norm,
        "top_eigen_gap_normalized": (eigenvalues[:, -1] - eigenvalues[:, -2]) / norm,
        "off_diagonal_frobenius_normalized": np.linalg.norm(off_diagonal, axis=(-2, -1))
        / norm,
    }


def _summary(values):
    a = np.asarray(values, np.float64)
    return {
        "min": float(a.min()),
        "median": float(np.median(a)),
        "p99": float(np.quantile(a, 0.99)),
        "max": float(a.max()),
    }


def covariance_report(correlations, labels, sweeps=SWEEPS):
    import jax
    import jax.numpy as jnp

    from foldjax.models.esmfold2.models.diffusion import _rotation_from_correlation

    h = np.asarray(correlations, np.float32)
    if h.ndim != 3 or h.shape[1:] != (3, 3) or not len(h) or not np.isfinite(h).all():
        raise ValueError("finite float32 covariance batch required")
    if np.asarray(labels).shape != (len(h),):
        raise ValueError("one label is required per covariance")
    if not sweeps or any(
        isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1
        for n in sweeps
    ):
        raise ValueError("sweep counts must be positive")
    oracle = svd_rotation64(h)
    arrays = {
        "correlation": h,
        "labels": np.asarray(labels),
        "rotation_oracle64": oracle,
    }
    report = {
        "cases": len(h),
        "platform": "cpu",
        "sweeps": list(sweeps),
        "oracle_scope": (
            "FP64 solver on the same FP32-rounded input; "
            "not recovery of lost input bits"
        ),
        "weak_gap_diagnostic_threshold": float(WEAK_GAP),
        "full_model_admission": None,
        "jax_version": jax.__version__,
        "production_source_sha256": hashlib.sha256(
            Path(inspect.getfile(_rotation_from_correlation)).read_bytes()
        ).hexdigest(),
        "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runs": {},
    }
    device = jax.devices("cpu")[0]
    with jax.default_device(device), jax.enable_x64(True):
        placed = jax.device_put(h, device)
        production = np.asarray(jax.jit(_rotation_from_correlation)(placed))
        arrays["rotation_production32"] = production
        production_metrics = rotation_metrics(h, production, oracle)
        report["production32"] = {
            name: _summary(value) for name, value in production_metrics.items()
        }
        for name, value in production_metrics.items():
            arrays[f"production32_{name}"] = value
        for dtype in (jnp.float32, jnp.float64):
            values = jax.device_put(h.astype(dtype), device)
            for count in sweeps:
                name = f"jacobi{32 if dtype == jnp.float32 else 64}_s{count:02}"
                rotation, quaternion, diagonal = jax.jit(
                    partial(jacobi_state, sweeps=count)
                )(values)
                rotation, quaternion, diagonal = map(
                    np.asarray, (rotation, quaternion, diagonal)
                )
                metrics = {
                    **rotation_metrics(h, rotation, oracle),
                    **eigen_metrics(h, quaternion, diagonal),
                    "probe_production_rotation_abs_diff": np.max(
                        np.abs(rotation - production), axis=(-2, -1)
                    ),
                }
                arrays[f"{name}_rotation"] = rotation
                for metric, value in metrics.items():
                    arrays[f"{name}_{metric}"] = value
                groups = {}
                for label in dict.fromkeys(np.asarray(labels).tolist()):
                    mask = np.asarray(labels) == label
                    groups[str(label)] = {
                        metric: _summary(value[mask])
                        for metric, value in metrics.items()
                    }
                report["runs"][name] = {
                    "metrics": {
                        metric: _summary(value) for metric, value in metrics.items()
                    },
                    "groups": groups,
                    "probe_vs_production_max_abs": float(
                        np.max(np.abs(rotation - production))
                    ),
                }
    gap = arrays[f"jacobi32_s{sweeps[-1]:02}_top_eigen_gap_normalized"]
    report["weak_gap_cases"] = int((gap <= WEAK_GAP).sum())
    report["scope"] = (
        "Eigen residual/objective distinguish convergence from a nonunique rotation. "
        "Probe outputs can change fusion, so production32 is measured separately. "
        "Captured full-model covariances and teacher-forced local replay "
        "are needed for attribution."
    )
    return report, arrays


def centering_report(seed=47):
    import jax

    from foldjax.models.esmfold2.models.diffusion import weighted_rigid_align

    rng = np.random.default_rng(seed)
    source = rng.normal(size=(8, 64, 3)).astype(np.float32)
    target = source @ np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float32)
    target += rng.normal(size=target.shape).astype(np.float32) * 0.01
    weights = rng.uniform(0.1, 1, size=source.shape[:2]).astype(np.float32)
    mask = (rng.uniform(size=weights.shape) > 0.2).astype(np.float32)
    results = []
    device = jax.devices("cpu")[0]
    with jax.default_device(device):
        for scale in (0.1, 1, 100):
            for translation in (0, 1e3, 1e6):
                x = (source * scale + translation).astype(np.float32)
                y = (target * scale - translation).astype(np.float32)
                w = (weights * mask).astype(np.float64)[..., None]
                denominator = np.maximum(w.sum(axis=1, keepdims=True), 1e-8)
                cx = (x.astype(np.float64) * w).sum(axis=1, keepdims=True) / denominator
                cy = (y.astype(np.float64) * w).sum(axis=1, keepdims=True) / denominator
                h = np.einsum("bni,bnj->bij", w * (y - cy), x - cx)
                oracle = (x - cx) @ svd_rotation64(h).transpose(0, 2, 1) + cy
                aligned = {}
                row = {
                    "scale": scale,
                    "translation": translation,
                    "reference": (
                        "FP64 centering/SVD of the same rounded FP32 coordinates"
                    ),
                }
                for backend in ("svd", "jacobi"):
                    value = np.asarray(
                        jax.jit(partial(weighted_rigid_align, backend=backend))(
                            x, y, weights, mask
                        )
                    )
                    aligned[backend] = value
                    error = np.where(mask[..., None] != 0, value - oracle, 0)
                    row[f"{backend}_valid_max_abs_error"] = float(np.max(np.abs(error)))
                    row[f"{backend}_valid_rms_error"] = float(
                        np.sqrt(np.sum(error**2) / (3 * mask.sum()))
                    )
                delta = np.where(
                    mask[..., None] != 0, aligned["svd"] - aligned["jacobi"], 0
                )
                row["svd_jacobi_valid_max_abs_error"] = float(np.max(np.abs(delta)))
                results.append(row)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="NPZ with correlation[...,3,3] and optional string labels",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-family", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    # The executable is deliberately CPU-only; it must never launch local GPU work.
    os.environ["JAX_PLATFORMS"] = "cpu"
    if args.input:
        covariance, labels, shape = load_covariances(args.input)
        identity = {
            "kind": "provided_covariances",
            "capture_origin_verified": False,
            "shape": list(shape),
            "sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        }
    else:
        covariance, labels = synthetic_panel(args.samples_per_family, args.seed)
        identity = {
            "kind": "synthetic",
            "seed": args.seed,
            "samples_per_family": args.samples_per_family,
        }
    if args.output.exists():
        parser.error("output must be a new directory")
    report, arrays = covariance_report(covariance, labels)
    report["input"] = identity
    report["centering_panel"] = centering_report()
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "arrays.npz", **arrays)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        name: report["runs"][name]["metrics"]["eigen_residual_normalized"]["max"]
        for name in ("jacobi32_s05", "jacobi32_s10", "jacobi32_s20", "jacobi64_s10")
    }
    print(
        json.dumps(
            {
                "cases": report["cases"],
                "max_eigen_residual": summary,
                "full_model_admission": None,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
