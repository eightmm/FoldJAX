"""Measure JAX Boltz-2 against a captured, noise-matched upstream run.

`capture_upstream_tape.py` records everything an upstream Boltz-2 prediction
draws -- the sampler's initial noise, its per-step churn noise, its per-step
rigid augmentation, and (when subsampling is on) the MSA row permutation of
every trunk pass -- plus the features the model consumed and the structure it
produced. This replays that tape through the JAX port and compares the
coordinates directly, atom for atom, with no alignment.

That directness is the point. On different tapes two correct implementations
produce different structures and can only be compared against each model's own
sample spread; a real defect then hides behind that spread while every
confidence score still agrees. On one tape the coordinates either match or they
do not.

The trunk tensors are compared too, and both the coordinate RMSD and the trunk
correlations are checked against thresholds -- printing a number is not the
same as testing it.

Measured on ``jobs/t0128.json`` (132 tokens, 1,013 resolved atoms, a 2,371-row
alignment) at 1 recycle / 20 diffusion steps / 1 sample, against an upstream
run captured at ``--precision 32 --no_kernels``, on an RTX PRO 6000:

    upstream trunk fed to the JAX sampler   0.00011 A
    JAX trunk, no MSA subsample            0.1489  A
    JAX trunk, MSA rows injected           0.1997  A
    JAX trunk, MSA rows NOT injected       ~2.4    A   (control, must fail)
    JAX trunk subsampling, upstream not    0.8744  A   (--jax-subsample-msa)

The first line is the verdict on the sampler: on one tape and one trunk the
JAX diffusion loop reproduces upstream's coordinates to 1e-4 A, so the whole
remaining gap is fp32 drift in the trunk (s correlates 0.99999996, z
0.99999960). The fourth line is the control: with the alignment rows left
unmatched the same harness reports about 2.4 A and fails, which is what makes the
0.1997 A above evidence rather than a number. The fifth is what FoldJAX's own
``boltz2_predict(subsample_msa=True)`` default costs against a `boltz predict`
that -- because ``--subsample_msa`` is a click flag, and a click flag defaults
to False whatever its Python signature says -- read all 2,371 rows.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.paths import weights_dir

#: ``tests/models/boltz2/scripts/`` -> the FoldJAX checkout root.
REPOSITORY = Path(__file__).resolve().parents[4]

#: ``foldjax.models.boltz2.models.predict._MSA_SUBSAMPLE_STREAM``, repeated so
#: an unmatched control run draws the rows production would draw.
_MSA_STREAM = 0xB012


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tape-dir", type=Path, required=True)
    parser.add_argument(
        "--weights",
        type=Path,
        default=weights_dir("boltz2") / "boltz2_conf",
        help="native Boltz-2 weights exported by `foldjax weights fetch`",
    )
    parser.add_argument("--out-dir", type=Path, default=REPOSITORY / "outputs/parity")
    parser.add_argument(
        "--triangle-backend",
        choices=("xla", "cueq", "tokamax", "pallas"),
        default="xla",
    )
    parser.add_argument(
        "--attention-backend", choices=("xla", "xla_sdpa", "cudnn"), default="xla"
    )
    parser.add_argument(
        "--compute-dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help=(
            "precision for the trunk and sampler. Upstream Boltz-2 runs "
            "bf16-mixed (`precision='bf16-mixed'` in boltz/main.py), so the "
            "float32 default here measures a port that is *more* precise than "
            "the run it is compared against -- which is not the same question "
            "as whether the port reproduces it"
        ),
    )
    parser.add_argument(
        "--max-rmsd",
        type=float,
        default=0.5,
        help=(
            "fail above this all-atom RMSD in angstrom. 0.5 is roughly 2.5x the "
            "measured fp32 trunk drift on the 132-token job and still an order "
            "of magnitude below any structural disagreement"
        ),
    )
    parser.add_argument(
        "--min-correlation",
        type=float,
        default=0.999,
        help="fail if any trunk tensor correlates below this with upstream's",
    )
    parser.add_argument("--no-scan", action="store_true")
    parser.add_argument(
        "--ignore-msa-rows",
        action="store_true",
        help=(
            "control run: keep upstream's tape but let the port draw its own "
            "MSA rows. The trunk correlations must collapse -- if they do not, "
            "this harness is not actually reading the alignment it thinks it is"
        ),
    )
    parser.add_argument(
        "--jax-subsample-msa",
        type=int,
        default=None,
        help=(
            "force the JAX trunk to subsample to N rows whatever the tape did, "
            "drawing them the way `boltz2_predict` does. Measures what the "
            "port's own subsample_msa=True default costs against an upstream "
            "run that -- as the shipped CLI defaults -- did not subsample"
        ),
    )
    parser.add_argument(
        "--trunk-source",
        choices=("jax", "upstream"),
        default="jax",
        help=(
            "'upstream' feeds the captured torch s/z/s_inputs straight into the "
            "JAX sampler, so the remaining coordinate error is the sampler's "
            "alone and trunk drift is excluded"
        ),
    )
    return parser.parse_args()


def rmsd(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((left - right) ** 2, axis=-1))))


def kabsch_rmsd(left: np.ndarray, right: np.ndarray) -> float:
    lc = left - left.mean(axis=0)
    rc = right - right.mean(axis=0)
    u, _, vt = np.linalg.svd(lc.T @ rc)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return rmsd(lc @ rotation.T, rc)


def array_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    x = np.asarray(left, dtype=np.float64).ravel()
    y = np.asarray(right, dtype=np.float64).ravel()
    return {
        "correlation": float(np.corrcoef(x, y)[0, 1]),
        "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
        "max_abs": float(np.max(np.abs(x - y))),
    }


def load_features(path: Path) -> dict[str, jnp.ndarray]:
    """Upstream's own feature tensors, as the JAX port wants them.

    Integer features are narrowed to int32 because JAX runs without x64; every
    id in a Boltz feature dict is far inside int32, and a silent numpy int64 ->
    int32 conversion inside jnp would happen anyway, so it is done explicitly.
    """
    features: dict[str, jnp.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        for name in archive.files:
            value = archive[name]
            if value.dtype == np.int64:
                value = value.astype(np.int32)
            elif value.dtype == np.float64:
                value = value.astype(np.float32)
            features[name] = jnp.asarray(value)
    return features


def load_tape(path: Path, meta: dict) -> dict[str, np.ndarray]:
    """Load the sampler tape and assert every shape the sampler will assume."""
    n_step = int(meta["n_step"])
    n_sample = int(meta["n_sample"])
    n_atom = int(meta["n_atom"])
    contracts = {
        "sigmas": (n_step + 1,),
        "init_noise": (n_sample, n_atom, 3),
        "step_noises": (n_step, n_sample, n_atom, 3),
        "rotations": (n_step, n_sample, 3, 3),
        "translations": (n_step, n_sample, 1, 3),
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = set(contracts).difference(archive.files)
        if missing:
            raise ValueError(f"tape is missing arrays: {sorted(missing)}")
        tape = {name: np.asarray(archive[name], np.float32) for name in contracts}
    for name, expected in contracts.items():
        if tape[name].shape != expected:
            raise ValueError(
                f"{name} expected shape {expected}, got {tape[name].shape}"
            )
    return tape


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_default_matmul_precision", "highest")

    from foldjax.models.boltz2.bridge.native import load_params
    from foldjax.models.boltz2.models.trunk_blocks.trunk import (
        _cast_float_feats,
        _cast_params,
        _sample_schedule,
        boltz2_sample_forward,
        boltz2_trunk_forward,
        relative_position_forward,
    )

    meta = json.loads((args.tape_dir / "tape.json").read_text(encoding="utf-8"))
    features = load_features(args.tape_dir / "features.npz")
    tape = load_tape(args.tape_dir / "tape.npz", meta)
    with np.load(args.tape_dir / "trunk.npz", allow_pickle=False) as archive:
        upstream_trunk = {name: archive[name] for name in archive.files}
    with np.load(args.tape_dir / "coordinate.npz", allow_pickle=False) as archive:
        upstream_coordinate = np.asarray(archive["coordinate"], np.float64)

    msa_rows = None
    if meta["subsample_msa"] and not args.ignore_msa_rows:
        msa_rows = jnp.asarray(np.load(args.tape_dir / "msa_rows.npy"), jnp.int32)
    subsample_msa = bool(meta["subsample_msa"])
    subsample_depth = int(meta["num_subsampled_msa"] or 1024)
    if args.jax_subsample_msa is not None:
        subsample_msa, subsample_depth, msa_rows = True, args.jax_subsample_msa, None

    # The schedule is a closed form on both sides, so a mismatch means the
    # diffusion constants disagree -- catch that here rather than reading it out
    # of a coordinate RMSD later.
    jax_sigmas = np.asarray(
        _sample_schedule(
            int(meta["n_step"]),
            sigma_min=0.0001,
            sigma_max=160.0,
            sigma_data=16.0,
            rho=7.0,
        )
    )
    schedule_max_abs = float(np.max(np.abs(jax_sigmas - tape["sigmas"])))

    params = load_params(args.weights)
    use_scan = not args.no_scan
    compute_dtype = jnp.dtype(args.compute_dtype)
    shared = dict(
        chunk_size=128,
        matmul_precision="highest",
        attention_backend=args.attention_backend,
        triangle_backend=args.triangle_backend,
        glu_backend="xla",
    )
    # `boltz2_trunk_forward` takes no dtype: `boltz2_predict` narrows the trunk
    # by casting the weights and the float features before the call, and the
    # comparison is only honest if this harness narrows it the same way.
    trunk_params = params["trunk"]
    trunk_features = features
    if compute_dtype != jnp.float32:
        trunk_params = _cast_params(trunk_params, compute_dtype)
        trunk_features = _cast_float_feats(features, compute_dtype)

    started = time.perf_counter()
    trunk = boltz2_trunk_forward(
        trunk_params,
        trunk_features,
        recycling_steps=int(meta["n_cycle"]),
        use_scan=use_scan,
        subsample_msa=subsample_msa,
        num_subsampled_msa=subsample_depth,
        msa_rows=msa_rows,
        # `boltz2_predict` folds a fixed stream id into the run key for this
        # draw; reuse it so an unmatched run reproduces the port's own rows
        # rather than some third choice.
        msa_key=(
            jax.random.fold_in(jax.random.PRNGKey(int(meta["seed"])), _MSA_STREAM)
            if msa_rows is None and subsample_msa
            else None
        ),
        **shared,
    )
    jax.block_until_ready(trunk["s"])
    trunk_seconds = time.perf_counter() - started

    sampler_trunk = trunk
    if args.trunk_source == "upstream":
        # The relative position encoding is a deterministic function of integer
        # features, so recomputing it here is exact; only the three learned
        # tensors need to come from the capture.
        sampler_trunk = {
            "s": jnp.asarray(upstream_trunk["s"], jnp.float32),
            "z": jnp.asarray(upstream_trunk["z"], jnp.float32),
            "s_inputs": jnp.asarray(upstream_trunk["s_inputs"], jnp.float32),
            "relative_position_encoding": relative_position_forward(
                params["trunk"]["rel_pos"], features
            ),
        }

    started = time.perf_counter()
    output = boltz2_sample_forward(
        params,
        features,
        jax.random.PRNGKey(int(meta["seed"])),
        trunk=sampler_trunk,
        recycling_steps=int(meta["n_cycle"]),
        num_sampling_steps=int(meta["n_step"]),
        multiplicity=int(meta["n_sample"]),
        step_scale=float(meta["step_scale"]),
        gamma_0=float(meta["gamma_0"]),
        gamma_min=float(meta["gamma_min"]),
        noise_scale=float(meta["noise_scale"]),
        sigma_data=float(meta["sigma_data"]),
        init_noise=jnp.asarray(tape["init_noise"]),
        step_noises=jnp.asarray(tape["step_noises"]),
        aug_transforms=(
            jnp.asarray(tape["rotations"]),
            jnp.asarray(tape["translations"]),
        ),
        use_scan=use_scan,
        compute_dtype=compute_dtype,
        **shared,
    )
    coordinate = np.asarray(
        jax.device_get(output["sample_atom_coords"]), dtype=np.float64
    )
    sample_seconds = time.perf_counter() - started

    if coordinate.shape != upstream_coordinate.shape:
        raise RuntimeError(
            f"coordinate shape {coordinate.shape} != upstream "
            f"{upstream_coordinate.shape}"
        )
    resolved = np.asarray(features["atom_pad_mask"])[0].astype(bool)
    left = coordinate[0]
    right = upstream_coordinate[0]

    metrics = {
        "tokens": int(meta["n_token"]),
        "atoms": int(meta["n_atom"]),
        "resolved_atoms": int(resolved.sum()),
        "msa_depth": int(meta["msa_depth"]),
        "recycles": int(meta["n_cycle"]),
        "diffusion_steps": int(meta["n_step"]),
        "samples": int(meta["n_sample"]),
        "upstream_precision": meta["precision"],
        "upstream_kernels": meta["kernels"],
        "upstream_subsample_msa": meta["subsample_msa"],
        "jax_subsample_msa": subsample_msa,
        "jax_subsample_depth": subsample_depth if subsample_msa else None,
        "msa_rows_injected": msa_rows is not None,
        "trunk_source": args.trunk_source,
        "triangle_backend": args.triangle_backend,
        "attention_backend": args.attention_backend,
        "schedule_max_abs": schedule_max_abs,
        "trunk_seconds": trunk_seconds,
        "sample_seconds": sample_seconds,
        "all_atom_rmsd_angstrom": rmsd(left[resolved], right[resolved]),
        "all_atom_kabsch_rmsd_angstrom": kabsch_rmsd(left[resolved], right[resolved]),
        "coordinate_max_abs_error_angstrom": float(
            np.max(np.abs(left[resolved] - right[resolved]))
        ),
        "padded_all_atom_rmsd_angstrom": rmsd(left, right),
        "s_inputs": array_metrics(
            np.asarray(trunk["s_inputs"]), upstream_trunk["s_inputs"]
        ),
        "s_trunk": array_metrics(np.asarray(trunk["s"]), upstream_trunk["s"]),
        "z_trunk": array_metrics(np.asarray(trunk["z"]), upstream_trunk["z"]),
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    np.save(args.out_dir / "coordinate.npy", coordinate)
    print(json.dumps(metrics, indent=2))

    failures = _verdict(
        metrics, max_rmsd=args.max_rmsd, min_correlation=args.min_correlation
    )
    if failures:
        print("\n".join(["", "PARITY FAILED:", *failures]))
        return 1
    print(
        f"\nPARITY OK: all-atom RMSD {metrics['all_atom_rmsd_angstrom']:.4f} A "
        f"<= {args.max_rmsd:g}, every trunk tensor >= {args.min_correlation:g}."
    )
    return 0


def _verdict(metrics: dict, *, max_rmsd: float, min_correlation: float) -> list[str]:
    """Which measurements fall short, as lines to print.

    Reporting a number is not checking it: a harness that only prints its RMSD
    cannot fail, and a check that cannot fail is how a reordered MSA block
    survives a whole release.
    """
    failures = []
    if not metrics["schedule_max_abs"] < 1e-4:
        failures.append(
            f"  noise schedule differs by {metrics['schedule_max_abs']:.3e}; "
            "the diffusion constants do not match upstream"
        )
    rmsd_value = metrics["all_atom_rmsd_angstrom"]
    if not rmsd_value <= max_rmsd:
        failures.append(f"  all-atom RMSD {rmsd_value:.4f} A > {max_rmsd:g} A")
    for name in ("s_inputs", "s_trunk", "z_trunk"):
        correlation = metrics.get(name, {}).get("correlation")
        if correlation is None:
            failures.append(f"  {name}: not reported")
        elif not correlation >= min_correlation:
            failures.append(
                f"  {name}: correlation {correlation:.9f} < {min_correlation:g}"
            )
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
