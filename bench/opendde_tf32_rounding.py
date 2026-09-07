"""Diagnose TF32 operand rounding on observed native Linear inputs; no model change."""

import argparse
import json
from pathlib import Path

import numpy as np

from bench.af3_closure_capture import save, sha


def tf32_round(value, mode):
    import jax
    import jax.numpy as jnp

    if value.dtype != jnp.float32:
        raise ValueError("TF32 rounding probe requires FP32 inputs")
    bits = jax.lax.bitcast_convert_type(value, jnp.uint32)
    if mode == "rne":
        increment = jnp.uint32(0xFFF) + ((bits >> 13) & jnp.uint32(1))
    elif mode == "rna":
        increment = jnp.uint32(0x1000)
    elif mode == "rtz":
        increment = jnp.uint32(0)
    else:
        raise ValueError("unknown rounding mode")
    rounded = (bits + increment) & jnp.uint32(0xFFFFE000)
    rounded = jnp.where((bits & jnp.uint32(0x7F800000)) == 0x7F800000, bits, rounded)
    return jax.lax.bitcast_convert_type(rounded, jnp.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    import jax
    import jax.numpy as jnp

    from foldjax.models.protenix.models.diffusion.diffusion import (
        inference_noise_schedule,
    )
    from foldjax.models.protenix.models.primitives.primitives import (
        LinearParams,
        linear,
    )

    with np.load(args.native / "torch/tape.npz") as archive:
        native_schedule = archive["noise_schedule"]
    schedule = np.asarray(inference_noise_schedule(num_steps=200))
    np.savez_compressed(
        args.out / "schedule.npz", native=native_schedule, foldjax=schedule
    )
    schedule_report = {
        "max_abs": float(np.max(abs(schedule - native_schedule))),
        "unequal": int(np.count_nonzero(schedule != native_schedule)),
    }
    print(json.dumps({"schedule": schedule_report}), flush=True)
    records = json.loads((args.native / "linear-policy.json").read_text())
    selected = [r for r in records if r["native_tf32_effect_max_abs"] != 0]
    functions = {}
    for precision in ("high", "highest"):
        with jax.default_matmul_precision(precision):
            for mode in ("none", "rne", "rna", "rtz"):

                def compute(x, weight, bias, mode=mode):
                    if mode != "none":
                        x, weight = tf32_round(x, mode), tf32_round(weight, mode)
                    return linear(x, LinearParams(weight, bias))

                functions[f"{precision}-{mode}"] = jax.jit(compute)
    for precision in ("high", "highest"):
        with jax.default_matmul_precision(precision):
            selected_functions = {
                key: fn
                for key, fn in functions.items()
                if key.startswith(precision + "-")
            }
            for record in selected:
                path = args.native / record["file"]
                with np.load(path) as archive:
                    values = dict(archive)
                operands = (
                    jnp.asarray(values["x"]),
                    jnp.asarray(values["weight"]),
                    jnp.asarray(values["bias"]) if "bias" in values else None,
                )
                metrics = record.setdefault("comparisons", {})
                arrays = {}
                for mode, function in selected_functions.items():
                    result = np.asarray(function(*operands))
                    arrays[mode] = result
                    error = result.astype(np.float64) - values["y"]
                    metrics[mode] = {
                        "max_abs": float(np.max(abs(error))),
                        "rmse": float(np.sqrt(np.mean(error**2))),
                        "unequal": int(np.count_nonzero(error)),
                    }
                np.savez_compressed(
                    args.out / f"{precision}-{record['file']}", **arrays
                )
                record.update(capture_sha256=sha(path))
                print(json.dumps(record), flush=True)
    save(
        args.out / "report.json",
        {
            "scope": __doc__,
            "schedule": schedule_report,
            "records": selected,
            "wrapper_sha256": sha(Path(__file__)),
            "jax_version": jax.__version__,
        },
    )


if __name__ == "__main__":
    main()
