"""Full same-LM, same-checkpoint JAX shim comparison; not model admission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench.boltz_amp_report import compare_arrays
from bench.esmfold2_lm_shim_probe import boundary_slice, load_reference
from bench.esmfold2_tape import _npz, _save_npz, _sha256


def validate_native(root, metadata, reference, weights):
    report = json.loads((root / "report.json").read_text())
    checks = {
        "source_sha256": metadata["binding"]["source"]["native"][
            "transformers/models/esmfold2/modeling_esmfold2_common.py"
        ],
        "checkpoint_sha256": _sha256(weights / "model.safetensors"),
        "config_sha256": _sha256(weights / "config.json"),
        "reference_sha256": _sha256(reference / "metadata.json"),
        "lm_sha256": metadata["lm_schema"]["sha256"],
        "archive_sha256": _sha256(root / "native.npz"),
    }
    if any(report.get(key) != value for key, value in checks.items()):
        raise ValueError("native shim source artifacts differ")
    if (
        report.get("autocast") != "bfloat16"
        or report.get("pair_dtype") != "torch.float32"
    ):
        raise ValueError("native shim dtype policy differs")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native", "reference", "weights", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    metadata, hidden = load_reference(args.reference)
    native = validate_native(args.native, metadata, args.reference, args.weights)
    args.output.mkdir(parents=True, exist_ok=False)
    import jax
    import jax.numpy as jnp
    from safetensors import safe_open

    from foldjax.models.esmfold2.models import model

    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise RuntimeError("candidate shim probe requires one GPU")
    jax.config.update("jax_default_matmul_precision", "highest")
    paths = (
        Path(model.__file__), Path(__file__),
        args.weights / "model.safetensors", args.native / "report.json",
        *sorted(Path(model.__file__).parent.rglob("*.py")),
    )
    before = [_sha256(path) for path in paths]
    with safe_open(args.weights / "model.safetensors", framework="numpy") as handle:
        params = {
            key: jnp.asarray(handle.get_tensor(key))
            for key in handle.keys() if key.startswith("language_model.")
        }
    run = jax.jit(
        lambda x, p: model.language_model_pair(x, p, compute_dtype=jnp.bfloat16)
    )
    output = run(jnp.asarray(hidden), params)
    output.block_until_ready()
    pair = np.asarray(output)
    if pair.dtype != np.float32:
        raise ValueError("native autocast shim must return FP32 pair")
    reference_pair = _npz(args.native / "native.npz")["pair"]
    comparison = compare_arrays({"pair": reference_pair}, {"pair": pair})

    def prefix_probe(x, p):
        normalized = model.layer_norm(
            x.astype(jnp.float32), p["language_model.base_z_linear.0.weight"],
            p["language_model.base_z_linear.0.bias"],
        )
        projected = model._lm_autocast_linear(
            normalized, p, "language_model.base_z_linear.1"
        )
        return {
            "base_z_linear.0": boundary_slice(normalized),
            "base_z_linear.1": boundary_slice(projected).astype(jnp.float32),
            "combine_softmax": jax.nn.softmax(p["language_model.base_z_combine"]),
        }

    prefix = {
        key: np.asarray(value)
        for key, value in jax.jit(prefix_probe)(jnp.asarray(hidden), params).items()
    }
    native_arrays = _npz(args.native / "native.npz")
    prefix_comparison = compare_arrays(
        {key: native_arrays[key] for key in prefix}, prefix
    )
    _save_npz(args.output / "candidate.npz", {"pair": pair, **prefix})
    validate_native(args.native, metadata, args.reference, args.weights)
    load_reference(args.reference)
    if before != [_sha256(path) for path in paths]:
        raise ValueError("bound shim inputs changed during run")
    report = {
        "scope": "full LM-shim pair only; identical native LM and checkpoint",
        "model_admission": None,
        "native": native,
        "jax": jax.__version__,
        "source_sha256": before[0],
        "source_tree_sha256": {
            str(path.relative_to(Path(model.__file__).parent)): digest
            for path, digest in zip(paths[4:], before[4:], strict=True)
        },
        "runner_sha256": before[1],
        "candidate_sha256": _sha256(args.output / "candidate.npz"),
        "pair_dtype": str(pair.dtype),
        "comparison": comparison,
        "prefix_comparison": prefix_comparison,
        "prefix_scope": (
            "separate JIT of the same normalization/projection helpers; "
            "up to three entries per non-channel axis; not a full execution trace"
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(comparison))


if __name__ == "__main__":
    main()
