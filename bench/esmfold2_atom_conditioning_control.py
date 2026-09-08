"""First learned atom stage: current route versus native-autocast control."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np

from bench.esmfold2_atom_encoder_probe import difference, validate_boundary
from bench.esmfold2_lm_encoder_candidate import compiler_control
from bench.esmfold2_tape import _npz, _save_npz, _sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("boundary", "reference", "weights", "source", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--compiler-profile",
        choices=(
            "default",
            "native-chunks-strict-rounding",
            "strict-rounding-only",
            "no-triton-gemm",
            "no-triton-no-padding",
        ),
        default="default",
    )
    args = parser.parse_args()
    _, arrays = validate_boundary(args.boundary)
    reference = json.loads((args.reference / "report.json").read_text())
    archive = args.reference / "native.npz"
    weight_path = args.weights / "model.safetensors"
    if _sha256(archive) != reference["archive_sha256"] or _sha256(
        weight_path
    ) != reference["bindings"].get(str(weight_path.resolve())):
        raise ValueError("native archive/weight identity differs")
    boundary_archive = args.boundary / "inputs.npz"
    if reference["bindings"].get(str(boundary_archive.resolve())) != _sha256(
        boundary_archive
    ):
        raise ValueError("native input identity differs")
    native = _npz(archive)
    import jax
    import jax.numpy as jnp
    from safetensors import safe_open

    from foldjax.models.esmfold2.models import atom, trunk

    source = Path(inspect.getfile(atom)).resolve()
    if source != (args.source / "src/foldjax/models/esmfold2/models/atom.py").resolve():
        raise ValueError("candidate source differs")
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise ValueError("one CUDA GPU required")
    paths = [
        Path(__file__),
        Path(inspect.getfile(validate_boundary)),
        Path(inspect.getfile(_sha256)),
        Path(inspect.getfile(compiler_control)),
        Path(__file__).with_name("esmfold2_input_boundary.py"),
        archive,
        args.reference / "report.json",
        boundary_archive,
        args.boundary / "report.json",
        weight_path,
        *source.parent.rglob("*.py"),
        args.source / "src/foldjax/models/_cp.py",
        args.source / "src/foldjax/models/boltz2/models/primitives/native_amp_norm.py",
    ]
    bindings = {str(path.resolve()): _sha256(path) for path in paths}
    args.output.mkdir(parents=True, exist_ok=False)
    prefix = "inputs_embedder.atom_attention_encoder"
    with safe_open(weight_path, framework="numpy") as handle:
        params = {
            key: jnp.asarray(handle.get_tensor(key))
            for key in handle.keys()
            if key.startswith(prefix + ".atom_linear.")
            or key.startswith(prefix + ".atom_norm.")
        }
    jax.config.update("jax_default_matmul_precision", "highest")

    def run(values, weights, native_projected):
        rounded = {
            k: v.astype(jnp.bfloat16) if jnp.issubdtype(v.dtype, jnp.floating) else v
            for k, v in values.items()
        }
        low_weights = {k: v.astype(jnp.bfloat16) for k, v in weights.items()}

        def features(v):
            return atom.atom_features(
                v["ref_pos"],
                v["ref_charge"],
                v["atom_attention_mask"],
                v["ref_element"],
                v["ref_atom_name_chars"],
            )

        old_features = features(rounded)
        old_linear = atom.linear(old_features, low_weights, prefix + ".atom_linear")
        old_norm = atom.layer_norm(
            old_linear,
            low_weights[prefix + ".atom_norm.weight"],
            low_weights[prefix + ".atom_norm.bias"],
        )
        full_features = features(values)
        projected = trunk._autocast_linear(
            full_features, weights, prefix + ".atom_linear"
        )
        normed = trunk._autocast_norm(projected, weights, prefix + ".atom_norm")
        norm_control = trunk._autocast_norm(
            native_projected, weights, prefix + ".atom_norm"
        )
        return {
            "legacy.features": old_features,
            "legacy.linear": old_linear,
            "legacy.norm": old_norm,
            "autocast.features": full_features,
            "autocast.linear": projected,
            "autocast.norm": normed,
            "native_linear.norm": norm_control,
        }

    native_prefix = "atom_attention_encoder"
    native_projected = native[native_prefix + ".atom_linear.output"]
    compile_options = (
        {"xla_allow_excess_precision": False}
        if args.compiler_profile == "strict-rounding-only"
        else compiler_control(args.compiler_profile)
    )
    values = jax.jit(run, compiler_options=compile_options)(
        {k: jnp.asarray(v) for k, v in arrays.items()},
        params,
        jnp.asarray(native_projected, dtype=jnp.bfloat16),
    )
    dtypes = {k: str(v.dtype) for k, v in values.items()}
    output = {k: np.asarray(v.astype(jnp.float32)) for k, v in values.items()}
    if not all(np.isfinite(v).all() for v in output.values()):
        raise ValueError("nonfinite candidate stage")
    comparisons = {}
    for name, value in output.items():
        stage = name.split(".")[1]
        key = {
            "features": ".atom_linear.input",
            "linear": ".atom_linear.output",
            "norm": ".atom_norm.output",
        }[stage]
        comparisons[name] = difference(value, native[native_prefix + key])
    _save_npz(args.output / "candidate.npz", output)
    if any(_sha256(Path(path)) != digest for path, digest in bindings.items()):
        raise ValueError("bound artifact changed during comparison")
    report = {
        "scope": (
            "shared native atom inputs; first Linear/LayerNorm only; "
            "control not wired into prediction"
        ),
        "full_model_admission": None,
        "jax": jax.__version__,
        "compiler_profile": args.compiler_profile,
        "compiler_options": compile_options,
        "bindings": bindings,
        "dtypes": dtypes,
        "comparisons": comparisons,
        "archive_sha256": _sha256(args.output / "candidate.npz"),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"dtypes": dtypes, "comparisons": comparisons}))


if __name__ == "__main__":
    main()
