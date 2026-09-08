"""Actual candidate InputsEmbedder against captured native learned boundaries."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np

from bench.esmfold2_atom_encoder_probe import difference, validate_boundary
from bench.esmfold2_input_boundary import INPUT_NAMES
from bench.esmfold2_lm_encoder_candidate import compiler_control
from bench.esmfold2_tape import _npz, _save_npz, _sha256


def capture(module, model, execute):
    """Return traced stage outputs while restoring every replaced function."""
    stored, originals = {}, []

    def save(key, value):
        if key in stored:
            raise ValueError(f"repeated atom boundary: {key}")
        stored[key] = value

    def wrap(name, prefix_index):
        original = getattr(module, name)

        def observe(*args, **kwargs):
            prefix = args[prefix_index].removeprefix("inputs_embedder.")
            save(prefix + ".input", args[0])
            output = original(*args, **kwargs)
            save(prefix + ".output", output)
            return output

        originals.append((module, name, original))
        setattr(module, name, observe)

    original_norm, original_encoder = module.layer_norm, model.atom_encoder

    def norm(*args, **kwargs):
        prefix = "atom_attention_encoder.atom_norm"
        save(prefix + ".input", args[0])
        output = original_norm(*args, **kwargs)
        save(prefix + ".output", output)
        return output

    def encoder(*args, **kwargs):
        output = original_encoder(*args, **kwargs)
        prefix = "atom_attention_encoder."
        for label, value in zip(
            ("tokens", "queries", "conditioning"), output[:3], strict=True
        ):
            save(prefix + label, value)
        save(prefix + "rope_cos", output[3][0])
        save(prefix + "rope_sin", output[3][1])
        return output

    try:
        for name, index in (
            ("_atom_linear", 2),
            ("swa_attention", 2),
            ("swiglu_ffn", 2),
            ("swa_block", 3),
        ):
            wrap(name, index)
        originals.extend(
            (
                (module, "layer_norm", original_norm),
                (model, "atom_encoder", original_encoder),
            )
        )
        module.layer_norm, model.atom_encoder = norm, encoder
        result = execute()
        return result, stored
    finally:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("boundary", "reference", "weights", "source", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _, arrays = validate_boundary(args.boundary)
    reference = json.loads((args.reference / "report.json").read_text())
    archive = args.reference / "native.npz"
    if _sha256(archive) != reference["archive_sha256"]:
        raise ValueError("native archive identity differs")
    config_path, weights_path = (
        args.weights / "config.json",
        args.weights / "model.safetensors",
    )
    for path in (config_path, weights_path, args.boundary / "inputs.npz"):
        if _sha256(path) != reference["bindings"].get(str(path.resolve())):
            raise ValueError("native weights/config/input identity differs")
    native = _npz(archive)
    import jax
    import jax.numpy as jnp
    from safetensors import safe_open

    from foldjax.models.esmfold2.models import atom, model

    source = Path(inspect.getfile(model)).resolve()
    if (
        source
        != (args.source / "src/foldjax/models/esmfold2/models/model.py").resolve()
    ):
        raise ValueError("candidate source differs")
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise ValueError("one CUDA GPU required")
    paths = [
        Path(__file__),
        Path(inspect.getfile(validate_boundary)),
        Path(inspect.getfile(compiler_control)),
        Path(inspect.getfile(_sha256)),
        Path(__file__).with_name("esmfold2_input_boundary.py"),
        archive,
        args.reference / "report.json",
        config_path,
        weights_path,
        args.boundary / "inputs.npz",
        args.boundary / "report.json",
        *source.parent.rglob("*.py"),
        args.source / "src/foldjax/models/_cp.py",
        args.source / "src/foldjax/models/boltz2/models/primitives/native_amp_norm.py",
    ]
    bindings = {str(p.resolve()): _sha256(p) for p in paths}
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(config_path.read_text())
    settings = model.settings_from_config(config)
    with safe_open(weights_path, framework="numpy") as handle:
        params = {
            key: jnp.asarray(handle.get_tensor(key))
            for key in handle.keys()
            if key.startswith("inputs_embedder.")
        }
    values = {key: jnp.asarray(value) for key, value in arrays.items()}
    jax.config.update("jax_default_matmul_precision", "highest")
    options = compiler_control("native-chunks-strict-rounding")

    def run(v, p):
        return model.inputs_embedding(
            *(v[name] for name in INPUT_NAMES),
            p,
            settings=settings,
            n_tokens=v["aatype"].shape[1],
            native_autocast=True,
        )

    compiled = jax.jit(run, compiler_options=options)
    baseline = np.asarray(compiled(values, params))
    repeated = np.asarray(compiled(values, params))
    observed, stages = jax.jit(
        lambda v, p: capture(atom, model, lambda: run(v, p)), compiler_options=options
    )(values, params)
    dtypes = {key: str(value.dtype) for key, value in stages.items()}
    stored = {
        key: np.asarray(value.astype(jnp.float32)) for key, value in stages.items()
    }
    if set(stored) != set(reference["dtypes"]):
        raise ValueError("captured native/candidate boundary sets differ")
    comparisons = {key: difference(value, native[key]) for key, value in stored.items()}
    stored.update(baseline=baseline, repeated=repeated, observed=np.asarray(observed))
    if not all(np.isfinite(value).all() for value in stored.values()):
        raise ValueError("nonfinite atom output")
    _save_npz(args.output / "candidate.npz", stored)
    if any(_sha256(Path(path)) != digest for path, digest in bindings.items()):
        raise ValueError("bound source/artifact changed during capture")
    report = {
        "scope": "shared native input learned atom encoder; no full-model admission",
        "full_model_admission": None,
        "bindings": bindings,
        "jax": jax.__version__,
        "compiler_options": options,
        "dtypes": dtypes,
        "dtype_equal": dtypes == reference["dtypes"],
        "comparisons": comparisons,
        "repeat": difference(baseline, repeated),
        "instrumentation": difference(baseline, stored["observed"]),
        "native_output": difference(baseline, native["baseline"]),
        "archive_sha256": _sha256(args.output / "candidate.npz"),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("dtype_equal", "repeat", "instrumentation", "native_output")
            }
        )
    )


if __name__ == "__main__":
    main()
