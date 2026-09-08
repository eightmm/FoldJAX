"""Actual predict prefix up to InputsEmbedder; no network or admission claim."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np

from bench.esmfold2_tape import _model_features, _npz, _save_npz, _sha256

INPUT_NAMES = (
    "aatype",
    "profile",
    "deletion_mean",
    "ref_pos",
    "atom_attention_mask",
    "ref_space_uid",
    "ref_charge",
    "ref_element",
    "ref_atom_name_chars",
    "atom_to_token",
)


class BoundaryReachedError(Exception):
    pass


def candidate_prefix(module, features, settings):
    """Trace the real prefix and stop before the first learned operation."""
    import jax

    original = module.inputs_embedding
    captured = []

    def stop(*args, **kwargs):
        if len(args) != len(INPUT_NAMES) + 1 or "settings" not in kwargs:
            raise ValueError("candidate input embedding signature differs")
        captured.append(dict(zip(INPUT_NAMES, args[: len(INPUT_NAMES)], strict=True)))
        raise BoundaryReachedError

    module.inputs_embedding = stop
    try:
        try:
            module.predict(
                jax.random.key(101), features, {}, settings=settings, n_chains=1
            )
        except BoundaryReachedError:
            pass
        if len(captured) != 1:
            raise ValueError("candidate input boundary was not reached exactly once")
        return captured[0]
    finally:
        module.inputs_embedding = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("native", "jax"), required=True)
    for name in ("reference", "features", "weights", "source", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads((args.reference / "metadata.json").read_text())
    config_path = args.weights / "config.json"
    config = json.loads(config_path.read_text())
    if (
        config != reference["config"]
        or _sha256(args.features) != reference["input_sha256"]
    ):
        raise ValueError("configuration or shared feature identity differs")
    paths = [
        Path(__file__),
        Path(inspect.getfile(_model_features)),
        args.features,
        config_path,
        args.reference / "metadata.json",
    ]
    bindings = {str(p.resolve()): _sha256(p) for p in paths}
    features = _model_features(_npz(args.features))
    args.output.mkdir(parents=True, exist_ok=False)
    if args.engine == "native":
        sys.path.insert(0, str(args.source / "src"))
        import torch
        from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config
        from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

        source = Path(inspect.getfile(ESMFold2Model)).resolve()
        expected = args.source / "src/transformers/models/esmfold2/modeling_esmfold2.py"
        if (
            source != expected.resolve()
            or _sha256(source)
            != reference["binding"]["source"]["native"][
                "transformers/models/esmfold2/modeling_esmfold2.py"
            ]
        ):
            raise ValueError("native source identity differs")
        if torch.cuda.device_count() != 1:
            raise ValueError("one CUDA GPU required")
        paths.extend(source.parent.rglob("*.py"))
        bindings.update(
            {
                str(p.resolve()): _sha256(p)
                for p in paths
                if str(p.resolve()) not in bindings
            }
        )
        # Weights are never consumed: the pre-hook stops before InputsEmbedder.
        # Meta construction avoids allocating or pretending to validate weights.
        with torch.device("meta"):
            model = ESMFold2Model(ESMFold2Config(**config))
        model.eval()
        captured = []

        def stop(module, positional, kwargs):
            del module
            if positional or set(kwargs) != set(INPUT_NAMES):
                raise ValueError("native input embedding signature differs")
            captured.append(kwargs)
            raise BoundaryReachedError

        hook = model.inputs_embedder.register_forward_pre_hook(stop, with_kwargs=True)
        try:
            with torch.inference_mode():
                try:
                    model(
                        **{
                            k: torch.as_tensor(v, device="cuda")
                            for k, v in features.items()
                        }
                    )
                except BoundaryReachedError:
                    pass
        finally:
            hook.remove()
        if len(captured) != 1:
            raise ValueError("native input boundary was not reached exactly once")
        values = captured[0]
        dtypes = {k: str(v.dtype).removeprefix("torch.") for k, v in values.items()}
        arrays = {k: v.detach().cpu().numpy() for k, v in values.items()}
        runtime = {"torch": torch.__version__, "torch_git": torch.version.git_version}
    else:
        import jax
        import jax.numpy as jnp

        from foldjax.models.esmfold2.models import model

        source = Path(inspect.getfile(model)).resolve()
        if (
            source
            != (args.source / "src/foldjax/models/esmfold2/models/model.py").resolve()
        ):
            raise ValueError("candidate source identity differs")
        if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
            raise ValueError("one CUDA GPU required")
        paths.extend(source.parent.rglob("*.py"))
        paths.append(args.source / "src/foldjax/models/_cp.py")
        paths.append(args.source / "src/foldjax/models/_random.py")
        bindings.update(
            {
                str(p.resolve()): _sha256(p)
                for p in paths
                if str(p.resolve()) not in bindings
            }
        )
        settings = model.settings_from_config(config)
        run = jax.jit(lambda f: candidate_prefix(model, f, settings))
        values = run({k: jnp.asarray(v) for k, v in features.items()})
        dtypes = {k: str(v.dtype) for k, v in values.items()}
        arrays = {
            k: np.asarray(v.astype(jnp.float32) if v.dtype == jnp.bfloat16 else v)
            for k, v in values.items()
        }
        runtime = {"jax": jax.__version__}
    if set(arrays) != set(INPUT_NAMES) or not all(
        np.isfinite(v).all() for v in arrays.values()
    ):
        raise ValueError("boundary values are missing or nonfinite")
    _save_npz(args.output / "inputs.npz", arrays)
    if any(_sha256(Path(p)) != h for p, h in bindings.items()):
        raise ValueError("bound source/input changed during capture")
    report = {
        "scope": (
            "shared-feature predict prefix only; "
            "no weights or learned operations executed"
        ),
        "full_model_admission": None,
        "engine": args.engine,
        "runtime": runtime,
        "bindings": bindings,
        "archive_sha256": _sha256(args.output / "inputs.npz"),
        "fields": {
            k: {
                "dtype": dtypes[k],
                "storage_dtype": str(v.dtype),
                "shape": list(v.shape),
            }
            for k, v in arrays.items()
        },
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["fields"]))


if __name__ == "__main__":
    main()
