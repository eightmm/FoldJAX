"""Pair initialization on one fixed native embedding, before MSA/recurrence."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np

from bench.esmfold2_lm_encoder_candidate import compiler_control
from bench.esmfold2_tape import (
    _model_features,
    _npz,
    _save_npz,
    _sha256,
    load_native_input_control,
)

PARTS = ("z_init_1", "z_init_2", "rel_pos", "token_bonds")
INDICES = ("residue_index", "asym_id", "sym_id", "entity_id", "token_index")


class PairBoundaryError(Exception):
    pass


def candidate_prefix(model, features, params, embedding, settings):
    """Trace actual predict to its second pair-sharding boundary, then stop."""
    import jax

    originals = {
        name: getattr(model, name)
        for name in (
            "inputs_embedding",
            "linear",
            "relative_position_encoding",
            "shard_pair_rows",
        )
    }
    captured, shards, inputs = {}, [], []

    def substitute(*args, **kwargs):
        inputs.append(True)
        return embedding

    def linear(x, p, prefix, *a, **kw):
        result = originals["linear"](x, p, prefix, *a, **kw)
        if prefix in ("z_init_1", "z_init_2", "token_bonds"):
            if prefix in captured:
                raise ValueError("duplicate pair projection")
            captured[prefix] = result
        return result

    def relative(*args, **kwargs):
        result = originals["relative_position_encoding"](*args, **kwargs)
        captured["rel_pos"] = result
        return result

    def shard(value, *args, **kwargs):
        shards.append(value)
        if len(shards) == 2:
            captured["z_init"] = value
            raise PairBoundaryError
        return originals["shard_pair_rows"](value, *args, **kwargs)

    try:
        model.inputs_embedding, model.linear = substitute, linear
        model.relative_position_encoding, model.shard_pair_rows = relative, shard
        try:
            model.predict(
                jax.random.key(101), features, params, settings=settings, n_chains=2
            )
        except PairBoundaryError:
            pass
        if (
            len(inputs) != 1
            or len(shards) != 2
            or set(captured) != set(PARTS) | {"z_init"}
        ):
            raise ValueError(
                "actual pair initialization boundary was not reached exactly once"
            )
        return captured
    finally:
        for name, value in originals.items():
            setattr(model, name, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("native", "jax"), required=True)
    for name in ("embedding", "reference", "features", "weights", "source", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads((args.reference / "metadata.json").read_text())
    config = json.loads((args.weights / "config.json").read_text())
    if (
        config != reference["config"]
        or _sha256(args.features) != reference["input_sha256"]
    ):
        raise ValueError("native config/features identity differs")
    features = _npz(args.features)
    shape = (*features["token_attention_mask"].shape, config["inputs"]["d_inputs"])
    embedding, bindings = load_native_input_control(
        args.embedding, args.reference, args.weights, shape
    )
    paths = [
        Path(__file__),
        Path(inspect.getfile(load_native_input_control)),
        Path(inspect.getfile(compiler_control)),
        args.features,
    ]
    bindings.update({str(p.resolve()): _sha256(p) for p in paths})
    args.output.mkdir(parents=True, exist_ok=False)
    if args.engine == "native":
        sys.path.insert(0, str(args.source / "src"))
        import torch
        from safetensors import safe_open
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
            raise ValueError("native source differs")
        if torch.cuda.device_count() != 1:
            raise ValueError("one CUDA device required")
        for p in source.parent.rglob("*.py"):
            bindings[str(p.resolve())] = _sha256(p)
        with torch.device("meta"):
            full = ESMFold2Model(ESMFold2Config(**config))
        parts = torch.nn.ModuleDict({name: getattr(full, name) for name in PARTS})
        del full
        parts.to_empty(device="cuda")
        with safe_open(args.weights / "model.safetensors", framework="pt") as handle:
            state = {
                key: handle.get_tensor(key)
                for key in handle.keys()
                if key.split(".")[0] in PARTS
            }
        parts.load_state_dict(state, strict=True)
        parts.eval()
        indices = {k: torch.as_tensor(features[k], device="cuda") for k in INDICES}
        bonds = torch.as_tensor(features["token_bonds"], device="cuda").float()
        if bonds.ndim == 3:
            bonds = bonds.unsqueeze(-1)
        x = torch.as_tensor(embedding, device="cuda")

        def run():
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                left, right = parts["z_init_1"](x), parts["z_init_2"](x)
                relative = parts["rel_pos"](**indices)
                bond = parts["token_bonds"](bonds)
                # Published eager composition, retaining each BF16 addition.
                z = left.unsqueeze(2) + right.unsqueeze(1)
                z = z + relative + bond
                return dict(
                    z_init_1=left,
                    z_init_2=right,
                    rel_pos=relative,
                    token_bonds=bond,
                    z_init=z,
                )

        values = run()
        repeated = run()
        repeat_equal = all(torch.equal(values[k], repeated[k]) for k in values)
        dtypes = {k: str(v.dtype).removeprefix("torch.") for k, v in values.items()}
        arrays = {k: v.float().cpu().numpy() for k, v in values.items()}
        runtime = {
            "torch": str(torch.__version__),
            "torch_git": torch.version.git_version,
        }
        options = None
    else:
        import jax
        import jax.numpy as jnp
        from safetensors import safe_open

        from foldjax.models.esmfold2.models import model

        source = Path(inspect.getfile(model)).resolve()
        if (
            source
            != (args.source / "src/foldjax/models/esmfold2/models/model.py").resolve()
        ):
            raise ValueError("candidate source differs")
        if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
            raise ValueError("one CUDA device required")
        paths = [
            *source.parent.rglob("*.py"),
            args.source / "src/foldjax/models/_cp.py",
            args.source / "src/foldjax/models/_random.py",
            args.source / "src/foldjax/models/esmfold2/data/all_atom.py",
        ]
        bindings.update({str(p.resolve()): _sha256(p) for p in paths})
        with safe_open(args.weights / "model.safetensors", framework="numpy") as handle:
            params = {
                key: jnp.asarray(handle.get_tensor(key))
                for key in handle.keys()
                if key.split(".")[0] in PARTS
            }
        settings = model.settings_from_config(config)
        jax.config.update("jax_default_matmul_precision", "highest")
        options = compiler_control("native-chunks-strict-rounding")
        compiled = jax.jit(
            lambda f, p, x: candidate_prefix(model, f, p, x, settings),
            compiler_options=options,
        )
        arguments = (
            {k: jnp.asarray(v) for k, v in _model_features(features).items()},
            params,
            jnp.asarray(embedding),
        )
        values = compiled(*arguments)
        repeated = compiled(*arguments)
        dtypes = {k: str(v.dtype) for k, v in values.items()}
        arrays = {k: np.asarray(v.astype(jnp.float32)) for k, v in values.items()}
        repeat_equal = all(
            np.array_equal(arrays[k], np.asarray(repeated[k].astype(jnp.float32)))
            for k in values
        )
        runtime = {"jax": jax.__version__}
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("nonfinite pair initialization")
    _save_npz(args.output / "pair.npz", arrays)
    if any(_sha256(Path(p)) != h for p, h in bindings.items()):
        raise ValueError("bound artifacts changed during pair capture")
    report = {
        "scope": (
            "fixed native embedding; pair initialization only; no full-model admission"
        ),
        "full_model_admission": None,
        "engine": args.engine,
        "runtime": runtime,
        "bindings": bindings,
        "dtypes": dtypes,
        "compiler_options": options,
        "repeat_equal": repeat_equal,
        "archive_sha256": _sha256(args.output / "pair.npz"),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"engine": args.engine, "dtypes": dtypes, "repeat_equal": repeat_equal}
        )
    )


if __name__ == "__main__":
    main()
