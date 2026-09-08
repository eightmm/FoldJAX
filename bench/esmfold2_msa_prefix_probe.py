"""First taped MSA loop through the first OPM projection; no model admission."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np

from bench.esmfold2_lm_encoder_candidate import compiler_control
from bench.esmfold2_tape import _npz, _save_npz, _sha256, load_native_input_control


def msa_inputs(features, tape):
    msa, mask = features["msa"], features["msa_attention_mask"]
    hd, dv = features["has_deletion"], features["deletion_value"]
    choices, keep = tape["msa_row_choices"], tape["msa_column_keep"]
    if choices.ndim != 2 or choices.shape[0] < 1:
        raise ValueError("first-loop MSA row tape is missing")
    rows = choices[0]
    if (
        msa.ndim != 3
        or mask.shape != msa.shape
        or hd.shape != msa.shape
        or dv.shape != msa.shape
        or msa.dtype.kind not in "iu"
        or not np.isin(mask, [0, 1]).all()
        or not np.isin(hd, [0, 1]).all()
        or not np.isfinite(dv).all()
        or np.any((msa < 0) | (msa >= 33))
        or rows.ndim != 1
        or rows.dtype.kind not in "iu"
        or rows.size < 1
        or rows[0] != 0
        or np.any(np.diff(rows) <= 0)
        or rows[-1] >= msa.shape[1]
        or keep.shape != (msa.shape[0], msa.shape[2])
        or keep.dtype != np.bool_
    ):
        raise ValueError("first-loop MSA feature/tape contract differs")
    # Published order: column mask once, preserve query, then select sorted rows.
    masked = mask.astype(bool) & keep[:, None]
    masked[:, 0] = mask[:, 0]
    selected_mask = masked[:, rows].swapaxes(1, 2).astype(np.float32)
    one_hot = np.eye(33, dtype=np.float32)[msa[:, rows].swapaxes(1, 2)]
    return {
        "msa_oh": one_hot * selected_mask[..., None],
        "msa_attention_mask": selected_mask,
        "has_deletion": hd[:, rows].swapaxes(1, 2).astype(np.float32),
        "deletion_value": dv[:, rows].swapaxes(1, 2).astype(np.float32),
    }


class MsaBoundaryError(Exception):
    pass


def candidate_prefix(
    embedders,
    trunk,
    inputs,
    embedding,
    pair,
    params,
    layers,
    native_opm=False,
    full_opm=False,
    wout_input=None,
    cuda_opm_norm=False,
):
    import jax.numpy as jnp

    saved = {}
    if wout_input is not None and not full_opm:
        raise ValueError("Wout input control requires full OPM capture")
    originals = (
        embedders.linear,
        trunk.linear,
        trunk.layer_norm,
        trunk._autocast_linear,
        embedders.outer_product_mean,
        trunk._autocast_norm,
    )

    def project(original, x, p, prefix, *a, **kw):
        key = prefix.removeprefix("msa_encoder.")
        if key + ".input" in saved:
            raise ValueError("duplicate MSA projection boundary")
        if key == "blocks.0.outer_product_mean.Wout" and wout_input is not None:
            if wout_input.shape != x.shape:
                raise ValueError("native Wout control shape differs")
            x = wout_input.astype(x.dtype)
        saved[key + ".input"] = x
        result = original(x, p, prefix, *a, **kw)
        saved[key + ".output"] = result
        if key == "blocks.0.outer_product_mean.W" and not full_opm:
            raise MsaBoundaryError
        return result

    def linear(x, p, prefix, *a, **kw):
        original = (
            originals[1] if prefix.startswith("msa_encoder.blocks.") else originals[0]
        )
        return project(original, x, p, prefix, *a, **kw)

    def native_linear(x, p, prefix):
        return project(originals[3], x, p, prefix)

    def native_norm(x, p, prefix, eps):
        if cuda_opm_norm:
            return norm(
                x.astype(jnp.float32),
                p[prefix + ".weight"],
                p[prefix + ".bias"],
                eps=eps,
            )
        key = "blocks.0.outer_product_mean.norm"
        if key + ".input" in saved:
            raise ValueError("duplicate MSA norm boundary")
        saved[key + ".input"] = x.astype(jnp.float32)
        result = originals[5](x, p, prefix, eps)
        saved[key + ".output"] = result
        return result

    def opm(*a, **kw):
        result = originals[4](*a, **kw)
        saved["blocks.0.outer_product_mean.output"] = result
        raise MsaBoundaryError

    def norm(x, *a, **kw):
        if "blocks.0.outer_product_mean.norm.input" in saved:
            raise ValueError("duplicate MSA norm boundary")
        saved["blocks.0.outer_product_mean.norm.input"] = x
        if cuda_opm_norm:
            from foldjax.models.boltz2.models.primitives.native_amp_norm import (
                _cuda_layer_norm,
            )

            if x.shape[-1] != 128:
                raise ValueError("OPM norm control requires width 128")
            result = _cuda_layer_norm(x, *a, **kw)[0]
        else:
            result = originals[2](x, *a, **kw)
        saved["blocks.0.outer_product_mean.norm.output"] = result
        return result

    embedders.linear = trunk.linear = linear
    if native_opm:
        trunk._autocast_norm = native_norm
    else:
        trunk.layer_norm = norm
    trunk._autocast_linear = native_linear
    if full_opm:
        embedders.outer_product_mean = opm
    try:
        try:
            embedders.msa_encoder(
                pair.astype(jnp.bfloat16),
                embedding.astype(jnp.bfloat16),
                inputs["msa_oh"].astype(jnp.bfloat16),
                inputs["has_deletion"].astype(jnp.bfloat16),
                inputs["deletion_value"].astype(jnp.bfloat16),
                inputs["msa_attention_mask"],
                {k: v.astype(jnp.bfloat16) for k, v in params.items()},
                "msa_encoder",
                n_layers=layers,
                native_opm_params=params if native_opm else None,
            )
        except MsaBoundaryError:
            pass
        if (
            len(saved) != (11 if full_opm else 8)
            or "blocks.0.outer_product_mean.W.output" not in saved
        ):
            raise ValueError("first MSA projection boundary missing or duplicated")
        return saved
    finally:
        (
            embedders.linear,
            trunk.linear,
            trunk.layer_norm,
            trunk._autocast_linear,
            embedders.outer_product_mean,
            trunk._autocast_norm,
        ) = originals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", choices=("native", "jax"), required=True)
    p.add_argument("--native-opm", action="store_true")
    p.add_argument("--full-opm", action="store_true")
    p.add_argument("--native-wout-input", type=Path)
    p.add_argument("--cuda-opm-norm", action="store_true")
    for name in (
        "embedding",
        "pair",
        "reference",
        "features",
        "weights",
        "source",
        "output",
    ):
        p.add_argument(f"--{name}", type=Path, required=True)
    args = p.parse_args()
    if args.cuda_opm_norm and (args.engine != "jax" or not args.native_opm):
        raise ValueError("CUDA OPM norm requires native-autocast candidate")
    if args.native_wout_input and (args.engine != "jax" or not args.full_opm):
        raise ValueError("native Wout input requires JAX full OPM capture")
    if args.native_opm and args.engine != "jax":
        raise ValueError("native-opm selects the candidate runtime policy only")
    reference = json.loads((args.reference / "metadata.json").read_text())
    config = json.loads((args.weights / "config.json").read_text())
    tape_path = args.reference / "tape.npz"
    if (
        config != reference["config"]
        or _sha256(args.features) != reference["input_sha256"]
        or _sha256(tape_path) != reference["tape_sha256"]
    ):
        raise ValueError("native config/feature/tape identity differs")
    features, tape = _npz(args.features), _npz(tape_path)
    embedding, bindings = load_native_input_control(
        args.embedding,
        args.reference,
        args.weights,
        (*features["token_attention_mask"].shape, config["inputs"]["d_inputs"]),
    )
    pair_report = json.loads((args.pair / "report.json").read_text())
    if (
        pair_report["engine"] != "native"
        or _sha256(args.pair / "pair.npz") != pair_report["archive_sha256"]
    ):
        raise ValueError("requires intact native pair capture")
    for path in (
        args.features,
        args.reference / "metadata.json",
        args.embedding / "native.npz",
    ):
        if pair_report["bindings"].get(str(path.resolve())) != _sha256(path):
            raise ValueError("native pair belongs to different inputs/reference")
    for path, digest in pair_report["bindings"].items():
        if _sha256(Path(path)) != digest:
            raise ValueError("native pair binding changed")
    bindings.update(pair_report["bindings"])
    pair = _npz(args.pair / "pair.npz")["z_init"]
    expected_pair = (
        embedding.shape[0],
        embedding.shape[1],
        embedding.shape[1],
        config["d_pair"],
    )
    if (
        pair.shape != expected_pair
        or pair.dtype != np.float32
        or not np.isfinite(pair).all()
    ):
        raise ValueError("native pair shape/dtype/finite contract differs")
    prepared = msa_inputs(features, tape)
    paths = [
        Path(__file__),
        Path(inspect.getfile(load_native_input_control)),
        Path(inspect.getfile(compiler_control)),
        args.features,
        tape_path,
        args.pair / "report.json",
        args.pair / "pair.npz",
    ]
    bindings.update({str(path.resolve()): _sha256(path) for path in paths})
    args.output.mkdir(parents=True, exist_ok=False)
    prefixes = (
        "msa_encoder.embed.",
        "msa_encoder.project_inputs.",
        "msa_encoder.blocks.0.outer_product_mean.norm.",
        "msa_encoder.blocks.0.outer_product_mean.W.",
    )
    if args.full_opm:
        prefixes += ("msa_encoder.blocks.0.outer_product_mean.Wout.",)
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
            raise ValueError("native source identity differs")
        if torch.cuda.device_count() != 1:
            raise ValueError("one CUDA GPU required")
        paths = list(source.parent.rglob("*.py"))
        bindings.update({str(path.resolve()): _sha256(path) for path in paths})
        with torch.device("meta"):
            model = ESMFold2Model(ESMFold2Config(**config)).msa_encoder
        modules = {
            name: model.get_submodule(name)
            for name in (
                "embed",
                "project_inputs",
                "blocks.0.outer_product_mean.norm",
                "blocks.0.outer_product_mean.W",
            )
        }
        if args.full_opm:
            modules["blocks.0.outer_product_mean.Wout"] = model.get_submodule(
                "blocks.0.outer_product_mean.Wout"
            )
        with safe_open(args.weights / "model.safetensors", framework="pt") as handle:
            for name, module in modules.items():
                module.to_empty(device="cuda")
                prefix = "msa_encoder." + name + "."
                state = {
                    k.removeprefix(prefix): handle.get_tensor(k)
                    for k in handle.keys()
                    if k.startswith(prefix)
                }
                module.load_state_dict(state, strict=True)
        model.eval()
        values, hooks = {}, []

        def hook_for(name):
            def hook(module, positional, output):
                if name + ".input" in values:
                    raise ValueError("duplicate native MSA boundary")
                values[name + ".input"] = positional[0]
                values[name + ".output"] = output
                if name == "blocks.0.outer_product_mean.W" and not args.full_opm:
                    raise MsaBoundaryError

            return hook

        try:
            if args.full_opm:

                def finish_opm(module, positional, output):
                    values["blocks.0.outer_product_mean.output"] = output
                    raise MsaBoundaryError

                hooks.append(
                    model.get_submodule(
                        "blocks.0.outer_product_mean"
                    ).register_forward_hook(finish_opm)
                )
            for name, module in modules.items():
                hooks.append(module.register_forward_hook(hook_for(name)))
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                try:
                    model(
                        x_pair=torch.as_tensor(pair, device="cuda").bfloat16(),
                        x_inputs=torch.as_tensor(embedding, device="cuda"),
                        **{
                            k: torch.as_tensor(v, device="cuda")
                            for k, v in prepared.items()
                        },
                    )
                except MsaBoundaryError:
                    pass
        finally:
            for hook in hooks:
                hook.remove()
        if len(values) != (11 if args.full_opm else 8):
            raise ValueError("native MSA boundaries missing")
        dtypes = {k: str(v.dtype).removeprefix("torch.") for k, v in values.items()}
        arrays = {k: v.detach().float().cpu().numpy() for k, v in values.items()}
        runtime = {
            "torch": str(torch.__version__),
            "torch_git": torch.version.git_version,
        }
    else:
        import jax
        import jax.numpy as jnp
        from safetensors import safe_open

        from foldjax.models.esmfold2.models import embedders, trunk

        source = Path(inspect.getfile(embedders)).resolve()
        if (
            source
            != (
                args.source / "src/foldjax/models/esmfold2/models/embedders.py"
            ).resolve()
        ):
            raise ValueError("candidate source differs")
        if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
            raise ValueError("one CUDA GPU required")
        paths = [
            *source.parent.rglob("*.py"),
            args.source / "src/foldjax/models/_cp.py",
        ]
        if args.cuda_opm_norm or args.native_opm:
            from foldjax.models.boltz2.models.primitives import native_amp_norm

            paths.append(Path(inspect.getfile(native_amp_norm)))
        bindings.update({str(path.resolve()): _sha256(path) for path in paths})
        with safe_open(args.weights / "model.safetensors", framework="numpy") as handle:
            params = {
                k: jnp.asarray(handle.get_tensor(k))
                for k in handle.keys()
                if k.startswith(prefixes)
            }
        control = None
        if args.native_wout_input:
            root = args.native_wout_input
            report = json.loads((root / "report.json").read_text())
            if report["engine"] != "native" or not report.get("full_opm"):
                raise ValueError("Wout control requires native full OPM capture")
            if _sha256(root / "prefix.npz") != report["archive_sha256"]:
                raise ValueError("Wout control archive changed")
            for path in (
                args.features,
                args.weights / "model.safetensors",
                args.reference / "metadata.json",
                args.pair / "pair.npz",
            ):
                if report["bindings"].get(str(path.resolve())) != _sha256(path):
                    raise ValueError("Wout control input binding differs")
            for path, digest in report["bindings"].items():
                if _sha256(Path(path)) != digest:
                    raise ValueError("Wout control source binding changed")
            bindings.update(report["bindings"])
            bindings.update(
                {
                    str((root / name).resolve()): _sha256(root / name)
                    for name in ("report.json", "prefix.npz")
                }
            )
            with np.load(root / "prefix.npz", allow_pickle=False) as archive:
                control = archive["blocks.0.outer_product_mean.Wout.input"]
            if (
                control.dtype != np.float32
                or not np.isfinite(control).all()
                or np.any(control.view(np.uint32) & 0xFFFF)
            ):
                raise ValueError("Wout control must be losslessly stored BF16")
        jax.config.update("jax_default_matmul_precision", "highest")
        run = jax.jit(
            lambda v, x, z, w, c: candidate_prefix(
                embedders,
                trunk,
                v,
                x,
                z,
                w,
                config["msa_encoder"]["n_layers"],
                args.native_opm,
                args.full_opm,
                c,
                args.cuda_opm_norm,
            ),
            compiler_options=compiler_control("native-chunks-strict-rounding"),
        )
        values = run(
            {k: jnp.asarray(v) for k, v in prepared.items()},
            jnp.asarray(embedding),
            jnp.asarray(pair),
            params,
            None if control is None else jnp.asarray(control),
        )
        dtypes = {k: str(v.dtype) for k, v in values.items()}
        arrays = {k: np.asarray(v.astype(jnp.float32)) for k, v in values.items()}
        runtime = {"jax": jax.__version__}
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("nonfinite MSA prefix")
    _save_npz(args.output / "prefix.npz", arrays)
    if any(_sha256(Path(p)) != h for p, h in bindings.items()):
        raise ValueError("bound artifacts changed during capture")
    report = {
        "scope": (
            "shared native embedding/pair and constructed first-loop tape features; "
            "MSA prefix only"
        ),
        "full_model_admission": None,
        "engine": args.engine,
        "native_opm": args.native_opm,
        "full_opm": args.full_opm,
        "cuda_opm_norm": args.cuda_opm_norm,
        "native_wout_input": str(args.native_wout_input)
        if args.native_wout_input
        else None,
        "runtime": runtime,
        "bindings": bindings,
        "dtypes": dtypes,
        "archive_sha256": _sha256(args.output / "prefix.npz"),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dtypes))


if __name__ == "__main__":
    main()
