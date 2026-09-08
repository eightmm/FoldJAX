"""First taped MSA loop through the first OPM projection; no model admission."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from contextlib import contextmanager, nullcontext, suppress
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bench.esmfold2_lm_encoder_candidate import compiler_control
from bench.esmfold2_tape import _npz, _save_npz, _sha256, load_native_input_control


def stack_compiler_control(disable_fusion=False):
    options = compiler_control("native-chunks-strict-rounding")
    if disable_fusion:
        options["xla_disable_hlo_passes"] += ",fusion"
    return options


def execute_capture(run, operands, output=None):
    if output is None:
        return run(*operands), {}
    lowered = run.lower(*operands)
    executable = lowered.compile()
    bindings = {}
    for name, content in (
        ("lowered.hlo.txt", lowered.as_text()),
        ("compiled.hlo.txt", executable.as_text()),
    ):
        path = output / name
        with path.open("x") as handle:
            handle.write(content)
        bindings[str(path.resolve())] = _sha256(path)
    return executable(*operands), bindings


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


def validate_transition_chunks(shapes, width, chunk_size):
    """Check observed calls, including the partial final native chunk."""
    if width < 1 or (chunk_size is not None and chunk_size < 1):
        raise ValueError("invalid native transition chunk dimensions")
    step = width if chunk_size is None else chunk_size
    expected = [min(step, width - i) for i in range(0, width, step)]
    if any(len(shape) < 2 for shape in shapes) or [s[1] for s in shapes] != expected:
        raise ValueError("native transition chunk geometry differs")


def observe_pwa_ops(torch, call, values, *args, **kwargs):
    """Observe only this PWA call, preserving native operations and arguments."""
    prefix = "blocks.0.msa_pair_weighted_averaging."
    softmax, einsum = torch.softmax, torch.einsum

    def record(name, tensor):
        key = prefix + name
        if key in values:
            raise ValueError("duplicate PWA operation")
        values[key] = tensor.clone()

    def observed_softmax(x, *a, **kw):
        record("softmax.input", x)
        out = softmax(x, *a, **kw)
        record("softmax.output", out)
        return out

    def observed_einsum(equation, *operands, **kw):
        if equation != "bijh,bjmhd,bimhd->bimhd" or len(operands) != 3:
            raise ValueError("unexpected native PWA contraction")
        for index, operand in enumerate(operands):
            record(f"einsum.input{index}", operand)
        out = einsum(equation, *operands, **kw)
        record("einsum.output", out)
        return out

    with (
        patch.object(torch, "softmax", observed_softmax),
        patch.object(torch, "einsum", observed_einsum),
    ):
        return call(*args, **kwargs)


@contextmanager
def embedding_barrier_control(embedders):
    import jax

    original = embedders.linear

    def linear(x, params, prefix):
        output = original(x, params, prefix)
        if prefix in ("msa_encoder.embed", "msa_encoder.project_inputs"):
            return jax.lax.optimization_barrier(output)
        return output

    with patch.object(embedders, "linear", linear):
        yield


def candidate_stack(
    embedders,
    inputs,
    embedding,
    pair,
    params,
    layers,
    first_only=False,
    capture_inputs=False,
    embedding_barriers=False,
):
    import jax.numpy as jnp

    saved = {}
    original = embedders.msa_encoder_block

    def block(msa, pair, p, prefix, **kw):
        if capture_inputs and prefix == "msa_encoder.blocks.0":
            saved["blocks.0.msa_input"] = msa
            saved["blocks.0.pair_input"] = pair
        result = original(msa, pair, p, prefix, **kw)
        key = prefix.removeprefix("msa_encoder.")
        if key + ".msa" in saved:
            raise ValueError("duplicate MSA stack boundary")
        saved[key + ".msa"], saved[key + ".pair"] = result
        if first_only:
            raise MsaBoundaryError
        return result

    with (
        patch.object(embedders, "msa_encoder_block", block),
        embedding_barrier_control(embedders) if embedding_barriers else nullcontext(),
        suppress(MsaBoundaryError) if first_only else nullcontext(),
    ):
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
            native_opm_params=params,
        )
    if len(saved) != (2 if first_only else 2 * layers) + (2 if capture_inputs else 0):
        raise ValueError("MSA stack boundaries missing")
    return saved


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
    full_pwa=False,
    full_block=False,
    stack=False,
    stack_first_only=False,
    stack_inputs=False,
    embedding_barriers=False,
):
    import jax.numpy as jnp

    if stack:
        return candidate_stack(
            embedders,
            inputs,
            embedding,
            pair,
            params,
            layers,
            stack_first_only,
            stack_inputs,
            embedding_barriers,
        )
    saved = {}
    if full_pwa and (not full_opm or not native_opm or cuda_opm_norm):
        raise ValueError("PWA propagation requires runtime native OPM without override")
    original_pwa = getattr(embedders, "msa_pair_weighted_averaging", None)
    original_transition = getattr(embedders, "transition", None)
    original_triangle = getattr(embedders, "triangle_multiplicative", None)
    if full_block and not full_pwa:
        raise ValueError("full block propagation requires PWA")

    def after_pwa(prefix):
        return full_block and any(
            part in prefix
            for part in (".msa_transition", ".tri_mul_", ".pair_transition")
        )

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
        if after_pwa(prefix):
            return original(x, p, prefix, *a, **kw)
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
        if after_pwa(prefix):
            return originals[5](x, p, prefix, eps)
        if cuda_opm_norm:
            return norm(
                x.astype(jnp.float32),
                p[prefix + ".weight"],
                p[prefix + ".bias"],
                eps=eps,
            )
        key = prefix.removeprefix("msa_encoder.")
        if key + ".input" in saved:
            raise ValueError("duplicate MSA norm boundary")
        saved[key + ".input"] = x.astype(jnp.float32)
        result = originals[5](x, p, prefix, eps)
        saved[key + ".output"] = result
        return result

    def opm(*a, **kw):
        result = originals[4](*a, **kw)
        saved["blocks.0.outer_product_mean.output"] = result
        if not full_pwa:
            raise MsaBoundaryError
        return result

    def pwa(msa, pair, params, prefix, *, pair_mask, **kw):
        result = original_pwa(msa, pair, params, prefix, pair_mask=pair_mask, **kw)
        key = "blocks.0.msa_pair_weighted_averaging."
        saved.update(
            {
                key + "msa_input": msa,
                key + "pair_input": pair,
                key + "mask_input": pair_mask,
                key + "output": result,
            }
        )
        if not full_block:
            raise MsaBoundaryError
        return result

    def update(original, x, p, prefix, **kw):
        key = prefix.removeprefix("msa_encoder.")
        result = original(x, p, prefix, **kw)
        saved[key + ".input"] = x
        saved[key + ".output"] = result
        if key == "blocks.0.pair_transition":
            raise MsaBoundaryError
        return result

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
    if full_pwa:
        embedders.msa_pair_weighted_averaging = pwa
    if full_block:
        embedders.transition = lambda *a, **kw: update(original_transition, *a, **kw)
        embedders.triangle_multiplicative = lambda *a, **kw: update(
            original_triangle, *a, **kw
        )
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
            len(saved)
            != (35 if full_block else 27 if full_pwa else 11 if full_opm else 8)
            or "blocks.0.outer_product_mean.W.output" not in saved
        ):
            raise ValueError("first MSA projection boundary missing or duplicated")
        return saved
    finally:
        if full_block:
            embedders.transition = original_transition
            embedders.triangle_multiplicative = original_triangle
        if full_pwa:
            embedders.msa_pair_weighted_averaging = original_pwa
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
    p.add_argument("--full-pwa", action="store_true")
    p.add_argument("--full-msa-transition", action="store_true")
    p.add_argument("--full-msa-block", action="store_true")
    p.add_argument("--ffi-library", type=Path)
    p.add_argument("--stack", action="store_true")
    p.add_argument("--disable-fusion", action="store_true")
    p.add_argument("--capture-hlo", action="store_true")
    p.add_argument("--stack-first-only", action="store_true")
    p.add_argument("--stack-inputs", action="store_true")
    p.add_argument("--embedding-barriers", action="store_true")
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
    if args.embedding_barriers and (args.engine != "jax" or not args.stack):
        raise ValueError("embedding barriers require JAX stack")
    if args.stack_inputs and (args.engine != "jax" or not args.stack):
        raise ValueError("stack input capture requires JAX stack")
    if args.stack_first_only and (args.engine != "jax" or not args.stack):
        raise ValueError("first-only control requires JAX stack capture")
    if args.capture_hlo and args.engine != "jax":
        raise ValueError("HLO capture requires JAX")
    if args.disable_fusion and (args.engine != "jax" or not args.stack):
        raise ValueError("fusion control requires JAX stack capture")
    if args.stack and (
        args.full_opm
        or args.full_pwa
        or args.full_msa_transition
        or args.full_msa_block
        or args.native_wout_input
        or args.cuda_opm_norm
    ):
        raise ValueError("stack capture cannot combine prefix controls")
    if args.stack and args.engine == "jax" and not args.native_opm:
        raise ValueError("JAX stack requires native autocast")
    if args.ffi_library and (args.engine != "jax" or not args.native_opm):
        raise ValueError("FFI capture requires native-autocast JAX")
    if args.full_msa_block and not args.full_msa_transition:
        raise ValueError("full MSA block requires native transition capture")
    if args.full_msa_transition and (
        (args.engine != "native" and not args.full_msa_block) or not args.full_pwa
    ):
        raise ValueError("MSA transition capture requires native full PWA")
    if args.full_pwa and not args.full_opm:
        raise ValueError("full-pwa requires full-opm capture")
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
    if args.full_pwa:
        prefixes += ("msa_encoder.blocks.0.msa_pair_weighted_averaging.",)
    if args.full_msa_block:
        prefixes += ("msa_encoder.blocks.0.",)
    if args.stack:
        prefixes = ("msa_encoder.",)
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
        if args.full_pwa:
            for leaf in (
                "norm_single",
                "compute_bias.0",
                "compute_bias.1",
                "Wv",
                "Wgate",
                "Wout",
            ):
                name = "blocks.0.msa_pair_weighted_averaging." + leaf
                modules[name] = model.get_submodule(name)
        if args.full_msa_transition:
            for leaf in ("norm", "ffn.w12", "ffn.w3"):
                name = "blocks.0.msa_transition." + leaf
                modules[name] = model.get_submodule(name)
        if args.full_msa_block:
            for leaf in ("tri_mul_out", "tri_mul_in", "pair_transition"):
                name = "blocks.0." + leaf
                modules[name] = model.get_submodule(name)
        if args.stack:
            modules = {
                name: model.get_submodule(name)
                for name in (
                    "embed",
                    "project_inputs",
                    *[f"blocks.{i}" for i in range(len(model.blocks))],
                )
            }
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
        transition_chunks = {}
        transition_chunk_geometry = {}
        pwa_module = None
        if args.full_pwa:
            pwa_module = model.get_submodule("blocks.0.msa_pair_weighted_averaging")
            original_pwa = pwa_module.forward
            pwa_module.forward = lambda *a, **kw: observe_pwa_ops(
                torch, original_pwa, values, *a, **kw
            )

        def hook_for(name):
            def hook(module, positional, output):
                if args.stack and name.startswith("blocks."):
                    if name + ".msa" in values:
                        raise ValueError("duplicate native stack boundary")
                    values[name + ".msa"] = output[0].clone()
                    values[name + ".pair"] = output[1].clone()
                    return
                if name.startswith("blocks.0.msa_transition."):
                    for suffix, value in (("input", positional[0]), ("output", output)):
                        transition_chunks.setdefault(name + "." + suffix, []).append(
                            value.clone()
                        )
                    return
                if name + ".input" in values:
                    raise ValueError("duplicate native MSA boundary")
                # PWA masked_fill_ mutates the bias projection output later.
                values[name + ".input"] = positional[0].clone()
                values[name + ".output"] = output.clone()
                if name == "blocks.0.pair_transition":
                    raise MsaBoundaryError
                if name == "blocks.0.outer_product_mean.W" and not args.full_opm:
                    raise MsaBoundaryError

            return hook

        try:
            if args.full_opm:

                def finish_opm(module, positional, output):
                    values["blocks.0.outer_product_mean.output"] = output
                    if not args.full_pwa:
                        raise MsaBoundaryError

                hooks.append(
                    model.get_submodule(
                        "blocks.0.outer_product_mean"
                    ).register_forward_hook(finish_opm)
                )
            if args.full_pwa:

                def finish_pwa(module, positional, output):
                    prefix = "blocks.0.msa_pair_weighted_averaging."
                    for key, value in zip(
                        ("msa_input", "pair_input", "mask_input"),
                        positional,
                        strict=True,
                    ):
                        values[prefix + key] = value.clone()
                    values[prefix + "output"] = output.clone()
                    if not args.full_msa_transition:
                        raise MsaBoundaryError

                hooks.append(
                    model.get_submodule(
                        "blocks.0.msa_pair_weighted_averaging"
                    ).register_forward_hook(finish_pwa)
                )
            if args.full_msa_transition:

                def finish_transition(module, positional, output):
                    chunk_size = module._chunk_size
                    width = positional[0].shape[1]
                    for key, chunks in transition_chunks.items():
                        validate_transition_chunks(
                            [v.shape for v in chunks], width, chunk_size
                        )
                        transition_chunk_geometry[key] = [list(v.shape) for v in chunks]
                        merged = torch.cat(chunks, dim=1)
                        if merged.shape[:2] != positional[0].shape[:2]:
                            raise ValueError("native transition chunk coverage differs")
                        values[key] = merged
                    values["blocks.0.msa_transition.output"] = output.clone()
                    if not args.full_msa_block:
                        raise MsaBoundaryError

                hooks.append(
                    model.get_submodule(
                        "blocks.0.msa_transition"
                    ).register_forward_hook(finish_transition)
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
            if pwa_module is not None:
                pwa_module.forward = original_pwa
        expected_count = (
            46
            if args.full_msa_block
            else 40
            if args.full_msa_transition
            else 33
            if args.full_pwa
            else 11
            if args.full_opm
            else 8
        )
        if args.stack:
            expected_count = 4 + 2 * len(model.blocks)
        if len(values) != expected_count:
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
        if args.ffi_library:
            from bench import native_cublaslt_ffi
            from bench.esmfold2_lm_encoder_candidate import ffi_output_linear
            from bench.esmfold2_tape import dispatch_bias_free_ffi

            ffi_paths = (
                args.ffi_library,
                Path(native_cublaslt_ffi.__file__),
                Path(native_cublaslt_ffi.__file__).with_suffix(".cc"),
            )
            bindings.update({str(p.resolve()): _sha256(p) for p in ffi_paths})
            ffi_target = native_cublaslt_ffi.register(args.ffi_library)
            trunk._autocast_linear = partial(
                dispatch_bias_free_ffi,
                ffi=partial(ffi_output_linear, target=ffi_target),
                fallback=trunk._autocast_linear,
            )
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
                args.full_pwa,
                args.full_msa_block,
                args.stack,
                args.stack_first_only,
                args.stack_inputs,
                args.embedding_barriers,
            ),
            compiler_options=stack_compiler_control(args.disable_fusion),
        )
        operands = (
            {k: jnp.asarray(v) for k, v in prepared.items()},
            jnp.asarray(embedding),
            jnp.asarray(pair),
            params,
            None if control is None else jnp.asarray(control),
        )
        values, hlo_bindings = execute_capture(
            run, operands, args.output if args.capture_hlo else None
        )
        bindings.update(hlo_bindings)
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
        "full_pwa": args.full_pwa,
        "full_msa_transition": args.full_msa_transition,
        "full_msa_block": args.full_msa_block,
        "stack": args.stack,
        "stack_first_only": args.stack_first_only,
        "stack_inputs": args.stack_inputs,
        "embedding_barriers": args.embedding_barriers,
        "compiler_options": stack_compiler_control(args.disable_fusion)
        if args.engine == "jax"
        else None,
        "ffi_native_dispatch": bool(args.ffi_library),
        "transition_chunk_geometry": transition_chunk_geometry
        if args.full_msa_transition and args.engine == "native"
        else None,
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
