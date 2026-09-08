"""Observe native Boltz outputs in sampler order before confidence-rank writing.

Development-only extension of the existing native tape recorder. No publisher
arithmetic or random draws are replaced. Detailed trunk observations are limited
to the first and last recycle; instrumented timing is not a performance result.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
import time
from collections import Counter
from collections.abc import Mapping
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bench.af3_closure_capture import sha
from bench.protenix_closure_capture import flatten_native, host_array, source_identity

SINGLE_LEAVES = (
    "pairformer_module.layers.0.pre_norm_s",
    *(f"pairformer_module.layers.0.attention.proj_{name}" for name in "qkvg"),
)


def save_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


def save_tree(out, name, value):
    """Store every leaf, retaining original BF16 dtype beside exact FP32 storage."""
    arrays, metadata = flatten_native(value)
    path = out / f"{name}.npz"
    tree_path = out / f"{name}.tree.json"
    if path.exists() or tree_path.exists():
        raise FileExistsError(f"refusing to overwrite observation {name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    save_new(tree_path, metadata)
    return {"arrays_sha256": sha(path), "tree_sha256": sha(tree_path)}


def writer_order(scores, actual_argsort):
    """Validate the actual writer's ordering, including its native tie behavior."""
    scores, order = host_array(scores), host_array(actual_argsort)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("writer confidence must be a finite sample vector")
    if order.dtype.kind not in "iu" or order.shape != scores.shape:
        raise ValueError("writer argsort must be one integer index per sample")
    if sorted(order.tolist()) != list(range(len(scores))):
        raise ValueError("writer argsort is not a sample permutation")
    if np.any(scores[order][1:] > scores[order][:-1]):
        raise ValueError("writer did not sort confidence in descending order")
    inverse = [0] * len(order)
    for rank, index in enumerate(order.tolist()):
        inverse[index] = rank
    return {
        "rank_to_sample_index": order.tolist(),
        "sample_index_to_rank": inverse,
        "confidence_score": scores.tolist(),
        "source_rule": "torch.argsort(confidence_score, descending=True)",
        "filename_rule": "<record.id>_model_<sample_index_to_rank[sample_index]>",
        "tie_policy": "actual publisher argsort result; no independent tie breaking",
        "comparison_order": "original sample index, never confidence rank",
    }


def native_locals(code):
    """Read the already-computed native frame; never reconstruct initial sums."""
    frame = sys._getframe(1)
    try:
        while frame is not None:
            if frame.f_code is code:
                return {
                    name: frame.f_locals[name] for name in ("s_init", "z_init", "i")
                }
            frame = frame.f_back
    finally:
        del frame
    raise RuntimeError("selected module hook did not originate in native forward")


def conditioning_tree(output):
    if not isinstance(output, tuple) or len(output) != 6:
        raise ValueError("unexpected publisher diffusion conditioning schema")
    q, c, to_keys, enc_bias, dec_bias, token_bias = output
    if not (
        isinstance(to_keys, partial)
        and to_keys.func.__module__ == "boltz.model.modules.encodersv2"
        and to_keys.func.__name__ == "single_to_keys"
        and not to_keys.args
        and set(to_keys.keywords) == {"indexing_matrix", "W", "H"}
    ):
        raise ValueError("unmapped native conditioning callback")
    return {
        "q": q,
        "c": c,
        "atom_enc_bias": enc_bias,
        "atom_dec_bias": dec_bias,
        "token_trans_bias": token_bias,
        "to_keys": {
            "native_type": "functools.partial",
            "function": f"{to_keys.func.__module__}.{to_keys.func.__name__}",
            "function_source_sha256": sha(Path(inspect.getsourcefile(to_keys.func))),
            "positional_argument_count": 0,
            "keywords": to_keys.keywords,
        },
    }


class NativeObserver:
    def __init__(self, out, *, samples, recycles, forward_code, input_details=False):
        self.out = out
        self.samples = samples
        self.recycles = recycles
        self.forward_code = forward_code
        self.input_details = input_details
        self.counts = Counter()
        self.artifacts = {}
        self.module_names = []

    def record(self, name, value):
        self.artifacts[name] = save_tree(self.out, name, value)

    def module_hook(self, name, cyclic=False):
        def observed(module, args, kwargs, output):
            index = self.counts[name]
            self.counts[name] += 1
            if cyclic:
                values = native_locals(self.forward_code)
                if values["i"] != index:
                    raise RuntimeError(
                        f"unexpected native recycle at {name}: {values['i']}"
                    )
                if name == "s_norm" and index == 0:
                    self.record(
                        "trunk-boundaries/initial",
                        {
                            "s_init": values["s_init"],
                            "z_init": values["z_init"],
                        },
                    )
                if index not in {0, self.recycles}:
                    return None
                label = f"trunk-boundaries/cycle-{index:02d}/{name}"
                if name == "msa_module":
                    output = {
                        "input_z": args[0], "input_emb": args[1],
                        "input_features": {key: args[2][key] for key in (
                            "msa", "has_deletion", "deletion_value", "msa_paired",
                            "msa_mask", "token_pad_mask",
                        )},
                        "delta_z": output,
                    }
                elif name == "msa_module.layers.0":
                    output = {"input_z": args[0], "input_m": args[1],
                              "token_mask": args[2], "msa_mask": args[3]}
                elif name in SINGLE_LEAVES:
                    output = {"input": args[0], "output": output}
                elif name == "pairformer_module":
                    output = {
                        "input_s": args[0],
                        "input_z": args[1],
                        "output_s": output[0],
                        "output_z": output[1],
                    }
            else:
                label = f"trunk-boundaries/{name}"
                if name == "diffusion_conditioning":
                    output = conditioning_tree(output)
                elif name in {
                    "input_embedder.atom_encoder",
                    "input_embedder.atom_attention_encoder",
                }:
                    # Only tensor outputs are this diagnostic's scope; the
                    # native callable is neither serialized nor replaced.
                    output = output[:3]
                elif name.startswith((
                    "input_embedder.atom_encoder.",
                    "input_embedder.atom_attention_encoder.atom_encoder.",
                )):
                    output = {"input": args[0], "output": output}
            self.record(label, output)
            # Returning the captured tensor would replace the publisher output.
            return None

        return observed

    def install_modules(self, model):
        handles = []
        names = [
            "input_embedder",
            "s_init",
            "z_init_1",
            "z_init_2",
            "rel_pos",
            "token_bonds",
            "contact_conditioning",
            "diffusion_conditioning",
        ]
        if model.bond_type_feature:
            names.append("token_bonds_type")
        if self.input_details:
            names.extend(f"input_embedder.{name}" for name in (
                "atom_encoder", "atom_enc_proj_z", "atom_attention_encoder",
                "res_type_encoding", "msa_profile_encoding",
                "method_conditioning_init", "modified_conditioning_init",
                "cyclic_conditioning_init", "mol_type_conditioning_init",
                "atom_encoder.embed_atompair_ref_pos",
                "atom_encoder.embed_atompair_ref_dist",
                "atom_encoder.embed_atompair_mask",
                "atom_encoder.c_to_p_trans_q", "atom_encoder.c_to_p_trans_k",
                "atom_encoder.p_mlp",
            ))
            prefix = (
                "input_embedder.atom_attention_encoder.atom_encoder."
                "diffusion_transformer.layers.0."
            )
            names.extend(prefix + name for name in (
                "adaln.a_norm", "adaln.s_norm", "adaln.s_scale", "adaln.s_bias",
                "pair_bias_attn.proj_q", "pair_bias_attn.proj_k",
                "pair_bias_attn.proj_v", "pair_bias_attn.proj_g",
                "pair_bias_attn.proj_o",
            ))
        cyclic = [
            "s_norm",
            "z_norm",
            "s_recycle",
            "z_recycle",
            "msa_module",
            "pairformer_module",
        ]
        self.module_names = names + cyclic
        if self.input_details:
            cyclic.append("msa_module.layers.0")
            self.module_names.append("msa_module.layers.0")
            cyclic.extend(SINGLE_LEAVES)
            self.module_names.extend(SINGLE_LEAVES)
        try:
            for name in self.module_names:
                module = model
                for component in name.split("."):
                    module = getattr(module, component)
                if name in {"msa_module", "pairformer_module"}:
                    module = getattr(module, "_orig_mod", module)
                handles.append(
                    module.register_forward_hook(
                        self.module_hook(name, name in cyclic),
                        with_kwargs=True,
                    )
                )
        except BaseException:
            for handle in handles:
                handle.remove()
            raise
        return handles

    def install(self, model_class, writer_class, torch):
        original_forward = model_class.forward
        original_predict = model_class.predict_step
        original_writer = writer_class.write_on_batch_end
        writer_signature = inspect.signature(original_writer)

        def forward(model, *args, **kwargs):
            self.counts["forward"] += 1
            if self.counts["forward"] != 1:
                raise RuntimeError(
                    "capture supports one structure forward, no affinity arm"
                )
            settings = {
                name: getattr(model, name)
                for name in (
                    "predict_args",
                    "steering_args",
                    "use_kernels",
                    "use_templates",
                    "bond_type_feature",
                    "confidence_prediction",
                    "run_trunk_and_structure",
                    "skip_run_structure",
                    "is_pairformer_compiled",
                    "is_msa_compiled",
                )
            }
            settings.update(
                {
                    "forward_keyword_options": kwargs,
                    "forward_positional_count": len(args),
                    "float32_matmul_precision": torch.get_float32_matmul_precision(),
                    "cuda_autocast_enabled": torch.is_autocast_enabled("cuda"),
                    "cuda_autocast_dtype": str(torch.get_autocast_dtype("cuda")),
                    "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                    "training": model.training,
                }
            )
            save_new(self.out / "effective-model-settings.json", settings)
            handles = self.install_modules(model)
            try:
                result = original_forward(model, *args, **kwargs)
            finally:
                for handle in handles:
                    handle.remove()
            self.record("forward-output", result)
            return result

        def predict(model, *args, **kwargs):
            self.counts["predict_step"] += 1
            result = original_predict(model, *args, **kwargs)
            if result is None or result.get("exception") is not False:
                raise RuntimeError(
                    "publisher predict_step did not return a successful prediction"
                )
            self.record("predict-step-output", result)
            if host_array(result["coords"]).shape[0] != self.samples:
                raise RuntimeError("publisher returned an unexpected sample count")
            return result

        def writer(*args, **kwargs):
            self.counts["writer"] += 1
            prediction = writer_signature.bind(*args, **kwargs).arguments["prediction"]
            scores = prediction["confidence_score"]
            original_sort = torch.argsort

            def argsort(*a, **k):
                result = original_sort(*a, **k)
                if (a[0] if a else k.get("input")) is scores:
                    self.counts["writer_argsort"] += 1
                    if k.get("descending") is not True:
                        raise RuntimeError("writer confidence argsort policy changed")
                    save_new(
                        self.out / "sample-rank-map.json", writer_order(scores, result)
                    )
                return result

            with patch.object(torch, "argsort", argsort):
                return original_writer(*args, **kwargs)

        contexts = ExitStack()
        contexts.enter_context(patch.object(model_class, "forward", forward))
        contexts.enter_context(patch.object(model_class, "predict_step", predict))
        contexts.enter_context(patch.object(writer_class, "write_on_batch_end", writer))
        return contexts.close

    def validate(self):
        expected = {
            name: 1 for name in ("forward", "predict_step", "writer", "writer_argsort")
        }
        for name in self.module_names:
            expected[name] = (
                self.recycles + 1
                if name
                in {
                    "s_norm",
                    "z_norm",
                    "s_recycle",
                    "z_recycle",
                    "msa_module",
                    "pairformer_module",
                    "msa_module.layers.0",
                    *SINGLE_LEAVES,
                }
                else 1
            )
        if dict(self.counts) != expected:
            raise RuntimeError(
                f"incomplete native observation: {dict(self.counts)} != {expected}"
            )


def load_legacy(path):
    spec = importlib.util.spec_from_file_location("boltz_legacy_tape_capture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--num-recycles", type=int, default=3)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--precision", choices=("bf16-mixed", "32"), default="bf16-mixed"
    )
    parser.add_argument("--no-kernels", action="store_true")
    parser.add_argument("--input-details", action="store_true")
    args = parser.parse_args(argv)
    if args.num_samples < 1 or args.num_steps < 1 or args.num_recycles < 0:
        parser.error("sample/step counts must be positive and recycles nonnegative")
    source, upstream = (
        args.source_root.resolve(strict=True),
        args.upstream_root.resolve(strict=True),
    )
    if Path(__file__).resolve() != source / "bench/boltz_closure_capture.py":
        raise ValueError(
            "execute this observer from the explicitly selected source root"
        )
    if args.input.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("this wrapper requires explicit native Boltz YAML input")
    import yaml

    document = yaml.safe_load(args.input.read_text())
    if not isinstance(document, Mapping) or document.get("properties"):
        raise ValueError(
            "capture requires a structure-only Boltz job without affinity properties"
        )
    checkpoint = args.cache / "boltz2_conf.ckpt"
    if not checkpoint.is_file() or not (args.cache / "mols").is_dir():
        raise FileNotFoundError(
            "existing Boltz confidence checkpoint and CCD mols cache required; "
            "no downloads"
        )
    args.out_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(upstream / "src"))
    import torch
    from boltz.data.feature import featurizerv2
    from boltz.data.write.writer import BoltzWriter
    from boltz.model.models.boltz2 import Boltz2

    if not Path(inspect.getfile(Boltz2)).resolve().is_relative_to(upstream / "src"):
        raise RuntimeError("imported Boltz source differs from explicit upstream root")
    legacy_path = source / "tests/models/boltz2/scripts/capture_upstream_tape.py"
    legacy = load_legacy(legacy_path)
    observer = NativeObserver(
        args.out_dir,
        samples=args.num_samples,
        recycles=args.num_recycles,
        forward_code=Boltz2.forward.__code__,
        input_details=args.input_details,
    )
    original_install = legacy._install_hooks
    original_augment = featurizerv2.center_random_augmentation
    preprocessing = {}

    def augment(*a, **k):
        def record(original, *inner, **inner_kw):
            result = original(*inner, **inner_kw)
            preprocessing[f"draw_{len(preprocessing):06d}"] = result.detach().cpu()
            return result

        randn, randn_like = torch.randn, torch.randn_like
        with (
            patch.object(torch, "randn", lambda *a, **k: record(randn, *a, **k)),
            patch.object(
                torch, "randn_like", lambda *a, **k: record(randn_like, *a, **k)
            ),
        ):
            return original_augment(*a, **k)

    def install(recorder, captured):
        undo, boltz_main = original_install(recorder, captured)
        undo.append(observer.install(Boltz2, BoltzWriter, torch))
        # The stock downloader also fetches unused affinity weights and a tarball.
        # Required confidence/CCD assets were validated above; missing components
        # still fail in the native loader rather than triggering a download.
        downloader = patch.object(boltz_main, "download_boltz2", lambda cache: None)
        downloader.start()
        undo.append(downloader.stop)
        return undo, boltz_main

    native_args = argparse.Namespace(
        input=args.input.resolve(),
        out_dir=args.out_dir,
        repo=upstream,
        cache=args.cache.resolve(),
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        num_recycles=args.num_recycles,
        seed=args.seed,
        precision=args.precision,
        kernels=not args.no_kernels,
        subsample_msa=False,
        num_subsampled_msa=1024,
    )
    provenance = {
        "upstream": source_identity(upstream),
        "input_sha256": sha(args.input),
        "checkpoint_sha256": sha(checkpoint),
        "wrapper_sha256": sha(Path(__file__)),
        "legacy_capture_sha256": sha(legacy_path),
        "helper_source_sha256": {
            str(Path(inspect.getfile(function)).resolve().relative_to(source)): sha(
                Path(inspect.getfile(function))
            )
            for function in (sha, flatten_native)
        },
        "upstream_python_source": {
            str(path.relative_to(upstream)): sha(path)
            for path in sorted((upstream / "src/boltz").rglob("*.py"))
        },
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "requested_settings": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(native_args).items()
        },
        "scope": (
            "native output/tape diagnostic; no parity, writer equivalence "
            "or performance admission"
        ),
        "asset_scope": (
            "checkpoint and input hashed; CCD component identities require "
            "a separate input audit"
        ),
        "downloads": (
            "disabled; existing confidence checkpoint and CCD directory required"
        ),
    }
    save_new(args.out_dir / "provenance.json", provenance)
    started = time.monotonic()
    with (
        patch.object(legacy, "parse_args", lambda: native_args),
        patch.object(legacy, "_install_hooks", install),
        patch.object(featurizerv2, "center_random_augmentation", augment),
    ):
        result = legacy.main()
    if result != 0:
        raise RuntimeError(f"legacy native capture failed: {result}")
    observer.validate()
    observer.record("preprocessing-tape", preprocessing)
    upstream_after = source_identity(upstream)
    if any(
        upstream_after[name] != provenance["upstream"][name]
        for name in ("commit", "tracked_diff_sha256")
    ):
        raise RuntimeError("publisher source changed during the native observation")
    save_new(
        args.out_dir / "capture-complete.json",
        {
            "passed": True,
            "not_parity_admission": True,
            "counts": dict(observer.counts),
            "artifacts": observer.artifacts,
            "selected_recycles": sorted({0, args.num_recycles}),
            "instrumented_seconds": time.monotonic() - started,
            "upstream_after": upstream_after,
        },
    )


if __name__ == "__main__":
    main()
