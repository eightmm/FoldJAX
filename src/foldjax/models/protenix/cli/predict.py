"""Standalone Protenix JAX prediction from native weights."""

from __future__ import annotations

import argparse
import json
import os
import shlex
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    feature_group = parser.add_mutually_exclusive_group(required=True)
    feature_group.add_argument("--features", type=Path)
    feature_group.add_argument("--input-json", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seed", type=int, help="Single seed (legacy spelling; default: 101)."
    )
    seed_group.add_argument(
        "--seeds", type=int, nargs="+", help="Run every job with each listed seed."
    )
    parser.add_argument(
        "--output-format",
        choices=("npz", "protenix", "both"),
        default="npz",
        help="Write legacy NPZ, ranked CIF/JSON output, or both.",
    )
    parser.add_argument("--n-sample", type=int, default=5)
    parser.add_argument("--n-step", type=int)
    parser.add_argument("--s-max", type=float, default=160.0)
    parser.add_argument("--s-min", type=float, default=4e-4)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--sigma-data", type=float, default=16.0)
    parser.add_argument("--n-cycle", type=int)
    parser.add_argument("--gamma0", type=float)
    parser.add_argument(
        "--eta",
        "--step-scale-eta",
        dest="eta",
        type=float,
        help="Diffusion step scale eta (model-specific default).",
    )
    parser.add_argument("--n-queries", type=int, default=32)
    parser.add_argument("--n-keys", type=int, default=128)
    parser.add_argument("--max-msa-rows", type=int, default=16384)
    parser.add_argument(
        "--msa-search",
        choices=("off", "local", "remote"),
        default="off",
        help="Optional MSA preprocessing backend (disabled by default).",
    )
    parser.add_argument("--msa-cache-dir", type=Path, default=Path("outputs/msa_cache"))
    parser.add_argument("--msa-search-version")
    parser.add_argument(
        "--msa-local-command",
        help=(
            "Shell-style local MSA wrapper command (arguments are not shell-executed)."
        ),
    )
    parser.add_argument("--msa-remote-url")
    parser.add_argument(
        "--rna-msa-local-command",
        help="Local nhmmer wrapper command producing rna_msa.a3m (off by default).",
    )
    parser.add_argument("--rna-msa-search-version")
    parser.add_argument(
        "--rna-msa-cache-dir",
        type=Path,
        default=Path("outputs/rna_msa_cache"),
    )
    parser.add_argument(
        "--template-search-command",
        help="Local template wrapper producing one .a3m or .hhr (off by default).",
    )
    parser.add_argument("--template-search-version")
    parser.add_argument(
        "--template-search-cache-dir",
        type=Path,
        default=Path("outputs/template_cache"),
    )
    parser.add_argument(
        "--template-mmcif-dir",
        type=Path,
        help="Existing local mmCIF coordinate database; never downloaded implicitly.",
    )
    msa_group = parser.add_mutually_exclusive_group()
    msa_group.add_argument(
        "--full-depth-msa",
        dest="full_depth_msa",
        action="store_true",
        help="Use the faster single-shape full MSA path.",
    )
    msa_group.add_argument(
        "--sample-msa-per-cycle",
        dest="full_depth_msa",
        action="store_false",
        help="Use upstream-style random MSA depths (slower on XLA).",
    )
    parser.set_defaults(full_depth_msa=True)
    parser.add_argument(
        "--msa-row-alignment",
        type=int,
        default=64,
        help="Align a nearly full MSA row count; set 0 to disable.",
    )
    parser.add_argument("--max-msa-padding-rows", type=int, default=8)
    parser.add_argument("--input-atom-heads", type=int, default=4)
    parser.add_argument("--atom-encoder-heads", type=int, default=4)
    parser.add_argument("--token-heads", type=int, default=16)
    parser.add_argument("--atom-decoder-heads", type=int, default=4)
    parser.add_argument("--triangle-mul-chunk-size", type=int)
    parser.add_argument("--triangle-att-q-chunk-size", type=int)
    parser.add_argument("--single-att-q-chunk-size", type=int)
    parser.add_argument("--token-q-chunk-size", type=int)
    parser.add_argument("--diffusion-chunk-size", type=int)
    parser.add_argument(
        "--trunk-dtype",
        choices=("bf16", "fp32"),
        default="bf16",
        help="Use upstream-style BF16 trunk with FP32 diffusion/confidence islands.",
    )
    parser.add_argument(
        "--chunk-policy",
        choices=("auto", "manual", "off"),
        default="auto",
        help="Resolve chunk knobs automatically, manually, or disable chunking.",
    )
    pairformer_scan_group = parser.add_mutually_exclusive_group()
    pairformer_scan_group.add_argument(
        "--pairformer-scan", dest="use_pairformer_scan", action="store_true"
    )
    pairformer_scan_group.add_argument(
        "--no-pairformer-scan", dest="use_pairformer_scan", action="store_false"
    )
    parser.set_defaults(use_pairformer_scan=False)
    parser.add_argument("--diffusion-scan", action="store_true")
    sampler_scan_group = parser.add_mutually_exclusive_group()
    sampler_scan_group.add_argument(
        "--sampler-scan", dest="sampler_scan", action="store_true"
    )
    sampler_scan_group.add_argument(
        "--no-sampler-scan", dest="sampler_scan", action="store_false"
    )
    parser.set_defaults(sampler_scan=True)
    parser.add_argument("--denoiser-jit", action="store_true")
    parser.add_argument(
        "--diffusion-attention-backend",
        choices=("xla", "xla_jit", "xla_sdpa"),
        default="xla_jit",
    )
    parser.add_argument(
        "--trunk-single-attention-backend",
        choices=("xla", "xla_jit", "xla_sdpa"),
        default="xla_jit",
    )
    parser.add_argument(
        "--trunk-triangle-attention-backend",
        choices=("xla", "xla_jit", "tokamax", "cueq", "cueq_jit"),
        default="cueq_jit",
    )
    confidence_scan_group = parser.add_mutually_exclusive_group()
    confidence_scan_group.add_argument(
        "--confidence-scan", dest="confidence_scan", action="store_true"
    )
    confidence_scan_group.add_argument(
        "--no-confidence-scan", dest="confidence_scan", action="store_false"
    )
    parser.set_defaults(confidence_scan=False)
    parser.add_argument("--no-confidence", action="store_true")
    parser.add_argument("--no-confidence-scores", action="store_true")
    parser.add_argument(
        "--no-graph-jit",
        action="store_true",
        help="trace the model op by op instead of as one compiled graph; "
        "much slower, kept for debugging and numerical comparison",
    )
    parser.add_argument("--include-trunk", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument(
        "--compile-cache",
        type=Path,
        default=Path("outputs/compile_cache"),
        help="Persistent XLA compilation cache dir (compile-time only, "
        "output-invariant).",
    )
    parser.add_argument(
        "--no-compile-cache",
        action="store_true",
        help="Disable the persistent compilation cache.",
    )
    parser.add_argument(
        "--prewarm-only",
        action="store_true",
        help="Compile and populate --compile-cache for each input shape, then "
        "exit without writing prediction files.",
    )
    parser.add_argument(
        "--model-name",
        default="auto",
        help="Known Protenix model name for runtime limits, or 'auto'.",
    )
    parser.add_argument(
        "--esm-checkpoint-dir",
        type=Path,
        help="Directory containing optional ESM/ISM preprocessing checkpoints.",
    )
    parser.add_argument(
        "--guidance-config",
        type=Path,
        help="JSON file containing the original Protenix TFG guidance mapping.",
    )
    args = parser.parse_args(argv)

    guidance_config = None
    if args.guidance_config is not None:
        try:
            guidance_config = json.loads(
                args.guidance_config.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid guidance config: {exc}") from exc
        if not isinstance(guidance_config, dict):
            raise SystemExit("guidance config must be a JSON object")
        if guidance_config.get("enable") and args.sampler_scan:
            args.sampler_scan = False
            print("TFG enabled: using the non-scan diffusion sampler")

    if args.cpu_only:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import jax
    import jax.numpy as jnp

    if not args.no_compile_cache:
        cache = args.compile_cache.expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(cache))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
        print(f"compile cache: {cache}")

    from foldjax.models.protenix.bridge.weights_io import load_native_weights
    from foldjax.models.protenix.chunking import resolve_chunk_config
    from foldjax.models.protenix.data.featurize_json import featurize_protein_json
    from foldjax.models.protenix.data.output import (
        sanitize_job_name,
        write_protenix_outputs,
    )
    from foldjax.models.protenix.data.static_io import (
        load_static_feature_npz,
        save_output_npz,
    )
    from foldjax.models.protenix.models.model import cast_trunk_params
    from foldjax.models.protenix.models.predict import protenix_predict_static
    from foldjax.models.protenix.models.trunk_blocks.msa import (
        pad_msa_features_to_bucket,
        sample_msa_cycle_features,
    )
    from foldjax.models.protenix.runtime_policy import (
        infer_model_name_from_path,
        model_inference_defaults,
        validate_inference_limits,
    )

    model_name = args.model_name
    if model_name == "auto":
        model_name = infer_model_name_from_path(args.weights)
    if model_name is None:
        sampler_defaults = {
            "n_cycle": 10,
            "n_step": 200,
            "gamma0": 0.8,
            "step_scale_eta": 1.5,
        }
    else:
        try:
            sampler_defaults = model_inference_defaults(model_name)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    n_cycle = args.n_cycle if args.n_cycle is not None else sampler_defaults["n_cycle"]
    n_step = args.n_step if args.n_step is not None else sampler_defaults["n_step"]
    gamma0 = args.gamma0 if args.gamma0 is not None else sampler_defaults["gamma0"]
    eta = args.eta if args.eta is not None else sampler_defaults["step_scale_eta"]

    msa_pipeline = None
    if args.msa_search != "off":
        if args.features is not None:
            raise SystemExit("MSA search requires --input-json")
        if not args.msa_search_version:
            raise SystemExit(
                "--msa-search-version is required when MSA search is enabled"
            )
        from foldjax.models.protenix.data.search import (
            LocalMsaClient,
            MsaSearchPipeline,
            RemoteMMseqs2Client,
        )

        if args.msa_search == "local":
            if not args.msa_local_command:
                raise SystemExit("--msa-local-command is required for local MSA search")
            command = shlex.split(args.msa_local_command)
            if not command:
                raise SystemExit("--msa-local-command must not be empty")
            msa_backend = LocalMsaClient(command, version=args.msa_search_version)
        else:
            if not args.msa_remote_url:
                raise SystemExit("--msa-remote-url is required for remote MSA search")
            msa_backend = RemoteMMseqs2Client(
                args.msa_remote_url, version=args.msa_search_version
            )
        msa_pipeline = MsaSearchPipeline(
            args.msa_cache_dir,
            msa_backend,
            options={"mode": args.msa_search},
        )

    rna_msa_pipeline = None
    if args.rna_msa_local_command is not None:
        if args.features is not None:
            raise SystemExit("RNA MSA search requires --input-json")
        if not args.rna_msa_search_version:
            raise SystemExit(
                "--rna-msa-search-version is required for RNA MSA search"
            )
        command = shlex.split(args.rna_msa_local_command)
        if not command:
            raise SystemExit("--rna-msa-local-command must not be empty")
        from foldjax.models.protenix.data.search import (
            LocalRnaMsaClient,
            RnaMsaSearchPipeline,
        )

        rna_msa_pipeline = RnaMsaSearchPipeline(
            args.rna_msa_cache_dir,
            LocalRnaMsaClient(command, version=args.rna_msa_search_version),
            options={"mode": "local-nhmmer"},
        )
    elif args.rna_msa_search_version:
        raise SystemExit(
            "--rna-msa-local-command is required with --rna-msa-search-version"
        )

    template_pipeline = None
    mmcif_dir = None
    if args.template_mmcif_dir is not None:
        mmcif_dir = args.template_mmcif_dir.expanduser().resolve()
        if not mmcif_dir.is_dir():
            raise SystemExit(f"template mmCIF directory does not exist: {mmcif_dir}")
        # Also enables coordinate resolution for an existing .a3m/.hhr
        # templatesPath when no automatic search command is requested.
        os.environ["PROTENIX_TEMPLATE_MMCIF_DIR"] = str(mmcif_dir)
    if args.template_search_command is not None:
        if args.features is not None:
            raise SystemExit("template search requires --input-json")
        if not args.template_search_version:
            raise SystemExit(
                "--template-search-version is required for template search"
            )
        if mmcif_dir is None:
            raise SystemExit("--template-mmcif-dir is required for template search")
        command = shlex.split(args.template_search_command)
        if not command:
            raise SystemExit("--template-search-command must not be empty")
        from foldjax.models.protenix.data.search import (
            LocalTemplateSearchClient,
            TemplateSearchPipeline,
        )

        template_pipeline = TemplateSearchPipeline(
            args.template_search_cache_dir,
            LocalTemplateSearchClient(
                command, version=args.template_search_version
            ),
            options={"mmcif_dir": str(mmcif_dir)},
        )
    elif args.template_search_version:
        raise SystemExit(
            "--template-search-command is required with template search options"
        )

    esm_provider = None
    if model_name is not None and ("_esm_" in model_name or "_ism_" in model_name):
        from foldjax.models.protenix.data.esm import FairEsmProvider

        esm_name = "esm2-3b-ism" if "_ism_" in model_name else "esm2-3b"
        esm_provider = FairEsmProvider(
            esm_name,
            checkpoint_dir=args.esm_checkpoint_dir or args.weights.parent,
        )

    try:
        if args.features is not None:
            jobs = [
                {
                    "name": args.features.stem,
                    "features": load_static_feature_npz(args.features),
                    "modelSeeds": None,
                }
            ]
        else:
            with args.input_json.open("r", encoding="utf-8") as handle:
                json_jobs = json.load(handle)
            if not isinstance(json_jobs, list) or not json_jobs:
                raise ValueError("input JSON must be a non-empty top-level list")
            jobs = []
            for index, job in enumerate(json_jobs):
                if not isinstance(job, dict):
                    raise ValueError(f"input JSON entry {index} must be an object")
                fallback_name = (
                    args.input_json.stem
                    if len(json_jobs) == 1
                    else f"{args.input_json.stem}_{index}"
                )
                if msa_pipeline is not None:
                    from foldjax.models.protenix.data.search import apply_msa_paths

                    job = apply_msa_paths(job, msa_pipeline)
                if template_pipeline is not None:
                    from foldjax.models.protenix.data.search import apply_template_paths

                    job = apply_template_paths(job, template_pipeline)
                if rna_msa_pipeline is not None:
                    from foldjax.models.protenix.data.search import apply_rna_msa_paths

                    job = apply_rna_msa_paths(job, rna_msa_pipeline)
                features = featurize_protein_json(
                    job,
                    base_dir=args.input_json.parent,
                    n_queries=args.n_queries,
                    n_keys=args.n_keys,
                    max_msa_rows=args.max_msa_rows,
                )
                if esm_provider is not None:
                    from foldjax.models.protenix.data.esm import add_esm_embeddings

                    esm_features = dict(features)
                    esm_features["residue_index"] = esm_features["residue_index"] - 1
                    features = add_esm_embeddings(
                        esm_features, job, provider=esm_provider
                    )
                    features["residue_index"] = esm_features["residue_index"] + 1
                jobs.append(
                    {
                        "name": str(job.get("name") or fallback_name),
                        "features": features,
                        "modelSeeds": job.get("modelSeeds"),
                    }
                )

        for job in jobs:
            features = job["features"]
            if (
                esm_provider is not None
                and args.features is not None
                and "esm_token_embedding" not in features
            ):
                raise ValueError(
                    "ESM/ISM model static features require esm_token_embedding"
                )
            n_token = int(features["restype"].shape[-2])
            validate_inference_limits(model_name=model_name, n_token=n_token)
            if args.full_depth_msa and args.msa_row_alignment > 0 and "msa" in features:
                original_msa_rows = int(features["msa"].shape[-2])
                features = pad_msa_features_to_bucket(
                    features,
                    bucket_size=args.msa_row_alignment,
                    max_padding_rows=args.max_msa_padding_rows,
                )
                job["features"] = features
                aligned_msa_rows = int(features["msa"].shape[-2])
                if aligned_msa_rows != original_msa_rows:
                    print(
                        f"{job['name']}: MSA rows aligned: "
                        f"{original_msa_rows} -> {aligned_msa_rows}"
                    )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.output_format != "npz" and (
        args.no_confidence or args.no_confidence_scores
    ):
        raise SystemExit("protenix output format requires confidence scores")

    params = load_native_weights(args.weights)
    trunk_dtype = None
    if args.trunk_dtype == "bf16":
        trunk_dtype = jnp.bfloat16
        params = cast_trunk_params(params, trunk_dtype)
    job_seeds = [_resolve_seeds(args, job.get("modelSeeds")) for job in jobs]
    legacy_npz = (
        args.output_format == "npz" and len(jobs) == 1 and len(job_seeds[0]) == 1
    )
    for job, seeds in zip(jobs, job_seeds, strict=True):
        features = job["features"]
        guidance_features = None
        if guidance_config is not None and guidance_config.get("enable"):
            from foldjax.models.protenix.data.geometry import (
                prepare_tfg_features,
                require_supported_geometry,
            )

            guidance_features = prepare_tfg_features(features)
            require_supported_geometry(guidance_features)
        n_token = int(features["restype"].shape[-2])
        chunk_config = resolve_chunk_config(
            n_token=n_token,
            n_sample=args.n_sample,
            policy=args.chunk_policy,
            triangle_mul_chunk_size=args.triangle_mul_chunk_size,
            triangle_att_q_chunk_size=args.triangle_att_q_chunk_size,
            single_att_q_chunk_size=args.single_att_q_chunk_size,
            token_q_chunk_size=args.token_q_chunk_size,
            diffusion_chunk_size=args.diffusion_chunk_size,
        )
        for seed in seeds:
            cycle_msa_features = None
            if not args.full_depth_msa:
                sampled = sample_msa_cycle_features(
                    features,
                    n_cycle=n_cycle,
                    seed=seed,
                )
                cycle_msa_features = sampled or None
            output = protenix_predict_static(
                params,
                features,
                key=jax.random.PRNGKey(seed),
                n_sample=args.n_sample,
                num_sampling_steps=n_step,
                s_max=args.s_max,
                s_min=args.s_min,
                rho=args.rho,
                sigma_data=args.sigma_data,
                recycling_steps=n_cycle,
                input_atom_heads=args.input_atom_heads,
                atom_encoder_heads=args.atom_encoder_heads,
                token_heads=args.token_heads,
                atom_decoder_heads=args.atom_decoder_heads,
                n_queries=args.n_queries,
                n_keys=args.n_keys,
                use_pairformer_scan=args.use_pairformer_scan,
                use_confidence_scan=args.confidence_scan,
                use_diffusion_scan=args.diffusion_scan,
                use_sampler_scan=args.sampler_scan,
                use_denoiser_jit=args.denoiser_jit,
                diffusion_attention_backend=args.diffusion_attention_backend,
                trunk_single_attention_backend=args.trunk_single_attention_backend,
                trunk_triangle_attention_backend=args.trunk_triangle_attention_backend,
                run_confidence=not args.no_confidence,
                run_confidence_scores=not args.no_confidence_scores,
                triangle_mul_chunk_size=chunk_config.triangle_mul_chunk_size,
                triangle_att_q_chunk_size=chunk_config.triangle_att_q_chunk_size,
                single_att_q_chunk_size=chunk_config.single_att_q_chunk_size,
                token_q_chunk_size=chunk_config.token_q_chunk_size,
                diffusion_chunk_size=chunk_config.diffusion_chunk_size,
                trunk_dtype=trunk_dtype,
                cycle_msa_features=cycle_msa_features,
                gamma0=gamma0,
                step_scale_eta=eta,
                guidance_config=guidance_config,
                guidance_features=guidance_features,
                graph_jit=not args.no_graph_jit,
            )
            if args.prewarm_only:
                jax.block_until_ready(output)
                print(
                    "prewarmed: "
                    f"job={job['name']} tokens={n_token} "
                    f"atoms={features['atom_to_token_idx'].shape[0]} "
                    f"samples={args.n_sample}"
                )
                break
            if args.output_format in ("protenix", "both"):
                paths = write_protenix_outputs(
                    args.out,
                    job_name=job["name"],
                    seed=seed,
                    output=output,
                    features=features,
                    include_raw=args.output_format == "both",
                    include_trunk=args.include_trunk,
                )
                print(f"wrote: {paths[0].parent}")
            else:
                if legacy_npz:
                    output_path = args.out
                else:
                    output_path = (
                        args.out
                        / sanitize_job_name(job["name"])
                        / f"seed_{seed}"
                        / "predictions"
                        / "raw_output.npz"
                    )
                print("  output arrays:")
                omitted = save_output_npz(
                    output_path,
                    output,
                    include_trunk=args.include_trunk,
                    report=print,
                )
                for name, nbytes in omitted:
                    print(
                        f"  left out {name} ({nbytes / 2**30:.1f} GiB): confidence "
                        "outputs are quadratic in token count, and at this size the "
                        "copy to host costs more than the prediction did"
                    )
                print(f"wrote: {output_path}")


def _resolve_seeds(args: argparse.Namespace, model_seeds: Any) -> list[int]:
    if args.seeds is not None:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    elif model_seeds is not None:
        if not isinstance(model_seeds, list) or not model_seeds:
            raise SystemExit("modelSeeds must be a non-empty list")
        seeds = [int(seed) for seed in model_seeds]
    else:
        seeds = [101]
    if len(set(seeds)) != len(seeds):
        raise SystemExit("seeds must be unique")
    return seeds


if __name__ == "__main__":
    main()
