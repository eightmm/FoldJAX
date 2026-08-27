"""Standalone Protenix JAX prediction from native weights."""

from __future__ import annotations

import argparse
import json
import os
import shlex
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from foldjax.models import _representations


def _collect_representations(output, wanted):
    """Filter the model output to the requested names.

    The model records them under the shared names already -- the tap is the
    name -- so nothing has to be translated here.
    """
    return {name: output[name] for name in wanted if name in output}


def main(
    argv: Sequence[str] | None = None,
    *,
    on_padding_plan: Callable[..., None] | None = None,
) -> list[Path]:
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
    parser.add_argument(
        "--num-samples",
        "--n-sample",
        dest="num_samples",
        type=int,
        default=5,
    )
    parser.add_argument("--num-steps", "--n-step", dest="num_steps", type=int)
    parser.add_argument("--s-max", type=float, default=160.0)
    parser.add_argument("--s-min", type=float, default=4e-4)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--sigma-data", type=float, default=16.0)
    parser.add_argument("--num-recycles", "--n-cycle", dest="num_recycles", type=int)
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
    parser.add_argument(
        "--max-msa-depth",
        "--max-msa-rows",
        dest="max_msa_depth",
        type=int,
        default=16384,
    )
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
    parser.add_argument(
        "--strict-token-limit",
        dest="strict_token_limit",
        action="store_true",
        help="refuse protenix-v2 above 2,560 tokens the way upstream does. "
        "That limit is a memory budget -- its own message says 'It might cause "
        "OOM' -- and nothing in the architecture is bounded by token count, so "
        "FoldJAX warns and runs instead. Use this to get upstream's behaviour",
    )
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
    parser.add_argument("--opm-chunk-size", type=int)
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
        # Left unset so the backend is resolved in one place, by
        # `_triangle_attention_backend()`, which honours
        # PROTENIX_TRIANGLE_BACKEND and otherwise picks the blocked XLA path.
        # This default used to be spelled "cueq_jit" here and in two module
        # signatures, so the documented and tested default was never the
        # effective one: every real caller passed cueq_jit, which does not
        # block rows and so does not bound the score tensor.
        default=None,
    )
    parser.add_argument(
        "--confidence-triangle-attention-backend",
        choices=("xla", "xla_jit", "tokamax", "cueq", "cueq_jit"),
        # Unset means "whatever the trunk runs". The head reads the same pair
        # tensor with the same head layout, so a different kernel there is a
        # bug rather than a choice -- it was one, and it cost a 39 GiB temp
        # arena at 2030 tokens. Overriding it separately is still allowed,
        # because the head and the trunk do not have to fit at the same moment.
        default=None,
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
    parser.add_argument(
        "--cp-devices",
        type=int,
        default=1,
        help="shard the pair representations across this many JAX devices "
        "(context parallelism, the JAX form of OpenDDE's Fold-CP); needs "
        "that many visible devices",
    )
    parser.add_argument(
        "--cp-layout",
        choices=("auto", "1d", "2d"),
        default="auto",
        help="how the pair axes are split: '2d' is Fold-CP's square grid "
        "(needs a square device count), '1d' splits rows only, 'auto' picks "
        "2d only when requested explicitly; 'auto' currently selects 1d.",
    )
    parser.add_argument("--include-trunk", action="store_true")
    parser.add_argument(
        "--representations-dir",
        type=Path,
        default=None,
        help=(
            "where to write the representation archive; defaults beside "
            "the run's own output. The common API pins this so that every "
            "model puts it in the same place."
        ),
    )
    parser.add_argument(
        "--stop-after",
        choices=("full", "trunk"),
        default="full",
        help=(
            "'trunk' stops once the representations exist, skipping the "
            "diffusion sampler and the confidence heads."
        ),
    )
    parser.add_argument(
        "--representations",
        default=None,
        help=(
            "comma-separated representation names to write beside the "
            "structures, or 'all'. These are the largest arrays a run "
            "produces -- the pair representation is quadratic in token "
            "count -- so nothing is written unless asked for. Names: "
            "single_inputs, single, pair."
        ),
    )
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
        help="Known Protenix model name, 'auto' to infer it from the weight "
        "filename, or 'unknown' to accept the base-model defaults for a "
        "checkpoint this build does not recognise.",
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
    parser.add_argument(
        "--padding",
        action="store_true",
        help="Pad generated feature axes to a reusable, fully masked shape profile.",
    )
    parser.add_argument("--pad-tokens", type=int)
    parser.add_argument("--pad-atoms", type=int)
    parser.add_argument("--pad-msa", type=int)
    parser.add_argument("--pad-templates", type=int)
    parser.add_argument("--pad-language-model-tokens", type=int)
    parser.add_argument(
        "--padding-overflow",
        choices=("error", "exact"),
        default="error",
    )
    args = parser.parse_args(argv)

    padding_requested = args.padding or any(
        value is not None
        for value in (
            args.pad_tokens,
            args.pad_atoms,
            args.pad_msa,
            args.pad_templates,
            args.pad_language_model_tokens,
        )
    )
    padding_config = None
    if padding_requested:
        from foldjax.schema import PaddingConfig

        padding_config = PaddingConfig(
            tokens=args.pad_tokens,
            atoms=args.pad_atoms,
            msa=args.pad_msa,
            templates=args.pad_templates,
            language_model_tokens=args.pad_language_model_tokens,
            overflow=args.padding_overflow,
        )
        if args.features is not None:
            raise SystemExit(
                "padding currently supports generated --input-json features only; "
                "static NPZ schemas may contain unregistered semantic axes"
            )
        if args.guidance_config is not None:
            raise SystemExit(
                "padding with TFG guidance is not yet supported; use one or the other"
            )
        if not args.full_depth_msa:
            raise SystemExit(
                "padding currently requires --full-depth-msa so random cycle "
                "sampling cannot select padded rows"
            )

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
    from foldjax.models.protenix.data.template_features import (
        dedup_templates,
    )
    from foldjax.models.protenix.models.model import cast_trunk_params
    from foldjax.models.protenix.models.predict import protenix_predict_static
    from foldjax.models.protenix.models.trunk_blocks.msa import (
        pad_msa_features_to_bucket,
        sample_msa_cycle_features,
    )
    from foldjax.models.protenix.runtime_policy import (
        KNOWN_MODEL_NAMES,
        infer_model_name_from_path,
        model_inference_defaults,
        validate_inference_limits,
    )

    model_name = args.model_name
    if model_name == "auto":
        model_name = infer_model_name_from_path(args.weights)
        if model_name is None:
            # The model name is not cosmetic. It selects the sampler schedule
            # (the mini models run 5 steps at gamma0=0, the base models 200 at
            # 0.8), it decides whether ESM/ISM conditioning is built at all,
            # and it carries the protenix-v2 token limit. Inferring it from the
            # filename means renaming a checkpoint silently changed all three:
            # an ISM model run from a renamed file quietly became a different
            # model, with no warning and a plausible-looking structure out.
            raise SystemExit(
                f"cannot tell which Protenix model {args.weights.name!r} is, "
                "and the name selects the sampler schedule, the ESM/ISM "
                "conditioning, and the token limit. Pass --model-name with one "
                f"of: {', '.join(KNOWN_MODEL_NAMES)}; or --model-name unknown "
                "to accept the base-model schedule with no ESM conditioning."
            )
    if model_name == "unknown":
        model_name = None
    if model_name is None:
        sampler_defaults = {
            "num_recycles": 10,
            "num_steps": 200,
            "gamma0": 0.8,
            "step_scale_eta": 1.5,
        }
    else:
        try:
            sampler_defaults = model_inference_defaults(model_name)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    num_recycles = (
        args.num_recycles
        if args.num_recycles is not None
        else sampler_defaults["num_recycles"]
    )
    num_steps = (
        args.num_steps
        if args.num_steps is not None
        else sampler_defaults["num_steps"]
    )
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

    esm_name = None
    esm_provider = None
    if model_name is not None and ("_esm_" in model_name or "_ism_" in model_name):
        esm_name = "esm2-3b-ism" if "_ism_" in model_name else "esm2-3b"
        # A static feature NPZ may already carry the publisher-derived
        # ``esm_token_embedding``.  In that case no language-model checkpoint
        # is opened (or even required); the feature is validated below.
        if args.features is None:
            from foldjax.models.protenix.data.esm import JaxEsmProvider

            esm_provider = JaxEsmProvider(
                esm_name,
                checkpoint_dir=args.esm_checkpoint_dir or args.weights.parent,
            )
    elif padding_config is not None and args.pad_language_model_tokens is not None:
        raise SystemExit(
            "--pad-language-model-tokens applies only to Protenix ESM/ISM variants"
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
                    max_msa_depth=args.max_msa_depth,
                )
                language_model_profile = None
                if esm_provider is not None and padding_config is not None:
                    from foldjax.padding import (
                        LANGUAGE_MODEL_TOKEN_BUCKETS,
                        resolve_axis,
                    )

                    protein_lengths = []
                    for wrapper in job.get("sequences", []):
                        if not isinstance(wrapper, dict):
                            continue
                        protein = wrapper.get("proteinChain")
                        if isinstance(protein, dict):
                            protein_lengths.append(
                                len(str(protein.get("sequence", "")))
                            )
                    if not protein_lengths or max(protein_lengths) < 1:
                        raise ValueError(
                            "ESM/ISM padding requires at least one protein sequence"
                        )
                    language_model_actual = max(protein_lengths)
                    language_model_target = resolve_axis(
                        language_model_actual,
                        padding_config,
                        "language_model_tokens",
                        buckets=tuple(
                            target
                            for target in LANGUAGE_MODEL_TOKEN_BUCKETS
                            if target <= esm_provider.max_sequence_length
                        )
                        + (esm_provider.max_sequence_length,),
                    )
                    language_model_profile = (
                        language_model_actual,
                        language_model_target,
                    )
                if esm_provider is not None:
                    from foldjax.models.protenix.data.esm import add_esm_embeddings

                    esm_features = dict(features)
                    esm_features["residue_index"] = esm_features["residue_index"] - 1
                    provider = esm_provider
                    if language_model_profile is not None:
                        _, language_model_target = language_model_profile

                        def provider(sequence: str) -> Any:
                            return esm_provider.embed(
                                sequence, target_length=language_model_target
                            )

                    features = add_esm_embeddings(
                        esm_features, job, provider=provider
                    )
                    features["residue_index"] = esm_features["residue_index"] + 1
                jobs.append(
                    {
                        "name": str(job.get("name") or fallback_name),
                        "features": features,
                        "modelSeeds": job.get("modelSeeds"),
                        "language_model_profile": language_model_profile,
                    }
                )

        for job in jobs:
            features = job["features"]
            if esm_name is not None:
                from foldjax.models.protenix.data.esm import validate_esm_embeddings

                validate_esm_embeddings(features)
            n_token = int(features["restype"].shape[-2])
            validate_inference_limits(
                model_name=model_name,
                n_token=n_token,
                strict_token_limit=args.strict_token_limit,
            )
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
            job["output_features"] = features
            if padding_config is not None:
                from foldjax.models.protenix.data.padding import (
                    pad_protenix_features,
                )

                features, padding_plan = pad_protenix_features(
                    features,
                    padding_config,
                    n_queries=args.n_queries,
                    n_keys=args.n_keys,
                )
                language_model_profile = job.get("language_model_profile")
                if language_model_profile is not None:
                    from foldjax.padding import PaddingPlan

                    language_model_actual, language_model_target = (
                        language_model_profile
                    )
                    padding_plan = PaddingPlan(
                        actual={
                            **padding_plan.actual,
                            "language_model_tokens": language_model_actual,
                        },
                        storage={
                            **(padding_plan.storage or padding_plan.actual),
                            "language_model_tokens": language_model_actual,
                        },
                        target={
                            **padding_plan.target,
                            "language_model_tokens": language_model_target,
                        },
                    )
                job["features"] = features
                job["padding_plan"] = padding_plan
                validate_inference_limits(
                    model_name=model_name,
                    n_token=padding_plan.target["tokens"],
                    strict_token_limit=args.strict_token_limit,
                )
                print(f"{job['name']}: {padding_plan.message('protenix')}")
                if on_padding_plan is not None:
                    valid_tokens = jnp.asarray(features["token_padding_mask"]).astype(
                        bool
                    )
                    valid_asym = jnp.asarray(features["asym_id"])[valid_tokens]
                    on_padding_plan(
                        padding_plan,
                        {"chains": int(jnp.max(valid_asym)) + 1},
                    )
            # A query with fewer than four template hits is padded up to four,
            # and the embedder runs the whole pairformer stack once per row.
            # The padded rows are identical to each other, so deduplicating
            # here buys their stack evaluations back; the survivors carry a
            # multiplicity so the average is unchanged. After padding, because
            # `pad_protenix_features` requires the native depth of four.
            # `output_features` was snapshotted above and keeps its rows.
            job["features"] = dedup_templates(job["features"])
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
    # Returned so a caller knows which files *this* run produced. FoldJAX used
    # to recover them by globbing the output tree, which cannot tell a
    # structure written now from one left by an earlier run into the same
    # directory.
    wanted_representations = _representations.resolve(
        args.representations, _representations.specs_for("protenix")
    )
    written: list[Path] = []
    # Only the raw-npz path reads the trunk representations or the full-bin
    # logits; the protenix cif+JSON path consumes the in-graph summaries alone.
    # Keeping unread [num_samples, N, N, 64] logits as program outputs held 21.6
    # GiB at 3,012 tokens (EXPERIMENT_LOG: the 84.1 vs 57.3 attribution).
    wants_raw = args.output_format in ("npz", "both")
    for job, seeds in zip(jobs, job_seeds, strict=True):
        features = job["features"]
        output_features = job.get("output_features", features)
        padding_plan = job.get("padding_plan")
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
            num_samples=args.num_samples,
            policy=args.chunk_policy,
            triangle_mul_chunk_size=args.triangle_mul_chunk_size,
            triangle_att_q_chunk_size=args.triangle_att_q_chunk_size,
            single_att_q_chunk_size=args.single_att_q_chunk_size,
            token_q_chunk_size=args.token_q_chunk_size,
            opm_chunk_size=args.opm_chunk_size,
            diffusion_chunk_size=args.diffusion_chunk_size,
        )
        for seed in seeds:
            init_noise = None
            step_noises = None
            preserve_prefix_rng = (
                padding_plan is not None and _prefix_rng_is_supported()
            )
            if padding_plan is not None and not preserve_prefix_rng:
                init_noise, step_noises = _padded_noise_tapes(
                    seed=seed,
                    num_samples=args.num_samples,
                    num_steps=num_steps,
                    actual_atom=padding_plan.actual["atoms"],
                    target_atom=padding_plan.target["atoms"],
                    diffusion_chunk_size=chunk_config.diffusion_chunk_size,
                )
            cycle_msa_features = None
            if not args.full_depth_msa:
                sampled = sample_msa_cycle_features(
                    features,
                    num_recycles=num_recycles,
                    seed=seed,
                )
                cycle_msa_features = sampled or None
            output = protenix_predict_static(
                params,
                features,
                key=jax.random.PRNGKey(seed),
                num_samples=args.num_samples,
                num_sampling_steps=num_steps,
                s_max=args.s_max,
                s_min=args.s_min,
                rho=args.rho,
                sigma_data=args.sigma_data,
                recycling_steps=num_recycles,
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
                confidence_triangle_attention_backend=(
                    args.confidence_triangle_attention_backend
                ),
                run_confidence=not args.no_confidence,
                run_confidence_scores=not args.no_confidence_scores,
                stop_after_trunk=args.stop_after == "trunk",
                capture_names=wanted_representations,
                return_trunk=(
                    (wants_raw and args.include_trunk)
                    or bool(wanted_representations)
                ),
                return_confidence_logits=wants_raw,
                triangle_mul_chunk_size=chunk_config.triangle_mul_chunk_size,
                triangle_att_q_chunk_size=chunk_config.triangle_att_q_chunk_size,
                single_att_q_chunk_size=chunk_config.single_att_q_chunk_size,
                token_q_chunk_size=chunk_config.token_q_chunk_size,
                opm_chunk_size=chunk_config.opm_chunk_size,
                diffusion_chunk_size=chunk_config.diffusion_chunk_size,
                trunk_dtype=trunk_dtype,
                cycle_msa_features=cycle_msa_features,
                gamma0=gamma0,
                step_scale_eta=eta,
                preserve_prefix_rng=preserve_prefix_rng,
                guidance_config=guidance_config,
                guidance_features=guidance_features,
                graph_jit=not args.no_graph_jit,
                cp_shards=args.cp_devices,
                cp_layout=args.cp_layout,
                padded_generated_schema=padding_plan is not None,
                init_noise=init_noise,
                step_noises=step_noises,
            )
            if args.prewarm_only:
                jax.block_until_ready(output)
                print(
                    "prewarmed: "
                    f"job={job['name']} tokens={n_token} "
                    f"atoms={features['atom_to_token_idx'].shape[0]} "
                    f"samples={args.num_samples}"
                )
                break
            if padding_plan is not None:
                from foldjax.models.protenix.data.padding import (
                    crop_protenix_outputs,
                )

                output = crop_protenix_outputs(output, padding_plan)
            if args.stop_after == "trunk":
                destination = args.representations_dir or (
                    args.out
                    if legacy_npz
                    else args.out
                    / sanitize_job_name(job["name"])
                    / f"seed_{seed}"
                    / "predictions"
                )
                archive = _representations.save(
                    destination,
                    _collect_representations(output, wanted_representations),
                    _representations.specs_for("protenix"),
                    model="protenix",
                )
                if archive is not None:
                    written.append(archive)
                    print(f"wrote: {archive}")
                continue
            if args.output_format in ("protenix", "both"):
                paths = write_protenix_outputs(
                    args.out,
                    job_name=job["name"],
                    seed=seed,
                    output=output,
                    features=output_features,
                    include_raw=args.output_format == "both",
                    include_trunk=args.include_trunk,
                )
                if wanted_representations:
                    archive = _representations.save(
                        args.representations_dir or paths[0].parent,
                        _collect_representations(output, wanted_representations),
                        _representations.specs_for("protenix"),
                        model="protenix",
                    )
                    if archive is not None:
                        written.append(archive)
                written.extend(paths)
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
                if wanted_representations:
                    archive = _representations.save(
                        args.representations_dir or output_path.parent,
                        _collect_representations(output, wanted_representations),
                        _representations.specs_for("protenix"),
                        model="protenix",
                    )
                    if archive is not None:
                        written.append(archive)
                written.append(output_path)
                print(f"wrote: {output_path}")
    return written


def _prefix_rng_is_supported() -> bool:
    """Whether masked padding draws preserve JAX's compact random prefix."""

    from foldjax.models._random import supports_masked_prefix_draw

    return supports_masked_prefix_draw()


def _padded_noise_tapes(
    *,
    seed: int,
    num_samples: int,
    num_steps: int,
    actual_atom: int,
    target_atom: int,
    diffusion_chunk_size: int | None,
) -> tuple[Any, Any]:
    """Generate the exact unpadded random stream, then right-pad it.

    Generating directly at ``target_atom`` preserves sample zero's flat prefix
    but changes every later sample's offset. Building the released real shape
    first keeps all real atoms of every sample identical to the default path.
    The per-step result is packed as ``[steps, samples, atoms, 3]`` for the
    scanned production sampler; direct library callers may still pass a
    sequence of step arrays.
    """

    import jax
    import jax.numpy as jnp

    from foldjax.models.protenix.data.padding import pad_atom_noise

    root_key = jax.random.PRNGKey(seed)
    if diffusion_chunk_size is None or diffusion_chunk_size <= 0:
        chunk_sizes = (num_samples,)
        chunk_keys = (root_key,)
    else:
        chunk_sizes = tuple(
            min(diffusion_chunk_size, num_samples - start)
            for start in range(0, num_samples, diffusion_chunk_size)
        )
        chunk_keys = tuple(jax.random.split(root_key, len(chunk_sizes)))

    init_chunks = []
    step_chunks: list[list[Any]] = [[] for _ in range(num_steps)]
    for chunk_size, chunk_key in zip(chunk_sizes, chunk_keys, strict=True):
        chunk_key, init_key = jax.random.split(chunk_key)
        init_chunks.append(
            jax.random.normal(
                init_key,
                (chunk_size, actual_atom, 3),
                dtype=jnp.float32,
            )
        )
        for step_index, step_key in enumerate(jax.random.split(chunk_key, num_steps)):
            step_chunks[step_index].append(
                jax.random.normal(
                    step_key,
                    (chunk_size, actual_atom, 3),
                    dtype=jnp.float32,
                )
            )
    init_noise = jnp.concatenate(init_chunks, axis=0)
    step_noises = jnp.stack(
        tuple(jnp.concatenate(chunks, axis=0) for chunks in step_chunks), axis=0
    )
    return (
        pad_atom_noise(init_noise, actual=actual_atom, target=target_atom),
        pad_atom_noise(step_noises, actual=actual_atom, target=target_atom),
    )


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
