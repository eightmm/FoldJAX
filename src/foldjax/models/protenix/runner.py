"""The Protenix prediction run, as a configuration and one entry point.

Everything here used to live in `cli/predict.py` beside the argument parser, so
the only way to run a prediction was to render argv and re-parse it. The parser
stays there and still produces the identical run; what it produces now is a
:class:`PredictionConfig`, which this module is the consumer of.

The body below is the CLI's, moved across with two lines adapted to a frozen
config -- the MSA-depth default resolves at the parse stage now, and TFG's
sampler-scan override goes through ``_replace`` -- and otherwise unchanged: the
same order of dtype resolution, memory admission, parameter preparation and
output writing, reading its options off the config where it read them off
``argparse``'s namespace.
"""

from __future__ import annotations

import json
import os
import shlex
from collections.abc import Callable
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

from foldjax import memory_policy
from foldjax.models import _representations
from foldjax.models._feature_storage import compact_msa_storage
from foldjax.models.protenix.amp_policy import (
    realise_amp_policy,
    requested_amp_policy,
)
from foldjax.models.protenix.data.compact_categories import (
    compact_ref_atom_category_storage,
    drop_dense_categories_from_writer_snapshot,
)

# Private backend capability: defer request-scoped reuse until after the CLI
# has parsed argv, selected its platform and materialised ESM conditioning.
PREPARED_PARAMS_LOADER_API = True

#: The featurizer's own paired/unpaired assembly cap
#: (`data/featurize_json.py`), resolved here rather than left unset so the
#: compile profile names the depth the run used. Padding does not appear in
#: this decision: it pads the MSA axis up to a bucket and never selects rows.
_DEFAULT_MSA_DEPTH = 16384


class PredictionConfig(NamedTuple):
    """One resolved Protenix run, one field per option the CLI parses.

    The fields are the parser's own destinations, in declaration order, so
    ``PredictionConfig(**vars(parser.parse_args(argv)))`` is exhaustive: a flag
    added to the parser without a field here is a ``TypeError`` rather than an
    option the run silently ignores. ``max_msa_depth`` is the one field the CLI
    resolves before handing it over, because ``None`` there means "this port's
    own depth" rather than "unset".
    """

    features: Path | None
    input_json: Path | None
    weights: Path
    out: Path
    seed: int | None
    seeds: list[int] | None
    msa_seed: int | None
    output_format: str
    num_samples: int
    num_steps: int | None
    s_max: float
    s_min: float
    rho: float
    sigma_data: float
    num_recycles: int | None
    gamma0: float | None
    eta: float | None
    n_queries: int
    n_keys: int
    max_msa_depth: int
    msa_search: str
    msa_cache_dir: Path
    msa_search_version: str | None
    msa_local_command: str | None
    msa_remote_url: str | None
    rna_msa_local_command: str | None
    rna_msa_search_version: str | None
    rna_msa_cache_dir: Path
    template_search_command: str | None
    template_search_version: str | None
    template_search_cache_dir: Path
    template_mmcif_dir: Path | None
    strict_token_limit: bool
    memory_check: str
    memory_budget_gib: float | None
    full_depth_msa: bool
    msa_row_alignment: int
    max_msa_padding_rows: int
    input_atom_heads: int
    atom_encoder_heads: int
    token_heads: int
    atom_decoder_heads: int
    triangle_mul_chunk_size: int | None
    triangle_att_q_chunk_size: int | None
    single_att_q_chunk_size: int | None
    token_q_chunk_size: int | None
    opm_chunk_size: int | None
    diffusion_chunk_size: int | None
    trunk_dtype: str
    amp_policy: str
    chunk_policy: str
    use_pairformer_scan: bool
    diffusion_scan: bool
    sampler_scan: bool
    denoiser_jit: bool
    deterministic_ops: str
    diffusion_attention_backend: str
    trunk_single_attention_backend: str
    trunk_triangle_attention_backend: str | None
    confidence_triangle_attention_backend: str | None
    glu_backend: str
    confidence_scan: bool
    no_confidence: bool
    no_confidence_scores: bool
    no_graph_jit: bool
    cp_devices: int
    cp_atom_windows: bool
    cp_layout: str
    include_trunk: bool
    representations_dir: Path | None
    stop_after: str
    representations: str | None
    cpu_only: bool
    compile_cache: Path
    no_compile_cache: bool
    prewarm_only: bool
    model_name: str
    esm_checkpoint_dir: Path | None
    guidance_config: Path | None
    padding: bool
    pad_tokens: int | None
    pad_atoms: int | None
    pad_msa: int | None
    pad_templates: int | None
    pad_language_model_tokens: int | None
    padding_overflow: str


def _resolve_msa_depth(value: int | None) -> int:
    return _DEFAULT_MSA_DEPTH if value is None else value


def _load_prepared_params(path: Path, trunk_dtype: str) -> Any:
    """Load and apply the same trunk cast as the native CLI."""

    from foldjax.models.protenix.bridge.weights_io import (
        _load_native_weights_with_field_dtype,
        load_native_weights,
    )

    if trunk_dtype == "bf16":
        import jax.numpy as jnp

        from foldjax.models.protenix.models.input_precision import (
            native_input_autocast_params,
        )

        params = _load_native_weights_with_field_dtype(
            path,
            jnp.bfloat16,
            frozenset({"pairformer_output"}),
        )
        return params._replace(
            input_embedder=native_input_autocast_params(params.input_embedder)
        )
    return load_native_weights(path)


def _amp_realised_params(params, policy, cache):
    """One parameter tree per realised policy, prepared once for the run.

    The policy is per job -- it is resolved from that job's token count -- but
    the checkpoint is loaded once, so the realisations are memoised rather than
    rebuilt for every job. A run whose jobs all land on the same side of the
    gate holds exactly one tree, as it did before this option existed.

    A stage the tree does not carry is left alone rather than reached through.
    Every released checkpoint carries both, and the model refuses the
    mismatch that a partial tree would produce -- ``confidence_autocast=True
    but these parameters carry no confidence stage to apply it to``, from
    ``_require_realised_amp_params``. That check owns the error, so preparing
    on top of it would only replace a sentence that names the stage with an
    ``AttributeError`` that names a tuple.
    """
    from foldjax.models.protenix.models.input_precision import (
        native_confidence_autocast_params,
        native_diffusion_autocast_params,
    )

    if policy in cache:
        return cache[policy]
    realised = params
    if policy.confidence_autocast and getattr(realised, "confidence", None) is not None:
        realised = realised._replace(
            confidence=native_confidence_autocast_params(realised.confidence)
        )
    if policy.diffusion_autocast and getattr(realised, "diffusion", None) is not None:
        realised = realised._replace(
            diffusion=native_diffusion_autocast_params(realised.diffusion)
        )
    cache[policy] = realised
    return realised


def _collect_representations(output, wanted):
    """Filter the model output to the requested names.

    The model records them under the shared names already -- the tap is the
    name -- so nothing has to be translated here.
    """
    return {name: output[name] for name in wanted if name in output}


def run_prediction(
    config: PredictionConfig,
    *,
    on_padding_plan: Callable[..., None] | None = None,
    _prepared_params_loader: Callable[[Path, str, bool], Any] | None = None,
) -> list[Path]:
    """Enter the shared compilation-cache scope, then run the prediction.

    The scope covers the whole run, and restores the process' previous JAX
    cache config when it ends. That matters because this is also called
    in-process -- by the FoldJAX backend, and by tests -- where a raw
    ``jax.config.update`` left the caller's setting overwritten.
    """

    with ExitStack() as cache_scope:
        return _run(
            config,
            cache_scope=cache_scope,
            on_padding_plan=on_padding_plan,
            _prepared_params_loader=_prepared_params_loader,
        )


def _run(
    config: PredictionConfig,
    *,
    cache_scope: ExitStack,
    on_padding_plan: Callable[..., None] | None = None,
    _prepared_params_loader: Callable[[Path, str, bool], Any] | None = None,
) -> list[Path]:
    deterministic = config.deterministic_ops == "on"

    padding_requested = config.padding or any(
        value is not None
        for value in (
            config.pad_tokens,
            config.pad_atoms,
            config.pad_msa,
            config.pad_templates,
            config.pad_language_model_tokens,
        )
    )
    padding_config = None
    if padding_requested:
        from foldjax.padding import cp_aligned_padding
        from foldjax.schema import PaddingConfig

        # The mesh this run will build decides what the automatic token and
        # atom targets have to divide; explicit --pad-* values are left as
        # written.  This is also where the neutral backend's request reaches
        # this port, so aligning here covers both entry points.
        padding_config = cp_aligned_padding(
            PaddingConfig(
                tokens=config.pad_tokens,
                atoms=config.pad_atoms,
                msa=config.pad_msa,
                templates=config.pad_templates,
                language_model_tokens=config.pad_language_model_tokens,
                overflow=config.padding_overflow,
            ),
            cp_devices=config.cp_devices,
            cp_layout=config.cp_layout,
        )
        if config.features is not None:
            raise SystemExit(
                "padding currently supports generated --input-json features only; "
                "static NPZ schemas may contain unregistered semantic axes"
            )
        if config.guidance_config is not None:
            raise SystemExit(
                "padding with TFG guidance is not yet supported; use one or the other"
            )
        if not config.full_depth_msa:
            raise SystemExit(
                "padding currently requires --full-depth-msa so random cycle "
                "sampling cannot select padded rows"
            )

    guidance_config = None
    if config.guidance_config is not None:
        try:
            guidance_config = json.loads(
                config.guidance_config.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid guidance config: {exc}") from exc
        if not isinstance(guidance_config, dict):
            raise SystemExit("guidance config must be a JSON object")
        if guidance_config.get("enable") and config.sampler_scan:
            config = config._replace(sampler_scan=False)
            print("TFG enabled: using the non-scan diffusion sampler")

    # Exact atom categories are a private managed-input optimization, not a new
    # public feature ABI. Static archives can be custom, and the eager/TFG
    # routes inspect the dense arrays outside the consolidated graph, so only a
    # generated JSON job on that graph is eligible.
    compact_generated_atom_categories = (
        config.input_json is not None
        and not config.no_graph_jit
        and not bool(guidance_config and guidance_config.get("enable"))
    )

    if config.cpu_only:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import jax
    import jax.numpy as jnp

    if not config.no_compile_cache:
        from foldjax.cache import compilation_cache_scope

        cache = config.compile_cache.expanduser().resolve()
        cache_scope.enter_context(compilation_cache_scope(cache))
        print(f"compile cache: {cache}")

    from foldjax.models.protenix.chunking import (
        PROTENIX_MEASURED_CHUNK_SIZE_THRESHOLDS,
        resolve_chunk_config,
    )
    from foldjax.models.protenix.data.featurize_json import featurize_protein_json
    from foldjax.models.protenix.data.output import (
        project_generated_writer_features,
        sanitize_job_name,
        write_protenix_outputs,
    )
    from foldjax.models.protenix.data.static_io import (
        load_static_feature_npz,
        save_output_npz,
    )
    from foldjax.models.protenix.data.template_features import (
        compact_zero_template_geometry,
        dedup_templates,
    )
    from foldjax.models.protenix.models.predict import protenix_predict_static
    from foldjax.models.protenix.models.trunk_blocks.msa import (
        pad_msa_features_to_bucket,
        sample_msa_cycle_index_tape,
    )
    from foldjax.models.protenix.runtime_policy import (
        KNOWN_MODEL_NAMES,
        infer_model_name_from_path,
        model_inference_defaults,
        validate_inference_limits,
    )

    model_name = config.model_name
    if model_name == "auto":
        model_name = infer_model_name_from_path(config.weights)
        if model_name is None:
            # The model name is not cosmetic. It selects the sampler schedule
            # (the mini models run 5 steps at gamma0=0, the base models 200 at
            # 0.8), it decides whether ESM/ISM conditioning is built at all,
            # and it carries the protenix-v2 token limit. Inferring it from the
            # filename means renaming a checkpoint silently changed all three:
            # an ISM model run from a renamed file quietly became a different
            # model, with no warning and a plausible-looking structure out.
            raise SystemExit(
                f"cannot tell which Protenix model {config.weights.name!r} is, "
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
        config.num_recycles
        if config.num_recycles is not None
        else sampler_defaults["num_recycles"]
    )
    num_steps = (
        config.num_steps
        if config.num_steps is not None
        else sampler_defaults["num_steps"]
    )
    gamma0 = config.gamma0 if config.gamma0 is not None else sampler_defaults["gamma0"]
    eta = config.eta if config.eta is not None else sampler_defaults["step_scale_eta"]

    msa_pipeline = None
    if config.msa_search != "off":
        if config.features is not None:
            raise SystemExit("MSA search requires --input-json")
        if not config.msa_search_version:
            raise SystemExit(
                "--msa-search-version is required when MSA search is enabled"
            )
        from foldjax.models.protenix.data.search import (
            LocalMsaClient,
            MsaSearchPipeline,
            RemoteMMseqs2Client,
        )

        if config.msa_search == "local":
            if not config.msa_local_command:
                raise SystemExit("--msa-local-command is required for local MSA search")
            command = shlex.split(config.msa_local_command)
            if not command:
                raise SystemExit("--msa-local-command must not be empty")
            msa_backend = LocalMsaClient(command, version=config.msa_search_version)
        else:
            if not config.msa_remote_url:
                raise SystemExit("--msa-remote-url is required for remote MSA search")
            msa_backend = RemoteMMseqs2Client(
                config.msa_remote_url, version=config.msa_search_version
            )
        msa_pipeline = MsaSearchPipeline(
            config.msa_cache_dir,
            msa_backend,
            options={"mode": config.msa_search},
        )

    rna_msa_pipeline = None
    if config.rna_msa_local_command is not None:
        if config.features is not None:
            raise SystemExit("RNA MSA search requires --input-json")
        if not config.rna_msa_search_version:
            raise SystemExit(
                "--rna-msa-search-version is required for RNA MSA search"
            )
        command = shlex.split(config.rna_msa_local_command)
        if not command:
            raise SystemExit("--rna-msa-local-command must not be empty")
        from foldjax.models.protenix.data.search import (
            LocalRnaMsaClient,
            RnaMsaSearchPipeline,
        )

        rna_msa_pipeline = RnaMsaSearchPipeline(
            config.rna_msa_cache_dir,
            LocalRnaMsaClient(command, version=config.rna_msa_search_version),
            options={"mode": "local-nhmmer"},
        )
    elif config.rna_msa_search_version:
        raise SystemExit(
            "--rna-msa-local-command is required with --rna-msa-search-version"
        )

    template_pipeline = None
    mmcif_dir = None
    if config.template_mmcif_dir is not None:
        mmcif_dir = config.template_mmcif_dir.expanduser().resolve()
        if not mmcif_dir.is_dir():
            raise SystemExit(f"template mmCIF directory does not exist: {mmcif_dir}")
        # Also enables coordinate resolution for an existing .a3m/.hhr
        # templatesPath when no automatic search command is requested.
        os.environ["PROTENIX_TEMPLATE_MMCIF_DIR"] = str(mmcif_dir)
    if config.template_search_command is not None:
        if config.features is not None:
            raise SystemExit("template search requires --input-json")
        if not config.template_search_version:
            raise SystemExit(
                "--template-search-version is required for template search"
            )
        if mmcif_dir is None:
            raise SystemExit("--template-mmcif-dir is required for template search")
        command = shlex.split(config.template_search_command)
        if not command:
            raise SystemExit("--template-search-command must not be empty")
        from foldjax.models.protenix.data.search import (
            LocalTemplateSearchClient,
            TemplateSearchPipeline,
        )

        template_pipeline = TemplateSearchPipeline(
            config.template_search_cache_dir,
            LocalTemplateSearchClient(
                command, version=config.template_search_version
            ),
            options={"mmcif_dir": str(mmcif_dir)},
        )
    elif config.template_search_version:
        raise SystemExit(
            "--template-search-command is required with template search options"
        )

    esm_name = None
    esm_provider = None
    provider = None
    if model_name is not None and ("_esm_" in model_name or "_ism_" in model_name):
        esm_name = "esm2-3b-ism" if "_ism_" in model_name else "esm2-3b"
        # A static feature NPZ may already carry the publisher-derived
        # ``esm_token_embedding``.  In that case no language-model checkpoint
        # is opened (or even required); the feature is validated below.
        if config.features is None:
            from foldjax.models.protenix.data.esm import JaxEsmProvider

            esm_provider = JaxEsmProvider(
                esm_name,
                checkpoint_dir=config.esm_checkpoint_dir or config.weights.parent,
                # The language-model encoder is its own executable, run before
                # the structure graph; a run asking for repeatable reductions
                # has to carry the option into it too.
                deterministic=deterministic,
            )
    elif padding_config is not None and config.pad_language_model_tokens is not None:
        raise SystemExit(
            "--pad-language-model-tokens applies only to Protenix ESM/ISM variants"
        )

    try:
        if config.features is not None:
            jobs = [
                {
                    "name": config.features.stem,
                    "features": load_static_feature_npz(config.features),
                    "modelSeeds": None,
                }
            ]
        else:
            with config.input_json.open("r", encoding="utf-8") as handle:
                json_jobs = json.load(handle)
            if not isinstance(json_jobs, list) or not json_jobs:
                raise ValueError("input JSON must be a non-empty top-level list")
            jobs = []
            for index, job in enumerate(json_jobs):
                if not isinstance(job, dict):
                    raise ValueError(f"input JSON entry {index} must be an object")
                fallback_name = (
                    config.input_json.stem
                    if len(json_jobs) == 1
                    else f"{config.input_json.stem}_{index}"
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
                    base_dir=config.input_json.parent,
                    n_queries=config.n_queries,
                    n_keys=config.n_keys,
                    max_msa_depth=config.max_msa_depth,
                )
                language_model_profile = None
                if esm_provider is not None and padding_config is not None:
                    from foldjax.padding import (
                        resolve_axis,
                        resolve_token_axis,
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
                    token_target = resolve_axis(
                        int(features["restype"].shape[0]), padding_config, "tokens"
                    )
                    language_model_target = resolve_token_axis(
                        language_model_actual,
                        padding_config,
                        "language_model_tokens",
                        token_target=token_target,
                        fixed_size=min(token_target, esm_provider.max_sequence_length),
                    )
                    if language_model_target > esm_provider.max_sequence_length:
                        raise ValueError(
                            f"language model padding target {language_model_target} "
                            "exceeds "
                            f"model limit {esm_provider.max_sequence_length}; "
                            "set --pad-language-model-tokens explicitly"
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
                        provider = partial(
                            esm_provider.embed,
                            target_length=language_model_target,
                        )

                    features = add_esm_embeddings(
                        esm_features, job, provider=provider
                    )
                    features["residue_index"] = esm_features["residue_index"] + 1
                    # ``dict(features)`` shares every dense atom-category array.
                    # The loop's final temporary would otherwise retain them
                    # through parameter loading and the whole prediction even
                    # after the managed model/output copies release them.
                    del esm_features
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
                strict_token_limit=config.strict_token_limit,
            )
            if (
                padding_config is None
                and config.full_depth_msa
                and config.msa_row_alignment > 0
                and "msa" in features
            ):
                original_msa_rows = int(features["msa"].shape[-2])
                features = pad_msa_features_to_bucket(
                    features,
                    bucket_size=config.msa_row_alignment,
                    max_padding_rows=config.max_msa_padding_rows,
                )
                job["features"] = features
                aligned_msa_rows = int(features["msa"].shape[-2])
                if aligned_msa_rows != original_msa_rows:
                    print(
                        f"{job['name']}: MSA rows aligned: "
                        f"{original_msa_rows} -> {aligned_msa_rows}"
                    )
            # Compact only private prediction storage; the public featurizer
            # remains publisher-compatible. Taking this snapshot after MSA
            # compaction also avoids retaining the wide native MSA arrays next
            # to their model-bound copies for the duration of the run.
            features = compact_msa_storage(features)
            keep_output_features = (
                config.output_format in ("protenix", "both")
                and config.stop_after == "full"
                and not config.prewarm_only
            )
            output_features = features if keep_output_features else None
            if output_features is not None and compact_generated_atom_categories:
                # Generated JSON carries complete explicit atom labels, so the
                # writer does not need to decode the dense categories. The
                # helper keeps both arrays if that metadata contract ever
                # becomes incomplete or shape-drifted.
                output_features = drop_dense_categories_from_writer_snapshot(
                    output_features
                )
            if output_features is not None and config.input_json is not None:
                output_features = project_generated_writer_features(output_features)
            if output_features is not None:
                job["output_features"] = output_features
            if padding_config is not None:
                from foldjax.models.protenix.data.padding import (
                    pad_protenix_features,
                )

                features, padding_plan = pad_protenix_features(
                    features,
                    padding_config,
                    n_queries=config.n_queries,
                    n_keys=config.n_keys,
                    max_msa_depth=config.max_msa_depth,
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
                    strict_token_limit=config.strict_token_limit,
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
            # A template-free query's four quadratic geometry tensors are
            # bitwise zero -- 5.9 GB of arguments at 4,100 tokens over the two
            # survivors -- and the trunk rebuilds them from a scalar. After
            # deduplication, because the dropped arrays are what distinguishes
            # the rows.
            model_features = compact_msa_storage(
                compact_zero_template_geometry(dedup_templates(job["features"]))
            )
            if compact_generated_atom_categories:
                # This is intentionally after every shape-changing operation:
                # padded all-zero rows become the v1 sentinels, and the graph
                # rebuilds the exact historical float32 arrays at entry.
                model_features = compact_ref_atom_category_storage(model_features)
            job["features"] = model_features
            # A Python loop variable outlives the loop. Point it at the compact
            # mapping too, otherwise the final job's pre-compaction (and, with
            # serving padding, separately allocated) dense arrays remain live
            # beside the model until this CLI returns.
            features = model_features
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    if config.output_format != "npz" and (
        config.no_confidence or config.no_confidence_scores
    ):
        raise SystemExit("protenix output format requires confidence scores")

    # Admission, before the structure checkpoint reaches the device below and
    # long before the first trace. Both shapes the peak law needs are final
    # here: `restype` carries the token count the program will be compiled for
    # (the padding target when padding is on), and `msa` carries the row count
    # it will receive -- after `--max-msa-depth`, after serving padding, and
    # after the row-alignment bucket, all of which have already run. Nothing is
    # narrowed to make a job fit; the only outcomes are proceed and refuse.
    memory_budget = memory_policy.device_memory_budget(
        override_gib=config.memory_budget_gib
    )
    off_profile = memory_policy.off_profile_reason(
        num_samples=config.num_samples,
        # Upstream's random per-cycle depths gather a narrower slice than the
        # argument carries, so the row count read below is an upper bound on
        # what the trunk actually holds and the estimate reads high.
        extras=()
        if config.full_depth_msa
        else ("--no-full-depth-msa, which samples fewer rows than it stores",),
    )
    for job in jobs:
        job_features = job["features"]
        msa = job_features.get("msa")
        memory_policy.admit(
            model="protenix",
            n_token=int(job_features["restype"].shape[-2]),
            msa_rows=None if msa is None else int(msa.shape[-2]),
            candidates=(("released", memory_policy.PROTENIX_PEAK),),
            budget=memory_budget,
            mode=config.memory_check,
            levers=(
                "--max-msa-depth lowers the estimate, but by changing the "
                "input: fewer alignment rows is a different prediction, not "
                "the same one in less memory",
            ),
            off_profile=off_profile,
        )

    # ESM/ISM conditioning is fully materialised in each job's compact
    # ``esm_token_embedding`` by this point.  Drop both the direct provider and
    # a possible ``partial`` that closes over it before the structure checkpoint
    # is loaded, otherwise the 3B encoder remains live beside the structure
    # parameters for the entire prediction.
    used_esm_provider = esm_provider is not None
    provider = None
    esm_provider = None

    trunk_dtype = None
    if config.trunk_dtype == "bf16":
        trunk_dtype = jnp.bfloat16
    # The callback is backend-internal. It receives the parser-validated weight
    # path and compute dtype, plus whether retaining the result is safe across
    # calls. A mini ESM/ISM provider is reconstructed per native invocation;
    # keeping the structure tree would overlap it on the next seed.
    params_loader = _prepared_params_loader or _load_prepared_params
    if _prepared_params_loader is None:
        params = params_loader(config.weights, config.trunk_dtype)
    else:
        params = params_loader(
            config.weights,
            config.trunk_dtype,
            not used_esm_provider,
        )
    job_seeds = [_resolve_seeds(config, job.get("modelSeeds")) for job in jobs]
    legacy_npz = (
        config.output_format == "npz" and len(jobs) == 1 and len(job_seeds[0]) == 1
    )
    # Returned so a caller knows which files *this* run produced. FoldJAX used
    # to recover them by globbing the output tree, which cannot tell a
    # structure written now from one left by an earlier run into the same
    # directory.
    wanted_representations = _representations.resolve(
        config.representations,
        (
            {"single_inputs": _representations.specs_for("protenix")["single_inputs"]}
            if config.stop_after == "inputs"
            else _representations.specs_for("protenix")
        ),
    )
    written: list[Path] = []
    amp_params_cache: dict[Any, Any] = {}
    # Only the raw-npz path reads the trunk representations or the full-bin
    # logits; the protenix cif+JSON path consumes the in-graph summaries alone.
    # Keeping unread [num_samples, N, N, 64] logits as program outputs held 21.6
    # GiB at 3,012 tokens (EXPERIMENT_LOG: the 84.1 vs 57.3 attribution).
    wants_raw = config.output_format in ("npz", "both")
    for job, seeds in zip(jobs, job_seeds, strict=True):
        features = job["features"]
        output_features = job.get("output_features")
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
        # Resolved here, from this job's own token count, because that is what
        # upstream's `update_inference_configs` keys on and what the port's own
        # table keys its diffusion half on -- a run with a small and a large
        # job in one `--input-json` gets two policies, and two executables,
        # exactly as upstream would build two configurations.
        amp_policy = realise_amp_policy(
            requested_amp_policy(config.amp_policy, n_token, model_name),
            trunk_is_bf16=trunk_dtype is not None,
        )
        job_params = _amp_realised_params(params, amp_policy, amp_params_cache)
        print(
            f"{job['name']}: amp policy {amp_policy.label()} "
            f"(--amp-policy {config.amp_policy}, n_token={n_token}, "
            f"trunk={config.trunk_dtype})"
        )
        chunk_config = resolve_chunk_config(
            n_token=n_token,
            num_samples=config.num_samples,
            policy=config.chunk_policy,
            # Measured, not upstream's table: this trunk's triangle ops are
            # fused, so chunking them costs time and saves no bytes. See the
            # table's own comment for the numbers.
            thresholds=PROTENIX_MEASURED_CHUNK_SIZE_THRESHOLDS,
            triangle_mul_chunk_size=config.triangle_mul_chunk_size,
            triangle_att_q_chunk_size=config.triangle_att_q_chunk_size,
            single_att_q_chunk_size=config.single_att_q_chunk_size,
            token_q_chunk_size=config.token_q_chunk_size,
            opm_chunk_size=config.opm_chunk_size,
            diffusion_chunk_size=config.diffusion_chunk_size,
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
                    num_samples=config.num_samples,
                    num_steps=num_steps,
                    actual_atom=padding_plan.actual["atoms"],
                    target_atom=padding_plan.target["atoms"],
                    diffusion_chunk_size=chunk_config.diffusion_chunk_size,
                )
            cycle_msa_index_tape = None
            if not config.full_depth_msa:
                cycle_msa_index_tape = sample_msa_cycle_index_tape(
                    features,
                    num_recycles=num_recycles,
                    # The only MSA draw in this loop. The diffusion RNG keeps
                    # `seed` -- both the `PRNGKey` below and the padded noise
                    # tape above -- so `--msa-seed` moves the row subset and
                    # nothing else.
                    seed=seed if config.msa_seed is None else config.msa_seed,
                )
            output = protenix_predict_static(
                job_params,
                features,
                key=jax.random.PRNGKey(seed),
                num_samples=config.num_samples,
                num_sampling_steps=num_steps,
                s_max=config.s_max,
                s_min=config.s_min,
                rho=config.rho,
                sigma_data=config.sigma_data,
                recycling_steps=num_recycles,
                input_atom_heads=config.input_atom_heads,
                atom_encoder_heads=config.atom_encoder_heads,
                token_heads=config.token_heads,
                atom_decoder_heads=config.atom_decoder_heads,
                n_queries=config.n_queries,
                n_keys=config.n_keys,
                use_pairformer_scan=config.use_pairformer_scan,
                use_confidence_scan=config.confidence_scan,
                use_diffusion_scan=config.diffusion_scan,
                use_sampler_scan=config.sampler_scan,
                use_denoiser_jit=config.denoiser_jit,
                diffusion_attention_backend=config.diffusion_attention_backend,
                trunk_single_attention_backend=config.trunk_single_attention_backend,
                trunk_triangle_attention_backend=config.trunk_triangle_attention_backend,
                confidence_triangle_attention_backend=(
                    config.confidence_triangle_attention_backend
                ),
                glu_backend=config.glu_backend,
                run_confidence=not config.no_confidence,
                run_confidence_scores=not config.no_confidence_scores,
                stop_after_trunk=config.stop_after == "trunk",
                stop_after_inputs=config.stop_after == "inputs",
                capture_names=wanted_representations,
                return_trunk=(
                    (wants_raw and config.include_trunk)
                    or bool(wanted_representations)
                ),
                return_confidence_logits=wants_raw,
                # The ranked CIF/JSON writer consumes summaries only. Raw NPZ
                # modes retain the historical pair-detail arrays.
                return_confidence_details=wants_raw,
                triangle_mul_chunk_size=chunk_config.triangle_mul_chunk_size,
                triangle_att_q_chunk_size=chunk_config.triangle_att_q_chunk_size,
                single_att_q_chunk_size=chunk_config.single_att_q_chunk_size,
                token_q_chunk_size=chunk_config.token_q_chunk_size,
                opm_chunk_size=chunk_config.opm_chunk_size,
                diffusion_chunk_size=chunk_config.diffusion_chunk_size,
                trunk_dtype=trunk_dtype,
                confidence_autocast=amp_policy.confidence_autocast,
                diffusion_autocast=amp_policy.diffusion_autocast,
                cycle_msa_index_tape=cycle_msa_index_tape,
                gamma0=gamma0,
                step_scale_eta=eta,
                preserve_prefix_rng=preserve_prefix_rng,
                guidance_config=guidance_config,
                guidance_features=guidance_features,
                graph_jit=not config.no_graph_jit,
                deterministic=deterministic,
                cp_shards=config.cp_devices,
                cp_layout=config.cp_layout,
                cp_atom_windows=config.cp_atom_windows,
                padded_generated_schema=padding_plan is not None,
                init_noise=init_noise,
                step_noises=step_noises,
            )
            if config.prewarm_only:
                jax.block_until_ready(output)
                print(
                    "prewarmed: "
                    f"job={job['name']} tokens={n_token} "
                    f"atoms={features['atom_to_token_idx'].shape[0]} "
                    f"samples={config.num_samples}"
                )
                break
            if padding_plan is not None:
                from foldjax.models.protenix.data.padding import (
                    crop_protenix_outputs,
                )

                output = crop_protenix_outputs(output, padding_plan)
            if config.stop_after in {"inputs", "trunk"}:
                destination = config.representations_dir or (
                    config.out
                    if legacy_npz
                    else config.out
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
            if config.output_format in ("protenix", "both"):
                if output_features is None:
                    raise RuntimeError("structured output feature snapshot is missing")
                paths = write_protenix_outputs(
                    config.out,
                    job_name=job["name"],
                    seed=seed,
                    output=output,
                    features=output_features,
                    include_raw=config.output_format == "both",
                    include_trunk=config.include_trunk,
                )
                if wanted_representations:
                    archive = _representations.save(
                        config.representations_dir or paths[0].parent,
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
                    output_path = config.out
                else:
                    output_path = (
                        config.out
                        / sanitize_job_name(job["name"])
                        / f"seed_{seed}"
                        / "predictions"
                        / "raw_output.npz"
                    )
                print("  output arrays:")
                omitted = save_output_npz(
                    output_path,
                    output,
                    include_trunk=config.include_trunk,
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
                        config.representations_dir or output_path.parent,
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


def _resolve_seeds(config: PredictionConfig, model_seeds: Any) -> list[int]:
    if config.seeds is not None:
        seeds = config.seeds
    elif config.seed is not None:
        seeds = [config.seed]
    elif model_seeds is not None:
        if not isinstance(model_seeds, list) or not model_seeds:
            raise SystemExit("modelSeeds must be a non-empty list")
        seeds = [int(seed) for seed in model_seeds]
    else:
        seeds = [101]
    if len(set(seeds)) != len(seeds):
        raise SystemExit("seeds must be unique")
    return seeds
