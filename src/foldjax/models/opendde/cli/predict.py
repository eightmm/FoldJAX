"""Torch-free OpenDDE prediction from JSON and native JAX weights."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from foldjax.models import _representations
from foldjax.models.protenix.chunking import ChunkPolicyName
from foldjax.models.protenix.data.template_features import dedup_templates
from foldjax.schema import PaddingConfig


def _load_jobs(path: Path) -> list[dict[str, Any]]:
    from foldjax.models.opendde.data.featurize_json import load_jobs

    return load_jobs(path)


def _featurize(job: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    from foldjax.models.opendde.data.featurize_json import featurize_opendde_json

    return featurize_opendde_json(job, **kwargs)


def _load_weights(path: Path) -> Any:
    from foldjax.models.opendde.bridge.weights_io import load_native_weights

    return load_native_weights(path)


def _predict(
    features: dict[str, Any],
    params: Any,
    *,
    seed: int,
    n_sample: int,
    n_step: int,
    n_cycle: int,
    n_queries: int,
    n_keys: int,
    diffusion_attention_backend: str,
    trunk_single_attention_backend: str,
    structural_single_attention_backend: str,
    chunk_policy: ChunkPolicyName = "auto",
    chunk_overrides: Mapping[str, int | None] | None = None,
    graph_jit: bool = True,
    cp_shards: int = 1,
    cp_layout: str = "auto",
    trunk_dtype: Any = None,
    # The summaries come out of the graph; the raw logits only when a raw dump
    # was asked for. `_score` passes precomputed summaries straight through.
    run_confidence_scores: bool = True,
    return_confidence_logits: bool = False,
    return_representations: bool = False,
    stop_after_trunk: bool = False,
    capture_names: tuple[str, ...] = (),
    n_chain: int | None = None,
    cycle_msa_features: tuple[dict[str, Any], ...] | None = None,
    init_noise: Any = None,
    step_noises: Sequence[Any] | None = None,
    rotations: Any = None,
    translations: Any = None,
) -> dict[str, Any]:
    import jax

    from foldjax.models.opendde.models.model import (
        opendde_infer_compiled,
        opendde_infer_static,
    )
    from foldjax.models.opendde.models.msa_sampling import (
        sample_opendde_msa_cycle_features,
    )
    from foldjax.models.protenix.chunking import resolve_chunk_config
    from foldjax.models.protenix.models.diffusion.diffusion import (
        inference_noise_schedule,
    )

    sampled = cycle_msa_features
    if sampled is None:
        sampled = sample_opendde_msa_cycle_features(
            features,
            n_cycle=n_cycle,
            seed=seed,
        )
    if cp_shards > 1 and not graph_jit:
        raise ValueError(
            "context parallelism requires the compiled graph; "
            "drop --no-graph-jit or --cp-devices"
        )
    infer = opendde_infer_compiled if graph_jit else opendde_infer_static
    # Rolling the repeated stacks into `lax.scan` only pays once the whole graph
    # is traced: unrolled, 20 diffusion steps become 20 copies of the module in
    # one executable, which is what made the compiled program slow to build and
    # to load back from the cache (105 s vs 187 s to compile, 5 s vs 21 s to
    # load). Op-by-op execution keeps the original unrolled form.
    #
    # `use_sampler_scan` was missing from this list, and it is the one whose
    # repeat count the user sets. At the released 200-step schedule the
    # unrolled sampler put 200 copies of the denoiser into one executable; at
    # the 20 steps this list was tuned against, it was 20, which is why the
    # omission did not show. Measured on a 132-residue job at 5 samples / 200
    # steps / 10 cycles, OpenDDE took 682 s against upstream's 15 s.
    scans = (
        dict(
            use_diffusion_scan=True,
            use_pairformer_scan=True,
            use_structural_refiner_scan=True,
            use_confidence_scan=True,
            use_sampler_scan=True,
        )
        if graph_jit
        else {}
    )
    # OpenDDE drives the Protenix trunk, whose triangle attention materialises a
    # [rows, heads, N, N] score tensor -- quadratic in token count, and unbounded
    # unless the query axis is blocked. Protenix resolves those chunk sizes from
    # its token count; OpenDDE never resolved them at all, so it ran the same
    # code unchunked and asked for 76 GiB on a 488-residue job that Protenix
    # completes in 12 GiB.
    #
    # The size to bound is not the residue count. OpenDDE is dual-branch: the
    # residue trunk and the structural refiner share one set of chunk knobs, and
    # the structural branch runs on sub-residue tokens with three times the
    # heads, so it is several times the larger of the two. This policy therefore
    # only proposes a chunk size; the triangle attention narrows it per call
    # site from that branch's own shape, which is the only place both the token
    # count and the head count are known. `--chunk-policy off` restores the
    # unbounded form.
    n_residue = int(features["restype"].shape[-2])
    # The residue count, as upstream feeds it: OpenDDE resolves the threshold
    # table twice, once with `N_token` for the residue trunk and once with
    # `len(parent_residue_idx)` for the structural branch (opendde.py:1481,
    # 1500-1502). Feeding `max(...)` handed the trunk the structural count,
    # and the structural token count is a near-constant 1.95x the residue one
    # -- 257/132, 948/490, 1886/970, 2978/1531 -- so the trunk landed a whole
    # threshold band or two too low:
    #
    #     tokens  ours(max)  upstream trunk
    #        970        256            None
    #       1531         32             512
    #
    # 32 rows at 1,531 tokens is 48 blocks where upstream runs 3, and the
    # blocked contraction re-reads the whole of `b` once per block, so the
    # traffic is `ceil(N/block) * N^2 * c`. That discontinuity -- None, then
    # 256, then 32 -- is the shape of OpenDDE's time curve, which grows as
    # N^2.71 against upstream's N^1.66.
    #
    # Passing the residue count loses nothing for the structural branch: every
    # call site re-narrows the policy value from its own shape and head count
    # (see the note above), and `_row_block` only ever narrows.
    chunks = resolve_chunk_config(
        n_token=n_residue,
        n_sample=n_sample,
        policy=chunk_policy,
        **{k: v for k, v in (chunk_overrides or {}).items() if v is not None},
    )
    return infer(
        features,
        params,
        inference_noise_schedule(n_step=n_step),
        key=jax.random.PRNGKey(seed),
        n_sample=n_sample,
        n_cycle=n_cycle,
        n_queries=n_queries,
        n_keys=n_keys,
        diffusion_attention_backend=diffusion_attention_backend,
        trunk_single_attention_backend=trunk_single_attention_backend,
        structural_single_attention_backend=structural_single_attention_backend,
        triangle_mul_chunk_size=chunks.triangle_mul_chunk_size,
        triangle_att_q_chunk_size=chunks.triangle_att_q_chunk_size,
        single_att_q_chunk_size=chunks.single_att_q_chunk_size,
        token_q_chunk_size=chunks.token_q_chunk_size,
        cycle_msa_features=sampled or None,
        cp_shards=cp_shards,
        cp_layout=cp_layout,
        trunk_dtype=trunk_dtype,
        run_confidence_scores=run_confidence_scores,
        return_confidence_logits=return_confidence_logits,
        return_representations=return_representations,
        capture_names=capture_names,
        stop_after_trunk=stop_after_trunk,
        n_chain=n_chain,
        init_noise=init_noise,
        step_noises=step_noises,
        rotations=rotations,
        translations=translations,
        **scans,
    )


def _score(
    output: Mapping[str, Any],
    features: Mapping[str, Any],
    *,
    num_recycles: int,
) -> dict[str, Any]:
    from foldjax.models.opendde.postprocess import opendde_confidence_scores

    return {
        **output,
        **opendde_confidence_scores(
            output,
            features,
            num_recycles=num_recycles,
        ),
    }


def _write(root: Path, **kwargs: Any) -> list[Path]:
    from foldjax.models.protenix.data.output import write_protenix_outputs

    return write_protenix_outputs(root, **kwargs)


def _job_seeds(job: Mapping[str, Any], explicit_seed: int | None) -> list[int]:
    if explicit_seed is not None:
        return [explicit_seed]
    seeds = job.get("modelSeeds") if "modelSeeds" in job else [101]
    if seeds == []:
        seeds = [101]
    if not isinstance(seeds, list):
        raise ValueError("modelSeeds must be a non-empty list")
    result = [int(seed) for seed in seeds]
    if len(set(result)) != len(result):
        raise ValueError("modelSeeds must be unique")
    return result


#: OpenDDE's own names for the arrays, against the shared vocabulary. The
#: structural-token trio is kept because it is what the diffusion module
#: conditions on, and it lives in a different space from the residue trio --
#: the manifest says which, so the two are not mistaken for each other.
def _collect_representations(output, wanted):
    """Pull the requested arrays out of the model output.

    The model records them under the shared names already -- the tap is the
    name -- so this only filters to what survived.
    """
    return {name: output[name] for name in wanted if name in output}


def main(
    argv: Sequence[str] | None = None,
    *,
    padding: PaddingConfig | None = None,
    padding_profiles: list[dict[str, Any]] | None = None,
) -> list[Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-sample", type=int, default=5)
    parser.add_argument("--n-step", type=int, default=200)
    parser.add_argument("--n-cycle", type=int, default=10)
    parser.add_argument("--n-queries", type=int, default=32)
    parser.add_argument("--n-keys", type=int, default=128)
    parser.add_argument("--max-msa-rows", type=int, default=16384)
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
        "--structural-single-attention-backend",
        choices=("xla", "xla_jit", "xla_sdpa"),
        default="xla_jit",
    )
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
        "(context parallelism, the JAX form of upstream's Fold-CP); needs "
        "that many visible devices and trades collective traffic for pair "
        "memory per device",
    )
    parser.add_argument(
        "--cp-layout",
        choices=("auto", "1d", "2d"),
        default="auto",
        help="how the pair axes are split: '2d' is Fold-CP's square grid "
        "(rows and columns, O(N^2/P) per device, needs a square device "
        "count), '1d' splits rows only. 'auto' picks 2d when the device "
        "count is a perfect square.",
    )
    parser.add_argument("--triangle-mul-chunk-size", type=int)
    parser.add_argument("--triangle-att-q-chunk-size", type=int)
    parser.add_argument("--single-att-q-chunk-size", type=int)
    parser.add_argument("--token-q-chunk-size", type=int)
    parser.add_argument(
        "--chunk-policy",
        choices=("auto", "manual", "off"),
        default="auto",
        help="block the query axis of the trunk's quadratic attentions by token "
        "count; 'off' materialises them whole and needs tens of gigabytes "
        "past a few hundred tokens",
    )
    parser.add_argument(
        "--trunk-dtype",
        choices=("bf16", "fp32"),
        default="fp32",
        help="element width of the embedder and both trunks; the diffusion "
        "sampler and the output heads stay FP32 either way",
    )
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument(
        "--representations-dir",
        type=Path,
        default=None,
        help=(
            "where to write the representation archive; defaults to the "
            "run's own prediction directory. The common API pins this so "
            "that every model puts it in the same place."
        ),
    )
    parser.add_argument(
        "--stop-after",
        choices=("full", "trunk"),
        default="full",
        help=(
            "'trunk' stops once the representations exist, skipping the "
            "diffusion sampler and the confidence heads. Only useful with "
            "--representations, since it writes no structure."
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
            "single_inputs, single, pair, structural_single_inputs, "
            "structural_single, structural_pair."
        ),
    )
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--compile-cache", type=Path)
    parser.add_argument(
        "--components-cif",
        type=Path,
        help="official components.cif used for arbitrary CCD atom identity",
    )
    parser.add_argument(
        "--ccd-rdkit-cache",
        type=Path,
        help="trusted official components.cif.rdkit_mol.pkl reference cache",
    )
    parser.add_argument(
        "--template-mmcif-dir",
        type=Path,
        help="local PDB mmCIF directory used to resolve template search hits",
    )
    parser.add_argument(
        "--template-release-dates",
        type=Path,
        help="official release_date_cache.json used for the model data cutoff",
    )
    parser.add_argument(
        "--template-obsolete-map",
        type=Path,
        help="official obsolete_to_successor.json used for template resolution",
    )
    parser.add_argument(
        "--kalign-binary",
        type=Path,
        help="Kalign 3.3.5 executable used for exact template realignment",
    )
    args = parser.parse_args(argv)

    if padding is not None:
        unsupported = sorted(
            set(padding.explicit_axes)
            - {"tokens", "atoms", "msa", "structural_tokens"}
        )
        if unsupported:
            raise ValueError(
                "OpenDDE does not support explicit padding axes: "
                + ", ".join(unsupported)
            )

    if args.weights.suffix.lower() in {".pt", ".pth", ".ckpt"}:
        raise SystemExit(
            "prediction requires native JAX weights; run "
            "opendde-jax-export-weights first"
        )
    if not args.input_json.is_file():
        raise SystemExit(f"missing input JSON: {args.input_json}")
    if not args.weights.is_file():
        raise SystemExit(f"missing native weights: {args.weights}")
    if args.n_sample < 1 or args.n_step < 1 or args.n_cycle < 1:
        raise SystemExit("n-sample, n-step, and n-cycle must be positive")
    if args.cp_devices < 1:
        raise SystemExit("cp-devices must be positive")
    for path, env_name, label in (
        (
            args.components_cif,
            "PROTENIX_CCD_COMPONENTS_FILE",
            "components.cif",
        ),
        (
            args.ccd_rdkit_cache,
            "PROTENIX_CCD_RDKIT_MOL_FILE",
            "CCD RDKit cache",
        ),
        (
            args.template_release_dates,
            "PROTENIX_TEMPLATE_RELEASE_DATES_FILE",
            "template release-date cache",
        ),
        (
            args.template_obsolete_map,
            "PROTENIX_TEMPLATE_OBSOLETE_FILE",
            "obsolete template map",
        ),
    ):
        if path is None:
            continue
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")
        os.environ[env_name] = str(path.expanduser().resolve())
    if args.template_mmcif_dir is not None:
        if not args.template_mmcif_dir.is_dir():
            raise SystemExit(
                f"missing template mmCIF directory: {args.template_mmcif_dir}"
            )
        os.environ["PROTENIX_TEMPLATE_MMCIF_DIR"] = str(
            args.template_mmcif_dir.expanduser().resolve()
        )
    if args.kalign_binary is not None:
        kalign_binary = args.kalign_binary.expanduser().resolve()
        if not kalign_binary.is_file() or not os.access(kalign_binary, os.X_OK):
            raise SystemExit(f"missing executable Kalign binary: {kalign_binary}")
        os.environ["PROTENIX_KALIGN_BINARY"] = str(kalign_binary)
    if args.cpu_only:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    if args.compile_cache is not None:
        import jax

        cache = args.compile_cache.expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(cache))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

    # Returned so a caller knows which files *this* run produced. FoldJAX used
    # to recover them by globbing the output tree, which cannot tell a
    # structure written now from one left by an earlier run into the same
    # directory.
    wanted_representations = _representations.resolve(
        args.representations, _representations.specs_for("opendde")
    )
    written: list[Path] = []
    try:
        jobs = _load_jobs(args.input_json)
        if not jobs:
            raise ValueError("input JSON must contain at least one job")
        params = _load_weights(args.weights)
        trunk_dtype = None
        if args.trunk_dtype == "bf16":
            import jax.numpy as jnp

            from foldjax.models.opendde.models.model import cast_trunk_params

            trunk_dtype = jnp.bfloat16
            params = cast_trunk_params(params, trunk_dtype)
        for job in jobs:
            job_name = str(job.get("name") or args.input_json.stem)
            for seed in _job_seeds(job, args.seed):
                features = _featurize(
                    job,
                    base_dir=args.input_json.parent,
                    n_queries=args.n_queries,
                    n_keys=args.n_keys,
                    max_msa_rows=args.max_msa_rows,
                    seed=seed,
                )
                output_features = features
                model_features = features
                cycle_msa_features = None
                padding_plan = None
                random_tapes: dict[str, Any] = {}
                n_chain = (
                    int(__import__("numpy").max(output_features["asym_id"])) + 1
                    if "asym_id" in output_features
                    else None
                )
                if padding is not None:
                    import jax

                    from foldjax.models.opendde.data.padding import (
                        pad_opendde_features,
                        select_opendde_model_features,
                    )
                    from foldjax.models.opendde.models.msa_sampling import (
                        sample_opendde_msa_cycle_features,
                    )
                    from foldjax.models.opendde.models.sampling import (
                        make_padded_random_tapes,
                    )

                    sampled = sample_opendde_msa_cycle_features(
                        features,
                        n_cycle=args.n_cycle,
                        seed=seed,
                    )
                    padded_features, cycle_msa_features, padding_plan = (
                        pad_opendde_features(
                            features,
                            sampled,
                            padding,
                            n_queries=args.n_queries,
                            n_keys=args.n_keys,
                        )
                    )
                    model_features = select_opendde_model_features(padded_features)
                    init_noise, step_noises, rotations, translations = (
                        make_padded_random_tapes(
                            key=jax.random.PRNGKey(seed),
                            n_sample=args.n_sample,
                            n_steps=args.n_step,
                            actual_atom=padding_plan.actual["atoms"],
                            target_atom=padding_plan.target["atoms"],
                        )
                    )
                    random_tapes = {
                        "init_noise": init_noise,
                        "step_noises": step_noises,
                        "rotations": rotations,
                        "translations": translations,
                    }
                    profile = padding_plan.summary()
                    if n_chain is not None:
                        # Confidence reductions unroll one loop per chain, so
                        # this value is part of the JIT executable identity even
                        # when every padded tensor axis is otherwise identical.
                        profile = {**profile, "static": {"chains": n_chain}}
                    if padding_profiles is not None:
                        padding_profiles.append(profile)
                    print(padding_plan.message("opendde"))
                # The residue trunk is Protenix's, template stack included, so
                # the four rows a template-free query is padded to are the same
                # four -- one all-gap row and three zero-padded ones identical
                # to each other. `template_embedder` already weights by
                # `template_multiplicity`; this is the call site that produces
                # it, which is per-port even though the mechanism is shared.
                # After padding, because `pad_opendde_features` delegates to
                # `pad_protenix_features`, which requires the native depth of
                # four. OpenDDE reads the templates every recycle, so the stack
                # evaluations come off once per cycle rather than once.
                model_features = dedup_templates(model_features)
                output = _predict(
                    model_features,
                    params,
                    seed=seed,
                    n_sample=args.n_sample,
                    n_step=args.n_step,
                    n_cycle=args.n_cycle,
                    n_queries=args.n_queries,
                    n_keys=args.n_keys,
                    diffusion_attention_backend=args.diffusion_attention_backend,
                    trunk_single_attention_backend=(
                        args.trunk_single_attention_backend
                    ),
                    structural_single_attention_backend=(
                        args.structural_single_attention_backend
                    ),
                    trunk_dtype=trunk_dtype,
                    chunk_policy=args.chunk_policy,
                    chunk_overrides={
                        "triangle_mul_chunk_size": args.triangle_mul_chunk_size,
                        "triangle_att_q_chunk_size": args.triangle_att_q_chunk_size,
                        "single_att_q_chunk_size": args.single_att_q_chunk_size,
                        "token_q_chunk_size": args.token_q_chunk_size,
                    },
                    graph_jit=not args.no_graph_jit,
                    cp_shards=args.cp_devices,
                    cp_layout=args.cp_layout,
                    # Without asym_id the per-chain loops cannot be sized on
                    # the host, and the in-graph summaries need exactly that;
                    # fall back to raw logits + host scoring, the old contract.
                    run_confidence_scores="asym_id" in model_features,
                    return_confidence_logits=(
                        args.include_raw or "asym_id" not in model_features
                    ),
                    capture_names=wanted_representations,
                    n_chain=n_chain,
                    cycle_msa_features=cycle_msa_features,
                    stop_after_trunk=args.stop_after == "trunk",
                    **random_tapes,
                )
                if args.stop_after == "trunk":
                    destination = args.representations_dir or (
                        args.out / job_name / f"seed_{seed}" / "predictions"
                    )
                    archive = _representations.save(
                        destination,
                        _collect_representations(output, wanted_representations),
                        _representations.specs_for("opendde"),
                        model="opendde",
                    )
                    if archive is not None:
                        written.append(archive)
                        print(f"wrote: {archive}")
                    continue
                if padding_plan is not None:
                    from foldjax.models.opendde.data.padding import (
                        crop_opendde_outputs,
                    )

                    output = crop_opendde_outputs(output, padding_plan)
                scored = _score(
                    output,
                    output_features,
                    num_recycles=args.n_cycle,
                )
                paths = _write(
                    args.out,
                    job_name=job_name,
                    seed=seed,
                    output=scored,
                    features=output_features,
                    include_raw=args.include_raw,
                    include_trunk=False,
                )
                if wanted_representations:
                    archive = _representations.save(
                        args.representations_dir or paths[0].parent,
                        _collect_representations(output, wanted_representations),
                        _representations.specs_for("opendde"),
                        model="opendde",
                    )
                    if archive is not None:
                        written.append(archive)
                written.extend(paths)
                print(f"wrote: {paths[0].parent}")
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return written


if __name__ == "__main__":
    main()
