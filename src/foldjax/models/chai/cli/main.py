"""Public command-line interface for standalone Chai-JAX prediction."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from foldjax.models.chai.cli.assets import add_assets_parser, run_assets
from foldjax.models.chai.inference import (
    InferenceConfig,
    _default_cache,
    run_inference,
)


def _existing_file(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chai-jax",
        description="Run standalone Chai-1 inference with JAX.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_assets_parser(subparsers)
    fold = subparsers.add_parser("fold", help="Run Chai-1 to fold a complex.")
    fold.add_argument("fasta_file", type=_existing_file)
    fold.add_argument("--output-dir", required=True, type=Path)
    fold.add_argument(
        "--bundle-path",
        required=True,
        type=Path,
        help="native Chai-JAX model bundle directory",
    )
    fold.add_argument(
        "--conformer-path",
        required=True,
        type=Path,
        help="native Chai-JAX conformer archive",
    )

    modalities = fold.add_argument_group("ESM, MSA, restraints, and templates")
    modalities.add_argument(
        "--use-esm-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use native ESM2 inference or --esm-embeddings-path (default: enabled)",
    )
    modalities.add_argument("--esm-embeddings-path", type=Path)
    modalities.add_argument(
        "--esm-model-path",
        type=Path,
        default=_default_cache("models", "esm2_t36_3B"),
    )
    modalities.add_argument(
        "--esm-cache-directory",
        type=Path,
        default=_default_cache("esm_embeddings"),
    )
    modalities.add_argument(
        "--esm-attention-implementation", choices=("xla", "cudnn"), default="xla"
    )
    modalities.add_argument(
        "--use-msa-server",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    modalities.add_argument(
        "--msa-server-url", default="https://api.colabfold.com"
    )
    modalities.add_argument(
        "--msa-server-version", default="colabfold-public-api-v1"
    )
    modalities.add_argument("--msa-directory", type=Path)
    modalities.add_argument("--msa-cache-directory", type=Path)
    modalities.add_argument("--constraint-path", type=Path)
    modalities.add_argument(
        "--use-templates-server",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    modalities.add_argument("--template-hits-path", type=Path)
    modalities.add_argument("--template-cif-directory", type=Path)
    modalities.add_argument(
        "--template-cache-directory",
        type=Path,
        default=_default_cache("templates"),
    )
    modalities.add_argument("--kalign-executable", default="kalign")

    inference = fold.add_argument_group("inference")
    inference.add_argument("--num-trunk-recycles", type=int, default=3)
    inference.add_argument("--recycle-msa-subsample", type=int, default=0)
    inference.add_argument("--num-diffn-timesteps", type=int, default=200)
    inference.add_argument("--num-diffn-samples", type=int, default=5)
    inference.add_argument("--num-trunk-samples", type=int, default=1)
    inference.add_argument("--seed", type=int)

    io = fold.add_argument_group("native assets and IO")
    io.add_argument("--expected-conformer-version")
    io.add_argument("--compilation-cache-dir", type=Path)
    io.add_argument(
        "--fasta-names-as-cif-chains",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def _fold(args: argparse.Namespace) -> int:
    config = InferenceConfig(
        num_trunk_recycles=args.num_trunk_recycles,
        recycle_msa_subsample=args.recycle_msa_subsample,
        num_diffusion_timesteps=args.num_diffn_timesteps,
        num_diffusion_samples=args.num_diffn_samples,
        num_trunk_samples=args.num_trunk_samples,
        seed=args.seed,
        use_esm_embeddings=args.use_esm_embeddings,
        esm_embeddings_path=args.esm_embeddings_path,
        esm_model_path=args.esm_model_path,
        esm_cache_directory=args.esm_cache_directory,
        esm_attention_implementation=args.esm_attention_implementation,
        use_msa_server=args.use_msa_server,
        msa_directory=args.msa_directory,
        msa_server_url=args.msa_server_url,
        msa_server_version=args.msa_server_version,
        msa_cache_directory=args.msa_cache_directory,
        use_templates_server=args.use_templates_server,
        template_hits_path=args.template_hits_path,
        template_cif_directory=args.template_cif_directory,
        template_cache_directory=args.template_cache_directory,
        kalign_executable=args.kalign_executable,
        constraint_path=args.constraint_path,
        fasta_names_as_cif_chains=args.fasta_names_as_cif_chains,
        compilation_cache_dir=args.compilation_cache_dir,
    )
    config.validate()
    candidates = run_inference(
        args.fasta_file,
        output_dir=args.output_dir,
        bundle_path=args.bundle_path,
        conformer_path=args.conformer_path,
        config=config,
        expected_conformer_version=args.expected_conformer_version,
    )
    for path in candidates.cif_paths:
        print(path)
    msa_plot_path = getattr(candidates, "msa_coverage_plot_path", None)
    if msa_plot_path is not None:
        print(msa_plot_path)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI, returning a process exit status for testability."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "fold":
            return _fold(args)
        if args.command == "assets":
            return run_assets(args)
    except Exception as error:
        print(f"chai-jax: error: {error}", file=sys.stderr)
        return 1
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
