"""Torch-free OpenDDE prediction from JSON and native JAX weights."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from foldjax.models import _predict_flags
from foldjax.models.opendde.runner import (
    DIFFUSION_DTYPE_CHOICES,
    PREPARED_PARAMS_LOADER_API,
    PredictionConfig,
    _load_prepared_params,
    _resolve_msa_depth,
    run_prediction,
)
from foldjax.schema import PaddingConfig

#: Re-exported, not re-implemented. The FoldJAX backend negotiates
#: request-scoped weight reuse by reading the sentinel off whichever module it
#: imports and calling that module's loader, and this is the module it imports.
#: The contract lives on the runner; this keeps the spelling the backend and
#: its test doubles were written against.
__all__ = ["PREPARED_PARAMS_LOADER_API", "_load_prepared_params", "main"]


def _boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def main(
    argv: Sequence[str] | None = None,
    *,
    padding: PaddingConfig | None = None,
    padding_profiles: list[dict[str, Any]] | None = None,
    _prepared_params_loader: Callable[[Path, str, bool], Any] | None = None,
) -> list[Path]:
    """Parse argv into a :class:`PredictionConfig`, then run the prediction."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    _predict_flags.add_weights_and_output(parser)
    parser.add_argument("--seed", type=int)
    _predict_flags.add_sample_count(parser)
    parser.add_argument(
        "--num-steps",
        "--n-step",
        dest="num_steps",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--num-recycles",
        "--n-cycle",
        dest="num_recycles",
        type=int,
        default=10,
    )
    _predict_flags.add_atom_neighbourhood(parser)
    parser.add_argument("--use-template", type=_boolean, default=False)
    parser.add_argument("--use-rna-msa", type=_boolean, default=False)
    _predict_flags.add_msa_depth(parser)
    _predict_flags.add_deterministic_ops(parser)
    _predict_flags.add_attention_backends(parser)
    parser.add_argument(
        "--structural-single-attention-backend",
        choices=("xla", "xla_jit", "xla_sdpa"),
        default="xla_jit",
    )
    _predict_flags.add_graph_jit(parser)
    parser.add_argument(
        "--cp-devices",
        type=int,
        default=1,
        help="shard the pair representations across this many JAX devices "
        "(context parallelism, the JAX form of upstream's Fold-CP); needs "
        "that many visible devices and trades collective traffic for pair "
        "memory per device",
    )
    # Spelled `true`/`false` rather than as Protenix's `--cp-atom-windows` /
    # `--no-cp-atom-windows` pair: this parser has no store_true/store_false
    # boolean anywhere, its two existing switches (`--use-template`,
    # `--use-rna-msa`) both take a value, and the adapter's flag loop renders
    # `--flag <value>` for every option it carries. The negative-flag
    # machinery Protenix' adapter needed exists only because its loop drops a
    # falsey value.
    parser.add_argument(
        "--cp-atom-windows",
        type=_boolean,
        default=True,
        help="distribute the diffusion atom graph (atom-pair cache, both atom "
        "transformer stacks, atom<->token routing) over the context-parallel "
        "rows; true by default and ignored without --cp-devices > 1. Needs the "
        "padded atom axis to be a multiple of n-queries times the row count "
        "and the structural token axis to divide the rows; a shape that "
        "cannot be split warns and runs replicated",
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
    # `resolve_chunk_config` has taken this override all along and `infer`
    # consumes it, but nothing could set it: the four below had flags and this
    # did not, so the only value it ever held was the policy's. It is the one
    # chunk knob measured to change whether a size runs -- serialising the
    # sample axis took OpenFold3 at 4,100 tokens from an unplaceable 107.85 GiB
    # request to a completed 76.4 GiB run.
    parser.add_argument("--diffusion-chunk-size", type=int)
    _predict_flags.add_trunk_chunk_sizes(parser)
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
        default="bf16",
        help="element width of the embedder and both trunks; the diffusion "
        "sampler, the distogram head and every confidence logit stay FP32 "
        "either way, and the confidence head's own stack is --confidence-dtype. "
        "Defaults to BF16, which on the eight-case panel is 13-39% faster, "
        "33-52% lighter and no further from upstream than FP32; pass fp32 for "
        "upstream's own trunk policy",
    )
    parser.add_argument(
        "--confidence-dtype",
        choices=("fp32", "bf16"),
        default="bf16",
        help="element width of the confidence head's re-embedding Pairformer "
        "and the three activations entering it. Defaults to BF16, which "
        "reproduces AlphaFold 3's confidence boundary: narrow stack, FP32 "
        "pLDDT/PAE/PDE/resolved logits. It moves scores only -- the head runs "
        "after the sampler and emits no coordinates, so no structure can move "
        "-- and is independent of --trunk-dtype; pass fp32 to keep the whole "
        "head wide",
    )
    # Native OpenDDE defaults to FP32. The five-sample, fixed-tape native
    # precision panel rejects BF16 on 5SAK/1URN; memory savings and a matched
    # best-ranked sample cannot authorize lowering every sample's precision.
    parser.add_argument(
        "--diffusion-dtype",
        choices=DIFFUSION_DTYPE_CHOICES,
        default="fp32",
        help="element width of the denoising network's matmuls -- upstream's "
        "skip_amp.sample_diffusion. A sibling of --trunk-dtype, not a second "
        "spelling of it: this narrows only what upstream's autocast narrows "
        "and keeps eleven geometry and conditioning projections FP32, the "
        "sampler state FP32, and every per-head pair bias delivered in FP32. "
        "Defaults to FP32, which is what upstream runs here and what every "
        "OpenDDE accuracy row was measured on; bf16 is opt-in, needs "
        "--trunk-dtype bf16, and is unsupported under context parallelism",
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
        choices=("full", "inputs", "trunk"),
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
    args.max_msa_depth = _resolve_msa_depth(args.max_msa_depth)
    return run_prediction(
        PredictionConfig(**vars(args)),
        padding=padding,
        padding_profiles=padding_profiles,
        _prepared_params_loader=_prepared_params_loader,
    )


if __name__ == "__main__":
    main()
