"""Standalone Protenix JAX prediction from native weights."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from foldjax import memory_policy
from foldjax.models import _predict_flags
from foldjax.models._glu import GLU_BACKENDS
from foldjax.models.protenix.amp_policy import (
    AMP_POLICY_CHOICES,
    DEFAULT_AMP_POLICY,
)
from foldjax.models.protenix.runner import (
    PREPARED_PARAMS_LOADER_API,
    PredictionConfig,
    _load_prepared_params,
    _resolve_msa_depth,
    run_prediction,
)

#: Re-exported, not re-implemented. The FoldJAX backend negotiates
#: request-scoped weight reuse by reading the sentinel off whichever module it
#: imports and calling that module's loader, and this is the module it imports.
#: The contract lives on the runner; this keeps the spelling the backend and
#: its test doubles were written against.
__all__ = ["PREPARED_PARAMS_LOADER_API", "_load_prepared_params", "main"]


def main(
    argv: Sequence[str] | None = None,
    *,
    on_padding_plan: Callable[..., None] | None = None,
    _prepared_params_loader: Callable[[Path, str, bool], Any] | None = None,
) -> list[Path]:
    """Parse argv into a :class:`PredictionConfig`, then run the prediction."""

    parser = argparse.ArgumentParser(description=__doc__)
    feature_group = parser.add_mutually_exclusive_group(required=True)
    feature_group.add_argument("--features", type=Path)
    feature_group.add_argument("--input-json", type=Path)
    _predict_flags.add_weights_and_output(parser)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seed", type=int, help="Single seed (legacy spelling; default: 101)."
    )
    seed_group.add_argument(
        "--seeds", type=int, nargs="+", help="Run every job with each listed seed."
    )
    # Deliberately outside `seed_group`: this is the other half of a pair, not
    # an alternative spelling of it. The per-cycle MSA row draw and the
    # diffusion noise both read `--seed` today, so varying the MSA subset also
    # redraws every sample and the two effects cannot be separated. Naming this
    # separately holds one fixed while the other moves.
    parser.add_argument(
        "--msa-seed",
        type=int,
        help="Seed for the per-cycle random MSA row draw "
        "(--sample-msa-per-cycle). Defaults to --seed; setting it leaves the "
        "diffusion RNG on --seed.",
    )
    parser.add_argument(
        "--output-format",
        choices=("npz", "protenix", "both"),
        default="npz",
        help="Write legacy NPZ, ranked CIF/JSON output, or both.",
    )
    _predict_flags.add_sample_count(parser)
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
    _predict_flags.add_atom_neighbourhood(parser)
    _predict_flags.add_msa_depth(parser)
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
    # Beside the token limit above because they answer the same question with a
    # measurement instead of a constant: this port's own fitted peak law against
    # the ceiling the allocator reports for this card.
    parser.add_argument(
        "--memory-check",
        choices=memory_policy.CHECK_MODES,
        default=memory_policy.DEFAULT_CHECK_MODE,
        help="what to do when the estimated peak does not fit the device: "
        "refuse before the weights are loaded, or warn and run anyway",
    )
    parser.add_argument(
        "--memory-budget-gib",
        type=float,
        help="plan against this much device memory instead of what the "
        "allocator reports; the smaller of the two is used",
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
    _predict_flags.add_trunk_chunk_sizes(parser)
    parser.add_argument("--opm-chunk-size", type=int)
    parser.add_argument("--diffusion-chunk-size", type=int)
    parser.add_argument(
        "--trunk-dtype",
        choices=("bf16", "fp32"),
        default="bf16",
        help="Use upstream-style BF16 trunk; which of the diffusion and "
        "confidence stages stay FP32 beside it is --amp-policy.",
    )
    parser.add_argument(
        "--amp-policy",
        choices=AMP_POLICY_CHOICES,
        default=DEFAULT_AMP_POLICY,
        help="which stages run under the BF16 autocast: 'auto' (default) "
        "runs the confidence head under it at every size and gates the "
        "diffusion sampler above 3840 tokens, 'upstream' reproduces "
        "upstream's own gate instead (confidence head only above 2560 "
        "tokens), 'fp32' and 'bf16' pin both stages at every size; realised "
        "only under --trunk-dtype bf16, since an FP32 trunk opens no "
        "autocast for a stage to run in",
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
    _predict_flags.add_deterministic_ops(parser)
    _predict_flags.add_attention_backends(
        parser,
        extra_backends=("tokamax",),
        # Measured on this port; see `models/predict.py`.
        diffusion_default="tokamax",
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
    parser.add_argument(
        "--glu-backend",
        choices=GLU_BACKENDS,
        default="xla",
        # Declared here rather than in `_predict_flags`, which is for flags
        # both ports already spelled identically. OpenDDE reaches these same
        # transitions through Protenix's primitives, so a flag added to the
        # shared builder would offer the kernel on a port nobody has measured
        # it on; the attention builder's docstring makes the same argument.
        help="which gated-linear-unit implementation every transition runs; "
        "'tokamax' is an opt-in fused Triton kernel that never writes the "
        "widened intermediate, unmeasured on this port and unavailable under "
        "context parallelism",
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
    _predict_flags.add_graph_jit(parser)
    parser.add_argument(
        "--cp-devices",
        type=int,
        default=1,
        help="shard the pair representations across this many JAX devices "
        "(context parallelism, the JAX form of OpenDDE's Fold-CP); needs "
        "that many visible devices",
    )
    atom_windows_group = parser.add_mutually_exclusive_group()
    atom_windows_group.add_argument(
        "--cp-atom-windows",
        dest="cp_atom_windows",
        action="store_true",
        help="distribute the diffusion atom graph (atom-pair cache, both atom "
        "transformer stacks, atom<->token routing) over the context-parallel "
        "rows; the default, and ignored without --cp-devices > 1",
    )
    atom_windows_group.add_argument(
        "--no-cp-atom-windows",
        dest="cp_atom_windows",
        action="store_false",
        help="keep the diffusion atom graph replicated on every "
        "context-parallel device",
    )
    parser.set_defaults(cp_atom_windows=True)
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
        choices=("full", "inputs", "trunk"),
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
    args.max_msa_depth = _resolve_msa_depth(args.max_msa_depth)
    return run_prediction(
        PredictionConfig(**vars(args)),
        on_padding_plan=on_padding_plan,
        _prepared_params_loader=_prepared_params_loader,
    )


if __name__ == "__main__":
    main()
