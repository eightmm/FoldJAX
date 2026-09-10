"""Flag declarations the ports' predict CLIs share verbatim.

Protenix and OpenDDE each publish a `foldjax-<port>-predict` console script,
and the FoldJAX backends drive those same parsers in-process by rendering argv
from their `_CLI_OPTIONS` sets. So a spelling, alias, type, default or choice
that drifts between the two ports is user-visible twice over: once in `--help`,
and once as a command one port accepts and the other rejects.

Only declarations that were already byte-identical in both parsers live here.
Flags the two ports spell the same but describe or default differently --
`--seed`, `--input-json`, `--num-steps`, `--num-recycles`, `--trunk-dtype`,
`--chunk-policy`, `--cp-devices`, `--cp-layout`, `--representations`,
`--representations-dir`, `--stop-after`, `--compile-cache`,
`--template-mmcif-dir` -- stay in their own parsers. Their prose and their
defaults are the per-port interface rather than duplication, and sharing them
would mean assembling help text from fragments.

Each function declares one run of flags that is contiguous in both parsers, so
a caller keeps the order its `--help` already prints. Declaration order is help
order, so calling one of these at the wrong point is itself a visible change;
`tests/models/test_shared_predict_flags.py` pins both the declarations and each
parser's full ordered flag list.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def add_weights_and_output(parser: argparse.ArgumentParser) -> None:
    """Where the native checkpoint is read from and predictions are written."""
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)


def add_sample_count(parser: argparse.ArgumentParser) -> None:
    """How many diffusion samples a job produces, under both spellings."""
    parser.add_argument(
        "--num-samples",
        "--n-sample",
        dest="num_samples",
        type=int,
        default=5,
    )


def add_atom_neighbourhood(parser: argparse.ArgumentParser) -> None:
    """The local atom-attention window both ports inherit from AF3."""
    parser.add_argument("--n-queries", type=int, default=32)
    parser.add_argument("--n-keys", type=int, default=128)


def add_msa_depth(parser: argparse.ArgumentParser) -> None:
    """The MSA row cap, under both spellings. Unset means the port's own."""
    parser.add_argument(
        "--max-msa-depth",
        "--max-msa-rows",
        dest="max_msa_depth",
        type=int,
        default=None,
    )


def add_trunk_chunk_sizes(parser: argparse.ArgumentParser) -> None:
    """The four trunk chunk overrides. Unset leaves them to `--chunk-policy`."""
    parser.add_argument("--triangle-mul-chunk-size", type=int)
    parser.add_argument("--triangle-att-q-chunk-size", type=int)
    parser.add_argument("--single-att-q-chunk-size", type=int)
    parser.add_argument("--token-q-chunk-size", type=int)


def add_attention_backends(parser: argparse.ArgumentParser) -> None:
    """The two attention kernels both ports expose, with the shared default."""
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


def add_graph_jit(parser: argparse.ArgumentParser) -> None:
    """The op-by-op escape hatch kept for debugging and numerical comparison."""
    parser.add_argument(
        "--no-graph-jit",
        action="store_true",
        help="trace the model op by op instead of as one compiled graph; "
        "much slower, kept for debugging and numerical comparison",
    )
