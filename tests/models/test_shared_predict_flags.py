"""Pin the flag declarations the Protenix and OpenDDE predict CLIs share.

Both ports expose their own ``foldjax-<port>-predict`` console script, and the
FoldJAX backends drive those same parsers in-process by rendering argv. So a
spelling, type, default or choice that drifts between the two is user-visible
twice over: once in ``--help`` and once as a rejected command.

These parsers are built inside ``main`` and parsed immediately, so there is no
factory to call. The tests capture the constructed parser the way
``tests/models/opendde/test_cache_profile.py`` already does, by intercepting
``parse_args``, and then assert against literal expectations rather than a
generated golden file: the point is that changing one of these values requires
editing this file, in both directions.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from foldjax.models.opendde.cli import predict as opendde_predict
from foldjax.models.protenix.cli import predict as protenix_predict
from tests._parser_capture import capture_parser, flag_order


def _spec(action: argparse.Action) -> dict[str, object]:
    return {
        "dest": action.dest,
        "type": action.type,
        "default": action.default,
        "choices": action.choices,
        "required": action.required,
        "nargs": action.nargs,
        "metavar": action.metavar,
        "const": action.const,
        "action": type(action).__name__,
        "help": action.help,
    }


def _by_option_strings(
    parser: argparse.ArgumentParser,
) -> dict[tuple[str, ...], dict[str, object]]:
    return {tuple(a.option_strings): _spec(a) for a in parser._actions}


#: Every flag both port parsers declare identically, down to the help string.
#: Flags the two ports spell the same but describe or default differently --
#: ``--trunk-dtype``, ``--chunk-policy``, ``--cp-devices``, ``--cp-layout``,
#: ``--stop-after``, ``--representations``, ``--representations-dir``,
#: ``--seed``, ``--num-steps``, ``--num-recycles``, ``--input-json``,
#: ``--compile-cache``, ``--template-mmcif-dir`` -- are deliberately absent:
#: their per-port prose and defaults are the interface, not duplication.
_SHARED_FLAG_SPECS: dict[tuple[str, ...], dict[str, object]] = {
    ("--weights",): {
        "dest": "weights",
        "type": Path,
        "default": None,
        "choices": None,
        "required": True,
        "nargs": None,
        "metavar": None,
        "const": None,
        "action": "_StoreAction",
        "help": None,
    },
    ("--out",): {
        "dest": "out",
        "type": Path,
        "default": None,
        "choices": None,
        "required": True,
        "nargs": None,
        "metavar": None,
        "const": None,
        "action": "_StoreAction",
        "help": None,
    },
    ("--num-samples", "--n-sample"): {
        "dest": "num_samples",
        "type": int,
        "default": 5,
        "choices": None,
        "required": False,
        "nargs": None,
        "metavar": None,
        "const": None,
        "action": "_StoreAction",
        "help": None,
    },
    ("--n-queries",): {
        "dest": "n_queries",
        "type": int,
        "default": 32,
        "choices": None,
        "required": False,
        "nargs": None,
        "metavar": None,
        "const": None,
        "action": "_StoreAction",
        "help": None,
    },
    ("--n-keys",): {
        "dest": "n_keys",
        "type": int,
        "default": 128,
        "choices": None,
        "required": False,
        "nargs": None,
        "metavar": None,
        "const": None,
        "action": "_StoreAction",
        "help": None,
    },
    ("--max-msa-depth", "--max-msa-rows"): {
        "dest": "max_msa_depth",
        "type": int,
        "default": None,
        "choices": None,
        "required": False,
        "nargs": None,
        "metavar": None,
        "const": None,
        "action": "_StoreAction",
        "help": None,
    },
    ("--triangle-mul-chunk-size",): {
        "dest": "triangle_mul_chunk_size",
        "type": int,
        "default": None,
        "choices": None,
        "required": False,
        "nargs": None,
        "metavar": None,
        "const": None,
        "action": "_StoreAction",
        "help": None,
    },
    ("--triangle-att-q-chunk-size",): {
        "dest": "triangle_att_q_chunk_size",
        "type": int,
        "default": None,
        "choices": None,
        "required": False,
        "nargs": None,
        "metavar": None,
        "const": None,
        "action": "_StoreAction",
        "help": None,
    },
    ("--single-att-q-chunk-size",): {
        "dest": "single_att_q_chunk_size",
        "type": int,
        "default": None,
        "choices": None,
        "required": False,
        "nargs": None,
        "metavar": None,
        "const": None,
        "action": "_StoreAction",
        "help": None,
    },
    ("--token-q-chunk-size",): {
        "dest": "token_q_chunk_size",
        "type": int,
        "default": None,
        "choices": None,
        "required": False,
        "nargs": None,
        "metavar": None,
        "const": None,
        "action": "_StoreAction",
        "help": None,
    },
    ("--diffusion-chunk-size",): {
        "dest": "diffusion_chunk_size",
        "type": int,
        "default": None,
        "choices": None,
        "required": False,
        "nargs": None,
        "metavar": None,
        "const": None,
        "action": "_StoreAction",
        "help": None,
    },
    ("--diffusion-attention-backend",): {
        "dest": "diffusion_attention_backend",
        "type": None,
        "default": "xla_jit",
        "choices": ("xla", "xla_jit", "xla_sdpa"),
        "required": False,
        "nargs": None,
        "metavar": None,
        "const": None,
        "action": "_StoreAction",
        "help": None,
    },
    ("--trunk-single-attention-backend",): {
        "dest": "trunk_single_attention_backend",
        "type": None,
        "default": "xla_jit",
        "choices": ("xla", "xla_jit", "xla_sdpa"),
        "required": False,
        "nargs": None,
        "metavar": None,
        "const": None,
        "action": "_StoreAction",
        "help": None,
    },
    ("--no-graph-jit",): {
        "dest": "no_graph_jit",
        "type": None,
        "default": False,
        "choices": None,
        "required": False,
        "nargs": 0,
        "metavar": None,
        "const": True,
        "action": "_StoreTrueAction",
        "help": (
            "trace the model op by op instead of as one compiled graph; "
            "much slower, kept for debugging and numerical comparison"
        ),
    },
    ("--cpu-only",): {
        "dest": "cpu_only",
        "type": None,
        "default": False,
        "choices": None,
        "required": False,
        "nargs": 0,
        "metavar": None,
        "const": True,
        "action": "_StoreTrueAction",
        "help": None,
    },
}

#: The full ordered flag list of each port parser. `--help` renders options in
#: declaration order, so this pins placement as well as membership: a shared
#: builder called at the wrong point in either `main` reorders the help output
#: even when every individual declaration still matches.
_PROTENIX_FLAG_ORDER: tuple[str, ...] = (
    "-h", "--help",
    "--features", "--input-json",
    "--weights", "--out",
    "--seed", "--seeds",
    "--output-format",
    "--num-samples", "--n-sample",
    "--num-steps", "--n-step",
    "--s-max", "--s-min", "--rho", "--sigma-data",
    "--num-recycles", "--n-cycle",
    "--gamma0",
    "--eta", "--step-scale-eta",
    "--n-queries", "--n-keys",
    "--max-msa-depth", "--max-msa-rows",
    "--msa-search", "--msa-cache-dir", "--msa-search-version",
    "--msa-local-command", "--msa-remote-url",
    "--rna-msa-local-command", "--rna-msa-search-version", "--rna-msa-cache-dir",
    "--template-search-command", "--template-search-version",
    "--template-search-cache-dir", "--template-mmcif-dir",
    "--strict-token-limit",
    "--full-depth-msa", "--sample-msa-per-cycle",
    "--msa-row-alignment", "--max-msa-padding-rows",
    "--input-atom-heads", "--atom-encoder-heads",
    "--token-heads", "--atom-decoder-heads",
    "--triangle-mul-chunk-size", "--triangle-att-q-chunk-size",
    "--single-att-q-chunk-size", "--token-q-chunk-size",
    "--opm-chunk-size", "--diffusion-chunk-size",
    "--trunk-dtype", "--chunk-policy",
    "--pairformer-scan", "--no-pairformer-scan",
    "--diffusion-scan",
    "--sampler-scan", "--no-sampler-scan",
    "--denoiser-jit",
    "--deterministic-ops",
    "--diffusion-attention-backend", "--trunk-single-attention-backend",
    "--trunk-triangle-attention-backend",
    "--confidence-triangle-attention-backend",
    "--confidence-scan", "--no-confidence-scan",
    "--no-confidence", "--no-confidence-scores",
    "--no-graph-jit",
    "--cp-devices", "--cp-layout",
    "--include-trunk",
    "--representations-dir", "--stop-after", "--representations",
    "--cpu-only",
    "--compile-cache", "--no-compile-cache", "--prewarm-only",
    "--model-name", "--esm-checkpoint-dir", "--guidance-config",
    "--padding", "--pad-tokens", "--pad-atoms", "--pad-msa",
    "--pad-templates", "--pad-language-model-tokens", "--padding-overflow",
)

_OPENDDE_FLAG_ORDER: tuple[str, ...] = (
    "-h", "--help",
    "--input-json", "--weights", "--out",
    "--seed",
    "--num-samples", "--n-sample",
    "--num-steps", "--n-step",
    "--num-recycles", "--n-cycle",
    "--n-queries", "--n-keys",
    "--use-template", "--use-rna-msa",
    "--max-msa-depth", "--max-msa-rows",
    "--diffusion-attention-backend", "--trunk-single-attention-backend",
    "--structural-single-attention-backend",
    "--no-graph-jit",
    "--cp-devices", "--cp-layout",
    "--diffusion-chunk-size",
    "--triangle-mul-chunk-size", "--triangle-att-q-chunk-size",
    "--single-att-q-chunk-size", "--token-q-chunk-size",
    "--chunk-policy", "--trunk-dtype",
    "--include-raw",
    "--representations-dir", "--stop-after", "--representations",
    "--cpu-only",
    "--compile-cache",
    "--components-cif", "--ccd-rdkit-cache",
    "--template-mmcif-dir", "--template-release-dates",
    "--template-obsolete-map", "--kalign-binary",
)

_PORTS: tuple[tuple[str, Callable[..., Any], tuple[str, ...]], ...] = (
    ("protenix", protenix_predict.main, _PROTENIX_FLAG_ORDER),
    ("opendde", opendde_predict.main, _OPENDDE_FLAG_ORDER),
)


@pytest.mark.parametrize(("port", "main", "expected"), _PORTS)
def test_port_predict_flag_order_is_pinned(
    port: str,
    main: Callable[..., Any],
    expected: Sequence[str],
) -> None:
    parser = capture_parser(main)
    assert tuple(flag_order(parser)) == tuple(expected), port


@pytest.mark.parametrize(("port", "main", "_expected_order"), _PORTS)
def test_shared_predict_flags_are_declared_as_pinned(
    port: str,
    main: Callable[..., Any],
    _expected_order: Sequence[str],
) -> None:
    declared = _by_option_strings(capture_parser(main))
    missing = sorted(set(_SHARED_FLAG_SPECS) - set(declared))
    assert not missing, f"{port} no longer declares {missing}"
    for option_strings, expected in _SHARED_FLAG_SPECS.items():
        assert declared[option_strings] == expected, (port, option_strings)


def test_shared_predict_flags_match_between_the_two_ports() -> None:
    protenix = _by_option_strings(capture_parser(protenix_predict.main))
    opendde = _by_option_strings(capture_parser(opendde_predict.main))
    for option_strings in _SHARED_FLAG_SPECS:
        assert protenix[option_strings] == opendde[option_strings], option_strings
