"""FoldJAX command-line interface."""

from __future__ import annotations

import argparse
import dataclasses
import errno
import json
import os
import sys
import time
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any, NoReturn

from foldjax import (
    assets,
    cache_gc,
    doctor,
    manifest,
    memory_policy,
    oom,
    paths,
    progress,
    report,
    tools_cli,
)
from foldjax.api import _write_failures, predict_batch, preflight, resolve_requests
from foldjax.input import is_jobs_file
from foldjax.job import Job
from foldjax.redaction import public_options
from foldjax.registry import (
    available_models,
    capabilities,
    model_info,
    normalize_model_name,
)
from foldjax.schema import (
    MSA_PAIRINGS,
    MSA_POLICIES,
    PRESETS,
    STOP_POINTS,
    TEMPLATE_POLICIES,
    BatchReport,
    JobSource,
    PaddingConfig,
    PredictionError,
    PredictionFailure,
    PredictionRequest,
    PredictionResult,
    expand_input_directories,
)


def _templates_value(value: str) -> str | Path:
    """``--templates``: a policy, or a private folder of mmCIF files."""
    if value in TEMPLATE_POLICIES:
        return value
    path = Path(value).expanduser()
    if path.is_dir():
        return path
    raise argparse.ArgumentTypeError(
        f"{value!r} is neither one of {', '.join(TEMPLATE_POLICIES)} nor a "
        "directory of mmCIF files"
    )


def _add_predict_arguments(
    parser: argparse.ArgumentParser,
    *,
    allow_no_cache: bool = True,
    cache_warm: bool = False,
) -> None:
    source = parser.add_argument_group(
        "input", "what to fold, and where its alignments come from"
    )
    weights_group = parser.add_argument_group(
        "weights", "which checkpoint and managed profile to run"
    )
    output_group = parser.add_argument_group("output", "where results are written")
    sampling = parser.add_argument_group(
        "sampling", "seeds and the model-neutral schedule knobs"
    )
    shapes = parser.add_argument_group(
        "padding", "opt-in shape normalization for executable reuse"
    )
    execution = parser.add_argument_group(
        "execution", "device memory, the compile cache, and native options"
    )
    source.add_argument(
        "--model",
        required=True,
        nargs="+",
        help=", ".join(available_models()) + "; several run each in turn",
    )
    source.add_argument(
        "--input",
        type=Path,
        nargs="+",
        help="job JSON/YAML, FASTA, a .pdb/.mmcif deposition to re-fold, a "
        "directory of them, or model-native input such as an OpenFold3 feature "
        ".npz; several run every model on every input. Use structure:PATH to "
        "read a .cif for its chemistry rather than as a job document. Omit it "
        "and give --sequence instead",
    )
    source.add_argument(
        "--sequence",
        nargs="+",
        default=[],
        metavar="SEQ",
        help="protein sequence(s) to fold without writing a job file; chains are "
        "named A, B, ... in the order given",
    )
    source.add_argument(
        "--dna", nargs="+", default=[], metavar="SEQ", help="DNA chain(s)"
    )
    source.add_argument(
        "--rna", nargs="+", default=[], metavar="SEQ", help="RNA chain(s)"
    )
    source.add_argument(
        "--ligand",
        nargs="+",
        default=[],
        metavar="CCD",
        help="ligand CCD code(s) for a --sequence job, for example ATP",
    )
    source.add_argument(
        "--ligand-smiles",
        nargs="+",
        default=[],
        metavar="SMILES",
        help="ligand SMILES for a --sequence job. Separate from --ligand because "
        "'CCO' is both a plausible CCD code and ethanol",
    )
    source.add_argument(
        "--name",
        help="what to call a --sequence job: its output directory is "
        "foldjax-outputs/NAME. Omitted, the job is called 'job' and its "
        "directory is foldjax-outputs/job-<first 8 hex digits of the job's "
        "SHA-256>, the same for the same chains on every run",
    )
    source.add_argument(
        "--affinity-binder",
        metavar="CHAIN",
        help="predict the binding affinity of this chain of a --sequence job. "
        "Boltz-2 is the only carried model with that head; the others refuse it",
    )
    weights_group.add_argument(
        "--weights",
        type=Path,
        help="model-native checkpoint or asset directory; resolved from the "
        "FoldJAX weight store if omitted",
    )
    weights_group.add_argument(
        "--profile",
        help="model-specific managed profile; selects matching weights and "
        "model variant (for example Protenix mini-esm-v0.5.0)",
    )
    output_group.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "omit to discard the warm-up prediction; set it to retain the "
            "result (batches add <model>/<input stem>)"
            if cache_warm
            else "defaults to foldjax-outputs/<input stem> for one run; "
            "batches add <model>/<input stem>"
        ),
    )
    source.add_argument(
        "--input-format",
        default="auto",
        help="auto (default), foldjax, native, or a model's own dialect",
    )
    sampling.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "representative seed (default: the model's default seed)"
            if cache_warm
            else "prediction seed. Omitted, each model uses its upstream's: "
            "Protenix 101, OpenFold3 42; OpenDDE and AlphaFold 3 run a native "
            "job's modelSeeds. Otherwise (Boltz-2, ESMFold2, and a job "
            "without modelSeeds) upstream seeds nothing, so a seed is drawn, "
            "printed and recorded in foldjax_run.json. `foldjax plan` shows "
            "which"
        ),
    )
    sampling.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help=(
            "prediction seed list; cache warm executes only the first because "
            "seed values do not change the compiled program. Mutually "
            "exclusive with --seed"
            if cache_warm
            else "run the job once per seed and return every structure "
            "together: each seed's native files and run manifest go in "
            "<output>/seed_<n>, its structures in <output>/seed-<n>_sample-<NN> "
            "(NN counts from 00 within each seed), and <output>/foldjax_run.json "
            "lists them all. The samples from one seed are correlated, so this "
            "is the usual way to get independent predictions. Mutually "
            "exclusive with --seed"
        ),
    )
    sampling.add_argument(
        "--num-seeds",
        type=int,
        help=(
            "accepted for request parity; cache warm still executes only the "
            "first seed. Mutually exclusive with --seeds"
            if cache_warm
            else "how many seeds to run, counting up from --seed or, without "
            "it, from the model's default seed. --seed 0 --num-seeds 3 is "
            "--seeds 0 1 2; mutually exclusive with --seeds"
        ),
    )
    source.add_argument(
        "--msa",
        choices=MSA_POLICIES,
        default="none",
        help="what to do about a protein chain with no alignment: refuse the "
        "job (default 'none', as upstream Boltz-2 does; ESMFold2, whose "
        "upstream folds without one, is exempt), 'single' to fold it from "
        "the single sequence on purpose, 'auto' to search and cache an "
        "alignment, or 'required' to fail rather than fall back when the "
        "search finds none. auto and required SEND THE SEQUENCE to the "
        "public ColabFold MMseqs2 server (FOLDJAX_MSA_SERVER_URL points at "
        "your own instead)",
    )
    source.add_argument(
        "--msa-pairing",
        choices=MSA_PAIRINGS,
        default="model",
        help="with --msa auto/required: how the searched alignment pairs a "
        "complex. 'model' (default) is each model's own: OpenFold3, Boltz-2, "
        "Protenix and OpenDDE one ColabFold pairgreedy search over the complex, "
        "AlphaFold 3 a per-chain alignment, ESMFold2 none. 'greedy' and "
        "'complete' pair the complex in one search with that ColabFold "
        "strategy, for every model but AlphaFold 3; 'none' delivers "
        "no paired alignment and skips the per-chain pairing search. Part of "
        "the MSA cache key and the run manifest",
    )
    source.add_argument(
        "--templates",
        type=_templates_value,
        default="none",
        metavar="{none,auto,required,DIR}",
        help="structural templates for protein chains that name none: 'none' "
        "(default) uses only the job's own, 'auto' searches the ColabFold "
        "MMseqs2 server's PDB70 hits, fetches the structures from RCSB and "
        "applies the model's released template filters and date cutoff, "
        "'required' does the same and fails the run when the search cannot run "
        "or keeps nothing. Both SEND THE SEQUENCE to that server "
        "(FOLDJAX_MSA_SERVER_URL points at "
        "your own; FOLDJAX_TEMPLATE_COMMAND runs a local search). A directory "
        "searches that private folder of mmCIF files on this machine instead "
        "(mmseqs on PATH, else Kalign; refused when neither is installed), as "
        "'auto', with no release-date cutoff unless --template-max-date is "
        "given. Protenix and OpenDDE need --option use_template=true; ESMFold2 "
        "has no template input",
    )
    source.add_argument(
        "--template-max-date",
        metavar="YYYY-MM-DD",
        help="with --templates auto/required: keep only templates released by "
        "this date, instead of the model's released default (AlphaFold 3, "
        "Protenix, OpenDDE: 2021-09-30; OpenFold3 and Boltz-2: none)",
    )
    parser.add_argument(
        "--representations",
        help=(
            "hand back the trunk arrays as well: a comma-separated list, or "
            "'all'. `foldjax capabilities --model M` lists what each model "
            "produces. These are the largest arrays a run makes -- a pair "
            "representation is quadratic in token count -- so they are never "
            "written unless asked for."
        ),
    )
    parser.add_argument(
        "--stop-after",
        choices=STOP_POINTS,
        default="full",
        help=(
            "'inputs' stops after the native input representation; "
            "'trunk' stops once the trunk representations exist, skipping the "
            "diffusion sampler and the confidence heads. It writes no "
            "structure, so it only makes sense with --representations."
        ),
    )
    sampling.add_argument(
        "--preset",
        choices=PRESETS,
        help="'fast': the reduced steps and recycles the model's publisher "
        "documents for the checkpoint being run, recorded in the manifest. "
        "Published only for Protenix's Mini profiles (5 steps, 4 recycles); "
        "refused elsewhere with the reason",
    )
    sampling.add_argument(
        "--num-samples", type=int, help="how many structures to generate"
    )
    sampling.add_argument("--num-steps", type=int, help="diffusion steps per structure")
    sampling.add_argument(
        "--num-recycles",
        type=int,
        help="trunk recycling iterations; omit to keep the selected model's default "
        "(also with --padding and cache warm)",
    )
    sampling.add_argument(
        "--max-msa-depth",
        type=int,
        help="cap how many MSA rows the model keeps. The trunk holds a "
        "[depth, tokens, channels] representation, so this is the dominant "
        "memory knob: capping a 13k-row alignment to 1024 halved Protenix's "
        "peak at 488 tokens. Omit to keep each backend's own default",
    )
    shapes.add_argument(
        "--padding",
        action="store_true",
        help="select a token bucket with derived atom capacity, and pad the MSA "
        "axis to the bucket at or above the rows the unpadded run would process "
        "(a profile depth is a floor, never a cap); template limits stay at the "
        "native depth. Disabled by default so existing scientific results and "
        "exact-shape execution are unchanged",
    )
    shapes.add_argument(
        "--pad-tokens",
        type=int,
        help="pin the padded token size for this run (also enables padding)",
    )
    shapes.add_argument(
        "--pad-atoms",
        type=int,
        help="pin the padded atom size; unsupported models reject it early",
    )
    shapes.add_argument(
        "--pad-msa",
        type=int,
        help=(
            "pin the padded MSA row capacity; it pads the rows the model "
            "already reads and is refused below them, so --max-msa-depth "
            "stays the only way to read fewer. Those rows are known only "
            "after featurization, so `foldjax plan` does not check a pin "
            "against them; predict does"
        ),
    )
    shapes.add_argument(
        "--pad-templates",
        type=int,
        help="pin the padded template count for models with a template axis",
    )
    shapes.add_argument(
        "--pad-structural-tokens",
        type=int,
        help="pin OpenDDE's secondary structural-token axis",
    )
    shapes.add_argument(
        "--pad-language-model-tokens",
        type=int,
        help=(
            "pin the language-model sequence width for ESMFold2/ESMC or "
            "Protenix ESM/ISM"
        ),
    )
    shapes.add_argument(
        "--padding-overflow",
        choices=("error", "exact"),
        help="when an automatic axis exceeds the standard grid: fail before "
        "compilation (default) or keep that exact size",
    )
    execution.add_argument(
        "--mem-fraction",
        type=float,
        help="fraction of the device JAX may preallocate. Defaults to "
        f"{oom.PREDICT_MEM_FRACTION} rather than JAX's {oom.DEFAULT_MEM_FRACTION}, "
        "because one prediction owns the process and a quarter of the card held "
        "in reserve is what stops jobs that would otherwise fit. Lower it to "
        "share the device with another process",
    )
    execution.add_argument(
        "--memory-check",
        choices=memory_policy.CHECK_MODES,
        default=memory_policy.DEFAULT_CHECK_MODE,
        help="what to do when a model's fitted peak law says the run does not "
        "fit this device: refuse before the weights and the graph (default), "
        "or warn and let the allocator answer. Boltz-2, Protenix, OpenFold3, "
        "OpenDDE and ESMFold2 carry a law; AlphaFold 3 has none and answers "
        "'unknown'",
    )
    execution.add_argument(
        "--memory-budget-gib",
        type=float,
        help="plan against this much device memory rather than what the "
        "allocator reports; the smaller of the two is used. Useful for asking "
        "whether a job would fit a card you are not on",
    )
    execution.add_argument(
        "--cache-dir",
        type=Path,
        help=f"compile cache root (default {paths.compile_cache_dir()})",
    )
    if allow_no_cache:
        execution.add_argument(
            "--no-cache",
            action="store_true",
            help="skip the persistent compile cache (slower, but writes nothing)",
        )
    execution.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="native option passed straight to the backend; repeatable",
    )


def _model_help() -> str:
    """Every model name, with the aliases each one also answers to."""
    from foldjax.portspec import PORTS

    names = []
    for name in available_models():
        aliases = PORTS[name].aliases
        names.append(f"{name} ({', '.join(aliases)})" if aliases else name)
    return "one of " + ", ".join(names)


class _Parser(argparse.ArgumentParser):
    """argparse, with the one refusal whose fix it does not say."""

    def error(self, message: str) -> NoReturn:
        # `--name -dash` reads the value as an option; argparse only says the
        # value is missing. Subcommand parsers inherit this class.
        if message.startswith("argument --name: expected one argument"):
            message += (
                "; a name that starts with '-' is attached with '=': --name=-dash"
            )
        super().error(message)


def _parser() -> argparse.ArgumentParser:
    from foldjax import __version__

    model_help = _model_help()

    parser = _Parser(
        prog="foldjax",
        description="Biomolecular structure prediction in JAX: one job file in, "
        "structures and confidence out.",
    )
    # The first line of every bug report, and it used to be reachable only
    # inside `foldjax doctor`.
    parser.add_argument("--version", action="version", version=f"foldjax {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    models = commands.add_parser("models", help="list available model backends")
    models.add_argument(
        "--json",
        action="store_true",
        help="include weight readiness, supported inputs, and execution options",
    )
    models.add_argument(
        "--for",
        dest="for_input",
        type=Path,
        metavar="JOB",
        help="report which models can run this job and why the others cannot. "
        "Answered from the input translation table, so it needs no weights. A "
        "multi-job file is answered per job",
    )
    models.add_argument(
        "--msa",
        choices=MSA_POLICIES,
        default="none",
        help="with --for: the alignment policy predict would run under "
        "(default 'none', which refuses a protein chain with no alignment "
        "except on ESMFold2)",
    )
    models.add_argument(
        "--profile",
        help="with --for: the managed weight profile predict would run, for the "
        "models that offer it (the others answer for their released weights); "
        "a profile decides, for example, whether Protenix can read a pocket",
    )
    home = commands.add_parser("home", help="show where FoldJAX keeps its files")
    home.add_argument(
        "--path",
        choices=tuple(paths.describe()),
        help="print just one location, for shell scripts",
    )

    runtime = commands.add_parser(
        "runtime", help="inspect or prepare model-specific native runtime artifacts"
    )
    runtime_commands = runtime.add_subparsers(dest="runtime_command", required=True)
    runtime_status = runtime_commands.add_parser(
        "status", help="report whether one model can start without preparation"
    )
    runtime_status.add_argument("--model", required=True, help=model_help)
    runtime_prepare = runtime_commands.add_parser(
        "prepare", help="prepare one model's generated native artifacts"
    )
    runtime_prepare.add_argument("--model", required=True, help=model_help)
    runtime_gc = runtime_commands.add_parser(
        "gc", help="inspect or remove older prepared runtime generations"
    )
    runtime_gc.add_argument("--model", required=True, help=model_help)
    runtime_gc.add_argument(
        "--keep-days",
        type=float,
        default=7.0,
        help="keep trees prepared within this many days even when unreachable, "
        "because a checkout at another commit has its own live tree and this "
        "store is shared between them (default: 7)",
    )
    runtime_gc.add_argument(
        "--all",
        dest="gc_all",
        action="store_true",
        help="ignore --keep-days and remove every tree but the current one",
    )
    runtime_gc.add_argument(
        "--apply",
        action="store_true",
        help="remove the reported generations; the default is a dry run because "
        "another checkout may still select an older generation",
    )
    runtime_gc.add_argument(
        "--dry-run",
        dest="apply",
        action="store_false",
        help=argparse.SUPPRESS,
    )

    describe = commands.add_parser(
        "capabilities",
        help="show what one backend accepts, and its sampling defaults",
    )
    describe.add_argument("--model", required=True, help=model_help)
    describe.add_argument(
        "--json",
        action="store_true",
        help="print JSON (the only format; accepted so scripts can say so)",
    )

    run = commands.add_parser("predict", help="run one prediction")
    _add_predict_arguments(run)
    run.add_argument(
        "--json",
        action="store_true",
        help="print the machine-readable result instead of the summary table. "
        "A non-interactive stdout already gets JSON, so pipes are unchanged",
    )
    run.add_argument(
        "--quiet",
        action="store_true",
        help="do not report progress on stderr (FOLDJAX_PROGRESS=0 does the same)",
    )
    run.add_argument(
        "--resume",
        action="store_true",
        help="skip any model/input pair whose output directory already holds a "
        f"finished {manifest.MANIFEST_NAME}",
    )
    run.add_argument(
        "--keep-going",
        action="store_true",
        help="run the rest of a batch when one model/input pair fails, and exit "
        "3 if any did",
    )

    show = commands.add_parser(
        "show", help="summarize finished runs in an output directory"
    )
    show.add_argument("path", type=Path, help="an output directory, or one run's own")
    show.add_argument("--json", action="store_true", help="print the manifests")
    show.add_argument(
        "--format",
        choices=("table", "csv", "json"),
        default="table",
        help="table: per-run summary for reading (default). csv/json: one row "
        "per model/input/seed/sample, failures included, with the common "
        "summary, native scores, structure path and SHA-256 "
        "(foldjax.results_table)",
    )
    show.add_argument(
        "--aggregate",
        action="store_true",
        help="with --format csv/json: count, median and spread per "
        "(input, model, configuration) instead of one row per sample; never "
        "across models",
    )

    compare = commands.add_parser(
        "compare",
        help="pairwise RMSD (CA for proteins, C4' for nucleic acids) and "
        "coverage between every structure for each input",
        description="Align every structure of each input in a finished output "
        "directory to every other one (all models, seeds and samples) with "
        "foldjax.align_structures, and write the RMSD, coverage and the residue "
        "correspondence used to compare.json and compare.csv (pairs), and one row "
        "per structure to compare_structures.csv. Proteins are fitted "
        "on CA, nucleic acids on C4'; ligands are carried, not fitted. Cost "
        "grows with the square of the structure count.",
    )
    compare.add_argument("path", type=Path, help="a finished output directory")
    compare.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where to write compare.json, compare.csv and compare_structures.csv "
        "(default: PATH/compare)",
    )
    compare.add_argument(
        "--samples",
        choices=("all", "best"),
        default="all",
        help="all samples (default), or each run's within-model best only",
    )

    plan = commands.add_parser(
        "plan", help="show the resolved request without running it"
    )
    _add_predict_arguments(plan)

    setup = commands.add_parser(
        "setup", help="fetch the default public models and report opt-in/manual ones"
    )
    setup.add_argument(
        "--download-only",
        action="store_true",
        help="fetch the released files but skip the JAX conversions",
    )
    setup.add_argument(
        "--all",
        dest="fetch_all",
        action="store_true",
        help="also fetch the models held back for their size, so one command "
        "installs every published checkpoint. AlphaFold 3 and Protenix v2 are "
        "still yours to supply: their parameters are licensed, not merely large",
    )

    weights = commands.add_parser(
        "weights", help="download released checkpoints and convert them for JAX"
    )
    weights_commands = weights.add_subparsers(dest="weights_command", required=True)
    weights_commands.add_parser("list", help="what is downloaded and converted")
    fetch = weights_commands.add_parser(
        "fetch", help="download and convert one model's weights"
    )
    fetch.add_argument("--model", required=True, help=", ".join(assets.available()))
    fetch.add_argument(
        "--profile",
        help="managed asset profile (model-specific; inspect with "
        "foldjax models --json)",
    )
    fetch.add_argument(
        "--download-only",
        action="store_true",
        help="fetch the released files but skip the JAX conversion",
    )
    where = weights_commands.add_parser(
        "path", help="print the managed prediction-ready asset path"
    )
    where.add_argument("--model", required=True, help=model_help)
    where.add_argument(
        "--profile",
        help="managed asset profile (defaults to the complete released bundle)",
    )

    doctor = commands.add_parser(
        "doctor", help="check the install, the accelerator, and what is missing"
    )
    doctor.add_argument(
        "--json", action="store_true", help="machine-readable, for bug reports"
    )

    cache = commands.add_parser(
        "cache", help="warm or trim the persistent JAX compilation cache"
    )
    cache_commands = cache.add_subparsers(dest="cache_command", required=True)
    collect = cache_commands.add_parser(
        "gc",
        help="remove old or corrupt compile-cache entries",
        description="Report what could be reclaimed from the compilation cache, "
        "and remove it only when --apply is given. Entries are keyed by "
        "accelerator, runtime, weights and shapes, so a deleted one costs one "
        "recompile and never a wrong result.",
    )
    collect.add_argument(
        "--older-than",
        type=int,
        metavar="DAYS",
        help="consider entries last used more than DAYS ago",
    )
    collect.add_argument(
        "--max-size",
        metavar="SIZE",
        help="keep the newest entries within SIZE (for example 20G, 500M)",
    )
    collect.add_argument(
        "--verify",
        action="store_true",
        help="decompress every JAX entry and select the ones that do not decode "
        "(a write cut short by a full disk or a kill). JAX only warns about such "
        "an entry and never replaces it, so every run recompiles that program. "
        "Entries written in the last 10 minutes are skipped: one may still be "
        "being written",
    )
    collect.add_argument(
        "--apply",
        action="store_true",
        help="actually delete. Without it this only reports",
    )
    warm = cache_commands.add_parser(
        "warm",
        help="execute a representative job once to populate its exact cache",
        description="Execute the same model path used by prediction once, including "
        "GPU kernel autotuning, and populate the persistent JAX cache. Prediction "
        "files are discarded unless --output-dir is supplied. Cache entries are "
        "specific to the accelerator, JAX runtime, input shapes, weights, profile, "
        "and static options.",
    )
    _add_predict_arguments(warm, allow_no_cache=False, cache_warm=True)
    tools_cli.register(commands, show=show, compare=compare, run=run, plan=plan)

    return parser


def _options(items: list[str]) -> dict[str, Any]:
    options = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"option must be KEY=VALUE: {item}")
        raw_key, value = item.split("=", 1)
        key = raw_key.strip()
        if not key:
            raise ValueError("option key must be non-empty")
        if key in options:
            raise ValueError(f"option {key!r} was set more than once")
        try:
            options[key] = json.loads(value)
        except json.JSONDecodeError:
            options[key] = value
    return options


def _memory_options(
    args: argparse.Namespace, options: dict[str, Any]
) -> dict[str, Any]:
    """Fold the two memory flags into the backend options, if they were given.

    Only when given. ``request.options`` is part of the identity a finished
    manifest is matched against, so injecting the default unconditionally would
    make every run already on disk stop matching and `--resume` redo it. The
    ports carry the same default themselves, which is what makes the omission
    and the spelling the same run.

    A flag and an ``--option`` of the same name are rejected rather than
    silently ordered, the way the neutral sampling knobs are.
    """
    given = {
        "memory_check": (
            None
            if getattr(args, "memory_check", memory_policy.DEFAULT_CHECK_MODE)
            == memory_policy.DEFAULT_CHECK_MODE
            else args.memory_check
        ),
        "memory_budget_gib": getattr(args, "memory_budget_gib", None),
    }
    for key, value in given.items():
        if value is None:
            continue
        if key in options:
            raise ValueError(
                f"--{key.replace('_', '-')} and --option {key}= set the same "
                "thing; pass one of them"
            )
        options[key] = value
    return options


#: Extensions read as FASTA. Converted to a common-schema job file before the
#: request is built, so `plan`, the manifest and the input digest all describe
#: the document the model actually saw.
_FASTA_SUFFIXES = frozenset({".fasta", ".fa", ".faa", ".fas", ".fna", ".mpfa"})

#: Deposited structures, read for their chemistry and never their coordinates.
#: `.cif` is deliberately absent: a FoldJAX or native job document may also be
#: `.cif`-adjacent in a workflow, and more importantly `--input x.cif` is
#: ambiguous between "fold this sequence again" and "this is native input".
#: `.pdb` and `.mmcif` are unambiguous; `.cif` is accepted only through the
#: explicit `structure:` prefix.
_STRUCTURE_SUFFIXES = frozenset({".pdb", ".ent", ".mmcif"})

#: What a directory of jobs may contain. A directory is expanded rather than
#: passed through: every backend reads files, and globbing in the shell drops
#: the sort order that makes a batch's output directories predictable.
_JOB_SUFFIXES = (
    frozenset({".json", ".yaml", ".yml"}) | _FASTA_SUFFIXES | _STRUCTURE_SUFFIXES
)


def _resolve_inputs(
    args: argparse.Namespace,
    *,
    jobs_root: Path | None = None,
    sources: dict[Path, JobSource] | None = None,
    unreadable: list[tuple[Path, Exception]] | None = None,
) -> list[Path]:
    """Turn everything the CLI accepts as input into common job/native files.

    Generated job documents go to ``jobs_root``, or to the store's
    ``runtime/jobs`` when it is None. ``sources`` receives, per generated job
    document, the FASTA or structure file it was written from. With
    ``unreadable`` given, a file that cannot be converted is listed there
    instead of ending the command -- `--keep-going` over a directory with one
    empty FASTA file in it.
    """
    sequences = bool(args.sequence or args.dna or args.rna)
    ligands = bool(args.ligand or args.ligand_smiles)
    if sequences or ligands:
        if args.input:
            raise ValueError(
                "--input and --sequence are alternatives; pass one of them"
            )
        if not sequences:
            raise ValueError("a ligand needs a --sequence to bind to")
        job = Job.from_sequences(
            args.sequence,
            dna=args.dna,
            rna=args.rna,
            ligand_ccd=args.ligand,
            ligand_smiles=args.ligand_smiles,
            name=args.name or "job",
            affinity_binder=args.affinity_binder,
        )
        return [job.store(jobs_root, stem=_sequence_stem(job, args.name))]
    if not args.input:
        raise ValueError("one of --input and --sequence is required")
    if args.name:
        raise ValueError("--name applies to a --sequence job; a job file names itself")
    if args.affinity_binder:
        raise ValueError(
            "--affinity-binder applies to a --sequence job; a job file says so "
            "with its own properties field"
        )

    selected: list[Path] = []
    for path in args.input:
        # `structure:` says "take the chemistry out of this file", which is the
        # one thing an extension cannot say for `.cif`: that suffix is also a
        # perfectly good name for a job document in a workflow directory. It is
        # not a path, so the directory expansion below leaves it alone.
        text = str(path)
        if text.startswith("structure:"):
            selected.append(Path(f"structure:{Path(text[10:]).resolve()}"))
            continue
        selected.append(path)
    # The same expansion a `PredictionRequest` performs, over the wider set the
    # CLI accepts: it can turn a FASTA or a deposited structure into a job
    # document, which a request cannot.
    expanded = expand_input_directories(selected, suffixes=_JOB_SUFFIXES)
    converted: list[Path] = []
    for path in expanded:
        try:
            job_file, source = _as_job_file(path, jobs_root=jobs_root)
        except Exception as error:
            if unreadable is None:
                raise
            unreadable.append((path, error))
            continue
        if source is not None and sources is not None:
            sources[job_file] = source
        converted.append(job_file)
    return converted


def _sequence_stem(job: Job, name: str | None) -> str:
    """The output stem of a ``--sequence`` job: its name, else job-<digest>.

    Stable by construction: the same chains always land in the same
    ``foldjax-outputs/<stem>``, whatever was run before, so ``--resume`` finds
    them; two different unnamed jobs never share a directory.
    ``<digest>`` is the first 8 hex digits of the SHA-256 of the job document
    (``json.dumps(job.to_document(), sort_keys=True)``).
    """
    if name:
        return name
    import hashlib

    document = json.dumps(job.to_document(), sort_keys=True)
    return f"job-{hashlib.sha256(document.encode()).hexdigest()[:8]}"


def _as_job_file(
    path: Path, *, jobs_root: Path | None = None
) -> tuple[Path, JobSource | None]:
    """Turn one accepted input into a file a request can carry.

    FASTA and deposited structures become ordinary common-schema documents,
    returned with the file they came from; everything else is already one, or
    is a model's own dialect, and passes through untouched.
    """
    text = str(path)
    if text.startswith("structure:"):
        original, kind = Path(text[10:]), "structure"
        job = Job.from_structure(original)
    elif path.suffix.lower() in _FASTA_SUFFIXES:
        original, kind = path, "fasta"
        job = Job.from_fasta(path)
    elif path.suffix.lower() in _STRUCTURE_SUFFIXES:
        original, kind = path, "structure"
        job = Job.from_structure(path)
    else:
        return path, None
    source = JobSource(path=original, index=0, name=job.name, kind=kind)
    return job.store(jobs_root), source


def _request(
    args: argparse.Namespace,
    *,
    jobs_root: Path | None = None,
    sources: dict[Path, JobSource] | None = None,
    unreadable: list[tuple[Path, Exception]] | None = None,
) -> PredictionRequest | None:
    """The request ``args`` describe; None only when every input is unreadable.

    ``sources`` and ``unreadable`` are :func:`_resolve_inputs`'s.
    """
    sources = {} if sources is None else sources
    inputs = _resolve_inputs(
        args, jobs_root=jobs_root, sources=sources, unreadable=unreadable
    )
    if not inputs:
        return None
    single_model = len(args.model) == 1
    # A multi-job file is several runs, like a directory, so it takes the
    # plural spelling even alone; the request expands it into its jobs. An
    # input that could not be converted still counts: which layout a batch
    # writes must not depend on how many of its files were readable.
    single_input = len(inputs) + len(unreadable or ()) == 1 and not (
        args.input_format in ("auto", "foldjax") and is_jobs_file(inputs[0])
    )
    single_input = single_input and not getattr(args, "_plural_inputs", False)
    if args.seed is not None and args.seeds:
        raise ValueError("--seed and --seeds are mutually exclusive")
    padding_values = {
        "tokens": args.pad_tokens,
        "atoms": args.pad_atoms,
        "msa": args.pad_msa,
        "templates": args.pad_templates,
        "structural_tokens": args.pad_structural_tokens,
        "language_model_tokens": args.pad_language_model_tokens,
    }
    padding_requested = args.padding or any(
        value is not None for value in padding_values.values()
    )
    if args.padding_overflow is not None and not padding_requested:
        raise ValueError("--padding-overflow requires --padding or a --pad-* target")
    padding = (
        PaddingConfig(
            **padding_values,
            overflow=args.padding_overflow or "error",
        )
        if padding_requested
        else None
    )
    templates = getattr(args, "templates", "none")
    return PredictionRequest(
        model=args.model[0] if single_model else None,
        models=None if single_model else tuple(args.model),
        input=inputs[0] if single_input else None,
        inputs=None if single_input else tuple(inputs),
        weights=args.weights,
        profile=args.profile,
        output_dir=args.output_dir,
        input_format=args.input_format,
        seed=args.seed,
        seeds=tuple(args.seeds) if args.seeds else None,
        num_seeds=args.num_seeds,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        num_recycles=args.num_recycles,
        max_msa_depth=args.max_msa_depth,
        cache_dir=args.cache_dir,
        use_compile_cache=not getattr(args, "no_cache", False),
        options=_memory_options(args, _options(args.option)),
        padding=padding,
        msa=args.msa,
        msa_pairing=getattr(args, "msa_pairing", "model"),
        templates="auto" if isinstance(templates, Path) else templates,
        template_dir=templates if isinstance(templates, Path) else None,
        template_max_date=getattr(args, "template_max_date", None),
        preset=getattr(args, "preset", None),
        representations=getattr(args, "representations", None),
        stop_after=getattr(args, "stop_after", "full"),
        resume=getattr(args, "resume", False),
        on_error="continue" if getattr(args, "keep_going", False) else "stop",
        source=sources.get(inputs[0]) if single_input else None,
    )


def _report(name: str, done: int, total: int | None) -> None:
    if total:
        share = f"{100 * done / total:5.1f}%  {done / 1e6:8.1f} / {total / 1e6:.1f} MB"
    else:
        share = f"{done / 1e6:8.1f} MB"
    print(f"\r  {name:<34s} {share}", end="", file=sys.stderr, flush=True)


def _format_bytes(size: int) -> str:
    """Format event sizes compactly without turning small files into 0.00 GB."""

    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            digits = 0 if unit == "B" else 1
            return f"{value:.{digits}f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


class _WeightReporter:
    """Render structured asset events without mixing them with stdout results."""

    def __init__(self) -> None:
        self._progress_active = False
        self._line_progress = os.environ.get("FOLDJAX_PROGRESS_MODE") == "lines"
        self._last_line_bytes: dict[str, int] = {}
        self._last_line_time: dict[str, float] = {}

    def progress(self, name: str, done: int, total: int | None) -> None:
        if sys.stderr.isatty():
            _report(name, done, total)
            self._progress_active = True
        elif self._line_progress:
            now = time.monotonic()
            previous_bytes = self._last_line_bytes.get(name, 0)
            previous_time = self._last_line_time.get(name, 0.0)
            byte_step = max(16 * 1024**2, (total or 0) // 100)
            if (
                done == total
                or done - previous_bytes >= byte_step
                or now - previous_time >= 2.0
            ):
                total_text = "" if total is None else str(total)
                print(
                    f"[foldjax-progress]\t{name}\t{done}\t{total_text}",
                    file=sys.stderr,
                    flush=True,
                )
                self._last_line_bytes[name] = done
                self._last_line_time[name] = now
        # Non-interactive logs use the structured start/done events. Emitting
        # byte callbacks too would duplicate every completed download line.

    def event(self, event: assets.AssetEvent) -> None:
        self.finish_progress()
        item = f"  {event.item}" if event.item else ""
        elapsed = (
            "" if event.elapsed_seconds is None else f"  {event.elapsed_seconds:.2f}s"
        )
        size = "" if event.bytes is None else f"  {_format_bytes(event.bytes)}"
        print(
            f"[weights] {event.model}/{event.profile} "
            f"{event.action} {event.status}{item}: {event.message}{size}{elapsed}",
            file=sys.stderr,
        )

    def finish_progress(self) -> None:
        if self._progress_active:
            print(file=sys.stderr)
            self._progress_active = False


def _template_report() -> list[str]:
    """What the template modality still needs, in the order it needs it.

    Two searches exist and they are configured apart. `--templates auto` /
    `required` is FoldJAX's (`foldjax.template_search`): its variables are
    `FOLDJAX_TEMPLATE_*` and its aligner the `kalign` Python module, probed
    with `find_spec` exactly as the search probes it. Protenix and OpenDDE
    also carry upstream's native search, configured through their own options
    (`template_mmcif_dir`, `kalign_binary`); those lines are labelled
    `protenix-native` so neither is read as the other.
    """
    from foldjax.doctor import install_command
    from foldjax.models.protenix.data.search import templates
    from foldjax.paths import assets_dir
    from foldjax.template_search import (
        TEMPLATE_ENVIRONMENT,
        template_search_backend,
    )

    lines = []
    search = template_search_backend()
    hits = search["hits"]
    lines.append(
        "search        "
        + (
            "local  " + " ".join(hits["command"])
            if hits["kind"] == "local"
            else f"remote {hits['host']}  (--templates auto; sequences leave "
            "this machine)"
        )
    )
    structures = search["structures"]
    lines.append(
        f"structures    {structures['local_dir'] or 'no local mirror'}, then "
        f"{structures['url'] or 'no download'}"
    )
    lines.append(f"realignment   {search['aligner']}")
    if search["aligner"] != "kalign-python":
        lines.append(f"              {install_command('templates')}")
    for name in TEMPLATE_ENVIRONMENT:
        value = os.environ.get(name)
        lines.append(f"  {name} " + ("unset" if value is None else repr(value)))

    native = "protenix-native"
    metadata = ["release_date_cache.json", "obsolete_to_successor.json"]
    missing = [name for name in metadata if not (assets_dir() / name).is_file()]
    lines.append(
        f"{native} metadata  missing: " + ", ".join(missing)
        if missing
        else f"{native} metadata  ready"
    )
    try:
        binary = templates._resolve_kalign_binary(None)
        lines.append(f"{native} kalign    ready  {binary}")
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        lines.append(f"{native} kalign    {error}")
    lines.append(
        f"{native} mmCIF     --option template_mmcif_dir=DIR (flat or "
        "PDB-divided, .cif or .cif.gz)"
    )
    return lines


def _run_setup(args: argparse.Namespace) -> int:
    """Fetch default assets and say exactly what opt-in/manual models need."""
    print(f"store: {paths.foldjax_home()}\n")
    print("weights")
    failed = False
    # Profiles, not just models. A model's alternative checkpoints are separate
    # bundles with their own `in_default_setup`, and Protenix's v2 is one: the
    # release and v2 are both supported, so iterating models alone would
    # silently skip the newer of the two -- here, its bring-your-own notice.
    targets: list[tuple[str, str | None]] = []
    for name in assets.available():
        for profile in assets.available_profiles(name):
            targets.append(
                (name, None if profile == assets.RELEASED_PROFILE else profile)
            )

    for name, profile in targets:
        spec = assets.assets_for(name, profile=profile)
        label = name if profile is None else f"{name}/{profile}"
        if not spec.in_default_setup and not args.fetch_all:
            state = "ready" if spec.ready() else "opt-in"
            print(f"  {label:<11s} {state}")
            if not spec.ready():
                suffix = "" if profile is None else f" --profile {profile}"
                print(f"    fetch: foldjax weights fetch --model {name}{suffix}")
                print(f"    {spec.notes}")
                print("    or run `foldjax setup --all` to take it with the rest")
            continue
        if not spec.downloads or (
            assets.missing_supplied(spec) and not spec.ready()
        ):
            # Gated or non-redistributable: the instruction differs per model,
            # so `notes` is the only honest text. A profile with a user-supplied
            # checkpoint stays here until the file is placed, and is then
            # converted by the ordinary fetch below.
            state = "ready" if spec.ready() else "manual"
            print(f"  {label:<11s} {state}")
            if not spec.ready():
                print(f"    {spec.notes}")
                print(f"    goes in: {assets.weights_dir(spec.model)}")
            continue
        try:
            reporter = _WeightReporter()
            result = assets.fetch(
                name,
                profile=profile,
                on_progress=reporter.progress,
                on_event=reporter.event,
                convert=not args.download_only,
            )
        except (RuntimeError, OSError, ValueError) as error:
            reporter.finish_progress()
            print(f"  {label:<11s} failed: {error}", file=sys.stderr)
            failed = True
            continue
        reporter.finish_progress()
        print(f"\r  {label:<11s} ready  {result}" + " " * 20)

    runtime = model_info("alphafold3").runtime
    state = "ready" if runtime.ready else "not ready"
    print("\nruntime")
    print(f"  {'alphafold3':<11s} {state}")
    if runtime.setup is not None:
        label = (
            "prepare"
            if runtime.setup.startswith("foldjax runtime prepare")
            else "action"
        )
        print(f"    {label}: {runtime.setup}")
    print(f"    {runtime.notes}")

    print("\nmsa           ready  remote MMseqs2 (ColabFold); no local database")
    print("\ntemplates")
    for line in _template_report():
        print(f"  {line}")
    return 1 if failed else 0


def _run_runtime_gc(args: argparse.Namespace) -> int:
    """Reclaim runtime trees left behind by earlier source or ABI generations.

    Every vendored-source edit and every interpreter move mints a new tree and
    abandons the old one in place, and each is around a gigabyte of chemistry
    rather than of code. Nothing collected them, so they accumulate for as long
    as the store lives.
    """
    if args.model != "alphafold3":
        raise ValueError(f"{args.model} has no FoldJAX-managed runtime generations")

    from foldjax.models.alphafold3 import build

    keep_days = None if args.gc_all else args.keep_days
    stale = build.stale_generations(keep_days=keep_days)
    # Counted before anything is removed. Asking again afterwards subtracts the
    # trees this run just deleted and reports the held-back count as far too
    # small -- it said one where six were held.
    older = len(build.stale_generations(keep_days=None))
    kept = [path for path in build.generations() if path not in stale]
    if not stale:
        print(f"nothing to remove; {len(kept)} generation(s) kept")
        if older and not args.gc_all:
            print(
                f"{older} older generation(s) kept as recent; "
                "--all removes them too"
            )
        return 0

    total = 0
    for path in stale:
        size = build.generation_bytes(path)
        total += size
        verb = "removed" if args.apply else "would remove"
        if args.apply:
            build.remove_generation(path)
        print(f"{verb}  {path.name}  {size / 2**30:.1f} GiB")
    print(f"{total / 2**30:.1f} GiB total; {len(kept)} generation(s) kept")
    if not args.gc_all:
        held = older - len(stale)
        if held:
            print(
                f"{held} older generation(s) kept as recent; "
                "--all removes them too"
            )
    if not args.apply:
        print("dry run; pass --apply to remove the reported generations")
    return 0


def _models_for_profiles(
    infos: Mapping[str, Any], profile: str | None
) -> dict[str, str | None]:
    """The profile each model answers ``models --for`` with; None is released."""
    if profile is None or profile == assets.RELEASED_PROFILE:
        return dict.fromkeys(infos)
    chosen = {
        name: profile
        if any(row["profile"] == profile for row in info.weight_profiles)
        else None
        for name, info in infos.items()
    }
    if not any(chosen.values()):
        raise ValueError(
            f"no model offers the weight profile {profile!r}; "
            "`foldjax models --json` lists each model's profiles"
        )
    return chosen


def _run_models_for(args: argparse.Namespace) -> int:
    """Say which models can run one job, before anything is downloaded.

    Whether a backend can express a document is knowable from the input layer
    alone, so this answers without weights, without a GPU, and without the
    fifteen minutes it takes to discover the same thing by running the job.
    """
    from foldjax.input import (
        _absolute_job_paths,
        compatibility,
        is_jobs_document,
        read_job_document,
        read_jobs_file,
    )
    from foldjax.registry import get_backend

    path = Path(args.for_input)
    if path.suffix.lower() in _FASTA_SUFFIXES:
        document = Job.from_fasta(path).to_document()
    else:
        document = read_job_document(path)
    # One row per job and model: a multi-job file is several questions.
    if is_jobs_document(document):
        base = path.parent.absolute()
        jobs = [
            (name, _absolute_job_paths(job, base)) for name, job in read_jobs_file(path)
        ]
    else:
        jobs = [(None, document)]
    rows = []
    infos = {name: model_info(name) for name in available_models()}
    profiles = _models_for_profiles(infos, getattr(args, "profile", None))
    for job_name, job in jobs:
        for name, info in infos.items():
            profile = profiles[name]
            reason = compatibility(job, name, msa=args.msa, base=path.parent)
            if reason is None and isinstance(job, dict):
                reason = get_backend(name).profile_refusal(job, profile)
            if profile is None:
                ready, setup = info.weights_ready, info.setup
            else:
                ready = assets.assets_for(name, profile=profile).ready()
                setup = (
                    None
                    if ready
                    else f"foldjax weights fetch --model {name} --profile {profile}"
                )
            row = {
                "model": name,
                "runs": reason is None,
                "reason": reason,
                "weights_ready": ready,
                "setup": setup,
            }
            if profile is not None:
                row["profile"] = profile
            if job_name is not None:
                row["job"] = job_name
            rows.append(row)
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    for index, (job_name, _job) in enumerate(jobs):
        if job_name is not None:
            print(("\n" if index else "") + f"job {job_name}")
        print(f"{'model':<11s}{'runs?':<7s}why")
        for row in rows:
            if row.get("job") != job_name:
                continue
            if not row["runs"]:
                note = row["reason"]
            elif row["weights_ready"]:
                note = ""
            else:
                note = f"weights not installed: {row['setup']}"
            print(f"{row['model']:<11s}{'yes' if row['runs'] else 'no':<7s}{note}")
    return 0


def _runtime_payload(name: str) -> dict[str, Any]:
    info = model_info(name)
    return {"model": info.model, **info.runtime.summary()}


def _run_runtime(args: argparse.Namespace) -> int:
    """Inspect or explicitly prepare runtime artifacts without hidden work."""
    if args.runtime_command == "gc":
        return _run_runtime_gc(args)
    payload = _runtime_payload(args.model)
    if args.runtime_command == "status" or payload["ready"]:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if payload["model"] != "alphafold3":
        raise ValueError(
            f"{payload['model']} has no FoldJAX-managed runtime preparation step"
        )

    from foldjax.models.alphafold3 import build

    blocker = build.runtime_blocker()
    if blocker is not None:
        raise PredictionError(blocker)
    print(
        "preparing AlphaFold 3 runtime; this may compile and download sources...",
        file=sys.stderr,
    )
    build.ensure_ready()
    prepared = _runtime_payload(args.model)
    if not prepared["ready"]:
        raise PredictionError(
            "AlphaFold 3 runtime preparation finished without a usable runtime"
        )
    print(json.dumps(prepared, indent=2, sort_keys=True))
    return 0


def _run_weights(args: argparse.Namespace) -> int:
    if args.weights_command == "list":
        for row in assets.status():
            mark = "ready" if row["converted"] else "     "
            print(
                f"{mark}  {row['model']:<9s} downloaded {row['downloaded']}  "
                f"{row['licence']}\n         {row['path']}"
            )
            profiles = assets.profile_status(str(row["model"]))
            if len(profiles) > 1:
                for profile in profiles:
                    state = "ready" if profile["ready"] else "missing"
                    size = profile["download_bytes"]
                    size_text = "unknown size" if size is None else f"{size} bytes"
                    supplied = str(profile.get("supplied", "0/0"))
                    supplied_text = (
                        "" if supplied.endswith("/0") else f", supplied {supplied}"
                    )
                    print(
                        f"         profile {profile['profile']}: {state}, "
                        f"downloaded {profile['downloaded']}{supplied_text}, "
                        f"{size_text}"
                    )
        return 0
    if args.weights_command == "path":
        print(assets.resolve_weights(args.model, profile=args.profile))
        return 0

    spec = assets.assets_for(args.model, profile=args.profile)
    public_model, profile = assets.public_target(spec, args.profile)
    print(f"{public_model}: {len(spec.downloads)} file(s) from {spec.source}")
    for item in spec.supplied:
        print(f"  plus {item.name}, supplied by you: {item.target(spec.model)}")
    print(f"licence: {spec.licence}")
    reporter = _WeightReporter()
    try:
        result = assets.fetch(
            public_model,
            profile=profile,
            on_progress=reporter.progress,
            on_event=reporter.event,
            convert=not args.download_only,
        )
    except (RuntimeError, OSError, ValueError) as error:
        # A missing or unfetchable asset is an ordinary outcome of this
        # command -- most often a model whose publisher releases the weights
        # only to applicants. It should read as an instruction, not as a
        # FoldJAX stack trace.
        reporter.finish_progress()
        print(str(error), file=sys.stderr)
        return 1
    reporter.finish_progress()
    state = "downloaded" if args.download_only else "ready"
    print(f"{state}: {result}")
    return 0


def _run_cache(args: argparse.Namespace) -> int:
    """Warm an exact persistent-cache profile and report what changed."""

    from foldjax.warmup import warm_cache

    request = _request(args)
    print(
        "[cache] warm uses execute_once: the model runs one representative seed; "
        "prediction files are discarded unless --output-dir is supplied.",
        file=sys.stderr,
    )
    started = resolve_requests(request)
    for item in started:
        backend = model_info(item.model).model
        print(
            f"[cache] {backend}: input={item.input} weights={item.weights} "
            f"root={item.cache_dir}",
            file=sys.stderr,
        )
    # Native runners are allowed to print human progress (OpenDDE reports its
    # output path, for example). Keep that useful text visible, but never let
    # it corrupt the machine-readable cache report on stdout.
    with _stdout_to_stderr():
        result = warm_cache(request)
    results = result if isinstance(result, tuple) else (result,)
    for item in results:
        peak = (
            "unknown"
            if item.peak_device_bytes is None
            else _format_bytes(item.peak_device_bytes)
        )
        print(
            f"[cache] {item.model}: {item.status}; namespace={item.cache_dir}; "
            f"new_files={item.new_files}; new_bytes={item.new_bytes}; "
            f"elapsed={item.seconds:.2f}s; peak_device={peak}",
            file=sys.stderr,
        )
    payload = (
        [item.summary() for item in result]
        if isinstance(result, tuple)
        else result.summary()
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _validate_mem_fraction(requested: float | None) -> None:
    """Validate the allocator option without changing process state."""
    if requested is not None and not 0.0 < requested <= 1.0:
        raise ValueError(f"--mem-fraction must be in (0, 1]; got {requested}")


def _apply_mem_fraction(requested: float | None) -> None:
    """Validate and set the pool fraction before anything imports JAX.

    Only here, and only for the CLI: this is a process-wide setting, and a
    library that changed it on import would be deciding for a host application
    that may be sharing the device. An explicit environment variable always wins
    -- someone who set it has a reason.
    """
    _validate_mem_fraction(requested)
    if requested is not None:
        oom.set_mem_fraction(requested, override=True)
    else:
        oom.set_mem_fraction(oom.PREDICT_MEM_FRACTION)


#: Collector thresholds for a process that owns one prediction run.
PREDICT_GC_THRESHOLD = (100_000, 50, 100)


def _apply_gc_threshold() -> None:
    """Collect cycles rarely in a prediction process, and only there.

    CPython's default (700, 10, 10) runs a full collection every 70,000 net
    allocations, and tracing a model graph allocates millions of short-lived
    objects while tens of millions of long-lived ones (jaxprs, MLIR, the
    parameter trees) sit in the oldest generation: each full pass walked
    them for ~0.3 s, 10-11 times per warm 254-token process -- 3.3-4.3 s on
    Boltz-2, Protenix and OpenFold3 (fixed-cost job 2333). These thresholds
    still collect cycles, only rarely, and left peak RSS where it was.

    Same restriction as `_apply_mem_fraction`: the collector is process-wide,
    so a library that retuned it on import would be deciding for its host.
    """
    import gc

    gc.set_threshold(*PREDICT_GC_THRESHOLD)


def _skip_release_reclaim() -> None:
    """Drop chemistry caches at a session end without the collect and trim.

    The library follows each release of a loaded cache with a full
    ``gc.collect()`` and ``malloc_trim``, handing the pages back to the
    operating system. A prediction process exits after its batch, or reuses
    that freed memory for the next model in it, so the pass is pure cost here:
    0.33 s of full collection in a warm 254-token ESMFold2 process (fixed-cost
    job 2391). Same restriction as `_apply_gc_threshold`.
    """
    from foldjax.models import _managed_memory

    _managed_memory.set_reclaim_at_release(False)


def _requested_cp_devices(args: argparse.Namespace) -> int:
    """How many devices this invocation asked context parallelism for.

    Read from the raw ``--option`` strings rather than from a resolved request,
    because the answer is needed before JAX is imported and resolving a request
    imports the backend. A malformed option is left at 1 and reported by the
    request build, which is where option errors are phrased.
    """
    try:
        options = _options(list(getattr(args, "option", None) or []))
    except ValueError:
        return 1
    try:
        return int(options.get("cp_devices", 1))
    except (TypeError, ValueError):
        return 1


def _apply_rendezvous_timeout(args: argparse.Namespace) -> None:
    """Bound XLA's collective rendezvous before anything imports JAX.

    Same argument as `_apply_mem_fraction`, and the same restriction to the
    CLI: this is process-wide, and a library that set it on import would be
    deciding for a host application. It has to happen here because XLA parses
    ``XLA_FLAGS`` when the backend initialises -- by the time
    `models/_cp.context_parallel` runs, a CLI prediction has long had one --
    and without it one device's OOM leaves the rest waiting forever
    (`foldjax.oom.CP_RENDEZVOUS_SECONDS`). A value the caller already set for
    the flag wins.
    """
    if _requested_cp_devices(args) > 1 and oom.gpu_is_possible():
        oom.set_rendezvous_timeout()


def _plan_summary(
    request: PredictionRequest, *, scratch: Path | None = None
) -> dict[str, Any]:
    from foldjax.msa_search import resolve_pairing
    from foldjax.presets import preset_record

    def shown(path: Path) -> str:
        # Written to scratch, not the store: the path predict will give it.
        # The store layout is content-keyed, so it is known without writing.
        if scratch is not None and Path(path).is_relative_to(scratch):
            return str(paths.runtime_dir("jobs") / Path(path).relative_to(scratch))
        return str(path)

    generated = None
    if scratch is not None and Path(request.input).is_relative_to(scratch):
        from foldjax.input import read_job_document

        generated = read_job_document(Path(request.input))
    source = request.source.summary() if request.source is not None else None
    if source is not None:
        source["path"] = shown(request.source.path)
    summary = {
        "model": request.model,
        "input": shown(request.input),
        "generated_input": generated,
        # The multi-job file and job this generated input came from.
        "source": source,
        "input_format": request.input_format,
        "weights": str(request.weights),
        "profile": request.profile,
        "output_dir": str(request.output_dir),
        "cache_dir": str(request.cache_dir) if request.cache_dir is not None else None,
        # None while the seed is still to be drawn: a plan does not draw one,
        # because the run would draw another. `seed_source` says which.
        "seeds": (
            None
            if request.seed is None and request.seeds is None
            else list(request.resolved_seeds)
        ),
        "seed_source": request.seed_source,
        "msa": request.msa,
        "msa_pairing": (
            resolve_pairing(request.model, request.msa_pairing)
            if request.msa in ("auto", "required")
            else None
        ),
        "templates": request.templates,
        "template_dir": (
            str(request.template_dir) if request.template_dir is not None else None
        ),
        "template_max_date": request.template_max_date,
        "preset": preset_record(request),
        "options": public_options(request.options),
        **_effective_sampling(request),
    }
    if request.padding is not None:
        summary["padding"] = request.padding.summary()
        bucket = _planned_token_bucket(request, generated)
        if bucket is not None:
            summary["padding_estimate"] = bucket
        # Plan refuses what predict refuses up to featurization; the MSA rows a
        # model stores are known only after it, so a pin below them is not.
        summary["not_checked"] = [
            "padding.msa against the MSA rows the model stores (known only "
            "after featurization; predict refuses a pin below them)"
        ]
    return summary


def _planned_token_bucket(
    request: PredictionRequest, generated: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """The token bucket ``--padding`` would pick, from the job's token estimate.

    The ``padding`` block shows the request, which leaves an unpinned axis
    null; the bucket itself is chosen from the featurized token count. That
    count is estimated here as `plan --json`'s Slurm block estimates it, so a
    ligand it cannot count is named. None for a native input, which carries no
    common job to count.
    """
    from foldjax.padding import resolve_axis
    from foldjax.slurm import estimate_tokens

    document = generated
    if document is None and request.input_format == "foldjax":
        from foldjax.input import read_job_document

        try:
            document = read_job_document(Path(request.input))
        except (OSError, ValueError):
            return None
    if not isinstance(document, Mapping):
        return None
    tokens, notes = estimate_tokens(document)
    record: dict[str, Any] = {"tokens": tokens}
    if notes:
        record["tokens_not_counted"] = notes
    try:
        record["token_bucket"] = resolve_axis(tokens, request.padding, "tokens")
    except ValueError as error:
        record["token_bucket"] = None
        record["reason"] = str(error)
    return record


def _effective_sampling(request: PredictionRequest) -> dict[str, Any]:
    """What each neutral sampling knob will run at, and where that comes from.

    ``request`` names the knob, ``option`` is a native option (an explicit
    ``--option`` or a managed profile's), ``default`` the adapter's released
    value, and ``checkpoint`` a value the checkpoint or its model variant
    decides -- null only where it cannot be read before the run. Values are
    in the neutral knobs' units (`Backend.sampling_resolution`).
    """
    from foldjax.registry import get_backend

    resolved = get_backend(request.model).sampling_resolution(request)
    return {
        "sampling": {knob: value for knob, (value, _) in resolved.items()},
        "sampling_source": {knob: source for knob, (_, source) in resolved.items()},
    }


def _run_predictions(
    request: PredictionRequest | None,
    *,
    sources: dict[Path, JobSource] | None = None,
    input_failures: tuple[PredictionFailure, ...] = (),
    failures_root: Path | None = None,
) -> BatchReport:
    """Execute a request and report what ran, what was reused and what failed.

    The resume and error policies live on the request rather than in this
    function, so `foldjax.predict_batch(...)` and `foldjax predict --resume
    --keep-going` are the same execution -- including at seed granularity,
    which is where the expensive repetition was. ``request`` is None when no
    input could be read at all; ``input_failures`` then is the whole report.
    """
    if request is None:
        report = BatchReport(failures=input_failures)
        _write_failures(failures_root or Path.cwd(), list(input_failures))
    else:
        # Only what is set, so the plain call keeps the one-argument shape.
        extra: dict[str, Any] = {}
        if sources:
            extra["sources"] = sources
        if input_failures:
            extra["input_failures"] = input_failures
        report = predict_batch(request, **extra)
    for path in report.skipped:
        print(f"[foldjax] reused finished run at {path}", file=sys.stderr)
    for failure in report.failures:
        seed = "" if failure.seed is None else f" seed {failure.seed}"
        # The file the caller wrote, not the job document generated from it.
        named = failure.source.describe() if failure.source else failure.input
        print(
            f"foldjax: {failure.model} · {named}{seed} failed: {failure.error}",
            file=sys.stderr,
        )
    return report


def _warn_no_structures(path: Path, command: str) -> None:
    """Say that a directory holds runs but no structure to read."""
    warnings.warn(
        f"{command} found 0 structures under {path}: its runs wrote none (a "
        "--stop-after run) or recorded none that exist",
        UserWarning,
        stacklevel=2,
    )


def _unreadable_failures(
    args: argparse.Namespace,
    unreadable: list[tuple[Path, Exception]],
    *,
    plural: bool,
) -> tuple[PredictionFailure, ...]:
    """One failure per model for each input that never became a job."""
    root = args.output_dir or Path("foldjax-outputs")
    failures: list[PredictionFailure] = []
    for path, error in unreadable:
        text = str(path)
        original = Path(text[10:]) if text.startswith("structure:") else path
        for model in args.model:
            name = normalize_model_name(model)
            failures.append(
                PredictionFailure(
                    model=name,
                    input=original,
                    seed=None,
                    output_dir=(
                        root / name / original.stem
                        if plural
                        else (args.output_dir or root / original.stem)
                    ),
                    error=str(error),
                    error_type=type(error).__name__,
                )
            )
    return tuple(failures)


def _render_predictions(results: list[PredictionResult]) -> str:
    """The summary tables for everything that just ran, or a plain fallback.

    A manifest whose provenance could not be described never fails a
    prediction (`foldjax.manifest.write`), so the renderer has to cope with its absence
    rather than assume the file it prefers to read.
    """
    entries: list[tuple[Path, dict[str, Any]]] = []
    seen: set[Path] = set()
    for result in results:
        if result.output_dir is None:
            continue
        directory = Path(result.output_dir)
        if directory in seen:
            continue
        seen.add(directory)
        entries.extend(report.read_manifests(directory))
    if entries:
        return report.render_all(entries)
    return json.dumps(
        [result.summary() for result in results], indent=2, sort_keys=True
    )


@contextmanager
def _stdout_to_stderr() -> Iterator[None]:
    """Send everything written to stdout, from Python or from C, to stderr.

    ``redirect_stdout`` alone covers ``print``; a native extension writes to
    file descriptor 1 directly, so that is pointed at stderr too, and restored
    afterwards. Where stdout has no descriptor (an embedding host, a test
    capture) only the Python-level redirect applies.
    """
    saved: int | None = None
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (AttributeError, OSError, ValueError):
            pass
    try:
        # Descriptors 1 and 2 themselves, not ``sys.stdout.fileno()``: native
        # code writes to 1 whatever Python object stands in for stdout.
        os.fstat(1)
        os.fstat(2)
        saved = os.dup(1)
        os.dup2(2, 1)
    except OSError:
        if saved is not None:
            os.close(saved)
        saved = None
    try:
        with redirect_stdout(sys.stderr):
            yield
    finally:
        if saved is not None:
            try:
                sys.stderr.flush()
            except (OSError, ValueError):
                pass
            os.dup2(saved, 1)
            os.close(saved)


def _format_warnings() -> None:
    """Print each warning once per command as ``foldjax: warning: <message>``.

    Python's default shows the warning's source file and the line that raised
    it, which in a terminal reads as a crash report, and repeats it for every
    seed. Installed by `entrypoint`, for the command's own process only and
    never on import: a library must not decide how its host shows warnings.
    """
    seen: set[tuple[type[Warning], str]] = set()

    def show(message, category, filename, lineno, file=None, line=None) -> None:
        text = str(message).strip()
        key = (category, text)
        if key in seen:
            return
        seen.add(key)
        try:
            print(f"foldjax: warning: {text}", file=file or sys.stderr, flush=True)
        except (OSError, ValueError):
            pass

    warnings.showwarning = show


def _run_plan(args: argparse.Namespace) -> int:
    """`foldjax plan`: resolve and check the request, writing nothing to the store.

    Generated job documents (a --sequence, FASTA or structure input, a shard's
    multi-job file, the jobs of a multi-job file) are written to scratch; the
    stems are the ones predict would use. `--json` adds the `slurm` block.
    """
    import tempfile

    from foldjax import slurm, tools_cli

    with tempfile.TemporaryDirectory(prefix="foldjax-plan-") as scratch:
        jobs = Path(scratch) / "jobs"
        tools_cli.prepare(args, jobs_root=jobs)
        handled = tools_cli.dispatch(args)
        if handled is not None:
            return handled
        requested = _request(args, jobs_root=jobs)
        resolved = resolve_requests(
            requested, draw_seeds=False, jobs_root=jobs / "split"
        )
        payload = []
        for item in resolved:
            preflight(item)
            summary = _plan_summary(item, scratch=jobs)
            if args.json:
                # Read while the scratch document still exists: the shown
                # input is the store path predict would write, not this one.
                summary["slurm"] = slurm.plan_slurm(
                    {**summary, "input": str(item.input)},
                    public_options(item.options) or {},
                )
            payload.append(summary)
    print(
        json.dumps(
            payload
            if requested.models is not None or requested.inputs is not None
            else payload[0],
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # Importing or embedding the CLI for discovery/plan commands must not
    # change the host's future JAX allocator. Prediction is the only command
    # that owns a model process and therefore the only one that applies it.
    if args.command == "predict" or (
        args.command == "cache" and args.cache_command == "warm"
    ):
        _apply_mem_fraction(args.mem_fraction)
        _apply_rendezvous_timeout(args)
        _apply_gc_threshold()
        _skip_release_reclaim()
    elif args.command == "plan":
        _validate_mem_fraction(args.mem_fraction)
        return _run_plan(args)
    tools_cli.prepare(args)
    handled = tools_cli.dispatch(args)
    if handled is not None:
        return handled
    if args.command == "models":
        if args.for_input is not None:
            return _run_models_for(args)
        if args.json:
            print(
                json.dumps(
                    [model_info(name).summary() for name in available_models()],
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(*available_models(), sep="\n")
        return 0
    if args.command == "home":
        locations = paths.describe()
        output = (
            locations[args.path]
            if args.path
            else json.dumps(locations, indent=2, sort_keys=True)
        )
        print(output)
        return 0
    if args.command == "capabilities":
        print(
            json.dumps(
                dataclasses.asdict(capabilities(args.model)), indent=2, sort_keys=True
            )
        )
        return 0
    if args.command == "runtime":
        return _run_runtime(args)
    if args.command == "setup":
        return _run_setup(args)
    if args.command == "weights":
        return _run_weights(args)
    if args.command == "doctor":
        return doctor.run_doctor(args)
    if args.command == "cache":
        if args.cache_command == "gc":
            return cache_gc.run_cache_gc(args)
        return _run_cache(args)
    if args.command == "compare":
        from foldjax.compare import compare_directory

        written = compare_directory(
            args.path, out=args.out, samples=args.samples
        )
        document = json.loads(written["json"].read_text(encoding="utf-8"))
        inputs = document.get("inputs") or []
        if not any(entry.get("structures") for entry in inputs):
            _warn_no_structures(args.path, "compare")
        elif not any(entry.get("pairs") for entry in inputs):
            warnings.warn(
                f"compare found no pair to compare under {args.path}: each input "
                "there has a single structure, so compare.csv lists no pairs. "
                "Compare several samples, seeds or models of one input, or "
                "pass --reference to score against a known structure",
                UserWarning,
                stacklevel=2,
            )
        print(json.dumps({key: str(value) for key, value in written.items()}, indent=2))
        return 0
    if args.command == "show" and (args.format != "table" or args.aggregate):
        from foldjax import results

        if args.json:
            raise ValueError("--json prints manifests; use --format json for rows")
        if args.format == "table":
            raise ValueError("--aggregate needs --format csv or --format json")
        rows = results.results_table(results.load_results(args.path))
        if not any(row.get("structure_path") for row in rows):
            _warn_no_structures(args.path, "show")
        if args.aggregate:
            rows = results.aggregate_table(rows)
        if args.format == "csv":
            sys.stdout.write(results.to_csv(rows))
        else:
            print(json.dumps(rows, indent=2, sort_keys=True, default=str))
        return 0
    if args.command == "show":
        from foldjax import results
        from foldjax.api import FAILURES_NAME

        entries = report.read_manifests(args.path)
        root = Path(args.path)
        failures = (
            list(results.load_results(root).failures)
            if not args.json
            and (
                root.name == FAILURES_NAME
                or (root.is_dir() and any(root.rglob(FAILURES_NAME)))
            )
            else []
        )
        if not entries and not failures:
            if not root.exists():
                # The same words `show --format csv`, compare and report use.
                raise FileNotFoundError(f"no such output directory: {args.path}")
            raise FileNotFoundError(
                f"no {manifest.MANIFEST_NAME} under {args.path}; a run writes one "
                "when it finishes"
            )
        if args.json:
            print(
                json.dumps(
                    [document for _path, document in entries],
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if entries and not any(
            sample.get("structure_path")
            for _path, document in entries
            for sample in document.get("samples") or []
            if isinstance(sample, dict)
        ):
            _warn_no_structures(args.path, "show")
        blocks = [report.render_all(entries)] if entries else []
        if failures:
            blocks.append(report.render_failures(failures))
        print("\n\n".join(blocks))
        return 0

    # Progress is on for this command, not for the host process: `main` is also
    # called in-process (tests, notebooks), and a caller that never asked for
    # stderr lines kept getting them after it returned.
    host_progress = (progress._enabled, progress._stream)
    if not args.quiet:
        progress.enable()
    try:
        # Stdout carries the result and nothing else: anything a backend or a
        # native library prints while the request resolves and runs goes to
        # stderr, so `foldjax predict ... > out.json` stays valid JSON.
        sources: dict[Path, JobSource] = {}
        unreadable: list[tuple[Path, Exception]] | None = (
            [] if getattr(args, "keep_going", False) else None
        )
        with _stdout_to_stderr():
            request = _request(args, sources=sources, unreadable=unreadable)
            plural = (
                request.models is not None or request.inputs is not None
                if request is not None
                else len(args.model) > 1
                or len(unreadable or ()) > 1
                or getattr(args, "_plural_inputs", False)
            )
            outcome = _run_predictions(
                request,
                sources=sources,
                input_failures=_unreadable_failures(
                    args, unreadable or [], plural=plural
                ),
                failures_root=args.output_dir,
            )
    finally:
        progress._enabled, progress._stream = host_progress
    results = list(outcome.results)
    if args.json or not sys.stdout.isatty():
        summaries = [result.summary() for result in results]
        payload: Any = summaries if plural else (summaries[0] if summaries else {})
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_render_predictions(results))
    formatted = tools_cli.finish_predict(args, results)
    # A batch that lost some of its runs is neither a success nor the same
    # failure as one that could not start; 3 says "partial" without pretending.
    return 3 if outcome.failures else formatted


#: Optional dependencies, and the extra that supplies each one. A missing
#: import is one of the few failures whose fix is a single exact command, and
#: the exception itself only ever names the module.
_EXTRA_FOR_MODULE = {
    "biotite": "openfold3-preprocess",
    "gemmi": "openfold3-preprocess",
    "rdkit": "openfold3-preprocess",
    "scipy": "openfold3-preprocess",
    "triton": "cuda13",
    "jaxlib": "cuda13",
    "kalign": "templates",
}


def _with_hint(error: BaseException) -> str:
    """The error, plus the next command -- when there is exactly one.

    Most FoldJAX failures already carry their own instruction: a missing
    checkpoint names its `weights fetch` line, an OOM names the knobs that
    change its cost. This fills the two gaps where the raise site cannot know
    the answer -- a missing optional package, and a disk that filled up.
    Anything else is returned unchanged rather than decorated with a guess.
    """
    message = str(error)
    if isinstance(error, ModuleNotFoundError) and error.name:
        extra = _EXTRA_FOR_MODULE.get(error.name.split(".")[0])
        if extra is not None:
            from foldjax.doctor import install_command

            return f"{message}\n  install it with: {install_command(extra)}"
        return message
    if isinstance(error, OSError) and error.errno == errno.ENOSPC:
        return (
            f"{message}\n"
            f"  the FoldJAX store is at {paths.foldjax_home()}\n"
            "  reclaim compile cache with: foldjax cache gc --older-than 30 --apply"
        )
    return message


#: Failures that mean "you asked for something that cannot work", as opposed to
#: a bug in FoldJAX. These get one clean line; anything else keeps its traceback
#: so a real defect stays debuggable.
#:
#: ``PermissionError`` and the other OSErrors are here because a read-only
#: output directory or a full disk is a fact about the machine, not a defect in
#: this package, and a stack trace through FoldJAX's internals says otherwise.
#: ``FileNotFoundError``, ``NotADirectoryError`` and ``IsADirectoryError`` are
#: OSError subclasses and stay listed for documentation.
_USER_ERRORS = (
    PredictionError,
    MemoryError,
    ValueError,
    FileNotFoundError,
    NotADirectoryError,
    IsADirectoryError,
    PermissionError,
    OSError,
    ModuleNotFoundError,
)


def entrypoint() -> None:
    _format_warnings()
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        # Cancelling a run that takes minutes is an ordinary thing to do, and a
        # traceback through JAX's internals reads as a crash rather than as the
        # answer to the key that was just pressed. 130 is what a shell expects
        # from a process that took SIGINT.
        print("\nfoldjax: interrupted", file=sys.stderr)
        raise SystemExit(130) from None
    except _USER_ERRORS as error:
        print(f"foldjax: {_with_hint(error)}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    # Without this, `python -m foldjax.cli` imports the module, defines
    # `main`, calls nothing, and exits 0 with no output -- which reads as a
    # prediction that produced nothing rather than as a command that never ran.
    entrypoint()
