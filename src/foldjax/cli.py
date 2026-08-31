"""FoldJAX command-line interface."""

from __future__ import annotations

import argparse
import dataclasses
import errno
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from foldjax import assets, manifest, oom, paths, progress, report
from foldjax.api import predict_batch, resolve_requests
from foldjax.job import Job
from foldjax.redaction import public_options
from foldjax.registry import available_models, capabilities, model_info
from foldjax.schema import (
    MSA_POLICIES,
    STOP_POINTS,
    BatchReport,
    PaddingConfig,
    PredictionError,
    PredictionRequest,
    PredictionResult,
    expand_input_directories,
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
        help="what to call a --sequence job in output file names (default 'job')",
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
        help="representative seed (default 0)" if cache_warm else None,
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
            else "run the job once per seed, into a seed_<n> directory each, "
            "and return every structure together. The samples from one seed "
            "are correlated, so this is the usual way to get independent "
            "predictions. Mutually exclusive with --seed"
        ),
    )
    sampling.add_argument(
        "--num-seeds",
        type=int,
        help=(
            "accepted for request parity; cache warm still executes only the "
            "first seed. Mutually exclusive with --seeds"
            if cache_warm
            else "how many seeds to run, counting up from --seed. --num-seeds "
            "3 is --seeds 0 1 2; mutually exclusive with --seeds"
        ),
    )
    source.add_argument(
        "--msa",
        choices=MSA_POLICIES,
        default="none",
        help="what to do about a protein chain with no alignment: fold it from "
        "the single sequence (default, unchanged), 'auto' to search and cache "
        "an alignment, or 'required' to fail rather than fall back. auto and "
        "required SEND THE SEQUENCE to the public ColabFold MMseqs2 server "
        "(FOLDJAX_MSA_SERVER_URL points at your own instead)",
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
            "'trunk' stops once the representations exist, skipping the "
            "diffusion sampler and the confidence heads. It writes no "
            "structure, so it only makes sense with --representations."
        ),
    )
    sampling.add_argument(
        "--num-samples", type=int, help="how many structures to generate"
    )
    sampling.add_argument("--num-steps", type=int, help="diffusion steps per structure")
    sampling.add_argument("--num-recycles", type=int, help="trunk recycling iterations")
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
        help="normalize every model-relevant dynamic axis to FoldJAX's standard "
        "shape buckets. Disabled by default so existing scientific results and "
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
        help="pin the padded MSA row count after max-MSA-depth is applied",
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


def _parser() -> argparse.ArgumentParser:
    from foldjax import __version__

    parser = argparse.ArgumentParser(
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
        "Answered from the input translation table, so it needs no weights",
    )
    home = commands.add_parser("home", help="show where FoldJAX keeps its files")
    home.add_argument(
        "--path",
        choices=("home", "downloads", "weights", "assets", "compile_cache", "runtime"),
        help="print just one location, for shell scripts",
    )

    runtime = commands.add_parser(
        "runtime", help="inspect or prepare model-specific native runtime artifacts"
    )
    runtime_commands = runtime.add_subparsers(dest="runtime_command", required=True)
    runtime_status = runtime_commands.add_parser(
        "status", help="report whether one model can start without preparation"
    )
    runtime_status.add_argument("--model", required=True)
    runtime_prepare = runtime_commands.add_parser(
        "prepare", help="prepare one model's generated native artifacts"
    )
    runtime_prepare.add_argument("--model", required=True)
    runtime_gc = runtime_commands.add_parser(
        "gc", help="inspect or remove older prepared runtime generations"
    )
    runtime_gc.add_argument("--model", required=True)
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

    describe = commands.add_parser("capabilities", help="show what one backend accepts")
    describe.add_argument("--model", required=True)

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
        "installs every published checkpoint. AlphaFold 3 is still yours to "
        "supply: its parameters are licensed, not merely large",
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
    where.add_argument("--model", required=True)
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
        help="remove old compile-cache entries",
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


def _resolve_inputs(args: argparse.Namespace) -> list[Path]:
    """Turn everything the CLI accepts as input into common job/native files."""
    sequences = bool(args.sequence or args.dna or args.rna)
    ligands = bool(args.ligand or args.ligand_smiles)
    if sequences or ligands:
        if args.input:
            raise ValueError(
                "--input and --sequence are alternatives; pass one of them"
            )
        if not sequences:
            raise ValueError("a ligand needs a --sequence to bind to")
        return [
            Job.from_sequences(
                args.sequence,
                dna=args.dna,
                rna=args.rna,
                ligand_ccd=args.ligand,
                ligand_smiles=args.ligand_smiles,
                name=args.name or "job",
                affinity_binder=args.affinity_binder,
            ).store()
        ]
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
    return [_as_job_file(path) for path in expanded]


def _as_job_file(path: Path) -> Path:
    """Turn one accepted input into a file a request can carry.

    FASTA and deposited structures become ordinary common-schema documents;
    everything else is already one, or is a model's own dialect, and passes
    through untouched.
    """
    text = str(path)
    if text.startswith("structure:"):
        return Job.from_structure(Path(text[10:])).store()
    suffix = path.suffix.lower()
    if suffix in _FASTA_SUFFIXES:
        return Job.from_fasta(path).store()
    if suffix in _STRUCTURE_SUFFIXES:
        return Job.from_structure(path).store()
    return path


def _request(args: argparse.Namespace) -> PredictionRequest:
    inputs = _resolve_inputs(args)
    single_model = len(args.model) == 1
    single_input = len(inputs) == 1
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
    return PredictionRequest(
        model=args.model[0] if single_model else None,
        models=None if single_model else tuple(args.model),
        input=inputs[0] if single_input else None,
        inputs=None if single_input else tuple(inputs),
        weights=args.weights,
        profile=args.profile,
        output_dir=args.output_dir,
        input_format=args.input_format,
        seed=0 if args.seed is None else args.seed,
        seeds=tuple(args.seeds) if args.seeds else None,
        num_seeds=args.num_seeds,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        num_recycles=args.num_recycles,
        max_msa_depth=args.max_msa_depth,
        cache_dir=args.cache_dir,
        use_compile_cache=not getattr(args, "no_cache", False),
        options=_options(args.option),
        padding=padding,
        msa=args.msa,
        representations=getattr(args, "representations", None),
        stop_after=getattr(args, "stop_after", "full"),
        resume=getattr(args, "resume", False),
        on_error="continue" if getattr(args, "keep_going", False) else "stop",
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
    """What the template modality still needs, in the order it needs it."""
    from foldjax.models.protenix.data.search import templates
    from foldjax.paths import assets_dir

    lines = []
    metadata = ["release_date_cache.json", "obsolete_to_successor.json"]
    missing = [name for name in metadata if not (assets_dir() / name).is_file()]
    lines.append(
        "metadata      missing: " + ", ".join(missing)
        if missing
        else "metadata      ready"
    )

    try:
        binary = templates._resolve_kalign_binary(None)
        lines.append(f"kalign        ready  {binary}")
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        lines.append(f"kalign        {error}")

    directory = os.environ.get("PROTENIX_TEMPLATE_MMCIF_DIR")
    if directory and Path(directory).is_dir():
        lines.append(f"coordinates   ready  {directory}")
    else:
        lines.append(
            "coordinates   set PROTENIX_TEMPLATE_MMCIF_DIR to a directory of "
            "mmCIF files (flat or PDB-divided, .cif or .cif.gz)"
        )
    return lines


def _run_setup(args: argparse.Namespace) -> int:
    """Fetch default assets and say exactly what opt-in/manual models need."""
    print(f"store: {paths.foldjax_home()}\n")
    print("weights")
    failed = False
    # Profiles, not just models. A model's alternative checkpoints are separate
    # bundles with their own `in_default_setup`, and Protenix's v2 is one: the
    # release and v2 are both supported and both fetched, so iterating models
    # alone would silently skip the newer of the two.
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
        if not spec.downloads:
            # Gated or non-redistributable: the instruction differs per model,
            # so `notes` is the only honest text.
            state = "ready" if spec.ready() else "manual"
            print(f"  {label:<11s} {state}")
            if not spec.ready():
                print(f"    {spec.notes}")
                print(f"    goes in: {assets.weights_dir(name)}")
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


def _run_models_for(args: argparse.Namespace) -> int:
    """Say which models can run one job, before anything is downloaded.

    Whether a backend can express a document is knowable from the input layer
    alone, so this answers without weights, without a GPU, and without the
    fifteen minutes it takes to discover the same thing by running the job.
    """
    from foldjax.input import compatibility, read_job_document

    path = Path(args.for_input)
    document = read_job_document(path)
    if path.suffix.lower() in _FASTA_SUFFIXES:
        document = Job.from_fasta(path).to_document()
    rows = []
    for name in available_models():
        reason = compatibility(document, name)
        info = model_info(name)
        rows.append(
            {
                "model": name,
                "runs": reason is None,
                "reason": reason,
                "weights_ready": info.weights_ready,
                "setup": info.setup,
            }
        )
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    print(f"{'model':<11s}{'runs?':<7s}why")
    for row in rows:
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
                    print(
                        f"         profile {profile['profile']}: {state}, "
                        f"downloaded {profile['downloaded']}, {size_text}"
                    )
        return 0
    if args.weights_command == "path":
        print(assets.resolve_weights(args.model, profile=args.profile))
        return 0

    spec = assets.assets_for(args.model, profile=args.profile)
    public_model = assets._public_model_name(spec.model)
    print(f"{public_model}: {len(spec.downloads)} file(s) from {spec.source}")
    print(f"licence: {spec.licence}")
    reporter = _WeightReporter()
    try:
        result = assets.fetch(
            public_model,
            profile=args.profile,
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


def _parse_size(text: str) -> int:
    """Accept 20G / 500M / 1024 the way every other disk tool does."""
    value = text.strip().upper().rstrip("B")
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    factor = 1
    if value and value[-1] in scale:
        factor, value = scale[value[-1]], value[:-1]
    try:
        size = float(value)
    except ValueError as error:
        raise ValueError(
            f"--max-size must look like 20G or 500M; got {text!r}"
        ) from error
    if not math.isfinite(size):
        raise ValueError(f"--max-size must look like 20G or 500M; got {text!r}")
    if size <= 0:
        raise ValueError("--max-size must be positive")
    return int(size * factor)


def _open_cache_root(root: Path) -> int | None:
    """Pin a real cache directory so later path swaps cannot widen GC scope."""
    required = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required) or not hasattr(os, "fwalk"):
        raise RuntimeError(
            "cache gc requires directory-descriptor and no-follow filesystem support"
        )
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        return os.open(root, flags)
    except OSError as error:
        if error.errno in {errno.ENOENT, errno.ENOTDIR, errno.ELOOP}:
            return None
        raise


_TOKAMAX_AUTOTUNE_DIRECTORIES = frozenset(
    (".tokamax-autotuning-v1", ".tokamax-autotuning-v2")
)
_CacheGcEntry = tuple[Path, float, int, int, int, int, str | None]


@dataclasses.dataclass(frozen=True)
class _CacheGcScan:
    entries: tuple[_CacheGcEntry, ...]
    protected_files: int
    protected_bytes: int

    @property
    def total_files(self) -> int:
        return len(self.entries) + self.protected_files

    @property
    def total_bytes(self) -> int:
        return sum(item[2] for item in self.entries) + self.protected_bytes


def _tokamax_temporary_lock_name(name: str) -> str | None:
    """Return the exact lock name for a v2 atomic-write temporary."""
    if not (name.startswith(".") and name.endswith(".tmp")):
        return None
    try:
        result_name, process_id, timestamp = name[1:-4].rsplit(".", maxsplit=2)
    except ValueError:
        return None
    if not result_name.endswith(".json"):
        return None
    signature = result_name.removesuffix(".json")
    if (
        len(signature) != 64
        or any(character not in "0123456789abcdef" for character in signature)
        or not process_id.isascii()
        or not process_id.isdecimal()
        or not timestamp.isascii()
        or not timestamp.isdecimal()
    ):
        return None
    return f"{signature}.lockfile"


def _acquire_tokamax_gc_lock(directory_fd: int, lock_name: str) -> int | None:
    """Non-blockingly pin an existing regular Tokamax lock for safe GC."""
    import fcntl
    import stat

    flags = os.O_RDWR | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        lock_fd = os.open(lock_name, flags, dir_fd=directory_fd)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
            os.close(lock_fd)
            return None
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, ValueError):
        os.close(lock_fd)
        return None
    return lock_fd


def _release_tokamax_gc_lock(lock_fd: int) -> None:
    import fcntl

    try:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
        os.close(lock_fd)


def _raise_cache_gc_walk_error(error: OSError) -> None:
    """Never turn an incomplete filesystem scan into a successful budget report."""
    raise error


def _cache_gc_usage(root_fd: int) -> tuple[int, int]:
    """Measure regular-file usage below an already pinned cache root."""
    import stat

    files = 0
    size = 0
    for _directory, _directories, names, directory_fd in os.fwalk(
        ".",
        topdown=True,
        onerror=_raise_cache_gc_walk_error,
        follow_symlinks=False,
        dir_fd=root_fd,
    ):
        for name in names:
            try:
                info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if stat.S_ISREG(info.st_mode):
                files += 1
                size += info.st_size
    return files, size


def _cache_gc_entries(
    root_fd: int,
) -> _CacheGcScan:
    """List stable regular-file identities under a pinned cache root.

    Tokamax lock files and unrecognised temporaries remain protected but count
    toward the real cache footprint. An exactly named v2 temporary is eligible
    only when its existing per-signature lock can be acquired without waiting;
    apply acquires that same lock again and holds it through unlink.
    """
    import stat

    entries: list[_CacheGcEntry] = []
    protected_files = 0
    protected_bytes = 0
    for directory, _directories, names, directory_fd in os.fwalk(
        ".",
        topdown=True,
        onerror=_raise_cache_gc_walk_error,
        follow_symlinks=False,
        dir_fd=root_fd,
    ):
        parent = Path(directory)
        for name in names:
            try:
                info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(info.st_mode):
                continue
            lock_name = None
            if not _TOKAMAX_AUTOTUNE_DIRECTORIES.isdisjoint(parent.parts):
                if name.endswith(".lockfile"):
                    protected_files += 1
                    protected_bytes += info.st_size
                    continue
                if name.endswith(".tmp"):
                    lock_name = (
                        _tokamax_temporary_lock_name(name)
                        if ".tokamax-autotuning-v2" in parent.parts
                        else None
                    )
                    lock_fd = (
                        None
                        if lock_name is None
                        else _acquire_tokamax_gc_lock(directory_fd, lock_name)
                    )
                    if lock_fd is None:
                        protected_files += 1
                        protected_bytes += info.st_size
                        continue
                    _release_tokamax_gc_lock(lock_fd)
            entries.append(
                (
                    parent / name,
                    info.st_mtime,
                    info.st_size,
                    info.st_dev,
                    info.st_ino,
                    info.st_mtime_ns,
                    lock_name,
                )
            )
    return _CacheGcScan(
        entries=tuple(entries),
        protected_files=protected_files,
        protected_bytes=protected_bytes,
    )


def _apply_cache_gc(
    root: Path,
    root_fd: int,
    doomed: list[_CacheGcEntry],
) -> tuple[int, int, int, int, int]:
    """Delete selected files through pinned parent descriptors.

    Returns ``(removed_files, removed_bytes, failed_files,
    already_absent_files, changed_files)``. A concurrent collector removing the
    same entry is success for the desired state, while a path replaced since
    the report was planned is left for a later GC pass.
    """
    pending = {
        relative: (size, device, inode, mtime_ns, lock_name)
        for relative, _mtime, size, device, inode, mtime_ns, lock_name in doomed
    }
    removed_files = 0
    removed_bytes = 0
    failed_files = 0
    already_absent_files = 0
    changed_files = 0
    for directory, directories, names, directory_fd in os.fwalk(
        ".",
        topdown=False,
        onerror=_raise_cache_gc_walk_error,
        follow_symlinks=False,
        dir_fd=root_fd,
    ):
        parent = Path(directory)
        for name in names:
            relative = parent / name
            if relative not in pending:
                continue
            size, device, inode, mtime_ns, lock_name = pending.pop(relative)
            lock_fd = (
                None
                if lock_name is None
                else _acquire_tokamax_gc_lock(directory_fd, lock_name)
            )
            if lock_name is not None and lock_fd is None:
                changed_files += 1
                continue
            try:
                try:
                    current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                except FileNotFoundError:
                    already_absent_files += 1
                    continue
                except OSError as error:
                    failed_files += 1
                    print(
                        f"[cache] could not inspect {root / relative}: {error}",
                        file=sys.stderr,
                    )
                    continue
                if (
                    current.st_dev,
                    current.st_ino,
                    current.st_mtime_ns,
                    current.st_size,
                ) != (device, inode, mtime_ns, size):
                    changed_files += 1
                    continue
                try:
                    os.unlink(name, dir_fd=directory_fd)
                except FileNotFoundError:
                    already_absent_files += 1
                except OSError as error:
                    failed_files += 1
                    print(
                        f"[cache] could not remove {root / relative}: {error}",
                        file=sys.stderr,
                    )
                else:
                    removed_files += 1
                    removed_bytes += size
            finally:
                if lock_fd is not None:
                    _release_tokamax_gc_lock(lock_fd)

        # Children have already been visited. ``rmdir`` is itself an atomic
        # emptiness and no-follow check; a symlink or concurrent writer simply
        # makes it fail without escaping this pinned parent descriptor.
        for name in directories:
            try:
                os.rmdir(name, dir_fd=directory_fd)
            except OSError:
                continue

    # Entries removed or renamed before the second descriptor walk no longer
    # need collection and are not failures of this invocation.
    already_absent_files += len(pending)
    return (
        removed_files,
        removed_bytes,
        failed_files,
        already_absent_files,
        changed_files,
    )


def _run_cache_gc(args: argparse.Namespace) -> int:
    """Reclaim compile-cache space, reporting first and deleting only on request.

    The cache is derived data keyed by accelerator, runtime, weight identity and
    shapes. Removing entries preserves the model and shape semantics, but costs
    a recompile or Tokamax retune; a retune may select a different valid
    floating-point schedule and therefore change last bits. It is still
    someone's disk, and a command that deletes gigabytes because it was run to
    see what was there is not a good trade, so the report is the default and
    `--apply` is the verb.
    """
    if args.older_than is None and args.max_size is None:
        raise ValueError("cache gc needs --older-than DAYS, --max-size SIZE, or both")
    # Parsed before the store is inspected, so a typo is reported the same way
    # whether or not a cache happens to exist yet.
    budget = None if args.max_size is None else _parse_size(args.max_size)
    if args.older_than is not None and args.older_than < 0:
        raise ValueError("--older-than must be a non-negative number of days")
    root = paths.compile_cache_dir()
    root_fd = _open_cache_root(root)
    if root_fd is None:
        print(f"[cache] nothing at {root}", file=sys.stderr)
        print(
            json.dumps(
                {
                    "root": str(root),
                    "applied": bool(args.apply),
                    "total_files": 0,
                    "total_bytes": 0,
                    "protected_files": 0,
                    "protected_bytes": 0,
                    "planned_removed_files": 0,
                    "planned_removed_bytes": 0,
                    "removed_files": 0,
                    "removed_bytes": 0,
                    "failed_files": 0,
                    "already_absent_files": 0,
                    "changed_files": 0,
                    "remaining_bytes": 0,
                    "budget_satisfied": True if budget is not None else None,
                }
            )
        )
        return 0

    import time

    try:
        scan = _cache_gc_entries(root_fd)
        entries = list(scan.entries)
        entries.sort(key=lambda item: item[1], reverse=True)

        doomed: list[_CacheGcEntry] = []
        if args.older_than is not None:
            cutoff = time.time() - args.older_than * 86400
            doomed = [item for item in entries if item[1] < cutoff]
        if budget is not None:
            kept = scan.protected_bytes
            over: list[_CacheGcEntry] = []
            for item in entries:
                if kept + item[2] <= budget:
                    kept += item[2]
                else:
                    over.append(item)
            chosen = {item[0] for item in doomed}
            doomed.extend(item for item in over if item[0] not in chosen)

        planned_removed_files = len(doomed)
        planned_removed_bytes = sum(item[2] for item in doomed)
        removed_files = planned_removed_files
        removed_bytes = planned_removed_bytes
        failed_files = 0
        already_absent_files = 0
        changed_files = 0
        if args.apply:
            (
                removed_files,
                removed_bytes,
                failed_files,
                already_absent_files,
                changed_files,
            ) = _apply_cache_gc(root, root_fd, doomed)
            _remaining_files, remaining_bytes = _cache_gc_usage(root_fd)
        else:
            remaining_bytes = scan.total_bytes - planned_removed_bytes
    finally:
        os.close(root_fd)
    action = "removed" if args.apply else "would remove"
    outcome = f"{removed_files} file(s), {_format_bytes(removed_bytes)}"
    if args.apply and failed_files:
        outcome += (
            f" (planned {planned_removed_files} file(s), "
            f"{_format_bytes(planned_removed_bytes)}; {failed_files} failed)"
        )
    elif args.apply and already_absent_files:
        outcome += f" ({already_absent_files} already absent)"
    if args.apply and changed_files:
        outcome += f" ({changed_files} changed since planning; kept)"
    print(
        f"[cache] {action} {outcome} of "
        f"{_format_bytes(scan.total_bytes)} under {root}"
        + ("" if args.apply else "; pass --apply to do it"),
        file=sys.stderr,
    )
    print(
        json.dumps(
            {
                "root": str(root),
                "applied": bool(args.apply),
                "total_files": scan.total_files,
                "total_bytes": scan.total_bytes,
                "protected_files": scan.protected_files,
                "protected_bytes": scan.protected_bytes,
                "planned_removed_files": planned_removed_files,
                "planned_removed_bytes": planned_removed_bytes,
                "removed_files": removed_files,
                "removed_bytes": removed_bytes,
                "failed_files": failed_files,
                "already_absent_files": already_absent_files,
                "changed_files": changed_files,
                "remaining_bytes": remaining_bytes,
                "budget_satisfied": (
                    None if budget is None else remaining_bytes <= budget
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


_OPTIONAL_RUNTIME_DISTRIBUTIONS = (
    "tokamax",
    "cuequivariance",
    "cuequivariance-jax",
    "cuequivariance-ops-jax-cu12",
    "cuequivariance-ops-jax-cu13",
    "jax-cuda12-plugin",
    "jax-cuda12-pjrt",
    "jax-cuda13-plugin",
    "jax-cuda13-pjrt",
    "triton",
)

# Extras are installation contracts, so distribution metadata is both cheaper
# and more faithful than importing their modules. In particular, probing these
# must not initialize Triton, cuEq kernels, or either raw-input pipeline.
_EXTRA_DISTRIBUTIONS = {
    "alphafold3": (
        ("absl-py", ">=2.3.1"),
        ("dm-haiku", "==0.0.17"),
        ("etils", ""),
        ("zstandard", ""),
    ),
    "openfold3-preprocess": (
        ("absl-py", ">=2.3.1"),
        ("awscrt", ""),
        ("biotite", ""),
        ("boto3", ""),
        ("click", ""),
        ("func-timeout", ""),
        ("ijson", ""),
        ("kalign-python", ""),
        ("lmdb", ""),
        ("memory-profiler", ""),
        ("ml-collections", ">=0.1.1"),
        ("networkx", ""),
        ("pdbeccdutils", ""),
        ("pydantic", ""),
    ),
}


def _distribution_version(name: str) -> str | None:
    """Inspect package metadata without importing an optional runtime."""
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _distribution_version_satisfies(version: str, specifier: str) -> bool | None:
    """Check an extra's version contract, or return None if it cannot be checked."""
    if not specifier:
        return True
    try:
        from packaging.specifiers import InvalidSpecifier, SpecifierSet
        from packaging.version import InvalidVersion, Version
    except ImportError:
        # ``packaging`` is normally present through the scientific stack, but
        # it is not an inference dependency of FoldJAX. Doctor must remain
        # usable in a deliberately minimal environment and fail closed here.
        return None
    try:
        return Version(version) in SpecifierSet(specifier)
    except (InvalidSpecifier, InvalidVersion):
        return None


def _runtime_versions() -> dict[str, str]:
    """Return installed core/optional runtime versions, with no module imports."""
    versions = {}
    for distribution in ("jax", "jaxlib", *_OPTIONAL_RUNTIME_DISTRIBUTIONS):
        version = _distribution_version(distribution)
        if version is not None:
            versions[distribution] = version
    return versions


def _weight_profile_readiness(info: Any) -> list[dict[str, Any]]:
    """Add an actionable reason and command to managed weight profile rows."""
    profiles = []
    for source in info.weight_profiles:
        row = dict(source)
        if row["ready"]:
            row["reason"] = None
            row["setup"] = None
            profiles.append(row)
            continue

        downloaded = str(row.get("downloaded", ""))
        try:
            present_text, total_text = downloaded.split("/", maxsplit=1)
            present, total = int(present_text), int(total_text)
        except (TypeError, ValueError):
            present = total = None

        profile = str(row.get("profile", "released"))
        if total == 0:
            reason = "manual weight installation is required"
            setup = info.setup if profile == "released" else row.get("notes")
        elif present is not None and total is not None and present < total:
            reason = f"managed downloads are incomplete ({present}/{total} present)"
            setup = f"foldjax weights fetch --model {info.model}"
        elif present is not None and total is not None:
            reason = (
                "all managed downloads are present, but conversion or staging "
                "is not ready"
            )
            setup = f"foldjax weights fetch --model {info.model}"
        else:
            reason = "the managed weight readiness check did not pass"
            setup = f"foldjax weights fetch --model {info.model}"
        if profile != "released" and total != 0:
            setup += f" --profile {profile}"
        row["reason"] = reason
        row["setup"] = setup
        profiles.append(row)
    return profiles


def _input_readiness(info: Any) -> dict[str, dict[str, Any]]:
    """Report whether each advertised input dialect can be preprocessed now."""
    rows = {}
    for input_format, requirement in info.capabilities.input_requirements.items():
        missing = []
        incompatible = []
        unknown_extras = []
        for extra in requirement.required_extras:
            distributions = _EXTRA_DISTRIBUTIONS.get(extra)
            if distributions is None:
                unknown_extras.append(extra)
                continue
            for distribution, specifier in distributions:
                version = _distribution_version(distribution)
                if version is None:
                    missing.append(distribution)
                    continue
                satisfies = _distribution_version_satisfies(version, specifier)
                if satisfies is False:
                    incompatible.append(
                        f"{distribution} {version} (requires {specifier})"
                    )
                elif satisfies is None:
                    incompatible.append(
                        f"{distribution} {version} (could not validate {specifier})"
                    )
        runtime_blocked = (
            requirement.preprocessing_runtime == "native" and not info.runtime.ready
        )
        ready = (
            not missing
            and not incompatible
            and not unknown_extras
            and not runtime_blocked
        )
        reasons = []
        if missing:
            reasons.append("missing distributions: " + ", ".join(missing))
        if incompatible:
            reasons.append("incompatible distributions: " + ", ".join(incompatible))
        if unknown_extras:
            reasons.append("unrecognized extras: " + ", ".join(unknown_extras))
        if runtime_blocked:
            reasons.append("the model's generated preprocessing runtime is not ready")
        setup_commands = (
            [f"uv sync --extra {extra}" for extra in requirement.required_extras]
            if missing or incompatible or unknown_extras
            else []
        )
        if runtime_blocked and info.runtime.setup:
            setup_commands.append(info.runtime.setup)
        rows[input_format] = {
            "ready": ready,
            "preprocessing_runtime": requirement.preprocessing_runtime,
            "required_extras": list(requirement.required_extras),
            "missing_distributions": missing,
            "incompatible_distributions": incompatible,
            "reason": "; ".join(reasons) or None,
            "setup": setup_commands or None,
        }
    return rows


def _run_doctor(args: argparse.Namespace) -> int:
    """Everything a first run needs, checked in one command.

    The information was all reachable already -- `models --json`, `home`,
    `runtime status`, the template section of `setup` -- across four commands
    and one that also downloads 3 GB. Someone whose first prediction fails
    should not have to know which of those to run.
    """
    import shutil as _shutil

    report_payload: dict[str, Any] = {
        "foldjax": __import__("foldjax").__version__,
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "home": str(paths.foldjax_home()),
    }

    from foldjax.cache import cache_snapshot, runtime_profile

    devices: list[str] = []
    backend_name = None
    runtime_identity: dict[str, str] | None = None
    try:
        import jax

        runtime_identity = runtime_profile()
        backend_name = runtime_identity["platform"]
        devices = [str(device) for device in jax.devices()]
    except Exception as error:  # noqa: BLE001 - a broken runtime is a finding
        report_payload["jax_error"] = str(error)
    report_payload["jax_backend"] = backend_name
    report_payload["devices"] = devices
    report_payload["jax_version"] = (
        None if runtime_identity is None else runtime_identity["jax"]
    )
    report_payload["jaxlib_version"] = (
        None if runtime_identity is None else runtime_identity["jaxlib"]
    )
    report_payload["device_kind"] = (
        None if runtime_identity is None else runtime_identity["device_kind"]
    )
    report_payload["device_topology"] = (
        None if runtime_identity is None else json.loads(runtime_identity["topology"])
    )
    runtime_versions = _runtime_versions()
    if runtime_identity is not None:
        # Module versions are the runtime actually initialized; prefer them to
        # metadata in case an embedding application has altered import paths.
        runtime_versions["jax"] = runtime_identity["jax"]
        runtime_versions["jaxlib"] = runtime_identity["jaxlib"]
    report_payload["runtime_versions"] = runtime_versions

    store = paths.foldjax_home()
    usage = _shutil.disk_usage(store if store.exists() else Path.cwd())
    report_payload["disk_free_bytes"] = usage.free
    compile_cache_path = paths.compile_cache_dir()
    compile_cache = cache_snapshot(compile_cache_path).summary()
    report_payload["compile_cache"] = {
        "path": str(compile_cache_path),
        **compile_cache,
    }

    models_payload = []
    for name in available_models():
        info = model_info(name)
        input_readiness = _input_readiness(info)
        models_payload.append(
            {
                "model": name,
                "weights_ready": info.weights_ready,
                "setup": info.setup,
                "runtime_ready": info.runtime.ready,
                "runtime_setup": info.runtime.setup,
                "weight_profiles": _weight_profile_readiness(info),
                "input_readiness": input_readiness,
            }
        )
    report_payload["models"] = models_payload
    report_payload["raw_preprocess"] = [
        {"model": row["model"], "input_format": input_format, **status}
        for row in models_payload
        for input_format, status in row["input_readiness"].items()
        if status["preprocessing_runtime"] not in {"base", "precomputed"}
    ]
    report_payload["templates"] = _template_report()
    from foldjax.input import msa_search_backend

    report_payload["msa"] = msa_search_backend()

    if args.json:
        print(json.dumps(report_payload, indent=2, sort_keys=True))
        return 0

    print(f"foldjax   {report_payload['foldjax']}  python {report_payload['python']}")
    if backend_name is None:
        print(f"jax       unavailable: {report_payload.get('jax_error')}")
    else:
        # Preserve the original backend/device line for people grepping doctor
        # output; the compiler identities are an additive detail below it.
        print(f"jax       {backend_name}  {', '.join(devices) or 'no devices'}")
        print(
            f"          jax {report_payload['jax_version']}, "
            f"jaxlib {report_payload['jaxlib_version']}, "
            f"device {report_payload['device_kind']}"
        )
        if backend_name == "cpu":
            print("          every model runs on CPU, slowly; check the CUDA extra")
    print(f"store     {report_payload['home']}  ({_format_bytes(usage.free)} free)")
    print(
        f"cache     {_format_bytes(compile_cache['bytes'])} in "
        f"{compile_cache['files']} file(s)  {compile_cache_path}"
    )
    optional_versions = {
        name: version
        for name, version in runtime_versions.items()
        if name not in {"jax", "jaxlib"}
    }
    if optional_versions:
        print(
            "runtimes  "
            + ", ".join(
                f"{name} {version}" for name, version in optional_versions.items()
            )
        )
    print("\nmodels")
    for row in models_payload:
        state = "ready" if row["weights_ready"] else "missing"
        print(f"  {row['model']:<11s}weights {state}")
        if not row["weights_ready"] and row["setup"]:
            print(f"    {row['setup']}")
        if not row["runtime_ready"] and row["runtime_setup"]:
            print(f"    runtime: {row['runtime_setup']}")
        for profile in row["weight_profiles"]:
            profile_state = "ready" if profile["ready"] else "missing"
            print(
                f"    profile {profile['profile']}: {profile_state}, "
                f"downloaded {profile['downloaded']}"
            )
            if profile["reason"]:
                print(f"      {profile['reason']}")
            if profile["setup"] and profile["setup"] != row["setup"]:
                print(f"      {profile['setup']}")
        input_groups: dict[tuple[Any, ...], list[str]] = {}
        for input_format, status in row["input_readiness"].items():
            if status["ready"] and status["preprocessing_runtime"] in {
                "base",
                "precomputed",
            }:
                continue
            key = (
                status["ready"],
                status["preprocessing_runtime"],
                tuple(status["required_extras"]),
                status["reason"],
                tuple(status["setup"] or ()),
            )
            input_groups.setdefault(key, []).append(input_format)
        for key, input_formats in input_groups.items():
            ready, preprocessing_runtime, extras, reason, setup = key
            state = "ready" if ready else "missing"
            detail = str(preprocessing_runtime)
            if extras:
                detail += "; extra " + ", ".join(extras)
            print(f"    input {', '.join(input_formats)}: {state} ({detail})")
            if reason:
                print(f"      {reason}")
            for command in setup:
                print(f"      {command}")
    print("\nmsa")
    for kind, entry in report_payload["msa"].items():
        if entry["kind"] == "local":
            detail = "local  " + " ".join(entry["command"])
        elif entry["kind"] == "remote":
            detail = f"remote {entry['host']}  (sequences leave this machine)"
        else:
            detail = f"none   set {entry['setup']} to a local workflow"
        print(f"  {kind:<11s}{detail}")
    print("\ntemplates")
    for line in report_payload["templates"]:
        print(f"  {line}")
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
    with redirect_stdout(sys.stderr):
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


def _plan_summary(request: PredictionRequest) -> dict[str, Any]:
    summary = {
        "model": request.model,
        "input": str(request.input),
        "input_format": request.input_format,
        "weights": str(request.weights),
        "profile": request.profile,
        "output_dir": str(request.output_dir),
        "cache_dir": str(request.cache_dir) if request.cache_dir is not None else None,
        "seeds": list(request.resolved_seeds),
        "msa": request.msa,
        "sampling": request.sampling,
        "options": public_options(request.options),
    }
    if request.padding is not None:
        summary["padding"] = request.padding.summary()
    return summary


def _result_summary(
    result: PredictionResult | tuple[PredictionResult, ...],
) -> dict[str, Any] | list[dict[str, Any]]:
    if isinstance(result, tuple):
        return [item.summary() for item in result]
    return result.summary()


def _run_predictions(request: PredictionRequest) -> BatchReport:
    """Execute a request and report what ran, what was reused and what failed.

    The resume and error policies live on the request rather than in this
    function, so `foldjax.predict_batch(...)` and `foldjax predict --resume
    --keep-going` are the same execution -- including at seed granularity,
    which is where the expensive repetition was.
    """
    report = predict_batch(request)
    for path in report.skipped:
        print(f"[foldjax] reused finished run at {path}", file=sys.stderr)
    for failure in report.failures:
        seed = "" if failure.seed is None else f" seed {failure.seed}"
        print(
            f"foldjax: {failure.model} · {failure.input}{seed} failed: {failure.error}",
            file=sys.stderr,
        )
    return report


def _render_predictions(results: list[PredictionResult]) -> str:
    """The summary tables for everything that just ran, or a plain fallback.

    A manifest that could not be written never fails a prediction
    (`foldjax.manifest.write`), so the renderer has to cope with its absence
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # Importing or embedding the CLI for discovery/plan commands must not
    # change the host's future JAX allocator. Prediction is the only command
    # that owns a model process and therefore the only one that applies it.
    if args.command == "predict" or (
        args.command == "cache" and args.cache_command == "warm"
    ):
        _apply_mem_fraction(args.mem_fraction)
    elif args.command == "plan":
        _validate_mem_fraction(args.mem_fraction)
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
        return _run_doctor(args)
    if args.command == "cache":
        if args.cache_command == "gc":
            return _run_cache_gc(args)
        return _run_cache(args)
    if args.command == "plan":
        requested = _request(args)
        resolved = resolve_requests(requested)
        payload = [_plan_summary(item) for item in resolved]
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
    if args.command == "show":
        entries = report.read_manifests(args.path)
        if not entries:
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
        else:
            print(report.render_all(entries))
        return 0

    if not args.quiet:
        progress.enable()
    request = _request(args)
    plural = request.models is not None or request.inputs is not None
    outcome = _run_predictions(request)
    results = list(outcome.results)
    if args.json or not sys.stdout.isatty():
        summaries = [result.summary() for result in results]
        payload: Any = summaries if plural else (summaries[0] if summaries else {})
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_render_predictions(results))
    # A batch that lost some of its runs is neither a success nor the same
    # failure as one that could not start; 3 says "partial" without pretending.
    return 3 if outcome.failures else 0


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
            return f"{message}\n  install it with: uv sync --extra {extra}"
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
