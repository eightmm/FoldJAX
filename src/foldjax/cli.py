"""FoldJAX command-line interface."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from foldjax import assets, oom, paths
from foldjax.api import predict, resolve_requests
from foldjax.redaction import public_options
from foldjax.registry import available_models, capabilities, model_info
from foldjax.schema import (
    PaddingConfig,
    PredictionError,
    PredictionRequest,
    PredictionResult,
)


def _add_predict_arguments(
    parser: argparse.ArgumentParser,
    *,
    allow_no_cache: bool = True,
    cache_warm: bool = False,
) -> None:
    parser.add_argument(
        "--model",
        required=True,
        nargs="+",
        help=", ".join(available_models()) + "; several run each in turn",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        nargs="+",
        help="job JSON/YAML or model-native input such as OpenFold3 feature .npz; "
        "several run every model on every input",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="model-native checkpoint or asset directory; resolved from the "
        "FoldJAX weight store if omitted",
    )
    parser.add_argument(
        "--profile",
        help="model-specific managed profile; selects matching weights and "
        "model variant (for example Protenix mini-esm-v0.5.0)",
    )
    parser.add_argument(
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
    parser.add_argument(
        "--input-format",
        default="auto",
        help="auto (default), foldjax, native, or a model's own dialect",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="representative seed (default 0)" if cache_warm else None,
    )
    parser.add_argument(
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
    parser.add_argument(
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
    parser.add_argument(
        "--num-samples", type=int, help="how many structures to generate"
    )
    parser.add_argument("--num-steps", type=int, help="diffusion steps per structure")
    parser.add_argument("--num-recycles", type=int, help="trunk recycling iterations")
    parser.add_argument(
        "--max-msa-depth",
        type=int,
        help="cap how many MSA rows the model keeps. The trunk holds a "
        "[depth, tokens, channels] representation, so this is the dominant "
        "memory knob: capping a 13k-row alignment to 1024 halved Protenix's "
        "peak at 488 tokens. Omit to keep each backend's own default",
    )
    parser.add_argument(
        "--padding",
        action="store_true",
        help="normalize every model-relevant dynamic axis to FoldJAX's standard "
        "shape buckets. Disabled by default so existing scientific results and "
        "exact-shape execution are unchanged",
    )
    parser.add_argument(
        "--pad-tokens",
        type=int,
        help="pin the padded token size for this run (also enables padding)",
    )
    parser.add_argument(
        "--pad-atoms",
        type=int,
        help="pin the padded atom size; unsupported models reject it early",
    )
    parser.add_argument(
        "--pad-msa",
        type=int,
        help="pin the padded MSA row count after max-MSA-depth is applied",
    )
    parser.add_argument(
        "--pad-templates",
        type=int,
        help="pin the padded template count for models with a template axis",
    )
    parser.add_argument(
        "--pad-structural-tokens",
        type=int,
        help="pin OpenDDE's secondary structural-token axis",
    )
    parser.add_argument(
        "--pad-language-model-tokens",
        type=int,
        help=(
            "pin the language-model sequence width for ESMFold2/ESMC or "
            "Protenix ESM/ISM"
        ),
    )
    parser.add_argument(
        "--padding-overflow",
        choices=("error", "exact"),
        help="when an automatic axis exceeds the standard grid: fail before "
        "compilation (default) or keep that exact size",
    )
    parser.add_argument(
        "--mem-fraction",
        type=float,
        help="fraction of the device JAX may preallocate. Defaults to "
        f"{oom.PREDICT_MEM_FRACTION} rather than JAX's {oom.DEFAULT_MEM_FRACTION}, "
        "because one prediction owns the process and a quarter of the card held "
        "in reserve is what stops jobs that would otherwise fit. Lower it to "
        "share the device with another process",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help=f"compile cache root (default {paths.compile_cache_dir()})",
    )
    if allow_no_cache:
        parser.add_argument(
            "--no-cache",
            action="store_true",
            help="skip the persistent compile cache (slower, but writes nothing)",
        )
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="native option passed straight to the backend; repeatable",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foldjax",
        description="Biomolecular structure prediction in JAX: one job file in, "
        "structures and confidence out.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    models = commands.add_parser("models", help="list available model backends")
    models.add_argument(
        "--json",
        action="store_true",
        help="include weight readiness, supported inputs, and execution options",
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
    runtime_commands = runtime.add_subparsers(
        dest="runtime_command", required=True
    )
    runtime_status = runtime_commands.add_parser(
        "status", help="report whether one model can start without preparation"
    )
    runtime_status.add_argument("--model", required=True)
    runtime_prepare = runtime_commands.add_parser(
        "prepare", help="prepare one model's generated native artifacts"
    )
    runtime_prepare.add_argument("--model", required=True)

    describe = commands.add_parser("capabilities", help="show what one backend accepts")
    describe.add_argument("--model", required=True)

    _add_predict_arguments(commands.add_parser("predict", help="run one prediction"))

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

    cache = commands.add_parser(
        "cache", help="warm the persistent JAX compilation cache"
    )
    cache_commands = cache.add_subparsers(dest="cache_command", required=True)
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


def _request(args: argparse.Namespace) -> PredictionRequest:
    single_model = len(args.model) == 1
    single_input = len(args.input) == 1
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
        input=args.input[0] if single_input else None,
        inputs=None if single_input else tuple(args.input),
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

    def progress(self, name: str, done: int, total: int | None) -> None:
        if sys.stderr.isatty():
            _report(name, done, total)
            self._progress_active = True
        # Non-interactive logs use the structured start/done events. Emitting
        # byte callbacks too would duplicate every completed download line.

    def event(self, event: assets.AssetEvent) -> None:
        self.finish_progress()
        item = f"  {event.item}" if event.item else ""
        elapsed = (
            ""
            if event.elapsed_seconds is None
            else f"  {event.elapsed_seconds:.2f}s"
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
    for name in assets.available():
        spec = assets.assets_for(name)
        if not spec.in_default_setup:
            state = "ready" if spec.ready() else "opt-in"
            print(f"  {name:<11s} {state}")
            if not spec.ready():
                print(f"    fetch: foldjax weights fetch --model {name}")
                print(f"    {spec.notes}")
            continue
        if not spec.downloads:
            # Gated or non-redistributable: the instruction differs per model,
            # so `notes` is the only honest text.
            state = "ready" if spec.ready() else "manual"
            print(f"  {name:<11s} {state}")
            if not spec.ready():
                print(f"    {spec.notes}")
                print(f"    goes in: {assets.weights_dir(name)}")
            continue
        try:
            reporter = _WeightReporter()
            result = assets.fetch(
                name,
                on_progress=reporter.progress,
                on_event=reporter.event,
                convert=not args.download_only,
            )
        except (RuntimeError, OSError, ValueError) as error:
            reporter.finish_progress()
            print(f"  {name:<11s} failed: {error}", file=sys.stderr)
            failed = True
            continue
        reporter.finish_progress()
        print(f"\r  {name:<11s} ready  {result}" + " " * 20)

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


def _runtime_payload(name: str) -> dict[str, Any]:
    info = model_info(name)
    return {"model": info.model, **info.runtime.summary()}


def _run_runtime(args: argparse.Namespace) -> int:
    """Inspect or explicitly prepare runtime artifacts without hidden work."""
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
        os.environ[oom.FRACTION_ENV] = str(requested)
    elif oom.FRACTION_ENV not in os.environ:
        os.environ[oom.FRACTION_ENV] = str(oom.PREDICT_MEM_FRACTION)


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
    if args.command == "cache":
        return _run_cache(args)
    if args.command == "plan":
        requested = _request(args)
        resolved = resolve_requests(requested)
        payload = [_plan_summary(item) for item in resolved]
        print(
            json.dumps(
                payload if requested.models is not None or requested.inputs is not None
                else payload[0],
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    print(
        json.dumps(
            _result_summary(predict(_request(args))), indent=2, sort_keys=True
        )
    )
    return 0


#: Failures that mean "you asked for something that cannot work", as opposed to
#: a bug in FoldJAX. These get one clean line; anything else keeps its traceback
#: so a real defect stays debuggable.
_USER_ERRORS = (
    PredictionError,
    MemoryError,
    ValueError,
    FileNotFoundError,
    NotADirectoryError,
    ModuleNotFoundError,
)


def entrypoint() -> None:
    try:
        raise SystemExit(main())
    except _USER_ERRORS as error:
        print(f"foldjax: {error}", file=sys.stderr)
        raise SystemExit(2) from None
