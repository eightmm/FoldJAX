"""FoldJAX command-line interface."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from foldjax import assets, oom, paths
from foldjax.api import predict, resolve_request
from foldjax.registry import available_models, capabilities
from foldjax.schema import PredictionRequest


def _add_predict_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help=", ".join(available_models()))
    parser.add_argument("--input", type=Path, required=True, help="job JSON or YAML")
    parser.add_argument(
        "--weights",
        type=Path,
        help="native JAX weights; resolved from the FoldJAX weight store if omitted",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="defaults to foldjax-outputs/<input stem>",
    )
    parser.add_argument(
        "--input-format",
        default="auto",
        help="auto (default), foldjax, native, or a model's own dialect",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="run the job once per seed, into a seed_<n> directory each, and "
        "return every structure together. The samples from one seed are "
        "correlated, so this is the usual way to get independent predictions. "
        "Mutually exclusive with --seed",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        help="how many seeds to run, counting up from --seed. --num-seeds 3 is "
        "--seeds 0 1 2; mutually exclusive with --seeds",
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

    commands.add_parser("models", help="list available model backends")
    commands.add_parser("home", help="show where FoldJAX keeps its files")

    describe = commands.add_parser("capabilities", help="show what one backend accepts")
    describe.add_argument("--model", required=True)

    _add_predict_arguments(commands.add_parser("predict", help="run one prediction"))

    plan = commands.add_parser(
        "plan", help="show the resolved request without running it"
    )
    _add_predict_arguments(plan)

    setup = commands.add_parser(
        "setup", help="fetch everything public in one go, and report the rest"
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
        "--download-only",
        action="store_true",
        help="fetch the released files but skip the JAX conversion",
    )
    where = weights_commands.add_parser("path", help="print converted weights path")
    where.add_argument("--model", required=True)

    return parser


def _options(items: list[str]) -> dict[str, Any]:
    options = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"option must be KEY=VALUE: {item}")
        key, value = item.split("=", 1)
        try:
            options[key] = json.loads(value)
        except json.JSONDecodeError:
            options[key] = value
    return options


def _request(args: argparse.Namespace) -> PredictionRequest:
    return PredictionRequest(
        model=args.model,
        input=args.input,
        weights=args.weights,
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
        use_compile_cache=not args.no_cache,
        options=_options(args.option),
    )


def _report(name: str, done: int, total: int | None) -> None:
    if total:
        share = f"{100 * done / total:5.1f}%  {done / 1e6:8.1f} / {total / 1e6:.1f} MB"
    else:
        share = f"{done / 1e6:8.1f} MB"
    print(f"\r  {name:<34s} {share}", end="", file=sys.stderr, flush=True)


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
    """Fetch every public asset, and say exactly what is left to do by hand."""
    print(f"store: {paths.foldjax_home()}\n")
    print("weights")
    failed = False
    for name in assets.available():
        spec = assets.assets_for(name)
        if not spec.downloads:
            # Gated or non-redistributable: there is nothing to fetch, and the
            # instruction differs per model, so `notes` is the only honest text.
            state = "ready" if spec.ready() else "manual"
            print(f"  {name:<11s} {state}")
            if not spec.ready():
                print(f"    {spec.notes}")
                print(f"    goes in: {assets.weights_dir(name)}")
            continue
        try:
            result = assets.fetch(
                name, on_progress=_report, convert=not args.download_only
            )
        except (RuntimeError, OSError, ValueError) as error:
            print(file=sys.stderr)
            print(f"  {name:<11s} failed: {error}", file=sys.stderr)
            failed = True
            continue
        print(f"\r  {name:<11s} ready  {result}" + " " * 20)

    print("\nmsa           ready  remote MMseqs2 (ColabFold); no local database")
    print("\ntemplates")
    for line in _template_report():
        print(f"  {line}")
    return 1 if failed else 0


def _run_weights(args: argparse.Namespace) -> int:
    if args.weights_command == "list":
        for row in assets.status():
            mark = "ready" if row["converted"] else "     "
            print(
                f"{mark}  {row['model']:<9s} downloaded {row['downloaded']}  "
                f"{row['licence']}\n         {row['path']}"
            )
        return 0
    if args.weights_command == "path":
        print(assets.resolve_weights(args.model))
        return 0

    spec = assets.assets_for(args.model)
    print(f"{spec.model}: {len(spec.downloads)} file(s) from {spec.source}")
    print(f"licence: {spec.licence}")
    try:
        result = assets.fetch(
            spec.model, on_progress=_report, convert=not args.download_only
        )
    except (RuntimeError, OSError, ValueError) as error:
        # A missing or unfetchable asset is an ordinary outcome of this
        # command -- most often a model whose publisher releases the weights
        # only to applicants. It should read as an instruction, not as a
        # FoldJAX stack trace.
        print(file=sys.stderr)
        print(str(error), file=sys.stderr)
        return 1
    print(file=sys.stderr)
    print(f"ready: {result}")
    return 0


def _apply_mem_fraction(requested: float | None) -> None:
    """Set the pool fraction before anything imports JAX.

    Only here, and only for the CLI: this is a process-wide setting, and a
    library that changed it on import would be deciding for a host application
    that may be sharing the device. An explicit environment variable always wins
    -- someone who set it has a reason.
    """
    if requested is not None and not 0.0 < requested <= 1.0:
        raise ValueError(f"--mem-fraction must be in (0, 1]; got {requested}")
    if requested is not None:
        os.environ[oom.FRACTION_ENV] = str(requested)
    elif oom.FRACTION_ENV not in os.environ:
        os.environ[oom.FRACTION_ENV] = str(oom.PREDICT_MEM_FRACTION)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _apply_mem_fraction(getattr(args, "mem_fraction", None))
    if args.command == "models":
        print(*available_models(), sep="\n")
        return 0
    if args.command == "home":
        print(json.dumps(paths.describe(), indent=2, sort_keys=True))
        return 0
    if args.command == "capabilities":
        print(
            json.dumps(
                dataclasses.asdict(capabilities(args.model)), indent=2, sort_keys=True
            )
        )
        return 0
    if args.command == "setup":
        return _run_setup(args)
    if args.command == "weights":
        return _run_weights(args)
    if args.command == "plan":
        resolved = resolve_request(_request(args))
        print(
            json.dumps(
                {
                    "model": resolved.model,
                    "input": str(resolved.input),
                    "input_format": resolved.input_format,
                    "weights": str(resolved.weights),
                    "output_dir": str(resolved.output_dir),
                    "cache_dir": str(resolved.cache_dir),
                    "seeds": list(resolved.resolved_seeds),
                    "sampling": resolved.sampling,
                    "options": {k: str(v) for k, v in resolved.options.items()},
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    print(json.dumps(predict(_request(args)).summary(), indent=2, sort_keys=True))
    return 0


#: Failures that mean "you asked for something that cannot work", as opposed to
#: a bug in FoldJAX. These get one clean line; anything else keeps its traceback
#: so a real defect stays debuggable.
_USER_ERRORS = (ValueError, FileNotFoundError, NotADirectoryError, ModuleNotFoundError)


def entrypoint() -> None:
    try:
        raise SystemExit(main())
    except _USER_ERRORS as error:
        print(f"foldjax: {error}", file=sys.stderr)
        raise SystemExit(2) from None
