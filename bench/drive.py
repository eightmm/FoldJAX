"""Run the comparison matrix, one measurement per process, results as they land.

Each row is written to its own JSON file the moment it completes, so a matrix
that is interrupted -- or a model that fails on the largest case -- still leaves
every row that did finish. Nothing is held in memory across runs.

Between runs the card is allowed to go idle. A process exiting does not hand
its device memory back synchronously, so a job started immediately after a
heavy one can fail an allocation that would otherwise fit, and report it as the
model being too large.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from bench.provenance import (
    CURRENT_RESULT_SCHEMA,
    artifact_identity,
    benchmark_identity,
    device_identity,
    execution_identity,
    foldjax_checkpoint_paths,
    foldjax_effective_environment,
    foldjax_implicit_asset_paths,
    portable_options,
    redact_machine_paths,
    reusable_result_file,
    runtime_identity,
    source_identity,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _redacted_diagnostic(text: str, *, work: Path, results: Path) -> str:
    return redact_machine_paths(
        text,
        {
            "benchmark-work": work,
            "benchmark-results": results,
            "foldjax-repo": REPO,
            "home": Path.home(),
        },
    )


def gpu_idle(threshold_mib: int = 2048, timeout_s: int = 300, settled: int = 3) -> None:
    """Wait for the previous run's memory to be back, and to stay back.

    One reading below the threshold is not enough. A process that has exited
    releases its memory to the driver over some interval, and `nvidia-smi` can
    report a small figure while the release is still in flight; the next run
    then asks for its pool -- 0.9 of the card, under the shipped setting -- and
    cannot get it. Requiring several consecutive quiet readings costs seconds
    and removes a failure that is indistinguishable from the model being too
    large for the card.

    Giving up is reported rather than silent: a run started on a busy card is
    still a measurement, but not of the thing the table claims.
    """
    waited = 0
    quiet = 0
    while waited < timeout_s:
        try:
            query = [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ]
            used = (
                subprocess.run(query, capture_output=True, text=True, timeout=30)
                .stdout.strip()
                .splitlines()
            )
        except (OSError, subprocess.SubprocessError):
            return
        if not used:
            return
        if int(used[0]) <= threshold_mib:
            quiet += 1
            if quiet >= settled:
                return
        else:
            quiet = 0
        time.sleep(5)
        waited += 5
    print(
        f"[warn] card still above {threshold_mib} MiB after {timeout_s}s; "
        "starting anyway",
        flush=True,
    )


def native_input(model: str, case, destination: Path) -> Path:
    """The job file this upstream reads, produced by FoldJAX's own translator.

    Both sides therefore run the same job naming the same alignment, rather
    than two inputs that are believed to say the same thing.
    """
    from foldjax.input import materialize_native_input
    from foldjax.registry import capabilities

    # Bench labels are (weights, schedule) pairs; FoldJAX's registry is keyed by
    # backend. Protenix's two supported checkpoints are two rows and one
    # backend, and they read the same input dialect, so the label resolves to
    # the backend before the translator is asked what it accepts.
    backend = "protenix" if model == "protenix-v2" else model
    return materialize_native_input(
        case.job, capabilities(backend), destination, seed=101
    )


def _comparison_options(model: str) -> list[str]:
    """`--option` flags that keep a row comparable with its upstream column.

    A model whose FoldJAX default has moved away from what upstream ships is
    pinned back here. Without this the table would quietly compare two
    precisions, which is not a comparison -- and it would do it silently,
    because a default is exactly the thing nobody passes.
    """
    from bench.spec import COMPARISON_OPTIONS

    pinned = COMPARISON_OPTIONS.get(model, {})
    return [
        flag
        for name, value in pinned.items()
        for flag in ("--option", f"{name}={value}")
    ]


def _expected_identity(
    model: str,
    impl: str,
    case,
    work: Path,
    *,
    expected_upstream_diff_sha256: str | None = None,
    traced: bool = False,
) -> dict[str, object]:
    """Resolve the same portable identity that the measured child will emit."""

    from bench.spec import COMPARISON_OPTIONS, SCHEDULE, SEED

    base_environment = dict(os.environ)
    base_environment["PYTHONPATH"] = str(REPO)
    if traced:
        hook = str(HERE / "peakhook")
        base_environment["PYTHONPATH"] = f"{hook}:{base_environment['PYTHONPATH']}"

    if impl == "foldjax":
        import foldjax
        from foldjax.schema import PredictionRequest

        backend, profile = model, None
        if model == "protenix-v2":
            backend, profile = "protenix", "v2"
        request = foldjax.resolve_request(
            PredictionRequest(
                model=backend,
                input=case.job,
                output_dir=work,
                seed=SEED,
                options=COMPARISON_OPTIONS.get(model, {}),
                profile=profile,
                **SCHEDULE,
            )
        )
        assert request.weights is not None
        options = COMPARISON_OPTIONS.get(model, {})
        artifacts = artifact_identity(
            job=Path(request.input),
            checkpoints=foldjax_checkpoint_paths(backend, Path(request.weights)),
            implicit_assets=foldjax_implicit_asset_paths(
                backend,
                Path(request.weights),
                options=request.options,
            ),
        )
        # Resolving some backends installs effective in-process defaults (for
        # example AlphaFold's bundled libcifpp data directory). Mirror those
        # mutations, while preserving the child PYTHONPATH/tracing projection.
        resolved_environment = dict(os.environ)
        resolved_environment["PYTHONPATH"] = base_environment["PYTHONPATH"]
        environment = foldjax_effective_environment(
            resolved_environment,
            model=backend,
            options=request.options,
        )
        return benchmark_identity(
            impl=impl,
            model=model,
            case=case.name,
            length=case.length,
            schedule=SCHEDULE,
            seed=SEED,
            options=portable_options(options),
            artifacts=artifacts,
            source=source_identity(REPO),
            runtime=runtime_identity(),
            device=device_identity(environment),
            execution=execution_identity(
                environment,
                timing_state="warm-after-successful-prefill",
                traced=traced,
            ),
        )

    from bench.run_upstream import (
        command,
        upstream_checkpoint_paths,
        upstream_environment,
        upstream_git_provenance,
        upstream_implicit_asset_paths,
        upstream_runtime_versions,
    )

    job = native_input(model, case, work / "input")
    argv, cwd, extra = command(model, job, work / "out", SCHEDULE, SEED)
    environment = upstream_environment(
        argv,
        extra,
        base_environment=base_environment,
        peak_file=work / "out/peak_bytes.txt",
    )
    # Review the checkout before importing any of its code to resolve dynamic
    # runtime-selected assets (currently Boltz's canonical molecule inventory).
    upstream_git = upstream_git_provenance(
        cwd,
        expected_diff_sha256=expected_upstream_diff_sha256,
    )
    upstream_runtime = upstream_runtime_versions(cwd)
    artifacts = artifact_identity(
        job=case.job,
        native_input=job,
        checkpoints=upstream_checkpoint_paths(model),
        implicit_assets=upstream_implicit_asset_paths(model, native_input=job),
    )
    return benchmark_identity(
        impl=impl,
        model=model,
        case=case.name,
        length=case.length,
        schedule=SCHEDULE,
        seed=SEED,
        artifacts=artifacts,
        source=source_identity(REPO),
        runtime=runtime_identity(),
        device=device_identity(environment),
        execution=execution_identity(
            environment,
            timing_state="cold-or-unspecified",
            traced=traced,
        ),
        upstream_git=upstream_git,
        upstream_runtime=upstream_runtime,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--impls", nargs="+", default=["foldjax", "upstream"])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument(
        "--upstream-patch",
        action="append",
        default=[],
        metavar="MODEL=SHA256",
        help="permit exactly one reviewed tracked diff for an upstream model; "
        "the digest is recorded in that result",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="reuse only successful current-schema rows with the same identity",
    )
    parser.add_argument(
        "--traces",
        type=Path,
        help="also record device memory over time, one file per measurement. "
        "The warm-up run is never traced: it is discarded work, and tracing it "
        "would put a compile-bound curve next to the ones being compared.",
    )
    args = parser.parse_args()

    upstream_patches: dict[str, str] = {}
    for entry in args.upstream_patch:
        try:
            model_name, digest = entry.split("=", 1)
        except ValueError:
            parser.error("--upstream-patch must be MODEL=SHA256")
        digest = digest.lower()
        if not model_name or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            parser.error("--upstream-patch must be MODEL followed by 64 hex digits")
        if model_name in upstream_patches:
            parser.error(f"duplicate --upstream-patch for {model_name}")
        upstream_patches[model_name] = digest

    sys.path.insert(0, str(REPO))
    from bench.spec import MODELS, REIMPLEMENTED
    from bench.spec import cases as all_cases

    unknown_models = sorted(set(args.models) - set(MODELS))
    if unknown_models:
        parser.error(f"unknown benchmark model(s): {', '.join(unknown_models)}")
    unknown_impls = sorted(set(args.impls) - {"foldjax", "upstream"})
    if unknown_impls:
        parser.error(f"unknown benchmark implementation(s): {', '.join(unknown_impls)}")
    unsupported_upstreams = sorted(
        set(args.models) - set(REIMPLEMENTED) if "upstream" in args.impls else ()
    )
    if unsupported_upstreams:
        parser.error(
            "no upstream benchmark runner for: " + ", ".join(unsupported_upstreams)
        )

    args.results.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    lookup = {case.name: case for case in all_cases()}
    unknown_cases = sorted(set(args.cases) - set(lookup))
    if unknown_cases:
        parser.error(f"unknown benchmark case(s): {', '.join(unknown_cases)}")

    for case_name in args.cases:
        case = lookup[case_name]
        for model in args.models:
            for impl in args.impls:
                row = args.results / f"{model}-{impl}-{case_name}.json"
                work = args.work / f"{model}-{impl}-{case_name}"
                if args.skip_existing and (row.exists() or row.is_symlink()):
                    try:
                        identity_before = _expected_identity(
                            model,
                            impl,
                            case,
                            work,
                            expected_upstream_diff_sha256=upstream_patches.get(model),
                            traced=args.traces is not None,
                        )
                    except (OSError, RuntimeError, ValueError) as error:
                        parser.error(str(error))
                    reusable = reusable_result_file(
                        row,
                        expected_identity=identity_before,
                    )
                    if reusable:
                        try:
                            identity_after = _expected_identity(
                                model,
                                impl,
                                case,
                                work,
                                expected_upstream_diff_sha256=upstream_patches.get(
                                    model
                                ),
                                traced=args.traces is not None,
                            )
                        except (OSError, RuntimeError, ValueError) as error:
                            parser.error(str(error))
                        if identity_after == identity_before and reusable_result_file(
                            row,
                            expected_identity=identity_after,
                        ):
                            print(f"[skip] {row.name}", flush=True)
                            continue
                    print(
                        f"[redo] {row.name}: legacy, failed, or identity changed",
                        flush=True,
                    )
                # A failed subprocess may not reach its JSON write. Remove a
                # prior row before starting so the failure handler cannot
                # mistake stale evidence for the run that just failed.
                row.unlink(missing_ok=True)
                subprocess.run(["rm", "-rf", str(work)], check=False)
                work.mkdir(parents=True, exist_ok=True)
                gpu_idle()
                print(f"[run ] {model} {impl} {case_name}", flush=True)

                # The interpreter this driver is running under, not a fixed
                # `.venv/bin/python`: a hard-coded path silently ignores
                # `UV_PROJECT_ENVIRONMENT`, so a matrix aimed at a second
                # environment measures the first one and says nothing.
                if impl == "foldjax":
                    # Fill the compile cache first and throw the run away. A
                    # cold JAX run is dominated by XLA compilation, which the
                    # torch side has no equivalent of, so timing one against
                    # the other measures the compiler rather than the model.
                    # A compatible readable cache entry may avoid recompiling;
                    # cache rejection/deserialization failure can still pay it.
                    warm = args.work / f"{model}-{impl}-{case_name}-warmup"
                    subprocess.run(["rm", "-rf", str(warm)], check=False)
                    warm.mkdir(parents=True, exist_ok=True)
                    print(f"[warm] {model} {case_name}", flush=True)
                    warm_started = time.perf_counter()
                    warm_argv = [
                        sys.executable,
                        "-m",
                        "bench.run_foldjax",
                        "--model",
                        model,
                        "--case",
                        case_name,
                        "--output-dir",
                        str(warm),
                        "--warmup",
                        *_comparison_options(model),
                    ]
                    try:
                        filled = subprocess.run(
                            warm_argv,
                            cwd=REPO,
                            capture_output=True,
                            text=True,
                            env={**os.environ, "PYTHONPATH": str(REPO)},
                            timeout=args.timeout + 600,
                        )
                    except subprocess.TimeoutExpired as error:
                        captured = error.stderr or ""
                        stderr = (
                            captured.decode(errors="replace")
                            if isinstance(captured, bytes)
                            else captured
                        )
                        filled = subprocess.CompletedProcess(
                            warm_argv,
                            124,
                            stdout="",
                            stderr=stderr,
                        )
                    except OSError as error:
                        filled = subprocess.CompletedProcess(
                            warm_argv,
                            127,
                            stdout="",
                            stderr=str(error),
                        )
                    # A warm-up that died is not a neutral event: the run it was
                    # supposed to prepare now pays compilation inside the
                    # measured time, and whatever killed it is likely to kill
                    # that run too. Say so, and keep the reason.
                    if filled.returncode != 0:
                        log = args.results / f"{model}-{impl}-{case_name}.warmup.txt"
                        log.write_text(filled.stderr or "")
                        failure = {
                            "schema": CURRENT_RESULT_SCHEMA,
                            "impl": impl,
                            "model": model,
                            "case": case_name,
                            "length": case.length,
                            "failed": True,
                            "reason": "warm-up failed; measurement not started",
                            "driver_returncode": filled.returncode,
                            "wall_s": round(time.perf_counter() - warm_started, 2),
                            "samples": [],
                            "stderr_bytes": len(filled.stderr or ""),
                            "stderr_path": log.name,
                            "stderr_tail": _redacted_diagnostic(
                                (filled.stderr or "")[-3000:],
                                work=args.work,
                                results=args.results,
                            ),
                        }
                        row.write_text(json.dumps(failure, sort_keys=True) + "\n")
                        print(
                            f"[FAIL] warm-up exited {filled.returncode}; "
                            f"reason in {log.name}",
                            flush=True,
                        )
                    subprocess.run(["rm", "-rf", str(warm)], check=False)
                    if filled.returncode != 0:
                        continue
                    gpu_idle()
                    argv = [
                        sys.executable,
                        "-m",
                        "bench.run_foldjax",
                        "--model",
                        model,
                        "--case",
                        case_name,
                        "--output-dir",
                        str(work),
                        "--json-out",
                        str(row),
                        "--timing-state",
                        "warm-after-successful-prefill",
                        *_comparison_options(model),
                    ]
                else:
                    job = native_input(model, case, work / "input")
                    argv = [
                        sys.executable,
                        "-m",
                        "bench.run_upstream",
                        "--model",
                        model,
                        "--case",
                        case_name,
                        "--job",
                        str(job),
                        "--output-dir",
                        str(work / "out"),
                        "--json-out",
                        str(row),
                        "--timeout",
                        str(args.timeout),
                        "--timing-state",
                        "cold-or-unspecified",
                    ]
                    if digest := upstream_patches.get(model):
                        argv.extend(("--expected-upstream-diff-sha256", digest))

                if args.traces:
                    argv.append("--traced")
                    # A wrapper, not a different runner: the measurement below
                    # is byte-for-byte the one an untraced matrix runs, so a
                    # trace and its table row can never be of different things.
                    argv = [
                        sys.executable,
                        "-m",
                        "bench.trace",
                        "run",
                        "--model",
                        model,
                        "--impl",
                        impl,
                        "--case",
                        case_name,
                        "--out",
                        str(args.traces / f"{model}-{impl}-{case_name}.jsonl"),
                        "--",
                        *argv,
                    ]

                started = time.perf_counter()
                try:
                    completed = subprocess.run(
                        argv,
                        cwd=REPO,
                        capture_output=True,
                        text=True,
                        env={**os.environ, "PYTHONPATH": str(REPO)},
                        timeout=args.timeout + 600,
                    )
                except subprocess.TimeoutExpired as error:
                    captured = error.stderr or ""
                    stderr = (
                        captured.decode(errors="replace")
                        if isinstance(captured, bytes)
                        else captured
                    )
                    completed = subprocess.CompletedProcess(
                        argv,
                        124,
                        stdout="",
                        stderr=stderr,
                    )
                except OSError as error:
                    completed = subprocess.CompletedProcess(
                        argv,
                        127,
                        stdout="",
                        stderr=str(error),
                    )
                if completed.returncode != 0 or not row.is_file():
                    # Keep all of it beside the row, not just the tail. An XLA
                    # allocation failure ends in a stack dump long enough to
                    # fill any tail on its own, so the line that says how many
                    # bytes were asked for -- the only part that identifies the
                    # failure -- is exactly what a truncated tail loses.
                    log = row.with_suffix(".stderr.txt")
                    log.write_text(completed.stderr or "")
                    if row.is_file():
                        # `run_upstream` writes its detailed provenance failure
                        # before returning 2. Preserve that evidence instead of
                        # replacing it with a generic driver error.
                        try:
                            failure = json.loads(row.read_text())
                        except (OSError, json.JSONDecodeError):
                            failure = {}
                    else:
                        failure = {}
                    failure.setdefault("schema", CURRENT_RESULT_SCHEMA)
                    failure.setdefault("impl", impl)
                    failure.setdefault("model", model)
                    failure.setdefault("case", case_name)
                    failure.setdefault("length", case.length)
                    failure.setdefault("failed", True)
                    failure.setdefault(
                        "wall_s", round(time.perf_counter() - started, 2)
                    )
                    failure.setdefault("reason", "benchmark subprocess failed")
                    failure["driver_returncode"] = completed.returncode
                    failure["stderr_bytes"] = len(completed.stderr or "")
                    failure["stderr_path"] = log.name
                    if completed.stderr:
                        failure["stderr_tail"] = _redacted_diagnostic(
                            completed.stderr[-3000:],
                            work=args.work,
                            results=args.results,
                        )
                    row.write_text(json.dumps(failure, sort_keys=True) + "\n")
                    print(f"[FAIL] {model} {impl} {case_name}", flush=True)
                else:
                    print(f"[ ok ] {row.read_text().strip()[:200]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
