"""Measure an external AlphaFold 3 checkout as a common-JAX reference.

This is not an independent-port speed comparison: FoldJAX's external-source
route executes the same AlphaFold 3 source.  It records a fresh child process
with that source, the locally licensed parameters, and (when supplied) the
same prebuilt cpp/CCD runtime used by the FoldJAX environment.

The optional supplied-MSA arm skips external database search while publisher
model featurisation and inference still run in the child process.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from bench.provenance import (
    CURRENT_RESULT_SCHEMA,
    ArtifactFingerprintError,
    _alphafold3_external_asset_paths,
    artifact_identity,
    benchmark_identity,
    device_identity,
    execution_identity,
    require_unchanged,
    runtime_identity,
    source_identity,
)
from bench.run_foldjax_cli import _read_peak, cli_environment, run_cli_child
from bench.run_upstream import produced_structures, upstream_runtime_versions

_BUCKETS = "256,512,768,1024,1280,1536,2048,2560,3072,3584,4096,4608,5120"
_STEPS = 200
_BOOTSTRAP = """
import importlib.util
import os
import runpy
import sys
from pathlib import Path

source, runtime, runner, *argv = map(Path, sys.argv[1:])
sys.path.insert(0, str(source / "src"))
import alphafold3
if not Path(alphafold3.__file__).resolve().is_relative_to(source / "src"):
    raise RuntimeError("external AlphaFold 3 source did not win import resolution")
os.environ["LIBCIFPP_DATA_DIR"] = str(runtime.parent / "share" / "libcifpp")
cpp = next(runtime.glob("cpp.*.so"))
spec = importlib.util.spec_from_file_location("alphafold3.cpp", cpp)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)
sys.modules["alphafold3.cpp"] = module
alphafold3.cpp = module
from alphafold3.common import resources
resources.ROOT = runtime
resources._DATA_ROOT = runtime
sys.argv = [str(runner), *(str(value) for value in argv)]
runpy.run_path(str(runner), run_name="__main__")
"""


def _provided_msa(protein: dict, path: Path, *, field: str, path_field: str) -> None:
    if field in protein and path_field in protein:
        raise ValueError(
            f"--native-input protein entries cannot set both {field} and {path_field}"
        )
    inline, external = protein.get(field), protein.get(path_field)
    if field in protein and isinstance(inline, str):
        return
    if path_field in protein and isinstance(external, str) and external:
        candidate = (path.parent / external).resolve()
        try:
            with candidate.open(encoding="utf-8"):
                return
        except OSError:
            pass
    raise ValueError(
        f"--native-input protein entries require a {field} string or "
        f"a readable {path_field}"
    )


def _provided_protein_inputs(document: dict, path: Path) -> None:
    sequences = document.get("sequences")
    if not isinstance(sequences, list):
        raise ValueError("--native-input sequences must be a list")
    for sequence in sequences:
        protein = sequence.get("protein") if isinstance(sequence, dict) else None
        if protein is None:
            continue
        if not isinstance(protein, dict):
            raise ValueError("--native-input protein entry must be an object")
        _provided_msa(
            protein,
            path,
            field="unpairedMsa",
            path_field="unpairedMsaPath",
        )
        _provided_msa(
            protein,
            path,
            field="pairedMsa",
            path_field="pairedMsaPath",
        )
        if not isinstance(protein.get("templates"), list):
            raise ValueError("--native-input protein entries require a templates list")


def _native_input(path: Path, *, require_provided: bool = False) -> Path:
    """Validate the native dialect and, for supplied-MSA arms, its inputs."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "--native-input must be a readable AlphaFold 3 JSON file"
        ) from error
    if (
        not isinstance(document, dict)
        or "entities" in document
        or "sequences" not in document
    ):
        raise ValueError(
            "--native-input must use the AlphaFold 3 sequences dialect, not the "
            "common entities dialect"
        )
    if require_provided:
        _provided_protein_inputs(document, path)
    return path


def _native_seed(path: Path) -> int:
    document = json.loads(path.read_text(encoding="utf-8"))
    seeds = document.get("modelSeeds") if isinstance(document, dict) else None
    if not isinstance(seeds, list) or len(seeds) != 1 or type(seeds[0]) is not int:
        raise ValueError(
            "--native-input must contain exactly one integer modelSeeds value"
        )
    return seeds[0]


def _parameter_paths(weights: Path) -> dict[str, Path]:
    """Bind exactly the parameter files selected by AlphaFold 3's loader."""
    from foldjax.backends.alphafold3 import _selected_parameter_files

    selected = _selected_parameter_files(weights)
    if not selected:
        raise ArtifactFingerprintError("cannot identify one AlphaFold 3 parameter set")
    return {f"parameter.{index}": path for index, path in enumerate(selected)}


def _assets(
    source: Path, common_runtime: Path | None, environment: dict[str, str]
) -> dict[str, Path]:
    selected = _alphafold3_external_asset_paths(source, environment=environment)
    if common_runtime is not None:
        cpp = sorted(common_runtime.glob("cpp.*.so"))
        if len(cpp) != 1:
            raise ArtifactFingerprintError(
                "common AlphaFold 3 runtime has no unique cpp extension"
            )
        for path in sorted(common_runtime.rglob("*")):
            if path.is_file():
                selected[
                    "alphafold3.common_runtime."
                    + path.relative_to(common_runtime).as_posix()
                ] = path
    return selected


def native_argv(args) -> list[str]:
    runner = args.source / "run_alphafold.py"
    flags = [
        "--json_path",
        str(args.native_input),
        "--output_dir",
        str(args.output_dir),
        "--model_dir",
        str(args.weights),
        "--jax_compilation_cache_dir",
        str(args.cache_dir),
        "--run_data_pipeline="
        + ("false" if getattr(args, "skip_database_search", False) else "true"),
        "--run_inference=true",
        "--jax_backend=gpu",
        "--gpu_device=0",
        f"--buckets={_BUCKETS}",
        "--flash_attention_implementation=triton",
        f"--num_diffusion_samples={args.num_samples}",
        f"--num_recycles={args.num_recycles}",
    ]
    if args.common_runtime is None:
        return [str(args.python), str(runner), *flags]
    return [
        str(args.python),
        "-c",
        _BOOTSTRAP,
        str(args.source),
        str(args.common_runtime),
        str(runner),
        *flags,
    ]


def _scores(sample_dir: Path, ranking_score: float) -> dict[str, float] | None:
    found = {"ranking_score": ranking_score}
    for path in sample_dir.rglob("*.json"):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(document, dict):
            found.update(
                {
                    key: float(value)
                    for key, value in document.items()
                    if isinstance(value, (int, float)) and math.isfinite(value)
                }
            )
    return found if all(math.isfinite(value) for value in found.values()) else None


def _samples(output_dir: Path) -> list[dict]:
    rows = []
    for ranking in sorted(output_dir.rglob("*_ranking_scores.csv")):
        try:
            with ranking.open(newline="", encoding="utf-8") as stream:
                rows.extend(csv.DictReader(stream))
        except (OSError, csv.Error):
            return []
    samples = []
    seen = set()
    for row in rows:
        try:
            seed, index = int(row["seed"]), int(row["sample"])
            score = float(row["ranking_score"])
        except (KeyError, TypeError, ValueError):
            return []
        if not math.isfinite(score):
            return []
        candidates = sorted(
            path
            for path in output_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".cif", ".mmcif", ".pdb"}
            and f"seed-{seed}_sample-{index}" in path.parts
        )
        if len(candidates) != 1 or candidates[0].resolve() in seen:
            return []
        structure = candidates[0].resolve()
        seen.add(structure)
        scores = _scores(structure.parent, score)
        if not scores:
            return []
        samples.append(
            {
                "seed": seed,
                "sample": index,
                "structure_path": str(structure.relative_to(output_dir.resolve())),
                "scores": scores,
            }
        )
    return samples


def _fresh_output(
    parser: argparse.ArgumentParser, output: Path, json_out: Path | None
) -> None:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--output-dir must be fresh and empty")
    if json_out is not None and (
        json_out.exists() or json_out.resolve().is_relative_to(output.resolve())
    ):
        parser.error("--json-out must name a new file outside --output-dir")
    output.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--native-input", type=Path, required=True)
    parser.add_argument("--common-job", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--common-runtime", type=Path)
    parser.add_argument("--skip-database-search", action="store_true")
    parser.add_argument("--num-recycles", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument(
        "--timing-state",
        choices=("cold-or-unspecified", "warm-after-successful-prefill"),
        default="cold-or-unspecified",
    )
    args = parser.parse_args()
    if min(args.timeout, args.num_recycles, args.num_samples) < 1:
        parser.error("--timeout, --num-recycles, and --num-samples must be positive")
    args.source = args.source.resolve()
    args.weights = args.weights.resolve()
    args.native_input = _native_input(
        args.native_input.resolve(), require_provided=args.skip_database_search
    )
    try:
        seed = _native_seed(args.native_input)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    if args.length < 1:
        parser.error("--length must be positive")
    args.common_job = args.common_job.resolve()
    if not args.common_job.is_file():
        parser.error("--common-job must be a readable common benchmark job")
    if (
        not (args.source / "run_alphafold.py").is_file()
        or not (args.source / "src/alphafold3/__init__.py").is_file()
    ):
        parser.error("--source must be an AlphaFold 3 checkout")
    _fresh_output(parser, args.output_dir, args.json_out)
    if args.common_runtime is not None:
        args.common_runtime = args.common_runtime.resolve()
        if not any(args.common_runtime.glob("cpp.*.so")):
            parser.error(
                "--common-runtime must contain the prebuilt AlphaFold 3 cpp extension"
            )
    peak_file = args.output_dir / "peak_bytes.txt"
    environment = cli_environment(peak_file)
    if args.common_runtime is not None:
        environment["LIBCIFPP_DATA_DIR"] = str(
            args.common_runtime.parent / "share" / "libcifpp"
        )
    source_python = str(args.source / "src")
    environment["PYTHONPATH"] = f"{source_python}:{environment['PYTHONPATH']}"
    schedule = {
        "num_samples": args.num_samples,
        "num_steps": _STEPS,
        "num_recycles": args.num_recycles,
    }
    try:
        checkpoints = _parameter_paths(args.weights)
        assets = _assets(args.source, args.common_runtime, environment)
        artifacts = artifact_identity(
            job=args.common_job,
            native_input=args.native_input,
            checkpoints=checkpoints,
            implicit_assets=assets,
        )
        source = source_identity(Path(__file__).resolve().parent.parent)
        parent_runtime = runtime_identity()
        runtime = upstream_runtime_versions(args.source, python=args.python)
        device = device_identity(environment)
        execution = execution_identity(
            environment, timing_state=args.timing_state, traced=False
        )
    except ArtifactFingerprintError as error:
        parser.error(str(error))
    identity = benchmark_identity(
        impl="alphafold3-common-jax-reference",
        model="alphafold3",
        case=args.case,
        length=args.length,
        schedule=schedule,
        seed=seed,
        options={
            "source": "external",
            "common_runtime_bootstrap": args.common_runtime is not None,
            "attention_backend": "triton",
            "data_pipeline": (
                "disabled_with_supplied_native_msa_templates"
                if args.skip_database_search
                else "publisher_database_search_pipeline"
            ),
            "native_featurization": "publisher_featurise_input",
        },
        artifacts=artifacts,
        source=source,
        runtime=runtime,
        device=device,
        execution=execution,
    )
    argv = native_argv(args)
    returncode, stdout, stderr, elapsed, launch_error = run_cli_child(
        argv, environment, args.timeout
    )
    (args.output_dir / "cli_stdout.txt").write_text(stdout, encoding="utf-8")
    (args.output_dir / "cli_stderr.txt").write_text(stderr, encoding="utf-8")
    postflight_error = None
    try:
        require_unchanged(
            artifacts,
            artifact_identity(
                job=args.common_job,
                native_input=args.native_input,
                checkpoints=checkpoints,
                implicit_assets=_assets(args.source, args.common_runtime, environment),
            ),
        )
        require_unchanged(
            source, source_identity(Path(__file__).resolve().parent.parent)
        )
        require_unchanged(
            runtime, upstream_runtime_versions(args.source, python=args.python)
        )
        require_unchanged(device, device_identity(environment))
        require_unchanged(
            execution,
            execution_identity(
                environment, timing_state=args.timing_state, traced=False
            ),
        )
    except (ArtifactFingerprintError, RuntimeError) as error:
        postflight_error = str(error)
    peak = _read_peak(peak_file)
    samples = _samples(args.output_dir)
    record = {
        "schema": CURRENT_RESULT_SCHEMA,
        "identity": identity,
        "impl": "alphafold3-common-jax-reference",
        "comparison_scope": "common_jax_runtime_reference_not_independent_port_speedup",
        "model": "alphafold3",
        "case": args.case,
        "length": args.length,
        "seed": seed,
        "schedule": schedule,
        "steps_fixed_to_publisher_default": _STEPS,
        "artifacts": artifacts,
        "source": source,
        "runtime": runtime,
        "parent_runtime": parent_runtime,
        "device": device,
        "execution": execution,
        "argv": argv,
        "timing_scope": "cli_subprocess",
        "wall_s": round(elapsed, 2),
        "peak_mib": None if peak is None else round(peak, 1),
        "returncode": returncode,
        "samples": samples,
        "structure_files": [sample["structure_path"] for sample in samples],
    }
    if postflight_error:
        record.update(
            failed=True,
            reason="benchmark provenance changed during prediction",
            provenance_error=postflight_error,
        )
    elif returncode != 0:
        record.update(
            failed=True,
            reason=launch_error or f"AlphaFold 3 exited {returncode}",
            stderr_tail=stderr[-2000:],
        )
    elif peak is None:
        record.update(
            failed=True, reason="AlphaFold 3 produced no JAX peak observer result"
        )
    elif len(samples) != args.num_samples or not produced_structures(args.output_dir):
        record.update(
            failed=True, reason="AlphaFold 3 outputs do not match requested samples"
        )
    text = json.dumps(record, allow_nan=False, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 1 if record.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
