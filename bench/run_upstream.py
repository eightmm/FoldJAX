"""Measure the same prediction from the model's own upstream repository.

Each upstream ships its own virtualenv and its own entry point, so every run is
a subprocess into that environment. Two things make the comparison controlled
rather than approximate:

* The job file is produced by FoldJAX's own input translator, which emits each
  upstream's native dialect. Both sides therefore read the same job, naming the
  same alignment file, rather than two hand-written inputs that are believed to
  match.
* Peak device memory is read the same way on both sides -- the live-bytes
  high-water mark, `max_memory_allocated` here and `peak_bytes_in_use` on the
  JAX side -- via `upstream_peak.py`, injected as `sitecustomize` so no upstream
  file is modified.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/jaemin/non-project/optimizing")
ASSETS = Path.home() / ".cache/foldjax/assets"
COMPONENTS = ASSETS / "components.cif"
RDKIT_CACHE = ASSETS / "components.cif.rdkit_mol.pkl"


def _cuda_home(repo: Path) -> Path | None:
    """The pip-installed CUDA toolkit inside ``repo``'s virtualenv, if any.

    torch ships the CUDA *runtime* as `nvidia/cu13`, but not the compiler, so
    an extension built on first use fails with "CUDA_HOME environment variable
    is not set". Installing `nvidia-cuda-nvcc` and `nvidia-cuda-cccl` at the
    same CUDA version into that virtualenv completes the toolkit in place,
    without touching the system or the other environments.
    """
    for lib in sorted(repo.glob(".venv/lib/python3.*/site-packages/nvidia/cu13")):
        if (lib / "bin" / "nvcc").is_file():
            return lib
    return None


def _toolkit_env(repo: Path) -> dict[str, str]:
    """CUDA_HOME and a PATH that finds nvcc, when the toolkit is present."""
    home = _cuda_home(repo)
    if home is None:
        # No compiler: fall back to the pure-torch layer norm so the run
        # happens at all, and let the report note it is not upstream's default.
        return {"LAYERNORM_TYPE": "torch"}
    return {"CUDA_HOME": str(home), "_BENCH_CUDA_BIN": str(home / "bin")}


def _protenix_family(
    repo: Path, checkpoint_flag: list[str], job: Path, out: Path, schedule, seed
) -> list[str]:
    """Protenix and OpenDDE share a config system and a runner layout."""
    return [
        str(repo / ".venv/bin/python"),
        "-m",
        "runner.inference",
        "--input_json_path",
        str(job),
        "--dump_dir",
        str(out),
        "--seeds",
        str(seed),
        "--model.N_cycle",
        str(schedule["num_recycles"]),
        "--sample_diffusion.N_step",
        str(schedule["num_steps"]),
        "--sample_diffusion.N_sample",
        str(schedule["num_samples"]),
        "--data.ccd_components_file",
        str(COMPONENTS),
        "--data.ccd_components_rdkit_mol_file",
        str(RDKIT_CACHE),
        *checkpoint_flag,
    ]


def command(model: str, job: Path, out: Path, schedule: dict, seed: int):
    """Return (argv, cwd, extra_env) for one upstream prediction."""
    if model == "boltz2":
        repo = ROOT / "boltz"
        return (
            [
                str(repo / ".venv/bin/boltz"),
                "predict",
                str(job),
                "--out_dir",
                str(out),
                "--cache",
                str(Path.home() / ".boltz"),
                "--model",
                "boltz2",
                "--recycling_steps",
                str(schedule["num_recycles"]),
                "--sampling_steps",
                str(schedule["num_steps"]),
                "--diffusion_samples",
                str(schedule["num_samples"]),
                "--seed",
                str(seed),
                "--accelerator",
                "gpu",
                "--devices",
                "1",
                "--output_format",
                "mmcif",
            ],
            repo,
            {},
        )
    if model == "protenix":
        repo = ROOT / "protenix"
        return (
            _protenix_family(
                repo,
                [
                    "--model_name",
                    "protenix_base_default_v1.0.0",
                    "--load_checkpoint_dir",
                    str(Path.home() / "protenix/checkpoint"),
                ],
                job,
                out,
                schedule,
                seed,
            ),
            repo,
            # Upstream's default layer norm is a CUDA extension compiled on
            # first use, so it needs a CUDA toolkit. There is no system one
            # here, but a matching nvcc/CCCL can be installed into the same
            # virtualenv as the CUDA runtime torch already ships -- see
            # `_cuda_home`. With that, Protenix runs its own fused kernel and
            # its time is a real measurement rather than a fallback's.
            {"PYTHONPATH": str(repo), **_toolkit_env(repo)},
        )
    if model == "opendde":
        repo = ROOT / "OpenDDE"
        return (
            _protenix_family(
                repo,
                [
                    "--load_checkpoint_path",
                    str(
                        Path.home()
                        / ".cache/foldjax/downloads/opendde/opendde.pt"
                    ),
                ],
                job,
                out,
                schedule,
                seed,
            ),
            repo,
            {"PYTHONPATH": str(repo)},
        )
    if model == "chai":
        repo = ROOT / "chai"
        script = Path(__file__).parent / "_chai_upstream.py"
        return (
            [
                str(repo / ".venv/bin/python"),
                str(script),
                "--fasta",
                str(job),
                # FoldJAX's translator writes Chai's sha256-addressed
                # `.aligned.pqt` here, from the same .a3m the other three read.
                "--msa-dir",
                str(job.parent / "chai_msa"),
                "--out",
                str(out),
                "--recycles",
                str(schedule["num_recycles"]),
                "--steps",
                str(schedule["num_steps"]),
                "--samples",
                str(schedule["num_samples"]),
                "--seed",
                str(seed),
            ],
            repo,
            {"CHAI_DOWNLOADS_DIR": str(repo / "downloads")},
        )
    raise ValueError(f"no upstream runner for {model}")


def scores(model: str, out: Path) -> list[dict]:
    """Per-sample confidence, read from whatever that upstream writes."""
    found: list[dict] = []
    if model == "boltz2":
        for path in sorted(out.rglob("confidence_*_model_*.json")):
            body = json.loads(path.read_text())
            found.append(
                {
                    key: float(body[key])
                    for key in ("confidence_score", "ptm", "iptm", "complex_plddt")
                    if isinstance(body.get(key), (int, float))
                }
            )
    elif model == "chai":
        import numpy as np

        for path in sorted(out.rglob("scores.model_idx_*.npz")):
            with np.load(path) as data:
                found.append(
                    {
                        key: float(np.asarray(data[key]).reshape(-1)[0])
                        for key in ("aggregate_score", "ptm", "iptm")
                        if key in data
                    }
                )
    else:
        for path in sorted(out.rglob("*_summary_confidence_sample_*.json")):
            body = json.loads(path.read_text())
            found.append(
                {
                    key: float(body[key])
                    for key in ("plddt", "ptm", "iptm", "ranking_score", "gpde")
                    if isinstance(body.get(key), (int, float))
                }
            )
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--job", type=Path, required=True, help="native input file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()

    from bench.spec import SCHEDULE, SEED, cases

    case = next(item for item in cases() if item.name == args.case)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    peak_file = args.output_dir / "peak_bytes.txt"

    argv, cwd, extra = command(
        args.model, args.job, args.output_dir, SCHEDULE, SEED
    )
    environment = dict(os.environ)
    environment.update(extra)
    environment["BENCH_PEAK_FILE"] = str(peak_file)
    # A directory holding nothing but `sitecustomize.py`, so putting it first
    # installs the peak-memory hook without shadowing anything the upstream
    # imports.
    hook = str(Path(__file__).parent / "peakhook")
    environment["PYTHONPATH"] = (
        f"{hook}:{environment['PYTHONPATH']}" if "PYTHONPATH" in environment else hook
    )
    # Invoking `.venv/bin/python` directly does not put that venv's `bin` on
    # PATH the way activating it would, so tools installed alongside the
    # interpreter are invisible. Protenix needs `ninja` there: its fused
    # layer-norm is a CUDA extension compiled on first use, and without it the
    # run dies at import. Running upstream on its pure-torch fallback instead
    # would be measuring a different thing than upstream's own default.
    venv_bin = str(Path(argv[0]).parent)
    # nvcc has to be found by name from inside the build, so its directory goes
    # on PATH alongside the venv's own bin (which is where ninja lives).
    cuda_bin = environment.pop("_BENCH_CUDA_BIN", None)
    prefix = f"{cuda_bin}:{venv_bin}" if cuda_bin else venv_bin
    environment["PATH"] = f"{prefix}:{environment.get('PATH', '')}"

    start = time.perf_counter()
    completed = subprocess.run(
        argv, cwd=cwd, env=environment, timeout=args.timeout,
        capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - start

    peak = 0.0
    if peak_file.is_file():
        peak = float(peak_file.read_text().strip() or 0) / 2**20

    found = scores(args.model, args.output_dir)
    record = {
        "impl": "upstream",
        "model": args.model,
        "case": case.name,
        "length": case.length,
        "schedule": dict(SCHEDULE),
        "seed": SEED,
        "wall_s": round(elapsed, 2),
        "peak_mib": round(peak, 1),
        "returncode": completed.returncode,
        "samples": [{"scores": entry} for entry in found],
    }
    if completed.returncode != 0:
        record["stderr_tail"] = completed.stderr[-2000:]
    elif not found:
        # An exit status of 0 is not evidence that anything was predicted.
        # OpenDDE's runner catches a CUDA OOM, writes it to an ERR/ file, and
        # returns 0 -- at 976 tokens it asked for 40.8 GiB on top of 58.5 GiB
        # already held and died, and without this the row would be published as
        # a 12-second run that beat everything else.
        record["failed"] = True
        record["reason"] = "exited 0 but produced no structures"
        errors = sorted(args.output_dir.rglob("ERR/*.txt"))
        if errors:
            record["stderr_tail"] = errors[0].read_text(errors="replace")[-2000:]
    text = json.dumps(record, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
