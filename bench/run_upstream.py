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
import ast
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

from bench.provenance import (
    CURRENT_RESULT_SCHEMA,
    ArtifactFingerprintError,
    artifact_identity,
    benchmark_identity,
    device_identity,
    execution_identity,
    fingerprint_path,
    redact_machine_paths,
    require_unchanged,
    reusable_result,
    runtime_identity,
    source_identity,
)

DEFAULT_UPSTREAM_ROOT = Path(__file__).resolve().parents[2]


def _upstream_root() -> Path:
    """Sibling-checkout root, portable by default and explicitly overridable."""

    configured = os.environ.get("FOLDJAX_BENCH_ROOT")
    return (
        Path(configured).expanduser().resolve() if configured else DEFAULT_UPSTREAM_ROOT
    )


def _git(repo: Path, *args: str, binary: bool = False) -> str | bytes:
    """Run one non-interactive provenance query in an upstream checkout."""

    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=not binary,
    )
    if completed.returncode != 0:
        stderr = completed.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        raise RuntimeError(f"cannot inspect upstream checkout {repo}: {stderr.strip()}")
    return completed.stdout


_IGNORED_EXECUTION_SUFFIXES = {
    ".bash",
    ".c",
    ".cc",
    ".cfg",
    ".conf",
    ".cpp",
    ".cu",
    ".cuh",
    ".dll",
    ".dylib",
    ".h",
    ".hpp",
    ".ini",
    ".json",
    ".ptx",
    ".py",
    ".pyd",
    ".pyi",
    ".sh",
    ".so",
    ".toml",
    ".yaml",
    ".yml",
}
_IGNORED_EXCLUDED_DIRECTORIES = {
    ".cache",
    ".git",
    ".mypy_cache",
    ".nox",
    ".oms",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".tool-cache",
    ".venv",
    "__pycache__",
    "node_modules",
    "tool-cache",
}
_MAX_IGNORED_ARTIFACTS = 4096
_MAX_IGNORED_ARTIFACT_BYTES = 256 << 20
_MAX_IGNORED_SINGLE_BYTES = 128 << 20
_MAX_SCANNED_ENTRIES = 100_000
_SECRET_NAME = re.compile(
    r"(^|[._-])(access[-_]?token|api[-_]?key|auth[-_]?token|credentials?|passwd|password|secrets?)([._-]|$)",
    re.IGNORECASE,
)


def _ignored_execution_candidates(repo: Path) -> list[Path]:
    """Find ignored execution-shaped entries, including specials git omits."""

    candidates: list[Path] = []
    scanned = 0

    def visit(directory: Path, prefix: Path) -> None:
        nonlocal scanned
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as error:
            raise RuntimeError(f"cannot inspect upstream checkout {repo}") from error
        for entry in entries:
            scanned += 1
            if scanned > _MAX_SCANNED_ENTRIES:
                raise RuntimeError("upstream checkout exceeds provenance scan cap")
            relative = prefix / entry.name
            try:
                metadata = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            is_directory = stat.S_ISDIR(metadata.st_mode)
            if is_directory and entry.name in _IGNORED_EXCLUDED_DIRECTORIES:
                continue
            secret_like = _SECRET_NAME.search(entry.name) or entry.name.startswith(
                ".env"
            )
            relevant = (
                secret_like
                or relative.suffix.lower() in _IGNORED_EXECUTION_SUFFIXES
                or stat.S_ISREG(metadata.st_mode)
                and bool(metadata.st_mode & 0o111)
                or not (stat.S_ISREG(metadata.st_mode) or is_directory)
            )
            if relevant:
                candidates.append(relative)
            if is_directory:
                visit(Path(entry.path), relative)

    visit(repo, Path())
    if not candidates:
        return []
    encoded = b"\0".join(os.fsencode(path) for path in candidates) + b"\0"
    completed = subprocess.run(
        ["git", "-C", str(repo), "check-ignore", "-z", "--stdin"],
        input=encoded,
        capture_output=True,
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(f"cannot inspect ignored files in upstream checkout {repo}")
    return sorted(
        (Path(os.fsdecode(item)) for item in completed.stdout.split(b"\0") if item),
        key=lambda path: path.as_posix(),
    )


def _ignored_runtime_artifacts(repo: Path) -> list[dict[str, object]]:
    """Fingerprint ignored in-tree code/config that can change execution.

    Virtualenv and exact tool-cache directories are represented by runtime
    versions. Other ignored source, executable, native, and configuration
    files can be imported or read by the runner, so their bytes are inputs.
    """

    artifacts: list[dict[str, object]] = []
    total_bytes = 0
    for relative in _ignored_execution_candidates(repo):
        if any(part in _IGNORED_EXCLUDED_DIRECTORIES for part in relative.parts):
            continue
        path = repo / relative
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        if _SECRET_NAME.search(relative.name) or relative.name.startswith(".env"):
            raise RuntimeError(
                "ignored execution config has a secret-like name and cannot be "
                f"safely recorded: {relative.as_posix()}"
            )
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(
                "ignored runtime artifact is a symlink and cannot be pinned by "
                f"target name alone: {path}; replace it with a regular file"
            )
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(
                f"ignored runtime artifact is not a regular file: {path}"
            )
        if metadata.st_size > _MAX_IGNORED_SINGLE_BYTES:
            raise RuntimeError(
                f"ignored execution artifact exceeds the provenance cap: {relative}"
            )
        if len(artifacts) >= _MAX_IGNORED_ARTIFACTS:
            raise RuntimeError("too many ignored execution artifacts to fingerprint")
        total_bytes += metadata.st_size
        if total_bytes > _MAX_IGNORED_ARTIFACT_BYTES:
            raise RuntimeError("ignored execution artifacts exceed provenance cap")
        try:
            fingerprint = fingerprint_path(path, allow_directory=False)
        except ArtifactFingerprintError as error:
            raise RuntimeError(str(error)) from error
        artifacts.append(
            {
                "path": relative.as_posix(),
                "kind": "file",
                "size": fingerprint["bytes"],
                "sha256": fingerprint["sha256"],
            }
        )
    return sorted(artifacts, key=lambda item: str(item["path"]))


def upstream_git_provenance(
    repo: Path,
    *,
    expected_diff_sha256: str | None = None,
) -> dict[str, object]:
    """Pin the exact tracked upstream source or refuse an ambiguous run."""

    if not repo.is_dir():
        raise RuntimeError(f"upstream checkout does not exist: {repo}")
    revision = str(_git(repo, "rev-parse", "--verify", "HEAD^{commit}")).strip()
    status = str(
        _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    ).rstrip("\n")
    status_lines = status.splitlines()
    untracked = [line[3:] for line in status_lines if line.startswith("?? ")]
    if untracked:
        preview = ", ".join(untracked[:5])
        if len(untracked) > 5:
            preview += f", ... ({len(untracked)} total)"
        raise RuntimeError(
            f"upstream checkout {repo} has untracked files: {preview}; move, "
            "remove, or intentionally ignore them before benchmarking"
        )
    status_lines = [line for line in status_lines if not line.startswith("?? ")]
    status = "\n".join(status_lines)
    dirty = bool(status)
    digest = None
    if dirty:
        patch = bytes(
            _git(
                repo,
                "diff",
                "--binary",
                "--no-ext-diff",
                "HEAD",
                "--",
                binary=True,
            )
        )
        digest = hashlib.sha256(patch).hexdigest()
        if expected_diff_sha256 is None:
            raise RuntimeError(
                f"upstream checkout {repo} has tracked changes; rerun only after "
                f"reviewing them and passing --expected-upstream-diff-sha256={digest}"
            )
        if expected_diff_sha256.lower() != digest:
            raise RuntimeError(
                "upstream tracked diff does not match the expected SHA-256: "
                f"expected {expected_diff_sha256.lower()}, found {digest}"
            )
    elif expected_diff_sha256 is not None:
        raise RuntimeError(
            "--expected-upstream-diff-sha256 was supplied, but the upstream "
            "checkout has no tracked changes"
        )
    return {
        "revision": revision,
        "tracked_dirty": dirty,
        "tracked_diff_sha256": digest,
        "tracked_status": status.splitlines(),
        "ignored_runtime_artifacts": _ignored_runtime_artifacts(repo),
    }


def upstream_runtime_versions(repo: Path) -> dict[str, object]:
    """Read environment versions without importing either tensor runtime."""

    python = repo / ".venv/bin/python"
    if not python.is_file():
        raise RuntimeError(f"upstream virtualenv is missing its Python: {python}")
    script = """
import importlib.metadata as metadata
import json
import platform

names = (
    "torch", "triton", "boltz", "protenix", "opendde", "openfold3", "biotite",
    "numpy", "scipy", "rdkit", "biopython", "pyyaml", "omegaconf",
    "hydra-core", "lightning", "pytorch-lightning",
    "cuequivariance", "cuequivariance-torch",
    "cuequivariance-ops-torch-cu12", "cuequivariance-ops-torch-cu13",
    "cuda-python", "nvidia-cuda-runtime", "nvidia-cuda-nvcc",
    "nvidia-cuda-cccl", "nvidia-cuda-crt", "nvidia-nvvm",
    "nvidia-cuda-runtime-cu12", "nvidia-cuda-nvcc-cu12",
    "nvidia-cuda-cccl-cu12", "nvidia-cuda-crt-cu12",
    "nvidia-cuda-runtime-cu13", "nvidia-cuda-nvcc-cu13",
    "nvidia-cuda-cccl-cu13", "nvidia-cuda-crt-cu13",
)
versions = {}
for name in names:
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        pass
print(json.dumps({
    "python": {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
    },
    "distributions": versions,
}))
"""
    completed = subprocess.run(
        [str(python), "-c", script],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"cannot inspect upstream runtime {python}: {completed.stderr.strip()}"
        )
    return json.loads(completed.stdout)


def _store() -> Path:
    """The FoldJAX store, wherever it currently is.

    These paths were `~/.cache/foldjax/...`, which is where the store lived
    until it moved inside the checkout on 2026-08-06. The assets happened to
    survive at the old location, so only the OpenDDE checkpoint went missing --
    and the harness reported that as a completed row with a peak of 0.0 MiB
    rather than as a failure, which reads in a table as "the model is free".
    Ask the package where its store is instead of naming it here.
    """
    from foldjax.paths import foldjax_home

    return foldjax_home()


ASSETS = _store() / "assets"
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


def upstream_checkpoint_paths(model: str) -> dict[str, Path]:
    """Exact checkpoint payload selected by one supported upstream command."""

    root = _upstream_root()
    if model == "boltz2":
        return {"model": Path.home() / ".boltz" / "boltz2_conf.ckpt"}
    if model in {"protenix", "protenix-v2"}:
        name = (
            "protenix-v2" if model == "protenix-v2" else "protenix_base_default_v1.0.0"
        )
        return {"model": Path.home() / "protenix" / "checkpoint" / f"{name}.pt"}
    if model == "openfold3":
        return {
            "model": root
            / "openfold3"
            / "openfold3_weights"
            / "checkpoints"
            / "of3_ft3_v1.pt"
        }
    if model == "opendde":
        return {"model": _store() / "downloads" / "opendde" / "opendde.pt"}
    raise ValueError(f"no upstream checkpoint for {model}")


def _upstream_boltz_canonical_tokens() -> tuple[str, ...]:
    """Read the exact canonical molecule inventory selected by upstream Boltz."""

    source = _upstream_root() / "boltz/src/boltz/data/const.py"
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    except (OSError, SyntaxError) as error:
        raise RuntimeError("cannot inspect upstream Boltz canonical tokens") from error

    definitions: list[tuple[ast.AST, ast.expr | None]] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            matching = [
                target
                for target in node.targets
                if isinstance(target, ast.Name) and target.id == "canonical_tokens"
            ]
            definitions.extend((target, node.value) for target in matching)
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "canonical_tokens"
        ):
            definitions.append((node.target, node.value))

    if len(definitions) != 1:
        raise RuntimeError(
            "upstream Boltz canonical tokens must have one literal definition"
        )
    target, expression = definitions[0]
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or node.id != "canonical_tokens":
            continue
        if node is target:
            continue
        parent = parents.get(node)
        grandparent = parents.get(parent) if parent is not None else None
        if not (
            isinstance(node.ctx, ast.Load)
            and isinstance(parent, ast.Starred)
            and isinstance(grandparent, (ast.List, ast.Tuple, ast.Set))
        ):
            raise RuntimeError(
                "upstream Boltz canonical tokens have unsupported mutation or use"
            )
    try:
        value = ast.literal_eval(expression)
    except (ValueError, TypeError) as error:
        raise RuntimeError(
            "upstream Boltz canonical tokens are not a literal sequence"
        ) from error

    if not isinstance(value, (list, tuple)) or not value:
        raise RuntimeError("upstream Boltz canonical tokens are unavailable")
    tokens = tuple(value)
    if (
        any(
            not isinstance(token, str)
            or re.fullmatch(r"[A-Z0-9]+", token) is None
            for token in tokens
        )
        or len(set(tokens)) != len(tokens)
    ):
        raise RuntimeError("upstream Boltz canonical tokens are invalid")
    _require_upstream_boltz_runtime_inventory(source, tokens)
    return tokens


def _require_upstream_boltz_runtime_inventory(
    expected_source: Path,
    expected_tokens: tuple[str, ...],
) -> None:
    """Bind the venv import used by the console script to the reviewed source."""

    repo = _upstream_root() / "boltz"
    python = repo / ".venv/bin/python"
    script = """
import json
from boltz.data import const

print(json.dumps({
    "source": const.__file__,
    "tokens": const.canonical_tokens,
}, sort_keys=True))
"""
    try:
        completed = subprocess.run(
            [str(python), "-c", script],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            "cannot inspect the upstream Boltz runtime inventory"
        ) from error
    if completed.returncode != 0:
        raise RuntimeError("cannot inspect the upstream Boltz runtime inventory")
    try:
        payload = json.loads(completed.stdout)
        actual_source = Path(payload["source"]).resolve(strict=True)
        actual_tokens = tuple(payload["tokens"])
        reviewed_source = expected_source.resolve(strict=True)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("invalid upstream Boltz runtime inventory") from error
    if actual_source != reviewed_source or actual_tokens != expected_tokens:
        raise RuntimeError(
            "upstream Boltz runtime inventory does not match the reviewed checkout"
        )


def _require_boltz_protein_benchmark(native_input: Path | None) -> None:
    """Limit the narrow asset inventory to our generated protein-only jobs."""

    if native_input is None:
        raise ArtifactFingerprintError(
            "upstream Boltz asset selection requires the generated native input"
        )
    try:
        document = json.loads(Path(native_input).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactFingerprintError(
            "upstream Boltz benchmark input is not generated JSON"
        ) from error
    if (
        not isinstance(document, dict)
        or set(document) != {"version", "sequences"}
        or type(document.get("version")) is not int
        or document["version"] != 1
        or not isinstance(document.get("sequences"), list)
        or not document["sequences"]
    ):
        raise ArtifactFingerprintError(
            "upstream Boltz provenance supports only pure unmodified protein jobs"
        )
    for entity in document["sequences"]:
        if not isinstance(entity, dict) or set(entity) != {"protein"}:
            raise ArtifactFingerprintError(
                "upstream Boltz provenance supports only pure unmodified protein jobs"
            )
        protein = entity["protein"]
        if (
            not isinstance(protein, dict)
            or set(protein) != {"id", "sequence", "msa"}
            or not isinstance(protein["sequence"], str)
            or not protein["sequence"].strip()
            or not isinstance(protein["msa"], str)
            or not protein["msa"].strip()
        ):
            raise ArtifactFingerprintError(
                "upstream Boltz provenance supports only pure unmodified protein jobs"
            )


def _upstream_biotite_ccd(repo: Path) -> Path:
    """Resolve the mutable Biotite CCD selected inside one upstream venv."""

    script = """
import json
from biotite.structure.info import ccd

print(json.dumps({"ccd": str(ccd._CCD_FILE)}, sort_keys=True))
"""
    try:
        completed = subprocess.run(
            [str(repo / ".venv/bin/python"), "-c", script],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("cannot inspect the upstream Biotite CCD") from error
    if completed.returncode != 0:
        raise RuntimeError("cannot inspect the upstream Biotite CCD")
    try:
        payload = json.loads(completed.stdout)
        ccd = Path(payload["ccd"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("invalid upstream Biotite CCD selection") from error
    if not ccd.is_absolute():
        raise RuntimeError("upstream Biotite CCD selection is not absolute")
    return ccd


def upstream_implicit_asset_paths(
    model: str,
    *,
    native_input: Path | None = None,
) -> dict[str, Path]:
    """Exact non-checkpoint assets selected by each upstream command."""

    if model == "boltz2":
        _require_boltz_protein_benchmark(native_input)
        cache = upstream_checkpoint_paths(model)["model"].parent
        for name, role in (
            ("mols.tar", "molecule archive"),
            ("boltz2_aff.ckpt", "affinity checkpoint"),
        ):
            try:
                metadata = (cache / name).lstat()
            except OSError as error:
                raise ArtifactFingerprintError(
                    f"upstream Boltz benchmark requires a pre-existing {role}"
                ) from error
            if not stat.S_ISREG(metadata.st_mode):
                raise ArtifactFingerprintError(
                    f"upstream Boltz benchmark requires a regular {role}"
                )
        mols = cache / "mols"
        return {
            f"boltz.canonical-protein.{token}": mols / f"{token}.pkl"
            for token in _upstream_boltz_canonical_tokens()
        }
    if model in {"protenix", "protenix-v2", "opendde"}:
        return {
            "ccd.components": COMPONENTS,
            "ccd.rdkit_cache": RDKIT_CACHE,
        }
    if model == "openfold3":
        return {
            "openfold3.runner_yaml": Path(__file__).parent / "openfold3_runner.yml",
            "openfold3.biotite_components": _upstream_biotite_ccd(
                _upstream_root() / "openfold3-v031"
            ),
        }
    raise ValueError(f"no upstream assets for {model}")


def command(model: str, job: Path, out: Path, schedule: dict, seed: int):
    """Return (argv, cwd, extra_env) for one upstream prediction."""
    root = _upstream_root()
    if model == "boltz2":
        repo = root / "boltz"
        checkpoint = upstream_checkpoint_paths(model)["model"]
        return (
            [
                str(repo / ".venv/bin/boltz"),
                "predict",
                str(job),
                "--out_dir",
                str(out),
                "--cache",
                str(checkpoint.parent),
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
    if model in {"protenix", "protenix-v2"}:
        repo = root / "protenix"
        # Upstream resolves the architecture from the name: `model_name` picks
        # the entry in configs_model_type, which for v2 widens c_z to 256 and
        # turns on hidden_scale_up before the checkpoint is loaded. The bench's
        # schedule flags are applied after that merge and still win.
        upstream_name = (
            "protenix-v2" if model == "protenix-v2" else "protenix_base_default_v1.0.0"
        )
        checkpoint = upstream_checkpoint_paths(model)["model"]
        return (
            _protenix_family(
                repo,
                [
                    "--model_name",
                    upstream_name,
                    "--load_checkpoint_dir",
                    str(checkpoint.parent),
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
    if model == "openfold3":
        # The 0.3.1 worktree, not the main checkout: `of3_ft3_v1.pt` -- the
        # weights the FoldJAX port runs -- is the legacy p1 checkpoint with
        # `version_compatibility="<0.4"` (entry_points/parameters.py). The
        # current 0.4.4 code refuses it (layer_norm_z / fourier_emb drift),
        # and running the >=0.4 default p2 weights instead would put two
        # different models in one table row.
        repo = root / "openfold3-v031"
        checkpoint = upstream_checkpoint_paths(model)["model"]
        return (
            [
                # `python -m`: 0.3.1's console script trips over a cutlass
                # import shim and presents the wrong argparse; the module
                # entry is the same `cli`. Underscore flags only in 0.3.1.
                str(repo / ".venv/bin/python"),
                "-m",
                "openfold3.run_openfold",
                "predict",
                "--query_json",
                str(job),
                "--inference_ckpt_path",
                str(checkpoint),
                "--num_diffusion_samples",
                str(schedule["num_samples"]),
                "--runner_yaml",
                str(Path(__file__).parent / "openfold3_runner.yml"),
                "--use_msa_server",
                "false",
                "--use_templates",
                "false",
                "--output_dir",
                str(out),
            ],
            repo,
            {},
        )
    if model == "opendde":
        repo = root / "OpenDDE"
        checkpoint = upstream_checkpoint_paths(model)["model"]
        return (
            _protenix_family(
                repo,
                [
                    "--load_checkpoint_path",
                    str(checkpoint),
                ],
                job,
                out,
                schedule,
                seed,
            ),
            repo,
            {"PYTHONPATH": str(repo)},
        )
    raise ValueError(f"no upstream runner for {model}")


def upstream_environment(
    argv: list[str],
    extra: Mapping[str, str],
    *,
    base_environment: Mapping[str, str] | None = None,
    peak_file: Path | None = None,
) -> dict[str, str]:
    """Build the exact effective subprocess environment for an upstream run."""

    environment = dict(os.environ if base_environment is None else base_environment)
    environment.update(extra)
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if peak_file is not None:
        environment["BENCH_PEAK_FILE"] = str(peak_file)
    hook = str(Path(__file__).parent / "peakhook")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{hook}:{existing_pythonpath}" if existing_pythonpath else hook
    )
    venv_bin = str(Path(argv[0]).parent)
    cuda_bin = environment.pop("_BENCH_CUDA_BIN", None)
    prefix = f"{cuda_bin}:{venv_bin}" if cuda_bin else venv_bin
    environment["PATH"] = f"{prefix}:{environment.get('PATH', '')}"
    return environment


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
    elif model == "openfold3":
        # `<case>/seed_<n>/
        #    <case>_seed_<n>_sample_<k>_confidences_aggregated.json`;
        # the runner YAML supplies the requested seed directly. Do not pass
        # ``--num_model_seeds``: pinned 0.3.1 interprets that as a request to
        # discard the YAML list and generate seeds from its hard-coded 42.
        # `ptm` is the key the FoldJAX column also writes, so the report can
        # compare like with like; `avg_plddt` is 0-100 where FoldJAX's
        # `mean_plddt` is 0-1, so both ride along unrenamed rather than
        # pretending to be the same number.
        for path in sorted(out.rglob("*_confidences_aggregated.json")):
            body = json.loads(path.read_text())
            found.append(
                {
                    key: float(body[key])
                    for key in (
                        "ptm",
                        "iptm",
                        "avg_plddt",
                        "gpde",
                        "disorder",
                        "sample_ranking_score",
                    )
                    if isinstance(body.get(key), (int, float))
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


def produced_structures(out: Path) -> list[Path]:
    """Nonempty structure files written by an upstream prediction."""

    found: list[Path] = []
    for path in sorted(out.rglob("*")):
        if path.suffix.lower() not in {".cif", ".mmcif", ".pdb"}:
            continue
        try:
            metadata = path.lstat()
        except OSError:
            continue
        if stat.S_ISREG(metadata.st_mode) and metadata.st_size > 0:
            found.append(path)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--job", type=Path, required=True, help="native input file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument(
        "--timing-state",
        choices=("cold-or-unspecified", "warm-after-successful-prefill"),
        default="cold-or-unspecified",
    )
    parser.add_argument("--traced", action="store_true")
    parser.add_argument(
        "--expected-upstream-diff-sha256",
        help="required exact git-diff SHA-256 when the selected upstream has "
        "tracked changes; clean checkouts must omit it",
    )
    args = parser.parse_args()

    from bench.spec import SCHEDULE, SEED, cases

    case = next(item for item in cases() if item.name == args.case)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    peak_file = args.output_dir / "peak_bytes.txt"

    argv, cwd, extra = command(args.model, args.job, args.output_dir, SCHEDULE, SEED)
    environment = upstream_environment(argv, extra, peak_file=peak_file)
    diagnostic_paths = {
        "native-input": args.job.parent,
        "output": args.output_dir,
        "upstream-checkout": cwd,
        "foldjax-repo": Path(__file__).resolve().parent.parent,
        "home": Path.home(),
    }

    def portable_diagnostic(message: str) -> str:
        return redact_machine_paths(message, diagnostic_paths)

    try:
        # Review the checkout before importing any of its code to resolve
        # runtime-selected assets (currently Boltz's canonical inventory).
        upstream_git = upstream_git_provenance(
            cwd,
            expected_diff_sha256=args.expected_upstream_diff_sha256,
        )
        upstream_runtime = upstream_runtime_versions(cwd)
        checkpoint_paths = upstream_checkpoint_paths(args.model)
        implicit_asset_paths = upstream_implicit_asset_paths(
            args.model,
            native_input=args.job,
        )
        artifacts = artifact_identity(
            job=case.job,
            native_input=args.job,
            checkpoints=checkpoint_paths,
            implicit_assets=implicit_asset_paths,
        )
        source = source_identity(Path(__file__).resolve().parent.parent)
        runtime = runtime_identity()
        device = device_identity(environment)
        execution = execution_identity(
            environment,
            timing_state=args.timing_state,
            traced=args.traced,
        )
    except (ArtifactFingerprintError, RuntimeError) as error:
        parser.error(str(error))
    identity = benchmark_identity(
        impl="upstream",
        model=args.model,
        case=case.name,
        length=case.length,
        schedule=SCHEDULE,
        seed=SEED,
        artifacts=artifacts,
        source=source,
        runtime=runtime,
        device=device,
        execution=execution,
        upstream_git=upstream_git,
        upstream_runtime=upstream_runtime,
    )
    # Near-capacity torch runs die on fragmentation, not on totals: Boltz-2 at
    # 3,012 tokens filled 81.3 GiB of a 95.6 GiB card and then failed a batch
    # allocation the free total easily covered. Expandable segments let the
    # caching allocator grow regions instead of hunting for contiguous blocks;
    # an allocator setting, not an upstream code change.
    start = time.perf_counter()
    launch_error: str | None = None
    stderr = ""
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            env=environment,
            timeout=args.timeout,
            capture_output=True,
            text=True,
        )
        returncode = completed.returncode
        stderr = completed.stderr
    except subprocess.TimeoutExpired as error:
        returncode = 124
        captured = error.stderr or ""
        stderr = (
            captured.decode(errors="replace")
            if isinstance(captured, bytes)
            else captured
        )
        launch_error = f"upstream prediction exceeded {args.timeout}s timeout"
    except OSError as error:
        returncode = 127
        launch_error = f"cannot launch upstream prediction: {error}"
        stderr = str(error)
    elapsed = time.perf_counter() - start

    postflight_errors: list[str] = []
    try:
        postflight = upstream_git_provenance(
            cwd,
            expected_diff_sha256=upstream_git["tracked_diff_sha256"],
        )
        if postflight != upstream_git:
            postflight_errors.append(
                "upstream checkout changed while prediction was running: "
                f"before={upstream_git!r}, after={postflight!r}"
            )
    except RuntimeError as error:
        postflight_errors.append(str(error))
    try:
        postflight_runtime = upstream_runtime_versions(cwd)
        if postflight_runtime != upstream_runtime:
            postflight_errors.append(
                "upstream runtime changed while prediction was running: "
                f"before={upstream_runtime!r}, after={postflight_runtime!r}"
            )
    except RuntimeError as error:
        postflight_errors.append(str(error))
    try:
        require_unchanged(
            artifacts,
            artifact_identity(
                job=case.job,
                native_input=args.job,
                checkpoints=checkpoint_paths,
                implicit_assets=upstream_implicit_asset_paths(
                    args.model,
                    native_input=args.job,
                ),
            ),
        )
    except ArtifactFingerprintError as error:
        postflight_errors.append(str(error))
    for label, before, inspect in (
        (
            "FoldJAX source/harness",
            source,
            lambda: source_identity(Path(__file__).resolve().parent.parent),
        ),
        ("driver runtime", runtime, runtime_identity),
        ("device", device, lambda: device_identity(environment)),
        (
            "execution controls",
            execution,
            lambda: execution_identity(
                environment,
                timing_state=args.timing_state,
                traced=args.traced,
            ),
        ),
    ):
        try:
            after = inspect()
            if after != before:
                postflight_errors.append(f"{label} changed during prediction")
        except (ArtifactFingerprintError, OSError, RuntimeError) as error:
            postflight_errors.append(str(error))
    postflight_error = "; ".join(postflight_errors) or None

    peak = 0.0
    if peak_file.is_file():
        peak = float(peak_file.read_text().strip() or 0) / 2**20

    found = scores(args.model, args.output_dir)
    structures = produced_structures(args.output_dir)
    record = {
        "schema": CURRENT_RESULT_SCHEMA,
        "identity": identity,
        "impl": "upstream",
        "model": args.model,
        "case": case.name,
        "length": case.length,
        "schedule": dict(SCHEDULE),
        "seed": SEED,
        "options": {},
        "artifacts": artifacts,
        "source": source,
        "runtime": runtime,
        "device": device,
        "execution": execution,
        "wall_s": round(elapsed, 2),
        "peak_mib": round(peak, 1),
        "returncode": returncode,
        "samples": [{"scores": entry} for entry in found],
        "upstream_git": upstream_git,
        "upstream_runtime": upstream_runtime,
    }
    if postflight_error is not None:
        record["failed"] = True
        record["reason"] = "benchmark provenance changed during prediction"
        record["provenance_error"] = portable_diagnostic(postflight_error)
    if returncode != 0:
        record["failed"] = True
        record.setdefault(
            "reason",
            portable_diagnostic(
                launch_error or f"upstream prediction exited {returncode}"
            ),
        )
        record["stderr_tail"] = portable_diagnostic(stderr[-2000:])
    elif not structures:
        # An exit status of 0 is not evidence that anything was predicted.
        # OpenDDE's runner catches a CUDA OOM, writes it to an ERR/ file, and
        # returns 0 -- the historical underprovisioned 970-token row asked for
        # 40.8 GiB on top of 58.5 GiB already held against PyTorch's 94.97 GiB
        # usable ceiling, and without this it would be published as a 12-second
        # run that beat everything else.
        record["failed"] = True
        record["reason"] = "exited 0 but produced no structures"
        # Each upstream puts its reason somewhere else: OpenDDE in `ERR/`,
        # OpenFold3 in its own `logs/predict_err_rank*.log`. Reading only one
        # of them made the same out-of-memory show up as two different results
        # -- one explained, one bare -- purely by where it had been written.
        errors = sorted(args.output_dir.rglob("ERR/*.txt")) or sorted(
            args.output_dir.rglob("logs/*err*.log")
        )
        if errors:
            record["stderr_tail"] = portable_diagnostic(
                errors[0].read_text(errors="replace")[-2000:]
            )
        else:
            # `ERR/` is OpenDDE's convention, not a general one. Boltz-2 also
            # exits 0 with nothing predicted and writes no such file, and this
            # branch used to drop the stderr it had already captured -- leaving
            # a `failed` row whose only evidence was a peak. Reading a failure
            # off a peak is guessing; at 4,926 tokens the peak was 92.2 GiB on a
            # 95.6 GiB `nvidia-smi` total, which looks like an out-of-memory and
            # is not proof of one.
            record["stderr_tail"] = portable_diagnostic(stderr[-2000:])
    elif len(structures) != SCHEDULE["num_samples"]:
        record["failed"] = True
        record["reason"] = (
            f"produced {len(structures)} structures for "
            f"{SCHEDULE['num_samples']} requested samples"
        )
    elif (
        len(found) != SCHEDULE["num_samples"]
        or any(not entry for entry in found)
        or any(not math.isfinite(value) for entry in found for value in entry.values())
    ):
        record["failed"] = True
        record["reason"] = (
            "produced structures but did not produce one usable score record "
            "per requested sample"
        )
    if not record.get("failed") and not reusable_result(
        record,
        expected_identity=identity,
    ):
        record["failed"] = True
        record["reason"] = (
            "measurement record has missing or non-finite timing, peak, or "
            "sample scores"
        )
    text = json.dumps(record, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    if postflight_error is not None:
        return 2
    return 1 if record.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
