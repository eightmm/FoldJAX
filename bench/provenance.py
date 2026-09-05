"""Portable, byte-exact provenance for one benchmark result row."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import stat
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

CURRENT_RESULT_SCHEMA = 2
_DIGEST_CHUNK_BYTES = 1 << 20
_MAX_RESULT_BYTES = 16 << 20
_SOURCE_EXCLUDED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".oms",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
}
_SOURCE_EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
_PATH_ENVIRONMENT_NAMES = {
    "CUDA_CACHE_PATH",
    "CUDA_HOME",
    "FOLDJAX_BENCH_DATA",
    "FOLDJAX_BENCH_ROOT",
    "FOLDJAX_HOME",
    "FOLDJAX_MSA_COMMAND",
    "FOLDJAX_RNA_MSA_COMMAND",
    "JAX_COMPILATION_CACHE_DIR",
    "LIBCIFPP_DATA_DIR",
    "PATH",
    "PROTENIX_CCD_COMPONENTS_FILE",
    "PROTENIX_CCD_RDKIT_MOL_FILE",
    "PROTENIX_KALIGN_BINARY",
    "PROTENIX_TEMPLATE_MMCIF_DIR",
    "PYTHONPATH",
    "XDG_CACHE_HOME",
}
_EXECUTION_ENVIRONMENT_NAMES = {
    *_PATH_ENVIRONMENT_NAMES,
    "BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_DEVICE_ORDER",
    "CUDA_MODULE_LOADING",
    "CUDA_VISIBLE_DEVICES",
    "CUEQ_DEFAULT_BACKEND",
    "CUEQ_DISABLE_JIT",
    "ESMFOLD2_ATOM_ATTENTION_BACKEND",
    "ESMFOLD2_ATOM_ROWS_PER_BLOCK",
    "FOLDJAX_SKIP_MESH_CHECK",
    "JAX_COMPILATION_CACHE_MAX_SIZE",
    "JAX_DEFAULT_MATMUL_PRECISION",
    "JAX_DISABLE_JIT",
    "JAX_ENABLE_X64",
    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES",
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
    "JAX_PLATFORM_NAME",
    "JAX_PLATFORMS",
    "LAYERNORM_TYPE",
    "MKL_NUM_THREADS",
    "NVIDIA_TF32_OVERRIDE",
    "OMP_NUM_THREADS",
    "OPENFOLD3_JAX_CACHE",
    "OPENFOLD3_TRIANGLE_BACKEND",
    "PROTENIX_TRIANGLE_BACKEND",
    "PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND",
    "PYTORCH_CUDA_ALLOC_CONF",
    "XLA_CLIENT_MEM_FRACTION",
    "XLA_FLAGS",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
}
_EXECUTION_ENVIRONMENT_PREFIXES = (
    "BOLTZ_",
    "CUBLAS_",
    "CUDA_",
    "CUDNN_",
    "CUEQ_",
    "ESMFOLD2_",
    "FOLDJAX_",
    "JAX_",
    "KMP_",
    "MKL_",
    "NCCL_",
    "NVIDIA_",
    "OMP_",
    "OPENFOLD3_",
    "PROTENIX_",
    "PYTORCH_",
    "TOKAMAX_",
    "TORCH_",
    "TRITON_",
    "XLA_",
)


class ArtifactFingerprintError(RuntimeError):
    """An input or checkpoint cannot be fingerprinted without following links."""


def _stat_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_mode,
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _fstat(file_descriptor: int, *, subject: str) -> os.stat_result:
    try:
        return os.fstat(file_descriptor)
    except OSError as error:
        raise ArtifactFingerprintError(f"cannot inspect {subject}") from error


def _close(file_descriptor: int, *, subject: str) -> None:
    try:
        os.close(file_descriptor)
    except OSError as error:
        raise ArtifactFingerprintError(f"cannot close {subject}") from error


def _read_regular_file(
    path_or_name: Path | str,
    *,
    directory_fd: int | None = None,
    expected: os.stat_result | None = None,
) -> tuple[int, str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        file_descriptor = os.open(path_or_name, flags, dir_fd=directory_fd)
    except OSError as error:
        raise ArtifactFingerprintError(
            f"cannot safely open benchmark artifact {path_or_name}"
        ) from error
    try:
        before = _fstat(file_descriptor, subject="benchmark artifact")
        if not stat.S_ISREG(before.st_mode):
            raise ArtifactFingerprintError("benchmark artifact is not a regular file")
        if expected is not None and _stat_identity(before) != _stat_identity(expected):
            raise ArtifactFingerprintError(
                "benchmark artifact changed while it was being opened"
            )
        digest = hashlib.sha256()
        size = 0
        while True:
            try:
                chunk = os.read(file_descriptor, _DIGEST_CHUNK_BYTES)
            except InterruptedError:
                continue
            except OSError as error:
                raise ArtifactFingerprintError(
                    "cannot read benchmark artifact"
                ) from error
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
        after = _fstat(file_descriptor, subject="benchmark artifact")
        if _stat_identity(after) != _stat_identity(before) or size != before.st_size:
            raise ArtifactFingerprintError(
                "benchmark artifact changed while it was being fingerprinted"
            )
        return size, digest.hexdigest()
    finally:
        _close(file_descriptor, subject="benchmark artifact")


def _directory_records(
    directory_fd: int,
    *,
    prefix: str = "",
    exclude: Callable[[str, bool], bool] | None = None,
) -> list[dict[str, Any]]:
    before = _fstat(directory_fd, subject="benchmark artifact directory")
    if not stat.S_ISDIR(before.st_mode):
        raise ArtifactFingerprintError("benchmark artifact is not a directory")
    try:
        names = sorted(os.listdir(directory_fd))
    except OSError as error:
        raise ArtifactFingerprintError(
            "cannot list benchmark artifact directory"
        ) from error

    records: list[dict[str, Any]] = []
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for name in names:
        relative = f"{prefix}/{name}" if prefix else name
        try:
            entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as error:
            raise ArtifactFingerprintError(
                f"cannot inspect benchmark artifact entry {relative}"
            ) from error
        if exclude is not None and exclude(relative, stat.S_ISDIR(entry.st_mode)):
            continue
        if stat.S_ISLNK(entry.st_mode):
            raise ArtifactFingerprintError(
                f"benchmark artifact entry is a symlink: {relative}"
            )
        if stat.S_ISREG(entry.st_mode):
            size, digest = _read_regular_file(
                name,
                directory_fd=directory_fd,
                expected=entry,
            )
            records.append(
                {
                    "path": relative,
                    "kind": "file",
                    "bytes": size,
                    "sha256": digest,
                }
            )
            continue
        if stat.S_ISDIR(entry.st_mode):
            try:
                child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
            except OSError as error:
                raise ArtifactFingerprintError(
                    f"cannot safely open benchmark artifact directory {relative}"
                ) from error
            try:
                opened = _fstat(child_fd, subject="benchmark artifact directory")
                if _stat_identity(opened) != _stat_identity(entry):
                    raise ArtifactFingerprintError(
                        "benchmark artifact directory changed while it was opened"
                    )
                records.append({"path": relative, "kind": "directory"})
                records.extend(
                    _directory_records(
                        child_fd,
                        prefix=relative,
                        exclude=exclude,
                    )
                )
            finally:
                _close(child_fd, subject="benchmark artifact directory")
            continue
        raise ArtifactFingerprintError(
            f"benchmark artifact entry is not a regular file or directory: {relative}"
        )

    after = _fstat(directory_fd, subject="benchmark artifact directory")
    if _stat_identity(after) != _stat_identity(before):
        raise ArtifactFingerprintError(
            "benchmark artifact directory changed while it was fingerprinted"
        )
    return records


def fingerprint_path(
    path: Path,
    *,
    allow_directory: bool = True,
    exclude: Callable[[str, bool], bool] | None = None,
) -> dict[str, Any]:
    """Hash one regular file or deterministic directory tree without links."""

    path = Path(path)
    try:
        initial = path.lstat()
    except OSError as error:
        raise ArtifactFingerprintError(
            f"benchmark artifact is unavailable: {path}"
        ) from error
    if stat.S_ISLNK(initial.st_mode):
        raise ArtifactFingerprintError(f"benchmark artifact is a symlink: {path}")
    if stat.S_ISREG(initial.st_mode):
        size, digest = _read_regular_file(path, expected=initial)
        try:
            final = path.lstat()
        except OSError as error:
            raise ArtifactFingerprintError(
                f"benchmark artifact changed after it was fingerprinted: {path}"
            ) from error
        if _stat_identity(final) != _stat_identity(initial):
            raise ArtifactFingerprintError(
                "benchmark artifact changed after it was fingerprinted"
            )
        return {"kind": "file", "bytes": size, "sha256": digest}
    if stat.S_ISDIR(initial.st_mode) and not allow_directory:
        raise ArtifactFingerprintError(
            f"benchmark input must be a regular file: {path}"
        )
    if not stat.S_ISDIR(initial.st_mode):
        raise ArtifactFingerprintError(
            f"benchmark artifact is not a regular file or directory: {path}"
        )

    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory_fd = os.open(path, flags)
    except OSError as error:
        raise ArtifactFingerprintError(
            f"cannot safely open benchmark artifact directory {path}"
        ) from error
    try:
        opened = _fstat(directory_fd, subject="benchmark artifact directory")
        if _stat_identity(opened) != _stat_identity(initial):
            raise ArtifactFingerprintError(
                "benchmark artifact directory changed while it was opened"
            )
        records = _directory_records(directory_fd, exclude=exclude)
    finally:
        _close(directory_fd, subject="benchmark artifact directory")
    try:
        final = path.lstat()
    except OSError as error:
        raise ArtifactFingerprintError(
            f"benchmark artifact directory changed after it was fingerprinted: {path}"
        ) from error
    if _stat_identity(final) != _stat_identity(initial):
        raise ArtifactFingerprintError(
            "benchmark artifact directory changed after it was fingerprinted"
        )
    payload = json.dumps(
        records,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    files = [record for record in records if record["kind"] == "file"]
    return {
        "kind": "directory",
        "bytes": sum(int(record["bytes"]) for record in files),
        "files": len(files),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def fingerprint_artifacts(
    paths: Mapping[str, Path],
    *,
    allow_directories: bool = True,
) -> dict[str, dict[str, Any]]:
    """Fingerprint named artifacts without persisting machine-specific paths."""

    return {
        label: fingerprint_path(Path(path), allow_directory=allow_directories)
        for label, path in sorted(paths.items())
    }


def foldjax_checkpoint_paths(model: str, weights: Path) -> dict[str, Path]:
    """Return the exact native checkpoint payloads selected by a FoldJAX run."""

    if model == "boltz2":
        from foldjax.models.boltz2.weights import resolve_native_weight_bundle

        bundle = resolve_native_weight_bundle(weights)
        if bundle is None:
            raise ArtifactFingerprintError("Boltz-2 native checkpoint is unavailable")
        checkpoint, sidecar = bundle
        if not sidecar.is_file():
            raise ArtifactFingerprintError("Boltz-2 checkpoint sidecar is unavailable")
        return {"model": checkpoint, "model_index": sidecar}
    if model == "esmfold2":
        root = weights.parent if weights.is_file() else weights
        return {"model": root / "model.safetensors"}
    return {"model": Path(weights)}


_ALPHAFOLD3_LIBCIFPP_FILES = (
    "components.cif",
    "mmcif_ddl.dic",
    "mmcif_ma.dic",
    "mmcif_pdbx.dic",
)


def _alphafold3_external_asset_paths(
    source: Any,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    """Execution bytes selected by AlphaFold 3's explicit checkout route."""

    from foldjax.manifest import alphafold3_effective_libcifpp_directory

    try:
        checkout = Path(source).resolve(strict=True)
    except (OSError, RuntimeError, TypeError) as error:
        raise ArtifactFingerprintError(
            "AlphaFold 3 external source checkout is unavailable"
        ) from error
    runner = checkout / "run_alphafold.py"
    package = checkout / "src/alphafold3"
    if not runner.is_file() or not (package / "__init__.py").is_file():
        raise ArtifactFingerprintError(
            "AlphaFold 3 external source is missing its runner or package"
        )

    selected = {"alphafold3.external.runner": runner}
    for path in sorted(package.rglob("*")):
        if (
            path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix.lower() not in {".pyc", ".pyo"}
        ):
            relative = path.relative_to(package).as_posix()
            selected[f"alphafold3.external.package.{relative}"] = path

    libcifpp = alphafold3_effective_libcifpp_directory(
        os.environ if environment is None else environment,
        package=package,
    )
    if libcifpp is None:
        raise ArtifactFingerprintError(
            "AlphaFold 3 external source has no verifiable libcifpp data"
        )
    for name in _ALPHAFOLD3_LIBCIFPP_FILES:
        selected[f"alphafold3.external.libcifpp.{name}"] = libcifpp / name
    return selected


def foldjax_implicit_asset_paths(
    model: str,
    weights: Path,
    *,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Return implicit assets selected by the FoldJAX benchmark request."""

    if model == "alphafold3":
        source = (options or {}).get("source")
        if source:
            return _alphafold3_external_asset_paths(source)
        from foldjax.manifest import alphafold3_runtime_dependency_paths
        from foldjax.models.alphafold3 import build

        try:
            build.ensure_ready()
            return alphafold3_runtime_dependency_paths()
        except (OSError, RuntimeError, TypeError) as error:
            raise ArtifactFingerprintError(
                "AlphaFold 3 managed runtime is unavailable"
            ) from error

    if model == "esmfold2":
        from foldjax.backends.esmfold2 import _esmc_asset_paths

        root = weights.parent if weights.is_file() else weights
        selected = {"esmfold2.structure_config": root / "config.json"}
        without_language_model = (options or {}).get("no_language_model", False)
        if type(without_language_model) is not bool:
            raise ArtifactFingerprintError(
                "ESMFold2 no_language_model must be a boolean"
            )
        if without_language_model:
            return selected
        configured = (options or {}).get("esmc_weights")
        esmc = Path(configured) if configured is not None else root / "esmc"
        assets = _esmc_asset_paths(esmc)
        if assets is None:
            raise ArtifactFingerprintError("ESMFold2 ESMC assets are unavailable")
        selected.update(
            {
                f"esmfold2.esmc.{index:04d}.{path.name}": path
                for index, path in enumerate(assets)
            }
        )
        return selected

    if model == "openfold3":
        configured = (options or {}).get("ccd_file_path")
        if configured is not None:
            try:
                ccd = Path(configured)
            except TypeError as error:
                raise ArtifactFingerprintError(
                    "OpenFold3 ccd_file_path is not a filesystem path"
                ) from error
        else:
            try:
                from biotite.structure.info import ccd as biotite_ccd

                ccd = Path(
                    biotite_ccd._CCD_FILE  # noqa: SLF001 - actual runtime input
                )
            except (AttributeError, ImportError, TypeError) as error:
                raise ArtifactFingerprintError(
                    "OpenFold3 Biotite CCD asset is unavailable"
                ) from error
        return {"openfold3.biotite_components": ccd}

    if model != "boltz2":
        return {}
    configured = (options or {}).get("mols")
    if configured is not None:
        mols = Path(str(configured))
    else:
        candidates = (weights.parent / "mols", weights.parent.parent / "mols")
        mols = next((candidate for candidate in candidates if candidate.is_dir()), None)
        if mols is None:
            raise ArtifactFingerprintError("Boltz-2 molecule assets are unavailable")

    # The FoldJAX protein benchmark loads the released canonical protein
    # molecules, not every CCD pickle in the 45k-entry bundle. Pin exactly
    # those role-selected payloads so provenance remains exact and affordable.
    from foldjax.models.boltz2.data.const import canonical_tokens
    from foldjax.models.boltz2.data.identifiers import resolve_molecule_pickle

    try:
        return {
            f"boltz.canonical-protein.{token}": resolve_molecule_pickle(mols, token)
            for token in canonical_tokens
        }
    except ValueError as error:
        raise ArtifactFingerprintError(str(error)) from error


def _input_artifact_paths(
    job: Path | None,
    *,
    native_input: Path | None,
) -> dict[str, Path]:
    from foldjax.manifest import (
        common_job_dependency_paths,
        native_input_dependency_paths,
    )

    inputs: dict[str, Path] = {}
    if job is not None:
        dependencies = common_job_dependency_paths(job)
        if dependencies is None:
            raise ArtifactFingerprintError(
                "benchmark job is not a valid FoldJAX common-schema document"
            )
        inputs["common.job"] = Path(job)
        inputs.update({f"common.{label}": path for label, path in dependencies.items()})
    if native_input is not None:
        native_dependencies = native_input_dependency_paths(native_input)
        if native_dependencies is None:
            raise ArtifactFingerprintError(
                "generated native benchmark input has unresolved dependencies"
            )
        inputs["native.document"] = Path(native_input)
        inputs.update(
            {f"native.{label}": path for label, path in native_dependencies.items()}
        )
    if not inputs:
        raise ArtifactFingerprintError(
            "benchmark identity needs a common job or native input"
        )
    return inputs


def artifact_identity(
    *,
    job: Path | None = None,
    checkpoints: Mapping[str, Path],
    native_input: Path | None = None,
    implicit_assets: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Fingerprint every external byte source controlled by the bench matrix."""

    return {
        "inputs": fingerprint_artifacts(
            _input_artifact_paths(
                Path(job) if job is not None else None,
                native_input=native_input,
            ),
            allow_directories=False,
        ),
        "checkpoints": fingerprint_artifacts(checkpoints),
        "implicit_assets": fingerprint_artifacts(implicit_assets or {}),
    }


def _source_excluded(relative: str, is_directory: bool) -> bool:
    parts = Path(relative).parts
    if is_directory and any(part in _SOURCE_EXCLUDED_DIRECTORIES for part in parts):
        return True
    return Path(relative).suffix.lower() in _SOURCE_EXCLUDED_SUFFIXES


def source_identity(repo: Path) -> dict[str, Any]:
    """Hash execution-affecting FoldJAX and benchmark source without paths."""

    repo = Path(repo)
    harness = {
        "alphafold3_sample_arm": repo
        / "tests/models/alphafold3/scripts/run_sample_shard_end_to_end_arm.py",
        "drive": repo / "bench/drive.py",
        "gap_ablation": repo / "bench/run_gap_ablation.py",
        "opendde_relp_arm": repo
        / "tests/models/opendde/scripts/run_structural_relp_end_to_end_arm.py",
        "openfold3_runner": repo / "bench/openfold3_runner.yml",
        "openfold3_confidence_arm": repo
        / "tests/models/openfold3/scripts/run_confidence_schedule_end_to_end_arm.py",
        "peakhook": repo / "bench/peakhook",
        "provenance": repo / "bench/provenance.py",
        "run_foldjax": repo / "bench/run_foldjax.py",
        "run_upstream": repo / "bench/run_upstream.py",
        "spec": repo / "bench/spec.py",
        "structures": repo / "bench/structures.py",
        "trace": repo / "bench/trace.py",
        "project": repo / "pyproject.toml",
        "lock": repo / "uv.lock",
    }
    return {
        "foldjax": fingerprint_path(
            repo / "src/foldjax",
            exclude=_source_excluded,
        ),
        "harness": {
            label: fingerprint_path(path, exclude=_source_excluded)
            for label, path in sorted(harness.items())
        },
    }


_RUNTIME_DISTRIBUTIONS = (
    "biopython",
    "biotite",
    "chex",
    "cuequivariance",
    "cuequivariance-jax",
    "cuequivariance-ops-jax-cu12",
    "cuequivariance-ops-jax-cu13",
    "foldjax",
    "gemmi",
    "jax-triton",
    "jax",
    "jax-cuda12-plugin",
    "jax-cuda12-pjrt",
    "jax-cuda13-plugin",
    "jax-cuda13-pjrt",
    "jaxlib",
    "ml-dtypes",
    "numpy",
    "nvidia-cuda-cccl-cu12",
    "nvidia-cuda-cccl-cu13",
    "nvidia-cuda-crt-cu12",
    "nvidia-cuda-crt-cu13",
    "nvidia-cuda-nvcc-cu12",
    "nvidia-cuda-nvcc-cu13",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-runtime-cu13",
    "optax",
    "pyyaml",
    "rdkit",
    "scipy",
    "torch",
    "tokamax",
    "triton",
)


def runtime_identity() -> dict[str, Any]:
    """Return path-free installed runtime versions without importing JAX."""

    versions: dict[str, str] = {}
    for name in _RUNTIME_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "distributions": versions,
    }


def _hash_environment_value(value: str) -> dict[str, str]:
    return {"sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()}


def portable_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Redact path- or secret-shaped option values while preserving identity."""

    portable: dict[str, Any] = {}
    for name, value in sorted(options.items()):
        sensitive = re.search(
            r"(credential|directory|file|home|key|mols|password|path|root|secret|source|token|weights)",
            name,
            re.IGNORECASE,
        )
        portable[name] = _hash_environment_value(str(value)) if sensitive else value
    return portable


def redact_machine_paths(
    text: str,
    paths: Mapping[str, Path],
) -> str:
    """Replace known absolute machine roots in persisted diagnostics."""

    replacements: list[tuple[str, str]] = []
    for label, path in paths.items():
        raw = str(Path(path).expanduser().absolute())
        if raw:
            replacements.append((raw, f"<{label}>"))
    redacted = text
    for raw, replacement in sorted(replacements, key=lambda item: -len(item[0])):
        redacted = redacted.replace(raw, replacement)
    return redacted


def foldjax_effective_environment(
    environment: Mapping[str, str] | None = None,
    *,
    model: str | None = None,
    options: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Mirror the benchmark runner's pre-import FoldJAX environment defaults."""

    effective = dict(os.environ if environment is None else environment)
    if not any(
        name in effective
        for name in ("XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_MEM_FRACTION")
    ):
        effective["XLA_CLIENT_MEM_FRACTION"] = "0.9"
    if model == "alphafold3":
        from foldjax.manifest import alphafold3_effective_libcifpp_directory

        source = (options or {}).get("source")
        package = None
        if source:
            try:
                package = Path(source).resolve(strict=True) / "src/alphafold3"
            except (OSError, RuntimeError, TypeError) as error:
                raise ArtifactFingerprintError(
                    "AlphaFold 3 external source checkout is unavailable"
                ) from error
        libcifpp = alphafold3_effective_libcifpp_directory(
            effective,
            package=package,
        )
        if libcifpp is not None:
            effective["LIBCIFPP_DATA_DIR"] = str(libcifpp)
    elif model == "boltz2":
        # The template-only path imports sklearn's threadpoolctl after the
        # benchmark snapshot. threadpoolctl and the loaded OpenMP runtime set
        # these defaults process-wide; project them up front so a deterministic
        # library initialization is not mistaken for a caller changing the
        # execution controls mid-prediction.
        effective.setdefault("KMP_DUPLICATE_LIB_OK", "True")
        effective.setdefault("KMP_INIT_AT_FORK", "FALSE")
    return effective


def execution_identity(
    environment: Mapping[str, str],
    *,
    timing_state: str,
    traced: bool,
) -> dict[str, Any]:
    """Capture non-secret effective execution controls without machine paths."""

    if timing_state not in {
        "cold-or-unspecified",
        "warm-after-successful-prefill",
    }:
        raise ValueError(f"unsupported benchmark timing state: {timing_state}")
    controls: dict[str, Any] = {}
    names = set(_EXECUTION_ENVIRONMENT_NAMES)
    names.update(
        name for name in environment if name.startswith(_EXECUTION_ENVIRONMENT_PREFIXES)
    )
    for name in sorted(names):
        value = environment.get(name)
        if value is None:
            continue
        # Device ordinals may be harmless, but CUDA also accepts stable GPU
        # UUIDs here. Hash the complete selector rather than leaking one.
        sensitive = (
            name not in _EXECUTION_ENVIRONMENT_NAMES
            or name in _PATH_ENVIRONMENT_NAMES
            or name
            in {
                "CUDA_VISIBLE_DEVICES",
                "XLA_FLAGS",
            }
        )
        if sensitive:
            controls[name] = _hash_environment_value(value)
        else:
            controls[name] = value
    return {
        "timing_state": timing_state,
        "traced": bool(traced),
        "environment": controls,
    }


def device_identity(
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Probe the selected device without initializing a CUDA compute context."""

    environment = os.environ if environment is None else environment
    forced = environment.get("JAX_PLATFORMS") or environment.get("JAX_PLATFORM_NAME")
    if forced and forced.split(",", 1)[0].strip().lower() == "cpu":
        return {"platform": "cpu"}
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() in {"", "-1"}:
        return {"platform": "cuda-disabled"}
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,compute_cap,memory.total,driver_version,uuid",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            query,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ArtifactFingerprintError(
            "cannot inspect benchmark GPU identity"
        ) from error
    if completed.returncode != 0:
        raise ArtifactFingerprintError("cannot inspect benchmark GPU identity")
    rows: list[tuple[str, str, str, str, str, str]] = []
    for line in completed.stdout.splitlines():
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 6:
            raise ArtifactFingerprintError("unexpected nvidia-smi identity output")
        rows.append(fields)  # type: ignore[arg-type]
    if not rows:
        raise ArtifactFingerprintError("no benchmark GPU is visible")
    selector = visible.split(",", 1)[0].strip() if visible else None
    chosen = rows[0]
    if selector:
        matches = [row for row in rows if selector in {row[0], row[5]}]
        if not matches:
            raise ArtifactFingerprintError("selected benchmark GPU is unavailable")
        chosen = matches[0]
    _index, name, compute_capability, memory_mib, driver, _uuid = chosen
    try:
        total_memory_mib = int(memory_mib)
    except ValueError as error:
        raise ArtifactFingerprintError("invalid nvidia-smi memory identity") from error
    return {
        "platform": "cuda",
        "model": name,
        "compute_capability": compute_capability,
        "total_memory_mib": total_memory_mib,
        "driver_version": driver,
    }


def benchmark_identity(
    *,
    impl: str,
    model: str,
    case: str,
    length: int,
    schedule: Mapping[str, int],
    seed: int,
    artifacts: Mapping[str, Any],
    source: Mapping[str, Any],
    runtime: Mapping[str, Any],
    device: Mapping[str, Any],
    execution: Mapping[str, Any],
    options: Mapping[str, Any] | None = None,
    upstream_git: Mapping[str, Any] | None = None,
    upstream_runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the portable identity that controls ``--skip-existing`` reuse."""

    identity: dict[str, Any] = {
        "impl": impl,
        "model": model,
        "case": case,
        "length": int(length),
        "schedule": dict(sorted(schedule.items())),
        "seed": int(seed),
        "options": dict(sorted((options or {}).items())),
        "artifacts": dict(artifacts),
        "source": dict(source),
        "runtime": dict(runtime),
        "device": dict(device),
        "execution": dict(execution),
    }
    if upstream_git is not None or upstream_runtime is not None:
        if upstream_git is None or upstream_runtime is None:
            raise ValueError("upstream git and runtime provenance must be paired")
        identity["upstream_git"] = dict(upstream_git)
        identity["upstream_runtime"] = dict(upstream_runtime)
    return identity


def reusable_result(record: Any, *, expected_identity: Mapping[str, Any]) -> bool:
    """Whether a successful row belongs to this exact current benchmark run."""

    if not isinstance(record, dict):
        return False
    if (
        type(record.get("schema")) is not int
        or record["schema"] != CURRENT_RESULT_SCHEMA
        or record.get("failed", False) is not False
    ):
        return False
    if "returncode" in record and (
        type(record["returncode"]) is not int or record["returncode"] != 0
    ):
        return False
    identity = dict(expected_identity)
    schedule = identity.get("schedule")
    if not isinstance(schedule, Mapping):
        return False
    expected_samples = schedule.get("num_samples")
    if type(expected_samples) is not int or expected_samples <= 0:
        return False
    samples = record.get("samples")
    if not isinstance(samples, list) or len(samples) != expected_samples:
        return False
    for sample in samples:
        if not isinstance(sample, dict):
            return False
        scores = sample.get("scores")
        if not isinstance(scores, dict) or not scores:
            return False
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in scores.values()
        ):
            return False
    for metric in ("wall_s", "peak_mib"):
        value = record.get(metric)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            return False
    if record.get("identity") != identity:
        return False
    return all(
        record.get(field) == identity.get(field)
        for field in (
            "impl",
            "model",
            "case",
            "length",
            "schedule",
            "seed",
            "options",
            "artifacts",
            "source",
            "runtime",
            "device",
            "execution",
            "upstream_git",
            "upstream_runtime",
        )
    )


def reusable_result_file(
    path: Path,
    *,
    expected_identity: Mapping[str, Any],
) -> bool:
    """Read a row defensively and apply the current-schema reuse contract."""

    try:
        path = Path(path)
        initial = path.lstat()
        if not stat.S_ISREG(initial.st_mode) or initial.st_size > _MAX_RESULT_BYTES:
            return False
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        file_descriptor = os.open(path, flags)
        try:
            before = _fstat(file_descriptor, subject="benchmark result")
            if (
                not stat.S_ISREG(before.st_mode)
                or _stat_identity(before) != _stat_identity(initial)
                or before.st_size > _MAX_RESULT_BYTES
            ):
                return False
            chunks: list[bytes] = []
            size = 0
            while True:
                chunk = os.read(file_descriptor, _DIGEST_CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                if size > _MAX_RESULT_BYTES:
                    return False
                chunks.append(chunk)
            after = _fstat(file_descriptor, subject="benchmark result")
            if (
                _stat_identity(after) != _stat_identity(before)
                or size != before.st_size
            ):
                return False
        finally:
            _close(file_descriptor, subject="benchmark result")
        final = path.lstat()
        if _stat_identity(final) != _stat_identity(initial):
            return False
        record = json.loads(b"".join(chunks).decode("utf-8"))
    except (
        ArtifactFingerprintError,
        OSError,
        RecursionError,
        UnicodeError,
        json.JSONDecodeError,
    ):
        return False
    return reusable_result(record, expected_identity=expected_identity)


def require_unchanged(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> None:
    """Reject a measurement whose external artifact bytes changed in flight."""

    if before != after:
        raise ArtifactFingerprintError(
            "benchmark provenance identity changed during prediction"
        )
