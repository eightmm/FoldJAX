"""Everything a first FoldJAX run needs, checked in one command.

`foldjax doctor` reports the runtime, the store, the compile cache, and per
model whether weights and each input dialect are ready -- information that used
to be spread across `models --json`, `home`, `runtime status` and the template
section of `setup`, one of which also downloads 3 GB.

Readiness is decided from distribution *metadata*, never by importing an
optional runtime: probing these must not initialize Triton, the cuEq kernels,
or either raw-input pipeline. The one runtime this module does touch is JAX,
imported inside `run_doctor` because a broken JAX is itself a finding -- so, as
with `foldjax.cache_gc`, nothing here imports it at module scope.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from foldjax import paths
from foldjax.registry import available_models, model_info

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


def run_doctor(args: argparse.Namespace) -> int:
    """Everything a first run needs, checked in one command.

    The information was all reachable already -- `models --json`, `home`,
    `runtime status`, the template section of `setup` -- across four commands
    and one that also downloads 3 GB. Someone whose first prediction fails
    should not have to know which of those to run.
    """
    import shutil as _shutil

    from foldjax.cli import _format_bytes, _template_report

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
