"""AlphaFold 3 in-process adapter for upstream's own runner.

The runner is `run_alphafold.py`, which upstream keeps at its repository root
instead of inside the `alphafold3` package, so installing the package does not
bring it along. A copy is vendored beside this module for that reason. The
managed vendored runtime is always the default; an external checkout is used
only when the request explicitly selects its ``source``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import re
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from foldjax.backends.base import Backend
from foldjax.manifest import path_stat_identity
from foldjax.padding import TOKEN_BUCKETS, PaddingPlan
from foldjax.schema import (
    InputRequirement,
    ModelCapabilities,
    PredictionError,
    PredictionRequest,
    PredictionResult,
    PredictionSample,
    _strict_boolean,
    _strict_integer,
)
from foldjax.scores import scalar_scores

#: Upstream's runner, carried because its wheel does not install it. See the
#: NOTICE beside it.
VENDORED_RUNNER = Path(__file__).with_name("_alphafold3_upstream") / "run_alphafold.py"

# Keep this order byte-for-byte aligned with upstream's ``select_model_files``.
# The first matching filename family wins, and a family containing more than
# one model name is ambiguous.  Reproducing that small selector locally lets a
# live session stat only the files the lazy parameter property will read,
# without importing AlphaFold 3 or opening its 1.1 GB payload.
_PARAMETER_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"(?P<model_name>.*)\.[0-9]+\.bin\.zst$",
        r"(?P<model_name>.*)\.bin\.zst\.[0-9]+$",
        r"(?P<model_name>.*)\.[0-9]+\.bin$",
        r"(?P<model_name>.*)\.bin\]\.[0-9]+$",
        r"(?P<model_name>.*)\.bin\.zst$",
        r"(?P<model_name>.*)\.bin$",
    )
)


def _selected_parameter_files(model_dir: Path) -> tuple[Path, ...] | None:
    """Return exactly the files upstream's parameter loader would select."""

    try:
        files = [path for path in Path(model_dir).iterdir() if path.is_file()]
    except (OSError, RuntimeError):
        return None
    for pattern in _PARAMETER_PATTERNS:
        models: dict[str, list[Path]] = {}
        for path in files:
            match = pattern.fullmatch(path.name)
            if match is not None:
                models.setdefault(match.group("model_name"), []).append(path)
        if not models:
            continue
        if len(models) != 1:
            return None
        return tuple(sorted(next(iter(models.values()))))
    return None


def _managed_asset_snapshot(
    weights: Path,
) -> tuple[tuple[str, str, str], ...] | None:
    """Stat identity of managed source, runner and parameter payloads."""

    parameter_files = _selected_parameter_files(weights)
    if not parameter_files:
        return None
    from foldjax.models.alphafold3 import build

    records: list[tuple[str, str, str]] = []
    paths = [(f"weight:{index}", path) for index, path in enumerate(parameter_files)]
    paths.extend(
        (
            ("runner", VENDORED_RUNNER),
            ("source", build.source_package()),
        )
    )
    for role, path in paths:
        identity = path_stat_identity(path)
        if identity is None:
            return None
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError):
            return None
        records.append(
            (
                role,
                str(resolved),
                json.dumps(identity, sort_keys=True, separators=(",", ":")),
            )
        )
    return tuple(records)


def _managed_source_key(weights: Path) -> tuple[str, str, str]:
    """Lexical identity keeps symlink retargeting inside one anchored source."""

    from foldjax.models.alphafold3 import build

    return (
        str(Path(weights).absolute()),
        str(VENDORED_RUNNER.absolute()),
        str(build.source_package().absolute()),
    )


def _device_identity(device: Any) -> tuple[str, ...]:
    """Stable identity of the concrete device pinned into ``ModelRunner``."""

    values = []
    for name in ("platform", "id", "process_index", "local_hardware_id", "device_kind"):
        value = getattr(device, name, None)
        try:
            value = value() if callable(value) else value
        except Exception:  # noqa: BLE001 - an opaque device must split the cache
            value = f"unavailable:{name}"
        values.append(f"{name}={value!s}")
    return (f"{type(device).__module__}.{type(device).__qualname__}", *values)


def _runner_identity(runner: Any, path: Path) -> tuple[str, ...]:
    """Process-local identity of the module that owns the runner/config ABI."""

    try:
        resolved = str(Path(path).resolve(strict=True))
    except (OSError, RuntimeError):
        resolved = str(Path(path).absolute())
    origin = getattr(runner, "__file__", None)
    if origin is not None:
        try:
            origin = str(Path(origin).resolve(strict=True))
        except (OSError, RuntimeError, TypeError):
            origin = str(origin)
    return (
        resolved,
        str(getattr(runner, "__name__", type(runner).__qualname__)),
        str(origin),
        str(id(runner)),
    )


def _cache_identity(path: Path | None) -> tuple[str, str] | None:
    """Identity of the active persistent compilation-cache namespace."""

    if path is None:
        return None
    lexical = str(Path(path).absolute())
    try:
        resolved = str(Path(path).resolve(strict=False))
    except (OSError, RuntimeError):
        resolved = lexical
    return lexical, resolved


def _runner_path(options: dict) -> Path:
    """Resolve an explicitly selected checkout or FoldJAX's managed runner.

    The ordinary path never probes or adopts an independently installed
    ``alphafold3`` package. A checkout is an external provenance boundary and
    is accepted only through the request's explicit ``source`` option.
    """
    explicit = options.pop("source", None)
    if explicit:
        checkout = Path(explicit).resolve()
        path = checkout / "run_alphafold.py"
        if not path.is_file():
            raise FileNotFoundError(f"AlphaFold 3 runner not found: {path}")
        package = checkout / "src" / "alphafold3"
        package_init = package / "__init__.py"
        if not package_init.is_file():
            raise FileNotFoundError(
                f"AlphaFold 3 package source not found beside runner: {package}"
            )
        from foldjax.models.alphafold3 import _upstream

        loaded = sys.modules.get("alphafold3")
        spec = None if loaded is not None else importlib.util.find_spec("alphafold3")
        origin = getattr(loaded, "__file__", None) or getattr(spec, "origin", None)
        if origin is None or Path(origin).resolve().parent != package:
            if loaded is not None or (spec is not None and spec.origin is not None):
                raise RuntimeError(
                    "the explicit AlphaFold 3 runner does not match the imported "
                    f"package; start a process with {checkout / 'src'} first on "
                    "PYTHONPATH, or remove --option source"
                )
            _upstream.ensure_registered(package=package)
        return path
    from foldjax.models.alphafold3 import build

    loaded = sys.modules.get("alphafold3")
    managed_package = getattr(loaded, "__foldjax_managed_package__", None)
    if managed_package is not None:
        # A process cannot safely swap a loaded pybind package to another
        # source/ABI runtime after changing FOLDJAX_HOME or upgrading FoldJAX.
        if Path(managed_package).resolve() != build.runtime_package().resolve():
            raise RuntimeError(
                "a different FoldJAX-managed AlphaFold 3 runtime is already "
                "imported; restart this Python process after changing "
                "FOLDJAX_HOME or upgrading FoldJAX"
            )
        build.register_runtime()
        return VENDORED_RUNNER
    if loaded is not None:
        origin = getattr(loaded, "__file__", None)
        if origin is None or not build.is_managed_origin(origin):
            blocker = build.runtime_blocker()
            raise RuntimeError(
                blocker
                or "an external AlphaFold 3 package is already imported; "
                "restart Python to use FoldJAX's managed runtime"
            )

    # `register_runtime` builds the generated native/CCD half once and gives
    # the source-and-ABI-keyed vendored package temporary import precedence.
    # Thus an arbitrary unimported site-package cannot change this selection.
    build.register_runtime()
    return VENDORED_RUNNER


def _load_runner(path: Path):
    resolved = path.resolve()
    digest = hashlib.sha256()
    digest.update(str(resolved).encode())
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    identity = digest.hexdigest()[:16]
    name = f"_foldjax_alphafold3_runner_{identity}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load AlphaFold 3 runner: {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        _settle_absl_flags()
    except BaseException:
        # Import machinery normally cleans a failed module itself; this module
        # is registered manually, so do the same or a retry receives a
        # half-executed runner with missing APIs.
        sys.modules.pop(name, None)
        raise
    return module


def _tokamax_kernel_fallback(strategy: str):
    """Let Tokamax measure its kernel configuration instead of guessing it.

    AlphaFold 3's transition blocks call ``tokamax.gated_linear_unit``, which
    prefers a Triton kernel and, on a cache miss, sizes it from heuristics.
    Those heuristics are not right for every device -- on an RTX PRO 6000
    Blackwell they ask for 108 KiB of shared memory against the 99 KiB it has,
    and the launch fails with ``RESOURCE_EXHAUSTED`` from inside the diffusion
    transformer. It fails *after* the kernel has compiled, so Tokamax's own
    per-implementation fallback, which only catches ``NotImplementedError``,
    never sees it.

    ``autotune`` is the mechanism Tokamax provides for exactly this: it
    benchmarks the candidate configurations on whatever device is in front of it
    and keeps one that fits, so the same command works across a fleet of
    different cards. It costs time on the first compile of each shape and is
    cached afterwards. Pass ``kernel_autotuning=heuristics`` for upstream's
    default, or ``error`` to make a cache miss loud.
    """
    from tokamax import config as tokamax_config

    return tokamax_config.autotuning_cache_miss_fallback(strategy)


def _model_device(device: Any):
    """Make lazy AlphaFold 3 parameter loads follow the selected device.

    ``ModelRunner`` pins its compiled function to ``device``, but its cached
    parameter property is first touched inside ``predict_structure``. Without
    this context, ``jnp.array`` in the upstream loader uses JAX's implicit
    default device and a nonzero GPU run briefly owns and copies the checkpoint
    from GPU 0.
    """
    import jax

    return jax.default_device(device)


def _settle_absl_flags() -> None:
    """Give absl its defaults instead of letting it parse FoldJAX's argv.

    ``run_alphafold.py`` is an absl program and defines its flags at import, and
    Tokamax reads its own flags lazily -- the first access does
    ``flags.FLAGS(sys.argv)`` if nothing has parsed yet. Reached from the
    FoldJAX CLI that argv is ``foldjax predict --model ...``, which absl rejects
    with "Unknown command line flag 'model'" from somewhere deep inside the
    diffusion transformer. FoldJAX's own CLI is the interface here, so the flags
    are marked parsed and keep their defaults; everything this backend cares
    about is passed to ``ModelRunner`` explicitly.
    """
    from absl import flags

    if not flags.FLAGS.is_parsed():
        flags.FLAGS.mark_as_parsed()


#: Where the two knobs `make_model_config` does not take live in the config it
#: returns. Named so a version that moves them fails with the path it looked
#: for rather than silently leaving the released default in place.
_DIFFUSION_STEPS = ("heads", "diffusion", "eval", "steps")
_MSA_DEPTH = ("evoformer", "num_msa")
_PREFIX_STABLE_NOISE_FEATURE = "__foldjax_prefix_stable_diffusion_noise"


def _validated_buckets(value: Any) -> tuple[int, ...]:
    """Normalize the legacy AF3 bucket option without deferring shape errors.

    An empty sequence keeps the historical meaning of disabling bucketing.
    Non-empty sequences follow the upstream runner's documented strictly
    increasing contract.
    """

    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError("buckets must be a sequence of positive integers")
    try:
        raw_buckets = tuple(value)
    except TypeError as error:
        raise ValueError("buckets must be a sequence of positive integers") from error
    buckets = tuple(
        _strict_integer(bucket, name="each bucket", minimum=1) for bucket in raw_buckets
    )
    if any(left >= right for left, right in zip(buckets, buckets[1:])):
        raise ValueError("buckets must be strictly increasing and unique")
    return buckets


def _prediction_buckets(
    request: PredictionRequest, options: dict[str, Any]
) -> tuple[int, ...] | None:
    """Resolve neutral padding onto AF3's native featurizer bucket list."""

    if request.padding is None:
        return _validated_buckets(options.pop("buckets", ())) or None
    if "buckets" in options:
        raise ValueError(
            "padding and the native AlphaFold3 option 'buckets' were both set; "
            "pass one of them"
        )
    if request.padding.tokens is not None:
        return (request.padding.tokens,)
    return TOKEN_BUCKETS


def _token_padding_plan(results: Any, buckets: tuple[int, ...]) -> PaddingPlan | None:
    """Recover the concrete native AF3 token shape from one job's results."""

    for results_for_seed in results:
        for inference_result in getattr(results_for_seed, "inference_results", ()):
            metadata = getattr(inference_result, "metadata", None)
            if not isinstance(metadata, dict):
                continue
            token_chain_ids = metadata.get("token_chain_ids")
            if token_chain_ids is None:
                continue
            actual = len(token_chain_ids)
            target = next((bucket for bucket in buckets if bucket >= actual), actual)
            return _resolved_token_padding_plan(
                actual=actual,
                target=target,
                buckets=buckets,
                overflow="exact",
            )
    return None


def _resolved_token_padding_plan(
    *,
    actual: int,
    target: int,
    buckets: tuple[int, ...],
    overflow: str,
    fixed_target: bool = False,
) -> PaddingPlan:
    """Validate one native AF3 bucket result before model compilation starts."""

    if target < actual:
        raise ValueError(
            f"AlphaFold3 featurization selected {target} tokens for an input "
            f"with {actual} tokens"
        )
    if actual > buckets[-1] and (overflow == "error" or fixed_target):
        raise ValueError(
            f"input tokens size {actual} exceeds the requested AlphaFold3 "
            f"padding limit {buckets[-1]}"
        )
    return PaddingPlan(
        actual={"tokens": actual},
        storage={"tokens": actual},
        target={"tokens": target},
    )


def _featurize_padded_structure(
    fold_input: Any,
    *,
    buckets: tuple[int, ...],
    overflow: str,
    fixed_target: bool = False,
) -> tuple[Any, PaddingPlan]:
    """Featurize one job and validate its resolved bucket without inference.

    The upstream convenience function combines featurization and model execution,
    so it cannot reject an undersized fixed bucket until after compilation.  The
    neutral path splits those same public stages, retaining the examples so the
    validation preflight never requires a second featurization.
    """

    from alphafold3.constants import chemical_components
    from alphafold3.data import featurisation

    print(f"Featurising data with {len(fold_input.rng_seeds)} seed(s)...")
    started = time.time()
    examples = featurisation.featurise_input(
        fold_input=fold_input,
        buckets=buckets,
        ccd=chemical_components.Ccd(user_ccd=fold_input.user_ccd),
        verbose=True,
    )
    plans: list[PaddingPlan] = []
    for example in examples:
        actual = int(np.asarray(example["seq_length"]).item())
        target = int(np.shape(example["seq_mask"])[0])
        plans.append(
            _resolved_token_padding_plan(
                actual=actual,
                target=target,
                buckets=buckets,
                overflow=overflow,
                fixed_target=fixed_target,
            )
        )
    if not plans:
        raise ValueError("AlphaFold3 featurization produced no examples")
    if any(plan.summary() != plans[0].summary() for plan in plans[1:]):
        raise ValueError("AlphaFold3 seeds resolved to different padding shapes")
    print(
        f"Featurising data with {len(fold_input.rng_seeds)} seed(s) took "
        f"{time.time() - started:.2f} seconds."
    )
    return examples, plans[0]


def _prepare_padded_jobs(
    jobs: tuple[tuple[Any, str, Path], ...],
    *,
    buckets: tuple[int, ...],
    overflow: str,
    fixed_target: bool = False,
) -> tuple[tuple[Any, str, Path, Any, PaddingPlan], ...]:
    """Resolve every neutral job before the first model invocation."""

    prepared = []
    for fold_input, job_name, job_dir in jobs:
        examples, plan = _featurize_padded_structure(
            fold_input,
            buckets=buckets,
            overflow=overflow,
            fixed_target=fixed_target,
        )
        prepared.append((fold_input, job_name, job_dir, examples, plan))
    return tuple(prepared)


def _predict_featurized_structure(
    fold_input: Any,
    examples: Any,
    model_runner: Any,
    runner: Any,
) -> tuple[Any, ...]:
    """Run inference over examples retained by the neutral padding preflight."""

    import jax

    all_results = []
    for seed, example in zip(fold_input.rng_seeds, examples, strict=True):
        model_example = dict(example)
        # Presence of this private, scalar feature selects the padded sampler's
        # prefix-stable random draws.  AlphaFold's typed Batch ignores unknown
        # fields, so the ordinary runner signature and the public feature ABI
        # remain unchanged.  The unpadded path never adds it.
        model_example[_PREFIX_STABLE_NOISE_FEATURE] = np.asarray(True)
        result = model_runner.run_inference(model_example, jax.random.PRNGKey(seed))
        inference_results = model_runner.extract_inference_results(
            batch=example,
            result=result,
            target_name=fold_input.name,
        )
        num_tokens = int(np.asarray(example["seq_length"]).item())
        all_results.append(
            runner.ResultsForSeed(
                seed=seed,
                inference_results=inference_results,
                full_fold_input=fold_input,
                embeddings=model_runner.extract_embeddings(
                    result=result, num_tokens=num_tokens
                ),
                distogram=model_runner.extract_distogram(
                    result=result, num_tokens=num_tokens
                ),
            )
        )
    return tuple(all_results)


def _shape_profile(plans: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Represent one concrete profile, or retain each native job's profile."""

    if not plans:
        return None
    if all(plan == plans[0] for plan in plans[1:]):
        return plans[0]
    return {"per_job": plans}


def _public_shape_profile(
    request: PredictionRequest, plans: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Only neutral padding changes AlphaFold3's public result envelope."""

    return _shape_profile(plans) if request.padding is not None else None


def _set_nested(config: Any, path: tuple[str, ...], value: Any) -> None:
    """Assign `value` at `path`, or explain which attribute is missing."""
    if value is None:
        return
    target = config
    for attribute in path[:-1]:
        target = getattr(target, attribute, None)
        if target is None:
            raise ValueError(
                f"this AlphaFold 3 build has no {'.'.join(path)} to set; "
                "the option cannot be honoured and is not being ignored"
            )
    if not hasattr(target, path[-1]):
        raise ValueError(
            f"this AlphaFold 3 build has no {'.'.join(path)} to set; "
            "the option cannot be honoured and is not being ignored"
        )
    setattr(target, path[-1], int(value))


def _validated_fold_jobs(
    fold_inputs: Any, *, output_dir: Path, seed: int
) -> tuple[tuple[Any, str, Path], ...]:
    """Resolve every native job directory before any prediction starts."""
    root = Path(output_dir).resolve()
    jobs: list[tuple[Any, str, Path]] = []
    directories: dict[Path, str] = {}
    for fold_input in fold_inputs:
        fold_input = dataclasses.replace(fold_input, rng_seeds=(seed,))
        job_name = fold_input.sanitised_name()
        if (
            not isinstance(job_name, str)
            or not job_name
            or job_name in {".", ".."}
            or "/" in job_name
            or "\\" in job_name
        ):
            raise ValueError(
                "AlphaFold 3 job output name must be one safe filename "
                f"component, got {job_name!r}"
            )
        job_dir = Path(output_dir) / job_name
        try:
            resolved = job_dir.resolve(strict=False)
        except OSError as error:
            raise ValueError(
                f"cannot resolve AlphaFold 3 output directory: {job_dir}"
            ) from error
        if job_dir.is_symlink() or not resolved.is_relative_to(root):
            raise ValueError(
                f"AlphaFold 3 job output escapes its run directory: {job_dir}"
            )
        previous = directories.get(resolved)
        if previous is not None:
            raise ValueError(
                f"AlphaFold 3 jobs {previous!r} and {job_name!r} share output "
                f"directory {job_dir}"
            )
        directories[resolved] = job_name
        jobs.append((fold_input, job_name, job_dir))
    return tuple(jobs)


class AlphaFold3Backend(Backend):
    name = "alphafold3"
    session_reuse = True
    padding_axes = ("tokens",)
    native_options = frozenset(
        {
            "buckets",
            "device",
            "kernel_autotuning",
            "platform",
            "return_distogram",
            "return_embeddings",
            "source",
        }
    )
    # `make_model_config` takes only four of these as parameters, but it returns
    # a plain mutable config and already sets `num_samples` by assignment. The
    # step count and the MSA depth live one level deeper -- at
    # `heads.diffusion.eval.steps` (200) and `evoformer.num_msa` (1024) -- and
    # are set the same way. They were previously reported as unsupported, which
    # made AlphaFold 3 the one backend that could not be held to the same
    # schedule as the others, and so could not be benchmarked against them.
    sampling_options: dict[str, str] = {
        "num_samples": "diffusion_samples",
        "num_steps": "diffusion_steps",
        "num_recycles": "recycles",
        "max_msa_depth": "max_msa_depth",
    }
    # AlphaFold 3 runs `bfloat16: 'all'` inside the model it ships, so there is
    # no dtype for a caller to choose here; the knob would be a lie.
    execution_options = {
        "attention_kernel": (
            "attention_backend",
            {"auto": "triton", "xla": "xla"},
        ),
    }
    compile_options = (
        "diffusion_samples",
        "diffusion_steps",
        "recycles",
        "max_msa_depth",
        "buckets",
        "attention_backend",
        "return_embeddings",
        "return_distogram",
        "kernel_autotuning",
    )

    def __init__(self) -> None:
        self._session_open = False
        self._session_active = False
        self._session_poisoned: str | None = None
        self._asset_anchors: dict[
            tuple[str, str, str], tuple[tuple[str, str, str], ...] | None
        ] = {}
        self._model_runner: Any | None = None
        self._model_runner_key: tuple[Any, ...] | None = None

    @contextmanager
    def session(self, requests: Sequence[PredictionRequest]) -> Iterator[Backend]:
        """Retain one managed ``ModelRunner`` inside one API request only."""

        if self._session_open:
            raise RuntimeError("nested AlphaFold3 backend sessions are not supported")
        self._session_open = True
        attempts = sum(len(request.resolved_seeds) for request in requests)
        # A scalar prediction has no reuse opportunity. Keeping it on the
        # historical path also preserves wrappers that instrument construction.
        self._session_active = attempts > 1
        try:
            yield self
        finally:
            self.invalidate_session()
            self._asset_anchors.clear()
            self._session_poisoned = None
            self._session_active = False
            self._session_open = False

    def invalidate_session(self) -> None:
        """Drop the runner, its lazy parameters, and its JIT callable."""

        self._model_runner = None
        self._model_runner_key = None

    def _raise_if_poisoned(self) -> None:
        if self._session_poisoned is not None:
            raise PredictionError(self._session_poisoned)

    def _poison(self, message: str) -> None:
        self.invalidate_session()
        self._session_poisoned = message
        raise PredictionError(message)

    def _managed_weights(self, request: PredictionRequest) -> Path | None:
        """Return weights only for the vendored route, without importing it."""

        self._raise_if_poisoned()
        options = self.apply_sampling(request)
        if options.get("source"):
            return None
        assert request.weights is not None
        return Path(request.weights)

    def _anchor_assets(
        self,
        weights: Path,
        *,
        require_verifiable: bool = False,
    ) -> tuple[tuple[str, str, str], tuple[tuple[str, str, str], ...] | None]:
        """Bind this session to one immutable managed model generation."""

        self._raise_if_poisoned()
        source = _managed_source_key(weights)
        snapshot = _managed_asset_snapshot(weights)
        missing = object()
        expected = self._asset_anchors.get(source, missing)
        if expected is missing:
            self._asset_anchors[source] = snapshot
        elif expected is None:
            # A source that could not be proven immutable stays on the fresh
            # compatibility path for the rest of this session.
            snapshot = None
        elif snapshot != expected:
            self._poison(
                "AlphaFold3 managed weights, runner or source changed while "
                "a prediction batch was active"
            )
        if require_verifiable and snapshot is None:
            self._poison(
                "AlphaFold3 cannot verify weights used by a resumed prediction"
            )
        return source, snapshot

    def validate_session(self, request: PredictionRequest) -> None:
        if not self._session_active:
            return
        weights = self._managed_weights(request)
        if weights is not None:
            self._anchor_assets(weights)

    def observe_resumed(self, request: PredictionRequest) -> None:
        if not self._session_active:
            return
        weights = self._managed_weights(request)
        if weights is not None:
            self._anchor_assets(weights, require_verifiable=True)

    def _select_model_runner(
        self,
        *,
        runner: Any,
        runner_path: Path,
        config: Any,
        config_identity: tuple[Any, ...],
        device: Any,
        request: PredictionRequest,
        buckets: tuple[int, ...] | None,
        kernel_fallback: str,
        anchor: tuple[tuple[str, str, str], tuple[tuple[str, str, str], ...] | None]
        | None,
    ) -> tuple[Any, tuple[Any, ...] | None]:
        """Construct lazily, returning a retention key only when safe."""

        key = None
        if anchor is not None and anchor[1] is not None:
            key = (
                anchor,
                _runner_identity(runner, runner_path),
                _device_identity(device),
                config_identity,
                _cache_identity(request.cache_dir),
                kernel_fallback,
                buckets,
                request.padding is not None,
            )
        if self._model_runner is not None:
            if key is not None and self._model_runner_key == key:
                return self._model_runner, key
            # Never construct generation B while generation A's 1.1 GB tree is
            # still retained by this backend.
            self.invalidate_session()
        return (
            runner.ModelRunner(
                config=config,
                device=device,
                model_dir=request.weights,
            ),
            key,
        )

    def _retain_model_runner(
        self,
        model_runner: Any,
        key: tuple[Any, ...] | None,
    ) -> None:
        if key is not None:
            self._model_runner = model_runner
            self._model_runner_key = key

    def validate_native_options(self, options: dict[str, Any]) -> None:
        for name in ("return_embeddings", "return_distogram"):
            _strict_boolean(options.get(name, False), name=name)
        if "buckets" in options:
            _validated_buckets(options["buckets"])
        if "device" in options:
            _strict_integer(options["device"], name="device", minimum=0)
        fallback = options.get("kernel_autotuning", "autotune")
        if fallback not in {"heuristics", "autotune", "error"}:
            raise ValueError(
                "kernel_autotuning must be one of 'heuristics', 'autotune', or 'error'"
            )

    def capabilities(self) -> ModelCapabilities:
        requirement = InputRequirement(
            preprocessing_runtime="native",
            required_extras=("alphafold3",),
            notes=(
                "Uses the vendored AlphaFold 3 input pipeline and its prepared "
                "ABI-specific runtime; inspect it with `foldjax runtime status "
                "--model alphafold3`."
            ),
        )
        return ModelCapabilities(
            model=self.name,
            sampling=dict(self.sampling_options),
            input_formats=("native", "alphafold3", "foldjax"),
            input_requirements={
                name: requirement for name in ("native", "alphafold3", "foldjax")
            },
            padding_axes=self.padding_axes,
        )

    def validate_request(self, request: PredictionRequest) -> None:
        if request.padding is not None and "buckets" in request.options:
            raise ValueError(
                "padding and the native AlphaFold3 option 'buckets' were both "
                "set; pass one of them"
            )
        if request.padding is not None and request.options.get("source") is not None:
            raise ValueError(
                "neutral AlphaFold3 padding requires FoldJAX's managed runtime; "
                "an external source cannot guarantee prefix-stable diffusion noise"
            )
        super().validate_request(request)

    def predict(self, request: PredictionRequest) -> PredictionResult:
        options = self.apply_sampling(request)
        buckets = _prediction_buckets(request, options)
        self._raise_if_poisoned()
        managed_weights = (
            Path(request.weights)
            if self._session_active and not options.get("source")
            else None
        )
        anchor = (
            self._anchor_assets(managed_weights)
            if managed_weights is not None
            else None
        )

        try:
            runner_path = _runner_path(options)
            runner = _load_runner(runner_path)
            import jax
            from alphafold3.common import folding_input

            jobs = _validated_fold_jobs(
                folding_input.load_fold_inputs_from_path(request.input),
                output_dir=request.output_dir,
                seed=request.seed,
            )
            if request.cache_dir is not None:
                request.cache_dir.mkdir(parents=True, exist_ok=True)
                jax.config.update("jax_compilation_cache_dir", str(request.cache_dir))
                jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
            # Selecting from the default backend keeps a CPU-only host usable
            # for smoke runs while still resolving the GPU on an accelerator.
            platform = options.pop("platform", None)
            devices = (
                jax.local_devices(backend=platform) if platform else jax.local_devices()
            )
            device = devices[int(options.pop("device", 0))]
            attention_backend = options.pop("attention_backend", "triton")
            diffusion_samples = int(options.pop("diffusion_samples", 5))
            recycles = int(options.pop("recycles", 10))
            return_embeddings = _strict_boolean(
                options.pop("return_embeddings", False), name="return_embeddings"
            )
            return_distogram = _strict_boolean(
                options.pop("return_distogram", False), name="return_distogram"
            )
            diffusion_steps = options.pop("diffusion_steps", None)
            max_msa_depth = options.pop("max_msa_depth", None)
            kernel_fallback = str(options.pop("kernel_autotuning", "autotune"))
            if options:
                raise ValueError(
                    f"unsupported AlphaFold 3 options: {', '.join(options)}"
                )
            config = runner.make_model_config(
                flash_attention_implementation=attention_backend,
                num_diffusion_samples=diffusion_samples,
                num_recycles=recycles,
                return_embeddings=return_embeddings,
                return_distogram=return_distogram,
            )
            _set_nested(config, _DIFFUSION_STEPS, diffusion_steps)
            _set_nested(config, _MSA_DEPTH, max_msa_depth)
            config_identity = (
                ("attention_backend", attention_backend),
                ("diffusion_samples", diffusion_samples),
                ("recycles", recycles),
                ("return_embeddings", return_embeddings),
                ("return_distogram", return_distogram),
                ("diffusion_steps", diffusion_steps),
                ("max_msa_depth", max_msa_depth),
            )
            model_runner, model_runner_key = self._select_model_runner(
                runner=runner,
                runner_path=runner_path,
                config=config,
                config_identity=config_identity,
                device=device,
                request=request,
                buckets=buckets,
                kernel_fallback=kernel_fallback,
                anchor=anchor,
            )
            all_results = []
            samples: list[PredictionSample] = []
            padding_plans: list[dict[str, Any]] = []
            if request.padding is not None:
                assert buckets is not None
                run_jobs = _prepare_padded_jobs(
                    jobs,
                    buckets=buckets,
                    overflow=request.padding.overflow,
                    fixed_target=request.padding.tokens is not None,
                )
            else:
                run_jobs = tuple(
                    (fold_input, job_name, job_dir, None, None)
                    for fold_input, job_name, job_dir in jobs
                )
            with _tokamax_kernel_fallback(kernel_fallback), _model_device(device):
                for (
                    fold_input,
                    job_name,
                    job_dir,
                    examples,
                    resolved_plan,
                ) in run_jobs:
                    if examples is not None:
                        results = _predict_featurized_structure(
                            fold_input,
                            examples,
                            model_runner,
                            runner,
                        )
                    else:
                        results = runner.predict_structure(
                            fold_input, model_runner, buckets=buckets
                        )
                    runner.write_outputs(results, job_dir, job_name)
                    all_results.extend(results)
                    if request.padding is not None:
                        assert resolved_plan is not None
                        padding_plans.append(resolved_plan.summary())
                    samples.extend(
                        _samples(
                            results,
                            job_dir,
                            job_name,
                            sample_offset=len(samples),
                        )
                    )
            shape_profile = _public_shape_profile(request, padding_plans)
            native_raw = tuple(all_results)
            result = PredictionResult(
                model=self.name,
                samples=tuple(samples),
                output_dir=request.output_dir,
                raw=(
                    {"results": native_raw, "padding": shape_profile}
                    if shape_profile is not None
                    else native_raw
                ),
                shape_profile=shape_profile,
            )
        except BaseException as error:
            # Never let a provenance recheck replace user cancellation.  A
            # runner involved in any abnormal exit is not safe to retain; for
            # ordinary prediction errors we still promote an observed asset
            # race to the more useful session-level PredictionError.
            self.invalidate_session()
            if isinstance(error, Exception) and managed_weights is not None:
                self._anchor_assets(managed_weights)
            raise

        if managed_weights is not None:
            self._anchor_assets(managed_weights)
        self._retain_model_runner(model_runner, model_runner_key)
        return result


def _samples(
    results: Any,
    job_dir: Path,
    job_name: str,
    *,
    sample_offset: int = 0,
) -> list[PredictionSample]:
    """Normalize AlphaFold 3 ``ResultsForSeed`` values into common samples.

    The sample layout is reconstructed from the runner's own naming rather than
    globbed, so every structure keeps the seed and ranking score that produced
    it instead of an arbitrary directory order.
    """
    samples = []
    for results_for_seed in results:
        seed = int(getattr(results_for_seed, "seed", 0))
        for index, result in enumerate(
            getattr(results_for_seed, "inference_results", ())
        ):
            prefix = f"{job_name}_seed-{seed}_sample-{index}"
            sample_dir = job_dir / f"seed-{seed}_sample-{index}"
            structure_path = sample_dir / f"{prefix}_model.cif"
            scores = scalar_scores(sample_dir / f"{prefix}_summary_confidences.json")
            metadata = getattr(result, "metadata", None)
            if isinstance(metadata, dict) and "ranking_score" in metadata:
                scores["ranking_score"] = float(metadata["ranking_score"])
            samples.append(
                PredictionSample(
                    seed=seed,
                    structure_path=(
                        structure_path if structure_path.is_file() else None
                    ),
                    scores=scores,
                    metadata={
                        "job": job_name,
                        "native_sample": index,
                        "sample": sample_offset + len(samples),
                    },
                )
            )
    return samples
