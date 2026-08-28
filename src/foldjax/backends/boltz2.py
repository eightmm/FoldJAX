"""Boltz2-JAX adapter."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from foldjax.backends._representations import _representations_result
from foldjax.backends.base import MATMUL_PRECISION_OPTION, Backend
from foldjax.manifest import document_uses_key, path_stat_identity
from foldjax.models import _representations
from foldjax.models.boltz2.weights import resolve_native_weight_bundle
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


def _weight_bundle_snapshot(path: Path) -> tuple[Any, ...] | None:
    """Cheap identity of the exact payload/sidecar selected by ``load_params``."""

    bundle = resolve_native_weight_bundle(path)
    if bundle is None:
        return None
    weights, sidecar = bundle
    if sidecar.exists() and not sidecar.is_file():
        return None
    paths = [weights, *([sidecar] if sidecar.is_file() else [])]
    records: list[tuple[str, str, str]] = []
    for candidate in paths:
        identity = path_stat_identity(candidate)
        if identity is None:
            return None
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            return None
        records.append(
            (
                str(candidate.absolute()),
                str(resolved),
                json.dumps(identity, sort_keys=True, separators=(",", ":")),
            )
        )
    missing: tuple[str, ...] = ()
    if not sidecar.exists():
        try:
            missing_sidecar = sidecar.parent.resolve(strict=True) / sidecar.name
        except (OSError, RuntimeError):
            return None
        if missing_sidecar.exists():
            return None
        missing = (str(missing_sidecar),)
    return (
        ("requested", str(Path(path))),
        ("selected", str(weights.absolute())),
        ("assets", tuple(records)),
        ("missing", missing),
    )


def _weight_source_key(role: str, path: Path) -> tuple[str, str]:
    """Keep a relative spelling stable even if an embedding process changes cwd."""

    return role, str(Path(path))


class _BoundedJitRunner:
    """Keep a request from retaining an executable for every input shape."""

    _MAX_EXECUTABLES = 8

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = self._runner(*args, **kwargs)
        cache_size = getattr(self._runner, "_cache_size", None)
        clear = getattr(self._runner, "clear_cache", None)
        if callable(cache_size) and callable(clear):
            try:
                if int(cache_size()) >= self._MAX_EXECUTABLES:
                    clear()
            except Exception:  # noqa: BLE001 - accounting must not hide a result
                pass
        return result

    def clear_cache(self) -> None:
        clear = getattr(self._runner, "clear_cache", None)
        if callable(clear):
            clear()


def _native_module():
    """Import the Boltz-2 port and resolve its lazy prediction entry point.

    Prediction and featurization are both part of the base, torch-free FoldJAX
    install. Resolve the lazy attribute here so an incomplete installation says
    which base dependency is missing instead of recommending a parity-only extra.
    """
    native = import_module("foldjax.models.boltz2")
    try:
        native.predict
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "the boltz2 backend is included in FoldJAX's base installation, but "
            "one of its required dependencies is missing; reinstall FoldJAX "
            f"(missing: {error.name})"
        ) from error
    return native


#: Scalar confidence the Boltz-2 confidence head reports, one value per
#: diffusion sample. `pae` and the `*_logits` arrays stay in `raw`.
#:
#: `complex_plddt` and `complex_iplddt` are the head's own aggregates,
#: `(plddt * token_pad_mask).sum() / token_pad_mask.sum()`. They are what
#: upstream writes into its confidence JSON, so they are the pLDDT that can be
#: compared against it -- unlike the plain mean below, which weights padding
#: and interface tokens the same as everything else.
_CONFIDENCE_FIELDS = (
    "ptm",
    "iptm",
    "ligand_iptm",
    "protein_iptm",
    "complex_plddt",
    "complex_iplddt",
)


def _sample_scores(
    output: Mapping[str, object],
    plddt: np.ndarray,
    index: int,
    sample_count: int,
) -> dict[str, float]:
    """Confidence for one diffusion sample.

    Every field here is produced per sample, so reading a whole-batch mean or
    always element 0 makes the samples indistinguishable. That is what used to
    happen: `mean_plddt` was the mean over *all* samples and `iptm` was always
    the first one's, and the identical dict was copied onto each sample. Asking
    Boltz-2 for five structures returned five identical score sets, so ranking
    them -- the reason to generate more than one -- silently could not work.
    """
    scores: dict[str, float] = {}
    if plddt.size:
        # [num_samples, n_token] when more than one was requested, else [n_token].
        per_sample = plddt[index] if plddt.ndim == 2 else plddt
        scores["mean_plddt"] = float(per_sample.mean())
    raw = output.get("raw")
    for field in _CONFIDENCE_FIELDS:
        value = output.get(field)
        if value is None and isinstance(raw, Mapping):
            value = raw.get(field)
        if value is None:
            continue
        flat = np.asarray(value).reshape(-1)
        if flat.size == 0:
            continue
        # A per-sample vector is indexed; a single value applies to them all.
        scores[field] = float(flat[index] if flat.size == sample_count else flat[0])
    return scores


def _detach_prediction_output(
    output: Mapping[str, Any],
    *,
    coords: np.ndarray | None = None,
    plddt: np.ndarray | None = None,
    drop_representations: Sequence[str] = (),
) -> dict[str, Any]:
    """Move a common-result tree to host memory without redundant copies.

    The native API deliberately returns JAX arrays in ``raw`` for callers that
    want to keep working on device.  The common API has already converted the
    coordinates, pLDDT, and scalar scores before it constructs a
    :class:`PredictionResult`; retaining the native tree there would keep every
    compiled output buffer alive for as long as that result is reachable.

    Top-level affinity fields alias leaves in the nested native output.  Leave
    those aliases out of the transfer and reconnect them afterwards, both to
    avoid a second host copy and to preserve their identity relationship.
    Likewise, reuse coordinates and pLDDT that are already NumPy arrays.  A
    trunk-only caller may name representation leaves already owned by the lazy
    on-disk archive; remove those before transfer rather than duplicating a
    potentially quadratic pair array in host memory.
    """

    import jax

    transfer = dict(output)
    if coords is not None:
        transfer.pop("coords", None)
    if plddt is not None:
        transfer.pop("plddt", None)

    nested = transfer.get("raw")
    if isinstance(nested, Mapping):
        nested = dict(nested)
        for name in drop_representations:
            nested.pop(name, None)
        transfer["raw"] = nested

    affinity_aliases = tuple(
        key
        for key in transfer
        if key.startswith("affinity_") and isinstance(nested, Mapping) and key in nested
    )
    for key in affinity_aliases:
        transfer.pop(key)

    transferred = dict(jax.device_get(transfer))
    transferred_nested = transferred.get("raw")
    detached: dict[str, Any] = {}
    for key in output:
        if key == "coords" and coords is not None:
            detached[key] = coords
        elif key == "plddt" and plddt is not None:
            detached[key] = plddt
        elif key in affinity_aliases:
            assert isinstance(transferred_nested, Mapping)
            detached[key] = transferred_nested[key]
        else:
            detached[key] = transferred[key]
    return detached


def _default_mols(weights: Path) -> Path | None:
    """Find the CCD molecule directory that `foldjax weights fetch` unpacked.

    Boltz reads per-molecule pickles from a directory rather than from the
    checkpoint, so the weight file alone is never enough. The fetcher puts it
    beside the weights; a hand-managed layout can keep it one level up.
    """
    for candidate in (weights.parent / "mols", weights.parent.parent / "mols"):
        if candidate.is_dir():
            return candidate
    return None


def _padding_shape_profile(metadata: object) -> dict[str, object] | None:
    """Report every independently compiled stage in a Boltz2 padded run."""

    if not isinstance(metadata, Mapping):
        return None
    primary = metadata.get("primary")
    if not isinstance(primary, Mapping):
        return None
    affinity = metadata.get("affinity")
    if isinstance(affinity, Mapping):
        return {"primary": dict(primary), "affinity": dict(affinity)}
    return dict(primary)


#: Compile-relevant defaults released by :func:`foldjax.models.boltz2.api.predict`.
#:
#: Keep this list beside the adapter rather than importing the native API while
#: planning a request: cache-directory selection must stay free of the model
#: runtime.  A signature drift test pins every value to the native authority.
_RELEASED_COMPILE_DEFAULTS: dict[str, object] = {
    "num_steps": 200,
    "num_recycles": 3,
    "num_samples": 1,
    "cp_atom_windows": True,
    "cp_devices": 1,
    "cp_layout": "auto",
    "affinity_num_steps": 200,
    "affinity_num_samples": 5,
    "compute_dtype": "bfloat16",
    "attention_backend": "xla",
    "triangle_backend": "cueq",
    "glu_backend": "xla",
    "bucket": False,
}


class Boltz2Backend(Backend):
    name = "boltz2"
    session_reuse = True
    padding_axes = ("tokens", "atoms", "msa")
    native_options = frozenset(
        {
            "affinity_num_samples",
            "affinity_mw_correction",
            "affinity_num_steps",
            "affinity_weights",
            "bucket",
            "cp_atom_windows",
            "cp_devices",
            "cp_layout",
            "diffusion_chunk_size",
            "feature_cache",
            "glu_backend",
            "mols",
            "msa_api_key_header",
            "msa_api_key_value",
            "msa_pairing_strategy",
            "msa_server_password",
            "msa_server_url",
            "msa_server_username",
            "return_confidence_logits",
            "steering_args",
            "use_msa_server",
            "write_fmt",
        }
    )
    sampling_options = {
        "num_samples": "num_samples",
        "num_steps": "num_steps",
        "num_recycles": "num_recycles",
        "max_msa_depth": "max_msa_depth",
    }
    # Neutral knob -> (this port's name, {neutral value: its value}). Boltz-2
    # already spells the values the neutral way; the names are its own.
    execution_options = {
        **MATMUL_PRECISION_OPTION,
        "dtype": ("compute_dtype", {"float32": "float32", "bfloat16": "bfloat16"}),
        "triangle_kernel": (
            "triangle_backend",
            {"auto": "cueq", "cueq": "cueq", "xla": "xla"},
        ),
        "attention_kernel": (
            "attention_backend",
            {"auto": "xla", "tokamax": "tokamax", "xla": "xla"},
        ),
    }
    compile_options = (
        "num_steps",
        "num_recycles",
        "num_samples",
        "cp_atom_windows",
        "cp_devices",
        "cp_layout",
        "affinity_num_steps",
        "affinity_num_samples",
        "compute_dtype",
        "max_msa_depth",
        "attention_backend",
        "triangle_backend",
        "glu_backend",
        "bucket",
    )

    def __init__(self) -> None:
        self._session_open = False
        self._session_active = False
        self._session_poisoned: str | None = None
        self._asset_anchors: dict[
            tuple[str, str], tuple[Any, ...] | None
        ] = {}
        self._params: dict[
            str, tuple[tuple[Any, ...], tuple[Any, ...] | None, Any]
        ] = {}
        self._runners: dict[str, tuple[tuple[Any, ...], Any]] = {}

    @contextmanager
    def session(self, requests: Sequence[PredictionRequest]) -> Iterator[Backend]:
        """Retain one primary tree, one affinity tree, and their JIT wrappers."""

        if self._session_open:
            raise RuntimeError("nested Boltz2 backend sessions are not supported")
        self._session_open = True
        attempts = sum(len(request.resolved_seeds) for request in requests)
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
        self._params.clear()
        for role in tuple(self._runners):
            self._drop_runner(role)

    def cache_profile(self, request: PredictionRequest) -> dict[str, Any]:
        """Keep explicit released defaults in the omitted cache namespace.

        The native API resolves omitted options to these exact values before it
        builds either retained runner identity.  Naming them explicitly must
        therefore not select another persistent cache directory.  Likewise,
        its context-parallel resolver maps both ``auto`` and ``1d`` to ``1d``
        for serial and multi-device runs.  Other conditional aliases remain
        distinct here even where a current control-flow branch ignores them.
        """

        profile = super().cache_profile(request)
        for name, default in _RELEASED_COMPILE_DEFAULTS.items():
            if name not in profile:
                continue
            value = profile[name]
            # ``bool`` is an ``int`` subclass. Preserve malformed or future
            # type variants rather than allowing True to alias integer 1.
            if type(value) is type(default) and value == default:
                profile.pop(name)
        if profile.get("cp_layout") == "1d":
            profile.pop("cp_layout")
        return profile

    def _drop_runner(self, role: str) -> None:
        cached = self._runners.pop(role, None)
        if cached is None:
            return
        clear = getattr(cached[1], "clear_cache", None)
        if callable(clear):
            try:
                clear()
            except Exception:  # noqa: BLE001 - cleanup must never hide a result
                pass

    def prepare(self, *, affinity_requested: bool) -> None:
        """Drop the optional second model before a primary-only prediction."""

        if not self._session_active or affinity_requested:
            return
        self._params.pop("affinity", None)
        self._drop_runner("affinity")

    def _raise_if_poisoned(self) -> None:
        if self._session_poisoned is not None:
            raise PredictionError(self._session_poisoned)

    def _poison(self, message: str) -> None:
        self.invalidate_session()
        self._session_poisoned = message
        raise PredictionError(message)

    def _request_weight_paths(
        self, request: PredictionRequest
    ) -> tuple[Path, Path | None]:
        self._raise_if_poisoned()
        assert request.weights is not None
        primary = Path(request.weights)
        affinity = None
        if request.stop_after != "trunk" and document_uses_key(request, "affinity"):
            configured = self.apply_sampling(request).get("affinity_weights")
            affinity = (
                Path(configured)
                if configured is not None
                else primary.with_name("boltz2_aff")
            )
        return primary, affinity

    def _anchor_weight(
        self,
        role: str,
        path: Path,
        *,
        require_verifiable: bool = False,
    ) -> tuple[tuple[str, str], tuple[Any, ...] | None]:
        self._raise_if_poisoned()
        source = _weight_source_key(role, path)
        snapshot = _weight_bundle_snapshot(path)
        missing = object()
        expected = self._asset_anchors.get(source, missing)
        if expected is missing:
            self._asset_anchors[source] = snapshot
        elif expected is None:
            snapshot = None
        elif snapshot != expected:
            self._poison(
                f"Boltz2 {role} weights changed while a prediction batch was active"
            )
        if require_verifiable and snapshot is None:
            self._poison(
                f"Boltz2 cannot verify {role} weights used by a resumed prediction"
            )
        return source, snapshot

    def validate_session(self, request: PredictionRequest) -> None:
        if not self._session_active:
            return
        primary, affinity = self._request_weight_paths(request)
        self._anchor_weight("primary", primary)
        if affinity is not None:
            self._anchor_weight("affinity", affinity)

    def observe_resumed(self, request: PredictionRequest) -> None:
        if not self._session_active:
            return
        primary, affinity = self._request_weight_paths(request)
        self._anchor_weight("primary", primary, require_verifiable=True)
        if affinity is not None:
            self._anchor_weight("affinity", affinity, require_verifiable=True)

    def load_params(
        self,
        role: str,
        path: Path,
        loader: Callable[[Path], Any],
        *,
        placement: tuple[Any, ...],
    ) -> Any:
        """Load lazily and retain only an exactly revalidated native bundle."""

        if not self._session_active:
            return loader(path)
        source, snapshot = self._anchor_weight(role, path)
        key = (source, snapshot, placement)
        cached = self._params.get(role)
        if cached is not None:
            if snapshot is not None and cached[0] == key:
                return cached[2]
            self._params.pop(role, None)
            self._drop_runner(role)
            # Do not keep the evicted multi-gigabyte tree alive in this local
            # while the replacement loader builds another one.
            del cached

        params = loader(path)
        after = _weight_bundle_snapshot(path)
        if snapshot is None or after is None:
            return params
        if after != snapshot:
            self._poison(f"Boltz2 {role} weights changed while they were loading")
        self._params[role] = (key, None, params)
        return params

    def place_params(
        self,
        role: str,
        params: Any,
        *,
        placement: tuple[Any, ...],
        placer: Callable[[Any], Any],
    ) -> Any:
        """Retain the mesh-placed tree instead of re-transferring it per seed."""

        if not self._session_active:
            return placer(params)
        self._raise_if_poisoned()
        cached = self._params.get(role)
        if cached is None or cached[2] is not params:
            # Unverifiable bundles deliberately never enter the parameter
            # cache. Preserve their historical fresh placement as well.
            return placer(params)
        if cached[1] == placement:
            return cached[2]
        placed = placer(params)
        self._params[role] = (cached[0], placement, placed)
        return placed

    def jit_runner(
        self,
        role: str,
        key: tuple[Any, ...],
        function: Callable[..., Any],
        jit_factory: Callable[[Callable[..., Any]], Any],
    ) -> Any:
        """Reuse one exact graph owner per primary/affinity role."""

        if not self._session_active:
            return jit_factory(function)
        self._raise_if_poisoned()
        cached = self._runners.get(role)
        if cached is not None and cached[0] == key:
            return cached[1]
        self._drop_runner(role)
        runner = _BoundedJitRunner(jit_factory(function))
        self._runners[role] = (key, runner)
        return runner

    def validate_native_options(self, options: dict[str, object]) -> None:
        if "num_steps" in options:
            # The published Karras schedule divides by ``num_steps - 1``.  One
            # step therefore produces a NaN schedule and only fails after an
            # expensive model compile; reject it while planning instead.
            _strict_integer(options["num_steps"], name="num_steps", minimum=2)
        if "cp_devices" in options:
            _strict_integer(options["cp_devices"], name="cp_devices", minimum=1)
        if "diffusion_chunk_size" in options:
            _strict_integer(
                options["diffusion_chunk_size"],
                name="diffusion_chunk_size",
                minimum=1,
            )
        if "cp_layout" in options and options["cp_layout"] not in {
            "auto",
            "1d",
            "2d",
        }:
            raise ValueError("cp_layout must be one of 'auto', '1d', or '2d'")
        for name in (
            "affinity_mw_correction",
            "bucket",
            "cp_atom_windows",
            "return_confidence_logits",
            "use_msa_server",
        ):
            if name in options:
                _strict_boolean(options[name], name=name)
        if "write_fmt" in options and options["write_fmt"] not in {
            None,
            "cif",
            "pdb",
        }:
            raise ValueError("write_fmt must be one of 'cif', 'pdb', or null")

    def capabilities(self) -> ModelCapabilities:
        requirement = InputRequirement(
            notes=(
                "NumPy featurization and JAX prediction are included in the base "
                "install and never select a second tensor runtime."
            )
        )
        return ModelCapabilities(
            representations=_representations.available("boltz2"),
            model=self.name,
            sampling=dict(self.sampling_options),
            input_formats=("native", "boltz", "foldjax"),
            input_requirements={
                name: requirement for name in ("native", "boltz", "foldjax")
            },
            supports_affinity=True,
            padding_axes=self.padding_axes,
        )

    def validate_request(self, request: PredictionRequest) -> None:
        if request.padding is not None and "bucket" in request.options:
            raise ValueError(
                "padding and the native Boltz2 option 'bucket' were both set; "
                "pass one of them"
            )
        super().validate_request(request)

    def predict(self, request: PredictionRequest) -> PredictionResult:
        self._raise_if_poisoned()
        if self._session_active:
            self.validate_session(request)
            # The native API confirms this from realized features, but the
            # structured job already lets a primary-only run release the
            # optional second model before featurization starts.
            _primary, affinity = self._request_weight_paths(request)
            self.prepare(affinity_requested=affinity is not None)
        options = self.apply_sampling(request)
        # Out before `**options` reaches the native signature: no model takes
        # this as an argument, the scope carries it.
        matmul_precision = self.matmul_precision(options)
        mols = options.pop("mols", None) or _default_mols(request.weights)
        if mols is None:
            raise ValueError(
                "Boltz2 prediction needs its CCD molecule directory. Run "
                "`foldjax weights fetch --model boltz2`, which unpacks it beside "
                "the weights, or pass --option mols=/path/to/mols"
            )
        native = _native_module()
        wanted = _representations.resolve(
            request.representations, _representations.specs_for("boltz2")
        )
        native_options: dict[str, Any] = dict(
            representations=wanted or None,
            representations_dir=request.output_dir,
            stop_after=request.stop_after,
            input=request.input,
            weights=request.weights,
            mols=Path(mols),
            out_dir=request.output_dir,
            seed=request.seed,
            compile_cache=request.cache_dir,
            padding=request.padding,
            write_fmt=options.pop("write_fmt", "cif"),
            **options,
        )
        if self._session_active:
            native_options["_runtime"] = self
        with matmul_precision():
            output = native.predict(**native_options)
        if request.stop_after == "trunk":
            # Nothing was folded, so there are no samples to describe.
            representations = _representations_result(
                self.name, request.output_dir, wanted
            )
            archived = (
                tuple(name for name in wanted if name in representations)
                if representations is not None
                else ()
            )
            output = _detach_prediction_output(
                output, drop_representations=archived
            )
            return PredictionResult(
                model=self.name,
                samples=(),
                output_dir=request.output_dir,
                raw=output,
                representations=representations,
            )
        coords = np.asarray(output["coords"])
        plddt = np.asarray(output.get("plddt", []))
        sample_count = coords.shape[0] if coords.ndim == 3 else 1
        paths = output.get("out_paths")
        if paths is None:
            paths = [output.get("out_path")] * sample_count
        samples = tuple(
            PredictionSample(
                seed=request.seed,
                structure_path=Path(paths[index]) if paths[index] else None,
                coordinates=coords[index] if coords.ndim == 3 else coords,
                scores=_sample_scores(output, plddt, index, sample_count),
            )
            for index in range(sample_count)
        )
        shape_profile = _padding_shape_profile(output.get("padding"))
        output = _detach_prediction_output(output, coords=coords, plddt=plddt)
        return PredictionResult(
            model=self.name,
            samples=samples,
            output_dir=request.output_dir,
            raw=output,
            shape_profile=shape_profile,
            representations=_representations_result(
                self.name, request.output_dir, wanted
            ),
        )
