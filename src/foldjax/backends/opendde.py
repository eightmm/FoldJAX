"""OpenDDE-JAX adapter."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from importlib import import_module
from pathlib import Path

from foldjax.backends._representations import _representations_result
from foldjax.backends._weight_session import PreparedWeightSession
from foldjax.backends.base import MATMUL_PRECISION_OPTION, Backend
from foldjax.models import _representations
from foldjax.models._managed_memory import lease as managed_memory_lease
from foldjax.schema import (
    InputRequirement,
    ModelCapabilities,
    PredictionRequest,
    PredictionResult,
    PredictionSample,
    _strict_boolean,
)
from foldjax.scores import scalar_scores

# Every environment variable the native CLI assigns. Asserted against its
# source in tests/test_native_contracts.py so a new export cannot escape.
_EXPORTED_ENVIRONMENT = (
    "JAX_PLATFORMS",
    "PROTENIX_CCD_COMPONENTS_FILE",
    "PROTENIX_CCD_RDKIT_MOL_FILE",
    "PROTENIX_KALIGN_BINARY",
    "PROTENIX_TEMPLATE_MMCIF_DIR",
    "PROTENIX_TEMPLATE_OBSOLETE_FILE",
    "PROTENIX_TEMPLATE_RELEASE_DATES_FILE",
)
_CLI_OPTIONS = {
    "ccd_rdkit_cache",
    "components_cif",
    "cp_devices",
    "cp_layout",
    "diffusion_attention_backend",
    "kalign_binary",
    "max_msa_depth",
    "token_q_chunk_size",
    "single_att_q_chunk_size",
    "triangle_att_q_chunk_size",
    "triangle_mul_chunk_size",
    "chunk_policy",
    "num_recycles",
    "n_keys",
    "n_queries",
    "num_samples",
    "num_steps",
    "structural_single_attention_backend",
    "template_mmcif_dir",
    "template_obsolete_map",
    "template_release_dates",
    "trunk_dtype",
    "trunk_single_attention_backend",
}
_CONFIDENCE_INFIX = "_summary_confidence_sample_"


class OpenDDEBackend(Backend):
    name = "opendde"
    session_reuse = True
    # OpenDDE has two token spaces: the residue trunk and the expanded
    # structural diffusion branch.  Both, the atom axis, and sampled MSA rows
    # have end-to-end masks and are cropped before public output.
    padding_axes = ("tokens", "atoms", "msa", "structural_tokens")
    native_options = frozenset(_CLI_OPTIONS | {"include_raw"})
    sampling_options = {
        "num_samples": "num_samples",
        "num_steps": "num_steps",
        "num_recycles": "num_recycles",
        "max_msa_depth": "max_msa_depth",
    }
    # OpenDDE has no triangle-kernel option of its own -- it drives Protenix's
    # trunk but exposes only the chunk sizes -- so `triangle_kernel` is absent
    # here and asking for it is an error rather than a silent no-op.
    execution_options = {
        **MATMUL_PRECISION_OPTION,
        "dtype": ("trunk_dtype", {"float32": "fp32", "bfloat16": "bf16"}),
        "attention_kernel": (
            "trunk_single_attention_backend",
            {"auto": "xla_jit", "xla": "xla_jit"},
        ),
    }
    compile_options = tuple(sorted(_CLI_OPTIONS))

    def __init__(self) -> None:
        self._weights = PreparedWeightSession(self.name)
        self._managed_memory: ExitStack | None = None
        self._ccd_memory_leased = False

    @contextmanager
    def session(self, requests: Sequence[PredictionRequest]) -> Iterator[Backend]:
        memory = ExitStack()
        try:
            with self._weights.session(requests):
                self._managed_memory = memory
                try:
                    yield self
                finally:
                    self._managed_memory = None
                    self._ccd_memory_leased = False
        finally:
            try:
                memory.close()
            except BaseException:
                pass

    @contextmanager
    def _ccd_memory_scope(self) -> Iterator[None]:
        """Lease shared Protenix/OpenDDE chemistry lazily."""

        from foldjax.models.protenix.data.featurize_json import (
            _release_external_ccd_cache,
        )

        memory = self._managed_memory
        if memory is not None:
            if not self._ccd_memory_leased:
                memory.enter_context(
                    managed_memory_lease(
                        "protenix_external_ccd", _release_external_ccd_cache
                    )
                )
                self._ccd_memory_leased = True
            yield
        else:
            with managed_memory_lease(
                "protenix_external_ccd", _release_external_ccd_cache
            ):
                yield

    def invalidate_session(self) -> None:
        self._weights.invalidate()

    def validate_session(self, request: PredictionRequest) -> None:
        if self._weights.active and request.weights is not None:
            self._weights.validate(Path(request.weights))

    def observe_resumed(self, request: PredictionRequest) -> None:
        if self._weights.active and request.weights is not None:
            self._weights.validate(Path(request.weights), resumed=True)

    def validate_native_options(self, options: dict[str, object]) -> None:
        _strict_boolean(options.get("include_raw", False), name="include_raw")

    def cache_profile(self, request: PredictionRequest) -> dict[str, object]:
        """Include the normalized graph-output profile in cache identity."""

        profile = super().cache_profile(request)
        options = self.apply_sampling(request)
        self.validate_native_options(options)
        profile["return_confidence_details"] = _strict_boolean(
            options.get("include_raw", False), name="include_raw"
        )
        return profile

    def capabilities(self) -> ModelCapabilities:
        native_requirement = InputRequirement(
            notes=(
                "NumPy/Gemmi/RDKit featurization and JAX prediction are included "
                "in the base install and do not import PyTorch. Native "
                "templatesPath and RNA unpairedMsaPath fields retain OpenDDE's "
                "released compatibility policy: warn and drop because its "
                "shipped defaults disable them."
            )
        )
        common_requirement = InputRequirement(
            notes=(
                "NumPy/Gemmi/RDKit featurization and JAX prediction are included "
                "in the base install and do not import PyTorch. FoldJAX common "
                "inputs reject templates and RNA MSAs because OpenDDE's shipped "
                "defaults would discard them; protein MSAs remain supported."
            )
        )
        return ModelCapabilities(
            representations=_representations.available("opendde"),
            model=self.name,
            sampling=dict(self.sampling_options),
            input_formats=("native", "opendde", "foldjax"),
            input_requirements={
                "native": native_requirement,
                "opendde": native_requirement,
                "foldjax": common_requirement,
            },
            # The implementation contains template machinery, but the shipped
            # inference contract fixes use_template=False and discards the
            # native field. Capabilities describe the runnable configuration,
            # not dormant code paths.
            supports_templates=False,
            padding_axes=self.padding_axes,
        )

    def predict(self, request: PredictionRequest) -> PredictionResult:
        if not request.weights.is_file():
            raise FileNotFoundError(
                f"OpenDDE-JAX weights must be a native weight file: {request.weights}"
            )
        if request.weights.suffix.lower() in {".pt", ".pth", ".ckpt"}:
            raise ValueError(
                "OpenDDE-JAX prediction requires converted native weights; "
                "run opendde-jax-export-weights first"
            )

        options = self.apply_sampling(request)
        # Out before the leftover-option check below: carried by the scope, not
        # by argv.
        matmul_precision = self.matmul_precision(options)
        include_raw = options.pop("include_raw", False)
        if not isinstance(include_raw, bool):
            raise ValueError("include_raw must be a boolean")

        argv = [
            "--input-json",
            str(request.input),
            "--weights",
            str(request.weights),
            "--out",
            str(request.output_dir),
            "--seed",
            str(request.seed),
        ]
        if request.cache_dir is not None:
            argv.extend(("--compile-cache", str(request.cache_dir)))
        for key in sorted(_CLI_OPTIONS):
            if key in options:
                argv.extend((f"--{key.replace('_', '-')}", str(options.pop(key))))
        if include_raw:
            argv.append("--include-raw")
        wanted = _representations.resolve(
            request.representations, _representations.specs_for("opendde")
        )
        if wanted:
            argv.extend(("--representations", ",".join(wanted)))
            # Pinned so that every backend puts the archive in the same place;
            # each model's own output tree is shaped differently.
            argv.extend(("--representations-dir", str(request.output_dir)))
        if request.stop_after == "trunk":
            argv.extend(("--stop-after", "trunk"))
        if options:
            raise ValueError(f"unsupported OpenDDE options: {', '.join(options)}")

        padding_profiles: list[dict[str, object]] = []
        native = import_module("foldjax.models.opendde.cli.predict")
        use_session_loader = self._weights.active and bool(
            getattr(native, "PREPARED_PARAMS_LOADER_API", False)
        )

        def session_params_loader(
            path: Path,
            trunk_dtype: str,
            cacheable: bool,
        ) -> object:
            if not cacheable:  # pragma: no cover - OpenDDE always passes True
                self._weights.invalidate()
                return native._load_prepared_params(Path(path), trunk_dtype)
            return self._weights.load(
                Path(path),
                lambda source: native._load_prepared_params(source, trunk_dtype),
                prepare_key=("trunk_dtype", trunk_dtype),
            )

        with matmul_precision(), _restored_environment(), self._ccd_memory_scope():
            if request.padding is None:
                # Keep the historical one-argument native entry point exact for
                # embedding applications and default-off predictions.
                written = (
                    native.main(
                        argv,
                        _prepared_params_loader=session_params_loader,
                    )
                    if use_session_loader
                    else native.main(argv)
                )
            else:
                unsupported = sorted(
                    set(request.padding.explicit_axes) - set(self.padding_axes)
                )
                if unsupported:
                    raise ValueError(
                        "opendde does not support explicit padding axes: "
                        + ", ".join(unsupported)
                    )
                written = (
                    native.main(
                        argv,
                        padding=request.padding,
                        padding_profiles=padding_profiles,
                        _prepared_params_loader=session_params_loader,
                    )
                    if use_session_loader
                    else native.main(
                        argv,
                        padding=request.padding,
                        padding_profiles=padding_profiles,
                    )
                )
        samples = tuple(
            PredictionSample(
                seed=request.seed,
                structure_path=path,
                scores=_scores(path),
            )
            for path in written
            if path.suffix == ".cif"
        )
        shape_profile = _shape_profile(
            padding_profiles,
            padded=request.padding is not None,
        )
        raw: dict[str, object] = {"argv": tuple(argv)}
        if shape_profile is not None:
            raw["padding"] = shape_profile
        return PredictionResult(
            model=self.name,
            samples=samples,
            output_dir=request.output_dir,
            raw=raw,
            shape_profile=shape_profile,
            representations=_representations_result(
                self.name, request.output_dir, wanted
            ),
        )


@contextmanager
def _restored_environment() -> Iterator[None]:
    """Undo the asset environment variables the native CLI exports.

    OpenDDE's CLI hands asset paths to the Protenix featurizer through
    ``os.environ``, which is process-scoped and therefore harmless for its own
    entry point. FoldJAX runs that CLI in-process, so without this a job that
    passes ``components_cif`` would silently leave it applied to every later
    prediction in the same session, including ones for other backends.
    """
    saved = {name: os.environ.get(name) for name in _EXPORTED_ENVIRONMENT}
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _scores(structure_path: Path) -> dict[str, float]:
    name, separator, rank = structure_path.stem.rpartition("_sample_")
    if not separator:
        return {}
    return scalar_scores(
        structure_path.with_name(f"{name}{_CONFIDENCE_INFIX}{rank}.json")
    )


def _shape_profile(
    profiles: list[dict[str, object]],
    *,
    padded: bool,
) -> dict[str, object] | None:
    """Collapse identical native-job profiles without hiding heterogeneous runs."""

    if not padded:
        return None
    if not profiles:
        raise RuntimeError(
            "OpenDDE padding completed without reporting a concrete shape profile"
        )
    first = profiles[0]
    if all(profile == first for profile in profiles[1:]):
        return dict(first)
    return {"per_run": [dict(profile) for profile in profiles]}
