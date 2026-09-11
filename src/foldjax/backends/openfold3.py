"""OpenFold3-JAX adapter.

Unlike the other vendored backends this drives the port's Python API rather
than its CLI, because OpenFold3 splits featurization from inference on purpose:
featurization delegates to upstream's data stack, inference needs only JAX and
a checkpoint. Shelling out would force both into one process for no benefit.

Both halves are vendored. Prediction from a self-contained feature ``.npz``
needs only FoldJAX's JAX runtime; building those features from JSON/YAML uses
the in-package NumPy/JAX preprocessing path and its chemistry dependencies in
the ``openfold3-preprocess`` extra.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from foldjax._openfold3_compile import (
    resolve_triangle_kernel,
)
from foldjax._openfold3_compile import (
    triangle_backend as _triangle_backend,
)
from foldjax.backends._ccd_session import WeightSessionHooks
from foldjax.backends._representations import _representations_result
from foldjax.backends._weight_session import PreparedWeightSession
from foldjax.backends.base import MATMUL_PRECISION_OPTION, Backend
from foldjax.cache import compilation_cache_scope
from foldjax.execution import DETERMINISTIC_API_OPTION
from foldjax.models import _representations
from foldjax.padding import PaddingPlan, resolve_axis, resolve_token_axis
from foldjax.schema import (
    InputRequirement,
    ModelCapabilities,
    PaddingConfig,
    PredictionRequest,
    PredictionResult,
    PredictionSample,
    _strict_boolean,
)

# Options that change the compiled program, so they belong in the cache namespace.
# Everything else here only affects what is written.
_COMPILE_OPTIONS = (
    "num_samples",
    "num_steps",
    "num_recycles",
    "pair_chunk_size",
    "diffusion_chunk_size",
    "max_msa_depth",
    "cp_devices",
    "cp_layout",
    "triangle_kernel",
    "glu_backend",
    "all_arrays",
    # Two runs that differ only in reduction policy compile different
    # programs, so they must not share one namespace.
    "deterministic",
    # Narrowing the trunk changes the element type of the compiled program's
    # largest buffers, so the two dtypes are two executables.
    "dtype",
    # Same for the confidence head's own knob: it is a second element type in
    # the same program, not a runtime branch.
    "confidence_dtype",
)

#: ``released_config``'s dtype, which `confidence_dtype` follows when unset.
#: Named rather than spelled twice so the namespace-stripping rule below
#: cannot drift from the model's default. Copied rather than imported for the
#: same reason as `_GLU_BACKENDS` -- resolving a cache directory must not
#: import the model package or JAX -- and a test pins the copy to
#: `released_config`'s signature.
_DEFAULT_DTYPE = "float32"

# ``released_config``'s model-side MSA subsampling depth. The public
# ``max_msa_depth`` knob is a cap, so asking for more cannot widen the released
# model's 1024-row input.
_RELEASED_MSA_DEPTH = 1024
#: The common API defaults to structures and normalized confidence scores; retaining
#: OpenFold3's quadratic per-bin pair distributions is an explicit ``all_arrays``
#: native option. Keep the same opt-in on the raw ``openfold3-jax-predict`` entry
#: point and do not make every managed prediction retain the arrays merely to write
#: an otherwise-unreferenced archive. ``released_config`` interprets this as a
#: pair-logit budget only; coordinates and scalar/per-atom confidence outputs are
#: still written regardless of this value.
_MANAGED_ARRAY_BUDGET_BYTES = 0

# Private marker inserted by ``data.compact_zero_template_pair_features``.
# Keep the spelling pinned here because the padding planner intentionally runs
# without importing the model package at module import time.
_ZERO_TEMPLATE_PAIR_MARKER = "_foldjax_zero_template_pair_features"

#: Values ``glu_backend`` accepts, copied rather than imported: the shared
#: :mod:`foldjax.models._glu` pulls in JAX, and this module resolves cache
#: directories without paying that import. A drift test pins the copy.
_GLU_BACKENDS = ("xla", "tokamax")

#: ``released_config`` values whose explicit spellings are identical to leaving
#: the public request unset.  Keep these lightweight copies beside the backend
#: so resolving a cache directory does not import the model/JAX runtime; a test
#: pins them to ``released_config``'s signature.
_RELEASED_COMPILE_DEFAULTS = {
    "num_samples": 5,
    "num_steps": 200,
    # The neutral request says three recycles; ``apply_sampling`` translates
    # that to the four executed trunk cycles stored in ``InferenceConfig``.
    "num_recycles": 4,
    "max_msa_depth": _RELEASED_MSA_DEPTH,
    # Not a ``released_config`` parameter: the reduction policy rides on the
    # executable rather than on the model configuration. It is here because
    # ``off`` has to name the same cache scope as an unasked run.
    "deterministic": False,
}


def _real_prefix_size(mask: np.ndarray, *, axis: str) -> int:
    """Count one padded axis and reject holes before model compilation."""

    mask = np.asarray(mask, dtype=bool).reshape(-1)
    size = int(np.count_nonzero(mask))
    if not np.array_equal(mask, np.arange(mask.size) < size):
        raise ValueError(f"OpenFold3 {axis} padding must be a contiguous suffix")
    return size


def _padding_plan(
    features: dict[str, Any],
    config: PaddingConfig,
    *,
    max_msa_depth: int = _RELEASED_MSA_DEPTH,
) -> PaddingPlan:
    """Resolve compiled axes from the token bucket and native capacities."""

    token_mask = np.asarray(features["token_mask"]) > 0
    atom_mask = np.asarray(features["atom_mask"]) > 0
    msa_mask = np.asarray(features["msa_mask"]) > 0
    cycle_indices = features.get("_foldjax_msa_cycle_indices")
    if cycle_indices is not None:
        # The union stays on the host. Only one native selection crosses the
        # compiled boundary, so its width determines the serving MSA capacity.
        msa_mask = msa_mask[:, np.asarray(cycle_indices)[0]]
    if _ZERO_TEMPLATE_PAIR_MARKER in features:
        # Trusted raw preprocessing may compact the four all-zero template
        # geometry tensors before serving padding.  The retained restype still
        # carries the released template storage width; semantically there are
        # no non-empty templates in this branch.
        template_restype = np.asarray(features["template_restype"])
        template_mask = np.zeros(template_restype.shape[:-1], dtype=bool)
    else:
        template_mask = (
            np.asarray(features["template_backbone_frame_mask"]) > 0
        ) | (np.asarray(features["template_pseudo_beta_mask"]) > 0)

    msa_rows = np.any(msa_mask, axis=(0, 2))
    template_rows = np.any(template_mask, axis=(0, 2))
    actual = {
        "tokens": _real_prefix_size(token_mask, axis="token"),
        "atoms": _real_prefix_size(atom_mask, axis="atom"),
        "msa": _real_prefix_size(msa_rows, axis="MSA-row"),
        "templates": _real_prefix_size(template_rows, axis="template-row"),
    }
    storage = {
        "tokens": int(token_mask.shape[-1]),
        "atoms": int(atom_mask.shape[-1]),
        "msa": int(msa_mask.shape[-2]),
        "templates": int(template_mask.shape[-2]),
    }
    target = {
        "tokens": resolve_axis(
            actual["tokens"], config, "tokens", minimum=storage["tokens"]
        )
    }
    for axis, fixed_size in (("atoms", None), ("msa", max_msa_depth), ("templates", 4)):
        target[axis] = resolve_token_axis(
            actual[axis],
            config,
            axis,
            token_target=target["tokens"],
            minimum=storage[axis],
            fixed_size=fixed_size,
        )
    return PaddingPlan(actual=actual, storage=storage, target=target)


def _sampler_noise_mask(plan: PaddingPlan, *, num_samples: int) -> np.ndarray:
    """Mask the source storage prefix so padding cannot change sample strides."""

    source = plan.storage or plan.actual
    target_atoms = plan.target["atoms"]
    stored_atom_prefix = np.arange(target_atoms) < source["atoms"]
    return np.broadcast_to(stored_atom_prefix, (num_samples, target_atoms))


def _compile_enabled(options: dict[str, Any]) -> bool:
    """Consume ``no_compile`` without applying Python's truthiness coercion."""
    return not _strict_boolean(options.pop("no_compile", False), name="no_compile")


class OpenFold3Backend(WeightSessionHooks, Backend):
    name = "openfold3"
    session_reuse = True
    padding_axes = ("tokens", "atoms", "msa", "templates")
    native_options = frozenset(
        {
            "ccd_file_path",
            # The confidence head's own dtype. `dtype` reaches this set through
            # `execution_options`; `confidence_dtype` has no neutral spelling,
            # so without naming it here `validate_request` rejects the one
            # option `predict` below is written to consume -- the same shape of
            # gap the `sampling_options` comment records, where a knob the
            # backend advertised could only ever fail.
            "confidence_dtype",
            "cp_devices",
            "cp_layout",
            "glu_backend",
            "no_compile",
            "pair_chunk_size",
            "diffusion_chunk_size",
            "all_arrays",
            "prefix",
            "query_id",
        }
    )
    # OpenFold3 spells these `num_steps` and `num_recycles`, which is what
    # `released_config` takes and what `predict` below pops. Mapping them onto
    # their own neutral names put `num_steps` and `num_recycles` into the option
    # dict, where nothing consumed them, so both knobs raised "unsupported
    # OpenFold3 options" -- capabilities advertised two knobs that could only
    # ever fail. `max_msa_depth` overrides `released_config`'s `msa_depth`, which
    # already carries upstream's own 1024; the knob narrows a setting the model
    # has rather than imposing one it lacks.
    sampling_options = {
        "num_samples": "num_samples",
        "num_steps": "num_steps",
        "num_recycles": "num_recycles",
        "max_msa_depth": "max_msa_depth",
    }
    # OpenFold3 selects its triangle kernel from an environment variable rather
    # than an argument, because the switch has to reach every triangle attention
    # in the model -- the template stack and the confidence head included --
    # without six signatures growing a parameter. The neutral knob is translated
    # into that variable in `predict` below, so a caller says the same thing here
    # as anywhere else.
    #
    # `dtype` defaults to `float32`, upstream's released inference precision
    # (`entry_points/validator.py:127`). `bfloat16` is opt-in and its value
    # depends on the size: 23-35% faster and 24-40% smaller at every size
    # measured, with per-chain structures identical to float32 at 2,096
    # tokens but a 4.7-5.8 A same-index drift at 3,012 against a 0.6 A
    # control spread. models/openfold3/dtype.py carries the table and the
    # leading hypothesis for the drift.
    #
    # `bfloat16` does **not** narrow the whole model -- a whole-trunk cast,
    # input embedder included, takes pLDDT from 0.858 to 0.466. It narrows the
    # token/pair representation track and leaves everything atom- or
    # coordinate-shaped float32: the input embedder, the entire denoiser (atom
    # encoder, atom decoder and the 24-block token transformer), the diffusion
    # conditioning's single branch, the confidence head's geometry
    # re-embedding, and every output head -- so every output logit is float32
    # in both profiles, while attention softmaxes inside the narrowed regions
    # do run bfloat16, as they do in both upstreams. That is AlphaFold 3's
    # released inference shape, and its two float32 islands are ones upstream
    # OpenFold3 pins itself while training this checkpoint under `bf16-mixed`.
    # `inference.cast_narrow_params` is the parameter split and
    # models/openfold3/dtype.py carries the reading with line numbers.
    #
    # The confidence head is a second narrowing group with its own native
    # knob, `confidence_dtype`, which follows `dtype` unless set -- so opting
    # into a narrow trunk narrows it too, and the knob is there to hold that
    # one region wide against a narrowed trunk, or narrow it against a wide
    # one. Narrowing it alone measures as nothing (+0.4% wall / -0.1% peak at
    # 1,003 tokens), which is why it is not its own default. It is
    # not a third neutral value, for the same reason Boltz-2's
    # `diffusion_compute_dtype` is not: the neutral vocabulary names what
    # every port means by "dtype", and a region only one port has does not
    # belong in it.
    #
    # Annotated because the shared deterministic entry carries `bool` native
    # values rather than the `str` the rest of this table maps to.
    execution_options: dict[str, tuple[str, dict[str, Any]]] = {
        **MATMUL_PRECISION_OPTION,
        **DETERMINISTIC_API_OPTION,
        "dtype": ("dtype", {"float32": "float32", "bfloat16": "bfloat16"}),
        "triangle_kernel": (
            "triangle_kernel",
            {
                "auto": "cueq",
                "cueq": "cueq",
                "cueq-full": "cueq-full",
                "xla": "xla",
            },
        ),
    }
    compile_options = _COMPILE_OPTIONS

    def __init__(self) -> None:
        self._weights = PreparedWeightSession(self.name)

    @contextmanager
    def session(self, requests: Sequence[PredictionRequest]) -> Iterator[Backend]:
        with self._weights.session(requests):
            yield self

    def cache_profile(self, request: PredictionRequest) -> dict[str, Any]:
        """Name static choices without splitting released-default aliases."""

        profile = super().cache_profile(request)
        options = self.apply_sampling(request)
        all_arrays = _strict_boolean(
            options.get("all_arrays", False), name="all_arrays"
        )
        profile.pop("all_arrays", None)
        # A trunk-only graph returns before every confidence/output head and the
        # raw-array writer, so retaining native pair distributions cannot affect
        # that graph. Do not create an otherwise-identical cache namespace for a
        # no-op output option.
        if all_arrays and request.stop_after not in ("trunk", "inputs"):
            profile["all_arrays"] = True
        # ``released_config`` supplies these exact values when the request
        # omits them.  Keeping an explicitly repeated default in the namespace
        # creates another cache scope; that scope is itself part of
        # ``_PredictGraphIdentity``, so an otherwise identical model config
        # receives a second in-process JIT owner as well as another persistent
        # directory.  Strip only the proven released defaults, preserving the
        # historical omitted-default namespace and every non-default graph.
        for name, default in _RELEASED_COMPILE_DEFAULTS.items():
            if name not in options:
                continue
            # `bool` before `int`, because `bool` is an `int`: coercing the
            # reduction policy would put `1` in the namespace instead of the
            # value the request and `predict` both carry.
            resolved = (
                _strict_boolean(options[name], name=name)
                if isinstance(default, bool)
                else int(options[name])
            )
            if resolved == default:
                profile.pop(name, None)
            else:
                # Native options accept integer spellings; use the same value
                # that ``predict`` passes into ``released_config``.
                profile[name] = resolved
        # `dtype` is a string, so it misses the int/bool coercion above. An
        # explicitly repeated default names the same program an unasked run
        # gets, and a second namespace for it would also split the in-process
        # JIT pool.
        if profile.get("dtype") == _DEFAULT_DTYPE:
            profile.pop("dtype", None)
        # `confidence_dtype` follows `dtype` when unset, so an explicit value
        # equal to the resolved one names the same program. Compared against
        # the request's `dtype` rather than against a literal for that reason:
        # under `dtype=float32` the alias to strip is `float32`, not the
        # shipped `bfloat16`.
        if profile.get("confidence_dtype") == options.get("dtype", _DEFAULT_DTYPE):
            profile.pop("confidence_dtype", None)
        cp_shards = int(options.get("cp_devices", 1))
        requested_layout = str(options.get("cp_layout", "auto"))
        profile["cp_devices"] = cp_shards
        profile["cp_layout"] = (
            "serial"
            if cp_shards <= 1
            else "1d"
            if requested_layout == "auto"
            else requested_layout
        )
        profile["triangle_kernel"] = resolve_triangle_kernel(
            options.get("triangle_kernel"), cp_shards=cp_shards
        )
        # The released SwiGLU is the unfused one, so a request that spells
        # that default out must name the namespace an omitted option names.
        # The fused value is a different program and keeps its own.
        if str(profile.get("glu_backend", "xla")) == "xla":
            profile.pop("glu_backend", None)
        profile["representations"] = _representations.resolve(
            request.representations, _representations.specs_for("openfold3")
        )
        profile["stop_after"] = request.stop_after
        profile["rng_route"] = "mask" if request.padding is not None else "native"
        return profile

    def validate_native_options(self, options: dict[str, Any]) -> None:
        _compile_enabled(dict(options))
        if "all_arrays" in options:
            _strict_boolean(options["all_arrays"], name="all_arrays")
        # The neutral translation already rejects an unknown `dtype`; this
        # reaches the native spelling, which bypasses it, and it is the only
        # check `confidence_dtype` gets before the config is built. Compared
        # against the option table rather than against the model layer, so
        # validating a request does not import JAX.
        allowed = self.execution_options["dtype"][1]
        for name in ("dtype", "confidence_dtype"):
            if name in options and str(options[name]) not in allowed:
                raise ValueError(
                    f"{name} must be one of {', '.join(allowed)}; "
                    f"got {options[name]!r}"
                )
        for name in ("num_samples", "num_steps", "num_recycles"):
            if name in options:
                try:
                    int(options[name])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{name} must be an integer") from error
        if "diffusion_chunk_size" in options:
            try:
                if int(options["diffusion_chunk_size"]) < 1:
                    raise ValueError(
                        "diffusion_chunk_size must be at least 1; it counts "
                        "diffusion samples denoised at once"
                    )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "diffusion_chunk_size must be a positive integer"
                ) from error
        if "pair_chunk_size" in options:
            try:
                int(options["pair_chunk_size"])
            except (TypeError, ValueError) as error:
                raise ValueError("pair_chunk_size must be an integer") from error
        if "glu_backend" in options:
            backend = options["glu_backend"]
            if backend not in _GLU_BACKENDS:
                raise ValueError(
                    f"glu_backend must be one of {_GLU_BACKENDS}; got {backend!r}"
                )

    def apply_sampling(self, request: PredictionRequest) -> dict[str, Any]:
        """Translate neutral semantics that differ from OpenFold3's literals."""
        options = super().apply_sampling(request)
        # Upstream exposes ``num_recycles`` but executes recycle + 1 trunk
        # cycles. A caller using the native ``num_recycles`` option has already
        # specified the executed count, so only the neutral knob gets +1.
        if request.num_recycles is not None:
            options["num_recycles"] = request.num_recycles + 1
        if options.get("max_msa_depth") is not None:
            options["max_msa_depth"] = min(
                _RELEASED_MSA_DEPTH, int(options["max_msa_depth"])
            )
        return options

    def capabilities(self) -> ModelCapabilities:
        raw = InputRequirement(
            preprocessing_runtime="jax",
            required_extras=("openfold3-preprocess",),
            notes=(
                "Runs FoldJAX's Torch-free OpenFold3 preprocessing pipeline "
                "before JAX prediction; no sibling checkout is required."
            ),
        )
        return ModelCapabilities(
            representations=_representations.available("openfold3"),
            input_representations=("single_inputs",),
            model=self.name,
            sampling=dict(self.sampling_options),
            input_formats=(
                "native",
                "openfold3",
                "openfold3-features",
                "foldjax",
            ),
            input_requirements={
                "native": raw,
                "openfold3": raw,
                "openfold3-features": InputRequirement(
                    preprocessing_runtime="precomputed",
                    notes=(
                        "A self-contained feature .npz with embedded chemistry; "
                        "prediction is JAX-only and needs no preprocessing extra."
                    ),
                ),
                "foldjax": raw,
            },
            padding_axes=self.padding_axes,
        )

    def predict(self, request: PredictionRequest) -> PredictionResult:
        options = self.apply_sampling(request)
        # Out before the leftover-option check: carried by the scope.
        matmul_precision = self.matmul_precision(options)
        # Both are request-level, so the pair is settled before featurization
        # rather than after it: an eager run dispatches operation by operation
        # and owns no executable to carry the reduction policy, and running it
        # anyway would report a deterministic run that was not one.
        compile_it = _compile_enabled(options)
        deterministic = bool(options.pop("deterministic", False))
        if deterministic and not compile_it:
            raise ValueError(
                "deterministic reductions are carried by the compiled graph; "
                "drop no_compile or deterministic"
            )
        # The port is vendored, so these are ordinary in-package imports. They
        # stay inside `predict` only to keep `import foldjax` off JAX's import
        # cost, which is the same reason the other vendored backends do it.
        data = import_module("foldjax.models.openfold3.data")
        inference = import_module("foldjax.models.openfold3.inference")
        output = import_module("foldjax.models.openfold3.output")
        chemistry = import_module("foldjax.models.openfold3.bridge.chemistry")
        checkpoint = import_module("foldjax.models.openfold3.bridge.checkpoint")
        mapping = import_module("foldjax.models.openfold3.bridge.torch_mapping")
        jax = import_module("jax")

        query_id = options.pop("query_id", None)
        ccd_file_path = options.pop("ccd_file_path", None)
        all_arrays = _strict_boolean(
            options.pop("all_arrays", False), name="all_arrays"
        )
        # Native samples inside each recycle.  Retain the compact precursor here;
        # the later cycle planner transfers only the union of selected rows.
        preprocess_msa_depth = None
        # Raw preprocessing otherwise builds the complete
        # [rows, tokens, 32] int32 one-hot before the identical host cut below.
        # Portable feature archives stay full-depth in the loader branch.
        features, table, output_metadata = _features_chemistry_and_metadata(
            request,
            data=data,
            query_id=query_id,
            ccd_file_path=ccd_file_path,
            msa_depth=preprocess_msa_depth,
            compact_empty_template_pairs=True,
            compact_msa=True,
        )
        precompacted_empty_templates = bool(
            getattr(data, "has_compact_zero_template_pair_features", lambda _: False)(
                features
            )
        )
        precompacted_msa = bool(
            getattr(data, "has_compact_msa_features", lambda _: False)(features)
        )
        # Portable archives may carry numeric host-side annotations in
        # addition to the model ABI.  They are useful to archive tooling but
        # must not enter the jitted pytree as undocumented compile axes.
        model_feature_names = getattr(data, "MODEL_FEATURES", None)
        if model_feature_names is not None:
            optional_feature_names = getattr(data, "OPTIONAL_MODEL_FEATURES", ())
            all_private_feature_names = tuple(
                getattr(data, "PRIVATE_MODEL_FEATURES", ())
            )
            compact_msa_feature_names = tuple(
                getattr(data, "COMPACT_MSA_PRIVATE_FEATURES", ())
            )
            private_feature_names: tuple[str, ...] = ()
            if precompacted_empty_templates:
                private_feature_names += tuple(
                    name
                    for name in all_private_feature_names
                    if name not in compact_msa_feature_names
                )
            if precompacted_msa:
                private_feature_names += compact_msa_feature_names
            features = {
                name: features[name]
                for name in (
                    *model_feature_names,
                    *optional_feature_names,
                    *private_feature_names,
                )
                if name in features
            }
        n_token = features["token_mask"].shape[-1]
        n_atom = features["atom_mask"].shape[-1]

        overrides = {
            key: int(options.pop(key))
            for key in ("num_samples", "num_steps", "num_recycles")
            if key in options
        }
        chunk = options.pop("pair_chunk_size", None)
        if chunk is not None:
            overrides["pair_chunk_size"] = int(chunk)
        # Left unset the config resolves it from the sample count, the way
        # Protenix does, so the shipped five-sample run is untouched and only a
        # caller who raises the count pays for the loop.
        sample_chunk = options.pop("diffusion_chunk_size", None)
        if sample_chunk is not None:
            overrides["diffusion_chunk_size"] = int(sample_chunk)
        depth = options.pop("max_msa_depth", None)
        if depth is not None:
            overrides["msa_depth"] = int(depth)
        cp_devices = options.pop("cp_devices", None)
        if cp_devices is not None:
            overrides["cp_shards"] = int(cp_devices)
        cp_layout = options.pop("cp_layout", None)
        if cp_layout is not None:
            overrides["cp_layout"] = str(cp_layout)
        glu_backend = options.pop("glu_backend", None)
        if glu_backend is not None:
            overrides["glu_backend"] = str(glu_backend)
        dtype = str(options.pop("dtype", _DEFAULT_DTYPE))
        overrides["dtype"] = dtype
        # Resolved to `dtype` here rather than left as `None`: this value is
        # also the weight-cache key below, and two spellings of one program
        # must not name two cached parameter trees. `released_config` applies
        # the same rule for direct callers.
        confidence_dtype = str(options.pop("confidence_dtype", dtype))
        overrides["confidence_dtype"] = confidence_dtype
        available = _representations.specs_for("openfold3")
        if request.stop_after == "inputs":
            available = {
                name: available[name]
                for name in self.capabilities().input_representations
            }
            # Validate each selector even when an earlier "all" expands first.
            for selector in request.representations or ():
                for name in selector.split(","):
                    _representations.resolve((name,), available)
        wanted = _representations.resolve(request.representations, available)
        overrides["returned_representations"] = wanted
        overrides["stop_after_inputs"] = request.stop_after == "inputs"
        overrides["stop_after_trunk"] = request.stop_after == "trunk"
        overrides["has_atomized_tokens"] = (
            request.stop_after not in ("trunk", "inputs")
            and data.has_atomized_tokens(features)
        )
        # PredictionResult exposes structures and normalized scores, so native
        # PAE/PDE/distogram bin distributions are opt-in. Decide this before
        # tracing: XLA can then DCE the unused PDE/distogram heads and keep PAE only
        # as long as pTM/ipTM need it. The raw CLI and direct inference API retain
        # their historical DEFAULT_ARRAY_BUDGET_BYTES / all-arrays behaviour.
        retain_pair_arrays = (
            all_arrays and request.stop_after not in ("trunk", "inputs")
        )
        array_budget_bytes = None if retain_pair_arrays else _MANAGED_ARRAY_BUDGET_BYTES
        overrides["max_array_bytes"] = array_budget_bytes
        config = inference.released_config(n_token=n_token, n_atom=n_atom, **overrides)
        features = data.prepare_msa_cycle_features(
            features,
            config.msa_depth,
            num_recycles=config.num_recycles,
            rng=np.random.default_rng(request.seed),
        )
        # A query with no templates is still featurized as the released
        # fixed-width axis of four identical empty ones, which the template
        # stack then embeds four times and averages. Dropping the duplicates
        # here -- before the padding plan reads the template axis, and before
        # anything reaches the device -- keeps the value and a quarter of the
        # work. Real, differing templates are left alone.
        features = data.collapse_identical_templates(features)
        padding_plan = None
        if request.padding is not None:
            padding_plan = _padding_plan(
                features, request.padding, max_msa_depth=config.msa_depth
            )
            features = data.pad_features(
                features,
                n_token=padding_plan.target["tokens"],
                n_atom=padding_plan.target["atoms"],
                # Retain the host union; the streamed scheduler pads each
                # selection after gathering, before device transfer.
                n_msa=None,
                n_templates=padding_plan.target["templates"],
            )
            n_token = padding_plan.target["tokens"]
            n_atom = padding_plan.target["atoms"]
            config = inference.released_config(
                n_token=n_token, n_atom=n_atom,
                **{**overrides, "msa_depth": padding_plan.target["msa"]},
            )
        features, n_chain = data.normalize_asym_ids(features)
        # Empty-template geometry is exact +0.  Compact it only after serving
        # padding has established the final template/token shapes; the helper's
        # private marker gives the resulting mapping its own JIT PyTree identity.
        if not precompacted_empty_templates:
            features = data.compact_zero_template_pair_features(features)
        if not precompacted_msa:
            # Archive inputs retain their portable dense ABI through validation,
            # row selection and serving padding, then cross the same private
            # model boundary as raw jobs.
            features = getattr(data, "compact_msa_features", lambda batch: batch)(
                features
            )
        # The exact structure sidecar validates atom names against the dense
        # public feature ABI, so keep that writer mapping intact. Only the
        # generated, validated, normalized model copy crosses the private
        # compact boundary before JIT/device transfer.
        output_features = features
        features = getattr(
            data,
            "compact_ref_atom_category_storage",
            lambda batch: batch,
        )(features)

        # The complete checkpoint remains visible to inspection and verification.
        # Only the inference path drops upstream's second registration of the
        # denoiser, after the model root is known and before host prestacking can
        # keep both owning NumPy copies live beside the device parameter tree.
        requested_prefix = options.pop("prefix", None)

        def load_params(path: Path) -> Any:
            checkpoint_state = checkpoint.load_checkpoint(path)
            model_prefix = mapping.resolve_model_prefix(
                checkpoint_state, requested_prefix
            )
            mapping.prune_sample_diffusion_aliases(
                checkpoint_state, prefix=model_prefix
            )
            loaded = mapping.map_inference_params(checkpoint_state, model_prefix)
            # Cast inside the loader so the session caches the narrowed tree
            # rather than the float32 one plus a fresh cast per request; both
            # dtypes are part of `prepare_key` for the same reason.
            return inference.cast_narrow_params(
                loaded, *inference.resolve_dtypes(config)
            )

        assert request.weights is not None
        params = self._weights.load(
            Path(request.weights),
            load_params,
            prepare_key=(
                "prefix", requested_prefix,
                "dtype", dtype,
                "confidence_dtype", confidence_dtype,
            ),
        )
        kernel = options.pop("triangle_kernel", None)
        if getattr(config, "cp_shards", 1) > 1 and not compile_it:
            raise ValueError(
                "context parallelism requires the compiled graph; drop "
                "no_compile or cp_devices"
            )
        if options:
            raise ValueError(f"unsupported OpenFold3 options: {', '.join(options)}")

        # This backend was the only one that ignored the request's cache
        # directory, which is the one it could least afford to: compiling the
        # released architecture takes minutes and grows with token count, so
        # without a persistent cache every process pays it again. `api.predict`
        # has already namespaced the directory per model, weight identity and
        # compile-relevant options, and opened that same scope; re-entering it
        # is what carries the one setting this port needs on top -- XLA's
        # default entry-size floor skips exactly the small-but-slow-to-compile
        # graphs it produces -- and restores it afterwards.
        compile_cache = (
            compilation_cache_scope(request.cache_dir, min_entry_size_bytes=-1)
            if compile_it and request.cache_dir is not None
            else nullcontext()
        )

        key = jax.random.key(request.seed)
        noise_mask = None
        if padding_plan is not None:
            # Preserve the ordinary sampler's *stored* row stride, not only its
            # semantic atom count.  A portable archive may already have a
            # masked suffix; compacting that suffix would move sample 2's first
            # draw directly behind sample 1's real atoms instead of behind the
            # full source storage width.
            noise_mask = _sampler_noise_mask(
                padding_plan,
                num_samples=config.num_samples,
            )
        if table is None:
            table = chemistry.representative_atom_table()
        # `_default_backend()` reads this environment variable per call. Keep
        # it set through tracing/execution so it reaches the template stack and
        # confidence head as well as the trunk, then restore the host value.
        with matmul_precision(), _triangle_backend(kernel), compile_cache:
            if padding_plan is not None:
                from foldjax.models.openfold3.streaming import compile_streamed_predict

                streamed = compile_streamed_predict(
                    config, table, n_chain=n_chain, triangle_kernel=kernel,
                    cache_scope=(
                        None if request.cache_dir is None else str(request.cache_dir)
                    ),
                    compiled=compile_it,
                    deterministic=deterministic,
                )
                prediction = streamed(key, features, params, noise_mask=noise_mask)
            elif compile_it:
                compiled = inference.compile_predict(
                    config,
                    table,
                    n_chain=n_chain,
                    triangle_kernel=kernel,
                    cache_scope=(
                        None if request.cache_dir is None else str(request.cache_dir)
                    ),
                    deterministic=deterministic,
                )
                prediction = (
                    compiled(key, features, params)
                    if noise_mask is None
                    else compiled(key, features, params, noise_mask=noise_mask)
                )
            else:
                prediction = (
                    inference.predict(
                        key,
                        features,
                        params,
                        config,
                        table,
                        n_chain=n_chain,
                    )
                    if noise_mask is None
                    else inference.predict(
                        key,
                        features,
                        params,
                        config,
                        table,
                        n_chain=n_chain,
                        noise_mask=noise_mask,
                    )
                )

        name = query_id or Path(request.input).stem
        shape_profile = None
        if padding_plan is not None:
            shape_profile = {
                **padding_plan.summary(),
                "static": {"chains": 1 if n_chain is None else int(n_chain)},
                "msa_execution": "host_streamed_cycles",
                "host_msa_union_rows": int(features["msa_mask"].shape[1]),
            }
        raw = {
            "features": {"n_token": n_token, "n_atom": n_atom},
            "output_metadata": (
                "exact" if output_metadata is not None else "canonical_fallback"
            ),
        }
        if shape_profile is not None:
            raw["padding"] = shape_profile
        # Saved before the structures because a trunk-only run has no
        # structures: the archive is the whole product of that graph.
        _representations.save(
            request.output_dir,
            {
                name: getattr(prediction, name)
                for name in wanted
                if getattr(prediction, name, None) is not None
            },
            _representations.specs_for("openfold3"),
            model="openfold3",
        )
        if request.stop_after in ("trunk", "inputs"):
            # The trunk graph returns before the sampler and the confidence
            # heads, so `prediction` carries no coordinates to write and there
            # are no samples to describe. Reading them raised IndexError.
            return PredictionResult(
                model=self.name,
                samples=(),
                output_dir=request.output_dir,
                raw=raw,
                shape_profile=shape_profile,
                representations=_representations_result(
                    self.name, request.output_dir, wanted
                ),
            )
        written = output.write_prediction_outputs(
            prediction,
            output_features,
            request.output_dir,
            name=name,
            max_array_bytes=array_budget_bytes,
            output_metadata=output_metadata,
        )
        scores = _scores(written["scores"])
        return PredictionResult(
            model=self.name,
            samples=tuple(
                PredictionSample(
                    seed=request.seed,
                    structure_path=path,
                    scores=scores.get(index, {}),
                )
                for index, path in enumerate(written["structures"])
            ),
            output_dir=request.output_dir,
            raw=raw,
            shape_profile=shape_profile,
            representations=_representations_result(
                self.name, request.output_dir, wanted
            ),
        )


def _features_and_chemistry(
    request: PredictionRequest,
    *,
    data: Any,
    query_id: str | None,
    ccd_file_path: str | Path | None,
) -> tuple[dict[str, Any], Any | None]:
    """Compatibility wrapper returning model features and chemistry only."""
    features, table, _metadata = _features_chemistry_and_metadata(
        request,
        data=data,
        query_id=query_id,
        ccd_file_path=ccd_file_path,
    )
    return features, table


def _features_chemistry_and_metadata(
    request: PredictionRequest,
    *,
    data: Any,
    query_id: str | None,
    ccd_file_path: str | Path | None,
    msa_depth: int | None = None,
    compact_empty_template_pairs: bool = False,
    compact_msa: bool = False,
) -> tuple[dict[str, Any], Any | None, Any | None]:
    """Load a JAX-only archive or run FoldJAX's NumPy raw-job preprocessor."""
    path = Path(request.input)
    if path.suffix.lower() == ".npz" or request.input_format == "openfold3-features":
        if ccd_file_path is not None:
            raise ValueError(
                "ccd_file_path applies only to raw OpenFold3 input; feature "
                "archives already contain fixed chemistry"
            )
        archive_loader = getattr(data, "load_feature_archive", None)
        if archive_loader is None:
            features, table = data.load_features(path)
            return features, table, None
        return archive_loader(path)

    spec = json.loads(path.read_text(encoding="utf-8"))
    features, output_metadata = data.featurize_query_with_metadata(
        spec,
        query_id=query_id,
        seed=request.seed,
        ccd_file_path=ccd_file_path,
        msa_depth=msa_depth,
        compact_empty_template_pairs=compact_empty_template_pairs,
        compact_msa=compact_msa,
    )
    return features, None, output_metadata


def _scores(path: Path) -> dict[int, dict[str, float]]:
    """Index the confidence JSON by sample, dropping non-numeric entries."""
    summary: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    indexed: dict[int, dict[str, float]] = {}
    for entry in summary.get("samples", []):
        index = int(entry.get("sample", -1))
        indexed[index] = {
            key: float(value)
            for key, value in entry.items()
            if key != "sample" and isinstance(value, (int, float))
        }
    return indexed
