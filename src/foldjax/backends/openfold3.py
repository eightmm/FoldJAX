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
from foldjax.backends._representations import _representations_result
from foldjax.backends.base import MATMUL_PRECISION_OPTION, Backend
from foldjax.models import _representations
from foldjax.padding import PaddingPlan, resolve_axis
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
    "max_msa_depth",
    "cp_devices",
    "cp_layout",
    "triangle_kernel",
)

# ``released_config``'s model-side MSA subsampling depth. The public
# ``max_msa_depth`` knob is a cap, so asking for more cannot widen the released
# model's 1024-row input.
_RELEASED_MSA_DEPTH = 1024


def _real_prefix_size(mask: np.ndarray, *, axis: str) -> int:
    """Count one padded axis and reject holes before model compilation."""

    mask = np.asarray(mask, dtype=bool).reshape(-1)
    size = int(np.count_nonzero(mask))
    if not np.array_equal(mask, np.arange(mask.size) < size):
        raise ValueError(f"OpenFold3 {axis} padding must be a contiguous suffix")
    return size


def _padding_plan(
    features: dict[str, Any], config: PaddingConfig
) -> PaddingPlan:
    """Resolve OpenFold3's four independently compiled feature axes."""

    token_mask = np.asarray(features["token_mask"]) > 0
    atom_mask = np.asarray(features["atom_mask"]) > 0
    msa_mask = np.asarray(features["msa_mask"]) > 0
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
        axis: resolve_axis(actual[axis], config, axis, minimum=storage[axis])
        for axis in ("tokens", "atoms", "msa", "templates")
    }
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


class OpenFold3Backend(Backend):
    name = "openfold3"
    padding_axes = ("tokens", "atoms", "msa", "templates")
    native_options = frozenset(
        {
            "ccd_file_path",
            "cp_devices",
            "cp_layout",
            "no_compile",
            "pair_chunk_size",
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
    # as anywhere else. There is no `dtype`: upstream runs `precision="32-true"`,
    # a whole-trunk bfloat16 cast destroys the prediction (pLDDT 0.858 -> 0.466),
    # and the partial profile that does work is one upstream never validated.
    execution_options = {
        **MATMUL_PRECISION_OPTION,
        "triangle_kernel": (
            "triangle_kernel", {"auto": "cueq", "cueq": "cueq", "xla": "xla"}
        ),
    }
    compile_options = _COMPILE_OPTIONS

    def cache_profile(self, request: PredictionRequest) -> dict[str, Any]:
        """Name every static OpenFold3 program choice, including defaults."""

        profile = super().cache_profile(request)
        options = self.apply_sampling(request)
        cp_shards = int(options.get("cp_devices", 1))
        requested_layout = str(options.get("cp_layout", "auto"))
        profile["cp_devices"] = cp_shards
        profile["cp_layout"] = (
            "serial"
            if cp_shards <= 1
            else "1d" if requested_layout == "auto" else requested_layout
        )
        profile["triangle_kernel"] = resolve_triangle_kernel(
            options.get("triangle_kernel"), cp_shards=cp_shards
        )
        profile["representations"] = _representations.resolve(
            request.representations, _representations.specs_for("openfold3")
        )
        profile["stop_after"] = request.stop_after
        profile["rng_route"] = "mask" if request.padding is not None else "native"
        return profile

    def validate_native_options(self, options: dict[str, Any]) -> None:
        _compile_enabled(dict(options))
        for name in ("num_samples", "num_steps", "num_recycles"):
            if name in options:
                try:
                    int(options[name])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{name} must be an integer") from error
        if "pair_chunk_size" in options:
            try:
                int(options["pair_chunk_size"])
            except (TypeError, ValueError) as error:
                raise ValueError("pair_chunk_size must be an integer") from error

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
        # The port is vendored, so these are ordinary in-package imports. They
        # stay inside `predict` only to keep `import foldjax` off JAX's import
        # cost, which is the same reason the other vendored backends do it.
        data = import_module("foldjax.models.openfold3.data")
        inference = import_module("foldjax.models.openfold3.inference")
        output = import_module("foldjax.models.openfold3.output")
        chemistry = import_module("foldjax.models.openfold3.bridge.chemistry")
        checkpoint = import_module("foldjax.models.openfold3.bridge.checkpoint")
        mapping = import_module("foldjax.models.openfold3.bridge.torch_mapping")
        compilation = import_module("foldjax.models.openfold3.compilation")
        jax = import_module("jax")

        query_id = options.pop("query_id", None)
        ccd_file_path = options.pop("ccd_file_path", None)
        requested_msa_depth = options.get("max_msa_depth")
        preprocess_msa_depth = (
            int(requested_msa_depth)
            if requested_msa_depth is not None
            else int(getattr(inference, "RELEASED_MSA_DEPTH", 1024))
        )
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
        )
        precompacted_empty_templates = bool(
            getattr(data, "has_compact_zero_template_pair_features", lambda _: False)(
                features
            )
        )
        # Portable archives may carry numeric host-side annotations in
        # addition to the model ABI.  They are useful to archive tooling but
        # must not enter the jitted pytree as undocumented compile axes.
        model_feature_names = getattr(data, "MODEL_FEATURES", None)
        if model_feature_names is not None:
            optional_feature_names = getattr(data, "OPTIONAL_MODEL_FEATURES", ())
            private_feature_names = (
                getattr(data, "PRIVATE_MODEL_FEATURES", ())
                if precompacted_empty_templates
                else ()
            )
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
        depth = options.pop("max_msa_depth", None)
        if depth is not None:
            overrides["msa_depth"] = int(depth)
        cp_devices = options.pop("cp_devices", None)
        if cp_devices is not None:
            overrides["cp_shards"] = int(cp_devices)
        cp_layout = options.pop("cp_layout", None)
        if cp_layout is not None:
            overrides["cp_layout"] = str(cp_layout)
        wanted = _representations.resolve(
            request.representations, _representations.specs_for("openfold3")
        )
        overrides["returned_representations"] = wanted
        overrides["stop_after_trunk"] = request.stop_after == "trunk"
        overrides["has_atomized_tokens"] = (
            request.stop_after != "trunk" and data.has_atomized_tokens(features)
        )
        config = inference.released_config(n_token=n_token, n_atom=n_atom, **overrides)
        # Upstream subsamples inside `MSAModuleEmbedder.forward`; this port does it
        # on the host, before the alignment reaches the device. Unconditional, so
        # the default path is the released one rather than a full-depth divergence.
        features = data.subsample_msa_rows(features, config.msa_depth)
        # A query with no templates is still featurized as the released
        # fixed-width axis of four identical empty ones, which the template
        # stack then embeds four times and averages. Dropping the duplicates
        # here -- before the padding plan reads the template axis, and before
        # anything reaches the device -- keeps the value and a quarter of the
        # work. Real, differing templates are left alone.
        features = data.collapse_identical_templates(features)
        padding_plan = None
        if request.padding is not None:
            padding_plan = _padding_plan(features, request.padding)
            features = data.pad_features(
                features,
                n_token=padding_plan.target["tokens"],
                n_atom=padding_plan.target["atoms"],
                n_msa=padding_plan.target["msa"],
                n_templates=padding_plan.target["templates"],
            )
            n_token = padding_plan.target["tokens"]
            n_atom = padding_plan.target["atoms"]
            config = inference.released_config(
                n_token=n_token, n_atom=n_atom, **overrides
            )
        features, n_chain = data.normalize_asym_ids(features)
        # Empty-template geometry is exact +0.  Compact it only after serving
        # padding has established the final template/token shapes; the helper's
        # private marker gives the resulting mapping its own JIT PyTree identity.
        if not precompacted_empty_templates:
            features = data.compact_zero_template_pair_features(features)

        # The complete checkpoint remains visible to inspection and verification.
        # Only the inference path drops upstream's second registration of the
        # denoiser, after the model root is known and before host prestacking can
        # keep both owning NumPy copies live beside the device parameter tree.
        checkpoint_state = checkpoint.load_checkpoint(request.weights)
        model_prefix = mapping.resolve_model_prefix(
            checkpoint_state, options.pop("prefix", None)
        )
        mapping.prune_sample_diffusion_aliases(
            checkpoint_state, prefix=model_prefix
        )
        params = mapping.map_inference_params(checkpoint_state, model_prefix)
        kernel = options.pop("triangle_kernel", None)
        compile_it = _compile_enabled(options)
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
        # compile-relevant options.
        #
        # `enable_compilation_cache` rather than a bare `jax.config.update`,
        # because it also lifts the minimum entry size -- XLA otherwise skips
        # exactly the small-but-slow-to-compile graphs this port produces.
        if compile_it and request.cache_dir is not None:
            compilation.enable_compilation_cache(request.cache_dir)

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
        with matmul_precision(), _triangle_backend(kernel):
            if compile_it:
                compiled = inference.compile_predict(
                    config,
                    table,
                    n_chain=n_chain,
                    triangle_kernel=kernel,
                    cache_scope=(
                        None if request.cache_dir is None else str(request.cache_dir)
                    ),
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
        if request.stop_after == "trunk":
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
            features,
            request.output_dir,
            name=name,
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
