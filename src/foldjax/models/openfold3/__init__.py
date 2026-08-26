"""Standalone JAX inference port for OpenFold3."""

# Take the pool up front. Growing the allocator on demand was the historical
# setting here, justified as stopping a large token bucket from failing before
# it had a chance to run; measured at 3,012 tokens with a cold compilation
# cache, it is what *caused* that failure. A grown pool is assembled piecewise
# and cannot serve one large contiguous request even with the card mostly
# free, so the run dies asking for a block that would have fit. Only the
# fraction is set, and only as a default: an explicit process-level setting
# still wins.
from foldjax import oom as _oom  # noqa: E402

_oom.set_mem_fraction(0.90)

# The matmul precision OpenFold3 needs is applied by `openfold3_precision`
# around `predict`, not here. `jax.config.update` is process-global, so setting
# it on import re-specified the numerics of every other model sharing the
# process -- and of every non-OpenFold3 test collected after this module.
from foldjax.models.openfold3.bridge.checkpoint import (  # noqa: E402
    describe,
    detect_fused_tri_mul,
    load_checkpoint,
)
from foldjax.models.openfold3.compilation import (  # noqa: E402
    default_cache_dir,
    enable_compilation_cache,
)
from foldjax.models.openfold3.inference import (  # noqa: E402
    InferenceConfig,
    InferenceParams,
    Prediction,
    compile_predict,
    openfold3_precision,
    predict,
)
from foldjax.models.openfold3.output import (  # noqa: E402
    atom_metadata,
    confidence_summary,
    write_prediction_outputs,
)

__version__ = "0.1.0"

__all__ = [
    "InferenceConfig",
    "InferenceParams",
    "Prediction",
    "atom_metadata",
    "compile_predict",
    "confidence_summary",
    "default_cache_dir",
    "describe",
    "detect_fused_tri_mul",
    "enable_compilation_cache",
    "load_checkpoint",
    "openfold3_precision",
    "predict",
    "write_prediction_outputs",
]
