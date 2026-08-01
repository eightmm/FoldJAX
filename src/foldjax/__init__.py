"""Biomolecular structure prediction in JAX.

One job file and a model name is the whole interface::

    from foldjax import PredictionRequest, predict

    result = predict(PredictionRequest(model="boltz2", input="job.yaml"))
    for sample in result.samples:
        print(sample.structure_path, sample.scores)

Weights, the output directory, the compile cache, and the input dialect are all
resolved for you. See ``foldjax.assets`` for the weight store and
``foldjax.paths`` for where everything lives.
"""

from foldjax import assets, paths
from foldjax.api import detect_input_format, predict, resolve_request
from foldjax.registry import available_models, capabilities, normalize_model_name
from foldjax.schema import (
    ModelCapabilities,
    PredictionRequest,
    PredictionResult,
    PredictionSample,
)

__all__ = [
    "ModelCapabilities",
    "PredictionRequest",
    "PredictionResult",
    "PredictionSample",
    "assets",
    "available_models",
    "capabilities",
    "detect_input_format",
    "normalize_model_name",
    "paths",
    "predict",
    "resolve_request",
]

__version__ = "0.3.0"
