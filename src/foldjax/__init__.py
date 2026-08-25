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

from foldjax import assets, paths, progress
from foldjax.alignment import (
    AlignmentReport,
    AtomSelection,
    RigidTransform,
    StructureAlignment,
    StructureAlignmentError,
    align_predictions,
    align_structures,
)
from foldjax.api import (
    detect_input_format,
    predict,
    predict_batch,
    resolve_request,
    resolve_requests,
)
from foldjax.job import (
    Bond,
    Dna,
    Job,
    Ligand,
    Modification,
    Protein,
    Rna,
    Template,
)
from foldjax.registry import (
    available_models,
    capabilities,
    model_info,
    normalize_model_name,
)
from foldjax.schema import (
    ERROR_POLICIES,
    MSA_POLICIES,
    BatchReport,
    InputRequirement,
    ModelCapabilities,
    ModelInfo,
    PaddingConfig,
    PredictionError,
    PredictionFailure,
    PredictionOutputError,
    PredictionRequest,
    PredictionResult,
    PredictionSample,
    RuntimeInfo,
)
from foldjax.warmup import CacheWarmResult, warm_cache

__all__ = [
    "AlignmentReport",
    "AtomSelection",
    "ERROR_POLICIES",
    "MSA_POLICIES",
    "BatchReport",
    "Bond",
    "CacheWarmResult",
    "Dna",
    "InputRequirement",
    "Job",
    "Ligand",
    "ModelCapabilities",
    "ModelInfo",
    "Modification",
    "PaddingConfig",
    "PredictionError",
    "PredictionFailure",
    "PredictionOutputError",
    "PredictionRequest",
    "PredictionResult",
    "PredictionSample",
    "Protein",
    "Rna",
    "RigidTransform",
    "RuntimeInfo",
    "StructureAlignment",
    "StructureAlignmentError",
    "Template",
    "assets",
    "align_predictions",
    "align_structures",
    "available_models",
    "capabilities",
    "detect_input_format",
    "model_info",
    "normalize_model_name",
    "paths",
    "predict",
    "predict_batch",
    "progress",
    "resolve_request",
    "resolve_requests",
    "warm_cache",
]

__version__ = "0.3.0"
