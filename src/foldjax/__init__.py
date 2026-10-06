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
    residue_correspondence,
)
from foldjax.api import (
    detect_input_format,
    predict,
    predict_batch,
    resolve_request,
    resolve_requests,
)
from foldjax.compare import compare_directory
from foldjax.confidence_arrays import ConfidenceArrays, load_confidence_arrays
from foldjax.job import (
    Bond,
    Contact,
    Dna,
    Job,
    Ligand,
    Modification,
    Pocket,
    Protein,
    Rna,
    Template,
)
from foldjax.model import ExecutionConfig, Model, ModelConfig, get_model
from foldjax.registry import (
    available_models,
    capabilities,
    model_info,
    normalize_model_name,
)
from foldjax.results import (
    FailureRecord,
    ResultsReport,
    RunRecord,
    SampleRecord,
    aggregate_table,
    load_results,
    results_table,
)
from foldjax.schema import (
    ERROR_POLICIES,
    MSA_POLICIES,
    TEMPLATE_POLICIES,
    BatchReport,
    InputRequirement,
    JobSource,
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
    "TEMPLATE_POLICIES",
    "BatchReport",
    "Bond",
    "CacheWarmResult",
    "Contact",
    "ConfidenceArrays",
    "Dna",
    "FailureRecord",
    "InputRequirement",
    "Job",
    "JobSource",
    "ExecutionConfig",
    "Model",
    "ModelConfig",
    "get_model",
    "Ligand",
    "ModelCapabilities",
    "ModelInfo",
    "Modification",
    "PaddingConfig",
    "Pocket",
    "PredictionError",
    "PredictionFailure",
    "PredictionOutputError",
    "PredictionRequest",
    "PredictionResult",
    "PredictionSample",
    "Protein",
    "Rna",
    "ResultsReport",
    "RigidTransform",
    "RunRecord",
    "RuntimeInfo",
    "SampleRecord",
    "StructureAlignment",
    "StructureAlignmentError",
    "Template",
    "assets",
    "align_predictions",
    "align_structures",
    "aggregate_table",
    "available_models",
    "capabilities",
    "compare_directory",
    "load_confidence_arrays",
    "detect_input_format",
    "load_results",
    "model_info",
    "normalize_model_name",
    "paths",
    "predict",
    "predict_batch",
    "progress",
    "residue_correspondence",
    "resolve_request",
    "resolve_requests",
    "results_table",
    "warm_cache",
]

__version__ = "0.1.1.dev0"
