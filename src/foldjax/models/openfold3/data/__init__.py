"""OpenFold3 query featurization and portable feature archives.

The production path uses FoldJAX-owned NumPy preprocessing for chemistry,
tokenization, MSA pairing, and local templates. It does not import PyTorch or
Lightning. Install the ``openfold3-preprocess`` extra for the chemistry/data
dependencies; the resulting arrays feed the JAX model directly or can be saved
as a pickle-free ``.npz`` archive.
"""

from foldjax.models.openfold3.data.cache import (
    save_preparsed_msas,
    save_template_cache,
)
from foldjax.models.openfold3.data.featurize import (
    MODEL_FEATURES,
    MSA_ROW_FEATURES,
    OPTIONAL_MODEL_FEATURES,
    OUTPUT_PREFIX,
    OutputMetadata,
    collapse_identical_templates,
    featurize_query,
    featurize_query_with_metadata,
    has_atomized_tokens,
    load_feature_archive,
    load_features,
    normalize_asym_ids,
    output_metadata_from_atom_array,
    pad_features,
    save_features,
    split_chemistry,
    split_output_metadata,
    subsample_msa_rows,
    validate_output_metadata,
)
from foldjax.models.openfold3.data.msa import MAIN_STEM, PAIRED_STEM, attach_msas

__all__ = [
    "MAIN_STEM",
    "MODEL_FEATURES",
    "MSA_ROW_FEATURES",
    "OPTIONAL_MODEL_FEATURES",
    "OUTPUT_PREFIX",
    "OutputMetadata",
    "PAIRED_STEM",
    "attach_msas",
    "collapse_identical_templates",
    "featurize_query",
    "featurize_query_with_metadata",
    "has_atomized_tokens",
    "load_feature_archive",
    "load_features",
    "normalize_asym_ids",
    "output_metadata_from_atom_array",
    "pad_features",
    "save_preparsed_msas",
    "save_template_cache",
    "save_features",
    "split_chemistry",
    "split_output_metadata",
    "subsample_msa_rows",
    "validate_output_metadata",
]
