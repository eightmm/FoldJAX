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
from foldjax.models.openfold3.data.compact_categories import (
    compact_ref_atom_category_storage,
)
from foldjax.models.openfold3.data.featurize import (
    COMPACT_MSA_PRIVATE_FEATURES,
    MODEL_FEATURES,
    MSA_ROW_FEATURES,
    OPTIONAL_MODEL_FEATURES,
    OUTPUT_PREFIX,
    PRIVATE_MODEL_FEATURES,
    OutputMetadata,
    collapse_identical_templates,
    compact_msa_features,
    compact_zero_template_pair_features,
    featurize_query,
    featurize_query_with_metadata,
    has_atomized_tokens,
    has_compact_msa_features,
    has_compact_zero_template_pair_features,
    load_feature_archive,
    load_features,
    normalize_asym_ids,
    output_metadata_from_atom_array,
    pad_features,
    prepare_msa_cycle_features,
    save_features,
    split_chemistry,
    split_output_metadata,
    subsample_msa_rows,
    validate_output_metadata,
)
from foldjax.models.openfold3.data.msa import MAIN_STEM, PAIRED_STEM, attach_msas

__all__ = [
    "COMPACT_MSA_PRIVATE_FEATURES",
    "MAIN_STEM",
    "MODEL_FEATURES",
    "MSA_ROW_FEATURES",
    "OPTIONAL_MODEL_FEATURES",
    "OUTPUT_PREFIX",
    "OutputMetadata",
    "PAIRED_STEM",
    "PRIVATE_MODEL_FEATURES",
    "attach_msas",
    "collapse_identical_templates",
    "compact_msa_features",
    "compact_ref_atom_category_storage",
    "compact_zero_template_pair_features",
    "featurize_query",
    "featurize_query_with_metadata",
    "has_compact_zero_template_pair_features",
    "has_compact_msa_features",
    "has_atomized_tokens",
    "load_feature_archive",
    "load_features",
    "normalize_asym_ids",
    "output_metadata_from_atom_array",
    "pad_features",
    "prepare_msa_cycle_features",
    "save_preparsed_msas",
    "save_template_cache",
    "save_features",
    "split_chemistry",
    "split_output_metadata",
    "subsample_msa_rows",
    "validate_output_metadata",
]
