"""Featurization: a query specification in, model features out.

The inference path in this package is torch-free by design, but featurization is
not part of it. Turning a sequence into OpenFold3 features means CCD lookups,
conformer generation, tokenization and template preprocessing -- thousands of
lines of chemistry that upstream already implements and that would be a second
porting project to reproduce.

So this module runs upstream's own ``InferenceDataset`` and converts its output
into what :func:`~foldjax.models.openfold3.inference.predict` consumes. That
keeps featurization exact rather than approximately right, and keeps torch out of
the inference path: features are produced once, and can be written to ``.npz``
and loaded later with no torch present at all.

That pipeline is *vendored*, at :mod:`foldjax.models.openfold3._upstream`, so
this needs no second checkout -- only the ``openfold3-preprocess`` extra, for the
torch and lightning upstream's code imports.

The other ports in this package use the same shape of interface
(``foldjax.models.boltz2.data.featurize_yaml`` returns a numpy feature dict from a
YAML), so this matches the local convention rather than inventing one.
"""

from foldjax.models.openfold3.data.featurize import (
    MODEL_FEATURES,
    MSA_ROW_FEATURES,
    featurize_query,
    pad_features,
    save_features,
    split_chemistry,
    subsample_msa_rows,
)
from foldjax.models.openfold3.data.msa import MAIN_STEM, PAIRED_STEM, attach_msas

__all__ = [
    "MAIN_STEM",
    "MODEL_FEATURES",
    "MSA_ROW_FEATURES",
    "PAIRED_STEM",
    "attach_msas",
    "featurize_query",
    "pad_features",
    "save_features",
    "split_chemistry",
    "subsample_msa_rows",
]
