"""Torch-free OpenDDE input featurization."""

from foldjax.models.opendde.data.compact_categories import (
    compact_ref_atom_category_storage,
)
from foldjax.models.opendde.data.featurize_json import featurize_opendde_json, load_jobs

__all__ = [
    "compact_ref_atom_category_storage",
    "featurize_opendde_json",
    "load_jobs",
]
