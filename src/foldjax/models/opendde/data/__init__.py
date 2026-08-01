"""Torch-free OpenDDE input featurization."""

from foldjax.models.opendde.data.featurize_json import featurize_opendde_json, load_jobs

__all__ = ["featurize_opendde_json", "load_jobs"]
