"""AlphaFold 3, carried in-package.

The source is pinned to one upstream release, with the small FoldJAX runtime
patch set enumerated in :mod:`foldjax.models.alphafold3.provenance`.  It is
imported under its own name through the ``._upstream`` shim, so the pipeline,
network and post-processing run from this repository alone. Model parameters
are the one thing that cannot ship: they are licensed separately by Google and
placed in the FoldJAX weight store by the user.
"""

from foldjax.models.alphafold3.provenance import (
    UPSTREAM_COMMIT,
    UPSTREAM_VERSION,
)

__all__ = ["UPSTREAM_COMMIT", "UPSTREAM_VERSION"]
