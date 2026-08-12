"""AlphaFold 3, carried in-package.

Upstream's source is vendored verbatim at ``._upstream`` (Apache-2.0; see the
LICENSE and NOTICE beside it) and imported under its own name through the shim
there, so the pipeline, network and post-processing run from this repository
alone. Model parameters are the one thing that cannot ship: they are licensed
separately by Google and fetched into the FoldJAX weight store by the user.
"""
