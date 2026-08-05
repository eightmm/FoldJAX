"""Model-independent search steps shared by the vendored ports.

Featurization cannot be shared between these models -- their feature vocabularies
barely intersect -- but the steps *below* it can: an MSA is an alignment whatever
consumes it. Two ports arrived carrying their own copy of the same ColabFold
MMseqs2 client, down to the private helper names, which is the duplication this
module removes: Protenix's copy was byte-identical and was folded back into this
one, and Chai's had diverged into a different implementation and left with the
port itself.

It has no third-party dependency at all, which is why it can sit here rather than
under one model: every port can reach it and none of them pulls anything in by
doing so. It was a separate `foldsearch` package while the ports were separate
repositories; that package was on no index, so the one backend using it -- the
OpenFold3 MSA search -- could only skip without it.
"""

from foldjax.search.msa import (
    HttpResponse,
    LocalMsaClient,
    LocalRnaMsaClient,
    MsaBackend,
    MsaPayload,
    MsaSearchPipeline,
    RemoteMMseqs2Client,
    RnaMsaBackend,
    RnaMsaPayload,
    RnaMsaSearchPipeline,
    SearchError,
)

__version__ = "0.1.0"

__all__ = [
    "HttpResponse",
    "LocalMsaClient",
    "LocalRnaMsaClient",
    "MsaBackend",
    "MsaPayload",
    "MsaSearchPipeline",
    "RemoteMMseqs2Client",
    "RnaMsaBackend",
    "RnaMsaPayload",
    "RnaMsaSearchPipeline",
    "SearchError",
    "__version__",
]
