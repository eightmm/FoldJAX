"""Model-independent search steps shared by the vendored ports.

Featurization cannot be shared between these models -- their feature vocabularies
barely intersect -- but the steps *below* it can: an MSA is an alignment whatever
consumes it. One carried port arrived with a byte-identical copy of the same
ColabFold MMseqs2 client, down to the private helper names. That copy was folded
back into this shared module; model-specific naming remains in each consumer.

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
