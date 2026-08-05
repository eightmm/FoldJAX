"""Real PDB entries for the test suite, and measurements that mean one thing.

Two things live here and nothing else: a curated set of real PDB entries with a
neutral description of each, and measurement helpers that mean the same thing for
torch and for JAX. Everything model-specific -- featurization, inference, output --
stays in the port it belongs to, because the models do not share those.

Synthetic inputs are excluded on purpose. A hand-built batch is uniform -- one
chain, one molecule type, no gaps -- and uniformity hides tokenization, masking
and pairing bugs; it also cannot show how cost scales, because the shapes are
whatever the author picked.

It is under ``tests/`` rather than ``src/`` because **entries are fetched from
RCSB over the network**, which is not something the installed package should be
able to do on import, and because the only callers are the OpenFold3 tests. It
ships in no wheel. This was a separate ``foldbench`` package while the ports
were separate repositories; that package was on no index, so the tests using it
could only skip.

The upstream CLI (``foldbench-targets``, which wrote OpenFold3 spec files) was
not carried over: nothing here calls it, and a console script under ``tests/``
would have nowhere to be installed from.
"""

from tests._foldbench.measure import Measurement, measure_jax, measure_torch
from tests._foldbench.rcsb import (
    Polymer,
    Target,
    UnsupportedEntityError,
    load_target,
    parse_target,
)
from tests._foldbench.specs import openfold3_query
from tests._foldbench.targets import TARGETS, TargetSpec, by_size

__version__ = "0.1.0"

__all__ = [
    "TARGETS",
    "Measurement",
    "Polymer",
    "Target",
    "TargetSpec",
    "UnsupportedEntityError",
    "by_size",
    "load_target",
    "measure_jax",
    "measure_torch",
    "openfold3_query",
    "parse_target",
    "__version__",
]
