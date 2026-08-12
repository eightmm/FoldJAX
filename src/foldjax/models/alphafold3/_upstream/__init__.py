"""Import the vendored AlphaFold 3 package under its own name.

Upstream's modules import each other absolutely (``from alphafold3.model import
...``), and its C++ extensions are compiled with ``alphafold3.*`` module names,
so rewriting the tree to a FoldJAX-prefixed spelling -- the OpenFold3 vendoring
approach -- would strand every compiled module. Registering the vendored root
as ``alphafold3`` keeps both the Python and the pybind halves verbatim.

An installed ``alphafold3`` distribution wins if one exists: the shim only
fills the name in when nothing else provides it, so an environment that carries
the real package keeps exactly the behaviour it had.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys


def ensure_registered() -> None:
    """Make ``import alphafold3`` resolve, preferring an installed copy."""
    if "alphafold3" in sys.modules:
        return
    if importlib.util.find_spec("alphafold3") is not None:
        return
    import os
    from pathlib import Path

    data = Path(__file__).parent / "share" / "libcifpp"
    if data.is_dir():
        # The compiled half statically links libcifpp and resolves its
        # component dictionaries through this variable when nothing is
        # installed system-wide.
        os.environ.setdefault("LIBCIFPP_DATA_DIR", str(data))
    module = importlib.import_module(f"{__name__}.alphafold3")
    sys.modules["alphafold3"] = module
