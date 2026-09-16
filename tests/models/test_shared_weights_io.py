"""The native weight reader/writer shared by Protenix and OpenDDE.

One OpenDDE checkpoint pickles both ports' parameter NamedTuples, so the
reader has to admit both ports' parameter modules and cannot belong to either.
These tests pin the two things that made the shared home safe to create: the
port modules still publish the same objects, and the restricted unpickler
admits exactly the two ``...models.`` subpackages -- through real weight files,
not only through ``find_class``.
"""

from __future__ import annotations

import io
import json
import pickle
import subprocess
import sys
from typing import Any, NamedTuple

import numpy as np
import pytest

from foldjax.models import _weights_io as shared
from foldjax.models.opendde.bridge import weights_io as opendde_io
from foldjax.models.opendde.models.structural_tokens import (
    StructuralTokenExpanderParams,
)
from foldjax.models.protenix.bridge import weights_io as protenix_io
from foldjax.models.protenix.models.model import cast_trunk_params
from foldjax.models.protenix.models.primitives.primitives import LinearParams

#: The published surface of ``protenix.bridge.weights_io``. The two private
#: names are part of it: Protenix' runner loads its prepared mixed-precision
#: tree through this module, and ``backends/protenix.py`` reaches the CLI
#: loader that calls it. OpenDDE's bridge imports the same two objects from
#: the shared module instead, which is what this consolidation removed.
_PROTENIX_FACADE_NAMES = (
    "_NativeWeightsUnpickler",
    "_PREPARED_CAST_INPUT_BATCH_BYTES",
    "_load_native_weights_with_field_dtype",
    "load_native_weights",
    "save_native_weights",
)
_OPENDDE_FACADE_NAMES = (
    "_PREPARED_CAST_INPUT_BATCH_BYTES",
    "_load_native_weights_with_field_dtype",
    "load_native_weights",
    "save_native_weights",
)


class ForeignParams(NamedTuple):
    """A parameter-shaped NamedTuple defined outside either port."""

    weight: Any


def _one_array_instance(cls: type) -> Any:
    values: list[Any] = [None] * len(cls._fields)
    values[0] = np.asarray([[1.0, 2.0]], dtype=np.float32)
    return cls._make(values)


@pytest.mark.parametrize("name", _PROTENIX_FACADE_NAMES)
def test_the_protenix_module_publishes_the_shared_object(name: str) -> None:
    assert getattr(protenix_io, name) is getattr(shared, name)
    assert name in protenix_io.__all__


@pytest.mark.parametrize("name", _OPENDDE_FACADE_NAMES)
def test_the_opendde_module_publishes_the_shared_object(name: str) -> None:
    assert getattr(opendde_io, name) is getattr(shared, name)


def test_a_weight_file_carrying_both_ports_classes_round_trips(tmp_path) -> None:
    """The reason the implementation is shared: one file, both ports' classes."""

    path = tmp_path / "mixed.pkl"
    params = {
        "protenix": _one_array_instance(LinearParams),
        "opendde": _one_array_instance(StructuralTokenExpanderParams),
    }
    shared.save_native_weights(path, params, compress=False)

    loaded = shared.load_native_weights(path)

    assert type(loaded["protenix"]) is LinearParams
    assert type(loaded["opendde"]) is StructuralTokenExpanderParams
    assert type(loaded["protenix"]).__module__.startswith(
        "foldjax.models.protenix.models."
    )
    assert type(loaded["opendde"]).__module__.startswith(
        "foldjax.models.opendde.models."
    )


def test_a_weight_file_naming_a_class_outside_both_ports_is_rejected(
    tmp_path,
) -> None:
    """A parameter-shaped class is not admitted just for being tuple-shaped."""

    path = tmp_path / "foreign.pkl"
    with open(path, "wb") as fh:
        pickle.dump(
            {"weights": ForeignParams(np.zeros((1, 2), dtype=np.float32))},
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    with pytest.raises(pickle.UnpicklingError, match="forbidden global") as excinfo:
        shared.load_native_weights(path)
    assert "ForeignParams" in str(excinfo.value)


def test_a_weight_file_naming_a_non_namedtuple_inside_a_port_is_rejected(
    tmp_path,
) -> None:
    """The admitted prefixes gate the module; the shape check gates the name."""

    assert cast_trunk_params.__module__.startswith("foldjax.models.protenix.models.")
    path = tmp_path / "callable.pkl"
    with open(path, "wb") as fh:
        pickle.dump(
            {"weights": cast_trunk_params}, fh, protocol=pickle.HIGHEST_PROTOCOL
        )

    with pytest.raises(pickle.UnpicklingError, match="forbidden global") as excinfo:
        shared.load_native_weights(path)
    assert "cast_trunk_params" in str(excinfo.value)


@pytest.mark.parametrize(
    "module",
    [
        # Inside the ports, outside their parameter subpackages: a weight file
        # may name where parameters are defined and nothing else.
        "foldjax.models.opendde.bridge.weights_io",
        "foldjax.models.protenix.bridge.weights_io",
        "foldjax.models.opendde.data.padding",
        "foldjax.models.protenix.cli.predict",
        # Other ports keep their own readers; this one admits two.
        "foldjax.models.boltz2.models.model",
        "foldjax.models.openfold3.models.model",
    ],
)
def test_only_the_two_ports_parameter_subpackages_are_admitted(module: str) -> None:
    unpickler = shared._NativeWeightsUnpickler(io.BytesIO(b""))
    with pytest.raises(pickle.UnpicklingError, match="forbidden global"):
        unpickler.find_class(module, "load_native_weights")


def test_the_shared_reader_imports_no_torch_and_no_port_model_module() -> None:
    """Reading a checkpoint resolves parameter classes by name, so importing
    the reader must not drag a port -- or torch -- in with it."""

    code = """
import json
import sys

import foldjax.models._weights_io

ports = ("alphafold3", "boltz2", "esmfold2", "opendde", "openfold3", "protenix")
print(json.dumps({
    "torch": "torch" in sys.modules,
    "ports": sorted(
        name
        for name in sys.modules
        if any(name.startswith("foldjax.models." + port) for port in ports)
    ),
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {"torch": False, "ports": []}
