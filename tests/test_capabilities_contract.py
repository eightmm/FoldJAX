"""`foldjax capabilities --json` is generated from the input translation table."""

from __future__ import annotations

import json

import pytest

from foldjax import cli
from foldjax.input import _NATIVE_ONLY, _TARGETS
from foldjax.registry import available_models


def test_every_model_has_a_native_only_entry() -> None:
    assert set(_NATIVE_ONLY) == set(_TARGETS) == set(available_models())


@pytest.mark.parametrize("model", sorted(_TARGETS))
def test_capabilities_json_reflects_the_targets(model: str, capsys) -> None:
    assert cli.main(["capabilities", "--model", model, "--json"]) == 0
    described = json.loads(capsys.readouterr().out)

    assert described["common_schema_features"] == sorted(_TARGETS[model].features)
    native_only = set(described["native_only_features"])
    assert _NATIVE_ONLY[model] - _TARGETS[model].features <= native_only
    # Nothing the common schema can carry is reported as native-only.
    assert not native_only & _TARGETS[model].features


def test_native_only_features_name_constraints_where_the_port_consumes_them(
    capsys,
) -> None:
    reported = {}
    common = {}
    for model in ("boltz2", "protenix", "openfold3", "esmfold2"):
        cli.main(["capabilities", "--model", model, "--json"])
        described = json.loads(capsys.readouterr().out)
        reported[model] = set(described["native_only_features"])
        common[model] = set(described["common_schema_features"])

    # A pocket is the common `constraints` field for the three that read one,
    # and a contact for Boltz-2 and Protenix; OpenFold3 applies a pocket as
    # pocket-guided sampling and has no contact constraint.
    for model in ("boltz2", "protenix", "openfold3"):
        assert "pocket_constraints" in common[model]
        assert "pocket_constraints" not in reported[model]
    for model in ("boltz2", "protenix"):
        assert "contact_constraints" in common[model]
        assert "contact_constraints" not in reported[model]
    assert "contact_constraints" not in reported["openfold3"]
    # ESMFold2 reads the common document itself: nothing is native-only.
    assert reported["esmfold2"] == set()
