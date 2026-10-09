"""`foldjax capabilities --json` is generated from the input translation table."""

from __future__ import annotations

import json

import pytest

from foldjax import cli
from foldjax.input import (
    _NATIVE_ONLY,
    _OPTION_GATED_FEATURES,
    _PROFILE_GATED_FEATURES,
    _TARGETS,
)
from foldjax.registry import available_models, capabilities


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
    # The two gated tiers are each a subset of common_schema_features, and
    # disjoint from native_only (a native-only field has no common spelling
    # to gate in the first place).
    for key in ("profile_gated_features", "option_gated_features"):
        gated = set(described[key])
        assert gated <= _TARGETS[model].features
        assert not gated & native_only


@pytest.mark.parametrize("model", sorted(_TARGETS))
def test_gated_tiers_match_their_tables(model: str) -> None:
    described = capabilities(model)
    features = _TARGETS[model].features
    assert set(described.profile_gated_features) == (
        _PROFILE_GATED_FEATURES.get(model, frozenset()) & features
    )
    assert set(described.option_gated_features) == (
        _OPTION_GATED_FEATURES.get(model, frozenset()) & features
    )


def test_protenix_pocket_and_contact_are_profile_gated() -> None:
    described = capabilities("protenix")
    assert "pocket_constraints" in described.common_schema_features
    assert "pocket_constraints" in described.profile_gated_features
    assert "contact_constraints" in described.profile_gated_features
    # Not also reported as a released default: gating is the honest half.
    assert "pocket_constraints" not in described.option_gated_features


@pytest.mark.parametrize("model", ["protenix", "opendde"])
def test_protenix_and_opendde_templates_are_option_gated(model: str) -> None:
    described = capabilities(model)
    assert "templates" in described.common_schema_features
    assert "templates" in described.option_gated_features
    assert "templates" not in described.profile_gated_features


def test_boltz2_and_openfold3_templates_are_not_gated() -> None:
    # Boltz-2 aligns a bare mmCIF itself (its feature is `templates_unmapped`,
    # not `templates`); OpenFold3 takes either form unconditionally. Neither
    # has a profile or option gate.
    boltz2 = capabilities("boltz2")
    assert "templates_unmapped" in boltz2.common_schema_features
    assert "templates_unmapped" not in boltz2.profile_gated_features
    assert "templates_unmapped" not in boltz2.option_gated_features
    openfold3 = capabilities("openfold3")
    assert "templates" in openfold3.common_schema_features
    assert "templates" not in openfold3.profile_gated_features
    assert "templates" not in openfold3.option_gated_features


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
