"""Graph changes make stale warm namespaces visible without flushing other models."""

from __future__ import annotations

import hashlib
import json

import pytest

from foldjax import cache


@pytest.mark.parametrize(
    "model,requires_warm",
    [
        ("protenix", True),
        ("opendde", True),
        ("openfold3", True),
        ("alphafold3", False),
        ("boltz2", False),
        ("esmfold2", False),
    ],
)
def test_graph_revision_only_invalidates_the_changed_models(
    tmp_path, model, requires_warm
):
    profile = {"runtime": "cpu", "options": {"tokens": 128, "dtype": "bf16"}}
    original = json.dumps(profile, sort_keys=True, separators=(",", ":"))
    previous_digest = hashlib.sha256(original.encode()).hexdigest()[:16]
    current = cache.cache_namespace(
        tmp_path, model=model, weight_id="weights", profile=profile
    )
    assert (current.name != previous_digest) is requires_warm
    assert (cache.cache_revision(model) is not None) is requires_warm
    assert json.dumps(profile, sort_keys=True, separators=(",", ":")) == original


def test_bumping_one_graph_revision_leaves_other_namespaces_unchanged(
    tmp_path, monkeypatch
):
    def namespace(model):
        return cache.cache_namespace(
            tmp_path, model=model, weight_id="weights", profile={"tokens": 128}
        )

    previous = {
        model: namespace(model) for model in ("protenix", "openfold3", "boltz2")
    }
    monkeypatch.setitem(cache._GRAPH_REVISIONS, "protenix", "next-graph-abi")
    assert namespace("protenix") != previous["protenix"]
    assert namespace("openfold3") == previous["openfold3"]
    assert namespace("boltz2") == previous["boltz2"]
