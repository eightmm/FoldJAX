"""Execution advisories use the same padded public-token axis as planning."""

from types import SimpleNamespace

import pytest

from foldjax import _gpu, api, memory_policy
from foldjax.job import Job
from foldjax.registry import get_backend
from foldjax.schema import PredictionRequest


@pytest.mark.parametrize(
    "model,length,target",
    [("alphafold3", 1400, 1536), ("opendde", 900, 1024), ("esmfold2", 650, 768)],
)
def test_preflight_records_padded_public_tokens_and_actual_card(
    tmp_path, monkeypatch, model, length, target
):
    source = Job.from_sequences(("A" * length,)).write(tmp_path / "job.json")
    request = PredictionRequest(
        model=model, input=source, input_format="foldjax", padding=True
    )
    device = SimpleNamespace(platform="gpu", device_kind="NVIDIA RTX A6000", id=2)
    monkeypatch.setattr(_gpu, "default_device", lambda request: device)
    monkeypatch.setattr(
        memory_policy,
        "device_memory_budget",
        lambda **kwargs: memory_policy.resolve_budget(
            pool_bytes=22 * 2**30, card_bytes=48 * 2**30, override_gib=None
        ),
    )
    with memory_policy.selected_device_scope(device):
        api._preflight_observed_capacity(request, get_backend(model))
        record = memory_policy.recorded()
    assessment = record["observed_capacity"]
    assert assessment["n_token"] == target
    assert assessment["shape_profile"]["estimate"]["tokens"] == length
    assert assessment["card_bytes"] == 48 * 2**30
    assert assessment["budget_bytes"] == 22 * 2**30
    assert assessment["observations"][0]["device_class"] == "A6000 48 GB"
    assert assessment["msa_rows"] is None
    assert record["selected"] is None  # No fitted profile decision was made.
    assert record["selected_device"]["id"] == 2


def test_capacity_preflight_does_not_probe_device_memory_on_cpu(tmp_path, monkeypatch):
    source = tmp_path / "job.json"
    source.write_text("{}")
    request = PredictionRequest(model="esmfold2", input=source)
    monkeypatch.setattr(
        _gpu, "default_device", lambda request: SimpleNamespace(platform="cpu")
    )

    def forbidden(**kwargs):
        raise AssertionError("CPU preflight must not probe GPU memory")

    monkeypatch.setattr(memory_policy, "device_memory_budget", forbidden)
    api._preflight_observed_capacity(request, get_backend("esmfold2"))
    assert memory_policy.recorded() is None
