"""A weight path that resolves to nothing has to say what it looked for.

Boltz-2 is the one model here whose weight option takes a bundle *stem* rather
than a directory, and the resolver appends a suffix to whatever it is handed.
Passing the directory -- the thing every other model takes -- makes it look for
a sibling file named after that directory, which nothing creates. The failure
is silent about all of this, and a benchmark panel lost three arms to it.
"""

from __future__ import annotations

import pytest

from foldjax.models.boltz2.weights import (
    native_weight_candidates,
    resolve_native_weight_bundle,
    unresolved_bundle_reason,
)


def test_a_directory_of_bundles_still_does_not_resolve(tmp_path) -> None:
    """Refusing is the point; the two bundles there are different models."""
    store = tmp_path / "boltz2"
    store.mkdir()
    (store / "boltz2_conf.safetensors").write_bytes(b"")
    (store / "boltz2_aff.safetensors").write_bytes(b"")

    assert resolve_native_weight_bundle(store) is None


def test_the_reason_names_the_bundles_that_are_there(tmp_path) -> None:
    """So the correction is reading the message, not reading the source."""
    store = tmp_path / "boltz2"
    store.mkdir()
    (store / "boltz2_conf.safetensors").write_bytes(b"")
    (store / "boltz2_aff.npz").write_bytes(b"")

    reason = unresolved_bundle_reason(store)
    assert "is a directory" in reason
    assert "boltz2_conf.safetensors" in reason
    assert "boltz2_aff.npz" in reason


def test_an_empty_directory_says_so_rather_than_listing_nothing(tmp_path) -> None:
    store = tmp_path / "boltz2"
    store.mkdir()
    assert "no bundle files" in unresolved_bundle_reason(store)


def test_a_missing_stem_lists_the_suffixes_that_were_tried(tmp_path) -> None:
    stem = tmp_path / "boltz2_conf"
    reason = unresolved_bundle_reason(stem)
    assert "tried" in reason
    for candidate in native_weight_candidates(stem):
        assert str(candidate) in reason


def test_the_named_bundle_resolves_with_its_sidecar(tmp_path) -> None:
    """The path that works stays working, suffix given or omitted."""
    store = tmp_path / "boltz2"
    store.mkdir()
    payload = store / "boltz2_conf.safetensors"
    payload.write_bytes(b"")

    for request in (payload, payload.with_suffix("")):
        weights, sidecar = resolve_native_weight_bundle(request)
        assert weights == payload
        assert sidecar == store / "boltz2_conf.safetensors.json"


def test_the_loader_raises_the_reason(tmp_path) -> None:
    """The message has to reach the caller, not just exist."""
    from foldjax.models.boltz2.bridge.native import load_params

    store = tmp_path / "boltz2"
    store.mkdir()
    (store / "boltz2_conf.safetensors").write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="is a directory"):
        load_params(store)
