from __future__ import annotations

import gc
import weakref

from foldjax.models.esmfold2.data import ccd


def test_ccd_cache_reuses_one_identity_and_releases_the_previous_generation(
    tmp_path, monkeypatch
) -> None:
    first_path = tmp_path / "first.pkl"
    second_path = tmp_path / "second.pkl"
    first_path.write_bytes(b"first")
    second_path.write_bytes(b"second")
    constructed = []

    class FakeStore:
        def __init__(self, path):
            self.path = path
            constructed.append(path)

    ccd._cached_store.cache_clear()
    monkeypatch.setattr(ccd, "CCDStore", FakeStore)
    try:
        first = ccd.get_ccd_store(first_path)
        assert ccd.get_ccd_store(first_path) is first
        first_ref = weakref.ref(first)

        second = ccd.get_ccd_store(second_path)
        del first
        gc.collect()

        assert first_ref() is None
        assert ccd.get_ccd_store(second_path) is second
        assert constructed == [str(first_path.resolve()), str(second_path.resolve())]
        assert ccd._cached_store.cache_info().currsize == 1
    finally:
        ccd._cached_store.cache_clear()


def test_release_helper_reports_only_a_loaded_cached_dictionary(tmp_path) -> None:
    path = tmp_path / "ccd.pkl"
    replacement_path = tmp_path / "replacement.pkl"
    path.write_bytes(b"identity")
    replacement_path.write_bytes(b"replacement")
    ccd._cached_store.cache_clear()
    try:
        unloaded = ccd.get_ccd_store(path)
        assert ccd._release_ccd_cache() is False
        del unloaded
        gc.collect()

        loaded = ccd.get_ccd_store(path)
        loaded._molecules = {}  # noqa: SLF001 - loaded-state contract
        assert ccd._release_ccd_cache() is True
        assert ccd._cached_store.cache_info().currsize == 0

        evicted_but_held = ccd.get_ccd_store(path)
        evicted_but_held._molecules = {}  # noqa: SLF001
        current_unloaded = ccd.get_ccd_store(replacement_path)
        assert ccd._release_ccd_cache() is False
        assert evicted_but_held._molecules == {}  # noqa: SLF001
        assert current_unloaded._molecules is None  # noqa: SLF001
    finally:
        ccd._cached_store.cache_clear()
