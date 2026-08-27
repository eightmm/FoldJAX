from __future__ import annotations

import gc
import os
import weakref
from types import SimpleNamespace

import pytest

from foldjax.backends._weight_session import PreparedWeightSession
from foldjax.schema import PredictionError


def _requests(count: int):
    return tuple(SimpleNamespace(resolved_seeds=(index,)) for index in range(count))


def test_session_reuses_one_prepared_generation_and_scalar_calls_do_not_cache(
    tmp_path,
) -> None:
    path = tmp_path / "weights.jax"
    path.write_bytes(b"weights")
    loads = []

    def loader(source):
        value = object()
        loads.append((source, value))
        return value

    session = PreparedWeightSession("toy")
    with session.session(_requests(1)):
        assert session.load(path, loader) is not session.load(path, loader)
    assert len(loads) == 2

    loads.clear()
    with session.session(_requests(2)):
        first = session.load(path, loader, prepare_key=("fp32",))
        second = session.load(path, loader, prepare_key=("fp32",))
        assert second is first
        session.invalidate()
        replacement = session.load(path, loader, prepare_key=("fp32",))
        assert replacement is not first
    assert len(loads) == 2


def test_session_exit_releases_the_tree_and_the_next_session_reloads(tmp_path) -> None:
    path = tmp_path / "weights.jax"
    path.write_bytes(b"weights")
    session = PreparedWeightSession("toy")
    refs = []

    class Tree:
        pass

    def loader(_path):
        value = Tree()
        refs.append(weakref.ref(value))
        return value

    with session.session(_requests(2)):
        first = session.load(path, loader)
        assert session.load(path, loader) is first
        del first

    gc.collect()
    assert refs[0]() is None

    with session.session(_requests(2)):
        second = session.load(path, loader)
        assert second is refs[1]()
    assert len(refs) == 2


def test_session_rejects_nesting_and_resets_poison_on_exit(tmp_path) -> None:
    path = tmp_path / "weights.jax"
    session = PreparedWeightSession("toy")

    with session.session(_requests(2)):
        with pytest.raises(RuntimeError, match="nested toy backend sessions"):
            with session.session(_requests(2)):
                pass
        with pytest.raises(PredictionError, match="cannot verify"):
            session.validate(path, resumed=True)

    path.write_bytes(b"weights")
    with session.session(_requests(2)):
        assert session.load(path, lambda source: source.read_bytes()) == b"weights"


def test_session_drops_an_incompatible_tree_before_loading_its_replacement(
    tmp_path,
) -> None:
    path = tmp_path / "weights.jax"
    path.write_bytes(b"weights")
    session = PreparedWeightSession("toy")
    first_ref = None

    class Tree:
        pass

    def first_loader(_path):
        nonlocal first_ref
        value = Tree()
        first_ref = weakref.ref(value)
        return value

    def replacement_loader(_path):
        gc.collect()
        assert first_ref is not None and first_ref() is None
        return Tree()

    with session.session(_requests(2)):
        session.load(path, first_loader, prepare_key=("first",))
        session.load(path, replacement_loader, prepare_key=("second",))


def test_session_refuses_a_retargeted_weight_symlink(tmp_path) -> None:
    first = tmp_path / "first.jax"
    second = tmp_path / "second.jax"
    link = tmp_path / "weights.jax"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    link.symlink_to(first)
    session = PreparedWeightSession("toy")

    with session.session(_requests(2)):
        session.load(link, lambda path: path.read_bytes())
        link.unlink()
        link.symlink_to(second)
        with pytest.raises(PredictionError, match="weights changed"):
            session.validate(link)
        session.invalidate()
        with pytest.raises(PredictionError, match="weights changed"):
            session.load(link, lambda path: path.read_bytes())


def test_resumed_run_requires_a_verifiable_weight_path(tmp_path) -> None:
    missing = tmp_path / "missing.jax"
    session = PreparedWeightSession("toy")

    with session.session(_requests(2)):
        with pytest.raises(PredictionError, match="cannot verify"):
            session.validate(missing, resumed=True)


def test_in_place_replacement_is_detected_even_when_size_is_unchanged(tmp_path) -> None:
    path = tmp_path / "weights.jax"
    path.write_bytes(b"first")
    session = PreparedWeightSession("toy")

    with session.session(_requests(2)):
        session.load(path, lambda source: source.read_bytes())
        original = path.stat()
        path.write_bytes(b"other")
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns + 1))
        with pytest.raises(PredictionError, match="weights changed"):
            session.validate(path)


def test_session_refuses_a_weight_changed_by_the_loader(tmp_path) -> None:
    path = tmp_path / "weights.jax"
    path.write_bytes(b"first")
    session = PreparedWeightSession("toy")

    def changing_loader(source):
        source.write_bytes(b"replacement")
        return object()

    with session.session(_requests(2)):
        with pytest.raises(PredictionError, match="changed while they were loading"):
            session.load(path, changing_loader)
