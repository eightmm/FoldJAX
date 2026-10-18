from __future__ import annotations

import dataclasses
import json
import os
from contextlib import contextmanager
from pathlib import Path

import jax
import numpy as np
import pytest

from foldjax import cache
from foldjax import executable_cache_verify as verify
from foldjax.api import resolve_cache_dir, resolve_requests
from foldjax.cli import main
from foldjax.registry import get_backend
from foldjax.schema import PaddingConfig, PredictionRequest


def _tiny_model(x):
    return x * x + np.float32(1)


@pytest.fixture
def resolved_request(tmp_path, monkeypatch):
    source = tmp_path / "job.json"
    source.write_text('{"version":1}')
    weights = tmp_path / "weights.jax"
    weights.write_bytes(b"installed weights")
    monkeypatch.setenv("FOLDJAX_PRIVATE_COMPILE_CACHE", str(tmp_path / "private"))
    request = PredictionRequest(
        model="boltz2",
        input=source,
        input_format="native",
        weights=weights,
        cache_dir=tmp_path / "shared",
        padding=PaddingConfig(tokens=128),
        seed=7,
    )
    return resolve_requests(request, draw_seeds=False)[0]


def _namespaces(request):
    backend = get_backend(request.model)
    return (
        resolve_cache_dir(request, backend),
        resolve_cache_dir(
            dataclasses.replace(request, cache_dir=cache.private_compile_cache_root()),
            backend,
        ),
    )


def _store(namespace, key, *, private=False):
    missing = []
    parent = namespace
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    for directory in reversed(missing):
        directory.mkdir(mode=0o700)
    namespace.chmod(0o700 if private else 0o755)
    entry = namespace / f"{key}-cache"
    entry.write_bytes(b"presence only: verification never opens executable contents")
    entry.chmod(0o600 if private else 0o644)
    return entry


def _forbid_compilation(monkeypatch):
    from jax._src import compilation_cache, compiler, dispatch

    def refuse(*_args, **_kwargs):
        pytest.fail("lowering-only verification compiled, executed or deserialized")

    monkeypatch.setattr(compiler, "compile_or_get_cached", refuse)
    monkeypatch.setattr(compiler, "backend_compile_and_load", refuse)
    monkeypatch.setattr(compilation_cache, "get_executable_and_time", refuse)
    monkeypatch.setattr(dispatch, "apply_primitive", refuse)


def test_key_matches_a_real_jax_persistent_entry(tmp_path, monkeypatch):
    """The expected filename comes from JAX's real CPU compilation, not a mock."""
    cpu = jax.devices("cpu")[0]
    pool = jax.jit(_tiny_model)
    shared = tmp_path / "compile"
    private = tmp_path / "private"
    monkeypatch.setattr(cache, "PERSISTENT_CACHE_MIN_COMPILE_SECS", 0)
    with (
        jax.default_device(cpu),
        cache.compilation_cache_scope(
            shared, private_directory=private, min_entry_size_bytes=0
        ),
    ):
        lowered = pool.lower(jax.ShapeDtypeStruct((13,), np.float32))
        expected = verify.lowered_cache_key(lowered)
        result = pool(np.ones((13,), np.float32))
        np.testing.assert_array_equal(result, np.full((13,), 2, np.float32))
        assert (shared / f"{expected}-cache").is_file()
        _forbid_compilation(monkeypatch)
        assert verify.lowered_cache_key(lowered) == expected
        assert verify._inspect_key(shared, expected, private=False)["status"] == (
            "key_present"
        )
    pool.clear_cache()


def test_key_includes_shape_and_compiler_options(monkeypatch):
    _forbid_compilation(monkeypatch)
    with jax.default_device(jax.devices("cpu")[0]):
        normal = jax.jit(_tiny_model)
        changed = jax.jit(
            _tiny_model, compiler_options={"xla_cpu_enable_fast_math": True}
        )
        small = normal.lower(jax.ShapeDtypeStruct((5,), np.float32))
        large = normal.lower(jax.ShapeDtypeStruct((7,), np.float32))
        different_options = changed.lower(jax.ShapeDtypeStruct((5,), np.float32))
        assert (
            len(
                {verify.lowered_cache_key(x) for x in (small, large, different_options)}
            )
            == 3
        )


@pytest.mark.parametrize("tier", ["shared", "private", "missing"])
def test_verify_matches_exact_key_and_uses_prediction_session(
    resolved_request, monkeypatch, tier
):
    request = resolved_request
    cpu = jax.devices("cpu")[0]
    seen = []
    backend_type = type(get_backend(request.model))

    @contextmanager
    def session(self, requests):
        assert requests == (request,)
        seen.append("session")
        self._verification_open = True
        yield self
        seen.append("closed")

    def lower(self, scalar):
        assert self._verification_open
        assert (
            dataclasses.replace(
                scalar, output_dir=request.output_dir, cache_dir=request.cache_dir
            )
            == request
        )
        assert scalar.cache_dir == _namespaces(request)[0]
        assert scalar.output_dir != request.output_dir
        seen.append("lower")
        return [
            (
                "jit(_tiny_model)",
                jax.jit(_tiny_model).lower(jax.ShapeDtypeStruct((5,), np.float32)),
            )
        ]

    monkeypatch.setattr(backend_type, "session", session)
    monkeypatch.setattr(backend_type, "lower_cache_graphs", lower, raising=False)
    _forbid_compilation(monkeypatch)
    shared, private = _namespaces(request)
    with jax.default_device(cpu):
        key = verify.lowered_cache_key(
            jax.jit(_tiny_model).lower(jax.ShapeDtypeStruct((5,), np.float32))
        )
        different_key = verify.lowered_cache_key(
            jax.jit(_tiny_model).lower(jax.ShapeDtypeStruct((7,), np.float32))
        )
        _store(shared, different_key)
        if tier != "missing":
            _store(
                shared if tier == "shared" else private, key, private=tier == "private"
            )
        snapshot = sorted(shared.iterdir())
        report = verify.verify_prediction_cache(request)
    assert seen == ["session", "lower", "closed"]
    graph = report["results"][0]["graphs"][0]
    assert graph["cache_key"] == key
    assert graph["source"] == (None if tier == "missing" else tier)
    assert report["status"] == ("missing" if tier == "missing" else "key_present")
    assert sorted(shared.iterdir()) == snapshot
    if tier != "private":
        assert not private.exists()


@pytest.mark.parametrize("unsafe", ["symlink", "hardlink", "mode"])
def test_exact_entry_uses_existing_trust_rules(tmp_path, unsafe):
    namespace = tmp_path / "shared"
    key = "jit_example-" + "a" * 64
    entry = _store(namespace, key)
    if unsafe == "symlink":
        entry.unlink()
        external = tmp_path / "external"
        external.write_bytes(b"executable")
        entry.symlink_to(external)
    elif unsafe == "hardlink":
        os.link(entry, tmp_path / "other-link")
    else:
        entry.chmod(0o666)
    assert verify._inspect_key(namespace, key, private=False)["status"] == "untrusted"


def test_pgle_is_reported_unsupported_without_compilation(monkeypatch):
    from jax._src import config

    _forbid_compilation(monkeypatch)
    lowered = jax.jit(_tiny_model).lower(jax.ShapeDtypeStruct((5,), np.float32))
    with (
        config.enable_pgle(True),
        pytest.raises(verify.UnsupportedCacheVerificationError, match="PGLE"),
    ):
        verify.lowered_cache_key(lowered)


def test_missing_lowering_adapter_is_not_a_namespace_hit(resolved_request, monkeypatch):
    from types import SimpleNamespace

    shared, _ = _namespaces(resolved_request)
    _store(shared, "jit_unrelated-" + "a" * 64)
    monkeypatch.setattr(
        verify, "get_backend", lambda _name: SimpleNamespace(name="boltz2")
    )
    report = verify.verify_prediction_cache(resolved_request)
    assert report["status"] == "unsupported"
    assert "no prediction lowering-only adapter" in report["results"][0]["reason"]


@pytest.mark.parametrize("present", [False, True])
def test_cli_verify_returns_status_for_the_actual_key(
    resolved_request, monkeypatch, capsys, present
):
    request = resolved_request
    backend_type = type(get_backend(request.model))

    def lower(_self, _request):
        return [
            (
                "jit(_tiny_model)",
                jax.jit(_tiny_model).lower(jax.ShapeDtypeStruct((5,), np.float32)),
            )
        ]

    monkeypatch.setattr(backend_type, "lower_cache_graphs", lower, raising=False)
    _forbid_compilation(monkeypatch)
    with jax.default_device(jax.devices("cpu")[0]):
        if present:
            shared, _ = _namespaces(request)
            key = verify.lowered_cache_key(lower(None, request)[0][1])
            _store(shared, key)
        status = main(
            [
                "cache",
                "verify",
                "--model",
                request.model,
                "--input",
                str(request.input),
                "--input-format",
                "native",
                "--weights",
                str(request.weights),
                "--cache-dir",
                str(request.cache_dir),
                "--pad-tokens",
                "128",
                "--seed",
                "7",
            ]
        )
    output = capsys.readouterr()
    payload = json.loads(output.out)
    assert status == (0 if present else 3)
    assert payload["strategy"] == "prediction_lowering"
    assert payload["status"] == ("key_present" if present else "missing")
    assert "not deserialized" in payload["limitation"]
    assert "execute_once" not in output.err


def test_verify_refuses_native_search_before_backend_preparation(resolved_request):
    request = dataclasses.replace(
        resolved_request, msa="auto", options={"use_msa_server": True}
    )
    with pytest.raises(ValueError, match="does not run a native MSA server search"):
        verify.verify_prediction_cache(request)


class _CachedMsaBackend:
    name, version = "verification-fixture", "1"

    def search(self, sequence):
        from foldjax.search.msa import MsaPayload

        return MsaPayload(
            paired=f">query\n{sequence}\n>paired\n{sequence}\n",
            unpaired=f">query\n{sequence}\n>unpaired\n{sequence}\n",
        )

    def search_complex(self, sequences):
        from foldjax.search.msa import ComplexPairPayload

        return ComplexPairPayload(
            tuple(
                f">query\n{sequence}\n>paired\n{sequence}\n" for sequence in sequences
            )
        )


def _common_msa_request(request, model, sequences, policy):
    request.input.write_text(
        json.dumps(
            {
                "name": "cached-probe",
                "entities": [
                    {"type": "protein", "id": chr(65 + index), "sequence": sequence}
                    for index, sequence in enumerate(sequences)
                ],
            }
        )
    )
    return dataclasses.replace(request, model=model, input_format="foldjax", msa=policy)


@pytest.mark.parametrize("model", ["boltz2", "esmfold2"])
@pytest.mark.parametrize("policy", ["auto", "required"])
@pytest.mark.parametrize("sequences", [("ACD",), ("ACD", "EFG")])
def test_cached_auto_msa_materializes_the_same_input_and_key_as_prediction(
    resolved_request, monkeypatch, model, policy, sequences
):
    import yaml

    from foldjax import msa_search
    from foldjax.input import materialize_native_input
    from foldjax.search import msa

    request = _common_msa_request(resolved_request, model, sequences, policy)
    backend = get_backend(model)
    provider = _CachedMsaBackend()
    pipeline = msa.MsaSearchPipeline(request.input.parent / "alignments", provider)
    monkeypatch.setattr(msa_search, "_msa_pipeline", lambda: pipeline)
    # Normal prediction materialization both fills the fixture store and gives
    # the independent native-input reference the lowering adapter must receive.
    normal_path = materialize_native_input(
        request.input,
        backend.capabilities(),
        request.input.parent / "prediction-input",
        seed=request.seed,
        msa=request.msa,
        options=backend.apply_sampling(request),
        msa_pairing=request.msa_pairing,
    )

    def signature(path):
        document = yaml.safe_load(path.read_text())
        proteins = (
            [entry["protein"] for entry in document["sequences"]]
            if model == "boltz2"
            else document["entities"]
        )
        msa_key = "msa" if model == "boltz2" else "unpaired_msa"
        return tuple(
            (protein["sequence"], Path(protein[msa_key]).read_text())
            for protein in proteins
        )

    expected = signature(normal_path)
    shape = (
        sum(len(sequence) for sequence, _ in expected),
        sum(len(text) for _, text in expected),
    )

    def lowered():
        return jax.jit(_tiny_model).lower(jax.ShapeDtypeStruct(shape, np.float32))

    def lower(_self, scalar):
        assert scalar.msa == policy
        assert signature(scalar.input) == expected
        return [("primary", lowered())]

    def forbid(*_args, **_kwargs):
        pytest.fail("verification searched or mutated the existing MSA cache")

    monkeypatch.setattr(provider, "search", forbid)
    monkeypatch.setattr(provider, "search_complex", forbid)
    for name in ("cache_key_lock", "quarantine_entry", "staging_directory"):
        monkeypatch.setattr(msa, name, forbid)
    monkeypatch.setattr(type(backend), "lower_cache_graphs", lower)
    snapshot = {
        path: path.read_bytes()
        for path in pipeline.cache_dir.rglob("*")
        if path.is_file()
    }
    _forbid_compilation(monkeypatch)
    with jax.default_device(jax.devices("cpu")[0]):
        key = verify.lowered_cache_key(lowered())
        normal_request = dataclasses.replace(
            request,
            input=normal_path,
            input_format="native"
            if "native" in backend.capabilities().input_formats
            else "foldjax",
        )
        _store(resolve_cache_dir(normal_request, backend), key)
        result = verify.verify_prediction_cache(request)
    assert result["status"] == "key_present"
    assert result["results"][0]["graphs"][0]["cache_key"] == key
    assert snapshot == {
        path: path.read_bytes()
        for path in pipeline.cache_dir.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("policy", ["auto", "required"])
@pytest.mark.parametrize("state", ["missing", "corrupt", "missing_complex"])
def test_auto_msa_cache_miss_refuses_without_fallback_search_or_repair(
    resolved_request, monkeypatch, policy, state
):
    from foldjax import msa_search
    from foldjax.search import msa

    sequences = ("ACD", "EFG") if state == "missing_complex" else ("ACD",)
    request = _common_msa_request(resolved_request, "boltz2", sequences, policy)
    provider = _CachedMsaBackend()
    pipeline = msa.MsaSearchPipeline(request.input.parent / "alignments", provider)
    if state != "missing":
        entries = pipeline.search(sequences)
        if state == "corrupt":
            Path(entries[0]["unpairedMsaPath"]).write_text("corrupt alignment")
    monkeypatch.setattr(msa_search, "_msa_pipeline", lambda: pipeline)

    def forbid(*_args, **_kwargs):
        pytest.fail(
            "cache-missing verification searched, repaired, or lowered a fallback"
        )

    monkeypatch.setattr(provider, "search", forbid)
    monkeypatch.setattr(provider, "search_complex", forbid)
    for name in ("cache_key_lock", "quarantine_entry", "staging_directory"):
        monkeypatch.setattr(msa, name, forbid)
    monkeypatch.setattr(type(get_backend("boltz2")), "lower_cache_graphs", forbid)
    snapshot = {
        path: path.read_bytes()
        for path in pipeline.cache_dir.rglob("*")
        if path.is_file()
    }
    with pytest.raises(ValueError, match="requires existing valid MSA cache entries"):
        verify.verify_prediction_cache(request)
    assert snapshot == {
        path: path.read_bytes()
        for path in pipeline.cache_dir.rglob("*")
        if path.is_file()
    }
    assert pipeline.cache_dir.exists() == (state != "missing")


@pytest.mark.parametrize("model", ["boltz2", "esmfold2"])
@pytest.mark.parametrize("msa_policy", ["single", "none", "auto", "required"])
def test_common_job_uses_prediction_materialization_and_temporary_outputs(
    resolved_request, monkeypatch, model, msa_policy
):
    import yaml

    request = resolved_request
    local_msa = msa_policy != "single"
    entity = {"type": "protein", "id": ["A"], "sequence": "ACD"}
    alignment = request.input.parent / "local.a3m"
    if local_msa:
        alignment.write_text(">query\nACD\n>hit\nACD\n")
        entity["unpaired_msa"] = alignment.name
    request.input.write_text(json.dumps({"name": "probe", "entities": [entity]}))
    output = request.input.parent / "existing-results"
    output.mkdir()
    (output / "keep").write_text("previous prediction")
    request = dataclasses.replace(
        request,
        model=model,
        input_format="foldjax",
        output_dir=output,
        msa=msa_policy,
        options={"pocket_sampling": "off"},
    )
    seen = []

    def lower(_self, scalar):
        assert "pocket_sampling" not in scalar.options
        assert scalar.output_dir != output
        document = yaml.safe_load(scalar.input.read_text())
        if model == "boltz2":
            assert scalar.input_format == "native"
            protein = document["sequences"][0]["protein"]
            assert protein["sequence"] == "ACD"
            assert protein["msa"] == (str(alignment) if local_msa else "empty")
        else:
            assert scalar.input_format == "foldjax"
            protein = document["entities"][0]
            assert protein["sequence"] == "ACD"
            if local_msa:
                assert protein["unpaired_msa"] == str(alignment)
        seen.append(scalar.input)
        return [
            (
                "primary",
                jax.jit(_tiny_model).lower(jax.ShapeDtypeStruct((3,), np.float32)),
            )
        ]

    monkeypatch.setattr(type(get_backend(model)), "lower_cache_graphs", lower)
    monkeypatch.setattr(
        "foldjax.msa_search._msa_pipeline",
        lambda: pytest.fail("supplied alignments must not query the search cache"),
    )
    _forbid_compilation(monkeypatch)
    with jax.default_device(jax.devices("cpu")[0]):
        result = verify.verify_prediction_cache(request)
    assert result["status"] == "missing"
    assert len(seen) == 1 and not seen[0].exists()
    assert list(output.iterdir()) == [output / "keep"]
    assert (output / "keep").read_text() == "previous prediction"


def test_resolved_chunking_is_current_and_never_inherited(
    resolved_request, monkeypatch
):
    from foldjax.execution import record_resolved_chunking

    write_record = [True]

    def record(model):
        record_resolved_chunking(
            model=model, n_token=128, chunks={"chunk_size": 64}, policy="token_rungs"
        )

    def lower(_self, _scalar):
        if write_record[0]:
            record("boltz2")
        return [
            (
                "primary",
                jax.jit(_tiny_model).lower(jax.ShapeDtypeStruct((3,), np.float32)),
            )
        ]

    record("old-model")
    monkeypatch.setattr(type(get_backend("boltz2")), "lower_cache_graphs", lower)
    _forbid_compilation(monkeypatch)
    with jax.default_device(jax.devices("cpu")[0]):
        first = verify.verify_prediction_cache(resolved_request)["results"][0]
        write_record[0] = False
        second = verify.verify_prediction_cache(resolved_request)["results"][0]
    assert first["resolved_chunking"]["profiles"] == [
        {
            "model": "boltz2",
            "n_token": 128,
            "policy": "token_rungs",
            "chunks": {"chunk_size": 64},
        }
    ]
    assert "resolved_chunking" not in second


@pytest.mark.parametrize("mode", ["auto", "required"])
def test_verify_never_searches_public_templates(resolved_request, mode):
    request = dataclasses.replace(resolved_request, templates=mode)
    with pytest.raises(ValueError, match="does not search public templates"):
        verify.verify_prediction_cache(request)
