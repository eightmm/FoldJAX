"""A finished manifest is reusable only when it proves this exact request."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

import foldjax
from foldjax.backends.base import Backend
from foldjax.manifest import MANIFEST_NAME
from foldjax.registry import backend_override
from foldjax.schema import (
    ModelCapabilities,
    PaddingConfig,
    PredictionRequest,
    PredictionResult,
    PredictionSample,
    Representations,
)


class _ResumeBackend(Backend):
    padding_axes = ("tokens",)
    sampling_options = {
        "num_samples": "samples",
        "num_steps": "steps",
        "num_recycles": "recycles",
        "max_msa_depth": "msa_depth",
    }

    def __init__(
        self,
        name: str,
        calls: list[tuple[str, str, int]],
        *,
        fail_stem: str | None = None,
    ) -> None:
        self.name = name
        self.calls = calls
        self.fail_stem = fail_stem

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model=self.name,
            input_formats=("foldjax", "native", "alternate-native"),
            padding_axes=self.padding_axes,
            representations=("single", "pair"),
        )

    def _representations(self, request: PredictionRequest) -> Representations | None:
        if not request.representations:
            return None
        requested = [
            name
            for entry in request.representations
            for name in entry.split(",")
            if name
        ]
        names = ["single", "pair"] if "all" in requested else requested
        arrays = {
            name: (
                np.arange(6, dtype=np.float32).reshape(2, 3)
                if name == "single"
                else np.arange(12, dtype=np.float32).reshape(2, 2, 3)
            )
            for name in names
        }
        archive = Path(request.output_dir) / "representations.npz"
        np.savez(archive, **arrays)
        entries = {
            name: {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "axes": ["token"] * (array.ndim - 1) + ["channel"],
                "space": "token",
                "description": f"mock {name}",
            }
            for name, array in arrays.items()
        }
        return Representations(model=self.name, path=archive, manifest=entries)

    def predict(self, request: PredictionRequest) -> PredictionResult:
        stem = Path(request.input).stem
        self.calls.append((self.name, stem, request.seed))
        if stem == self.fail_stem:
            raise MemoryError("planned failure")

        representations = self._representations(request)
        shape_profile = (
            {"tokens": request.padding.tokens or 16}
            if request.padding is not None
            else None
        )
        if request.stop_after == "trunk":
            return PredictionResult(
                model=self.name,
                output_dir=request.output_dir,
                representations=representations,
                shape_profile=shape_profile,
            )

        structure = Path(request.output_dir) / f"native-{request.seed}.cif"
        structure.write_text("data_mock\n_entry.id mock\n#\n", encoding="utf-8")
        return PredictionResult(
            model=self.name,
            samples=(
                PredictionSample(
                    seed=request.seed,
                    structure_path=structure,
                    scores={"confidence": 0.75},
                ),
            ),
            output_dir=request.output_dir,
            representations=representations,
            shape_profile=shape_profile,
        )


def _file(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def _request(tmp_path: Path, **updates) -> PredictionRequest:
    _file(tmp_path / "alignment.a3m", b">query\nAAAA\n")
    job = tmp_path / "job.json"
    job.write_text(
        json.dumps(
            {
                "name": "job",
                "entities": [
                    {
                        "type": "protein",
                        "id": "A",
                        "sequence": "AAAA",
                        "unpaired_msa": "alignment.a3m",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    values = {
        "model": "boltz2",
        "input": job,
        "input_format": "foldjax",
        "weights": _file(tmp_path / "weights.bin", b"weights-a\n"),
        "output_dir": tmp_path / "out",
        "profile": None,
        "seed": 7,
        "num_steps": 3,
        "msa": "none",
        "options": {"mode": "baseline"},
        "padding": PaddingConfig(tokens=8),
        "representations": ("single",),
        "use_compile_cache": False,
    }
    values.update(updates)
    return PredictionRequest(**values)


@contextmanager
def _backends(
    calls: list[tuple[str, str, int]], *, fail_stem: str | None = None
) -> Iterator[None]:
    boltz = _ResumeBackend("boltz2", calls, fail_stem=fail_stem)
    opendde = _ResumeBackend("opendde", calls, fail_stem=fail_stem)
    esmfold = _ResumeBackend("esmfold2", calls, fail_stem=fail_stem)
    protenix = _ResumeBackend("protenix", calls, fail_stem=fail_stem)
    with (
        backend_override("boltz2", lambda: boltz),
        backend_override("opendde", lambda: opendde),
        backend_override("esmfold2", lambda: esmfold),
        backend_override("protenix", lambda: protenix),
    ):
        yield


def test_exact_request_reuses_nonempty_recorded_artifacts(tmp_path: Path) -> None:
    calls: list[tuple[str, str, int]] = []
    request = _request(tmp_path)
    with _backends(calls):
        first = foldjax.predict_batch(request)
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 1
    assert first.skipped == ()
    assert resumed.skipped == (request.output_dir,)
    assert resumed.results[0].samples[0].structure_path.is_file()
    assert resumed.results[0].representations is not None
    np.testing.assert_array_equal(
        resumed.results[0].representations["single"],
        np.arange(6, dtype=np.float32).reshape(2, 3),
    )


@pytest.mark.parametrize(
    "identity",
    [
        "input_format",
        "model",
        "weights",
        "profile",
        "seed",
        "msa",
        "sampling",
        "options",
        "padding",
        "representations",
        "stop_after",
    ],
)
def test_changed_request_identity_forces_a_rerun(
    tmp_path: Path, identity: str
) -> None:
    calls: list[tuple[str, str, int]] = []
    request = _request(tmp_path)
    changes = {
        "input_format": {"input_format": "alternate-native"},
        "model": {"model": "opendde"},
        "weights": {
            "weights": _file(tmp_path / "other-weights.bin", b"weights-b\n")
        },
        "profile": {"profile": "released"},
        "seed": {"seed": 8},
        "msa": {"msa": "required"},
        "sampling": {"num_steps": 4},
        "options": {"options": {"mode": "changed"}},
        "padding": {"padding": PaddingConfig(tokens=16)},
        "representations": {"representations": ("pair",)},
        "stop_after": {"stop_after": "trunk"},
    }[identity]

    with _backends(calls):
        foldjax.predict(request)
        resumed = foldjax.predict_batch(
            dataclasses.replace(request, resume=True, **changes)
        )

    assert len(calls) == 2
    assert resumed.skipped == ()


def test_editing_the_input_in_place_forces_a_rerun(tmp_path: Path) -> None:
    calls: list[tuple[str, str, int]] = []
    request = _request(tmp_path)
    with _backends(calls):
        foldjax.predict(request)
        document = json.loads(request.input.read_text(encoding="utf-8"))
        document["entities"][0]["sequence"] = "AAAT"
        request.input.write_text(json.dumps(document), encoding="utf-8")
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 2
    assert resumed.skipped == ()


@pytest.mark.parametrize(
    "missing",
    [
        "schema",
        "artifact_paths",
        "model",
        "input.path",
        "input.resolved_path",
        "input.format",
        "input.sha256",
        "input_dependencies",
        "weights.identity",
        "weights.profile",
        "weights.sha256",
        "seeds",
        "msa",
        "sampling",
        "options",
        "options_verifiable",
        "padding",
        "requested_representations",
        "stop_after",
        "representations",
        "samples",
    ],
)
def test_legacy_or_incomplete_manifest_is_not_reused(
    tmp_path: Path, missing: str
) -> None:
    calls: list[tuple[str, str, int]] = []
    request = _request(tmp_path)
    with _backends(calls):
        foldjax.predict(request)
        path = request.output_dir / MANIFEST_NAME
        document = json.loads(path.read_text(encoding="utf-8"))
        parent = document
        parts = missing.split(".")
        for part in parts[:-1]:
            parent = parent[part]
        parent.pop(parts[-1])
        path.write_text(json.dumps(document), encoding="utf-8")
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 2
    assert resumed.skipped == ()


@pytest.mark.parametrize("damage", ["missing", "empty"])
def test_missing_or_empty_structure_is_not_reused(
    tmp_path: Path, damage: str
) -> None:
    calls: list[tuple[str, str, int]] = []
    request = _request(tmp_path)
    with _backends(calls):
        result = foldjax.predict(request)
        structure = result.samples[0].structure_path
        if damage == "missing":
            structure.unlink()
        else:
            structure.write_bytes(b"")
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 2
    assert resumed.skipped == ()
    assert resumed.results[0].samples[0].structure_path.stat().st_size > 0


def test_trunk_archive_is_recorded_and_restored_on_resume(tmp_path: Path) -> None:
    calls: list[tuple[str, str, int]] = []
    request = _request(tmp_path, stop_after="trunk")
    with _backends(calls):
        first = foldjax.predict(request)
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    document = json.loads((request.output_dir / MANIFEST_NAME).read_text())
    assert document["schema"] == 1
    assert document["stop_after"] == "trunk"
    assert document["requested_representations"] == ["single"]
    assert document["representations"]["path"] == "representations.npz"
    assert document["representations"]["names"] == ["single"]
    assert document["samples"] == []
    assert len(calls) == 1
    assert resumed.skipped == (request.output_dir,)
    restored = resumed.results[0]
    assert restored.samples == ()
    assert restored.representations is not None
    np.testing.assert_array_equal(
        restored.representations["single"], first.representations["single"]
    )


def test_full_run_cannot_succeed_without_requested_representations(
    tmp_path: Path,
) -> None:
    class _MissingRepresentations(_ResumeBackend):
        def _representations(
            self, request: PredictionRequest
        ) -> Representations | None:
            return None

    calls: list[tuple[str, str, int]] = []
    backend = _MissingRepresentations("boltz2", calls)
    request = _request(tmp_path)

    with backend_override("boltz2", lambda: backend), pytest.raises(
        foldjax.PredictionOutputError, match="no requested representations"
    ):
        foldjax.predict(request)

    assert len(calls) == 1
    assert not (request.output_dir / MANIFEST_NAME).exists()


@pytest.mark.parametrize("damage", ["missing", "empty"])
def test_missing_or_empty_trunk_archive_is_not_reused(
    tmp_path: Path, damage: str
) -> None:
    calls: list[tuple[str, str, int]] = []
    request = _request(tmp_path, stop_after="trunk")
    with _backends(calls):
        first = foldjax.predict(request)
        archive = first.representations.path
        if damage == "missing":
            archive.unlink()
        else:
            archive.write_bytes(b"")
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 2
    assert resumed.skipped == ()
    assert resumed.results[0].representations.path.stat().st_size > 0


def test_redacted_options_never_create_a_false_resume_match(tmp_path: Path) -> None:
    calls: list[tuple[str, str, int]] = []
    request = _request(tmp_path, options={"api_key": "first-secret"})
    with _backends(calls):
        foldjax.predict(request)
        resumed = foldjax.predict_batch(
            dataclasses.replace(
                request,
                resume=True,
                options={"api_key": "different-secret"},
            )
        )

    document = (request.output_dir / MANIFEST_NAME).read_text(encoding="utf-8")
    assert "different-secret" not in document
    assert len(calls) == 2
    assert resumed.skipped == ()


def test_prior_pair_failures_do_not_block_a_later_multiseed_manifest(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, int]] = []
    first = _file(tmp_path / "first.native", b"first\n")
    second = _file(tmp_path / "second.native", b"second\n")
    output = tmp_path / "batch"
    request = PredictionRequest(
        model="boltz2",
        inputs=(first, second),
        input_format="native",
        weights=_file(tmp_path / "weights.bin", b"weights\n"),
        output_dir=output,
        seeds=(1, 2),
        on_error="continue",
        use_compile_cache=False,
    )

    with _backends(calls, fail_stem="first"):
        report = foldjax.predict_batch(request)

    first_output = output / "boltz2" / "first"
    second_output = output / "boltz2" / "second"
    assert len(report.failures) == 2
    assert not (first_output / MANIFEST_NAME).exists()
    top = json.loads((second_output / MANIFEST_NAME).read_text())
    assert top["seeds"] == [1, 2]
    assert len(top["samples"]) == 2


def test_same_size_weight_overwrite_forces_a_rerun(tmp_path: Path) -> None:
    calls: list[tuple[str, str, int]] = []
    request = _request(tmp_path)
    with _backends(calls):
        foldjax.predict(request)
        before = json.loads(
            (request.output_dir / MANIFEST_NAME).read_text(encoding="utf-8")
        )["weights"]["sha256"]
        request.weights.write_bytes(b"weights-b\n")
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    after = json.loads(
        (request.output_dir / MANIFEST_NAME).read_text(encoding="utf-8")
    )["weights"]["sha256"]
    assert len(calls) == 2
    assert resumed.skipped == ()
    assert before != after


def test_nested_weight_tree_overwrite_forces_a_rerun(tmp_path: Path) -> None:
    calls: list[tuple[str, str, int]] = []
    weights = tmp_path / "weights"
    nested = weights / "trunk"
    nested.mkdir(parents=True)
    shard = _file(nested / "params.bin", b"version-a")
    request = _request(tmp_path, weights=weights)
    with _backends(calls):
        foldjax.predict(request)
        shard.write_bytes(b"version-b")
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 2
    assert resumed.skipped == ()


def test_multiseed_hashes_a_checkpoint_once_even_with_many_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from foldjax import manifest

    request = _request(
        tmp_path,
        representations=None,
        seed=0,
        seeds=(0, 1),
    )
    document = json.loads(request.input.read_text(encoding="utf-8"))
    entities = []
    for index in range(40):
        alignment = _file(
            tmp_path / f"alignment-{index}.a3m",
            f">query-{index}\nAAAA\n".encode(),
        )
        entities.append(
            {
                "type": "protein",
                "id": f"A{index}",
                "sequence": "AAAA",
                "unpaired_msa": alignment.name,
            }
        )
    document["entities"] = entities
    request.input.write_text(json.dumps(document), encoding="utf-8")

    manifest._cached_weight_content_digest.cache_clear()
    manifest._cached_path_content_digest.cache_clear()
    original = manifest._hash_weight_contents
    weight_hashes = 0

    def counted(path: Path, signature):
        nonlocal weight_hashes
        if Path(path).resolve() == request.weights.resolve():
            weight_hashes += 1
        return original(path, signature)

    monkeypatch.setattr(manifest, "_hash_weight_contents", counted)
    calls: list[tuple[str, str, int]] = []
    with _backends(calls):
        foldjax.predict(request)

    assert len(calls) == 2
    assert weight_hashes == 1


def test_same_content_at_a_different_input_path_is_not_reused(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, int]] = []
    request = _request(tmp_path)
    alternate = tmp_path / "same-job.json"
    alternate.write_bytes(request.input.read_bytes())
    with _backends(calls):
        foldjax.predict(request)
        resumed = foldjax.predict_batch(
            dataclasses.replace(request, input=alternate, resume=True)
        )

    assert len(calls) == 2
    assert resumed.skipped == ()


def test_common_alignment_mutation_forces_a_rerun(tmp_path: Path) -> None:
    calls: list[tuple[str, str, int]] = []
    request = _request(tmp_path)
    alignment = tmp_path / "alignment.a3m"
    with _backends(calls):
        foldjax.predict(request)
        alignment.write_bytes(b">query\nTTTT\n")
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 2
    assert resumed.skipped == ()


def test_explicit_alignment_keeps_auto_msa_resume_verifiable(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, int]] = []
    request = _request(tmp_path, msa="auto")
    with _backends(calls):
        foldjax.predict(request)
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 1
    assert resumed.skipped == (request.output_dir,)


def test_path_like_option_tree_mutation_forces_a_rerun(tmp_path: Path) -> None:
    mols = tmp_path / "custom-mols"
    mols.mkdir()
    component = _file(mols / "ATP.pkl", b"component-a")
    request = _request(tmp_path, options={"mols": str(mols)})
    calls: list[tuple[str, str, int]] = []
    with _backends(calls):
        foldjax.predict(request)
        component.write_bytes(b"component-b")
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 2
    assert resumed.skipped == ()


def test_implicit_boltz_mols_tree_mutation_forces_a_rerun(
    tmp_path: Path,
) -> None:
    mols = tmp_path / "mols"
    mols.mkdir()
    component = _file(mols / "ATP.pkl", b"component-a")
    request = _request(tmp_path)
    calls: list[tuple[str, str, int]] = []
    with _backends(calls):
        foldjax.predict(request)
        component.write_bytes(b"component-b")
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 2
    assert resumed.skipped == ()


def test_boltz_scalar_weight_sidecar_mutation_forces_a_rerun(
    tmp_path: Path,
) -> None:
    weights = _file(tmp_path / "boltz2_conf.npz", b"arrays-a")
    sidecar = _file(tmp_path / "boltz2_conf.npz.json", b'{"scalars": {}}')
    request = _request(tmp_path, weights=weights)
    calls: list[tuple[str, str, int]] = []
    with _backends(calls):
        foldjax.predict(request)
        sidecar.write_bytes(b'{"scalars": 1}')
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 2
    assert resumed.skipped == ()


def test_missing_boltz_scalar_sidecar_is_a_stable_resume_identity(
    tmp_path: Path,
) -> None:
    weights = _file(tmp_path / "boltz2_conf.npz", b"arrays-a")
    request = _request(tmp_path, weights=weights)
    calls: list[tuple[str, str, int]] = []
    with _backends(calls):
        foldjax.predict(request)
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 1
    assert resumed.skipped == (request.output_dir,)


def test_adding_a_missing_boltz_scalar_sidecar_forces_a_rerun(
    tmp_path: Path,
) -> None:
    weights = _file(tmp_path / "boltz2_conf.npz", b"arrays-a")
    sidecar = tmp_path / "boltz2_conf.npz.json"
    request = _request(tmp_path, weights=weights)
    calls: list[tuple[str, str, int]] = []
    with _backends(calls):
        foldjax.predict(request)
        sidecar.write_bytes(b'{"scalars": {}}')
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 2
    assert resumed.skipped == ()


def test_replacing_a_missing_boltz_sidecar_with_a_directory_forces_a_rerun(
    tmp_path: Path,
) -> None:
    weights = _file(tmp_path / "boltz2_conf.npz", b"arrays-a")
    sidecar = tmp_path / "boltz2_conf.npz.json"
    request = _request(tmp_path, weights=weights)
    calls: list[tuple[str, str, int]] = []
    with _backends(calls):
        foldjax.predict(request)
        sidecar.mkdir()
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 2
    assert resumed.skipped == ()


def test_implicit_boltz_affinity_weight_mutation_forces_a_rerun(
    tmp_path: Path,
) -> None:
    weights = _file(tmp_path / "boltz2_conf.npz", b"confidence")
    _file(tmp_path / "boltz2_conf.npz.json", b'{"scalars": {}}')
    affinity = _file(tmp_path / "boltz2_aff.npz", b"affinity-a")
    _file(tmp_path / "boltz2_aff.npz.json", b'{"scalars": {}}')
    native = tmp_path / "native-job.json"
    native.write_text(
        json.dumps({"name": "job", "properties": [{"affinity": {"binder": "L"}}]}),
        encoding="utf-8",
    )
    request = _request(
        tmp_path,
        input=native,
        input_format="native",
        weights=weights,
    )
    calls: list[tuple[str, str, int]] = []
    with _backends(calls):
        foldjax.predict(request)
        affinity.write_bytes(b"affinity-b")
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 2
    assert resumed.skipped == ()


@pytest.mark.parametrize("companion", ["config", "esmc"])
def test_esmfold2_companion_weight_mutation_forces_a_rerun(
    tmp_path: Path, companion: str
) -> None:
    root = tmp_path / "esmfold2"
    root.mkdir()
    weights = _file(root / "model.safetensors", b"structure")
    config = _file(root / "config.json", b'{"model": "a"}')
    esmc = root / "esmc"
    esmc.mkdir()
    shard = _file(esmc / "model-00001.safetensors", b"language-a")
    request = _request(tmp_path, model="esmfold2", weights=weights)
    calls: list[tuple[str, str, int]] = []
    with _backends(calls):
        foldjax.predict(request)
        if companion == "config":
            config.write_bytes(b'{"model": "b"}')
        else:
            shard.write_bytes(b"language-b")
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 2
    assert resumed.skipped == ()


def test_native_protenix_template_search_is_conservatively_rerun(
    tmp_path: Path,
) -> None:
    search = _file(tmp_path / "templates.a3m", b">query\nAAAA\n")
    native = tmp_path / "native-job.json"
    native.write_text(
        json.dumps(
            {
                "name": "job",
                "sequences": [
                    {
                        "proteinChain": {
                            "sequence": "AAAAA",
                            "count": 1,
                            "templatesPath": str(search),
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    request = _request(
        tmp_path,
        model="protenix",
        input=native,
        input_format="native",
    )
    calls: list[tuple[str, str, int]] = []
    with _backends(calls):
        foldjax.predict(request)
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    document = json.loads((request.output_dir / MANIFEST_NAME).read_text())
    assert document["input_dependencies"]["verifiable"] is False
    assert len(calls) == 2
    assert resumed.skipped == ()


def test_implicit_protenix_language_model_weight_mutation_forces_a_rerun(
    tmp_path: Path,
) -> None:
    weights = _file(
        tmp_path / "protenix_mini_esm_v0.5.0.bin",
        b"structure",
    )
    language_model = _file(
        tmp_path / "esm2_t36_3B_UR50D.safetensors",
        b"language-a",
    )
    request = _request(tmp_path, model="protenix", weights=weights)
    calls: list[tuple[str, str, int]] = []
    with _backends(calls):
        foldjax.predict(request)
        language_model.write_bytes(b"language-b")
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 2
    assert resumed.skipped == ()


def test_implicit_protenix_ccd_asset_mutation_forces_a_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    components = _file(tmp_path / "components.cif", b"component-a")
    rdkit = _file(tmp_path / "components.cif.rdkit_mol.pkl", b"rdkit-a")
    monkeypatch.setenv("PROTENIX_CCD_COMPONENTS_FILE", str(components))
    monkeypatch.setenv("PROTENIX_CCD_RDKIT_MOL_FILE", str(rdkit))
    request = _request(tmp_path, model="opendde")
    document = json.loads(request.input.read_text(encoding="utf-8"))
    document["entities"].append(
        {"type": "ligand", "id": "L", "ccd": "ATP"}
    )
    request.input.write_text(json.dumps(document), encoding="utf-8")
    calls: list[tuple[str, str, int]] = []
    with _backends(calls):
        foldjax.predict(request)
        components.write_bytes(b"component-b")
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 2
    assert resumed.skipped == ()


def test_secret_path_option_is_neither_recorded_nor_reused(tmp_path: Path) -> None:
    secret = _file(tmp_path / "credential.pem", b"private")
    request = _request(tmp_path, options={"api_key_file": str(secret)})
    calls: list[tuple[str, str, int]] = []
    with _backends(calls):
        foldjax.predict(request)
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    payload = (request.output_dir / MANIFEST_NAME).read_text(encoding="utf-8")
    assert str(secret) not in payload
    assert len(calls) == 2
    assert resumed.skipped == ()


def test_self_contained_native_input_can_resume(tmp_path: Path) -> None:
    native = _file(tmp_path / "job.fasta", b">A\nAAAA\n")
    request = _request(tmp_path, input=native, input_format="native")
    calls: list[tuple[str, str, int]] = []
    with _backends(calls):
        foldjax.predict(request)
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 1
    assert resumed.skipped == (request.output_dir,)


def test_native_file_ligand_mutation_forces_a_rerun(tmp_path: Path) -> None:
    ligand = _file(tmp_path / "ligand.sdf", b"ligand-a")
    native = tmp_path / "native-job.json"
    native.write_text(
        json.dumps({"name": "job", "ligand": f"FILE_{ligand.name}"}),
        encoding="utf-8",
    )
    request = _request(tmp_path, input=native, input_format="native")
    calls: list[tuple[str, str, int]] = []
    with _backends(calls):
        foldjax.predict(request)
        ligand.write_bytes(b"ligand-b")
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 2
    assert resumed.skipped == ()


@pytest.mark.parametrize(
    "payload",
    [b'{"source_url":"https://example.invalid/data"}', b"{not-json"],
)
def test_opaque_structured_native_input_is_conservatively_rerun(
    tmp_path: Path, payload: bytes
) -> None:
    native = _file(tmp_path / "native-job.json", payload)
    request = _request(tmp_path, input=native, input_format="native")
    calls: list[tuple[str, str, int]] = []
    with _backends(calls):
        foldjax.predict(request)
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 2
    assert resumed.skipped == ()


class _BrokenRepresentationBackend(_ResumeBackend):
    def __init__(
        self,
        calls: list[tuple[str, str, int]],
        damage: str,
    ) -> None:
        super().__init__("boltz2", calls)
        self.damage = damage

    def _representations(self, request: PredictionRequest):
        value = super()._representations(request)
        assert value is not None
        if self.damage == "wrong-type":
            return object()
        if self.damage == "wrong-model":
            return Representations("opendde", value.path, value.manifest)
        if self.damage == "missing":
            value.path.unlink()
            return value
        if self.damage == "empty":
            value.path.write_bytes(b"")
            return value
        if self.damage == "garbage":
            value.path.write_bytes(b"not a zip archive")
            return value
        if self.damage == "empty-manifest":
            return Representations(self.name, value.path, {})
        if self.damage == "wrong-name":
            return Representations(
                self.name,
                value.path,
                {"pair": value.manifest["single"]},
            )
        entry = dict(value.manifest["single"])
        if self.damage == "shape":
            entry["shape"] = [3, 2]
        elif self.damage == "dtype":
            entry["dtype"] = "float64"
        return Representations(self.name, value.path, {"single": entry})


@pytest.mark.parametrize(
    "damage",
    [
        "wrong-type",
        "wrong-model",
        "missing",
        "empty",
        "garbage",
        "empty-manifest",
        "wrong-name",
        "shape",
        "dtype",
    ],
)
def test_invalid_initial_representation_artifact_is_rejected(
    tmp_path: Path, damage: str
) -> None:
    calls: list[tuple[str, str, int]] = []
    backend = _BrokenRepresentationBackend(calls, damage)
    request = _request(tmp_path)

    with backend_override("boltz2", lambda: backend), pytest.raises(
        foldjax.PredictionOutputError, match="invalid representations"
    ):
        foldjax.predict(request)

    assert len(calls) == 1
    assert not (request.output_dir / MANIFEST_NAME).exists()


@pytest.mark.parametrize("location", ["external", "symlink"])
def test_initial_representation_must_be_a_regular_artifact_under_output_root(
    tmp_path: Path, location: str
) -> None:
    class _EscapingBackend(_ResumeBackend):
        def _representations(
            self, request: PredictionRequest
        ) -> Representations | None:
            value = super()._representations(request)
            assert value is not None
            external = request.output_dir.parent / "external-representations.npz"
            value.path.replace(external)
            if location == "external":
                path = external
            else:
                value.path.symlink_to(external)
                path = value.path
            return Representations(self.name, path, value.manifest)

    calls: list[tuple[str, str, int]] = []
    backend = _EscapingBackend("boltz2", calls)
    request = _request(tmp_path)
    with backend_override("boltz2", lambda: backend), pytest.raises(
        foldjax.PredictionOutputError, match="outside its output root"
    ):
        foldjax.predict(request)


@pytest.mark.parametrize(
    ("selectors", "expected"),
    [
        (("single", "pair"), ("single", "pair")),
        (("all",), ("single", "pair")),
    ],
)
def test_multi_representation_order_survives_sorted_manifest_json(
    tmp_path: Path,
    selectors: tuple[str, ...],
    expected: tuple[str, ...],
) -> None:
    request = _request(tmp_path, representations=selectors)
    calls: list[tuple[str, str, int]] = []
    with _backends(calls):
        foldjax.predict(request)
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    record = json.loads(
        (request.output_dir / MANIFEST_NAME).read_text(encoding="utf-8")
    )["representations"]
    assert tuple(record["entries"]) != tuple(record["names"])
    assert resumed.results[0].representations.names() == expected
    assert len(calls) == 1


@pytest.mark.parametrize("damage", ["shape", "dtype", "identity", "model"])
def test_tampered_representation_manifest_forces_a_rerun(
    tmp_path: Path, damage: str
) -> None:
    request = _request(tmp_path)
    calls: list[tuple[str, str, int]] = []
    with _backends(calls):
        foldjax.predict(request)
        path = request.output_dir / MANIFEST_NAME
        document = json.loads(path.read_text(encoding="utf-8"))
        record = document["representations"]
        if damage in {"shape", "dtype"}:
            record["entries"]["single"][damage] = (
                [3, 2] if damage == "shape" else "float64"
            )
        elif damage == "identity":
            record["identity"] = "0" * 64
        else:
            record["model"] = "opendde"
        path.write_text(json.dumps(document), encoding="utf-8")
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 2
    assert resumed.skipped == ()


def test_rewritten_representation_payload_forces_a_rerun(tmp_path: Path) -> None:
    request = _request(tmp_path)
    calls: list[tuple[str, str, int]] = []
    with _backends(calls):
        first = foldjax.predict(request)
        np.savez(
            first.representations.path,
            single=np.full((2, 3), 7, dtype=np.float32),
        )
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 2
    assert resumed.skipped == ()


@pytest.mark.parametrize("damage", ["unsupported-version", "encrypted"])
def test_malformed_representation_zip_forces_a_rerun(
    tmp_path: Path, damage: str
) -> None:
    request = _request(tmp_path)
    calls: list[tuple[str, str, int]] = []
    with _backends(calls):
        first = foldjax.predict(request)
        archive = first.representations.path
        payload = bytearray(archive.read_bytes())
        central = payload.index(b"PK\x01\x02")
        if damage == "unsupported-version":
            # ZipFile.open raises NotImplementedError for an unsupported
            # version-needed field in the central-directory entry.
            payload[central + 6] = 0xFF
        else:
            # An unexpected encryption bit raises RuntimeError when no password
            # exists. Both are malformed artifacts, not reasons for resume to
            # crash before it can conservatively rerun the backend.
            payload[central + 8] |= 1
        archive.write_bytes(payload)
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 2
    assert resumed.skipped == ()


def test_representation_validation_reads_headers_but_not_array_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from foldjax import manifest

    request = _request(tmp_path, representations=("single", "pair"))
    original_digest = manifest._sha256_file

    def no_representation_digest(path: Path) -> str:
        if Path(path).name == "representations.npz":
            raise AssertionError("representation payload was fully hashed")
        return original_digest(path)

    def no_numpy_load(*_args, **_kwargs):
        raise AssertionError("representation array was materialised")

    monkeypatch.setattr(manifest, "_sha256_file", no_representation_digest)
    monkeypatch.setattr(np, "load", no_numpy_load)
    calls: list[tuple[str, str, int]] = []
    with _backends(calls):
        foldjax.predict(request)
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 1
    assert resumed.skipped == (request.output_dir,)


def test_empty_representation_tuple_is_a_full_run_without_capture(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, representations=())
    calls: list[tuple[str, str, int]] = []
    with _backends(calls):
        first = foldjax.predict(request)
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert first.representations is None
    assert resumed.results[0].representations is None
    assert len(calls) == 1


def test_bfloat16_representation_round_trips_and_resumes(tmp_path: Path) -> None:
    import ml_dtypes

    class _Bfloat16Backend(_ResumeBackend):
        def _representations(
            self, request: PredictionRequest
        ) -> Representations | None:
            if not request.representations:
                return None
            array = np.asarray(
                [[1.0, 2.0], [3.0, 4.0]], dtype=ml_dtypes.bfloat16
            )
            archive = request.output_dir / "representations.npz"
            np.savez(archive, single=array)
            return Representations(
                self.name,
                archive,
                {
                    "single": {
                        "shape": [2, 2],
                        "dtype": "bfloat16",
                        "axes": ["token", "channel"],
                        "space": "token",
                        "description": "mock bfloat16",
                    }
                },
            )

    request = _request(tmp_path)
    calls: list[tuple[str, str, int]] = []
    backend = _Bfloat16Backend("boltz2", calls)
    with backend_override("boltz2", lambda: backend):
        first = foldjax.predict(request)
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    expected = np.asarray([[1, 2], [3, 4]], dtype=ml_dtypes.bfloat16)
    assert first.representations["single"].dtype == ml_dtypes.bfloat16
    assert resumed.results[0].representations["single"].dtype == ml_dtypes.bfloat16
    np.testing.assert_array_equal(
        resumed.results[0].representations["single"], expected
    )
    assert len(calls) == 1


def test_representation_module_load_restores_bfloat16(tmp_path: Path) -> None:
    import ml_dtypes

    from foldjax.models._representations import (
        RepresentationSpec,
        load,
        save,
    )

    expected = np.asarray([[1, 2], [3, 4]], dtype=ml_dtypes.bfloat16)
    specs = {
        "single": RepresentationSpec(
            "single",
            ("token", "channel"),
            "token",
            "mock",
        )
    }
    save(tmp_path, {"single": expected}, specs, model="boltz2")

    restored = load(tmp_path)["single"]
    assert restored.dtype == ml_dtypes.bfloat16
    np.testing.assert_array_equal(restored, expected)


def test_fresh_process_validates_bfloat16_without_dtype_registration(
    tmp_path: Path,
) -> None:
    import os
    import subprocess
    import sys

    import ml_dtypes

    archive = tmp_path / "representations.npz"
    np.savez(
        archive,
        single=np.asarray([[1, 2]], dtype=ml_dtypes.bfloat16),
    )
    script = """
import sys
from pathlib import Path
assert 'ml_dtypes' not in sys.modules
from foldjax.api import _representation_artifact_error
from foldjax.schema import Representations
assert 'ml_dtypes' not in sys.modules
value = Representations(
    'boltz2',
    Path(sys.argv[1]),
    {'single': {'shape': [1, 2], 'dtype': 'bfloat16'}},
)
error = _representation_artifact_error(
    value,
    expected_model='boltz2',
    expected_names=('single',),
)
if error is not None:
    raise SystemExit(error)
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script, str(archive)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr


def test_same_size_structure_mutation_forces_a_rerun(tmp_path: Path) -> None:
    request = _request(tmp_path)
    calls: list[tuple[str, str, int]] = []
    with _backends(calls):
        first = foldjax.predict(request)
        structure = first.samples[0].structure_path
        structure.write_bytes(b"x" * structure.stat().st_size)
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 2
    assert resumed.skipped == ()


@pytest.mark.parametrize("damage", ["duplicate-structure", "duplicate-slot"])
def test_duplicate_manifest_sample_identity_forces_a_rerun(
    tmp_path: Path, damage: str
) -> None:
    request = _request(tmp_path)
    calls: list[tuple[str, str, int]] = []
    with _backends(calls):
        first = foldjax.predict(request)
        path = request.output_dir / MANIFEST_NAME
        document = json.loads(path.read_text(encoding="utf-8"))
        duplicate = json.loads(json.dumps(document["samples"][0]))
        if damage == "duplicate-structure":
            duplicate["metadata"]["sample"] = 1
        else:
            document["samples"][0]["metadata"]["sample"] = 0
            duplicate["metadata"]["sample"] = 0
            source = first.samples[0].structure_path
            second = request.output_dir / "second.cif"
            second.write_bytes(source.read_bytes())
            duplicate["structure_path"] = second.name
        document["samples"].append(duplicate)
        path.write_text(json.dumps(document), encoding="utf-8")
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 2
    assert resumed.skipped == ()


def test_manifest_relative_artifacts_restore_after_cwd_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "nested").mkdir()
    output = tmp_path / "nested" / ".." / "out"
    request = _request(tmp_path, output_dir=output)
    calls: list[tuple[str, str, int]] = []
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    with _backends(calls):
        foldjax.predict(request)
        monkeypatch.chdir(elsewhere)
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 1
    assert resumed.skipped == (output,)
    assert resumed.results[0].samples[0].structure_path.is_file()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("structure", "../../external.cif"),
        ("structure", "bad\x00name.cif"),
        ("representations", "bad\x00name.npz"),
    ],
)
def test_malformed_or_escaping_manifest_artifact_path_forces_a_rerun(
    tmp_path: Path, field: str, value: str
) -> None:
    request = _request(tmp_path)
    calls: list[tuple[str, str, int]] = []
    external = _file(tmp_path / "external.cif", b"data_external\n")
    del external
    with _backends(calls):
        foldjax.predict(request)
        path = request.output_dir / MANIFEST_NAME
        document = json.loads(path.read_text(encoding="utf-8"))
        if field == "structure":
            document["samples"][0]["structure_path"] = value
        else:
            document["representations"]["path"] = value
        path.write_text(json.dumps(document), encoding="utf-8")
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 2
    assert resumed.skipped == ()


def test_symlinked_restored_structure_forces_a_rerun(tmp_path: Path) -> None:
    request = _request(tmp_path)
    calls: list[tuple[str, str, int]] = []
    with _backends(calls):
        first = foldjax.predict(request)
        structure = first.samples[0].structure_path
        payload = structure.read_bytes()
        external = _file(tmp_path / "external.cif", payload)
        structure.unlink()
        structure.symlink_to(external)
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 2
    assert resumed.skipped == ()


def test_invalid_utf8_manifest_forces_a_rerun(tmp_path: Path) -> None:
    request = _request(tmp_path)
    calls: list[tuple[str, str, int]] = []
    with _backends(calls):
        foldjax.predict(request)
        (request.output_dir / MANIFEST_NAME).write_bytes(b"\xff\xfe")
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 2
    assert resumed.skipped == ()


@pytest.mark.parametrize("score", [True, "0.75", 10**400])
def test_manifest_score_type_drift_forces_a_rerun(
    tmp_path: Path, score: object
) -> None:
    request = _request(tmp_path)
    calls: list[tuple[str, str, int]] = []
    with _backends(calls):
        foldjax.predict(request)
        path = request.output_dir / MANIFEST_NAME
        document = json.loads(path.read_text(encoding="utf-8"))
        document["samples"][0]["scores"]["confidence"] = score
        path.write_text(json.dumps(document), encoding="utf-8")
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(calls) == 2
    assert resumed.skipped == ()
