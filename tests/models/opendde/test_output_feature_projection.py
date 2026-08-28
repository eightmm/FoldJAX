"""OpenDDE keeps a bounded, reader-owned postprocessing feature view."""

from __future__ import annotations

import gc
import hashlib
import json
import weakref
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models._feature_storage import compact_msa_storage
from foldjax.models.opendde.cli import predict as predict_impl
from foldjax.models.opendde.data.compact_categories import (
    compact_ref_atom_category_storage,
)
from foldjax.models.opendde.data.featurize_json import featurize_opendde_json
from foldjax.models.opendde.postprocess import (
    OPENDDE_GENERATED_OUTPUT_FEATURE_FIELDS,
    opendde_confidence_scores,
    project_generated_output_features,
)
from foldjax.models.protenix.data.template_features import dedup_templates
from foldjax.schema import PaddingConfig


class _ReaderPoison(Mapping[str, Any]):
    def __init__(self, values: Mapping[str, Any], allowed: frozenset[str]):
        self._values = dict(values)
        self._allowed = allowed

    def _check(self, key: str) -> None:
        if key not in self._allowed:
            raise AssertionError(f"reader reached non-admitted field {key!r}")

    def __getitem__(self, key: str) -> Any:
        self._check(key)
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        self._check(key)
        return key in self._values

    def get(self, key: str, default: Any = None) -> Any:
        self._check(key)
        return self._values.get(key, default)


def _features(sequence: str = "ACDE") -> dict[str, Any]:
    return featurize_opendde_json(
        {"sequences": [{"proteinChain": {"sequence": sequence}}]},
        n_queries=2,
        n_keys=4,
        augment_reference=False,
    )


def _output(features: Mapping[str, Any]) -> dict[str, np.ndarray]:
    n_atom = len(features["atom_to_token_idx"])
    return {
        "coordinate": np.arange(n_atom * 3, dtype=np.float32).reshape(1, n_atom, 3)
        * 0.01,
        "atom_plddt": np.full((1, n_atom), 0.75, dtype=np.float32),
        "summary_plddt": np.asarray([0.75], dtype=np.float32),
        "summary_ranking_score": np.asarray([0.5], dtype=np.float32),
    }


def _non_npz_hashes(root: Path) -> dict[Path, str]:
    return {
        path.relative_to(root): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file() and path.suffix != ".npz"
    }


def _tree_bytes_signature(tree: Any) -> tuple[str, tuple[tuple[Any, ...], ...]]:
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    signature = []
    for leaf in leaves:
        array = np.asarray(leaf)
        payload = (
            repr(array.tolist()).encode()
            if array.dtype.hasobject
            else array.tobytes(order="C")
        )
        signature.append(
            (array.dtype.str, array.shape, hashlib.sha256(payload).hexdigest())
        )
    return str(treedef), tuple(signature)


def _template_probe_hlo_hash(features: Mapping[str, Any]) -> bytes:
    def graph_probe(restype, distogram):
        return jnp.sum(restype) + jnp.sum(distogram[..., 0])

    lowered = jax.jit(graph_probe).lower(
        jnp.asarray(features["restype"]),
        jnp.asarray(features["template_distogram"]),
    )
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    return hashlib.sha256(stablehlo.encode()).digest()


def test_opendde_projection_is_separate_shallow_and_nonretaining() -> None:
    features = _features("ACDEFGHIK")
    template_ref = weakref.ref(features["template_distogram"])
    projected = project_generated_output_features(features)

    assert projected is not features
    assert set(projected) == OPENDDE_GENERATED_OUTPUT_FEATURE_FIELDS & features.keys()
    assert all(projected[name] is features[name] for name in projected)
    del features
    gc.collect()
    assert template_ref() is None


def test_opendde_projection_falls_back_by_identity() -> None:
    features = _features()
    missing = dict(features)
    del missing["output_atom_element"]
    assert project_generated_output_features(missing) is missing

    mismatched = dict(features)
    mismatched["output_atom_chain_id"] = mismatched["output_atom_chain_id"][:-1]
    assert project_generated_output_features(mismatched) is mismatched


def test_shape_score_poison_and_writer_bytes_pin_the_reader_union(tmp_path) -> None:
    from foldjax.models.opendde.cli.predict import _write

    job = {
        "name": "mixed",
        "sequences": [
            {"proteinChain": {"sequence": "GACE"}},
            {"ligand": {"ligand": "CCD_ATP"}},
            {"dnaSequence": {"sequence": "GA"}},
            {"rnaSequence": {"sequence": "UC"}},
            {"ion": {"ion": "MG"}},
        ],
    }
    features = featurize_opendde_json(
        job,
        n_queries=2,
        n_keys=4,
        augment_reference=False,
    )
    projected = project_generated_output_features(features)
    poisoned = _ReaderPoison(projected, OPENDDE_GENERATED_OUTPUT_FEATURE_FIELDS)
    output = _output(features)

    dense_scores = opendde_confidence_scores(
        output,
        features,
        num_recycles=2,
        return_confidence_details=False,
    )
    projected_scores = opendde_confidence_scores(
        output,
        poisoned,
        num_recycles=2,
        return_confidence_details=False,
    )
    assert dense_scores.keys() == projected_scores.keys()
    for name in dense_scores:
        np.testing.assert_array_equal(dense_scores[name], projected_scores[name])

    dense_root = tmp_path / "dense"
    projected_root = tmp_path / "projected"
    _write(
        dense_root,
        job_name="mixed",
        seed=3,
        output={**output, **dense_scores},
        features=features,
        include_raw=True,
        include_trunk=False,
    )
    _write(
        projected_root,
        job_name="mixed",
        seed=3,
        output={**output, **projected_scores},
        features=poisoned,
        include_raw=True,
        include_trunk=False,
    )
    assert _non_npz_hashes(dense_root) == _non_npz_hashes(projected_root)
    dense_npz = next(dense_root.rglob("*.npz"))
    projected_npz = next(projected_root.rglob("*.npz"))
    with np.load(dense_npz) as dense, np.load(projected_npz) as compact:
        assert dense.files == compact.files
        for name in dense.files:
            assert dense[name].tobytes() == compact[name].tobytes(), name


def test_raw_confidence_fallback_reads_only_admitted_linear_fields() -> None:
    features = _features("AC")
    projected = project_generated_output_features(features)
    poisoned = _ReaderPoison(projected, OPENDDE_GENERATED_OUTPUT_FEATURE_FIELDS)
    n_token = len(features["asym_id"])
    n_atom = len(features["atom_to_token_idx"])
    output = {
        "coordinate": jnp.zeros((1, n_atom, 3), dtype=jnp.float32),
        "plddt": jnp.zeros((1, n_atom, 50), dtype=jnp.float32),
        "pae": jnp.zeros((1, n_token, n_token, 64), dtype=jnp.float32),
        "pde": jnp.zeros((1, n_token, n_token, 64), dtype=jnp.float32),
        "distogram_logits": jnp.zeros((n_token, n_token, 96), dtype=jnp.float32),
    }

    scores = opendde_confidence_scores(
        output,
        poisoned,
        num_recycles=1,
        include_shape_complementarity=False,
        return_confidence_details=False,
    )

    assert scores["atom_plddt"].shape == (1, n_atom)
    assert scores["summary_ranking_score"].shape == (1,)


def test_padded_cli_projection_never_reaches_model_bound_features(
    tmp_path, monkeypatch
) -> None:
    from foldjax.models.opendde import postprocess as postprocess_impl
    from foldjax.models.opendde.data.padding import (
        pad_opendde_features,
        select_opendde_model_features,
    )
    from foldjax.models.opendde.models.msa_sampling import (
        sample_opendde_msa_cycle_features,
    )

    input_path = tmp_path / "job.json"
    job = {
        "name": "job",
        "sequences": [{"proteinChain": {"sequence": "ACDE"}}],
    }
    input_path.write_text(json.dumps([job]), encoding="utf-8")
    weights_path = tmp_path / "weights.npz"
    weights_path.write_bytes(b"fixture")
    padding = PaddingConfig(msa=8)

    expected_source = predict_impl._featurize(
        job,
        base_dir=input_path.parent,
        n_queries=2,
        n_keys=4,
        max_msa_depth=16384,
        seed=101,
    )
    expected_source = compact_msa_storage(expected_source)
    expected_cycles = sample_opendde_msa_cycle_features(
        expected_source,
        num_recycles=2,
        seed=101,
    )
    expected_padded, expected_cycles, _ = pad_opendde_features(
        expected_source,
        expected_cycles,
        padding,
        n_queries=2,
        n_keys=4,
    )
    expected_model = select_opendde_model_features(expected_padded)
    expected_model = dedup_templates(expected_model)
    expected_model = compact_ref_atom_category_storage(expected_model)
    captured: dict[str, Any] = {}

    def writer_only_projection(features):
        return {
            name: features[name]
            for name in (
                "output_atom_name",
                "output_atom_element",
                "output_atom_res_name",
                "output_atom_chain_id",
                "output_atom_res_id",
            )
        }

    def fake_predict(model_features, _params, **kwargs):
        captured["model"] = model_features
        captured["cycles"] = kwargs["cycle_msa_features"]
        n_atom = len(model_features["atom_padding_mask"])
        return {"coordinate": np.zeros((1, n_atom, 3), dtype=np.float32)}

    def fake_score(output, _output_features, **_kwargs):
        n_atom = len(expected_source["atom_to_token_idx"])
        return {
            **output,
            "atom_plddt": np.ones((1, n_atom), dtype=np.float32),
            "summary_ranking_score": np.ones((1,), dtype=np.float32),
        }

    def fake_write(root, **kwargs):
        captured["writer"] = kwargs["features"]
        return [Path(root) / "job" / "prediction.cif"]

    monkeypatch.setattr(
        postprocess_impl, "project_generated_output_features", writer_only_projection
    )
    monkeypatch.setattr(predict_impl, "_predict", fake_predict)
    monkeypatch.setattr(predict_impl, "_score", fake_score)
    monkeypatch.setattr(predict_impl, "_write", fake_write)

    predict_impl.main(
        [
            "--input-json",
            str(input_path),
            "--weights",
            str(weights_path),
            "--out",
            str(tmp_path / "out"),
            "--n-sample",
            "1",
            "--n-step",
            "1",
            "--n-cycle",
            "2",
            "--n-queries",
            "2",
            "--n-keys",
            "4",
        ],
        padding=padding,
        _prepared_params_loader=lambda _path, _dtype, _cacheable: (),
    )

    assert set(captured["writer"]) == {
        "output_atom_name",
        "output_atom_element",
        "output_atom_res_name",
        "output_atom_chain_id",
        "output_atom_res_id",
    }
    assert _tree_bytes_signature(captured["model"]) == _tree_bytes_signature(
        expected_model
    )
    assert _tree_bytes_signature(captured["cycles"]) == _tree_bytes_signature(
        expected_cycles
    )
    assert _template_probe_hlo_hash(captured["model"]) == _template_probe_hlo_hash(
        expected_model
    )


def test_padded_cli_releases_source_intermediates_and_model_before_score(
    tmp_path, monkeypatch
) -> None:
    from foldjax.models.opendde.data import padding as padding_impl
    from foldjax.models.opendde.models import msa_sampling as msa_impl

    input_path = tmp_path / "job.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "name": "job",
                    "sequences": [{"proteinChain": {"sequence": "ACD"}}],
                }
            ]
        ),
        encoding="utf-8",
    )
    weights_path = tmp_path / "weights.npz"
    weights_path.write_bytes(b"fixture")
    source_refs: list[weakref.ReferenceType[np.ndarray]] = []
    sampled_refs: list[weakref.ReferenceType[np.ndarray]] = []
    padded_refs: list[weakref.ReferenceType[np.ndarray]] = []
    model_refs: list[weakref.ReferenceType[np.ndarray]] = []
    actual_atoms: list[int] = []
    real_featurize = predict_impl._featurize
    real_sample = msa_impl.sample_opendde_msa_cycle_features
    real_pad = padding_impl.pad_opendde_features

    def recording_featurize(job, **kwargs):
        features = real_featurize(job, **kwargs)
        actual_atoms.append(len(features["atom_to_token_idx"]))
        source_refs.append(weakref.ref(features["template_distogram"]))
        return features

    def recording_sample(features, **kwargs):
        sampled = real_sample(features, **kwargs)
        sampled_refs.extend(weakref.ref(cycle["msa"]) for cycle in sampled)
        return sampled

    def recording_pad(features, sampled, *args, **kwargs):
        padded, cycles, plan = real_pad(features, sampled, *args, **kwargs)
        padded_refs.append(weakref.ref(padded["template_distogram"]))
        return padded, cycles, plan

    monkeypatch.setattr(predict_impl, "_featurize", recording_featurize)
    monkeypatch.setattr(msa_impl, "sample_opendde_msa_cycle_features", recording_sample)
    monkeypatch.setattr(padding_impl, "pad_opendde_features", recording_pad)

    def fake_predict(model_features, _params, **_kwargs):
        gc.collect()
        assert source_refs and all(reference() is None for reference in source_refs)
        assert sampled_refs and all(reference() is None for reference in sampled_refs)
        assert padded_refs and all(reference() is None for reference in padded_refs)
        model_refs.append(weakref.ref(model_features["template_distogram"]))
        n_atom = len(model_features["atom_padding_mask"])
        return {"coordinate": np.zeros((1, n_atom, 3), dtype=np.float32)}

    def fake_score(output, output_features, **_kwargs):
        gc.collect()
        assert model_refs and all(reference() is None for reference in model_refs)
        assert set(output_features) <= OPENDDE_GENERATED_OUTPUT_FEATURE_FIELDS
        return {
            **output,
            "atom_plddt": np.ones((1, actual_atoms[0]), dtype=np.float32),
            "summary_ranking_score": np.ones((1,), dtype=np.float32),
        }

    writes: list[Mapping[str, Any]] = []
    monkeypatch.setattr(predict_impl, "_predict", fake_predict)
    monkeypatch.setattr(predict_impl, "_score", fake_score)
    monkeypatch.setattr(
        predict_impl,
        "_write",
        lambda root, **kwargs: writes.append(kwargs)
        or [Path(root) / "job" / "prediction.cif"],
    )
    predict_impl.main(
        [
            "--input-json",
            str(input_path),
            "--weights",
            str(weights_path),
            "--out",
            str(tmp_path / "out"),
            "--n-sample",
            "1",
            "--n-step",
            "1",
            "--n-cycle",
            "2",
            "--n-queries",
            "2",
            "--n-keys",
            "4",
            "--include-raw",
        ],
        padding=PaddingConfig(msa=8),
        _prepared_params_loader=lambda _path, _dtype, _cacheable: (),
    )

    assert writes[0]["include_raw"] is True
    assert set(writes[0]["features"]) <= OPENDDE_GENERATED_OUTPUT_FEATURE_FIELDS


def test_trunk_only_cli_does_not_build_an_output_projection(
    tmp_path, monkeypatch
) -> None:
    from foldjax.models.opendde import postprocess as postprocess_impl

    input_path = tmp_path / "job.json"
    input_path.write_text(
        '[{"name":"job","sequences":[{"proteinChain":{"sequence":"A"}}]}]',
        encoding="utf-8",
    )
    weights_path = tmp_path / "weights.npz"
    weights_path.write_bytes(b"fixture")
    monkeypatch.setattr(
        postprocess_impl,
        "project_generated_output_features",
        lambda _features: (_ for _ in ()).throw(
            AssertionError("trunk-only must not retain output features")
        ),
    )
    monkeypatch.setattr(
        predict_impl,
        "_predict",
        lambda *_args, **_kwargs: {"coordinate": np.zeros((1, 5, 3), np.float32)},
    )
    predict_impl.main(
        [
            "--input-json",
            str(input_path),
            "--weights",
            str(weights_path),
            "--out",
            str(tmp_path / "out"),
            "--stop-after",
            "trunk",
            "--n-sample",
            "1",
            "--n-step",
            "1",
            "--n-cycle",
            "1",
            "--n-queries",
            "2",
            "--n-keys",
            "4",
        ],
        _prepared_params_loader=lambda _path, _dtype, _cacheable: (),
    )
