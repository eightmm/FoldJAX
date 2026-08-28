"""Managed writer snapshots retain only fields their concrete readers use."""

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
import pytest

from foldjax.models._feature_storage import compact_msa_storage
from foldjax.models.protenix.cli import predict as predict_impl
from foldjax.models.protenix.data.compact_categories import (
    compact_ref_atom_category_storage,
)
from foldjax.models.protenix.data.featurize_json import featurize_protein_json
from foldjax.models.protenix.data.output import (
    PROTENIX_GENERATED_OUTPUT_FEATURE_FIELDS,
    project_generated_writer_features,
    write_protenix_outputs,
)
from foldjax.models.protenix.data.static_io import save_static_feature_npz
from foldjax.models.protenix.data.template_features import dedup_templates


class _ReaderPoison(Mapping[str, Any]):
    """Raise if a concrete reader reaches outside its admitted field set."""

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


def _protein_features(sequence: str = "AG") -> dict[str, Any]:
    return featurize_protein_json(
        {"sequences": [{"proteinChain": {"sequence": sequence}}]},
        n_queries=2,
        n_keys=4,
    )


def _prediction(features: Mapping[str, Any]) -> dict[str, np.ndarray]:
    n_atom = len(features["atom_to_token_idx"])
    return {
        "coordinate": np.arange(n_atom * 3, dtype=np.float32).reshape(1, n_atom, 3),
        "atom_plddt": np.full((1, n_atom), 0.75, dtype=np.float32),
        "summary_plddt": np.asarray([75.0], dtype=np.float32),
        "summary_ranking_score": np.asarray([0.75], dtype=np.float32),
    }


def _file_hashes(root: Path) -> dict[Path, str]:
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
    def graph_probe(distogram, unit_vector):
        return jnp.sum(distogram[..., 0]) + jnp.sum(unit_vector[..., 0])

    lowered = jax.jit(graph_probe).lower(
        jnp.asarray(features["template_distogram"]),
        jnp.asarray(features["template_unit_vector"]),
    )
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    return hashlib.sha256(stablehlo.encode()).digest()


def test_generated_writer_projection_is_explicit_shallow_and_nonretaining() -> None:
    features = _protein_features("ACDEFGHIK")
    template_ref = weakref.ref(features["template_distogram"])
    projected = project_generated_writer_features(features)

    assert projected is not features
    assert set(projected) == PROTENIX_GENERATED_OUTPUT_FEATURE_FIELDS & features.keys()
    assert all(projected[name] is features[name] for name in projected)
    del features
    gc.collect()
    assert template_ref() is None


def test_generated_writer_projection_falls_back_by_identity() -> None:
    features = _protein_features()
    missing = dict(features)
    del missing["output_atom_name"]
    assert project_generated_writer_features(missing) is missing

    mismatched = dict(features)
    mismatched["output_atom_res_id"] = mismatched["output_atom_res_id"][:-1]
    assert project_generated_writer_features(mismatched) is mismatched

    no_atom_axis = dict(features)
    del no_atom_axis["atom_to_token_idx"]
    assert project_generated_writer_features(no_atom_axis) is no_atom_axis


def test_writer_poison_and_covalent_bytes_pin_the_complete_reader_set(tmp_path) -> None:
    features = featurize_protein_json(
        {
            "name": "bonded",
            "sequences": [
                {"proteinChain": {"sequence": "C", "id": ["P"]}},
                {"ion": {"ion": "MG", "id": ["M"]}},
            ],
            "covalent_bonds": [
                {
                    "entity1": 1,
                    "copy1": 1,
                    "position1": 1,
                    "atom1": "SG",
                    "entity2": 2,
                    "copy2": 1,
                    "position2": 1,
                    "atom2": "MG",
                }
            ],
        },
        n_queries=2,
        n_keys=4,
    )
    projected = project_generated_writer_features(features)
    poisoned = _ReaderPoison(projected, PROTENIX_GENERATED_OUTPUT_FEATURE_FIELDS)
    dense_root = tmp_path / "dense"
    projected_root = tmp_path / "projected"
    output = _prediction(features)

    write_protenix_outputs(
        dense_root,
        job_name="bonded",
        seed=7,
        output=output,
        features=features,
        include_raw=True,
    )
    write_protenix_outputs(
        projected_root,
        job_name="bonded",
        seed=7,
        output=output,
        features=poisoned,
        include_raw=True,
    )

    assert _file_hashes(dense_root) == _file_hashes(projected_root)
    dense_npz = next(dense_root.rglob("*.npz"))
    projected_npz = next(projected_root.rglob("*.npz"))
    with np.load(dense_npz) as dense, np.load(projected_npz) as compact:
        assert dense.files == compact.files
        for name in dense.files:
            assert dense[name].tobytes() == compact[name].tobytes(), name


def test_generated_cli_projection_never_reaches_model_bound_features(
    tmp_path, monkeypatch
) -> None:
    from foldjax.models.protenix.data import output as output_impl

    input_path = tmp_path / "job.json"
    job = {
        "name": "job",
        "sequences": [{"proteinChain": {"sequence": "ACDE"}}],
    }
    input_path.write_text(json.dumps([job]), encoding="utf-8")
    expected = compact_msa_storage(_protein_features("ACDE"))
    expected = compact_msa_storage(dedup_templates(expected))
    expected = compact_ref_atom_category_storage(expected)
    captured: dict[str, Mapping[str, Any]] = {}

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

    def fake_predict(_params, model_features, **_kwargs):
        captured["model"] = model_features
        return _prediction(model_features)

    def fake_writer(root, **kwargs):
        captured["writer"] = kwargs["features"]
        return [Path(root) / "job" / "prediction.cif"]

    monkeypatch.setattr(
        output_impl, "project_generated_writer_features", writer_only_projection
    )
    monkeypatch.setattr(
        "foldjax.models.protenix.models.predict.protenix_predict_static", fake_predict
    )
    monkeypatch.setattr(output_impl, "write_protenix_outputs", fake_writer)

    predict_impl.main(
        [
            "--model-name",
            "unknown",
            "--weights",
            str(tmp_path / "unused.jax"),
            "--input-json",
            str(input_path),
            "--out",
            str(tmp_path / "out"),
            "--seeds",
            "7",
            "--output-format",
            "protenix",
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
            "--cpu-only",
            "--no-compile-cache",
        ],
        _prepared_params_loader=lambda _path, _dtype, _cacheable: (),
    )

    assert set(captured["writer"]) == {
        "output_atom_name",
        "output_atom_element",
        "output_atom_res_name",
        "output_atom_chain_id",
        "output_atom_res_id",
    }
    assert _tree_bytes_signature(captured["model"]) == _tree_bytes_signature(expected)
    assert _template_probe_hlo_hash(captured["model"]) == _template_probe_hlo_hash(
        expected
    )


def test_cli_releases_every_full_template_snapshot_before_params_load(
    tmp_path, monkeypatch
) -> None:
    input_path = tmp_path / "jobs.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "name": name,
                    "sequences": [{"proteinChain": {"sequence": sequence}}],
                }
                for name, sequence in (("first", "AG"), ("second", "ACD"))
            ]
        ),
        encoding="utf-8",
    )
    template_refs: list[weakref.ReferenceType[np.ndarray]] = []
    real_dedup = dedup_templates

    def recording_dedup(features):
        for name in (
            "template_distogram",
            "template_unit_vector",
            "template_pseudo_beta_mask",
            "template_backbone_frame_mask",
        ):
            template_refs.append(weakref.ref(features[name]))
        return real_dedup(features)

    monkeypatch.setattr(
        "foldjax.models.protenix.data.template_features.dedup_templates",
        recording_dedup,
    )

    def params_loader(_path, _dtype, _cacheable):
        gc.collect()
        assert len(template_refs) == 8
        assert all(reference() is None for reference in template_refs)
        return ()

    monkeypatch.setattr(
        "foldjax.models.protenix.models.predict.protenix_predict_static",
        lambda _params, features, **_kwargs: _prediction(features),
    )
    predict_impl.main(
        [
            "--model-name",
            "unknown",
            "--weights",
            str(tmp_path / "unused.jax"),
            "--input-json",
            str(input_path),
            "--out",
            str(tmp_path / "out"),
            "--seeds",
            "7",
            "--output-format",
            "both",
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
            "--cpu-only",
            "--no-compile-cache",
        ],
        _prepared_params_loader=params_loader,
    )


def test_static_writer_snapshot_keeps_the_full_custom_mapping(
    tmp_path, monkeypatch
) -> None:
    features_path = tmp_path / "features.npz"
    features = _protein_features()
    features["custom_writer_field"] = np.asarray([17], dtype=np.int32)
    save_static_feature_npz(features_path, features)
    captured: dict[str, Mapping[str, Any]] = {}

    monkeypatch.setattr(
        "foldjax.models.protenix.models.predict.protenix_predict_static",
        lambda _params, model_features, **_kwargs: _prediction(model_features),
    )

    def fake_writer(root, **kwargs):
        captured["features"] = kwargs["features"]
        return [Path(root) / "static" / "prediction.cif"]

    monkeypatch.setattr(
        "foldjax.models.protenix.data.output.write_protenix_outputs", fake_writer
    )
    predict_impl.main(
        [
            "--model-name",
            "unknown",
            "--weights",
            str(tmp_path / "unused.jax"),
            "--features",
            str(features_path),
            "--out",
            str(tmp_path / "out"),
            "--output-format",
            "protenix",
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
            "--cpu-only",
            "--no-compile-cache",
        ],
        _prepared_params_loader=lambda _path, _dtype, _cacheable: (),
    )

    assert "custom_writer_field" in captured["features"]
    assert "template_distogram" in captured["features"]


@pytest.mark.parametrize(
    ("mode_args", "output_name"),
    [
        (("--output-format", "npz"), "raw.npz"),
        (("--output-format", "both", "--prewarm-only"), "prewarm"),
        (("--output-format", "protenix", "--stop-after", "trunk"), "trunk"),
    ],
)
def test_non_writer_branches_do_not_retain_an_output_projection(
    tmp_path, monkeypatch, mode_args, output_name
) -> None:
    from foldjax.models.protenix.data import output as output_impl

    input_path = tmp_path / f"{output_name}.json"
    input_path.write_text(
        '[{"sequences":[{"proteinChain":{"sequence":"A"}}]}]',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        output_impl,
        "project_generated_writer_features",
        lambda _features: (_ for _ in ()).throw(
            AssertionError("non-writer branch must not retain writer features")
        ),
    )
    monkeypatch.setattr(
        "foldjax.models.protenix.models.predict.protenix_predict_static",
        lambda _params, features, **_kwargs: _prediction(features),
    )

    predict_impl.main(
        [
            "--model-name",
            "unknown",
            "--weights",
            str(tmp_path / "unused.jax"),
            "--input-json",
            str(input_path),
            "--out",
            str(tmp_path / output_name),
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
            "--cpu-only",
            "--no-compile-cache",
            *mode_args,
        ],
        _prepared_params_loader=lambda _path, _dtype, _cacheable: (),
    )


def test_padding_crops_before_the_projected_writer_view(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "job.json"
    input_path.write_text(
        '[{"name":"job","sequences":[{"proteinChain":{"sequence":"A"}}]}]',
        encoding="utf-8",
    )
    model_atoms: list[int] = []
    writer_calls: list[dict[str, Any]] = []

    def fake_predict(_params, features, **_kwargs):
        n_atom = len(features["atom_to_token_idx"])
        model_atoms.append(n_atom)
        return {
            "coordinate": np.zeros((1, n_atom, 3), dtype=np.float32),
            "atom_plddt": np.ones((1, n_atom), dtype=np.float32),
            "summary_ranking_score": np.ones((1,), dtype=np.float32),
        }

    def fake_writer(root, **kwargs):
        writer_calls.append(kwargs)
        return [Path(root) / "job" / "prediction.cif"]

    monkeypatch.setattr(
        "foldjax.models.protenix.models.predict.protenix_predict_static", fake_predict
    )
    monkeypatch.setattr(
        "foldjax.models.protenix.data.output.write_protenix_outputs", fake_writer
    )
    predict_impl.main(
        [
            "--model-name",
            "unknown",
            "--weights",
            str(tmp_path / "unused.jax"),
            "--input-json",
            str(input_path),
            "--out",
            str(tmp_path / "out"),
            "--output-format",
            "protenix",
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
            "--padding",
            "--pad-tokens",
            "8",
            "--pad-atoms",
            "8",
            "--pad-msa",
            "4",
            "--pad-templates",
            "4",
            "--cpu-only",
            "--no-compile-cache",
        ],
        _prepared_params_loader=lambda _path, _dtype, _cacheable: (),
    )

    assert model_atoms == [8]
    writer_features = writer_calls[0]["features"]
    assert set(writer_features) <= PROTENIX_GENERATED_OUTPUT_FEATURE_FIELDS
    actual_atoms = len(writer_features["output_atom_name"])
    assert writer_calls[0]["output"]["coordinate"].shape == (1, actual_atoms, 3)
    assert writer_calls[0]["output"]["atom_plddt"].shape == (1, actual_atoms)
