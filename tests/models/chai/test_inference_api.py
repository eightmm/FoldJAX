from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import foldjax.models.chai.inference as inference
from foldjax.models.chai.bridge.bundle_io import NativeBundle
from foldjax.models.chai.bridge.conformer_io import ConformerData
from foldjax.models.chai.data.msa import RESIDUE_ORDER, expected_aligned_pqt_basename


def _conformer() -> ConformerData:
    names = ("N", "CA", "C", "CB")
    return ConformerData(
        position=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            np.float32,
        ),
        element=np.asarray([7, 6, 6, 6], np.int32),
        charge=np.zeros(4, np.int32),
        atom_names=names,
        bonds=((0, 1), (1, 2), (1, 3)),
        symmetries=np.arange(4, dtype=np.int64)[:, None],
    )


def _assets() -> inference.InferenceAssets:
    return inference.InferenceAssets(
        bundle=NativeBundle(manifest={}, components={}),
        conformers={"ALA": _conformer()},
    )


def test_model_padded_inputs_skip_raw_restraint_controls() -> None:
    prepared = SimpleNamespace(
        padded_inputs={
            "token_exists_mask": np.asarray([[True, False]]),
            "contact_constraints": [{"distance_threshold": 8.0}],
            "docking_constraints": [None],
            "pocket_constraints": [None],
        }
    )

    values = inference._model_padded_inputs(prepared)

    assert set(values) == {"token_exists_mask"}
    np.testing.assert_array_equal(values["token_exists_mask"], [[True, False]])


def test_model_padded_inputs_reject_other_non_arrays() -> None:
    prepared = SimpleNamespace(padded_inputs={"unexpected": [None]})

    with pytest.raises(TypeError, match="unexpected.*NumPy array"):
        inference._model_padded_inputs(prepared)


def test_model_padded_inputs_keep_raw_msa_on_host() -> None:
    prepared = SimpleNamespace(
        padded_inputs={
            "token_exists_mask": np.ones((1, 4), dtype=np.bool_),
            "msa_tokens": np.zeros((1, 16384, 4), dtype=np.uint8),
            "main_msa_tokens": np.zeros((1, 16384, 4), dtype=np.uint8),
            "msa_mask": np.ones((1, 16384, 4), dtype=np.bool_),
        }
    )

    values = inference._model_padded_inputs(prepared)

    assert set(values) == {"token_exists_mask", "msa_mask"}
    assert "msa_tokens" not in values
    assert "main_msa_tokens" not in values


def test_staged_trunk_preserves_outer_msa_pair_residual(monkeypatch) -> None:
    """The memory-bounded production path must match the exported trunk graph."""
    block = SimpleNamespace(
        outer_product_mean=object(),
        msa_transition=object(),
        weighted_averaging=object(),
        pair=object(),
    )
    params = SimpleNamespace(
        msa=SimpleNamespace(linear_s2m_weight=object(), blocks=(block,)),
        pairformer_blocks=object(),
    )
    monkeypatch.setattr(
        inference,
        "_compiled_recycle_template",
        lambda *args, **kwargs: (args[0], args[3] + 3),
    )
    monkeypatch.setattr(
        inference, "_compiled_msa_embedding", lambda *args: np.zeros((1, 1, 1, 1))
    )
    monkeypatch.setattr(inference, "_compiled_outer_product_mean", lambda *args: 7)
    monkeypatch.setattr(inference, "_compiled_msa_transition", lambda *args: 0)
    monkeypatch.setattr(inference, "_compiled_msa_weighted", lambda *args: 0)
    monkeypatch.setattr(inference, "_compiled_msa_pair", lambda pair, *args: pair + 11)
    monkeypatch.setattr(
        inference,
        "_compiled_pairformer_stack",
        lambda blocks, single, pair, *args: (single, pair),
    )

    single, pair = inference._run_staged_trunk(
        np.ones((1, 1, 1)),
        np.ones((1, 1, 1, 1)),
        np.ones((1, 1, 1)),
        np.full((1, 1, 1, 1), 2),
        np.zeros((1, 1, 1, 1)),
        np.ones((1, 1, 1), dtype=bool),
        np.zeros((1, 1, 1, 1, 1)),
        np.ones((1, 1, 1, 1), dtype=bool),
        np.ones((1, 1), dtype=bool),
        np.ones((1, 1, 1), dtype=bool),
        params,
        skip_templates=False,
    )

    np.testing.assert_array_equal(single, np.ones((1, 1, 1)))
    # recycle/template pair=5; MSA inner result=(5+7)+11=23; outer result=5+23.
    np.testing.assert_array_equal(pair, np.full((1, 1, 1, 1), 28))


def test_prepare_inference_builds_real_padded_feature_contract(tmp_path: Path) -> None:
    fasta = tmp_path / "query.fasta"
    fasta.write_text(">protein|name=target\nAA\n", encoding="utf-8")

    prepared = inference.prepare_inference_from_assets(
        fasta, _assets(), config=inference.InferenceConfig(use_esm_embeddings=False)
    )

    assert prepared.model_size == 256
    assert prepared.structure_context.num_tokens == 2
    assert prepared.structure_context.num_atoms == 8
    assert len(prepared.features) == 32
    assert prepared.padded_inputs["token_exists_mask"].shape == (1, 256)
    assert prepared.padded_inputs["atom_exists_mask"].shape == (1, 5888)
    assert prepared.padded_inputs["block_atom_pair_mask"].shape == (1, 184, 32, 128)
    assert prepared.padded_inputs["msa_tokens"].shape == (1, 16384, 256)
    assert prepared.padded_inputs["template_restype"].shape == (1, 4, 256)
    np.testing.assert_array_equal(
        prepared.padded_inputs["token_residue_type"][0, :3], [0, 0, 32]
    )
    assert prepared.features["ResidueType"].shape == (1, 256)

    query = prepared.padded_inputs["block_atom_pair_q_idces"]
    key = prepared.padded_inputs["block_atom_pair_kv_idces"]
    reference_space = prepared.padded_inputs["atom_ref_space_uid"]
    same_reference_space = (
        reference_space[:, query][..., :, None]
        == reference_space[:, key][..., None, :]
    )
    mask = prepared.padded_inputs["block_atom_pair_mask"]
    assert np.all(same_reference_space[mask])
    assert np.any(~same_reference_space)


def test_reference_only_glycan_atoms_remain_in_block_pair_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leaving atoms are reference features even when absent from output coords."""
    fasta = tmp_path / "glycan.fasta"
    fasta.write_text(
        ">glycan|name=sugar\nNAG(4-1 NAG)\n", encoding="utf-8"
    )
    names = ("C1", "O1", "C2", "O4")
    conformer = ConformerData(
        position=np.asarray(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 1.4, 0.0], [0.0, 2.8, 0.0]],
            np.float32,
        ),
        element=np.asarray([6, 8, 6, 8], np.int32),
        charge=np.zeros(4, np.int32),
        atom_names=names,
        bonds=((0, 1), (0, 2), (2, 3)),
        symmetries=np.arange(4, dtype=np.int64)[:, None],
    )
    assets = inference.InferenceAssets(
        bundle=NativeBundle(manifest={}, components={}), conformers={"NAG": conformer}
    )
    prepared = inference.prepare_inference_from_assets(
        fasta, assets, config=inference.InferenceConfig(use_esm_embeddings=False)
    )
    values = prepared.padded_inputs
    reference_only = values["atom_ref_mask"] & ~values["atom_exists_mask"]
    atom = int(np.flatnonzero(reference_only[0])[0])
    query = values["block_atom_pair_q_idces"]
    key = values["block_atom_pair_kv_idces"]
    locations = np.argwhere((query[..., :, None] == atom) & (key[..., None, :] == atom))

    assert locations.size
    assert any(
        values["block_atom_pair_mask"][0, block, query_offset, key_offset]
        for block, query_offset, key_offset in locations
    )

    captured: dict[str, np.ndarray] = {}
    embedded = {
        "TOKEN": np.zeros((1, 256, 1), np.float32),
        "TOKEN_PAIR": np.zeros((1, 256, 256, 2), np.float32),
        "ATOM": np.zeros((1, 5888, 2), np.float32),
        "ATOM_PAIR": np.zeros((1, 184, 32, 128, 2), np.float32),
        "MSA": np.zeros((1, 1, 256, 1), np.float32),
        "TEMPLATES": np.zeros((1, 4, 256, 256, 1), np.float32),
    }
    monkeypatch.setattr(inference, "_compiled_embed_features", lambda *args: embedded)

    def token_embedder(inputs, *args, **kwargs):
        del args, kwargs
        captured["atom_single_mask"] = np.asarray(inputs["atom_single_mask"])
        return (
            np.zeros((1, 256, 1), np.float32),
            np.zeros((1, 256, 1), np.float32),
            np.zeros((1, 256, 256, 1), np.float32),
        )

    monkeypatch.setattr(inference, "_compiled_token_embedder", token_embedder)
    components = SimpleNamespace(
        feature_embedding={},
        bond_loss_input_proj={"weight": np.zeros((2, 1), np.float32)},
        token_embedder={},
    )
    diffusion_inputs, *_ = inference._embed_and_initialize(prepared, components)

    assert not captured["atom_single_mask"][0, atom]
    assert not np.asarray(diffusion_inputs["atom_single_mask"])[0, atom]


def test_esm_requires_native_model_or_precomputed_archive() -> None:
    config = inference.InferenceConfig(
        use_esm_embeddings=True,
        esm_model_path=None,
    )
    with pytest.raises(ValueError, match="esm_embeddings_path or esm_model_path"):
        config.validate()


def test_template_server_requires_msa_server() -> None:
    with pytest.raises(ValueError, match="requires use_msa_server"):
        inference.InferenceConfig(use_templates_server=True).validate()


def test_native_esm_template_and_restraint_paths_are_supported() -> None:
    inference.InferenceConfig(
        use_esm_embeddings=True,
        esm_embeddings_path=Path("esm.npz"),
        template_hits_path=Path("templates.npz"),
        constraint_path=Path("restraints.csv"),
    ).validate()


def test_native_esm_model_generation_reaches_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fasta = tmp_path / "query.fasta"
    fasta.write_text(">protein|name=target\nAA\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def generate(inputs: object, **kwargs: object) -> dict[str, np.ndarray]:
        captured.update(inputs=inputs, **kwargs)
        return {"AA": np.ones((2, 2560), np.float32)}

    monkeypatch.setattr(
        inference, "generate_native_esm_embeddings_for_inputs", generate
    )
    prepared = inference.prepare_inference_from_assets(
        fasta,
        _assets(),
        config=inference.InferenceConfig(
            esm_model_path=tmp_path / "esm2-native",
            esm_cache_directory=tmp_path / "esm-cache",
        ),
    )

    np.testing.assert_array_equal(
        prepared.padded_inputs["esm_embeddings"][0, :2], 1.0
    )
    assert captured["model_path"] == tmp_path / "esm2-native"
    assert captured["cache_dir"] == tmp_path / "esm-cache"


def test_local_aligned_parquet_reaches_joined_and_profile_features(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "query.fasta"
    fasta.write_text(">protein|name=target\nAA\n", encoding="utf-8")
    msa_dir = tmp_path / "msas"
    msa_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "sequence": ["AA", "A-"],
                "source_database": ["query", "uniref90"],
                "pairing_key": ["", ""],
                "comment": ["query", "hit"],
            }
        ),
        msa_dir / expected_aligned_pqt_basename("AA"),
    )

    prepared = inference.prepare_inference_from_assets(
        fasta,
        _assets(),
        config=inference.InferenceConfig(
            msa_directory=msa_dir, use_esm_embeddings=False
        ),
    )

    expected = [[RESIDUE_ORDER["A"], RESIDUE_ORDER["A"]], [RESIDUE_ORDER["A"], 31]]
    np.testing.assert_array_equal(
        prepared.padded_inputs["msa_tokens"][0, :2, :2], expected
    )
    np.testing.assert_array_equal(
        prepared.padded_inputs["main_msa_tokens"][0, :2, :2], expected
    )


def test_injected_search_pipeline_reaches_inference_without_torch(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "query.fasta"
    fasta.write_text(">protein|name=target\nAA\n", encoding="utf-8")
    paired = tmp_path / "pairing.a3m"
    unpaired = tmp_path / "non_pairing.a3m"
    paired.write_text(">query\nAA\n", encoding="utf-8")
    unpaired.write_text(">query\nAA\n>hit\nA-\n", encoding="utf-8")

    class Pipeline:
        def search(self, sequences: list[str]) -> list[dict[str, str]]:
            assert sequences == ["AA"]
            return [
                {
                    "pairedMsaPath": str(paired),
                    "unpairedMsaPath": str(unpaired),
                }
            ]

    prepared = inference.prepare_inference_from_assets(
        fasta,
        _assets(),
        config=inference.InferenceConfig(use_esm_embeddings=False),
        msa_search_pipeline=Pipeline(),
    )

    assert prepared.padded_inputs["msa_mask"][0, :2, :2].all()


def test_compilation_cache_is_explicit_and_persistent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    updates: list[tuple[str, object]] = []
    monkeypatch.setattr(
        inference.jax.config,
        "update",
        lambda name, value: updates.append((name, value)),
    )
    cache = tmp_path / "nested" / "compile-cache"

    inference.InferenceConfig(compilation_cache_dir=cache).configure_compilation_cache()

    assert cache.is_dir()
    assert updates == [
        ("jax_compilation_cache_dir", str(cache.resolve())),
        ("jax_persistent_cache_min_compile_time_secs", 1.0),
    ]


def test_diffusion_translations_broadcast_over_atoms_for_multiple_samples() -> None:
    translations = inference._random_translations(
        inference.jax.random.PRNGKey(7), 5
    )

    assert translations.shape == (5, 1, 3)
    np.testing.assert_array_equal(
        translations,
        inference._random_translations(inference.jax.random.PRNGKey(7), 5),
    )


def test_ligand_modality_reaches_padded_feature_contract(tmp_path: Path) -> None:
    fasta = tmp_path / "ligand.fasta"
    fasta.write_text(">ligand|name=lig\nCCO\n", encoding="utf-8")

    prepared = inference.prepare_inference_from_assets(
        fasta, _assets(), config=inference.InferenceConfig(use_esm_embeddings=False)
    )

    assert prepared.structure_context.num_tokens == 3
    assert prepared.features["ResidueType"].shape == (1, 256)


def test_run_inference_maps_components_and_calls_explicit_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fasta = tmp_path / "query.fasta"
    fasta.write_text(">protein|name=target\nA\n", encoding="utf-8")
    fake_components = object()
    monkeypatch.setattr(inference, "load_inference_assets", lambda *a, **k: _assets())
    monkeypatch.setattr(inference, "_random_seed", lambda: 12345)
    monkeypatch.setattr(
        inference, "map_model_components", lambda bundle: fake_components
    )
    received: dict[str, Any] = {}

    def backend(
        prepared: inference.PreparedInference, components: Any, config: Any
    ) -> str:
        received.update(prepared=prepared, components=components, config=config)
        return "prediction"

    result = inference.run_inference(
        fasta,
        output_dir=tmp_path / "outputs",
        bundle_path="bundle",
        conformer_path="conformers.npz",
        config=inference.InferenceConfig(use_esm_embeddings=False),
        backend=backend,
    )
    assert result == "prediction"
    assert received["components"] is fake_components
    assert received["prepared"].model_size == 256
    assert received["prepared"].realized_seed == 12345
    assert received["config"].seed == 12345


def test_run_inference_uses_native_backend_and_output_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fasta = tmp_path / "query.fasta"
    fasta.write_text(">protein|name=target\nA\n", encoding="utf-8")
    monkeypatch.setattr(inference, "load_inference_assets", lambda *a, **k: _assets())
    components = object()
    prediction = object()
    candidates = object()
    monkeypatch.setattr(inference, "map_model_components", lambda bundle: components)
    monkeypatch.setattr(
        inference,
        "execute_prepared_inference",
        lambda prepared, received, config: prediction,
    )
    monkeypatch.setattr(
        inference,
        "write_prediction_outputs",
        lambda output_dir, predictions: candidates,
    )

    result = inference.run_inference(
        fasta,
        output_dir=tmp_path / "outputs",
        bundle_path="bundle",
        conformer_path="conformers.npz",
        config=inference.InferenceConfig(use_esm_embeddings=False),
    )

    assert result is candidates


def test_run_inference_increments_seed_for_each_trunk_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fasta = tmp_path / "query.fasta"
    fasta.write_text(">protein|name=target\nA\n", encoding="utf-8")
    monkeypatch.setattr(inference, "load_inference_assets", lambda *a, **k: _assets())
    monkeypatch.setattr(inference, "map_model_components", lambda _bundle: object())
    predictions: list[object] = []
    received_seeds: list[int | None] = []

    def execute(_prepared: object, _components: object, config: object) -> object:
        prediction = object()
        predictions.append(prediction)
        received_seeds.append(config.seed)
        return prediction

    candidates = object()
    written: list[object] = []
    monkeypatch.setattr(inference, "execute_prepared_inference", execute)
    monkeypatch.setattr(
        inference,
        "write_prediction_outputs",
        lambda _output_dir, trunk_predictions: (
            written.extend(trunk_predictions) or candidates
        ),
    )

    result = inference.run_inference(
        fasta,
        output_dir=tmp_path / "outputs",
        bundle_path="bundle",
        conformer_path="conformers.npz",
        config=inference.InferenceConfig(
            num_trunk_samples=3, seed=41, use_esm_embeddings=False
        ),
    )

    assert result is candidates
    assert received_seeds == [41, 42, 43]
    assert written == predictions


def test_trunk_sample_seed_is_realized_when_seed_is_unspecified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fasta = tmp_path / "query.fasta"
    fasta.write_text(">protein|name=target\nA\n", encoding="utf-8")
    monkeypatch.setattr(inference, "load_inference_assets", lambda *a, **k: _assets())
    monkeypatch.setattr(inference, "_random_seed", lambda: 8765)
    monkeypatch.setattr(inference, "map_model_components", lambda _bundle: object())
    seeds: list[int | None] = []
    monkeypatch.setattr(
        inference,
        "execute_prepared_inference",
        lambda _prepared, _components, config: seeds.append(config.seed) or object(),
    )
    monkeypatch.setattr(
        inference, "write_prediction_outputs", lambda *_args, **_kwargs: object()
    )

    inference.run_inference(
        fasta,
        output_dir=tmp_path / "outputs",
        bundle_path="bundle",
        conformer_path="conformers.npz",
        config=inference.InferenceConfig(
            num_trunk_samples=2, use_esm_embeddings=False
        ),
    )

    assert seeds == [8765, 8766]


def test_run_inference_builds_content_addressed_remote_msa_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = object()
    captured: dict[str, Any] = {}

    def remote(url: str, *, version: str) -> tuple[str, str]:
        return url, version

    def search_pipeline(cache_dir: Path, backend: object) -> object:
        captured.update(cache_dir=cache_dir, backend=backend)
        return pipeline

    def prepare(*_args: object, **kwargs: object) -> tuple[object, Any]:
        captured["pipeline"] = kwargs["msa_search_pipeline"]
        return SimpleNamespace(realized_seed=71), _assets()

    monkeypatch.setattr(inference, "RemoteMMseqs2Backend", remote)
    monkeypatch.setattr(inference, "MsaSearchPipeline", search_pipeline)
    monkeypatch.setattr(inference, "prepare_inference", prepare)
    monkeypatch.setattr(inference, "map_model_components", lambda _bundle: object())

    result = inference.run_inference(
        tmp_path / "query.fasta",
        output_dir=tmp_path / "outputs",
        bundle_path="bundle",
        conformer_path="conformers",
        config=inference.InferenceConfig(use_msa_server=True),
        backend=lambda *_args: "prediction",
    )

    assert result == "prediction"
    assert captured == {
        "cache_dir": tmp_path / "outputs/msas",
        "backend": ("https://api.colabfold.com", "colabfold-public-api-v1"),
        "pipeline": pipeline,
    }


def test_run_inference_enables_templates_on_remote_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def remote(url: str, *, version: str, use_templates: bool) -> object:
        captured.update(url=url, version=version, use_templates=use_templates)
        return object()

    monkeypatch.setattr(inference, "RemoteMMseqs2Backend", remote)
    monkeypatch.setattr(
        inference,
        "MsaSearchPipeline",
        lambda cache_dir, backend: captured.update(cache_dir=cache_dir) or object(),
    )
    monkeypatch.setattr(
        inference,
        "prepare_inference",
        lambda *_args, **_kwargs: (SimpleNamespace(realized_seed=72), _assets()),
    )
    monkeypatch.setattr(inference, "map_model_components", lambda _bundle: object())

    inference.run_inference(
        tmp_path / "query.fasta",
        output_dir=tmp_path / "outputs",
        bundle_path="bundle",
        conformer_path="conformers",
        config=inference.InferenceConfig(
            use_msa_server=True,
            use_templates_server=True,
        ),
        backend=lambda *_args: "prediction",
    )

    assert captured == {
        "url": "https://api.colabfold.com",
        "version": "colabfold-public-api-v1",
        "use_templates": True,
        "cache_dir": tmp_path / "outputs/msas",
    }
