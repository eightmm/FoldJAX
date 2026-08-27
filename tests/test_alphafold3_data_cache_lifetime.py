from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace


def _runtime_modules():
    from foldjax.models.alphafold3 import build

    build.register_runtime()
    from alphafold3.constants import chemical_components
    from alphafold3.data import pipeline

    return chemical_components, pipeline


def test_ccd_loader_cache_has_a_fixed_process_bound() -> None:
    chemical_components, _pipeline = _runtime_modules()

    assert chemical_components._load_ccd_pickle_cached.cache_parameters() == {
        "maxsize": 1,
        "typed": False,
    }


def test_replaced_ccd_cache_keeps_only_active_callers_alive(
    tmp_path, monkeypatch
) -> None:
    chemical_components, _pipeline = _runtime_modules()

    class WeakDictionary(dict):
        pass

    loads = 0

    def fake_load(_handle):
        nonlocal loads
        value = WeakDictionary(ALA={"name": [str(loads)]})
        loads += 1
        return value

    first_path = tmp_path / "first.pickle"
    second_path = tmp_path / "second.pickle"
    first_path.write_bytes(b"first")
    second_path.write_bytes(b"second")
    cache = chemical_components._load_ccd_pickle_cached
    cache.cache_clear()
    monkeypatch.setattr(chemical_components.safe_pickle, "load", fake_load)
    try:
        first = chemical_components.Ccd(ccd_pickle_path=first_path)
        first_base = weakref.ref(first._dict)
        reused = chemical_components.Ccd(ccd_pickle_path=first_path)
        assert reused._dict is first._dict
        del reused

        second = chemical_components.Ccd(ccd_pickle_path=second_path)
        assert first["ALA"] == {"name": ["0"]}
        assert second["ALA"] == {"name": ["1"]}
        assert cache.cache_info().currsize == 1

        del first
        gc.collect()
        assert first_base() is None
    finally:
        cache.cache_clear()


def test_user_ccd_does_not_mutate_the_process_global_base(monkeypatch) -> None:
    chemical_components, _pipeline = _runtime_modules()
    base = {"ALA": {"name": ["base"]}}
    user = {"ALA": {"name": ["override"]}, "LIG": {"name": ["ligand"]}}
    monkeypatch.setattr(
        chemical_components,
        "_load_ccd_pickle_cached",
        lambda _path: base,
    )
    monkeypatch.setattr(
        chemical_components.cif_dict,
        "parse_multi_data_cif",
        lambda _text: {
            name: SimpleNamespace(to_dict=lambda value=value: value)
            for name, value in user.items()
        },
    )

    custom = chemical_components.Ccd(user_ccd="data_LIG")
    default = chemical_components.Ccd()

    assert custom["ALA"] == user["ALA"]
    assert custom["LIG"] == user["LIG"]
    assert default["ALA"] == {"name": ["base"]}
    assert "LIG" not in default
    assert base == {"ALA": {"name": ["base"]}}
    assert default._dict is base
    assert custom._dict is not base


def test_component_info_cache_is_bounded_by_the_ccd_lifetime(monkeypatch) -> None:
    chemical_components, _pipeline = _runtime_modules()
    base = {f"LIG{index}": {} for index in range(130)}
    monkeypatch.setattr(
        chemical_components,
        "_load_ccd_pickle_cached",
        lambda _path: base,
    )
    monkeypatch.setattr(
        chemical_components,
        "mmcif_to_info",
        lambda component: component,
    )
    ccd = chemical_components.Ccd()

    for name in base:
        assert chemical_components.component_name_to_info(ccd, name) is base[name]

    assert len(ccd._component_info_cache) == 128
    assert "LIG0" not in ccd._component_info_cache
    assert "LIG129" in ccd._component_info_cache

    ccd_ref = weakref.ref(ccd)
    del ccd
    gc.collect()
    assert ccd_ref() is None


def test_pipeline_search_cache_is_reused_only_within_one_job(monkeypatch) -> None:
    _chemical_components, pipeline = _runtime_modules()
    calls = {"templates": 0, "protein": 0, "rna": 0}

    class Result:
        pass

    def templates(*_args, **_kwargs):
        calls["templates"] += 1
        return Result()

    def protein(*_args, **_kwargs):
        calls["protein"] += 1
        return Result()

    def rna(*_args, **_kwargs):
        calls["rna"] += 1
        return Result()

    monkeypatch.setattr(pipeline, "_get_protein_templates", templates)
    monkeypatch.setattr(pipeline, "_get_protein_msa_and_templates", protein)
    monkeypatch.setattr(pipeline, "_get_rna_msa", rna)

    caches = pipeline._PipelineRunCaches.create()
    template = caches.protein_templates("SEQ", "A3M", True, "cfg", "pdb")
    assert caches.protein_templates("SEQ", "A3M", True, "cfg", "pdb") is template
    protein_result = caches.protein_msa_and_templates("SEQ", False, "configs")
    assert (
        caches.protein_msa_and_templates("SEQ", False, "configs")
        is protein_result
    )
    rna_result = caches.rna_msa("RNA", "configs")
    assert caches.rna_msa("RNA", "configs") is rna_result
    assert calls == {"templates": 1, "protein": 1, "rna": 1}

    template_ref = weakref.ref(template)
    del template, protein_result, rna_result, caches
    gc.collect()
    assert template_ref() is None

    next_job = pipeline._PipelineRunCaches.create()
    next_job.protein_templates("SEQ", "A3M", True, "cfg", "pdb")
    assert calls["templates"] == 2


def test_data_pipeline_passes_one_cache_bundle_per_input_job(monkeypatch) -> None:
    _chemical_components, pipeline = _runtime_modules()
    from alphafold3.common import folding_input

    seen = []

    def process_protein_chain(_self, chain, *, run_caches=None):
        seen.append(run_caches)
        return chain

    monkeypatch.setattr(
        pipeline.DataPipeline,
        "process_protein_chain",
        process_protein_chain,
    )
    chains = tuple(
        folding_input.ProteinChain(
            id=chain_id,
            sequence="AAAA",
            ptms=(),
            paired_msa="",
            unpaired_msa="",
            templates=(),
        )
        for chain_id in ("A", "B")
    )
    fold_input = folding_input.Input(
        name="homomer",
        chains=chains,
        rng_seeds=(0,),
    )
    data_pipeline = object.__new__(pipeline.DataPipeline)

    data_pipeline.process(fold_input)
    data_pipeline.process(fold_input)

    assert seen[0] is seen[1]
    assert seen[2] is seen[3]
    assert seen[0] is not seen[2]
