"""Exercise AF3's actual stage control flow without loading its native runtime."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from foldjax.backends.alphafold3 import (
    AlphaFold3Backend,
    _predict_common_representations,
)
from foldjax.models import _representations
from foldjax.models.alphafold3.build import source_package
from foldjax.schema import PaddingConfig, PredictionRequest


def _native_forward(stop_after: str, wanted: tuple[str, ...]):
    source = source_package() / "model/model.py"
    tree = ast.parse(source.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Model"
    )
    forward = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__"
    )
    forward.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            forward,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    calls = {"trunk": 0, "diffusion": 0, "confidence": 0}
    target = np.arange(12, dtype=np.float32).reshape(3, 4)

    class Evoformer:
        def __init__(self, config, global_config):
            self.config = config

        def __call__(self, *, batch, prev, target_feat, key):
            calls["trunk"] += 1
            return {**prev, "single": prev["single"] + 1, "pair": prev["pair"] + 2}

    def fori_loop(start, end, body, state):
        for i in range(start, end):
            state = body(i, state)
        return state

    def confidence(*args):
        def run(**kwargs):
            calls["confidence"] += 1
            return {"confidence": np.ones(3)}

        return run

    def sample(*args, **kwargs):
        calls["diffusion"] += 1
        return {"atom_positions": np.zeros((1, 3, 24, 3))}

    namespace = {
        "jnp": np,
        "jax": SimpleNamespace(
            random=SimpleNamespace(split=lambda key: (key + 1, key + 2))
        ),
        "hk": SimpleNamespace(
            next_rng_key=lambda: 0, running_init=lambda: False, fori_loop=fori_loop
        ),
        "feat_batch": SimpleNamespace(
            Batch=SimpleNamespace(
                from_data_dict=lambda _: SimpleNamespace(
                    num_res=3,
                    token_features=SimpleNamespace(
                        mask=np.ones(3), asym_id=np.zeros(3)
                    ),
                    pseudo_beta_info=SimpleNamespace(token_atoms_to_pseudo_beta=None),
                )
            )
        ),
        "evoformer_network": SimpleNamespace(Evoformer=Evoformer),
        "create_target_feat_embedding": lambda **kwargs: target,
        "confidence_head": SimpleNamespace(ConfidenceHead=confidence),
        "mapping": SimpleNamespace(sharded_map=lambda fn, in_axes: lambda x: fn(x[0])),
        "distogram_head": SimpleNamespace(
            DistogramHead=lambda *args: lambda *a, **k: {}
        ),
    }
    exec(compile(module, str(source), "exec"), namespace)
    config = SimpleNamespace(
        evoformer=SimpleNamespace(pair_channel=2, seq_channel=4),
        foldjax_return_representations=wanted,
        foldjax_stop_after=stop_after,
        num_recycles=3,
        heads=SimpleNamespace(
            diffusion=SimpleNamespace(eval=None), confidence=None, distogram=None
        ),
        return_distogram=False,
        return_embeddings=False,
    )
    self = SimpleNamespace(config=config, global_config=None, _sample_diffusion=sample)
    return namespace["__call__"](self, {}, key=0), calls


def test_native_inputs_stop_matches_full_inputs_and_skips_trunk_and_heads():
    full, _ = _native_forward("full", ("single_inputs", "single", "pair"))
    early, calls = _native_forward("inputs", ("single_inputs",))
    np.testing.assert_array_equal(
        early["representations"]["single_inputs"],
        full["representations"]["single_inputs"],
    )
    assert calls == {"trunk": 0, "diffusion": 0, "confidence": 0}
    assert set(early) == {"representations"}


def test_native_trunk_stop_matches_full_trunk_and_skips_heads():
    full, _ = _native_forward("full", ("single", "pair"))
    early, calls = _native_forward("trunk", ("single", "pair"))
    for name in ("single", "pair"):
        np.testing.assert_array_equal(
            early["representations"][name], full["representations"][name]
        )
    assert calls == {"trunk": 4, "diffusion": 0, "confidence": 0}


def test_native_default_does_not_return_additional_arrays():
    baseline, _ = _native_forward("full", ())
    exposed, _ = _native_forward("full", ("single_inputs", "single", "pair"))
    assert "representations" not in baseline
    for name in baseline:
        if not isinstance(baseline[name], dict):
            np.testing.assert_array_equal(baseline[name], exposed[name])


@pytest.mark.parametrize(
    "stop_after,wanted", [("inputs", ("single_inputs",)), ("trunk", ("single", "pair"))]
)
def test_common_archive_crops_padding_and_skips_structure_extraction(
    tmp_path: Path, stop_after, wanted
):
    (tmp_path / "input.json").write_text("{}")
    request = PredictionRequest(
        model="alphafold3",
        input=tmp_path / "input.json",
        output_dir=tmp_path,
        stop_after=stop_after,
        representations=wanted,
        padding=PaddingConfig(tokens=8),
    )
    seen = {}
    arrays = {
        "single_inputs": np.ones((8, 4)),
        "single": np.ones((8, 6)),
        "pair": np.ones((8, 8, 2)),
    }

    def run(batch, key):
        seen.update(batch)
        return {"representations": arrays}

    result = _predict_common_representations(
        SimpleNamespace(rng_seeds=(0,), name="job"),
        [{"seq_length": np.array(3)}],
        SimpleNamespace(run_inference=run),
        SimpleNamespace(),
        request=request,
        wanted=wanted,
        buckets=(8,),
    )
    assert result == ()
    assert "__foldjax_prefix_stable_diffusion_noise" in seen
    with np.load(tmp_path / _representations.ARCHIVE_NAME) as archive:
        assert set(archive.files) == set(wanted)
        for name in wanted:
            assert archive[name].shape[:2] == (
                (3, 3) if name == "pair" else (3, arrays[name].shape[-1])
            )


def test_external_runtime_cannot_claim_common_representations(tmp_path: Path):
    (tmp_path / "input.json").write_text("{}")
    request = PredictionRequest(
        model="alphafold3",
        input=tmp_path / "input.json",
        output_dir=tmp_path,
        representations=("single",),
        options={"source": str(tmp_path)},
    )
    with pytest.raises(ValueError, match="managed runtime"):
        AlphaFold3Backend().validate_request(request)


def test_cache_profile_canonicalizes_inputs_all_and_separates_stop_points(tmp_path):
    import dataclasses

    (tmp_path / "input.json").write_text("{}")
    base = PredictionRequest(
        model="alphafold3",
        input=tmp_path / "input.json",
        output_dir=tmp_path,
        stop_after="inputs",
        representations=("single_inputs",),
    )
    backend = AlphaFold3Backend()
    assert backend.cache_profile(base) == backend.cache_profile(
        dataclasses.replace(base, representations=("all",))
    )
    assert backend.cache_profile(base) != backend.cache_profile(
        dataclasses.replace(base, stop_after="full")
    )
