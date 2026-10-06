"""A backend that writes structures itself must skip the writer in trunk mode.

`stop_after="trunk"` compiles a graph that returns before the sampler, so the
prediction it hands back has no coordinates. Two backends write structures
themselves rather than delegating to their model's CLI, and both reached the
writer anyway: OpenFold3 had no branch at all (`IndexError` on an empty
coordinate shape) and ESMFold2 had one placed *below* the writer, which is the
same as not having it (`KeyError: 'sample_atom_coords'`).

Both are checked by running them: the native model is a stub, the writer
raises, and a trunk-only prediction must come back with its representations
and no samples. ESMFold2's run is
`test_esmfold2_output_lifetime.py::test_trunk_only_releases_inputs_before_representation_export`;
OpenFold3's is here. Backends that delegate to a model CLI do not call the
writer here -- the branch lives in that CLI instead.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import foldjax.backends
from foldjax.schema import PredictionRequest

#: Backends that call `write_prediction_outputs` themselves, each with the
#: behavioural trunk test that covers it.
_COVERED = {"openfold3.py", "esmfold2.py"}


def test_every_backend_that_writes_structures_has_a_trunk_test() -> None:
    """A third backend that writes its own structures needs its own trunk run."""
    directory = pathlib.Path(inspect.getfile(foldjax.backends)).parent
    writers = {
        path.name
        for path in directory.glob("*.py")
        if not path.name.startswith("_")
        and any(
            isinstance(node, ast.Attribute) and node.attr == "write_prediction_outputs"
            for node in ast.walk(ast.parse(path.read_text()))
        )
    }
    assert writers == _COVERED


def test_openfold3_trunk_only_never_reaches_the_writer(
    tmp_path: Path, monkeypatch
) -> None:
    from foldjax.backends.openfold3 import OpenFold3Backend
    from foldjax.models.openfold3.data import has_atomized_tokens

    features = {
        "token_mask": np.ones((1, 2), dtype=np.float32),
        "atom_mask": np.ones((1, 2), dtype=np.float32),
        "asym_id": np.zeros((1, 2), dtype=np.int64),
        "is_atomized": np.zeros((1, 2), dtype=np.int32),
        "msa_mask": np.ones((1, 1, 2), dtype=np.float32),
        "template_backbone_frame_mask": np.ones((1, 1, 2), dtype=np.float32),
        "template_pseudo_beta_mask": np.ones((1, 1, 2), dtype=np.float32),
    }
    single = np.arange(2 * 384, dtype=np.float32).reshape(1, 2, 384)
    configs: list[dict] = []

    def released_config(**overrides):
        configs.append(overrides)
        return SimpleNamespace(msa_depth=1024, num_samples=2, num_recycles=4)

    def fake_compile(config, table, *, n_chain=None, **compile_options):
        def compiled(key, batch, params, *, noise_mask=None):
            # The trunk graph's product: representations, and no coordinates.
            return SimpleNamespace(single=single)

        return compiled

    def writer(*args, **kwargs):
        pytest.fail("a trunk-only prediction reached the structure writer")

    modules = {
        "foldjax.models.openfold3.data": SimpleNamespace(
            pocket_sampling_config=lambda batch: None,
            featurize_query_with_metadata=lambda *args, **kwargs: (features, None),
            prepare_msa_cycle_features=lambda batch, depth, **kwargs: batch,
            collapse_identical_templates=lambda batch: batch,
            compact_zero_template_pair_features=lambda batch: batch,
            has_atomized_tokens=has_atomized_tokens,
            normalize_asym_ids=lambda batch: (batch, 1),
        ),
        "foldjax.models.openfold3.inference": SimpleNamespace(
            resolve_dtypes=lambda config: (None, None),
            cast_narrow_params=lambda params, dtype, confidence: params,
            released_config=released_config,
            compile_predict=fake_compile,
        ),
        "foldjax.models.openfold3.output": SimpleNamespace(
            write_prediction_outputs=writer
        ),
        "foldjax.models.openfold3.bridge.chemistry": SimpleNamespace(
            representative_atom_table=lambda: object()
        ),
        "foldjax.models.openfold3.bridge.checkpoint": SimpleNamespace(
            load_checkpoint=lambda path: {}
        ),
        "foldjax.models.openfold3.bridge.torch_mapping": SimpleNamespace(
            resolve_model_prefix=lambda state, prefix=None: "",
            prune_sample_diffusion_aliases=lambda state, *, prefix: 0,
            map_inference_params=lambda state, prefix: object(),
        ),
        "jax": SimpleNamespace(random=SimpleNamespace(key=lambda seed: seed)),
        "jax.numpy": SimpleNamespace(
            asarray=np.asarray,
            broadcast_to=np.broadcast_to,
        ),
    }
    monkeypatch.setattr(
        "foldjax.backends.openfold3.import_module", lambda name: modules[name]
    )
    monkeypatch.setattr(
        "foldjax.models.openfold3.streaming.compile_streamed_predict", fake_compile
    )
    (tmp_path / "job.json").write_text("{}")
    (tmp_path / "weights").mkdir()
    request = PredictionRequest(
        model="openfold3",
        input=tmp_path / "job.json",
        weights=tmp_path / "weights",
        output_dir=tmp_path / "out",
        seed=5,
        cache_dir=tmp_path / "cache",
        stop_after="trunk",
        representations=("single",),
    )

    result = OpenFold3Backend().predict(request)

    assert all(config["stop_after_trunk"] is True for config in configs)
    assert result.samples == ()
    assert result.representations is not None
    # The archive keeps the token and channel axes, not the batch of one.
    np.testing.assert_array_equal(result.representations["single"], single[0])
