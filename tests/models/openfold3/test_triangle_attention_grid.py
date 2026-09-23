"""`triangle_attention_grid` on OpenFold3: the scope, the identity, the refusal.

The option selects the 2-D triangle-attention algorithm -- the released ring,
or the streamed gather (`_cp_attention.gather_triangle_attention_2d_from_pair`)
-- and reaches both 2-D entries through a scope, because no signature between
the adapter and them has a use for it. Three things make that safe here, and
each is asserted below:

* the adapter enters the scope around the whole native call, and resolves
  what `gather` runs locally on the host before featurization, so a GPU
  process without cuEquivariance is refused rather than downgraded;
* the value partitions `_PredictGraphIdentity`, the in-process pool's key, so
  a `gather` call can never be handed the ring program an earlier call in the
  same process compiled (the scope is a ContextVar no `jax.jit` key reads);
* `ring` is the default of that field, so every identity built before the
  option existed is unchanged.

The option surface -- vocabulary, refusal off the grid, the cache namespace --
is in `tests/test_cp_option_surface.py`; the arithmetic of the two entries in
`tests/models/test_cp_gather_triangle.py`.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.backends import openfold3 as openfold3_backend
from foldjax.backends.openfold3 import OpenFold3Backend
from foldjax.models import _cp_attention
from foldjax.models._cp_attention import (
    triangle_attention_grid,
    triangle_attention_grid_scope,
)
from foldjax.models.openfold3 import inference, streaming
from foldjax.schema import PredictionRequest
from tests.models.openfold3.test_deterministic_ops import _identity, _RecordedPool
from tests.models.openfold3.test_stable_compile import _config, _table


def test_the_identity_defaults_to_the_ring_and_partitions_on_gather() -> None:
    assert _identity() == _identity(triangle_attention_grid="ring")
    assert _identity().triangle_attention_grid == "ring"
    assert _identity(triangle_attention_grid="gather") != _identity()


def test_compile_predict_reads_the_scope_into_the_identity(monkeypatch) -> None:
    pool = _RecordedPool()
    monkeypatch.setattr(inference, "_compiled_predict", pool)
    args = (jax.random.key(0), {"x": jnp.ones((4,), dtype=jnp.float32)}, ())

    inference.compile_predict(_config(), _table())(*args)
    with triangle_attention_grid_scope("gather"):
        inference.compile_predict(_config(), _table())(*args)
    with triangle_attention_grid_scope("ring"):
        inference.compile_predict(_config(), _table())(*args)

    grids = [identity.triangle_attention_grid for identity in pool.identities]
    assert grids == ["ring", "gather", "ring"]
    assert pool.identities[0] == pool.identities[2] != pool.identities[1]


def test_compile_streamed_predict_reads_the_scope_into_the_identity(
    monkeypatch,
) -> None:
    pool = _RecordedPool()
    monkeypatch.setattr(streaming, "_compiled_streams", pool)
    config = _config(msa_depth=8)
    args = (jax.random.key(0), {"x": jnp.ones((4,), dtype=jnp.float32)}, ())

    streaming.compile_streamed_predict(config, _table())(*args)
    with triangle_attention_grid_scope("gather"):
        streaming.compile_streamed_predict(config, _table())(*args)

    grids = [identity.triangle_attention_grid for identity in pool.identities]
    assert grids == ["ring", "gather"]


def _request(tmp_path: Path, **options: Any) -> PredictionRequest:
    input_path = tmp_path / "job.yaml"
    input_path.write_text("version: 1\n", encoding="utf-8")
    weights = tmp_path / "weights"
    weights.mkdir(exist_ok=True)
    return PredictionRequest(
        model="openfold3",
        input=input_path,
        weights=weights,
        options={"cp_devices": 4, "pair_chunk_size": 0, **options},
    )


class _NativeCallReachedError(Exception):
    """Raised by the compiled-predict double: everything before it ran."""


def _drive_to_the_native_call(monkeypatch, seen: list[str]) -> list[str]:
    """Replace everything `predict` imports with doubles that record the scope.

    Featurization and the model modules are stand-ins; what runs for real is
    the adapter's own control flow, from option handling to the `with` block
    around the compiled call, whose scope the double reads.
    """

    featurized: list[str] = []
    features = {
        "token_mask": np.ones((1, 4), dtype=np.float32),
        "atom_mask": np.ones((1, 4), dtype=np.float32),
    }

    def features_chemistry_and_metadata(*_args: Any, **_kwargs: Any):
        featurized.append("featurized")
        return features, None, {}

    def compile_predict(*_args: Any, **_kwargs: Any):
        def run(*_call_args: Any, **_call_kwargs: Any):
            seen.append(triangle_attention_grid())
            raise _NativeCallReachedError

        return run

    config = SimpleNamespace(msa_depth=None, num_recycles=1, cp_shards=4)
    modules = {
        "foldjax.models.openfold3.data": SimpleNamespace(
            prepare_msa_cycle_features=lambda batch, *a, **k: batch,
            collapse_identical_templates=lambda batch: batch,
            normalize_asym_ids=lambda batch: (batch, None),
            compact_zero_template_pair_features=lambda batch: batch,
            has_atomized_tokens=lambda batch: False,
        ),
        "foldjax.models.openfold3.inference": SimpleNamespace(
            released_config=lambda **_kwargs: config,
            compile_predict=compile_predict,
            cast_narrow_params=lambda params, *_dtypes: params,
            resolve_dtypes=lambda _config: ("bfloat16", "bfloat16"),
        ),
        "foldjax.models.openfold3.output": SimpleNamespace(),
        "foldjax.models.openfold3.bridge.chemistry": SimpleNamespace(
            representative_atom_table=lambda: None,
        ),
        "foldjax.models.openfold3.bridge.checkpoint": SimpleNamespace(
            load_checkpoint=lambda _path: {},
        ),
        "foldjax.models.openfold3.bridge.torch_mapping": SimpleNamespace(
            resolve_model_prefix=lambda _state, prefix: prefix,
            prune_sample_diffusion_aliases=lambda _state, prefix: None,
            map_inference_params=lambda _state, _prefix: {},
        ),
        "jax": jax,
    }
    monkeypatch.setattr(openfold3_backend, "import_module", modules.__getitem__)
    monkeypatch.setattr(
        openfold3_backend,
        "_features_chemistry_and_metadata",
        features_chemistry_and_metadata,
    )
    return featurized


def test_predict_hands_the_native_call_the_algorithm_it_resolved(
    tmp_path: Path, monkeypatch
) -> None:
    seen: list[str] = []
    _drive_to_the_native_call(monkeypatch, seen)

    for options, expected in (
        ({}, "ring"),
        ({"triangle_attention_grid": "ring"}, "ring"),
        ({"triangle_attention_grid": "gather"}, "gather"),
    ):
        with pytest.raises(_NativeCallReachedError):
            OpenFold3Backend().predict(_request(tmp_path, **options))
        assert seen.pop() == expected
    # The scope is the run's alone.
    assert triangle_attention_grid() == "ring"


def test_gather_without_cuequivariance_on_a_gpu_is_refused_before_featurizing(
    tmp_path: Path, monkeypatch
) -> None:
    seen: list[str] = []
    featurized = _drive_to_the_native_call(monkeypatch, seen)

    def refused() -> str:
        raise RuntimeError("triangle_attention_grid='gather' needs cuEquivariance")

    monkeypatch.setattr(_cp_attention, "resolve_gather_attention_body", refused)
    with pytest.raises(RuntimeError, match="gather"):
        OpenFold3Backend().predict(
            _request(tmp_path, triangle_attention_grid="gather")
        )
    assert featurized == [] and seen == []
    # The ring never asks.
    with pytest.raises(_NativeCallReachedError):
        OpenFold3Backend().predict(_request(tmp_path))
    assert featurized == ["featurized"] and seen == ["ring"]
