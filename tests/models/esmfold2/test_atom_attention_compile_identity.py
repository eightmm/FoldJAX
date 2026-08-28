"""Atom-attention environment choices belong to the whole-model JIT identity."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.esmfold2 import inference
from foldjax.models.esmfold2.data import all_atom as all_atom_featurisation
from foldjax.models.esmfold2.data.features import build_features
from foldjax.models.esmfold2.models import atom
from foldjax.models.esmfold2.models import model as structure_model


def test_environment_change_retraces_the_resolved_atom_attention_graph(
    monkeypatch,
) -> None:
    """A prior blocked trace must not pin a later dense or narrower run."""

    features = build_features([("AG", "A", 0, 0)])
    settings = dataclasses.replace(
        structure_model.ModelSettings(), trunk_n_layers=0, coda_n_layers=0
    )
    loaded = inference.LoadedModel(
        parameters={
            "token_bonds.weight": jnp.ones(
                (settings.d_pair, 1), dtype=jnp.bfloat16
            )
        },
        settings=settings,
    )
    traces: list[int] = []

    def tiny_predict(key, model_features, parameters, **kwargs):
        del key, parameters, kwargs
        rows = atom._resolve_rows_per_block(None)
        traces.append(rows)
        return {"probe": model_features["asym_id"].astype(jnp.int32) + rows}

    monkeypatch.setattr(structure_model, "predict", tiny_predict)
    inference.compiled_predict.cache_clear()

    def run(*, backend: str | None, rows: str | None) -> np.ndarray:
        if backend is None:
            monkeypatch.delenv("ESMFOLD2_ATOM_ATTENTION_BACKEND", raising=False)
        else:
            monkeypatch.setenv("ESMFOLD2_ATOM_ATTENTION_BACKEND", backend)
        if rows is None:
            monkeypatch.delenv("ESMFOLD2_ATOM_ROWS_PER_BLOCK", raising=False)
        else:
            monkeypatch.setenv("ESMFOLD2_ATOM_ROWS_PER_BLOCK", rows)
        result = inference.predict(jax.random.key(0), features, loaded)
        jax.block_until_ready(result)
        return np.asarray(result["probe"])

    try:
        default = run(backend=None, rows=None)
        explicit_default = run(backend="blocked", rows="256")
        narrower = run(backend="blocked", rows="128")
        dense = run(backend="dense", rows="999")
        repeated_dense = run(backend="dense", rows="128")

        np.testing.assert_array_equal(default, explicit_default)
        np.testing.assert_array_equal(dense, repeated_dense)
        assert int(default[0, 0]) == 256
        assert int(narrower[0, 0]) == 128
        assert int(dense[0, 0]) == 0
        assert traces == [256, 128, 0]
        assert inference._compiled_predict_pool._entry_count() == 3  # noqa: SLF001

        compact_bonds = inference._has_compact_token_bond_encoding(
            features,
            loaded.parameters,
            pair_width=settings.d_pair,
            compute_dtype=settings.trunk_dtype,
        )
        model_features = inference._model_bound_features(
            features, compact_token_bond_encoding=compact_bonds
        )
        arrays = {
            name: jnp.asarray(value)
            for name, value in model_features.items()
            if name not in all_atom_featurisation.OUTPUT_METADATA_FEATURES
        }
        contiguous_atoms = inference._has_contiguous_atom_groups(features)

        def stablehlo(rows_per_block: int) -> str:
            runner = inference.compiled_predict(
                settings,
                1,
                False,
                1,
                (),
                False,
                contiguous_atoms,
                compact_bonds,
                True,
                False,
                rows_per_block,
            )
            lowered = runner.lower(
                jax.random.key(0),
                arrays,
                loaded.parameters,
                None,
                settings,
                1,
                False,
                1,
                (),
                False,
                contiguous_atoms,
                compact_bonds,
                True,
                False,
                rows_per_block,
            )
            return str(lowered.compiler_ir(dialect="stablehlo"))

        default_hlo = stablehlo(256)
        explicit_default_hlo = stablehlo(256)
        narrower_hlo = stablehlo(128)
        dense_hlo = stablehlo(0)
        assert default_hlo == explicit_default_hlo
        assert default_hlo != narrower_hlo
        assert default_hlo != dense_hlo
        assert narrower_hlo != dense_hlo
        # The ambient environment is still dense here. Each lower must see
        # the static identity it was given, not that later process value.
        assert traces[-4:] == [256, 256, 128, 0]
    finally:
        inference.compiled_predict.cache_clear()
