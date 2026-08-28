"""Managed ESMFold2 returns only what its writer and representation API use."""

from __future__ import annotations

import dataclasses
import inspect
import subprocess
import sys
import textwrap
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2 import inference
from foldjax.models.esmfold2 import output as output_module
from foldjax.models.esmfold2.data.features import build_features
from foldjax.models.esmfold2.models import heads as structure_heads
from foldjax.models.esmfold2.models import model as structure_model
from tests.models.cp_probe_env import inherited_environment


def _prediction(*, samples: int = 2, tokens: int = 4, atoms: int = 6):
    plddt = jnp.arange(samples * tokens, dtype=jnp.float32).reshape(samples, tokens)
    plddt = plddt.at[0, 0].set(jnp.asarray(-0.0, dtype=jnp.float32))
    return {
        "single": jnp.arange(tokens * 3, dtype=jnp.float32).reshape(1, tokens, 3),
        "pair": jnp.arange(tokens * tokens, dtype=jnp.float32).reshape(
            1, tokens, tokens, 1
        ),
        "sample_atom_coords": jnp.arange(
            samples * atoms * 3, dtype=jnp.float32
        ).reshape(samples, atoms, 3),
        "plddt": plddt,
        "plddt_per_atom": jnp.arange(samples * atoms, dtype=jnp.float32).reshape(
            samples, atoms
        ),
        "complex_plddt": jnp.linspace(0.25, 0.75, samples),
        "complex_iplddt": jnp.linspace(0.75, 0.25, samples),
        "ptm": jnp.linspace(0.1, 0.2, samples),
        "iptm": jnp.linspace(0.3, 0.4, samples),
        "atom_pad_mask": jnp.ones((1, atoms), dtype=jnp.float32),
        "residue_index": jnp.arange(tokens, dtype=jnp.int32)[None],
        "entity_id": jnp.ones((1, tokens), dtype=jnp.int32),
        "plddt_ca": plddt,
        "pair_chains_iptm": jnp.arange(samples * 4, dtype=jnp.float32).reshape(
            samples, 2, 2
        ),
    }


def _assert_raw_equal(actual: object, expected: object, *, name: str) -> None:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    assert actual_array.dtype == expected_array.dtype, name
    assert actual_array.shape == expected_array.shape, name
    assert actual_array.tobytes() == expected_array.tobytes(), name


def test_direct_defaults_retain_native_auxiliary_outputs() -> None:
    for function in (structure_model.predict, inference.predict):
        assert (
            inspect.signature(function).parameters["return_auxiliary_outputs"].default
            is True
        )

    original = _prediction()
    public = structure_model._project_prediction_outputs(  # noqa: SLF001
        original,
        return_auxiliary_outputs=True,
    )
    managed = structure_model._project_prediction_outputs(  # noqa: SLF001
        original,
        return_auxiliary_outputs=False,
    )

    assert set(public) == set(original)
    assert set(public) - set(managed) == structure_model.MANAGED_AUXILIARY_OUTPUTS
    assert set(original) == set(public)
    for name, value in managed.items():
        _assert_raw_equal(value, public[name], name=name)


def test_writer_cif_and_json_are_exact_without_auxiliary_outputs(tmp_path) -> None:
    features = build_features([("AG", "A", 0, 0)])
    tokens = int(features["token_attention_mask"].shape[-1])
    atoms = int(features["atom_attention_mask"].shape[-1])
    prediction = _prediction(samples=2, tokens=tokens, atoms=atoms)

    public = structure_model._project_prediction_outputs(  # noqa: SLF001
        prediction,
        return_auxiliary_outputs=True,
    )
    managed = structure_model._project_prediction_outputs(  # noqa: SLF001
        prediction,
        return_auxiliary_outputs=False,
    )
    public_written = output_module.write_prediction_outputs(
        public, features, tmp_path / "public", name="same"
    )
    managed_written = output_module.write_prediction_outputs(
        managed, features, tmp_path / "managed", name="same"
    )

    assert public_written["summary"] == managed_written["summary"]
    assert (
        public_written["scores"].read_bytes() == managed_written["scores"].read_bytes()
    )
    assert len(public_written["structures"]) == len(managed_written["structures"])
    for public_path, managed_path in zip(
        public_written["structures"], managed_written["structures"], strict=True
    ):
        assert public_path.read_bytes() == managed_path.read_bytes()


@partial(jax.jit, static_argnames=("return_auxiliary_outputs",))
def _compiled_projection(
    pair,
    plddt,
    atom_mask,
    residue_index,
    entity_id,
    *,
    return_auxiliary_outputs,
):
    samples, tokens, _ = pair.shape
    chain_membership = jax.nn.one_hot(
        jnp.arange(tokens, dtype=jnp.int32) % 8,
        8,
        dtype=jnp.float32,
    )
    chains = jnp.broadcast_to(
        jnp.swapaxes(chain_membership, 0, 1)[None],
        (samples, 8, tokens),
    )
    numerator = jnp.einsum("bij,bci,bdj->bcd", pair, chains, chains)
    denominator = (
        jnp.einsum("bci,bdj->bcd", chains, chains) + structure_heads.EPS
    )
    pair_chains = numerator / denominator
    output = {
        "sample_atom_coords": jnp.zeros((samples, atom_mask.shape[-1], 3)),
        "plddt": plddt,
        "plddt_per_atom": plddt[:, : atom_mask.shape[-1]],
        "complex_plddt": jnp.mean(plddt, axis=-1),
        "complex_iplddt": jnp.max(plddt, axis=-1),
        "ptm": jnp.min(plddt, axis=-1),
        "iptm": jnp.sum(plddt, axis=-1),
        "atom_pad_mask": atom_mask,
        "residue_index": residue_index,
        "entity_id": entity_id,
        "plddt_ca": plddt,
        "pair_chains_iptm": pair_chains,
    }
    return structure_model._project_prediction_outputs(  # noqa: SLF001
        output,
        return_auxiliary_outputs=return_auxiliary_outputs,
    )


def test_projection_dces_chain_matrix_and_reduces_compiled_output_bytes() -> None:
    samples, tokens, atoms, chains = 2, 32, 24, 8
    pair = jnp.arange(samples * tokens * tokens, dtype=jnp.float32).reshape(
        samples, tokens, tokens
    )
    plddt = jnp.arange(samples * tokens, dtype=jnp.float32).reshape(samples, tokens)
    atom_mask = jnp.ones((1, atoms), dtype=jnp.float32)
    residue_index = jnp.arange(tokens, dtype=jnp.int32)[None]
    entity_id = jnp.ones((1, tokens), dtype=jnp.int32)

    full = _compiled_projection.lower(
        pair,
        plddt,
        atom_mask,
        residue_index,
        entity_id,
        return_auxiliary_outputs=True,
    )
    compact = _compiled_projection.lower(
        pair,
        plddt,
        atom_mask,
        residue_index,
        entity_id,
        return_auxiliary_outputs=False,
    )
    full_executable = full.compile()
    compact_executable = compact.compile()
    full_result = full_executable(pair, plddt, atom_mask, residue_index, entity_id)
    compact_result = compact_executable(
        pair, plddt, atom_mask, residue_index, entity_id
    )

    full_dot_count = full.as_text().count("stablehlo.dot_general")
    compact_dot_count = compact.as_text().count("stablehlo.dot_general")
    assert full_dot_count == 3
    assert compact_dot_count == 0
    for name, value in compact_result.items():
        _assert_raw_equal(value, full_result[name], name=name)

    full_bytes = full_executable.memory_analysis().output_size_in_bytes
    compact_bytes = compact_executable.memory_analysis().output_size_in_bytes
    auxiliary_payload = (
        samples * chains * chains * 4
        + samples * tokens * 4
        + atoms * 4
        + 2 * tokens * 4
    )
    full_payload = sum(np.asarray(value).nbytes for value in full_result.values())
    compact_payload = sum(
        np.asarray(value).nbytes for value in compact_result.values()
    )
    assert full_payload - compact_payload == auxiliary_payload
    assert full_bytes - compact_bytes >= auxiliary_payload


def test_auxiliary_choice_has_a_distinct_bounded_compiled_identity() -> None:
    assert "return_auxiliary_outputs" in inference._COMPILED_PREDICT_STATIC_ARGNAMES
    assert inference._compiled_predict_pool._limit == 8  # noqa: SLF001
    settings = dataclasses.replace(
        structure_model.ModelSettings(), trunk_n_layers=0, coda_n_layers=0
    )
    inference.compiled_predict.cache_clear()
    try:
        public = inference.compiled_predict(settings, 1, return_auxiliary_outputs=True)
        managed = inference.compiled_predict(
            settings, 1, return_auxiliary_outputs=False
        )
        assert public is not managed
        assert public is inference.compiled_predict(
            settings, 1, return_auxiliary_outputs=True
        )

        for n_chains in range(1, 6):
            for return_auxiliary_outputs in (True, False):
                inference.compiled_predict(
                    settings,
                    n_chains,
                    return_auxiliary_outputs=return_auxiliary_outputs,
                )
        assert inference.compiled_predict.cache_info().currsize == 8
    finally:
        inference.compiled_predict.cache_clear()


def test_inference_routes_auxiliary_choice_into_factory_and_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = build_features([("AG", "A", 0, 0)])
    settings = structure_model.ModelSettings()
    loaded = inference.LoadedModel(
        parameters={
            "token_bonds.weight": jnp.ones((settings.d_pair, 1), dtype=jnp.bfloat16)
        },
        settings=settings,
    )
    seen: list[tuple[str, bool]] = []

    def fake_compiled_predict(*identity, return_auxiliary_outputs=True):
        del identity
        seen.append(("factory", return_auxiliary_outputs))

        def run(*args, return_auxiliary_outputs=True):
            del args
            seen.append(("runtime", return_auxiliary_outputs))
            return {}

        return run

    monkeypatch.setattr(inference, "compiled_predict", fake_compiled_predict)
    inference.predict(
        jax.random.key(0),
        features,
        loaded,
        return_auxiliary_outputs=False,
    )

    assert seen == [("factory", False), ("runtime", False)]


_CP_PROBE = textwrap.dedent(
    r"""
    import os

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel, shard_pair_rows
    from foldjax.models.esmfold2.models import model as structure_model

    assert jax.device_count() == 4, jax.devices()
    layout = os.environ["FOLDJAX_CP_PROBE_LAYOUT"]
    samples, tokens, chains = 2, 8, 2
    pair = jnp.arange(samples * tokens * tokens, dtype=jnp.float32).reshape(
        samples, tokens, tokens, 1
    )

    def build(return_auxiliary_outputs):
        def run(value):
            value = shard_pair_rows(value)
            scalar_pair = value[..., 0]
            plddt = jnp.sum(scalar_pair, axis=-1)
            membership = jax.nn.one_hot(
                jnp.arange(tokens, dtype=jnp.int32) % chains,
                chains,
                dtype=jnp.float32,
            ).T
            output = {
                "sample_atom_coords": jnp.repeat(plddt[..., None], 3, axis=-1),
                "plddt": plddt,
                "plddt_per_atom": plddt,
                "complex_plddt": jnp.mean(plddt, axis=-1),
                "complex_iplddt": jnp.max(plddt, axis=-1),
                "ptm": jnp.min(plddt, axis=-1),
                "iptm": jnp.sum(plddt, axis=-1),
                "atom_pad_mask": jnp.ones((1, tokens), dtype=jnp.float32),
                "residue_index": jnp.arange(tokens, dtype=jnp.int32)[None],
                "entity_id": jnp.ones((1, tokens), dtype=jnp.int32),
                "plddt_ca": plddt,
                "pair_chains_iptm": jnp.einsum(
                    "bij,ci,dj->bcd", scalar_pair, membership, membership
                ),
            }
            return structure_model._project_prediction_outputs(
                output,
                return_auxiliary_outputs=return_auxiliary_outputs,
            )
        return run

    def host(tree):
        return jax.tree.map(np.asarray, tree)

    with context_parallel(4, layout=layout):
        full_executable = jax.jit(build(True)).lower(pair).compile()
        compact_executable = jax.jit(build(False)).lower(pair).compile()
        full = host(full_executable(pair))
        compact = host(compact_executable(pair))
        full_hlo = full_executable.as_text().lower()
        compact_hlo = compact_executable.as_text().lower()

    expected = {
        name: value
        for name, value in full.items()
        if name not in structure_model.MANAGED_AUXILIARY_OUTPUTS
    }
    assert compact.keys() == expected.keys()
    for name in compact:
        np.testing.assert_array_equal(compact[name], expected[name], err_msg=name)
    for marker in (
        " all-gather(",
        " all-reduce(",
        " collective-permute(",
        " reduce-scatter(",
    ):
        assert compact_hlo.count(marker) <= full_hlo.count(marker), (
            marker,
            compact_hlo.count(marker),
            full_hlo.count(marker),
        )
    print("ESMFOLD2_AUX_OUTPUT_CP_OK", layout)
    """
)


@pytest.mark.parametrize("layout", ["1d", "2d"])
def test_managed_projection_preserves_forced_cpu_cp_layouts(layout: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _CP_PROBE],
        capture_output=True,
        text=True,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
            "FOLDJAX_CP_PROBE_LAYOUT": layout,
            **inherited_environment(),
        },
        timeout=240,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "ESMFOLD2_AUX_OUTPUT_CP_OK" in completed.stdout
