"""Static output profiles for Protenix/OpenDDE confidence pair details."""

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

from foldjax.api import resolve_cache_dir
from foldjax.backends.opendde import OpenDDEBackend
from foldjax.backends.protenix import ProtenixBackend
from foldjax.models.opendde.cli.predict import _score as opendde_score
from foldjax.models.opendde.models import model as opendde_model
from foldjax.models.opendde.postprocess import (
    CONFIDENCE_DETAIL_KEYS,
    SHAPE_COMPLEMENTARITY_SCORE_KEYS,
)
from foldjax.models.protenix.models import model as protenix_model
from foldjax.models.protenix.models.diffusion.diffusion import (
    inference_noise_schedule,
)
from foldjax.models.protenix.models.heads.confidence import (
    confidence_scores_from_logits,
)
from foldjax.schema import PredictionRequest
from tests.models.cp_probe_env import inherited_environment
from tests.models.protenix.test_model import _toy_features, _toy_params


def _request(tmp_path, model: str, **options: object) -> PredictionRequest:
    source = tmp_path / f"{model}.json"
    source.write_text("{}\n", encoding="utf-8")
    weights = tmp_path / f"{model}.jax"
    weights.write_bytes(b"fixture")
    return PredictionRequest(
        model=model,
        input=source,
        weights=weights,
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        options=options,
    )


def _assert_same_array(actual: object, expected: object, *, name: str) -> None:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    assert actual_array.dtype == expected_array.dtype, name
    assert actual_array.shape == expected_array.shape, name
    np.testing.assert_array_equal(
        np.isnan(actual_array)
        if np.issubdtype(actual_array.dtype, np.inexact)
        else actual_array,
        np.isnan(expected_array)
        if np.issubdtype(expected_array.dtype, np.inexact)
        else expected_array,
        err_msg=f"non-finite class drifted for {name}",
    )
    if np.issubdtype(actual_array.dtype, np.inexact):
        for predicate in (np.isposinf, np.isneginf):
            np.testing.assert_array_equal(
                predicate(actual_array),
                predicate(expected_array),
                err_msg=f"infinity class drifted for {name}",
            )
        actual_array = np.nan_to_num(actual_array)
        expected_array = np.nan_to_num(expected_array)
    np.testing.assert_array_equal(actual_array, expected_array, err_msg=name)


def _confidence_case(
    asym_id: np.ndarray,
    token_mask: np.ndarray,
    *,
    nonfinite: bool,
) -> dict[str, jnp.ndarray | int | float]:
    samples, bins = 2, 4
    n_token = int(asym_id.size)
    rng = np.random.default_rng(20260828 + n_token)
    plddt = rng.normal(size=(samples, n_token, bins)).astype(np.float32)
    pae = rng.normal(size=(samples, n_token, n_token, bins)).astype(np.float32)
    pde = rng.normal(size=(samples, n_token, n_token, bins)).astype(np.float32)
    distogram = rng.normal(size=(n_token, n_token, bins)).astype(np.float32)
    if nonfinite:
        plddt[0, 0, 0] = np.nan
        pae[0, 0, 0, 0] = np.inf
        pae[1, -1, -1, :] = -np.inf
        pde[0, -1, 0, 1] = np.nan
        distogram[0, -1, 0] = np.inf
    coordinate = np.stack(
        [
            np.column_stack(
                (
                    np.arange(n_token, dtype=np.float32),
                    np.full(n_token, sample, dtype=np.float32),
                    np.zeros(n_token, dtype=np.float32),
                )
            )
            for sample in range(samples)
        ]
    )
    elements = np.zeros((n_token, 128), dtype=np.float32)
    elements[:, 5] = 1.0
    return {
        "plddt_logits": jnp.asarray(plddt),
        "pae_logits": jnp.asarray(pae),
        "pde_logits": jnp.asarray(pde),
        "distogram_logits": jnp.asarray(distogram),
        "plddt_max_bin": 4.0,
        "pae_max_bin": 4.0,
        "pde_max_bin": 4.0,
        "distogram_min_bin": 0.0,
        "distogram_max_bin": 4.0,
        "token_has_frame": jnp.ones((n_token,), dtype=bool),
        "token_asym_id": jnp.asarray(asym_id, dtype=jnp.int32),
        "atom_to_token_idx": jnp.arange(n_token, dtype=jnp.int32),
        "atom_coordinate": jnp.asarray(coordinate),
        "elements_one_hot": jnp.asarray(elements),
        "mol_id": jnp.arange(n_token, dtype=jnp.int32),
        "token_is_ligand": jnp.zeros((n_token,), dtype=bool),
        "token_mask": jnp.asarray(token_mask, dtype=bool),
        "atom_mask": jnp.asarray(token_mask, dtype=bool),
        "num_recycles": 3,
        "n_chain": int(asym_id.max()) + 1,
    }


@pytest.mark.parametrize(
    ("asym_id", "token_mask", "nonfinite"),
    [
        (np.asarray([0, 0]), np.asarray([False, False]), False),
        (np.asarray([0, 0, 0]), np.asarray([True, True, True]), False),
        (np.asarray([0, 1, 2, 2]), np.asarray([True, True, True, True]), False),
        (np.asarray([0, 1, 1]), np.asarray([True, True, True]), True),
    ],
    ids=("empty-mask", "monomer", "multimer", "nonfinite"),
)
def test_elision_changes_only_the_three_detail_leaves(
    asym_id: np.ndarray,
    token_mask: np.ndarray,
    nonfinite: bool,
) -> None:
    kwargs = _confidence_case(asym_id, token_mask, nonfinite=nonfinite)
    full = confidence_scores_from_logits(**kwargs, return_confidence_details=True)
    compact = confidence_scores_from_logits(**kwargs, return_confidence_details=False)

    assert set(full) - set(compact) == CONFIDENCE_DETAIL_KEYS
    assert not (set(compact) & CONFIDENCE_DETAIL_KEYS)
    for name, value in compact.items():
        _assert_same_array(value, full[name], name=name)


def test_protenix_model_coordinates_and_summaries_are_exact() -> None:
    features = _toy_features()
    params = _toy_params()
    init_noise = jnp.ones((2, 3, 3), dtype=jnp.float32)
    step_noise = jnp.zeros_like(init_noise)

    def run(return_details: bool):
        return protenix_model.protenix_infer_static(
            features,
            params,
            inference_noise_schedule(num_steps=1, sigma_data=4.0),
            key=None,
            num_samples=2,
            init_noise=init_noise,
            step_noises=(step_noise,),
            num_recycles=1,
            input_atom_heads=1,
            atom_encoder_heads=1,
            token_heads=1,
            atom_decoder_heads=1,
            n_queries=2,
            n_keys=4,
            sigma_data=4.0,
            centre_each_step=False,
            return_trunk=False,
            return_confidence_logits=False,
            return_confidence_details=return_details,
        )

    full = run(True)
    compact = run(False)
    assert set(full) - set(compact) == CONFIDENCE_DETAIL_KEYS
    for name, value in compact.items():
        _assert_same_array(value, full[name], name=name)


def test_opendde_score_preserves_coordinates_ranking_and_shape_fields() -> None:
    samples, tokens = 2, 3
    output: dict[str, object] = {
        "coordinate": jnp.arange(samples * tokens * 3, dtype=jnp.float32).reshape(
            samples, tokens, 3
        ),
        "plddt": jnp.zeros((samples, tokens, 50), dtype=jnp.float32),
        "pae": jnp.zeros((samples, tokens, tokens, 64), dtype=jnp.float32),
        "pde": jnp.zeros((samples, tokens, tokens, 64), dtype=jnp.float32),
        "distogram_logits": jnp.zeros((tokens, tokens, 96), dtype=jnp.float32),
        "shape_comp_token_pred": jnp.arange(samples * tokens).reshape(samples, tokens),
        "shape_comp_token_mask": jnp.asarray(
            [[True, False, True], [False, True, True]]
        ),
        "shape_comp_global_pred": jnp.asarray([0.25, 0.75]),
        "shape_comp_pair_mean_pred": jnp.asarray([0.2, 0.7]),
        "shape_comp_pair_topk_mean_pred": jnp.asarray([0.3, 0.8]),
        "shape_comp_valid_pair_frac_pred": jnp.asarray([0.5, 1.0]),
        "shape_comp_uses_structural_tokens": jnp.asarray(False),
    }
    features = {
        "has_frame": jnp.ones((tokens,), dtype=bool),
        "asym_id": jnp.asarray([0, 1, 1], dtype=jnp.int32),
        "atom_to_token_idx": jnp.arange(tokens, dtype=jnp.int32),
        "is_protein": jnp.ones((tokens,), dtype=bool),
    }

    full = opendde_score(
        output, features, num_recycles=2, return_confidence_details=True
    )
    compact = opendde_score(
        output, features, num_recycles=2, return_confidence_details=False
    )

    assert set(full) - set(compact) == CONFIDENCE_DETAIL_KEYS
    assert SHAPE_COMPLEMENTARITY_SCORE_KEYS <= compact.keys()
    for name, value in compact.items():
        _assert_same_array(value, full[name], name=name)


def test_output_profile_is_static_and_has_distinct_bounded_jit_identity() -> None:
    assert "return_confidence_details" in protenix_model.GRAPH_STATIC_ARGNAMES
    assert "return_confidence_details" in opendde_model.GRAPH_STATIC_ARGNAMES
    for function in (
        protenix_model.protenix_infer_static,
        opendde_model.opendde_infer_static,
    ):
        assert (
            inspect.signature(function).parameters["return_confidence_details"].default
            is True
        )
    for pool in (
        protenix_model._compiled_protenix_infer,
        opendde_model._compiled_opendde_infer,
    ):
        full = pool._identity((), {"return_confidence_details": True})
        compact = pool._identity((), {"return_confidence_details": False})
        assert full != compact


def test_backend_cache_profile_normalizes_writer_modes(tmp_path) -> None:
    protenix = ProtenixBackend()
    protenix_default = _request(tmp_path, "protenix")
    protenix_named = dataclasses.replace(
        protenix_default, options={"output_format": "protenix"}
    )
    protenix_raw = dataclasses.replace(
        protenix_default, options={"output_format": "npz"}
    )
    protenix_both = dataclasses.replace(
        protenix_default, options={"output_format": "both"}
    )
    assert protenix.cache_profile(protenix_default) == protenix.cache_profile(
        protenix_named
    )
    assert (
        protenix.cache_profile(protenix_default)["return_confidence_details"] is False
    )
    assert protenix.cache_profile(protenix_raw)["return_confidence_details"] is True
    assert protenix.cache_profile(protenix_both) == protenix.cache_profile(protenix_raw)
    assert resolve_cache_dir(protenix_default, protenix) == resolve_cache_dir(
        protenix_named, protenix
    )
    assert resolve_cache_dir(protenix_default, protenix) != resolve_cache_dir(
        protenix_raw, protenix
    )

    opendde = OpenDDEBackend()
    opendde_default = _request(tmp_path, "opendde")
    opendde_named = dataclasses.replace(opendde_default, options={"include_raw": False})
    opendde_raw = dataclasses.replace(opendde_default, options={"include_raw": True})
    assert opendde.cache_profile(opendde_default) == opendde.cache_profile(
        opendde_named
    )
    assert opendde.cache_profile(opendde_default)["return_confidence_details"] is False
    assert opendde.cache_profile(opendde_raw)["return_confidence_details"] is True
    assert resolve_cache_dir(opendde_default, opendde) == resolve_cache_dir(
        opendde_named, opendde
    )
    assert resolve_cache_dir(opendde_default, opendde) != resolve_cache_dir(
        opendde_raw, opendde
    )


@partial(jax.jit, static_argnames=("return_details",))
def _compiled_confidence_profile(plddt, pae, pde, distogram, *, return_details):
    n_token = int(pae.shape[-3])
    return confidence_scores_from_logits(
        plddt_logits=plddt,
        pae_logits=pae,
        pde_logits=pde,
        distogram_logits=distogram,
        token_has_frame=jnp.ones((n_token,), dtype=bool),
        token_asym_id=jnp.arange(n_token, dtype=jnp.int32) % 2,
        token_is_ligand=jnp.zeros((n_token,), dtype=bool),
        n_chain=2,
        return_confidence_details=return_details,
    )


def test_elision_reduces_compiled_output_bytes_and_jit_cache_is_distinct() -> None:
    samples, tokens, bins = 2, 64, 8
    plddt = jnp.zeros((samples, tokens, bins), dtype=jnp.float32)
    pair = jnp.zeros((samples, tokens, tokens, bins), dtype=jnp.float32)
    distogram = jnp.zeros((tokens, tokens, bins), dtype=jnp.float32)

    _compiled_confidence_profile.clear_cache()
    _compiled_confidence_profile(plddt, pair, pair, distogram, return_details=True)[
        "summary_ptm"
    ].block_until_ready()
    _compiled_confidence_profile(plddt, pair, pair, distogram, return_details=False)[
        "summary_ptm"
    ].block_until_ready()
    assert _compiled_confidence_profile._cache_size() == 2

    full = _compiled_confidence_profile.lower(
        plddt, pair, pair, distogram, return_details=True
    )
    compact = _compiled_confidence_profile.lower(
        plddt, pair, pair, distogram, return_details=False
    )
    full_bytes = full.compile().memory_analysis().output_size_in_bytes
    compact_bytes = compact.compile().memory_analysis().output_size_in_bytes
    detail_payload = (2 * samples + 1) * tokens * tokens * 4
    assert full_bytes - compact_bytes >= detail_payload

    # Inputs still contain two [S,N,N] pair-logit tensors. The public result
    # signature contains two more only in the full profile.
    pair_type = f"tensor<{samples}x{tokens}x{tokens}xf32>"
    full_signature = next(
        line for line in full.as_text().splitlines() if "func.func public @main" in line
    )
    compact_signature = next(
        line
        for line in compact.as_text().splitlines()
        if "func.func public @main" in line
    )
    assert full_signature.split(" -> ", maxsplit=1)[0] == compact_signature.split(
        " -> ", maxsplit=1
    )[0]
    for name in CONFIDENCE_DETAIL_KEYS:
        marker = f"result['{name}']"
        assert marker in full_signature
        assert marker not in compact_signature
    assert full_signature.count(pair_type) == compact_signature.count(pair_type) + 2


_CP_PROBE = textwrap.dedent(
    r"""
    import os

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel, shard_pair_rows
    from foldjax.models.opendde.postprocess import opendde_confidence_scores
    from foldjax.models.protenix.models.heads.confidence import (
        confidence_scores_from_logits,
    )

    assert jax.device_count() == 4, jax.devices()
    layout = os.environ["FOLDJAX_CP_PROBE_LAYOUT"]
    samples, tokens = 2, 8
    rng = np.random.default_rng(20260828)
    plddt = jnp.asarray(rng.normal(size=(samples, tokens, 50)), jnp.float32)
    pae = jnp.asarray(
        rng.normal(size=(samples, tokens, tokens, 64)), jnp.float32
    )
    pde = jnp.asarray(
        rng.normal(size=(samples, tokens, tokens, 64)), jnp.float32
    )
    distogram = jnp.asarray(
        rng.normal(size=(tokens, tokens, 96)), jnp.float32
    )
    has_frame = jnp.asarray([True, True, False, True, True, True, True, True])
    asym_id = jnp.asarray([0, 0, 0, 0, 1, 1, 1, 1], jnp.int32)
    atom_to_token_idx = jnp.arange(tokens, dtype=jnp.int32)
    coordinates = jnp.arange(samples * tokens * 3, dtype=jnp.float32).reshape(
        samples, tokens, 3
    )

    def build(*, model, return_details):
        def run(plddt_value, pae_value, pde_value, distogram_value):
            pae_value = shard_pair_rows(pae_value)
            pde_value = shard_pair_rows(pde_value)
            distogram_value = shard_pair_rows(distogram_value)
            if model == "opendde":
                return opendde_confidence_scores(
                    {
                        "coordinate": coordinates,
                        "plddt": plddt_value,
                        "pae": pae_value,
                        "pde": pde_value,
                        "distogram_logits": distogram_value,
                    },
                    {
                        "has_frame": has_frame,
                        "asym_id": asym_id,
                        "atom_to_token_idx": atom_to_token_idx,
                        "is_protein": jnp.ones((tokens,), dtype=bool),
                    },
                    num_recycles=3,
                    n_chain=2,
                    include_shape_complementarity=False,
                    return_confidence_details=return_details,
                )
            return confidence_scores_from_logits(
                plddt_logits=plddt_value,
                pae_logits=pae_value,
                pde_logits=pde_value,
                distogram_logits=distogram_value,
                token_has_frame=has_frame,
                token_asym_id=asym_id,
                token_is_ligand=jnp.zeros((tokens,), dtype=bool),
                n_chain=2,
                return_confidence_details=return_details,
            )
        return run

    def ready(tree):
        return jax.tree.map(lambda value: np.asarray(value), tree)

    def compare_exact(actual, expected):
        assert actual.keys() == expected.keys()
        for name in actual:
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)

    def compare_cp(actual, expected):
        assert actual.keys() == expected.keys()
        for name in actual:
            np.testing.assert_allclose(
                actual[name],
                expected[name],
                rtol=1e-5,
                atol=1e-5,
                err_msg=name,
            )

    for model in ("protenix", "opendde"):
        serial_full = ready(
            jax.jit(build(
                model=model,
                return_details=True,
            ))(plddt, pae, pde, distogram)
        )
        serial_compact = {
            name: value
            for name, value in serial_full.items()
            if name not in {"token_pair_pae", "token_pair_pde", "contact_probs"}
        }

        jax.clear_caches()
        with context_parallel(4, layout=layout):
            full_executable = jax.jit(build(
                model=model,
                return_details=True,
            )).lower(plddt, pae, pde, distogram).compile()
            compact_executable = jax.jit(build(
                model=model,
                return_details=False,
            )).lower(plddt, pae, pde, distogram).compile()
            cp_full = ready(full_executable(plddt, pae, pde, distogram))
            cp_compact = ready(compact_executable(plddt, pae, pde, distogram))
            full_hlo = full_executable.as_text().lower()
            compact_hlo = compact_executable.as_text().lower()

        # Context-parallel reductions have a different summation tree from a
        # serial executable, hence the ordinary confidence tolerance. The two
        # output profiles on the *same* mesh must remain exactly equal.
        compare_cp(cp_full, serial_full)
        compare_cp(cp_compact, serial_compact)
        compare_exact(cp_compact, {
            name: value
            for name, value in cp_full.items()
            if name not in {"token_pair_pae", "token_pair_pde", "contact_probs"}
        })
        for marker in (
            " all-gather(",
            " all-reduce(",
            " collective-permute(",
            " reduce-scatter(",
        ):
            assert compact_hlo.count(marker) == full_hlo.count(marker), (
                model,
                marker,
                compact_hlo.count(marker),
                full_hlo.count(marker),
            )
    print("CONFIDENCE_DETAIL_CP_OK")
    """
)


@pytest.mark.parametrize("layout", ["1d", "2d"])
def test_confidence_detail_elision_preserves_forced_cpu_cp(layout: str) -> None:
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
    assert "CONFIDENCE_DETAIL_CP_OK" in completed.stdout
