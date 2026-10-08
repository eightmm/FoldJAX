"""The base-constraint checkpoint's transformer ``SubstructureEmbedder``.

``fixtures/substructure_upstream.npz`` is upstream's own ``ConstraintEmbedder``
in that configuration (``scripts/substructure_upstream_fixture.py``).
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.protenix.bridge.torch_mapping import (
    map_constraint_embedder_state_dict,
)
from foldjax.models.protenix.models.primitives.primitives import LinearParams
from foldjax.models.protenix.models.trunk_blocks import trunk as trunk_module
from foldjax.models.protenix.models.trunk_blocks.embedders import (
    ConstraintEmbedderParams,
    SubstructureMlpParams,
    SubstructureTransformerParams,
    constraint_embedder,
    require_complete_constraint_embedder,
    substructure_transformer,
    validate_zero_substructure,
)

FIXTURE = Path(__file__).parent / "fixtures" / "substructure_upstream.npz"
CHANNELS = ("pocket", "contact", "contact_atom", "substructure")


@pytest.fixture(scope="module")
def upstream() -> dict[str, np.ndarray]:
    with np.load(FIXTURE) as data:
        return {key: data[key] for key in data.files}


def _params(upstream) -> ConstraintEmbedderParams:
    state = {
        f"constraint_embedder.{key.removeprefix('param/')}": value
        for key, value in upstream.items()
        if key.startswith("param/")
    }
    return map_constraint_embedder_state_dict(state, "constraint_embedder")


def _features(upstream, case: str) -> dict[str, jnp.ndarray]:
    return {channel: jnp.asarray(upstream[f"{case}_{channel}"]) for channel in CHANNELS}


def test_converter_maps_the_transformer_form(upstream) -> None:
    params = _params(upstream)
    substructure = params.substructure_z
    assert isinstance(substructure, SubstructureTransformerParams)
    assert len(substructure.layers) == 2
    assert substructure.layers[0].self_attn_in_proj.weight.shape == (24, 8)
    assert params.pocket_z is not None and params.contact_atom_z is not None


def test_converter_refuses_a_substructure_layout_it_cannot_map() -> None:
    state = {"constraint_embedder.substructure_z_embedder.mystery.weight": np.ones(2)}
    with pytest.raises(ValueError, match="neither the MLP nor the transformer"):
        map_constraint_embedder_state_dict(state, "constraint_embedder")


@pytest.mark.parametrize("case", ["off", "pocket"])
def test_constraint_embedder_matches_upstream(upstream, case) -> None:
    actual = constraint_embedder(_features(upstream, case), _params(upstream))
    np.testing.assert_allclose(actual, upstream[f"{case}_out"], rtol=1e-5, atol=1e-5)


def test_substructure_embedder_matches_upstream_on_the_zero_map(upstream) -> None:
    params = _params(upstream).substructure_z
    zero = jnp.asarray(upstream["off_substructure"])
    np.testing.assert_allclose(
        substructure_transformer(zero, params),
        upstream["substructure_out"],
        rtol=1e-5,
        atol=1e-5,
    )
    traced = jax.jit(substructure_transformer)(zero, params)
    np.testing.assert_allclose(traced, upstream["substructure_out"], atol=1e-5)


def _recorded_z_constraint(monkeypatch, features, constraint):
    """The ``z_constraint`` the trunk hands ``trunk_initial_embeddings``."""
    from tests.models.protenix.test_trunk import _pairformer_output_params

    seen = []
    original = trunk_module.trunk_initial_embeddings

    def recording(*args, z_constraint=None, **kwargs):
        seen.append(z_constraint)
        # The tiny trunk's pair width is not the fixture's; the term itself is
        # what is under test, and `trunk_initial_embeddings` only adds it.
        return original(*args, **kwargs)

    monkeypatch.setattr(trunk_module, "trunk_initial_embeddings", recording)
    n_token = 5
    trunk_module.pairformer_output_from_s_inputs(
        {
            "relp": jnp.zeros((n_token, n_token, 2), dtype=jnp.float32),
            "token_bonds": jnp.zeros((n_token, n_token), dtype=jnp.float32),
            **features,
        },
        jnp.ones((n_token, 2), dtype=jnp.float32),
        _pairformer_output_params()._replace(constraint=constraint),
        num_recycles=1,
    )
    (z_constraint,) = seen
    return z_constraint


def test_trunk_adds_upstreams_term_without_a_constraint(upstream, monkeypatch) -> None:
    # Upstream attaches all-zero constraint maps to every job, so a job with no
    # constraint still gains the substructure embedder's constant.
    z_constraint = _recorded_z_constraint(monkeypatch, {}, _params(upstream))
    np.testing.assert_allclose(
        jnp.broadcast_to(z_constraint, upstream["off_out"].shape),
        upstream["off_out"],
        rtol=1e-5,
        atol=1e-5,
    )


def test_trunk_adds_upstreams_term_with_a_pocket(upstream, monkeypatch) -> None:
    z_constraint = _recorded_z_constraint(
        monkeypatch,
        {"constraint_feature": _features(upstream, "pocket")},
        _params(upstream),
    )
    np.testing.assert_allclose(
        z_constraint, upstream["pocket_out"], rtol=1e-5, atol=1e-5
    )


def test_trunk_adds_nothing_for_weights_without_a_constraint_embedder(
    monkeypatch,
) -> None:
    assert _recorded_z_constraint(monkeypatch, {}, ConstraintEmbedderParams()) is None


def test_trunk_init_gains_exactly_the_term(upstream) -> None:
    from tests.models.protenix.test_trunk import _pairformer_output_params

    initial = _pairformer_output_params().trunk.initial
    s_inputs = jnp.asarray(np.arange(10, dtype=np.float32).reshape(5, 2))
    relp = jnp.zeros((5, 5, 2), dtype=jnp.float32)
    bonds = jnp.zeros((5, 5), dtype=jnp.float32)
    term = jnp.asarray(upstream["off_out"][..., :2])
    _, without = trunk_module.trunk_initial_embeddings(s_inputs, relp, bonds, initial)
    _, with_term = trunk_module.trunk_initial_embeddings(
        s_inputs, relp, bonds, initial, z_constraint=term
    )
    np.testing.assert_array_equal(with_term, without + term)


def test_a_nonzero_substructure_map_is_refused(upstream) -> None:
    params = _params(upstream)
    features = _features(upstream, "pocket")
    features["substructure"] = features["substructure"].at[0, 1, 2].set(1.0)
    with pytest.raises(ValueError, match="nonzero substructure feature is refused"):
        constraint_embedder(features, params)
    # Before tracing, where the embedder cannot see values.
    with pytest.raises(ValueError, match="nonzero substructure feature is refused"):
        validate_zero_substructure({"constraint_feature": features}, params)
    validate_zero_substructure(
        {"constraint_feature": _features(upstream, "pocket")}, params
    )
    # The MLP form reads any map.
    mlp = ConstraintEmbedderParams(
        substructure_z=SubstructureMlpParams(
            layers=(LinearParams(weight=jnp.ones((2, 4))),)
        )
    )
    validate_zero_substructure({"constraint_feature": features}, mlp)


def test_an_earlier_incomplete_conversion_is_named(upstream) -> None:
    stale = _params(upstream)._replace(substructure_z=None)
    with pytest.raises(ValueError, match="converted before FoldJAX mapped"):
        constraint_embedder(_features(upstream, "pocket"), stale)

    class _Params:
        class pairformer_output:  # noqa: N801 - attribute access only
            constraint = stale

    with pytest.raises(ValueError, match="Reconvert them"):
        require_complete_constraint_embedder(_Params)
    require_complete_constraint_embedder(None)
    with pytest.raises(ValueError, match="carry no constraint embedder"):
        constraint_embedder(
            {"pocket": jnp.zeros((2, 2, 1))}, ConstraintEmbedderParams()
        )


def test_a_featurized_pocket_job_reaches_the_embedder(upstream) -> None:
    from foldjax.models.protenix.data.featurize_json import featurize_protein_json

    features = featurize_protein_json(
        {
            "name": "pocket",
            "sequences": [
                {"proteinChain": {"sequence": "ACDEF", "count": 1}},
                {"proteinChain": {"sequence": "GHK", "count": 1}},
            ],
            "constraint": {
                "pocket": {
                    "binder_chain": {"entity": 2, "copy": 1},
                    "contact_residues": [{"entity": 1, "copy": 1, "position": 3}],
                    "max_distance": 6,
                }
            },
        }
    )
    constraint = {
        key: jnp.asarray(value)
        for key, value in features["constraint_feature"].items()
    }
    params = _params(upstream)
    validate_zero_substructure(features, params)
    actual = constraint_embedder(constraint, params)
    pocket = np.asarray(constraint["pocket"])
    expected = (
        pocket @ upstream["param/pocket_z_embedder.weight"].T
        + upstream["off_out"][0, 0]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_loading_an_incomplete_conversion_names_it(upstream, tmp_path) -> None:
    from foldjax.models.protenix.bridge.weights_io import save_native_weights
    from foldjax.models.protenix.models.model import ProtenixInferenceParams
    from foldjax.models.protenix.models.trunk_blocks.trunk import (
        PairformerOutputParams,
    )
    from foldjax.models.protenix.runner import _load_prepared_params

    def weights(constraint: ConstraintEmbedderParams) -> Path:
        path = tmp_path / f"{constraint.substructure_z is None}.jax"
        save_native_weights(
            path,
            ProtenixInferenceParams(
                input_embedder=None,
                pairformer_output=PairformerOutputParams(
                    trunk=None,
                    constraint=constraint,
                    template=None,
                    msa=None,
                    pairformer_stack=None,
                ),
                diffusion=None,
                distogram=None,
                confidence=None,
            ),
            compress=False,
        )
        return path

    complete = _params(upstream)
    loaded = _load_prepared_params(weights(complete), "fp32")
    assert isinstance(
        loaded.pairformer_output.constraint.substructure_z,
        SubstructureTransformerParams,
    )
    stale = weights(complete._replace(substructure_z=None))
    with pytest.raises(ValueError, match="Reconvert them"):
        _load_prepared_params(stale, "fp32")
