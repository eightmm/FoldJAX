"""Training-free guidance against upstream Protenix 2.0.0, on CPU.

``fixtures/tfg_upstream.npz`` holds upstream's own outputs
(``scripts/tfg_upstream_fixture.py``): one ``TFGEngine.step`` of the default
guidance config, and what its ``GeometryFeaturizer(...,
exclude_std_residue=True)`` returns for three jobs. OpenDDE ships the same
engine, potentials and featurizer, and FoldJAX runs one port for both.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.protenix.data.featurize_json import featurize_protein_json
from foldjax.models.protenix.data.geometry import prepare_tfg_features
from foldjax.models.protenix.tfg.config import (
    parse_tfg_config,
    upstream_guidance_config,
)
from foldjax.models.protenix.tfg.engine import TFGEngine
from foldjax.models.protenix.tfg.potentials import PairwiseDistancePotential

FIXTURE = Path(__file__).parent / "fixtures" / "tfg_upstream.npz"

GEOMETRY_KEYS = (
    "interchain_bond_index",
    "pairwise_distance_index",
    "pairwise_distance_upper_bound",
    "pairwise_distance_lower_bound",
    "pairwise_distance_is_bond",
    "pairwise_distance_is_angle",
    "experimental_torsion_index",
    "experimental_torsion_force_constant",
    "experimental_torsion_sign",
    "linear_triple_bond_index",
    "chiral_index",
    "chiral_orientation",
    "stereo_bond_index",
    "stereo_bond_orientation",
    "planar_improper_index",
    "planar_improper_is_carbonyl",
)
_BOUNDS = ("pairwise_distance_upper_bound", "pairwise_distance_lower_bound")


@pytest.fixture(scope="module")
def upstream() -> dict[str, np.ndarray]:
    with np.load(FIXTURE) as data:
        return {key: data[key] for key in data.files}


def _step(engine: TFGEngine, upstream: dict[str, np.ndarray], features) -> np.ndarray:
    return np.asarray(
        engine.step(
            lambda x, _t_hat: 0.8 * x + 0.1,
            x=jnp.asarray(upstream["engine_x"]),
            t_hat=jnp.full((2,), 2.5),
            c_tau=jnp.full((2,), 2.0),
            step_scale_eta=1.5,
            step_i=150,
            num_diffusion_steps=200,
            input_feature_dict=features,
        )
    )


def test_one_guided_step_matches_upstream(upstream) -> None:
    """Projection only out of range, then refinement, Chiral -> Pairwise.

    Before those three fixes the port's step sat 0.087 A from upstream's on
    a 0.16 A step; with them it is float32 round-off.
    """

    features = {
        key.removeprefix("engine_feat_"): jnp.asarray(value)
        for key, value in upstream.items()
        if key.startswith("engine_feat_")
    }
    engine = TFGEngine(parse_tfg_config(upstream_guidance_config()))

    step = _step(engine, upstream, features)

    error = np.linalg.norm(step - upstream["engine_step"], axis=-1)
    assert error.max() <= 1e-4, error.max()


def test_pairwise_projection_moves_only_out_of_range_pairs() -> None:
    features = {
        "pairwise_distance_index": jnp.asarray([[0], [1]]),
        "pairwise_distance_is_bond": jnp.asarray([1]),
        "pairwise_distance_is_angle": jnp.asarray([0]),
        "pairwise_distance_lower_bound": jnp.asarray([1.0]),
        "pairwise_distance_upper_bound": jnp.asarray([1.5]),
        "ref_element": jnp.zeros((2, 128)).at[:, 5].set(1.0),
    }
    params = {"bond_buffer": 0.0, "angle_buffer": 0.0, "clash_buffer": 0.0}
    potential = PairwiseDistancePotential()

    def projected_distance(distance: float) -> float:
        coords = jnp.asarray([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
        moved = coords + potential.project(coords, features, params)
        return float(jnp.linalg.norm(moved[1] - moved[0]))

    assert projected_distance(1.2) == pytest.approx(1.2, abs=1e-6)
    assert projected_distance(2.0) == pytest.approx(1.5, abs=1e-5)
    assert projected_distance(0.7) == pytest.approx(1.0, abs=1e-5)


def _job(upstream: dict[str, np.ndarray], case: str) -> dict:
    return json.loads(str(upstream["geometry_jobs"]))[case]


def _featurize(upstream: dict[str, np.ndarray], case: str) -> dict:
    features = featurize_protein_json(_job(upstream, case), n_queries=2, n_keys=4)
    np.testing.assert_array_equal(
        features["output_atom_name"], upstream[f"geometry_{case}_atom_name"]
    )
    np.testing.assert_array_equal(
        np.asarray(features["output_atom_polymer_type"]) == "non-polymer",
        upstream[f"geometry_{case}_hetero"],
    )
    return features


def _assert_upstream_geometry(
    upstream: dict[str, np.ndarray], case: str, result: dict, *, bounds_atol: float
) -> None:
    for key in GEOMETRY_KEYS:
        expected = upstream[f"geometry_{case}_{key}"]
        actual = np.asarray(result[key])
        if expected.size == 0:
            assert actual.size == 0, key
            continue
        assert actual.shape == expected.shape, key
        if key in _BOUNDS:
            np.testing.assert_allclose(actual, expected, atol=bounds_atol, err_msg=key)
        else:
            np.testing.assert_array_equal(actual, expected, err_msg=key)


def _no_ccd_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(code, **_kwargs):
        raise AssertionError(f"the CCD cache was opened for {code!r}")

    monkeypatch.setattr(
        "foldjax.models.protenix.data.featurize_json._external_ccd_molecule", refuse
    )


@pytest.mark.parametrize("case", ["smiles", "protein"])
def test_geometry_matches_upstreams_featurizer(upstream, case, monkeypatch) -> None:
    """Standard residues skipped; SMILES ligands from their hydrogenated graph."""

    features = _featurize(upstream, case)
    _no_ccd_cache(monkeypatch)

    result = prepare_tfg_features(features)

    _assert_upstream_geometry(upstream, case, result, bounds_atol=1e-6)


def _use_official_ccd_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    root = next(
        (
            parent / "protenix" / "common"
            for parent in Path(__file__).resolve().parents
            if (parent / "protenix" / "common" / "components.cif").is_file()
        ),
        None,
    )
    if root is None or not (root / "components.cif.rdkit_mol.pkl").is_file():
        pytest.skip("official components.cif/RDKit CCD assets are unavailable")
    monkeypatch.setenv("PROTENIX_CCD_COMPONENTS_FILE", str(root / "components.cif"))
    monkeypatch.setenv(
        "PROTENIX_CCD_RDKIT_MOL_FILE", str(root / "components.cif.rdkit_mol.pkl")
    )


def test_ccd_geometry_matches_upstreams_featurizer(upstream, monkeypatch) -> None:
    """A CCD ligand and a modified residue from the cache; the metal ion skipped.

    The bounds tolerance is RDKit's, not the port's: on the identical cached
    ATP molecule, `GetMoleculeBoundsMatrix` of RDKit 2026.03.5 (FoldJAX) and
    2025.09.3 (upstream's environment) differ on 29 of its 465 heavy-atom
    pairs, by at most 0.052 A. Every index, flag and torsion is exact.
    """

    _use_official_ccd_assets(monkeypatch)
    features = _featurize(upstream, "ccd")

    result = prepare_tfg_features(features)

    _assert_upstream_geometry(upstream, "ccd", result, bounds_atol=0.06)
    statuses = {
        record["res_name"]: record["status"]
        for record in result["geometry_provenance"]["residues"]
    }
    assert statuses == {"SEP": "featurized", "ATP": "featurized", "MG": "metal"}


def test_a_guided_step_leaves_a_protein_only_job_bonded() -> None:
    """No constraint reaches a standard residue, so refinement is the identity.

    The per-chain featurization this replaces gave every unbonded atom pair a
    clash floor and stretched N-CA from 1.45 to 2.96 A in one step.
    """

    job = {
        "name": "protein",
        "modelSeeds": [101],
        "sequences": [{"proteinChain": {"sequence": "ACDEFGHIK", "count": 1}}],
    }
    features = featurize_protein_json(job, n_queries=2, n_keys=4)
    guided = prepare_tfg_features(features)
    arrays = {
        key: jnp.asarray(value)
        for key, value in guided.items()
        if isinstance(value, np.ndarray) and value.dtype.kind in "biuf"
    }
    tokens = np.asarray(features["atom_to_token_idx"])
    names = np.asarray(features["output_atom_name"])
    # Each residue's own reference conformer, residues laid 6 A apart.
    coords = (
        np.asarray(features["ref_pos"]) + tokens[:, None] * np.asarray([6.0, 0.0, 0.0])
    ).astype(np.float32)
    engine = TFGEngine(parse_tfg_config(upstream_guidance_config()))

    refined = np.asarray(
        engine.refine(jnp.asarray(coords[None]), arrays, t=0.25, step_i=150)
    )[0]

    nitrogen, alpha = names == "N", names == "CA"
    before = np.linalg.norm(coords[nitrogen] - coords[alpha], axis=-1)
    after = np.linalg.norm(refined[nitrogen] - refined[alpha], axis=-1)
    np.testing.assert_array_equal(after, before)
    np.testing.assert_array_equal(refined, coords)
