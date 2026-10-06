"""OpenFold3 pocket-guided sampling: featurization, proposal search, wiring.

Upstream v0.5.0 turns a query's ``pocket_constraint`` into ``pocket_sampling_*``
features and a second, partial diffusion rollout seeded from ligand poses
proposed in the pocket. ``fixtures/pocket_upstream.npz`` holds upstream's own
CPU outputs (``scripts/pocket_upstream_fixture.py``): its features for two
queries and its proposal search on three synthetic cases, with every random
draw recorded so the JAX search can replay them.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax import memory_policy
from foldjax.models.openfold3 import inference
from foldjax.models.openfold3.data import (
    featurize_query,
    load_feature_archive,
    pocket_sampling_config,
    save_features,
)
from foldjax.models.openfold3.data import pocket_constraints as pocket_data
from foldjax.models.openfold3.models.pocket_constraints import (
    PocketProposalDraws,
    PocketSamplingConfig,
    PocketSamplingInputs,
    build_pocket_sampling_seeds,
    draw_pocket_proposals,
)
from foldjax.models.openfold3.models.sampler import sample_diffusion

FIXTURE = Path(__file__).parent / "fixtures" / "pocket_upstream.npz"


def _upstream() -> dict[str, np.ndarray]:
    with np.load(FIXTURE) as loaded:
        return {name: loaded[name] for name in loaded.files}


def _query_spec(name: str) -> dict:
    """One of the fixture's queries, in upstream's query-set wrapper."""
    queries = json.loads(str(_upstream()["feature_queries"]))
    return {"queries": {name: queries[name]}}


def _vendored_query_and_structure(name: str):
    from foldjax.models.openfold3._upstream.openfold3.core.data.primitives.structure.query import (  # noqa: E501
        structure_with_ref_mols_from_query,
    )
    from foldjax.models.openfold3.data.featurize import _query_set

    query = _query_set(_query_spec(name)).queries[name]
    return query, structure_with_ref_mols_from_query(query)


def _settings(**overrides):
    from foldjax.models.openfold3._upstream.openfold3.core.config.pocket_sampling_config import (  # noqa: E501
        PocketSamplingSettings,
    )

    return PocketSamplingSettings(**overrides)


# --- featurization -----------------------------------------------------------


@pytest.mark.parametrize("name", ["smiles", "ccd_order"])
def test_features_match_upstream_bit_for_bit(name) -> None:
    upstream = _upstream()
    prefix = f"feature_{name}_"
    expected = {
        key[len(prefix) :]: value
        for key, value in upstream.items()
        if key.startswith(prefix)
    }
    query, structure = _vendored_query_and_structure(name)
    ours = pocket_data.create_pocket_sampling_features(
        query=query,
        atom_array=structure.atom_array,
        processed_reference_molecules=structure.processed_reference_mols,
        settings=_settings(rdkit_num_conformers=0),
    )
    assert set(ours) == set(expected)
    for key, value in expected.items():
        assert ours[key].dtype == value.dtype, key
        np.testing.assert_array_equal(ours[key], value, err_msg=key)


def test_featurize_query_carries_upstreams_released_settings() -> None:
    features = featurize_query(_query_spec("smiles"))
    config = pocket_sampling_config(features)
    assert config == PocketSamplingConfig(
        n_ligand_atoms=12,
        n_pocket_atoms=int(features["pocket_sampling_pocket_atom_mask"].sum()),
        n_conformers=features["pocket_sampling_conformer_rels"].shape[1],
        num_parents=16,
        candidates=1024,
        start_frac=0.75,
        ligand_jitter=0.25,
        center_jitter=4.0,
        surface_jitter=1.5,
        vdw_buffer=float(np.float32(0.225)),
        diversity_rmsd=0.5,
        contact_distance=4.0,
    )
    rels = features["pocket_sampling_conformer_rels"][0]
    assert 1 <= rels.shape[0] <= 32 and rels.shape[1:] == (12, 3)
    np.testing.assert_allclose(rels.mean(axis=1), 0.0, atol=1e-5)
    assert config.start_step(200) == 150


def test_a_query_without_a_constraint_gets_no_pocket_features() -> None:
    spec = _query_spec("smiles")
    del spec["queries"]["smiles"]["pocket_constraint"]
    features = featurize_query(spec)
    assert not pocket_data.has_pocket_sampling_features(features)
    assert pocket_sampling_config(features) is None


def test_disabled_settings_produce_no_features() -> None:
    query, structure = _vendored_query_and_structure("smiles")
    assert (
        pocket_data.create_pocket_sampling_features(
            query, structure.atom_array, settings=_settings(enabled=False)
        )
        == {}
    )


def test_a_pocket_residue_without_atoms_is_refused() -> None:
    spec = _query_spec("smiles")
    spec["queries"]["smiles"]["pocket_constraint"]["pocket_residues"].append(["A", 99])
    with pytest.raises(ValueError, match="A:99 does not match any atoms"):
        featurize_query(spec)


def test_unknown_elements_take_the_carbon_radius() -> None:
    query, structure = _vendored_query_and_structure("smiles")
    atoms = structure.atom_array.copy()
    atoms.element[0] = "Xx"
    features = pocket_data.create_pocket_sampling_features(
        query, atoms, settings=_settings(rdkit_num_conformers=0)
    )
    assert features["pocket_sampling_vdw_radii"][0] == pytest.approx(1.70)


def test_conformer_failure_falls_back_to_parent_conformations(monkeypatch) -> None:
    query, structure = _vendored_query_and_structure("smiles")

    def fail(**_kwargs):
        raise ValueError("bad mapping")

    monkeypatch.setattr(pocket_data, "_atom_order_from_reference_molecule", fail)
    features = pocket_data.create_pocket_sampling_features(
        query, structure.atom_array, structure.processed_reference_mols
    )
    assert pocket_data.POCKET_SAMPLING_CONFORMERS not in features
    assert "pocket_sampling_ligand_atom_mask" in features


def test_archives_keep_the_constraint(tmp_path) -> None:
    features = featurize_query(_query_spec("smiles"))
    path = save_features(features, tmp_path / "pocket.npz")
    loaded, _table, _metadata = load_feature_archive(path)
    assert pocket_sampling_config(loaded) == pocket_sampling_config(features)


def test_a_malformed_archive_is_refused_at_the_boundary(tmp_path) -> None:
    features = featurize_query(_query_spec("smiles"))
    features["pocket_sampling_vdw_radii"] = features["pocket_sampling_vdw_radii"][
        :, :-1
    ]
    path = save_features(features, tmp_path / "pocket.npz")
    with pytest.raises(ValueError, match="pocket_sampling_vdw_radii"):
        load_feature_archive(path)
    partial = featurize_query(_query_spec("smiles"))
    del partial["pocket_sampling_candidates"]
    with pytest.raises(ValueError, match="pocket_sampling_candidates"):
        pocket_sampling_config(partial)


# --- proposal search ---------------------------------------------------------


@pytest.mark.parametrize(
    "case", ["conformers", "parent_conformation", "fill_duplicates"]
)
def test_seed_search_replays_upstreams_draws(case) -> None:
    upstream = _upstream()
    prefix = f"seed_{case}_"
    n_conf, num_parents, candidates, diversity, _samples = upstream[f"{prefix}settings"]
    lig = upstream[f"{prefix}ligand_atom_mask"]
    pocket = upstream[f"{prefix}pocket_atom_mask"]
    config = PocketSamplingConfig(
        n_ligand_atoms=int(lig.sum()),
        n_pocket_atoms=int(pocket.sum()),
        n_conformers=int(n_conf),
        num_parents=int(num_parents),
        candidates=int(candidates),
        start_frac=0.75,
        ligand_jitter=0.25,
        center_jitter=4.0,
        surface_jitter=1.5,
        vdw_buffer=float(np.float32(0.225)),
        diversity_rmsd=float(np.float32(diversity)),
        contact_distance=4.0,
    )
    inputs = PocketSamplingInputs(
        ligand_atom_mask=jnp.asarray(lig),
        pocket_atom_mask=jnp.asarray(pocket),
        vdw_radii=jnp.asarray(upstream[f"{prefix}vdw_radii"]),
        conformer_rels=(
            jnp.asarray(upstream[f"{prefix}conformer_rels"]) if n_conf else None
        ),
    )
    draws = PocketProposalDraws(
        *(
            jnp.asarray(upstream[f"{prefix}draw_{name}"])
            for name in PocketProposalDraws._fields
        )
    )
    xl = jnp.asarray(upstream[f"{prefix}xl"])
    seeds = jax.jit(build_pocket_sampling_seeds, static_argnums=3)(
        xl, jnp.ones(xl.shape[1]), inputs, config, draws
    )
    # Same proposals chosen in the same order; the rest is float32 rounding
    # (upstream's cdist uses the matmul expansion above 25 rows).
    np.testing.assert_allclose(seeds, upstream[f"{prefix}seeds"], atol=1e-4, rtol=0)


def test_ties_break_in_upstreams_key_order() -> None:
    """Each secondary key decides only among candidates tied on the ones before.

    Continuous centroid distances almost never tie, so the replay cases above
    cannot see the secondary keys; here every key is tied in turn.
    """
    from foldjax.models.openfold3.models.pocket_constraints import _candidate_order

    rng = np.random.default_rng(0)
    rows = rng.integers(0, 2, size=(64, 5)).astype(np.float32)
    com, vdw, min_prot, lig_atoms, contact = rows.T
    # upstream _candidate_sort_key, applied by Python's stable sort
    expected = sorted(
        range(64),
        key=lambda i: (com[i], -lig_atoms[i], contact[i], vdw[i], -min_prot[i]),
    )
    order = _candidate_order(*(jnp.asarray(column) for column in rows.T))
    np.testing.assert_array_equal(order, expected)


def test_generated_conformer_candidates_are_rigid_and_centred() -> None:
    """Upstream's test_build_pocket_sampling_seeds_uses_generated_conformer..."""
    xl = jnp.zeros((2, 5, 3)).at[:, 2].set(jnp.array([20.0, 0.0, 0.0]))
    xl = xl.at[:, 3].set(jnp.array([10.0, 0.0, 0.0]))
    xl = xl.at[:, 4].set(jnp.array([11.0, 0.0, 0.0]))
    conformer = jnp.array([[[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    config = PocketSamplingConfig(
        n_ligand_atoms=2,
        n_pocket_atoms=2,
        n_conformers=1,
        num_parents=2,
        candidates=6,
        start_frac=0.5,
        ligand_jitter=0.0,
        center_jitter=0.0,
        surface_jitter=0.0,
        vdw_buffer=0.0,
        diversity_rmsd=0.0,
        contact_distance=4.0,
    )
    inputs = PocketSamplingInputs(
        ligand_atom_mask=jnp.array([0.0, 0, 0, 1, 1]),
        pocket_atom_mask=jnp.array([1.0, 1, 0, 0, 0]),
        vdw_radii=jnp.full((5,), 1.7),
        conformer_rels=conformer,
    )
    draws = draw_pocket_proposals(
        jax.random.key(0), candidates=6, n_conformers=1, n_pocket_atoms=2
    )
    seeds = build_pocket_sampling_seeds(xl, jnp.ones(5), inputs, config, draws)
    ligand = seeds[:, 3:5]
    np.testing.assert_allclose(
        jnp.linalg.norm(ligand.mean(axis=1), axis=-1), 0, atol=1e-5
    )
    np.testing.assert_allclose(
        jnp.linalg.norm(ligand[:, 0] - ligand[:, 1], axis=-1), 4.0, atol=1e-5
    )
    np.testing.assert_array_equal(seeds[:, :3], xl[:, :3])


# --- sampler and predict wiring ------------------------------------------------


def test_x_start_resumes_a_rollout_from_given_coordinates() -> None:
    start = jnp.arange(24, dtype=jnp.float32).reshape(2, 4, 3)
    schedule = jnp.array([0.5, 0.25, 0.1])
    out = sample_diffusion(
        jax.random.key(1),
        schedule,
        (2, 4, 3),
        lambda x, t: x,  # identity denoiser: zero update
        gamma_0=0.0,
        gamma_min=1.0,
        noise_scale=0.0,
        step_scale=1.0,
        x_start=start,
    )
    init_key = jax.random.split(jax.random.key(1), 3)[0]
    expected = start + 0.5 * jax.random.normal(init_key, (2, 4, 3))
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match="x_start cannot be combined"):
        sample_diffusion(
            jax.random.key(1),
            schedule,
            (2, 4, 3),
            lambda x, t: x,
            gamma_0=0.0,
            gamma_min=1.0,
            noise_scale=0.0,
            step_scale=1.0,
            x_start=start,
            noise_mask=jnp.ones((2, 4)),
        )


_POCKET = PocketSamplingConfig(
    n_ligand_atoms=2,
    n_pocket_atoms=2,
    n_conformers=0,
    num_parents=16,
    candidates=8,
    start_frac=0.5,
    ligand_jitter=0.25,
    center_jitter=4.0,
    surface_jitter=1.5,
    vdw_buffer=0.225,
    diversity_rmsd=0.5,
    contact_distance=4.0,
)


def _wiring_config(pocket):
    return inference.InferenceConfig(
        n_atom=6,
        n_token=3,
        num_samples=3,
        num_steps=4,
        n_query=2,
        n_key=4,
        atom_heads=1,
        token_heads=1,
        no_heads_msa=1,
        no_heads_pair=1,
        no_heads_pair_bias=1,
        max_relative_idx=2,
        max_relative_chain=2,
        num_recycles=1,
        max_atoms_per_token=2,
        plddt_bins=4,
        pae_bins=4,
        pae_bin_max=4.0,
        msa_depth=None,
        pocket_sampling=pocket,
    )


def _wiring_batch(pocket: bool):
    batch = {"atom_mask": jnp.ones((1, 6)), "token_mask": jnp.ones((1, 3))}
    if pocket:
        batch.update(
            pocket_sampling_ligand_atom_mask=jnp.array([[0.0, 0, 0, 0, 1, 1]]),
            pocket_sampling_pocket_atom_mask=jnp.array([[1.0, 1, 0, 0, 0, 0]]),
            pocket_sampling_vdw_radii=jnp.full((1, 6), 1.7),
            pocket_sampling_candidates=jnp.array([[8]]),
        )
    return batch


class _SamplerDoneError(Exception):
    pass


def _run_wiring(monkeypatch, pocket, *, calls_before_stop, jit=False):
    single, pair = jnp.zeros((1, 3, 4)), jnp.zeros((1, 3, 3, 4))
    monkeypatch.setattr(inference, "trunk", lambda *a, **k: (single, single, pair))
    monkeypatch.setattr(inference, "pair_conditioning", lambda *a, **k: pair)
    monkeypatch.setattr(inference, "single_conditioning", lambda *a, **k: single)
    monkeypatch.setattr(inference, "denoise", lambda batch, x, *a, **k: x * 0.25)
    calls = []

    def observe(key, schedule, shape, denoiser, **kwargs):
        result = sample_diffusion(key, schedule, shape, denoiser, **kwargs)
        calls.append(
            {
                "schedule_length": schedule.shape[0],
                "x_start": kwargs.get("x_start"),
                "result": result,
            }
        )
        if len(calls) == calls_before_stop:
            raise _SamplerDoneError
        return result

    monkeypatch.setattr(inference, "sample_diffusion", observe)
    params = SimpleNamespace(trunk=None, diffusion_conditioning=None, denoiser=None)
    config = _wiring_config(_POCKET if pocket else None)

    def run(key, batch):
        return inference.predict(key, batch, params, config, None)

    with pytest.raises(_SamplerDoneError):
        (jax.jit(run) if jit else run)(jax.random.key(7), _wiring_batch(pocket))
    return calls


def test_pocket_runs_a_second_rollout_from_the_schedule_tail(monkeypatch) -> None:
    first, second = _run_wiring(monkeypatch, True, calls_before_stop=2)
    assert first["x_start"] is None and first["schedule_length"] == 5
    # start_step = round(0.5 * 4) = 2: steps 2 and 3 remain.
    assert second["schedule_length"] == 3
    seeds = np.asarray(second["x_start"])
    parents = np.asarray(first["result"])
    assert seeds.shape == (3, 6, 3)
    for seed in seeds:
        # Non-ligand atoms come from one parent; the ligand is a rigid copy
        # of a parent ligand (no RDKit conformers here), plus a translation.
        assert any(np.allclose(seed[:4], parent[:4]) for parent in parents)
        bond = np.linalg.norm(seed[4] - seed[5])
        assert np.isclose(
            bond, np.linalg.norm(parents[:, 4] - parents[:, 5], axis=-1), atol=1e-4
        ).any()


def test_pocket_search_traces_under_jit(monkeypatch) -> None:
    _first, second = _run_wiring(monkeypatch, True, calls_before_stop=2, jit=True)
    assert second["schedule_length"] == 3


def test_the_first_rollout_is_untouched_by_a_pocket(monkeypatch) -> None:
    (plain,) = _run_wiring(monkeypatch, False, calls_before_stop=1)
    (constrained, _second) = _run_wiring(monkeypatch, True, calls_before_stop=2)
    np.testing.assert_array_equal(plain["result"], constrained["result"])


def test_features_and_config_must_agree() -> None:
    params = SimpleNamespace(trunk=None)
    with pytest.raises(ValueError, match="pocket_sampling=None"):
        inference.predict(
            jax.random.key(0), _wiring_batch(True), params, _wiring_config(None), None
        )
    with pytest.raises(ValueError, match="lacks"):
        inference.predict(
            jax.random.key(0),
            _wiring_batch(False),
            params,
            _wiring_config(_POCKET),
            None,
        )
    with pytest.raises(ValueError, match="cannot be combined with noise_fn"):
        inference.predict(
            jax.random.key(0),
            _wiring_batch(True),
            params,
            _wiring_config(_POCKET),
            None,
            noise_mask=jnp.ones((3, 6)),
        )


# --- admission and the adapter -------------------------------------------------


def test_a_pocket_run_is_admitted_as_unknown_not_fits() -> None:
    budget = memory_policy.resolve_budget(
        pool_bytes=90 * 2**30, card_bytes=96 * 2**30, override_gib=None
    )
    memory_policy.reset_warnings()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inference.released_config(
            n_token=512, n_atom=512 * 24, memory_budget=budget, pocket_sampling=_POCKET
        )
    block = memory_policy.recorded()
    assert block["state"] == "unknown"
    assert block["exceeds_profile"] == ["pocket-guided sampling"]


@pytest.mark.parametrize(
    ("padding", "options", "named"),
    [(True, {}, "padding"), (None, {"cp_devices": 2}, "cp_devices > 1")],
)
def test_the_adapter_refuses_options_the_pocket_sampler_lacks(
    tmp_path, monkeypatch, padding, options, named
) -> None:
    from foldjax.backends import openfold3 as adapter
    from foldjax.schema import PredictionRequest

    features = {
        "token_mask": np.ones((1, 8), dtype=np.float32),
        "atom_mask": np.ones((1, 16), dtype=np.float32),
    }
    modules = {
        "foldjax.models.openfold3.data": SimpleNamespace(
            featurize_query_with_metadata=lambda *args, **kwargs: (features, None),
            pocket_sampling_config=lambda batch: _POCKET,
        ),
        "foldjax.models.openfold3.inference": SimpleNamespace(),
        "foldjax.models.openfold3.output": SimpleNamespace(),
        "foldjax.models.openfold3.bridge.chemistry": SimpleNamespace(),
        "foldjax.models.openfold3.bridge.checkpoint": SimpleNamespace(),
        "foldjax.models.openfold3.bridge.torch_mapping": SimpleNamespace(),
        "jax": SimpleNamespace(),
    }
    monkeypatch.setattr(adapter, "import_module", lambda name: modules[name])
    job = tmp_path / "query.json"
    job.write_text('{"queries": {}}')
    weights = tmp_path / "openfold3.pt"
    weights.write_bytes(b"native")
    with pytest.raises(ValueError, match=f"pocket_constraint.*{named}"):
        adapter.OpenFold3Backend().predict(
            PredictionRequest(
                model="openfold3",
                input=job,
                input_format="openfold3",
                weights=weights,
                output_dir=tmp_path / "out",
                padding=padding,
                options=options,
            )
        )


def test_a_native_pocket_query_reaches_the_program_and_the_writer(
    tmp_path, monkeypatch
) -> None:
    """The real adapter path: featurize, filter, admit, compact, run, write.

    Only the checkpoint and the compiled program are stand-ins; the latter
    records what it was handed and returns zeros of the right shapes, so the
    real output writer runs on them.
    """
    import importlib

    from foldjax.backends import openfold3 as adapter
    from foldjax.schema import PredictionRequest

    seen: dict = {}

    def fake_compile(config, table, **_kwargs):
        def compiled(key, batch, params, **_options):
            seen["config"] = config
            seen["keys"] = set(batch)
            samples, atoms = config.num_samples, config.n_atom
            return inference.Prediction(
                coordinates=np.zeros((samples, atoms, 3), np.float32),
                plddt=np.full((samples, atoms), 50.0, np.float32),
                ptm=np.full((samples,), 0.5, np.float32),
                iptm=np.full((samples,), 0.5, np.float32),
                chain_pair_iptm=None,
                pae_logits=None,
                pde_logits=None,
                distogram_logits=None,
            )

        return compiled

    inference_proxy = SimpleNamespace(
        **{
            name: getattr(inference, name)
            for name in dir(inference)
            if not name.startswith("__")
        }
    )
    inference_proxy.compile_predict = fake_compile
    inference_proxy.cast_narrow_params = lambda params, *_dtypes: params
    stubs = {
        "foldjax.models.openfold3.inference": inference_proxy,
        "foldjax.models.openfold3.bridge.checkpoint": SimpleNamespace(
            load_checkpoint=lambda path: {}
        ),
        "foldjax.models.openfold3.bridge.torch_mapping": SimpleNamespace(
            resolve_model_prefix=lambda state, prefix=None: "",
            prune_sample_diffusion_aliases=lambda state, *, prefix: 0,
            map_inference_params=lambda state, prefix: object(),
        ),
    }
    monkeypatch.setattr(
        adapter,
        "import_module",
        lambda name: stubs.get(name) or importlib.import_module(name),
    )
    job = tmp_path / "query.json"
    job.write_text(json.dumps(_query_spec("smiles")))
    weights = tmp_path / "openfold3.pt"
    weights.write_bytes(b"native")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = adapter.OpenFold3Backend().predict(
            PredictionRequest(
                model="openfold3",
                input=job,
                input_format="openfold3",
                weights=weights,
                output_dir=tmp_path / "out",
                seed=1,
                options={"num_samples": 2, "num_steps": 4},
            )
        )
    config = seen["config"].pocket_sampling
    assert config is not None and config.n_ligand_atoms == 12
    assert set(pocket_data.POCKET_SAMPLING_ARRAY_FEATURES) <= seen["keys"]
    assert pocket_data.POCKET_SAMPLING_CONFORMERS in seen["keys"]
    assert result.raw["pocket_sampling"]["candidates"] == 1024
    assert len(result.samples) == 2


# --- review follow-ups ---------------------------------------------------------


def test_the_search_never_holds_a_parent_per_candidate() -> None:
    """[candidates, N_atom, 3] parents were 239 MiB of temp at 20,000 atoms.

    Gathering ligand and pocket atoms per sample before per candidate leaves
    under 10 MiB on CPU XLA; the bound sits well between the two.
    """
    atoms, samples, n_lig, n_pocket = 20000, 5, 40, 80
    config = _POCKET._replace(
        n_ligand_atoms=n_lig, n_pocket_atoms=n_pocket, n_conformers=32, candidates=1024
    )
    inputs = PocketSamplingInputs(
        ligand_atom_mask=jnp.zeros(atoms).at[-n_lig:].set(1.0),
        pocket_atom_mask=jnp.zeros(atoms).at[100 : 100 + n_pocket].set(1.0),
        vdw_radii=jnp.full(atoms, 1.7),
        conformer_rels=jnp.zeros((32, n_lig, 3)),
    )
    draws = draw_pocket_proposals(
        jax.random.key(0), candidates=1024, n_conformers=32, n_pocket_atoms=n_pocket
    )
    compiled = (
        jax.jit(build_pocket_sampling_seeds, static_argnums=3)
        .lower(jnp.zeros((samples, atoms, 3)), jnp.ones(atoms), inputs, config, draws)
        .compile()
    )
    parents_bytes = 1024 * atoms * 3 * 4
    assert compiled.memory_analysis().temp_size_in_bytes < parents_bytes // 4


def test_pocket_sampling_is_refused_under_context_parallelism() -> None:
    with pytest.raises(ValueError, match="context parallelism"):
        inference.released_config(
            n_token=64, n_atom=512, cp_shards=2, pocket_sampling=_POCKET
        )
    config = _wiring_config(_POCKET)._replace(cp_shards=2)
    with pytest.raises(ValueError, match="context parallelism"):
        inference._split_pocket_sampling_inputs(
            _wiring_batch(True), config, replay=False
        )
    # Unconstrained CP configs are untouched.
    inference.released_config(n_token=64, n_atom=512, cp_shards=2)


_SAMPLER_VARIANTS = {
    "plain": {},
    "augmented": {"augment": True},
    "chunked": {"diffusion_chunk_size": 2},
    "masked": {"noise_mask": True},
}


def _sampler_call(sampler, variant: str):
    from foldjax.models.openfold3.models.augmentation import (
        centre_random_augmentation,
    )

    options = dict(_SAMPLER_VARIANTS[variant])
    mask = jnp.array([[1.0, 1.0, 0.0, 1.0]] * 3)
    if options.pop("augment", False):
        options["augment_fn"] = lambda key, x: centre_random_augmentation(key, x, mask)
    if options.pop("noise_mask", False):
        options["noise_mask"] = mask.astype(bool)

    def run(key):
        return sampler(
            key,
            jnp.array([4.0, 2.0, 1.0, 0.5]),
            (3, 4, 3),
            lambda x, t: x * 0.25 + 0.1,
            gamma_0=0.8,
            gamma_min=1.0,
            noise_scale=1.003,
            step_scale=1.5,
            **options,
        )

    return jax.jit(run)


@pytest.mark.parametrize("variant", sorted(_SAMPLER_VARIANTS))
def test_an_unconstrained_rollout_is_the_d1a4eb1_rollout(variant) -> None:
    """``x_start=None`` must leave the sampler exactly as it was before it.

    ``_sampler_d1a4eb1.py`` is the pre-pocket sampler, frozen: the lowered
    programs are the same text and the outputs the same bits.
    """
    from . import _sampler_d1a4eb1 as base

    ours = _sampler_call(sample_diffusion, variant)
    theirs = _sampler_call(base.sample_diffusion, variant)
    key = jax.random.key(13)
    assert ours.lower(key).as_text() == theirs.lower(key).as_text()
    np.testing.assert_array_equal(np.asarray(ours(key)), np.asarray(theirs(key)))


def test_an_unconstrained_predict_samples_once_from_the_callers_key(
    monkeypatch,
) -> None:
    keys = []
    real = sample_diffusion

    def observe(key, schedule, shape, denoiser, **kwargs):
        keys.append((key, schedule.shape[0], kwargs.get("x_start")))
        result = real(key, schedule, shape, denoiser, **kwargs)
        raise _SamplerDoneError(result)

    single, pair = jnp.zeros((1, 3, 4)), jnp.zeros((1, 3, 3, 4))
    monkeypatch.setattr(inference, "trunk", lambda *a, **k: (single, single, pair))
    monkeypatch.setattr(inference, "pair_conditioning", lambda *a, **k: pair)
    monkeypatch.setattr(inference, "single_conditioning", lambda *a, **k: single)
    monkeypatch.setattr(inference, "denoise", lambda batch, x, *a, **k: x * 0.25)
    monkeypatch.setattr(inference, "sample_diffusion", observe)
    params = SimpleNamespace(trunk=None, diffusion_conditioning=None, denoiser=None)
    key = jax.random.key(7)
    with pytest.raises(_SamplerDoneError):
        inference.predict(key, _wiring_batch(False), params, _wiring_config(None), None)
    ((seen_key, length, x_start),) = keys
    assert seen_key is key and length == 5 and x_start is None
