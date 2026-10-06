"""Upstream's full confidence outputs: expected PAE/PDE, gPDE, chain pTM, bespoke ipTM.

Two kinds of evidence. ``fixtures/full_confidence_upstream.npz`` holds upstream
OpenFold3 v0.5.0's own CPU results (``scripts/full_confidence_upstream_fixture.py``)
for the arithmetic, and for the RASA disorder term of the ranking score. The
tiny monkeypatched program from ``test_pae_metric_sinking`` then checks that
every confidence schedule returns the same values and that turning the arrays
on changes no coordinate.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3 import inference, rasa
from foldjax.models.openfold3.models.confidence import compute_chain_pair_iptm
from tests.models.openfold3.test_pae_metric_sinking import (
    PAE_BINS,
    _install_tiny_predict,
    _tiny_inference_config,
)

FIXTURE = Path(__file__).parent / "fixtures" / "full_confidence_upstream.npz"
PTM = {"bin_min": 0.0, "bin_max": 32.0, "no_bins": 64}
#: float32 softmax/expectation against torch's, summed over 64 bins of 0-32 A.
ATOL = 2e-5


@pytest.fixture(scope="module")
def upstream():
    with np.load(FIXTURE) as archive:
        return {name: archive[name] for name in archive.files}


def test_expected_errors_and_gpde_match_upstream(upstream) -> None:
    pae = inference._expected_pair_error(
        jnp.asarray(upstream["conf_pae_logits"]), bin_max=32.0
    )
    pde = inference._expected_pair_error(
        jnp.asarray(upstream["conf_pde_logits"]), bin_max=inference.PDE_BIN_MAX
    )
    n_token = pae.shape[-1]
    contact = inference._contact_probabilities(
        jnp.asarray(upstream["conf_distogram_logits"]), jnp.ones((n_token, n_token))
    )
    np.testing.assert_allclose(pae, upstream["conf_pae"], rtol=0, atol=ATOL)
    np.testing.assert_allclose(pde, upstream["conf_pde"], rtol=0, atol=ATOL)
    np.testing.assert_allclose(contact, upstream["conf_contact_probs"], atol=1e-6)
    np.testing.assert_allclose(
        inference._global_pde(pde, contact), upstream["conf_gpde"], atol=ATOL
    )


def test_padded_tokens_do_not_weight_the_global_pde(upstream) -> None:
    pde = jnp.asarray(upstream["conf_pde"])
    logits = jnp.asarray(upstream["conf_distogram_logits"])
    padded_pde = jnp.pad(pde, ((0, 0), (0, 3), (0, 3)), constant_values=31.0)
    padded_logits = jnp.pad(logits, ((0, 3), (0, 3), (0, 0)))
    real = jnp.arange(pde.shape[-1] + 3) < pde.shape[-1]
    contact = inference._contact_probabilities(
        padded_logits, real[:, None] & real[None, :]
    )
    np.testing.assert_allclose(
        inference._global_pde(padded_pde, contact), upstream["conf_gpde"], atol=ATOL
    )


def test_chain_scores_match_upstream_on_every_schedule(upstream) -> None:
    logits = jnp.asarray(upstream["conf_pae_logits"])
    has_frame = jnp.asarray(upstream["conf_has_frame"])
    asym_id = jnp.asarray(upstream["conf_asym_id"])
    is_ligand = jnp.asarray(upstream["conf_is_ligand"])
    token_mask = jnp.ones(asym_id.shape[0])
    n_chain = 3

    # The serial schedule's compact reduction, one sample per map step.
    _, _, chain_pair, chain_ptm = jax.lax.map(
        lambda one: jax.tree.map(
            lambda leaf: leaf[0],
            inference._compact_confidence_metrics_from_pae(
                one[0][None],
                one[1][None],
                token_mask,
                asym_id,
                n_chain=n_chain,
                return_chain_ptm=True,
                **PTM,
            ),
        ),
        (logits, has_frame),
    )
    np.testing.assert_allclose(chain_ptm, upstream["conf_chain_ptm"], atol=ATOL)
    np.testing.assert_allclose(chain_pair, upstream["conf_chain_pair_iptm"], atol=ATOL)
    # The batched schedule's helpers on all samples at once.
    batched_pair = compute_chain_pair_iptm(
        logits, has_frame, token_mask, asym_id, n_chain=n_chain, **PTM
    )
    bespoke = inference._bespoke_iptm(
        batched_pair, has_frame, token_mask, asym_id, is_ligand, n_chain=n_chain
    )
    np.testing.assert_allclose(bespoke, upstream["conf_bespoke_iptm"], atol=ATOL)
    # Sample 1's second chain has no frame: per-sample averaging is what upstream
    # does, and a frame pooled over samples would give a different answer here.
    assert not np.allclose(
        upstream["conf_bespoke_iptm"][1, 0, 2], upstream["conf_bespoke_iptm"][0, 0, 2]
    )


def _tiny(monkeypatch, **changes):
    batch, params = _install_tiny_predict(monkeypatch)
    batch["is_ligand"] = jnp.asarray([[0, 0, 0, 1]], dtype=jnp.int32)
    config = _tiny_inference_config(per_sample_token_cutoff=3, returned_pair_logits=())
    return batch, params, config._replace(**changes)


def _predict(batch, params, config):
    return inference.predict(jax.random.key(5), batch, params, config, None, n_chain=3)


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"returned_pair_logits": ("pae_logits", "pde_logits")},
        {"per_sample_token_cutoff": None},
    ],
    ids=["serial-sink", "serial-with-logits", "batched"],
)
def test_every_schedule_returns_the_same_full_confidence(monkeypatch, changes) -> None:
    batch, params, base = _tiny(monkeypatch)
    reference = _predict(batch, params, base)
    candidate = _predict(batch, params, base._replace(**changes))
    samples, n_token = base.num_samples, base.n_token

    assert reference.pae.shape == reference.pde.shape == (samples, n_token, n_token)
    assert reference.gpde.shape == (samples,)
    assert reference.chain_ptm.shape == (samples, 3)
    assert reference.bespoke_iptm.shape == (samples, 3, 3)
    for name in ("pae", "pde", "gpde", "chain_ptm", "bespoke_iptm"):
        np.testing.assert_allclose(
            getattr(candidate, name), getattr(reference, name), atol=1e-5, err_msg=name
        )
    if candidate.pae_logits is not None:
        expected = inference._expected_pair_error(candidate.pae_logits, bin_max=32.0)
        np.testing.assert_allclose(candidate.pae, expected, atol=1e-5)
        expected = inference._expected_pair_error(candidate.pde_logits, bin_max=32.0)
        np.testing.assert_allclose(candidate.pde, expected, atol=1e-5)


def test_one_sample_takes_the_batched_path_and_still_returns_them(monkeypatch) -> None:
    batch, params, base = _tiny(monkeypatch, num_samples=1)
    # The harness's frame mask is written for three samples.
    monkeypatch.setattr(
        inference,
        "token_frame_atoms",
        lambda *args, **kwargs: ((None, None, None), jnp.ones((1, base.n_token), bool)),
    )
    result = _predict(batch, params, base)
    assert result.pae.shape == (1, base.n_token, base.n_token)
    assert result.gpde.shape == (1,)


def test_turning_the_arrays_off_changes_no_coordinate(monkeypatch) -> None:
    batch, params, base = _tiny(monkeypatch)
    on = _predict(batch, params, base)
    off = _predict(batch, params, base._replace(return_expected_errors=False))

    assert off.pae is off.pde is off.gpde is None
    for name in ("coordinates", "plddt", "ptm", "iptm", "chain_pair_iptm"):
        np.testing.assert_array_equal(
            np.asarray(getattr(on, name)), np.asarray(getattr(off, name)), err_msg=name
        )
    np.testing.assert_array_equal(on.chain_ptm, off.chain_ptm)


def test_released_config_returns_them_by_default() -> None:
    config = inference.released_config(n_token=16, n_atom=64)
    assert config.return_expected_errors is True
    assert (
        inference.released_config(
            n_token=16, n_atom=64, return_expected_errors=False
        ).return_expected_errors
        is False
    )


def test_pde_bins_follow_the_logit_width() -> None:
    logits = jnp.zeros((1, 2, 2, PAE_BINS))
    # A uniform distribution's expectation is the mean bin centre: half the range.
    np.testing.assert_allclose(
        inference._expected_pair_error(logits, bin_max=32.0), 16.0, atol=1e-5
    )


# --- RASA disorder ---------------------------------------------------------------


def test_rasa_constants_match_upstream(upstream) -> None:
    assert rasa.WINDOW == int(upstream["rasa_window"])
    assert rasa.DEFAULT_MAX_ACC == float(upstream["rasa_default_max_acc"])
    assert rasa.DISORDER_THRESHOLD == float(upstream["rasa_threshold"])
    assert rasa.DISORDER_THRESHOLD == float(upstream["rasa_process_threshold"])
    assert rasa.VDW_RADII == str(upstream["rasa_vdw_radii"])
    assert (
        dict(
            zip(
                upstream["rasa_sander_names"].tolist(),
                upstream["rasa_sander_values"].tolist(),
                strict=True,
            )
        )
        == rasa.SANDER_MAX_ACC
    )
    np.testing.assert_allclose(
        rasa.smooth_rasa(np.linspace(0.0, 1.0, 8)), upstream["rasa_smoothed_probe"]
    )


def test_rasa_constants_match_the_vendored_upstream_scale() -> None:
    from foldjax.models.openfold3._upstream.openfold3.core.data.resources import (
        residues,
    )

    assert residues.RESIDUE_SASA_SCALES["Sander"] == rasa.SANDER_MAX_ACC


def _labels(upstream):
    return {
        "atom_name": upstream["rasa_atom_name"],
        "element": upstream["rasa_element"],
        "residue_name": upstream["rasa_res_name"],
        "residue_id": upstream["rasa_res_id"],
        "chain_id": upstream["rasa_chain_id"],
        "is_protein": upstream["rasa_is_protein"],
    }


def test_disorder_matches_upstream(upstream) -> None:
    pytest.importorskip("biotite")
    labels = _labels(upstream)
    residue_rasa = rasa.protein_residue_rasa(upstream["rasa_coords"][:1], **labels)
    # One short chain (8 residues, under the 12-residue reflect half-window)
    # beside the 129-residue one; the ligand copy is filtered out.
    assert residue_rasa[0].shape == upstream["rasa_residue_rasa_0"].shape == (137,)
    np.testing.assert_allclose(
        residue_rasa[0], upstream["rasa_residue_rasa_0"], rtol=0, atol=1e-4
    )
    # A fraction of 137 residues; upstream returns it as float32.
    np.testing.assert_allclose(
        rasa.protein_disorder(upstream["rasa_coords"], **labels),
        upstream["rasa_disorder"],
        rtol=0,
        atol=1e-6,
    )


def test_disorder_without_protein_is_nan(upstream) -> None:
    pytest.importorskip("biotite")
    labels = _labels(upstream)
    labels["is_protein"] = np.zeros_like(labels["is_protein"])
    assert np.isnan(rasa.protein_disorder(upstream["rasa_coords"], **labels)).all()


def test_missing_biotite_keeps_the_partial_score(monkeypatch) -> None:
    from foldjax.models.openfold3 import output
    from tests.models.openfold3.test_output import _prediction, _ranking_features

    monkeypatch.setattr(rasa, "unavailable_reason", lambda: "biotite is absent")
    summary = output.confidence_summary(
        _prediction(64, n_samples=2), _ranking_features(64, has_protein=True)
    )
    assert summary["disorder_unavailable"] == "biotite is absent"
    assert "ranked_samples" not in summary
    for entry in summary["samples"]:
        assert "sample_ranking_score" not in entry
        assert "sample_ranking_score_no_disorder" in entry


def test_protein_ranking_score_includes_the_disorder_term(upstream) -> None:
    pytest.importorskip("biotite")
    from foldjax.models.openfold3 import output
    from tests.test_mmcif_label_fields import openfold3_case

    prediction, features, metadata = openfold3_case(samples=2)
    coordinates = np.zeros((2, 5, 3), dtype=np.float32)
    coordinates[:, :, 0] = np.arange(5) * 3.8
    coordinates[1] *= 2.0
    prediction = prediction._replace(coordinates=coordinates)
    exact = output.atom_metadata(features, metadata)
    summary = output.confidence_summary(prediction, features, metadata=exact)

    assert "disorder_unavailable" not in summary
    expected = rasa.protein_disorder(
        coordinates,
        atom_name=exact.name,
        element=exact.element,
        residue_name=exact.residue_name,
        residue_id=exact.residue_id,
        chain_id=exact.chain_id,
        is_protein=np.asarray(exact.molecule_type_id) == 0,
    )
    for entry, disorder in zip(summary["samples"], expected, strict=True):
        assert entry["disorder"] == pytest.approx(disorder)
        assert entry["sample_ranking_score"] == pytest.approx(
            0.8 * entry["iptm"]
            + 0.2 * entry["ptm"]
            + 0.5 * disorder
            - 100.0 * entry["has_clash"]
        )
    assert summary["ranked_samples"] == sorted(
        range(2), key=lambda index: -summary["samples"][index]["sample_ranking_score"]
    )
