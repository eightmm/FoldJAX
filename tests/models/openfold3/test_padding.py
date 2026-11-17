"""OpenFold3's complete token/atom/MSA/template serving profile."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.backends.openfold3 import _padding_plan, _sampler_noise_mask
from foldjax.models.openfold3.bridge.torch_mapping import map_atom_logit_head
from foldjax.models.openfold3.data import (
    collapse_identical_templates,
    compact_zero_template_pair_features,
    pad_features,
)
from foldjax.models.openfold3.data.featurize import _ZERO_TEMPLATE_PAIR_MARKER
from foldjax.models.openfold3.data.validation import validate_features
from foldjax.models.openfold3.inference import Prediction
from foldjax.models.openfold3.models import template_module
from foldjax.models.openfold3.models.confidence import compute_plddt
from foldjax.models.openfold3.models.heads import atom_logit_head
from foldjax.models.openfold3.models.sampler import (
    padded_noise_tape,
    sample_diffusion,
)
from foldjax.models.openfold3.output import confidence_summary, crop_prediction
from foldjax.schema import PaddingConfig
from tests.models.openfold3.feature_fixture import minimal_features


def test_padding_covers_all_four_dynamic_axes() -> None:
    features = minimal_features(tokens=4, atoms=7, msa_rows=2, templates=1)
    padded = pad_features(
        features, n_token=8, n_atom=13, n_msa=5, n_templates=3
    )

    assert padded["token_mask"].shape == (1, 8)
    assert padded["atom_mask"].shape == (1, 13)
    assert padded["msa"].shape == (1, 5, 8, 32)
    assert padded["msa_mask"].shape == (1, 5, 8)
    assert padded["template_restype"].shape == (1, 3, 8, 32)
    assert padded["template_distogram"].shape == (1, 3, 8, 8, 39)
    assert padded["template_unit_vector"].shape == (1, 3, 8, 8, 3)
    assert float(padded["token_mask"].sum()) == 4
    assert float(padded["atom_mask"].sum()) == 7
    assert float(padded["msa_mask"].sum()) == 8
    assert float(padded["template_backbone_frame_mask"].sum()) == 4
    np.testing.assert_array_equal(
        padded["template_padding_mask"], np.asarray([[1.0, 0.0, 0.0]])
    )


def test_zero_template_compaction_runs_after_final_padding_shape() -> None:
    features = minimal_features(tokens=4, atoms=7, msa_rows=2, templates=4)
    features["template_pseudo_beta_mask"].fill(0.0)
    features["template_backbone_frame_mask"].fill(0.0)
    collapsed = collapse_identical_templates(features)
    assert collapsed["template_restype"].shape == (1, 1, 4, 32)
    padded = pad_features(
        collapsed, n_token=8, n_atom=13, n_msa=5, n_templates=3
    )

    compact = compact_zero_template_pair_features(padded)

    assert compact["template_restype"].shape == (1, 3, 8, 32)
    np.testing.assert_array_equal(
        compact["template_padding_mask"], np.asarray([[1.0, 0.0, 0.0]])
    )
    assert compact[_ZERO_TEMPLATE_PAIR_MARKER].shape == ()
    for name in (
        "template_distogram",
        "template_pseudo_beta_mask",
        "template_unit_vector",
        "template_backbone_frame_mask",
    ):
        assert name not in compact
    validate_features(padded)

    repadded = pad_features(
        padded, n_token=8, n_atom=13, n_msa=5, n_templates=4
    )
    np.testing.assert_array_equal(
        repadded["template_padding_mask"],
        np.asarray([[1.0, 0.0, 0.0, 0.0]]),
    )


def test_padding_plan_reports_real_storage_and_standard_targets() -> None:
    features = minimal_features(tokens=4, atoms=7, msa_rows=2, templates=1)
    plan = _padding_plan(features, PaddingConfig())

    assert plan.actual == {"tokens": 4, "atoms": 7, "msa": 2, "templates": 1}
    assert plan.storage == plan.actual
    assert plan.target == {
        "tokens": 128,
        # The stored atoms on the atom grid, not 24 per token (3,072).
        "atoms": 256,
        "msa": 1024,
        # The one stored row, not the released width of four.
        "templates": 1,
    }


def test_padding_plan_accepts_compacted_empty_template_geometry() -> None:
    features = minimal_features(tokens=4, atoms=7, msa_rows=2, templates=4)
    features["template_pseudo_beta_mask"].fill(0.0)
    features["template_backbone_frame_mask"].fill(0.0)
    compact = compact_zero_template_pair_features(
        collapse_identical_templates(features)
    )

    plan = _padding_plan(compact, PaddingConfig())

    assert plan.actual["templates"] == 0
    assert plan.storage["templates"] == 1
    assert plan.target["templates"] == 1


@pytest.mark.parametrize(
    ("stored", "expected"), [(1, 1), (2, 2), (3, 4), (4, 4)]
)
def test_padding_plan_keeps_stored_templates_on_the_template_ladder(
    stored: int, expected: int
) -> None:
    """Rows that differ are kept and padded to the next of 1, 2 and 4.

    A template-free query reaches the plan already collapsed to one row; padding
    it back to four would run the template stack four times per cycle for the
    value one row gives.
    """
    features = minimal_features(tokens=4, atoms=7, msa_rows=2, templates=stored)
    # Distinct distograms, so `collapse_identical_templates` keeps every row.
    for row in range(stored):
        features["template_distogram"][0, row, 0, 1, row] = 1.0
    assert collapse_identical_templates(features)["template_restype"].shape[1] == stored

    plan = _padding_plan(features, PaddingConfig())

    assert plan.actual["templates"] == plan.storage["templates"] == stored
    assert plan.target["templates"] == expected


def test_explicit_template_pin_still_decides_the_padded_width() -> None:
    features = minimal_features(tokens=4, atoms=7, msa_rows=2, templates=1)

    plan = _padding_plan(features, PaddingConfig(templates=4))

    assert plan.target["templates"] == 4
    with pytest.raises(ValueError, match="padding.templates=2 is smaller"):
        _padding_plan(
            minimal_features(tokens=4, atoms=7, msa_rows=2, templates=4),
            PaddingConfig(templates=2),
        )


def test_existing_archive_padding_is_never_shrunk() -> None:
    features = minimal_features(tokens=4, atoms=7, msa_rows=2, templates=1)
    padded = pad_features(
        features, n_token=8, n_atom=13, n_msa=5, n_templates=3
    )
    plan = _padding_plan(padded, PaddingConfig())

    assert plan.actual == {"tokens": 4, "atoms": 7, "msa": 2, "templates": 1}
    assert plan.storage == {"tokens": 8, "atoms": 13, "msa": 5, "templates": 3}
    assert all(plan.target[name] >= size for name, size in plan.storage.items())


def test_public_crop_removes_atom_and_pair_padding() -> None:
    features = pad_features(
        minimal_features(tokens=4, atoms=7),
        n_token=8,
        n_atom=13,
        n_msa=2,
        n_templates=1,
    )
    prediction = Prediction(
        coordinates=np.zeros((2, 13, 3), dtype=np.float32),
        plddt=np.zeros((2, 13), dtype=np.float32),
        ptm=np.zeros(2, dtype=np.float32),
        iptm=np.zeros(2, dtype=np.float32),
        chain_pair_iptm=None,
        pae_logits=np.zeros((2, 8, 8, 64), dtype=np.float32),
        pde_logits=np.zeros((2, 8, 8, 64), dtype=np.float32),
        distogram_logits=np.zeros((1, 8, 8, 64), dtype=np.float32),
        experimentally_resolved_logits=np.zeros((2, 13, 2), dtype=np.float32),
        plddt_logits=np.zeros((2, 13, 50), dtype=np.float32),
    )

    cropped = crop_prediction(prediction, features)

    assert cropped.coordinates.shape == (2, 7, 3)
    assert cropped.plddt.shape == (2, 7)
    assert cropped.pae_logits.shape == (2, 4, 4, 64)
    assert cropped.pde_logits.shape == (2, 4, 4, 64)
    assert cropped.distogram_logits.shape == (1, 4, 4, 64)
    assert cropped.experimentally_resolved_logits.shape == (2, 7, 2)
    assert cropped.plddt_logits.shape == (2, 7, 50)


def test_crop_rejects_interleaved_padding() -> None:
    features = minimal_features(tokens=4, atoms=7)
    features["token_mask"] = np.asarray([[1, 0, 1, 1]], dtype=np.float32)
    prediction = Prediction(
        coordinates=np.zeros((1, 7, 3), dtype=np.float32),
        plddt=np.zeros((1, 7), dtype=np.float32),
        ptm=np.zeros(1, dtype=np.float32),
        iptm=np.zeros(1, dtype=np.float32),
        chain_pair_iptm=None,
        pae_logits=None,
        pde_logits=None,
        distogram_logits=None,
    )

    with pytest.raises(ValueError, match="token padding must be a suffix"):
        crop_prediction(prediction, features)


def test_plan_rejects_interleaved_archive_padding_before_execution() -> None:
    features = minimal_features(tokens=4, atoms=7)
    features["msa_mask"] = np.asarray(
        [[[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]]],
        dtype=np.float32,
    )
    for name in ("msa", "has_deletion", "deletion_value"):
        value = features[name]
        features[name] = np.concatenate(
            [value[:, :1], np.zeros_like(value[:, :1]), value[:, :1]], axis=1
        )

    with pytest.raises(ValueError, match="MSA-row padding"):
        _padding_plan(features, PaddingConfig())


def test_padded_sampler_tape_keeps_the_exact_real_atom_prefix() -> None:
    import jax

    exact = padded_noise_tape(
        jax.random.key(17),
        n_steps=3,
        num_samples=2,
        actual_atoms=5,
        target_atoms=5,
    )
    padded = padded_noise_tape(
        jax.random.key(17),
        n_steps=3,
        num_samples=2,
        actual_atoms=5,
        target_atoms=11,
    )

    np.testing.assert_array_equal(np.asarray(padded[:, :, :5]), np.asarray(exact))
    np.testing.assert_array_equal(
        np.asarray(padded[:, :, 5:]), np.zeros((4, 2, 6, 3), np.float32)
    )


def test_masked_sampler_keeps_unpadded_coordinates_and_zeros_suffix() -> None:
    import jax

    schedule = jnp.asarray([2.0, 1.0], dtype=jnp.float32)
    kwargs = {
        "noise_schedule": schedule,
        "denoise_fn": lambda coordinates, _t: coordinates,
        "gamma_0": 0.0,
        "gamma_min": 1.0,
        "noise_scale": 1.0,
        "step_scale": 1.0,
        "augment_fn": None,
    }
    exact = sample_diffusion(jax.random.key(19), shape=(2, 5, 3), **kwargs)
    mask = jnp.asarray([[1] * 5 + [0] * 6] * 2, dtype=bool)
    padded = sample_diffusion(
        jax.random.key(19),
        shape=(2, 11, 3),
        noise_mask=mask,
        **kwargs,
    )

    np.testing.assert_array_equal(np.asarray(padded[:, :5]), np.asarray(exact))
    np.testing.assert_array_equal(
        np.asarray(padded[:, 5:]), np.zeros((2, 6, 3), np.float32)
    )


def test_sampler_mask_preserves_existing_storage_stride_across_samples() -> None:
    import jax

    features = pad_features(
        minimal_features(tokens=4, atoms=7),
        n_token=8,
        n_atom=13,
        n_msa=2,
        n_templates=1,
    )
    plan = _padding_plan(features, PaddingConfig(atoms=32))
    noise_mask = _sampler_noise_mask(plan, num_samples=2)

    assert np.all(noise_mask[:, :13])
    assert not np.any(noise_mask[:, 13:])

    schedule = jnp.asarray([2.0, 1.0], dtype=jnp.float32)
    kwargs = {
        "noise_schedule": schedule,
        "denoise_fn": lambda coordinates, _t: coordinates,
        "gamma_0": 0.0,
        "gamma_min": 1.0,
        "noise_scale": 1.0,
        "step_scale": 1.0,
        "augment_fn": None,
    }
    exact = sample_diffusion(jax.random.key(29), shape=(2, 13, 3), **kwargs)
    padded = sample_diffusion(
        jax.random.key(29),
        shape=(2, 32, 3),
        noise_mask=jnp.asarray(noise_mask),
        **kwargs,
    )

    np.testing.assert_array_equal(np.asarray(padded[:, :13]), np.asarray(exact))
    np.testing.assert_array_equal(
        np.asarray(padded[:, 13:]), np.zeros((2, 19, 3), np.float32)
    )


@pytest.mark.parametrize("scan_templates", [True, False])
def test_template_padding_rows_are_neutral_in_both_reduction_paths(
    monkeypatch, scan_templates: bool
) -> None:
    values = jnp.asarray([2.0, 4.0, 100.0], dtype=jnp.float32).reshape(1, 3, 1, 1)
    batch = {
        "template_restype": values,
        "template_padding_mask": jnp.asarray([[1.0, 1.0, 0.0]]),
    }
    params = SimpleNamespace(
        template_pair_embedder=object(),
        template_pair_stack=object(),
        linear_t=object(),
    )
    # The embedding is built one template at a time inside the scan, so the
    # stub has to be per-template too: it reads back the value this step's
    # `template_restype` row carries, which is what makes the scan path
    # actually see three different templates rather than one repeated.
    monkeypatch.setattr(
        template_module,
        "_embed_pair_state",
        lambda _z, _params, **_kwargs: jnp.zeros((1, 1, 1, 1), dtype=jnp.float32),
    )
    monkeypatch.setattr(
        template_module,
        "_template_pair_features",
        lambda batch, _z, _params, **_kwargs: batch["template_restype"][..., None],
    )
    monkeypatch.setattr(
        template_module,
        "template_pair_stack",
        lambda value, _params, **_kwargs: value,
    )
    monkeypatch.setattr(
        template_module, "linear", lambda value, _params: value
    )

    result = template_module.template_embedder(
        batch,
        jnp.zeros((1, 1, 1, 1), dtype=jnp.float32),
        params,
        pair_mask=jnp.ones((1, 1, 1), dtype=jnp.float32),
        no_heads=1,
        scan_templates=scan_templates,
    )

    np.testing.assert_array_equal(np.asarray(result), np.asarray([[[[3.0]]]]))


@pytest.mark.parametrize("scan_templates", [True, False])
def test_template_reduction_without_padding_mask_keeps_legacy_average(
    monkeypatch, scan_templates: bool
) -> None:
    values = jnp.asarray([2.0, 4.0, 99.0], dtype=jnp.float32).reshape(1, 3, 1, 1)
    batch = {"template_restype": values}
    params = SimpleNamespace(
        template_pair_embedder=object(),
        template_pair_stack=object(),
        linear_t=object(),
    )
    # The embedding is built one template at a time inside the scan, so the
    # stub has to be per-template too: it reads back the value this step's
    # `template_restype` row carries, which is what makes the scan path
    # actually see three different templates rather than one repeated.
    monkeypatch.setattr(
        template_module,
        "_embed_pair_state",
        lambda _z, _params, **_kwargs: jnp.zeros((1, 1, 1, 1), dtype=jnp.float32),
    )
    monkeypatch.setattr(
        template_module,
        "_template_pair_features",
        lambda batch, _z, _params, **_kwargs: batch["template_restype"][..., None],
    )
    monkeypatch.setattr(
        template_module,
        "template_pair_stack",
        lambda value, _params, **_kwargs: value,
    )
    monkeypatch.setattr(
        template_module, "linear", lambda value, _params: value
    )

    result = template_module.template_embedder(
        batch,
        jnp.zeros((1, 1, 1, 1), dtype=jnp.float32),
        params,
        pair_mask=jnp.ones((1, 1, 1), dtype=jnp.float32),
        no_heads=1,
        scan_templates=scan_templates,
    )

    np.testing.assert_array_equal(np.asarray(result), np.asarray([[[[35.0]]]]))


def test_native_msa_cap_and_token_profile_are_stable() -> None:
    plans = [
        _padding_plan(
            minimal_features(
                tokens=tokens, atoms=atoms, msa_rows=rows, templates=templates
            ),
            PaddingConfig(),
            max_msa_depth=100,
        )
        for tokens, atoms, rows, templates in ((4, 7, 2, 1), (8, 20, 80, 4))
    ]
    # Only the template axis follows its stored rows; every other axis of the
    # profile is the token bucket's.
    assert {k: v for k, v in plans[0].target.items() if k != "templates"} == {
        k: v for k, v in plans[1].target.items() if k != "templates"
    }
    assert [plan.target["templates"] for plan in plans] == [1, 4]
    assert plans[0].target["msa"] == 100


def test_native_template_capacity_does_not_truncate_archive() -> None:
    with pytest.raises(ValueError, match="above the released width of 4"):
        _padding_plan(
            minimal_features(tokens=4, atoms=7, msa_rows=2, templates=5),
            PaddingConfig(),
        )


def _head_prediction(features, s, head, coordinates) -> Prediction:
    n_atom = features["atom_mask"].shape[-1]
    logits = atom_logit_head(
        s,
        head,
        jnp.asarray(features["max_atom_per_token_mask"]),
        max_atoms_per_token=23,
        c_out=50,
        n_atom=n_atom,
    )
    stored = np.zeros((coordinates.shape[0], n_atom, 3), dtype=np.float32)
    stored[:, : coordinates.shape[1]] = coordinates
    return Prediction(
        coordinates=stored,
        plddt=np.asarray(compute_plddt(logits)),
        ptm=np.full(coordinates.shape[0], 0.7, dtype=np.float32),
        iptm=np.zeros(coordinates.shape[0], dtype=np.float32),
        chain_pair_iptm=None,
        pae_logits=None,
        pde_logits=None,
        distogram_logits=None,
    )


def _past_the_token_slots():
    """Exact and served predictions whose atom bucket exceeds tokens * 23."""
    rng = np.random.default_rng(0)
    exact = minimal_features(tokens=4, atoms=7)
    padded = pad_features(exact, n_token=8, n_atom=8 * 23 + 16, n_msa=2)
    head = map_atom_logit_head(
        {
            "layer_norm.weight": rng.normal(size=8).astype(np.float32),
            "layer_norm.bias": rng.normal(size=8).astype(np.float32),
            "linear.weight": rng.normal(size=(23 * 50, 8)).astype(np.float32),
        }
    )
    s = rng.normal(size=(2, 8, 8)).astype(np.float32)
    coordinates = rng.normal(size=(2, 7, 3)).astype(np.float32) * 10.0

    reference = _head_prediction(exact, jnp.asarray(s[:, :4]), head, coordinates)
    served = _head_prediction(padded, jnp.asarray(s), head, coordinates)
    return exact, reference, padded, served


def test_atom_bucket_past_the_token_slots_keeps_the_public_plddt() -> None:
    """Serving may pad atoms past tokens * 23: 3OG2 pads 1003 tokens to 1024
    (23552 slots) but 7749 atoms to 24576. The per-atom heads must still return
    every padded atom row, so the atom mask applies and the public mean pLDDT is
    the unpadded one rather than one diluted by padding rows at exactly 50."""
    exact, reference, padded, served = _past_the_token_slots()

    assert served.plddt.shape == (2, 8 * 23 + 16)
    np.testing.assert_allclose(served.plddt[:, :7], reference.plddt, rtol=1e-6)
    expected = [
        entry["mean_plddt"]
        for entry in confidence_summary(reference, exact)["samples"]
    ]
    actual = [
        entry["mean_plddt"]
        for entry in confidence_summary(served, padded)["samples"]
    ]
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    assert 1.0 < min(expected) and max(expected) <= 100.0


def test_atom_bucket_past_the_token_slots_keeps_the_b_factors(tmp_path) -> None:
    """The padded run's CIF B-factors were all 0: the writer dropped them."""
    gemmi = pytest.importorskip("gemmi")
    from foldjax.models.openfold3.output import write_prediction_outputs

    _, reference, padded, served = _past_the_token_slots()
    with pytest.warns(RuntimeWarning, match="no exact output metadata"):
        written = write_prediction_outputs(served, padded, tmp_path, name="pad")
    for index, path in enumerate(written["structures"]):
        structure = gemmi.read_structure(str(path))
        b_factors = [
            atom.b_iso
            for model in structure
            for chain in model
            for residue in chain
            for atom in residue
        ]
        np.testing.assert_allclose(
            b_factors, reference.plddt[index] * 100.0, atol=1e-2
        )


def test_summary_refuses_plddt_the_atom_mask_cannot_cover() -> None:
    """A length mismatch used to skip the mask and average the padding in."""
    features = pad_features(minimal_features(tokens=4, atoms=7), n_token=8, n_atom=13)
    prediction = Prediction(
        coordinates=np.zeros((1, 13, 3), dtype=np.float32),
        plddt=np.full((1, 12), 0.9, dtype=np.float32),
        ptm=np.zeros(1, dtype=np.float32),
        iptm=None,
        chain_pair_iptm=None,
        pae_logits=None,
        pde_logits=None,
        distogram_logits=None,
    )
    with pytest.raises(ValueError, match="pLDDT"):
        confidence_summary(prediction, features)


def test_padded_polymer_ligand_complex_preserves_clash_and_sample_ranking() -> None:
    """Default padding + ``normalize_asym_ids`` preserves clash and ranking scores."""
    from foldjax.models.openfold3.data import normalize_asym_ids
    from foldjax.models.openfold3.models.clash import compute_has_clash

    exact = minimal_features(tokens=3, atoms=6)
    exact["asym_id"] = np.asarray([[1, 2, 3]], dtype=np.int32)
    exact["is_protein"] = np.zeros((1, 3), dtype=np.int32)
    exact["is_rna"] = np.asarray([[1, 1, 0]], dtype=np.int32)
    exact["is_dna"] = np.zeros((1, 3), dtype=np.int32)
    exact["is_ligand"] = np.asarray([[0, 0, 1]], dtype=np.int32)
    exact["atom_to_token_index"] = np.asarray([[0, 0, 1, 1, 2, 2]], dtype=np.int32)

    padded = pad_features(exact, n_token=6, n_atom=10)
    exact_norm, _ = normalize_asym_ids(exact)
    padded_norm, n_chain = normalize_asym_ids(padded)

    pair_base = np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32)
    ligand_far = np.asarray([[50.0, 0.0, 0.0], [53.0, 0.0, 0.0]], dtype=np.float32)
    sample_clash = np.concatenate((pair_base, pair_base + 0.1, ligand_far), axis=0)
    sample_clean = np.concatenate((pair_base, pair_base + 20.0, ligand_far), axis=0)
    exact_coords = np.stack((sample_clash, sample_clean))
    padded_coords = np.pad(exact_coords, ((0, 0), (0, 4), (0, 0)))

    exact_pred = Prediction(
        coordinates=exact_coords,
        plddt=np.full((2, 6), 0.85, dtype=np.float32),
        ptm=np.asarray([0.9, 0.7], dtype=np.float32),
        iptm=np.asarray([0.9, 0.7], dtype=np.float32),
        chain_pair_iptm=None,
        pae_logits=None,
        pde_logits=None,
        distogram_logits=None,
    )
    padded_pred = exact_pred._replace(
        coordinates=padded_coords,
        plddt=np.pad(exact_pred.plddt, ((0, 0), (0, 4))),
    )
    exact_summary = confidence_summary(exact_pred, exact_norm)
    padded_summary = confidence_summary(padded_pred, padded_norm)
    assert exact_summary["ranked_samples"] == padded_summary["ranked_samples"] == [1, 0]
    for exact_entry, padded_entry in zip(
        exact_summary["samples"], padded_summary["samples"], strict=True
    ):
        assert exact_entry == pytest.approx(padded_entry)

    # Also verify direct callers whose padded atoms point to padded tokens (asym_id 0).
    padded_unvalidated = {
        **padded_norm,
        "atom_to_token_index": np.asarray(
            [[0, 0, 1, 1, 2, 2, 3, 4, 5, 5]], dtype=np.int32
        ),
    }
    unval_summary = confidence_summary(padded_pred, padded_unvalidated)
    assert unval_summary["ranked_samples"] == [1, 0]
    owners = padded_unvalidated["atom_to_token_index"][0]
    polymer_tok = (
        padded_unvalidated["is_rna"][0].astype(bool)
        & padded_unvalidated["token_mask"][0].astype(bool)
    )
    jax_clash = compute_has_clash(
        jnp.asarray(padded_unvalidated["asym_id"][0][owners]),
        jnp.asarray(padded_coords),
        jnp.asarray(padded_unvalidated["atom_mask"][0]),
        jnp.asarray(polymer_tok[owners]),
        n_chain=n_chain,
    )
    np.testing.assert_array_equal(np.asarray(jax_clash), [1.0, 0.0])


def test_unsupported_masked_prefix_draw_guards_sampler_and_falls_back_to_tape(
    tmp_path, monkeypatch
) -> None:
    """When masked prefix draw is unsupported, sampler guards and adapter uses tape."""
    import importlib
    import warnings

    import jax

    from foldjax.backends import openfold3 as adapter
    from foldjax.models import _random
    from foldjax.models.openfold3 import inference
    from foldjax.models.openfold3.models.sampler import sample_diffusion
    from foldjax.schema import PredictionRequest

    schedule = jnp.asarray([4.0, 2.0, 1.0], dtype=jnp.float32)
    mask = jnp.asarray([[True, True, False], [True, True, False]])

    monkeypatch.setattr(_random, "supports_masked_prefix_draw", lambda: False)
    with pytest.raises(ValueError, match="prefix-preserving padding noise requires"):
        sample_diffusion(
            jax.random.key(0),
            schedule,
            (2, 3, 3),
            lambda x, _t: x * 0.5,
            gamma_0=0.8,
            gamma_min=1.0,
            noise_scale=1.003,
            step_scale=1.5,
            noise_mask=mask,
        )

    # Verify OpenFold3Backend.predict falls back to padded_noise_tape and matches
    # the masked-prefix rollout on the real atom prefix.
    features = minimal_features(tokens=4, atoms=6)
    seen_routes: list[tuple[bool, bool]] = []

    def run_adapter(supports_mask: bool):
        monkeypatch.setattr(
            _random, "supports_masked_prefix_draw", lambda: supports_mask
        )

        def fake_streamed_factory(config, _table, *, n_chain=None, **_kwargs):
            def call(key, batch, _params, *, noise_mask=None, noise_tape=None):
                seen_routes.append((noise_mask is not None, noise_tape is not None))
                sched = jnp.asarray([2.0, 1.0, 0.5], dtype=jnp.float32)
                coords = sample_diffusion(
                    key,
                    sched,
                    (config.num_samples, config.n_atom, 3),
                    lambda x, _t: x * 0.5,
                    gamma_0=0.8,
                    gamma_min=1.0,
                    noise_scale=1.003,
                    step_scale=1.5,
                    noise_mask=noise_mask,
                    noise_tape=noise_tape,
                )
                return inference.Prediction(
                    coordinates=coords,
                    plddt=jnp.full((config.num_samples, config.n_atom), 0.8),
                    ptm=jnp.full((config.num_samples,), 0.7),
                    iptm=jnp.full((config.num_samples,), 0.6),
                    chain_pair_iptm=None,
                    pae_logits=None,
                    pde_logits=None,
                    distogram_logits=None,
                )

            return call

        data_mod = importlib.import_module("foldjax.models.openfold3.data")
        data_ns = {
            name: getattr(data_mod, name)
            for name in dir(data_mod)
            if not name.startswith("__")
        }
        data_ns["featurize_query_with_metadata"] = lambda *_a, **_k: (
            {k: v.copy() for k, v in features.items()},
            None,
        )
        inf_mod = importlib.import_module("foldjax.models.openfold3.inference")
        inf_ns = {
            n: getattr(inf_mod, n) for n in dir(inf_mod) if not n.startswith("__")
        }
        inf_ns["cast_narrow_params"] = lambda params, *_d: params
        stubs = {
            "foldjax.models.openfold3.data": SimpleNamespace(**data_ns),
            "foldjax.models.openfold3.inference": SimpleNamespace(**inf_ns),
            "foldjax.models.openfold3.bridge.checkpoint": SimpleNamespace(
                load_checkpoint=lambda _path: {}
            ),
            "foldjax.models.openfold3.bridge.torch_mapping": SimpleNamespace(
                resolve_model_prefix=lambda _state, prefix=None: "",
                prune_sample_diffusion_aliases=lambda _state, *, prefix: 0,
                map_inference_params=lambda _state, _prefix: object(),
            ),
        }
        monkeypatch.setattr(
            adapter,
            "import_module",
            lambda name: stubs.get(name) or importlib.import_module(name),
        )
        monkeypatch.setattr(
            "foldjax.models.openfold3.streaming.compile_streamed_predict",
            fake_streamed_factory,
        )
        job = tmp_path / "job.json"
        job.write_text('{"queries": {}}', encoding="utf-8")
        weights = tmp_path / "w.pt"
        weights.write_bytes(b"x")
        out_dir = tmp_path / ("out-mask" if supports_mask else "out-tape")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adapter.OpenFold3Backend().predict(
                PredictionRequest(
                    model="openfold3",
                    input=job,
                    input_format="openfold3",
                    weights=weights,
                    output_dir=out_dir,
                    seed=11,
                    padding=PaddingConfig(tokens=8, atoms=16, msa=4, templates=1),
                    options={"num_samples": 2, "num_steps": 2, "all_arrays": True},
                )
            )
        with np.load(out_dir / "job_raw.npz") as raw:
            return raw["coordinates"][:, :6, :]

    tape_coords = run_adapter(False)
    mask_coords = run_adapter(True)
    assert seen_routes == [(False, True), (True, False)]
    np.testing.assert_allclose(tape_coords, mask_coords, rtol=1e-6, atol=1e-6)
