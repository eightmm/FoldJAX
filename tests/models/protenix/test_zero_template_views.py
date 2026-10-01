"""Template-free geometry costs one word on the host, and changes no input bit.

``_as_protenix_dict`` used to allocate the four quadratic template geometry
fields densely for every slot, real or padded: ``4 x 44 x 4 B x N^2``, 30 GB at
6,568 tokens, before ``dedup_templates`` copied the survivors and
``compact_zero_template_geometry`` threw all of it away. A slot with no
observed atom now contributes no computation, and a query where every slot is
such a slot gets zero-stride views that padding, deduplication, compaction and
the archive writer all accept without materialising.

These tests hold the model-facing features to the old dense path *bit for
bit*: the reference below is the old loop verbatim, substituted for the new
fields, and both representations then run through the same downstream steps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from foldjax.models.opendde.data.featurize_json import featurize_opendde_json
from foldjax.models.opendde.data.padding import (
    pad_opendde_features,
    select_opendde_model_features,
)
from foldjax.models.opendde.models.msa_sampling import (
    sample_opendde_msa_cycle_features,
)
from foldjax.models.protenix.data.featurize_json import featurize_protein_json
from foldjax.models.protenix.data.padding import pad_protenix_features
from foldjax.models.protenix.data.static_io import (
    load_static_feature_npz,
    save_static_feature_npz,
)
from foldjax.models.protenix.data.template_features import (
    _BACKBONE_FRAME,
    _DGRAM_NUM_BINS,
    ZERO_TEMPLATE_GEOMETRY_FIELDS,
    ZERO_TEMPLATE_GEOMETRY_MARKER,
    _as_protenix_dict,
    _dgram_from_positions,
    _pseudo_beta,
    _unit_vector,
    broadcast_scalar,
    compact_zero_template_geometry,
    dedup_templates,
    has_compact_zero_template_geometry,
)
from foldjax.schema import PaddingConfig

_FIXTURE = Path(__file__).parent / "fixtures" / "template_small.json"


def _reference_as_protenix_dict(
    aatype: np.ndarray, atom_positions: np.ndarray, atom_mask: np.ndarray
) -> dict[str, np.ndarray]:
    """The pre-change ``_as_protenix_dict``, verbatim: dense, every slot computed."""
    num_t, num_res = aatype.shape
    pb_masks = np.empty((num_t, num_res, num_res), dtype=np.float32)
    dgrams = np.empty((num_t, num_res, num_res, _DGRAM_NUM_BINS), dtype=np.float32)
    unit_vectors = np.empty((num_t, num_res, num_res, 3), dtype=np.float32)
    bb_masks = np.empty((num_t, num_res, num_res), dtype=np.float32)
    bool_mask = atom_mask.astype(bool)
    for i in range(num_t):
        pos = atom_positions[i] * bool_mask[i][..., None]
        pb_pos, pb_mask = _pseudo_beta(aatype[i], pos, bool_mask[i])
        pb_mask_2d = pb_mask[:, None] * pb_mask[None, :]
        dgrams[i] = _dgram_from_positions(pb_pos) * pb_mask_2d[..., None]
        pb_masks[i] = pb_mask_2d
        uv, bb_mask_2d = _unit_vector(aatype[i], pos, bool_mask[i])
        unit_vectors[i] = uv * bb_mask_2d[..., None]
        bb_masks[i] = bb_mask_2d
    return {
        "template_aatype": aatype,
        "template_atom_positions": atom_positions,
        "template_atom_mask": atom_mask.astype(bool),
        "template_pseudo_beta_mask": pb_masks,
        "template_distogram": dgrams,
        "template_unit_vector": unit_vectors,
        "template_backbone_frame_mask": bb_masks,
    }


def _as_old_path(features: dict[str, Any]) -> dict[str, Any]:
    """``features`` with its geometry rebuilt the way the old featurizer did."""
    reference = _reference_as_protenix_dict(
        np.asarray(features["template_aatype"]),
        np.asarray(features["template_atom_positions"]),
        np.asarray(features["template_atom_mask"]),
    )
    old = dict(features)
    for name in ZERO_TEMPLATE_GEOMETRY_FIELDS:
        old[name] = reference[name]
    return old


def _assert_bit_identical(new: Any, old: Any, path: str = "") -> None:
    if isinstance(old, dict):
        assert isinstance(new, dict), path
        assert set(new) == set(old), (path, set(new) ^ set(old))
        for key in old:
            _assert_bit_identical(new[key], old[key], f"{path}.{key}")
        return
    if old is None or isinstance(old, (bool, int, float, str)):
        assert new == old, path
        return
    new_array, old_array = np.asarray(new), np.asarray(old)
    assert new_array.dtype == old_array.dtype, (path, new_array.dtype, old_array.dtype)
    assert new_array.shape == old_array.shape, (path, new_array.shape, old_array.shape)
    if old_array.dtype.hasobject:
        assert new_array.tolist() == old_array.tolist(), path
    else:
        assert np.ascontiguousarray(new_array).tobytes() == old_array.tobytes(), path


def _two_template_json(tmp_path: Path) -> Path:
    """The fixture hit, plus a second, distinct hit that maps only three residues."""
    (hit,) = json.loads(_FIXTURE.read_text())
    partial = dict(hit)
    partial["queryIndices"] = hit["queryIndices"][:3]
    partial["templateIndices"] = hit["templateIndices"][:3]
    path = tmp_path / "two_templates.json"
    path.write_text(json.dumps([hit, partial]))
    return path


def _job(case: str, tmp_path: Path) -> dict[str, Any]:
    if case == "none":
        first: dict[str, Any] = {"sequence": "ACDEFGHIK", "count": 2}
    else:
        path = _FIXTURE if case == "one" else _two_template_json(tmp_path)
        first = {"sequence": "AGSRF", "count": 1, "templatesPath": str(path)}
    return {
        "name": case,
        "sequences": [
            {"proteinChain": first},
            {"proteinChain": {"sequence": "MKVLAG", "count": 1}},
        ],
    }


CASES = ("none", "one", "two")


def _protenix(case: str, tmp_path: Path) -> dict[str, Any]:
    return featurize_protein_json(
        _job(case, tmp_path), n_queries=2, n_keys=4, use_template=case != "none"
    )


def _opendde(case: str, tmp_path: Path) -> dict[str, Any]:
    return featurize_opendde_json(
        _job(case, tmp_path),
        n_queries=2,
        n_keys=4,
        seed=7,
        use_template=case != "none",
    )


def _model_bound(features: dict[str, Any]) -> dict[str, Any]:
    return dict(compact_zero_template_geometry(dedup_templates(features)))


def _protenix_padded(features: dict[str, Any]) -> dict[str, Any]:
    n_token = int(np.asarray(features["restype"]).shape[0])
    padded, _ = pad_protenix_features(
        features,
        PaddingConfig(tokens=n_token + 5),
        n_queries=2,
        n_keys=4,
    )
    return padded


def _opendde_padded(features: dict[str, Any]) -> dict[str, Any]:
    sampled = sample_opendde_msa_cycle_features(features, num_recycles=1, seed=7)
    padded, _, _ = pad_opendde_features(
        features,
        sampled,
        PaddingConfig(tokens=int(features["restype"].shape[0]) + 5),
        n_queries=2,
        n_keys=4,
    )
    return select_opendde_model_features(padded)


@pytest.mark.parametrize("case", CASES)
def test_featurizer_geometry_is_the_old_dense_geometry(case, tmp_path) -> None:
    features = _protenix(case, tmp_path)
    old = _as_old_path(features)
    for name in ZERO_TEMPLATE_GEOMETRY_FIELDS:
        _assert_bit_identical(features[name], old[name], name)
        scalar = broadcast_scalar(np.asarray(features[name]))
        # The property itself, not a proxy for it: a template-free query's
        # geometry holds one stored word, a query with a hit is dense.
        assert (scalar is not None) == (case == "none"), name


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize(
    ("featurize", "pad"),
    [
        (_protenix, None),
        (_protenix, _protenix_padded),
        (_opendde, None),
        (_opendde, _opendde_padded),
    ],
    ids=["protenix", "protenix-padded", "opendde", "opendde-padded"],
)
def test_model_bound_features_match_the_old_path_bit_for_bit(
    case, featurize, pad, tmp_path
) -> None:
    features = featurize(case, tmp_path)
    old = _as_old_path(features)
    if pad is not None:
        features, old = pad(features), pad(old)
    new_bound, old_bound = _model_bound(features), _model_bound(old)

    _assert_bit_identical(new_bound, old_bound)
    assert has_compact_zero_template_geometry(new_bound) == (case == "none")
    assert has_compact_zero_template_geometry(
        old_bound
    ) == has_compact_zero_template_geometry(new_bound)
    if case == "none":
        # Nothing on the way to the marker materialised the views.
        for name in ZERO_TEMPLATE_GEOMETRY_FIELDS:
            assert broadcast_scalar(np.asarray(features[name])) is not None, name
            deduped = dedup_templates(features)[name]
            assert broadcast_scalar(np.asarray(deduped)) is not None, name
        assert np.asarray(new_bound[ZERO_TEMPLATE_GEOMETRY_MARKER]).nbytes == 4


def test_the_archive_writes_the_old_bytes(tmp_path) -> None:
    features = _protenix("none", tmp_path)
    save_static_feature_npz(tmp_path / "new.npz", features)
    save_static_feature_npz(tmp_path / "old.npz", _as_old_path(features))
    new = load_static_feature_npz(tmp_path / "new.npz")
    old = load_static_feature_npz(tmp_path / "old.npz")
    _assert_bit_identical(new, old)


def test_an_unobserved_slot_with_nonzero_positions_is_still_computed() -> None:
    """Why emptiness needs bitwise +0.0 positions, not just an all-False mask.

    ``positions * mask`` turns an infinite position into ``NaN``, and the
    unit vector carries it through ``uv * bb_mask``. Treating this slot as
    empty would emit ``+0.0`` there; it has to take the dense path, and the
    ``NaN`` has to keep it out of the compact marker as before.
    """
    n_res = 4
    aatype = np.zeros((1, n_res), np.int32)
    positions = np.zeros((1, n_res, 24, 3), np.float32)
    positions[0, 1, 1] = (np.inf, -2.0, 3.0)  # residue 1, CA
    positions[0, 2, 2] = (-1.0, -0.0, 5.0)  # residue 2, C: signed values
    mask = np.zeros((1, n_res, 24), np.int32)
    with np.errstate(invalid="ignore"):
        reference = _reference_as_protenix_dict(aatype, positions, mask)
        produced = _as_protenix_dict(aatype, positions, mask)
    assert np.isnan(reference["template_unit_vector"]).any(), "tripwire: no NaN"

    for name in ZERO_TEMPLATE_GEOMETRY_FIELDS:
        _assert_bit_identical(produced[name], reference[name], name)
        assert broadcast_scalar(np.asarray(produced[name])) is None, name
    assert not has_compact_zero_template_geometry(
        compact_zero_template_geometry(produced)
    )


def test_an_out_of_range_restype_still_raises_on_an_empty_slot() -> None:
    n_res = 3
    aatype = np.full((1, n_res), len(_BACKBONE_FRAME), np.int32)
    with pytest.raises(IndexError):
        _reference_as_protenix_dict(
            aatype,
            np.zeros((1, n_res, 24, 3), np.float32),
            np.zeros((1, n_res, 24), np.int32),
        )
    with pytest.raises(IndexError):
        _as_protenix_dict(
            aatype,
            np.zeros((1, n_res, 24, 3), np.float32),
            np.zeros((1, n_res, 24), np.int32),
        )
