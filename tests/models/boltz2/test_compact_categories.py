"""Boltz-2 private compact storage for categorical pair and atom features.

The load-bearing proof is :func:`test_fixture_restores_to_bit_identical_arrays`:
the compiled graph rebuilds arrays that are equal to the dense publisher inputs
in value *and* dtype, so every consumer downstream runs an unchanged program.
The remaining model-level tests execute the real consumers so a silently
dropped feature cannot pass.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import foldjax.models.boltz2.api as api
from foldjax.models.boltz2.data.bucket import pad_feats, select_model_features
from foldjax.models.boltz2.data.compact_categories import (
    COMPACT_CONTACT_CONDITIONING,
    COMPACT_REF_ATOM_CATEGORIES,
    COMPACT_TOKEN_BONDS,
    CONTACT_CONDITIONING_IDS,
    REF_ATOM_NAME_CHAR_IDS,
    REF_ELEMENT_IDS,
    TOKEN_BONDS_FLAGS,
    TYPE_BONDS_IDS,
    compact_category_storage,
)
from foldjax.models.boltz2.models._compact_categories import (
    restore_compact_categories,
)

_FIXTURE = Path(__file__).parent / "fixtures/1UBQ_A.npz"

_DENSE_NAMES = (
    "contact_conditioning",
    "token_bonds",
    "type_bonds",
    "ref_element",
    "ref_atom_name_chars",
)
_PRIVATE_NAMES = (
    COMPACT_CONTACT_CONDITIONING,
    CONTACT_CONDITIONING_IDS,
    COMPACT_TOKEN_BONDS,
    TOKEN_BONDS_FLAGS,
    TYPE_BONDS_IDS,
    COMPACT_REF_ATOM_CATEGORIES,
    REF_ELEMENT_IDS,
    REF_ATOM_NAME_CHAR_IDS,
)


def _one_hot(ids: np.ndarray, classes: int) -> np.ndarray:
    """Publisher-shaped ``int64`` one-hot; ``classes`` marks an all-zero row."""

    dense = np.zeros((*ids.shape, classes), dtype=np.int64)
    live = ids < classes
    index = np.nonzero(live)
    dense[(*index, ids[live])] = 1
    return dense


def _synthetic(tokens: int = 3, atoms: int = 4) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260910)
    contact_ids = rng.integers(0, 5, size=(1, tokens, tokens)).astype(np.uint8)
    element_ids = rng.integers(0, 128, size=(1, atoms)).astype(np.uint8)
    char_ids = rng.integers(0, 64, size=(1, atoms, 4)).astype(np.uint8)
    return {
        "token_pad_mask": np.ones((1, tokens), dtype=np.float32),
        "atom_pad_mask": np.ones((1, atoms), dtype=np.float32),
        "contact_conditioning": _one_hot(contact_ids, 5),
        "type_bonds": rng.integers(0, 4, size=(1, tokens, tokens)).astype(np.int64),
        "token_bonds": rng.integers(0, 2, size=(1, tokens, tokens, 1)).astype(
            np.float32
        ),
        "ref_element": _one_hot(element_ids, 128),
        "ref_atom_name_chars": _one_hot(char_ids, 64),
    }


def _fixture_features() -> dict[str, np.ndarray]:
    with np.load(_FIXTURE) as archive:
        return {
            name: np.asarray(archive[name])
            for name in (*_DENSE_NAMES, "token_pad_mask", "atom_pad_mask")
        }


def test_native_categories_compact_without_mutating_public_features() -> None:
    public = _synthetic()
    original = {name: public[name] for name in _DENSE_NAMES}

    compact = compact_category_storage(public)

    assert compact is not public
    for name in _DENSE_NAMES:
        assert name not in compact
        assert public[name] is original[name]
    for name in _PRIVATE_NAMES:
        assert name not in public
        assert name in compact
    for marker in (
        COMPACT_CONTACT_CONDITIONING,
        COMPACT_TOKEN_BONDS,
        COMPACT_REF_ATOM_CATEGORIES,
    ):
        assert compact[marker].shape == ()
        assert compact[marker].dtype == np.uint8
        assert int(compact[marker]) == 1
    assert compact[CONTACT_CONDITIONING_IDS].dtype == np.uint8
    assert compact[TYPE_BONDS_IDS].dtype == np.uint8
    assert compact[TOKEN_BONDS_FLAGS].dtype == np.bool_
    assert compact[REF_ELEMENT_IDS].dtype == np.uint8
    assert compact[REF_ATOM_NAME_CHAR_IDS].dtype == np.uint8
    assert compact[CONTACT_CONDITIONING_IDS].shape == (1, 3, 3)
    assert compact[TOKEN_BONDS_FLAGS].shape == (1, 3, 3, 1)
    assert compact[REF_ELEMENT_IDS].shape == (1, 4)
    assert compact[REF_ATOM_NAME_CHAR_IDS].shape == (1, 4, 4)


def _assert_restores_exactly(dense: dict[str, np.ndarray]) -> None:
    compact = compact_category_storage(dense)
    for name in _DENSE_NAMES:
        assert name not in compact

    restored = jax.jit(restore_compact_categories)(
        {name: jnp.asarray(value) for name, value in compact.items()}
    )

    for name in _DENSE_NAMES:
        expected = jnp.asarray(dense[name])
        assert restored[name].dtype == expected.dtype, name
        assert np.array_equal(np.asarray(restored[name]), np.asarray(expected)), name
    for name in _PRIVATE_NAMES:
        assert name not in restored


def test_synthetic_categories_restore_to_bit_identical_arrays() -> None:
    _assert_restores_exactly(_synthetic())


def test_fixture_restores_to_bit_identical_arrays() -> None:
    dense = _fixture_features()
    _assert_restores_exactly(dense)

    compact = compact_category_storage(dense)
    dense_bytes = sum(dense[name].nbytes for name in _DENSE_NAMES)
    compact_bytes = sum(np.asarray(compact[name]).nbytes for name in _PRIVATE_NAMES)
    assert compact_bytes * 8 < dense_bytes


def test_padded_categories_restore_to_the_dense_padded_arrays() -> None:
    dense = _fixture_features()
    tokens = int(dense["token_pad_mask"].shape[-1]) + 20
    atoms = int(dense["atom_pad_mask"].shape[-1]) + 32

    padded_dense, _ = pad_feats(dense, tokens, atoms)
    padded_compact, _ = pad_feats(compact_category_storage(dense), tokens, atoms)
    restored = jax.jit(restore_compact_categories)(padded_compact)

    for name in _DENSE_NAMES:
        assert restored[name].dtype == padded_dense[name].dtype, name
        assert np.array_equal(
            np.asarray(restored[name]), np.asarray(padded_dense[name])
        ), name


def _fallback_cases() -> list[tuple[str, dict[str, np.ndarray]]]:
    cases: list[tuple[str, dict[str, np.ndarray]]] = []

    narrow = _synthetic()
    narrow["contact_conditioning"] = narrow["contact_conditioning"].astype(np.int32)
    cases.append(("int32 contact one-hot", narrow))

    multi_hot = _synthetic()
    multi_hot["contact_conditioning"][0, 0, 0, :2] = 1
    cases.append(("multi-hot contact row", multi_hot))

    scaled = _synthetic()
    scaled["contact_conditioning"][0, 0, 0, 0] = 2
    cases.append(("non-binary contact value", scaled))

    ranked = _synthetic()
    ranked["contact_conditioning"] = ranked["contact_conditioning"][0]
    cases.append(("unbatched contact one-hot", ranked))

    wide = _synthetic()
    wide["type_bonds"] = wide["type_bonds"] + 256
    cases.append(("type_bonds beyond uint8", wide))

    signed_zero = _synthetic()
    signed_zero["token_bonds"][0, 0, 0, 0] = -0.0
    cases.append(("negative zero token_bonds", signed_zero))

    fractional = _synthetic()
    fractional["token_bonds"][0, 0, 0, 0] = 0.5
    cases.append(("fractional token_bonds", fractional))

    infinite = _synthetic()
    infinite["token_bonds"][0, 0, 0, 0] = np.inf
    cases.append(("non-finite token_bonds", infinite))

    element = _synthetic()
    element["ref_element"][0, 0, :2] = 1
    cases.append(("multi-hot ref_element", element))

    chars = _synthetic()
    chars["ref_atom_name_chars"] = chars["ref_atom_name_chars"].astype(np.float32)
    cases.append(("float ref_atom_name_chars", chars))
    return cases


@pytest.mark.parametrize("label,features", _fallback_cases(), ids=lambda v: str(v)[:40])
def test_custom_layouts_fall_back_to_the_dense_contract(label, features) -> None:
    compact = compact_category_storage(features)

    stale = [name for name in _PRIVATE_NAMES if name in compact]
    remaining = [name for name in _DENSE_NAMES if name in compact]
    assert remaining, f"{label}: nothing stayed dense"
    for name in remaining:
        assert np.array_equal(compact[name], features[name])
    # A group either compacts completely or keeps its dense members; no group
    # may leave one dense member behind next to its own private payload.
    assert not (set(stale) & set(_group_private(remaining)))


def _group_private(dense_names: list[str]) -> set[str]:
    groups = {
        "contact_conditioning": (
            COMPACT_CONTACT_CONDITIONING,
            CONTACT_CONDITIONING_IDS,
        ),
        "token_bonds": (COMPACT_TOKEN_BONDS, TOKEN_BONDS_FLAGS, TYPE_BONDS_IDS),
        "type_bonds": (COMPACT_TOKEN_BONDS, TOKEN_BONDS_FLAGS, TYPE_BONDS_IDS),
        "ref_element": (COMPACT_REF_ATOM_CATEGORIES, REF_ELEMENT_IDS),
        "ref_atom_name_chars": (
            COMPACT_REF_ATOM_CATEGORIES,
            REF_ATOM_NAME_CHAR_IDS,
        ),
    }
    private: set[str] = set()
    for name in dense_names:
        private.update(groups[name])
    return private


def test_dense_input_takes_precedence_over_stale_private_provenance() -> None:
    dense = _synthetic()
    stale = dict(dense)
    stale[CONTACT_CONDITIONING_IDS] = np.zeros((1, 3, 3), dtype=np.uint8)
    stale[COMPACT_CONTACT_CONDITIONING] = np.asarray(1, dtype=np.uint8)
    stale[REF_ELEMENT_IDS] = np.zeros((1, 4), dtype=np.uint8)

    compact = compact_category_storage(stale)
    assert (
        compact[CONTACT_CONDITIONING_IDS].tobytes()
        != np.zeros((1, 3, 3), dtype=np.uint8).tobytes()
    )

    restored = restore_compact_categories({**dense, CONTACT_CONDITIONING_IDS: 1})
    assert CONTACT_CONDITIONING_IDS not in restored
    assert np.array_equal(
        restored["contact_conditioning"], dense["contact_conditioning"]
    )


@pytest.mark.parametrize(
    "private",
    [
        {COMPACT_CONTACT_CONDITIONING: np.asarray(1, dtype=np.uint8)},
        {CONTACT_CONDITIONING_IDS: np.zeros((1, 3, 3), dtype=np.uint8)},
        {COMPACT_TOKEN_BONDS: np.asarray(1, dtype=np.uint8)},
        {TYPE_BONDS_IDS: np.zeros((1, 3, 3), dtype=np.uint8)},
        {COMPACT_REF_ATOM_CATEGORIES: np.asarray(1, dtype=np.uint8)},
        {REF_ELEMENT_IDS: np.zeros((1, 4), dtype=np.uint8)},
    ],
)
def test_incomplete_private_representation_is_rejected(private) -> None:
    features = _synthetic()
    for name in _DENSE_NAMES:
        del features[name]
    features.update(private)

    with pytest.raises((KeyError, ValueError)):
        compact_category_storage(features)
    with pytest.raises((KeyError, ValueError)):
        restore_compact_categories(features)


@pytest.mark.parametrize("marker", [COMPACT_CONTACT_CONDITIONING, COMPACT_TOKEN_BONDS])
def test_wrong_version_marker_is_rejected(marker) -> None:
    compact = dict(compact_category_storage(_synthetic()))
    compact[marker] = np.asarray(2, dtype=np.uint8)

    with pytest.raises(ValueError, match="version"):
        compact_category_storage(compact)
    with pytest.raises(ValueError, match="version"):
        restore_compact_categories(compact)


def test_model_feature_whitelist_keeps_the_private_categories() -> None:
    compact = compact_category_storage(_synthetic())
    compact["mol_type"] = np.asarray([[0, 3, 0]], dtype=np.int64)

    kept = select_model_features(compact)

    for name in _PRIVATE_NAMES:
        assert name in kept


# --- model-level: the real consumers must run on both forms ----------------

_CONSUMER_FEATURES = (
    "atom_pad_mask",
    "atom_to_token",
    "contact_conditioning",
    "contact_threshold",
    "cyclic_period",
    "deletion_mean",
    "method_feature",
    "modified",
    "mol_type",
    "profile",
    "ref_atom_name_chars",
    "ref_charge",
    "ref_element",
    "ref_pos",
    "ref_space_uid",
    "res_type",
    "token_bonds",
    "token_pad_mask",
    "type_bonds",
)

_PAIR_CHANNELS = 4
_FOURIER = 4


def _consumer_features(*, compact: bool, mutate: str = "") -> dict[str, jnp.ndarray]:
    """Fixture features with varied pair categories.

    1UBQ_A carries no contact constraints and no token bonds, so its pair
    categories are uniform: comparing them dense against compact would pass
    whatever the graph did with them. Real, varied values are installed here so
    the comparison has something to see (see the tripwire test below).
    """

    with np.load(_FIXTURE) as archive:
        feats: dict[str, np.ndarray] = {
            name: np.asarray(archive[name]) for name in _CONSUMER_FEATURES
        }
    rng = np.random.default_rng(20260912)
    tokens = int(feats["token_pad_mask"].shape[-1])
    feats["contact_conditioning"] = _one_hot(
        rng.integers(0, 5, size=(1, tokens, tokens)).astype(np.uint8), 5
    )
    feats["type_bonds"] = rng.integers(0, 4, size=(1, tokens, tokens)).astype(np.int64)
    feats["token_bonds"] = rng.integers(0, 2, size=(1, tokens, tokens, 1)).astype(
        np.float32
    )
    if mutate:
        feats[mutate] = _mutated(feats[mutate], mutate)
    if compact:
        feats = dict(compact_category_storage(feats))
    return {name: jnp.asarray(value) for name, value in feats.items()}


def _mutated(value: np.ndarray, name: str) -> np.ndarray:
    """Change one publisher array without leaving its exact contract."""

    changed = value.copy()
    if name == "type_bonds":
        changed[0, 0, 0] = (int(changed[0, 0, 0]) + 1) % 4
    elif name == "token_bonds":
        changed[0, 0, 0, 0] = 1.0 - float(changed[0, 0, 0, 0])
    else:
        classes = changed.shape[-1]
        row = changed[0, 0, 0] if changed.ndim == 4 else changed[0, 0]
        current = int(np.argmax(row)) if int(np.sum(row)) else classes
        row[...] = 0
        row[(current + 1) % classes] = 1
    return changed


def _consumer_params() -> dict[str, object]:
    from tests.models.boltz2.test_atom_cp_model_integration import _native_params

    rng = np.random.default_rng(20260911)

    def weight(rows: int, cols: int):
        return jnp.asarray(rng.normal(scale=0.05, size=(rows, cols)), jnp.float32)

    native = _native_params()
    conditioning = native["diffusion_conditioning"]
    score = native["score_model"]
    channels = 8
    input_embedder = {
        "atom_encoder": conditioning["atom_encoder"],
        "atom_enc_proj_z": conditioning["atom_enc_proj_z"][0],
        "atom_attention_encoder": score["atom_attention_encoder"],
        "res_type_encoding": {"kernel": weight(33, channels)},
        "msa_profile_encoding": {"kernel": weight(34, channels)},
        "method_conditioning_init": jnp.zeros((2, channels), jnp.float32),
        "modified_conditioning_init": jnp.zeros((1, channels), jnp.float32),
        "cyclic_conditioning_init": {"kernel": jnp.zeros((1, channels), jnp.float32)},
        "mol_type_conditioning_init": jnp.zeros((1, channels), jnp.float32),
    }
    return {
        "trunk": {
            "input_embedder": input_embedder,
            "contact_conditioning": {
                "fourier_embedding": {
                    "proj": {
                        "kernel": weight(1, _FOURIER),
                        "bias": jnp.asarray(
                            rng.normal(scale=0.01, size=(_FOURIER,)), jnp.float32
                        ),
                    }
                },
                "encoder": {
                    "kernel": weight(4 + _FOURIER, _PAIR_CHANNELS),
                    "bias": jnp.asarray(
                        rng.normal(scale=0.01, size=(_PAIR_CHANNELS,)), jnp.float32
                    ),
                },
                "encoding_unspecified": jnp.asarray(
                    rng.normal(scale=0.1, size=(_PAIR_CHANNELS,)), jnp.float32
                ),
                "encoding_unselected": jnp.asarray(
                    rng.normal(scale=0.1, size=(_PAIR_CHANNELS,)), jnp.float32
                ),
            },
            "token_bonds": {"kernel": weight(1, _PAIR_CHANNELS)},
            "token_bonds_type": weight(256, _PAIR_CHANNELS),
        }
    }


def _consumer_trunk(params, feats, **_kwargs):
    """Run the real consumers of every compacted feature on the model input."""

    from foldjax.models.boltz2.models.trunk_blocks.input_embedder import (
        input_embedder_forward,
    )
    from foldjax.models.boltz2.models.trunk_blocks.trunk import (
        _linear,
        contact_conditioning_forward,
    )

    s_inputs = input_embedder_forward(params["input_embedder"], feats)
    z = contact_conditioning_forward(params["contact_conditioning"], feats)
    z = z + _linear(
        feats["token_bonds"].astype(jnp.float32), params["token_bonds"]["kernel"]
    )
    z = z + params["token_bonds_type"][feats["type_bonds"].astype(jnp.int32)]
    return {"s_inputs": s_inputs, "s": s_inputs, "z": z}


def _run_trunk_only(feats, monkeypatch):
    import foldjax.models.boltz2.models.predict as predict_module

    monkeypatch.setattr(predict_module, "boltz2_trunk_forward", _consumer_trunk)
    return jax.jit(
        lambda model_feats: predict_module.boltz2_predict(
            _consumer_params(),
            model_feats,
            jax.random.PRNGKey(0),
            stop_after_trunk=True,
            return_representations=("single_inputs", "single", "pair"),
        )
    )(feats)


def test_trunk_only_output_is_bitwise_identical_with_and_without_compaction(
    monkeypatch,
) -> None:
    dense = _run_trunk_only(_consumer_features(compact=False), monkeypatch)
    compact = _run_trunk_only(_consumer_features(compact=True), monkeypatch)

    assert set(dense) == {"single_inputs", "single", "pair"}
    for name, expected in dense.items():
        assert compact[name].dtype == expected.dtype, name
        assert np.array_equal(np.asarray(compact[name]), np.asarray(expected)), name
    assert np.ptp(np.asarray(dense["pair"])) > 0


@pytest.mark.parametrize("name", _DENSE_NAMES)
def test_every_compacted_feature_reaches_the_trunk_output(name, monkeypatch) -> None:
    """A dropped feature would make the bitwise comparison above vacuous."""

    baseline = _run_trunk_only(_consumer_features(compact=False), monkeypatch)
    mutated = _run_trunk_only(
        _consumer_features(compact=False, mutate=name), monkeypatch
    )
    compact_mutated = _run_trunk_only(
        _consumer_features(compact=True, mutate=name), monkeypatch
    )

    assert any(
        not np.array_equal(np.asarray(mutated[key]), np.asarray(value))
        for key, value in baseline.items()
    ), name
    for key, value in mutated.items():
        assert np.array_equal(np.asarray(compact_mutated[key]), np.asarray(value)), (
            name,
            key,
        )


def _run_full(feats, monkeypatch):
    import foldjax.models.boltz2.models.predict as predict_module

    monkeypatch.setattr(predict_module, "boltz2_trunk_forward", _consumer_trunk)
    monkeypatch.setattr(
        predict_module,
        "boltz2_sample_forward",
        lambda _params, _feats, _key, *, trunk, **_kwargs: {
            "sample_atom_coords": jnp.sum(trunk["z"], axis=2)[..., :3]
            + jnp.sum(trunk["s_inputs"], axis=-1, keepdims=True)
        },
    )
    monkeypatch.setattr(
        predict_module,
        "affinity_module_forward",
        lambda _params, **kwargs: {
            "affinity_pred_value": jnp.sum(kwargs["z"], axis=(1, 2, 3))
        },
    )
    return jax.jit(
        lambda model_feats: predict_module.boltz2_predict(
            _consumer_params(),
            model_feats,
            jax.random.PRNGKey(0),
            run_confidence=False,
            run_distogram=False,
            affinity_params={"head": {"kernel": jnp.ones((1,))}},
        )
    )(feats)


def test_affinity_path_output_is_bitwise_identical_with_and_without_compaction(
    monkeypatch,
) -> None:
    dense = _run_full(_consumer_features(compact=False), monkeypatch)
    compact = _run_full(_consumer_features(compact=True), monkeypatch)

    assert "affinity_pred_value" in dense
    assert "sample_atom_coords" in dense
    for name, expected in dense.items():
        actual, expected = np.asarray(compact[name]), np.asarray(expected)
        assert actual.dtype == expected.dtype, name
        assert np.array_equal(actual, expected), name
        assert np.any(expected != 0), f"{name} is a vacuous all-zero comparison"


# --- managed API: private on the model input, dense in public output -------


def _api_features(*, affinity: bool) -> dict[str, np.ndarray]:
    features = _synthetic(tokens=2, atoms=3)
    features["mol_type"] = np.asarray([[0, 3]], dtype=np.int32)
    features["token_to_rep_atom"] = np.asarray([[[1, 0, 0], [0, 0, 1]]], np.int64)
    features["atom_to_token"] = np.asarray([[[1, 0], [1, 0], [0, 1]]], np.int64)
    features["affinity_token_mask"] = np.asarray(
        [[0.0, 1.0 if affinity else 0.0]], dtype=np.float32
    )
    return features


def _managed_model_features(
    tmp_path, monkeypatch, *, affinity: bool = False, **predict_kwargs
):
    public = _api_features(affinity=affinity)
    seen: list[dict] = []
    monkeypatch.setattr(api, "featurize", lambda **kwargs: (public, "job", tmp_path))
    affinity_weights = tmp_path / "boltz2_aff"
    if affinity:
        affinity_weights.with_suffix(".npz").touch()
        monkeypatch.setattr(
            api,
            "_prepare_affinity_features",
            lambda **kwargs: _api_features(affinity=True),
            raising=False,
        )
        predict_kwargs["affinity_weights"] = affinity_weights

    def fake_load_params(path):
        if Path(path) == affinity_weights:
            return {"trunk": {}, "affinity": {"modules": [{"head": jnp.ones((1,))}]}}
        return {"trunk": {}}

    monkeypatch.setattr(
        "foldjax.models.boltz2.bridge.native.load_params", fake_load_params
    )

    def fake_predict(params, model_features, _key, **kwargs):
        seen.append(dict(model_features))
        tokens = model_features["token_pad_mask"].shape[-1]
        samples = int(kwargs.get("multiplicity", 1))
        if kwargs.get("stop_after_trunk"):
            return {
                "single": jnp.ones((1, tokens, 4), jnp.float32),
                "pair": jnp.ones((1, tokens, tokens, 4), jnp.float32),
            }
        out = {
            "sample_atom_coords": jnp.zeros((samples, 3, 3)),
            "plddt": jnp.ones((samples, 3)),
            "iptm": jnp.zeros((samples,)),
        }
        if "affinity" in params:
            out["affinity_pred_value"] = jnp.zeros((1,))
        return out

    monkeypatch.setattr(
        "foldjax.models.boltz2.models.predict.boltz2_predict", fake_predict
    )
    api.predict(
        seq=["AC"],
        weights=tmp_path / "boltz2_conf",
        mols=tmp_path,
        out_dir=tmp_path,
        num_steps=1,
        **predict_kwargs,
    )
    return public, seen


def test_managed_full_path_hands_the_model_private_categories(
    tmp_path, monkeypatch
) -> None:
    public, seen = _managed_model_features(tmp_path, monkeypatch)

    assert len(seen) == 1
    model_features = seen[0]
    for name in _DENSE_NAMES:
        assert name in public
        assert name not in model_features
    for name in _PRIVATE_NAMES:
        assert name not in public
        assert name in model_features
    assert model_features[CONTACT_CONDITIONING_IDS].dtype == jnp.uint8
    assert model_features[TOKEN_BONDS_FLAGS].dtype == jnp.bool_
    assert model_features[TYPE_BONDS_IDS].dtype == jnp.uint8
    assert model_features[REF_ELEMENT_IDS].dtype == jnp.uint8
    assert model_features[REF_ATOM_NAME_CHAR_IDS].dtype == jnp.uint8


def test_managed_trunk_only_path_hands_the_model_private_categories(
    tmp_path, monkeypatch
) -> None:
    _, seen = _managed_model_features(
        tmp_path, monkeypatch, stop_after="trunk", representations=("single",)
    )

    model_features = seen[0]
    for name in _DENSE_NAMES:
        assert name not in model_features
    for name in _PRIVATE_NAMES:
        assert name in model_features


def test_managed_affinity_path_hands_the_model_private_categories(
    tmp_path, monkeypatch
) -> None:
    _, seen = _managed_model_features(tmp_path, monkeypatch, affinity=True)

    assert len(seen) == 2
    for model_features in seen:
        for name in _DENSE_NAMES:
            assert name not in model_features
        for name in _PRIVATE_NAMES:
            assert name in model_features


def test_eager_steering_keeps_the_dense_categories(tmp_path, monkeypatch) -> None:
    """Steering runs eagerly and owns the dense contract, so it must not compact."""

    public, seen = _managed_model_features(
        tmp_path, monkeypatch, steering_args={"fk_steering": True}
    )

    model_features = seen[0]
    for name in _DENSE_NAMES:
        assert name in model_features
        assert np.array_equal(np.asarray(model_features[name]), public[name])
    for name in _PRIVATE_NAMES:
        assert name not in model_features


def test_public_features_survive_a_managed_prediction_unchanged(
    tmp_path, monkeypatch
) -> None:
    """The caller's featurizer dict is never the compacted model input."""

    before = {
        name: value.copy() for name, value in _api_features(affinity=False).items()
    }
    public, seen = _managed_model_features(tmp_path, monkeypatch)

    assert public is not seen[0]
    assert set(public) == set(before)
    for name, value in before.items():
        assert np.array_equal(public[name], value), name
    for name in _PRIVATE_NAMES:
        assert name not in public
        assert name in seen[0]


@pytest.mark.parametrize(
    "private",
    [
        {COMPACT_CONTACT_CONDITIONING: np.asarray(1, dtype=np.uint8)},
        {CONTACT_CONDITIONING_IDS: np.zeros((1, 2, 2), dtype=np.uint8)},
        {TOKEN_BONDS_FLAGS: np.zeros((1, 2, 2, 1), dtype=np.bool_)},
        {REF_ATOM_NAME_CHAR_IDS: np.zeros((1, 3, 4), dtype=np.uint8)},
    ],
)
def test_managed_api_rejects_incomplete_private_categories(
    tmp_path, monkeypatch, private
) -> None:
    features = _api_features(affinity=False)
    for name in _DENSE_NAMES:
        del features[name]
    features.update(private)
    monkeypatch.setattr(api, "featurize", lambda **kwargs: (features, "job", tmp_path))
    monkeypatch.setattr(
        "foldjax.models.boltz2.bridge.native.load_params", lambda path: {"trunk": {}}
    )

    with pytest.raises(ValueError, match="private group"):
        api.predict(
            seq=["AC"],
            weights=tmp_path / "boltz2_conf",
            mols=tmp_path,
            out_dir=tmp_path,
            num_steps=1,
        )
