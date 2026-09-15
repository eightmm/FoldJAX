from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from foldjax.backends.base import Backend
from foldjax.cli import _parser, _request
from foldjax.models._random import masked_prefix_draw
from foldjax.padding import PaddingPlan, resolve_axis
from foldjax.registry import capabilities
from foldjax.schema import (
    ModelCapabilities,
    PaddingConfig,
    PredictionRequest,
    PredictionResult,
)


def _input(tmp_path: Path) -> Path:
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"entities": []}), encoding="utf-8")
    return path


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "256"])
def test_padding_targets_are_strict_positive_integers(value: object) -> None:
    with pytest.raises(ValueError, match=r"padding\.tokens"):
        PaddingConfig(tokens=value)  # type: ignore[arg-type]


def test_padding_config_normalizes_numpy_integers_and_summarizes() -> None:
    config = PaddingConfig(tokens=np.int64(512), msa=np.int32(128))

    assert config.tokens == 512 and type(config.tokens) is int
    assert config.msa == 128 and type(config.msa) is int
    assert config.explicit_axes == ("tokens", "msa")
    assert config.summary() == {
        "tokens": 512,
        "atoms": None,
        "msa": 128,
        "templates": None,
        "structural_tokens": None,
        "language_model_tokens": None,
        "overflow": "error",
    }


@pytest.mark.parametrize("value", ["grow", 1, None])
def test_padding_overflow_is_strict(value: object) -> None:
    with pytest.raises(ValueError, match="padding.overflow"):
        PaddingConfig(overflow=value)  # type: ignore[arg-type]


def test_request_accepts_simple_boolean_and_mapping_padding(tmp_path: Path) -> None:
    path = _input(tmp_path)

    automatic = PredictionRequest(model="boltz2", input=path, padding=True)
    pinned = PredictionRequest(
        model="boltz2",
        input=path,
        padding={"tokens": 512, "overflow": "exact"},
    )
    disabled = PredictionRequest(model="boltz2", input=path, padding=False)

    assert automatic.padding == PaddingConfig()
    assert pinned.padding == PaddingConfig(tokens=512, overflow="exact")
    assert disabled.padding is None


def test_request_rejects_unknown_padding_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported padding fields: 'token'"):
        PredictionRequest(
            model="boltz2",
            input=_input(tmp_path),
            padding={"token": 512},
        )


def test_resolve_axis_selects_standard_pinned_and_overflow_targets() -> None:
    automatic = PaddingConfig()
    assert resolve_axis(300, automatic, "tokens") == 512
    assert resolve_axis(300, PaddingConfig(tokens=768), "tokens") == 768
    assert resolve_axis(9000, PaddingConfig(overflow="exact"), "tokens") == 9000
    with pytest.raises(ValueError, match="largest standard bucket 8192"):
        resolve_axis(9000, automatic, "tokens")
    with pytest.raises(ValueError, match="smaller than the input size 600"):
        resolve_axis(
            300,
            PaddingConfig(tokens=512),
            "tokens",
            minimum=600,
        )


def test_padding_plan_reports_real_storage_and_target_shapes() -> None:
    plan = PaddingPlan(
        actual={"tokens": 300, "atoms": 2000},
        storage={"tokens": 320, "atoms": 2016},
        target={"tokens": 512, "atoms": 3072},
    )

    assert plan.changed is True
    assert plan.summary()["target"] == {"tokens": 512, "atoms": 3072}
    assert "tokens 300 (stored 320) -> 512" in plan.message("model")


def test_masked_random_draw_preserves_compact_matrix_stream() -> None:
    import jax
    import jax.numpy as jnp

    key = jax.random.key(23)
    exact = jax.random.normal(key, (2, 3, 3, 4))
    token_mask = jnp.asarray([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]], dtype=bool)
    pair_mask = token_mask[:, :, None] & token_mask[:, None, :]
    padded = masked_prefix_draw(
        lambda draw_key, shape: jax.random.normal(draw_key, shape),
        key,
        pair_mask,
        trailing_shape=(4,),
    )

    np.testing.assert_array_equal(np.asarray(padded[:, :3, :3]), np.asarray(exact))
    assert not np.any(np.asarray(padded)[:, 3:])
    assert not np.any(np.asarray(padded)[:, :, 3:])


def test_result_summary_only_adds_concrete_shape_profile_when_present() -> None:
    plain = PredictionResult(model="boltz2")
    padded = PredictionResult(
        model="boltz2",
        shape_profile={
            "actual": {"tokens": 300},
            "target": {"tokens": 512},
            "changed": True,
        },
    )

    assert "shape_profile" not in plain.summary()
    assert padded.summary()["shape_profile"] == padded.shape_profile


def test_all_builtin_models_advertise_their_complete_padding_profile() -> None:
    expected = {
        "alphafold3": ("tokens",),
        "boltz2": ("tokens", "atoms", "msa"),
        "esmfold2": ("tokens", "atoms", "msa", "language_model_tokens"),
        "opendde": ("tokens", "atoms", "msa", "structural_tokens"),
        "openfold3": ("tokens", "atoms", "msa", "templates"),
        "protenix": (
            "tokens",
            "atoms",
            "msa",
            "templates",
            "language_model_tokens",
        ),
    }

    assert {model: capabilities(model).padding_axes for model in expected} == expected


def test_cli_padding_shortcut_and_exact_targets(tmp_path: Path) -> None:
    args = _parser().parse_args(
        [
            "predict",
            "--model",
            "boltz2",
            "--input",
            str(_input(tmp_path)),
            "--pad-tokens",
            "512",
            "--pad-atoms",
            "4096",
            "--pad-msa",
            "128",
            "--padding-overflow",
            "exact",
        ]
    )

    request = _request(args)

    assert request.padding == PaddingConfig(
        tokens=512,
        atoms=4096,
        msa=128,
        overflow="exact",
    )

    automatic_args = _parser().parse_args(
        [
            "cache",
            "warm",
            "--model",
            "boltz2",
            "--input",
            str(_input(tmp_path)),
            "--padding",
        ]
    )
    assert _request(automatic_args).padding == PaddingConfig()


def test_cli_rejects_overflow_policy_without_padding(tmp_path: Path) -> None:
    args = _parser().parse_args(
        [
            "plan",
            "--model",
            "boltz2",
            "--input",
            str(_input(tmp_path)),
            "--padding-overflow",
            "exact",
        ]
    )
    with pytest.raises(ValueError, match="requires --padding"):
        _request(args)


class _NoPaddingBackend(Backend):
    name = "third-party"

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(model=self.name, input_formats=("foldjax",))

    def predict(self, request: PredictionRequest) -> PredictionResult:
        raise AssertionError("not called")


class _TokenPaddingBackend(_NoPaddingBackend):
    padding_axes = ("tokens",)


def test_backend_padding_support_is_fail_closed(tmp_path: Path) -> None:
    path = _input(tmp_path)
    request = PredictionRequest(
        model="third-party",
        input=path,
        input_format="foldjax",
        padding=True,
    )
    with pytest.raises(ValueError, match="does not support input padding"):
        _NoPaddingBackend().validate_request(request)

    request = PredictionRequest(
        model="third-party",
        input=path,
        input_format="foldjax",
        padding=PaddingConfig(atoms=512),
    )
    with pytest.raises(ValueError, match="explicit padding axes: atoms"):
        _TokenPaddingBackend().validate_request(request)


def test_host_padding_policy_does_not_fragment_cache_namespace(tmp_path: Path) -> None:
    path = _input(tmp_path)
    backend = _TokenPaddingBackend()
    exact = PredictionRequest(model="third-party", input=path)
    automatic = PredictionRequest(model="third-party", input=path, padding=True)
    pinned = PredictionRequest(
        model="third-party", input=path, padding=PaddingConfig(tokens=512)
    )

    assert backend.cache_profile(exact) == backend.cache_profile(automatic)
    assert backend.cache_profile(exact) == backend.cache_profile(pinned)


@pytest.mark.parametrize(
    "axis,expected",
    [("atoms", 6144), ("structural_tokens", 512), ("language_model_tokens", 256)],
)
def test_token_profile_capacity_is_independent_of_real_size(axis, expected):
    from foldjax.padding import resolve_token_axis

    for actual in (1, 17, 100):
        assert (
            resolve_token_axis(actual, PaddingConfig(), axis, token_target=256)
            == expected
        )


def test_token_profile_preserves_pins_and_rejects_storage_overflow():
    from foldjax.padding import resolve_token_axis

    assert (
        resolve_token_axis(3, PaddingConfig(atoms=32), "atoms", token_target=256) == 32
    )
    assert (
        resolve_token_axis(3, PaddingConfig(), "msa", token_target=256, fixed_size=1280)
        == 1280
    )
    assert resolve_token_axis(3, PaddingConfig(), "atoms", token_target=3) == 96
    with pytest.raises(ValueError, match="smaller than"):
        resolve_token_axis(
            3, PaddingConfig(overflow="exact"), "atoms", token_target=256, minimum=6145
        )
    with pytest.raises(ValueError, match="fixed_size"):
        resolve_token_axis(3, PaddingConfig(), "msa", token_target=256)


@pytest.mark.parametrize(
    "model", ["alphafold3", "boltz2", "protenix", "opendde", "openfold3", "esmfold2"]
)
def test_padding_never_spells_an_msa_depth_and_preserves_overrides(model, tmp_path):
    """A shape switch must not select which alignment rows a model reads.

    Deriving the depth from the padding profile is how ``--padding`` came to
    cap a 16,384-row Protenix alignment at 1,024 rows -- measured as a 19%
    lower peak and coordinates 0.2-0.3 A away from the exact run, none of it
    from padding.  Padding now pads that axis up instead, and the depth stays
    where the port's own default and ``max_msa_depth`` put it.
    """

    from foldjax.registry import get_backend

    path = _input(tmp_path)
    backend = get_backend(model)
    request = PredictionRequest(model=model, input=path, padding=True)
    assert "max_msa_depth" not in backend.apply_sampling(request)
    # An MSA padding target is a capacity for that axis, not a row selection.
    pinned = PredictionRequest(model=model, input=path, padding=PaddingConfig(msa=64))
    assert "max_msa_depth" not in backend.apply_sampling(pinned)
    # ... so an MSA target names the same program the token axis does, with
    # the concrete padded shapes joining the cache namespace at run time.
    assert backend.cache_profile(pinned) == backend.cache_profile(
        PredictionRequest(model=model, input=path, padding=PaddingConfig(tokens=512))
    )
    overridden = PredictionRequest(
        model=model, input=path, padding=True, max_msa_depth=128
    )
    assert backend.apply_sampling(overridden)["max_msa_depth"] == 128
    # A named depth is a different input, and therefore a different program.
    assert backend.cache_profile(overridden) != backend.cache_profile(request)
    native = PredictionRequest(
        model=model,
        input=path,
        padding=True,
        options={"max_msa_depth": 128},
    )
    assert backend.apply_sampling(native)["max_msa_depth"] == 128
    unpadded = PredictionRequest(model=model, input=path)
    assert "max_msa_depth" not in backend.apply_sampling(unpadded)


def test_msa_axis_pads_up_to_a_bucket_and_refuses_a_target_below_storage():
    """The MSA axis follows the token and atom rule: the bucket that fits."""

    from foldjax.padding import (
        MSA_BUCKETS,
        MSA_PROFILE_DEPTH,
        OPENDDE_MSA_PROFILE_DEPTH,
        cp_aligned_padding,
        resolve_msa_axis,
    )

    automatic = PaddingConfig()
    # Deeper than the profile floor: the next bucket up, never a crop.
    assert resolve_msa_axis(3000, automatic, minimum=3000) == 4096
    assert resolve_msa_axis(16384, automatic, minimum=16384) == 16384
    # Shallower: the profile's preferred floor, so one token band shares one
    # executable.  OpenDDE's floor is its released per-cycle depth.
    assert resolve_msa_axis(300, automatic, minimum=300) == MSA_PROFILE_DEPTH
    assert (
        resolve_msa_axis(
            300, automatic, minimum=300, profile_depth=OPENDDE_MSA_PROFILE_DEPTH
        )
        == OPENDDE_MSA_PROFILE_DEPTH
    )
    assert (
        resolve_msa_axis(
            2000, automatic, minimum=2000, profile_depth=OPENDDE_MSA_PROFILE_DEPTH
        )
        == 2048
    )
    # An active input cap bounds the masked capacity but never the rows.
    assert resolve_msa_axis(2, automatic, minimum=2, input_depth=128) == 128
    assert resolve_msa_axis(5, automatic, minimum=5, input_depth=8) == 8
    with pytest.raises(ValueError, match="deeper than max_msa_depth=128"):
        resolve_msa_axis(129, automatic, minimum=129, input_depth=128)
    # A pin is a target.  Below the stored rows it is refused, and the message
    # names the option that does change the input.
    assert resolve_msa_axis(300, PaddingConfig(msa=2048), minimum=300) == 2048
    with pytest.raises(ValueError, match=r"never truncates.*--max-msa-depth"):
        resolve_msa_axis(3000, PaddingConfig(msa=2048), minimum=3000)
    # Above the grid the overflow policy decides, as on every other axis.
    with pytest.raises(ValueError, match="exceeds the largest standard bucket"):
        resolve_msa_axis(MSA_BUCKETS[-1] + 1, automatic, minimum=MSA_BUCKETS[-1] + 1)
    assert (
        resolve_msa_axis(20000, PaddingConfig(overflow="exact"), minimum=20000) == 20000
    )
    # The mesh never splits this axis, so alignment must not inflate it.
    for devices in (2, 3, 4):
        assert (
            resolve_msa_axis(
                300, cp_aligned_padding(automatic, cp_devices=devices), minimum=300
            )
            == MSA_PROFILE_DEPTH
        )


def test_token_grid_reaches_past_one_card_and_still_refuses_above_it() -> None:
    """The grid has to cover what a mesh runs, not what one card runs."""

    from foldjax.padding import TOKEN_BUCKETS

    assert TOKEN_BUCKETS[0] == 256
    assert TOKEN_BUCKETS[-1] == 8192
    assert list(TOKEN_BUCKETS) == sorted(set(TOKEN_BUCKETS))
    automatic = PaddingConfig()
    # 4,888 tokens is the largest target in the scale set, and the reason the
    # grid reaches past one card: it was refused outright while a four-device
    # mesh exists to run exactly that size.
    assert resolve_axis(4888, automatic, "tokens") == 5120
    assert resolve_axis(5121, automatic, "tokens") == 5376
    assert resolve_axis(6145, automatic, "tokens") == 6400
    with pytest.raises(ValueError, match="largest standard bucket 8192"):
        resolve_axis(8193, automatic, "tokens")
    assert resolve_axis(8193, PaddingConfig(overflow="exact"), "tokens") == 8193


def test_a_token_bucket_costs_at_most_one_256_token_step() -> None:
    """The grid's contract is the bound, not the list of sizes.

    A geometric grid let a job pay for a whole doubling: 2,096 tokens landed
    on 3,072, measured at +97% wall and +44% peak against the exact shape on
    Protenix.  A constant step bounds that at one step everywhere, which is
    what lets padding be the normal way to run rather than an option.
    """

    from foldjax.padding import TOKEN_BUCKETS

    assert len(TOKEN_BUCKETS) == 32
    assert {
        later - earlier for earlier, later in zip(TOKEN_BUCKETS, TOKEN_BUCKETS[1:])
    } == {256}
    automatic = PaddingConfig()
    assert resolve_axis(2096, automatic, "tokens") == 2304
    assert resolve_axis(3012, automatic, "tokens") == 3072
    assert resolve_axis(4888, automatic, "tokens") == 5120
    # The bound itself, over every size the grid accepts.
    for actual in range(1, TOKEN_BUCKETS[-1] + 1):
        assert resolve_axis(actual, automatic, "tokens") - actual < 256
    # ... and the worst case at 2k is one step of quadratic work, not a
    # doubling: 12.5% of 2,048 rather than 50%.
    assert (resolve_axis(2049, automatic, "tokens") - 2048) / 2048 == 0.125


@pytest.mark.parametrize("cp_devices,cp_layout", [(3, "auto"), (3, "1d"), (9, "2d")])
def test_three_mesh_rows_compose_with_a_256_token_grid(
    cp_devices: int, cp_layout: str
) -> None:
    """256 is not a multiple of 3, so pin what three rows actually do.

    ``_cp_aligned_target`` rounds a bucket hit up to the next multiple of
    ``rows``; it never refuses one.  Ten of the thirty-two buckets are
    multiples of 3 already -- the bucket index has to carry the factor, since
    256 does not -- and 2,304 (``256 * 9``) is one of them, so at 2,096 tokens
    three rows get the bucket itself.  For the other twenty-two the target
    lands one or two tokens above its bucket, which the tail of this test
    pins, because that is the case a 256 grid does not cover by construction.
    """

    from foldjax.padding import cp_aligned_padding, resolve_token_axis

    config = cp_aligned_padding(
        PaddingConfig(), cp_devices=cp_devices, cp_layout=cp_layout
    )
    tokens = resolve_axis(2096, config, "tokens")
    assert tokens == 2304
    assert tokens % 3 == 0
    # 24 * 2304 = 55,296 is a multiple of 32 and of 32 * 3, so the atom axis
    # needs no rounding either.
    assert resolve_token_axis(1, config, "atoms", token_target=tokens) == 55296
    assert resolve_token_axis(1, config, "structural_tokens", token_target=tokens) == (
        4608
    )

    # A bucket three rows do not divide: 2,000 -> 2,048 -> 2,049, and the atom
    # target then derives from 2,049 and does need its own rounding.
    off_grid = resolve_axis(2000, config, "tokens")
    assert off_grid == 2049
    assert resolve_token_axis(1, config, "atoms", token_target=off_grid) == 49248


def test_a_bucket_derives_an_atom_target_every_plausible_mesh_divides() -> None:
    """``24 * 256 = 6,144 = 32 * 192``, so a bucket's atom target divides
    ``32 * rows`` for every ``rows`` that divides 192.  The geometric grid had
    this too -- every one of its buckets was a multiple of 256 -- and the
    256-step grid preserves it rather than gaining it."""

    from foldjax.padding import cp_aligned_padding, resolve_token_axis

    assert resolve_token_axis(1, PaddingConfig(), "atoms", token_target=2304) == 55296
    for cp_devices in (2, 3, 4, 6, 8):
        config = cp_aligned_padding(PaddingConfig(), cp_devices=cp_devices)
        assert resolve_token_axis(1, config, "atoms", token_target=2304) == 55296
    # Five rows do not divide 192; the shared rule then rounds up, as it did
    # before this grid existed.
    five = cp_aligned_padding(PaddingConfig(), cp_devices=5)
    assert resolve_token_axis(1, five, "atoms", token_target=2304) == 55360


def test_derived_grids_cover_the_largest_token_bucket() -> None:
    """No axis may refuse a size the token axis accepts."""

    from foldjax.padding import (
        ATOM_BUCKETS,
        LANGUAGE_MODEL_TOKEN_BUCKETS,
        STRUCTURAL_TOKEN_BUCKETS,
        TOKEN_BUCKETS,
    )

    largest = TOKEN_BUCKETS[-1]
    assert ATOM_BUCKETS[-1] >= 24 * largest
    assert STRUCTURAL_TOKEN_BUCKETS[-1] >= 2 * largest
    assert LANGUAGE_MODEL_TOKEN_BUCKETS[-1] >= largest


def test_cp_mesh_rows_matches_the_layout_the_real_resolver_builds() -> None:
    """Pin the host-side reimplementation against the JAX-side resolver."""

    import math

    from foldjax.models._cp import resolve_cp_layout
    from foldjax.padding import cp_mesh_rows

    for devices in (1, 2, 3, 4, 6, 8, 9, 16):
        for layout in ("auto", "1d", "2d"):
            try:
                resolved = resolve_cp_layout(layout, devices)
            except ValueError:
                with pytest.raises(ValueError):
                    cp_mesh_rows(devices, layout)
                continue
            expected = math.isqrt(devices) if resolved == "2d" else devices
            assert cp_mesh_rows(devices, layout) == expected
    with pytest.raises(ValueError, match="must be positive"):
        cp_mesh_rows(0)
    with pytest.raises(ValueError, match="must be an integer"):
        cp_mesh_rows(True)
    with pytest.raises(ValueError, match="'auto', '1d', or '2d'"):
        cp_mesh_rows(4, "grid")


@pytest.mark.parametrize(
    "cp_devices,cp_layout,rows", [(4, "1d", 4), (4, "2d", 2), (3, "auto", 3)]
)
def test_mesh_alignment_rounds_automatic_token_atom_and_structural_targets(
    cp_devices: int, cp_layout: str, rows: int
) -> None:
    from foldjax.padding import (
        cp_aligned_padding,
        resolve_axis,
        resolve_token_axis,
    )

    config = cp_aligned_padding(
        PaddingConfig(), cp_devices=cp_devices, cp_layout=cp_layout
    )
    tokens = resolve_axis(4888, config, "tokens")
    assert tokens % rows == 0
    atoms = resolve_token_axis(117312, config, "atoms", token_target=tokens)
    assert atoms % (32 * rows) == 0
    structural = resolve_token_axis(1, config, "structural_tokens", token_target=tokens)
    assert structural % rows == 0

    # A token pin the mesh divides but whose derived atom count it does not:
    # ((24 * 3012 + 31) // 32) * 32 == 72288, a multiple of 32 and of nothing
    # larger. This is the case the whole change exists for.
    pinned = cp_aligned_padding(
        PaddingConfig(tokens=3012), cp_devices=cp_devices, cp_layout=cp_layout
    )
    pinned_tokens = resolve_axis(3012, pinned, "tokens")
    assert pinned_tokens == 3012
    assert resolve_token_axis(1, pinned, "atoms", token_target=pinned_tokens) == (
        ((72288 + 32 * rows - 1) // (32 * rows)) * 32 * rows
    )
    # ... and a token pin whose doubled structural count the mesh does not
    # divide either.
    odd = cp_aligned_padding(
        PaddingConfig(tokens=1003), cp_devices=cp_devices, cp_layout=cp_layout
    )
    assert resolve_token_axis(1, odd, "structural_tokens", token_target=1003) == (
        ((2006 + rows - 1) // rows) * rows
    )


def test_mesh_alignment_leaves_explicit_pins_and_unsharded_axes_alone() -> None:
    """A pin is a statement about the compiled shape, not a starting point."""

    from foldjax.padding import (
        MSA_PROFILE_DEPTH,
        cp_aligned_padding,
        resolve_axis,
        resolve_token_axis,
    )

    config = cp_aligned_padding(
        PaddingConfig(tokens=1003, atoms=72288, structural_tokens=2006),
        cp_devices=4,
    )
    assert resolve_axis(1003, config, "tokens") == 1003
    assert resolve_token_axis(1, config, "atoms", token_target=1003) == 72288
    assert resolve_token_axis(1, config, "structural_tokens", token_target=1003) == 2006
    # The mesh never splits these, so alignment must not inflate them.
    automatic = cp_aligned_padding(PaddingConfig(), cp_devices=3)
    assert (
        resolve_token_axis(
            1, automatic, "msa", token_target=1003, fixed_size=MSA_PROFILE_DEPTH
        )
        == MSA_PROFILE_DEPTH
    )
    assert (
        resolve_token_axis(1, automatic, "templates", token_target=1003, fixed_size=4)
        == 4
    )
    assert (
        resolve_token_axis(1, automatic, "language_model_tokens", token_target=1003)
        == 1003
    )


@pytest.mark.parametrize("tokens", [254, 1003, 2096, 3012, 4888])
def test_a_single_device_resolves_the_profile_every_port_had_before(
    tokens: int,
) -> None:
    """``cp_devices=1`` must reach the same numbers, axis by axis and port by
    port, that the documented formulas give.  The expectations here are written
    out rather than read back from the resolver, so alignment leaking into a
    serial run would fail this."""

    from foldjax.padding import (
        MSA_PROFILE_DEPTH,
        OPENDDE_MSA_PROFILE_DEPTH,
        cp_aligned_padding,
        resolve_axis,
        resolve_msa_axis,
        resolve_token_axis,
    )
    from foldjax.registry import get_backend

    plain = PaddingConfig()
    serial = cp_aligned_padding(plain, cp_devices=1)
    assert serial is plain
    # Not even a layout a mesh would refuse makes a one-device request
    # resolve differently; the ports still reject that combination.
    assert cp_aligned_padding(plain, cp_devices=1, cp_layout="2d") is plain
    with pytest.raises(ValueError, match="must be an integer"):
        cp_aligned_padding(plain, cp_devices=1.0)  # type: ignore[arg-type]
    token_target = resolve_axis(tokens, serial, "tokens")
    for model in (
        "alphafold3",
        "boltz2",
        "protenix",
        "opendde",
        "openfold3",
        "esmfold2",
    ):
        depth = OPENDDE_MSA_PROFILE_DEPTH if model == "opendde" else MSA_PROFILE_DEPTH
        expected = {
            "tokens": token_target,
            "atoms": ((24 * token_target + 31) // 32) * 32,
            "structural_tokens": 2 * token_target,
            "language_model_tokens": token_target,
            "msa": depth,
            "templates": 4,
        }
        for axis in get_backend(model).padding_axes:
            if axis == "tokens":
                resolved = resolve_axis(tokens, serial, axis)
            elif axis == "msa":
                resolved = resolve_msa_axis(1, serial, profile_depth=depth)
            elif axis == "templates":
                resolved = resolve_token_axis(
                    1, serial, axis, token_target=token_target, fixed_size=4
                )
            else:
                resolved = resolve_token_axis(
                    1, serial, axis, token_target=token_target
                )
            assert resolved == expected[axis], (model, axis)


def test_protenix_mesh_request_resolves_aligned_targets_at_4888_tokens(
    tmp_path: Path,
) -> None:
    """The neutral request's own option pipeline, then the shared policy.

    The port's own CLI applies the same helper to the profile it builds from
    ``--pad-*``; that line needs weights to reach and is not exercised here.
    """

    from foldjax.padding import cp_aligned_padding, resolve_axis, resolve_token_axis
    from foldjax.registry import get_backend

    backend = get_backend("protenix")
    request = PredictionRequest(
        model="protenix",
        input=_input(tmp_path),
        input_format="foldjax",
        padding=True,
        options={"cp_devices": 4},
    )
    backend.validate_request(request)
    options = backend.apply_sampling(request)
    config = cp_aligned_padding(
        request.padding,
        cp_devices=int(options["cp_devices"]),
        cp_layout=str(options.get("cp_layout", "auto")),
    )
    tokens = resolve_axis(4888, config, "tokens")
    atoms = resolve_token_axis(117312, config, "atoms", token_target=tokens)
    assert tokens == 5120
    assert tokens % 4 == 0
    assert atoms % 128 == 0


def test_the_boltz2_plan_aligner_agrees_with_the_shared_mesh_policy() -> None:
    """Boltz-2 aligns its resolved plan again; the two must not disagree."""

    from foldjax.models.boltz2.data.bucket import (
        align_padding_plan_for_context_parallel,
        resolve_padding_plan,
    )
    from foldjax.padding import cp_aligned_padding

    feats = {
        "token_pad_mask": np.ones((1, 3012), dtype=np.float32),
        "atom_pad_mask": np.ones((1, 71000), dtype=np.float32),
        "msa": np.ones((1, 1, 3012), dtype=np.int32),
    }
    pinned = PaddingConfig(tokens=3012)
    shared = resolve_padding_plan(
        feats,
        cp_aligned_padding(pinned, cp_devices=4),
        max_msa_depth=1024,
    )
    native = align_padding_plan_for_context_parallel(
        feats,
        resolve_padding_plan(feats, pinned, max_msa_depth=1024),
        cp_rows=4,
        cp_cols=1,
    )
    assert shared.target["tokens"] == native.target["tokens"]
    assert shared.target["atoms"] == native.target["atoms"] == 72320


def test_the_opendde_backend_hands_its_native_path_a_mesh_aware_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from foldjax.backends.opendde import OpenDDEBackend
    from foldjax.padding import cp_rows

    input_path = tmp_path / "job.json"
    weights_path = tmp_path / "opendde.jax"
    input_path.write_text("{}", encoding="utf-8")
    weights_path.write_bytes(b"native")
    seen: dict[str, object] = {}

    def native_main(argv, **kwargs):
        seen["argv"] = argv
        seen.update(kwargs)
        kwargs["padding_profiles"].append({"target": {"msa": 1280}})
        return []

    monkeypatch.setattr(
        "foldjax.backends.opendde.import_module",
        lambda _name: SimpleNamespace(main=native_main),
    )
    OpenDDEBackend().predict(
        PredictionRequest(
            model="opendde",
            input=input_path,
            weights=weights_path,
            output_dir=tmp_path / "out",
            padding=True,
            options={"cp_devices": 4},
        )
    )

    assert "--cp-devices" in seen["argv"]
    assert cp_rows(seen["padding"]) == 4


def test_the_boltz2_backend_hands_its_native_api_a_mesh_aware_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from foldjax.backends.boltz2 import Boltz2Backend
    from foldjax.padding import cp_rows

    mols = tmp_path / "mols"
    mols.mkdir()
    weights = tmp_path / "weights"
    weights.mkdir()
    seen: dict[str, object] = {}

    def native_predict(**kwargs):
        seen.update(kwargs)
        return {
            "coords": np.zeros((2, 3)),
            "plddt": np.asarray([0.7, 0.8]),
            "iptm": np.asarray([0.6]),
            "out_path": tmp_path / "out" / "job.cif",
        }

    monkeypatch.setattr(
        "foldjax.backends.boltz2.import_module",
        lambda _name: SimpleNamespace(predict=native_predict),
    )
    Boltz2Backend().predict(
        PredictionRequest(
            model="boltz2",
            input=_input(tmp_path),
            weights=weights,
            output_dir=tmp_path / "out",
            padding=True,
            options={"mols": mols, "cp_devices": 4},
        )
    )

    assert cp_rows(seen["padding"]) == 4
