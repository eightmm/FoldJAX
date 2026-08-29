"""PAE metrics may cross the serial-confidence map without PAE logits."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3 import inference
from foldjax.models.openfold3.models.clash import sample_ranking_score
from foldjax.models.openfold3.models.confidence import (
    compute_chain_pair_iptm,
    compute_ptm,
)
from tests.models.cp_probe_env import inherited_environment

MAX_CONFIDENCE_DELTA = 1e-3
SAMPLES = 5
PAE_BINS = 64


def _legacy_metrics(logits, has_frame, token_mask, asym_id, *, n_chain):
    kwargs = {"bin_min": 0.0, "bin_max": 32.0, "no_bins": PAE_BINS}
    ptm = compute_ptm(logits, has_frame, token_mask, **kwargs)
    iptm = (
        jnp.zeros_like(ptm)
        if n_chain == 1
        else compute_ptm(
            logits,
            has_frame,
            token_mask,
            asym_id=asym_id,
            interface=True,
            **kwargs,
        )
    )
    chain_pair = None
    if n_chain is not None and n_chain > 1:
        chain_pair = compute_chain_pair_iptm(
            logits,
            has_frame,
            token_mask,
            asym_id,
            n_chain=n_chain,
            **kwargs,
        )
    return ptm, iptm, chain_pair


def _compact_mapped_metrics(logits, has_frame, token_mask, asym_id, *, n_chain):
    kwargs = {"bin_min": 0.0, "bin_max": 32.0, "no_bins": PAE_BINS}
    return jax.lax.map(
        lambda one: jax.tree.map(
            lambda leaf: leaf[0],
            inference._compact_confidence_metrics_from_pae(
                one[0][None],
                one[1][None],
                token_mask,
                asym_id,
                n_chain=n_chain,
                **kwargs,
            ),
        ),
        (logits, has_frame),
    )


def _assert_nonfinite_and_finite_delta(actual, expected) -> None:
    if actual is None or expected is None:
        assert actual is expected is None
        return
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    assert np.isfinite(actual[finite]).all()
    if finite.any():
        assert np.max(np.abs(actual[finite] - expected[finite])) <= MAX_CONFIDENCE_DELTA


def test_compact_map_cpu_fuzz_stays_within_confidence_delta_contract() -> None:
    # Five graph shapes cover unknown/monomer/multimer paths; eight runtime inputs
    # per graph vary chain occupancy, padding and frame masks without recompiling.
    for n_token, n_chain in ((1, None), (2, 1), (9, 2), (33, 4), (65, 3)):
        chain_count = max(1, n_chain or 3)
        legacy_run = jax.jit(
            lambda pair, frames, mask, chains: _legacy_metrics(
                pair, frames, mask, chains, n_chain=n_chain
            )
        )
        compact_run = jax.jit(
            lambda pair, frames, mask, chains: _compact_mapped_metrics(
                pair, frames, mask, chains, n_chain=n_chain
            )
        )
        for seed in range(8):
            rng = np.random.default_rng(1000 * n_token + seed)
            logits = jnp.asarray(
                rng.normal(size=(SAMPLES, n_token, n_token, PAE_BINS)),
                dtype=jnp.float32,
            )
            has_frame = jnp.asarray(rng.random((SAMPLES, n_token)) > 0.25)
            token_mask = jnp.asarray(rng.random(n_token) > 0.15)
            asym_id = jnp.asarray(rng.integers(0, chain_count, size=n_token))
            legacy = legacy_run(logits, has_frame, token_mask, asym_id)
            compact = compact_run(logits, has_frame, token_mask, asym_id)

            for actual, expected in zip(compact, legacy, strict=True):
                _assert_nonfinite_and_finite_delta(actual, expected)


def test_bfloat16_metrics_stay_within_the_same_contract() -> None:
    rng = np.random.default_rng(20260828)
    n_token, n_chain = 33, 4
    logits = jnp.asarray(
        rng.normal(size=(SAMPLES, n_token, n_token, PAE_BINS)),
        dtype=jnp.bfloat16,
    )
    has_frame = jnp.asarray(rng.random((SAMPLES, n_token)) > 0.2)
    token_mask = jnp.asarray(rng.random(n_token) > 0.1)
    asym_id = jnp.asarray(np.arange(n_token) % n_chain)
    legacy = jax.jit(
        lambda pair, frames: _legacy_metrics(
            pair, frames, token_mask, asym_id, n_chain=n_chain
        )
    )(logits, has_frame)
    compact = jax.jit(
        lambda pair, frames: _compact_mapped_metrics(
            pair, frames, token_mask, asym_id, n_chain=n_chain
        )
    )(logits, has_frame)
    for actual, expected in zip(compact, legacy, strict=True):
        _assert_nonfinite_and_finite_delta(actual, expected)


@pytest.mark.parametrize("n_token", [257, 259, 511])
def test_bfloat16_large_token_count_keeps_legacy_denominator(n_token) -> None:
    # ``compute_ptm`` casts the considered-token count to the logits dtype.
    # BF16 first rounds above 256, so this catches an easy-to-miss >1e-3 drift
    # if the compact reduction instead divides in the promoted FP32 dtype.
    logits = jnp.zeros((1, n_token, n_token, 2), dtype=jnp.bfloat16)
    has_frame = jnp.ones((1, n_token), dtype=bool)
    token_mask = jnp.ones(n_token, dtype=bool)
    asym_id = jnp.zeros(n_token, dtype=jnp.int32)
    # A narrow range keeps the expected TM weight close to one, making the
    # rounded-count denominator error exceed the public 1e-3 confidence bound.
    kwargs = {"bin_min": 0.0, "bin_max": 1.0, "no_bins": 2}

    legacy = compute_ptm(logits, has_frame, token_mask, **kwargs)
    compact, _, _ = inference._compact_confidence_metrics_from_pae(
        logits,
        has_frame,
        token_mask,
        asym_id,
        n_chain=1,
        **kwargs,
    )

    np.testing.assert_array_equal(np.asarray(compact), np.asarray(legacy))


def test_empty_masks_and_nonfinite_logits_keep_their_semantics() -> None:
    n_token = 5
    logits = np.zeros((SAMPLES, n_token, n_token, PAE_BINS), dtype=np.float32)
    logits[0, 0, 0, 0] = np.nan
    # An active row with a masked column is still undefined historically:
    # compute_ptm multiplies the NaN softmax result by the pair mask, and
    # IEEE NaN * 0 remains NaN.
    logits[1, 1, 2, 3] = np.inf
    # A single -Inf among finite bins is a valid zero-probability category.
    logits[2, 3, 1, 7] = -np.inf
    # All -Inf bins are undefined, including at a masked column.
    logits[3, 0, 4, :] = -np.inf
    # Multiple +Inf values are equally undefined. On a masked row they are
    # suppressed by the final has-frame/token-row selection and must not poison
    # another row.
    logits[4, 2, 0, 9] = np.inf
    logits[4, 2, 0, 10] = np.inf
    logits = jnp.asarray(logits)
    frames = jnp.asarray(
        [[False] * n_token, [True] * n_token, [True, False, True, False, True]]
        + [[True] * n_token] * 2
    )
    masks = (
        jnp.zeros(n_token, dtype=bool),
        jnp.asarray([True, True, False, True, False]),
    )
    asym_id = jnp.asarray([0, 0, 1, 2, 2])
    for token_mask in masks:
        legacy = jax.jit(
            lambda pair, valid: _legacy_metrics(
                pair, valid, token_mask, asym_id, n_chain=3
            )
        )(logits, frames)
        compact = jax.jit(
            lambda pair, valid: _compact_mapped_metrics(
                pair, valid, token_mask, asym_id, n_chain=3
            )
        )(logits, frames)
        for actual, expected in zip(compact, legacy, strict=True):
            _assert_nonfinite_and_finite_delta(actual, expected)

        if bool(jnp.any(token_mask)):
            legacy_ptm = np.asarray(legacy[0])
            np.testing.assert_array_equal(
                np.isnan(legacy_ptm),
                np.asarray([False, True, False, True, False]),
            )

    empty = _compact_mapped_metrics(
        logits,
        frames,
        masks[0],
        asym_id,
        n_chain=3,
    )
    for value in empty:
        np.testing.assert_array_equal(np.asarray(value), 0.0)
        assert not np.signbit(np.asarray(value)).any()


def test_scalar_poison_matches_the_legacy_active_row_truth_table(monkeypatch) -> None:
    all_valid = [True, True, True, True]
    all_framed = [[True, True, True, True]]
    no_chain_nan = np.zeros((3, 3), dtype=bool)
    row_chain_0_nan = np.asarray(
        [[False, True, True], [True, False, False], [True, False, False]]
    )
    row_chain_2_nan = np.asarray(
        [[False, False, True], [False, False, True], [True, True, False]]
    )
    only_0_1_nan = np.asarray(
        [[False, True, False], [True, False, False], [False, False, False]]
    )
    cases = (
        (
            "active-cross",
            "posinf",
            0,
            2,
            all_valid,
            all_framed,
            3,
            True,
            row_chain_0_nan,
        ),
        (
            "active-same-chain",
            "nan",
            0,
            1,
            all_valid,
            all_framed,
            3,
            True,
            row_chain_0_nan,
        ),
        (
            "active-all-neginf",
            "all_neginf",
            3,
            0,
            all_valid,
            all_framed,
            3,
            True,
            row_chain_2_nan,
        ),
        (
            "masked-column",
            "multi_posinf",
            0,
            3,
            [True, True, True, False],
            all_framed,
            3,
            True,
            only_0_1_nan,
        ),
        (
            "masked-row",
            "nan",
            3,
            0,
            [True, True, True, False],
            all_framed,
            3,
            False,
            no_chain_nan,
        ),
        (
            "unframed-row",
            "posinf",
            0,
            2,
            all_valid,
            [[False, True, True, True]],
            3,
            False,
            no_chain_nan,
        ),
        ("monomer-iptm-zero", "posinf", 0, 1, all_valid, all_framed, 1, True, None),
    )
    prepared = []
    for case in cases:
        name, kind, row, column, mask, frames, n_chain, *_ = case
        values = np.zeros((1, 4, 4, PAE_BINS), dtype=np.float32)
        if kind == "nan":
            values[0, row, column, 0] = np.nan
        elif kind == "posinf":
            values[0, row, column, 0] = np.inf
        elif kind == "multi_posinf":
            values[0, row, column, :2] = np.inf
        else:
            assert kind == "all_neginf"
            values[0, row, column, :] = -np.inf
        logits = jnp.asarray(values)
        token_mask = jnp.asarray(mask)
        has_frame = jnp.asarray(frames)
        asym_id = (
            jnp.zeros(4, dtype=jnp.int32) if n_chain == 1 else jnp.asarray([0, 0, 1, 2])
        )
        legacy = _legacy_metrics(
            logits, has_frame, token_mask, asym_id, n_chain=n_chain
        )
        legacy_classes = tuple(
            None if value is None else np.isnan(np.asarray(value)) for value in legacy
        )
        prepared.append((case, logits, has_frame, token_mask, asym_id, legacy_classes))

    # Simulate the CUDA lowering that returned finite probabilities for an
    # undefined row. The scalar poison must recover the historical class without
    # relying on NaN surviving the pair-mask multiply.
    monkeypatch.setattr(jax.nn, "softmax", lambda value, axis=-1: jnp.zeros_like(value))
    for case, logits, has_frame, token_mask, asym_id, legacy_classes in prepared:
        name, _, _, _, _, _, n_chain, expected_nan, expected_chain = case
        compact = inference._compact_confidence_metrics_from_pae(
            logits,
            has_frame,
            token_mask,
            asym_id,
            n_chain=n_chain,
            bin_min=0.0,
            bin_max=32.0,
            no_bins=PAE_BINS,
        )
        compact_classes = tuple(
            None if value is None else np.isnan(np.asarray(value)) for value in compact
        )
        for actual, expected in zip(compact_classes, legacy_classes, strict=True):
            if actual is None or expected is None:
                assert actual is expected is None, name
            else:
                np.testing.assert_array_equal(actual, expected, err_msg=name)
        np.testing.assert_array_equal(compact_classes[0], [expected_nan], err_msg=name)
        np.testing.assert_array_equal(
            compact_classes[1], [False if n_chain == 1 else expected_nan], err_msg=name
        )
        if expected_chain is not None:
            np.testing.assert_array_equal(
                compact_classes[2][0], expected_chain, err_msg=name
            )


def _config(**changes):
    config = inference.released_config(
        n_token=8,
        n_atom=8,
        num_samples=3,
        num_steps=1,
        per_sample_token_cutoff=7,
    )
    return config._replace(**changes)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"returned_pair_logits": ()}, True),
        ({"returned_pair_logits": ("pde_logits",)}, True),
        ({"returned_pair_logits": ("pae_logits",)}, False),
        ({"returned_pair_logits": (), "per_sample_token_cutoff": None}, False),
        ({"returned_pair_logits": (), "num_samples": 1}, False),
    ],
)
def test_sink_requires_serial_confidence_and_unreturned_pae(changes, expected) -> None:
    assert inference._sink_pae_metrics(_config(**changes)) is expected


def test_high_sample_schedule_sinks_only_unreturned_pae() -> None:
    base = inference.released_config(
        n_token=8,
        n_atom=8,
        num_samples=6,
        num_steps=1,
        per_sample_token_cutoff=750,
    )

    assert inference._per_sample_confidence(base)
    assert not inference._sink_pae_metrics(base)
    assert inference._sink_pae_metrics(base._replace(returned_pair_logits=()))
    assert not inference._per_sample_confidence(
        base._replace(per_sample_token_cutoff=None)
    )


def _project(pair, weight):
    return jnp.einsum("...c,bc->...b", pair, weight)


def _lowered_pair(n_token: int, *, n_chain: int = 2):
    channels = 16
    pair = jax.ShapeDtypeStruct(
        (SAMPLES, n_token, n_token, channels), jnp.float32
    )
    frames = jax.ShapeDtypeStruct((SAMPLES, n_token), jnp.bool_)
    weight = jax.ShapeDtypeStruct((PAE_BINS, channels), jnp.float32)
    token_mask = jnp.ones(n_token, dtype=bool)
    asym_id = jnp.asarray(np.arange(n_token) % n_chain)

    def legacy(z, valid, projection):
        logits = jax.lax.map(lambda one: _project(one, projection), z)
        return _legacy_metrics(
            logits, valid, token_mask, asym_id, n_chain=n_chain
        )

    def compact(z, valid, projection):
        return jax.lax.map(
            lambda one: jax.tree.map(
                lambda leaf: leaf[0],
                inference._compact_confidence_metrics_from_pae(
                    _project(one[0], projection)[None],
                    one[1][None],
                    token_mask,
                    asym_id,
                    n_chain=n_chain,
                    bin_min=0.0,
                    bin_max=32.0,
                    no_bins=PAE_BINS,
                ),
            ),
            (z, valid),
        )

    return (
        jax.jit(legacy).lower(pair, frames, weight),
        jax.jit(compact).lower(pair, frames, weight),
    )


def test_compact_stablehlo_has_no_all_sample_pae_stack() -> None:
    n_token = 32
    legacy, compact = _lowered_pair(n_token)
    pae_stack = f"tensor<{SAMPLES}x{n_token}x{n_token}x{PAE_BINS}xf32>"
    assert pae_stack in legacy.as_text()
    assert pae_stack not in compact.as_text()


@pytest.mark.parametrize("n_chain", [2, 4])
def test_compact_metrics_reduce_compiled_cpu_temporary_memory(n_chain) -> None:
    # Four chains exercise six independently weighted pair reductions, not just
    # the single off-diagonal pair present in a two-chain complex.
    legacy, compact = _lowered_pair(128, n_chain=n_chain)
    legacy_temp = legacy.compile().memory_analysis().temp_size_in_bytes
    compact_temp = compact.compile().memory_analysis().temp_size_in_bytes
    assert legacy_temp > 4 * compact_temp, (legacy_temp, compact_temp)


def test_compact_metrics_keep_both_cpu_context_parallel_layouts() -> None:
    probe = textwrap.dedent(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np

        from foldjax.models._cp import context_parallel, shard_pair_rows
        from foldjax.models.openfold3.inference import (
            _compact_confidence_metrics_from_pae,
        )

        assert jax.device_count() == 4, jax.devices()
        rng = np.random.default_rng(20260828)
        n_token, n_chain, n_bins = 16, 4, 64
        logits = jnp.asarray(
            rng.normal(size=(1, n_token, n_token, n_bins)), dtype=jnp.float32
        )
        frames = jnp.asarray(rng.random((1, n_token)) > 0.2)
        token_mask = jnp.asarray(rng.random(n_token) > 0.1)
        asym_id = jnp.asarray(np.arange(n_token) % n_chain)

        def run(pair, valid):
            return _compact_confidence_metrics_from_pae(
                pair,
                valid,
                token_mask,
                asym_id,
                n_chain=n_chain,
                bin_min=0.0,
                bin_max=32.0,
                no_bins=n_bins,
            )

        serial = jax.jit(run)(logits, frames)
        for layout in ("1d", "2d"):
            jax.clear_caches()
            with context_parallel(4, layout=layout):
                sharded = jax.jit(
                    lambda pair, valid: run(shard_pair_rows(pair), valid)
                )(logits, frames)
            for actual, expected in zip(sharded, serial, strict=True):
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-3)
        """
    )
    environment = {
        **inherited_environment(),
        "FOLDJAX_SKIP_MESH_CHECK": "1",
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
    }
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _install_tiny_predict(monkeypatch) -> tuple[dict[str, jax.Array], object]:
    n_token = n_atom = 4

    def fake_trunk(batch, params, **kwargs):
        del batch, params, kwargs
        values = jnp.arange(n_token * 4, dtype=jnp.float32).reshape(1, n_token, 4)
        pair = jnp.zeros((1, n_token, n_token, 4), dtype=jnp.float32)
        return values / 17.0, values / 13.0, pair

    def fake_sample(key, schedule, shape, denoise_fn, **kwargs):
        del key, schedule, denoise_fn, kwargs
        samples, atoms, _ = shape
        sample = jnp.arange(samples, dtype=jnp.float32)[:, None]
        atom = jnp.arange(atoms, dtype=jnp.float32)[None, :]
        x = sample + atom / 7.0
        return jnp.stack((x, x / 3.0, -x / 5.0), axis=-1)

    def fake_representatives(batch, coordinates, atom_mask, table):
        del batch, atom_mask, table
        return coordinates[:, :n_token], jnp.ones((coordinates.shape[0], n_token))

    def fake_pairformer(si_input, si, zij, x_pred, params, **kwargs):
        del si_input, zij, params, kwargs
        signal = x_pred[..., 0]
        single = si + signal[..., None] / 19.0
        delta = signal[..., :, None] - signal[..., None, :]
        pair = jnp.stack((delta, delta**2, -delta, delta / 11.0), axis=-1)
        return single, pair

    def fake_pae(pair, params):
        del params
        bins = jnp.arange(PAE_BINS, dtype=pair.dtype)
        return pair[..., :1] + bins / 23.0

    def fake_pde(pair, params):
        del params
        bins = jnp.arange(PAE_BINS, dtype=pair.dtype)
        return pair[..., 1:2] + bins / 29.0

    def fake_atom_head(single, params, mask, *, c_out, n_atom, **kwargs):
        del params, mask, kwargs
        bins = jnp.arange(c_out, dtype=single.dtype)
        return single[..., :n_atom, :1] + bins / max(c_out, 1)

    frame_mask = jnp.asarray(
        [[True, True, False, True], [True, False, True, True], [False] * n_token]
    )

    monkeypatch.setattr(inference, "trunk", fake_trunk)
    monkeypatch.setattr(inference, "noise_schedule", lambda *args, **kwargs: None)
    monkeypatch.setattr(inference, "sample_diffusion", fake_sample)
    monkeypatch.setattr(
        inference, "pair_conditioning", lambda batch, z, params, **kwargs: z
    )
    monkeypatch.setattr(
        inference,
        "single_conditioning",
        lambda *args, **kwargs: jnp.zeros((1, n_token, 4)),
    )
    monkeypatch.setattr(inference, "token_representative_atoms", fake_representatives)
    monkeypatch.setattr(
        inference,
        "distogram_head",
        lambda z, params: jnp.broadcast_to(
            jnp.arange(PAE_BINS, dtype=z.dtype),
            (1, n_token, n_token, PAE_BINS),
        ),
    )
    monkeypatch.setattr(inference, "pairformer_embedding", fake_pairformer)
    monkeypatch.setattr(inference, "predicted_aligned_error_head", fake_pae)
    monkeypatch.setattr(inference, "predicted_distance_error_head", fake_pde)
    monkeypatch.setattr(inference, "atom_logit_head", fake_atom_head)
    monkeypatch.setattr(
        inference,
        "token_frame_atoms",
        lambda *args, **kwargs: ((None, None, None), frame_mask),
    )

    batch = {
        "token_mask": jnp.asarray([[1, 1, 0, 1]], dtype=jnp.float32),
        "asym_id": jnp.asarray([[0, 0, 1, 2]], dtype=jnp.int32),
        "atom_mask": jnp.ones((1, n_atom), dtype=jnp.float32),
        "max_atom_per_token_mask": jnp.arange(n_atom)[None],
    }
    params = SimpleNamespace(
        trunk=None,
        diffusion_conditioning=None,
        denoiser=None,
        pairformer_embedding=None,
        plddt_head=None,
        pae_head=None,
        pde_head=None,
        distogram_head=None,
        experimentally_resolved_head=object(),
    )
    return batch, params


def _tiny_inference_config(
    *,
    per_sample_token_cutoff: int | None,
    returned_pair_logits: tuple[str, ...] = (
        "pae_logits",
        "pde_logits",
        "distogram_logits",
    ),
) -> inference.InferenceConfig:
    return inference.InferenceConfig(
        n_token=4,
        n_atom=4,
        n_query=1,
        n_key=1,
        atom_heads=1,
        token_heads=1,
        no_heads_msa=1,
        no_heads_pair=1,
        no_heads_pair_bias=1,
        max_relative_idx=1,
        max_relative_chain=1,
        num_recycles=1,
        num_samples=3,
        max_atoms_per_token=1,
        plddt_bins=50,
        pae_bins=PAE_BINS,
        pae_bin_max=32.0,
        num_steps=1,
        per_sample_token_cutoff=per_sample_token_cutoff,
        returned_pair_logits=returned_pair_logits,
        has_atomized_tokens=False,
    )


@pytest.mark.parametrize(
    "per_sample_token_cutoff", [3, None], ids=["serial-sink", "batched-control"]
)
def test_full_predict_drops_logits_and_keeps_shared_outputs(
    monkeypatch, per_sample_token_cutoff
) -> None:
    batch, params = _install_tiny_predict(monkeypatch)
    base = _tiny_inference_config(
        per_sample_token_cutoff=per_sample_token_cutoff,
    )
    kept = inference.predict(
        jax.random.key(0), batch, params, base, None, n_chain=3
    )
    dropped = inference.predict(
        jax.random.key(0),
        batch,
        params,
        base._replace(returned_pair_logits=("distogram_logits",)),
        None,
        n_chain=3,
    )
    pde_only = inference.predict(
        jax.random.key(0),
        batch,
        params,
        base._replace(returned_pair_logits=("pde_logits",)),
        None,
        n_chain=3,
    )
    pae_only = inference.predict(
        jax.random.key(0),
        batch,
        params,
        base._replace(returned_pair_logits=("pae_logits",)),
        None,
        n_chain=3,
    )

    assert dropped.pae_logits is dropped.pde_logits is None
    assert kept.pae_logits.shape == (3, 4, 4, PAE_BINS)
    assert kept.pde_logits.shape == (3, 4, 4, PAE_BINS)
    assert pde_only.pae_logits is None
    assert pae_only.pde_logits is None
    np.testing.assert_array_equal(pde_only.pde_logits, kept.pde_logits)
    np.testing.assert_array_equal(pae_only.pae_logits, kept.pae_logits)
    for name in (
        "coordinates",
        "plddt",
        "distogram_logits",
        "experimentally_resolved_logits",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(dropped, name)), np.asarray(getattr(kept, name))
        )
    for candidate in (dropped, pde_only, pae_only):
        for name in ("ptm", "iptm", "chain_pair_iptm"):
            _assert_nonfinite_and_finite_delta(
                getattr(candidate, name), getattr(kept, name)
            )

    zeros = jnp.zeros(base.num_samples)
    legacy_ranking = sample_ranking_score(kept.ptm, kept.iptm, zeros)
    compact_ranking = sample_ranking_score(dropped.ptm, dropped.iptm, zeros)
    np.testing.assert_allclose(
        compact_ranking, legacy_ranking, rtol=0.0, atol=MAX_CONFIDENCE_DELTA
    )
    legacy_host = np.asarray(legacy_ranking)
    compact_host = np.asarray(compact_ranking)
    for i in range(base.num_samples):
        for j in range(i + 1, base.num_samples):
            if abs(legacy_host[i] - legacy_host[j]) > 2 * MAX_CONFIDENCE_DELTA:
                assert np.sign(legacy_host[i] - legacy_host[j]) == np.sign(
                    compact_host[i] - compact_host[j]
                )


def test_serial_schedule_keeps_the_batched_output_contract(monkeypatch) -> None:
    """Scheduling one confidence sample at a time changes no retained field."""
    batch, params = _install_tiny_predict(monkeypatch)
    batched_config = _tiny_inference_config(
        per_sample_token_cutoff=None,
        returned_pair_logits=(),
    )
    serial_config = batched_config._replace(per_sample_token_cutoff=0)

    batched = inference.predict(
        jax.random.key(17), batch, params, batched_config, None, n_chain=3
    )
    serial = inference.predict(
        jax.random.key(17), batch, params, serial_config, None, n_chain=3
    )

    for name in (
        "coordinates",
        "plddt",
        "experimentally_resolved_logits",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(serial, name)),
            np.asarray(getattr(batched, name)),
            err_msg=name,
        )
    for name in ("ptm", "iptm", "chain_pair_iptm"):
        _assert_nonfinite_and_finite_delta(
            getattr(serial, name), getattr(batched, name)
        )
    for name in ("pae_logits", "pde_logits", "distogram_logits"):
        assert getattr(serial, name) is getattr(batched, name) is None


def test_high_sample_trunk_stop_returns_before_confidence(monkeypatch) -> None:
    batch, params = _install_tiny_predict(monkeypatch)
    config = _tiny_inference_config(
        per_sample_token_cutoff=750,
        returned_pair_logits=(),
    )._replace(num_samples=10, stop_after_trunk=True)

    def unexpected_confidence(_config):
        raise AssertionError("trunk-only prediction reached confidence scheduling")

    monkeypatch.setattr(inference, "_per_sample_confidence", unexpected_confidence)
    result = inference.predict(jax.random.key(23), batch, params, config, None)

    for name in (
        "coordinates",
        "plddt",
        "ptm",
        "iptm",
        "chain_pair_iptm",
        "pae_logits",
        "pde_logits",
        "distogram_logits",
        "experimentally_resolved_logits",
    ):
        assert getattr(result, name) is None


@pytest.mark.torch_parity
def test_composed_inference_sink_matches_returned_logits_control(
    openfold3_source, randomized
) -> None:
    from tests.models.openfold3.test_inference_end_to_end import (
        _batch,
        _params,
        _representative_atoms,
        _torch,
    )
    from tests.models.openfold3.test_inference_end_to_end import (
        _config as composed_config,
    )

    del openfold3_source
    torch = _torch()
    batch = _batch(torch)
    params = _params(torch, randomized)
    table = _representative_atoms()
    base = composed_config()
    base = base._replace(per_sample_token_cutoff=base.n_token - 1)
    kept = inference.predict(
        jax.random.key(11), batch, params, base, table, n_chain=2
    )
    compact = inference.predict(
        jax.random.key(11),
        batch,
        params,
        base._replace(returned_pair_logits=("distogram_logits",)),
        table,
        n_chain=2,
    )

    assert compact.pae_logits is compact.pde_logits is None
    for name in (
        "coordinates",
        "plddt",
        "distogram_logits",
        "experimentally_resolved_logits",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(compact, name)), np.asarray(getattr(kept, name))
        )
    for name in ("ptm", "iptm", "chain_pair_iptm"):
        _assert_nonfinite_and_finite_delta(
            getattr(compact, name), getattr(kept, name)
        )


def test_ranking_contract_identifies_unprotected_near_ties() -> None:
    scores = np.asarray([0.5, 0.5 + 1.5 * MAX_CONFIDENCE_DELTA])
    assert abs(scores[1] - scores[0]) <= 2 * MAX_CONFIDENCE_DELTA
