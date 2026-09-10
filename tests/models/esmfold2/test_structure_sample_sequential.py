"""One diffusion sample at a time, without rewriting the random stream.

The batched sampler denoises `batch * num_samples` rollouts in one call, and
the denoiser is the only place in the structure head where that axis costs
more than atom width: the token transformer's attention logits are
`[rows, tokens, tokens, heads]` float32. Everything else in `sample` -- the
initial draw, the per-step augmentation, the churn normal, the Kabsch align --
is `[rows, atoms, 3]`.

So this option maps the denoiser call and leaves the rest of the sampler at the
batched width. That is what makes the equivalence argument short. Every draw
still happens once per step, at the full width, from the same key, in the same
order, so "the same key per sample" is a property of the code rather than a
claim each draw site has to be tested for. The tests below pin the two halves
of that: the denoiser really is entered one row at a time, and the rollout that
comes out is the batched one.

The denoiser here is a stand-in. The released one needs the checkpoint, and
what could go wrong in this change is the plumbing around it -- cache width,
mask width, the shape the mapped results are put back into -- not the
denoiser's own arithmetic, which is untouched. The stand-in is nonlinear and
row-local so that a row fed another row's coordinates, or its own twice, lands
somewhere else and the comparison fails.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2.models import diffusion as d
from foldjax.models.esmfold2.models import model as structure_model

N_ATOMS = 4
N_TOKENS = 2
C_TOKEN = 2


def _sampler_fixture(monkeypatch: pytest.MonkeyPatch, *, steps: int = 2):
    """A sampler whose denoiser records the width it was entered at.

    Two schedule steps by default, and that is a measurement rather than a
    default. The schedule spans sigma 256 down to 0.0064, and the Euler step
    cancels a magnitude-250 intermediate down to a magnitude-2 coordinate; at
    two steps the two paths agree to 3e-8, and at three -- where the clip
    leaves an intermediate sigma of 42.8 and the cancellation happens twice --
    they agree only to 5e-6, which is the sampler's own float32 floor and not
    a property of this option. `test_the_three_step_schedule_...` below pins
    that case against a measured floor instead of a chosen tolerance.
    """
    settings = d.DiffusionSettings(num_steps=steps, c_token=C_TOKEN, noise_scale=1.003)
    single_mask = jnp.asarray([[1.0, 1.0, 1.0, 0.0]], jnp.float32)
    widths: list[int] = []

    def denoise(x, sigma, s_inputs, cache, params, prefix="", **kwargs):
        del s_inputs, cache, params, prefix
        widths.append(int(x.shape[0]))
        # Bounded on purpose. An amplifying stand-in run over three steps
        # reaches 1e13, where the two paths still agree to 1e-5 *relatively*
        # and disagree by 1e8 absolutely -- which measures the fixture's
        # conditioning rather than this change. `tanh` keeps the row-local
        # coupling and drops the growth.
        energy = jnp.tanh(jnp.sum(x * x, axis=(1, 2), keepdims=True))
        denoised = jnp.tanh(x) * (1.0 + energy) / (1.0 + sigma[:, None, None])
        token_repr = jnp.broadcast_to(
            jnp.mean(x, axis=(1, 2))[:, None, None],
            (x.shape[0], N_TOKENS, C_TOKEN),
        )
        return denoised, token_repr

    monkeypatch.setattr(d, "diffusion_module", denoise)

    def run(key, *, samples, sequential, **draws):
        # `build_cache` spreads the atom-level entries to `batch * samples`
        # when it batches and leaves them at one sample when it does not; the
        # cache widths here are the two it produces.
        mask = single_mask if sequential else jnp.repeat(single_mask, samples, axis=0)
        cache = SimpleNamespace(atom_mask=mask, n_tokens=N_TOKENS)
        return d.sample(
            key,
            jnp.zeros((1, N_TOKENS, C_TOKEN), jnp.float32),
            cache,
            {},
            settings=settings,
            num_samples=samples,
            sample_sequential=sequential,
            **draws,
        )

    return run, widths


def _tape(rows: int, steps: int) -> dict[str, jnp.ndarray]:
    rng = np.random.default_rng(11)
    shapes = (
        ("diffusion_initial_normal", (rows, N_ATOMS, 3)),
        ("diffusion_rotation_quaternions", (steps, rows, 4)),
        ("diffusion_translations", (steps, rows, 1, 3)),
        ("diffusion_churn_normals", (steps, rows, N_ATOMS, 3)),
    )
    return {
        name: jnp.asarray(rng.normal(size=shape), jnp.float32)
        for name, shape in shapes
    }


def test_the_option_is_off_by_default() -> None:
    """Sequencing reorders execution, and the released numbers were batched.

    Unlike `confidence_sample_sequential`, this one has no released command it
    rescues: the schedules in this repository all complete batched. It is a
    capacity control for callers who raise the sample count, so it stays off
    until a caller asks and the batched program remains the shipped one.
    """
    assert structure_model.ModelSettings().structure_sample_sequential is False


def test_the_batched_path_enters_the_denoiser_once_at_full_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, widths = _sampler_fixture(monkeypatch)
    run(jax.random.key(0), samples=3, sequential=False)
    # Traced once inside the schedule scan, at every rollout at once.
    assert widths and set(widths) == {3}


def test_the_sequential_path_enters_the_denoiser_one_row_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tripwire: without it the comparison below could measure dead code.

    A branch that silently kept batching would agree with the batched path
    perfectly, so equality on its own certifies nothing. This is the assertion
    that the branch fires.
    """
    run, widths = _sampler_fixture(monkeypatch)
    run(jax.random.key(0), samples=3, sequential=True)
    assert widths and set(widths) == {1}


def test_sequencing_reproduces_the_batched_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same key, same coordinates, same token representation.

    Not asserted bitwise: the mapped denoiser runs on a one-row array where
    the batched one runs on three, and a per-row reduction is free to pick a
    different order at a different shape. The tolerance is far below what any
    row-mixing error would produce.
    """
    run, _ = _sampler_fixture(monkeypatch)
    batched = run(jax.random.key(7), samples=3, sequential=False)
    sequential = run(jax.random.key(7), samples=3, sequential=True)
    for one, other in zip(batched, sequential, strict=True):
        assert one.shape == other.shape
        np.testing.assert_allclose(one, other, rtol=1e-6, atol=1e-6)


def test_sequencing_is_not_a_reordering_of_the_rollouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rows have to come back where they went in, and they are distinct.

    Guards the comparison above from passing on rollouts that happen to agree:
    if the three rows were identical, a permutation would be invisible.
    """
    run, _ = _sampler_fixture(monkeypatch)
    coords, _ = run(jax.random.key(7), samples=3, sequential=True)
    for row in range(1, coords.shape[0]):
        assert not np.allclose(coords[0], coords[row], rtol=1e-3, atol=1e-3)


def test_the_native_tape_still_drives_both_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replay tape is consumed at the batched width whichever path runs.

    Sequencing does not touch the draw sites, so the tape needs no per-sample
    slicing and its shape contract is unchanged -- which is the point of
    mapping the denoiser rather than the sampler.
    """
    run, _ = _sampler_fixture(monkeypatch)
    tape = _tape(rows=3, steps=2)
    batched = run(jax.random.key(0), samples=3, sequential=False, **tape)
    # A different key, because a tape is supposed to make the key irrelevant.
    sequential = run(jax.random.key(41), samples=3, sequential=True, **tape)
    for one, other in zip(batched, sequential, strict=True):
        np.testing.assert_allclose(one, other, rtol=1e-6, atol=1e-6)


def test_the_three_step_schedule_agrees_to_the_samplers_own_float32_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Where the rollout amplifies, the comparison has to be against a floor.

    A third step leaves an intermediate sigma of 42.8 in the clipped schedule
    and cancels the magnitude-250 opening intermediate twice, so the float32
    error it carries arrives on a magnitude-2 coordinate near 1e-6 whatever
    the execution order. Measured here rather than written down: a one-ULP
    change to a single entry of the initial draw moves the *batched* result on
    its own, and the gap between the paths is held to that scale. The rows
    differ from each other by order one, so a plumbing error sits four orders
    of magnitude above both numbers and neither can hide it.
    """
    run, _ = _sampler_fixture(monkeypatch, steps=3)
    tape = _tape(rows=3, steps=3)
    batched, _ = run(jax.random.key(0), samples=3, sequential=False, **tape)
    sequential, _ = run(jax.random.key(0), samples=3, sequential=True, **tape)

    nudged = np.asarray(tape["diffusion_initial_normal"]).copy()
    nudged[0, 0, 0] = np.nextafter(nudged[0, 0, 0], np.float32(np.inf))
    moved, _ = run(
        jax.random.key(0),
        samples=3,
        sequential=False,
        **dict(tape, diffusion_initial_normal=jnp.asarray(nudged)),
    )

    floor = float(np.max(np.abs(np.asarray(batched) - np.asarray(moved))))
    gap = float(np.max(np.abs(np.asarray(batched) - np.asarray(sequential))))
    assert gap <= max(10.0 * floor, 1e-5), (gap, floor)


def test_the_sequential_sampler_compiles(monkeypatch: pytest.MonkeyPatch) -> None:
    """One traced program: the row loop nests inside the schedule loop."""
    run, widths = _sampler_fixture(monkeypatch)
    compiled = jax.jit(
        lambda key: run(key, samples=3, sequential=True),
    )
    eager = run(jax.random.key(3), samples=3, sequential=True)
    traced = compiled(jax.random.key(3))
    assert set(widths) == {1}
    for one, other in zip(eager, traced, strict=True):
        np.testing.assert_allclose(one, other, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# What `predict` asks the sampler for
# ---------------------------------------------------------------------------


def _cheap_features() -> dict[str, jax.Array]:
    token = jnp.arange(N_TOKENS, dtype=jnp.int32)[None]
    atom = jnp.arange(N_TOKENS, dtype=jnp.int32)[None]
    return {
        "token_index": token,
        "residue_index": token,
        "asym_id": jnp.zeros_like(token),
        "sym_id": jnp.zeros_like(token),
        "entity_id": jnp.ones_like(token),
        "mol_type": jnp.zeros_like(token),
        "res_type": token,
        "token_bonds": jnp.zeros((1, N_TOKENS, N_TOKENS, 1), dtype=jnp.float32),
        "token_attention_mask": jnp.ones((1, N_TOKENS), dtype=bool),
        "ref_pos": jnp.zeros((1, N_TOKENS, 3), dtype=jnp.float32),
        "ref_element": jnp.ones((1, N_TOKENS), dtype=jnp.int32),
        "ref_charge": jnp.zeros((1, N_TOKENS), dtype=jnp.float32),
        "ref_atom_name_chars": jnp.zeros((1, N_TOKENS, 4), dtype=jnp.int32),
        "ref_space_uid": atom,
        "atom_attention_mask": jnp.ones((1, N_TOKENS), dtype=bool),
        "atom_to_token": atom,
        "distogram_atom_idx": atom,
    }


def _predict_fixture(
    monkeypatch: pytest.MonkeyPatch, *, samples: int, sequential: bool, confidence: bool
):
    """Run `predict` with the trunk stubbed out, recording the sampler call."""
    settings = dataclasses.replace(
        structure_model.ModelSettings(),
        d_pair=2,
        d_inputs=2,
        trunk_n_layers=0,
        lm_encoder_n_layers=None,
        coda_n_layers=0,
        confidence_n_layers=0,
        msa_n_layers=None,
        num_recycles=0,
        num_samples=samples,
        confidence_sample_sequential=confidence,
        structure_sample_sequential=sequential,
        trunk_dtype="float32",
    )
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        structure_model,
        "one_hot_atom_features",
        lambda *args, **kwargs: (
            jnp.zeros((1, N_TOKENS, 128), dtype=jnp.float32),
            jnp.zeros((1, N_TOKENS, 4, 64), dtype=jnp.float32),
        ),
    )
    monkeypatch.setattr(
        structure_model,
        "inputs_embedding",
        lambda *args, **kwargs: jnp.asarray(
            [[[1.0, -2.0], [3.0, -4.0]]], dtype=jnp.float32
        ),
    )
    monkeypatch.setattr(
        structure_model,
        "relative_position_encoding",
        lambda *args, **kwargs: jnp.zeros(
            (1, N_TOKENS, N_TOKENS, 2), dtype=jnp.float32
        ),
    )
    monkeypatch.setattr(
        structure_model,
        "_token_bonds_encoding",
        lambda *args, **kwargs: jnp.zeros(
            (1, N_TOKENS, N_TOKENS, 2), dtype=jnp.float32
        ),
    )
    monkeypatch.setattr(
        structure_model, "run_loops", lambda key, z, z_init, *a, **k: z_init
    )
    monkeypatch.setattr(
        structure_model, "folding_trunk", lambda value, *a, **k: value
    )
    monkeypatch.setattr(
        structure_model,
        "linear",
        lambda value, params, prefix: (
            jnp.sum(value, axis=-1, keepdims=True)
            if prefix == "distogram_head"
            else value
        ),
    )

    def fake_build_cache(*args, **kwargs):
        seen["cache_samples"] = kwargs["num_samples"]
        return {"pair": args[7]}

    def fake_sample(key, single, cache, *args, **kwargs):
        del key, single, args
        seen["sample_samples"] = kwargs["num_samples"]
        seen["sample_sequential"] = kwargs["sample_sequential"]
        signal = jnp.sum(cache["pair"])
        rows = jnp.arange(samples, dtype=jnp.float32)[:, None, None]
        return jnp.broadcast_to(signal, (samples, N_TOKENS, 3)) + rows, None

    monkeypatch.setattr(structure_model.diffusion, "build_cache", fake_build_cache)
    monkeypatch.setattr(structure_model.diffusion, "sample", fake_sample)

    def fake_confidence(single, pair, coords, *args, **kwargs):
        del single, args, kwargs
        signal = jnp.sum(pair) + jnp.sum(coords, axis=(1, 2))
        return {
            "plddt": jnp.broadcast_to(signal[:, None], (coords.shape[0], N_TOKENS)),
            "plddt_per_atom": jnp.broadcast_to(
                signal[:, None], (coords.shape[0], N_TOKENS)
            ),
            "plddt_ca": jnp.broadcast_to(
                signal[:, None], (coords.shape[0], N_TOKENS)
            ),
            "complex_plddt": signal,
            "ptm": signal,
            "pair_chains_iptm": jnp.broadcast_to(
                signal[:, None, None], (coords.shape[0], 1, 1)
            ),
        }

    monkeypatch.setattr(structure_model, "confidence_head", fake_confidence)

    def call(features=None, *, pair_batch=1):
        return structure_model.predict(
            jax.random.key(0),
            _cheap_features() if features is None else features,
            {"token_bonds.weight": jnp.ones((2, 1), dtype=jnp.float32)},
            settings=settings,
            initial_pair_state=jnp.zeros(
                (pair_batch, N_TOKENS, N_TOKENS, 2), dtype=jnp.float32
            ),
            n_chains=1,
        )

    return call, seen


@pytest.mark.parametrize("confidence", [False, True])
def test_predict_builds_a_one_sample_cache_and_asks_the_sampler_to_sequence(
    monkeypatch: pytest.MonkeyPatch, confidence: bool
) -> None:
    """The saving needs both halves: a narrow cache and a mapped denoiser.

    `build_cache` repeats the atom-level conditioning to `batch * num_samples`,
    so leaving it batched would keep a factor the mapped denoiser then cannot
    use. Parametrised over the confidence flag because the two options are
    independent and both orders have to run.
    """
    call, seen = _predict_fixture(
        monkeypatch, samples=3, sequential=True, confidence=confidence
    )
    call()
    assert seen["cache_samples"] == 1
    assert seen["sample_samples"] == 3
    assert seen["sample_sequential"] is True


@pytest.mark.parametrize("confidence", [False, True])
def test_predict_leaves_the_batched_call_alone_when_the_option_is_off(
    monkeypatch: pytest.MonkeyPatch, confidence: bool
) -> None:
    call, seen = _predict_fixture(
        monkeypatch, samples=3, sequential=False, confidence=confidence
    )
    call()
    assert seen["cache_samples"] == 3
    assert seen["sample_samples"] == 3
    assert seen["sample_sequential"] is False


@pytest.mark.parametrize("confidence", [False, True])
def test_the_two_paths_return_the_same_outputs_through_the_confidence_head(
    monkeypatch: pytest.MonkeyPatch, confidence: bool
) -> None:
    """Whatever the sampler returns has to reach the head the same way.

    The stand-in sampler produces the same coordinates on both paths, so any
    difference here is the model's own handling of the result -- which is where
    a wrong sample layout would show up.
    """
    off, _ = _predict_fixture(
        monkeypatch, samples=3, sequential=False, confidence=confidence
    )
    batched = off()
    on, _ = _predict_fixture(
        monkeypatch, samples=3, sequential=True, confidence=confidence
    )
    sequential = on()
    assert set(batched) == set(sequential)
    for name, value in batched.items():
        np.testing.assert_array_equal(np.asarray(value), np.asarray(sequential[name]))


def test_a_single_sample_run_never_reaches_the_sequential_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One sample is already one at a time; the batched program is the same one."""
    call, seen = _predict_fixture(
        monkeypatch, samples=1, sequential=True, confidence=True
    )
    call()
    assert seen["cache_samples"] == 1
    assert seen["sample_sequential"] is False


def test_a_batched_input_is_refused_rather_than_sliced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two inputs in one call would need the pair cache sliced per rollout.

    That slice is a copy of the largest tensor the model holds, taken inside
    the loop this option exists to shrink, so the combination is refused
    instead. Nothing in the port produces a batch above one today; the check
    is here so that a caller who builds one is told, rather than handed a
    silently wrong shape.
    """
    call, _ = _predict_fixture(
        monkeypatch, samples=3, sequential=True, confidence=True
    )
    doubled = {
        name: jnp.concatenate([value, value], axis=0)
        for name, value in _cheap_features().items()
    }
    with pytest.raises(ValueError, match="structure_sample_sequential"):
        call(doubled, pair_batch=2)
