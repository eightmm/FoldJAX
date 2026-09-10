"""Tape-pinned CPU parity for Protenix against the stored 1UBQ native capture.

Two tiers over one 76-token case, both driven by
``protenix-master-native-20260909-9yET4f/protein_1ubq`` (see the manifest for
the capture path):

* **Tier A** stops the port at the trunk boundary and compares its three
  representations against the ``trunk.npz`` the native run wrote.
* **Tier B** replays the whole stored sampler tape to coordinates and measures
  per-sample entity RMSD against the native ``prediction.npz``.

What these tests certify, and what they do not
----------------------------------------------
The input is the port's own **pre-featurized** ``foldjax-input.npz``, so a
featurizer regression is invisible here: what is under test is the input
embedder, the trunk, and the diffusion sampler. The MSA cycle tape is built
with ``bench.protenix_foldjax_capture.msa_cycle_index_tape`` -- the same
function the GPU replay harness calls -- which re-checks, row by row, that the
stored input reproduces exactly the rows the native run selected, so a stale
pairing of the two fixture files fails there rather than as a parity number.

Why the model API and not the harness
-------------------------------------
``bench.protenix_foldjax_capture.replay`` drives the CLI and refuses a tape
whose ``init_noise.shape[0]`` is not 5. Nothing is sliced here -- the full
n5/200-step schedule is replayed -- so that guard is satisfied in spirit; what
the CLI route additionally requires is the input *document* plus the 490 MB
chemistry assets to re-featurize it, which is precisely the layer these
fixtures replace. The tape injection is therefore re-implemented against
``protenix_predict_static``: the four sampler arrays and the MSA cycle tape go
in as wrapper arguments, exactly as the harness passes them
(``bench/protenix_foldjax_capture.py:377-387``), and the noise schedule is
pinned by patching the module global the harness patches.

The schedule has to be pinned rather than regenerated: on CPU
``inference_noise_schedule`` reproduces 186 of the capture's 201 entries
bitwise and the other 15 to one float32 ulp (1.1e-7 relative). The GPU capture
recorded a bitwise match, so regenerating here would quietly replace the tape
with a slightly different one.

Which kernels this runs
-----------------------
The triangle-attention backend is left unset, which is what the CLI does and
what the capture ran: ``_triangle_attention_backend()`` resolves it to
``cueq_jit`` and the pairformer keeps triangle multiplication and the pair
transition on the un-jitted path. It is recorded in the manifest tripwire, so
a host that resolves something else (``PROTENIX_TRIANGLE_BACKEND``) fails as a
stale fixture rather than quietly measuring another arm.

That does mean this CPU test needs cuEquivariance to import -- a declared
dependency of the package, and ``foldjax.models._cueq.load_cueq`` refuses to
fall back on purpose. Forcing ``xla_jit`` instead would remove that dependency
and cost accuracy: measured on this fixture, the ``xla_jit`` arm's entity RMSD
against the capture is 0.017/0.114/0.048/0.092/0.135 A where this one gives
0.015/0.113/0.015/0.030/0.051, because this one is the kernel family the
capture itself ran. (The trunk barely notices -- 0.01968 against 0.01963
relative RMS on the pair stream -- so the whole difference is what 200
diffusion steps do with it.)

Two CPU replays of this fixture agree bit for bit, in the same process and
across two processes pinned to disjoint core sets, so neither tolerance carries
a rerun margin.

The one deviation from the captured arm is that the confidence heads are off.
They read the trunk and the coordinates and nothing reads them back, so the
structure is unaffected; leaving them on costs wall time this subset does not
have.

Both tolerances are closed and were set from a CPU calibration run recorded in
``tests/parity/manifest/protenix.json``; see that file's ``set_from`` for the
number and ``notes`` for the detection floor.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_parity

PORT = "protenix"
CASE = "protein_1ubq"
CHECKPOINT = "protenix_base_default_v1.0.0.jax"

#: The released inference schedule the capture ran, reproduced exactly.
RECYCLES = 10
SAMPLING_STEPS = 200
SAMPLES = 5

#: Canonical trunk tap -> the name the native capture stored it under.
TRUNK_ARRAYS = {"single_inputs": "s_inputs", "single": "s", "pair": "z"}

#: What ``sampler-tape.npz`` must contain for the replay to be tape-pinned.
SAMPLER_TAPE_KEYS = frozenset(
    {"init_noise", "step_noises", "rotations", "translations", "noise_schedule"}
)


def _arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _array(path: Path, name: str) -> np.ndarray:
    """One member of an archive. ``prediction.npz`` holds 141 of them."""
    with np.load(path, allow_pickle=False) as archive:
        return archive[name]


def _nest(flat: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Undo the dotted flattening the capture wrote the feature dict with."""
    nested: dict[str, Any] = {}
    for key, value in flat.items():
        *parents, leaf = key.split(".")
        node = nested
        for parent in parents:
            node = node.setdefault(parent, {})
        node[leaf] = value
    return nested


def _checkpoint() -> Path:
    from foldjax.paths import weights_dir

    path = weights_dir("protenix") / CHECKPOINT
    if not path.is_file():
        pytest.fail(
            "CPU parity needs the released Protenix checkpoint\n"
            f"  looked for: {path}\n"
            "  fetch it with `foldjax weights fetch --model protenix`, or set\n"
            "  FOLDJAX_HOME to a store that already has it -- a git worktree\n"
            "  has no .foldjax/ of its own, so the main checkout's store has\n"
            "  to be named explicitly.",
            pytrace=False,
        )
    return path


@functools.lru_cache(maxsize=1)
def _params() -> Any:
    """The checkpoint under the CLI's own bf16 trunk cast.

    Through the CLI helper rather than a second copy of the cast: the capture
    was produced by that CLI, and two implementations of "which fields are
    bfloat16" would be free to drift apart without any test noticing.
    """
    from foldjax.models.protenix.cli.predict import _load_prepared_params

    return _load_prepared_params(_checkpoint(), "bf16")


def _require_cpu() -> None:
    """Every tolerance in the manifest was calibrated against CPU XLA."""
    import jax

    backend = jax.default_backend()
    if backend != "cpu":
        pytest.fail(
            f"this subset is calibrated on CPU XLA and is running on {backend!r}; "
            "set JAX_PLATFORMS=cpu",
            pytrace=False,
        )


def _observed_schema() -> dict[str, str]:
    """What this checkout can consume, for the manifest's tripwire.

    Read off the objects the fixtures are fed into, not spelled out here: a
    tripwire whose ``observed`` side is a literal certifies nothing.
    """
    from bench.protenix_foldjax_capture import _TAPE_ARGUMENTS
    from foldjax.models._representations import available
    from foldjax.models.protenix.models.predict import protenix_predict_static
    from foldjax.models.protenix.models.triangle.triangle import (
        _triangle_attention_backend,
    )
    from foldjax.models.protenix.models.trunk_blocks.msa import MSACycleIndexTape

    accepted = inspect.signature(protenix_predict_static).parameters
    return {
        "tape_arguments": ",".join(
            sorted(name for name in _TAPE_ARGUMENTS if name in accepted)
        ),
        "msa_cycle_index_tape_fields": ",".join(MSACycleIndexTape._fields),
        "trunk_representations": ",".join(available(PORT)),
        "triangle_attention_backend": _triangle_attention_backend(),
    }


def _msa_cycles(features: Mapping[str, Any], msa_tape: Mapping[str, np.ndarray]) -> Any:
    """The native run's per-cycle row choices, re-verified against the input."""
    from bench.protenix_foldjax_capture import msa_cycle_index_tape

    return msa_cycle_index_tape(features, msa_tape)


def _relative_rms(port: np.ndarray, native: np.ndarray) -> float:
    """RMS of the difference over the RMS of the reference.

    Not max-abs: both sides are bfloat16 activations whose largest entries are
    in the thousands, so a plain maximum reports the ulp at the largest value
    (64 on the pair stream) and says nothing about the other 739,327 entries.
    """
    left = np.asarray(port, dtype=np.float64)
    right = np.asarray(native, dtype=np.float64)
    return float(np.sqrt(np.mean((left - right) ** 2)) / np.sqrt(np.mean(right**2)))


def _run_trunk(features: Mapping[str, Any], cycles: Any) -> dict[str, np.ndarray]:
    """The port's three trunk representations, and nothing downstream of them.

    ``key=None``: with the MSA cycles supplied and MC dropout off (this capture
    recorded ``mc_dropout_applied: false``) nothing in the trunk draws, and the
    sampler -- the only consumer of the key -- is never reached.
    """
    import jax
    import jax.numpy as jnp

    from foldjax.models.protenix.models.predict import protenix_predict_static

    output = protenix_predict_static(
        _params(),
        dict(features),
        None,
        num_samples=1,
        recycling_steps=RECYCLES,
        trunk_dtype=jnp.bfloat16,
        cycle_msa_index_tape=jax.tree.map(jnp.asarray, cycles),
        stop_after_trunk=True,
        capture_names=tuple(TRUNK_ARRAYS),
        run_confidence=False,
    )
    output = jax.block_until_ready(output)
    return {
        name: np.asarray(jax.device_get(value), dtype=np.float64)
        for name, value in output.items()
    }


def _replay_coordinates(
    features: Mapping[str, Any],
    cycles: Any,
    tape: Mapping[str, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> np.ndarray:
    """The full stored tape, replayed to coordinates through the wrapper."""
    import jax
    import jax.numpy as jnp

    from foldjax.models.protenix.models import predict as prediction

    schedule = jnp.asarray(tape["noise_schedule"])
    calls: list[dict[str, Any]] = []

    def pinned(**kwargs: Any) -> jnp.ndarray:
        calls.append(kwargs)
        return schedule

    monkeypatch.setattr(prediction, "inference_noise_schedule", pinned)
    output = prediction.protenix_predict_static(
        _params(),
        dict(features),
        None,
        num_samples=SAMPLES,
        num_sampling_steps=SAMPLING_STEPS,
        recycling_steps=RECYCLES,
        trunk_dtype=jnp.bfloat16,
        init_noise=jnp.asarray(tape["init_noise"]),
        step_noises=jnp.asarray(tape["step_noises"]),
        rotations=jnp.asarray(tape["rotations"]),
        translations=jnp.asarray(tape["translations"]),
        cycle_msa_index_tape=jax.tree.map(jnp.asarray, cycles),
        # What the capture ran: the harness reads it out of the native
        # effective config (`enable_efficient_fusion`), the wrapper default is
        # off, and the two are different programs.
        use_diffusion_efficient_fusion=True,
        run_confidence=False,
        return_confidence_logits=False,
        return_trunk=False,
    )
    output = jax.block_until_ready(output)
    # A patch that never fired would leave the CPU-generated schedule in place
    # and this test would still pass, one ulp away from the tape it claims to
    # replay.
    assert len(calls) == 1, f"the schedule pin fired {len(calls)} times, expected 1"
    return np.asarray(jax.device_get(output["coordinate"]), dtype=np.float64)


def _entity_rmsd(
    native: np.ndarray, port: np.ndarray, chains: list[Any]
) -> dict[Any, list[float]]:
    """Per-chain, per-sample RMSD under one whole-system Kabsch per sample.

    The semantics the master panel published, through the function that
    produced it (``bench/protenix_master_diff.py`` calls the same helper).
    """
    from bench.entity_parity import compare_entity_parity

    keys = list(range(port.shape[1]))
    mask = np.ones(port.shape[:2], dtype=bool)
    report = compare_entity_parity(native, port, keys, keys, chains, chains, mask, mask)
    return report["entity_rmsd"]


def test_the_trunk_matches_the_native_capture(
    parity_case: Callable[..., Any],
) -> None:
    """Tier A: the port's trunk boundary against the native run's own."""
    case = parity_case(PORT, CASE, tier="A")
    case.assert_tripwire(_observed_schema())
    _require_cpu()
    assert case.entry.tolerance_metric == "relative_rms", (
        "this test measures relative RMS; the manifest records "
        f"{case.entry.tolerance_metric!r}"
    )

    features = _nest(_arrays(case.path("foldjax-input.npz")))
    cycles = _msa_cycles(features, _arrays(case.path("msa-tape.npz")))
    native = _arrays(case.path("trunk.npz"))
    port = _run_trunk(features, cycles)

    residuals = {}
    for name, stored in TRUNK_ARRAYS.items():
        assert port[name].shape == native[stored].shape, (
            f"{name}: port {port[name].shape}, capture {native[stored].shape}"
        )
        residuals[name] = _relative_rms(port[name], native[stored])
    worst = max(residuals.items(), key=lambda item: item[1])
    assert worst[1] <= case.entry.tolerance_value, (
        f"Protenix trunk drifted from {case.entry.capture_provenance}\n"
        f"  relative RMS: {residuals}\n"
        f"  worst {worst[0]} = {worst[1]:.6g} > {case.entry.tolerance_value} "
        f"({case.entry.tolerance_set_from})"
    )


def test_the_trunk_comparison_notices_a_reversed_cycle_order(
    parity_case: Callable[..., Any],
) -> None:
    """Tier A's negative control: the assertion above can actually fail.

    The ten recycles consume their MSA rows in reverse order -- cycle 9's
    selection at cycle 0 and back -- which is what a port that indexed the tape
    backwards would compute. Everything else (weights, features, the rows
    themselves, the recycle count) is what the passing test uses.

    This perturbation and not a smaller one, because a smaller one does not
    show. Measured on this fixture, in relative RMS of the pair stream against
    a 0.01963 calibration: swapping one row of one cycle moves the trunk 0.01178
    and leaves it 0.01972 from the capture -- indistinguishable, because the
    CPU-against-GPU gap is already 0.01963 and the perturbation is not
    orthogonal to it. Shifting every row index by one moves the trunk 0.03003
    (0.03311 from the capture) and reversing the cycle order 0.04637 (0.04824).
    So the floor this tier resolves is a trunk movement of roughly 0.03 relative
    RMS; anything finer has to be caught at Tier B or on GPU, and the manifest
    says so.
    """
    from foldjax.models.protenix.models.trunk_blocks.msa import MSACycleIndexTape

    case = parity_case(PORT, CASE, tier="A")
    _require_cpu()
    features = _nest(_arrays(case.path("foldjax-input.npz")))
    cycles = _msa_cycles(features, _arrays(case.path("msa-tape.npz")))
    native = _arrays(case.path("trunk.npz"))

    reversed_cycles = MSACycleIndexTape(
        row_indices=np.asarray(cycles.row_indices)[::-1].copy(),
        row_mask=np.asarray(cycles.row_mask)[::-1].copy(),
    )
    assert not np.array_equal(reversed_cycles.row_indices, cycles.row_indices), (
        "the ten cycles selected the same rows, so reversing them changes nothing"
    )
    port = _run_trunk(features, reversed_cycles)

    residuals = {
        name: _relative_rms(port[name], native[stored])
        for name, stored in TRUNK_ARRAYS.items()
    }
    assert max(residuals.values()) > case.entry.tolerance_value, (
        "a reversed MSA cycle order left the trunk inside the parity tolerance: "
        f"{residuals} vs {case.entry.tolerance_value}"
    )


def test_the_replayed_coordinates_match_the_native_capture(
    parity_case: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tier B: the whole stored tape, replayed to coordinates."""
    from foldjax.models.protenix.models.diffusion.diffusion import (
        inference_noise_schedule,
    )

    case = parity_case(PORT, CASE, tier="B")
    case.assert_tripwire(_observed_schema())
    _require_cpu()
    assert case.entry.tolerance_metric == "rmsd_angstrom", (
        "this test measures entity RMSD in angstrom; the manifest records "
        f"{case.entry.tolerance_metric!r}"
    )

    stored_input = _arrays(case.path("foldjax-input.npz"))
    features = _nest(stored_input)
    cycles = _msa_cycles(features, _arrays(case.path("msa-tape.npz")))
    tape = _arrays(case.path("sampler-tape.npz"))
    assert set(tape) == set(SAMPLER_TAPE_KEYS), f"unexpected tape schema: {set(tape)}"
    assert tape["init_noise"].shape[0] == SAMPLES
    assert tape["step_noises"].shape[:2] == (SAMPLING_STEPS, SAMPLES)
    assert tape["noise_schedule"].shape == (SAMPLING_STEPS + 1,)

    # Not bitwise on CPU (15 of 201 entries differ by one float32 ulp), which
    # is why the replay below runs on the stored schedule and not this one.
    generated = np.asarray(
        inference_noise_schedule(
            num_steps=SAMPLING_STEPS,
            s_max=160.0,
            s_min=4.0e-4,
            rho=7.0,
            sigma_data=16.0,
        )
    )
    assert np.allclose(generated, tape["noise_schedule"], rtol=1e-6, atol=0.0), (
        "this checkout's noise schedule is not the one the capture ran"
    )

    native = _array(case.path("prediction.npz"), "coordinate")
    port = _replay_coordinates(features, cycles, tape, monkeypatch)
    assert port.shape == native.shape, f"port {port.shape}, capture {native.shape}"

    chains = stored_input["output_atom_chain_id"].tolist()
    per_chain = _entity_rmsd(np.asarray(native, dtype=np.float64), port, chains)
    excluded = set(case.entry.excluded_samples)
    failures = [
        f"chain {chain} sample {index}: {value:.4g} A"
        for chain, values in sorted(per_chain.items())
        for index, value in enumerate(values)
        if index not in excluded and value > case.entry.tolerance_value
    ]
    assert not failures, (
        f"Protenix replay drifted from {case.entry.capture_provenance}\n"
        f"  entity RMSD: {per_chain}\n"
        f"  excluded samples: {sorted(excluded)} ({case.entry.notes})\n"
        f"  over {case.entry.tolerance_value} A: {failures}\n"
        f"  tolerance: {case.entry.tolerance_set_from}"
    )


def test_the_replay_notices_one_changed_msa_row(
    parity_case: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tier B's negative control: the assertion above can actually fail.

    One row index in the first recycle's MSA selection is replaced by another
    row of the same cycle -- one row of 160, in one cycle of ten. Everything
    else is what the passing test replays.

    This is the perturbation Tier A cannot see: at the trunk boundary it moves
    the pair stream 0.0118 in relative RMS, under the 0.0196 the CPU-against-GPU
    gap already costs. Two hundred diffusion steps turn it into something a
    coordinate tolerance can resolve -- the asserted samples move to 0.0171,
    0.0419 and 0.1387 A from 0.0147, 0.0148 and 0.0508 -- so sample 4 crosses
    the 0.06 A tolerance at 2.3x while samples 0 and 2 stay inside it. The
    assertion below is therefore "some asserted sample notices", not "all of
    them do": which sample carries a small perturbation is chaos, not a
    property of the port.

    A single noise value is the wrong perturbation to reach for here, which is
    why it is not the one used: adding 1.0 to one of the 9,030 entries of
    ``init_noise`` moved sample 0 by 4.9e-6 A and left the other four bitwise.
    The sampler contracts toward its trajectory instead of amplifying that draw.
    """
    from foldjax.models.protenix.models.trunk_blocks.msa import MSACycleIndexTape

    case = parity_case(PORT, CASE, tier="B")
    _require_cpu()
    stored_input = _arrays(case.path("foldjax-input.npz"))
    features = _nest(stored_input)
    cycles = _msa_cycles(features, _arrays(case.path("msa-tape.npz")))
    tape = _arrays(case.path("sampler-tape.npz"))

    rows = np.array(cycles.row_indices, copy=True)
    assert rows[0, 0] != rows[0, 1], "the first cycle selected one row twice"
    rows[0, 0] = rows[0, 1]
    port = _replay_coordinates(
        features,
        MSACycleIndexTape(row_indices=rows, row_mask=cycles.row_mask),
        tape,
        monkeypatch,
    )

    native = _array(case.path("prediction.npz"), "coordinate")
    per_chain = _entity_rmsd(
        np.asarray(native, dtype=np.float64),
        port,
        stored_input["output_atom_chain_id"].tolist(),
    )
    excluded = set(case.entry.excluded_samples)
    asserted = {
        f"{chain}/{index}": value
        for chain, values in sorted(per_chain.items())
        for index, value in enumerate(values)
        if index not in excluded
    }
    assert max(asserted.values()) > case.entry.tolerance_value, (
        "one swapped MSA row left every asserted sample inside the parity "
        f"tolerance: {asserted} vs {case.entry.tolerance_value} A"
    )
