"""Tier B: replay ESMFold2's stored native tape on CPU, to coordinates.

ESMFold2 is stochastic by design -- three deliberate draw sites, so two native
runs of the same input land 0.006 A apart -- and the stored tape is what makes
a comparison possible at all: every forward draw the native run made is
replayed into this port instead of being re-drawn.

What this certifies is the *shared core*: the tape's ``features.npz`` is the
native feature dictionary already built, so nothing here exercises either
featurizer, and ``upstream_lm.npz`` is the native ESM-C output injected as
``lm_hidden_states``, so nothing here runs the 25.4 GB language model. What is
left is the trunk, the diffusion sampler and the atom decoder, at the released
five-sample schedule, against native coordinates from the same draws.

There is no Tier A for this port: the ESMFold2 capture stores no trunk
boundary (``tape.npz`` holds ``initial_pair_state``, an *input*, not the
trunk's output), so coordinates are the only stored native reference.

Injection path: this module does not shell out to ``bench/esmfold2_tape.py
replay``; it calls the same ``structure_model.predict`` with the same
arguments the harness builds, reusing that module's ``SAMPLES``, ``SEED``,
``_replay_settings`` and ``_model_features`` and running its three tape
validators. What it does not reuse is the CLI's sha256 binding layer (the
manifest and ``FixtureStore`` verify the same bytes) and ``_read_lm`` (the
fixture is the same file, read directly). Two harnesses answer two questions:
the CLI answers "was this GPU run bound to that capture", this answers "does
this checkout still fold that tape to those coordinates".

The second test flips one bit of the same tape and requires the comparison to
fail. It is here rather than in a session note because a parity test that
cannot fail looks exactly like one that passes.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ._fixtures import ResolvedCase, sha256_of

pytestmark = pytest.mark.cpu_parity

PORT = "esmfold2"
CASE = "protein_1ubq"
TIER = "B"

#: The eight classified tape arrays, exactly as ``bench/esmfold2_tape.py``
#: requires them. Named here so a rename in the port trips the tripwire
#: instead of silently replaying fewer draws.
TAPE_INPUTS = (
    "diffusion_churn_normals",
    "diffusion_initial_normal",
    "diffusion_rotation_quaternions",
    "diffusion_translations",
    "initial_pair_state",
    "lm_dropout_masks",
    "msa_column_keep",
    "msa_row_choices",
)

FIXTURE_FEATURES = "features.npz"
FIXTURE_TAPE = "tape.npz"
FIXTURE_LM = "upstream_lm.npz"
FIXTURE_COORDS = "upstream_coords.npz"


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: np.asarray(loaded[name]) for name in loaded.files}


def _weights_directory() -> Path:
    """The released structure checkpoint, or a failure that says where.

    Missing weights fail rather than skip: a skipped parity test reads exactly
    like a passing one in a summary line.
    """
    from foldjax import paths

    directory = paths.weights_dir(PORT)
    if not (directory / "model.safetensors").is_file():
        pytest.fail(
            f"ESMFold2 weights are not at {directory}. Fetch them with "
            "`foldjax weights fetch --model esmfold2`, or point FOLDJAX_HOME "
            "at a store that has them. ESM-C is NOT needed here -- this test "
            f"injects the native language-model output from {FIXTURE_LM}.",
            pytrace=False,
        )
    return directory


def observed_schema(weights: Path, settings: Any) -> dict[str, str]:
    """What *this checkout and checkpoint* say, for the manifest to disagree with.

    Read from the port and the weights directory, never from the fixture: a
    tripwire fed by the stored file would compare the capture against itself.
    """
    import inspect

    from bench.esmfold2_tape import SAMPLES, SEED
    from foldjax.models.esmfold2.models import model as structure_model

    accepted = set(inspect.signature(structure_model.predict).parameters)
    return {
        "tape_inputs": ",".join(sorted(set(TAPE_INPUTS) & accepted)),
        "samples": str(SAMPLES),
        "seed": str(SEED),
        "trunk_dtype": str(settings.trunk_dtype),
        "checkpoint_config_sha256": sha256_of(weights / "config.json"),
    }


def load_structure_model(weights: Path) -> Any:
    """The released structure checkpoint, without ESM-C.

    ``language_model=False`` is not a degraded mode here: the native language
    model's output is replayed from the capture, so loading the 25.4 GB
    checkpoint would only produce a value this run then discards.
    """
    from foldjax.models.esmfold2 import inference

    return inference.load(
        weights, dtype="float32", esmc_dtype="bfloat16", language_model=False
    )


@lru_cache(maxsize=1)
def _compiled_predict() -> Any:
    """One jit wrapper for the module.

    The tape arrays are traced values, not static, so the perturbed replay in
    the tripwire test reuses this program instead of compiling a second one --
    which at this size is most of the wall time.
    """
    import jax

    from bench.esmfold2_lm_encoder_candidate import compiler_control
    from foldjax.models.esmfold2.models import model as structure_model

    return jax.jit(
        structure_model.predict,
        static_argnames=("settings", "n_chains"),
        compiler_options=compiler_control("default"),
    )


def replay_to_coordinates(
    case: ResolvedCase,
    loaded: Any,
    *,
    perturb: Callable[[dict[str, np.ndarray]], None] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    """Fold the stored tape to coordinates. Returns (coords, features, seconds).

    ``perturb`` edits the tape in memory before it is handed to the model; it
    exists so the test can prove its own assertion can fail (a file-level edit
    would prove the digest check works, which is a different check).
    """
    import jax

    from bench.esmfold2_tape import SAMPLES, SEED, _model_features, _replay_settings
    from foldjax.models.esmfold2.models import model as structure_model

    features = _npz(case.path(FIXTURE_FEATURES))
    tape = _npz(case.path(FIXTURE_TAPE))
    if set(tape) != set(TAPE_INPUTS):
        pytest.fail(
            f"stored tape holds {sorted(tape)}, this test replays "
            f"{sorted(TAPE_INPUTS)}",
            pytrace=False,
        )
    if perturb is not None:
        perturb(tape)
    lm = _npz(case.path(FIXTURE_LM))["lm_hidden_states"]

    settings = _replay_settings(loaded.settings)
    if settings.trunk_dtype != "bfloat16":
        # The capture was taken under the BF16 trunk policy and the harness
        # refuses any other; a checkpoint that stopped saying so would make
        # this a comparison of two different models.
        pytest.fail(
            f"replay requires the BF16 trunk policy, checkpoint says "
            f"{settings.trunk_dtype!r}",
            pytrace=False,
        )

    batch, tokens = features["token_attention_mask"].shape
    structure_model.validate_initial_pair_state(
        tape["initial_pair_state"], batch=batch, tokens=tokens, width=settings.d_pair
    )
    structure_model.validate_msa_tape(
        tape["msa_column_keep"],
        tape["msa_row_choices"],
        batch=batch,
        tokens=tokens,
        depth=features["msa"].shape[1] if "msa" in features else None,
        loops=max(1, settings.num_recycles + 1),
        settings=settings,
    )
    structure_model.diffusion.validate_diffusion_tape(
        tape["diffusion_initial_normal"],
        tape["diffusion_rotation_quaternions"],
        tape["diffusion_translations"],
        tape["diffusion_churn_normals"],
        steps=len(structure_model.diffusion.noise_schedule(settings.diffusion)) - 1,
        batch=batch * SAMPLES,
        atoms=features["atom_attention_mask"].shape[-1],
    )

    arrays = {
        name: jax.numpy.asarray(value)
        for name, value in _model_features(features).items()
    }
    dynamic = {
        "lm_hidden_states": jax.numpy.asarray(lm),
        **{name: jax.numpy.asarray(value) for name, value in tape.items()},
    }
    static = {"settings": settings, "n_chains": int(features["asym_id"].max()) + 1}
    predict = _compiled_predict()
    # Process-global, like the harness (:1008); restored so the rest of the
    # session is not silently run at a precision it did not ask for.
    previous = jax.config.jax_default_matmul_precision
    jax.config.update("jax_default_matmul_precision", "highest")
    try:
        started = time.perf_counter()
        output = predict(
            jax.random.key(SEED), arrays, loaded.parameters, **static, **dynamic
        )
        jax.block_until_ready(output["sample_atom_coords"])
        elapsed = time.perf_counter() - started
    finally:
        jax.config.update("jax_default_matmul_precision", previous)
    return np.asarray(output["sample_atom_coords"], np.float32), features, elapsed


def per_sample_rmsd(
    native: np.ndarray, candidate: np.ndarray, features: Mapping[str, np.ndarray]
) -> dict[str, list[float]]:
    """Entity RMSD per sample, under the panel's own comparator."""
    from bench.esmfold2_tape_report import compare_coordinates

    report = compare_coordinates(native, candidate, dict(features))
    return {str(label): list(values) for label, values in report["entity_rmsd"].items()}


def _worst_asserted(
    rmsd: Mapping[str, list[float]], excluded: tuple[int, ...]
) -> float:
    """The largest entity RMSD over the samples this entry asserts on."""
    values = [
        value
        for series in rmsd.values()
        for index, value in enumerate(series)
        if index not in excluded
    ]
    if not values:
        pytest.fail(f"every sample is excluded ({sorted(excluded)})", pytrace=False)
    return max(values)


def _table(rmsd: Mapping[str, list[float]], excluded: tuple[int, ...]) -> str:
    lines = []
    for label, values in sorted(rmsd.items()):
        for index, value in enumerate(values):
            mark = " (excluded)" if index in excluded else ""
            lines.append(f"  {label} sample {index}: {value:.6f} A{mark}")
    return "\n".join(lines)


def test_tape_replay_matches_native_coordinates(
    parity_case: Callable[..., ResolvedCase],
) -> None:
    """The stored native draws, folded here, must land where native landed."""
    import jax

    case = parity_case(PORT, CASE, tier=TIER)
    entry = case.entry
    weights = _weights_directory()
    loaded = load_structure_model(weights)
    case.assert_tripwire(observed_schema(weights, loaded.settings))
    assert jax.default_backend() == "cpu", (
        "the tolerance in the manifest was calibrated on CPU XLA; on another "
        f"backend it means nothing (this process is on {jax.default_backend()})"
    )

    native = _npz(case.path(FIXTURE_COORDS))["coords"]
    candidate, features, elapsed = replay_to_coordinates(case, loaded)
    rmsd = per_sample_rmsd(native, candidate, features)

    worst = _worst_asserted(rmsd, entry.excluded_samples)
    report = (
        f"{PORT}/{CASE} tier {TIER}: worst asserted entity RMSD {worst:.6f} A "
        f"vs tolerance {entry.tolerance_value} A ({entry.tolerance_set_from}); "
        f"replay {elapsed:.1f} s on "
        f"{os.environ.get('OMP_NUM_THREADS', 'default')} threads\n"
        f"{_table(rmsd, entry.excluded_samples)}"
    )
    print("\n" + report)
    assert worst <= entry.tolerance_value, report


def test_one_flipped_msa_column_trips_the_comparison(
    parity_case: Callable[..., ResolvedCase],
) -> None:
    """The assertion above must be able to fail.

    A parity test that cannot fail certifies nothing, and every part of this
    one is shared with a passing run: the same fixture, the same compiled
    program, the same comparator, the same tolerance. One bit of the tape is
    flipped -- ``msa_column_keep[0, 0]``, which is whether the native run
    masked the first MSA column -- and the trunk state that reaches every
    sample changes with it.

    The size of the perturbation is chosen from measurement, not taste. This
    sampler is strongly contracting at 76 tokens: adding 1.0 to a single
    element of ``diffusion_initial_normal`` -- one number out of 9,120, where
    sigma is 160 A -- moves its sample only 0.0166 A -> 0.0262 A, *inside*
    this tolerance. A tripwire built on that would have proved nothing. The
    flipped MSA column moves the asserted samples to 0.029-0.057 A and the
    excluded one to 0.278 A, so it clears the tolerance on the strength of a
    real change rather than a large one.
    """
    case = parity_case(PORT, CASE, tier=TIER)
    entry = case.entry
    weights = _weights_directory()
    loaded = load_structure_model(weights)

    def flip_one_msa_column(tape: dict[str, np.ndarray]) -> None:
        tape["msa_column_keep"][0, 0] = ~tape["msa_column_keep"][0, 0]

    native = _npz(case.path(FIXTURE_COORDS))["coords"]
    candidate, features, _ = replay_to_coordinates(
        case, loaded, perturb=flip_one_msa_column
    )
    rmsd = per_sample_rmsd(native, candidate, features)
    worst = _worst_asserted(rmsd, entry.excluded_samples)
    assert worst > entry.tolerance_value, (
        "flipping one MSA column of the tape left the worst asserted sample at "
        f"{worst:.6f} A, inside the {entry.tolerance_value} A tolerance -- this "
        "comparison would not notice a real regression either\n"
        f"{_table(rmsd, entry.excluded_samples)}"
    )


def test_the_tolerance_is_tighter_than_the_defects_the_panel_recorded() -> None:
    """A tolerance wide enough to pass anything detects nothing.

    The GPU panel's coordinate gate for this port is 0.05 A
    (``native-A-vs-port-A.json``), and the recorded native A-vs-B floor for
    this case is 0.0059 A. A tolerance at or above the gate would stop
    distinguishing a passing run from the 0.122 A one the panel failed, so the
    manifest's number has to sit below it -- and above the CPU calibration,
    which ``_manifest.py`` already enforces.
    """
    from ._manifest import entry_for

    entry = entry_for(PORT, CASE, TIER)
    assert entry.tolerance_value < 0.05, (
        f"tolerance {entry.tolerance_value} A is at or above the panel's "
        "coordinate gate; it could not fail the run the panel failed"
    )
    if entry.excluded_samples:
        assert entry.notes.strip(), (
            f"{PORT}/{CASE} excludes samples {list(entry.excluded_samples)} "
            "with no reason recorded in notes"
        )
