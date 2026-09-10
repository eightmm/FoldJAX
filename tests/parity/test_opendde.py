"""Tier B: replay the stored OpenDDE tape on CPU, atom for atom.

The capture is `opendde-master-20260909-Ayx8BN/protein_1ubq/native-A`: one
upstream OpenDDE run at 76 tokens / 602 atoms, 10 recycles, 200 diffusion
steps, 5 samples, seed 101, recorded together with everything that run drew --
the sampler's initial noise, its per-step churn noise, its per-step rigid
augmentation, and the MSA rows its trunk read on each of the ten passes. Given
that tape the model is a function, so the port either returns upstream's
structure or it does not, and no alignment-free comparison has to be argued
about.

What this certifies, and what it does not
-----------------------------------------

The fixture is upstream's *own* featurized input (`native-input.npz` plus the
derived local-atom geometry in `native-derived.npz`), so this is a core-only
test: a featurizer regression is invisible to it by construction. The capture's
`input-audit.json` is what says the two featurizers agree on this case, and the
per-port featurize suites are where that stays covered.

It is also not the harness the GPU panel ran. The panel's FoldJAX arm went
through the CLI backend with the tape injected; this test calls the model API
directly, exactly as `tests/models/opendde/scripts/parity_matched_tape.py`
does -- same entry point, same `xla` attention backends, same
`cycle_msa_features` injection, `run_confidence=False`. Two harnesses answer
two questions: this one answers "does the core still reproduce the captured
run", not "does the backend still wire it up correctly".

Precision: the port's shipped default is `high`, and CPU cannot run it --
`dot_general` on CPU rejects `TF32_TF32_F32`, because there is no TF32 there.
So the CPU arm is `highest`, and the GPU arm it should be read against is
`fj-highest` (sample 0: 0.00178 A), not the shipped-default `fj-high-A`
(0.00223 A). Both are in the manifest.

One sample, and why that is the same computation
------------------------------------------------

The stored tape holds five samples; this replays sample 0 by slicing every tape
array on its sample axis. That is not an approximation of the five-sample run:
`opendde_infer_static` itself splits the tape this way when `diffusion_chunk_size`
is set (`src/foldjax/models/opendde/models/model.py:954-970` slices
`init_noise`, `step_noises`, `rotations` and `translations` per sample chunk and
concatenates the results), so a one-sample slice is a chunk the shipped code
already runs. The trunk is computed once either way and the diffusion samples do
not interact. Slicing the *step* axis would not be valid -- the noise schedule is
the sampler's clock -- and nothing here does it.

What the assertion is sensitive to
----------------------------------

Measured, not assumed. Negating the sample's 200 per-step churn draws moves the
structure to 0.7317 A -- 146x the tolerance, and the same distance the capture's
own five samples sit from each other (0.32-0.69 A), so that perturbation simply
produces a different valid sample and the assertion fails on it. Negating the
sample's whole initial-noise draw does **not**: it still lands inside the
tolerance, as does flipping one scalar of it. That is not the hook failing to
fire -- on a one-step schedule the same one-scalar flip moves atom 0 by 976 A,
and the whole-draw negation moves coordinates by 5,795 A -- it is the sampler
forgetting where it started: at sigma_max = 2,560 A the initial coordinates are
noise, and 200 churn draws overwrite them. So a defect confined to how
`init_noise` is consumed would not be caught here. Everything the schedule
carries afterwards is.

Sample 0 is one of the four tight samples (native A-B floor 0.00207 A, GPU port
0.00223 A). Sample 1 is the one loose sample on this case in every arm (native
A-B floor 0.00632 A, port 0.01225 A), so it is not what a tolerance should be
calibrated on. Samples 2-4 are as tight as sample 0 and are left out for the CPU
budget: each additional sample is another 200 denoising steps.
"""

from __future__ import annotations

import hashlib
import inspect
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bench import entity_parity

from ._fixtures import ResolvedCase

pytestmark = pytest.mark.cpu_parity

PORT = "opendde"
CASE = "protein_1ubq"
TIER = "B"

#: Which stored sample the sliced tape replays. Recorded in the manifest as the
#: complement of `excluded_samples`, and asserted against it below.
REPLAYED_SAMPLE = 0

#: `high` is TF32 and CPU `dot_general` refuses it, so the CPU arm runs the
#: only fp32 policy that exists here. Applied as a context manager rather than
#: `jax.config.update`, which is process-global and would leak into whatever
#: else shares the session.
MATMUL_PRECISION = "highest"

#: The MSA fields one recorded trunk pass replaces. `msa_mask` is not among them:
#: upstream's mask is all-ones and only prioritises rows during selection, so
#: once the rows are selected every one of them enters the module unmasked
#: (the same reconstruction `parity_matched_tape.build_cycle_msa` makes).
_CYCLE_MSA_FIELDS = ("msa", "has_deletion", "deletion_value")

#: Nested under `pad_info` in the feature dict; flat with this prefix in the npz.
_PAD_INFO_PREFIX = "pad_info."

#: The injection contract this fixture is shaped for. Recorded in the manifest
#: tripwire as the names that are actually present, so a rename drops one and
#: the mismatch is reported instead of being replayed into a wrong answer.
_TAPE_INJECTION = (
    "noise_schedule",
    "init_noise",
    "step_noises",
    "rotations",
    "translations",
    "cycle_msa_features",
)


def observed_tripwire() -> dict[str, str]:
    """Schema identity of *this checkout*, to compare with the capture's.

    Both keys are computed, never spelled: a literal in the test and the same
    literal in the manifest would agree forever (memory:
    a-literal-label-certifies-absence).
    """
    from foldjax.models.opendde.data.padding import _MODEL_FEATURES
    from foldjax.models.opendde.models.model import opendde_infer_static

    contract = ",".join(sorted(_MODEL_FEATURES))
    digest = hashlib.sha256(contract.encode()).hexdigest()[:16]
    parameters = inspect.signature(opendde_infer_static).parameters
    return {
        "model_feature_contract": f"{len(_MODEL_FEATURES)}:{digest}",
        "tape_injection": ",".join(
            name for name in _TAPE_INJECTION if name in parameters
        ),
    }


def _fail(case: ResolvedCase, problem: str) -> None:
    pytest.fail(
        f"{case.entry.port}/{case.entry.case} tier {case.entry.tier}: {problem}\n"
        f"  capture: {case.entry.capture_provenance}",
        pytrace=False,
    )


def load_features(case: ResolvedCase) -> dict[str, Any]:
    """Upstream's featurized input, in the shape the model reads it.

    `native-derived.npz` carries the local-atom geometry (`d_lm`, `v_lm`) and
    the query/key trunking metadata the atom encoder needs; the npz stores that
    metadata flat, the model wants it nested under `pad_info`. `v_lm` is stored
    as bool and is read as float32, which is the dtype the port's own featurizer
    emits for it (the capture's `input-audit.json` records that difference and
    no value difference).
    """
    import jax.numpy as jnp

    features: dict[str, Any] = {}
    with np.load(case.path("native-input.npz"), allow_pickle=False) as archive:
        for name in archive.files:
            features[name] = jnp.asarray(archive[name])
    pad_info: dict[str, Any] = {}
    with np.load(case.path("native-derived.npz"), allow_pickle=False) as archive:
        for name in archive.files:
            value = archive[name]
            if name == "v_lm":
                value = value.astype(np.float32)
            if name.startswith(_PAD_INFO_PREFIX):
                pad_info[name.removeprefix(_PAD_INFO_PREFIX)] = jnp.asarray(value)
            else:
                features[name] = jnp.asarray(value)
    if not pad_info:
        _fail(case, "native-derived.npz carries no pad_info.* arrays")
    features["pad_info"] = pad_info
    return features


def load_tape(case: ResolvedCase, *, sample: int) -> tuple[Any, dict[str, Any]]:
    """The stored tape, sliced to one sample, as `opendde_infer_static` kwargs.

    The shape contract is checked against itself -- steps, samples and atoms are
    read off the arrays and required to agree -- so a fixture whose sample axis
    moved fails as a fixture rather than as a parity regression.
    """
    import jax.numpy as jnp

    with np.load(case.path("tape.npz"), allow_pickle=False) as archive:
        tape = {name: np.asarray(archive[name], np.float32) for name in archive.files}
    missing = {
        "noise_schedule",
        "init_noise",
        "step_noises",
        "rotations",
        "translations",
    }.difference(tape)
    if missing:
        _fail(case, f"tape.npz is missing {sorted(missing)}")
    steps, samples, atoms = tape["step_noises"].shape[:3]
    expected = {
        "noise_schedule": (steps + 1,),
        "init_noise": (samples, atoms, 3),
        "step_noises": (steps, samples, atoms, 3),
        "rotations": (steps, samples, 3, 3),
        "translations": (steps, samples, 3),
    }
    wrong = [
        f"{name} is {tape[name].shape}, the contract is {shape}"
        for name, shape in expected.items()
        if tape[name].shape != shape
    ]
    if wrong:
        _fail(case, "tape.npz does not match the sampler contract: " + "; ".join(wrong))
    if not 0 <= sample < samples:
        _fail(case, f"sample {sample} is outside the stored {samples}")
    return jnp.asarray(tape["noise_schedule"]), {
        "init_noise": jnp.asarray(tape["init_noise"][sample : sample + 1]),
        "step_noises": tuple(
            jnp.asarray(tape["step_noises"][step, sample : sample + 1])
            for step in range(steps)
        ),
        "rotations": jnp.asarray(tape["rotations"][:, sample : sample + 1]),
        "translations": jnp.asarray(tape["translations"][:, sample : sample + 1]),
    }


def load_cycle_msa(case: ResolvedCase) -> tuple[dict[str, Any], ...]:
    """The rows upstream's trunk actually read, one draw per recycle.

    Upstream re-samples the alignment on every pass, so feeding the port the
    whole alignment is not the same model on the same input; these are the
    selected rows themselves, not indices applied to a re-featurized alignment,
    which would compare this checkout against itself.
    """
    import jax.numpy as jnp

    with np.load(case.path("msa.npz"), allow_pickle=False) as archive:
        stored = {name: np.asarray(archive[name]) for name in archive.files}
    if "rows" not in stored:
        _fail(case, "msa.npz records no per-cycle row draws")
    cycles = int(stored["rows"].shape[0])
    draws: list[dict[str, Any]] = []
    for cycle in range(cycles):
        selected: dict[str, Any] = {}
        for field in _CYCLE_MSA_FIELDS:
            key = f"selected_{field}"
            if key not in stored:
                _fail(case, f"msa.npz is missing {key}")
            value = stored[key][cycle]
            selected[field] = jnp.asarray(
                value.astype(np.int32 if field == "msa" else np.float32)
            )
        selected["msa_mask"] = jnp.ones(selected["msa"].shape, dtype=jnp.float32)
        draws.append(selected)
    return tuple(draws)


def native_coordinates(case: ResolvedCase) -> np.ndarray:
    with np.load(case.path("coordinate.npz"), allow_pickle=False) as archive:
        return np.asarray(archive["coordinate"], np.float64)


def weights_path() -> Path:
    """The released OpenDDE weights, or a failure that says where to put them."""
    from foldjax.paths import foldjax_home, weights_dir

    path = weights_dir(PORT) / "opendde.jax"
    if not path.is_file():
        pytest.fail(
            f"the CPU parity replay needs the released OpenDDE weights at {path}\n"
            f"  FoldJAX home is {foldjax_home()} (override with FOLDJAX_HOME)\n"
            "  fetch them with `foldjax weights fetch opendde`",
            pytrace=False,
        )
    return path


def replay(
    case: ResolvedCase,
    *,
    sample: int = REPLAYED_SAMPLE,
    perturb: Callable[[dict[str, Any], dict[str, Any], tuple], tuple] | None = None,
) -> tuple[np.ndarray, float]:
    """Coordinates for one stored sample, plus the wall seconds it took.

    `perturb` receives `(features, tape, cycle_msa)` and returns them changed.
    It is how the assertion below is shown to be able to fail: the fixture files
    are sha256-checked, so editing one on disk trips the resolver rather than
    the parity comparison, and a tripwire that cannot fire proves nothing
    (memory: patch-a-function-arm-needs-a-tripwire).
    """
    import jax

    from foldjax.models.opendde.bridge.weights_io import load_native_weights
    from foldjax.models.opendde.models.model import opendde_infer_static

    features = load_features(case)
    schedule, tape = load_tape(case, sample=sample)
    cycle_msa = load_cycle_msa(case)
    if perturb is not None:
        features, tape, cycle_msa = perturb(features, tape, cycle_msa)
    params = load_native_weights(weights_path())

    started = time.perf_counter()
    with jax.default_matmul_precision(MATMUL_PRECISION):
        output = opendde_infer_static(
            features,
            params,
            schedule,
            key=None,
            num_samples=1,
            num_recycles=len(cycle_msa),
            run_confidence=False,
            cycle_msa_features=cycle_msa,
            diffusion_attention_backend="xla",
            trunk_single_attention_backend="xla",
            trunk_triangle_attention_backend="xla",
            structural_single_attention_backend="xla",
            structural_triangle_attention_backend="xla",
            **tape,
        )
        coordinate = np.asarray(jax.device_get(output["coordinate"]), np.float64)
    seconds = time.perf_counter() - started

    while coordinate.ndim > 3 and coordinate.shape[0] == 1:
        coordinate = coordinate[0]
    if coordinate.shape[0] != 1:
        _fail(case, f"expected one sample of coordinates, got {coordinate.shape}")
    return coordinate[0], seconds


def entity_rmsds(
    case: ResolvedCase, mine: np.ndarray, native: np.ndarray
) -> Mapping[Any, float]:
    """Per-entity RMSD under one global superposition -- the panel's own metric.

    `bench.entity_parity` is what produced the `entity_rmsd` numbers in the
    capture's comparison JSONs, so the value here is comparable to the recorded
    GPU residual by construction rather than by resemblance. Entities are the
    chains: `asym_id` gathered to atoms.
    """
    with np.load(case.path("native-input.npz"), allow_pickle=False) as archive:
        asym_id = np.asarray(archive["asym_id"])
        atom_to_token = np.asarray(archive["atom_to_token_idx"])
    labels = [int(asym_id[token]) for token in atom_to_token]
    keys = list(range(len(labels)))
    mask = np.ones((1, len(labels)), dtype=bool)
    comparison = entity_parity.compare_entity_parity(
        mine[None, ...],
        native[None, ...],
        keys,
        keys,
        labels,
        labels,
        mask,
        mask,
    )
    return comparison["entity_max_rmsd"]


def test_tape_pinned_replay_matches_the_native_capture(parity_case) -> None:
    case = parity_case(PORT, CASE, tier=TIER)
    entry = case.entry
    case.assert_tripwire(observed_tripwire())
    assert REPLAYED_SAMPLE not in entry.excluded_samples, (
        f"the manifest excludes sample {REPLAYED_SAMPLE}, which is the one this "
        "test replays"
    )

    mine, seconds = replay(case, sample=REPLAYED_SAMPLE)
    native = native_coordinates(case)
    if native.shape[0] <= REPLAYED_SAMPLE:
        _fail(case, f"coordinate.npz holds {native.shape[0]} samples")
    measured = entity_rmsds(case, mine, native[REPLAYED_SAMPLE])
    worst = max(measured.values())
    # Printed on the way past, not only on failure: a regression detector that
    # reports nothing while it passes cannot show the residual drifting towards
    # its tolerance, and cannot show an improvement either.
    print(
        f"\nopendde/{CASE} sample {REPLAYED_SAMPLE}: entity RMSD {worst:.7f} A "
        f"(tolerance {entry.tolerance_value:g} A, CPU calibration "
        f"{entry.cpu_residual:.7f} A), replay {seconds:.1f} s"
    )

    assert worst <= entry.tolerance_value, (
        f"OpenDDE {CASE} sample {REPLAYED_SAMPLE}: entity RMSD "
        f"{worst:.6f} A > tolerance {entry.tolerance_value:g} A "
        f"({entry.tolerance_set_from})\n"
        f"  per entity: "
        + ", ".join(f"{label}={value:.6f}" for label, value in sorted(measured.items()))
        + f"\n  CPU calibration was {entry.cpu_residual:.6f} A in "
        f"{entry.cpu_wall_seconds:.0f} s; this run took {seconds:.0f} s\n"
        f"  capture: {entry.capture_provenance}"
    )
