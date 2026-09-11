"""CPU parity replay of the smallest stored Boltz-2 native capture.

``protein_dna_ion_1aay`` -- PDB 1AAY, the Zif268 zinc-finger peptide on its DNA
duplex with three Zn ions, 115 tokens / 1,216 atoms -- is the smallest Boltz-2
master capture on this host (``docs/ion-case-1aay-master-2026-09-09.md``); no
1UBQ/3GCA/1URN Boltz-2 capture exists here, and the 76-token matched-tape npz
was stripped from the mirror. The capture is ``native-A``: upstream ``b1ebfc4``,
cuEquivariance torch kernels, bf16-mixed, ``float32_matmul_precision=highest``,
no MSA subsampling, n=5 x 200 steps x 3 recycles, seed 101.

This module deliberately pins the **legacy** ``glu_backend="xla"`` arm, which
is no longer the product default -- ``tokamax`` is. The fused kernel is pinned
to Triton with no fallback, so it cannot run on CPU, and every residual below
was calibrated against the XLA gate. What is certified here is therefore the
XLA gated linear unit; the fused default is a GPU-panel question. See
``SHARED_OPTIONS``.

Two tiers, split at the trunk boundary, because that is where the case's known
defect lives:

* **Tier A** runs the port trunk on the captured features and compares the four
  stored trunk arrays.
* **Tier B** injects the *captured native* trunk into the port's diffusion
  sampler and replays the stored 200-step tape to coordinates.

Tier B deliberately does not chain the port's own trunk into the sampler. That
was measured first (same code, ``trunk_source="jax"``): per-sample all-atom RMSD
0.057/0.114/0.147/0.067/0.024 A, i.e. a residual band the size of the very
defects this module exists to catch -- the panel's 0.05 A coordinate gate and
the 0.175 A bistable sample. A tolerance covering that band could not detect
either. Split at the boundary instead, each half gets a real detection floor:
Tier A bounds the trunk drift directly, Tier B bounds the sampler and the tape
injection at ~0.01 A. The composition of the two -- port trunk feeding port
sampler -- is therefore NOT asserted end to end here; that remains a GPU-panel
question (``docs/boltz2-master-kernel-toggle-2026-09-09.md`` "1AAY").

Both tiers go through the same loaders the bench harness uses: this module
imports ``load_features``/``load_tape``/``captured_sampler_trunk`` from
``tests/models/boltz2/scripts/parity_matched_tape.py``, which is the file
``bench/boltz_foldjax_capture.py`` loads as its "legacy" loader, and the entity
comparison is ``bench.boltz_amp_report.compare_coordinates`` -- the function
that produced the recorded GPU numbers in this manifest.

What these assertions can and cannot see, measured by perturbing the fixture in
memory and re-running (numbers and the rest of the calibration are in
``manifest/boltz2.json``):

===================================================  ==================  ======
perturbation                                         effect              result
===================================================  ==================  ======
tier A: MSA masked beyond 1024 rows                  0.0027 -> 0.0571    fails
tier A: one MSA row of 8,192 duplicated              0.0027 -> 0.0028    passes
tier B: tape slices of samples 0 and 1 swapped       0.0019 -> 0.568 A   fails
tier B: one ``step_noises`` scalar of 3.65M +1 A     max|dx| 0.112 A     fails
tier B: ``init_noise`` negated                       0.0019 -> 0.0019 A  passes
===================================================  ==================  ======

The last row is why the guard below is on the unaligned per-atom maximum and
not only on the entity RMSD, and why ``init_noise`` is not used as a tripwire:
at ``sigma_max`` 160 A with ``alignment_reverse_diff`` on, this case's
trajectory does not remember its starting noise, so an assertion that only
watched ``init_noise`` would certify nothing.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.cpu_parity

PORT = "boltz2"
CASE = "protein_dna_ion_1aay"

#: The stored trunk boundary, in the order ``captured_sampler_trunk`` requires.
TRUNK_ARRAYS = ("s", "z", "s_inputs", "relative_position_encoding")

#: Triangle *multiplication* has its own backend switch, read from the
#: environment at call time and defaulting to ``cueq``. On this CPU host the
#: cuEquivariance import is not even stable across processes (it raised
#: ``libcue_ops.so: cannot open shared object file`` in one probe and loaded in
#: the next), so a run that did not pin it would be calibrated against whichever
#: backend happened to import. Pinned to the XLA einsum, which is what
#: ``parity_matched_tape.py`` defaults to for its FP32 control.
TRIANGLE_MULTIPLICATION_ENV = "BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND"

#: Backends and chunking shared by both tiers. ``triangle_backend="cueq"`` is
#: what the GPU capture ran; the fused kernel is a CUDA FFI call, so the CPU
#: replay is a third implementation on this axis and the residuals below are
#: calibrated as such (memory: fusion-decisions-are-backend-specific).
#:
#: ``glu_backend="xla"`` is now the *legacy* arm: the product default is
#: ``tokamax``. It is pinned deliberately and no second arm is added, for two
#: reasons. The fused path is pinned to Triton with no fallback, so it cannot
#: run in a CPU replay at all; and the residuals asserted below were
#: calibrated against the XLA gate, whose activation rounds in float32 where
#: the kernel rounds at its own width. A tokamax arm therefore belongs to the
#: GPU panel, not here. See ``docs/cli.md`` for the measurement that made it
#: the default.
#: ``pair_residual_dtype="float32"`` is pinned for the same reason, one step
#: further: the product default is now ``auto``, which stores the pair
#: residual in bfloat16 on this bfloat16 trunk. That deviates from the AMP
#: placement the capture was taken under -- upstream's eval-mode dropout mask
#: is float32 and promotes every Pairformer residual sum -- so replaying
#: against this capture under the default would compare two policies and
#: charge the difference to the port. The pinned spelling emits no cast at
#: all, so it lowers to the same program, byte for byte, that these residuals
#: were calibrated on. Unpinned, tier A's ``z`` lands at relative RMSE
#: 1.7405e-02 against this case's 5.0e-03 tolerance -- 3.5x over, next to a
#: 2.7360e-03 calibration -- so this is measured, not precautionary. Whether
#: the released default is close enough to upstream is a GPU-panel question
#: and was answered there (5DEI at 2,096 tokens; see ``docs/cli.md``), not by
#: widening a tolerance here.
#:
#: The pin routes around a hole in the tripwire below rather than closing it:
#: ``assert_tripwire`` reads ``cyclic_pos_enc`` and ``fix_sym_check`` and
#: nothing else, so a trunk default that changes the output does not trip it.
#: It surfaced here only because the tolerance was tight enough. The next
#: trunk default to move will meet the same gap.
SHARED_OPTIONS: Mapping[str, Any] = {
    "chunk_size": 128,
    "matmul_precision": "highest",
    "attention_backend": "xla",
    "triangle_backend": "xla",
    "glu_backend": "xla",
    "pair_residual_dtype": "float32",
}

#: Entity-blind secondary guard for tier B: the largest single-atom coordinate
#: difference over the resolved atoms, unaligned. Entity RMSD averages over an
#: entity, so one protein atom moved 1 A inside chain A's 755 atoms would only
#: add ~0.04 A to that entity's RMSD. Calibrated on CPU at 0.03699 A; 0.10 A is
#: 2.7x that, and it is what makes the tape itself testable: shifting ONE scalar
#: of `step_noises` (of 3.65M) by 1 A moves this to 0.1121 A while the entity
#: metric only goes 0.00188 -> 0.00348 A, inside its tolerance.
MAX_ABS_COORDINATE_ANGSTROM = 0.10


def _weights() -> Path:
    """The released confidence bundle, or a failure that says where to put it."""
    from foldjax.paths import weights_dir

    path = weights_dir("boltz2") / "boltz2_conf"
    bundle = path.with_suffix(".safetensors")
    if not bundle.is_file():
        pytest.fail(
            f"Boltz-2 released weights not found at {bundle}.\n"
            "  `foldjax weights fetch boltz2` writes them, or point FOLDJAX_HOME "
            "at a checkout that already has them (a git worktree has no "
            "`.foldjax/` of its own, so the store falls back to "
            "~/.cache/foldjax).",
            pytrace=False,
        )
    return path


def _capture_metadata(case) -> tuple[dict, dict]:
    """Read and enforce the capture policy, as ``bench`` does before a replay.

    The tolerances below are calibrated against ONE policy. A capture taken at
    a different precision, with the kernels off, or with the MSA subsampled is
    a different measurement wearing the same file names, so it fails here
    rather than quietly moving the residual.
    """
    meta = json.loads(case.path("tape.json").read_text())
    effective = json.loads(case.path("effective-model-settings.json").read_text())
    schedule = (meta["num_samples"], meta["num_steps"], meta["num_recycles"])
    assert schedule == (5, 200, 3), f"capture schedule is {schedule}, not n5/200/3"
    assert meta["precision"] == "bf16-mixed", meta["precision"]
    assert meta["kernels"] is True
    assert meta["subsample_msa"] is False
    assert effective["float32_matmul_precision"] == "highest"
    assert effective["cuda_autocast_enabled"] is True
    assert effective["cuda_autocast_dtype"] == "torch.bfloat16"
    assert effective["steering_args"]["fk_steering"] is False, (
        "FK steering draws a resampling tape this fixture does not carry"
    )
    return meta, effective


def _port_trunk(params, features, meta, effective):
    """The port's own trunk at the capture's recycle count, in the capture's AMP.

    ``_cast_trunk_params`` is how ``parity_matched_tape.py`` reproduces the
    native operator-selective autocast: bf16 parameters against fp32 features,
    so the port rounds where upstream rounds (memory:
    match-native-rounding-not-accuracy).
    """
    import jax.numpy as jnp

    from foldjax.models.boltz2.models.trunk_blocks.trunk import (
        _cast_trunk_params,
        boltz2_trunk_forward,
    )

    return boltz2_trunk_forward(
        _cast_trunk_params(params["trunk"], jnp.bfloat16),
        features,
        recycling_steps=int(meta["num_recycles"]),
        use_scan=True,
        subsample_msa=False,
        use_template=bool(effective["use_templates"]),
        **SHARED_OPTIONS,
    )


def _relative_rmse(port, native) -> float:
    import numpy as np

    left = np.asarray(port, np.float64)
    right = np.asarray(native, np.float64)
    assert left.shape == right.shape, f"{left.shape} != {right.shape}"
    scale = float(np.sqrt(np.mean(right**2)))
    assert scale > 0.0, "the captured array is all zero; nothing to normalise by"
    return float(np.sqrt(np.mean((left - right) ** 2)) / scale)


def _assert_schedule(meta, captured) -> None:
    """A closed form on both sides, so read it directly rather than through drift.

    The sampler builds its own schedule from these constants; the capture stored
    the one upstream built. If they disagree the diffusion constants disagree,
    and that is worth naming here instead of arriving later as coordinate
    residual.

    Compared pointwise-relative, not by absolute difference. This schedule opens
    at ``sigma_max * sigma_data`` = 2560, where one float32 ULP is 2.4e-4, so
    ``parity_matched_tape._verdict``'s absolute ``1e-4`` -- calibrated on a
    20-step run -- is below the representation itself here and fails on an
    agreeing schedule. Measured agreement is 1.1e-7, one ULP; 1e-6 is eight of
    them, and a wrong ``rho``/``sigma_max``/``sigma_data`` moves this by orders
    of magnitude.
    """
    import numpy as np

    from foldjax.models.boltz2.models.trunk_blocks.trunk import _sample_schedule

    generated = np.asarray(
        _sample_schedule(
            int(meta["num_steps"]),
            sigma_min=0.0001,
            sigma_max=160.0,
            sigma_data=float(meta["sigma_data"]),
            rho=7.0,
        )
    )
    sigmas = np.asarray(captured, np.float64)
    assert generated.shape == sigmas.shape == (int(meta["num_steps"]) + 1,)
    nonzero = sigmas > 0.0
    difference = float(
        np.max(np.abs(generated[nonzero] - sigmas[nonzero]) / sigmas[nonzero])
    )
    assert difference < 1e-6, (
        f"noise schedule differs from the capture by {difference:.3e} relative; "
        "the diffusion constants do not match upstream"
    )
    assert generated[~nonzero].tolist() == sigmas[~nonzero].tolist()


def test_trunk_boundary_matches_the_native_capture(
    parity_case, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tier A: the port trunk against the four stored native trunk arrays.

    The metric is relative RMSE per array -- ``||port - native|| / ||native||``,
    dimensionless -- because the trunk activations carry no length unit and
    their scales differ by two orders of magnitude between ``s`` and
    ``s_inputs``. The assertion is on the worst array, so a regression confined
    to one of them cannot hide behind the others.
    """
    import inspect

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models.boltz2.bridge.native import load_params
    from foldjax.models.boltz2.models.trunk_blocks.trunk import boltz2_trunk_forward
    from tests.models.boltz2.scripts.parity_matched_tape import load_features

    case = parity_case(PORT, CASE, tier="A")
    meta, effective = _capture_metadata(case)
    monkeypatch.setenv(TRIANGLE_MULTIPLICATION_ENV, "xla")
    weights = _weights()

    trunk_defaults = inspect.signature(boltz2_trunk_forward).parameters
    features = load_features(case.path("features.npz"))
    with np.load(case.path("trunk.npz"), allow_pickle=False) as archive:
        native = {name: archive[name] for name in archive.files}
    assert set(native) == set(TRUNK_ARRAYS), sorted(native)

    params = load_params(weights)
    started = time.perf_counter()
    with jax.default_matmul_precision("highest"):
        trunk = _port_trunk(params, features, meta, effective)
        jax.block_until_ready(trunk["s"])
    wall_seconds = time.perf_counter() - started

    case.assert_tripwire(
        {
            "trunk_checkpoint_flags": (
                f"cyclic_pos_enc={trunk_defaults['cyclic_pos_enc'].default},"
                f"fix_sym_check={trunk_defaults['fix_sym_check'].default}"
            ),
            "trunk_boundary_arrays": ",".join(
                sorted(set(trunk).intersection(TRUNK_ARRAYS))
            ),
        }
    )

    residuals = {
        name: _relative_rmse(trunk[name], native[name]) for name in TRUNK_ARRAYS
    }
    max_abs = {
        name: float(np.max(np.abs(np.asarray(trunk[name], np.float64) - native[name])))
        for name in TRUNK_ARRAYS
    }
    worst = max(residuals, key=residuals.__getitem__)
    report = "  ".join(
        f"{name} relrmse={residuals[name]:.3e} maxabs={max_abs[name]:.3e}"
        for name in TRUNK_ARRAYS
    )
    print(f"\n[tier A] {wall_seconds:.1f} s  {report}")
    assert jnp.isfinite(trunk["s"]).all() and jnp.isfinite(trunk["z"]).all()
    assert residuals[worst] <= case.entry.tolerance_value, (
        f"trunk boundary drifted: worst array {worst} at relative RMSE "
        f"{residuals[worst]:.4e} > tolerance {case.entry.tolerance_value:.4e}\n"
        f"  {report}\n"
        f"  CPU calibration was {case.entry.cpu_residual:.4e} at "
        f"{case.entry.cpu_source_commit}"
    )


def test_tape_pinned_sampler_replay_matches_native_coordinates(
    parity_case, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tier B: 200 stored steps x 5 samples from the captured trunk boundary.

    Per-entity RMSD under one whole-system Kabsch per sample -- the metric the
    GPU panel reports, computed by the same function -- so the numbers here and
    the ``gpu_residual_angstrom`` beside them are the same quantity. 1AAY splits
    into six entities and three of them are a single Zn atom, which is why an
    entity metric is the right one: an ion displacement is invisible in an
    all-atom RMSD over 1,216 atoms and is its own entity here.

    ``excluded_samples`` is empty and is expected to stay empty: sample index 2
    is bistable end to end (0.175 A on GPU, 0.147 A in the rejected CPU full
    replay), but the bifurcation is driven by trunk drift, and with the captured
    trunk injected that sample lands at the same ~0.01 A as the other four.
    """
    import inspect

    import jax
    import jax.numpy as jnp
    import numpy as np

    from bench.boltz_amp_report import compare_coordinates
    from foldjax.models.boltz2.bridge.native import load_params
    from foldjax.models.boltz2.models.trunk_blocks.trunk import (
        boltz2_sample_forward,
    )
    from tests.models.boltz2.scripts.parity_matched_tape import (
        captured_sampler_trunk,
        load_features,
        load_tape,
    )

    case = parity_case(PORT, CASE, tier="B")
    meta, effective = _capture_metadata(case)
    monkeypatch.setenv(TRIANGLE_MULTIPLICATION_ENV, "xla")
    weights = _weights()

    sampler_defaults = inspect.signature(boltz2_sample_forward).parameters
    case.assert_tripwire(
        {
            "sampler_tape_kwargs": ",".join(
                sorted(
                    name
                    for name in ("init_noise", "step_noises", "aug_transforms")
                    if name in sampler_defaults
                )
            ),
            "diffusion_schedule_constants": (
                f"rho={sampler_defaults['rho'].default},"
                f"sigma_max={sampler_defaults['sigma_max'].default},"
                f"sigma_min={sampler_defaults['sigma_min'].default}"
            ),
        }
    )

    features = load_features(case.path("features.npz"))
    tape = load_tape(case.path("tape.npz"), meta)
    with np.load(case.path("trunk.npz"), allow_pickle=False) as archive:
        trunk = captured_sampler_trunk({name: archive[name] for name in archive.files})
    with np.load(case.path("coordinate.npz"), allow_pickle=False) as archive:
        native_coordinate = np.asarray(archive["coordinate"], np.float64)
    with np.load(case.path("features.npz"), allow_pickle=False) as archive:
        raw_features = {name: archive[name] for name in archive.files}

    _assert_schedule(meta, tape["sigmas"])
    params = load_params(weights)
    started = time.perf_counter()
    with jax.default_matmul_precision("highest"):
        output = boltz2_sample_forward(
            params,
            features,
            jax.random.PRNGKey(int(meta["seed"])),
            trunk=trunk,
            recycling_steps=int(meta["num_recycles"]),
            num_sampling_steps=int(meta["num_steps"]),
            multiplicity=int(meta["num_samples"]),
            step_scale=float(meta["step_scale"]),
            gamma_0=float(meta["gamma_0"]),
            gamma_min=float(meta["gamma_min"]),
            noise_scale=float(meta["noise_scale"]),
            sigma_data=float(meta["sigma_data"]),
            init_noise=jnp.asarray(tape["init_noise"]),
            step_noises=jnp.asarray(tape["step_noises"]),
            aug_transforms=(
                jnp.asarray(tape["rotations"]),
                jnp.asarray(tape["translations"]),
            ),
            use_scan=True,
            compute_dtype=jnp.bfloat16,
            **SHARED_OPTIONS,
        )
        coordinate = np.asarray(
            jax.device_get(output["sample_atom_coords"]), np.float64
        )
    wall_seconds = time.perf_counter() - started

    assert coordinate.shape == native_coordinate.shape
    assert np.isfinite(coordinate).all()
    report = compare_coordinates(coordinate, native_coordinate, raw_features)
    kept = [
        index
        for index in range(int(meta["num_samples"]))
        if index not in set(case.entry.excluded_samples)
    ]
    assert kept, "every sample is excluded; the entry asserts nothing"
    per_entity = {
        label: max(values[index] for index in kept)
        for label, values in report["entity_rmsd"].items()
    }
    worst = max(per_entity, key=per_entity.__getitem__)
    resolved = raw_features["atom_pad_mask"][0].astype(bool)
    max_abs = float(
        np.max(np.abs(coordinate[:, resolved] - native_coordinate[:, resolved]))
    )
    print(
        f"\n[tier B] {wall_seconds:.1f} s  worst entity {worst} "
        f"{per_entity[worst]:.5f} A  max|dx| {max_abs:.5f} A"
    )
    for label, values in report["entity_rmsd"].items():
        print(f"  {label}: " + " ".join(f"{value:.5f}" for value in values))

    assert per_entity[worst] <= case.entry.tolerance_value, (
        f"tape-pinned replay drifted: entity {worst} at "
        f"{per_entity[worst]:.5f} A > tolerance "
        f"{case.entry.tolerance_value:.5f} A over samples {kept}\n"
        f"  per entity: {per_entity}\n"
        f"  CPU calibration was {case.entry.cpu_residual:.5f} A at "
        f"{case.entry.cpu_source_commit}"
    )
    assert max_abs <= MAX_ABS_COORDINATE_ANGSTROM, (
        f"one atom moved {max_abs:.5f} A > {MAX_ABS_COORDINATE_ANGSTROM} A "
        "while the entity RMSDs stayed inside tolerance -- an entity average "
        "hides a single-atom move"
    )
