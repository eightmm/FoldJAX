"""CPU parity for the OpenFold3 port against one stored native OpenBind capture.

The capture is ``protein_rna_1urn`` (118 tokens, 1,225 atoms), the smallest case
the cuEq master panel admits: 1UBQ and 3GCA are excluded there because native
cuEq attention falls back to plain attention below 100 tokens, so their numbers
describe a kernel the panel is not measuring
(``docs/openbind-master-cueq-panel-2026-09-09.md``).

Two tiers over the same fixture and the same tape:

* **Tier A** stops the released program after the trunk
  (``stop_after_trunk``) and compares ``single_inputs``/``single``/``pair``
  against the native ``trunk-00.npz``. It costs one trunk (four cycles, 1,024
  MSA rows) and no diffusion.
* **Tier B** replays the whole tape to coordinates and measures the per-entity
  RMSD the master panel measures, under one whole-system Kabsch fit per sample.

Both go through the bench harness's own tape path -- ``load_capture``,
``prepare_core_features``, ``model_feature_batch``, ``ForwardTape.prepare_features``
and ``native_trunk_arrays`` are imported from ``bench.openbind_core_replay`` and
``bench.openbind_tape_adapter``, and the coordinate metric is
``bench.entity_parity.compare_entity_parity`` called with the arguments
``bench/openbind_core_report.py`` calls it with (atom keys from the capture's
five annotation columns, entities grouped by ``chain_id``, five samples paired
positionally). Only ``bench.openbind_core_replay.main``'s file plumbing --
argparse, ``preflight.json``, ``finished.json``, the source-hash re-checks --
is left out; the executed program and the arrays it is fed are the harness's
(memory: two-harnesses-answer-two-questions).

What this certifies and what it does not:

* It certifies the model core from the pre-featurized native batch onward. The
  featurizer is *not* exercised: ``input.npz`` was written by upstream, so a
  regression in this checkout's featurization is invisible here.
* The native capture ran cuEq triangle kernels on a GPU; a CPU replay must run
  the XLA ones. The kernel family therefore differs from the capture by
  construction, which is one reason the tolerance is calibrated on CPU and the
  test refuses to run anywhere else.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest

from ._fixtures import ResolvedCase, sha256_of

pytestmark = pytest.mark.cpu_parity

PORT = "openfold3"
CASE = "protein_rna_1urn"

#: Fixture file name -> path inside a capture directory. The fixture layout is
#: flat (one directory per case, plain file names), so the one file that lives
#: in a subdirectory of the capture is stored under its base name and put back
#: where the adapter looks for it.
CAPTURE_LAYOUT = {"model_config.json": "predictions/model_config.json"}

#: Released checkpoint the capture was produced with. Overridable because the
#: managed store is per machine; never downloaded (PROJECT.md forbids implicit
#: downloads), and a missing file fails the test rather than skipping it.
CHECKPOINT_ENV = "FOLDJAX_PARITY_OF3_CHECKPOINT"
CHECKPOINT_NAME = "of3-ob-2025-06-30-174k.pt"


def resolve_checkpoint() -> Path:
    """The released OpenFold3 checkpoint, from the environment or the store."""
    override = os.environ.get(CHECKPOINT_ENV)
    if override:
        return Path(override).expanduser()
    from foldjax import paths

    return paths.weights_dir(PORT) / CHECKPOINT_NAME


def capture_shaped_dir(files: Mapping[str, Path], root: Path) -> Path:
    """Rebuild the capture directory shape the bench adapter reads.

    ``load_capture`` opens ``runner.yml``, ``effective-model.json``,
    ``predictions/model_config.json``, ``input.npz`` and ``tape.npz`` by name
    relative to a capture root, and it is the harness's own validation of the
    native configuration (FP32, n5/200/four passes, fixed-depth all-MSA
    selection). Symlinking the verified fixture files into that shape reuses it
    instead of restating it here.
    """
    for name, source in files.items():
        target = root / CAPTURE_LAYOUT.get(name, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
    return root


def tape_layout(tape) -> str:
    """The parsed tape's shape signature, as the tripwire's schema identity.

    A property of what ``parse_forward_tape`` produced, not the spelling of the
    code that produced it: if the tape parser starts handing the model a
    different number of cycles, samples, steps or atoms, this string moves.
    """
    return ";".join(
        f"{name}={tuple(np.shape(getattr(tape, name)))}"
        for name in ("msa_indices", "noise", "quaternions", "translations")
    )


def model_feature_schema() -> str:
    """Digest of the model ABI the stored batch has to satisfy."""
    from foldjax.models.openfold3.data.featurize import (
        MODEL_FEATURES,
        OPTIONAL_MODEL_FEATURES,
    )

    payload = "|".join(
        (",".join(sorted(MODEL_FEATURES)), ",".join(sorted(OPTIONAL_MODEL_FEATURES)))
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def observed_tripwire(tape, checkpoint: Path) -> dict[str, str]:
    """What this checkout and this machine report for the capture's schema."""
    from bench.openbind_tape_adapter import UPSTREAM_COMMIT

    return {
        "upstream_commit": UPSTREAM_COMMIT,
        "tape_layout": tape_layout(tape),
        "model_feature_schema": model_feature_schema(),
        "checkpoint_sha256": sha256_of(checkpoint),
    }


def native_batch(capture_dir: Path):
    """Load the capture through the harness and return ``(tape, effective, batch)``."""
    from bench.openbind_core_replay import model_feature_batch
    from bench.openbind_tape_adapter import load_capture, prepare_core_features

    tape, effective = load_capture(capture_dir)
    with np.load(capture_dir / "input.npz", allow_pickle=False) as archive:
        features = prepare_core_features(dict(archive), max_atoms_per_token=23)
    return tape, effective, tape.prepare_features(model_feature_batch(features))


def replay_config(features, tape, effective, *, stop_after_trunk: bool):
    """The fused replay's configuration: `openbind_core_replay.main`, verbatim.

    ``max_array_bytes=None`` is what ``array_budget_bytes`` returns off the
    streamed path, and the sampling knobs come from the capture, not from a
    default that could drift away from what the native run did.
    """
    from foldjax.models.openfold3.inference import released_config

    return released_config(
        n_token=features["token_mask"].shape[-1],
        n_atom=features["atom_mask"].shape[-1],
        num_recycles=4,
        num_samples=5,
        num_steps=200,
        msa_depth=tape.msa_indices.shape[1],
        max_array_bytes=None,
        returned_representations=(
            ("single_inputs", "single", "pair") if stop_after_trunk else ()
        ),
        stop_after_trunk=stop_after_trunk,
        per_sample_token_cutoff=effective["settings"]["memory"]["eval"][
            "per_sample_token_cutoff"
        ],
    )


@functools.cache
def inference_params(checkpoint: str):
    """Map the released checkpoint once per process; both tiers reuse it."""
    from foldjax.models.openfold3.bridge.checkpoint import load_checkpoint
    from foldjax.models.openfold3.bridge.torch_mapping import (
        map_inference_params,
        prune_sample_diffusion_aliases,
        resolve_model_prefix,
    )

    state = load_checkpoint(checkpoint)
    prefix = resolve_model_prefix(state, None)
    prune_sample_diffusion_aliases(state, prefix=prefix)
    return map_inference_params(state, prefix)


def run_replay(capture_dir: Path, checkpoint: Path, *, stop_after_trunk: bool):
    """Compile and run the fused replay program; return ``(prediction, seconds)``.

    ``triangle_kernel="xla"`` where the capture ran cuEq: the fused cuEq path is
    a CUDA kernel and this test runs on CPU. That difference is part of what the
    CPU calibration measures, and it is why the tolerance may not be reused on
    GPU (memory: cpu-calibrated-tolerance-fails-on-gpu).
    """
    import jax

    from foldjax.models.openfold3.bridge.chemistry import representative_atom_table
    from foldjax.models.openfold3.inference import compile_predict

    tape, effective, features = native_batch(capture_dir)
    config = replay_config(features, tape, effective, stop_after_trunk=stop_after_trunk)
    run = compile_predict(config, representative_atom_table(), triangle_kernel="xla")
    started = time.perf_counter()
    result = jax.device_get(
        run(
            jax.random.key(101),
            features,
            inference_params(str(checkpoint)),
            noise_tape=tape.noise,
            augmentation_tape=tape.augmentation(),
        )
    )
    return result, time.perf_counter() - started


def trunk_residuals(port: Sequence[np.ndarray], native: Sequence[np.ndarray]):
    """Per-array max |delta| divided by the native array's own scale.

    The three trunk arrays span five orders of magnitude (``single`` reaches
    1.6e5, ``single_inputs`` 9), so an absolute maximum would be a report about
    ``single`` alone. Dividing by ``max|native|`` gives one number per array
    that means the same thing in each.
    """
    names = ("single_inputs", "single", "pair")
    residuals = {}
    for name, left, right in zip(names, port, native, strict=True):
        scale = float(np.abs(right).max())
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"native trunk array {name} has no scale")
        delta = np.abs(left.astype(np.float64) - right.astype(np.float64)).max()
        residuals[name] = float(delta / scale)
    return residuals


def entity_rmsd(capture_dir: Path, coordinates: np.ndarray):
    """The master panel's coordinate metric, called the way the panel calls it.

    ``bench/openbind_core_report.py:compare`` builds the atom keys from the five
    stored annotation columns, groups entities by ``chain_id``, broadcasts the
    captured atom mask over the five samples and pairs samples positionally.
    """
    from bench.entity_parity import compare_entity_parity

    with np.load(capture_dir / "input.npz", allow_pickle=False) as features:
        columns = [
            features["atom_array.0.annotation." + name].tolist()
            for name in ("chain_id", "res_id", "res_name", "atom_name", "element")
        ]
        atom_mask = features["atom_mask"]
    keys = list(zip(*columns, strict=True))
    labels = columns[0]
    with np.load(capture_dir / "coordinate.npz", allow_pickle=False) as native:
        native_coordinates = native["coordinate"]
    samples = native_coordinates.shape[0]
    if coordinates.shape != native_coordinates.shape:
        raise ValueError(
            f"replayed {coordinates.shape}, capture holds {native_coordinates.shape}"
        )
    mask = np.broadcast_to(atom_mask.astype(bool), (samples, len(keys)))
    return compare_entity_parity(
        native_coordinates, coordinates, keys, keys, labels, labels, mask, mask
    )


def prepared_case(case: ResolvedCase, tmp_path: Path) -> tuple[Path, Path]:
    """Verified fixtures in capture shape, plus the checkpoint they were made with.

    Fails -- never skips -- when the checkpoint is absent: a skipped parity test
    is indistinguishable from a passing one in a summary line.
    """
    import jax

    if jax.default_backend() != "cpu":
        pytest.fail(
            "the CPU parity subset asserts a CPU-calibrated tolerance; this "
            f"process is on {jax.default_backend()!r}. Set JAX_PLATFORMS=cpu.",
            pytrace=False,
        )
    checkpoint = resolve_checkpoint()
    if not checkpoint.is_file():
        pytest.fail(
            f"released OpenFold3 checkpoint missing: {checkpoint}\n"
            f"  point {CHECKPOINT_ENV} at a copy of {CHECKPOINT_NAME}; the "
            "capture in this manifest was produced with it and the tripwire "
            "checks its sha256.",
            pytrace=False,
        )
    return capture_shaped_dir(case.files, tmp_path / "capture"), checkpoint


def test_trunk_matches_native_capture(parity_case, tmp_path: Path) -> None:
    """Tier A: the trunk boundary, against the native ``trunk-00.npz``."""
    from bench.openbind_core_replay import native_trunk_arrays

    case = parity_case(PORT, CASE, tier="A")
    capture_dir, checkpoint = prepared_case(case, tmp_path)
    tape, _, features = native_batch(capture_dir)
    case.assert_tripwire(observed_tripwire(tape, checkpoint))

    prediction, seconds = run_replay(capture_dir, checkpoint, stop_after_trunk=True)
    port = (prediction.single_inputs, prediction.single, prediction.pair)
    if any(array is None for array in port):
        raise AssertionError("the trunk program returned no representations")
    native, _ = native_trunk_arrays(capture_dir, features["token_mask"].shape[-1])
    residuals = trunk_residuals(port, native)

    entry = case.entry
    worst = max(residuals.values())
    assert worst <= entry.tolerance_value, (
        f"OpenFold3 trunk boundary moved: {entry.tolerance_metric} "
        f"{worst:.3e} > {entry.tolerance_value:.3e} "
        f"({entry.tolerance_set_from}); per array "
        + ", ".join(f"{k} {v:.3e}" for k, v in residuals.items())
        + f"; CPU calibration {entry.cpu_residual:.3e} in "
        f"{entry.cpu_wall_seconds:.0f} s, this run {seconds:.0f} s"
    )


def test_replay_coordinates_match_native(parity_case, tmp_path: Path) -> None:
    """Tier B: the whole tape replayed to coordinates, per-entity RMSD."""
    case = parity_case(PORT, CASE, tier="B")
    capture_dir, checkpoint = prepared_case(case, tmp_path)
    tape, _, _ = native_batch(capture_dir)
    case.assert_tripwire(observed_tripwire(tape, checkpoint))

    prediction, seconds = run_replay(capture_dir, checkpoint, stop_after_trunk=False)
    report = entity_rmsd(capture_dir, np.asarray(prediction.coordinates))
    entry = case.entry
    per_sample = report["entity_rmsd"]
    kept = [
        (entity, index, value)
        for entity, values in per_sample.items()
        for index, value in enumerate(values)
        if index not in entry.excluded_samples
    ]
    assert kept, "every sample was excluded; the entry asserts nothing"
    entity, index, worst = max(kept, key=lambda item: item[2])
    assert worst <= entry.tolerance_value, (
        f"OpenFold3 replay coordinates moved: entity {entity} sample {index} "
        f"{worst:.4f} A > {entry.tolerance_value:.4f} A "
        f"({entry.tolerance_set_from}); per entity "
        + json.dumps(report["entity_max_rmsd"])
        + f"; CPU calibration {entry.cpu_residual:.4f} A in "
        f"{entry.cpu_wall_seconds:.0f} s, this run {seconds:.0f} s"
    )
