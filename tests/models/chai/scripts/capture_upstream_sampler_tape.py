"""Record upstream Chai-1's end-to-end sampler tape and the structure it made.

Chai already had two parity routes and neither one matches the diffusion noise
on a real job:

* `benchmark_sampler_parity.py` matches the tape, but drives the sampler and
  the diffusion module on a *synthetic* context. Its number answers "does the
  EDM loop agree given identical inputs", not "does a prediction agree".
* `export_scientific_parity.py` runs the real end-to-end job, but its own
  metadata says the sampler randomness is only `seed-and-distribution`
  matched: Torch's Philox and JAX's Threefry draw different streams from the
  same seed, so the two structures are different samples and can only be
  compared statistically.
* `reference_augmentation_tape.py` covers one thing, and says so in its
  docstring: the rigid augmentation applied to non-standard residue reference
  conformers during featurization. Not the sampler.

This script closes that gap. It runs one real upstream prediction through
`chai_lab`'s own entry points and intercepts every random draw the diffusion
sampler makes, so `run_matched_tape_parity.py` can replay exactly those numbers
into the JAX port and the two structures become directly comparable.

Chai's sampler (`chai_lab/chai1.py`, `run_folding_on_context`) draws:

* `sigmas[0] * torch.randn(b * ds, n_atom, 3)` once, for the initial positions.
  Recorded unscaled, because the JAX side multiplies by `sigmas[0]` itself.
* inside `center_random_augmentation`, once per step: `random_rotations(b)` and
  a `torch.randn_like(centroid)` translation of shape `[b, 1, 3]`.
* `torch.randn(atom_pos.shape)` once per step, the churn noise. Recorded
  unscaled: upstream folds `S_noise = 1.003` into the draw, the JAX port
  multiplies by 1.003 inside its step, so the raw normal is the portable form.

Two traps are handled explicitly.

The model draws random numbers outside the sampler -- ESM, MSA handling and the
reference-conformer augmentation all run before it -- so a blanket `torch.randn`
hook records garbage. Recording is gated on the immediate caller being
`run_folding_on_context`'s own code object, which is exactly "the sampler is on
the stack" and nothing looser.

The initial draw and the per-step churn draw have the *same* shape, so shape
cannot separate them the way it can for OpenDDE. They are separated by order
instead: the initial draw is the first one the sampler makes, every later one
is a churn draw, and the count is checked against the schedule length.

Writes `tape.npz` (what the replay reads), `reference.npz` (the coordinates to
compare against) and `trunk.npz` (the trunk representations handed to the
diffusion module, so a coordinate mismatch can be attributed to the trunk or to
the sampler). Run it with the upstream checkout's own virtualenv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

#: Trunk outputs handed to the diffusion module. Recording these makes a
#: coordinate mismatch attributable: if they already disagree the sampler is
#: not the thing to look at.
TRUNK_KEYS = (
    "token_single_initial_repr",
    "token_pair_initial_repr",
    "token_single_trunk_repr",
    "token_pair_trunk_repr",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--msa-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--recycles", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--no-esm",
        action="store_true",
        help="disable the ESM embedding feature on both sides",
    )
    parser.add_argument(
        "--reference-augmentation-tape",
        type=Path,
        help="tape for the non-standard-residue conformer augmentation; only "
        "needed when the job actually has non-standard residues",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[5] / "chai",
        help="upstream Chai checkout; defaults to a sibling of this repo",
    )
    return parser.parse_args()


def _numpy(value: object) -> np.ndarray:
    return np.asarray(value.detach().float().cpu().numpy(), dtype=np.float32)


def _drop_batch(array: np.ndarray, rank: int) -> np.ndarray:
    """Strip leading singleton axes down to ``rank``, as float32.

    Upstream carries axes the JAX port does not. The axes being dropped are
    always size one, so this loses nothing, and it fails loudly rather than
    reshaping when that stops being true.
    """
    while array.ndim > rank and array.shape[0] == 1:
        array = array[0]
    if array.ndim != rank:
        raise RuntimeError(f"cannot reduce shape {array.shape} to rank {rank}")
    return np.ascontiguousarray(array, dtype=np.float32)


class SamplerTape:
    """Classify and record every random draw Chai's diffusion sampler makes."""

    def __init__(self) -> None:
        self.sampler_code: object = None
        self.in_augment = False
        self.coordinate_draws: list[np.ndarray] = []
        self.rotations: list[np.ndarray] = []
        self.translations: list[np.ndarray] = []
        self.final_coordinates: list[np.ndarray] = []
        self.trunk: dict[str, np.ndarray] = {}
        self.sigmas: np.ndarray | None = None
        self.conformer_augmentations = 0
        self.esm: dict[str, np.ndarray] = {}

    def in_sampler(self, depth: int = 2) -> bool:
        """True when the direct caller at ``depth`` is the sampler itself."""
        frame = sys._getframe(depth)
        return frame.f_code is self.sampler_code

    def randn(self, original, *args, **kwargs):
        value = original(*args, **kwargs)
        if self.sampler_code is not None and self.in_sampler(3):
            self.coordinate_draws.append(_numpy(value))
        return value

    def randn_like(self, original, *args, **kwargs):
        value = original(*args, **kwargs)
        if self.in_augment:
            self.translations.append(_numpy(value))
        return value


def _install(chai1, utils, residue, esm, tape: SamplerTape):
    """Patch every draw site, returning a callable that restores them all."""

    target = chai1.run_folding_on_context
    tape.sampler_code = getattr(target, "__wrapped__", target).__code__

    original_randn = torch.randn
    original_randn_like = torch.randn_like
    original_augment = chai1.center_random_augmentation
    original_frames = chai1.get_frames_and_mask
    original_schedule = chai1.InferenceNoiseSchedule.get_schedule
    original_forward = chai1.ModuleWrapper.forward
    original_conformer_augment = residue.center_random_augmentation
    original_esm = esm._get_esm_contexts_for_sequences

    def randn(*a, **k):
        return tape.randn(original_randn, *a, **k)

    def randn_like(*a, **k):
        return tape.randn_like(original_randn_like, *a, **k)

    def augment(atom_coords, atom_single_mask, s_trans=1.0, rotations=None):
        # `center_random_augmentation` draws the rotation itself when it is not
        # given one, then draws the translation. Drawing it here first and
        # handing it in keeps the RNG stream in exactly the same order while
        # making the matrix observable.
        if rotations is None:
            rotations = utils.random_rotations(
                atom_coords.shape[0], device=atom_coords.device
            )
        tape.rotations.append(_numpy(rotations))
        before = len(tape.translations)
        tape.in_augment = True
        try:
            result = original_augment(
                atom_coords, atom_single_mask, s_trans=s_trans, rotations=rotations
            )
        finally:
            tape.in_augment = False
        if len(tape.translations) != before + 1:
            raise RuntimeError(
                "expected exactly one translation draw per augmentation, saw "
                f"{len(tape.translations) - before}"
            )
        return result

    def frames(atom_pos, *a, **k):
        # The ranking loop is the first place the finished coordinates are
        # visible by name after the sampler returns.
        tape.final_coordinates.append(_numpy(atom_pos))
        return original_frames(atom_pos, *a, **k)

    def schedule(self, *a, **k):
        value = original_schedule(self, *a, **k)
        tape.sigmas = _numpy(value)
        return value

    def forward(self, *a, **k):
        # The trunk's own forward also takes `token_single_trunk_repr` (it
        # recycles it), so that name does not identify the diffusion module.
        # `atom_noised_coords` does, and only the diffusion module receives
        # the finished trunk representations.
        if not tape.trunk and "atom_noised_coords" in k:
            tape.trunk = {name: _numpy(k[name]) for name in TRUNK_KEYS}
        return original_forward(self, *a, **k)

    def conformer_augment(*a, **k):
        tape.conformer_augmentations += 1
        return original_conformer_augment(*a, **k)

    def esm_contexts(*a, **k):
        # Record the embeddings the real run used, keyed by the sequence it
        # keyed them by, rather than recomputing them from the FASTA. The JAX
        # port has its own ESM2; handing it Torch's output removes a
        # divergence that has nothing to do with the fold model under test.
        value = original_esm(*a, **k)
        tape.esm = {
            sequence: np.ascontiguousarray(
                context.esm_embeddings.detach().float().cpu().numpy(), np.float32
            )
            for sequence, context in value.items()
        }
        return value

    torch.randn = randn
    torch.randn_like = randn_like
    chai1.center_random_augmentation = augment
    chai1.get_frames_and_mask = frames
    chai1.InferenceNoiseSchedule.get_schedule = schedule
    chai1.ModuleWrapper.forward = forward
    residue.center_random_augmentation = conformer_augment
    esm._get_esm_contexts_for_sequences = esm_contexts

    def restore() -> None:
        torch.randn = original_randn
        torch.randn_like = original_randn_like
        chai1.center_random_augmentation = original_augment
        chai1.get_frames_and_mask = original_frames
        chai1.InferenceNoiseSchedule.get_schedule = original_schedule
        chai1.ModuleWrapper.forward = original_forward
        residue.center_random_augmentation = original_conformer_augment
        esm._get_esm_contexts_for_sequences = original_esm

    return restore


def _write_esm(path: Path, embeddings: dict[str, np.ndarray], model_path: Path) -> None:
    """Write the recorded Torch ESM embeddings with their model provenance."""
    digest = hashlib.sha256()
    with model_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    ordered = sorted(embeddings)
    np.savez_compressed(
        path,
        manifest_json=np.asarray(
            json.dumps(
                {
                    "sequences": ordered,
                    "model_id": "esm2_t36_3B_UR50D",
                    "model_revision": "chai-traced-sdpa-fp16",
                    "source_sha256": digest.hexdigest(),
                }
            )
        ),
        **{
            f"embedding_{index:06d}": embeddings[sequence]
            for index, sequence in enumerate(ordered)
        },
    )


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.repo))

    import chai_lab.chai1 as chai1
    import chai_lab.data.dataset.embeddings.esm as esm
    import chai_lab.data.parsing.structure.residue as residue
    import chai_lab.model.utils as utils

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.out_dir / "upstream"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"refusing to reuse a populated run directory: {run_dir}")

    tape = SamplerTape()
    restore = _install(chai1, utils, residue, esm, tape)
    replay = None
    if args.reference_augmentation_tape is not None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from reference_augmentation_tape import replay_torch

        replay = replay_torch(args.reference_augmentation_tape)
    try:
        context = chai1.make_all_atom_feature_context(
            fasta_file=args.fasta,
            output_dir=run_dir,
            use_esm_embeddings=not args.no_esm,
            use_msa_server=False,
            msa_directory=args.msa_dir,
            constraint_path=None,
            use_templates_server=False,
            templates_path=None,
            esm_device=torch.device(args.device),
        )
        if replay is not None:
            replay.__enter__()
        chai1.run_folding_on_context(
            context,
            output_dir=run_dir,
            num_trunk_recycles=args.recycles,
            num_diffn_timesteps=args.steps,
            num_diffn_samples=args.samples,
            seed=args.seed,
            device=torch.device(args.device),
            low_memory=True,
        )
    finally:
        if replay is not None:
            replay.__exit__(None, None, None)
        restore()

    if tape.sigmas is None:
        raise RuntimeError("the diffusion noise schedule was never built")
    # Chai's schedule holds one sigma per requested timestep and the sampler
    # walks consecutive pairs, so `--steps 20` is 20 sigmas and 19 EDM steps.
    if len(tape.sigmas) != args.steps:
        raise RuntimeError(
            f"expected {args.steps} sigmas, schedule has {len(tape.sigmas)}"
        )
    steps = len(tape.sigmas) - 1
    if len(tape.coordinate_draws) != steps + 1:
        raise RuntimeError(
            f"expected {steps + 1} coordinate draws inside the sampler, got "
            f"{len(tape.coordinate_draws)}"
        )
    if len(tape.rotations) != steps or len(tape.translations) != steps:
        raise RuntimeError(
            f"expected {steps} augmentations, got {len(tape.rotations)} rotations "
            f"and {len(tape.translations)} translations"
        )
    if len(tape.final_coordinates) != args.samples:
        raise RuntimeError(
            f"expected {args.samples} final coordinate sets, got "
            f"{len(tape.final_coordinates)}"
        )
    if tape.conformer_augmentations and args.reference_augmentation_tape is None:
        raise RuntimeError(
            f"{tape.conformer_augmentations} reference-conformer augmentations ran "
            "unmatched; generate one with reference_augmentation_tape.py and pass "
            "--reference-augmentation-tape"
        )

    init = _drop_batch(tape.coordinate_draws[0], 3)
    atom_count = init.shape[1]
    if init.shape[0] != args.samples:
        raise RuntimeError(
            f"initial noise has shape {init.shape}, expected a leading sample "
            f"axis of {args.samples}"
        )
    np.savez_compressed(
        args.out_dir / "tape.npz",
        sigmas=np.asarray(tape.sigmas, dtype=np.float32),
        init_noise=init,
        step_noises=np.stack([_drop_batch(a, 3) for a in tape.coordinate_draws[1:]]),
        rotations=np.stack([_drop_batch(a, 3) for a in tape.rotations]),
        # `[b, 1, 3]` upstream; the port wants one vector per sample and
        # broadcasts it itself.
        translations=np.stack(
            [_drop_batch(np.squeeze(a, axis=-2), 2) for a in tape.translations]
        ),
    )
    np.savez_compressed(
        args.out_dir / "reference.npz",
        coordinate=np.concatenate(
            [_drop_batch(a, 3) for a in tape.final_coordinates], axis=0
        ),
    )
    if tape.trunk:
        np.savez_compressed(args.out_dir / "trunk.npz", **tape.trunk)
    if not args.no_esm:
        if not tape.esm:
            raise RuntimeError("ESM embeddings were requested but never computed")
        _write_esm(
            args.out_dir / "esm.npz",
            tape.esm,
            esm.downloads_path / "esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt",
        )

    summary = {
        "steps": steps,
        "samples": args.samples,
        "recycles": args.recycles,
        "seed": args.seed,
        "atom_count": int(atom_count),
        "use_esm_embeddings": not args.no_esm,
        "esm_sequences": sorted(tape.esm),
        "conformer_augmentations": tape.conformer_augmentations,
        "trunk_recorded": sorted(tape.trunk),
        "out": str(args.out_dir),
    }
    (args.out_dir / "tape.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
