"""Record upstream OpenDDE's sampler tape and the coordinates it produced.

`run_official_smoke.py` can already *replay* a tape -- init noise, per-step
noise, and the per-step rigid augmentation -- so the JAX sampler draws exactly
the numbers the torch one drew. What was missing is the capture. Without it the
two implementations run on different random tapes and their structures can only
be compared statistically, against each implementation's own sample spread,
which is a far weaker statement than "same input, same output".

Every random draw in `opendde.model.generator.sample_diffusion` is intercepted
here:

* the initial `torch.randn` scaled by `noise_schedule[0]`, recorded unscaled
  because that is the form the JAX side multiplies itself,
* `uniform_random_rotation` and the translation `torch.randn` inside
  `centre_random_augmentation`, once per step,
* the churn `torch.randn` added to reach `t_hat`, once per step.

The two `torch.randn` sites are told apart by shape: the coordinate draws carry
an atom axis, the translation draws do not.

Writes `tape.npz` (the arrays `run_official_smoke.py --random-tape` expects) and
`coordinate.npy` (the reference structure to compare against).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-sample", type=int, default=1)
    parser.add_argument("--n-step", type=int, default=20)
    parser.add_argument("--n-cycle", type=int, default=1)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[5] / "OpenDDE",
        help="upstream OpenDDE checkout; defaults to a sibling of this repo",
    )
    return parser.parse_args()


def _drop_batch(array: np.ndarray, rank: int) -> np.ndarray:
    """Strip leading singleton axes down to `rank`, as float32.

    Upstream carries a batch axis its JAX counterpart does not: a translation
    is `[1, 1, 3]` there and `[1, 3]` here. The axes being dropped are always
    size one, so this loses nothing -- and it fails loudly rather than
    reshaping if that stops being true.
    """
    while array.ndim > rank and array.shape[0] == 1:
        array = array[0]
    if array.ndim != rank:
        raise RuntimeError(f"cannot reduce shape {array.shape} to rank {rank}")
    return array.astype(np.float32)


class TapeRecorder:
    """Classify and record every random draw the sampler makes."""

    def __init__(self) -> None:
        self.draws: list[np.ndarray] = []
        self.rotations: list[np.ndarray] = []
        # The model draws random numbers of its own outside the sampler --
        # `embedders.py` builds a random Fourier basis at construction time, for
        # one -- and those are not part of the tape. Record only while
        # `sample_diffusion` is on the stack.
        self.active = False

    def randn(self, original, *args, **kwargs):
        value = original(*args, **kwargs)
        if self.active:
            array = value.detach().float().cpu().numpy()
            if array.ndim >= 2 and array.shape[-1] == 3:
                self.draws.append(array)
        return value

    def split(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Separate coordinate draws from translation draws, by atom axis.

        Both are trailing-3 tensors and both carry the batch and sample axes,
        so rank does not tell them apart -- with a batch axis present a
        translation is `[1, 1, 3]`, rank 3, exactly like a coordinate tensor
        that has been squeezed. What does separate them is the axis before the
        last: for a coordinate draw it is the atom count, for a translation it
        is the sample count, and the atom count is larger in every real job.
        """
        widths = {array.shape[-2] for array in self.draws}
        if len(widths) != 2:
            raise RuntimeError(
                f"expected two distinct draw widths, saw {sorted(widths)}"
            )
        atoms = max(widths)
        coordinates = [a for a in self.draws if a.shape[-2] == atoms]
        translations = [a for a in self.draws if a.shape[-2] != atoms]
        return coordinates, translations

    def rotation(self, original, *args, **kwargs):
        value = original(*args, **kwargs)
        if self.active:
            self.rotations.append(value.detach().float().cpu().numpy())
        return value


def main() -> int:
    args = parse_args()
    os.environ.setdefault("PYTHONPATH", str(args.repo))
    import sys

    sys.path.insert(0, str(args.repo))

    from opendde.model import generator, utils

    recorder = TapeRecorder()
    original_randn = torch.randn
    original_rotation = utils.uniform_random_rotation

    def randn(*a, **k):
        return recorder.randn(original_randn, *a, **k)

    def rotation(*a, **k):
        return recorder.rotation(original_rotation, *a, **k)

    # `centre_random_augmentation` resolves `uniform_random_rotation` through
    # its own module globals, so patching the name in `utils` reaches it even
    # though `generator` imported the augmentation itself by value.
    torch.randn = randn
    utils.uniform_random_rotation = rotation

    captured: dict[str, object] = {}
    original_sample = generator.sample_diffusion

    import inspect

    signature = inspect.signature(original_sample)

    def sample_diffusion(*a, **k):
        recorder.active = True
        try:
            result = original_sample(*a, **k)
        finally:
            recorder.active = False
        # `noise_schedule` reaches this function positionally from the model's
        # own wrapper and by keyword from other callers; bind rather than guess.
        bound = signature.bind(*a, **k)
        schedule = bound.arguments["noise_schedule"]
        captured["schedule"] = schedule.detach().float().cpu().numpy()
        captured["coordinate"] = result.detach().float().cpu().numpy()
        return result

    generator.sample_diffusion = sample_diffusion
    try:
        _run(args)
    finally:
        torch.randn = original_randn
        utils.uniform_random_rotation = original_rotation
        generator.sample_diffusion = original_sample

    if "coordinate" not in captured:
        raise RuntimeError("sample_diffusion was never called")
    coordinate = captured["coordinate"]
    schedule = captured["schedule"]

    n_step = len(schedule) - 1
    coordinate_draws, translations = recorder.split()
    if len(coordinate_draws) != n_step + 1:
        raise RuntimeError(
            f"expected {n_step + 1} coordinate draws, got {len(coordinate_draws)}"
        )
    if len(recorder.rotations) != n_step or len(translations) != n_step:
        raise RuntimeError(
            f"expected {n_step} augmentations, got {len(recorder.rotations)} "
            f"rotations and {len(translations)} translations"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "tape.npz",
        noise_schedule=np.asarray(schedule, dtype=np.float32),
        init_noise=_drop_batch(coordinate_draws[0], 3),
        step_noises=np.stack([_drop_batch(a, 3) for a in coordinate_draws[1:]]),
        rotations=np.stack([_drop_batch(a, 3) for a in recorder.rotations]),
        translations=np.stack([_drop_batch(a, 2) for a in translations]),
    )
    # `run_official_smoke.py --reference-coordinates` reads an archive keyed
    # `coordinate`, so write that form rather than a bare array.
    np.savez_compressed(
        args.out_dir / "coordinate.npz",
        coordinate=_drop_batch(coordinate, 3)
        if coordinate.ndim > 3
        else coordinate.astype(np.float32),
    )
    (args.out_dir / "tape.json").write_text(
        json.dumps(
            {
                "n_step": n_step,
                "n_sample": args.n_sample,
                "n_cycle": args.n_cycle,
                "seed": args.seed,
                "coordinate_shape": list(coordinate.shape),
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "steps": n_step,
                "coordinate_draws": len(coordinate_draws),
                "rotations": len(recorder.rotations),
                "translations": len(translations),
                "coordinate_shape": list(coordinate.shape),
                "out": str(args.out_dir),
            },
            indent=2,
        )
    )
    return 0


#: The upstream runner is a config-driven CLI; this is the same argv
#: `bench/run_upstream.py` drives OpenDDE with, so the capture and the
#: benchmark are running the same program.
def _run(args) -> None:
    """Drive one upstream prediction in-process, with the hooks installed."""

    import sys

    home = Path.home()
    argv = [
        "runner.inference",
        "--input_json_path",
        str(args.input_json),
        "--dump_dir",
        str(args.out_dir / "upstream"),
        "--seeds",
        str(args.seed),
        "--model.N_cycle",
        str(args.n_cycle),
        "--sample_diffusion.N_step",
        str(args.n_step),
        "--sample_diffusion.N_sample",
        str(args.n_sample),
        "--data.ccd_components_file",
        str(home / ".cache/foldjax/assets/components.cif"),
        "--data.ccd_components_rdkit_mol_file",
        str(home / ".cache/foldjax/assets/components.cif.rdkit_mol.pkl"),
        "--load_checkpoint_path",
        str(args.checkpoint),
    ]
    from runner.inference import run

    saved = sys.argv
    sys.argv = argv
    try:
        run()
    finally:
        sys.argv = saved


if __name__ == "__main__":
    raise SystemExit(main())
