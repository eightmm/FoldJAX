"""Record upstream Boltz-2's features, sampler tape, trunk, and coordinates.

The JAX sampler already accepts a tape -- ``boltz2_sample_forward`` takes
``init_noise``, ``step_noises`` and ``aug_transforms`` -- so it can draw exactly
the numbers the torch one drew. What was missing is the capture. Without it the
two implementations run on different random tapes and their structures can only
be compared statistically, which is a far weaker statement than "same input,
same output": a real defect can hide behind sample spread while every
confidence score still agrees.

Everything that makes an upstream Boltz-2 prediction non-deterministic is
intercepted here while `boltz predict` runs in this process:

* ``AtomDiffusion.sample``'s initial ``torch.randn(shape)``, recorded UNSCALED
  because ``sigmas[0] *`` is what the JAX side multiplies itself,
* ``compute_random_augmentation``'s rotation and translation, once per step,
* the churn ``torch.randn(shape)`` that lifts sigma to ``t_hat``, once per step,
  again unscaled,
* ``MSAModule.forward``'s ``torch.randperm(n_msa)[:k]`` row draw, once per
  trunk pass, but only when subsampling is actually on (see below).

Two ``torch.randn`` sites inside the sampler share a rank: the coordinate draws
are ``[multiplicity, n_atom, 3]`` and the translation draws are
``[multiplicity, 1, 3]``. They are told apart by the axis before the last --
the atom count for one, one for the other -- never by rank.

The features the model actually consumed are captured too, so the JAX side
runs the model on byte-identical input and the comparison isolates the model
from the featurizer.

On the MSA subsample: `boltz predict` declares ``--subsample_msa`` as a click
flag, and a click flag defaults to False no matter what the Python signature
says, so the SHIPPED CLI default is subsample_msa=False and MSAModule draws no
permutation at all. Pass ``--subsample-msa`` here to exercise the subsampling
path; the row indices are then captured per trunk pass so the JAX side can be
handed the same rows.

Runs under the upstream checkout's own virtualenv, e.g.

    ../boltz/.venv/bin/python tests/models/boltz2/scripts/capture_upstream_tape.py \
        --input ../foldjax-bench/jobs/t0128.json --out-dir /tmp/boltz2-tape
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

#: ``tests/models/boltz2/scripts/`` -> the workspace that holds both checkouts.
WORKSPACE = Path(__file__).resolve().parents[5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="a Boltz job YAML, or a FoldJAX common-schema JSON to translate",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-sample", type=int, default=1)
    parser.add_argument("--n-step", type=int, default=20)
    parser.add_argument("--n-cycle", type=int, default=1)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--precision",
        default="32",
        help=(
            "Lightning precision for the run. The shipped CLI uses bf16-mixed; "
            "32 makes the trunk fp32 so a coordinate comparison measures the "
            "port rather than bf16 rounding."
        ),
    )
    parser.add_argument(
        "--subsample-msa",
        action="store_true",
        help="pass --subsample_msa upstream and capture the per-pass rows",
    )
    parser.add_argument("--num-subsampled-msa", type=int, default=1024)
    parser.add_argument(
        "--kernels",
        action="store_true",
        help="leave upstream's fused triangle kernels on (off by default: they "
        "are written for bf16 and this capture defaults to fp32)",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=WORKSPACE / "boltz",
        help="upstream Boltz checkout; defaults to a sibling of this repo",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path.home() / ".boltz",
        help="upstream weight/molecule cache",
    )
    return parser.parse_args()


def _drop_batch(array: np.ndarray, rank: int) -> np.ndarray:
    """Strip leading singleton axes down to ``rank``, as float32.

    Upstream carries axes its JAX counterpart does not. The axes being dropped
    are always size one, so this loses nothing -- and it fails loudly rather
    than reshaping when that stops being true.
    """
    while array.ndim > rank and array.shape[0] == 1:
        array = array[0]
    if array.ndim != rank:
        raise RuntimeError(f"cannot reduce shape {array.shape} to rank {rank}")
    return array.astype(np.float32)


def _to_boltz_document(path: Path) -> dict:
    """Translate a FoldJAX common-schema job into Boltz's own YAML dialect.

    Mirrors ``foldjax.input._boltz`` for the entity kinds a parity job uses.
    FoldJAX is not importable from the upstream virtualenv, so the subset is
    reproduced here rather than imported, and anything outside it is rejected
    instead of guessed at.
    """
    job = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    sequences = []
    for entity in job["entities"]:
        kind = entity["type"]
        if kind not in ("protein", "dna", "rna", "ligand"):
            raise RuntimeError(f"unsupported entity type for this capture: {kind}")
        ids = entity.get("id")
        ids = [ids] if isinstance(ids, str) else list(ids)
        body: dict[str, object] = {"id": ids}
        if kind == "ligand":
            if entity.get("ccd"):
                body["ccd"] = str(entity["ccd"])
            else:
                body["smiles"] = str(entity["smiles"])
        else:
            body["sequence"] = str(entity["sequence"])
            if entity.get("modifications"):
                raise RuntimeError("modifications are outside this capture's subset")
            if entity.get("unpaired_msa"):
                msa = Path(str(entity["unpaired_msa"]))
                body["msa"] = str(msa if msa.is_absolute() else (base / msa).resolve())
            elif kind == "protein":
                body["msa"] = "empty"
        sequences.append({kind: body})
    if job.get("bonds"):
        raise RuntimeError("covalent bonds are outside this capture's subset")
    return {"version": 1, "sequences": sequences}


class TapeRecorder:
    """Classify and record every random draw an upstream prediction makes."""

    def __init__(self) -> None:
        self.draws: list[np.ndarray] = []
        self.rotations: list[np.ndarray] = []
        self.translations: list[np.ndarray] = []
        self.msa_rows: list[np.ndarray] = []
        # The model draws random numbers of its own outside the sampler --
        # weight init and the Fourier bases at construction time, for two -- and
        # those are not part of the tape. Record only while the sampler, or the
        # MSA module, is on the stack.
        self.sampler_active = False
        self.msa_active = False

    def randn(self, original, *args, **kwargs):
        value = original(*args, **kwargs)
        if self.sampler_active:
            array = value.detach().float().cpu().numpy()
            # The rotation quaternions are drawn here too and are trailing-4.
            if array.ndim >= 2 and array.shape[-1] == 3:
                self.draws.append(array)
        return value

    def randperm(self, original, *args, **kwargs):
        value = original(*args, **kwargs)
        if self.msa_active:
            self.msa_rows.append(value.detach().cpu().numpy().astype(np.int64))
        return value

    def augmentation(self, original, *args, **kwargs):
        rotation, translation = original(*args, **kwargs)
        if self.sampler_active:
            self.rotations.append(rotation.detach().float().cpu().numpy())
            # Recorded post-scaling: this is exactly the tensor upstream added,
            # including ``s_trans``, so the JAX side needs no scale of its own.
            self.translations.append(translation.detach().float().cpu().numpy())
        return rotation, translation

    def split(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Separate coordinate draws from translation draws, by atom axis.

        Both are trailing-3 tensors of the same rank, so rank does not tell
        them apart. What does is the axis before the last: for a coordinate
        draw it is the atom count, for a translation it is one.
        """
        widths = {array.shape[-2] for array in self.draws}
        if len(widths) != 2:
            raise RuntimeError(f"expected two draw widths, saw {sorted(widths)}")
        atoms = max(widths)
        coordinates = [a for a in self.draws if a.shape[-2] == atoms]
        translations = [a for a in self.draws if a.shape[-2] != atoms]
        return coordinates, translations


def _install_hooks(recorder: TapeRecorder, captured: dict) -> tuple[list, object]:
    """Patch every random and observable site.

    Returns the undo callables and the imported ``boltz.main`` module, so the
    caller neither imports it a second time nor reaches past the hooks.
    """

    import boltz.main as boltz_main
    from boltz.model.models import boltz2 as boltz2_model
    from boltz.model.modules import diffusionv2, trunkv2

    undo = []

    original_randn = torch.randn
    torch.randn = lambda *a, **k: recorder.randn(original_randn, *a, **k)
    undo.append(lambda: setattr(torch, "randn", original_randn))

    original_randperm = torch.randperm
    torch.randperm = lambda *a, **k: recorder.randperm(original_randperm, *a, **k)
    undo.append(lambda: setattr(torch, "randperm", original_randperm))

    # ``AtomDiffusion.sample`` resolves this through its own module globals, so
    # patching the name in ``diffusionv2`` reaches it.
    original_augmentation = diffusionv2.compute_random_augmentation
    diffusionv2.compute_random_augmentation = (
        lambda *a, **k: recorder.augmentation(original_augmentation, *a, **k)
    )
    undo.append(
        lambda: setattr(
            diffusionv2, "compute_random_augmentation", original_augmentation
        )
    )

    original_msa_forward = trunkv2.MSAModule.forward

    def msa_forward(self, *a, **k):
        recorder.msa_active = True
        try:
            return original_msa_forward(self, *a, **k)
        finally:
            recorder.msa_active = False

    trunkv2.MSAModule.forward = msa_forward
    undo.append(lambda: setattr(trunkv2.MSAModule, "forward", original_msa_forward))

    original_sample = diffusionv2.AtomDiffusion.sample
    sample_signature = inspect.signature(original_sample)

    def sample(self, *a, **k):
        recorder.sampler_active = True
        try:
            result = original_sample(self, *a, **k)
        finally:
            recorder.sampler_active = False
        bound = sample_signature.bind(self, *a, **k)
        bound.apply_defaults()
        steps = bound.arguments["num_sampling_steps"]
        captured["sigmas"] = (
            self.sample_schedule(steps).detach().float().cpu().numpy()
        )
        captured["step_scale"] = float(self.step_scale)
        captured["gamma_0"] = float(self.gamma_0)
        captured["gamma_min"] = float(self.gamma_min)
        captured["noise_scale"] = float(self.noise_scale)
        captured["sigma_data"] = float(self.sigma_data)
        return result

    diffusionv2.AtomDiffusion.sample = sample
    undo.append(lambda: setattr(diffusionv2.AtomDiffusion, "sample", original_sample))

    original_forward = boltz2_model.Boltz2.forward

    def forward(self, feats, *a, **k):
        if "feats" not in captured:
            captured["feats"] = {
                name: value.detach().cpu().numpy()
                for name, value in feats.items()
                if torch.is_tensor(value)
            }
        def record_s_inputs(_module, _inputs, output) -> None:
            # A forward hook that returns a value REPLACES the module output,
            # so this must return None and only record.
            captured.setdefault("s_inputs", output.detach().float().cpu().numpy())

        handle = self.input_embedder.register_forward_hook(record_s_inputs)
        try:
            out = original_forward(self, feats, *a, **k)
        finally:
            handle.remove()
        if "coordinate" not in captured:
            for name in ("s", "z", "sample_atom_coords"):
                captured[name] = out[name].detach().float().cpu().numpy()
            captured["coordinate"] = captured.pop("sample_atom_coords")
        return out

    boltz2_model.Boltz2.forward = forward
    undo.append(lambda: setattr(boltz2_model.Boltz2, "forward", original_forward))

    return undo, boltz_main


def _force_precision(boltz_main, precision: str) -> Callable[[], None]:
    """Pin the Lightning precision the CLI would otherwise choose itself."""

    original_trainer = boltz_main.Trainer

    class PinnedTrainer(original_trainer):
        def __init__(self, *a, **k):
            k["precision"] = precision
            super().__init__(*a, **k)

    boltz_main.Trainer = PinnedTrainer
    return lambda: setattr(boltz_main, "Trainer", original_trainer)


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.repo / "src"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.input.suffix.lower() in (".yaml", ".yml"):
        job_path = args.input
    else:
        import yaml

        job_path = args.out_dir / "job.yaml"
        job_path.write_text(
            yaml.safe_dump(_to_boltz_document(args.input), sort_keys=False),
            encoding="utf-8",
        )

    recorder = TapeRecorder()
    captured: dict[str, object] = {}
    undo, boltz_main = _install_hooks(recorder, captured)
    undo.append(_force_precision(boltz_main, args.precision))

    argv = [
        "predict",
        str(job_path),
        "--out_dir",
        str(args.out_dir / "upstream"),
        "--cache",
        str(args.cache),
        "--model",
        "boltz2",
        "--recycling_steps",
        str(args.n_cycle),
        "--sampling_steps",
        str(args.n_step),
        "--diffusion_samples",
        str(args.n_sample),
        "--seed",
        str(args.seed),
        "--accelerator",
        "gpu",
        "--devices",
        "1",
        "--num_workers",
        "0",
        "--output_format",
        "mmcif",
        "--override",
    ]
    if not args.kernels:
        argv.append("--no_kernels")
    if args.subsample_msa:
        argv += [
            "--subsample_msa",
            "--num_subsampled_msa",
            str(args.num_subsampled_msa),
        ]

    try:
        boltz_main.cli.main(args=argv, standalone_mode=False)
    finally:
        for revert in reversed(undo):
            revert()

    return _write(args, recorder, captured)


def _write(args, recorder: TapeRecorder, captured: dict) -> int:
    for name in ("coordinate", "feats", "sigmas", "s", "z", "s_inputs"):
        if name not in captured:
            raise RuntimeError(f"upstream never produced {name!r}")

    n_step = args.n_step
    if len(captured["sigmas"]) != n_step + 1:
        raise RuntimeError(
            f"expected {n_step + 1} sigmas, got {len(captured['sigmas'])}"
        )
    coordinate_draws, translation_draws = recorder.split()
    if len(coordinate_draws) != n_step + 1:
        raise RuntimeError(
            f"expected {n_step + 1} coordinate draws, got {len(coordinate_draws)}"
        )
    if len(recorder.rotations) != n_step or len(translation_draws) != n_step:
        raise RuntimeError(
            f"expected {n_step} augmentations, got {len(recorder.rotations)} "
            f"rotations and {len(translation_draws)} translation draws"
        )

    tape = {
        "sigmas": captured["sigmas"].astype(np.float32),
        "init_noise": _drop_batch(coordinate_draws[0], 3),
        "step_noises": np.stack([_drop_batch(a, 3) for a in coordinate_draws[1:]]),
        "rotations": np.stack([_drop_batch(a, 3) for a in recorder.rotations]),
        # The JAX sampler broadcasts a [multiplicity, 1, 3] translation, which
        # is the rank upstream already uses; keep it rather than squeezing.
        "translations": np.stack([_drop_batch(a, 3) for a in recorder.translations]),
    }
    np.savez_compressed(args.out_dir / "tape.npz", **tape)

    passes = args.n_cycle + 1
    rows = None
    if args.subsample_msa:
        if len(recorder.msa_rows) != passes:
            raise RuntimeError(
                f"expected {passes} MSA row draws, got {len(recorder.msa_rows)}"
            )
        rows = np.stack([draw[: args.num_subsampled_msa] for draw in recorder.msa_rows])
        np.save(args.out_dir / "msa_rows.npy", rows)
    elif recorder.msa_rows:
        raise RuntimeError(
            "upstream drew MSA permutations without --subsample-msa; the "
            "capture's assumption about the CLI default is wrong"
        )

    np.savez_compressed(args.out_dir / "features.npz", **captured["feats"])
    np.savez_compressed(
        args.out_dir / "trunk.npz",
        s=captured["s"],
        z=captured["z"],
        s_inputs=captured["s_inputs"],
    )
    np.savez_compressed(
        args.out_dir / "coordinate.npz",
        coordinate=captured["coordinate"].astype(np.float32),
    )

    summary = {
        "n_step": n_step,
        "n_sample": args.n_sample,
        "n_cycle": args.n_cycle,
        "seed": args.seed,
        "precision": args.precision,
        "kernels": bool(args.kernels),
        "subsample_msa": bool(args.subsample_msa),
        "num_subsampled_msa": args.num_subsampled_msa if args.subsample_msa else None,
        "msa_rows_shape": None if rows is None else list(rows.shape),
        "msa_depth": int(captured["feats"]["msa"].shape[1]),
        "n_token": int(captured["feats"]["token_pad_mask"].shape[1]),
        "n_atom": int(captured["feats"]["atom_pad_mask"].shape[1]),
        "coordinate_shape": list(captured["coordinate"].shape),
        "step_scale": captured["step_scale"],
        "gamma_0": captured["gamma_0"],
        "gamma_min": captured["gamma_min"],
        "noise_scale": captured["noise_scale"],
        "sigma_data": captured["sigma_data"],
        "out": str(args.out_dir),
    }
    (args.out_dir / "tape.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
