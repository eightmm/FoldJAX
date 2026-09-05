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

The sampler is not the only place OpenDDE draws random numbers, and matching it
alone is not enough. `MSAModule._prepare_msa_sample` calls
`subsample_msa_feature_dict_valid_first` on EVERY trunk pass, and that function
says in its own docstring that "each call re-samples the order"; with
`msa_depth = 1280` and an alignment deeper than that, each pass reads a
different 1,280 rows. A port that reads the whole alignment is then not running
the same model on the same input, and no amount of matched sampler noise can
make the coordinates agree. So the row draw is captured too, per pass, both as
indices and as the selected feature rows, and the indices are verified against
the rows the sampler actually returned rather than assumed.

The residue trunk's per-stage state is recorded as well -- `s_inputs`, `z_init`,
and `z` leaving the recycle projection, the template embedder, the MSA module
and the Pairformer. Every one of those stages is shape-preserving in `z`, so a
whole-trunk comparison can only say that something is wrong; the first stage
that drifts names the module.

Writes, into `--out-dir`:

    tape.npz         sampler tape, as `run_official_smoke.py --random-tape` wants
    coordinate.npz   the reference structure
    msa.npz          the alignment upstream read, and the rows each pass drew
    trunk.npz        the residue trunk's per-stage tensors
    tape.json        shapes and counts, for the parity harness to check against
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--num-samples", "--n-sample", dest="num_samples", type=int, default=1
    )
    parser.add_argument(
        "--num-steps", "--n-step", dest="num_steps", type=int, default=20
    )
    parser.add_argument(
        "--num-recycles", "--n-cycle", dest="num_recycles", type=int, default=1
    )
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[5] / "OpenDDE",
        help="upstream OpenDDE checkout; defaults to a sibling of this repo",
    )
    parser.add_argument(
        "--dtype",
        choices=("fp32", "bf16"),
        default="fp32",
        help=(
            "upstream's inference precision. `fp32` is its own default "
            "(`opendde/config/model_base.py:37`); `bf16` exists so the port's "
            "bf16 trunk can be compared against a bf16 reference rather than "
            "against an fp32 one, which is the mismatched comparison this "
            "project reported for a while"
        ),
    )
    parser.add_argument(
        "--skip-trunk-stages",
        action="store_true",
        help=(
            "record the sampler tape and the structure, but not the residue "
            "trunk's per-stage tensors. Those are ~26 GiB and go through "
            "`savez_compressed` on one core, which costs over an hour; a "
            "coordinate comparison does not read them. The replay skips any "
            "stage absent from the tape, so this yields the RMSD and an empty "
            "stage table rather than an error"
        ),
    )
    parser.add_argument(
        "--disable-tf32",
        action="store_true",
        help=(
            "capture with `enable_tf32 false`, which upstream does NOT default "
            "to (`opendde/config/inference_defaults.py:24`). This is a control, "
            "not a mode: upstream's fp32 matmuls carry ten mantissa bits, worth "
            "2.941e-04 relative on this card, and the port's residual against "
            "it is a per-block ~3e-04 that raising the port's own precision to "
            "HIGHEST does not remove. If that residual is upstream's own "
            "rounding, turning it off is what makes it disappear -- and if it "
            "does not disappear, the port has a real defect to find"
        ),
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


def _translation_samples(array: np.ndarray) -> np.ndarray:
    """Normalize one upstream translation draw to ``[samples, 3]``."""
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[-2:] != (1, 3):
        raise RuntimeError(
            "translation must have shape [samples, 1, 3] after optional "
            f"batch axes, got {array.shape}"
        )
    return array[..., 0, :].astype(np.float32)


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


class MSARowRecorder:
    """Record which alignment rows each trunk pass drew, and check them.

    The upstream sampler returns the selected features but not the indices that
    produced them, and the indices are what a port needs in order to read the
    same alignment. They are reconstructed here from the `torch.randperm` draws
    the function made -- and then verified by re-selecting with them and
    comparing against what the function actually returned. A reconstruction
    that is merely plausible would be worse than none: it would put a
    confident, wrong alignment into the comparison.
    """

    def __init__(self) -> None:
        self.rows: list[np.ndarray] = []
        self.selected: list[dict[str, np.ndarray]] = []
        self.source: dict[str, np.ndarray] | None = None
        self.permutations: list[torch.Tensor] = []
        self.active = False

    def randperm(self, original, *args, **kwargs):
        value = original(*args, **kwargs)
        if self.active:
            self.permutations.append(value.detach().clone())
        return value

    def subsample(self, original, collapse, *args, **kwargs):
        import inspect

        bound = inspect.signature(original).bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments

        self.permutations.clear()
        self.active = True
        try:
            result = original(*args, **kwargs)
        finally:
            self.active = False

        indices = self._reconstruct(collapse, arguments)
        feat_dict = arguments["feat_dict"]
        dim_dict = arguments["dim_dict"]
        for name, dim in dim_dict.items():
            expected = torch.index_select(feat_dict[name], dim=dim, index=indices)
            if not torch.equal(expected, result[name]):
                raise RuntimeError(
                    f"reconstructed MSA rows do not reproduce upstream's {name!r}; "
                    "the row capture cannot be trusted"
                )
        if self.source is None:
            self.source = {
                name: feat_dict[name].detach().cpu().numpy()
                for name in (*dim_dict, "msa_mask")
                if name in feat_dict
            }
        self.rows.append(indices.detach().cpu().numpy().astype(np.int64))
        self.selected.append(
            {name: result[name].detach().cpu().numpy() for name in dim_dict}
        )
        return result

    def _reconstruct(self, collapse, arguments: dict) -> torch.Tensor:
        """Replay the selection upstream just made, from its own permutations.

        This mirrors `subsample_msa_feature_dict_valid_first` exactly; the
        verification in `subsample` is what makes mirroring safe, because a
        divergence in either the validity rule or the take order shows up as a
        mismatch against the returned rows rather than as a silent skew.
        """
        feat_dict = arguments["feat_dict"]
        msa = feat_dict["msa"]
        msa_dim = arguments["dim_dict"]["msa"]
        msa_len = msa.size(dim=msa_dim)
        device = msa.device
        num_msa = max(0, min(int(arguments["num_msa"]), msa_len))
        if num_msa == 0:
            return torch.empty(0, dtype=torch.long, device=device)

        msa_mask = arguments.get("msa_mask")
        gap_token = arguments.get("gap_token")
        row_valid = None
        if msa_mask is not None:
            row_valid = collapse(msa_mask.bool().any(dim=-1))
        if gap_token is not None and (row_valid is None or torch.all(row_valid)):
            row_valid = collapse((msa != gap_token).any(dim=-1))
        if row_valid is None:
            row_valid = torch.ones(msa_len, dtype=torch.bool, device=device)

        valid_idx = row_valid.nonzero(as_tuple=False).squeeze(-1)
        invalid_idx = (~row_valid).nonzero(as_tuple=False).squeeze(-1)
        selected = []
        drawn = iter(self.permutations)
        take_valid = min(valid_idx.numel(), num_msa)
        if take_valid > 0:
            selected.append(valid_idx[next(drawn)][:take_valid])
        take_invalid = num_msa - take_valid
        if take_invalid > 0 and invalid_idx.numel() > 0:
            selected.append(invalid_idx[next(drawn)][:take_invalid])
        if selected:
            return torch.cat(selected, dim=0)
        return torch.empty(0, dtype=torch.long, device=device)


class StageRecorder:
    """Keep the last value each named residue-trunk stage carried.

    Only the last recycle is kept, because that is the state the rest of the
    model consumes; earlier cycles differ from it only through the carry, which
    the last cycle already reflects.
    """

    def __init__(self) -> None:
        self.stages: dict[str, np.ndarray] = {}

    def record(self, name: str, tensor: torch.Tensor) -> None:
        self.stages[name] = tensor.detach().float().cpu().numpy()

    def install(self, model) -> list:
        """Hook the four residue-trunk submodules, returning removable handles.

        `template_embedder` runs only when it has blocks, so `after_template` is
        taken from what the MSA module was handed rather than from the template
        embedder's own output -- that way the stage exists, and means the same
        thing, whether or not templates are configured.
        """
        handles = []

        def on_embedder(_module, _inputs, output):
            self.record("s_inputs", output)

        def on_z_cycle(_module, _inputs, output):
            self.record("recycle_delta_z", output)

        def on_template(_module, inputs, _output):
            self.record("after_recycle_z", inputs[1])

        def on_msa(_module, inputs, output):
            self.record("after_template", inputs[1])
            self.record("after_msa", output)

        def on_pairformer(_module, inputs, output):
            self.record("after_recycle_s", inputs[0])
            self.record("after_pairformer_s", output[0])
            self.record("after_pairformer_z", output[1])

        handles.append(model.input_embedder.register_forward_hook(on_embedder))
        handles.append(model.linear_no_bias_z_cycle.register_forward_hook(on_z_cycle))
        if getattr(model.template_embedder, "n_blocks", 0) > 0:
            handles.append(model.template_embedder.register_forward_hook(on_template))
        handles.append(model.msa_module.register_forward_hook(on_msa))
        handles.append(model.pairformer_stack.register_forward_hook(on_pairformer))

        # Pairformer block 0, in and out. The stack-level stage says z lost
        # four digits while the s beside it kept them, but not which block or
        # which operation. Block 0 is where to start, and recording its INPUT
        # as well as its output is what makes the reading interpretable: a
        # probe whose input already differs measures nothing about the block.
        block = model.pairformer_stack.blocks[0]

        def on_block0(_module, inputs, output):
            self.record("block0_input_z", inputs[1])
            self.record("block0_out_s", output[0])
            self.record("block0_out_z", output[1])

        handles.append(block.register_forward_hook(on_block0))

        # Inside the MSA module. `after_msa` is where the first digits go: every
        # stage before it agrees to ten, and the Pairformer only amplifies what
        # it is handed -- block 0 turns a 0.00081 relative error into 0.00087,
        # and forty-eight of those reach 0.01299. So the MSA module's blocks are
        # where a real difference can still enter.
        #
        # The module's input z is recorded too, for the reason block 0 recorded
        # its own: a comparison whose starting points differ measures the
        # starting points, which is how the previous probe returned correlations
        # of 0.067 for operations that were fine.
        for index, msa_block in enumerate(model.msa_module.blocks):

            def on_msa_block(_module, inputs, output, index=index):
                if index == 0 and len(inputs) > 1:
                    self.record("msa_input_z", inputs[1])
                pair = output[1] if isinstance(output, (tuple, list)) else output
                self.record(f"msa_block{index}_z", pair)

            handles.append(msa_block.register_forward_hook(on_msa_block))

            # Inside each block. The block-level stage says z moved; it does
            # not say through which of the two paths that can move it. OpenDDE
            # runs `m = msa_stack(m, z)`, then `z = z + outer_product_mean(m)`,
            # then `z = pair_stack(z)`, so recording m on both sides of the
            # stack and z between the mean and the pair stack separates three
            # possibilities that the block-level number cannot: m arrived
            # wrong, the stack made it wrong, or the pair stack amplified z.
            #
            # `m` has never been measured on either side. Every stage recorded
            # so far is z, and z enters this module agreeing to five digits at
            # both 132 and 970 tokens -- so if the module is where the size
            # dependence lives, the evidence is in m.
            # A *pre*-hook for the input, because `MSAStack.inference_forward`
            # writes through it: `m[start:end, :, :] += msa_update`, then
            # returns the same tensor. A forward hook fires afterwards, so its
            # `inputs[0]` is the mutated buffer -- the output under the name of
            # the input. That is not a subtle failure: it reported the module's
            # input as correlating 0.41 with the port's while the output it
            # produced from that input correlated 0.99999999.
            def on_msa_stack_in(_module, inputs, index=index):
                self.record(f"msa_block{index}_m_in", inputs[0])

            def on_msa_stack_out(_module, _inputs, output, index=index):
                self.record(f"msa_block{index}_m_out", output)

            handles.append(
                msa_block.msa_stack.register_forward_pre_hook(on_msa_stack_in)
            )
            handles.append(
                msa_block.msa_stack.register_forward_hook(on_msa_stack_out)
            )

            # `pair_stack` is called entirely by keyword, so a plain forward
            # hook sees an empty `inputs` tuple and would record nothing.
            def on_pair_stack(_module, args, kwargs, _output, index=index):
                pair = kwargs.get("z")
                if pair is None and len(args) > 1:
                    pair = args[1]
                if pair is not None:
                    self.record(f"msa_block{index}_z_after_opm", pair)

            handles.append(
                msa_block.pair_stack.register_forward_hook(
                    on_pair_stack, with_kwargs=True
                )
            )
        return handles

    def finish(self) -> dict[str, np.ndarray]:
        stages = dict(self.stages)
        stages.setdefault("after_recycle_z", stages.get("after_template"))
        # `z_init` is not a module output anywhere, but the first thing every
        # cycle does is `z_init + linear_z_cycle(layernorm_z_cycle(z))`, so it
        # is exactly the recycled state minus that projection. Recovering it
        # this way needs no reimplementation of the embedding arithmetic.
        if "after_recycle_z" in stages and "recycle_delta_z" in stages:
            stages["z_init"] = stages["after_recycle_z"] - stages["recycle_delta_z"]
        return {name: value for name, value in stages.items() if value is not None}


def main() -> int:
    global torch

    import torch

    args = parse_args()
    os.environ.setdefault("PYTHONPATH", str(args.repo))
    import sys

    sys.path.insert(0, str(args.repo))

    from opendde.model import generator, msa_sampling, utils
    from opendde.model import opendde as opendde_model
    from opendde.model.modules import pairformer as pairformer_module

    recorder = TapeRecorder()
    msa_recorder = MSARowRecorder()
    stage_recorder = StageRecorder()
    original_randn = torch.randn
    original_randperm = torch.randperm
    original_rotation = utils.uniform_random_rotation

    def randn(*a, **k):
        return recorder.randn(original_randn, *a, **k)

    def randperm(*a, **k):
        return msa_recorder.randperm(original_randperm, *a, **k)

    def rotation(*a, **k):
        return recorder.rotation(original_rotation, *a, **k)

    # `centre_random_augmentation` resolves `uniform_random_rotation` through
    # its own module globals, so patching the name in `utils` reaches it even
    # though `generator` imported the augmentation itself by value.
    torch.randn = randn
    torch.randperm = randperm
    utils.uniform_random_rotation = rotation

    # `pairformer` imported the sampler by value, so the name has to be replaced
    # in the module that calls it, not in the one that defines it.
    original_subsample = pairformer_module.subsample_msa_feature_dict_valid_first
    collapse = msa_sampling._collapse_msa_row_mask

    def subsample(*a, **k):
        return msa_recorder.subsample(original_subsample, collapse, *a, **k)

    pairformer_module.subsample_msa_feature_dict_valid_first = subsample

    # Hooking the trunk's submodules only while `get_pairformer_output` is on
    # the stack keeps the structural branch's own Pairformer out of the record:
    # it is a different instance, but the same class, so a class-level patch
    # would conflate the two.
    original_pairformer_output = opendde_model.OpenDDE.get_pairformer_output

    def get_pairformer_output(self, *a, **k):
        handles = stage_recorder.install(self)
        try:
            return original_pairformer_output(self, *a, **k)
        finally:
            for handle in handles:
                handle.remove()

    opendde_model.OpenDDE.get_pairformer_output = get_pairformer_output

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

    # `opendde/model/opendde.py` does `from opendde.model.generator import
    # sample_diffusion`, so the name it calls is its own module global, bound at
    # import time. Replacing only `generator.sample_diffusion` leaves that
    # binding untouched and the hook never fires -- which is exactly what this
    # script did until now, and why it could raise "sample_diffusion was never
    # called" on a run that completed successfully. Patch the caller's name too.
    generator.sample_diffusion = sample_diffusion
    opendde_model.sample_diffusion = sample_diffusion
    try:
        # Nothing here is trained, and a forward hook keeps its inputs
        # reachable, so without this every intermediate the probes touch stays
        # alive on the autograd graph. Upstream already peaks near 58 GiB on a
        # 970-token job; adding per-block hooks then asked for another 40.8 GiB
        # and the capture died before it reached the sampler.
        #
        # `inference_mode` rather than `no_grad`: it also drops version
        # counters and view tracking, and nothing downstream ever
        # differentiates or mutates these tensors in place.
        with torch.inference_mode():
            _run(args)
    finally:
        torch.randn = original_randn
        torch.randperm = original_randperm
        utils.uniform_random_rotation = original_rotation
        generator.sample_diffusion = original_sample
        opendde_model.sample_diffusion = original_sample
        pairformer_module.subsample_msa_feature_dict_valid_first = original_subsample
        opendde_model.OpenDDE.get_pairformer_output = original_pairformer_output

    if "coordinate" not in captured:
        raise RuntimeError("sample_diffusion was never called")
    coordinate = captured["coordinate"]
    schedule = captured["schedule"]

    num_steps = len(schedule) - 1
    coordinate_draws, translations = recorder.split()
    if len(coordinate_draws) != num_steps + 1:
        raise RuntimeError(
            f"expected {num_steps + 1} coordinate draws, got {len(coordinate_draws)}"
        )
    if len(recorder.rotations) != num_steps or len(translations) != num_steps:
        raise RuntimeError(
            f"expected {num_steps} augmentations, got {len(recorder.rotations)} "
            f"rotations and {len(translations)} translations"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "tape.npz",
        noise_schedule=np.asarray(schedule, dtype=np.float32),
        init_noise=_drop_batch(coordinate_draws[0], 3),
        step_noises=np.stack([_drop_batch(a, 3) for a in coordinate_draws[1:]]),
        rotations=np.stack([_drop_batch(a, 3) for a in recorder.rotations]),
        translations=np.stack([_translation_samples(a) for a in translations]),
    )
    # `run_official_smoke.py --reference-coordinates` reads an archive keyed
    # `coordinate`, so write that form rather than a bare array.
    np.savez_compressed(
        args.out_dir / "coordinate.npz",
        coordinate=_drop_batch(coordinate, 3)
        if coordinate.ndim > 3
        else coordinate.astype(np.float32),
    )
    msa_summary = _write_msa(args.out_dir, msa_recorder, num_recycles=args.num_recycles)
    stages = stage_recorder.finish()
    if not stages:
        raise RuntimeError("no trunk stages were recorded")
    if args.skip_trunk_stages:
        # `savez_compressed` on ~26 GiB of stage tensors runs zlib on a single
        # core and takes over an hour -- an hour spent on arrays that only the
        # stage table reads. A coordinate comparison needs the sampler tape
        # (1.8 MiB) and the structure (85 KiB), both already written above.
        stages = {}
    else:
        np.savez_compressed(args.out_dir / "trunk.npz", **stages)

    (args.out_dir / "tape.json").write_text(
        json.dumps(
            {
                "num_steps": num_steps,
                "num_samples": args.num_samples,
                "num_recycles": args.num_recycles,
                "seed": args.seed,
                "coordinate_shape": list(coordinate.shape),
                "stages": {
                    name: list(value.shape) for name, value in sorted(stages.items())
                },
                **msa_summary,
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "steps": num_steps,
                "coordinate_draws": len(coordinate_draws),
                "rotations": len(recorder.rotations),
                "translations": len(translations),
                "coordinate_shape": list(coordinate.shape),
                "stages": sorted(stages),
                **msa_summary,
                "out": str(args.out_dir),
            },
            indent=2,
        )
    )
    return 0


def _write_msa(
    out_dir: Path, msa_recorder: MSARowRecorder, *, num_recycles: int
) -> dict:
    """Write the alignment upstream read and the rows every pass drew.

    Both are written. The indices alone would let a port re-select from its own
    featurizer's alignment, which is only equivalent if the two featurizers
    emit the same rows in the same order -- an assumption, not a fact. The
    selected rows themselves make the parity run independent of it.
    """
    if not msa_recorder.rows:
        raise RuntimeError(
            "the MSA subsampler was never called; either this build does not "
            "subsample or the patch did not reach it"
        )
    if len(msa_recorder.rows) != num_recycles:
        raise RuntimeError(
            f"expected {num_recycles} MSA row draws, one per trunk pass, got "
            f"{len(msa_recorder.rows)}"
        )
    widths = {row.shape for row in msa_recorder.rows}
    if len(widths) != 1:
        raise RuntimeError(f"MSA draws differ in depth: {sorted(widths)}")

    source = msa_recorder.source or {}
    arrays: dict[str, np.ndarray] = {
        "rows": np.stack(msa_recorder.rows),
        **{f"input_{name}": value for name, value in source.items()},
        **{
            f"selected_{name}": np.stack(
                [cycle[name] for cycle in msa_recorder.selected]
            )
            for name in msa_recorder.selected[0]
        },
    }
    np.savez_compressed(out_dir / "msa.npz", **arrays)

    rows = arrays["rows"]
    identical = bool(np.all(rows == rows[0]))
    return {
        "msa_source_rows": int(np.shape(source.get("msa", np.empty(0)))[0])
        if "msa" in source
        else None,
        "msa_sampled_rows": int(rows.shape[-1]),
        "msa_passes": int(rows.shape[0]),
        # If the draws were identical across passes there would be nothing to
        # match and the whole question would be moot; say so rather than leave
        # it to be inferred from the arrays.
        "msa_rows_identical_across_passes": identical,
    }


#: The upstream runner is a config-driven CLI; this is the same argv
#: `bench/run_upstream.py` drives OpenDDE with, so the capture and the
#: benchmark are running the same program.
def _run(args) -> None:
    """Drive one upstream prediction in-process, with the hooks installed."""

    import sys

    from foldjax.paths import assets_dir

    assets = assets_dir()
    argv = [
        "runner.inference",
        "--input_json_path",
        str(args.input_json),
        "--dump_dir",
        str(args.out_dir / "upstream"),
        "--seeds",
        str(args.seed),
        "--model.N_cycle",
        str(args.num_recycles),
        "--sample_diffusion.N_step",
        str(args.num_steps),
        "--sample_diffusion.N_sample",
        str(args.num_samples),
        "--data.ccd_components_file",
        str(assets / "components.cif"),
        "--data.ccd_components_rdkit_mol_file",
        str(assets / "components.cif.rdkit_mol.pkl"),
        "--load_checkpoint_path",
        str(args.checkpoint),
    ]
    if args.dtype != "fp32":
        argv += ["--dtype", args.dtype]
    if args.disable_tf32:
        argv += ["--enable_tf32", "false"]
    from runner.inference import run

    saved = sys.argv
    sys.argv = argv
    try:
        run()
    finally:
        sys.argv = saved


if __name__ == "__main__":
    raise SystemExit(main())
