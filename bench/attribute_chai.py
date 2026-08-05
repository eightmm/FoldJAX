"""Say which stage of a Chai run holds the peak, not merely how big it is.

`peak_bytes_in_use` is one number for the whole process, so it names a size and
nothing else. This wraps the stage entry points and records live bytes on both
sides of each, which turns that one number into a profile: the stage whose exit
raises the running peak is the stage that owns it.

Live bytes, not the arena. A program's temp arena can be huge without touching
the process peak -- removing an 8 GiB arena from this very model once made the
peak go *up* -- so the quantity to watch is what stays resident across a stage
boundary, which is exactly what `bytes_in_use` reports.

    python -m bench.attribute_chai --case t1531 --out chai-1531-stages.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

# Must precede the first JAX import, or the allocator reserves the card and the
# numbers below describe the pool instead of the model.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="t1531")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/claude-1000/attribute-chai"),
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="also wrap the compiled pieces inside the trunk",
    )
    parser.add_argument(
        "--recycles",
        type=int,
        help="override the bench schedule; the peak is set by the first recycle, "
        "so one or two reproduce it at a fraction of the wall time",
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--samples", type=int)
    args = parser.parse_args()

    import jax

    import foldjax
    from bench.spec import SCHEDULE, SEED, cases
    from foldjax.models.chai import inference as inf
    from foldjax.schema import PredictionRequest

    device = jax.local_devices()[0]
    marks: list[dict[str, object]] = []

    def mark(label: str) -> None:
        stats = device.memory_stats() or {}
        marks.append(
            {
                "label": label,
                "live_mib": round(stats.get("bytes_in_use", 0) / 2**20, 1),
                "peak_mib": round(stats.get("peak_bytes_in_use", 0) / 2**20, 1),
            }
        )

    def wrap(name: str, counter: dict[str, int]) -> None:
        """Record live bytes around one stage, collapsing repeated calls.

        Trunk recycles and diffusion steps run hundreds of times; recording
        every one would bury the profile, so only the first and last of each
        stage are kept. The peak column still comes from the process, so a
        middle iteration that spiked would show up in the last mark anyway.
        """
        original = getattr(inf, name)

        def wrapped(*call_args, **call_kwargs):
            index = counter.get(name, 0)
            counter[name] = index + 1
            first = index == 0
            if first:
                mark(f"{name}[0]:enter")
            before = (device.memory_stats() or {}).get("peak_bytes_in_use", 0)
            result = original(*call_args, **call_kwargs)
            jax.block_until_ready(result)
            after = (device.memory_stats() or {}).get("peak_bytes_in_use", 0)
            # Nested stages count inside their caller as well as on their own
            # line: a trunk that raised the mark by 30 GiB and one MSA block
            # inside it that raised it by 30 GiB are the same 30 GiB.
            raised[name] = raised.get(name, 0.0) + (after - before) / 2**20
            mark(f"{name}[{index}]:exit")
            if not first and len(marks) > 2:
                # Keep the running profile short: drop the previous exit unless
                # it moved the peak, so the file holds first/last plus anything
                # that actually mattered.
                previous, current = marks[-2], marks[-1]
                same_stage = str(previous["label"]).startswith(f"{name}[")
                if same_stage and previous["peak_mib"] == current["peak_mib"]:
                    marks.pop(-2)
            return result

        setattr(inf, name, wrapped)

    #: How much each wrapped call raised the process high-water mark. A stage
    #: that never raises it does not own the peak no matter how long it runs or
    #: how much it leaves resident, which is the distinction the flat number
    #: cannot make.
    raised: dict[str, float] = {}

    counter: dict[str, int] = {}
    stages = [
        "_embed_and_initialize",
        "_run_staged_trunk",
        "_run_diffusion_pair_conditioning",
        "_run_diffusion_step_staged",
        "_compiled_diffusion_step",
        "_compiled_confidence_initialize",
        "_run_staged_confidence_block",
    ]
    if args.deep:
        # The pieces `_run_staged_trunk` calls. Patching the module attribute
        # works even for the ones it binds to a local first, because that bind
        # happens per call.
        stages += [
            "_compiled_recycle_template",
            "_compiled_msa_embedding",
            "_compiled_outer_product_mean",
            "_compiled_msa_transition",
            "_compiled_msa_weighted",
            "_compiled_msa_weighted_128",
            "_compiled_residual_add",
            "_run_msa_pair_block_low_memory",
            "_compiled_msa_pair",
            "_run_pairformer_block_low_memory",
            "_compiled_pairformer_stack",
            # Inside the two stages that own the peak. `_embed_and_initialize`
            # raises the process high-water mark by 24,187 MiB of a 33,611 MiB
            # run -- against 856 for the whole staged trunk -- and the two
            # compiled pieces below account for 9,536 of it. The rest of this
            # group exists to find the other 14,600, which is the difference
            # between this port and upstream's peak.
            "_compiled_embed_features",
            "_compiled_embed_non_msa_features",
            "_compiled_token_embedder",
            "_compiled_split_and_bond",
            "_model_padded_inputs",
            "_token_bond_feature",
            "_msa_row_indices_and_mask",
            "_chunked_triangle_multiplication",
            "_chunked_pair_transition",
            "_chunked_triangle_attention",
            "_compiled_msa_pair_pre_attention",
            "_compiled_msa_pair_add_attention",
            "_compiled_pair_attention_bias",
        ]
    for name in stages:
        if hasattr(inf, name):
            wrap(name, counter)

    schedule = dict(SCHEDULE)
    for key, value in (
        ("num_recycles", args.recycles),
        ("num_steps", args.steps),
        ("num_samples", args.samples),
    ):
        if value is not None:
            schedule[key] = value

    case = next(item for item in cases() if item.name == args.case)
    # Chai refuses to write into a directory that already holds a prediction,
    # and it checks *after* the run, so a leftover from the previous attempt
    # throws away the whole computation at the last step.
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    mark("start")
    foldjax.predict(
        PredictionRequest(
            model="chai",
            input=case.job,
            output_dir=args.output_dir,
            seed=SEED,
            options={},
            **schedule,
        )
    )
    mark("end")

    stats = device.memory_stats() or {}
    report = {
        "case": args.case,
        "length": case.length,
        "schedule": schedule,
        "peak_mib": round(stats.get("peak_bytes_in_use", 0) / 2**20, 1),
        "calls": counter,
        "raised_mib": {
            name: round(value, 1)
            for name, value in sorted(
                raised.items(), key=lambda item: -item[1]
            )
        },
        "marks": marks,
    }
    print(json.dumps(report, indent=2))
    print("\npeak raised by, MiB (nested stages also count in their caller):")
    for name, value in report["raised_mib"].items():
        if value > 0:
            print(f"  {value:>10.1f}  {name}")
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
