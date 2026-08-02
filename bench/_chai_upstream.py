"""Run one upstream Chai-1 prediction. Executed by Chai's own virtualenv.

Chai has no CLI knob for the trunk recycle count in older releases and its
`fold` command is a thin Typer wrapper over `run_inference`, so calling the
function directly is both simpler and exactly the same code path.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--msa-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--recycles", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    from chai_lab.chai1 import run_inference

    # `run_inference` asserts the output directory is empty if it exists.
    out = args.out / "chai"
    if out.exists():
        raise SystemExit(f"refusing to reuse a populated output dir: {out}")

    run_inference(
        fasta_file=args.fasta,
        output_dir=out,
        msa_directory=args.msa_dir,
        num_trunk_recycles=args.recycles,
        num_diffn_timesteps=args.steps,
        num_diffn_samples=args.samples,
        num_trunk_samples=1,
        seed=args.seed,
        device="cuda:0",
        use_esm_embeddings=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
