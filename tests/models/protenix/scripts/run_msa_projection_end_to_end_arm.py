"""Run one full Protenix benchmark arm with dense or compact MSA projection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("dense", "compact"), required=True)
    parser.add_argument("--case", default="L1000_3og2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    from foldjax.models.protenix.models.trunk_blocks import msa as msa_module

    if args.arm == "dense":
        msa_module._msa_input_projection = msa_module._dense_msa_input_projection

    from bench import run_foldjax

    sys.argv = [
        "bench.run_foldjax",
        "--model",
        "protenix",
        "--case",
        args.case,
        "--output-dir",
        str(args.output_dir),
        "--json-out",
        str(args.json_out),
        "--seed",
        "101",
        "--label",
        f"protenix-msa-{args.arm}",
    ]
    return run_foldjax.main()


if __name__ == "__main__":
    raise SystemExit(main())
