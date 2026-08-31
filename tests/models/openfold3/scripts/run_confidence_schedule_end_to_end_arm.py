"""Run one OpenFold3 benchmark arm with batched or serial confidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--confidence-arm",
        choices=("batched", "serial"),
        required=True,
    )
    args, remaining = parser.parse_known_args()

    # Keep module import outside FoldJAX's timed predict call in both arms.
    import foldjax.models.openfold3.inference as inference

    if args.confidence_arm == "batched":
        inference._per_sample_confidence = lambda _config: False
    else:
        inference._per_sample_confidence = lambda config: config.num_samples > 1

    from bench.run_foldjax import main as benchmark_main

    sys.argv = [sys.argv[0], *remaining]
    return benchmark_main()


if __name__ == "__main__":
    raise SystemExit(main())
