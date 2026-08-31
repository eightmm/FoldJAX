"""Run one AlphaFold 3 benchmark arm with batched or sharded samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--sample-arm",
        choices=("batched", "sharded"),
        required=True,
    )
    args, remaining = parser.parse_known_args()

    from foldjax.models.alphafold3 import build

    build.register_runtime()
    from alphafold3.model import model as model_impl

    if args.sample_arm == "batched":
        model_impl._diffusion_sample_shard_size = lambda _config, _sample: None

    from bench.run_foldjax import main as benchmark_main

    sys.argv = [sys.argv[0], *remaining]
    return benchmark_main()


if __name__ == "__main__":
    raise SystemExit(main())
