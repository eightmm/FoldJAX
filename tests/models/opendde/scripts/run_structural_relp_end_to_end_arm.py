"""Run one OpenDDE benchmark arm with dense or direct structural relp."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--relp-arm", choices=("dense", "direct"), required=True)
    args, remaining = parser.parse_known_args()

    # Import the production module before FoldJAX's timer in both arms. Only
    # the selected implementation should differ between A/B processes.
    import foldjax.models.opendde.models.model as model_impl

    if args.relp_arm == "dense":
        from foldjax.models.protenix.models.trunk_blocks.embedders import (
            relative_position_encoding,
            relative_position_features,
        )

        def dense_projection(features, params):
            return relative_position_encoding(
                relative_position_features(features),
                params,
            )

        model_impl.relative_position_encoding_from_features = dense_projection

    from bench.run_foldjax import main as benchmark_main

    sys.argv = [sys.argv[0], *remaining]
    return benchmark_main()


if __name__ == "__main__":
    raise SystemExit(main())
