"""Perturbed-trunk Boltz sampler counterfactual: is one sample a near-tie?

Wraps :mod:`bench.boltz_downstream_probe`. After the probe cross-binds the
native trunk it adds Gaussian noise to ``s``, ``z`` and ``s_inputs`` at a
requested relative RMSE (the band the port's own trunk sits at), so the
sampler sees a random perturbation of the same size as the port's. The probe
records the perturbed arrays' identities in its policy, so the output names
what it sampled from. Not parity evidence; a sensitivity control.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from bench import boltz_downstream_probe as probe

_DEFAULT_BANDS = {"s": 1.022e-3, "z": 2.698e-3, "s_inputs": 7.8e-5}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--band",
        action="append",
        default=[],
        metavar="NAME=REL",
        help="relative RMSE per trunk leaf; default s/z/s_inputs at the port band",
    )
    args, rest = parser.parse_known_args(argv)
    bands = dict(_DEFAULT_BANDS)
    for item in args.band:
        name, value = item.split("=", 1)
        bands[name] = float(value)
    original = probe.load_trunk

    def perturbed(root, meta):
        trunk, tree = original(root, meta)
        rng = np.random.default_rng(args.seed)
        realised = {}
        for key, value in trunk.items():
            rel = bands.get(key, 0.0)
            if rel <= 0:
                continue
            base = value.astype(np.float64)
            scale = float(np.sqrt((base**2).mean()))
            noisy = base + rng.standard_normal(base.shape) * rel * scale
            trunk[key] = noisy.astype(np.float32)
            realised[key] = float(np.sqrt(((trunk[key] - base) ** 2).mean()) / scale)
        print(f"perturbed trunk seed={args.seed} realised relative RMSE {realised}")
        return trunk, tree

    probe.load_trunk = perturbed
    return probe.main(rest)


if __name__ == "__main__":
    sys.exit(main())
