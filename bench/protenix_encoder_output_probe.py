"""Isolated real-weight JAX encoder output probe; shared-input diagnostic only."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.protenix.cli.predict import _load_prepared_params
from foldjax.models.protenix.models.trunk_blocks.embedders import input_feature_embedder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError("encoder discriminator requires GPU")
    params = _load_prepared_params(args.weights, "bf16").input_embedder
    features = {}
    for name in ("native-input.npz", "native-derived.npz"):
        with np.load(args.reference / name, allow_pickle=False) as archive:
            for key in archive.files:
                parts = key.split(".")
                target = features
                for part in parts[:-1]:
                    target = target.setdefault(part, {})
                value = archive[key]
                target[parts[-1]] = (
                    value.item() if value.ndim == 0 else jnp.asarray(value)
                )
    n_token = features["restype"].shape[0]
    with jax.default_matmul_precision("high"):
        infer = jax.jit(
            lambda: input_feature_embedder(
                features,
                params,
                n_token=n_token,
                n_heads=4,
                n_queries=32,
                n_keys=128,
                use_scan=True,
            )
        )
        result = np.asarray(infer())
    with np.load(args.reference / "trunk.npz", allow_pickle=False) as archive:
        native = archive["s_inputs"]
    embedding = result[:, :384]
    rounded = np.asarray(
        jnp.asarray(embedding).astype(jnp.bfloat16).astype(jnp.float32)
    )
    delta = result.astype(np.float64) - native.astype(np.float64)
    print(
        json.dumps(
            {
                "shape": list(result.shape),
                "off_bf16_grid": int(np.count_nonzero(embedding != rounded)),
                "s_inputs_rmse": float(np.sqrt(np.mean(delta**2))),
                "s_inputs_max_abs": float(np.max(np.abs(delta))),
                "shared_input_diagnostic_only": True,
            }
        )
    )


if __name__ == "__main__":
    main()
