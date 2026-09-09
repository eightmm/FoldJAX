"""Probe whether JIT widening preserves the BF16 aggregation output grid."""

import json

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.protenix.models.diffusion.atom import aggregate_atom_to_token


def main():
    if jax.default_backend() != "gpu":
        raise RuntimeError("GPU discriminator requires GPU")
    for count in (3, 7, 11, 23):
        x = jnp.linspace(0.1, 0.9, count * 16).reshape(count, 16).astype(jnp.bfloat16)
        index = jnp.zeros(count, jnp.int32)

        def infer(value):
            return aggregate_atom_to_token(value, index, n_token=1).astype(jnp.float32)

        for compiled in (False, True):
            y = np.asarray((jax.jit(infer) if compiled else infer)(x))
            rounded = np.asarray(
                jnp.asarray(y).astype(jnp.bfloat16).astype(jnp.float32)
            )
            print(
                json.dumps(
                    {
                        "atoms": count,
                        "jit": compiled,
                        "off_bf16_grid": int(np.count_nonzero(y != rounded)),
                        "max_rounding_error": float(np.max(np.abs(y - rounded))),
                    }
                )
            )


if __name__ == "__main__":
    main()
