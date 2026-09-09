"""Small fixed-input GPU atom reduction discriminator; not model admission."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("native", "foldjax"))
    parser.add_argument("--fp32-accumulation", action="store_true")
    args = parser.parse_args()
    values = [1.0] + [2**-8] * 256
    if args.arm == "native":
        import torch
        from protenix.model.utils import aggregate_atom_to_token

        for dtype in (torch.float32, torch.bfloat16):
            x = torch.tensor(values, device="cuda", dtype=dtype).reshape(-1, 1)
            index = torch.zeros(len(values), device="cuda", dtype=torch.int64)
            for reduction in ("sum", "mean"):
                result = [
                    aggregate_atom_to_token(
                        x.float() if args.fp32_accumulation else x,
                        index,
                        n_token=1,
                        reduce=reduction,
                    )
                    .to(dtype)
                    .float()
                    .item()
                    for _ in range(3)
                ]
                print(
                    json.dumps(
                        [
                            args.arm,
                            str(dtype),
                            reduction,
                            args.fp32_accumulation,
                            result,
                        ]
                    )
                )
    else:
        import jax
        import jax.numpy as jnp

        from foldjax.models.protenix.models.diffusion.atom import (
            aggregate_atom_to_token,
        )

        if jax.default_backend() != "gpu":
            raise RuntimeError("this discriminator requires a GPU")
        for dtype in (jnp.float32, jnp.bfloat16):
            x = jnp.asarray(values, dtype=dtype).reshape(-1, 1)
            index = jnp.zeros(len(values), dtype=jnp.int32)
            for reduction in ("sum", "mean"):
                infer = jax.jit(
                    lambda x: aggregate_atom_to_token(
                        x.astype(jnp.float32) if args.fp32_accumulation else x,
                        index,
                        n_token=1,
                        reduce=reduction,
                    ).astype(dtype)
                )
                result = [float(infer(x)[0, 0]) for _ in range(3)]
                print(
                    json.dumps(
                        [
                            args.arm,
                            str(dtype),
                            reduction,
                            args.fp32_accumulation,
                            result,
                        ]
                    )
                )


if __name__ == "__main__":
    main()
