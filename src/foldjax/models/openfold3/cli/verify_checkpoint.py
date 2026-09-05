"""Verify a real OpenFold3 checkpoint against this port.

Runs the checks that cannot be done without weights, in the order that fails
cheapest first:

1. block counts against :data:`RELEASED_BLOCK_COUNTS`, so a config mismatch is
   caught before any mapping is attempted,
2. ``map_inference_params``, which raises on a missing or unexpectedly-shaped
   key in any of the seven parameter groups,
3. optionally the full ``predict`` path on the real parameters.

Step 3 takes the self-contained ``.npz`` made by ``openfold3-jax-featurize``.
Its token and atom sizes are read from the archive; ``--tokens``/``--atoms`` are
optional assertions for scripts that want to pin the expected bucket.

    openfold3-jax-verify-checkpoint of3-ob-2025-06-30-174k.pt
    openfold3-jax-verify-checkpoint of3-ob-2025-06-30-174k.pt --batch batch.npz
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openfold3-jax-verify-checkpoint",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--prefix",
        default="",
        help="model root prefix, e.g. 'model' (default: auto-detect)",
    )
    parser.add_argument(
        "--batch", type=Path, default=None, help="featurized batch as .npz"
    )
    parser.add_argument(
        "--tokens", type=int, default=None, help="optional expected token count"
    )
    parser.add_argument(
        "--atoms", type=int, default=None, help="optional expected atom count"
    )
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cycles", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    batch_np = None
    table = None
    if args.batch is not None:
        # Validate and size the archive before reading a multi-gigabyte
        # checkpoint. This also strips chemistry metadata from the model batch.
        from foldjax.models.openfold3.data import load_features

        try:
            batch_np, table = load_features(args.batch)
        except (OSError, ValueError) as error:
            print(f"cannot load batch: {error}")
            return 1
        n_token = batch_np["token_mask"].shape[-1]
        n_atom = batch_np["atom_mask"].shape[-1]
        for label, expected, actual in (
            ("tokens", args.tokens, n_token),
            ("atoms", args.atoms, n_atom),
        ):
            if expected is not None and expected != actual:
                print(
                    f"batch has {actual} {label}, but --{label} expected {expected}"
                )
                return 1

    from foldjax.models.openfold3.bridge.checkpoint import (
        detect_fused_tri_mul,
        load_checkpoint,
    )
    from foldjax.models.openfold3.cli.inspect_checkpoint import count_blocks
    from foldjax.models.openfold3.inference import RELEASED_BLOCK_COUNTS

    state = load_checkpoint(args.checkpoint)
    print(f"loaded {len(state)} tensors from {args.checkpoint}")

    layout = detect_fused_tri_mul(state)
    print(
        "triangular multiplication: "
        f"{ {True: 'fused', False: 'unfused', None: 'not found'}[layout] }"
    )

    print("\nblock counts vs the released architecture:")
    mismatched = False
    for root, expected in RELEASED_BLOCK_COUNTS.items():
        found = count_blocks(state, root)
        if found is None:
            print(f"  {root:<36} MISSING (expected {expected})")
            mismatched = True
        elif found != expected:
            print(f"  {root:<36} {found} != {expected}  MISMATCH")
            mismatched = True
        else:
            print(f"  {root:<36} {found} ok")
    if mismatched:
        print(
            "\nThis checkpoint does not match released_config(); do not use that "
            "preset with it. Inspect it with openfold3-jax-inspect-checkpoint."
        )
        return 1

    prefix = args.prefix if args.prefix else None
    print("\nmapping every parameter group ...")
    from foldjax.models.openfold3.bridge.torch_mapping import map_inference_params

    params = map_inference_params(state, prefix)
    print(f"  Pairformer blocks   {len(params.trunk.pairformer_stack.blocks)}")
    print(f"  MSA blocks          {len(params.trunk.msa_module.blocks)}")
    print(f"  diffusion blocks    {len(params.denoiser.diffusion_transformer.blocks)}")
    template = params.trunk.template_embedder
    print(
        "  template tower      "
        + (
            f"{len(template.template_pair_stack.blocks)} blocks"
            if template is not None
            else "absent"
        )
    )
    print(
        "  confidence embed    "
        f"{len(params.pairformer_embedding.pairformer_stack.blocks)} blocks"
    )
    print("  heads               plddt, pae, pde, distogram, exp-resolved")

    if args.batch is None:
        print(
            "\nNo --batch given, so the forward pass was not run. Supply a "
            "featurized batch to check predicted structures."
        )
        return 0

    import jax
    import jax.numpy as jnp
    import numpy as onp

    from foldjax.models.openfold3.data import (
        normalize_asym_ids,
        prepare_msa_cycle_features,
    )
    from foldjax.models.openfold3.inference import predict, released_config

    config = released_config(
        n_token=n_token,
        n_atom=n_atom,
        num_recycles=args.cycles,
        num_samples=args.samples,
        num_steps=args.steps,
    )
    batch_np = prepare_msa_cycle_features(
        batch_np,
        config.msa_depth,
        num_recycles=config.num_recycles,
        rng=onp.random.default_rng(0),
    )
    batch_np, n_chain = normalize_asym_ids(batch_np)
    batch = {key: jnp.asarray(value) for key, value in batch_np.items()}
    print(f"\nrunning predict on {jax.devices()[0]} ...")
    prediction = predict(
        jax.random.key(0),
        batch,
        params,
        config,
        table,
        n_chain=n_chain,
    )
    ok = True
    for name, value in prediction._asdict().items():
        if value is None:
            print(f"  {name:<16} absent")
            continue
        array = onp.asarray(value)
        finite = bool(onp.isfinite(array).all())
        ok = ok and finite
        print(f"  {name:<16} {array.shape}  finite={finite}")
    if not ok:
        print("\nnon-finite outputs; the mapping or the config is wrong")
        return 1
    print("\npredict ran on real weights with finite outputs. Structural accuracy")
    print("still needs a reference structure; this checks neither RMSD nor clashes.")
    return 0


def entrypoint() -> None:
    raise SystemExit(main())

if __name__ == "__main__":  # ``python -m foldjax.models.openfold3.cli.<name>``
    entrypoint()
