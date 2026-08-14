"""One-time converter: published Boltz-2 checkpoints -> JAX native weights.

Loads the Lightning ``.ckpt`` container through FoldJAX's restricted NumPy
archive reader, builds every JAX parameter pytree through the bridge mappers,
and writes the native format (safetensors plus a JSON sidecar, or an ``.npz``
fallback). PyTorch is neither installed nor imported for this conversion.

Prediction never calls this: it reads the converted file. Run it once per
checkpoint, or let ``foldjax weights fetch --model boltz2`` run it for you.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import jax.numpy as jnp

from foldjax.models.boltz2.bridge.checkpoint import load_checkpoint_state_dict
from foldjax.models.boltz2.bridge.confidence_mapping import (
    map_confidence_module_state_dict,
)
from foldjax.models.boltz2.bridge.native import save_params
from foldjax.models.boltz2.bridge.torch_mapping import (
    map_bfactor_state_dict,
    map_boltz2_graph_state_dict,
    map_distogram_state_dict,
)

_DTYPES = {"fp32": None, "bf16": jnp.bfloat16, "fp16": jnp.float16}
_SUFFIX = {"fp32": "", "bf16": "_bf16", "fp16": "_fp16"}

# The released Boltz-2 architecture these mappers were written against.
MSA_LAYERS = 4
PAIRFORMER_LAYERS = 64
TOKEN_LAYERS = 24
TOKEN_TRANSFORMER_HEADS = 16
CONFIDENCE_PF_LAYERS = 8


def _graph(state: dict) -> dict:
    return map_boltz2_graph_state_dict(
        state,
        num_msa_layers=MSA_LAYERS,
        num_pairformer_layers=PAIRFORMER_LAYERS,
        num_token_layers=TOKEN_LAYERS,
        token_transformer_heads=TOKEN_TRANSFORMER_HEADS,
    )


def export_confidence(checkpoint: Path, out_dir: Path, dtype: str = "fp32") -> Path:
    """Convert ``boltz2_conf.ckpt`` into native weights and return their path."""
    state = load_checkpoint_state_dict(checkpoint)
    params = {
        **_graph(state),
        # The distogram, bfactor, and confidence heads ship inside the same
        # checkpoint, so they are bundled rather than written separately.
        "distogram": map_distogram_state_dict(state)["distogram"],
        "confidence": map_confidence_module_state_dict(
            state, "confidence_module", num_pairformer_layers=CONFIDENCE_PF_LAYERS
        ),
    }
    if any(key.startswith("bfactor_module.") for key in state):
        params["bfactor"] = map_bfactor_state_dict(state)["bfactor"]

    out_dir.mkdir(parents=True, exist_ok=True)
    info = save_params(params, out_dir / f"boltz2_conf{_SUFFIX[dtype]}", _DTYPES[dtype])
    return Path(info["weights_path"])


def export_affinity(checkpoint: Path, out_dir: Path) -> Path | None:
    """Convert ``boltz2_aff.ckpt``; return None when it holds no affinity head."""
    from foldjax.models.boltz2.bridge.affinity_mapping import (
        map_affinity_ensemble_state_dict,
    )

    state = load_checkpoint_state_dict(checkpoint)
    if not any(key.startswith("affinity_module") for key in state):
        return None
    params = {
        **_graph(state),
        "distogram": map_distogram_state_dict(state)["distogram"],
        "confidence": map_confidence_module_state_dict(
            state, "confidence_module", num_pairformer_layers=CONFIDENCE_PF_LAYERS
        ),
        "affinity": map_affinity_ensemble_state_dict(state),
    }
    if any(key.startswith("bfactor_module.") for key in state):
        params["bfactor"] = map_bfactor_state_dict(state)["bfactor"]

    out_dir.mkdir(parents=True, exist_ok=True)
    return Path(save_params(params, out_dir / "boltz2_aff")["weights_path"])


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conf-ckpt", type=Path, required=True)
    parser.add_argument("--aff-ckpt", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--dtype",
        choices=sorted(_DTYPES),
        default="fp32",
        help="storage dtype for the confidence weights; casts float arrays only",
    )
    args = parser.parse_args(argv)

    if not args.conf_ckpt.is_file():
        raise SystemExit(f"missing checkpoint: {args.conf_ckpt}")
    weights = export_confidence(args.conf_ckpt, args.out_dir, args.dtype)
    print(f"wrote native weights: {weights}")

    if args.aff_ckpt is not None:
        if not args.aff_ckpt.is_file():
            raise SystemExit(f"missing checkpoint: {args.aff_ckpt}")
        affinity = export_affinity(args.aff_ckpt, args.out_dir)
        if affinity is None:
            print(f"no affinity head in {args.aff_ckpt}; skipped")
        else:
            print(f"wrote native weights: {affinity}")


if __name__ == "__main__":
    main()
