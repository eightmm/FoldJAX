"""Developer converter for Boltz checkpoints and legacy feature fixtures.

Loads Lightning ``.ckpt`` files through FoldJAX's restricted NumPy reader,
builds every JAX parameter pytree via the existing bridge mappers, and writes
them to ``outputs/native_weights/`` in the native format (safetensors + JSON
sidecar, or .npz fallback).

The test-only publisher runtime is used solely to convert legacy real-feature
``.pt`` fixtures into plain ``.npz`` files.

Run:
    uv run python scripts/export_native_weights.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import torch

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

# Match the layer/head config used by the parity + benchmark scripts.
MSA_LAYERS = 4
PAIRFORMER_LAYERS = 64
TOKEN_LAYERS = 24
TOKEN_TRANSFORMER_HEADS = 16
CONFIDENCE_PF_LAYERS = 8


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TiB"


def export_conf(ckpt: Path, out_dir: Path, dtype_name: str = "fp32") -> None:
    print(f"[conf] loading {ckpt}")
    state = load_checkpoint_state_dict(ckpt)

    graph = map_boltz2_graph_state_dict(
        state,
        num_msa_layers=MSA_LAYERS,
        num_pairformer_layers=PAIRFORMER_LAYERS,
        num_token_layers=TOKEN_LAYERS,
        token_transformer_heads=TOKEN_TRANSFORMER_HEADS,
    )
    # Distogram + bfactor heads share the conf checkpoint; bundle alongside.
    params = {
        **graph,
        "distogram": map_distogram_state_dict(state)["distogram"],
    }
    if any(k.startswith("bfactor_module.") for k in state):
        params["bfactor"] = map_bfactor_state_dict(state)["bfactor"]
    # Confidence head shares the conf checkpoint; bundle under "confidence".
    params["confidence"] = map_confidence_module_state_dict(
        state, "confidence_module", num_pairformer_layers=CONFIDENCE_PF_LAYERS
    )

    out = out_dir / f"boltz2_conf{_SUFFIX[dtype_name]}"
    info = save_params(params, out, dtype=_DTYPES[dtype_name])
    print(
        f"[conf] {info['backend']} -> {info['weights_path']} "
        f"({_fmt_bytes(info['bytes'])}, {info['num_arrays']} arrays, "
        f"{info['num_scalars']} scalars)"
    )


def export_affinity(ckpt: Path, out_dir: Path) -> None:
    if not ckpt.exists():
        print(f"[aff] skip (missing {ckpt})")
        return
    # Affinity mapper imports boltz types lazily; only import here (dev-time).
    from foldjax.models.boltz2.bridge.affinity_mapping import (
        map_affinity_ensemble_state_dict,
    )

    print(f"[aff] loading {ckpt}")
    state = load_checkpoint_state_dict(ckpt)
    if not any(key.startswith("affinity_module") for key in state):
        print("[aff] no affinity_module weights found; skip")
        return
    graph = map_boltz2_graph_state_dict(
        state,
        num_msa_layers=MSA_LAYERS,
        num_pairformer_layers=PAIRFORMER_LAYERS,
        num_token_layers=TOKEN_LAYERS,
        token_transformer_heads=TOKEN_TRANSFORMER_HEADS,
    )
    params = {
        **graph,
        "distogram": map_distogram_state_dict(state)["distogram"],
        "confidence": map_confidence_module_state_dict(
            state,
            "confidence_module",
            num_pairformer_layers=CONFIDENCE_PF_LAYERS,
        ),
        "affinity": map_affinity_ensemble_state_dict(state),
    }
    if any(key.startswith("bfactor_module.") for key in state):
        params["bfactor"] = map_bfactor_state_dict(state)["bfactor"]
    out = out_dir / "boltz2_aff"
    info = save_params(params, out)
    print(
        f"[aff] {info['backend']} -> {info['weights_path']} "
        f"({_fmt_bytes(info['bytes'])}, {info['num_arrays']} arrays)"
    )


def export_features(pt_path: Path) -> None:
    if not pt_path.exists():
        print(f"[feat] skip (missing {pt_path})")
        return
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    feats: dict[str, np.ndarray] = {}
    for key, value in obj.items():
        if key.startswith("_") or not torch.is_tensor(value):
            continue
        feats[key] = value.detach().cpu().numpy()
    out = pt_path.with_suffix(".npz")
    np.savez(out, **feats)
    print(f"[feat] {pt_path.name} -> {out.name} ({len(feats)} tensors)")


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent
    cache = root / ".cache" / "boltz"
    parser.add_argument(
        "--conf-ckpt", type=Path, default=cache / "boltz2_conf.ckpt"
    )
    parser.add_argument(
        "--aff-ckpt", type=Path, default=cache / "boltz2_aff.ckpt"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=root / "outputs" / "native_weights"
    )
    parser.add_argument(
        "--dtype", choices=["fp32", "bf16", "fp16"], default="fp32",
        help="storage dtype for conf weights; casts FLOAT arrays only, writes "
             "boltz2_conf{,_bf16,_fp16}.safetensors",
    )
    parser.add_argument(
        "--features",
        type=Path,
        nargs="*",
        default=[Path("outputs/real_features/1UBQ_A.pt")],
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    export_conf(args.conf_ckpt, args.out_dir, dtype_name=args.dtype)
    if args.dtype == "fp32":
        # Affinity/features are dtype-agnostic for this task; only export them
        # on the default fp32 pass to avoid redundant rewrites.
        export_affinity(args.aff_ckpt, args.out_dir)
        for feat in args.features:
            export_features(feat)
    print("done.")


if __name__ == "__main__":
    main()
