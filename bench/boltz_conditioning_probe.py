"""Matched-native-trunk conditioning stages; no sampler/model admission.

The separate-projection arm is a diagnostic counterfactual only. It keeps BF16
operators and native weights, but separates the transition's two input GEMMs.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bench.af3_closure_capture import sha
from bench.boltz_closure_capture import save_new
from bench.boltz_downstream_probe import bound_arrays, load_trunk, source_identity
from bench.boltz_foldjax_capture import native_settings
from bench.boltz_relpos_probe import comparison

FEATURES = (
    "ref_pos",
    "ref_charge",
    "ref_element",
    "ref_atom_name_chars",
    "atom_pad_mask",
    "ref_space_uid",
    "atom_to_token",
)
OUTPUTS = ("q", "c", "atom_enc_bias", "atom_dec_bias", "token_trans_bias")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--features-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source, root = args.source_root.resolve(), args.reference.resolve()
    if Path(__file__).resolve() != source / "bench/boltz_conditioning_probe.py":
        raise ValueError("execute from the explicitly selected source")
    if args.out.exists():
        raise FileExistsError(args.out)
    meta, _ = native_settings(root)
    trunk, _ = load_trunk(root, meta)
    expected, _ = bound_arrays(root, "trunk-boundaries/diffusion_conditioning", OUTPUTS)
    with np.load(root / "features.npz", allow_pickle=False) as archive:
        features = {k: archive[k] for k in FEATURES}
    # load_trunk/bound_arrays verify the captured stage hashes. Bind the raw
    # features to the prior matched-capture report's digest (legacy native
    # completion records do not themselves bind features.npz).
    provenance = json.loads((root / "provenance.json").read_text())
    feature_hash = sha(root / "features.npz")
    from bench.boltz_msa_probe import verify_bound_file

    verify_bound_file(root / "features.npz", args.features_sha256)

    import jax
    import jax.numpy as jnp
    from safetensors import safe_open

    from foldjax.models.boltz2.bridge.native import unflatten_pytree
    from foldjax.models.boltz2.compile_policy import compiler_options
    from foldjax.models.boltz2.models.diffusion import diffusion_conditioning as module
    from foldjax.models.boltz2.models.trunk_blocks import conditioning

    if not Path(inspect.getfile(module)).resolve().is_relative_to(source / "src"):
        raise ValueError("imported a different conditioning source")
    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise RuntimeError("conditioning probe requires exactly one GPU")
    sources = source_identity(source)
    prefix = "d:conditioned_diffusion/d:diffusion_conditioning/"
    with safe_open(args.weights, framework="numpy") as archive:
        encoded = {
            k[len(prefix) :]: archive.get_tensor(k)
            for k in archive.keys()
            if k.startswith(prefix)
        }
    if not encoded:
        raise ValueError("missing conditioning weights")
    params = unflatten_pytree(encoded, {})
    device_trunk = jax.tree.map(jnp.asarray, trunk)
    device_feats = {
        k: jnp.asarray(v.astype(np.int32) if v.dtype == np.int64 else v)
        for k, v in features.items()
    }
    reference_identity = {
        "capture_complete_sha256": sha(root / "capture-complete.json"),
        "features_sha256": feature_hash,
        "weights_sha256": sha(args.weights),
        "native_checkpoint_sha256": provenance["checkpoint_sha256"],
    }
    args.out.mkdir(parents=True, exist_ok=False)
    original_transition = conditioning.transition_forward
    arms = {}
    for separate in (False, True):
        label = "separate_transition_projections" if separate else "production"

        def transition(params, x, **kwargs):
            if separate:
                kwargs["chunk_size"] = params["fc3"]["kernel"].shape[0]
            return original_transition(params, x, **kwargs)

        def run(params, trunk, feats):
            result = module.diffusion_conditioning_forward(
                params,
                trunk["s"],
                trunk["z"],
                trunk["relative_position_encoding"],
                feats,
                compute_dtype=jnp.bfloat16,
                lazy_token_trans_bias=False,
            )
            return {k: result[k] for k in OUTPUTS}

        with (
            patch.object(conditioning, "transition_forward", transition),
            jax.default_matmul_precision("highest"),
        ):
            result = jax.jit(run, compiler_options=compiler_options("bfloat16"))(
                params, device_trunk, device_feats
            )
            jax.block_until_ready(result)
        stored = {k: np.asarray(v.astype(jnp.float32)) for k, v in result.items()}
        with (args.out / f"{label}.npz").open("xb") as stream:
            np.savez(stream, **stored)
        arms[label] = {
            "comparisons": {k: comparison(stored[k], expected[k]) for k in OUTPUTS},
            "arrays_sha256": sha(args.out / f"{label}.npz"),
        }
    if (
        sources != source_identity(source)
        or feature_hash != sha(root / "features.npz")
        or reference_identity["weights_sha256"] != sha(args.weights)
    ):
        raise ValueError("source/input/weights changed during probe")
    save_new(
        args.out / "report.json",
        {
            "capture_complete": True,
            "not_model_parity_admission": True,
            "materialized_bias_only": True,
            "arms": arms,
            "source": sources,
            **reference_identity,
            "compiler_options": compiler_options("bfloat16"),
        },
    )
    print(json.dumps(arms))


if __name__ == "__main__":
    main()
