"""Isolate the OpenDDE confidence head with identical native inputs and weights.

This intentionally supplies native intermediates, so it cannot establish
independent-input end-to-end parity. The legacy-axis arm is a one-variable
counterfactual to the corrected shared implementation, never a production mode.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from bench.af3_closure import compare_confidence
from bench.af3_closure_capture import save, sha
from bench.opendde_closure_report import arrays


def validate_representatives(values):
    mask = values["selected_distogram_rep_atom_mask"]
    atoms = values["x_pred_coords"].shape[-2]
    tokens = values["s_inputs"].shape[-2]
    if (
        mask.shape != (atoms,)
        or not np.isin(mask, [0, 1]).all()
        or int(mask.sum()) != tokens
    ):
        raise ValueError("representative mask must select exactly one atom per token")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    import jax
    import jax.numpy as jnp

    from foldjax.models.opendde.bridge.weights_io import load_native_weights
    from foldjax.models.protenix.models.heads.confidence import confidence_head

    values = arrays(args.native / "confidence-input-0.npz")
    native = arrays(args.native / "confidence-output-0.npz")
    provenance = json.loads((args.native / "provenance.json").read_text())
    if len(provenance["confidence_calls"]) != 1:
        raise ValueError("expected one complete five-sample native confidence call")
    if values["x_pred_coords"].shape[0] != 5:
        raise ValueError("requires five samples without a batch dimension")
    validate_representatives(values)
    features = {
        key: jnp.asarray(values[f"input_feature_dict.{key}"])
        for key in (
            "distogram_rep_atom_mask",
            "atom_to_token_idx",
            "atom_to_tokatom_idx",
        )
    }
    features["distogram_rep_atom_mask"] = jnp.asarray(
        values["selected_distogram_rep_atom_mask"]
    )
    params = load_native_weights(args.weights).confidence
    inputs = {
        key: jnp.asarray(values[key])
        for key in ("s_inputs", "s_trunk", "z_trunk", "x_pred_coords")
    }
    inputs["pair_mask"] = (
        jnp.asarray(values["pair_mask"]) if "pair_mask" in values else None
    )

    def execute(parameters):
        return confidence_head(features, **inputs, params=parameters, use_scan=False)

    records = {}
    for precision in ("high", "highest"):
        with jax.default_matmul_precision(precision):
            compiled = jax.jit(execute)
            for axis, parameters in (
                ("native-axis", params),
                (
                    "legacy-axis",
                    params._replace(
                        linear_s1=params.linear_s2, linear_s2=params.linear_s1
                    ),
                ),
            ):
                result = jax.device_get(compiled(parameters))
                name = f"{precision}-{axis}"
                np.savez_compressed(args.out / f"{name}.npz", **result)
                records[name] = compare_confidence(native, result)
                print(
                    json.dumps(
                        {
                            "arm": name,
                            "passed": records[name]["passed"],
                            "max_errors": {
                                key: leaf["max_absolute_error"]
                                for key, leaf in records[name]["leaves"].items()
                            },
                        }
                    ),
                    flush=True,
                )
    save(
        args.out / "report.json",
        {
            "scope": __doc__,
            "native_confidence_policy": provenance["confidence_calls"][0],
            "native_input_sha256": sha(args.native / "confidence-input-0.npz"),
            "native_output_sha256": sha(args.native / "confidence-output-0.npz"),
            "native_provenance_sha256": sha(args.native / "provenance.json"),
            "weights_sha256": sha(args.weights),
            "source_sha256": sha(
                Path(__file__).resolve().parents[1]
                / "src/foldjax/models/protenix/models/heads/confidence.py"
            ),
            "jax_version": jax.__version__,
            "source_files": {
                str(path.relative_to(Path(__file__).resolve().parents[1])): sha(path)
                for path in sorted(
                    (Path(__file__).resolve().parents[1] / "src").rglob("*.py")
                )
            },
            "comparisons": records,
        },
    )


if __name__ == "__main__":
    main()
