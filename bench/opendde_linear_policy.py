"""Bounded native Linear observations and same-input JAX precision controls."""

import argparse
import collections
import json
import os
from pathlib import Path

import numpy as np

from bench.af3_closure_capture import save, sha


class LinearPolicyObserver:
    """Record up to six distinct shapes per native component, at most 64 MiB each.

    A second direct forward with TF32 disabled is a diagnostic intervention.
    It does not replace the returned result, and these timings are inadmissible.
    """

    def __init__(self, out):
        self.out = Path(out)
        self.names, self.records, self.seen = {}, [], set()
        self.counts = collections.Counter()

    def __enter__(self):
        import torch

        self.pre = torch.nn.modules.module.register_module_forward_pre_hook(self.before)
        self.post = torch.nn.modules.module.register_module_forward_hook(self.after)
        return self

    def __exit__(self, *args):
        self.pre.remove()
        self.post.remove()
        save(self.out / "linear-policy.json", self.records)

    def before(self, module, args):
        if type(module).__name__ == "OpenDDE":
            self.names.update({id(m): name for name, m in module.named_modules()})

    def after(self, module, args, result):
        import torch

        if not isinstance(module, torch.nn.Linear) or not args:
            return
        x = args[0]
        if x.dtype != torch.float32 or result.dtype != torch.float32:
            return
        name = self.names.get(id(module))
        if name is None:
            return
        group = name.split(".", 1)[0]
        signature = (group, tuple(x.shape), tuple(module.weight.shape))
        if signature in self.seen or self.counts[group] >= 6 or len(self.records) >= 64:
            return
        size = (x.numel() + result.numel() * 2 + module.weight.numel()) * 4
        if size > 64 * 2**20:
            return
        self.seen.add(signature)
        self.counts[group] += 1
        enabled = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            precise = module.forward(x)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = enabled
        values = {
            "x": x.detach().cpu().numpy(),
            "y": result.detach().cpu().numpy(),
            "fp32_control": precise.detach().cpu().numpy(),
            "weight": module.weight.detach().cpu().numpy(),
        }
        if module.bias is not None:
            values["bias"] = module.bias.detach().cpu().numpy()
        filename = f"linear-policy-{len(self.records)}.npz"
        np.savez_compressed(self.out / filename, **values)
        self.records.append(
            {
                "name": name,
                "group": group,
                "input_shape": list(x.shape),
                "weight_shape": list(module.weight.shape),
                "file": filename,
                "matmul_allow_tf32": enabled,
                "native_tf32_effect_max_abs": float(
                    np.max(abs(values["y"] - values["fp32_control"]))
                ),
            }
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    import jax
    import jax.numpy as jnp

    from foldjax.models.protenix.models.primitives.primitives import (
        LinearParams,
        linear,
    )

    records = json.loads((args.native / "linear-policy.json").read_text())
    if not records:
        raise ValueError("no native Linear observations")
    for record in records:
        path = args.native / record["file"]
        with np.load(path, allow_pickle=False) as archive:
            values = dict(archive)
        params = LinearParams(
            jnp.asarray(values["weight"]),
            jnp.asarray(values["bias"]) if "bias" in values else None,
        )
        record["capture_sha256"] = sha(path)
        record["comparisons"] = {}
        for precision in ("high", "highest"):
            with jax.default_matmul_precision(precision):
                executable = (
                    jax.jit(linear).lower(jnp.asarray(values["x"]), params).compile()
                )
                result = np.asarray(executable(jnp.asarray(values["x"]), params))
                if record["file"] in (
                    "linear-policy-14.npz",
                    "linear-policy-25.npz",
                    "linear-policy-46.npz",
                ):
                    (args.out / f"{precision}-{record['file']}.hlo.txt").write_text(
                        executable.as_text()
                    )
            record["comparisons"][precision] = {
                reference: {
                    "max_abs": float(np.max(abs(result - values[reference]))),
                    "rmse": float(
                        np.sqrt(
                            np.mean(
                                (result.astype(np.float64) - values[reference]) ** 2
                            )
                        )
                    ),
                }
                for reference in ("y", "fp32_control")
            }
        print(json.dumps(record), flush=True)
    save(
        args.out / "report.json",
        {
            "scope": __doc__,
            "native_provenance_sha256": sha(args.native / "provenance.json"),
            "jax_version": jax.__version__,
            "records": records,
            "wrapper_sha256": sha(Path(__file__)),
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
        },
    )


if __name__ == "__main__":
    main()
