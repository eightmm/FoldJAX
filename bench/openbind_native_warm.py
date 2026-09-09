"""Native prediction-step warm timing with shared inputs and ordinary native RNG.

``--triangle-backend`` selects upstream's own Triton kernels or the
cuEquivariance torch kernels the same way the parity captures do, and the
kernel census proves which one executed (cuEq attention silently falls back
to plain torch at or below 100 tokens).
"""

import argparse
import contextlib
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bench.boltz_historical_replay import digest, save_new
from bench.openbind_core_replay import positive_count
from bench.openbind_native_capture import KernelCensus
from bench.openbind_native_outputs import configured_backend
from bench.openbind_warm import measure_calls


def copy_containers(value):
    """Native forward pops mapping entries; preserve tensor storage, copy containers."""
    if isinstance(value, dict):
        return {key: copy_containers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [copy_containers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(copy_containers(item) for item in value)
    return value


def assert_shared_features_unchanged(actual, expected):
    if actual.keys() - expected.keys():
        raise ValueError("unexpected shared native feature")
    for name, value in actual.items():
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise ValueError(f"shared feature schema changed: {name}")
        np.testing.assert_array_equal(value, expected[name], err_msg=name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--warm-repeats", type=positive_count, default=3)
    parser.add_argument("--triangle-backend", choices=("cueq", "triton", "xla"))
    args = parser.parse_args(argv)
    sys.path.insert(0, str(args.upstream_root))

    import torch
    from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
    from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom
    from openfold3.run_openfold import cli

    original_config = OF3ProjectEntry.get_model_config_with_update
    config_calls = []

    def model_config(self, update):
        config = configured_backend(
            lambda value: original_config(self, value), update, args.triangle_backend
        )
        config_calls.append(True)
        return config

    census = KernelCensus()

    args.out_dir.mkdir(parents=True, exist_ok=False)
    runner = args.out_dir / "runner.yaml"
    save_new(
        runner,
        {
            "model_update": {"presets": ["predict"]},
            "experiment_settings": {"seeds": [101]},
            "pl_trainer_args": {"precision": "32-true"},
        },
    )
    input_path = args.capture / "input.npz"
    with np.load(input_path, allow_pickle=False) as archive:
        captured = dict(archive)
    sources = {
        str(p.relative_to(args.upstream_root)): digest(p)
        for p in sorted((args.upstream_root / "openfold3").rglob("*.py"))
    }
    if not sources:
        raise ValueError("no native source files")
    identity = {
        "native_source": sources,
        "checkpoint": digest(args.checkpoint),
        "input": digest(args.input),
        "features": digest(input_path),
        "harness": digest(Path(__file__)),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    original = OpenFold3AllAtom.predict_step
    completed = []

    def predict_step(self, batch, batch_idx):
        if completed:
            raise RuntimeError("expected one native query batch")
        completed.append(True)
        batch = copy_containers(batch)
        replaced = []
        for name, value in batch.items():
            if isinstance(value, torch.Tensor) and name in captured:
                saved = torch.as_tensor(captured[name], device=value.device)
                if saved.shape != value.shape or saved.dtype != value.dtype:
                    raise ValueError(f"native feature schema mismatch: {name}")
                batch[name] = saved
                replaced.append(name)
        if "ref_pos" not in replaced or "msa" not in replaced:
            raise ValueError("native input boundary not matched")
        torch.cuda.synchronize()
        save_new(
            args.out_dir / "preflight.json",
            {
                **identity,
                "shared_feature_keys": sorted(replaced),
                "config": self.config.to_dict(),
                "requested_triangle_backend": args.triangle_backend,
                "scope": "native predict_step includes confidence; excludes writer",
                "rng": "native reseed 101 per call; no tape or internal observers",
                "memory_scope": "allocator lifetime peak; not reset warm-only peak",
                "device": torch.cuda.get_device_name(),
            },
        )
        public, last, prepared = [], [], []

        def prepare():
            prepared[:] = [copy_containers(batch)]

        def observe(result):
            if result is None:
                raise RuntimeError("native predict_step returned no prediction")
            assert_shared_features_unchanged(
                {name: batch[name].detach().cpu().numpy() for name in replaced},
                captured,
            )
            _, outputs = result
            scores = outputs["confidence_scores"]
            tensors = {
                "coordinates": outputs["atom_positions_predicted"],
                **{name: scores[name] for name in ("plddt", "ptm", "iptm")},
            }
            arrays = {
                name: value.detach().cpu().numpy().copy()
                for name, value in tensors.items()
            }
            if not all(np.isfinite(value).all() for value in arrays.values()):
                raise ValueError("nonfinite native public output")
            public.append(arrays)
            if len(public) == args.warm_repeats + 1:
                last.append(result)
            return {
                "allocator": {
                    "peak_bytes_in_use": torch.cuda.max_memory_allocated(),
                    "bytes_in_use": torch.cuda.memory_allocated(),
                }
            }

        rows = measure_calls(
            lambda: original(self, prepared.pop(), batch_idx),
            lambda _: torch.cuda.synchronize(),
            observe,
            prepare=prepare,
            warm_repeats=args.warm_repeats,
        )
        archives = []
        for index, arrays in enumerate(public):
            path = args.out_dir / f"public-{index}.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            archives.append(
                {
                    "file": path.name,
                    "sha256": digest(path),
                    "equal_to_first": {
                        k: bool(np.array_equal(v, public[0][k]))
                        for k, v in arrays.items()
                    },
                }
            )
        save_new(
            args.out_dir / "measurements.json",
            {
                "calls": rows,
                "archives": archives,
                "warm_median_seconds": float(
                    np.median([r["seconds"] for r in rows[1:]])
                ),
                "shared_inputs_unchanged_after_every_call": True,
                "kernel_calls": dict(census.calls),
            },
        )
        return last[0]

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch.object(OpenFold3AllAtom, "predict_step", predict_step)
        )
        stack.enter_context(
            patch.object(OF3ProjectEntry, "get_model_config_with_update", model_config)
        )
        for active in census.patches():
            stack.enter_context(active)
        cli.main(
            args=[
                "predict",
                "--query_json",
                str(args.input),
                "--inference_ckpt_path",
                str(args.checkpoint),
                "--num_diffusion_samples",
                "5",
                "--runner_yaml",
                str(runner),
                "--use_msa_server",
                "false",
                "--use_templates",
                "false",
                "--output_dir",
                str(args.out_dir / "predictions"),
            ],
            standalone_mode=False,
        )
    if not completed or not (args.out_dir / "measurements.json").exists():
        raise RuntimeError("native warm measurement did not complete")
    if args.triangle_backend is not None and not config_calls:
        raise RuntimeError("native triangle backend configuration hook was not called")
    for relative, expected in sources.items():
        if digest(args.upstream_root / relative) != expected:
            raise RuntimeError("native source changed")
    for path, key in (
        (args.checkpoint, "checkpoint"),
        (args.input, "input"),
        (input_path, "features"),
        (Path(__file__), "harness"),
    ):
        if digest(path) != identity[key]:
            raise RuntimeError(f"native warm artifact changed: {key}")
    save_new(
        args.out_dir / "finished.json",
        {
            "measurements_sha256": digest(args.out_dir / "measurements.json"),
            "parity_admitted": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
