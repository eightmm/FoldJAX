"""Extend an existing native RNG observer with raw outputs and trunk arrays.

Diagnostic synchronization changes execution context; an observer bridge is
required before this capture can establish ordinary execution equivalence.
"""

import argparse
import copy
import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np


def triangle_backend_update(custom, backend):
    if backend not in ("cueq", "triton", "xla"):
        raise ValueError(f"unsupported native triangle backend: {backend}")
    result = copy.deepcopy(custom)
    memory = result.setdefault("settings", {}).setdefault("memory", {}).setdefault(
        "eval", {}
    )
    memory["use_cueq_triangle_kernels"] = backend == "cueq"
    memory["use_triton_triangle_kernels"] = backend == "triton"
    return result


def configured_backend(original, update, backend):
    if backend is None:
        return original(update)
    update = update.model_copy(update={
        "custom": triangle_backend_update(update.custom, backend)
    })
    config = original(update)
    memory = config["settings"]["memory"]["eval"]
    expected = (backend == "cueq", backend == "triton")
    actual = (memory["use_cueq_triangle_kernels"],
              memory["use_triton_triangle_kernels"])
    if actual != expected:
        raise ValueError("native triangle backend override was not applied")
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-wrapper", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--triangle-backend", choices=("cueq", "triton", "xla"))
    parser.add_argument("--capture-first-pair-block", action="store_true")
    parser.add_argument("--capture-pair-boundaries", action="store_true")
    args = parser.parse_args(argv)
    if args.capture_pair_boundaries and not args.capture_first_pair_block:
        parser.error("--capture-pair-boundaries requires --capture-first-pair-block")
    spec = importlib.util.spec_from_file_location("native_capture", args.native_wrapper)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    from openfold3.core.utils.chunk_utils import ChunkSizeTuner
    from openfold3.projects.of3_all_atom.model import OpenFold3
    from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry

    original_config = OF3ProjectEntry.get_model_config_with_update
    config_calls = []

    def model_config(self, update):
        config = configured_backend(
            lambda value: original_config(self, value), update, args.triangle_backend
        )
        config_calls.append(True)
        return config

    original_forward = OpenFold3.forward
    original_trunk = OpenFold3.run_trunk
    original_tune = ChunkSizeTuner.tune_chunk_size
    names, chunks, trunks, pair_blocks, boundaries = {}, [], [], [], []

    def tune(self, representative_fn, args, max_chunk_size):
        value = original_tune(self, representative_fn, args, max_chunk_size)
        chunks.append({"module": names.get(id(self), "unknown"),
                       "cap": max_chunk_size, "chosen": value})
        return value

    def trunk(self, *positional, **keyword):
        for name, module in self.named_modules():
            tuner = getattr(module, "chunk_size_tuner", None)
            if tuner is not None:
                names[id(tuner)] = name
        result = original_trunk(self, *positional, **keyword)
        path = args.out / f"trunk-{len(trunks):02d}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **native.flatten(result))
        trunks.append({"file": path.name, "sha256": native.driver.digest_file(path)})
        return result

    def forward(self, batch):
        handles = []
        observed = False

        def before_pair(module, positional, keyword):
            if observed:
                return
            z = keyword.get("z", positional[0] if positional else None)
            if z is None or "pair_mask" not in keyword:
                raise ValueError("first pair block requires z and pair_mask")
            path = args.out / "first-pair-input.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, **native.flatten({
                    "z": z, "pair_mask": keyword["pair_mask"],
                }))

        def after_pair(module, positional, keyword, output):
            nonlocal observed
            if observed:
                return
            path = args.out / "first-pair-output.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, **native.flatten({"z": output}))
            pair_blocks.append({
                "module": "pairformer_stack.blocks.0.pair_stack",
                "input": native.driver.digest_file(args.out / "first-pair-input.npz"),
                "output": native.driver.digest_file(path),
                "scope": (
                    "first invocation only; observer synchronization changes execution"
                ),
            })
            observed = True

        if args.capture_first_pair_block:
            block = self.pairformer_stack.blocks[0].pair_stack
            handles = [block.register_forward_pre_hook(before_pair, with_kwargs=True),
                       block.register_forward_hook(after_pair, with_kwargs=True)]
            if args.capture_pair_boundaries:
                def attach_boundary(name):
                    def before(module, positional, keyword):
                        if observed:
                            return
                        if not positional or "mask" not in keyword:
                            raise ValueError(
                                "pair boundary requires positional z and mask"
                            )
                        path = args.out / f"pair-{name}-input.npz"
                        with path.open("xb") as stream:
                            np.savez_compressed(stream, **native.flatten({
                                "z": positional[0], "mask": keyword["mask"],
                            }))

                    def after(module, positional, keyword, output):
                        if observed:
                            return
                        path = args.out / f"pair-{name}-output.npz"
                        with path.open("xb") as stream:
                            np.savez_compressed(stream, **native.flatten({"z": output}))
                        boundaries.append({
                            "name": name,
                            "input": native.driver.digest_file(
                                args.out / f"pair-{name}-input.npz"
                            ),
                            "output": native.driver.digest_file(path),
                            "scalar_kwargs": {
                                k: v for k, v in keyword.items()
                                if v is None or isinstance(v, (bool, int, float, str))
                            },
                        })

                    child = getattr(block, name)
                    handles.extend([
                        child.register_forward_pre_hook(before, with_kwargs=True),
                        child.register_forward_hook(after, with_kwargs=True),
                    ])

                for name in ("tri_mul_out", "tri_mul_in", "tri_att_start",
                             "tri_att_end", "pair_transition"):
                    attach_boundary(name)
        try:
            result = original_forward(self, batch)
        finally:
            for handle in handles:
                handle.remove()
        with (args.out / "raw-output.npz").open("xb") as stream:
            np.savez_compressed(stream, **native.flatten(result[1]))
        return result

    with (
        patch.object(OpenFold3, "forward", forward),
        patch.object(OF3ProjectEntry, "get_model_config_with_update", model_config),
        patch.object(OpenFold3, "run_trunk", trunk),
        patch.object(ChunkSizeTuner, "tune_chunk_size", tune),
    ):
        native.capture(args.input, args.out, "32-true")
    if args.triangle_backend is not None and not config_calls:
        raise ValueError("native triangle backend configuration hook was not called")
    if args.capture_first_pair_block and not pair_blocks:
        raise ValueError("first pair block observer was not called")
    if args.capture_pair_boundaries and len(boundaries) != 5:
        raise ValueError("incomplete pair boundary capture")
    native.control.save(args.out / "trace.json", {
        "chunks": chunks, "trunk_arrays": trunks,
        "first_pair_blocks": pair_blocks,
        "pair_boundaries": boundaries,
        "requested_triangle_backend": args.triangle_backend,
        "native_source": native.driver.source_identity(native.UP),
        "wrapper": native.driver.digest_file(Path(__file__)),
        "base_wrapper": native.driver.digest_file(args.native_wrapper),
        "raw_output_sha256": native.driver.digest_file(args.out / "raw-output.npz"),
        "public_artifacts": {
            str(path.relative_to(args.out)): native.driver.digest_file(path)
            for path in [args.out / "coordinate.npz", *sorted(
                (args.out / "predictions").rglob("*confidences*.json")
            )]
        },
        "scope": "observed native outputs; no observer neutrality claim",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
