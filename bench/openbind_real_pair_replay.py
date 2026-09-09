"""Teacher-forced first PairBlock diagnostic, never full-model admission."""

import argparse
import importlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from bench.boltz_historical_replay import digest, save_new, source_hashes
from bench.openbind_core_replay import positive_count
from bench.openbind_tape_adapter import capture_provenance


def map_private_samples(fn, z, mask):
    """Keep private B=1 kernel arithmetic while mapping independent samples."""
    import jax

    if z.ndim != 4 or mask.shape != z.shape[:-1] or z.shape[0] < 1:
        raise ValueError(f"invalid private batch shapes: {z.shape}, {mask.shape}")
    if z.shape[0] == 1:
        return fn(z, mask)
    return jax.lax.map(lambda one: fn(one[0][None], one[1][None])[0], (z, mask))


@contextmanager
def private_pair_operators():
    """Trace native-faithful kernels through the existing block's residual flow."""
    block = importlib.import_module("foldjax.models.openfold3.models.pair_block")
    from foldjax.models.openfold3.models import native_triangle_ops as ops

    original_mul, original_att = block.tri_mul_out_in, block.triangle_attention

    def multiplication(z, params, *, pair_mask, eps=1e-5):
        def one(z, mask):
            z = ops.native_triangle_multiplication_residual(
                z, params.tri_mul_out, outgoing=True, mask=mask, eps=eps
            )
            return ops.native_triangle_multiplication_residual(
                z, params.tri_mul_in, outgoing=False, mask=mask, eps=eps
            )

        return map_private_samples(one, z, pair_mask)

    def attention(z, params, *, no_heads, mask, chunk_size=None, **kwargs):
        if no_heads != 4:
            raise ValueError("private native attention requires four heads")
        return map_private_samples(
            lambda z, mask: ops.native_triangle_attention_update(
                z, params, mask=mask, chunk_size=chunk_size or 1024, **kwargs
            ),
            z,
            mask,
        )

    try:
        block.tri_mul_out_in, block.triangle_attention = multiplication, attention
        yield
    finally:
        block.tri_mul_out_in, block.triangle_attention = original_mul, original_att


def load_pair_capture(root):
    trace = json.loads((root / "trace.json").read_text())
    records = trace["first_pair_blocks"]
    if len(records) != 1:
        raise ValueError("expected one observed pair block")
    record = records[0]
    if record["module"] != "pairformer_stack.blocks.0.pair_stack":
        raise ValueError("unexpected pair block identity")
    values = []
    for side in ("input", "output"):
        path = root / f"first-pair-{side}.npz"
        identity = record[side]
        if (
            digest(path) != identity["sha256"]
            or path.stat().st_size != identity["bytes"]
        ):
            raise ValueError("pair activation identity mismatch")
        with np.load(path, allow_pickle=False) as archive:
            values.append(dict(archive))
    z, mask, expected = values[0]["z"], values[0]["pair_mask"], values[1]["z"]
    if (
        z.ndim != 4
        or z.shape[0] != 1
        or z.shape[1] != z.shape[2]
        or z.shape[-1] != 128
        or expected.shape != z.shape
        or mask.shape != z.shape[:-1]
        or any(
            a.dtype != np.float32 or not np.isfinite(a).all()
            for a in (z, mask, expected)
        )
        or not np.isin(mask, (0, 1)).all()
    ):
        raise ValueError("invalid real pair activation")
    return z, mask, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "checkpoint", "source-root", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--chunk-size", type=positive_count)
    parser.add_argument("--boundaries", action="store_true")
    parser.add_argument(
        "--block-backend",
        choices=("xla", "native-private", "source-native-private"),
        default="xla",
    )
    parser.add_argument(
        "--boundary-backend", choices=("xla", "native-private"), default="xla"
    )
    args = parser.parse_args()
    if args.boundary_backend != "xla" and not args.boundaries:
        parser.error("--boundary-backend requires --boundaries")
    z, mask, expected = load_pair_capture(args.capture)
    checkpoint_hash = digest(args.checkpoint)
    provenance = capture_provenance(args.capture, checkpoint_hash)
    source = source_hashes(args.source_root)
    import jax

    from foldjax._openfold3_compile import triangle_backend
    from foldjax.models.openfold3.bridge.checkpoint import load_checkpoint
    from foldjax.models.openfold3.bridge.torch_mapping import (
        map_pair_block,
        resolve_model_prefix,
    )
    from foldjax.models.openfold3.models.pair_block import pair_block

    state = load_checkpoint(args.checkpoint)
    prefix = resolve_model_prefix(state, None)
    block_prefix = ".".join(
        filter(None, (prefix, "pairformer_stack.blocks.0.pair_stack"))
    )
    params = map_pair_block(state, block_prefix)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    save_new(
        args.out_dir / "preflight.json",
        {
            "scope": "native first-block activation injection; XLA vs native Triton",
            "capture": provenance,
            "checkpoint_sha256": checkpoint_hash,
            "source": source,
            "runner_sha256": digest(Path(__file__)),
            "matmul_precision": "high",
            "no_heads_pair": 4,
            "chunk_size": args.chunk_size,
            "boundary_backend": args.boundary_backend,
            "block_backend": args.block_backend,
        },
    )

    def forward(z, mask, params):
        selected = (
            "native-private" if args.block_backend == "source-native-private" else "xla"
        )
        with triangle_backend(selected), jax.default_matmul_precision("high"):
            if args.block_backend == "native-private":
                with private_pair_operators():
                    return pair_block(
                        z,
                        params,
                        pair_mask=mask,
                        no_heads_pair=4,
                        chunk_size=args.chunk_size,
                    )
            return pair_block(
                z, params, pair_mask=mask, no_heads_pair=4, chunk_size=args.chunk_size
            )

    run = jax.jit(forward)
    outputs = [np.asarray(run(z, mask, params)) for _ in range(3)]
    if not all(np.isfinite(a).all() for a in outputs):
        raise ValueError("nonfinite pair output")
    with (args.out_dir / "outputs.npz").open("xb") as stream:
        np.savez_compressed(stream, **{str(i): a for i, a in enumerate(outputs)})
    delta = outputs[0].astype(np.float64) - expected.astype(np.float64)
    if source_hashes(args.source_root) != source:
        raise ValueError("source changed during pair replay")
    save_new(
        args.out_dir / "report.json",
        {
            "max_absolute_error": float(np.abs(delta).max()),
            "rmse": float(np.sqrt(np.mean(delta**2))),
            "repeat_equal": [bool(np.array_equal(outputs[0], a)) for a in outputs[1:]],
            "outputs_sha256": digest(args.out_dir / "outputs.npz"),
            "model_admitted": False,
        },
    )
    if args.boundaries:
        from foldjax.models.openfold3.models.primitives import swiglu_transition
        from foldjax.models.openfold3.models.triangle import triangle_multiplication
        from foldjax.models.openfold3.models.triangle_attention import (
            triangle_attention,
        )

        records = provenance["trace"]["pair_boundaries"]
        names = (
            "tri_mul_out",
            "tri_mul_in",
            "tri_att_start",
            "tri_att_end",
            "pair_transition",
        )
        if [r["name"] for r in records] != list(names):
            raise ValueError("incomplete or reordered pair boundaries")
        reports = {}
        for record in records:
            name = record["name"]
            arrays = []
            for side in ("input", "output"):
                path = args.capture / f"pair-{name}-{side}.npz"
                identity = record[side]
                if (
                    digest(path) != identity["sha256"]
                    or path.stat().st_size != identity["bytes"]
                ):
                    raise ValueError("pair boundary identity mismatch")
                with np.load(path, allow_pickle=False) as archive:
                    arrays.append(dict(archive))
            boundary_z, boundary_mask = arrays[0]["z"], arrays[0]["mask"]
            boundary_expected = arrays[1]["z"]
            options = record["scalar_kwargs"]

            def operation(z, mask, weights):
                with triangle_backend("xla"), jax.default_matmul_precision("high"):
                    if args.boundary_backend == "native-private":
                        from foldjax.models.openfold3.models import native_triangle_ops

                        ops = native_triangle_ops

                        if name.startswith("tri_mul"):
                            residual = options.get("inplace_safe") and options.get(
                                "_add_with_inplace"
                            )
                            fn = (
                                ops.native_triangle_multiplication_residual
                                if residual
                                else ops.native_triangle_multiplication_update
                            )
                            return fn(
                                z, weights, mask=mask, outgoing=name == "tri_mul_out"
                            )
                        if name.startswith("tri_att"):
                            # Child inputs already have ending-node orientation.
                            return native_triangle_ops.native_triangle_attention_update(
                                z,
                                weights,
                                mask=mask,
                                transpose_bias=options.get("transpose_bias", False),
                            )
                        if name == "pair_transition":
                            return ops.native_swiglu_transition_update(
                                z, weights, mask=mask
                            )
                    if name.startswith("tri_mul"):
                        update = triangle_multiplication(
                            z, weights, mask=mask, outgoing=name == "tri_mul_out"
                        )
                        if options.get("inplace_safe") and options.get(
                            "_add_with_inplace"
                        ):
                            return z + update
                        return update
                    if name.startswith("tri_att"):
                        return triangle_attention(
                            z,
                            weights,
                            mask=mask,
                            no_heads=4,
                            transpose_bias=options.get("transpose_bias", False),
                            chunk_size=options.get("chunk_size"),
                        )
                    return swiglu_transition(z, weights, mask=mask)

            boundary_run = jax.jit(operation)
            repeats = [
                np.asarray(
                    boundary_run(boundary_z, boundary_mask, getattr(params, name))
                )
                for _ in range(3)
            ]
            measured = repeats[0]
            if (
                measured.shape != boundary_expected.shape
                or not np.isfinite(measured).all()
            ):
                raise ValueError("invalid pair boundary result")
            delta = measured.astype(np.float64) - boundary_expected.astype(np.float64)
            reports[name] = {
                "max_absolute_error": float(np.abs(delta).max()),
                "rmse": float(np.sqrt(np.mean(delta**2))),
            }
            with (args.out_dir / f"{name}.npz").open("xb") as stream:
                np.savez_compressed(stream, output=measured)
            reports[name]["outputs_sha256"] = digest(args.out_dir / f"{name}.npz")
            reports[name]["repeat_equal"] = [
                bool(np.array_equal(measured, value)) for value in repeats[1:]
            ]
        if source_hashes(args.source_root) != source:
            raise ValueError("source changed during boundary replay")
        save_new(
            args.out_dir / "boundaries.json",
            {
                "scope": "native teacher inputs; per-operator errors, not accumulated",
                "boundary_backend": args.boundary_backend,
                "model_admitted": False,
                "operators": reports,
            },
        )


if __name__ == "__main__":
    main()
