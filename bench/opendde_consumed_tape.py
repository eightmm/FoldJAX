"""Execution observers for the finite unpadded OpenDDE scan replay.

Callback instrumentation changes the compiled graph. A separate uninstrumented
structure/confidence bridge is mandatory; these runs are not performance data.
Native MSA references must be explicitly normalized to consumer storage before
construction (not silently cast by this comparator).
"""

import hashlib
import inspect
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np


class ConsumedTape:
    def __init__(self, expected):
        self.expected = expected
        self.seen = set()
        self.errors = []

    def observe(self, kind, index, values):
        index = np.asarray(index)
        if index.shape != () or index.dtype.kind not in "iu":
            self.errors.append("event index must be an integer scalar")
            return
        key = (kind, int(index))
        if key in self.seen:
            self.errors.append(f"duplicate event {key}")
            return
        self.seen.add(key)
        if key not in self.expected:
            self.errors.append(f"unexpected event {key}")
            return
        reference = self.expected[key]
        if reference.keys() != values.keys():
            self.errors.append(f"field mismatch {key}")
            return
        for name in reference:
            a, b = np.asarray(reference[name]), np.asarray(values[name])
            if (
                a.shape != b.shape
                or a.dtype != b.dtype
                or not np.isfinite(a).all()
                or not np.isfinite(b).all()
                or a.tobytes() != b.tobytes()
            ):
                self.errors.append(f"consumed value mismatch {key}:{name}")

    def finish(self):
        missing = self.expected.keys() - self.seen
        return {
            "passed": bool(self.expected) and not missing and not self.errors,
            "missing": sorted(missing),
            "errors": list(self.errors),
            "events": len(self.seen),
            "value_comparison": "finite_bitwise_bytes",
            "requires_uninstrumented_output_bridge": True,
        }


def expected_events(tape, cycles):
    """Construct exact consumer references, retaining each native cycle index."""
    shapes = {
        "noise_schedule": (201,),
        "rotations": (200, 5, 3, 3),
        "translations": (200, 5, 3),
    }
    initial = np.asarray(tape["init_noise"])
    if (
        initial.ndim != 3
        or initial.shape[0] != 5
        or initial.shape[1] == 0
        or initial.shape[-1] != 3
    ):
        raise ValueError("initial noise requires [5, atom, 3]")
    shapes.update(init_noise=initial.shape, step_noises=(200, *initial.shape))
    for name, shape in shapes.items():
        value = np.asarray(tape[name])
        if (
            value.shape != shape
            or value.dtype != np.float32
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"invalid native FP32 tape {name}")
    if len(cycles) != 10 or any(not value for value in cycles):
        raise ValueError("ten nonempty consumer-normalized MSA cycles required")
    for cycle in cycles:
        if cycle.keys() != cycles[0].keys() or any(
            np.asarray(value).dtype.kind not in "biuf" or not np.isfinite(value).all()
            for value in cycle.values()
        ):
            raise ValueError(
                "MSA cycles require identical fields and finite numeric values"
            )
    result = {("initial", 0): {"noise": initial, "schedule": tape["noise_schedule"]}}
    for i in range(200):
        result["step", i] = {
            "previous": tape["noise_schedule"][i],
            "current": tape["noise_schedule"][i + 1],
            "noise": tape["step_noises"][i],
            "rotation": tape["rotations"][i],
            "translation": tape["translations"][i],
        }
    result.update({("msa", i): values for i, values in enumerate(cycles)})
    return result


@contextmanager
def observe_consumption(recorder):
    """Install before fresh tracing; call effects_barrier before finish().

    Only this sampler's scan and this trunk's cycle scan are intercepted.
    Missing cached/unsupported routes fail through absent expected events.
    """
    import jax
    import jax.numpy as jnp

    from foldjax.models.opendde.models import model, sampling
    from foldjax.models.protenix.models.trunk_blocks import trunk

    sampler, scan, msa = sampling.sample_diffusion, jax.lax.scan, trunk.msa_module
    files = [
        Path(inspect.getsourcefile(fn))
        for fn in (sampler, trunk.pairformer_output_from_s_inputs)
    ]
    hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }
    cycle = []

    def emit(kind, index, values):
        jax.debug.callback(
            lambda i, v: recorder.observe(kind, i, v), index, values, ordered=False
        )

    def observed_sampler(*args, **kwargs):
        bound = inspect.signature(sampler).bind(*args, **kwargs)
        bound.apply_defaults()
        p = bound.arguments
        if p["atom_mask"] is not None or p["preserve_prefix_rng"] or p["batch_shape"]:
            raise ValueError("observer supports only unpadded, unbatched sample tape")
        if not p["use_scan"] or p["num_samples"] != 5 or p["dtype"] != jnp.float32:
            raise ValueError("observer requires FP32 n5 scanned sampler")
        if any(
            p[name] is None
            for name in ("init_noise", "step_noises", "rotations", "translations")
        ):
            raise ValueError("complete explicit sampler tape required")
        emit(
            "initial",
            0,
            {
                "noise": jnp.asarray(p["init_noise"], jnp.float32),
                "schedule": jnp.asarray(p["noise_schedule"], jnp.float32),
            },
        )
        return sampler(*args, **kwargs)

    def observed_scan(fn, init, xs=None, **kwargs):
        code = getattr(fn, "__code__", None)
        sampler_body = (
            code is not None
            and code.co_filename == str(files[0])
            and fn.__qualname__ == "sample_diffusion.<locals>.body"
        )
        msa_body = (
            code is not None
            and code.co_filename == str(files[1])
            and fn.__qualname__ == "pairformer_output_from_s_inputs.<locals>.one_cycle"
        )
        if not (sampler_body or msa_body):
            return scan(fn, init, xs, **kwargs)
        count = 200 if sampler_body else 10
        if kwargs.get("reverse", False):
            raise ValueError("reversed scan is outside replay contract")
        if xs is None or any(leaf.shape[0] != count for leaf in jax.tree.leaves(xs)):
            raise ValueError("unexpected consumed scan length/structure")

        def indexed(carry, pair):
            index, values = pair
            if sampler_body:
                if not isinstance(values, tuple) or len(values) != 5:
                    raise ValueError("unexpected sampler scan xs")
                emit(
                    "step",
                    index,
                    dict(
                        zip(
                            ("previous", "current", "noise", "rotation", "translation"),
                            values,
                            strict=True,
                        )
                    ),
                )
                return fn(carry, values)
            cycle.append(index)
            try:
                return fn(carry, values)
            finally:
                cycle.pop()

        return scan(indexed, init, (jnp.arange(count), xs), **kwargs)

    def observed_msa(features, *args, **kwargs):
        if cycle:
            fields = recorder.expected["msa", 0].keys()
            if fields != features.keys():
                raise ValueError(
                    "actual MSA consumer fields differ from native reference"
                )
            emit("msa", cycle[-1], {name: features[name] for name in fields})
        return msa(features, *args, **kwargs)

    with (
        patch.object(model, "sample_diffusion", observed_sampler),
        patch.object(sampling, "sample_diffusion", observed_sampler),
        patch.object(jax.lax, "scan", observed_scan),
        patch.object(trunk, "msa_module", observed_msa),
    ):
        yield hashes
        jax.effects_barrier()
    if any(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() != value
        for path, value in hashes.items()
    ):
        raise RuntimeError("observed source changed during execution")
