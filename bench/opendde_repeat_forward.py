"""Three synchronized OpenDDE forwards at a fixed public or compiled boundary.

Use the arguments of ``python -m bench.opendde_closure_capture foldjax`` with
this module instead. The first raw result is returned unchanged to that capture.
``--repeat-boundary compiled`` normalizes inputs once before repeating the pool
dispatch; the default ``public`` repeats the underlying ``cli._predict`` call.
This is a same-process diagnostic, not a performance run or parity admission.
Host-side JIT-owner observations do not prove runtime-executable identity.
"""

import argparse
import hashlib
import inspect
import json
from contextlib import ExitStack, contextmanager
from functools import wraps
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bench.af3_closure_capture import flatten, save, sha

TAPE_NAMES = (
    "init_noise",
    "step_noises",
    "rotations",
    "translations",
    "cycle_msa_features",
)


def compare_arrays(reference, candidate):
    """Compare every leaf, without tolerances or shape/dtype coercion."""
    leaves = {}
    for name in sorted(reference.keys() & candidate.keys()):
        a, b = np.asarray(reference[name]), np.asarray(candidate[name])
        shape_equal, dtype_equal = a.shape == b.shape, a.dtype == b.dtype
        finite_a, finite_b = np.isfinite(a), np.isfinite(b)
        finite = bool(finite_a.all() and finite_b.all())
        equal = shape_equal and dtype_equal and a.tobytes() == b.tobytes()
        metrics = {"max_abs": None, "rmse": None}
        if shape_equal and dtype_equal and finite:
            dtype = np.complex128 if a.dtype.kind == "c" else np.float64
            delta = np.abs(a.astype(dtype) - b.astype(dtype))
            metrics = {
                "max_abs": float(delta.max(initial=0)),
                "rmse": float(np.sqrt(np.mean(delta**2))) if delta.size else 0.0,
            }
        leaves[name] = {
            "reference_shape": list(a.shape),
            "candidate_shape": list(b.shape),
            "reference_dtype": str(a.dtype),
            "candidate_dtype": str(b.dtype),
            "shape_equal": shape_equal,
            "dtype_equal": dtype_equal,
            "reference_nonfinite": int(a.size - finite_a.sum()),
            "candidate_nonfinite": int(b.size - finite_b.sum()),
            "finite": finite,
            "bitwise_equal": equal,
            **metrics,
        }
    missing = sorted(reference.keys() - candidate.keys())
    extra = sorted(candidate.keys() - reference.keys())
    complete = bool(leaves) and not missing and not extra
    return {
        "reference_leaves": len(reference),
        "candidate_leaves": len(candidate),
        "missing": missing,
        "extra": extra,
        "leaves": leaves,
        "all_bitwise_equal": complete
        and all(leaf["bitwise_equal"] for leaf in leaves.values()),
        "all_finite_bitwise_equal": complete
        and all(leaf["finite"] and leaf["bitwise_equal"] for leaf in leaves.values()),
    }


def _structure(value):
    if isinstance(value, dict):
        if any(not isinstance(key, str) or "." in key for key in value):
            raise ValueError("raw/tape dictionary keys must be unambiguous strings")
        return ["dict", [[key, _structure(value[key])] for key in sorted(value)]]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [_structure(child) for child in value]]
    return "leaf"


def _object_identity(value):
    if isinstance(value, dict):
        children = tuple((key, _object_identity(value[key])) for key in sorted(value))
    elif isinstance(value, (tuple, list)):
        children = tuple(_object_identity(child) for child in value)
    else:
        children = ()
    return id(value), children


def tape_digest(kwargs, to_host):
    """Hash only explicit stochastic inputs; never copy weights or full features."""
    tree = {name: kwargs[name] for name in TAPE_NAMES}
    if any(value is None for value in tree.values()):
        raise ValueError(
            "repeat control requires complete explicit tape and MSA cycles"
        )
    return _tree_digest(tree, to_host)


def _tree_digest(tree, to_host):
    host = to_host(tree)
    digest = hashlib.sha256(json.dumps(_structure(host)).encode())
    for name, value in sorted(flatten(host).items()):
        digest.update(json.dumps([name, str(value.dtype), value.shape]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def normalized_inputs(args, kwargs):
    """Select the pinned graph signature's non-weight dynamic input arrays."""
    if len(args) != 3 or "key" not in kwargs:
        raise ValueError(
            "compiled boundary requires features, parameters, schedule/key"
        )
    return {
        "features": args[0],
        "noise_schedule": args[2],
        "key": kwargs["key"],
        **{name: kwargs[name] for name in TAPE_NAMES},
    }


@contextmanager
def observe_jit_owner(pool):
    """Observe actual host dispatch leases, returning each real owner unchanged."""
    acquire = pool._acquire
    records, owners = [], []

    def observed(identity):
        owner = acquire(identity)
        owners.append(owner)
        records.append(
            {
                "pool_id": id(pool),
                "owner_id": id(owner),
                "identity_sha256": hashlib.sha256(repr(identity).encode()).hexdigest(),
                "owner_cache_size_before": int(owner._cache_size()),
            }
        )
        return owner

    with patch.object(pool, "_acquire", observed):
        yield records
        for record, owner in zip(records, owners, strict=True):
            record["owner_cache_size_after"] = int(owner._cache_size())


def owner_evidence(runs, boundary="public"):
    records = [run["jit_dispatches"] for run in runs]
    single = len(records) == 3 and all(len(group) == 1 for group in records)
    owners = {
        (record["pool_id"], record["owner_id"], record["identity_sha256"])
        for group in records
        for record in group
    }
    same = single and len(owners) == 1
    warm = same and all(
        records[i][0]["owner_cache_size_before"]
        == records[i][0]["owner_cache_size_after"]
        == 1
        for i in (1, 2)
    )
    return {
        "same_jit_owner_observed": same,
        "warm_single_entry_owner_observed_on_repeats": warm,
        "runtime_executable_identity_verified": False,
        "limitation": (
            "Owner object ids and cache sizes are host-dispatch evidence only; "
            "no runtime executable handle or binary was inspected. "
        )
        + (
            "_predict rebuilds normalized features, schedule and key per repeat."
            if boundary == "public"
            else "Normalized argument objects are reused directly at pool dispatch."
        ),
    }


def repeat_forward(
    infer, args, kwargs, *, out, synchronize, to_host, observe, boundary="public"
):
    """Reuse argument values three times and return the first original object."""
    if boundary not in ("public", "compiled"):
        raise ValueError("unknown repeat boundary")
    argument_ids = _object_identity(args), _object_identity(kwargs)
    tape_ids = {name: _object_identity(kwargs[name]) for name in TAPE_NAMES}
    expected_digest = tape_digest(kwargs, to_host)
    expected_normalized = (
        _tree_digest(normalized_inputs(args, kwargs), to_host)
        if boundary == "compiled"
        else None
    )
    runs, arrays, structures = [], [], []
    first = None
    for index in range(3):
        before = tape_digest(kwargs, to_host)
        if before != expected_digest:
            raise RuntimeError("explicit tape changed before repeated forward")
        normalized_before = (
            _tree_digest(normalized_inputs(args, kwargs), to_host)
            if boundary == "compiled"
            else None
        )
        if normalized_before != expected_normalized:
            raise RuntimeError("normalized input values changed before forward")
        with observe() as dispatches:
            result = infer(*args, **kwargs)
            synchronize(result)
        after = tape_digest(kwargs, to_host)
        if after != expected_digest or tape_ids != {
            name: _object_identity(kwargs[name]) for name in TAPE_NAMES
        }:
            raise RuntimeError("explicit tape values or objects mutated during forward")
        if argument_ids != (
            _object_identity(args),
            _object_identity(kwargs),
        ):
            raise RuntimeError("prediction argument objects changed during forward")
        normalized_after = (
            _tree_digest(normalized_inputs(args, kwargs), to_host)
            if boundary == "compiled"
            else None
        )
        if normalized_after != expected_normalized:
            raise RuntimeError("normalized input values mutated during forward")
        host = to_host(result)
        structures.append(_structure(host))
        snapshot = {name: value.copy() for name, value in flatten(host).items()}
        if not snapshot:
            raise ValueError("repeated forward returned no raw output leaves")
        arrays.append(snapshot)
        path = out / f"repeat-forward-{index + 1}.npz"
        np.savez(path, **snapshot)
        runs.append(
            {
                "repeat": index + 1,
                "raw_file": path.name,
                "raw_sha256": sha(path),
                "tape_sha256_before": before,
                "tape_sha256_after": after,
                "normalized_input_sha256_before": normalized_before,
                "normalized_input_sha256_after": normalized_after,
                "jit_dispatches": dispatches,
            }
        )
        if index == 0:
            first = result
        del result, host
    pairs = {}
    for i, j in combinations(range(3), 2):
        comparison = compare_arrays(arrays[i], arrays[j])
        comparison["structure_equal"] = structures[i] == structures[j]
        comparison["all_finite_bitwise_equal"] &= comparison["structure_equal"]
        comparison["all_bitwise_equal"] &= comparison["structure_equal"]
        pairs[f"{i + 1}_vs_{j + 1}"] = comparison
    evidence = {
        "scope": "same-process raw forward diagnostic; no closure or performance claim",
        "repetitions": 3,
        "repeat_boundary": boundary,
        "same_argument_value_objects": True,
        "same_normalized_argument_objects": True if boundary == "compiled" else None,
        "normalized_inputs_unchanged": True if boundary == "compiled" else None,
        "normalized_input_fields": (
            list(normalized_inputs(args, kwargs)) if boundary == "compiled" else []
        ),
        "weight_bytes_hashed": False,
        "argument_mapping_note": (
            "The same args tuple and kwargs mapping are reused at this call site. "
            "Python **kwargs recreates the callee mapping, retaining value objects."
        ),
        "explicit_tapes_unchanged": True,
        "tape_fields": list(TAPE_NAMES),
        "first_result_returned_unchanged": True,
        "runs": runs,
        "pairwise": pairs,
        "all_finite_bitwise_equal": all(
            value["all_finite_bitwise_equal"] for value in pairs.values()
        ),
        "compiled_owner_evidence": owner_evidence(runs, boundary),
        "wrapper_sha256": sha(Path(__file__)),
    }
    save(out / "repeat-forward.json", evidence)
    return first


def repeat_compiled_forward(infer, args, kwargs, *, pool, **repeat_options):
    """Normalize once, intercept only this pool, and preserve all other dispatches."""
    original_call = type(pool).__call__
    seen = []

    def dispatch(*a, **kw):
        return original_call(pool, *a, **kw)

    def intercepted(self, *a, **kw):
        if self is not pool:
            return original_call(self, *a, **kw)
        if seen:
            raise RuntimeError("compiled repeat expects exactly one target pool call")
        seen.append(True)
        return repeat_forward(dispatch, a, kw, boundary="compiled", **repeat_options)

    with patch.object(type(pool), "__call__", intercepted):
        result = infer(*args, **kwargs)
    if not seen:
        raise RuntimeError("compiled repeat did not observe the target pool")
    return result


def validate_control(parser, args):
    if args.arm != "foldjax":
        parser.error("repeat-forward supports only the foldjax arm")
    if args.capture_consumed_tape or args.capture_trunk_boundary:
        parser.error(
            "repeat-forward forbids consumed-tape and trunk-boundary observers"
        )


def main():
    from bench import opendde_closure_capture as capture

    original_parse = argparse.ArgumentParser.parse_args
    with ExitStack() as stack:

        def checked_parse(parser, *args, **kwargs):
            # Reuse capture's real parser, including its abbreviation rules, and
            # install before it binds cli._predict as its local replay target.
            parser.add_argument(
                "--repeat-boundary", choices=("public", "compiled"), default="public"
            )
            arguments = original_parse(parser, *args, **kwargs)
            validate_control(parser, arguments)
            stack.enter_context(
                patch.object(argparse.ArgumentParser, "parse_args", original_parse)
            )
            from foldjax.models.opendde.cli import predict as cli

            infer = cli._predict
            seen = []

            @wraps(infer)
            def repeated(*a, **kw):
                if seen:
                    raise RuntimeError("repeat control expects one capture prediction")
                seen.append(True)
                import jax

                from foldjax.models.opendde.models import model

                pool = model._compiled_opendde_infer

                def synchronize(result):
                    jax.block_until_ready(result)
                    jax.effects_barrier()

                source = {
                    "cli_predict": infer,
                    "compiled_predict": model.opendde_infer_compiled,
                    "jit_pool_dispatch": type(pool).__call__,
                }
                save(
                    arguments.out / "repeat-forward-source.json",
                    {
                        "repeat_boundary": arguments.repeat_boundary,
                        **{
                            name: {
                                "qualname": fn.__qualname__,
                                "line": inspect.getsourcelines(fn)[1],
                                "source_sha256": sha(Path(inspect.getsourcefile(fn))),
                            }
                            for name, fn in source.items()
                        },
                    },
                )
                repeat = (
                    repeat_forward
                    if arguments.repeat_boundary == "public"
                    else repeat_compiled_forward
                )
                return repeat(
                    infer,
                    a,
                    kw,
                    out=arguments.out,
                    synchronize=synchronize,
                    to_host=jax.device_get,
                    observe=lambda: observe_jit_owner(pool),
                    **(
                        {"pool": pool}
                        if arguments.repeat_boundary == "compiled"
                        else {}
                    ),
                )

            stack.enter_context(patch.object(cli, "_predict", repeated))
            return arguments

        stack.enter_context(
            patch.object(argparse.ArgumentParser, "parse_args", checked_parse)
        )
        capture.main()


if __name__ == "__main__":
    main()
