"""Independently capture AF3 source trees with explicitly shared binary/CCD assets."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def metadata_value(value):
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "values": metadata_value(value.tolist()),
        }
    if isinstance(value, np.generic):
        return value.item()
    if dataclasses.is_dataclass(value):
        return {
            field.name: metadata_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if hasattr(value, "to_mmcif_dict"):
        return metadata_value(dict(value.to_mmcif_dict()))
    if isinstance(value, Mapping):
        return {str(key): metadata_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [metadata_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unhandled native metadata type: {type(value)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--native-source", type=Path)
    parser.add_argument("--buckets", type=int, nargs="+")
    args = parser.parse_args()
    from foldjax.models.alphafold3.build import active_package

    runtime = active_package()
    if args.native_source is None:
        from foldjax.models.alphafold3._upstream import ensure_registered

        ensure_registered()
    else:
        source = (args.native_source / "src").resolve()
        sys.path.insert(0, str(source))
        import alphafold3

        if not Path(alphafold3.__file__).resolve().is_relative_to(source):
            raise RuntimeError("native AF3 source did not win import resolution")
        os.environ["LIBCIFPP_DATA_DIR"] = str(runtime.parent / "share/libcifpp")
        cpp = next(runtime.glob("cpp.*.so"))
        spec = importlib.util.spec_from_file_location("alphafold3.cpp", cpp)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules["alphafold3.cpp"] = module
        from alphafold3.common import resources

        # Only generated assets are shared; all Python modules stay native.
        resources.ROOT = runtime
        resources._DATA_ROOT = runtime
    from alphafold3.common import folding_input
    from alphafold3.constants import chemical_components
    from alphafold3.data import featurisation

    jobs = list(folding_input.load_fold_inputs_from_path(args.input))
    if len(jobs) != 1 or len(jobs[0].rng_seeds) != 1:
        raise ValueError("capture requires exactly one job and one seed")
    features = featurisation.featurise_input(
        jobs[0],
        chemical_components.Ccd(user_ccd=jobs[0].user_ccd),
        buckets=args.buckets,
    )[0]
    arrays, metadata = {}, {}
    for key, value in features.items():
        if isinstance(value, np.ndarray) and value.dtype != object:
            arrays[key] = value
        else:
            metadata[key] = metadata_value(value)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    args.out.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "numeric_or_string_features": len(arrays),
                "metadata_features": len(metadata),
                "seed": jobs[0].rng_seeds[0],
                "shared_binary_and_ccd_assets": True,
                "buckets": args.buckets,
                "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
