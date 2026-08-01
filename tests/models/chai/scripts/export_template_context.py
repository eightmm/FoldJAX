#!/usr/bin/env python3
"""Convert an official Chai preprocessing fixture into a native template asset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from foldjax.models.chai.data.templates import (
    TemplateContext,
    save_native_template_context,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--query-identity", type=Path, required=True)
    args = parser.parse_args()

    fixture_bytes = args.fixture.read_bytes()
    query_identity = json.loads(args.query_identity.read_text(encoding="utf-8"))
    if not isinstance(query_identity, dict):
        raise SystemExit("query identity JSON must contain an object")
    with np.load(args.fixture, allow_pickle=False) as archive:
        arrays = {}
        for name in (
            "template_restype",
            "template_pseudo_beta_mask",
            "template_backbone_frame_mask",
            "template_distances",
            "template_unit_vector",
        ):
            key = f"inputs/{name}"
            if key not in archive.files:
                raise SystemExit(f"official preprocessing fixture is missing {key}")
            value = np.asarray(archive[key])
            if value.shape[0] != 1:
                raise SystemExit(f"template fixture batch must be exactly one: {key}")
            arrays[name] = np.array(value[0], copy=True)
    save_native_template_context(
        TemplateContext(**arrays),
        args.destination,
        source_id=str(args.fixture.resolve()),
        source_sha256=hashlib.sha256(fixture_bytes).hexdigest(),
        query_identity=query_identity,
    )
    print(f"exported template context to {args.destination.resolve()}")


if __name__ == "__main__":
    main()
