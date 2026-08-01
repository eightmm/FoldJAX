"""Featurize every OpenDDE JSON job under a directory and record a matrix."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def _load_jobs(path: Path) -> list[dict[str, Any]]:
    from foldjax.models.opendde.data.featurize_json import load_jobs

    return load_jobs(path)


def _featurize(job: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    from foldjax.models.opendde.data.featurize_json import featurize_opendde_json

    return featurize_opendde_json(job, **kwargs)


def _seed(job: dict[str, Any]) -> int:
    from foldjax.models.opendde.data.featurize_json import _resolve_seed

    return _resolve_seed(job, None)


def _shape(value: Any) -> list[int]:
    return list(np.asarray(value).shape)


def _job_summary(
    path: Path,
    root: Path,
    job_index: int,
    job: dict[str, Any],
    features: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    checked = ["restype", "ref_pos", "msa"]
    if "template_atom_positions" in features:
        checked.extend(("template_atom_positions", "template_atom_mask"))
    all_finite = all(
        bool(np.isfinite(np.asarray(features[name])).all()) for name in checked
    )
    summary: dict[str, Any] = {
        "source": str(path.relative_to(root)),
        "job_index": job_index,
        "name": str(job.get("name") or path.stem),
        "seed": seed,
        "n_token": int(np.asarray(features["restype"]).shape[0]),
        "n_atom": int(np.asarray(features["ref_pos"]).shape[0]),
        "n_structural": int(np.asarray(features["parent_residue_idx"]).shape[0]),
        "msa_shape": _shape(features["msa"]),
        "all_finite": all_finite,
    }
    if "template_atom_mask" in features:
        mask = np.asarray(features["template_atom_mask"])
        summary["template_shape"] = _shape(features["template_atom_positions"])
        summary["template_mask_sums"] = [
            int(value) for value in mask.sum(axis=tuple(range(1, mask.ndim)))
        ]
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-queries", type=int, default=32)
    parser.add_argument("--n-keys", type=int, default=128)
    parser.add_argument("--max-msa-rows", type=int, default=16384)
    args = parser.parse_args(argv)

    root = args.input_root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"missing input directory: {root}")
    paths = sorted(root.rglob("*.json"))
    if not paths:
        raise SystemExit(f"no JSON inputs found under: {root}")

    summaries = []
    for path in paths:
        jobs = _load_jobs(path)
        for job_index, job in enumerate(jobs):
            seed = _seed(job)
            features = _featurize(
                job,
                base_dir=path.parent,
                seed=seed,
                n_queries=args.n_queries,
                n_keys=args.n_keys,
                max_msa_rows=args.max_msa_rows,
                augment_reference=False,
            )
            summary = _job_summary(path, root, job_index, job, features, seed)
            if not summary["all_finite"]:
                raise RuntimeError(
                    f"non-finite features for {summary['source']}:{job_index}"
                )
            summaries.append(summary)

    report = {
        "input_root": str(root),
        "n_files": len(paths),
        "n_jobs": len(summaries),
        "all_finite": all(job["all_finite"] for job in summaries),
        "torch_imported": "torch" in sys.modules,
        "jobs": summaries,
    }
    output = args.out.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
