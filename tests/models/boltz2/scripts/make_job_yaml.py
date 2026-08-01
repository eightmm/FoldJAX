"""CLI to build a Boltz-2 job YAML from bare sequences / ligand codes.

Thin wrapper over ``foldjax.models.boltz2.data.job_yaml.build_job_yaml``.

  uv run python scripts/make_job_yaml.py --seq MKQLED... --ligand-ccd ATP \
    --out job.yaml
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from foldjax.models.boltz2.data.job_yaml import build_job_yaml


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seq", action="append", default=[], help="protein sequence (repeatable)"
    )
    p.add_argument("--dna", action="append", default=[], help="DNA sequence")
    p.add_argument("--rna", action="append", default=[], help="RNA sequence")
    p.add_argument(
        "--ligand-ccd", action="append", default=[], help="ligand CCD code"
    )
    p.add_argument(
        "--ligand-smiles", action="append", default=[], help="ligand SMILES"
    )
    p.add_argument("--use-msa-server", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    text = build_job_yaml(
        args.seq,
        args.ligand_ccd,
        dna=args.dna,
        rna=args.rna,
        ligands_smiles=args.ligand_smiles,
        use_msa_server=args.use_msa_server,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
