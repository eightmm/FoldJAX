"""Turn a query specification into a feature ``.npz`` using NumPy preprocessing.

Chemistry, tokenization, MSA, and local-template preprocessing run without PyTorch
or Lightning. Running this once produces a portable, pickle-free archive for JAX
inference.

    openfold3-jax-featurize query.json -o ubq.npz
    openfold3-jax-featurize query.json -o ubq.npz --query-id 7cnx \\
        --pad-tokens 512 --pad-atoms 4096
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openfold3-jax-featurize",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("spec", type=Path, help="query JSON")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--query-id", default=None, help="required when the file holds several"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="deterministic preprocessing seed (default: 0)",
    )
    parser.add_argument(
        "--ccd", type=Path, default=None, help="CCD file; biotite's copy by default"
    )
    parser.add_argument(
        "--pad-tokens",
        type=int,
        default=None,
        help="pad to this token count so one compiled predict serves many targets",
    )
    parser.add_argument("--pad-atoms", type=int, default=None)
    parser.add_argument(
        "--msa-server",
        action="store_true",
        help="search alignments with the public ColabFold MMseqs2 server "
        "(needs the 'msa' extra and network access)",
    )
    parser.add_argument(
        "--alignment-dir",
        type=Path,
        default=None,
        help="where --msa-server writes alignments; defaults to <output>.msa",
    )
    parser.add_argument(
        "--paired-msa",
        action="store_true",
        help="also use the paired alignment. Only for multimers, and only when the "
        "search returns taxonomy-annotated headers: an unpairable alignment "
        "silently collapses the MSA to the query sequence alone",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if (args.pad_tokens is None) != (args.pad_atoms is None):
        parser.error("--pad-tokens and --pad-atoms must be given together")

    import json

    from foldjax.models.openfold3.data import (
        featurize_query_with_metadata,
        pad_features,
        save_features,
    )

    spec: object = args.spec
    if args.msa_server:
        from foldjax.models.openfold3.data import attach_msas

        alignment_dir = args.alignment_dir or args.output.with_suffix(".msa")
        print(f"searching alignments into {alignment_dir} ...")
        spec = attach_msas(
            json.loads(args.spec.read_text()),
            alignment_dir=alignment_dir,
            paired=args.paired_msa,
            query_id=args.query_id,
        )

    features, output_metadata = featurize_query_with_metadata(
        spec, query_id=args.query_id, seed=args.seed, ccd_file_path=args.ccd
    )
    tokens = features["token_mask"].shape[-1]
    atoms = features["atom_mask"].shape[-1]
    msa_rows = features["msa"].shape[1]
    print(f"featurized {tokens} tokens, {atoms} atoms, {msa_rows} MSA rows")
    if msa_rows == 1:
        print(
            "  only the query sequence is in the MSA -- accuracy will be much lower "
            "than with alignments. Add main_msa_file_paths to the chains."
        )

    if args.pad_tokens is not None:
        features = pad_features(
            features, n_token=args.pad_tokens, n_atom=args.pad_atoms
        )
        print(f"padded to {args.pad_tokens} tokens, {args.pad_atoms} atoms")

    path = save_features(features, args.output, output_metadata=output_metadata)
    print(f"wrote {path}")
    print(
        "  run it with: released_config(n_token="
        f"{features['token_mask'].shape[-1]}, n_atom="
        f"{features['atom_mask'].shape[-1]})"
    )
    return 0


def entrypoint() -> None:
    raise SystemExit(main())

if __name__ == "__main__":  # ``python -m foldjax.models.openfold3.cli.<name>``
    entrypoint()
