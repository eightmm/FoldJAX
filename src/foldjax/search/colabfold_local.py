#!/usr/bin/env python3
"""Reference local MSA search for ``FOLDJAX_MSA_COMMAND``, on ColabFold's pipeline.

FoldJAX runs a configured local search as::

    <FOLDJAX_MSA_COMMAND> --input query.fasta --output DIR

and expects ``DIR/non_pairing.a3m`` and ``DIR/pairing.a3m``, each starting
with the query (`foldjax.search.msa.LocalMsaClient`). This wrapper produces
them with the same MMseqs2 steps ``colabfold_search`` runs against locally
built ColabFold databases (``setup_databases.sh``), so nothing leaves the
machine:

- ``non_pairing.a3m``: ``mmseqs_search_monomer`` -- UniRef30 and, by default,
  the environmental database, merged as the server's ``env`` ticket merges
  them;
- ``pairing.a3m``: ``mmseqs_search_pair`` over UniRef for this one query,
  what the server's single-query ``paircomplete`` ticket returns -- or the
  query alone when FoldJAX sets ``FOLDJAX_MSA_PAIRING=none``
  (``--msa-pairing none``), which skips that search.

It deliberately imports nothing from FoldJAX: run it with the interpreter
that has ColabFold installed (``pip install colabfold``, which provides
``colabfold.mmseqs.search``) and ``mmseqs`` on ``PATH``, by its file path::

    export FOLDJAX_MSA_COMMAND="/opt/colabfold/bin/python \\
        $(foldjax msa wrapper) --db /data/colabfold_db --threads 16"
    export FOLDJAX_MSA_LOCAL_VERSION="uniref30_2302+envdb_202108"

``--gpu`` runs MMseqs2's GPU search (``--gpu 1`` in ``colabfold_search``,
which needs GPU-indexed databases: ``setup_databases.sh`` with
``GPU=1``). Set ``FOLDJAX_MSA_LOCAL_VERSION`` to name the databases: it is
part of FoldJAX's cache key, so new databases never reuse old alignments.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

#: Set by FoldJAX (`foldjax.search.msa.MSA_PAIRING_ENV`).
PAIRING_ENV = "FOLDJAX_MSA_PAIRING"


def _query(path: Path) -> str:
    """The first FASTA record's sequence."""
    sequence: list[str] = []
    seen = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if seen:
                break
            seen = True
        elif seen:
            sequence.append(line.strip())
    query = "".join(sequence).upper()
    if not query:
        raise SystemExit(f"{path}: no query sequence")
    return query


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, required=True, help="query FASTA")
    parser.add_argument("--output", type=Path, required=True, help="result directory")
    parser.add_argument(
        "--db", type=Path, required=True, help="ColabFold database directory"
    )
    parser.add_argument("--db1", default="uniref30_2302_db", help="UniRef database")
    parser.add_argument(
        "--db3", default="colabfold_envdb_202108_db", help="environmental database"
    )
    parser.add_argument(
        "--use-env", type=int, choices=(0, 1), default=1, help="search --db3 too"
    )
    parser.add_argument("--mmseqs", default="mmseqs", help="mmseqs executable")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--db-load-mode", type=int, default=0)
    parser.add_argument(
        "-s",
        type=float,
        default=None,
        help="sensitivity (CPU only; default: the server's)",
    )
    parser.add_argument(
        "--gpu", action="store_true", help="MMseqs2 GPU search (GPU-indexed databases)"
    )
    return parser


def _write_query(base: Path, query: str, mmseqs: str, search) -> None:
    """``qdb`` and its lookup, as ``colabfold_search`` creates them."""
    fasta = base / "query.fas"
    fasta.write_text(f">101\n{query}\n", encoding="utf-8")
    search.run_mmseqs(
        mmseqs,
        ["createdb", fasta, base / "qdb", "--shuffle", "0", "--dbtype", "1"],
    )
    (base / "qdb.lookup").write_text("0\t101\t0\n", encoding="utf-8")


def main(argv: list[str] | None = None, *, search=None) -> int:
    args = _parser().parse_args(argv)
    if search is None:
        try:
            from colabfold.mmseqs import search  # type: ignore[no-redef]
        except ImportError:
            print(
                "colabfold_local: this interpreter has no ColabFold "
                "(`pip install colabfold`); run the wrapper with the Python "
                "that has it",
                file=sys.stderr,
            )
            return 2
    if shutil.which(args.mmseqs) is None and not Path(args.mmseqs).is_file():
        print(f"colabfold_local: {args.mmseqs!r} is not on PATH", file=sys.stderr)
        return 2
    query = _query(args.input)
    pair = os.environ.get(PAIRING_ENV, "").strip().lower() != "none"
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="colabfold-local-") as raw:
        base = Path(raw)
        _write_query(base, query, args.mmseqs, search)
        search.mmseqs_search_monomer(
            dbbase=args.db,
            base=base,
            uniref_db=Path(args.db1),
            metagenomic_db=Path(args.db3),
            mmseqs=Path(args.mmseqs),
            use_env=bool(args.use_env),
            use_templates=False,
            s=args.s,
            db_load_mode=args.db_load_mode,
            threads=args.threads,
            gpu=int(args.gpu),
            unpack=True,
        )
        unpaired = base / "0.a3m"
        if not unpaired.is_file():
            print("colabfold_local: the monomer search wrote no 0.a3m", file=sys.stderr)
            return 1
        shutil.copyfile(unpaired, args.output / "non_pairing.a3m")
        paired = base / "0.paired.a3m"
        if pair:
            search.mmseqs_search_pair(
                dbbase=args.db,
                base=base,
                uniref_db=Path(args.db1),
                mmseqs=Path(args.mmseqs),
                pair_env=False,
                s=args.s,
                db_load_mode=args.db_load_mode,
                threads=args.threads,
                gpu=bool(args.gpu),
                unpack=True,
            )
        if pair and paired.is_file() and paired.read_text(encoding="utf-8").strip():
            shutil.copyfile(paired, args.output / "pairing.a3m")
        else:
            # No pairing asked for, or no pairing hit: the query alone, which
            # is what a paired block with no partner rows holds.
            (args.output / "pairing.a3m").write_text(
                f">101\n{query}\n", encoding="utf-8"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
