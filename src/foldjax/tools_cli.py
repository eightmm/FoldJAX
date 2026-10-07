"""Analysis and workflow commands, registered onto `foldjax.cli`'s parser.

Kept out of ``cli.py`` so the prediction CLI stays one file to read: this adds
``report``, ``interfaces``, ``check`` and ``jobs``, and extends ``show``
(``--interfaces``, ``--screen``), ``compare`` (``--reference``, ``--metrics``),
``predict`` (``--structure-format``, ``--shard``) and ``plan`` (``--json``,
``--shard``). `foldjax.cli` calls `register` while building its parser,
`prepare` before it resolves inputs, `dispatch` for the commands handled here,
and `finish_predict` after a prediction batch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def register(
    commands: Any,
    *,
    show: argparse.ArgumentParser,
    compare: argparse.ArgumentParser,
    run: argparse.ArgumentParser,
    plan: argparse.ArgumentParser,
) -> None:
    from foldjax.interfaces import DEFAULT_DIST_CUTOFF, DEFAULT_PAE_CUTOFF
    from foldjax.structure_format import FORMATS

    show.add_argument(
        "--interfaces",
        action="store_true",
        help="with --format csv/json: add per-chain-pair interface columns "
        "derived from each sample's PAE (derived.ipsae.A-B, derived.pdockq.A-B, "
        "derived.pdockq2.A-B, derived.lis.A-B) beside the model's own "
        "native.chain_pair_iptm.A-B; see `foldjax interfaces`",
    )
    show.add_argument(
        "--rank-by",
        metavar="KEY",
        help="order the samples by KEY within each model (and configuration) "
        "only, with a rank_within_model column: plddt, ptm, iptm, ranking (the "
        "model's own ranking score) or a numeric column such as "
        "score.<native name> (derived.* with --interfaces). KEY:asc ranks the "
        "smallest first. Ranks never cross models: each model's scores keep "
        "their own calibration",
    )
    show.add_argument(
        "--screen",
        action="store_true",
        help="with --format table/csv/json: one row per model and job from the "
        "sample each model ranks first, with its affinity outputs where it has "
        "an affinity head (Boltz-2), ranked within each model only",
    )

    compare.add_argument(
        "--reference",
        type=Path,
        help="also score every structure against this deposited mmCIF/PDB "
        "(accuracy, comparable across models): rows reference:<name> in "
        "compare.csv and columns in compare_structures.csv",
    )
    compare.add_argument(
        "--metrics",
        help="with --reference: comma-separated lddt, lddt_ca, tm, rmsd_ca, "
        "dockq, lig_rmsd or all (default lddt,lddt_ca,tm,rmsd_ca). dockq runs "
        "the DockQ tool (uv tool install --python 3.12 DockQ==2.1.3)",
    )

    for parser in (run, plan):
        parser.add_argument(
            "--shard",
            metavar="I/N",
            help="run only shard I of N (0-based) of a batch: every N-th plain "
            "input or job of a multi-job file, round-robin, each into the "
            "directory the whole batch would use. 'auto' or 'auto/N' reads "
            "SLURM_ARRAY_TASK_ID (and SLURM_ARRAY_TASK_COUNT)",
        )
    run.add_argument(
        "--structure-format",
        choices=FORMATS,
        default="cif",
        help="'pdb' or 'both' also write a .pdb beside each canonical mmCIF "
        "(the mmCIF is always kept: the manifest and confidence arrays refer to "
        "it). PDB cannot hold >99,999 atoms, chain ids longer than one "
        "character, or residue names longer than three: 'pdb' refuses such a "
        "job before running and fails after it if a structure still does not "
        "fit; 'both' warns and writes no PDB for it",
    )
    plan.add_argument(
        "--json",
        action="store_true",
        help="add a 'slurm' block per run: --gres and the minimum card memory "
        "from the model's fitted peak law (device memory only; --mem is left "
        "unset because no host-memory law is calibrated)",
    )

    report = commands.add_parser(
        "report",
        help="write a static, self-contained HTML report of an output directory",
        description="One page, no external scripts or fonts: per input, one "
        "card per model with run metadata, the model's own scores per sample, "
        "a per-residue pLDDT plot and a PAE heatmap where the model wrote PAE.",
    )
    report.add_argument("path", type=Path, help="a finished output directory")
    report.add_argument(
        "--out", type=Path, help="where to write it (default PATH/foldjax_report.html)"
    )

    interfaces = commands.add_parser(
        "interfaces",
        help="ipSAE, pDockQ, pDockQ2 and LIS per chain pair, from each sample's PAE",
        description="Derived by FoldJAX from each sample's PAE, pLDDT and "
        "structure with the definitions of ipsae.py v4 (Dunbrack 2025), next "
        "to the model's own chain-pair ipTM where it has one. Within-model "
        "quantities: each PAE has its model's own calibration. Samples without "
        "PAE are listed with the reason.",
    )
    interfaces.add_argument("path", type=Path, help="a finished output directory")
    interfaces.add_argument(
        "--pae-cutoff",
        type=float,
        default=DEFAULT_PAE_CUTOFF,
        help=f"ipSAE PAE cutoff in A (default {DEFAULT_PAE_CUTOFF:g}, the "
        "reference's recommendation)",
    )
    interfaces.add_argument(
        "--dist-cutoff",
        type=float,
        default=DEFAULT_DIST_CUTOFF,
        help=f"CB distance in A for the interface-residue counts dist1/dist2 "
        f"only; it changes no score (default {DEFAULT_DIST_CUTOFF:g})",
    )
    interfaces.add_argument(
        "--format", choices=("table", "csv", "json"), default="table"
    )
    interfaces.add_argument(
        "--out", type=Path, help="write to a file instead of stdout"
    )

    check = commands.add_parser(
        "check",
        help="PoseBusters checks of every predicted ligand (optional extra)",
        description="Runs PoseBusters (`uv sync --extra posebusters`) on each "
        "ligand of each sample: pb_valid and one pb.<check> column per test.",
    )
    check.add_argument("path", type=Path, help="a finished output directory")
    check.add_argument("--format", choices=("table", "csv", "json"), default="table")
    check.add_argument("--out", type=Path, help="write to a file instead of stdout")

    from foldjax.schema import MSA_PAIRINGS

    msa = commands.add_parser(
        "msa", help="search and cache alignments ahead of a prediction"
    )
    msa_commands = msa.add_subparsers(dest="msa_command", required=True)
    prefetch = msa_commands.add_parser(
        "prefetch",
        help="search and cache every chain's alignment, and nothing else",
        description="Runs the --msa auto search for every protein chain (and "
        "RNA chain, when FOLDJAX_RNA_MSA_COMMAND is set) of the inputs into the "
        "shared MSA cache, so a later `foldjax predict --msa auto` -- on a node "
        "without network, say -- reads them from there. No weights load and no "
        "output is written. Without --model, the per-chain search every model "
        "shares; with --model, the search predict runs for that model "
        "(OpenFold3's complex pairing included). The public ColabFold server "
        "receives the sequences unless FOLDJAX_MSA_COMMAND or "
        "FOLDJAX_MSA_SERVER_URL says otherwise. Exits 3 when any chain's "
        "search failed.",
    )
    prefetch.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="job JSON/YAML, multi-job files, FASTA, .pdb/.mmcif, or directories",
    )
    prefetch.add_argument(
        "--model", nargs="+", help="search as predict would for these models"
    )
    prefetch.add_argument(
        "--msa-pairing",
        choices=MSA_PAIRINGS,
        default="model",
        help="as for predict; without --model, greedy/complete also run the "
        "complex pairing search",
    )
    msa_commands.add_parser(
        "wrapper",
        help="print the path of the reference local search wrapper",
        description="Prints the file path of foldjax/search/colabfold_local.py, "
        "a FOLDJAX_MSA_COMMAND wrapper that runs ColabFold's MMseqs2 search "
        "against local databases (optional --gpu). Run it with the Python that "
        "has ColabFold installed; see docs/cli.md.",
    )

    jobs = commands.add_parser("jobs", help="generate multi-job files for screens")
    jobs_commands = jobs.add_subparsers(dest="jobs_command", required=True)
    expand = jobs_commands.add_parser(
        "expand",
        help="one job per ligand of a library against one target",
        description='Writes {"jobs": [...]}: the target job plus one SMILES '
        "ligand each, named <target>__<ligand name> (a digest of the canonical "
        "SMILES when the record has no name), with the target's alignment paths "
        "made absolute so every job reuses them. Run it with --padding and the "
        "persistent compile cache so the screen compiles once per size bucket.",
    )
    expand.add_argument("--target", type=Path, required=True, help="one common job")
    expand.add_argument(
        "--ligands",
        type=Path,
        required=True,
        help="library: .smi (SMILES [name]) or .sdf",
    )
    expand.add_argument(
        "--affinity",
        action="store_true",
        help="ask for binding affinity of each ligand (Boltz-2 only)",
    )
    expand.add_argument(
        "--skip-invalid",
        action="store_true",
        help="leave out unreadable records instead of refusing the library",
    )
    expand.add_argument(
        "--out", type=Path, help="jobs file to write (default <target>__<library>.json)"
    )
    pull = jobs_commands.add_parser(
        "pulldown",
        help="protein-protein jobs: baits x candidates, or all pairs",
        description="One two-chain job per pair (A = first, B = second), named "
        "<first>__<second> from the FASTA ids. Without --msa-dir run with "
        "--msa auto: alignments are cached per sequence.",
    )
    pull.add_argument("--baits", type=Path, help="FASTA of bait proteins")
    pull.add_argument("--candidates", type=Path, help="FASTA of candidate proteins")
    pull.add_argument(
        "--all-vs-all",
        action="store_true",
        help="every unordered pair of all proteins given (no self pairs)",
    )
    pull.add_argument(
        "--msa-dir",
        type=Path,
        help="use <dir>/<fasta id>.a3m as each chain's unpaired_msa",
    )
    pull.add_argument(
        "--out", type=Path, help="jobs file to write (default pulldown.json)"
    )


def _emit(text: str, out: Path | None) -> None:
    if out is None:
        sys.stdout.write(text if text.endswith("\n") else text + "\n")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    print(str(out))


def prepare(args: argparse.Namespace, *, jobs_root: Path | None = None) -> None:
    """Apply ``--shard`` and the ``--structure-format pdb`` pre-flight.

    Runs before `foldjax.cli` builds the request, so a shard is just a
    smaller batch and a job PDB cannot hold is refused before any GPU time.
    A shard's multi-job files go below ``jobs_root`` (`plan`'s scratch), or
    the store's ``runtime/jobs`` when it is None.
    """
    if args.command not in {"predict", "plan"}:
        return
    if getattr(args, "shard", None):
        from foldjax import cli, slurm
        from foldjax.schema import expand_input_directories

        if not args.input:
            raise ValueError("--shard splits --input; a --sequence job is one run")
        index, count = slurm.parse_shard(args.shard)
        expanded = expand_input_directories(
            [Path(p) for p in args.input], suffixes=cli._JOB_SUFFIXES
        )
        selected, summary = slurm.shard_inputs(
            expanded, index, count, jobs_root=jobs_root
        )
        print(
            f"[foldjax] shard {index}/{count}: {summary['units_in_shard']} of "
            f"{summary['units_total']} unit(s)",
            file=sys.stderr,
        )
        if not selected:
            args._empty_shard = summary
        args.input = selected
        # A shard is part of a batch: it keeps the batch's <model>/<input>
        # layout even when it holds a single input.
        args._plural_inputs = True
    if args.command == "predict" and getattr(args, "structure_format", "cif") == "pdb":
        from foldjax.cli import _FASTA_SUFFIXES
        from foldjax.input import read_job_document
        from foldjax.job import Job
        from foldjax.structure_format import preflight

        documents = []
        for path in args.input or []:
            path = Path(path)
            if not path.is_file():
                continue
            if path.suffix.lower() in _FASTA_SUFFIXES:
                # A header such as ">AB" becomes a two-character chain id.
                documents.append(Job.from_fasta(path).to_document())
                continue
            if path.suffix.lower() not in {".json", ".yaml", ".yml"}:
                continue
            try:
                document = read_job_document(path)
            except (OSError, ValueError):
                continue
            if isinstance(document, dict) and isinstance(document.get("jobs"), list):
                documents.extend(j for j in document["jobs"] if isinstance(j, dict))
            elif isinstance(document, dict):
                documents.append(document)
        problems = preflight(documents)
        if problems:
            raise ValueError(
                "--structure-format pdb cannot hold this job: "
                + "; ".join(problems[:5])
                + ". Use --structure-format both to keep the mmCIF and skip the PDB"
            )


def dispatch(args: argparse.Namespace) -> int | None:
    """Handle the commands and flags registered here; None for the rest."""
    command = args.command
    if getattr(args, "_empty_shard", None) and command in {"predict", "plan"}:
        print(
            json.dumps({"shard": args._empty_shard, "runs": []}, indent=2),
        )
        return 0
    if command == "msa":
        return _msa(args)
    if command == "report":
        from foldjax.html_report import write_report

        print(str(write_report(args.path, args.out)))
        return 0
    if command == "interfaces":
        from foldjax import interfaces

        if args.format == "json":
            text = interfaces.to_json(
                interfaces.interfaces_document(
                    args.path, pae_cutoff=args.pae_cutoff, dist_cutoff=args.dist_cutoff
                )
            )
        else:
            rows = interfaces.interface_rows(
                args.path, pae_cutoff=args.pae_cutoff, dist_cutoff=args.dist_cutoff
            )
            text = (
                interfaces.to_csv(rows)
                if args.format == "csv"
                else interfaces.render_table(rows)
            )
        _emit(text, args.out)
        return 0
    if command == "check":
        from foldjax import checks

        rows = checks.check_directory(args.path)
        if args.format == "json":
            text = json.dumps(rows, indent=2, sort_keys=True, default=str)
        elif args.format == "csv":
            text = checks.to_csv(rows)
        else:
            text = checks.render_table(rows)
        _emit(text, args.out)
        return 0
    if command == "jobs":
        from foldjax import job_generators

        if args.jobs_command == "expand":
            document, summary = job_generators.expand_ligands(
                args.target,
                args.ligands,
                affinity=args.affinity,
                skip_invalid=args.skip_invalid,
            )
            out = args.out or Path(f"{args.target.stem}__{args.ligands.stem}.json")
        else:
            document, summary = job_generators.pulldown(
                args.baits,
                args.candidates,
                all_vs_all=args.all_vs_all,
                msa_dir=args.msa_dir,
            )
            out = args.out or Path("pulldown.json")
        written = job_generators.write_jobs(document, out)
        print(json.dumps({"jobs_file": str(written), **summary}, indent=2))
        return 0
    if command == "show" and (args.interfaces or args.screen or args.rank_by):
        return _show(args)
    if command == "compare" and (args.reference is not None or args.metrics):
        from foldjax.compare import compare_directory

        written = compare_directory(
            args.path,
            out=args.out,
            samples=args.samples,
            reference=args.reference,
            metrics=args.metrics,
        )
        print(json.dumps({key: str(value) for key, value in written.items()}, indent=2))
        return 0
    return None


def _msa(args: argparse.Namespace) -> int:
    if args.msa_command == "wrapper":
        from foldjax.search import colabfold_local

        print(Path(colabfold_local.__file__).resolve())
        return 0
    from foldjax import progress
    from foldjax.msa_prefetch import prefetch, render

    was_enabled = progress.enabled()
    progress.enable()
    try:
        records = _prefetch(args, prefetch)
    finally:
        if not was_enabled:
            progress.disable()
    print(render(records))
    failed = [
        record
        for record in records
        if record.get("error") or record.get("paired_error")
    ]
    if failed:
        print(
            f"foldjax: {len(failed)} chain search(es) failed; see 'error' above",
            file=sys.stderr,
        )
        return 3
    return 0


def _prefetch(args: argparse.Namespace, prefetch: Any) -> list[dict[str, Any]]:
    import tempfile

    from foldjax import cli

    with tempfile.TemporaryDirectory(prefix="foldjax-prefetch-jobs-") as scratch:
        # FASTA and structures become job documents in scratch, as `plan`
        # does: a prefetch writes nothing into the store but the MSA cache.
        namespace = argparse.Namespace(
            input=list(args.inputs),
            sequence=[],
            dna=[],
            rna=[],
            ligand=[],
            ligand_smiles=[],
            name=None,
            affinity_binder=None,
        )
        inputs = cli._resolve_inputs(namespace, jobs_root=Path(scratch))
        return prefetch(inputs, models=args.model, pairing=args.msa_pairing)


def _show(args: argparse.Namespace) -> int:
    from foldjax import results

    if args.json:
        raise ValueError("--json prints manifests; use --format json for rows")
    rows = results.results_table(results.load_results(args.path))
    if args.rank_by and (args.screen or args.aggregate):
        raise ValueError(
            "--rank-by orders samples; --screen and --aggregate are tables of "
            "their own. Drop one of them"
        )
    if args.rank_by and not args.interfaces:
        return _emit_ranked(results.rank_rows(rows, args.rank_by), args)
    if args.screen:
        if args.interfaces or args.aggregate:
            raise ValueError("--screen is its own table; drop --interfaces/--aggregate")
        from foldjax.job_generators import screen_table

        table = screen_table(rows)
        if args.format == "csv":
            sys.stdout.write(results.to_csv(table))
        elif args.format == "json":
            print(json.dumps(table, indent=2, sort_keys=True, default=str))
        else:
            print(_render_screen(table))
        return 0
    if args.format == "table":
        raise ValueError("--interfaces adds columns: use --format csv or json")
    if args.aggregate:
        raise ValueError("--interfaces is per sample; drop --aggregate")
    from foldjax.interfaces import interface_columns, interface_rows

    folded = interface_columns(interface_rows(args.path))
    for row in rows:
        key = (
            row.get("model"),
            row.get("input"),
            row.get("configuration"),
            row.get("job"),
            row.get("seed"),
            row.get("sample"),
        )
        row.update(folded.get(key, {}))
    if args.rank_by:
        return _emit_ranked(results.rank_rows(rows, args.rank_by), args)
    if args.format == "csv":
        sys.stdout.write(results.to_csv(rows))
    else:
        print(json.dumps(rows, indent=2, sort_keys=True, default=str))
    return 0


def _emit_ranked(rows: list[dict[str, Any]], args: argparse.Namespace) -> int:
    from foldjax import results

    if args.format == "csv":
        sys.stdout.write(results.to_csv(rows))
    elif args.format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True, default=str))
    else:
        print(results.render_ranked(rows, args.rank_by))
    return 0


def _render_screen(table: list[dict[str, Any]]) -> str:
    lines = [
        "ranked within each model only; a rank is not comparable across models",
    ]
    current = None
    for row in table:
        group = (row["model"], row["configuration"])
        if group != current:
            current = group
            lines.append("")
            lines.append(
                f"{row['model']} ({row['configuration']}), by {row['rank_basis']}"
            )
            lines.append(
                f"  {'rank':>4s}  {'job':<32s}"
                f"{row.get('ranking_key') or 'ranking':>20s}"
                f"{'affinity_pred':>15s}{'p(binder)':>11s}"
            )

        def fmt(value: Any) -> str:
            return "-" if not isinstance(value, (int, float)) else f"{value:.3f}"

        lines.append(
            f"  {str(row['rank_within_model'] or '-'):>4s}  {str(row['job'])[:31]:<32s}"
            f"{fmt(row.get('ranking_value')):>20s}"
            f"{fmt(row.get('score.affinity_pred_value')):>15s}"
            f"{fmt(row.get('score.affinity_probability_binary')):>11s}"
        )
    return "\n".join(lines).lstrip("\n")


def finish_predict(args: argparse.Namespace, results: Any) -> int:
    """Write PDB copies after a batch; nonzero when ``pdb`` could not."""
    fmt = getattr(args, "structure_format", "cif")
    if fmt == "cif":
        return 0
    from foldjax.structure_format import write_formats

    structures = [
        sample.structure_path
        for result in results
        for sample in result.samples
        if sample.structure_path is not None
    ]
    written, refused = write_formats(structures, fmt)
    for message in refused:
        print(f"foldjax: {message}", file=sys.stderr)
    if written:
        print(f"[foldjax] wrote {len(written)} PDB file(s)", file=sys.stderr)
    return 2 if refused and fmt == "pdb" else 0
