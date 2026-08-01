"""Self-contained Boltz-2 input preprocessing for Boltz-JAX.

Extracted from ``boltz/src/boltz/main.py`` (functions ``check_inputs``,
``process_input``, ``process_inputs``) with the excluded paths removed or
guarded: model download and prediction/CLI. Templates and the ColabFold MMseqs2
MSA-server search are vendored under ``foldjax.models.boltz2.data``; ``msa: empty`` remains
an explicitly offline path with no network and no ``import boltz``.

``process_inputs`` writes the standard ``out_dir/processed/`` tree
(``structures/``, ``records/``, ``msa/``, ``constraints/``, ``templates/``,
``mols/`` + ``manifest.json``) consumable by
``foldjax.models.boltz2.data.module.inferencev2.PredictionDataset``.
"""

from __future__ import annotations

import pickle
from functools import partial
from pathlib import Path

from rdkit import Chem
from tqdm import tqdm

from foldjax.models.boltz2.data import const
from foldjax.models.boltz2.data.mol import load_canonicals
from foldjax.models.boltz2.data.parse.fasta import parse_fasta
from foldjax.models.boltz2.data.parse.yaml import parse_yaml
from foldjax.models.boltz2.data.types import Manifest, Record


def check_inputs(data: Path) -> list[Path]:
    """Validate the input path and return the list of input files."""
    if data.is_dir():
        files: list[Path] = list(data.glob("*"))
        for d in files:
            if d.is_dir():
                msg = f"Found directory {d} instead of .fasta or .yaml."
                raise RuntimeError(msg)
            if d.suffix.lower() not in (".fa", ".fas", ".fasta", ".yml", ".yaml"):
                msg = (
                    f"Unable to parse filetype {d.suffix}, "
                    "please provide a .fasta or .yaml file."
                )
                raise RuntimeError(msg)
        return files
    return [data]


def compute_msa(
    data: dict[str, str],
    target_id: str,
    msa_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
    msa_server_username: str | None = None,
    msa_server_password: str | None = None,
    api_key_header: str | None = None,
    api_key_value: str | None = None,
) -> None:
    """Query the MSA server for ``data`` and dump per-entity ``{name}.csv``.

    Vendored from ``boltz/main.py``; uses the vendored ``run_mmseqs2`` (colabfold
    API). ``requests`` is imported lazily so the no-MSA path keeps no network dep.
    """
    from foldjax.models.boltz2.data.msa.mmseqs2 import run_mmseqs2

    auth_headers = None
    if api_key_value:
        key = api_key_header if api_key_header else "X-API-Key"
        auth_headers = {"Content-Type": "application/json", key: api_key_value}

    if len(data) > 1:
        paired_msas = run_mmseqs2(
            list(data.values()),
            msa_dir / f"{target_id}_paired_tmp",
            use_env=True,
            use_pairing=True,
            host_url=msa_server_url,
            pairing_strategy=msa_pairing_strategy,
            msa_server_username=msa_server_username,
            msa_server_password=msa_server_password,
            auth_headers=auth_headers,
        )
    else:
        paired_msas = [""] * len(data)

    unpaired_msa = run_mmseqs2(
        list(data.values()),
        msa_dir / f"{target_id}_unpaired_tmp",
        use_env=True,
        use_pairing=False,
        host_url=msa_server_url,
        pairing_strategy=msa_pairing_strategy,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        auth_headers=auth_headers,
    )

    for idx, name in enumerate(data):
        paired = paired_msas[idx].strip().splitlines()
        paired = paired[1::2]  # ignore headers
        paired = paired[: const.max_paired_seqs]
        keys = [i for i, s in enumerate(paired) if s != "-" * len(s)]
        paired = [s for s in paired if s != "-" * len(s)]

        unpaired = unpaired_msa[idx].strip().splitlines()
        unpaired = unpaired[1::2]
        unpaired = unpaired[: (const.max_msa_seqs - len(paired))]
        if paired:
            unpaired = unpaired[1:]  # query already present

        seqs = paired + unpaired
        keys = keys + [-1] * len(unpaired)
        csv_str = ["key,sequence"] + [f"{k},{s}" for k, s in zip(keys, seqs)]
        (msa_dir / f"{name}.csv").write_text("\n".join(csv_str))


def process_input(
    path: Path,
    ccd: dict,
    msa_dir: Path,
    mol_dir: Path,
    boltz2: bool,
    use_msa_server: bool,
    processed_msa_dir: Path,
    processed_constraints_dir: Path,
    processed_templates_dir: Path,
    processed_mols_dir: Path,
    structure_dir: Path,
    records_dir: Path,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    msa_server_username: str | None = None,
    msa_server_password: str | None = None,
    api_key_header: str | None = None,
    api_key_value: str | None = None,
) -> None:
    """Parse a single input file and dump its processed structure/record tree.

    Protein chains with no explicit MSA are sent to the configured server when
    ``use_msa_server`` is true; ``msa: empty`` remains single-sequence mode.
    """
    # Parse data
    if path.suffix.lower() in (".fa", ".fas", ".fasta"):
        target = parse_fasta(path, ccd, mol_dir, boltz2)
    elif path.suffix.lower() in (".yml", ".yaml"):
        target = parse_yaml(path, ccd, mol_dir, boltz2)
    else:
        msg = (
            f"Unable to parse filetype {path.suffix}, "
            "please provide a .fasta or .yaml file."
        )
        raise RuntimeError(msg)

    target_id = target.record.id

    # Decide whether any protein entity needs remote MSA generation.
    to_generate = {}
    prot_id = const.chain_type_ids["PROTEIN"]
    for chain in target.record.chains:
        if (chain.mol_type == prot_id) and (chain.msa_id == 0):
            entity_id = chain.entity_id
            msa_id = f"{target_id}_{entity_id}"
            to_generate[msa_id] = target.sequences[entity_id]
            chain.msa_id = msa_dir / f"{msa_id}.csv"
        elif chain.msa_id == 0:
            chain.msa_id = -1

    if to_generate and not use_msa_server:
        msg = (
            "Missing MSA's in input and --use-msa-server not set. Use "
            "`msa: empty` per protein chain, provide a precomputed a3m/csv, or "
            "enable the MSA server."
        )
        raise RuntimeError(msg)

    if to_generate:
        compute_msa(
            data=to_generate,
            target_id=target_id,
            msa_dir=msa_dir,
            msa_server_url=msa_server_url,
            msa_pairing_strategy=msa_pairing_strategy,
            msa_server_username=msa_server_username,
            msa_server_password=msa_server_password,
            api_key_header=api_key_header,
            api_key_value=api_key_value,
        )

    # Parse generated or precomputed MSA data (csv/a3m).
    msas = sorted({c.msa_id for c in target.record.chains if c.msa_id != -1})
    msa_id_map = {}
    for msa_idx, msa_id in enumerate(msas):
        msa_path = Path(msa_id)
        if not msa_path.exists():
            msg = f"MSA file {msa_path} not found."
            raise FileNotFoundError(msg)
        processed = processed_msa_dir / f"{target_id}_{msa_idx}.npz"
        msa_id_map[msa_id] = f"{target_id}_{msa_idx}"
        if not processed.exists():
            if msa_path.suffix == ".a3m":
                from foldjax.models.boltz2.data.parse.a3m import parse_a3m

                msa = parse_a3m(msa_path, taxonomy=None, max_seqs=8192)
            elif msa_path.suffix == ".csv":
                from foldjax.models.boltz2.data.parse.csv import parse_csv

                msa = parse_csv(msa_path, max_seqs=8192)
            else:
                msg = f"MSA file {msa_path} not supported, only a3m or csv."
                raise RuntimeError(msg)
            msa.dump(processed)

    for c in target.record.chains:
        if (c.msa_id != -1) and (c.msa_id in msa_id_map):
            c.msa_id = msa_id_map[c.msa_id]

    # Dump templates (empty for the protein no-MSA path).
    for template_id, template in target.templates.items():
        name = f"{target.record.id}_{template_id}.npz"
        template.dump(processed_templates_dir / name)

    # Dump constraints.
    constraints_path = processed_constraints_dir / f"{target.record.id}.npz"
    target.residue_constraints.dump(constraints_path)

    # Dump extra molecules.
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    with (processed_mols_dir / f"{target.record.id}.pkl").open("wb") as f:
        pickle.dump(target.extra_mols, f)

    # Dump structure and record.
    target.structure.dump(structure_dir / f"{target.record.id}.npz")
    target.record.dump(records_dir / f"{target.record.id}.json")


def process_inputs(
    data: list[Path],
    out_dir: Path,
    ccd_path: Path,
    mol_dir: Path,
    use_msa_server: bool = False,
    boltz2: bool = True,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    msa_server_username: str | None = None,
    msa_server_password: str | None = None,
    api_key_header: str | None = None,
    api_key_value: str | None = None,
) -> Manifest:
    """Process the input data, writing the ``processed/`` tree + manifest.

    Returns the loaded :class:`Manifest`.
    """
    msa_dir = out_dir / "msa"
    records_dir = out_dir / "processed" / "records"
    structure_dir = out_dir / "processed" / "structures"
    processed_msa_dir = out_dir / "processed" / "msa"
    processed_constraints_dir = out_dir / "processed" / "constraints"
    processed_templates_dir = out_dir / "processed" / "templates"
    processed_mols_dir = out_dir / "processed" / "mols"
    predictions_dir = out_dir / "predictions"

    for d in (
        out_dir,
        msa_dir,
        records_dir,
        structure_dir,
        processed_msa_dir,
        processed_constraints_dir,
        processed_templates_dir,
        processed_mols_dir,
        predictions_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)

    # Load CCD / canonical molecules.
    if boltz2:
        ccd = load_canonicals(mol_dir)
    else:
        with ccd_path.open("rb") as file:
            ccd = pickle.load(file)  # noqa: S301

    process_input_partial = partial(
        process_input,
        ccd=ccd,
        msa_dir=msa_dir,
        mol_dir=mol_dir,
        boltz2=boltz2,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        api_key_header=api_key_header,
        api_key_value=api_key_value,
        processed_msa_dir=processed_msa_dir,
        processed_constraints_dir=processed_constraints_dir,
        processed_templates_dir=processed_templates_dir,
        processed_mols_dir=processed_mols_dir,
        structure_dir=structure_dir,
        records_dir=records_dir,
    )

    for path in tqdm(data):
        process_input_partial(path)

    records = [Record.load(p) for p in records_dir.glob("*.json")]
    manifest = Manifest(records)
    manifest.dump(out_dir / "processed" / "manifest.json")
    return manifest
