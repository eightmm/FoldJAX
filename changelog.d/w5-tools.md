### Added

- **`foldjax report DIR`.** One static, self-contained HTML page per output
  directory (no external script, stylesheet or font): per input, one card per
  model with the run metadata and warnings, the model's own scores per sample,
  a per-residue pLDDT plot and, where the model wrote PAE, a PAE heatmap. A
  run without PAE says why.
- **`foldjax interfaces DIR`** and **`show --format csv|json --interfaces`.**
  ipSAE (with its d0chn/d0dom variants and the PAE-only ipTM), pDockQ, pDockQ2
  and LIS per sample and chain pair, ported from the reference `ipsae.py` v4
  and matching its output; the model's own chain-pair ipTM is reported beside
  them as `native`, the rest as `derived`. Samples without PAE are listed with
  the reason.
- **`foldjax compare DIR --reference X.cif --metrics ...`.** Accuracy against a
  deposited structure -- all-atom lDDT, CA-lDDT, TM-score, CA RMSD, DockQ and
  ligand RMSD -- under a homomer-aware chain assignment, as `reference:<name>`
  rows of `compare.csv` and columns of `compare_structures.csv`. Agrees with
  US-align `-TMscore 1`, DockQ 2.1.3 and the benchmark's lDDT and ligand RMSD
  to four decimals on the jctc-v3 FoldJAX outputs. DockQ pins `numpy<2`, so it
  runs as a separately installed tool (`uv tool install --python 3.12
  DockQ==2.1.3`, or `FOLDJAX_DOCKQ`); without it the column is empty and says
  how to get it.
- **`foldjax jobs expand --target T --ligands LIB [--affinity]`** and
  **`foldjax jobs pulldown --baits A --candidates B [--all-vs-all]`** write
  multi-job files for ligand and protein-protein screens, with names that do
  not depend on library order and the target's alignments reused.
  **`show --screen`** tabulates each job's top sample with its affinity
  outputs, ranked within each model only.
- **`--structure-format {cif,pdb,both}`** (predict) writes a PDB copy beside
  each mmCIF and refuses structures the PDB format cannot hold (more than
  99,999 atoms, multi-character chain ids, residue names over three
  characters, atom names over four, residue numbers past 9,999), which gemmi
  would otherwise write as hybrid-36 or shifted columns. The mmCIF stays the
  record.
- **`foldjax check DIR`**: PoseBusters checks of every predicted ligand
  (`pb_valid`, `pb.<check>`), with the new optional extra `posebusters`.
- **`--shard I/N` / `--shard auto`** (predict, plan) runs one round-robin shard
  of a batch, splitting multi-job files per job, into the same directories and
  resume identities as the whole batch; `auto` reads Slurm's array variables.
  **`plan --json`** adds a `slurm` block with `--gres` and the minimum card
  memory from the model's fitted peak law (`--mem` is left unset: no host
  memory law is calibrated).
- **`results_table(..., as_frame=True)`** returns a pandas DataFrame.
