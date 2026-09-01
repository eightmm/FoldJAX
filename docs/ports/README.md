# Porting records

Each model here began as its own repository, and each of those carried notes
that the vendoring did not: an experiment log, and the gates a port had to pass
before it was called done. The code moved into `src/foldjax/models/<name>/` and
the test suites into `tests/models/<name>/`; this directory is where the
*reasoning* moved, so the sibling checkouts can be deleted without losing it.

The dated benchmark write-ups those gates were once argued from have been
removed: they measured revisions from July 2026 and were being read as current
numbers. [`docs/benchmark.md`](../benchmark.md) is the one measured comparison,
and it says which revision each of its rows was taken at.

| port | what is here |
|---|---|
| [`boltz2/`](boltz2/) | `EXPERIMENTS.jsonl` (30 KB), `RELEASE_GATES.md`, `PERFORMANCE_POLICY.md`, the original README |
| [`esmfold2/`](esmfold2/) | diffusion specification and implementation notes |
| [`opendde/`](opendde/) | `EXPERIMENTS.jsonl` (17 KB), `OFFICIAL_ASSETS.json`, README and PROJECT notes |
| [`openfold3/`](openfold3/) | README and PROJECT notes |
| [`protenix/`](protenix/) | `EXPERIMENTS.jsonl` (66 KB), `PERFORMANCE_POLICY.md`, the original README |

## Read these as history, not as instructions

**Every file here began before vendoring.** Their historical parity and
benchmark narratives are intentionally preserved; only the archive notices and
obviously obsolete public setup/runtime snippets are kept aligned with the
current package. Three things follow:

- **The module names are historical.** Names such as
  `protenix_jax.models.model` and `boltz_jax.data.featurize_yaml` are now
  `foldjax.models.<name>.*`. Nothing in the current package answers to the old
  names except restricted weight loaders, which map them deliberately so files
  exported before the move still load.
- **Historical commands inside dated experiments are not current setup
  instructions.** Each port had its own `pyproject.toml` and virtualenv; there
  is one of each now. The commands in the top-level [README](../../README.md)
  are authoritative.
- **Some conclusions have since been overturned**, and the overturning is
  recorded in the top-level README rather than here. The clearest case is
  OpenDDE's speed against upstream: figures taken at bfloat16 against an fp32
  upstream were not a matched comparison, and the corrected numbers are
  different enough to change the conclusion rather than shade it.

What these files are good for is the question a current file cannot answer:
*why is it built this way?* The parity gates each port had to pass, the
measurements that set a default, and the wrong turns that were tried first are
all here, and several of them are the only record that a given approach was
tested and rejected.

For the current state of any port, read the top-level [README](../../README.md);
for the ongoing experiment log across all of them, read `EXPERIMENT_LOG.md` in
the workspace root.
