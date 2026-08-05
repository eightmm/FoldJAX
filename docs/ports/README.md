# Porting records

Each model here began as its own repository, and each of those carried notes
that the vendoring did not: an experiment log, a porting plan, the gates a port
had to pass before it was called done, and the benchmark reports those gates
were argued from. The code moved into `src/foldjax/models/<name>/` and the test
suites into `tests/models/<name>/`; this directory is where the *reasoning*
moved, so the sibling checkouts can be deleted without losing it.

| port | what is here |
|---|---|
| [`boltz2/`](boltz2/) | `EXPERIMENTS.jsonl` (30 KB), `RELEASE_GATES.md`, `PERFORMANCE_POLICY.md`, the original README |
| [`chai/`](chai/) | `SCIENTIFIC_PARITY.md` (16 KB), `BENCHMARKS.md` (11 KB), `COMPLETION_GATES.md`, `PERFORMANCE_POLICY.md`, README and PROJECT notes |
| [`opendde/`](opendde/) | `EXPERIMENTS.jsonl` (17 KB), `OFFICIAL_ASSETS.json`, README and PROJECT notes |
| [`openfold3/`](openfold3/) | README and PROJECT notes |
| [`protenix/`](protenix/) | `PORTING_PLAN.md` (29 KB), `EXPERIMENTS.jsonl` (66 KB), a parity benchmark, a performance profile, a production inference benchmark, the original README |

## Read these as history, not as instructions

**Every file here predates vendoring**, and none has been rewritten to match the
package as it stands. Three things follow:

- **The module names are the old ones.** `protenix_jax.models.model`,
  `chai_jax.output`, `boltz_jax.data.featurize_yaml` and the rest are now
  `foldjax.models.<name>.*`. Nothing in the current package answers to the old
  names except the weight loader, which maps them deliberately so that files
  exported before the move still load.
- **The install and test commands do not work.** Each port had its own
  `pyproject.toml` and virtualenv; there is one of each now. The commands in the
  top-level [README](../../README.md) are the current ones.
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
