# Chai scientific-parity fixtures

Inputs and recorded results for `scripts/scientific_parity.py`, recovered from
the standalone `chai_jax` repository when it was vendored. `fixtures/` holds the
FASTA and restraint files each case runs on; `results/` holds what those runs
reported on the dates in their names; `release_thresholds.json` is what a run
had to beat to count as a pass.

These are a record rather than a fixture the suite loads: nothing under
`tests/models/chai/` reads this directory, so it neither runs nor rots on its
own. It is kept because the thresholds and the dated results are the evidence
behind Chai's parity claims, and re-deriving them means re-running the
comparison against upstream.
