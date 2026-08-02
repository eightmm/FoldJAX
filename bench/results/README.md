# Recorded rows

One JSON per (model, implementation, size), exactly as `drive.py` wrote it.
They are committed so the table in the top-level README can be regenerated and
audited without a GPU:

```bash
python -m bench.report --results bench/results
```

`results-before/` holds rows superseded by a fix, kept as the "before" half of
a measured improvement rather than deleted.
