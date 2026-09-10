# CPU parity fixture manifests

One `<port>.json` per port. The JSON is the only part of a fixture that lives
in git: the `.npz` files it describes are 7-16 MB per case and are fetched by
digest into a directory outside the checkout
(`python -m tests.parity.fetch --from <capture dir>`, see
[`docs/parity-cpu.md`](../../../docs/parity-cpu.md)).

`tests/parity/_manifest.py` is what enforces this schema, and it rejects
unknown keys rather than ignoring them: a manifest that quietly dropped a
misspelled key would certify a fixture nobody checked. `tests/parity/test_gate.py`
additionally requires that the tests an entry names are actually collected
under `-m cpu_parity`.

```json
{
  "schema_version": 1,
  "port": "protenix",
  "entries": [
    {
      "case": "protein_1ubq",
      "tier": "A",
      "tokens": 76,
      "capture": {
        "provenance": "/abs/path/<capture>/protein_1ubq/native-A/provenance.json",
        "commit": "34c342d"
      },
      "tripwire": {
        "tape_schema": "protenix-tape-v1",
        "featurizer_commit": "04ad081"
      },
      "files": {
        "trunk.npz": {"sha256": "<64 lowercase hex>", "bytes": 5578638}
      },
      "gpu_residual_angstrom": {"max": 0.037, "per_sample": [0.01, 0.02, 0.037]},
      "cpu_calibration": {
        "residual_angstrom": 0.021,
        "wall_seconds": 96.4,
        "source_commit": "34c342d",
        "host": "workstation, taskset -c 0-3, OMP_NUM_THREADS=4",
        "recorded": "2026-09-10"
      },
      "tolerance": {
        "metric": "rmsd_angstrom",
        "value": 0.05,
        "set_from": "CPU calibration 0.021 A + margin >= the native A-B floor"
      },
      "test_node_ids": [
        "tests/parity/test_protenix_trunk_boundary.py::test_trunk_matches_capture"
      ],
      "excluded_samples": [],
      "notes": ""
    }
  ]
}
```

## Fields

| field | meaning |
| --- | --- |
| `schema_version` | version of *this file format*. The loader speaks exactly one. |
| `port` | must equal the file stem. |
| `case` | the capture's case directory name, e.g. `protein_1ubq`. |
| `tier` | `A` = trunk boundary (minutes), `B` = full tape replay to coordinates (nightly). |
| `tokens` | token count of the case, so a runtime surprise is attributable. |
| `capture.provenance` | absolute path of the capture's `provenance.json`. Its parent is what the fetch helper is pointed at, and it is what a reader opens to see how the capture was produced. |
| `capture.commit` | source commit the capture was taken at. |
| `tripwire` | schema identity of the *stored* artefacts (tape layout version, featurizer/upstream commit). The test compares this against what the checkout reports now and fails; it must never re-featurize, which would compare the checkout against itself. |
| `files` | plain file names, each with `sha256` (64 lowercase hex) and `bytes`. Verified before a path reaches a test. |
| `gpu_residual_angstrom` | what the GPU panel measured for this case (`max`, and `per_sample` where the panel recorded it). Recorded for context; it is **not** where the tolerance comes from. |
| `cpu_calibration` | the first CPU run of this fixture: residual, wall seconds, source commit, and (optionally) host and date. Required -- no entry may ship before its calibration run exists. |
| `tolerance` | `metric`, `value` (> 0, and >= the CPU calibration residual), and `set_from`: the sentence that says how the number was chosen. |
| `test_node_ids` | `tests/parity/...::test_...` node ids that consume this entry. The gate asserts they are collected. |
| `excluded_samples` | optional sample indices this entry does not assert on (a bistable sample, say), with the reason in `notes`. |
| `notes` | optional prose. |

## Why the tolerance cannot come from the GPU residual

Every recorded residual is a GPU port (cuEq triangle kernels, bf16 where the
port uses it) against a GPU native run. CPU XLA is a third implementation: it
has no tensor-core bf16 and materialises converts the GPU fuses, and fusion
decisions are backend-specific. So the number in `tolerance.value` is set from
a CPU calibration run of this exact fixture and the loader refuses an entry
whose tolerance is below its own calibration residual. Zero is refused too:
chunk and scan choices are arithmetically equal, not bitwise.
