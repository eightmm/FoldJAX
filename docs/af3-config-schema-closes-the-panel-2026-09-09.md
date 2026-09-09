# The AF3 audit panel closes once both arms record the same config schema

## The defect

The panel marked all six cases `"passed": false`. Reading the per-check verdicts
rather than the summary flag: each case runs 38 checks, and in all six the
**same 36 pass and the same 2 fail** -- `config.json` and
`effective-config.json`. `coordinates`, `tape`, `preprocessing_tape`,
`tape_coverage`, `identity.npz` and every provenance check were already passing.

Diffing the two configs structurally -- a flatten-and-diff misses one of them,
because an empty list emits no leaves -- leaves exactly two keys out of 126:

| Key | native | foldjax |
| --- | --- | --- |
| `foldjax_stop_after` | *(absent)* | `'full'` |
| `foldjax_return_representations` | *(absent)* | `[]` |

Both are port-only fields on the port's copy of the upstream config dataclass,
at the values that mean "captured nothing extra". Upstream's dataclass has
neither, so the two arms serialized different schemas and the comparison could
never match.

## Why the fix is not in the comparison

`bench/af3_closure.py` already contains `config_difference_kind`, which
enumerates these two keys with these defaults and classifies them
`allowlisted_extensions_only`. Its docstring states the policy:

> Describe exact top-level default extensions; never grant parity admission.

and `test_default_extension_is_explained_without_waiving_config_gate` enforces
it. An earlier attempt in this session normalized the comparison instead; every
config comparison flipped to equal and both tripwires still fired, and the
pre-existing guard failed. That attempt was reverted.

## The fix

`bench/af3_closure_capture.py` gains `config_record()`, applied at both places
a config is serialized. It writes the documented no-op values on an arm that
lacks them. For a run that captured nothing extra, those values are a true
statement about the native run too: it ran to completion and returned no
additional representations. The comparison stays strict.

A port run that stopped early or returned representations still differs from
native and still fails -- asserted in both directions by
`tests/test_af3_closure_capture.py`.

## Verified end to end

`protein_1ubq`, both arms re-run from a snapshot carrying the fix, compared with
the unmodified `compare_arms`:

```
passed = True
n_checks = 38
failed = []
```

The patch is confirmed to have fired rather than something else having changed:

| Arm | `foldjax_stop_after` | `foldjax_return_representations` |
| --- | --- | --- |
| native, before the fix | *(absent)* | *(absent)* |
| native, after | `full` | `[]` |
| foldjax, after | `full` | `[]` |

The pre-existing guard suite is green at 73 passed, comparison untouched.

## Effect on the cross-model standard

AlphaFold 3's row in `common-error-standard-2026-09-09.md` was `n/a`, then
wrongly retracted, then restored on the grounds that the gate was deliberate.
The gate was deliberate; what it was refusing to admit was a schema difference
that should not have existed in the artifact. With the records made
schema-identical the panel admits the case, on a strict comparison, with the
policy and its guard intact.

## All six cases, re-run and scored

Every case re-run on both arms from the fixed snapshot, scored with the
unmodified `compare_arms`:

```
protein_1ubq             passed=True  38 checks  failed=[]
protein_dna_7r6r         passed=True  38 checks  failed=[]
protein_ligand_5sak      passed=True  38 checks  failed=[]
protein_protein_7st3     passed=True  38 checks  failed=[]
protein_rna_1urn         passed=True  38 checks  failed=[]
protein_rna_ligand_3v7e  passed=True  38 checks  failed=[]
```

Six of six, 38 of 38, zero failed checks. The panel went from all-six-failing to
all-six-passing on a strict comparison, with the allowlist policy and its guard
test untouched.

Artifacts: `foldjax-bench/af3-schemafix-20260909`, source snapshot
`foldjax-bench/af3-schemafix-src-20260909`.
