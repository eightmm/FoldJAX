# OpenBind independent input identity

External reference pin: `c4771653c5d0a3ebb0b3af71b05efd64bc44ee86`. The
candidate is the current FoldJAX implementation, not a claim that its entire
vendored data closure is identical to that commit.

The preprocessing parity test now records the actual native quaternion and
translation draws, then replays only those draws through independent NumPy
featurization. It asserts exact pre-augmentation positions, masks, event order
and consumption. Native output coordinates are not injected into the candidate.
The fixture aligns Python RNG with the candidate's conformer seed scope and
uses the recognized `uniref90_hits.a3m` dummy-MSA basename. This is a controlled
comparison: native dataloader workers have their own RNG initialization, and
the runner reseeds prediction only after featurization. CPU Torch RNG is scoped
and seeded to make the recorded augmentation draws reproducible.

Independence here means separate executions without shared feature arrays or
injected coordinates. Candidate atom construction/tokenization uses an adapted
vendored upstream data closure; atom identity/bond checks detect vendor drift,
not independently authored chemistry algorithms. The tensorizer and reference
augmentation arithmetic are NumPy implementations.

## Verified scope

All seven original panel JSON inputs passed at seed 101: 1UBQ, 5SAK, 1URN,
3GCA, 7R6R, 3V7E and 7ST3. Checks cover the 33 shared model features (including
reference positions, shape and dtype; numerical tolerance 1e-6), plus exact
ordered atom names, elements, residue names/IDs, chain IDs, entity IDs,
molecule types, bond endpoints and bond types. Each case builds both input
paths separately; pytest enumerates all seven names and fails missing inputs.

The final test also checks the complete runtime feature key set, including
`cyclic_mask` and `is_ligand` outside `MODEL_FEATURES`. Native-only tensor keys
must be exactly dataset bookkeeping (`num_paired_seqs`, `seed`,
`repeated_sample`, `valid_sample`); candidate-only keys must be exactly the
derived padded atom mask. Seed, validity and non-repetition metadata are
checked explicitly. Pairing-row counts are omitted from the portable ABI and
are not claimed as independently compared. The final full-file run with these
key-set checks passed all 11 tests.

The permanent tests also compare the candidate-only padded atom mask against
the native confidence-head helper. The seven-case panel is now parametrized in
pytest, with explicit case names rather than a glob that might omit cases.
Set `FOLDJAX_OPENBIND_INPUT_PANEL` to the native panel work directory. An absent
environment variable skips these external-data tests; a configured directory
with missing cases fails. No external panel data is shipped in the package.

Run in the development environment containing the pinned native checkout:

```sh
JAX_PLATFORMS=cpu python -m pytest \
  tests/models/openfold3/test_torch_parity_preprocessing.py -k independent -q
```

With the panel environment variable set, all seven cases passed including the
derived mask check, alongside the two self-contained protein/3GCA tests.

Verification: focused preprocessing pytest: 9 passed, 2 deselected; then the
entire file passed 11 tests after strengthening integer feature checks to exact
equality. The native checkout HEAD matched the pin and its worktree was clean.
Ruff and `git diff --check` passed. Query-only MSA warnings are expected.

## Limits

This proves the inspected input panel, not all chemistry or arbitrary MSA and
template configurations. Replay captures new preprocessing draws, not the
historical full-model capture's preprocessing tape. Final structure/confidence
parity and warm performance remain separate gates. No production kernel change
or remote publication is certified by this test evidence.

Independent review identified and prompted correction of the vendored-code
independence qualifier and native-worker seeding description above. The panel
does not cover MSA truncation, covalent inter-chain chemistry, or non-empty
templates; the separate direct-template test is not full template-input parity.
