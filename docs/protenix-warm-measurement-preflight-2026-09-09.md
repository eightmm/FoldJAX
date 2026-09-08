# Protenix warm harness preflight

The initial preflight below predates the completed measurements recorded later.
Existing production benchmark
scripts have explicit workload differences that must be resolved before a
native-default comparison:

- Torch defaults to native random MSA sampling; JAX defaults to full-depth MSA.
  JAX requires `--sample-msa-per-cycle` for the corresponding random arm.
  Equal seeds alone are not a proof of equal selected rows.
- JAX pads MSA rows to a bucket of 64 by default. Exact-shape comparison
  requires `--msa-row-bucket 0` unless padding is explicitly a tested profile.
- Torch enables efficient fusion by default; the JAX script disables it by
  default. Resolve effective runtime policy against the current public runner,
  rather than silently calling the script defaults production parity.
- Both scripts retain the preceding measured output while the next run is
  evaluated. That output-lifetime policy affects peak allocation and must be
  stated or harmonized before interpreting memory savings.

Memory metadata has been clarified without changing existing numeric fields:
Torch retains the legacy warm-region `peak_vram_gb` and records the maximum
of pre-reset and post-reset allocator peaks as `lifetime_peak_allocated_bytes`.
JAX labels its peak as lifetime-scoped and reports warm-only peak as null.
Neither includes allocations outside its framework allocator. Three warm
repeats exclude first inference/compilation but do not establish statistically
significant speed differences.

Verification so far: both edited scripts parse and pass scoped Ruff and diff
checks. Real GPU validation and effective-option parity remain pending.

Further source comparison found the benchmark passed `matmul_precision="default"`
while the current CLI omits the argument and the called prediction function
defaults to `high`. The benchmark now exposes `--matmul-precision`, defaults
to `high`, and records the requested value; `--matmul-precision default`
retains the historical benchmark intervention. No production model code was
changed. A CPU parser check verified both default and legacy override; Ruff
and diff checks passed. Existing historical results are not relabeled.

Jobs 852/853 submit the first native/JAX 3GCA warm pair from a fresh shared
source snapshot, five samples, ten cycles, 200 steps and three warm repeats.
Both use the completed candidate's shared feature archive. Native loads its
pinned base checkpoint through the existing capture's checkpoint symlink.
JAX explicitly uses per-cycle MSA sampling, zero additional row padding and
`high` matmul precision. This is ordinary RNG performance, not tape parity.
Native keeps its efficient fusion; JAX retains its current disabled fusion.
Results must retain that policy difference and full-output retention scope.
Submission does not establish success or comparable peak-memory savings.

Queued-input preflight confirmed 46 tokens, 719 atoms, two MSA rows and an
explicit `is_ligand` field, so the Torch loader's heuristic ligand fallback
is not used for this input. The only dotted feature key is
`pad_info.mask_trunked`; nonnumeric fields are five output atom annotations.
Native 852 began after the existing Claude job 851 completed; JAX 853 remains
queued behind it. No native warm measurements have yet been emitted.

Native job 852 subsequently exited 0. Result root:
`protenix-warm-current-20260909-4XKaRc`. Warm seconds were
`[3.0752237939741462, 3.0838017690111883, 3.089270121010486]`, median
3.0838017690111883. Both warm-region and lifetime allocator peaks were
3,183,086,080 bytes. Coordinate shape was `[5,719,3]`; a checksum alone is
not a structural parity check. Candidate 853 is running; do not compute or
claim a paired speedup before its result and workload-scope review.

## Completed pilot warm pair

Job 853 exited 0. Same external result root contains `native.json` and
`foldjax.json`. Both report five samples, ten cycles, 200 steps and
coordinate shape `[5,719,3]`.

| Metric | Native Torch | FoldJAX |
| --- | ---: | ---: |
| Warm repeat 1 (s) | 3.07522379 | 1.10426841 |
| Warm repeat 2 (s) | 3.08380177 | 1.12392385 |
| Warm repeat 3 (s) | 3.08927012 | 1.12504008 |
| Warm median (s) | 3.08380177 | 1.12392385 |
| Lifetime allocator peak (bytes) | 3,183,086,080 | 1,864,699,904 |

Observed median ratio is approximately 2.74x; measured lifetime allocation
is approximately 41.4 percent lower. These are pilot numbers for this tiny
46-token shared-feature core workload, not general speed or memory claims.
The JAX first call took 78.74945958 seconds and is excluded from warm timing.
JAX reports ten MSA cycles of two rows each; native row selection is not
observed in this uninstrumented arm. Fusion policies remain different as
recorded above. Torch retains its GPU-loaded checkpoint/state_dict alongside
the copied model parameters; this harness's allocation includes that lifecycle
and is not solely model working memory.

The performance scripts save only coordinate shape/checksum, not coordinates
and raw/public confidence arrays. Their checksums differ under ordinary RNG;
this neither proves nor disproves fixed-tape fidelity. An output-retaining,
provenance-bound follow-up and harmonized parameter/output lifetime are needed
before scientific/performance admission. The separately measured fixed-tape
Protenix RNA gray-zone and strict confidence failure remain unchanged.

## Parameter/output lifetime follow-up

The Torch benchmark now loads checkpoint tensors on CPU, copies through
`load_state_dict`, then releases checkpoint/state_dict before warmup. Both
benchmarks release the previous output before the next timed call. These are
measurement-lifecycle changes, not model arithmetic changes; metrics record
the policy explicitly. Syntax, scoped Ruff and diff checks pass.

Jobs 858/859 rerun native/JAX with the same model source snapshot as 852/853,
same features, settings and warm count, replacing only the two benchmark
scripts. Results remain pending. The previous 41.4 percent pilot allocation
reduction must not be treated as the corrected lifecycle comparison.

Both benchmark scripts now accept optional `--prediction-dir`, created with
exclusive directory semantics before inference. After timing and peak capture,
the last ordinary-RNG prediction is exported with the existing native/JAX
tree serializers, original dtype metadata and archive/tree hashes. Export is
outside measured timing/peak. This preserves the complete returned prediction,
not unreturned native internal logits. Existing jobs 858/859 predate this
option and cannot be retroactively claimed to contain these outputs.
Scoped Ruff and 12 existing Protenix capture tests passed; actual GPU export
and a dedicated end-to-end option test remain pending.

## Completed lifecycle-corrected pair

Jobs 858/859 both exited 0. Result root:
`protenix-warm-lifetime-20260909-HpcvmU`; `native.json` and `foldjax.json`
retain the effective settings. Model source, features and workload match the
pilot; checkpoint and previous-output lifetime policies changed as above.

| Metric | Native Torch | FoldJAX |
| --- | ---: | ---: |
| Warm repeat 1 (s) | 3.10321610 | 1.11032400 |
| Warm repeat 2 (s) | 3.11217479 | 1.12548079 |
| Warm repeat 3 (s) | 3.11398127 | 1.12729704 |
| Warm median (s) | 3.11217479 | 1.12548079 |
| Lifetime allocator peak (bytes) | 1,649,330,688 | 1,921,323,008 |
| Warm-only allocator peak (bytes) | 1,649,330,688 | unavailable |

The observed median ratio is 2.77x. FoldJAX's lifetime allocation is now
16.5 percent higher, not lower. The pilot's 41.4 percent reduction is not an
admissible memory-saving claim: Torch's unnecessary GPU checkpoint retention
was a measurement confound. Both lifetime changes were applied together, so
this pair does not separately quantify their individual contributions.

These are framework allocator peaks, not total process VRAM. The JAX peak
includes compilation/first inference and cannot be called warm-only memory.
The small shared-feature workload, different fusion policies, MSA preparation
timing boundary and ordinary RNG remain limitations. Neither successful exit
nor coordinate checksums establish structural/confidence parity. Prediction
export was not enabled in these two runs. No model closure or general speed/
memory optimization claim follows from this pair.

Verification: queue jobs 858 and 859 exited 0; both JSON artifacts contain
five samples, ten cycles, 200 steps, three warm timings, `[5,719,3]`
coordinates and `previous_output_retained_during_next_run=false`.

## Output-retaining GPU follow-up

Jobs 867/868 exited 0 using the same isolated model source and workload with
`--prediction-dir`. Result root: `protenix-warm-export-20260909-EkDtIR`.
Both NPZ archives and tree metadata match the SHA256 values in their metrics.
Coordinates have shape `[5,719,3]`; every numeric array is finite. Native
exports 141 flattened leaves and FoldJAX 34; native per-sample lists versus
candidate stacked arrays account for different layouts, not an equality test.
Both include returned pLDDT/PAE/PDE/resolved/contact outputs. Native internal
distogram logits are not returned and were not intercepted for this arm.

Warm medians are 3.07808959 s native and 1.13425694 s FoldJAX. Lifetime
allocator peaks are 1,649,330,688 and 1,864,699,904 bytes respectively.
FoldJAX remains higher in measured lifetime allocation. Its peak differs from
job 859 despite the same lifecycle policy; no cause is established by these
two runs, and no warm-only memory claim is made.

Source inspection confirms another scope difference: this JAX benchmark uses
precomputed `cycle_msa_features`, while the current public CLI passes
`cycle_msa_index_tape`. A CLI-aligned follow-up is required before claiming
these timings represent the current default route. Ordinary RNG outputs are
preserved for analysis, not admitted as actual-tape parity.

Verification: jobs 867/868 exited 0; both archive/tree hashes and finite-array
checks passed. The new benchmark tests cover exclusive-output-directory
preflight and default/overridden matmul policy. Those and existing capture
tests total 16 passed; scoped Ruff passed. Full release validation is pending.
