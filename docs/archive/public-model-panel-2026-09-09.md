# Six-model default-path comparison, 2026-09-09

Status: in progress, not a completed benchmark.

Requested panel: 1UBQ, 5SAK, 1URN, 3GCA, 7R6R, 3V7E and 7ST3 for
AlphaFold3, Boltz2, ESMFold2, OpenDDE, OpenBind and Protenix. Unsupported
combinations require an explicit source-backed exclusion, not silent omission.
Each model retains its own pinned native schedule and dtype, with five samples.
Do not use `bench.spec.SCHEDULE` indiscriminately: it forces ten recycles even
for models with different native defaults.

Accuracy and performance are separate arms. Accuracy uses actual matched RNG
tapes, one whole-system Kabsch fit per sample, per-entity RMSD and all retained
raw/public confidence leaves. Intermediate native embeddings, LM outputs,
shims and development FFI/operator substitutions are excluded from candidates.
Sharing native features is only a core comparison: independently generated
input identity must be reported separately.

Performance uses ordinary inference without observation hooks, after completed
compilation and warmup, with device synchronization around timed inference.
Report warm time, peak allocator memory and scope consistently on both sides;
a JAX process-lifetime peak must not be presented as a resettable warm-only
peak against Torch's reset warm peak. AF3's native side is JAX, not Torch.
No crystal-coordinate comparison and no tolerance relaxation.

## Execution ledger

- Snapshot: `public-panel-20260909-9V6yUI` (external artifact root).
- tsp 800: ESMFold2 / 5SAK, default compiler/operator policy, no native
  intermediate substitution or development FFI. Existing native tape,
  five samples, two same-executable forwards. Shared native features mean
  this is the first accuracy core arm, not independent preprocessing or
  a measured warm-performance arm. Running state verified at submission.
- Other model/case arms and warm performance arms: not yet launched.

Existing `bench.esmfold2_compare` needs measurement-scope review before reuse:
its JAX peak is lifetime-scoped, Torch resets its peak after warmup, and
full-output synchronization must precede timing. Its ordinary-RNG structures
must not be reported as fixed-tape parity.
