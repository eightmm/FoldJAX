# Token-derived padding profiles

With `--padding`, the token bucket determines automatic storage capacities.
Padding remains opt-in, and explicit axis targets take precedence. This policy
reduces shape variants without changing model weights or model inputs: the MSA
axis is padded *up* to a bucket, and no padded run reads fewer alignment rows
than the same job run unpadded.

| Backend | Atoms | MSA | Other capacity |
| --- | --- | --- | --- |
| AlphaFold 3 | native 24T, 32-aligned | native 16384-row pool / released trunk crop | native templates |
| Boltz2 | 24T, 32-aligned | bucket >= stored rows, floor 1024 | existing supported axes |
| Protenix | 24T, 32-aligned | bucket >= stored rows, floor 1024 | templates 4; LM min(T, provider limit) |
| OpenDDE | 24T, 32-aligned | bucket >= the released 1280-row cycle sample | structural tokens 2T |
| OpenFold3 | 24T, 32-aligned | 1024 per streamed cycle | templates 4 |
| ESMFold2 | 24T, 32-aligned | 1024 | LM 3T for per-chain BOS/EOS |

T is the selected token bucket, not the real sequence length. The MSA buckets
are 1, 64, 128, 256, 512, 768, 1024, 1280, 2048, 3072, 4096, 6144, 8192, 10240,
12288, 14336 and 16384 -- the released profile depths below 2,048 rows, then a
constant 2,048-row step, so no padded run pays more than one step over the rows
it stored -- and the 1024-row floor (1280 for OpenDDE) only widens a shallower
alignment so jobs in one token band still share an executable. A 3,000-row
alignment therefore pads to 3,072 rows with every row kept and the suffix
masked. `max_msa_depth` is the one option that selects fewer rows;
`padding.msa` / `--pad-msa` is a capacity for the axis and a target below the
stored rows is refused, naming `--max-msa-depth` as the option that changes the
input.

Native MSA selection/cropping still precedes padding where applicable: OpenDDE
takes its released 1,280-row valid-first cycle sample -- the same call the
unpadded route makes -- and OpenFold3 its 1,024 rows per streamed cycle, and
padding then pads what they selected. Shallow
MSAs are masked to the chosen storage size. Protenix retains its existing
full-depth MSA requirement. ESMFold2 variants without an active row-selection
cap reject oversized MSA storage unless explicitly pinned; padding never invents
a new sampling route. Padding OFF is unchanged.

## OpenFold3 MSA execution

The padded managed backend keeps the native recycle selections and their union
on the CPU. It gathers and pads one selection to the requested MSA capacity
before device transfer. The index tape and variable-depth union are absent from
all compiled stage inputs. Input embedding runs once, a stable cycle JIT updates
the single/pair carry, and the shared diffusion/confidence tail runs once.

The scheduler explicitly stages common inputs once, admits at most one MSA
lookahead, and waits for each cycle before admitting another. Carry donation
allows the pair-state buffer to be reused. This bounds scheduler-owned MSA
buffers; it does not promise lower allocator reservation or a measured GPU peak.

The host union may exceed 1024 without exceeding the per-cycle capacity. Shape
metadata reports the cycle capacity plus `host_msa_union_rows` and
`msa_execution=host_streamed_cycles`. Padding OFF keeps the fused native path.
See [implementation and verification](openfold3-streamed-msa-2026-09-08.md).

## Recycling defaults

The common API uses the [paper inference recycling policy](recycling-defaults.md).
Padding and cache warming keep that same policy. Boltz2 defaults to five
additional recycles; ESMFold2 defaults to nine additional recycles (ten total
loops). Explicit values take precedence. Other managed model defaults are
AF3 3 (four total passes, explicitly choosing Algorithm 1), Protenix base 10, OpenDDE 10 and OpenFold3/OpenBind 3. The linked audit
separates verified paper settings from publisher fallbacks and explains
initial-pass counting; not every code default is a paper benchmark setting.

## Capacity reasoning

The carried Boltz2 standard residue table has at most 23 atoms per token;
Protenix's terminal nucleotide table reaches 24. Nonstandard components are
atom-tokenized. Native input therefore satisfies atoms <=24T. Materialized
feature archives still receive explicit storage-capacity checks and are never
silently shortened. OpenDDE emits at most backbone plus sidechain/base per
token, giving its structural bound 2T. ESMFold2 may have T one-residue protein
chains, each with BOS/EOS, giving packed LM bound 3T. Protenix embeds chains
separately, so its native provider length cap can bound the automatic LM target.

## The grid, and the context-parallel mesh

The token grid steps by 256 from 256 to 8,192 -- 32 buckets -- so a bucket
costs at most one step of padded work over the exact shape: 256 tokens, 12.5%
at 2k. That bound is the rule; the list of sizes follows from it. A geometric
grid had no such bound, and its gap between 2,048 and 3,072 was the whole cost
of padding: a 2,096-token Protenix job landed on 3,072 and measured +97% wall
and +44% peak against its exact shape (376 s / 30.6 GiB versus 191 s / 21.2
GiB), with the deposited structure unchanged. It now lands on 2,304. The price
is 32 executables per model to bake rather than 11, which `cache warm`
amortises: a bucket is baked once and then hit by every job in its 256-token
band, which is what lets a deployment pre-bake per bucket and run padded by
default.

The grid ends at 8,192 rather than at what one card folds on purpose: context
parallelism exists to run the targets that do not fit one card, so a grid
ending at 4,096 refused exactly the sizes the mesh was added for -- 4,888
tokens with `--padding` was an error rather than a 5,120 profile. Each derived
grid covers that ceiling's own derivation (24x for atoms, 2x for structural
tokens). Above the last bucket `overflow="error"` still refuses rather than
compiling an unplanned shape.

When a request carries `cp_devices > 1`, the automatic targets are rounded up
to what the mesh divides: the atom target to a multiple of `32 * cp_rows`, the
token bucket to a multiple of `cp_rows`, and OpenDDE's structural target to a
multiple of `cp_rows` (`cp_rows` is `cp_devices` under `1d`, its square root
under `2d`). Those are the distributed diffusion atom graph's own requirements
-- see [context parallelism](context_parallel.md) -- and the margins are
small: a pinned `tokens=3012` derives 72,288 atoms, a multiple of 32 and of
nothing larger, which four rows cannot take.

Alignment composes with a 256-step grid, and it is the one thing that can move
an automatic target off the grid. A bucket `256 * k` divides every
power-of-two row count, and its derived atom target `6,144 * k` divides
`32 * cp_rows` for every `cp_rows` that divides 192, so two, three, four, six
and eight rows leave a bucket's own atom target alone. A row count with an odd
factor the bucket index lacks does round the token target past its bucket, by
at most `cp_rows - 1` tokens: with three rows 2,048 becomes 2,049 (and its
atom target derives from 2,049), while 2,304 -- `256 * 9`, the bucket a
2,096-token job selects -- is one of the ten buckets three rows already
divide, so a 3x3 mesh gets the bucket itself, 2,304 tokens and 55,296 atoms.
The one-step bound is therefore a statement about the bucket, not about every
mesh width.

Explicit pins are never rewritten. A misaligned pin keeps its existing
outcome, a warning naming the multiple and a replicated atom graph, because a
pin is a statement about the compiled shape. `cp_devices=1` resolves exactly
as it did before this policy existed.

## Compatibility and verification scope

Existing masks, atom-to-token mappings, crop functions, and prefix-preserving
random draws are reused. Explicit pins and padding-off behavior remain intact.
A token-grid `overflow="exact"` choice never bypasses derived-capacity checks.
The older Boltz-specific `bucket=True` interface retains its legacy policy;
this change applies to the common `PaddingConfig` / `--padding` interface.

Equal padded dimensions are necessary but not sufficient for a cache hit.
Device/JAX identity, dtype, native static options, chain/chemistry configuration,
and auxiliary graph signatures can still require separate executables. The
change does not promise exactly eight executable files per model.

Focused CPU regression coverage checks automatic capacities, explicit overrides,
storage overflow, masks/cropping, native MSA handling, and random-prefix behavior.
Real-weight GPU parity, cache hits, memory, and speed under the new policy have
not been measured. Masked slots can still consume memory and computation.

## The MSA axis pads up (2026-09-15)

`--padding` used to pass the profile's depth to each port's own row-selection
control, so a padded run read a different alignment than the unpadded one:
Protenix and OpenDDE capped their featurizer's 16,384-row assembly at 1,024 and
1,280 rows, Boltz-2's `max_msa_seqs` fell from 16,384 to 1,024, AlphaFold 3's
featurizer cropped its 16,384-row pool to 1,024, and the shared managed
`apply_sampling` injected the same depth for all six models. A 2,096-token
Protenix job padded to 2,304 measured a 19% lower peak and coordinates
0.2-0.3 A from its exact run -- all of it the cap, none of it the padding. An
MSA cap of 4,096 on Protenix had already failed the seed-block accuracy
admission, so a silent 1,024 is a scientific change wearing a shape option's
clothes.

The axis now resolves like tokens and atoms, in one shared
`padding.resolve_msa_axis`: the smallest MSA bucket that holds the stored rows,
with the profile depth as a floor and an explicit `padding.msa` as a target
that is refused below the storage. Consequences worth stating: a padded run on
a deep alignment now costs what the unpadded one costs (the cap was buying the
memory), the memory admission check sees the real row count, and
`ExecutionConfig(padding=...)` no longer requires `ModelConfig(msa_depth=...)`.
OpenFold3 and ESMFold2 were already shape-only -- their per-cycle selection
runs before padding in both routes -- and OpenDDE's released 1,280-row
per-cycle sampling is likewise its own policy, unchanged here; what changed
there is that `--pad-msa` no longer resamples it.

## Initial token-policy validation (2026-09-08, before MSA unification)

- Full CPU suite: `JAX_PLATFORMS=cpu uv run pytest -q` — 5518 passed,
  404 skipped, 67 warnings, 692.27 seconds. This was run in the existing working
  tree, which also contains unrelated development changes.
- The Protenix provider-limit correction was separately checked by 40 CLI/ESM
  tests after the full-suite process had started.
- Combined API/cache/AF3/Boltz2/ESMFold2/OpenFold3 tests: 414 passed.
- Additional Protenix/OpenDDE padding/RNG tests: 55 passed.
- Python compilation, repository Ruff, lock consistency, and diff checks passed.
- Independent source review identified the Protenix 4096-token/4094-LM boundary;
  automatic capacity now clamps to the provider limit, with a CLI regression.

The skipped tests did not execute and are not counted as passing evidence.
No new real-weight GPU run or GPU cache-hit measurement was performed, and
this record is not a release/parity approval.

## MSA-1024 follow-up validation (2026-09-08)

At this intermediate stage, the common padded default was 1024 for every
backend (OpenDDE was subsequently restored to 1280 below). Boltz2, Protenix,
and OpenDDE native padded entrypoints also resolve the same default while
preserving their padding-off defaults. AF3's source wrapper accepts an explicit
MSA crop size and its vendored modification is registered in provenance.

- Final affected CPU suite: 628 passed, 4 warnings, 35.35 seconds. It includes
  backend/cache/default/override tests, AF3 sessions and wrapper configuration,
  all model padding routes, and OpenDDE deep-MSA selection before padding.
- The AF3 mock session fixture now supplies its own libcifpp directory. A
  controlled probe established its resume failure depended on an installed
  runtime directory; production resume validation was not weakened.
- A native AF3 feature-generation test would rebuild the runtime after the
  source change. That test/build was stopped and its subprocesses terminated;
  actual native feature generation remains unverified in this follow-up.
- Changed-file Ruff, syntax, lock, and diff checks passed. Repository-wide Ruff
  reported an unrelated E501 at tests/test_esmfold2_shim_interchange.py:93 in the
  concurrently changing working tree; that file was left untouched.
- The full CPU suite was not repeated for this follow-up. Earlier full-suite
  counts above apply to the preceding token-padding change only.
- No GPU prediction, cache-hit, or prediction-parity measurement was run.

## Model-specific MSA policy

The profile floor is 1,280 rows for OpenDDE and 1,024 for the other models, and
it is a floor rather than a capacity: above it the axis tracks the input-depth
bucket, so a deeper alignment pads to the bucket that holds it instead of being
cropped. Shallower inputs use the masked floor; explicit `--pad-msa` selects a
wider capacity. Native `--max-msa-depth` keeps its existing source-selection
semantics and is the only way to read fewer rows. Each requested profile is
compiled on demand; `cache warm` executes only the requested profile, not a
sweep of MSA depths.

```bash
uv run foldjax cache warm --model opendde --input job.yaml --padding
uv run foldjax cache warm --model opendde --input job.yaml --pad-msa 2048
```

The second command explicitly requests another MSA capacity. Use matching
options for subsequent prediction. Neither command was executed during this
CPU-only verification.

Validation of this final policy: 382 affected CPU tests passed in 7.49 seconds,
including all-model implicit/explicit-default cache equivalence and OpenDDE
native/common default, deep-MSA sampling, masking, and override routes. Changed
files passed Ruff and Python compilation; lock and diff checks passed. The
full CPU suite and GPU inference were not repeated for this bounded correction.
