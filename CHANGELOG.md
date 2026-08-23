# Changelog

What changed between releases, and why. Entries name the behaviour a user can
observe; the reasoning lives beside the code.

FoldJAX follows [semantic versioning](https://semver.org). Model weights,
upstream defaults and scientific results are covered by the project contract in
`PROJECT.md` rather than by this file: a release never changes what a recorded
command predicts unless it says so here, in its own paragraph.

## Unreleased

### Added

- **Protenix carries its relative-position one-hot compactly without changing
  trunk or diffusion arithmetic.** The `[N_token, N_token, 139]` feature contains
  only exact zeroes and ones, so its host/device representation is now int8
  (4.70 GiB to 1.17 GiB at 3,012 tokens). Promotion through the projection
  weights preserves the historical split: bfloat16 in the default trunk and
  float32 in diffusion. A compile-only GPU probe at 1,003 tokens reduced total
  reported memory from 1,056.7 to 624.6 MiB; released-weight peak memory remains
  a deployment gate, and the CPU compiler may materialize the widening
  conversion instead of fusing it.

- **OpenDDE warns before a job that probably will not fit.** OpenDDE ships
  float32, and float32 has a wall near 1,400 residues on a 96 GB card that no
  backend choice rescues -- at 1,531 residues the fused path asks 93.40 GiB and
  the blocked path 80.72, and both die. The failure presented as a half-hour
  hang, because the allocator retries every ten seconds before giving up. A
  preflight now estimates the arena from the realized trunk dtype and the
  structural token count, compares it against the pool the allocator took, and
  warns -- naming `--trunk-dtype bf16` as the lever and saying plainly that it
  is an opt-in divergence from upstream's float32 storage. It never refuses,
  it is silent for every job that fits, and it is silent whenever it cannot
  establish the pool, the token count or the dtype. The message says the
  estimate is not a guarantee in either direction, because the practical
  ceiling sits below the arithmetic one.

- **A requested triangle-multiplication chunk size now says when it is
  ignored.** The fused kernel has no parameter of that family, so the option
  was silently inert whenever it ran -- and unlike the attention path, where
  the fused kernel never builds the tensor a chunk size would bound, this one
  does build it: 5,299 MiB apiece on OpenDDE's structural branch at 1,003
  tokens. The warning names that size and points at the backend that honours
  the request.

- **OpenFold3 reuses a bounded, correctly partitioned in-process JIT pool across
  repeated factories.** Multi-seed/backend calls with the same graph choices
  now share a trace instead of constructing a new `jax.jit` cache each time,
  while an eight-executable LRU bound prevents plural unpadded workloads from
  retaining every program shape for the lifetime of the process. The static
  identity includes configuration, chain count, resolved CP layout and ordered
  devices, effective triangle kernel, RNG route, representations/stop boundary,
  and persistent-cache scope; the representative-atom table stays validated
  dynamic data. Deleted or replaced persistent-cache entries force repopulation,
  and `lower().compile()` retains the public three-argument callable plus its CP
  placement. Tiny serial and forced four-device CPU tests cover exact parity,
  trace reuse and eviction, distinct 1-D/2-D HLO, cache repair and partitioning.
  Released-weight GPU compile latency and memory remain deployment gates, so no
  GPU speedup is claimed here.
- **ESMFold2 keeps one verified model and one input's ESMC states for a
  request.** A multi-seed or multi-input batch used to construct a fresh
  backend for every scalar run, reload the 26,347,754,143-byte released bundle,
  and repeat the seed-independent ESMC-6B forward. Backends can now opt into a
  single-threaded, request-scoped session; ESMFold2 loads lazily, retains at
  most one input's final scattered language-model state, and releases both
  model and state before the next model group. At 1,000 tokens that retained
  BF16 state is 414,720,000 bytes, against 25,408,148,888 source bytes for the
  ESMC checkpoint. An all-resumed request loads no weights.

  Reuse is deliberately narrower than path equality. The session binds every
  structure-model file, ESMC config/index/shard, device placement and the exact
  normalized feature bytes; it refuses to combine seeds if those assets change
  during resume, load, prediction or manifest writing. Backends that do not opt
  in keep the historical fresh-instance-per-scalar behavior. CPU contract tests
  cover two inputs by three seeds, lazy resume, checkpoint replacement,
  continued failures, cleanup and wrapper compatibility. Released-weight
  CUDA latency, peak-memory and numerical parity remain deployment gates, so
  no end-to-end GPU speedup is claimed here.
- **ESMFold2 casts checkpoint leaves before transferring them.** The released
  structure and ESMC bundles contain 1,594 tensors across 49 shapes and 802
  selected tensors across six shapes, respectively. Loading them through
  `jnp.asarray` and then casting ESMC on device created a shape-specialized
  staging program for the structure leaves and both staging and conversion
  programs for ESMC. The loader now casts the published float32 ESMC leaves to
  the requested dtype on the host and transfers both bundles with `device_put`.
  This preserves the historical values exactly, removes all of those loader
  compilations, and avoids an extra full-width device copy of the current ESMC
  leaf (up to 141,557,760 bytes in the released checkpoint). In a separate
  128 MiB CPU transfer probe the bfloat16 path's maximum RSS fell from 909,652
  to 598,180 KiB (34.2%); released-checkpoint GPU peak memory remains a
  deployment gate.
- **AlphaFold 3 reuses one managed ModelRunner inside a request.** Multi-seed
  and multi-input managed runs now load the selected parameter generation and
  construct its shape-keyed JIT owner once, while scalar runs and explicit
  external `source=` checkouts preserve the fresh-run path. Reuse is lazy, so
  an entirely resumed request does not import the runner or load parameters.
  Weight files, the vendored runner and vendored model source are anchored in
  both the live session and run manifests; a partial resume cannot combine
  seeds produced by different generations. CPU mock tests cover reuse,
  cleanup, cache/config/device splits, mutation, resume and cancellation.
  Released-weight GPU latency, memory and numerical parity remain deployment
  gates, so no production speedup is claimed here.
- **Boltz-2 reuses its verified parameters and JIT owners inside a request.**
  Multi-seed and multi-input runs now load the confidence checkpoint once and
  retain one bounded compiled owner. Consecutive affinity runs do the same for
  the complete affinity model; that optional tree is released before a later
  primary-only input. Reuse is lazy, so an all-resumed request does not import
  the native runtime or load either checkpoint. The session
  binds the loader's selected `.safetensors`/`.npz` file, scalar sidecar and
  sidecar absence, device/CP topology, cache namespace and every trace-time
  model option, including JAX dtype/PRNG mode and the effective triangle
  multiplication backend. A changed or newly preferred checkpoint generation
  poisons the session instead of mixing seeds, while a continued ordinary
  failure drops params and executables before the next seed. Each role retains
  at most eight compiled shapes and all state is released at the model-group
  boundary. CPU JAX tests cover primary and affinity load/trace reuse, resume
  without runtime import, mutation, continued failure and executable eviction.
  Released-weight CUDA latency, peak memory and numerical parity remain
  deployment gates, so no GPU speedup is claimed here.
- **Boltz-2 drops the flat host checkpoint mapping before layer stacking.** The
  nested parameter tree already owns the same NumPy leaves; retaining the flat
  dictionary kept the unstacked per-layer arrays alive while their stacked
  replacements were built and transferred. On the local released confidence
  checkpoint, the loader's maximum RSS fell from 6,687,764 KiB to 4,848,852 KiB
  (1,838,912 KiB, 27.5%) while the resulting parameter tree remained
  2,026,900,352 bytes and value-identical. A lifetime regression test verifies
  that the flat owner is gone before every pre-stack operation. The final host
  transfer now uses `device_put` as well: a separate 12-shape CPU probe built
  12 `jit(stage)` programs through `jnp.asarray` and zero through the direct
  transfer. Float64, float32, int64, int32 and boolean leaves retained the same
  values and JAX-canonical dtypes.
- **OpenFold3 repeated-layer weights are stacked before device transfer.**
  The released checkpoint contains seven homogeneous scanned stacks whose
  parameter leaves total 1,444,290,304 bytes (1.35 GiB). They previously
  arrived at `jit` as one argument per layer, so XLA emitted `concatenate`
  copies of the weights into its temporary arena. Mapping now keeps checkpoint
  arrays on the host, stacks each homogeneous sequence once, and transfers only
  the stacked tree. The in-graph copies really do go: a 1,003-token lowering
  carries 18 concatenate operations with pre-stacking against 491 without it,
  436 of them parameter-shaped. What that does *not* buy is peak memory. At
  1,003 and 3,012 tokens the compiled arena, arguments and output are identical
  to four decimals either way, and eight measured runs all peaked at 9,997 MiB
  -- at real sizes the peak is set elsewhere, and a copy removed underneath it
  does not move it. Both statements hold at once; the earlier "drops a full
  stacked copy from temporary memory", measured on a 4-layer CPU compilation,
  is true there and does not surface at model scale.

  What it does buy is time: about 1.5 s of a 21.6 s warm run at 1,003 tokens,
  roughly 7%, with the arms disjoint and the slower arm slower in both orders
  including when it ran first on a cooler card. The mechanism is unproven -- a
  single 1.35 GiB copy is milliseconds at HBM bandwidth, so what fits is the
  24-block diffusion transformer being re-stacked inside the sampling loop and
  paid once per rollout step. Separately, it costs about 0.16 s per process on
  the host at checkpoint mapping; that is a different machine and a different
  frequency from the device saving and the two should not be netted. The
  heterogeneous four-block MSA path and the two unscanned
  conditioning-transition loops keep their original layouts.
- **AlphaFold 3 parameters follow the explicitly selected device.** A run on
  device 1 used to load the 1,146,811,260-byte checkpoint on JAX's implicit
  device 0 before the device-pinned model copied it across. Native prediction
  now runs inside the selected default-device context, so lazy parameter loads
  and other unpinned allocations start on the requested device. A forced
  two-device probe places all 405 checkpoint array leaves directly on device 1.
- **`confidence_sample_sequential` for ESMFold2**, off by default. Runs the
  confidence head one structure at a time instead of batching every sample
  through it. This model's peak is a single temp arena sized
  `num_samples * L^2 * 4*c_z`, and that head is where the sample factor enters
  it: at 1,003 tokens with five samples on a 96 GB card, peak falls from 56.0
  to 30.2 GiB and the arena from 35.99 to 11.31, with warm time unchanged at
  28.3-28.6 s either way. Coordinates are bit-identical across all five
  samples; the only difference a user could see is up to 1e-5 of pLDDT, which
  a float32 control collapses to 3.6e-7, so it is bfloat16 reduction-order
  rounding from the changed matmul shapes rather than anything about sample
  independence -- and it was measured on one target at one seed, so it is not
  a bound.

  Read the 45% as a property of that measurement and not of the option: the
  arena is quadratic and this divides only the head's share of it, so the
  fraction moves with length and sample count. It does not raise the size the
  model can reach, and that is now a named tensor rather than an inference --
  at 2,096 tokens the allocator asks for `bf16[1, 2096, 2096, 2048]` on top of
  64.6 GiB already resident and never gets it. Dividing a quadratic by 3.18
  buys its square root in reachable length, about 1.8x, which was not enough.
  Protenix carries the same switch and defaults it on; whether this one should
  follow is a decision about that 1e-5 rather than about the memory.
- **OpenFold3 compiles 3 programs before predicting instead of 104.** Every
  `jnp.asarray` on a NumPy array is a traced operation, so it compiles one
  `jit(stage)` program per distinct shape and dtype -- and none of them is
  slow enough to clear `jax_persistent_cache_min_compile_time_secs`, which
  FoldJAX leaves at JAX's one-second default, so they are compiled again on
  every process start rather than served from the persistent cache. Loading
  weights accounted for 86 of them and the CLI's feature dictionary for 15.
  Mapping the checkpoint now uses `jax.device_put`, which transfers rather than
  traces, and the CLI hands its features to the jitted call as NumPy, placing
  them only under `--no-compile`, which dispatches per operation and would
  otherwise re-transfer. Around 0.55 s comes off every process start.

  Read that as startup latency and not as throughput: the cost is per distinct
  shape, not per byte, so it is a fixed half-second -- about 8% of a 76-token
  job and about 0.25% of a 3,012-token one. The parameter tree hashes
  identically to before, dtype canonicalisation is unchanged, and the arrays
  stay uncommitted with the sharding they had, so the context-parallel path is
  untouched. The lowered StableHLO of the CLI's graph is identical with placed
  and with NumPy features, and both entry points write byte-identical outputs.
- **A query with no templates no longer runs the template stack four times.**
  Upstream's released width is four templates, so a query without any is
  featurized as four *identical* empty ones -- the same all-gap restype, the
  same zero distogram, masks and unit vectors. The stack runs once per row and
  averages, and nothing downstream distinguishes rows, so it computed one
  tensor four times and divided the sum by four. Collapsing runs that are all
  equal to a single row saves 484 MiB of entry arguments at 1,003 tokens and
  4.26 GiB at 3,012, and drops three of four template-stack evaluations in
  every recycling cycle: peak falls 415 MiB at 1,003 tokens with no time
  difference the measurement resolves, and 6.3 GiB with 16.3 s -- 10.4% and
  6.8% -- at 3,012, the latter confirmed with the arms run in both orders so a
  warming card cannot explain it. This is a divergence from upstream in work
  and not in value: with `n` identical rows the reduction is `n * x / n`, and
  the written structures and confidences come out byte-identical at both sizes,
  including a four-chain complex, against a rerun floor of exactly zero. It
  declines to collapse when any row differs and when the padding mask is not
  all-active, and the padded path was checked too -- a collapsed feature padded
  back to four rows carries a mask of `[1, 0, 0, 0]`, so the denominator stays
  one.
- **The MSA outer product carries a byte ceiling, like every other widening
  operation here.** `outer_product_mean` widens `[M, N, C]` into `[N, N, C, C]`
  before `linear_out` narrows it, and at 1,003 tokens that intermediate is
  1,965 MiB and exists twice. It was the one widening operation without a
  ceiling of its own -- the transitions and the triangle projections both have
  one -- and it blocked only when the chunk policy handed it a size, which
  upstream's table does not do at or below 1,024 tokens. So the sizes where
  this tensor dominated were exactly the sizes that built it whole. The
  program's buffers at or above 512 MiB fall from 15 totalling 16.30 GiB to 12
  totalling 10.55 GiB, and measured peak at 1,003 tokens falls 8.25 to 7.79 GiB
  with warm time unchanged at 57.5 s. The chunk table is untouched: a policy
  asking for a smaller block still gets it, and `chunk_size=0` still builds it
  whole.

  The blocks are sequenced by a loop rather than emitted as a list, and that is
  the saving rather than a style choice -- measured, the list form leaves 1,281
  MiB simultaneously live against the loop's 519 MiB, although both name the
  same tensors. Blocking is bit-identical to the dense form in float64 across
  60 shape-and-block pairs including non-multiples and the clamped last block;
  in float32 it differs by 1-2 ulp from GEMM re-tiling, which moves a
  1,003-token structure 0.119 Å against a 0.197 Å floor between two runs of the
  unchanged code. Under a context-parallel mesh the ceiling is deliberately
  inert: turning it on there halves the widest per-device buffer and doubles the
  collective traffic, a trade an interconnect has to settle.
- **Features the graph never reads are no longer copied to the device.**
  Boltz-2's featurizer produces 78 arrays and the compiled program reads 31.
  `jax.jit` already dropped the other 47 from the program -- and drops them
  before the rest are transferred -- but the call site placed the whole
  dictionary first, so those buffers existed for the whole run anyway: 306 MiB
  of them at 1,003 tokens and 1,326 MiB at 2,096, since the term is quadratic
  in token count. `disto_target`, a training label of `f32[N, N, 1, 64]` that
  no inference path reads, was 246 MiB of it on its own. Placing them also
  compiled one transfer program per array -- 142 of the 143 programs a cold run
  built. The compiled path now hands the featurizer's NumPy arrays straight to
  `jax.jit`; the two callers that cannot use the pruning, eager steering and
  context parallelism, still place theirs. The lowered StableHLO of the whole
  graph hashes identically either way, so no prediction changes.
- **A context-parallel mesh proves it moves data before anything runs on it.**
  Entering `--cp-devices P` now shards a four-mebibyte array whose rows carry
  their own index, asks for it back replicated, and refuses to continue if the
  rows another device owned did not come home. A device count is not an
  interconnect: on one tested node every context-parallel run compiled, ran to
  completion, and returned finite numbers with one shard's rows correct and the
  rest zero, while `jax.device_count()` reported four the whole time. The check
  costs about twelve milliseconds after the first mesh of a given shape, and
  `FOLDJAX_SKIP_MESH_CHECK=1` turns it off for anyone who would rather have the
  silence.
- **Context parallelism (`--cp-devices P` / `cp_devices`).** Five backends --
  OpenDDE, Protenix, Boltz-2, OpenFold3, and ESMFold2 -- can shard their pair
  representations across `P` local JAX devices, the JAX form of OpenDDE
  upstream's Fold-CP `1 x P` mode. One process, no launcher: each consolidated
  graph carries `with_sharding_constraint` row shardings and the SPMD
  partitioner places the collectives; only triangle attention routes through
  `shard_map`, because its blocked row loop iterates the sharded axis
  (ESMFold2 has no triangle attention and needs no `shard_map` at all). Like
  upstream's mode it runs the XLA triangle kernels in CP mode -- the fused
  cueq/tokamax kernels are FFI calls the partitioner cannot split, and asking
  for one fails loudly; single-device runs keep their fused defaults
  unchanged. Verified numerically against the single-device programs on a
  forced four-device CPU mesh: module-level pair-stack parity for every
  covered backend at a token count indivisible by the mesh, plus real-weight
  end-to-end runs where local weights exist (OpenDDE 6.1e-4 Å, Protenix
  7.3e-4 Å max coordinate diff at fp32; Boltz-2 sits inside its own
  documented fp32 reduction-order floor, so its CP gate is the confidence
  summaries, which agree). Boltz-2 rejects steering and affinity jobs under
  CP for now. AlphaFold 3 is excluded: its vendored upstream implementation
  is authoritative and runs a Triton attention kernel the partitioner cannot
  split, and it is the least memory-bound of the six. Measured on real
  hardware (4x RTX A5000 24 GB, Slurm): a 1,003-residue OpenDDE job that
  OOMs one 24 GB card completes at CP=4 with balanced per-device peaks of
  11.7 GiB (bf16) / 18.0 GiB (fp32) -- 2.53x below the 45.5 GiB single-device
  fp32 peak, the gap to 4x being the triangle-mult all-gather that a future
  ring pass reclaims. Multi-GPU runs need `NCCL_P2P_DISABLE=1` on the tested
  nodes, and homogeneous cards (a mixed H100/RTX mesh is refused by XLA).
  **Multi-GPU context parallelism is not usable, and no result from it should
  be trusted without a single-device run beside it.** Measured 2026-08-20 on
  four RTX A5000s: sixteen context-parallel runs of one fixed Boltz-2 command
  -- a 254-residue target with its alignment, at the released sampling
  defaults -- returned sixteen structures 34-41 Å from the single-device
  answer, against a 0.36-0.60 Å floor between reruns of the single-device
  command itself, with complex pLDDT 0.96 falling to 0.09-0.44. Both layouts
  fail, with and without an alignment, at two diffusion steps as at two
  hundred, at one recycle as at three, and the trunk representations the same
  runs capture differ from the single-device ones by about the magnitude of
  the arrays. OpenFold3 fails the same way, so it is not one port's mistake.
  Every one of those outputs was finite.

  It is intermittent and the rate depends on the node: four RTX 6000 Ada cards
  gave three good runs and three ruined ones out of six for Boltz-2, and three
  good and one ruined out of four for OpenFold3, while the A5000 node ruined
  everything. The ruined runs agree with *each other* rather than scattering,
  so this is a deterministic wrong computation that is sometimes selected, not
  noise.

  The distributed algorithms are not what is wrong, and neither is this
  package. On four forced CPU devices the same released configuration -- real
  alignment, 200 steps, three recycles -- reproduces the single-device summary
  to four decimals in both layouts. And twenty lines of JAX containing no
  FoldJAX at all -- one `float32[N, 256]` split by rows across two GPUs of the
  A5000 node and gathered by a single replication constraint -- return the
  second half wrong from 0.25 MiB upward and identically zero from 4 MiB
  upward, on the N/2 boundary every time. That node hangs with NCCL
  peer-to-peer enabled and silently drops half the payload with it disabled.
  So the multi-GPU numbers above measure that machine, not this code.

  On hardware that passes that reproduction, it works. Four 2080ti returned
  four complete Boltz-2 predictions -- two at `1d`, two at `2d` -- within
  0.14-0.52 Å CA RMSD of the single-card run against a 0.14 Å floor between two
  single-card runs, the grid's best run landing exactly on that floor. Two RTX
  6000 Pro, which is hardware anyone would deploy on -- 96 GB, bfloat16, fused
  kernels, the released sampling defaults, a real target with its alignment --
  returned three context-parallel runs at 0.130, 0.295 and 0.342 Å against a
  0.379 Å floor, complex pLDDT agreeing to four decimals in every one, and
  trunk arrays 1.40e-2 relative from the single-card run where two single-card
  runs differ from each other by 1.4e-2. Not distinguishable from the machine's
  own noise.

  Neither run claims memory or speed: 254 tokens is below the size where
  splitting a pair tensor pays, and the 6000 Pro node holds three identical
  cards, so the square grid is out of reach there.

  The older reservation this replaces is kept below, because its observations
  still stand and one of them is now explained: silently non-finite output was
  the visible tail of this, and a finiteness check never was the gate.

  Two things are known. A node
  that hands out four cards where only three initialise returns NaN from
  every run on it, so `jax.device_count()` must equal the requested shard
  count before a result is worth reading; that check is necessary and not
  sufficient. On one node where all four cards did initialise, CP=4 returned
  non-finite confidence and coordinates in 11 of 20 runs of the same input,
  at the same rate with and without XLA autotune flags -- which refutes the
  earlier entries in this file that blamed `--xla_gpu_autotune_level`, whose
  evidence was a single run either way against what is now known to be a
  coin-flip base rate. A second node of the same card model returned six
  finite runs from six. Whether the difference is that node's hardware or a
  timing-sensitive defect in the context-parallel path is under
  investigation; single-device runs are unaffected either way.

  What the search has settled so far: the corruption appears after the trunk.
  In a failing run every trunk array is finite and bit-identical to the ones
  the passing runs used -- so nothing the sharded pair code computes is wrong
  -- and only the diffusion coordinates are bad. It is also not always loud:
  Boltz-2 produced structures 34 Å from the single-device answer with every
  array in them finite, so **a finiteness check is the wrong gate**. Mutual
  agreement between repeats does not replace it: the two corrupted Boltz-2
  runs agree with *each other* to 0.224 Å, inside the clean rerun floor, so
  a group of repeats that are all corrupted looks consistent. Only agreement
  with a single-device run settles it. That coincidence is also the sharpest
  clue so far -- a race would scatter two bad runs, and these land in the
  same wrong place, so the corruption is a deterministic computation that is
  sometimes selected rather than noise. Ten hypotheses have been measured and closed --
  autotune flags, a bad node, the step count, the diffusion-side sharding
  constraints, the sampler's `lax.scan`, NCCL's transport, the allocator,
  asynchronous collectives, and pinning the trunk/diffusion handover
  replicated.

  One candidate has been tried and measured: running the sampler inside a
  fully replicated `shard_map`, which the compiled HLO confirms removes every
  collective from the diffusion region and which reproduces the single-device
  structure to 0.0002 Å. Against the unmodified code in the same job, on the
  same cards, twelve rounds each at the released sampling defaults, it did
  not fix anything -- seven failures against four, which at that sample size
  is not a difference (p = 0.41) -- and the surviving runs of the modified
  arm still contained one 9.8 Å from its own siblings. It has been reverted:
  it costs the pair tensor's memory on every device and buys nothing
  measurable. What it did establish is worth keeping: a sharding constraint
  is a preference the partitioner may route around, `shard_map` is the
  guarantee, and the corruption survives the guarantee -- so it is not in
  anything the sampler communicates.

  Known v1 boundary: OpenDDE at ~3,000 residues still exceeds
  3x 96 GB because the fp32 diffusion side's atom-window gathers pull full
  pair tensors per device. Sharding those consumers was the planned next
  lever and is now on hold: the evidence above says that sharding *into*
  the diffusion is what the port cannot yet make reliable.
- **The square context-parallel grid (`--cp-layout 2d`).** Fold-CP's own
  layout, on OpenDDE, Protenix, Boltz-2 and OpenFold3: pair rows *and*
  columns split across a square device grid, with the triangle contraction
  run as Cannon's algorithm -- skew both operands once, then one local
  product per ring hop. Per-device pair cost falls from `O(N^2/P_rows)` with
  a full-width column axis to `O(N^2/P_total)`, and no all-gather appears in
  the contraction: the transient is two tiles. Triangle attention no longer
  falls back to row sharding under the grid; see the next entry.
  **`--cp-layout auto` deliberately stays on the 1-D layout**: every
  published number for context parallelism was measured there, and a default
  that silently changed the program would make those numbers describe a
  configuration nobody can reproduce, so the grid is opt-in until it has GPU
  evidence of its own. Verified on forced CPU meshes at 2x2 *and* 3x3 -- the
  3x3 case exists because modulo 2 a forward ring hop and a backward one are
  the same permutation, so a 2x2 grid cannot falsify a sign anywhere in the
  schedule. Every context-parallel parity gate is now mutation-tested and
  asserts from a trace-time witness that the sharded program really was
  traced.

- **Gather-free triangle attention on the square grid.** Under `--cp-layout
  2d`, Boltz-2, Protenix and OpenFold3 now run triangle attention as Fold-CP's
  own ring instead of gathering the column axis back into whole rows. The
  query tile stays resident; key, value, mask and the pair bias rotate through
  `lax.ppermute` for `sqrt(P)` hops while an online softmax accumulator folds
  each partial exactly. This closes the hole the grid entry above used to
  admit: the contraction was `O(N^2/P_total)` but attention rebuilt an
  `O(N^2/P_rows)` tensor immediately after it, so the grid's memory bound did
  not survive a whole block. It now does. Axes that do not divide the mesh
  side are padded at the `shard_map` boundary and sliced back -- a global
  array padded before it is sharded gathers nothing -- so a 253-residue chain
  still runs on a 2x2 mesh, and the padded columns are pushed below the
  smallest score the caller's mask already carries rather than zeroed, since
  they sit on the axis the softmax spans. Verified on forced CPU meshes at
  2x2 and 3x3, at token counts that divide the grid and at thirteen, which
  divides neither; each gate reads the compiled HLO and fails if an
  `all-gather` appears or the ring's `collective-permute` does not. **No GPU
  measurement yet**: the memory this is supposed to save has not been weighed
  on real cards, and `--cp-layout auto` still stays on the 1-D layout.
  OpenDDE and ESMFold2 are not wired to the ring.

- **Trunk representations (`representations=`, `stop_after="trunk"`).** Every
  model here builds a per-token *single* stream and a token-pair state before
  it predicts coordinates, and until now only two of them could hand those
  back, each under its own name and only through its native command line.
  `PredictionRequest(representations=("single", "pair"))` returns them from
  all five, and `stop_after="trunk"` stops once they exist rather than paying
  for a structure that will be discarded. The result carries a
  `Representations` handle: `result.representations["pair"]` reads one array
  from disk on access, `describe` gives its shape, dtype, axis names and --
  the part that keeps a cross-model comparison honest -- the *space* those
  axes count. Two models both call their pair state `pair`, but OpenDDE's
  `structural_pair` indexes a structural-token space its residue space does
  not cover, and nothing about the array says so. `foldjax capabilities
  --model M` lists what each model produces; ESMFold2 offers two names rather
  than three because it carries one single stream, not an input embedding and
  a trunk output.

  Underneath is a capture mechanism rather than a flag per tensor: a model
  annotates a value with `capture("pair", z)` and the traced program hands
  back the ones that were asked for. Off is free -- an un-requested tap leaves
  no trace in the jaxpr -- and the requested set is a static argument, so a
  different set is a different program rather than a stale cache hit. On costs
  liveness rather than bytes: a captured value becomes an entry output, and
  the temp arena is reused across a run where an entry output is not, which is
  the whole of the 15.2 GiB that returning Protenix's confidence logits used
  to cost at 3,012 tokens. **Nothing is returned by default**, because a pair
  representation is quadratic in token count -- half a gigabyte at a thousand
  tokens, nearly five at three thousand. A tap inside a scanned loop is
  refused unless the caller says what it is willing to spend: stacking
  OpenDDE's structural pair state over 48 refiner blocks is 63 GiB at 488
  residues, and learning that from an OOM twenty minutes in is much worse than
  learning it from the error.
- **Alignment search.** `--msa auto` (`PredictionRequest(msa="auto")`) searches
  for a protein chain that arrived without one and caches the result under
  `$FOLDJAX_HOME/msa/`, keyed by sequence and search provenance rather than by
  model, so one target run through three backends searches once. `--msa
  required` refuses to fall back to single sequence. The default stays `none`,
  which is the previous behaviour — but it now warns once per run instead of
  silently folding a protein from its own sequence. **`auto` and `required`
  send the sequence to the configured server**; `FOLDJAX_MSA_COMMAND` and
  `FOLDJAX_RNA_MSA_COMMAND` point at a local workflow instead, and RNA is
  searchable only that way.
- **Input that is not a job file.** `--sequence`/`--dna`/`--rna`/`--ligand`/
  `--ligand-smiles`, FASTA files, `.pdb`/`.mmcif` depositions (re-folded from
  their chemistry, never their coordinates), `structure:PATH` for a `.cif`, and
  a directory as a batch. Each becomes an ordinary common-schema document under
  the runtime store, so `plan` names it and the manifest hashes it.
  `Job.from_sequences`, `Job.from_fasta` and `Job.from_structure` are the
  Python spellings.
- **Templates and binding affinity in the common schema.** A polymer takes
  `templates: [{mmcif, query_indices, template_indices}]` and a job takes
  `properties: [{affinity: {binder}}]`. The residue map is required by
  AlphaFold 3, Protenix and OpenDDE and refused by Boltz-2, which aligns the
  mmCIF itself; affinity reaches Boltz-2 alone. Whatever a backend cannot
  express is refused, never dropped.
- **`foldjax show`, `foldjax doctor`, `foldjax cache gc`, `foldjax models
  --for JOB`, `foldjax --version`.** `models --for` answers from the input
  translation table, so it needs no weights, GPU or network. `cache gc` reports
  by default and deletes only with `--apply`.
- **Progress on stderr and a summary table on stdout.** Stage lines during a
  run, and the model's own top-ranked structure at the end. stdout stays JSON
  whenever it is not a terminal, so pipes are unchanged; `--json` forces it and
  `--quiet` (or `FOLDJAX_PROGRESS=0`) silences the stage lines.
- **`--resume` and `--keep-going`**, and their `PredictionRequest(resume=...,
  on_error="continue")` spellings. Resume works at *seed* granularity: a
  five-seed job that died on the fourth repeats only what is missing. Failures
  are recorded in `foldjax_failures.json` beside the runs that succeeded, and
  `foldjax.predict_batch` returns results, skips and failures together.
- **`cost.phases` in the run manifest** — preparing input, predicting and
  writing, measured separately, because one number for the whole run cannot say
  which of them to act on.
- **`py.typed`.** The public surface is annotated; downstream type checkers
  now see it instead of `Any`.

### Changed

- **AlphaFold 3 reuses chain confidence aggregates during host
  postprocessing.** `pae_metrics` already computes the minimum chain-pair PAE,
  chain-pair ipTM, per-chain ipTM and cross-chain ipTM that the result writer
  needs; the same arrays were then computed a second time and discarded. The
  writer now consumes the first results directly. On a 3,012-token four-chain
  result with the default five samples, the duplicate PAE and ipTM passes took
  1.34 s on the measured CPU. The reused arrays are exactly equal, including
  masked and NaN cases. This removes transient allocation churn but does not
  claim a lower process peak, because the necessary first pass remains.

- **Protenix and OpenDDE compute each distinct template once instead of once
  per row.** A query with fewer than four template hits is padded up to four,
  and the padding is zeros rather than the gap restype, so a query with no hits
  at all is one all-gap row plus three identical zero rows -- two distinct
  templates, not four and not one. The template stack ran the whole pairformer
  once per row and divided by the row count; it now runs once per distinct row,
  weights each by how many it stands for, and divides by their sum. At 3,012
  tokens Protenix's entry arguments fall 12.404 -> 9.430 GiB and its total
  50.145 -> 47.171; the measured run peak at 1,003 tokens is 8,953 -> 6,970
  MiB. OpenDDE reads templates once per recycle, so its stack evaluations come
  off per cycle. Structures are not bit-identical and are not meant to be: the
  divisor carries a `1e-7` epsilon that does not scale, and the running sum
  re-associates. Paired CA RMSD against a base run is 0.065 A median against a
  0.061 A rerun floor, both about 30x below the model's own 1.86 A sampling
  spread.

- **OpenDDE's bfloat16 trunk now runs the blocked triangle product rather than
  the fused kernel.** Measured on a cooled card, warmup discarded, three timed
  runs per arm alternating: 166.27 -> 143.97 s and 22.44 -> 19.85 GiB at 1,003
  residues, 405.07 -> 373.23 s and 52.10 -> 45.76 GiB at 1,531. Per-arm spread
  was 0.19-0.56% against gaps of 8-13%. The default is keyed on the trunk
  dtype: float32 is what OpenDDE ships and is unchanged, because the blocked
  path's accumulators only narrow when the trunk does. An explicit
  `PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND` still wins on either dtype.

- **The blocked triangle product holds its output at the trunk's width.** It
  was pinned to float32 on the grounds that a narrower destination would round
  every block on the way in. It would, and that is the rounding the caller
  already performs on the next line, so the results are bit-identical. On a
  bfloat16 trunk this removes the largest buffer in that program -- 4,471 ->
  2,235 MiB on Protenix's template stack at 3,012 tokens, moving the temp arena
  39.102 -> 37.367 GiB. A float32 trunk is unaffected.

- Sequences are normalized where the document is validated: all whitespace
  removed (so a YAML block scalar works), upper-cased, and nucleic acids
  checked against IUPAC with the offending position named. A chain id may be
  omitted and is assigned in document order; an id that is present but blank is
  still an error.
- Unknown document fields name the field that was probably meant.
- `capabilities` gained `common_schema_features` and `native_only_features`.
  `supports_templates` describes the model; those two describe what a common
  job document can reach, which is not the same thing and used to be implied.
- `foldjax weights fetch` refuses a download the disk cannot hold and warns
  when conversion will be tight.
- The MSA client identifies itself to the search server as `foldjax/<version>`
  rather than `protenix-jax/0.1.0`.

### Fixed

- **ESMFold2's folding trunk ran in float32 while every setting said
  bfloat16.** `ModelSettings.trunk_dtype` defaults to `bfloat16`, the released
  `config.json` sets it, and a test pinned that field -- but the field is not
  the tensors. Weights were cast, and three feature tensors were not: one
  float32 term in `z_init = z_init + rel_pos + token_bonds_encoding` widened
  `z_init`, and the `z.astype(z_init.dtype)` after it carried float32 through
  all forty-eight trunk layers. Casting those three to the compute dtype makes
  the port run the configuration it advertises. Warm wall time at 1,003 tokens
  falls from 36.5-38.8 s to 28.2-28.8 s, about 23%, measured with an ordering
  control so it is not a cooler-card artefact. Peak does not move -- 56,228
  against 56,230 MiB, where two runs of the unchanged code at different seeds
  differ by 427 MiB -- which says the pair path accounts for essentially none
  of this model's footprint; its language model is 27.8% of it. Structures
  move 0.065 Å at the median against a rerun floor of exactly 0.000 Å (a fixed
  seed is bit-deterministic here) and a sampling spread of 0.49-1.53 Å, so
  nothing a user reads changes. The gate that let this ship asserted the
  setting; the new one spies the trunk and asserts the dtypes the graph
  actually built, in both bfloat16 and float32, because pinning one would pass
  for a port that hard-coded it.
- **Resume now proves what it reuses.** A finished manifest must match the
  exact scalar request, lexical and resolved input path, input and referenced
  input content, checkpoint and referenced-local-asset stat/tree identities,
  padding, representation selection, and stop point. Any relevant metadata or
  tree change triggers a conservative rerun without rereading large checkpoint
  or asset payloads merely to write or check a manifest. The recorded structure
  must remain non-empty with the same SHA-256; representation archives must
  remain contained beneath the run root with the same names, order, shapes,
  logical dtypes, ZIP metadata, and stat identity. Legacy/incomplete records
  rerun conservatively. Trunk and BF16 archives are restored lazily, and a
  failure from an earlier batch pair no longer suppresses a later successful
  multi-seed manifest. Inputs, checkpoints, and referenced assets must remain
  immutable during backend execution; resume detects changes between runs but
  does not lock those files against concurrent replacement.
- **Built-in writers refuse non-finite public coordinates.** Boltz-2,
  ESMFold2, OpenFold3, Protenix, and the Protenix writer shared by OpenDDE now
  fail before creating a structure when any exposed atom is NaN or infinite.
  Serving-padding atoms remain outside that check, and Boltz-2's guard is an
  explicit exception that remains active under optimized Python.
- **Distributed attention and Cannon accumulation keep their numerical
  contracts at edge cases.** Boltz-2's serial, chunked, fused, 1-D, and 2-D
  pair-biased attention paths return finite zero for an all-masked key set
  instead of a uniform mean or NaN. OpenFold3's 2-D Cannon reduction now uses
  compensated fp32 tile accumulation like the other supported pair trunks. A
  forced four-device model-level gate exercises Boltz-2 conditioning through
  its atom encoder, token transformer, and atom decoder without an all-gather.
- **Unsupported representation requests now fail during request validation.**
  Capability validation understands comma-separated names and `all`, so an
  unsupported array is rejected by `plan`/resolution before model weights or a
  native runtime are loaded. A backend that exposes no representations rejects
  `all`, and an empty comma/whitespace-only selector is refused instead of
  running a trunk graph that returns no arrays. Representation capture is also
  single-seed until the public result can carry more than one archive handle;
  multi-seed capture now fails early instead of silently losing handles.
- **OpenDDE common inputs no longer advertise fields its released runtime
  discards.** Mapped templates and RNA MSAs now fail in compatibility checking
  and materialization; protein MSAs are unchanged. Native OpenDDE passthrough
  retains the existing `templatesPath`/RNA `unpairedMsaPath` warning/drop
  behavior required to match the shipped `use_template=False` and
  `use_rna_msa=False` defaults.
- **The hermetic CI gate no longer waits on RCSB.** Tests marked `network`
  remain available as an opt-in integration surface, but the canonical CPU
  suite excludes them instead of depending on an external service and its
  120-second timeout.
- **OpenFold3 predicted nothing at all.** Every OpenFold3 run raised
  `TypeError: released_config() got an unexpected keyword argument
  'returned_representations'` before the model was reached, with no option
  needed to trigger it and through both `foldjax predict --model openfold3`
  and `openfold3-jax-predict`. The backend passed the trunk-representation
  request into a constructor that had never been given the two parameters,
  though `InferenceConfig` carried both fields already. Weights are
  access-gated, so no test that runs a prediction could have caught it; the
  gate that now does reads the two sources and asserts every key one passes is
  a parameter the other accepts.
- **A run that stopped at the trunk still tried to write a structure.** The
  trunk graph returns before the sampler, so there are no coordinates:
  OpenFold3 raised `IndexError` reading an empty coordinate shape and ESMFold2
  raised `KeyError: 'sample_atom_coords'`. Both are backends that write
  structures themselves rather than delegating to their model's command line,
  and both are now fixed -- ESMFold2 had the branch but below the writer, which
  is the same as not having it, so the new gate checks the order and not just
  the presence.
- **The square context-parallel grid was unreachable for OpenDDE through the
  common option.** `opendde-jax-predict --cp-layout 2d` ran while `foldjax
  --model opendde --option cp_layout=2d` was refused as an unsupported option,
  because the backend's option set was missing the entry the other three
  two-dimensional models had.
- **A representation archive disagreed with its own manifest.** Boltz-2,
  OpenFold3 and ESMFold2 wrote a leading batch axis of one, so `shape` carried
  four numbers beside three names in `axes`, while Protenix and OpenDDE wrote
  three of each -- one feature with two contracts, in the file a consumer reads
  to learn what it is holding. Arrays are now conformed to the axes their spec
  names, and an array that is not its spec is refused rather than written under
  a description that does not fit it.
- `python -m foldjax.cli` runs the command line. It used to import the module,
  define `main`, call nothing, and exit 0 with no output, which reads as a
  prediction that produced nothing rather than as a command that never ran.
- Ctrl-C ends a run with one line and exit code 130 instead of a traceback
  through FoldJAX's internals.
- A read-only output directory, a full disk and other `OSError`s are reported
  as the machine facts they are rather than as defects. A full disk names the
  store and the command that reclaims space; a missing optional package names
  the extra that provides it.
- The AlphaFold 3 runtime error names `foldjax runtime prepare --model
  alphafold3` instead of an internal function.
- `foldjax.progress` resolves its stream at write time. Holding the stream
  meant writing to whatever stderr *was*, which could fail a prediction that
  was otherwise fine.

## 0.3.0

FoldJAX became one standalone package: AlphaFold 3, Boltz-2, ESMFold2, OpenDDE,
OpenFold3 and Protenix inference with no sibling repository, plus opt-in
mask-aware padding for reusable JAX executables. See the git history and
`docs/ports/` for how each port got there.
