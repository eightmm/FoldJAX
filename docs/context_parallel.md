# Context parallelism

FoldJAX exposes context parallelism for the quadratic pair representation used
by AF3-family folding models and, for Boltz-2, for the atom-window diffusion
path. The implementation follows Fold-CP's central distributed algorithms in
JAX:

- pair rows may be split over one mesh axis (`1d`);
- a square mesh may tile both pair axes (`2d`);
- triangle multiplication uses Cannon-style tile rotation with compensated
  fp32 tile accumulation;
- triangle attention keeps queries resident and rotates key, value, mask, and
  pair-bias tiles;
- triangle attention uses a two-pass global-maximum softmax and compensated
  fp32 numerator/denominator accumulation;
- Boltz-2 atom/query windows are distributed over CP rows and exchange only
  fixed-width halos and sparse token/pair data.

The two-dimensional pair hot paths do not reconstruct a full token axis. HLO
regression tests reject an `all-gather` in ring attention and verify the
expected explicit collectives for the atom-window adapters.

## Supported model paths

| Model | 1D pair CP | 2D pair CP | Atom-window CP | Notes |
|---|---:|---:|---:|---|
| Boltz-2 | yes | yes | yes | Cannon/ring pair core, CP-row atom windows, halo exchange, sparse token/pair routing; the 2-D layout also splits the MSA alignment depth on the grid's column axis |
| Protenix | yes | yes | yes | Pair trunk and confidence pair path use the common pair core; the diffusion atom graph is distributed over CP rows (`cp_atom_windows`, default on) |
| OpenDDE | yes | yes | yes | Structural-token refinement uses the Protenix pair primitives; its diffusion module calls the same Protenix denoiser with the atom graph distributed over CP rows (`cp_atom_windows`, default on), aligned on the *structural* token axis |
| OpenFold3 | yes | yes | yes | Pair stack, template stack, confidence pair re-embedding; the diffusion atom graph is distributed over CP rows (`cp_atom_windows`, default on) with an index-driven ring gather instead of a halo |
| ESMFold2 | yes | yes | no | Pair-row constraint path; the grid runs the shared Cannon schedule for the triangle contraction and blocks the pair transition inside the shard. This trunk has no triangle attention, so there is no ring to run; at 3,012 tokens the grid is the only arm that finishes, so `auto` picks it on a square count |
| AlphaFold3 | no | no | no | The vendored publisher runtime is not rewritten for FoldJAX CP |

`cp_layout` chooses the mesh, and `auto` is resolved per port:

| Model | `auto` on a perfect-square count (4, 9, 16, ...) | `auto` on any other count |
|---|---|---|
| OpenDDE | `2d` | `1d` |
| Boltz-2 | `2d` | `1d` |
| OpenFold3 | `2d` | `1d` |
| ESMFold2 | `2d` | `1d` |
| Protenix | `1d` | `1d` |

An explicit `1d` or `2d` is passed through unchanged on every port; `2d` on a
non-square device count is refused rather than degraded to rows; and
`cp_devices=1` is the serial program whatever the layout says. What runs is
recorded resolved rather than as spelled -- the compiled program's static
topology, the out-of-memory diagnosis, and the compile-cache namespace the
adapters write -- so an omitted layout shares a namespace with the explicit
spelling that builds the same mesh, and the other spelling keeps its own.

OpenDDE, Boltz-2, OpenFold3 and ESMFold2 pick the grid on the strength of the
same four-card deployment node (4 x 96 GiB, 2x2 mesh), each on its own target.
On a 2,096-token 5DEI, OpenDDE completes there at 32.1 GiB per device where the
serial run, 1-D on two cards and 1-D on four cards all run out of memory, at
0.59-0.75 Å CA RMSD to the deposited structure against a 0.007 Å two-process
floor. Boltz-2 runs at 16.6 GiB per device against 19.0 in the 1-D layout and
18.5 serial, with coordinates within 0.11 Å of its serial run on one sample and
deposited RMSD unchanged to two decimals (0.44 0.43 0.42 0.47 0.43). Protenix
measured the other way on the same node -- 11.6 GiB per device in the grid
against 10.7 in the 1-D layout -- so it keeps rows.

**OpenFold3's row is a completion where nothing else completes.** Its target
is a 6,568-token one -- eight 821-residue chains -- and at that size a single
card runs out of memory and so does the 1-D layout on four cards, on *every*
rank, each asking for a 101 GiB arena. The grid finishes it: 42,209 MiB per
device in 10,554 s (2 h 56 min), job 1518, measured before two further memory
fixes landed on the same day (the pair-transition local-row chunk `5ec4a91`
and the context-parallel `diffusion_chunk_size=1` default `67ddd48`), so the
per-device figure is an upper bound on what the same run costs now. A CPU SPMD
probe attributes the gap to the 1-D triangle multiplication's full-width
operand all-gather -- `f32[1, 128, N, N]` twelve times over, 20.6 GiB at 6.5k
-- which the Cannon path replaces with half-width tiles. Because the other
arms do not run, there is no serial-versus-1-D-versus-2-D parity arm at this
size for this port: what is answered here is the per-device ceiling (item 2 of
the deployment list below), not item 1. **The grid is also the slower layout
for OpenFold3 and its wall time at ordinary sizes is unmeasured** -- there is
no 2,096-token row yet -- and it is chosen for the memory ceiling per the
project rule, the same trade the other two ports take.

**ESMFold2's grid is the only arm that finishes, and it is not slower.** Both
pair axes go on the mesh, the triangle contraction runs the same Cannon
schedule as the other pair cores, and the pair transition takes its row block
inside the shard; its trunk has no triangle attention, so the ring schedule the
other ports need has nothing to run here. The arena the grid quarters is a
single one -- a folding-trunk pair tensor, quadratic in the token count and
with no sample axis in it -- and a serial run exhausts 96 GiB at 3,012 tokens.
On the four-card node at that size the grid completed both benchmark passes in
30 minutes: 772 s and 23,430 MiB per device for five structures. The 1-D layout
on the same four cards did not finish a single pass in one to two hours, at
3,012 tokens or at 4,100, so **the comparison here is not a peak against a
peak, it is a completion against a wall clock that ran out** -- there is no 1-D
number to quote at either size. The mechanism is this trunk's own: the
language-model encoder's pair stack keeps a dense einsum and an all-reduce
under the row mesh, which the Cannon schedule replaces with half-width tiles.
Alongside the card, the arithmetic and the partitioning are checked on forced
CPU meshes: the pair trunk and the MSA encoder block agree with the unsharded
program to 1.1e-5 on a 2x2 grid and 1.8e-5 on a 3x3 one against a 3e-5
tolerance, a device holds `(N/side)^2` of the pair state, and the partitioned
trunk contains no `all-gather` and no value carrying both token axes at full
width.

**The grid is the slower program everywhere it has been timed except
ESMFold2.** Boltz-2 at 2,096 tokens costs about 3.5x the serial wall time in
it, and the 1-D layout is not free either; the OpenFold3 figure at that size
is not measured at all. On those ports the layout is chosen for the memory
ceiling and nothing else, which is what context parallelism is for here:
fitting targets that otherwise do not run, so on a square device count where
the job already fits and wall time is what matters, ask for `1d` explicitly.
**ESMFold2 is the exception**: at 3,012 tokens its 1-D arm is the one that does
not finish, so asking for `1d` there buys nothing -- see its paragraph above.
Below the sizes that need a mesh at all, no ESMFold2 wall-time comparison
between the two layouts has been recorded either way.

`cp_layout` is not the only default a mesh moves. A mesh is asked for because a
target does not otherwise fit, so under one a capacity-first default beats a
speed-first one, and the knobs that answer to that are resolved per port too:

| Model | Knob | Omitted, serially | Omitted, `cp_devices > 1` |
|---|---|---|---|
| OpenFold3 | `diffusion_chunk_size` | unchunked at every released schedule (the width engages above five samples) | `1` |

**Why OpenFold3's rollout denoises one sample at a time under a mesh.** The
sample axis is the one axis sharding does not touch, and the rollout's widest
value hangs off it: the diffusion pair conditioning is hoisted out of the
sampler loop at a leading axis of one and widened to the rollout's sample width
at the point of use, which at 6,568 tokens on four devices is
`f32[5, N/4, N, 128]` -- **25.7 GiB per rank**, held by the rollout and by the
24-block diffusion transformer. `diffusion_chunk_size=1` takes the same value
to 5.1 GiB, and it takes nothing else with it: the conditioning is constructed
per chunk rather than retained outside the loop, every noise draw is narrowed
from the full sample width rather than redrawn, and augmentation still sees
every sample. What it costs is wall time: the denoiser runs one sample where it
would have run five. How much is not measured on this port at the sizes a mesh
is for -- at 4,100 tokens the unchunked arm has no wall time because it does not
run at all, and the nearest number for this knob is Boltz-2's `+33%` at 3k
tokens in
[docs/scale-rows-master-2026-09-10.md](scale-rows-master-2026-09-10.md).

A caller with the room says so, and an explicit value always wins: any width at
or above the sample count is the unchunked rollout -- `--option
diffusion_chunk_size=5` at the released five samples -- as is `None` through
the Python API. Serial runs are untouched. The resolved width, not the
spelling, is what the compile-cache namespace records, so an omitted option
under a mesh and an explicit `1` are one namespace while the unchunked rollout
keeps its own.

The gates are in `tests/models/openfold3/test_diffusion_chunk_cp.py`: the
resolution rule and the recorded identity in-process, and the program itself in
a four-device CPU subprocess, where no `[samples, N/4, N, channels]` value
survives the chunked rollout, one does survive without the chunk -- the
tripwire that gives the bound its power -- and all five samples come back
within the chunk loop's own float32 tolerance. The census is read on the
per-device SPMD text, because the pre-partition HLO shows a local tile only
inside a `shard_map` body and carries global shapes everywhere else; it is
counted there too, and both numbers are printed.

## Runtime and input placement

The active CP runtime is immutable and task-local, so concurrent requests
cannot overwrite one another's mesh or leak a topology into an unrelated JIT
trace. Its layout, shard count, grid shape, and mesh-axis names form an explicit
compilation identity.

Model parameters and unrecognised inputs remain replicated. A conservative,
name-based registry places only known pair features directly on the pair mesh
when both semantic axes divide evenly. Boltz-2 may additionally place explicitly
whitelisted linear atom features directly on CP-row shards. Dense coupled
atom/token maps are intentionally excluded from generic independent sharding;
the production diffusion path instead uses model-specific sparse routing.

Precomputed Boltz-2 diffusion noise tapes also enter the compiled program
already sharded on their atom axis. They are not first copied in full to every
device. Under a two-dimensional mesh, atom/query data are sharded over CP rows
and replicated over CP columns.

## Failing on a device that runs out of memory

XLA's collective rendezvous waits forever by default
(`xla_gpu_nccl_termination_timeout_seconds` is `-1`), so a device whose
allocator failed used to leave the others waiting there until the scheduler
killed the job. `foldjax predict --option cp_devices=N` and `foldjax cache
warm` compose `--xla_gpu_nccl_termination_timeout_seconds=600` into
`XLA_FLAGS` before JAX loads; `FOLDJAX_CP_RENDEZVOUS_TIMEOUT` changes that
number, a negative value declines the bound, and a value already in
`XLA_FLAGS` is kept.

The flag is read once, when the backend initialises. A run that reaches
`context_parallel` with a backend already up -- the Python API, and the native
`--cp-devices` command lines, which load weights first -- cannot be given the
bound from inside the process; it is warned instead, and the fix is
`XLA_FLAGS=--xla_gpu_nccl_termination_timeout_seconds=600` in that run's
launcher environment. A process pinned to the CPU is left alone: there is no
NCCL rendezvous to bound, and `XLA_FLAGS` would reach every child it spawns.

An out-of-memory failure that surfaces as a Python error also reports which
mesh produced it, since the figures it quotes are one device's budget rather
than the job's. When the bound fires instead, XLA ends the process from its
own rendezvous and that log carries XLA's allocator message.

## The two-dimensional ring's row block

Triangle attention is a batch of independent attentions, one per row of the
pair representation, and nothing in its softmax crosses a row. The serial and
1-D paths use that to bound memory: they loop over blocks of rows, projecting
`q`/`k`/`v` for one block inside the loop, so the projections and the
`[rows, heads, N, N]` score tensor are bounded together.

The 2-D ring does the same, with the whole rotation inside the row loop. One
block of local rows projects its own `q`, `k`, `v` and gate, runs both ring
passes over every key tile, and writes its rows of the output; the loop is a
`lax.scan`, because an unrolled Python loop lets XLA schedule the blocks
together and keep every block's tile live at once. Communication is unchanged
in bytes -- every skew and hop keeps a device's grid row, so slicing the rows
before the skew sends the same payload in more, smaller `collective-permute`s
-- and the two loop-invariant bias skews are hoisted above the loop. Wall time
may grow; the layout exists for targets that otherwise do not run.

What the loop then holds is the output destination, at the value dtype, plus
the pre-projection pair tile it slices and the bias. Blocking the *query* axis
instead, which this ring did before, bounded only the score tile: at 1,024
OpenDDE structural tokens on a 2x2 CPU mesh the pass-2 loop carried five
`f32[512, 12, 512, 32]` tensors -- `q`, `k`, `v`, the accumulator and its
Neumaier correction -- 1,969.0 MiB, quadratic in the local width and 67.5 GiB
per device at 6,144 structural tokens. The row block leaves 627.3 MiB of that
carry, of which 192 MiB is the destination.

The block is one knob, and it is the one each port already had:
`triangle_att_q_chunk_size` (Protenix, OpenDDE), `q_chunk_size` (Boltz-2) and
`chunk_size` (OpenFold3) reach `resolve_ring_row_block`, which follows the
serial path's rule -- `rows * heads` near 288, capped at 64 rows, narrowed
further when one block's score tile would pass 8 GiB, floored at 8 rows. A
non-positive value asks for one block, which is the unblocked ring and the
program the ring lowered to before blocking existed.

## The ring tile kernel (experimental, GPU only, unmeasured)

Under a mesh every fused kernel in the trunk resolves to an XLA path, because
a kernel that consumes the whole token axis cannot be partitioned, and that
resolution is most of the 2-D wall penalty: Boltz-2 at 2,096 tokens runs 231 s
serial with its fused kernels, 555 s serial with the XLA ones, and 973 s on
the 2x2 grid. The ring's per-step tile, though, is a *local* attention over
`[rows, heads, queries, keys]` -- a shape a fused kernel can take.

`--option triangle_attention_ring_kernel=tokamax` (Boltz-2 and Protenix)
replaces the tile with tokamax's Pallas/Triton attention, called with
`normalize_output=False` and `return_residuals=True` so it hands back the
tile's unnormalised numerator together with the softmax maximum and
denominator. Those three are exactly what a statistics merge needs, so the
tiles combine without the tile ever materialising its score tensor -- which is
also what the ring's two-pass body could not do, because a kernel normalises
its tile before the ring gets to see it.

It is a **different program and different arithmetic**, not a faster spelling
of the default:

- one rotation rather than two, with `V` travelling with `K` from the first
  step, because there is no global maximum to fix in advance;
- each tile is normalised against its own maximum and rescaled onto the
  running one, which is the repeated rescaling the two-pass body was written
  to remove. The Neumaier compensation on both the numerator and the
  denominator is carried across the merge;
- a query row with no valid key anywhere in the ring comes out as zeros.
  Tokamax masks with `finfo.min` rather than `-inf`, so such a row is forced
  back to `(-inf, 0, 0)` before the merge sees it. The default path instead
  leaves a row whose keys are all *mask-biased* (`-1e9`, not absent) reducing
  over those keys as the serial path does; only genuinely absent keys are
  `-inf` there. The two therefore differ on rows the model masks away
  downstream, and nowhere else.

`xla` is the default and remains the program every 2-D measurement in this
repository describes. The option is refused rather than downgraded: off the
GPU backend, without tokamax installed, or without a 2-D layout to be a body
of, it raises. It also forks the compilation-cache namespace, because it is a
different program -- but the value travels in a `ContextVar`, which no `jax.jit`
cache key carries, so the retained in-process runner does **not** fork on it:
one value per process.

No wall time or peak is recorded here yet. The experiment that would record it
runs one layer's ring on four cards in both kernels, asserts the fused
dispatch, compares the outputs, and only then times a full prediction pass.
Until it has run, the option is an implementation with a CPU-proved merge and
no measurement.

## Boltz-2's MSA stack on the grid

Everything above shards a quadratic `[N, N, C]` pair state. Boltz-2's MSA
module carries a second large tensor that is not pair-shaped at all: `m` is
`[B, M, N, C]`, with `M` the alignment depth -- 8,192 rows at full depth,
because upstream subsamples only when `--subsample_msa` is given. Sharding
only its token axis leaves that tensor merely halved on a 2x2 grid while the
pair stack is quartered, and the MSA stack is what owns the peak: of the
16,956 MiB per-device peak-live set measured at 2,096 tokens on four cards,
the residual stream is two `f32[1,8192,1048,64]` (2,096 MiB each), the MSA
transition's hidden-chunk accumulators eight BF16 tiles of 1,048 MiB, and
PairWeightedAveraging's output one more `bf16[1,8192,1048,64]`.

Under the 2-D layout the MSA tensor therefore uses **both** grid axes, and its
column axis means something different from the pair tensor's:

```text
MSA  P(None, cp_col, cp_row, None) -> [1, M/side, N/side, c_m]
pair P(None, cp_row, cp_col, None) -> [1, N/side, N/side, c_z]
```

Rank `(r, s)` owns alignment rows `M_s` against token positions `N_r`, while
the pair tile at that same rank owns token rows `N_r` against token columns
`N_s`. Reusing the column axis for two meanings is sound -- they are different
tensors -- but it is not a reshard, so the two operators that read one layout
and write the other bridge them explicitly:

- **PairWeightedAveraging** derives its weights from the pair tensor per
  `(i, j)` and applies them to every alignment row independently. Each rank
  projects its own pair tile to head logits, masks them *there* -- on the rank
  that owns the key block, so the gathered value is the logit the serial
  softmax is fed -- and `all-gather`s those logits along the column axis. The
  softmax then normalises over the complete key axis. The widened values ride
  a `collective-permute` ring along the row axis, which keeps the alignment
  shard fixed; after `t` hops a rank holds token block `(r - t) % side` and
  contracts it against that block of the gathered weights. Nothing is summed
  along the column axis: those ranks hold different alignment sequences. What
  travels is `[B, heads, N/side, N]` -- 70 MiB at 2,096 tokens and eight heads
  -- against the 4.4 GiB the widened values would have cost.
- **OuterProductMean** needs, for output block `(I, J)`, the `a` operand of
  token block `I` and the `b` operand of block `J` over the whole alignment.
  `b` and its mask ride the same row-axis ring, so at step `t` every rank of
  grid row `r` holds key block `(r - t) % side` -- the same block for the
  entire grid column, which is what makes the column reduction coherent. Each
  rank contributes its own alignment shard's numerator and its own mask count,
  both are `psum`ed along the column axis, the mean is taken *after* that
  reduction against the reduced count and the same clamp, and the output bias
  is added once. Exactly one rank of each reduction group owns the block the
  group computed and keeps it. Output rows and, on the AMP path, the hidden
  axis are blocked, so no collective carries more than the byte budget one
  serial block carried -- and the blocks are chained through
  `optimization_barrier`, because a loop of *independent* collectives is one
  XLA merges into a single collective whose every operand and every result is
  co-live. Measured at 2,112 tokens on a 2x2 CPU mesh, the 272 reductions of
  this loop and the ring merged into two all-reduces of 256 and 16 members,
  8,712 MiB of results; chained, the largest is one block's 33 MiB. It is the
  same trade the triangle ring's row block took when it became a `lax.scan`:
  one block's communication no longer overlaps the next block's arithmetic.
- **The transition, the channel norms and the gates** are elementwise in both
  split axes and contract only over channels, so they need no collective --
  but the transition's row block now has to be taken *inside* the shard, for
  the reason the pair transition's does: axis 1 is a sharded axis there.
  `cp_msa=True` still declares that the block survives; on the grid it marks
  the local tile.

The alignment depth is the one axis no padding plan aligns, so the module pads
it once, for the whole stack, at the entry features -- `m` is never returned,
so those rows are never removed again, and the masked padding keeps them out
of OuterProductMean's mean and so out of every pair value the module returns.
The token axis is already grid-aligned by
`align_padding_plan_for_context_parallel`, so its padding is inert on the
deployment path -- a trunk-only capture or `cp_atom_windows=false` skips that
alignment, and an unaligned width then takes the per-layer pad, which is
correct and costs one copy of the pair tensor a layer. One detail is load-bearing there:
the pair mask's token padding is `-1` rather than `0`, which puts a padded key
column one penalty level below a masked real one. A query row with any
unmasked key cannot tell the two apart, and a row with *none* must stay a
softmax over the real tokens rather than pick up columns that do not exist.

Neither bridge is bitwise equal to the serial program and neither can be:
summing a shard at a time reassociates both reductions. The contract is
CP-versus-serial tolerance, and `tests/models/boltz2/test_msa_grid_bridges.py`
pins it on 2x2 and 3x3 CPU grids -- both operators against serial in FP32 and
BF16 and on both sides of `_NATIVE_CHUNK_THRESHOLD`, so the released regime's
per-head groups and hand-added bias are covered; uneven padding on both axes;
an alignment mask that zeroes every row of one depth shard; an empty mask
whose output must be the bias and nothing else; the whole module against both
the serial and the 1-D residual; and a before/after census of the per-device
shape, the widest value carrying the stream's channel width, and every
`all-gather` in the program.

Compiled through the released CLI path on four forced CPU devices at 2,112
tokens, every MSA value in the grid program falls from `[1,960,1056,*]` to
`[1,480,1056,*]` -- 1,150 of the 32-channel shape and 97 of the 64-channel one
become 430 and 67, with none of the full-depth shape left -- and its temp
arena falls 29.5%, from 18,423,041,840 to 12,995,040,856 bytes (31.1% at
1,088). The serial and 1-D arms are byte-identical before and after, arena
included.

## Boltz-2 atom-window path

Boltz-2 distributes the atom diffusion graph rather than only the pair trunk:

- fixed-width half-window halos use `collective-permute`;
- token-to-atom gathers rotate linear source shards;
- atom-to-token means use reduce-scatter semantics and accumulate sums/counts in
  fp32 before restoring the model dtype;
- token-pair values are looked up from rotating two-dimensional pair tiles;
- query-window outputs remain CP-row sharded through the atom transformer;
- the confidence frame boundary performs one explicit linear-size coordinate
  replication, while the quadratic pair state remains distributed.

Inputs are padded to a complete query-window partition and halo width, then all
public coordinates, confidence outputs, and captured representations are
cropped back to the biological prefix.

These `shard_map` bodies are also why `attention_backend=tokamax` is refused
under a mesh rather than resolved. A Pallas kernel declares its outputs with no
`manual_axis_type`, which a checked `shard_map` requires of everything produced
inside it, so the fused attention raises from the partitioner once the atom
windows are active -- and they are, whenever `cp_atom_windows` is on and the
shapes line up. Unlike `glu_backend`, `triangle_backend` and
`diffusion_attention_backend`, this knob ships `xla`, so there is no released
default to protect and nothing to resolve: an omitted request is already the
partitionable spelling, and any other value was named. The refusal lands at
plan time on all three surfaces -- the adapter's option dict, `api.predict`,
and the model entry -- and it has to be the base knob that says so, because
`trunk_atom_attention_backend` and `diffusion_attention_backend` canonicalize
an explicit value equal to the global one to "unset" before their own mesh
check, which is what used to carry a globally named fused kernel past both.

## Protenix atom-window path

Protenix distributes the same four operations over CP rows, through the shared
primitives in `models/_cp_atom.py` plus Protenix-specific adapters in
`models/protenix/models/diffusion/_cp.py`:

- the atom-pair cache `p_lm` and the atom single cache `c_l` are built already
  split, from window- and atom-sharded reference features, rather than built
  whole and sliced;
- the projected token-pair tensor the atom windows read stays split on both
  pair axes; its `[n_windows, n_queries, n_keys, c]` slice is assembled by
  rotating pair-row tiles and reducing over pair columns, so no device ever
  holds the whole projection. This is the operation that previously forced a
  full-pair gather per device;
- the atom-pair conditioning and both atom transformer stacks run inside
  `shard_map` bodies with a fixed-width half-window halo. The trunk builder is
  installed task-locally rather than threaded through four signatures --
  `models/protenix/models/primitives/atom_windows_cp.py` -- so the serial
  attention site keeps exactly one code path;
- the atom->token mean is a `psum_scatter` over CP rows in Protenix' own
  arithmetic (`sum / max(count, 1)` in the activation dtype), and the
  token->atom gather rotates the linear token shards;
- the token transformer's queries follow the scatter onto CP rows, and its
  per-block pair bias is a projection of the pair-sharded conditioning cache,
  so under the square grid the bias and the token logits are split on both
  pair axes. The global token attention still collects K/V, which is linear in
  the token count.

Two shapes have to divide the mesh, and there is no fallback that hides it: the
atom axis must be a multiple of `n_queries * cp_rows` and the token axis must
divide the rows (and the columns under `2d`). A request that cannot be split
resolves to the replicated path **with a warning naming the multiple to pad
to** -- a silent fallback would leave a run that reads as distributed and costs
what the replicated one costs. Automatic padding (`--padding` with
`cp_devices > 1`) derives targets the mesh divides, so it supplies the
alignment by itself; pin `PaddingConfig(atoms=..., tokens=...)` when a run
needs one exact shape, and pin it aligned, because a pin is taken as written.

The sampler loop and its RNG tape are unchanged: the noise is drawn and carried
replicated exactly as before, the denoiser reshards the `[samples, atoms, 3]`
coordinates it is handed, and the coordinate update is replicated again on the
way out. Nothing about the diffusion tape moved.

Inside a sharded body `xla_jit` runs as `xla`: the window plan holds that
body's tracers and cannot cross an inner `jit`, and there is no eager dispatch
overhead left inside one traced program for the wrapper to remove. That is the
arm the released CP path takes, because the adapter resolves an omitted
`diffusion_attention_backend` to `xla_jit` when `cp_devices > 1`.

`cp_atom_windows` is a compile option on Protenix, defaulting to on. It is
ignored without a mesh, and the internal keyword it threads through
`atom_attention_encoder`, `atom_attention_decoder` and
`diffusion_module_f_forward` defaults to *off* -- because the request has to be
resolved against the shapes before it is honoured, and only a model entry point
knows them. A direct caller of those functions therefore gets the replicated
program rather than a `require_atom_windows` failure.

## OpenDDE atom-window path

OpenDDE's diffusion module calls the same Protenix denoiser, so it distributes
the same operations through the same adapters. Its option surface carries
`cp_atom_windows` too -- `--cp-atom-windows true|false` on
`foldjax-opendde-predict`, `--option cp_atom_windows=false` through the unified
CLI -- defaulting to on. The spelling differs from Protenix' `--no-…` pair
because this parser expresses every boolean as a value (`--use-template`,
`--use-rna-msa`), and its adapter's flag loop renders `--flag <value>`.

One thing is genuinely different, and it is the thing to get right: **the token
axis that has to divide the mesh is the structural one, not the residue one.**
OpenDDE diffuses over its expanded structural tokens, so
`diffusion_module_forward` receives `n_token=n_structural_token` and the
alignment requirement is `n_structural_token % cp_rows == 0` (and `% cp_cols`
under `2d`). Automatic padding derives the structural target as twice the token
bucket, so an aligned residue count does not imply an aligned structural one;
automatic padding therefore rounds the structural target up to the rows too,
and the misalignment warning names
`PaddingConfig(atoms=..., structural_tokens=...)` / `--pad-structural-tokens`
for the pinned case.
The atom requirement is the shared one: a multiple of `n_queries * cp_rows`,
i.e. 32 times the row count with the released windows.

The second difference is an operand Protenix never supplies: OpenDDE always
passes `extra_attn_bias`, a replicated `[N_structural, N_structural]` role-pair
bias, into the token attention whose queries the distributed atom graph now
delivers CP-row sharded. The compiled SPMD module still contains no full-width
token pair bias and no full-width token attention logits, so the replicated
bias is sliced rather than forcing the logits back together
(`tests/models/opendde/scripts/atom_cp_parity.py`).

`diffusion_dtype=bf16` remains refused under a mesh, unchanged by this: that
guard is about the denoising network's precision, not its placement.

## OpenFold3 atom-window path

OpenFold3's diffusion atom graph is split over CP rows too, and the mechanism
differs from Boltz-2's and Protenix' in one place that decides everything else.

**OpenFold3's key windows are not a fixed offset from their query block.**
`models/openfold3/models/atom_blocks.py:block_indices` *shifts* a window rather
than clipping it: a block that would start before atom 0 slides right, and a
block that would run past the last real atom slides left to end there. So a
query block lying entirely inside the atom padding reads the last `n_key`
*real* atoms, however far away they are — measured on 96 atoms with 60 real and
a 4/8 window, blocks 15..23 all read atoms 52..59, three whole blocks away. The
shift is derived from `jnp.sum(atom_mask)`, so it is a traced value: there is no
static halo width that covers it. The key side is therefore an **index-driven
ring gather** over rotating atom shards
(`models/_cp_atom.py:gather_atom_windows_local`), which puts no bound at all on
where a key may live. The query side is a pure local reshape.

The second difference is structural: the whole atom stage runs inside *one*
`shard_map` body per encoder/decoder rather than one per arithmetic stage.
OpenFold3 reaches its blocking through four shared primitives, and each of them
dispatches on an `AtomBlockPlan` installed task-locally by the enclosing
sharded body (`models/openfold3/models/atom_cp.py`) — the same `ContextVar`
device Protenix uses for its window trunk builder, and for the same reason: the
plan holds that body's tracers, so it cannot be threaded through signatures the
input embedder, the trunk and the frame builder also call. With no plan
installed all four keep their historical single path.

Per stage (`file:line` at the dispatch):

- key blocks and the block pair mask — `atom_blocks.py:122`
  (`single_rep_to_blocks`), five call sites between the reference-feature
  embedder, the atom-pair conditioning and both cross-attention stacks;
- the atom-pair conditioning cache is born window-sharded and never assembled:
  every `[N_blocks, N_query, N_key, c]` tensor in the encoder is a CP-row slice;
- the token-pair gather — `atom_blocks.py:176` (`pair_rep_to_blocks`) — rotates
  pair-row tiles and reduces over pair columns, so the projected
  `[N_token, N_token, c_atom_pair]` tensor is never held whole. This is the
  operation that previously forced a full-pair gather per device;
- the token→atom broadcast — `atomize.py:48` — rotates the linear token shards.
  `token_mask` and `num_atoms_per_token` enter the body *replicated*, so the
  prefix-validity threshold `positions < sum(counts)` is the same whole
  reduction the serial path performs rather than a per-shard count;
- the atom→token mean — `atomize.py:147` — keeps OpenFold3's deliberate
  one-hot contraction (a GPU scatter-add's summation order is not reproducible
  and the rollout amplifies it) and reduce-scatters `totals` and `counts` over
  CP rows. The overflow bin masked atoms are routed to is dropped *before* the
  `psum_scatter`, because `n_token + 1` does not divide the rows;
- the token transformer's queries follow the scatter onto CP rows
  (`denoiser.py:140`, `:156`) and its per-block pair bias is a projection of the
  pair-sharded conditioning (`denoiser.py:116`), so under the square grid the
  bias and the token logits are split on both pair axes.

Two shapes have to divide the mesh, and there is no fallback that hides it: the
atom axis must be a multiple of `n_query * cp_rows` and the token axis must
divide the rows (and the columns under `2d`). Unlike the halo paths there is no
requirement on `n_key`: a CP row owning a single query block is legal. A request
that cannot be split resolves to the replicated path **with a warning naming the
multiples to pad to**; automatic padding supplies the alignment, and a pin is
taken as written. At the released `n_query=32` on four devices in the 1-D
layout that is `atoms % 128 == 0` and `tokens % 4 == 0`; on a 2x2 grid --
which is what four devices build here unless `cp_layout=1d` asks otherwise --
`atoms % 64 == 0` and `tokens % 2 == 0`.

The sampler loop and its RNG tape are unchanged: the coordinate state stays
replicated (`[samples, atoms, 3]` is linear in the atom count and three channels
wide), the encoder's `shard_map` reshards the coordinates it is handed, and the
decoder's update is replicated again on the way out.

`cp_atom_windows` is a compile option on OpenFold3, defaulting to on, resolved
once in `inference.py:_predict_from_trunk` before the rollout body is defined so
the misalignment warning is emitted once and outside `lax.scan`. The internal
keyword it threads through `denoise`, `atom_attention_encoder` and
`atom_attention_decoder` defaults to *off*, which is what keeps the input
embedder's atom encoder — and every direct caller — on the program it had. Both
atom transformer stacks already default to `scan_blocks=True`, so the ring
gathers' `ppermute` runs inside a `lax.scan` body inside the `shard_map`.

## Trunk-only representation capture

`stop_after="trunk"` is a distinct compiled graph. It returns before the
sampler and confidence heads and therefore does not read coordinate or pLDDT
fields that do not exist in that graph. Requested `single`, `single_inputs`, and
`pair` representations are cropped to the unpadded token count before being
persisted.

## Numerical contracts

- pair and atom padding are masked and sliced away;
- reductions that combine communication tiles accumulate in fp32;
- atom-to-token BF16 reductions accumulate in fp32 and cast only the final mean
  back to BF16;
- ring attention uses one global maximum before exponentiation;
- compensated summation limits tile-order drift without changing the
  gather-free communication schedule;
- an all-masked ring tile contributes zero mass rather than evaluating
  `exp(-inf - -inf)`;
- pair-biased attention uses an exact `-inf` key mask, so a globally all-masked
  query returns a finite zero output rather than a uniform distribution;
- a mean whose numerator and denominator are reduced across shards is
  normalised after the reduction, never averaged from locally normalised
  means, and its output bias is added once;
- token padding introduced to divide a mesh is masked one penalty level below
  a masked real position, so an all-masked query row stays a distribution over
  the real positions;
- 2x2 and 3x3 meshes are tested because modulo two cannot distinguish opposite
  ring directions;
- serial, 1D, and 2D programs are traced through fresh closures to prevent a
  cached serial executable from producing a vacuous parity pass.

## CPU proof gates

The final proof suite runs the following model and runtime tests on forced CPU
meshes:

```bash
uv run pytest -q \
  tests/models/test_cp_runtime.py \
  tests/models/test_cp_ring_attention.py \
  tests/models/test_cp_masked_ring.py \
  tests/models/boltz2/test_context_parallel.py \
  tests/models/boltz2/test_msa_grid_bridges.py \
  tests/models/boltz2/test_atom_context_parallel.py \
  tests/models/boltz2/test_atom_cp_numerics.py \
  tests/models/boltz2/test_atom_cp_model_integration.py \
  tests/models/boltz2/test_atom_cp_padding.py \
  tests/models/boltz2/test_api.py \
  tests/models/protenix/test_context_parallel.py \
  tests/models/protenix/test_atom_context_parallel.py \
  tests/models/opendde/test_context_parallel.py \
  tests/models/opendde/test_atom_context_parallel.py \
  tests/models/openfold3/test_context_parallel.py \
  tests/models/openfold3/test_atom_context_parallel.py \
  tests/models/esmfold2/test_context_parallel.py \
  tests/models/esmfold2/test_cp_layout_option.py
```

On August 20, 2026, the cross-model branch gate passed 66 tests. The subsequent
atom-numerics hardening gate passed 20 focused tests, including forced 4-device
and 9-device Pairformer parity, BF16 reduce-scatter parity, and an all-masked
pair-bias attention case. Ruff formatting/linting, Python compilation, and
`git diff --check` also passed. No numerical tolerance was loosened.

These gates prove numerical parity, padding behavior, model wiring, entry
placement, and HLO collective structure on multi-device CPU meshes. The
model-level atom gate composes conditioning, the atom encoder, token
transformer, and atom decoder under both layouts rather than testing only the
routing primitives. These gates do not claim CUDA/NCCL throughput or
multi-node reliability.

## Deployment validation: what the 2-D default rests on, and what is still open

The list below is what a deployment topology has to be measured for before
`2d` is selected automatically on it, or before that configuration is called
production-ready:

1. serial versus 1D versus 2D output parity with identical parameters and
   random tapes;
2. per-device peak memory and warm latency, reported separately for pair and
   atom stages;
3. compiled communication volume and reshard boundaries;
4. repeated-run finiteness and determinism, especially through OpenDDE
   diffusion;
5. 2, 4, and 8 GPUs, plus multi-node runs when those are intended.

Items 1 and 2 are answered for OpenDDE and Boltz-2 on one topology -- the
four-card 96 GiB node, a 2x2 mesh, one 2,096-token target -- and that is the
whole basis of their `auto = 2d`, quoted above. It is a memory-ceiling result:
the grid runs a job the other layouts do not, and it is slower. OpenFold3's
`auto = 2d` rests on **item 2 only**, on the same node and one 6,568-token
target: the grid's per-device peak is recorded and the run completes, while
the arms item 1 would compare it against -- serial, and 1-D on four cards --
both run out of memory there, so there is no parity comparison to make at that
size and no wall-time comparison either. ESMFold2's `auto = 2d` rests on
**item 2 only** as well, on the same node and one 3,012-token target, and for
a different reason: the grid's per-device peak and wall time are recorded
(23,430 MiB, 772 s per pass) and both passes complete, while serial runs out
of 96 GiB and the 1-D arm on four cards was still in its first pass after one
to two hours, so neither arm produced coordinates to compare against or a
latency to compare with. Everything else on the list is open, for every port:
8 GPUs, multi-node, other targets and token counts, and the run-to-run
determinism reservation recorded in `PROJECT.md`. Protenix has the measurement
and it went the other way. No default moves without its own numbers.

For Boltz-2, Protenix, OpenDDE and OpenFold3, the pair trunk scales over both
two-dimensional mesh axes, while atom windows scale over CP rows and are
replicated over CP columns. The *Atom-window CP* column of the table above is
the authority on which models distribute their atom graph and which still hold
it whole on every device.

Protenix' atom-window path has CPU parity, HLO-structure and serial-invariance
gates (`tests/models/protenix/test_atom_context_parallel.py`, 18 tests on 1-,
4- and 9-device CPU meshes, in both layouts, on both the `xla` and the
`xla_jit` denoiser attention the adapter resolves an omitted backend to under
a mesh, and -- because `cp_shards > 1` requires `graph_jit`, which sets
`use_diffusion_scan=True` -- with the block stack inside `lax.scan` inside the
`shard_map` body, with and without a token query chunk) and no GPU measurement
yet. The per-device claim it makes is
structural -- the compiled SPMD module contains no full-width atom activation,
atom-pair window cache, or projected token-pair tensor -- not a measured
peak.

OpenDDE's has the same shape of evidence and the same limit
(`tests/models/opendde/test_atom_context_parallel.py`, 14 tests: 1-D x4 and 2x2
CPU meshes, both denoiser attention arms, the scanned block stack and scanned
step loop `graph_jit` actually compiles, a token query chunk, the serial
lowering pinned byte-identical to `main` in the same interpreter with zero
collectives and zero sharding ops, the structural-axis misalignment warning,
and the compile-namespace spelling). Also structural, also no GPU measurement.
The target it exists for -- a job that does not fit one card -- is exactly the
one no CPU mesh can measure, so the memory claim stays a claim about what the
compiled module does not contain.

OpenFold3's atom-window path has the same three kinds of gate
(`tests/models/openfold3/test_atom_context_parallel.py`, 11 tests): the serial
denoiser lowering is pinned byte-identical to `git archive main` and carries no
collective or sharding annotation; a live four-device mesh with
`cp_atom_windows` off lowers to that same program, so the option rather than
the mesh is what distributes the graph; and one denoiser step, the encoder, the
decoder and a three-step sampler agree with serial on 1-D x4, 2x2 and 3x3
meshes to FP32 reduction-order tolerance. Two of those arms pad the atom axis
so that query blocks fall off the real atoms (90 and 60 real of 96), which is
the property an index-driven gather satisfies and a halo of any static width
does not -- an all-real fixture would pass against either. The per-device claim
is structural in the same sense as Protenix': the compiled SPMD module contains
no `[S, N_atom, c_atom]` activation, no `[S, N_blocks, N_query, N_key, c_pair]`
cache, no projected `[S, N_token, N_token, c_pair]` tensor, no
`[S, H, N_token, N_token]` token bias and no `[S, N_atom, N_token + 1]`
atom-by-token one-hot. That gate lowers with the features as *traced arguments*
rather than closed-over literals: with them as constants XLA folds
`atom_to_token_index` through the aggregate's one-hot and the block tables, and
a full-width tensor then appears as a `constant` — or vanishes — for reasons the
real program does not share. Measured: the literal-batch module carries a
folded `f32[S, N_atom, N_token]` constant the traced one does not.

What is still full-width per device, deliberately: the token attention's
gathered K/V (`[S, H, N_token, d]`, linear in the token count, the same
remainder Protenix has), the replicated `[S, N_atom, 3]` coordinate update the
sampler carries, and the token-shaped `token_mask` / `num_atoms_per_token` the
sharded bodies need whole. No GPU measurement yet.
