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
| Boltz-2 | yes | yes | yes | Cannon/ring pair core, CP-row atom windows, halo exchange, sparse token/pair routing |
| Protenix | yes | yes | yes | Pair trunk and confidence pair path use the common pair core; the diffusion atom graph is distributed over CP rows (`cp_atom_windows`, default on) |
| OpenDDE | yes | yes | yes | Structural-token refinement uses the Protenix pair primitives; its diffusion module calls the same Protenix denoiser with the atom graph distributed over CP rows (`cp_atom_windows`, default on), aligned on the *structural* token axis |
| OpenFold3 | yes | yes | no | Pair stack, template stack, and confidence pair re-embedding |
| ESMFold2 | yes | no | no | Pair-row constraint path; no two-dimensional triangle-attention ring |
| AlphaFold3 | no | no | no | The vendored publisher runtime is not rewritten for FoldJAX CP |

`auto` currently resolves to `1d`. Use `2d` explicitly only with a
perfect-square device count. This preserves the published one-dimensional
configuration until the square-grid path has been measured on the deployment
GPUs.

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
what the replicated one costs. Pin `PaddingConfig(atoms=..., tokens=...)` to
supply the alignment.

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
the misalignment warning accordingly names
`PaddingConfig(atoms=..., structural_tokens=...)` / `--pad-structural-tokens`.
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
  tests/models/esmfold2/test_context_parallel.py
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

## Deployment validation still required

Before selecting `2d` automatically or describing a particular cluster
configuration as production-ready, measure on that deployment topology:

1. serial versus 1D versus 2D output parity with identical parameters and
   random tapes;
2. per-device peak memory and warm latency, reported separately for pair and
   atom stages;
3. compiled communication volume and reshard boundaries;
4. repeated-run finiteness and determinism, especially through OpenDDE
   diffusion;
5. 2, 4, and 8 GPUs, plus multi-node runs when those are intended.

For Boltz-2, Protenix and OpenDDE, the pair trunk scales over both
two-dimensional mesh axes, while atom windows scale over CP rows and are
replicated over CP columns. OpenFold3 deliberately retains pair-only CP until
its atom graph receives a model-specific distributed contract and
checkpoint-level validation.

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
