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
| Protenix | yes | yes | no | Pair trunk and confidence pair path use the common pair core; atom streams remain replicated |
| OpenDDE | yes | yes | no | Structural-token refinement uses the Protenix pair primitives; diffusion atom streams remain replicated |
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
- atom-to-token means use reduce-scatter semantics;
- token-pair values are looked up from rotating two-dimensional pair tiles;
- query-window outputs remain CP-row sharded through the atom transformer;
- the confidence frame boundary performs one explicit linear-size coordinate
  replication, while the quadratic pair state remains distributed.

Inputs are padded to a complete query-window partition and halo width, then all
public coordinates, confidence outputs, and captured representations are
cropped back to the biological prefix.

## Trunk-only representation capture

`stop_after="trunk"` is a distinct compiled graph. It returns before the
sampler and confidence heads and therefore does not read coordinate or pLDDT
fields that do not exist in that graph. Requested `single`, `single_inputs`, and
`pair` representations are cropped to the unpadded token count before being
persisted.

## Numerical contracts

- pair and atom padding are masked and sliced away;
- reductions that combine communication tiles accumulate in fp32;
- ring attention uses one global maximum before exponentiation;
- compensated summation limits tile-order drift without changing the
  gather-free communication schedule;
- an all-masked ring tile contributes zero mass rather than evaluating
  `exp(-inf - -inf)`;
- a globally all-masked query returns a finite zero output;
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
  tests/models/boltz2/test_atom_cp_padding.py \
  tests/models/boltz2/test_api.py \
  tests/models/protenix/test_context_parallel.py \
  tests/models/opendde/test_context_parallel.py \
  tests/models/openfold3/test_context_parallel.py \
  tests/models/esmfold2/test_context_parallel.py
```

On August 20, 2026, the final branch gate passed all 66 tests. Separate staged
probes compared every residual sub-step of a two-layer Boltz-2 Pairformer on
4-device and 9-device CPU meshes; all reported zero tolerance violations. The
gate also passed Ruff formatting/linting, Python compilation, and
`git diff --check`.

These gates prove numerical parity, padding behavior, model wiring, entry
placement, and HLO collective structure on multi-device CPU meshes. They do not
claim CUDA/NCCL throughput or multi-node reliability.

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

For Boltz-2, the pair trunk scales over both two-dimensional mesh axes, while
atom windows scale over CP rows and are replicated over CP columns. Protenix,
OpenDDE, and OpenFold3 deliberately retain pair-only CP until their atom graphs
receive model-specific distributed contracts and checkpoint-level validation.
