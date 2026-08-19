# Context parallelism

FoldJAX exposes context parallelism for the quadratic pair representation used
by AF3-family folding models. The implementation follows Fold-CP's two central
pair algorithms in JAX:

- pair rows may be split over one mesh axis (`1d`);
- a square mesh may tile both pair axes (`2d`);
- triangle multiplication uses Cannon-style tile rotation;
- triangle attention keeps queries resident and rotates key, value, mask, and
  pair-bias tiles while accumulating an fp32 online softmax.

The two-dimensional attention hot path does not reconstruct a full token axis.
Its regression tests inspect HLO and reject an `all-gather`.

## Supported model paths

| Model | 1D pair CP | 2D pair CP | Notes |
|---|---:|---:|---|
| Boltz-2 | yes | yes | Cannon multiplication and ring triangle attention |
| Protenix | yes | yes | The same pair core is used by its trunk and confidence path |
| OpenDDE | yes | yes | Uses the Protenix pair primitives for structural-token refinement |
| OpenFold3 | yes | yes | Pair stack, template stack, and confidence re-embedding |
| ESMFold2 | yes | no | Pair-row constraint path; no triangle-attention ring |
| AlphaFold3 | no | no | The vendored publisher runtime is not rewritten for FoldJAX CP |

`auto` currently resolves to `1d`. Use `2d` explicitly only with a
perfect-square device count. This keeps published one-dimensional runs
reproducible until the square-grid path has been measured on the target GPUs.

## Input placement

The active CP runtime is task-local, so concurrent requests cannot overwrite
one another's mesh. Model parameters and unrecognised inputs remain
replicated. A small, explicit registry places known pair features directly on
the pair mesh when both axes divide evenly. Uneven pair features and every
atom/single-stream feature stay replicated at entry.

That last point is deliberate: FoldJAX currently distributes the quadratic
pair trunk, not the complete atom-window diffusion graph. No capability flag
claims otherwise. Atom-window tensors remain a linear-memory replicated stage
until a model-specific halo-exchange implementation is validated.

## Numerical contracts

- pair padding is masked and sliced away;
- online softmax accumulates in fp32;
- an all-masked ring tile contributes zero mass rather than evaluating
  `exp(-inf - -inf)`;
- a globally all-masked query returns a finite zero output;
- 2x2 and 3x3 meshes are tested because modulo two cannot distinguish opposite
  ring directions;
- serial, 1D, and 2D programs are traced through fresh closures to prevent a
  cached serial JAX program from producing a vacuous parity pass.

## CPU proof gates

```bash
uv run pytest -q \
  tests/models/test_cp_runtime.py \
  tests/models/test_cp_ring_attention.py \
  tests/models/test_cp_masked_ring.py \
  tests/models/boltz2/test_context_parallel.py \
  tests/models/protenix/test_context_parallel.py \
  tests/models/opendde/test_context_parallel.py \
  tests/models/openfold3/test_context_parallel.py \
  tests/models/esmfold2/test_context_parallel.py
```

These tests force multiple CPU devices in subprocesses. They prove numerical
and HLO structure, not CUDA/NCCL performance.

## GPU validation still required

Before selecting `2d` automatically or describing it as production-ready,
measure on the deployment topology:

1. serial versus 1D versus 2D output parity with identical parameters and
   random tapes;
2. per-device peak memory and warm latency;
3. HLO/trace communication volume;
4. repeated-run finiteness and determinism, especially through OpenDDE
   diffusion;
5. 2, 4, and 8 GPUs, plus multi-node runs when those are intended.

The absence of atom-window CP should be included when interpreting end-to-end
memory scaling: the quadratic trunk scales with CP, while the later linear
atom stage does not yet scale across devices.
