# OpenFold3 host-streamed MSA recycling

The padded managed path previously padded the MSA union to the per-cycle limit,
then gathered a variable-width cycle index inside the trunk. Shallow MSAs
therefore still changed the compiled MSA width, and a native multi-cycle union
larger than the per-cycle limit failed padding.

The implementation now retains native row selections and order on the host,
pads each gathered selection with zero-mask rows, and transfers only the current
and next selection. Compact categorical padding uses category 32, the existing
zero-vector sentinel. No union-sized array or cycle index reaches a stage JIT.
There is no additional truncation of the candidate pool in this scheduler.

Three stage functions share a bounded process cache: initialization, one trunk
cycle, and diffusion/confidence. Parameters remain dynamic. Common features and
parameters are explicitly device-placed once per invocation; the initial single
and pair embeddings survive recycling. The previous single/pair carry is donated
to the next cycle, and the initial single/pair carries are explicitly sharded under CP.
The scheduler fences each cycle after enqueuing one lookahead so asynchronous
calls cannot retain the entire MSA schedule. Transfer/compute overlap is possible,
not a measured guarantee.

The backend selects this path with `--padding`, including its existing eager
diagnostic option. Padding OFF retains the fused path. Native row selection,
recycle count, template arithmetic, diffusion RNG routes, precision scope,
confidence heads, and writer mapping remain shared. Padding metadata separates
`host_msa_union_rows` from the compiled cycle width.

## Verification

- CPU feature tests cover source depths 1, 17, 1024, 1025, and 4097 with four
  native selections and capacity 1024. Every selected row is preserved in order;
  shallow suffix rows have zero mask. Deep unions no longer fail the cycle cap.
- Dense/compact MSA embedding equivalence, real JAX stage trace counts, lookahead
  admission order, invalid-index rejection, eager execution, and chain/cache-scope
  partitioning are covered by `tests/models/openfold3/test_streamed_msa.py`.
- OpenFold3 plus common backend/padding/cache/resume tests: **1355 passed,
  356 skipped, 25 warnings**. Skipped optional publisher/GPU tests are not evidence
  of correctness. The final scheduler follow-up has **12 passed**.
- A separate CPU full-model probe reused the existing reduced end-to-end fixture:
  synthetic randomized parameters (scale 0.1), 3 tokens, 12 atoms, 4 cycles,
  2 samples, 3 diffusion steps, MSA capacity 8, source depths 1/3/11, selection
  seed 2 and JAX key 17. The Torch fixture was exported as host arrays from the
  already-installed external publisher environment, then mapped and executed
  in the JAX-only project environment; no dependencies were installed.
- The fused baseline also used the unmodified trunk from
  `c44a116f97214fbc63c1de4b6253d1b55b10e819`, avoiding a comparison solely between
  two callers of the newly extracted cycle. Coordinates, confidence outputs,
  logits, and returned single/pair representations had maximum absolute
  difference **0** for each depth. The streamed single-device JIT pool retained
  **3 executables** across all three depths.
- Final focused CPU checks: **127 passed, 10 skipped, 150 deselected**.
- A forced two-device CPU probe initially produced a second cycle executable:
  the zero single carry was replicated but subsequent carries were row-sharded.
  Explicit initial single/pair sharding removed that split, leaving **3 stage
  executables** across source depths 1/3/11. Finite output elements matched the
  legacy fused baseline exactly. Both paths produced matching nonfinite atom-head
  outputs with this padded synthetic fixture, so this is **not** a full CP
  confidence-parity acceptance. The persistent CP regression compares finite
  trunk representations; full confidence on real inputs remains a separate gate.
- Changed-file Ruff, `uv lock --check`, Python syntax and diff checks passed.
  Repository-wide Ruff initially passed; the final repeat found two unrelated
  errors (I001/E501) in concurrently added `bench/esmfold2_pair_init_probe.py`.
  That file was left untouched.

The reduced synthetic probe is an execution-equivalence check, not a published
checkpoint scientific-parity gate. Real-weight GPU outputs, warm latency,
peak allocated memory, transfer overlap, and cross-process persistent-cache hits
have not been measured for this change. Persistent JAX caching uses the existing
namespace, invalidation and access-tracking infrastructure.

## Remaining compile-profile work

1. Static chain counts in Protenix, OpenDDE, OpenFold3 and ESMFold2.
2. Boltz2 template count/presence, which the common padding policy does not fix.
3. Compact/dense feature and optional RNG-route variants, including ESMFold2
   atom-group and token-bond representations.
4. Select explicit warmup profiles for samples/steps, outputs, dtype, kernels,
   model/LM variants and device topology; these remain configuration axes.
