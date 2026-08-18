
## Protenix 3012-token memory gap (84,106 vs upstream 57,310 MiB) -- ruled out, open

Record: `foldjax-bench/results-lengths/protenix-foldjax-L3000_6ztx.json`, `options: {}`
(so every default is the shipped one).

Ruled out by reading the defaults both sides actually run:

- **dtype**: not it. `cli/predict.py:126` defaults `--trunk-dtype bf16`; upstream
  `configs/configs_base.py:135` is `"dtype": "bf16"`. Both trunks are bfloat16.
- **triangle kernel**: not it. `models/triangle/triangle.py:50` already resolves to
  `cueq_jit` with no env var, matching upstream's `triangle_attention:
  "cuequivariance"`. The stale comment at `cli/predict.py:167-171` still says the
  unset default "picks the blocked XLA path" -- it does not, and that comment
  should go.
- **diffusion sample batching**: not it. Both default `diffusion_chunk_size=None`
  and denoise all 5 samples in one call (`models/model.py:202` /
  upstream `protenix/model/generator.py:268`).

So the 1.47x is structural, not a knob left unset. Next, in order:

1. Whether upstream's *inference runner* passes `diffusion_chunk_size` /
   `attn_chunk_size` from its infer config even though `generator.py`'s own
   defaults are None. That is the only remaining place a knob asymmetry could hide.
2. Otherwise: XLA sizes one temp arena for the whole program and holds it, while
   torch allocates and frees per module. Attribute it the way the boltz2 OPM was
   attributed -- compile the protenix program alone at n_token=3012, compile-only,
   and read `memory_analysis()` (see `compile-one-module-to-attribute-an-arena`).
   Do not read a whole-run peak for this; it cannot attribute a buffer.

Also landed while checking: **opendde foldjax L3000 OOM'd** after 3597 s (16.11 GiB
allocation refused), so the 3012 opendde row is now both-implementations-fail, not
"foldjax still running".

## Protenix 3012 gap, continued: the runner config is symmetric too

Checked the one remaining knob asymmetry (upstream's *runner* filling chunk
sizes its generator defaults leave None). It does fill them -- and the port
already mirrors it:

- Upstream `infer_setting`: `chunk_size` 256 with `dynamic_chunk_size: True`,
  thresholds {1024:-1, 1536:512, 2048:256, 2560:128}, beyond -> 32
  (`protenix.py:443-468`), and `sample_diffusion_chunk_size: 5`.
- Port: `chunking.py:20-27` transcribes the same table (`PROTENIX_EXTREME_CHUNK
  _SIZE = 32`, diffusion chunk 5), `chunk_policy=auto` is the CLI default, and
  the bench row ran `options: {}`.

At 3012 tokens both sides therefore run the whole trunk chunked at 32 and one
5-sample diffusion batch. Nothing configurable is left; the 84.1 vs 57.3 GiB
is the program itself. `foldjax-bench/protenix_arena_probe.py` now intercepts
`_compiled_protenix_infer` at the CLI's own call (same features, same bf16
cast, same resolved chunks), `lower().compile()`s it and writes
`memory_analysis()` to `protenix3012-arena.json`, with an XLA dump for
`arena_attr.py`. Queued behind the opendde L3000 retry currently holding the
card.

Stale comment found on the way: `cli/predict.py:167-171` still claims the
unset triangle default "picks the blocked XLA path"; `triangle.py:50` has
defaulted to cueq_jit since the kernel-default commit. Fix when touching that
file next.

## Protenix 3012 gap, attributed (compile-only probe + buffer assignment)

`memory_analysis()` of the exact bench program: arguments 12.40 + outputs
26.36 + temp 43.28 = 82.04 GiB, against the measured 84,106 MiB peak. XLA's
own rematerializer agrees it cannot get below 83.0 GiB (hlo_rematerialization
warning in the probe log).

**Outputs, 26.36 GiB.** Allocations 0 and 1 are 11,612,344,320 B each =
5 x 3012^2 x 64 x 4 B: the full-bin **PAE and PDE logits, f32[5,N,N,64]**.
`model.py:259` does `output.update(confidence_logits)` -- and then *also*
computes the summaries in-graph at `model.py:262` (`confidence_scores_from_
logits`). The raw logits ride along as program outputs the CLI's cif/JSON path
never needs, and XLA entry outputs are live for the whole execution, on top of
the temp arena. pae+pde alone are 21.6 GiB; z_trunk/s_trunk/distogram make up
most of the rest.

**Temp arena, 43.28 GiB.** Top buffers are five distinct **f32 pair-sized
[N^2,128] buffers (4.43 GiB each, ~22 GiB)** around `triton_kernel_call` /
`jit(triangle_attention)` inside the recycle/pairformer scan, plus f32
triangle-attention operands (~13 GiB): the cueq kernels take float32, so the
bf16 trunk pays f32 up-casts of the pair tensor several times over per block,
and buffer assignment keeps several alive at once. The bf16[16384,3012,64] MSA
stack (6.3 GiB x2) shares the arena.

So the 84.1-vs-57.3 split vs upstream torch: ~22 GiB of returned logits torch
reduces per-sample and frees, plus f32 staging around the fused kernels that
torch's bf16-native cueq never allocates. Cheapest fix, in order: stop
returning pae/pde (and distogram/z/s unless asked) once scores are computed
in-graph -- that alone is ~21.6 GiB and closes most of the gap.

## Output-gating fix, verified compile-only (protenix, 3012)

Same probe, program after `return_confidence_logits`/`return_trunk` gating:

    before  argument 12.404  output 26.357  temp 43.282  total 82.04 GiB
    after   argument 12.404  output  0.374  temp 54.109  total 66.89 GiB

Outputs collapsed 26.36 -> 0.37 GiB. Temp grew 43.28 -> 54.11: the pae/pde
logits are still computed for the in-graph summaries, and one logits-sized
buffer (10.8 GiB) that used to be written straight into an entry output now
needs temp arena space instead. Net -15.2 GiB. The remaining gap to upstream
(66.9 vs 57.3) is the f32 staging around the float32-only cueq kernels plus
that one in-flight logits buffer; halving the latter means reducing the
confidence head sample-sequentially with in-scan summarization (what boltz2's
`confidence_sequentially` and AF3's `sharded_map(shard_size=1)` do).

Boltz2 got the same gating (`0484e4f`): backend reads six scalars + plddt +
coords; pae/pde/plddt/resolved logits and pdistogram are now opt-in
(`return_confidence_logits=True`).

## Same output-logits audit: openfold3 and opendde

**openfold3 -- same disease, no consumer at all above 4 GiB.** The jitted
`Prediction` (inference.py:576-586) always carries `pae_logits`/`pde_logits`
f32[n_sample,N,N,64] plus distogram/resolved logits. The only reader is
`output.write_arrays`, which has a 4 GiB budget (output.py:279-328) and drops
exactly these arrays largest-first when they exceed it -- at 3,012 tokens the
program hauls ~22 GiB out as entry outputs so the writer can throw them away.
ptm/iptm/plddt are already computed in-graph. Fix: decide the writer's keep-set
*before* the program runs (same budget rule, helper next to write_arrays so
they cannot drift), request only the survivors from the jit; npz content is
then bit-identical at every size while the large-N peak drops.

**opendde -- same tensors, but a real host consumer.** `return_representations`
is already gated off (model.py:605), but `run_confidence` returns the raw
plddt/pae/pde logits, and `postprocess.py:88-124` computes the released
summaries on the *host* from them (`confidence_scores_from_logits` outside the
graph). So the logits cannot just be dropped: the fix is protenix's shape --
move `confidence_scores_from_logits` inside the graph behind a
`run_confidence_scores` static flag, teach postprocess to accept precomputed
summaries, then gate the logits. Note opendde is the port that OOMs at 3,012:
21.6 GiB of logits as entry outputs is plausibly part of why, so this fix is
also the next lever on that failed cell.

## Why protenix still holds 21.6 GiB more than boltz2 at 3012 (68.6 vs 46.9)

Attributed from the post-fix probe dump (arena 54.1 GiB): the top three
buffers are **f32[5,3012,3012,64] at three distinct offsets -- 10.8 GiB each,
32.2 GiB live at once**: the pae and pde logits stacked over all five samples
(two `stack` fusions) plus a third consumed by the in-graph scores. The
confidence stack processes the whole sample batch at once.

Boltz-2 runs the same-shaped heads one sample at a time
(`confidence_sequentially`, lax.map -- upstream production behaviour), so its
in-flight logits are [1,N,N,64] = 2.2 GiB; AF3 does the same with
`sharded_map(shard_size=1)`. So the gap is not model shape: it is protenix's
confidence sample batching, and the fix is the same shape as boltz2's --
per-sample head + per-sample `confidence_scores_from_logits` inside the loop,
never materializing the [5, N, N, 64] stacks. Expected landing: near boltz2's
46.9.

Also verified this round: opendde in-graph scores now trace and reproduce the
host path exactly (L500: ranking 0.1946111 vs 0.1946113, peak 13,216 ->
12,649 MiB), but L3000 still dies on the same single 16.11 GiB allocation --
the structural branch's ~2x-token pair buffer, which is model shape
(`foldjax-remaining-memory-bottleneck`), not the logits.

## Sequential confidence scoring, verified (protenix, 3012)

    outputs-gated          argument 12.404  output 0.374  temp 54.109  peak 68,558 MiB
    + per-sample scoring   argument 12.404  output 0.374  temp 46.483  peak 60,759 MiB

Wall 804 s (unchanged), ranking 0.955322 vs 0.955337, plddt 94.296 vs 94.285
-- inside the rerun floor. Cumulative from the start of this audit: 84,106 ->
60,759 MiB, -28%, at zero time or score cost (commit 60f1e08; equivalence test
pins sequential == batched per key, including sample-axis-free contact_probs).

The new arena's top is no longer confidence: four bf16[128, 4*N^2] operands
(8.86 GiB each, ~35 GiB) around the recycle loop's triangle-multiplication
dot_generals, plus f32[N,N,256] cueq staging. The remaining 13.9 GiB to
boltz2 (60.8 vs 46.9) is ~12.4 GiB of entry arguments (the 16,384-row MSA
feature stack rides in the args) plus that tri-mul working set -- model and
feature shape, no longer an unread output or a redundant stack. Next lever if
ever needed: bf16-native cueq bindings or narrower tri-mul staging; expect
diminishing returns.

## Small-cell re-measurement after the logits fixes (tsp 29-32)

    boltz2   499: 5,090 -> 4,781 MiB (1.77x)   1003: 10,027 -> 7,402 (2.19x)
    protenix 499: 5,706 -> 5,058 MiB (1.01x)   1003: 10,531 -> 8,711 (1.50x)

Protenix 499 flips from the table's only losing memory cell (0.90x) to parity;
boltz2 1003 more than doubles its lead. Scores unchanged. Every FoldJAX cell
in the table is now post-fix and >= 0.94x memory against its upstream.

## AF3 and OpenFold3 columns filled (tsp 33-38, 2026-08-12)

    alphafold3  499: 285s/3,036   1003: 368s/6,901    3012: 1,013s/42,277 MiB
    openfold3   499:  39s/10,442  1003: 126s/12,043   3012: 1,374s/66,264 MiB

Upstream columns stay empty by spec: AF3's would be the same code, OpenFold3's
venv is not provisioned. AF3 has the flattest memory curve in the table --
bf16 'all', flash attention, shard_size=1 confidence, summaries-only outputs,
i.e. every practice the ports adopted this session. OpenFold3 at 3,012 runs
66.3 GiB *with* the pair-logit gate; the ungated program would have carried
21.6 GiB more entry outputs, 87.9 GiB against an 85.5 GiB pool -- over it.

## Open: OF3-vs-protenix memory comparison + OpenFold3 upstream column

Goal (2026-08-12): (1) explain OF3's curve against protenix's -- 10.4 vs 5.1
GiB at 499, 66.3 vs 60.8 at 3012, despite protenix carrying a 16k-row MSA;
(2) fill the OpenFold3 upstream cells, the only fillable blanks (AF3's column
is the same code by design; boltz2/opendde 3012 upstreams are real OOMs).

In flight: tsp 39 runs `probe_memory.py --model openfold3 --tokens 499 1003
3012` (memory_analysis splits -> of3_probe.log in the session scratchpad);
a background `uv pip install -e .` provisions `openfold3/.venv` (log:
of3_venv.log). Next after those land: compare the OF3 splits against
protenix's (12.404 args / 0.374 out / 46.483 temp at 3012), attribute the
fattest OF3 buffers via an XLA dump if the split alone does not answer, then
add an `openfold3` branch to `bench/run_upstream.py` (checkpoint is in the
foldjax store; upstream CLI entry needs reading) and run the three cells
through tsp.

## OpenFold3 upstream wiring, state so far (2026-08-12)

- venv: `openfold3/.venv` provisioned clean (`uv pip install -e .`, exit 0).
- CLI: `run_openfold predict --query-json Q --inference-ckpt-path CKPT
  --num-diffusion-samples N --num-model-seeds K --runner-yaml Y --output-dir O`;
  checkpoints live in `openfold3/openfold3_weights/checkpoints/`.
- Input: the bench job is FoldJAX-neutral; `foldjax.input.materialize_native_input`
  with the openfold3 capabilities emits upstream's own query-set JSON (the
  backend already uses it, and the fixed `probe_memory.py` now does too).
- Schedule: CLI covers samples/seeds only. Recycles (`num_recycles: 3` default,
  `projects/of3_all_atom/config/model_config.py:177`) and the rollout step
  count must be pinned to 10/200 through the runner yaml's `model_update`
  presets/overrides -- exact override paths still to be read from how
  `model_update` is applied. MSA server must stay off (a3m paths ride in the
  query json).
- Probes: per-size `probe_memory.py` runs queued (of3p-499/1003/3012); the
  first attempt taught it the neutral->native translation and that it pads up
  only, so each size probes from its own job.

## OF3 vs protenix memory, compared on the real bench configs (msa 1024)

    tokens   openfold3 (args/out/temp = total | measured)   protenix
    499      1.60/0.18/ 2.83 =  4.60 | 10,442 MiB           5,058 MiB
    1003     2.15/0.72/ 7.28 = 10.16 | 12,043               8,711
    3012     7.52/2.16/53.09 = 62.77 | 66,264               12.40/0.37/46.48 = 59.26 | 60,759

(First probe run measured a different program: `probe_memory.py --msa-depth`
default None overrode the released 1024-row subsample -- 26 GiB at 499. The
flag now must be passed; the numbers above are the bench configuration.)

Where OF3's extra weight actually is:

- **Trunk dtype, the structural one.** OF3's upstream contract is
  `precision="32-true"`: every pair activation in the trunk is f32 where
  protenix's is bf16 ([[openfold3-bf16-embedder-island]] -- bf16 destroys the
  prediction, so this is not a knob). temp 53.1 vs 46.5 at 3012 despite
  protenix hauling a 16k-row MSA stack.
- **Outputs 2.16 GiB = distogram_logits [N,N,64] f32.** Not waste: at 3012 it
  fits the npz budget and is written; protenix's format never writes it.
- **args 7.5 vs 12.4** favours OF3 (1024-row MSA features, but f32 weights
  against protenix's bf16 cast).
- The 499 measured-vs-analysis gap (10.4 vs 4.6 GiB) is process-lifetime
  high-water: checkpoint-load staging dominates when the program itself is
  small.

Verdict: no protenix-style redundancy left in OF3; the curve difference is the
fp32 trunk contract plus a consumer-backed distogram output.

## OpenFold3 upstream column filled; table complete (2026-08-12)

    tokens  FoldJAX          upstream (0.3.1, plain torch)   speed  memory
    499     39 s / 10,442    97 s / 19,193                   2.46x  1.84x
    1003    126 / 12,043     323 / 25,620                    2.56x  2.13x
    3012    1,374 / 66,264   6,782 / 81,011                  4.94x  1.22x

ptm agrees per row (0.9325/0.9274, 0.9315/0.9300, 0.8762/0.8749): same model,
same weights, both sides. Caveat recorded in upstream-environments.md: this
upstream ran kernel-less (DS4Sci refuses sm_120; its cueq flag is broken at
n_samples>1), so the speed column is generous to FoldJAX here in a way the
other rows are not. The remaining blanks are all structural: AF3's column is
the same code by design, boltz2/opendde 3012 upstreams are real OOMs.

## Protenix 3012 gap: multi-agent + two-family peer synthesis (2026-08-12)

Consulted codex (gpt-5.6-sol) and antigravity via `oms peer-ask`, cross-checked
by three local read-only agents against the on-disk buffer assignment. Ground
truth overturned both peers' top guesses:

1. **The four 8.65-GiB bf16[128, 4*N^2] buffers are the TEMPLATE stack's
   tri-mul a-side projections** (gate+value x2; K=64=template c_z identifies
   it -- triangle.py:104-117's cueq guard rejects exactly this stack). XLA
   merged the 32-row chunked gemms back into whole-tensor gemms across all 95
   blocks and all 4 unrolled templates (concat-of-gemms -> gemm-of-concat),
   so the chunk knob demonstrably reaches the code and buys nothing there.
   ~34.6 GiB of the 46.5 GiB arena, live at once.
2. **The bench job has NO templates.** Args hold 5.98 GiB of literal zeros
   (template_distogram 5.27 all-zero, masks, token_bonds), and the template
   stack burns its 34.6 GiB computing on those zeros. MAX_TEMPLATES=4 shapes
   everything.
3. **The args story was misattributed in this log**: MSA is 0.551 GiB (4.4%),
   not the 12.4; `relp` (4.70 GiB) is a one-hot reconstructible in-graph from
   60 KB of metadata (consumers are bias-free linears = embedding gathers);
   cycle_msa_features contribute zero (bench runs full-depth MSA) and would
   ADD ~6.7 GiB as implemented.
4. **bf16-native cueq FFI is a phantom lever**: the binding is already
   dtype-polymorphic (verified bf16-in/bf16-out on this card); the only f32
   residual is the vendor's bias promotion, forced on cc 12.0 (no sm100f) and
   unreachable from FoldJAX. `_cueq.py:16`'s "takes float32" comment misleads.
5. Donation: alias 0.0 today, outputs (0.37) cannot absorb 5-GiB args, and
   the CLI reuses `features` across seeds -- not a lever (codex agreed).

Plan, ranked by GiB per risk, for the next session:
  P1  Host-side template gate (boltz2's `use_template` pattern, api.py:275):
      skip the template stack and its zero features when no template is real.
      First verify what upstream torch does with zero templates -- if its
      masked template reduction yields an exactly-zero pair update, skipping
      is numerics-identical; if upstream always runs the stack, match
      upstream instead and rely on P2. Expected for template-free jobs:
      arena -34.6, args -5.98.
  P2  Make `_triangle_contract`'s block loop a real loop (fori/scan writing
      via .at[].set, boltz2 triangle.py:120-123 shape) so XLA cannot invert
      the chunking; bounds the a-side operand at [32,N,128] ~ 25 MiB even
      with real templates. Watch the 3012 = 94*32+4 remainder.
  P3  Reconstruct `relp` in-graph from the five s32[3012] vectors (-4.70 GiB,
      ~1 ulp; JAX prunes the dead argument automatically).
  P4  u8-narrow the remaining one-hot args, cast back in-graph (bit-identical,
      ~-2 GiB on top of P1/P3).
  P5  Template python loop -> fori_loop carrying u (complementary bound).

## P1' (in-graph zero template features) reverted -- it traded 5.95 for +28

Measured, P2+P1' together: args 12.404 -> 6.456 exactly as planned, but temp
39.10 -> 67.54 GiB and the real peak 60.8 -> 77.9 GiB (wall 804 -> 1,178 s).
The folding assumption failed: the template feature concat mixes the rebuilt
zeros with nonzero aatype one-hots, so `linear(at)` cannot fold, and the
[N,N,108] per-template tensors that used to be read out of resident entry
parameters through slice fusions now materialize in the temp arena instead.
Entry-args bytes are cheaper than temp bytes here. Reverted (1349409); P2 (the
tri-mul scan, -7.4 GiB real) stays. Same caveat now attached to P3: rebuilding
`relp` in-graph must be the weight-split gather form that never materializes
[N,N,139], or it will repeat this failure.

## Protenix memory fight: won at every size (final, 2026-08-12)

    tokens  FoldJAX (this session's start -> end)   upstream   memory ratio
    499     5,058 -> 5,045 MiB                      5,130      1.02x
    1003    8,711 -> 8,468                          13,025     1.54x
    3012    60,759 -> 53,197  (wall 804 -> 795 s)   57,310     0.94x -> 1.08x

Cumulative at 3,012 across the whole audit: 84,106 -> 53,197 MiB (-37%), zero
wall cost, ranking_score 0.9554 vs upstream 0.9502 throughout. What landed:
output gating, per-sample confidence scoring, and the tri-mul block scan (P2).
What was tried and reverted with reasons recorded: the zero-template arg drop
(P1', traded 5.95 args for +28 temp). Deferred with a design constraint: relp
gather (P3) and u8 args (P4). The table now has no FoldJAX memory cell below
1.0x anywhere.

## AlphaFold 3 vendoring, phase 1 (2026-08-12)

Upstream AF3 source is Apache-2.0 (weights terms are separate), so it now
lives verbatim at `models/alphafold3/_upstream/alphafold3` -- 1.7 MB: 86 .py,
19 .cc, CMakeLists, LICENSE, NOTICE; excluded: test_data, the two 513 MB CCD
pickles (store-bound), compiled artifacts. Imports resolve through a
`sys.modules` shim (`ensure_registered`) rather than the OpenFold3-style
rewrite, because the pybind modules are compiled as `alphafold3.*` and a
rewritten tree would strand them; an installed distribution still wins.
The `alphafold3` extra now carries its runtime (haiku/absl/etils/tqdm/zstd);
jax/rdkit/tokamax were already compatible, and upstream's jax[cuda12] pin
never applies to vendored source.

Verified: the model stack imports from the vendored tree up to the compiled
boundary -- `constants/chemical_components -> alphafold3.cpp.cif_dict` is the
first pybind touch. Remaining phases: (2) fetch-time extension build (CMake
FetchContents abseil/pybind11/libcifpp/dssp; same class of one-time install
step as a weight download) with artifacts in the store; (3) CCD pickles into
the store; (4) backend in-process path through the vendored code, replacing
the external-install requirement.

Also learned the hard way: `uv sync` without `--inexact` removes the manually
installed external alphafold3 -- it did, mid-session; a rebuild is running.

## OF3 pair-chunk sweep at 3,012: closed, no gain

    auto(58): 1,374 s / 66,264 MiB    96: 1,370 / 66,309    128: 1,372 / 66,324
    (ptm 0.8762 / 0.8743 / 0.8747 -- inside the rerun floor)

The freed logits headroom bought nothing here: wall and peak are flat across a
2.2x chunk range, so the 3,012-token cost is the fp32 trunk's arithmetic, not
score-tensor traffic. `auto_pair_chunk_size` stays as is.

## AlphaFold 3 standalone: proven end-to-end (2026-08-12)

With no `alphafold3` distribution installed, `foldjax predict --model
alphafold3` completed the 499-token bench case on the vendored tree alone:
296.8 s / 3,033 MiB / ptm 0.97 / ranking 0.9671, against 285 s / 3,036 MiB on
the previously-installed distribution -- wall inside run noise, memory
identical. The path exercised everything vendoring added: the sys.modules
shim, the in-place-built `alphafold3.cpp` (uv build staging: static version,
LICENSE/README/terms files, libcifpp dictionaries beside the tree), the
store-copied CCD pickles, and the terms documents inside the package where
post-processing ships them from.

## ESMFold2 JAX port: started (2026-08-13)

The torch backend landed first and is honest about what it is -- upstream's
PyTorch model, a forked transformers, and a 24 GB ESMC-6B download -- which
makes it the only backend here that is not JAX, not torch-free and not
standalone. The port replaces it.

Sizing: the structure model is **234.8M parameters** (safetensors, measured),
smaller than Protenix's 368M: structure_head 131.5M, folding_trunk 75.6M,
msa_encoder 8.2M, confidence_head 7.8M, lm_encoder 6.3M, rest 5.0M. The wall
is ESMC-6B, a 6B transformer LM -- large, but an ordinary encoder, and MIT.

Landed:
* `models/esmfold2/models/primitives.py` -- linear, layer_norm, swiglu,
  transition_layer, adaptive_layer_norm, fourier_embedding
* `models/esmfold2/models/trunk.py` -- triangle multiplicative (both flows),
  transition (both residual conventions), pair update block, folding trunk,
  outer product mean, MSA pair-weighted averaging
* `tests/models/esmfold2/parity.py` -- written before any module, because
  that is what made the other four ports trustworthy
* 14 parity tests, all matching torch at 2e-5

Specs, read off the vendored source and kept because the source lives in a
fork that will move: `docs/ports/esmfold2/{trunk,diffusion}-spec.md`. The two
findings that change how the port is written: the trunk is a **linear
recurrence** (`z = a*z + inject @ b_mat.T`, 21 iterations, only z carried),
not AF2 recycling; and **inference is stochastic by design** -- random initial
pair state, LM dropout forced `training=True` at p=0.25, MSA rows resubsampled
every iteration.

Next, in order: the atom encoder/decoder and 3D-RoPE sliding-window attention;
the diffusion transformer and its EDM preconditioning; the sampler (Euler with
a Kabsch realignment, 48 real steps from 68 nominal); the confidence head; the
checkpoint mapper; then ESMC-6B. Each lands only with parity against torch.

## OpenDDE context parallelism: Fold-CP as sharding annotations (2026-08-16)

Upstream OpenDDE's Fold-CP is ~20 files of hand-written torch collectives
because torch has no partitioner. The JAX port needed none of that code --
only its layout: pair tensors row-sharded, transposes re-sharded, everything
token-linear replicated. `--cp-devices P` builds a 1D `cp` mesh in one
process (no torchrun equivalent) and the consolidated graph carries
`with_sharding_constraint` at the seams: trunk `z_init` and recycle carry,
every `pairformer_block` entry/transpose/exit, structural expansion,
diffusion conditioning `pair_z`, structural `relp`, confidence `z_pair`.

Two places could not be annotation-only, both because a loop iterates the
sharded axis:

* **Triangle attention** runs under `shard_map`: per-row attention is local
  to a row shard, the `[heads, N, N]` triangle bias is projected from the
  sharded tensor and passed in replicated, and the existing blocked row loop
  runs unchanged on each device's local rows. Rows pad to a multiple of the
  mesh and slice back, with a re-shard after the slice -- without it the
  partitioner returned a replicated result after every block.
* **The role-pair projection** (`_pair_project_by_role`) swaps its row-wise
  `lax.map` for a sum over the 7 column roles under CP: seven batched
  matmuls partition cleanly; the row scan would have gathered the whole
  structural pair tensor per step.

Routing rules under CP: cueq/tokamax are FFI calls the partitioner cannot
split -- unset triangle backends resolve to `xla_jit`, explicit fused ones
fail loudly (upstream's own rule: torch kernels only in distributed mode).
The blocked triangle-mult contraction and the inner `jit` wrappers are also
bypassed: the former iterates the sharded axis, the latter's trace cache
does not key on the mesh.

Evidence, all on a forced 4-device CPU mesh (`cp_shards` is a static arg, so
mesh changes are retraces): module parity for the pairformer stack and
structural expander at N=13 (indivisible by 4) is exact; end-to-end tiny.json
with real weights matches the single-device run at 6.1e-4 A max coordinate
diff and confidence summaries equal to 5 decimals; the compiled 2-block stack
at N=512 shows row-sharded input/output, `f32[128,512,128]` tiles, 15
all-gathers and 18 all-to-alls, and exactly 3 full-size pair tensors left in
the module -- the triangle-mult `b` operands, the known v1 concession.

Not measured: real multi-GPU memory and time (this host has one GPU; CPU
`memory_analysis` reports the host-shared total, which is invariant by
construction). Next lever if the all-gather hurts on small cards: a
`ppermute` ring for the triangle-mult contraction, ~50 lines inside
`shard_map`.

## Context parallelism rolled out to five backends (2026-08-16)

The OpenDDE recipe generalized in one pass: Protenix (by hand -- its modules
already carried the seams, since OpenDDE reuses them), then Boltz-2,
OpenFold3, and ESMFold2 by three parallel agents working from the recipe.
AlphaFold 3 is excluded on three grounds: the vendored upstream source is
authoritative per the project contract, its default attention is a Triton
kernel the partitioner cannot split anyway, and it is the leanest model of
the six.

What transferred unchanged: the seam list (block entry / transpose exchange /
born-sharded creation sites), the fused-kernel gate, the inner-jit bypass,
`cp_shards` as a static argument, and the pad+slice+re-shard shard_map
pattern. What each port forced:

* **Protenix**: nothing new -- entry activation, `pair_z` and host-`relp`
  seams, CLI. Its bf16 default trunk makes CP-vs-single coordinate diffs
  look large (0.53 A at 2 steps); at fp32 the same comparison is 7.3e-4 A,
  so the spread is bf16 reduction-order amplification, not CP.
* **Boltz-2**: rank-4 `[B,N,N,C]` triangle attention with an internal
  transpose; already had opt-in mesh plumbing in its trunk, which now
  defaults to the ambient CP mesh. Its fp32 trunk is documented as
  reduction-order chaotic through the rigid-alignment sampler -- the CP e2e
  gate there is confidence summaries (agree to 3e-3), not coordinates, and
  the control (a mathematically exact reorder alone) diverges *more* than
  CP does. Steering and affinity jobs are rejected under CP v1.
* **OpenFold3**: rank-general CP triangle attention (leading batch axis
  everywhere, more on template/confidence paths); chunking split three ways
  -- pair-transition chunk off under CP, triangle-attention chunk kept
  (local to the shard), OPM chunk untouched (replicated operands; turning
  it off would rebuild the 3,000-token OPM wall).
* **ESMFold2**: constraint-only -- no triangle attention, so no shard_map;
  the recurrence carry, MSA encoder pair, LM pair, diffusion condition pair
  and confidence pair are pinned. Module parity is bit-exact; end-to-end is
  not a valid gate (stochastic by design). The ESMC-6B checkpoint
  replicates per device.

Every backend's parity runs on a forced 4-device CPU mesh at N=13
(indivisible by the mesh, so the padding path is always exercised). Full
suite after the rollout: all green. Real multi-GPU still unmeasured -- one
card in this host.

## CP on real multi-GPU: the 24 GB wall falls at CP=4 (2026-08-16, SNU cluster)

First real multi-GPU execution of the context-parallel path, on a Slurm
cluster (`snucluster`): gpu2 = 4x RTX A5000 24 GB, gpu1 = H100 PCIe + 3x RTX
PRO 6000 Blackwell Max-Q 96 GB (the desktop's card). Getting it to run took
three fixes the desktop could never have surfaced, then it delivered the
number the whole project was after.

**The result.** OpenDDE at 1,003 residues on the A5000 node:

    single 24 GB card, bf16:  OOM              (control)
    CP=4 bf16:                completed, per-device peak 11.72-11.73 GiB
    CP=4 fp32:                completed, per-device peak 18.02-18.05 GiB

Per-device peaks agree across the four cards to 0.01 GiB -- the row sharding
is balanced. Against the desktop's single-device 45.5 GiB (fp32, same size):
2.53x per-device, and the shortfall from 4x is exactly the v1 concession --
45.5/4 = 11.4 sharded + ~5.5 all-gathered triangle-mult projection + margin
= 18.0. The fp32/bf16 confidence summaries agree to 4 decimals. A job that
cannot run on one 24 GB card runs on four of them.

**The three fixes** (all recorded in `cp-gpu-deployment-gotchas`):

1. NCCL P2P deadlock -- CP runs spun at 100% GPU forever on the A5000 node;
   `NCCL_P2P_DISABLE=1` (SHM transport) turned a 34-minute hang into an
   84-second run.
2. Triton GEMM autotune -- per-config `tt.dot` errors escalated to a stall /
   allocator churn on CP graphs; `--xla_gpu_enable_triton_gemm=false
   --xla_gpu_autotune_level=1`.
3. The incoming triangle multiplication contracted over the *sharded* axis,
   which the partitioner realised as a full-size f32 partial (49 GiB per
   device at 3,012 residues) plus an all-reduce. Fixed in all four ports by
   projecting the transpose and contracting in the outgoing form -- same
   arithmetic, sharded partials. Module parity stays green everywhere.

Also settled empirically: a mixed-architecture mesh (H100 sm90 + 6000 Pro
sm120) is refused outright -- "executable is built for device ... cannot run
it on device ..." -- so CP meshes must be homogeneous.

**The honest boundary.** 3,012 residues on 3x 96 GB still fails in v1: the
trunk shards, but the diffusion side is f32 by design and its atom-window
gathers pull full pair tensors per device (49 GiB), with one 84 GiB
allocation beyond any pool. v2 items, in value order: shard the diffusion
conditioning consumers (atom-encoder gather, per-step pair bias), then the
triangle-mult ppermute ring to reclaim the remaining all-gather, then cueq
under shard_map. Also operational: 24 GB cards need
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.92`; the default 0.75 pool is what failed,
not the cards.

## The v1 CP ceiling, measured: 2,800 residues on 3x 96 GB (2026-08-16)

Walking the gap between the 1,003-residue pass and the 3,012-residue failure,
bf16 CP=3 on the three RTX PRO 6000 Max-Q cards (pool 0.90 = 86.4 GiB),
per-device peaks from the allocator:

    2,000 residues:  43.2 GiB/device   completed
    2,500 residues:  67.1 GiB/device   completed
    2,800 residues:  84.7 GiB/device   completed  (1.7 GiB under the pool)
    2,000 fp32:      74.1 GiB/device   completed

The curve is cleanly quadratic -- 43.2 x (2500/2000)^2 predicts 67.5 against
67.1 measured, and extrapolates to ~98 GiB at 3,012, which is exactly why
that size OOMs. All 38 minutes of sweep, every size balanced across the
three cards to 0.1 GiB, confidence summaries sane at every size. So v1
context parallelism on this hardware runs OpenDDE to ~2,800 residues in
bf16 (~2,100 in fp32); past that, the unsharded fp32 diffusion side is the
binding constraint and v2's diffusion-consumer sharding is the lever.

## Fold-CP's own design, read from NVIDIA's boltz-cp (2026-08-17)

The Fold-CP this port has been chasing turns out to be NVIDIA's, published
with a paper (arXiv 2603.14806) and code (NVIDIA-Digital-Bio/boltz-cp); the
OpenDDE implementation is downstream of it. Reading the source corrected
three assumptions this port was built on:

* **The mesh is square, and 1xP is not a supported configuration** -- four
  assertions in `comm.py` refuse a non-square grid and the CLI `isqrt`s a
  scalar `size_cp`. Our 1-D row sharding is outside their design space, which
  is exactly why we still hold full-size buffers.
* **Triangle multiplication is Cannon's algorithm**, not an all-gather: skew
  both operands, then P steps of local matmul with a one-hop shift. The
  per-device transient is two double-buffered tiles, O((N/P)^2), never a
  panel. Both directions are the same code with a transpose-which-operand
  flag -- our transposed-outgoing rewrite is unnecessary once the ring
  exists.
* **The triangle-attention bias is never gathered.** It is redistributed by
  two P2P stages and then rotates one hop per ring step, so each rank holds
  one [H, I/P, J/P] tile. Our replicated [heads, N, N] bias is the 1-D
  compromise.

The diffusion side, which is our remaining wall, is solved structurally
rather than by dtype: the atom pair is never [N_atom, N_atom] but is born
window-batched (B, K, W, H, C) with the window index sharded, and the token
pair reaches it only through a bounded-rectangle gather after projection to
the narrow atom_z width. Their fp32 islands are the same as ours -- the
defect was never the dtype, it was the consumer pulling a whole pair tensor.

For a JAX port the good news is that every schedule is a *static*
permutation, so `lax.ppermute` covers the transpose, the Cannon skew and the
ring; `psum` covers the OPM mask denominator and the confidence reductions;
and everything placement-preserving stays plain `with_sharding_constraint`
(they hand-wrote ~40 DTensor wrappers only because torch has no GSPMD). The
one primitive that does not port cleanly is their data-dependent rectangular
gather, which needs runtime interval arithmetic; a fixed halo or an
all-gather of the *projected* z is the static-shape substitute.

Order to implement, each step parity-testable against the current 1-D code:
mesh + placements (with MSA depth on cp0, tokens on cp1) -> grid transpose
primitive -> Cannon triangle multiplication -> rotating-bias triangle
attention -> OPM and pair-weighted averaging -> the atom/diffusion path ->
confidence and distogram heads last. Two testing ideas worth stealing: a
per-parameter gradient parity check through `full_tensor()`, and comparing
fp32 error *distributions* by percentile against an fp64 oracle instead of
hand-tuning a tolerance.

## The square grid lands, and two ways a CP test lies (2026-08-17)

Fold-CP's own layout -- rows and columns both split on a square grid, with
the triangle contraction run as Cannon's algorithm -- now works in the
Protenix/OpenDDE shared modules, in OpenFold3, and in Boltz-2. Per-device
pair cost drops from `O(N^2/P_rows)` with a full-width column axis to
`O(N^2/P_total)`, and the ring never materialises a full-width operand: the
transient is two tiles.

Getting there produced two lessons worth more than the feature.

**A grid permutation does not transpose a tile.** The first implementation
read the moved operand as if its axes had been swapped and used the matmul
einsum `ikd,kjd->ijd`. It runs, it type-checks, and it computes a different
function -- 15.9 off the dense result on a value scale of 13.4. Cannon moves
whole tiles and leaves their internal axes alone, so the local contraction
stays the dense einsum *of that direction*: `ikd,jkd->ijd` outgoing,
`kid,kjd->ijd` incoming. (Adding a local `swapaxes` after the grid hop is the
equally valid other spelling, which is what the Boltz-2 port uses.)

**Every context-parallel parity test in the repo was vacuous.** The pattern
was `jax.jit(run)(z)` once outside the mesh and once inside it. `jit` caches
its jaxpr on the callable, and the mesh is a module global read at trace
time, so the second call replayed the unsharded program: the test compared
the unsharded program against itself. The tell was in plain sight and had
been read as good news -- an *exactly zero* difference, which a real
`shard_map` cannot produce because it reorders float32 accumulation. Fixed
with `jax.clear_caches()` before each mesh block, and proved non-vacuous by
mutation: the wrong einsum now fails the gate it used to pass.

**A 2x2 mesh cannot falsify a sign.** Every shift in the schedule is +-1 or
+-coord, and modulo 2 a forward hop and a backward hop are the same
permutation, so skew signs and ring directions were untested by construction.
Nine forced CPU devices separate them. Measured:

    mutation                       2x2     3x3
    lhs ring hop reversed          pass    FAIL
    row-skew sign flipped          pass    FAIL
    col-skew sign flipped          pass    FAIL
    *both* ring hops reversed      pass    pass   <- not a defect

The last row is the trap: reversing both hops walks the k blocks in the
opposite order with the operands still paired at every step, so it is an
equivalent schedule at any size. Only breaking the pairing is a bug.

`--cp-layout` selects the layout and **`auto` stays on 1-D**. The grid is the
better design, but every published number for this feature -- the size
ladders, the per-device peaks, the 1,003-residue GPU parity -- was measured
on the 1-D layout, and a default that silently changes the program would make
those numbers describe a configuration nobody can reproduce. It flips when
the grid has its own GPU evidence.

**Postscript: the whole-stack run found what the module probes could not.**
The 2-D layout passed every module-level parity probe and then failed the
first real end-to-end run: triangle attention's `shard_map` named the 1-D
mesh axis literally, and under the square grid that axis is called something
else. Nothing caught it because the attention had only ever been traced under
a 1-D mesh -- the module probes exercised the multiplication on the grid and
the attention off it, never both at once. Fixed by deriving the spec from the
layout (`pair_row_spec`), and locked in with a whole-stack probe at 2x2 and
3x3 that fails when the literal is reinstated. Real weights end to end,
1-D against 2-D on the same four devices: 3.7e-4 A max coordinate difference,
confidence summaries equal to four decimals.

## The workaround was the bug (2026-08-18)

Context-parallel runs on GPU returned NaN for every confidence output while
coordinates, distogram and both trunk representations stayed finite. A day of
narrowing ruled out the obvious suspects one at a time -- kernel (XLA and
cueq both), shard count (2 and 4), token count (9 and 1,003), layout (1-D and
2-D), padding (a token count divisible by the mesh fails too), card (A5000
and RTX 6000 Ada), and the code itself (CPU meshes are finite at every
setting). An instrumented run then showed every *input* to the confidence
head finite and every output NaN, which pointed inside the head.

It was none of those. It was `XLA_FLAGS`:

    --xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=1  -> NaN
    the same run with XLA_FLAGS removed                            -> finite

`--xla_gpu_autotune_level` performs a correctness check on autotuned results
at level 3 and above; at level 1 an algorithm is selected without it, and a
GEMM that returns garbage is never rejected. Nothing in the port was wrong.

The deeper mistake is procedural, and it is worth more than the fix. When the
first multi-GPU runs hung, two workarounds went in at once -- these XLA flags
and `NCCL_P2P_DISABLE=1` -- and both were then carried as required. Only one
was: the hang was the NCCL peer-to-peer deadlock, and a run with P2P disabled
and no XLA flags completes fine. The unnecessary half sat there quietly
corrupting a whole class of outputs, and it was written into the CHANGELOG as
advice. Two workarounds applied together must be removed one at a time before
either is believed.

**Correction, an hour later: the flags were not the cause.** The A/B above was
a single pair of runs, and it read cleanly -- flags on, NaN; flags off,
finite. A later run of the *same* flag-free configuration produced NaN again,
this time including the coordinates. So the failure is intermittent, the
one-pair comparison was luck, and the autotune flag is exonerated as the sole
cause. What is established is narrower and worse: context-parallel runs on
these GPUs intermittently produce non-finite output, single-device runs do
not, and CPU meshes never do. The next measurement is frequency -- the same
job repeated five times at CP=4 and three times at CP=1 -- because a rate is
what tells a race apart from a threshold. The CHANGELOG now says multi-GPU
output is unvalidated rather than naming a cause it cannot support.

**Second correction: the retraction was premature too.** With the flags gone,
six consecutive context-parallel runs on a node whose four cards all
initialise came back finite. The run that had refuted the flag hypothesis --
flag-free and still NaN -- turns out to have been on a node in a degraded
state: Slurm handed the job four A5000s, JAX could initialise only three, one
card carried a leftover CUDA context with no owning process, and every
context-parallel run there returned non-finite output regardless of settings.
So there are two independent ways to get NaN, and each one masked the other:

    clean node, no flags   6 runs   all finite
    clean node, flags      1 run    NaN        (repeat under way)
    degraded node, either  many     NaN

Three self-inflicted lessons, in the order they were learned. A single A/B
cannot establish a cause when the failure is intermittent -- that produced the
first wrong conclusion. A single counter-example cannot retract one either --
that produced the second. And a diagnostic that does not verify its own
preconditions measures nothing: two "failure rate" jobs reported five failed
runs each, which turned out to be `only 3 JAX devices visible` rather than any
numerical failure, because the node had not released the cards and the harness
threw the error away with `>/dev/null`. The device-count guard that now opens
those jobs is the cheapest thing in this entire investigation.

The degraded node is worth reporting on its own: after a day of `scancel`,
one A5000 on that node no longer initialises and another holds an orphaned
context. Anything measured there today is suspect.

**Third correction, and the one the evidence actually supports: the fault is
in the program.** Two controls settled it. The flag hypothesis died on a
same-node A/B -- on the node that had produced NaN with the autotune flags,
removing them left the rate unchanged (7 of 14 runs non-finite with, 4 of 6
without), so the single-pair comparison that first accused the flags was a
coin flip against a base rate near one half. The bad-node hypothesis died the
same afternoon: the node that had returned six finite runs from six returned
a NaN run on the seventh attempt. There is no environment left to blame.

Before spending more GPU time on rates, every retained run directory on the
cluster was re-checked -- 63 of them, most never checked at all, because a job
that exits 0 and writes a structure had been read as a success, which is
exactly the reading a silent NaN passes. The map that came back is the most
useful artifact of the day:

    single-device runs, every size            12 / 12 finite
    OpenDDE ladder 1,003 / 2,000 / 2,500 / 2,800   all finite
    Protenix 3,012 / 4,500                    all finite
    Boltz-2 2,000 ... 5,500                   all finite
    OpenDDE short context-parallel probes     19 non-finite, about half

So the published per-device peaks stand, and the failure is not a function of
size, layout, shard count, node or flags. It is also not the sampler settings:
the finite and the non-finite runs use the same `--n-step 2`.

Localisation, from a probe that returns the trunk representations alongside
the outputs, is unambiguous. In a failing run `s_inputs`, `s_trunk`,
`z_trunk`, the structural refiner's outputs and the distogram logits -- which
read `z_trunk` alone -- are all fully finite, and `coordinate` is fully NaN,
taking the confidence heads with it. Everything sharded is right; the failure
appears in diffusion. Calling the same compiled program five times inside one
process settles when the coin is flipped: the first process returned five
finite calls, but the second returned finite, NaN, NaN, NaN, finite -- same
executable, same inputs, same devices. So the outcome is decided per
execution, not per compilation, which rules out an autotuned algorithm choice
being the thing that varies and leaves only two candidates: a buffer the
program reads without having written it, or a race between an asynchronous
collective and its consumer.

That leaves one suspect with a mechanism. Context parallelism reaches past the
trunk exactly once, at `pair_z = shard_pair_rows(pair_z)` in the diffusion
conditioning, and the tensor it constrains is then read by an arbitrary
two-axis gather, `z_token[..., idx_q, idx_k, :]`, over a row count that does
not divide by the shard count. An ablation with that single constraint removed
and nothing else changed is running from an isolated checkout.

The methodological entry for the day is not the third correction but what it
cost: three published causes, two of them wrong, all three argued from runs
that were never repeated enough to distinguish a cause from a coin. The base
rate should have been measured first. It takes six runs.
