
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
