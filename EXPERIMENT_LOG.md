
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
