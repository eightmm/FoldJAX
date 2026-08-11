
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
