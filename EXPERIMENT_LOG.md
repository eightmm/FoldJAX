
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
