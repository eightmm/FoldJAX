<!-- W6 (performance, memory, scale) entries for CHANGELOG.md's Unreleased section. -->

### Changed

- **AlphaFold 3 warm processes can read their persistent-cache entry back.**
  XLA refused AlphaFold 3 GPU cache entries on load (`RET_CHECK ... Invalid
  metadata payload id 201 with payloads size 192`), so the warm process
  recompiled the main program (jctc-v3: 143 of 160 AlphaFold 3 warm processes,
  all 70 of DeepMind's runner and 73 of 90 FoldJAX ones). The failing check
  reads only Tokamax's HLO metadata payloads; the executed program is now traced
  without them, while the autotuning discovery lowering keeps them under a
  separate trace. The compiled kernels are unchanged; cache keys change once.

- **OpenFold3 denoises one sample at a time above 4,888 tokens on a serial
  run** when `diffusion_chunk_size` is omitted. 4,888 is the largest serial
  size with a completed unchunked rollout; above it the `f32[5, N, N, 128]`
  pair conditioning alone is 102.9 GiB at 6,568 tokens. Every measured size
  keeps its program; an explicit width, `None` included, still wins.

- **Padding steps its token grid by 128 below 1,024 tokens** (128 ... 896,
  then 1,024 ... 8,192 by 256; 36 buckets). The derived axes follow: OpenDDE's
  8RG4 pads 373 -> 384 tokens, 720 -> 768 structural tokens and 3,058 -> 9,216
  atoms, where it padded to 512 / 1,024 / 12,288 at 1.62x its unpadded wall.
  Padded executables below 1,024 tokens are new shapes and compile once.

- **Admission says `fits` for measured small OpenFold3 and OpenDDE runs, and
  admits OpenDDE at 4,040 structural tokens.** Below their fitted domains
  (OpenFold3 from 129 tokens, OpenDDE bf16 from 246 structural tokens) every
  completed jctc-v3 run sits under the law's upper estimate, so admission now
  reports `fits` there -- and still never refuses in that band. At exactly
  4,040 structural tokens (2,096-residue L2000_5dei), measured completing at
  78,616 MiB, the bound is that measurement plus its 28 MiB repeat spread,
  131 MiB under the 0.9-pool threshold, instead of the law's 80.9 GiB refusal.

- **Fewer small compiled programs per process.** Weight trees move to the
  device with `device_put` and are narrowed in one grouped convert; integer
  feature checks run in NumPy. On two small CPU inputs Protenix compiled 145 ->
  23 programs, OpenDDE 209 -> 36, ESMFold2 177 -> 159, with every output array,
  structure and confidence file bitwise identical.

- **`kernel_autotuning=heuristics` runs AlphaFold 3 on sm_120 cards.** Tokamax's
  heuristic GLU tile asked for more than the 99 KiB of shared memory per block;
  on compute capability 12.x it now drops pipeline stages (then tile width)
  until it fits. The default stays `autotune`.

### Added

- **`cost.breakdown` in `foldjax_run.json`:** seconds spent in weight load,
  featurize, the language model, trace, lower, compile and cache restore
  (from JAX's own monitoring events), the derived `execute and host`
  remainder of `predict`, and counts of compiled programs and cache hits.

### Known issues found

- The OpenFold3 blocked-pair peak law under-estimates a nucleic/ligand-heavy
  input inside its domain: mixed_3k_5npk at 3,061 tokens peaked at 41,260 MiB
  against an upper estimate of 29,876 MiB, so admission called a run 11 GiB
  over its estimate a fit. The law keys on tokens only.
- In-bucket recompiles under `--padding` come from the per-input MSA bucket
  (Boltz-2, Protenix, OpenFold3; by design, to bound peak -- `--pad-msa` pins
  it) and, for Protenix, the chain count, which the confidence summaries take
  as a static argument. The MSA cycle tape width is already pinned under
  padding.
