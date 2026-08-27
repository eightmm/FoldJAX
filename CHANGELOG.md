# Changelog

What changed between releases, and why. Entries name the behaviour a user can
observe; the reasoning lives beside the code.

FoldJAX follows [semantic versioning](https://semver.org). Model weights,
upstream defaults and scientific results are covered by the project contract in
`PROJECT.md` rather than by this file: a release never changes what a recorded
command predicts unless it says so here, in its own paragraph.

## Unreleased

### Changed

- **`matmul_precision` is a knob you can actually turn, on all six models.**
  It had been in the neutral vocabulary all along, and all six backends
  answered `does not support matmul_precision` -- the one knob the vocabulary
  advertised and no adapter had ever implemented. It could not be delivered by
  renaming, which is how every other neutral knob works: four ports open their
  own `jax.default_matmul_precision` scope inside the call, so a scope the
  adapter opened around them was overridden, and the other two opened none, so
  there was nothing to reach. The adapter now records the request in a
  ContextVar and the port that pins reads it where it used to name its own
  constant. Every default is unchanged -- Boltz-2 stays `highest` because
  upstream's `main.py:1096` asks for it, OpenFold3 and Protenix stay `high`
  because their upstreams select TF32, and the three that pinned nothing still
  pin nothing until asked -- so a run nobody asked to change is the run it
  already was. It is scoped rather than latched, for the reason those pins
  already record: a process-global update leaves the next unrelated thing in
  the caller's precision.

- **A neutral knob cannot be advertised and unimplemented again.**
  `tests/test_execution_knob_coverage.py` checks the backends against the
  promise `KNOBS` makes rather than against each other -- which is what let
  this through, since all six agreed. It also checks that the value survives
  translation, that the adapter takes the option out of the dict before it can
  reach a native signature, and that the scope reaches the ports that pin
  nothing, by reading `highest` back out of the lowered HLO.

- **Four OpenFold3 config fields had escaped the guard built to stop exactly
  that.** `test_no_field_is_left_unchecked` exists so that adding a field to
  `InferenceConfig` cannot silently skip the upstream comparison -- but the
  test is itself `torch_parity`, so it ran nowhere, and `cp_layout`,
  `cp_shards`, `returned_representations` and `stop_after_trunk` had gone
  undeclared. All four are FoldJAX's own; `grep` finds none of them anywhere in
  the OpenFold3 checkout. The defect was the missing declaration rather than
  the fields, so each is now listed with the reason it has no upstream
  counterpart, and the set is named `foldjax_only` instead of `shapes` -- which
  three of its entries already were not.

- **A parity test failed where it should have skipped.**
  `test_torch_parity_preprocessing.py` guarded `torch` with `importorskip` and
  not `pytorch_lightning`, which upstream's `InferenceDataset` imports on the
  way to the tensorizer. An environment with torch and without Lightning --
  what `--extra openfold3-preprocess` produces -- got a failure rather than a
  skip with a reason. Lightning is now guarded too, and provisioning it lets
  the test run for the first time: the NumPy tensorizer matches upstream's.

- **The OpenFold3 parity environment is written down, with what it costs to
  get wrong.** 396 of that suite's tests are `torch_parity` and run nowhere by
  default. `bench/upstream-environments.md` carries the recipe -- including
  that `uv sync` prunes anything outside the lockfile, so the extras go first
  and torch second -- and records 395 passed, 3 skipped at `a29e4a3`. A first
  attempt reported 189 failed and 64 errors in 21.8 seconds; none of it was
  real. 21.8 seconds is 189 imports collapsing, 252 of them on `biotite`. A
  parity run that finishes far too quickly is reporting on its own
  provisioning.

- **A Boltz-2 parity test asserted something the backend never promised, and
  no ordinary run could see it.** 24 Boltz-2 modules leave collection when
  torch is absent, so the shipped environment collects 166 tests where a torch
  environment collects 267; the missing 101 run in no CI job either, because CI
  has no torch and the comparison needs the publisher's 2.28 GB checkpoint. Run
  against the released weights, one failed:
  `test_checkpoint_diffusion_transformer_layer_chunk_matches_unchunked`
  demanded that a chunked and an unchunked diffusion-transformer layer agree
  bit for bit, and they differ by 1.7e-5 absolute on 72% of elements. Nothing
  regressed -- the repository's first commit fails it too. Chunking slices the
  query axis, which changes the einsum extent, and XLA picks a different
  contraction schedule for the smaller operand; a twenty-line script
  reproduces it against the bare primitive in both environments, with and
  without torch imported. The test now asserts closeness at 1e-3, about sixty
  times the measured deviation, with the measurement recorded beside it.
  Chunking stays a live production path, so the test still fails if it ever
  changes the answer for real.

- **What the Boltz-2 parity tolerances actually achieve is now written down.**
  Instrumenting every `assert_allclose` in that suite recorded 111
  comparisons: 87 use less than 1% of the tolerance they declare, and 23 are
  bit-identical while declaring as much as 2e-3.
  `bench/upstream-environments.md` carries the numbers, the command that
  reproduces them, and the reason they are not an argument for tightening all
  87 -- they come from one CPU backend, and a threshold calibrated to this
  machine's contraction schedules would fail on the next one for no real
  reason.

- **The ESMFold2 atom encoder is checked against upstream's released weights,
  for the first time.** That test had never run: it imported
  `modeling_esmfold2_common`, a path in no released `transformers`, so it
  reported "skipped" everywhere. `transformers` and the published checkpoint
  have diverged in naming -- 2,226 names against 1,594, 12 in both -- but they
  hold the same trained tensors, so each side now loads its own artifact and
  nothing is mapped. The embedding agrees to 4.8e-7. Past it the two differ by
  1.3e-2, all of it one deliberate cast: this port drops q/k/v to bfloat16 for a
  float32 caller, which the reference it was ported from did and the current
  library does not. Removing that cast alone takes the layers to 4.9e-6 and the
  token output to 2.8e-5, so nothing else disagrees -- and the cast does not
  fire on the released bfloat16 path, where float32 scores would be 29,524 MiB.

- **Why the ESMFold2 encoder parity test does not run is now a measurement
  rather than a guess.** `transformers` and the published checkpoint have
  diverged in parameter naming: against transformers 5.16.1 the library's
  `EsmFold2Model` carries 2,226 names, the released safetensors carries 1,594,
  and 12 appear in both. Loading the published file through the library reports
  every tensor as unexpected. FoldJAX follows the checkpoint -- which is why
  `assets.py` converts nothing -- so a parity run against the library's modules
  needs the library's own re-exported weights, a separate artifact. The rotary
  comparison works because that module holds no parameters at all.

- **The ESMFold2 rotary parity test asserts bit-identity instead of a
  tolerance three times the one its own file uses.** It compared at
  `atol=rtol=3e-3`, without the reason `PROJECT.md` requires above 1e-3, in a
  file whose constant is 2e-5. The gap was real -- 1.9e-3 measured -- and it
  was a dtype mismatch rather than a disagreement: FoldJAX returns these tables
  in bfloat16 because upstream hardcodes that, and the comparison took
  upstream's float32. 2e-3 is what bfloat16 costs on values in [-1, 1], so the
  tolerance was the size of the cast and hid anything smaller. Upstream takes
  the dtype as an argument now, so the two are compared as the same thing and
  come out bit-identical. The window test, which is pure JAX, no longer skips
  because torch is absent: the import guard was module-level and took it along.

- **The out-of-memory explainer told people to raise a fraction that could not
  help.** Its branch compared the total against the *device* and then reported
  the *pool*, so an allocation that fits both printed "This is the pool's limit
  and not the card's" and advised `=0.95`. Measured, a 90.5 GiB need failed
  inside a 93.1 GiB pool and failed again at 0.98: the pool is assembled
  piecewise and can hold a total it cannot serve in one block. There are three
  outcomes now -- past the device, past the pool, and fits both -- and the third
  says it is fragmentation and that a larger fraction will not help.

- **`bench/structures.py` had never once reported on `protenix-v2`.** It split
  each run directory on every hyphen and read the second field as the
  implementation, so `protenix-v2-foldjax-...` became model `protenix` with
  implementation `v2`, matched neither side, and was dropped without a word. It
  splits on the implementation now. `bench/drive.py` likewise spawned a
  hard-coded `.venv/bin/python`, which silently ignored
  `UV_PROJECT_ENVIRONMENT` and measured the wrong environment; it uses the
  interpreter it is running under.

- **ESMFold2 is in the benchmark table, and OpenDDE's precision lever is
  written down.** ESMFold2 had been measured under the same schedule all along
  and the table dropped its rows, so a reader could not tell "not measured"
  from "left out". The published table now matches what `bench/report.py`
  emits. The structure-level gate, which the doc said it had not spent a card
  on repeating, was repeated at 1,354 tokens for free from structures the run
  had already written.

- **Importing FoldJAX no longer costs you the GPU for following jaxlib's
  documentation.** jaxlib renamed the pool-fraction variable to
  `XLA_CLIENT_MEM_FRACTION`; either spelling works alone, and both together
  raise inside the CUDA plugin's `initialize()`, after which JAX reports "an
  NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not
  installed" and runs on the CPU. The jaxlib is installed, and nothing in that
  message names the pair of variables. FoldJAX added its own spelling with
  `setdefault`, so anyone who had set the current name lost the GPU by
  importing `foldjax.models.openfold3` -- silently, and with a message pointing
  somewhere else. The fraction is now set through one place that respects
  whichever spelling is already present and never adds a second, and the
  out-of-memory explainer names the variable actually in effect rather than a
  constant.

- **Two more benchmark sizes, and the one where FoldJAX runs what its upstream
  cannot.** At 1,354 tokens OpenDDE finishes in 529 s at 77.8 GiB while upstream
  OpenDDE holds 91.15 GiB, asks for 9.82 more, and dies on a 94.97 GiB card. The
  size was chosen rather than found: upstream's 1,003-token peak is 59.5 GiB
  against FoldJAX's 42.9, which puts the two ceilings about 250 tokens apart,
  and a first attempt at 1,531 confirmed the upper edge by failing on both
  sides. At 4,926 tokens nothing here runs, on either side, which is also a
  result -- and not one failure but three: three FoldJAX runs ask for more than
  the device holds, Protenix asks for less and still fails with the pool raised
  to 0.98, and upstream Protenix v2 refuses above 2,560 tokens before
  allocating anything.

- **The figure no longer says "out of memory" about runs that never said so.**
  Its label for a missing bar was "refused, or else out of memory", which put a
  cause on cells that have no evidence for one: upstream Boltz-2 at 4,926 tokens
  reaches 92.2 GiB, produces nothing, exits 0 and writes no error anywhere. That
  is now "no structure". The refusal is also read from the recorded text rather
  than from prose someone remembered to write, which is why the same refusal
  showed as a refusal at 3,012 tokens and as an out-of-memory at 4,926.

- **An upstream that exits 0 with nothing predicted keeps its reason.**
  `bench/run_upstream.py` looked only in `ERR/`, which is OpenDDE's convention.
  Boltz-2 writes no such file and had its captured stderr discarded, leaving a
  failed row whose only evidence was a peak; OpenFold3 writes to its own
  `logs/predict_err_rank*.log` and had its out-of-memory reported as a bare
  failure. Both are read now, and the captured stderr is the fallback.

- **Long-lived processes now bound the largest inference and preprocessing
  caches.** Protenix and OpenDDE whole-model graphs are owned by an eight-entry
  least-recently-used JIT pool; evicting an identity explicitly clears its JAX
  executable rather than retaining every unpadded shape and static option ever
  seen. AlphaFold 3 protein/RNA search results are now memoized only for the
  duration of one input job, which keeps homomer reuse without retaining full
  MSAs and templates across unrelated jobs. Its process-wide CCD loader keeps
  at most two base dictionaries, and a user CCD is applied to a shallow copy so
  one job can no longer alter the chemical-component table seen by later jobs.
  Component metadata keeps its 128-entry reuse window on that CCD instance, so
  the helper cache cannot retain job-specific dictionary copies after the job.

- **OpenFold3 uses its validated atom-owner table for token-to-atom
  broadcasts.** The high-level feature validator already proves that
  `atom_to_token_index`, atom counts and masks describe the same layout. The
  noisy-position encoder and atom decoder now gather through that table instead
  of reconstructing it with an `[atom, token]` comparison and reduction on
  every trace. The public count-based primitive remains the fallback for direct
  callers. FP32/BF16, padded, nonfinite and signed-zero CPU regressions are
  byte-identical; an isolated 256-atom/64-token StableHLO no longer contains the
  `[1, 256, 64]` predicate or its reduction. This is a source-level graph and
  temporary reduction, not a whole-model or GPU peak claim.

- **A directory of jobs is now a batch from Python, not only from the CLI.**
  `PredictionRequest(inputs=(directory,))` expands to the job documents
  directly inside it, sorted, so running a sweep no longer starts with a glob
  the caller writes themselves. Sorting is half the point: `iterdir` returns
  filesystem order, so a hand-written batch runs in an order that depends on
  which machine created the files. The CLI keeps expanding the wider set it can
  convert -- FASTA and deposited structures become job documents first -- and
  both spellings now go through one implementation. `input` is one run, so it
  refuses a directory and names `inputs` in the message rather than reporting
  "is not a file" and leaving the caller to guess.

- **OpenDDE forms shape-complementarity chain masks inside their existing
  chunks.** The floating-point calculation was already bounded to 128 token
  rows, but two NumPy-derived boolean masks were still materialized in full as
  `[token, atom]` and `[token, token]` constants. Keeping only the two chain-id
  vectors and comparing each row chunk removes those quadratic constants
  without changing any arithmetic. A synthetic CPU density-stage probe at 512
  tokens and 4,096 atoms reduced StableHLO from about 4.21 million to 57,000
  characters and warm time from 8.73 to 6.67 ms; outputs were byte-identical.

- **OpenFold3 computes each symmetric chain-pair ipTM only once.** The union
  mask, chain-presence test and interface pTM call are identical for `(i, j)`
  and `(j, i)`, so the upper-triangle result is now reused for its mirror while
  the diagonal remains zero. This preserves bytes while halving the unrolled
  pair calls and roughly halved the isolated 16-chain StableHLO. Runtime may
  not move because XLA could already common-subexpression-eliminate the two
  copies; this is principally a trace and compile-size reduction.

- **Boltz-2 selects three ligand-frame atoms with stable top-k on the ordinary
  finite path.** The confidence head previously sorted every atom distance and
  kept only three indices. Equal finite distances and infinities retain their
  historical input order; rows containing NaNs keep the full stable argsort so
  nonfinite direct-call semantics are unchanged. In an isolated CPU probe with
  three samples, 128 representative rows and 1,024 atoms, warm time fell from
  9.65 to 2.03 ms. This is a confidence-stage result, not a whole-model or GPU
  latency claim.

- **Multi-seed ESMFold2 sessions retain the combined language-model embedding,
  not all 81 ESMC layer states.** A separately compiled layer-normalize,
  projection, softmax and layer-combination prefix turns the seed-independent
  `[B, N, 81, 2560]` stack into `[B, N, 256]` before it enters the session
  cache. At 1,000 tokens in BF16 that is 414,720,000 retained bytes versus
  512,000, while the pair lift and every seed-dependent structure operation
  remain in the structure graph. Direct/model APIs still accept and default to
  raw hidden states, and legacy split wrappers recompute rather than retaining
  them. FP32 and BF16 CPU split-boundary tests are byte-identical; fresh-process
  released-weight CUDA peak and latency remain deployment gates because the
  extra compiled boundary can affect device fusion and initial peak.

- **`uv sync` with no arguments is now the environment to have.** The CUDA 13
  runtime also lives in a `gpu` dependency group, and uv enables that group by
  default, so the documented install lost both of its flags: `uv sync --extra
  cuda13 --group dev` is now `uv sync`, and `dev` had in fact been a default
  group all along. The `cuda12` and `cuda13` extras are unchanged and remain
  the published surface, because a wheel or a `foldjax[cuda12] @ git+...`
  install can only ask for an extra, never a group; the CUDA 13 pins therefore
  exist in both places and a test keeps the two copies equal. Two departures
  still take a flag. A CPU-only machine syncs `--no-default-groups --group
  dev`, which is what CI now does. A CUDA 12 node exports `UV_NO_GROUP=gpu` for
  its whole shell rather than passing a flag once, because `uv run` re-syncs
  the default groups before it runs anything and would otherwise reinstall the
  CUDA 13 wheels beside the CUDA 12 ones; forgetting it stops with a conflict
  error instead of installing both generations.

- **Protenix v2 runs past upstream's 2,560-token limit, and says what that
  costs.** Upstream refuses protenix-v2 above 2,560 tokens by assertion, before
  allocating anything, and its own message gives the reason: "It might cause
  OOM." That is a memory budget rather than an architectural bound — the
  relative position encoding buckets residue separation with
  `clip(delta, 0, 2 * r_max)` at r_max=32, so nothing in the model is bounded
  by token count — and a budget calibrated on one implementation does not
  transfer to another that uses 1.2x to 1.3x less of it. FoldJAX now warns and
  runs where upstream refuses: measured, 3,012 tokens completed five samples at
  78.2 GiB. The warning carries an expected size, fitted on this port's own
  measurements as `2.1 GiB + 0.0086 MiB × tokens²`, which reproduces the
  1,003-token peak to within 5%; and it says plainly that neither
  implementation validates the model at that length, because a run that fits is
  not thereby a run that is right. `strict_token_limit=true` restores upstream's
  refusal unchanged.

  A switch option reached the native CLI as `--flag value`, which argparse
  rejects with exit 2 and a usage dump naming no argument. Switch-style options
  now render as a bare flag when on and nothing when off, and a value that is
  neither true nor false is refused with a message that says which option.

- **Protenix v2 is supported and fetched, alongside the release.**
  `--profile v2`, or `model_name=protenix-v2`, selects the 464M model announced
  2026-04-08 against the release's 368M, and `foldjax setup` now takes both:
  it iterates a model's profiles rather than the model alone, so a second
  supported checkpoint is no longer silently skipped. The port needed no
  architecture change. v2 is the same blocks at wider dimensions — c_z 256 with
  `hidden_scale_up`, which sets triangle multiplication hidden to c_z and
  triangle attention heads to c_z // 32 — and these blocks read those widths
  from the parameters they are handed rather than from a config. What the model
  name carries is the rest of v2: its sampler schedule and the 2,560-token
  limit.

  The checkpoint comes from a mirror, and that is stated rather than hidden.
  ByteDance's own CDN key stopped serving `protenix-v2.pt` the day of the
  announcement — it answers anonymous requests with `AccessDenied` where the v1
  checkpoints serve normally, and a key that does not exist answers 404 — and
  upstream says accessibility is under company-level internal review with no
  timeline (bytedance/Protenix#294, #295, both open since 2026-04-08). The
  mirrored archive was verified before its SHA-256 was pinned here: 464,442,431
  parameters against the 464.44 M upstream documents, Pairformer weights of
  `tri_mul_out.linear_a_p (256, 256)` and `tri_att_start.linear (8, 256)`, and
  a clean load through the port's own mapping. A substituted file fails on the
  hash before it is ever loaded.

- **Protenix gains its other published base checkpoint.**
  `--profile base-20250630` fetches
  `protenix_base_20250630_v1.0.0.pt`: the same 368M architecture as the
  release, trained to a 2025-06-30 wwPDB cutoff instead of AlphaFold 3's
  2021-09-30. Upstream recommends it for practical use and keeps the release
  for benchmarks, because comparing against AlphaFold 3 needs AlphaFold 3's
  cutoff; the release therefore stays the default and this is opt-in. It gets
  its own storage root because its published file is also named `protenix.pt`
  and a shared root would have the second fetch convert over the first, and it
  does not pick up the ESM/ISM staging the mini profiles need, having no
  encoder beside it. Its SHA-256 is FoldJAX's own, recorded from the file at
  that URL on 2026-08-25; the publisher serves no checksum.

- **`foldjax setup` now fetches OpenFold3's weights, leaving AlphaFold 3 as the
  only model anyone has to supply by hand.** The port's `of3_ft3_v1.pt` was
  already declared with the publisher's own URL, size and SHA-256, but was
  marked opt-in and reported as such; the key is served unsigned, so there was
  nothing to opt into. `setup` adds 2.29 GB and one more `ready` line. ESMFold2
  stays opt-in on the reason that field exists for — its full bundle is about
  26.8 GB, not a licence — and AlphaFold 3 stays manual because DeepMind
  releases its parameters only to applicants who accept their terms and they
  may not be redistributed.

- **Template realignment accepts Kalign 3.3.5 *or newer*, and finds the wheel.**
  The check was an exact `== 3.3.5` against a binary named exactly `kalign`.
  Neither Protenix nor AlphaFold 3 asks for a version, no commit here recorded
  why 3.3.5, and PyPI's `kalign-python` wheel publishes 3.4.8 through 3.6.0 and
  nothing older — so the pin made templates reachable only through a system
  package, while a working `kalign-py` sat unused in the same virtualenv. The
  resolver now looks for `kalign-py` as well as `kalign`, reads the `kalign-py
  X.Y.Z` banner its wheel prints, and compares version ordering against a 3.3.5
  floor. An unparsable banner still fails, rather than passing because it did
  not match a string. What is **not** established is that versions above the
  floor align identically; Kalign realigns the query onto the template chain,
  so a different aligner can give different templates and a different
  structure. Pin an exact binary with `PROTENIX_KALIGN_BINARY` when a run has
  to be reproduced against one.

- **`foldjax runtime gc` reclaims abandoned AlphaFold 3 runtime trees.** A
  prepared tree is keyed by the vendored source *and* the interpreter ABI, so
  every edit to a vendored file and every Python move mints a new one and
  abandons the old in place — and nothing collected them. Each is about a
  gigabyte, of which only the compiled extension depends on the key: the rest
  is a ~490 MB `components.cif` and the ~510 MB `ccd.pickle` generated from it.
  The store this was found in held twelve trees and 12 GB, one of them
  reachable. `gc` keeps the live tree always, and by default keeps anything
  prepared in the last 7 days as well, because the store is shared between
  checkouts and a worktree at another commit has its own live tree;
  `--keep-days` moves that guard and `--all` drops it. `--dry-run` reports
  without removing. Removing a tree costs a rebuild rather than data, with one
  caveat worth knowing: libcifpp fetches the current CCD at build time, so a
  rebuild months later is a different chemical dictionary, not the same one.

- **`foldjax setup --all` installs every published checkpoint in one command.**
  Size was the only thing keeping ESMFold2 out of `setup`, and there was no way
  to say "take it too" without naming it; the opt-in line now says the flag
  exists. AlphaFold 3 is not included by any flag, because there is nothing to
  download: its parameters are licensed to the person who requested them.

- **OpenFold3's per-sample confidence is reported differently, and its values
  moved.** Carrying the port's own featurization changed three things a caller
  can see. `mean_plddt` is now on the public 0--100 scale rather than the
  head's [0, 1] output, so a value that read 0.9103 reads 91.15. `iptm` and
  `has_clash` are reported where earlier runs of the same four-chain jobs
  produced neither. And pTM on the same job moves by up to 0.005: measured
  under the benchmark schedule at 3,012 tokens it is 0.87029 where the earlier
  build gave 0.87563, and at 2,096 tokens 0.90455 against 0.90708, both
  reproducible to five decimals across independent runs. Against the recorded
  upstream run the direction depends on the statistic — at 3,012 tokens the
  sample mean is closer than before and the best sample is farther, because
  upstream's own five samples spread 0.0105 there — so this is a different
  computation rather than a more or less accurate one. The new `iptm` lands
  within 0.011 of upstream's. Peak memory and wall time are unaffected by this
  entry; they are covered by the benchmark.

- **ESMFold2 now accepts the complete released all-biomolecule input
  contract.** FoldJAX's Torch-free NumPy adapter now builds protein, DNA, RNA,
  CCD-ligand, SMILES-ligand, modified-residue, and explicit covalent-bond
  features instead of rejecting every non-protein job. Protein-only inputs
  retain their established feature path. Mixed-complex outputs are written
  directly as mmCIF so arbitrary chain identifiers, CCD residue names, and all
  folded atoms survive. The managed released and structure-only profiles now
  include Biohub's pinned 417,306,584-byte `ccd.pkl`; the reader rechecks its
  SHA-256 before deserializing it. RDKit moves to the 2026.03 serialization
  generation used by that snapshot. The complete public download is 26.77 GB;
  the structure-without-ESMC profile is 1.36 GB.

- **FoldJAX now targets Python 3.13 and the current JAX/cuEquivariance runtime
  family.** The package metadata, lockfile, lint target, and CPU CI move from
  Python 3.12 to 3.13. JAX/JAXLIB and both CUDA plugin families move from
  0.10.1 to 0.11.1; cuEquivariance core, JAX, and CUDA ops are aligned at
  0.11.1; Tokamax moves to 0.0.13; AlphaFold 3's Haiku extra moves to 0.0.17;
  and Flax is pinned at 0.12.9 because earlier 0.12 releases resolve but fail at
  import against JAX 0.11.1's HiType API. Qwix 0.1.8 and Triton 3.7.1 complete
  the pinned kernel stack. AlphaFold 3's first-use extension build now passes
  the active Python executable to uv explicitly, preventing a machine with
  multiple installed Python minors from producing the wrong ABI wheel.
  CUDA 12 and CUDA 13 remain conflicting extras so a checkout can serve both
  driver generations through separate environments. This is a runtime support
  migration, not a change to model weights or sampling defaults. JAX 0.11 also
  changed the implementation of 2-by-2 and 3-by-3 determinant operations;
  alignment and rotation paths can therefore differ in low-order floating-point
  bits even with the same inputs. CPU tests cover the base runtime, and locked
  dependency resolution covers both CUDA extras; released-model CUDA numerical
  and kernel-performance checks remain deployment gates.

### Fixed

- **OpenFold3 p1 is now a managed public download instead of being mislabeled
  as access-gated.** Upstream's own p1 release path publishes
  `of3_ft3_v1.pt` from an unsigned S3 bucket. The registry now pins that exact
  2,288,027,095-byte checkpoint and its SHA-256, stages it atomically for direct
  loading, and leaves it opt-in because the preprocessing extra is not part of
  the base runtime. The Colab workflow fetches it automatically when OpenFold3
  is selected and exposes no manual OpenFold3 path field. Current
  upstream OpenBind v0.5 and older p2 checkpoints target newer architectures;
  FoldJAX never substitutes them for its p1 port. AlphaFold 3 remains the only
  carried model whose parameters must be supplied manually.

- **OpenFold3 no longer forces the allocator to grow on demand, which is what
  made long targets fail.** Importing the port set
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` process-wide, documented as stopping a
  large token bucket from failing before it had a chance to run. A cold-cache
  A/B at 3,012 tokens, both arms setting the variable explicitly and both at
  `MEM_FRACTION=0.90`, measured the opposite: with preallocation off the run
  died after 324.3 s asking for 54.34 GiB, having reached a high-water mark of
  only 28.5 GiB; with it on the same command completed in 1,058.4 s at a peak
  of 59.8 GiB and logged no allocator failure. A pool grown on demand is
  assembled piecewise, so it cannot serve one large contiguous request even
  with the card mostly free. Only the fraction is now set, still via
  `setdefault`, so an explicit operator setting continues to win. This also
  re-enables `foldjax.oom`'s pool-based diagnostic for OpenFold3, which
  declines to report whenever preallocation is off. The recorded symptom of a
  3,012-token first invocation failing while a second process succeeded is
  explained by the same mechanism.

### Added

- **Predictions can now be compared in one rigid coordinate frame without
  duplicating their structure files.** `align_predictions` accepts common
  `PredictionResult` objects and `align_structures` accepts named PDB/mmCIF
  paths; either another prediction or an external structure can be the
  reference. The default chemistry-aware Kabsch fit uses protein CA and
  DNA/RNA C4' atoms, carries ligands with the whole complex, refuses to guess
  ambiguous homomer permutations, and reports chain correspondence, RMSD, and
  atom coverage. Its durable representation is only a small JSON transform
  report with source SHA-256 provenance. Transformed mmCIF text or files are
  produced only through explicit `aligned_cif`/`write_aligned` calls, so the
  ordinary multi-model comparison does not double output storage. This is a
  post-processing API and deliberately adds no CLI mode or cross-model score
  ranking.

- **The Google Colab example is now a practical multi-model workflow rather
  than a single-model smoke test.** A setup form selects models, sampling, MSA,
  and storage policy before idempotent installation and checkpoint preparation;
  a later form accepts protein, DNA, RNA, CCD ligands, and SMILES ligands.
  Independent checkboxes expose all six carried models. Its default
  builds one compact synthetic protein+RNA+ATP job and sends that exact
  recorded input to Protenix, OpenDDE, and OpenFold3; model-specific requests
  preserve each backend's actual portable execution vocabulary before their
  reports are combined. ESMFold2
  accepts the same all-biomolecule common schema and its complete public bundle
  is 26.77 GB. OpenFold3 is selected by default and fetches or reuses the exact
  managed public p1 checkpoint; AlphaFold 3 accepts a user-provided parameter
  directory. Their
  optional preprocessing/runtime dependencies install only when selected.
  Parameter readiness and runtime state are checked before public downloads.
  Input entity types and ligand representations are checked immediately after
  the later biomolecule form and before prediction. A matching install marker,
  verified managed-weight state, and finished manifests make repeated Run-all
  execution skip completed work. FoldJAX runs the large sessions sequentially,
  preserves successful models
  when an in-scope failure is configured to continue, validates every emitted
  structure and finite score, and presents a native-score table plus an
  interactive model/sample viewer. A single archive contains the common input,
  complete batch report, comparison CSV, structures, confidence artifacts, and
  reproducibility manifests. Model scores remain separately named and are not
  collapsed into a scientifically unsupported universal ranking. Before first
  use the notebook shows each publisher, licence, download size, cache state,
  and conservative disk estimate, then fetches public checkpoints in sequence.
  Setup, execution, validation, visualization, and export now use consistent
  status cards and styled tables, so folded implementation cells still present
  a readable end-to-end workflow instead of raw diagnostic output.
  It detects the active accelerator, checks the immutable JAX
  0.11.1/cuEquivariance 0.11.1 stack, and rejects silent CPU fallback before
  prediction. Fast mode remains clearly labelled as 1 sample, 20 steps, and 1
  recycle; released defaults are opt-in.
  MSA transfer, Colab/Drive retention, and cached-versus-local storage are
  disclosed where the user makes those choices. Separate toggles persist the
  reusable asset cache and prediction outputs: the former covers weights,
  shared assets, generated runtimes, and sequence-and-search-provenance MSA
  entries, while the latter keeps output trees and validated resume manifests.
  A cache inventory reports each component's file count and logical size. An
  input preflight groups identical chains and reports verified MSA hits, misses,
  disabled search, and unavailable RNA search without displaying sequences;
  cached protein alignments are shared across model runs. Compilation stays on
  the active VM because its identity includes the accelerator and runtime.

- **OpenFold3 projects template residue types before expanding them across token
  pairs.** The template pair embedder previously broadcast each
  `[N_template, N_token, 32]` input to
  `[N_template, N_token, N_token, 32]` and only then applied its two
  projections. It now evaluates the same two linears once per token and lets
  their existing left-associated additions broadcast the projected values over
  the other pair axis. The public call, parameter tree, context-parallel
  identity, dot reduction width, projection order, and addition order are
  unchanged. FP32/BF16, leading-axis, signed-zero, non-finite, dense/compact
  template, and forced four-CPU 1-D/2-D context-parallel oracle tests retain
  exact bytes on the dynamic inference path. A locked-environment CPU probe of
  the projection and its two left-associated additions used an
  `[1, 1, 256, 256, 64]` accumulator, `[1, 1, 256, 32]` residue types, and two
  dynamic `[64, 32]` weights. Output bytes were identical while compiled
  temporary memory fell from 32.0 to 0.125 MiB in FP32 (33,554,432 to 131,072
  bytes) and from 32.008 to 0.102 MiB in BF16 (33,562,624 to 106,496 bytes);
  argument and output sizes were unchanged. Manually closing both arbitrary
  residue types and weights into a JIT can instead make XLA constant-fold the
  two forms differently and produce small FP32 deltas; supported one-hot
  direct/full closures and the production dynamic route were byte-exact. These
  are synthetic CPU stage measurements, not released-weight end-to-end GPU
  numerical, latency, or whole-model peak-memory evidence.

- **OpenDDE stops carrying the raw alignment after its recycle samples are
  complete.** The CLI's exact/default path already sampled every recycle to at
  most 1,280 rows, but still passed the much deeper source `msa`,
  `has_deletion`, `deletion_value`, and optional `msa_mask` through preflight,
  JIT identity, and context-parallel placement even though the trunk reads only
  the sampled cycles. It now removes only those source fields before model
  inference when every sampled cycle has all four non-empty, shape-consistent
  arrays. Missing, empty, incomplete, or malformed cycles keep the original
  mapping and direct low-level model APIs are unchanged; `profile`,
  `deletion_mean`, constraints, templates, writer data, and custom metadata are
  preserved. An actual two-cycle CPU Pairformer probe with non-finite raw
  deletion sentinels produced byte-identical outputs and byte-identical
  StableHLO in the matched probe, while changing source depth caused three
  traces before pruning and one after.
  A forced four-CPU-device placement probe at 256 tokens and 4,096 source rows
  measured 48 MiB of replicated source storage across the mesh for the common
  three-array source. At the nominal 16,384-row source maximum and 1,531
  tokens, those three arrays at the default device dtypes are 287.1 MiB per
  device, or 1.12 GiB on four devices. The latter is a shape-and-dtype
  extrapolation, not an observed peak; released-weight end-to-end GPU numerical
  parity, latency, and peak memory remain deployment gates.

- **ESMFold2 does not return a native distogram that the common backend never
  reads.** The direct model and inference APIs still return
  `distogram_logits` by default for parity and debugging, but the common
  backend now gives both its exact-shape and padded/session graphs a separate
  static output choice. Its structure writer, confidence summary, normalized
  result, raw metadata and representation exporter consume none of this
  `[batch, tokens, tokens, 64]` float32 array, so the backend graph omits both
  the entry output and its independent symmetric-pair projection. The choice
  is part of the in-process JIT identity and is not exposed as a sampling
  override. At one batch and 1,003 tokens the removed entry output is
  257,538,304 bytes (245.6 MiB); at 3,012 it is 2,322,468,864 bytes
  (2.16 GiB). In an isolated
  256-token, 256-pair-channel CPU stage, output memory fell from 16,778,256 to
  1,024 bytes, temporary memory from 83,886,080 to zero, reported floating
  point work from 2,168,455,168 operations to none, the StableHLO dot count
  from one to zero, and warm median time from 32.07 to 0.005 ms. The retained
  carry was byte-identical. These are synthetic CPU stage measurements;
  released-weight end-to-end GPU latency and whole-model peak memory remain
  deployment measurements.

- **ESMFold2 replaces a proven zero token-bond projection with its compact
  signed-zero vector.** The supported protein featurizer supplies an all-`+0`
  `[batch, tokens, tokens, 1]` feature, and the released projection is a finite
  `[256, 1]` weight with no bias, but the structure graph still projected that
  input to 256 pair channels and added the zero result before the trunk and
  again in the confidence head. The inference wrapper now verifies the full
  host shape and values plus the compatible post-cast finite, unbiased weight
  and routes that proof into the compiled-function identity. The compact graph
  derives a `[256]` zero vector with each compute-cast weight channel's sign
  and keeps both original left-associated broadcast additions; simply dropping
  them is not bitwise exact for signed zero. Nonzero, negative-zero, malformed,
  non-finite, bias-bearing, integer-weight, incompatible-dtype, and unverified
  direct/custom inputs retain the generic projection. FP32/BF16 raw bytes,
  supported source-to-compute cast edges, post-cast overflow fallback, padding,
  context-parallel routing, and the two cache identities are regression-gated.
  In an isolated 256-token BF16 CPU pair stage that reuses the encoding on both
  sides of a four-step scan, outputs were byte-identical, the StableHLO dot
  count fell from one to zero, compiled temporary memory fell from 67,108,932
  to 33,554,500 bytes, and floating-point work reported by XLA fell from
  285,344,000 to 184,551,936 operations. These are synthetic CPU stage
  measurements; released-weight end-to-end GPU numerical parity, latency, and
  whole-model peak memory remain deployment gates.

- **OpenFold3 drops the released checkpoint's redundant diffusion registration
  before inference mapping.** Upstream registers one denoiser under both
  `diffusion_module.*` and `sample_diffusion.*`; Torch preserves their shared
  storage, while the current restricted NumPy reader materializes both.
  After the full checkpoint is loaded and its model prefix is resolved, the
  common-backend and standalone-predict inference paths now remove the sampler
  group only when its key set and every contiguous tensor byte exactly match
  the canonical group. Partial, non-matching, nested-unrelated, and
  inspection/verification loads remain complete. In the released checkpoint
  this removes 767 arrays totaling 812,960,240 bytes before host prestacking
  and device transfer. Across two fresh CPU processes, maximum RSS fell from
  5,222,868--5,417,916 KiB to 4,646,056--4,662,020 KiB; mapping time was
  0.37--0.38 s in both arms, while a fresh-process run of the final
  conservative byte check took 0.13 s. The resulting 593-leaf,
  1,473,188,784-byte mapped tree was bitwise identical. This does not reduce
  the restricted reader's own earlier deserialization peak, and
  released-weight GPU latency and peak memory were not measured.

- **Boltz-2 drops training and writer-only features before non-steering
  inference owns them.** The model-input allowlist previously ran only for
  explicit neutral padding. Exact-shape, legacy-bucket, and context-parallel
  calls therefore retained the complete featurizer mapping; context
  parallelism also placed every array before JIT could discard unread inputs.
  A 1,003-token probe found 47 JIT-dead arrays totaling 306 MiB, including the
  246 MiB quadratic `disto_target` training label. The conservative host
  allowlist now removes training and writer-only inputs such as that label
  before the exact, bucketed, context-parallel, and always-jitted affinity
  routes take ownership, while retaining conditional model inputs. Active
  steering still receives its variable-length constraint arrays, padding keeps
  its template-row validation, and public featurization plus the low-level
  model API remain unchanged. A minimal CPU DCE probe produces identical HLO,
  executable memory analysis, outputs, and non-finite masks before and after
  filtering; a forced four-CPU mesh also gates that removed features never
  reach placement. No model arithmetic, static runtime identity, or
  compilation profile changes.

- **Protenix and OpenDDE confidence project released distance bins without a
  quadratic one-hot.** Their shared head previously materialized
  `[N_token, N_token, 39]` float indicators before a linear whose input has at
  most one nonzero. Production compiled wrappers now validate the concrete
  checkpoint's finite, strictly ordered, exactly adjacent open intervals and
  finite projection before selecting the same weight row directly; direct API
  calls retain the dense default, while custom parameters with overlapping,
  gapped, malformed, or non-finite layouts fall back to it. Canonical finite,
  adjacent custom layouts can use the compact path. Exact edges, out-of-range
  and non-finite distances,
  signed zero, FP32/BF16, both CP layouts, and wrapper cache identity are
  regression-gated. With released weights, isolated full distance-embedding
  CPU temporary memory fell from 519,100,644 to 24,144,488 bytes at 1,003
  Protenix tokens, from 4,681,226,304 to 217,731,560 bytes at 3,012, and from
  366,741,760 to 5,715,560 bytes at 488 OpenDDE tokens; output memory was
  unchanged, and a 256-token Protenix warm median fell from 9.32 to 4.04 ms.
  Full released confidence-head outputs were bitwise identical on CPU at the
  default JAX matmul precision, but its total arena remains
  Pairformer-dominated. These are CPU confidence-stage
  measurements; released-weight end-to-end GPU numerical parity, latency, and
  peak memory remain deployment gates.

- **OpenFold3 confidence projects distance bins without a quadratic one-hot.**
  The confidence re-embedding previously materialized
  `[samples, N_token, N_token, 39]` float distance indicators only to pass each
  one-hot row through a bias-free linear. It now locates the same strict bin and
  selects that projection-weight row directly. Exact bin edges and distances
  outside the open final bound still select no row; coordinate NaN/Inf,
  arbitrary NaN/Inf weights, and negative-zero weights retain the dense dot's
  finite and non-finite results. At 3,012 tokens in OpenFold3's released FP32
  path, the removed one-hot is 1.32 GiB per confidence sample. In an isolated
  512-token, 128-output FP32 CPU projection,
  compiled temporary memory fell from 40,894,620 to 6,292,456 bytes while the
  134,217,728-byte output and 26,112 argument bytes were unchanged, and the
  39-wide StableHLO dot was removed. This is a synthetic confidence-stage CPU
  measurement; released-weight end-to-end GPU latency and peak memory remain
  deployment gates.

- **Boltz-2 confidence constructs ligand frames from representative atom rows.**
  The inference head previously formed every atom-by-atom distance row and then
  gathered the one row selected by each token's existing representative-atom
  `argmax`. It now gathers those representative coordinates, masks, and
  atom-derived chain IDs first, while retaining the full atom-axis stable sort,
  `[:3]` selection, and `[1, 0, 2]` frame order. Dense-reference tests keep
  representative ties and all-zero rows, malformed low-level mappings, padded
  atoms, batch/sample axes, and non-finite coordinate behaviour bitwise equal.
  In an isolated three-sample, 128-token, 1,024-atom CPU frame-selection probe,
  compiled temporary memory fell from 25,166,416 to 3,147,344 bytes and warm
  median time from 197.1 to 25.2 ms. These are confidence-frame stage CPU
  measurements; released-weight GPU latency and whole-model peak memory remain
  deployment gates.

- **Padding no longer rebuilds a 200-leaf diffusion-noise tuple inside the
  scanned OpenDDE and Protenix samplers.** Both padding producers now preserve
  the historical per-key draws, sample-chunk order, and real-atom prefix while
  carrying step noise as one `[steps, samples, atoms, 3]` array. The scan reads
  that array directly; direct tuple/list callers still take the historical
  stack path, and calls without an explicit tape retain the historical RNG
  path. In an isolated CPU sampler probe at 200 steps, 5 samples, and 32 atoms,
  the 384,000 input bytes and outputs were unchanged bitwise. OpenDDE StableHLO
  text fell from 117,452 to 82,930 bytes and compiled temporary memory from
  386,392 to 2,392 bytes; Protenix fell from 42,190 to 7,672 bytes and from
  384,260 to 260 bytes. These are synthetic sampler-only CPU measurements;
  released-weight end-to-end GPU latency and peak memory remain deployment
  gates.

- **OpenFold3 carries template-free pair geometry as one verified zero scalar.**
  Duplicate empty templates were already collapsed to one row, but that row
  still carried four exact-zero geometric inputs through the entry boundary,
  led by `[N_token, N_token, 39]` float32 distogram storage. Production now
  verifies the final post-padding host arrays are bitwise positive zero before
  replacing them with a private scalar marker; nonzero, non-finite,
  negative-zero, differently typed, malformed, and direct low-level inputs
  retain the dense path. The compact graph evaluates every bias-free zero
  projection once and broadcasts it in the historical add order, rather than
  deleting the terms, so `0 * NaN/Inf` parameter behaviour is retained. The
  removed entry data is 161.2 MiB at 1,003 tokens and 1.42 GiB at 3,012 tokens.
  In an isolated 128-token, 8-channel CPU pair-embedder probe, feature bytes
  including `asym_id` fell from 2,770,432 to 16,900, StableHLO text from 10,222
  to 7,983 bytes, compiled argument memory from 3,298,496 to 544,452 bytes, and
  compiled temporary memory from 5,308,416 to 4,784,128 bytes. Full-model
  released-weight GPU latency and peak memory remain deployment gates.

- **Large float32 CPU transitions compile as one bounded loop.** Protenix and
  OpenDDE historically emitted one independent transition subgraph per leading
  chunk, which let CPU XLA keep several widened blocks live together and made
  StableHLO grow with sequence length. Serial CPU float32 transitions with at
  least four chunks now use one `lax.map`; shorter block lists, reduced
  precision, GPU/TPU, and both context-parallel layouts retain the historical
  graph. The choice follows the actual JAX lowering target, including explicit
  JIT devices/backends and portable export, rather than the process default.
  At OpenDDE's released transition widths and 946 structural tokens, a CPU
  compile-only probe reduced StableHLO text from 27,211 to 13,709 bytes,
  seven-fresh-process median lower-and-compile time from 152.3 to 110.9 ms,
  and compiled temporary memory from 5,243.6 to 2,804.8 MiB. The loop can
  retile full float32 chunks and is not bitwise-identical: the measured
  max-norm relative difference
  (`max(abs(delta)) / max(abs(reference))`) was 6.4e-7;
  non-finite classifications and the separately evaluated short tail were
  unchanged. These are CPU compiler measurements only; released-weight GPU
  latency and memory remain deployment gates.

- **Protenix carries its relative-position one-hot compactly without changing
  trunk or diffusion arithmetic.** The `[N_token, N_token, 139]` feature contains
  only exact zeroes and ones, so its host/device representation is now int8
  (4.70 GiB to 1.17 GiB at 3,012 tokens). Promotion through the projection
  weights preserves the historical split: bfloat16 in the default trunk and
  float32 in diffusion. A compile-only GPU probe at 1,003 tokens reduced total
  reported memory from 1,056.7 to 624.6 MiB; released-weight peak memory remains
  a deployment gate, and the CPU compiler may materialize the widening
  conversion instead of fusing it.

- **OpenDDE warns before a job that probably will not fit.** OpenDDE ships
  float32, and float32 has a wall near 1,400 residues on a 96 GB card that no
  backend choice rescues -- at 1,531 residues the fused path asks 93.40 GiB and
  the blocked path 80.72, and both die. The failure presented as a half-hour
  hang, because the allocator retries every ten seconds before giving up. A
  preflight now estimates the arena from the realized trunk dtype and the
  structural token count, compares it against the pool the allocator took, and
  warns -- naming `--trunk-dtype bf16` as the lever and saying plainly that it
  is an opt-in divergence from upstream's float32 storage. It never refuses,
  it is silent for every job that fits, and it is silent whenever it cannot
  establish the pool, the token count or the dtype. The message says the
  estimate is not a guarantee in either direction, because the practical
  ceiling sits below the arithmetic one.

- **A requested triangle-multiplication chunk size now says when it is
  ignored.** The fused kernel has no parameter of that family, so the option
  was silently inert whenever it ran -- and unlike the attention path, where
  the fused kernel never builds the tensor a chunk size would bound, this one
  does build it: 5,299 MiB apiece on OpenDDE's structural branch at 1,003
  tokens. The warning names that size and points at the backend that honours
  the request.

- **OpenFold3 reuses a bounded, correctly partitioned in-process JIT pool across
  repeated factories.** Multi-seed/backend calls with the same graph choices
  now share a trace instead of constructing a new `jax.jit` cache each time,
  while an eight-executable LRU bound prevents plural unpadded workloads from
  retaining every program shape for the lifetime of the process. The static
  identity includes configuration, chain count, resolved CP layout and ordered
  devices, effective triangle kernel, RNG route, representations/stop boundary,
  and persistent-cache scope; the representative-atom table stays validated
  dynamic data. Deleted or replaced persistent-cache entries force repopulation,
  and `lower().compile()` retains the public three-argument callable plus its CP
  placement. Tiny serial and forced four-device CPU tests cover exact parity,
  trace reuse and eviction, distinct 1-D/2-D HLO, cache repair and partitioning.
  Released-weight GPU compile latency and memory remain deployment gates, so no
  GPU speedup is claimed here.
- **OpenFold3 omits atomized-frame neighbour search for polymer-only inputs.**
  Protein, RNA and DNA frame atoms come from their named atoms, but the generic
  confidence path also built sample-by-token-by-atom distances and a `top_k`
  for atomized ligands and modified residues even when no active token needed
  it. Production entry points now derive that graph choice from masked host
  features and carry it in the compilation identity; direct callers retain the
  conservative generic default. A three-sample, 128-token, 1,024-atom CPU
  stage probe kept frame positions and validity bitwise equal while reducing
  compiled temporary memory from 3,214,032 to 63,888 bytes, warm median time
  from 58.0 to 1.17 ms, and `top_k` operations from one to zero. These are
  confidence-frame stage measurements; released-weight full-model GPU latency
  and peak memory remain deployment gates.
- **OpenFold3 omits aggregate interface-TM work for known monomers.** Active
  chain IDs are now normalized to an explicit static count, so production
  monomer graphs return the mathematically identical all-zero ipTM vector
  without tracing its interface mask and reductions. Direct callers that leave
  the chain count unknown retain the generic path. In a five-sample, 128-token
  optimized CPU graph, HLO text fell from 33,067 to 22,033 bytes, fusions from
  25 to 18, and reductions from nine to six; XLA had already shared the pair
  softmax, so no full-model latency or peak-memory improvement is claimed.
- **ESMFold2 keeps one verified model and one input's ESMC states for a
  request.** A multi-seed or multi-input batch used to construct a fresh
  backend for every scalar run, reload the 26,347,754,143-byte released bundle,
  and repeat the seed-independent ESMC-6B forward. Backends can now opt into a
  single-threaded, request-scoped session; ESMFold2 loads lazily, retains at
  most one input's final scattered language-model state, and releases both
  model and state before the next model group. At 1,000 tokens that retained
  BF16 state is 414,720,000 bytes, against 25,408,148,888 source bytes for the
  ESMC checkpoint. An all-resumed request loads no weights.

  Reuse is deliberately narrower than path equality. The session binds every
  structure-model file, ESMC config/index/shard, device placement and the exact
  normalized feature bytes; it refuses to combine seeds if those assets change
  during resume, load, prediction or manifest writing. Backends that do not opt
  in keep the historical fresh-instance-per-scalar behavior. CPU contract tests
  cover two inputs by three seeds, lazy resume, checkpoint replacement,
  continued failures, cleanup and wrapper compatibility. Released-weight
  CUDA latency, peak-memory and numerical parity remain deployment gates, so
  no end-to-end GPU speedup is claimed here.
- **ESMFold2 casts checkpoint leaves before transferring them.** The released
  structure and ESMC bundles contain 1,594 tensors across 49 shapes and 802
  selected tensors across six shapes, respectively. Loading them through
  `jnp.asarray` and then casting ESMC on device created a shape-specialized
  staging program for the structure leaves and both staging and conversion
  programs for ESMC. The loader now casts the published float32 ESMC leaves to
  the requested dtype on the host and transfers both bundles with `device_put`.
  This preserves the historical values exactly, removes all of those loader
  compilations, and avoids an extra full-width device copy of the current ESMC
  leaf (up to 141,557,760 bytes in the released checkpoint). In a separate
  128 MiB CPU transfer probe the bfloat16 path's maximum RSS fell from 909,652
  to 598,180 KiB (34.2%); released-checkpoint GPU peak memory remains a
  deployment gate.
- **ESMFold2 confidence gathers each token's contiguous atom scores instead of
  materializing an atom-by-token one-hot matrix.** The normalized production
  featurizer guarantees a masked suffix, sorted owners, and at most 23 atoms
  per token, so the pLDDT head can use a fixed-order streaming gather with no
  colliding scatter-add. Direct low-level confidence calls retain the generic
  dense fallback. In a one-batch, 8,256-atom, 1,024-token float32 CPU stage
  probe, compiled temporary memory fell from 33,857,792 to 103,912 bytes and
  warm median time from 6.230 to 0.046 ms. At the released 32-sample,
  7,776-atom, 1,003-token shape, the avoided float32 owner matrix alone is
  998,313,984 bytes. Bfloat16 was bitwise equal in focused probes; float32
  per-token means differed by at most two unit-scale epsilons, and historical
  NaN/infinity propagation is preserved. Released-weight GPU latency and
  whole-model peak memory remain deployment gates.
- **Protenix and OpenDDE build local atom windows with one indexed gather.**
  Their shared encoder, decoder, and local-attention paths previously cloned
  one static slice into the JAX graph for every query block and concatenated
  those slices. One axis-aware primitive now produces the identical overlapping
  windows, including bfloat16 and non-finite values, without changing masks or
  attention arithmetic. In a 2,048-atom, 4-head, 64-channel CPU local-attention
  graph, outputs were bitwise equal while StableHLO text fell from 172,987 to
  142,929 bytes, compile time from 125.3 to 46.8 ms, and warm median time from
  1.380 to 1.327 ms. The isolated window extraction's 8,388,608-byte temporary
  disappeared, although the complete graph's 10,485,760-byte temporary arena
  was unchanged. Released-weight GPU compile, latency, and whole-model peak
  memory remain deployment gates.
- **AlphaFold 3 reuses one managed ModelRunner inside a request.** Multi-seed
  and multi-input managed runs now load the selected parameter generation and
  construct its shape-keyed JIT owner once, while scalar runs and explicit
  external `source=` checkouts preserve the fresh-run path. Reuse is lazy, so
  an entirely resumed request does not import the runner or load parameters.
  Weight files, the vendored runner and vendored model source are anchored in
  both the live session and run manifests; a partial resume cannot combine
  seeds produced by different generations. CPU mock tests cover reuse,
  cleanup, cache/config/device splits, mutation, resume and cancellation.
  Released-weight GPU latency, memory and numerical parity remain deployment
  gates, so no production speedup is claimed here.
- **AlphaFold 3 broadcasts its host PAE frame mask instead of copying its
  square.** The postprocessing path only reads this mask, so a zero-stride
  NumPy view preserves its shape, dtype, values, and downstream pTM/PAE metrics
  exactly. At 3,072 tokens this removes a 9,437,184-byte square data buffer,
  replacing it with an O(1) NumPy view. This is a host postprocessing
  allocation reduction and does not change model predictions.
- **AlphaFold 3 avoids repeated full-square host copies while summarizing a
  monomer's PAE.** The monomer path now builds the same final advanced-index
  buffer directly instead of first selecting both square axes separately, so
  NumPy keeps the historical reduction order and bitwise result. For five
  samples at 2,048 tokens, an isolated float32-contact CPU stage fell from
  692.9 to 171.6 ms and from 507,546,480 to 302,025,716 peak traced bytes.
  Multimer processing is unchanged.
- **AlphaFold 3 ranks NumPy samples without retaining their products
  together.** C-contiguous multi-sample PDE results are reduced one sample at
  a time; unusual non-C direct inputs and the JAX path retain the original
  vectorized expression. For an isolated five-sample, 2,048-token CPU stage
  with float32 contact probabilities, peak traced bytes fell from 83,920,368
  to 16,811,696 and time from 19.1 to 13.5 ms, with bitwise-identical scores.
- **AlphaFold 3 skips host interface-TM scoring when no interface exists.**
  Monomer ipTM is defined by the existing implementation as a same-shaped NaN
  vector, so postprocessing now emits that value after pTM validation instead
  of constructing and reducing an all-false pair mask for every sample.
  Multimer scoring is unchanged. For five samples at 3,072 tokens, an isolated
  CPU host-stage probe reduced median time from 23.938 to 0.022 ms and peak
  traced allocation from 9,569,504 to 19,231 bytes. This does not affect model
  compilation, device memory, or forward-pass numerics.
- **Boltz-2 reuses its verified parameters and JIT owners inside a request.**
  Multi-seed and multi-input runs now load the confidence checkpoint once and
  retain one bounded compiled owner. Consecutive affinity runs do the same for
  the complete affinity model; that optional tree is released before a later
  primary-only input. Reuse is lazy, so an all-resumed request does not import
  the native runtime or load either checkpoint. The session
  binds the loader's selected `.safetensors`/`.npz` file, scalar sidecar and
  sidecar absence, device/CP topology, cache namespace and every trace-time
  model option, including JAX dtype/PRNG mode and the effective triangle
  multiplication backend. A changed or newly preferred checkpoint generation
  poisons the session instead of mixing seeds, while a continued ordinary
  failure drops params and executables before the next seed. Each role retains
  at most eight compiled shapes and all state is released at the model-group
  boundary. CPU JAX tests cover primary and affinity load/trace reuse, resume
  without runtime import, mutation, continued failure and executable eviction.
  Released-weight CUDA latency, peak memory and numerical parity remain
  deployment gates, so no GPU speedup is claimed here.
- **Boltz-2 drops the flat host checkpoint mapping before layer stacking.** The
  nested parameter tree already owns the same NumPy leaves; retaining the flat
  dictionary kept the unstacked per-layer arrays alive while their stacked
  replacements were built and transferred. On the local released confidence
  checkpoint, the loader's maximum RSS fell from 6,687,764 KiB to 4,848,852 KiB
  (1,838,912 KiB, 27.5%) while the resulting parameter tree remained
  2,026,900,352 bytes and value-identical. A lifetime regression test verifies
  that the flat owner is gone before every pre-stack operation. The final host
  transfer now uses `device_put` as well: a separate 12-shape CPU probe built
  12 `jit(stage)` programs through `jnp.asarray` and zero through the direct
  transfer. Float64, float32, int64, int32 and boolean leaves retained the same
  values and JAX-canonical dtypes.
- **OpenFold3 repeated-layer weights are stacked before device transfer.**
  The released checkpoint contains seven homogeneous scanned stacks whose
  parameter leaves total 1,444,290,304 bytes (1.35 GiB). They previously
  arrived at `jit` as one argument per layer, so XLA emitted `concatenate`
  copies of the weights into its temporary arena. Mapping now keeps checkpoint
  arrays on the host, stacks each homogeneous sequence once, and transfers only
  the stacked tree. The in-graph copies really do go: a 1,003-token lowering
  carries 18 concatenate operations with pre-stacking against 491 without it,
  436 of them parameter-shaped. What that does *not* buy is peak memory. At
  1,003 and 3,012 tokens the compiled arena, arguments and output are identical
  to four decimals either way, and eight measured runs all peaked at 9,997 MiB
  -- at real sizes the peak is set elsewhere, and a copy removed underneath it
  does not move it. Both statements hold at once; the earlier "drops a full
  stacked copy from temporary memory", measured on a 4-layer CPU compilation,
  is true there and does not surface at model scale.

  What it does buy is time: about 1.5 s of a 21.6 s warm run at 1,003 tokens,
  roughly 7%, with the arms disjoint and the slower arm slower in both orders
  including when it ran first on a cooler card. The mechanism is unproven -- a
  single 1.35 GiB copy is milliseconds at HBM bandwidth, so what fits is the
  24-block diffusion transformer being re-stacked inside the sampling loop and
  paid once per rollout step. Separately, it costs about 0.16 s per process on
  the host at checkpoint mapping; that is a different machine and a different
  frequency from the device saving and the two should not be netted. The
  heterogeneous four-block MSA path and the two unscanned
  conditioning-transition loops keep their original layouts.
- **AlphaFold 3 parameters follow the explicitly selected device.** A run on
  device 1 used to load the 1,146,811,260-byte checkpoint on JAX's implicit
  device 0 before the device-pinned model copied it across. Native prediction
  now runs inside the selected default-device context, so lazy parameter loads
  and other unpinned allocations start on the requested device. A forced
  two-device probe places all 405 checkpoint array leaves directly on device 1.
- **`confidence_sample_sequential` for ESMFold2**, **on by default** since the
  released 32-sample configuration cannot run without it (see Fixed). Runs the
  confidence head one structure at a time instead of batching every sample
  through it. This model's peak is a single temp arena sized
  `num_samples * L^2 * 4*c_z`, and that head is where the sample factor enters
  it: at 1,003 tokens with five samples on a 96 GB card, peak falls from 56.0
  to 30.2 GiB and the arena from 35.99 to 11.31, with warm time unchanged at
  28.3-28.6 s either way. Coordinates are bit-identical across all five
  samples; the only difference a user could see is up to 1e-5 of pLDDT, which
  a float32 control collapses to 3.6e-7, so it is bfloat16 reduction-order
  rounding from the changed matmul shapes rather than anything about sample
  independence -- and it was measured on one target at one seed, so it is not
  a bound.

  Read the 45% as a property of that measurement and not of the option: the
  arena is quadratic and this divides only the head's share of it, so the
  fraction moves with length and sample count. It does not raise the size the
  model can reach, and that is now a named tensor rather than an inference --
  at 2,096 tokens the allocator asks for `bf16[1, 2096, 2096, 2048]` on top of
  64.6 GiB already resident and never gets it. Dividing a quadratic by 3.18
  buys its square root in reachable length, about 1.8x, which was not enough.
  Protenix carries the same switch and defaults it on; whether this one should
  follow is a decision about that 1e-5 rather than about the memory.
- **OpenFold3 compiles 3 programs before predicting instead of 104.** Every
  `jnp.asarray` on a NumPy array is a traced operation, so it compiles one
  `jit(stage)` program per distinct shape and dtype -- and none of them is
  slow enough to clear `jax_persistent_cache_min_compile_time_secs`, which
  FoldJAX leaves at JAX's one-second default, so they are compiled again on
  every process start rather than served from the persistent cache. Loading
  weights accounted for 86 of them and the CLI's feature dictionary for 15.
  Mapping the checkpoint now uses `jax.device_put`, which transfers rather than
  traces, and the CLI hands its features to the jitted call as NumPy, placing
  them only under `--no-compile`, which dispatches per operation and would
  otherwise re-transfer. Around 0.55 s comes off every process start.

  Read that as startup latency and not as throughput: the cost is per distinct
  shape, not per byte, so it is a fixed half-second -- about 8% of a 76-token
  job and about 0.25% of a 3,012-token one. The parameter tree hashes
  identically to before, dtype canonicalisation is unchanged, and the arrays
  stay uncommitted with the sharding they had, so the context-parallel path is
  untouched. The lowered StableHLO of the CLI's graph is identical with placed
  and with NumPy features, and both entry points write byte-identical outputs.
- **A query with no templates no longer runs the template stack four times.**
  Upstream's released width is four templates, so a query without any is
  featurized as four *identical* empty ones -- the same all-gap restype, the
  same zero distogram, masks and unit vectors. The stack runs once per row and
  averages, and nothing downstream distinguishes rows, so it computed one
  tensor four times and divided the sum by four. Collapsing runs that are all
  equal to a single row saves 484 MiB of entry arguments at 1,003 tokens and
  4.26 GiB at 3,012, and drops three of four template-stack evaluations in
  every recycling cycle: peak falls 415 MiB at 1,003 tokens with no time
  difference the measurement resolves, and 6.3 GiB with 16.3 s -- 10.4% and
  6.8% -- at 3,012, the latter confirmed with the arms run in both orders so a
  warming card cannot explain it. This is a divergence from upstream in work
  and not in value: with `n` identical rows the reduction is `n * x / n`, and
  the written structures and confidences come out byte-identical at both sizes,
  including a four-chain complex, against a rerun floor of exactly zero. It
  declines to collapse when any row differs and when the padding mask is not
  all-active, and the padded path was checked too -- a collapsed feature padded
  back to four rows carries a mask of `[1, 0, 0, 0]`, so the denominator stays
  one.
- **The MSA outer product carries a byte ceiling, like every other widening
  operation here.** `outer_product_mean` widens `[M, N, C]` into `[N, N, C, C]`
  before `linear_out` narrows it, and at 1,003 tokens that intermediate is
  1,965 MiB and exists twice. It was the one widening operation without a
  ceiling of its own -- the transitions and the triangle projections both have
  one -- and it blocked only when the chunk policy handed it a size, which
  upstream's table does not do at or below 1,024 tokens. So the sizes where
  this tensor dominated were exactly the sizes that built it whole. The
  program's buffers at or above 512 MiB fall from 15 totalling 16.30 GiB to 12
  totalling 10.55 GiB, and measured peak at 1,003 tokens falls 8.25 to 7.79 GiB
  with warm time unchanged at 57.5 s. The chunk table is untouched: a policy
  asking for a smaller block still gets it, and `chunk_size=0` still builds it
  whole.

  The blocks are sequenced by a loop rather than emitted as a list, and that is
  the saving rather than a style choice -- measured, the list form leaves 1,281
  MiB simultaneously live against the loop's 519 MiB, although both name the
  same tensors. Blocking is bit-identical to the dense form in float64 across
  60 shape-and-block pairs including non-multiples and the clamped last block;
  in float32 it differs by 1-2 ulp from GEMM re-tiling, which moves a
  1,003-token structure 0.119 Å against a 0.197 Å floor between two runs of the
  unchanged code. Under a context-parallel mesh the ceiling is deliberately
  inert: turning it on there halves the widest per-device buffer and doubles the
  collective traffic, a trade an interconnect has to settle.
- **Features the graph never reads are no longer copied to the device.**
  Boltz-2's featurizer produces 78 arrays and the compiled program reads 31.
  `jax.jit` already dropped the other 47 from the program -- and drops them
  before the rest are transferred -- but the call site placed the whole
  dictionary first, so those buffers existed for the whole run anyway: 306 MiB
  of them at 1,003 tokens and 1,326 MiB at 2,096, since the term is quadratic
  in token count. `disto_target`, a training label of `f32[N, N, 1, 64]` that
  no inference path reads, was 246 MiB of it on its own. Placing them also
  compiled one transfer program per array -- 142 of the 143 programs a cold run
  built. The compiled path now hands the featurizer's NumPy arrays straight to
  `jax.jit`; the two callers that cannot use the pruning, eager steering and
  context parallelism, still place theirs. The lowered StableHLO of the whole
  graph hashes identically either way, so no prediction changes.
- **A context-parallel mesh proves it moves data before anything runs on it.**
  Entering `--cp-devices P` now shards a four-mebibyte array whose rows carry
  their own index, asks for it back replicated, and refuses to continue if the
  rows another device owned did not come home. A device count is not an
  interconnect: on one tested node every context-parallel run compiled, ran to
  completion, and returned finite numbers with one shard's rows correct and the
  rest zero, while `jax.device_count()` reported four the whole time. The check
  costs about twelve milliseconds after the first mesh of a given shape, and
  `FOLDJAX_SKIP_MESH_CHECK=1` turns it off for anyone who would rather have the
  silence.
- **Context parallelism (`--cp-devices P` / `cp_devices`).** Five backends --
  OpenDDE, Protenix, Boltz-2, OpenFold3, and ESMFold2 -- can shard their pair
  representations across `P` local JAX devices, the JAX form of OpenDDE
  upstream's Fold-CP `1 x P` mode. One process, no launcher: each consolidated
  graph carries `with_sharding_constraint` row shardings and the SPMD
  partitioner places the collectives; only triangle attention routes through
  `shard_map`, because its blocked row loop iterates the sharded axis
  (ESMFold2 has no triangle attention and needs no `shard_map` at all). Like
  upstream's mode it runs the XLA triangle kernels in CP mode -- the fused
  cueq/tokamax kernels are FFI calls the partitioner cannot split, and asking
  for one fails loudly; single-device runs keep their fused defaults
  unchanged. Verified numerically against the single-device programs on a
  forced four-device CPU mesh: module-level pair-stack parity for every
  covered backend at a token count indivisible by the mesh, plus real-weight
  end-to-end runs where local weights exist (OpenDDE 6.1e-4 Å, Protenix
  7.3e-4 Å max coordinate diff at fp32; Boltz-2 sits inside its own
  documented fp32 reduction-order floor, so its CP gate is the confidence
  summaries, which agree). Boltz-2 rejects steering and affinity jobs under
  CP for now. AlphaFold 3 is excluded: its vendored upstream implementation
  is authoritative and runs a Triton attention kernel the partitioner cannot
  split, and it is the least memory-bound of the six. Measured on real
  hardware (4x RTX A5000 24 GB, Slurm): a 1,003-residue OpenDDE job that
  OOMs one 24 GB card completes at CP=4 with balanced per-device peaks of
  11.7 GiB (bf16) / 18.0 GiB (fp32) -- 2.53x below the 45.5 GiB single-device
  fp32 peak, the gap to 4x being the triangle-mult all-gather that a future
  ring pass reclaims. Multi-GPU runs need `NCCL_P2P_DISABLE=1` on the tested
  nodes, and homogeneous cards (a mixed H100/RTX mesh is refused by XLA).
  **Multi-GPU context parallelism is not usable, and no result from it should
  be trusted without a single-device run beside it.** Measured 2026-08-20 on
  four RTX A5000s: sixteen context-parallel runs of one fixed Boltz-2 command
  -- a 254-residue target with its alignment, at the released sampling
  defaults -- returned sixteen structures 34-41 Å from the single-device
  answer, against a 0.36-0.60 Å floor between reruns of the single-device
  command itself, with complex pLDDT 0.96 falling to 0.09-0.44. Both layouts
  fail, with and without an alignment, at two diffusion steps as at two
  hundred, at one recycle as at three, and the trunk representations the same
  runs capture differ from the single-device ones by about the magnitude of
  the arrays. OpenFold3 fails the same way, so it is not one port's mistake.
  Every one of those outputs was finite.

  It is intermittent and the rate depends on the node: four RTX 6000 Ada cards
  gave three good runs and three ruined ones out of six for Boltz-2, and three
  good and one ruined out of four for OpenFold3, while the A5000 node ruined
  everything. The ruined runs agree with *each other* rather than scattering,
  so this is a deterministic wrong computation that is sometimes selected, not
  noise.

  The distributed algorithms are not what is wrong, and neither is this
  package. On four forced CPU devices the same released configuration -- real
  alignment, 200 steps, three recycles -- reproduces the single-device summary
  to four decimals in both layouts. And twenty lines of JAX containing no
  FoldJAX at all -- one `float32[N, 256]` split by rows across two GPUs of the
  A5000 node and gathered by a single replication constraint -- return the
  second half wrong from 0.25 MiB upward and identically zero from 4 MiB
  upward, on the N/2 boundary every time. That node hangs with NCCL
  peer-to-peer enabled and silently drops half the payload with it disabled.
  So the multi-GPU numbers above measure that machine, not this code.

  On hardware that passes that reproduction, it works. Four 2080ti returned
  four complete Boltz-2 predictions -- two at `1d`, two at `2d` -- within
  0.14-0.52 Å CA RMSD of the single-card run against a 0.14 Å floor between two
  single-card runs, the grid's best run landing exactly on that floor. Two RTX
  6000 Pro, which is hardware anyone would deploy on -- 96 GB, bfloat16, fused
  kernels, the released sampling defaults, a real target with its alignment --
  returned three context-parallel runs at 0.130, 0.295 and 0.342 Å against a
  0.379 Å floor, complex pLDDT agreeing to four decimals in every one, and
  trunk arrays 1.40e-2 relative from the single-card run where two single-card
  runs differ from each other by 1.4e-2. Not distinguishable from the machine's
  own noise.

  Neither run claims memory or speed: 254 tokens is below the size where
  splitting a pair tensor pays, and the 6000 Pro node holds three identical
  cards, so the square grid is out of reach there.

  The older reservation this replaces is kept below, because its observations
  still stand and one of them is now explained: silently non-finite output was
  the visible tail of this, and a finiteness check never was the gate.

  Two things are known. A node
  that hands out four cards where only three initialise returns NaN from
  every run on it, so `jax.device_count()` must equal the requested shard
  count before a result is worth reading; that check is necessary and not
  sufficient. On one node where all four cards did initialise, CP=4 returned
  non-finite confidence and coordinates in 11 of 20 runs of the same input,
  at the same rate with and without XLA autotune flags -- which refutes the
  earlier entries in this file that blamed `--xla_gpu_autotune_level`, whose
  evidence was a single run either way against what is now known to be a
  coin-flip base rate. A second node of the same card model returned six
  finite runs from six. Whether the difference is that node's hardware or a
  timing-sensitive defect in the context-parallel path is under
  investigation; single-device runs are unaffected either way.

  What the search has settled so far: the corruption appears after the trunk.
  In a failing run every trunk array is finite and bit-identical to the ones
  the passing runs used -- so nothing the sharded pair code computes is wrong
  -- and only the diffusion coordinates are bad. It is also not always loud:
  Boltz-2 produced structures 34 Å from the single-device answer with every
  array in them finite, so **a finiteness check is the wrong gate**. Mutual
  agreement between repeats does not replace it: the two corrupted Boltz-2
  runs agree with *each other* to 0.224 Å, inside the clean rerun floor, so
  a group of repeats that are all corrupted looks consistent. Only agreement
  with a single-device run settles it. That coincidence is also the sharpest
  clue so far -- a race would scatter two bad runs, and these land in the
  same wrong place, so the corruption is a deterministic computation that is
  sometimes selected rather than noise. Ten hypotheses have been measured and closed --
  autotune flags, a bad node, the step count, the diffusion-side sharding
  constraints, the sampler's `lax.scan`, NCCL's transport, the allocator,
  asynchronous collectives, and pinning the trunk/diffusion handover
  replicated.

  One candidate has been tried and measured: running the sampler inside a
  fully replicated `shard_map`, which the compiled HLO confirms removes every
  collective from the diffusion region and which reproduces the single-device
  structure to 0.0002 Å. Against the unmodified code in the same job, on the
  same cards, twelve rounds each at the released sampling defaults, it did
  not fix anything -- seven failures against four, which at that sample size
  is not a difference (p = 0.41) -- and the surviving runs of the modified
  arm still contained one 9.8 Å from its own siblings. It has been reverted:
  it costs the pair tensor's memory on every device and buys nothing
  measurable. What it did establish is worth keeping: a sharding constraint
  is a preference the partitioner may route around, `shard_map` is the
  guarantee, and the corruption survives the guarantee -- so it is not in
  anything the sampler communicates.

  Known v1 boundary: OpenDDE at ~3,000 residues still exceeds
  3x 96 GB because the fp32 diffusion side's atom-window gathers pull full
  pair tensors per device. Sharding those consumers was the planned next
  lever and is now on hold: the evidence above says that sharding *into*
  the diffusion is what the port cannot yet make reliable.
- **The square context-parallel grid (`--cp-layout 2d`).** Fold-CP's own
  layout, on OpenDDE, Protenix, Boltz-2 and OpenFold3: pair rows *and*
  columns split across a square device grid, with the triangle contraction
  run as Cannon's algorithm -- skew both operands once, then one local
  product per ring hop. Per-device pair cost falls from `O(N^2/P_rows)` with
  a full-width column axis to `O(N^2/P_total)`, and no all-gather appears in
  the contraction: the transient is two tiles. Triangle attention no longer
  falls back to row sharding under the grid; see the next entry.
  **`--cp-layout auto` deliberately stays on the 1-D layout**: every
  published number for context parallelism was measured there, and a default
  that silently changed the program would make those numbers describe a
  configuration nobody can reproduce, so the grid is opt-in until it has GPU
  evidence of its own. Verified on forced CPU meshes at 2x2 *and* 3x3 -- the
  3x3 case exists because modulo 2 a forward ring hop and a backward one are
  the same permutation, so a 2x2 grid cannot falsify a sign anywhere in the
  schedule. Every context-parallel parity gate is now mutation-tested and
  asserts from a trace-time witness that the sharded program really was
  traced.

- **Gather-free triangle attention on the square grid.** Under `--cp-layout
  2d`, Boltz-2, Protenix and OpenFold3 now run triangle attention as Fold-CP's
  own ring instead of gathering the column axis back into whole rows. The
  query tile stays resident; key, value, mask and the pair bias rotate through
  `lax.ppermute` for `sqrt(P)` hops while an online softmax accumulator folds
  each partial exactly. This closes the hole the grid entry above used to
  admit: the contraction was `O(N^2/P_total)` but attention rebuilt an
  `O(N^2/P_rows)` tensor immediately after it, so the grid's memory bound did
  not survive a whole block. It now does. Axes that do not divide the mesh
  side are padded at the `shard_map` boundary and sliced back -- a global
  array padded before it is sharded gathers nothing -- so a 253-residue chain
  still runs on a 2x2 mesh, and the padded columns are pushed below the
  smallest score the caller's mask already carries rather than zeroed, since
  they sit on the axis the softmax spans. Verified on forced CPU meshes at
  2x2 and 3x3, at token counts that divide the grid and at thirteen, which
  divides neither; each gate reads the compiled HLO and fails if an
  `all-gather` appears or the ring's `collective-permute` does not. **No GPU
  measurement yet**: the memory this is supposed to save has not been weighed
  on real cards, and `--cp-layout auto` still stays on the 1-D layout.
  OpenDDE and ESMFold2 are not wired to the ring.

- **Trunk representations (`representations=`, `stop_after="trunk"`).** Every
  model here builds a per-token *single* stream and a token-pair state before
  it predicts coordinates, and until now only two of them could hand those
  back, each under its own name and only through its native command line.
  `PredictionRequest(representations=("single", "pair"))` returns them from
  all five, and `stop_after="trunk"` stops once they exist rather than paying
  for a structure that will be discarded. The result carries a
  `Representations` handle: `result.representations["pair"]` reads one array
  from disk on access, `describe` gives its shape, dtype, axis names and --
  the part that keeps a cross-model comparison honest -- the *space* those
  axes count. Two models both call their pair state `pair`, but OpenDDE's
  `structural_pair` indexes a structural-token space its residue space does
  not cover, and nothing about the array says so. `foldjax capabilities
  --model M` lists what each model produces; ESMFold2 offers two names rather
  than three because it carries one single stream, not an input embedding and
  a trunk output.

  Underneath is a capture mechanism rather than a flag per tensor: a model
  annotates a value with `capture("pair", z)` and the traced program hands
  back the ones that were asked for. Off is free -- an un-requested tap leaves
  no trace in the jaxpr -- and the requested set is a static argument, so a
  different set is a different program rather than a stale cache hit. On costs
  liveness rather than bytes: a captured value becomes an entry output, and
  the temp arena is reused across a run where an entry output is not, which is
  the whole of the 15.2 GiB that returning Protenix's confidence logits used
  to cost at 3,012 tokens. **Nothing is returned by default**, because a pair
  representation is quadratic in token count -- half a gigabyte at a thousand
  tokens, nearly five at three thousand. A tap inside a scanned loop is
  refused unless the caller says what it is willing to spend: stacking
  OpenDDE's structural pair state over 48 refiner blocks is 63 GiB at 488
  residues, and learning that from an OOM twenty minutes in is much worse than
  learning it from the error.
- **Alignment search.** `--msa auto` (`PredictionRequest(msa="auto")`) searches
  for a protein chain that arrived without one and caches the result under
  `$FOLDJAX_HOME/msa/`, keyed by sequence and search provenance rather than by
  model, so one target run through three backends searches once. `--msa
  required` refuses to fall back to single sequence. The default stays `none`,
  which is the previous behaviour — but it now warns once per run instead of
  silently folding a protein from its own sequence. **`auto` and `required`
  send the sequence to the configured server**; `FOLDJAX_MSA_COMMAND` and
  `FOLDJAX_RNA_MSA_COMMAND` point at a local workflow instead, and RNA is
  searchable only that way.
- **Input that is not a job file.** `--sequence`/`--dna`/`--rna`/`--ligand`/
  `--ligand-smiles`, FASTA files, `.pdb`/`.mmcif` depositions (re-folded from
  their chemistry, never their coordinates), `structure:PATH` for a `.cif`, and
  a directory as a batch. Each becomes an ordinary common-schema document under
  the runtime store, so `plan` names it and the manifest hashes it.
  `Job.from_sequences`, `Job.from_fasta` and `Job.from_structure` are the
  Python spellings.
- **Templates and binding affinity in the common schema.** A polymer takes
  `templates: [{mmcif, query_indices, template_indices}]` and a job takes
  `properties: [{affinity: {binder}}]`. The residue map is required by
  AlphaFold 3, Protenix and OpenDDE and refused by Boltz-2, which aligns the
  mmCIF itself; affinity reaches Boltz-2 alone. Whatever a backend cannot
  express is refused, never dropped.
- **`foldjax show`, `foldjax doctor`, `foldjax cache gc`, `foldjax models
  --for JOB`, `foldjax --version`.** `models --for` answers from the input
  translation table, so it needs no weights, GPU or network. `cache gc` reports
  by default and deletes only with `--apply`.
- **Progress on stderr and a summary table on stdout.** Stage lines during a
  run, and the model's own top-ranked structure at the end. stdout stays JSON
  whenever it is not a terminal, so pipes are unchanged; `--json` forces it and
  `--quiet` (or `FOLDJAX_PROGRESS=0`) silences the stage lines.
- **`--resume` and `--keep-going`**, and their `PredictionRequest(resume=...,
  on_error="continue")` spellings. Resume works at *seed* granularity: a
  five-seed job that died on the fourth repeats only what is missing. Failures
  are recorded in `foldjax_failures.json` beside the runs that succeeded, and
  `foldjax.predict_batch` returns results, skips and failures together.
- **`cost.phases` in the run manifest** — preparing input, predicting and
  writing, measured separately, because one number for the whole run cannot say
  which of them to act on.
- **`py.typed`.** The public surface is annotated; downstream type checkers
  now see it instead of `Any`.

### Changed

- **ESMFold2's atom attention runs at the width upstream runs it, and windows
  instead of scoring densely.** Two changes, both on the diffusion atom stack.

  Upstream casts q/k/v to bfloat16 unconditionally right after the rotary
  (`common.py:573-575`) and restores the caller's dtype after the context
  einsum; the port did neither, so every atom-attention call ran float32. That
  is a parity defect on its own terms. It is worth 65,485 -> 35,945 MiB of temp
  arena at 1,003 residues and the released 32 diffusion samples.

  **Those arena figures, and the 35,945 -> 11,312 MiB below, were measured with
  `confidence_sample_sequential=True` -- which was not the shipped default when
  they were taken and is now.** They describe the shipped product. The note that
  stood here said the opposite, and was correct until the default flipped: at
  `False` the confidence head builds `bf16[32, L^2, 2048]` at 122.8 GiB against
  a 95.6 GiB device, so no measurement of that configuration was ever going to
  be a figure for a working product. Both
  A/Bs held the setting constant, so the differences are real at that setting;
  whether they are the same at the default is being measured.

  The attention is a sliding window of +-64 in packed rank and was computed as
  a full `[batch, heads, atoms, atoms]` score matrix with the window applied as
  a mask -- 59,049 MiB, 90.2% of that arena, to carry at most 129 non-zero
  terms per row. It is now a `lax.scan` over row blocks with a `dynamic_slice`
  of `R + 2*half_window` keys, so the window falls out of the slice bounds:
  35,945 -> 11,312 MiB. **Behind `ESMFOLD2_ATOM_ATTENTION_BACKEND`, and now the
  default**: at 1,003 residues and 32 samples it is 46.73 s against dense's
  49.30 and 24.460 GiB against 48.516, three timed runs per arm with the arms
  alternating and warmups discarded. The separation is clean rather than a
  difference of means -- the slowest blocked run beats the fastest dense one --
  and dense held the higher mean SM clock and still lost, so the margin
  understates. It shipped defaulting to `dense`, because flipping a default
  should follow a timing measurement rather than precede one.

  Together 65,485 -> 11,312 MiB, 82.7%. The blocked path is not bit-identical
  to the dense one -- the two contract over different widths, so XLA tiles the
  reductions differently. Against a float64 reference in a ten-seed CPU sweep
  over four window/block shapes, its RMS error was 0.951--1.204 times the dense
  path's; the largest blocked error was 9.1e-7. The gate therefore checks both
  a bounded RMS ratio and a float32-epsilon/output-scale max-error budget,
  rather than comparing two runner-sensitive extreme errors by a fixed ratio.

- **ESMFold2 no longer returns confidence logits nothing reads.** The model
  handed back `pae_logits`, `pde_logits`, `plddt_logits`, `resolved_logits`,
  `pae` and `pde`; `write_prediction_outputs` consumes three arrays and four
  scalars, `representations` offers only single and pair, and no option exposed
  them, so a caller could not have asked for them. They are now behind
  `return_confidence_logits`, defaulting False, named after Boltz-2's flag of
  the same name and default.

  Measured at 1,003 tokens and the released 32 diffusion samples: entry outputs
  fall 16,264 -> 250 MiB. **The net saving is 12,189 MiB, not the 16,014 the
  outputs column suggests** -- the arena rises 3,826 MiB in exchange, because
  dropping the logits does not delete them. `ptm` and `iptm` come from
  `softmax(pae_logits)` and `complex_plddt` from `plddt_logits`, so the tensors
  migrate out of the output allocation and into the arena, where XLA overlaps
  them with other tenants. An earlier estimate of 15.36 GiB counted the outputs
  and not the exchange.

  `distogram_logits` is deliberately outside the flag: it is 245 MiB, flat in
  sample count, and read by a torch-gated parity test that cannot run here.
  `pae` and `pde` are derived matrices rather than raw logits, and in this
  family of tools a PAE matrix is something users want -- so the flag is also
  how to turn them back on, and PAE from ESMFold2 is one argument away rather
  than unsupported.

- **AlphaFold 3 reuses chain confidence aggregates during host
  postprocessing.** `pae_metrics` already computes the minimum chain-pair PAE,
  chain-pair ipTM, per-chain ipTM and cross-chain ipTM that the result writer
  needs; the same arrays were then computed a second time and discarded. The
  writer now consumes the first results directly. On a 3,012-token four-chain
  result with the default five samples, the duplicate PAE and ipTM passes took
  1.34 s on the measured CPU. The reused arrays are exactly equal, including
  masked and NaN cases. This removes transient allocation churn but does not
  claim a lower process peak, because the necessary first pass remains.

- **Protenix and OpenDDE compute each distinct template once instead of once
  per row.** A query with fewer than four template hits is padded up to four,
  and the padding is zeros rather than the gap restype, so a query with no hits
  at all is one all-gap row plus three identical zero rows -- two distinct
  templates, not four and not one. The template stack ran the whole pairformer
  once per row and divided by the row count; it now runs once per distinct row,
  weights each by how many it stands for, and divides by their sum. At 3,012
  tokens Protenix's entry arguments fall 12.404 -> 9.430 GiB and its total
  50.145 -> 47.171; the measured run peak at 1,003 tokens is 8,953 -> 6,970
  MiB. OpenDDE reads templates once per recycle, so its stack evaluations come
  off per cycle; its entry arguments fall 3.455 -> 3.126 GiB at 1,003 tokens,
  measured against a base that differs from main by the dedup call alone. Structures are not bit-identical and are not meant to be: the
  divisor carries a `1e-7` epsilon that does not scale, and the running sum
  re-associates. Paired CA RMSD against a base run is 0.065 A median against a
  0.061 A rerun floor, both about 30x below the model's own 1.86 A sampling
  spread.

- **OpenDDE's bfloat16 trunk now runs the blocked triangle product rather than
  the fused kernel.** Measured on a cooled card, warmup discarded, three timed
  runs per arm alternating: 166.27 -> 143.97 s and 22.44 -> 19.85 GiB at 1,003
  residues, 405.07 -> 373.23 s and 52.10 -> 45.76 GiB at 1,531. Per-arm spread
  was 0.19-0.56% against gaps of 8-13%. The default is keyed on the trunk
  dtype: float32 is what OpenDDE ships and is unchanged, because the blocked
  path's accumulators only narrow when the trunk does. An explicit
  `PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND` still wins on either dtype.

- **The blocked triangle product holds its output at the trunk's width.** It
  was pinned to float32 on the grounds that a narrower destination would round
  every block on the way in. It would, and that is the rounding the caller
  already performs on the next line, so the results are bit-identical. On a
  bfloat16 trunk this removes the largest buffer in that program -- 4,471 ->
  2,235 MiB on Protenix's template stack at 3,012 tokens, moving the temp arena
  39.102 -> 37.367 GiB. A float32 trunk is unaffected.

- Sequences are normalized where the document is validated: all whitespace
  removed (so a YAML block scalar works), upper-cased, and nucleic acids
  checked against IUPAC with the offending position named. A chain id may be
  omitted and is assigned in document order; an id that is present but blank is
  still an error.
- Unknown document fields name the field that was probably meant.
- `capabilities` gained `common_schema_features` and `native_only_features`.
  `supports_templates` describes the model; those two describe what a common
  job document can reach, which is not the same thing and used to be implied.
- `foldjax weights fetch` refuses a download the disk cannot hold and warns
  when conversion will be tight.
- The MSA client identifies itself to the search server as `foldjax/<version>`
  rather than `protenix-jax/0.1.0`.

### Fixed

- **ESMFold2's released configuration did not run.** At 32 diffusion samples --
  what `config.json` ships and `backends/esmfold2.py` carries -- a 1,003-residue
  job terminated with an allocation failure after 71 minutes. The
  confidence head runs its own folding trunk on the spread pair, and batched
  over 32 samples that trunk builds `bf16[32, L^2, 2048]` at **122.8 GiB on a
  95.6 GiB device** -- 27.2 GiB over on a single buffer. That is arithmetic
  rather than fragmentation, so it holds whatever the allocator does: no
  rearrangement makes one allocation fit that is larger than the device. The
  terminal run was taken under `XLA_PYTHON_CLIENT_PREALLOCATE=false`; there is
  no completed-or-failed run of that configuration under default preallocation,
  and the arithmetic is what makes one unnecessary.

  `confidence_sample_sequential` now defaults to on, which divides that and
  every other batched intermediate by the sample count.


- **ESMFold2's folding trunk ran in float32 while every setting said
  bfloat16.** `ModelSettings.trunk_dtype` defaults to `bfloat16`, the released
  `config.json` sets it, and a test pinned that field -- but the field is not
  the tensors. Weights were cast, and three feature tensors were not: one
  float32 term in `z_init = z_init + rel_pos + token_bonds_encoding` widened
  `z_init`, and the `z.astype(z_init.dtype)` after it carried float32 through
  all forty-eight trunk layers. Casting those three to the compute dtype makes
  the port run the configuration it advertises. Warm wall time at 1,003 tokens
  falls from 36.5-38.8 s to 28.2-28.8 s, about 23%, measured with an ordering
  control so it is not a cooler-card artefact. Peak does not move -- 56,228
  against 56,230 MiB, where two runs of the unchanged code at different seeds
  differ by 427 MiB -- which says the pair path accounts for essentially none
  of this model's footprint; its language model is 27.8% of it. Structures
  move 0.065 Å at the median against a rerun floor of exactly 0.000 Å (a fixed
  seed is bit-deterministic here) and a sampling spread of 0.49-1.53 Å, so
  nothing a user reads changes. The gate that let this ship asserted the
  setting; the new one spies the trunk and asserts the dtypes the graph
  actually built, in both bfloat16 and float32, because pinning one would pass
  for a port that hard-coded it.
- **Resume now proves what it reuses.** A finished manifest must match the
  exact scalar request, lexical and resolved input path, input and referenced
  input content, checkpoint and referenced-local-asset stat/tree identities,
  padding, representation selection, and stop point. Any relevant metadata or
  tree change triggers a conservative rerun without rereading large checkpoint
  or asset payloads merely to write or check a manifest. The recorded structure
  must remain non-empty with the same SHA-256; representation archives must
  remain contained beneath the run root with the same names, order, shapes,
  logical dtypes, ZIP metadata, and stat identity. Legacy/incomplete records
  rerun conservatively. Trunk and BF16 archives are restored lazily, and a
  failure from an earlier batch pair no longer suppresses a later successful
  multi-seed manifest. Inputs, checkpoints, and referenced assets must remain
  immutable during backend execution; resume detects changes between runs but
  does not lock those files against concurrent replacement.
- **Built-in writers refuse non-finite public coordinates.** Boltz-2,
  ESMFold2, OpenFold3, Protenix, and the Protenix writer shared by OpenDDE now
  fail before creating a structure when any exposed atom is NaN or infinite.
  Serving-padding atoms remain outside that check, and Boltz-2's guard is an
  explicit exception that remains active under optimized Python.
- **Distributed attention and Cannon accumulation keep their numerical
  contracts at edge cases.** Boltz-2's serial, chunked, fused, 1-D, and 2-D
  pair-biased attention paths return finite zero for an all-masked key set
  instead of a uniform mean or NaN. OpenFold3's 2-D Cannon reduction now uses
  compensated fp32 tile accumulation like the other supported pair trunks. A
  forced four-device model-level gate exercises Boltz-2 conditioning through
  its atom encoder, token transformer, and atom decoder without an all-gather.
- **Unsupported representation requests now fail during request validation.**
  Capability validation understands comma-separated names and `all`, so an
  unsupported array is rejected by `plan`/resolution before model weights or a
  native runtime are loaded. A backend that exposes no representations rejects
  `all`, and an empty comma/whitespace-only selector is refused instead of
  running a trunk graph that returns no arrays. Representation capture is also
  single-seed until the public result can carry more than one archive handle;
  multi-seed capture now fails early instead of silently losing handles.
- **OpenDDE common inputs no longer advertise fields its released runtime
  discards.** Mapped templates and RNA MSAs now fail in compatibility checking
  and materialization; protein MSAs are unchanged. Native OpenDDE passthrough
  retains the existing `templatesPath`/RNA `unpairedMsaPath` warning/drop
  behavior required to match the shipped `use_template=False` and
  `use_rna_msa=False` defaults.
- **The hermetic CI gate no longer waits on RCSB.** Tests marked `network`
  remain available as an opt-in integration surface, but the canonical CPU
  suite excludes them instead of depending on an external service and its
  120-second timeout.
- **OpenFold3 predicted nothing at all.** Every OpenFold3 run raised
  `TypeError: released_config() got an unexpected keyword argument
  'returned_representations'` before the model was reached, with no option
  needed to trigger it and through both `foldjax predict --model openfold3`
  and `openfold3-jax-predict`. The backend passed the trunk-representation
  request into a constructor that had never been given the two parameters,
  though `InferenceConfig` carried both fields already. The checkpoint was not
  installed in the hermetic test environment, so no prediction test caught it;
  the gate that now does reads the two sources and asserts every key one passes
  is a parameter the other accepts.
- **A run that stopped at the trunk still tried to write a structure.** The
  trunk graph returns before the sampler, so there are no coordinates:
  OpenFold3 raised `IndexError` reading an empty coordinate shape and ESMFold2
  raised `KeyError: 'sample_atom_coords'`. Both are backends that write
  structures themselves rather than delegating to their model's command line,
  and both are now fixed -- ESMFold2 had the branch but below the writer, which
  is the same as not having it, so the new gate checks the order and not just
  the presence.
- **The square context-parallel grid was unreachable for OpenDDE through the
  common option.** `opendde-jax-predict --cp-layout 2d` ran while `foldjax
  --model opendde --option cp_layout=2d` was refused as an unsupported option,
  because the backend's option set was missing the entry the other three
  two-dimensional models had.
- **A representation archive disagreed with its own manifest.** Boltz-2,
  OpenFold3 and ESMFold2 wrote a leading batch axis of one, so `shape` carried
  four numbers beside three names in `axes`, while Protenix and OpenDDE wrote
  three of each -- one feature with two contracts, in the file a consumer reads
  to learn what it is holding. Arrays are now conformed to the axes their spec
  names, and an array that is not its spec is refused rather than written under
  a description that does not fit it.
- `python -m foldjax.cli` runs the command line. It used to import the module,
  define `main`, call nothing, and exit 0 with no output, which reads as a
  prediction that produced nothing rather than as a command that never ran.
- Ctrl-C ends a run with one line and exit code 130 instead of a traceback
  through FoldJAX's internals.
- A read-only output directory, a full disk and other `OSError`s are reported
  as the machine facts they are rather than as defects. A full disk names the
  store and the command that reclaims space; a missing optional package names
  the extra that provides it.
- The AlphaFold 3 runtime error names `foldjax runtime prepare --model
  alphafold3` instead of an internal function.
- `foldjax.progress` resolves its stream at write time. Holding the stream
  meant writing to whatever stderr *was*, which could fail a prediction that
  was otherwise fine.

## 0.3.0

FoldJAX became one standalone package: AlphaFold 3, Boltz-2, ESMFold2, OpenDDE,
OpenFold3 and Protenix inference with no sibling repository, plus opt-in
mask-aware padding for reusable JAX executables. See the git history and
`docs/ports/` for how each port got there.
