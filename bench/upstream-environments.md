# Provisioning the upstream environments

When a fast path supported by the reference and this hardware is merely
unprovisioned, comparing that fallback is a handicap match. Each upstream
repository here runs in its own virtualenv, and two arrived incomplete in ways
that inflated their numbers. This is what it took to provision those paths and
how to check them. Hardware-incompatible paths, such as OpenFold3 0.3.1's
sm_120 limitation, remain explicitly qualified rather than described as a
missing install.

Nothing here touches the system: every install goes into that project's own
`.venv`, and nothing is removed.

## Source provenance gate

Environment fixes and source patches are different controls. `run_upstream.py`
requires a clean tracked checkout by default and records its commit, runtime
versions, path-free device/execution identity, and tracked status in every
current-schema result. It hashes the common job and every referenced
MSA/template, the generated native input and its referenced companions, the
selected checkpoint, and model-specific implicit assets before inference and
verifies the same fingerprints afterwards. When a
reviewed compatibility patch such as Protenix's sm_120 architecture entry is
required, hash `git diff --binary --no-ext-diff HEAD --` and pass that exact
digest through `bench.drive --upstream-patch MODEL=SHA256`. A later edit changes
the digest and fails before GPU work starts. Ordinary untracked files are
rejected because Python can import them. Git-ignored in-tree source, config,
executables, and native extensions are runtime inputs: each is byte-hashed into
the row, and a change during inference invalidates the measurement. Symlinks
and special files fail closed. `--skip-existing` also requires the requested
sample count, finite nonempty scores, valid measurements, and an exact identity
match rechecked around the safe result read.

The historical result files already checked into `bench/results*` predate this
schema and artifact gate. They document dated measurements, not an exact
source/runtime/checkpoint capsule, and are never accepted by current
`--skip-existing` solely because the filename exists.

## OpenDDE — cuEquivariance kernels

OpenDDE resolves `triangle_multiplicative` and `triangle_attention` from a
config default of `"auto"`, which picks cuEquivariance when it is importable
and plain torch when it is not. Its `.venv` had no cuEquivariance at all, so it
was running the torch path — which materialises the triangle tensors that the
fused kernels avoid, and that is most of a 91 GiB peak at 490 tokens.

Its `pyproject.toml` pins the **cu12** builds, but the installed torch is
`2.12.0+cu130`, so those would not have loaded. The cu13 builds at the same
version are what matches:

```bash
uv pip install --python .venv/bin/python \
    cuequivariance==0.10.0 cuequivariance-torch==0.10.0 cuequivariance-ops-torch-cu13
```

Check it resolved, rather than assuming:

```python
import torch
from opendde.utils.environment import get_cuequivariance_runtime_status
print(get_cuequivariance_runtime_status(torch.device("cuda:0")))
# CuEquivarianceRuntimeStatus(unavailable_reason=None, requires_cc7_fallback=False)
```

## Protenix — the fused CUDA layer norm

Protenix's default layer norm is a CUDA extension compiled on first use. torch
ships the CUDA *runtime* as `nvidia/cu13` but not the compiler, so on a host
with no system CUDA the import dies with "CUDA_HOME environment variable is not
set" and the only way to run is `LAYERNORM_TYPE=torch` — upstream's slow path.

The toolkit can be completed inside the same virtualenv, but **the versions
have to match the runtime torch shipped**, or nvcc and the headers disagree
("CUDA compiler and CUDA toolkit headers are incompatible"). Here torch carries
`nvidia-cuda-runtime 13.0.96`, so everything else is pinned to the 13.0 line:

```bash
uv pip install --python .venv/bin/python \
    nvidia-cuda-nvcc==13.0.88 nvidia-cuda-cccl==13.0.85 \
    nvidia-cuda-crt==13.0.88 nvidia-nvvm==13.0.88
```

Two link-time gaps remain in the pip layout, because the wheels ship versioned
libraries and torch links `-L$CUDA_HOME/lib64 -lcudart`:

```bash
CH=.venv/lib/python3.11/site-packages/nvidia/cu13
ln -sf libcudart.so.13 "$CH/lib/libcudart.so"
ln -s lib "$CH/lib64"
```

`run_upstream.py` finds this toolkit itself (`_cuda_home`) and exports
`CUDA_HOME` plus a `PATH` containing both `nvcc` and the venv's `ninja`. If it
is absent it falls back to `LAYERNORM_TYPE=torch` and the report says so, so a
machine without the toolkit still produces a run — just not upstream's default.

### The architecture list stops before this GPU

With the toolkit in place the extension builds and then fails at every launch
with `cudaErrorNoKernelImageForDevice`. `torch_ext_compile.py` enumerates the
architectures it will emit and the list ends at `compute_100`; this card is
sm_120. It also *overwrites* `TORCH_CUDA_ARCH_LIST` from that same list, so no
environment variable can steer it. One line in that checkout fixes it:

```python
_wanted = [..., ("100", "100"), ("120", "120")]
```

nvcc 13.0 lists `compute_120` as supported, so nothing else is needed.

### Verifying it, properly

Checking that the call returns is **not** enough. A kernel launched for the
wrong architecture fails asynchronously, so the Python call returns normally
and the error surfaces at some unrelated op later — that is exactly how this
was missed the first time. Synchronise, and compare against a reference:

```python
import torch
from protenix.model.layer_norm.layer_norm import FusedLayerNorm
m = FusedLayerNorm(384).cuda()
x = torch.randn(4, 16, 384, device="cuda")
out = m(x)
torch.cuda.synchronize()                       # without this, a bad build passes
ref = torch.nn.functional.layer_norm(x, (384,), m.weight, m.bias)
print(float((out - ref).abs().max()))          # 4.77e-07 here
```

Also delete `protenix/model/layer_norm/build.ninja` and the objects beside it
when changing architectures: torch reuses that build directory and will happily
keep a module compiled for the old list.

## The accelerated path that was already complete

- **Boltz-2** has cuEquivariance installed and runs `bf16-mixed` by default.

It needed no environment repair.

## What precision each upstream actually runs

Not one answer for the family — the two Protenix-derived repos disagree, and
assuming otherwise put a wrong premise into this project's task list ("opendde
still fp32 vs upstream bf16", which is backwards).

| upstream | storage dtype | TF32 matmuls | where |
|---|---|---|---|
| Protenix | `bf16` | on | `configs/configs_base.py:135`, `configs_inference.py:32` |
| OpenDDE | `fp32` | on | `opendde/config/model_base.py:37`, `inference_defaults.py:24` |
| Boltz-2 | `bf16-mixed` | — | its own default |

The benchmark passes no dtype argument to any of them, so these defaults are
what it measures against. Two consequences worth stating plainly:

- A FoldJAX run at `fp32` against Protenix is not a matched comparison; it asks
  more of this port than of its reference.
- Against OpenDDE the *storage* dtypes match, so the only precision question
  left is the matmul one — `enable_tf32: True` means upstream's fp32 matmuls
  carry ten mantissa bits, and JAX's default on this card is a separate
  question that has to be measured rather than assumed.

## A note on torch versions

Both OpenDDE and Protenix pin torch versions in their `pyproject.toml`
(`2.7.1` for OpenDDE) that predate this GPU. Those wheels carry no SM_120
kernels and fail with "no kernel image is available" — OpenDDE's runner
reporting success while doing so. Both virtualenvs were already upgraded to
`2.12.0+cu130` before this work; that is a deviation from their pins and is the
only way either runs on this card.

## OpenFold3 (added 2026-08-12)

Runs from the `openfold3-v031` worktree (tag 0.3.1): `of3_ft3_v1.pt` -- the
weights the FoldJAX port runs -- is the legacy p1 checkpoint with
`version_compatibility="<0.4"`, and the 0.4.4 checkout refuses it. The
`pae_enabled` preset must be on (the checkpoint carries the PAE head; 0.3.1
builds it only under that preset).

Seed audit (2026-08-31): the original recorded rows also passed
`--num_model_seeds 1`. In 0.3.1 that replaces the runner YAML's `[101]` with a
seed generated from the hard-coded start 42 (`2746317213`). Time and peak
memory remain shape-valid, but confidence is not a seed-matched parity point.
The harness now omits the flag so the YAML seed is authoritative.

The original command also left `--use_templates` at 0.3.1's `true` default
even though these native jobs supplied no template path. The model ultimately
received zero-template features, so outputs and GPU inference are unaffected,
but wall time included avoidable template/CCD initialization. Current runs
pass `--use_templates false`; historical timing retains this disclosed caveat.

Kernels: none. The default DS4Sci `evoformer_attn` fails its cutlass JIT on
sm_120, and the experimental `use_cueq_triangle_kernels` path crashes in the
confidence stack at `num_diffusion_samples > 1` (bias batched over samples
where the kernel demands (B, 1, H, S, S)). Plain torch attention is the only
configuration 0.3.1 completes on this card, so its timing column carries that
caveat -- unlike the other upstreams, which all reached their fast paths.

### The two 3,012-token questions, answered by measurement (2026-08-12)

**Boltz-2's OOM is capacity, not fragmentation.** With
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` the run reached 96,191.6 MiB
(93.9 GiB) against the card's 97,887 MiB (95.6 GiB) `nvidia-smi` total --
12.6 GiB beyond its earlier fragmented peak -- and still died. PyTorch reports
94.97 GiB usable on this device. The setting stays (it is how near-capacity
torch should run) but the verdict stands: this model does not fit this card at
3,012 tokens upstream.

**OpenFold3's 1.9-hour cell is dominated by upstream's shipped chunking, not
harness overhead.**
Disabling the confidence-head host offload (predict preset, token_cutoff
2,800) moved 6,782 s to 6,722 s -- sixty seconds. The time lives in
`chunk_size: 4`, which the predict preset applies at every size; 0.3.1 ships
no larger-card preset. The offload stays off (peak 81.0 -> 83.2 GiB, still
fits), so the row is an explicitly disclosed harness variant of the upstream
prediction preset rather than an untouched-default claim.

## ESMFold2 — the `transformers` reference for the atom parity test

`tests/models/esmfold2/test_atom_parity.py` compares the atom stack against
`transformers`. FoldJAX's own environment is Torch-free on purpose, so the two
tests that need it skip there and only the window test runs. To run all of
them:

```bash
UV_PROJECT_ENVIRONMENT=.venv-parity uv sync --extra cuda13
uv pip install --python .venv-parity/bin/python torch transformers
JAX_PLATFORMS=cpu .venv-parity/bin/python -m pytest \
    tests/models/esmfold2/test_atom_parity.py
```

**`uv pip install` reads `--python` or `VIRTUAL_ENV`, not
`UV_PROJECT_ENVIRONMENT`.** Setting only the latter installs Torch into the
project's own `.venv` -- the one environment this repository keeps Torch out of,
and the one every other measurement runs in. `uv sync` puts it back, but a
benchmark taken in between would have been taken somewhere else.

Verified 2026-08-27 with `transformers` 5.16.1 and `torch` 2.13.0: the rotary
tables are bit-identical, and the encoder test skips with its reason. The
import path matters -- upstream moved this from
`modeling_esmfold2_common`, which held free functions taking explicit widths,
to `modeling_esmfold2`, which builds modules from a config. The old path is
still importable-looking enough that the file simply reported "skipped" on a
machine that could have run it.

### The two artifacts, and why both are needed

`transformers` and the published checkpoint have diverged in naming. Against
`transformers` 5.16.1 and the released `model.safetensors`: the library's
`EsmFold2Model` carries 2,226 parameter names, the checkpoint carries 1,594, and
12 appear in both. Loading the published file through the library reports every
tensor as unexpected. The checkpoint has
`inputs_embedder.atom_attention_encoder.atom_transformer.blocks.0.adaln_modulation.1.weight`
where the library has `layers.0.adaln_linear.weight`. ESMFold2 entered
`transformers` at 5.16.0 and has been named this way since; no released version
matches the checkpoint, and `modeling_esmfold2_common`, which an earlier version
of the parity test imported, is in none of them.

**They are the same trained weights.** `atom_linear.weight`,
`atom_norm.weight`, `atom_to_token_linear.weight` and the adaLN projection are
bit-identical across the two artifacts, and the checkpoint's fused `Wqkv` is
exactly `concat(q_proj, k_proj, v_proj)` from the re-export. So the parity test
gives each side its own file and maps nothing -- FoldJAX loads the published
checkpoint, upstream loads its re-export, and only the outputs are compared.
A name mapping would reintroduce the rename table that naming after the
checkpoint was chosen to delete, and would pass just as happily lining up the
wrong tensors.

The library's side is `biohub/ESMFold2-hf`, 24.5 GB of which almost all is the
language model. safetensors keeps byte offsets in its header, so the 28
atom-encoder tensors come out with range requests:

```bash
python tests/models/esmfold2/fetch_hf_atom_weights.py \
    "$FOLDJAX_BENCH_DATA/esmfold2-hf/atom_encoder.npz"
curl -sLo "$FOLDJAX_BENCH_DATA/esmfold2-hf/config.json" \
    https://huggingface.co/biohub/ESMFold2-hf/resolve/main/config.json
ESMFOLD2_HF_ATOM_WEIGHTS="$FOLDJAX_BENCH_DATA/esmfold2-hf/atom_encoder.npz" \
JAX_PLATFORMS=cpu .venv-parity/bin/python -m pytest \
    tests/models/esmfold2/test_atom_parity.py
```

3.6 MiB rather than the 4.7 GiB shard they live in.

### What the encoder comparison found

The atom embedding agrees to 4.8e-7 -- float32 exact. Past it the two diverge
by 1.3e-2, and all of it is one deliberate cast: this port drops q/k/v to
bfloat16 when the caller is float32, which the reference it was ported from did
and `transformers` 5.16.1 does not. Removing that cast alone takes the layers
from 4.2e-3 to 4.9e-6 and the token output from 1.3e-2 to 2.8e-5, so nothing
else in the stack disagrees.

The cast is not a defect. At the released 32 samples and 7,776 atoms the
float32 scores would be 29,524 MiB, and the released path runs bfloat16, where
the cast does not fire. It is visible only to a float32 caller.

## Boltz-2 — the 101 tests that no ordinary run collects

`tests/models/conftest.py` drops 24 Boltz-2 modules from collection when torch
is absent, because they import it at module scope. That gate is correct and
documented, but it is worth saying what it costs: the shipped environment
collects 166 tests under `tests/models/boltz2/`, a torch environment collects
267. The missing 101 run in no CI job either -- CI has no torch, and the
comparison needs the publisher's 2.28 GB `boltz2_conf.ckpt`, which is not
something a runner fetches. They are local-only by construction.

```bash
UV_PROJECT_ENVIRONMENT=.venv-parity uv sync --extra cuda13
uv pip install --python .venv-parity/bin/python torch
JAX_PLATFORMS=cpu .venv-parity/bin/python -m pytest tests/models/boltz2/
```

The suite locates upstream as `Path(__file__).parents[4] / "boltz"`, so it wants
the sibling checkout at its usual place; a git worktree elsewhere needs that
path symlinked or every checkpoint test skips instead of running.

Verified 2026-08-27 at `a7751e2`: 261 passed, 5 skipped, 1 failed in 115 s.

### The one that failed, and why it was not a regression

`test_checkpoint_diffusion_transformer_layer_chunk_matches_unchunked` asserted
`rtol=0, atol=0` between a chunked and an unchunked diffusion-transformer
layer. At `a7751e2` it differed by 1.7e-5 absolute, 1.1e-4 relative, on 72% of
3,072 elements.

Bisect pointed at a context-parallel commit, which was wrong: the bisect had
been narrowed to `src/foldjax/models/boltz2` and so never tested the commits in
between. The parent failed too. So did the commit that introduced the gate, and
so did the repository's first commit. Nothing regressed.

The premise was false from the start. Chunking slices the query axis, that
changes the `bihd,bjhd->bhij` extent, and XLA picks a different contraction
schedule for the smaller operand. A twenty-line script against the bare
primitive reproduces it at 3.6e-7 with chunk 2 and 2.9e-7 with chunk 4 --
identically in the shipped `.venv` and in `.venv-parity`, with and without
torch imported first, so it is neither this port's arithmetic nor torch's
threading. The test now asserts closeness at 1e-3, roughly sixty times the
measured deviation, with the measurement written down beside it.

### What the tolerances are actually achieving

Instrumenting every `assert_allclose` in the suite recorded 111 comparisons.
87 of them use **less than 1%** of the tolerance they declare, and 23 are
bit-identical -- deviation exactly zero -- while declaring as much as 2e-3.
Only six use more than a tenth of theirs; the tightest is
`test_micro_modules.py:54`, at 49% of 1e-3.

This is worth knowing before trusting a pass, because a tolerance can be the
size of the defect it is meant to catch. It is **not** an argument for
tightening all 87 to their measured values: these numbers come from one CPU
backend, and a threshold calibrated against this machine's contraction
schedules is a threshold that fails on the next one for no real reason. The
failing test above is the case where the number was demonstrably wrong rather
than merely generous.

## OpenFold3 — 396 parity tests, and the guard that was switched off

OpenFold3 gates differently from Boltz-2, and better: `openfold3_source` skips
inside a fixture instead of dropping modules at collection, so the tests stay
visible and countable. The shipped environment reports 371 passed and 362
skipped; `-m torch_parity` selects 396 of them, and those 396 run nowhere by
default.

They need torch, `ml_collections`, the chemistry stack behind
`--extra openfold3-preprocess`, and Lightning, which upstream's
`InferenceDataset` imports. Order matters, because `uv sync` prunes anything
outside the lockfile:

```bash
UV_PROJECT_ENVIRONMENT=.venv-parity uv sync --extra cuda13 \
    --extra openfold3-preprocess
uv pip install --python .venv-parity/bin/python \
    torch ml_collections pytorch_lightning
JAX_PLATFORMS=cpu OPENFOLD3_SOURCE=/path/to/openfold3 \
    .venv-parity/bin/python -m pytest tests/models/openfold3/ -m torch_parity
```

`OPENFOLD3_SOURCE` is worth using rather than relying on the sibling-directory
search: from a git worktree the fallback resolves to the worktree's parent, and
every parity test skips instead of running.

Verified 2026-08-27 at `a29e4a3`: 395 passed, 3 skipped, 4 min 40 s.

A first attempt reported 189 failed and 64 errors in 21.8 seconds. None of it
was real -- 21.8 seconds is not 189 comparisons failing, it is 189 imports
collapsing, 252 of them on `biotite`, because that environment had only
`--extra cuda13`. A parity run that finishes far too quickly is reporting on
its own provisioning.

### The two that were real

`test_no_field_is_left_unchecked` exists so that adding a field to
`InferenceConfig` cannot silently escape the upstream comparison. It is itself
`torch_parity`, so it had not run, and four fields had escaped:
`cp_layout`, `cp_shards`, `returned_representations` and `stop_after_trunk`.
All four are FoldJAX's own -- `grep` finds none of them anywhere in the
OpenFold3 checkout -- so the defect was the missing declaration, not the
fields. They are now declared with the reason each has no upstream counterpart,
and the set is called `foldjax_only` rather than `shapes`, which three of its
entries already were not.

`test_torch_parity_preprocessing.py` guarded `torch` with `importorskip` and
not `pytorch_lightning`, which upstream pulls in on the way to the tensorizer.
On an environment with torch and without Lightning -- exactly what
`--extra openfold3-preprocess` produces -- it failed where it should have
skipped. Both are fixed: Lightning is now guarded, and installing it lets the
test actually run, which it had never done. It passes: the NumPy tensorizer
matches upstream's.
