# Provisioning the upstream environments

A comparison against an upstream that is missing its own accelerated paths is
not a comparison, it is a handicap match. Each upstream repository here runs in
its own virtualenv, and two of them arrived incomplete in ways that inflated
their numbers. This is what it took to give each one its intended fast path,
and how to check it.

Nothing here touches the system: every install goes into that project's own
`.venv`, and nothing is removed.

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

## The two that were already complete

- **Boltz-2** has cuEquivariance installed and runs `bf16-mixed` by default.
- **Chai** does not use cuEquivariance at all; its components are traced
  TorchScript modules.

Neither needed anything.

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

Kernels: none. The default DS4Sci `evoformer_attn` fails its cutlass JIT on
sm_120, and the experimental `use_cueq_triangle_kernels` path crashes in the
confidence stack at `num_diffusion_samples > 1` (bias batched over samples
where the kernel demands (B, 1, H, S, S)). Plain torch attention is the only
configuration 0.3.1 completes on this card, so its timing column carries that
caveat -- unlike the other upstreams, which all reached their fast paths.

### The two 3,012-token questions, answered by measurement (2026-08-12)

**Boltz-2's OOM is capacity, not fragmentation.** With
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` the run reached 96.2 GiB of
the 97.9 GiB card -- 15 GiB beyond its earlier fragmented peak -- and still
died. The setting stays (it is how near-capacity torch should run) but the
verdict stands: this model does not fit this card at 3,012 tokens upstream.

**OpenFold3's 1.9-hour cell is its own shipped default, not our harness.**
Disabling the confidence-head host offload (predict preset, token_cutoff
2,800) moved 6,782 s to 6,722 s -- sixty seconds. The time lives in
`chunk_size: 4`, which the predict preset applies at every size; 0.3.1 ships
no larger-card preset. The offload stays off (peak 81.0 -> 83.2 GiB, still
fits), and the wall-clock column simply is what upstream does here.
