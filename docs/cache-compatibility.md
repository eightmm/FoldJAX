# Compilation cache compatibility

## 2026-10-19: buffered ESMFold2 states and fresh AF3 warm verification

No default model namespace revision changes. The experimental ESMFold2
`lm_state_storage=buffer` option changes host-orchestrated state storage while
retaining every layer, dtype and the original layer combination. Both storage
choices feed the same structure JIT signature/key; buffer-update helpers have
their own executable keys. Omitted/default requests retain the original stack
path, including already warmed buckets up to 1024. GPU capacity and full-model
accuracy admission for the candidate remain pending.

AF3 process-isolated warming separates preparation from execution verification
without editing the carried runner or source hashed by its tuning store. The
prepared model executable and Tokamax store retain their existing identities.
No rewarm of working default caches is requested for this lifecycle change.
See [cache warming](cache-warm.md) and the
[GPU measurement follow-up](cluster-memory-followup.md).

## 2026-10-19: experimental OpenDDE structural projection row chunks

**OpenDDE rewarm required only for the new opt-in profile.**
`--option structural_triangle_att_row_chunk_size=N` uses a separate namespace;
pass the same width when warming and predicting. The default omits this option
from the profile and retains the existing graph and namespace. No default
rewarm is requested for any model, including AF3/3090. Pure host staging
changes preserve model input values/dtypes and the AF3 portable tuning identity.

The experimental width bounds structural-refiner fused-attention projections.
It has CPU contract coverage but has not passed real fused GPU accuracy,
latency or capacity admission. See the [cluster measurement handoff](cluster-memory-handoff.md).
This is not a new published memory limit or a new numerical default.

## 2026-10-19: candidate cache exclusion and experimental ESMFold2 alignment

AF3 benchmark candidates no longer read or write the persistent JAX cache.
The guard applies inside Tokamax's compiler workers; the real model executable
and persistent tuning records retain their existing cache paths. Existing
`jit_forward-*` files are not automatically deleted. No AF3 rewarm or namespace
revision is required for this change.

ESMFold2's default `svd` alignment and namespace are preserved. The experimental
`rigid_align_backend=jacobi` option selects a separate compilation profile and
must be passed to both `cache warm` and `predict` when used. It does not replace
the measured default or establish a new memory limit.

## 2026-10-18: portable AF3 tuning identity and shared permissions

**AF3 rewarm required after upgrading from 596369b2.** Tokamax's source identity
now binds parameter bytes, runner bytes, and sorted relative source-file names
with their sizes and SHA-256 digests. Mount device numbers, inodes, timestamps,
permissions and absolute install paths no longer enter that identity. The same
assets therefore produce the same source digest across NFS clients. Device,
driver/runtime, call shape and configuration still have to match.

The JAX namespace revisions are unchanged for all six models. Existing JAX
executables remain candidates for reuse, but old AF3 tuning records use the old
source identity and are not migrated. Warm the serving buckets again to publish
the content-based tuning records:

```bash
foldjax cache warm --model alphafold3 --token-buckets auto
```

Each process hashes unchanged assets once and uses stat generations only to
invalidate that in-memory hash memo and the loaded-weight/session anchors.
Generated Python bytecode is excluded from the source digest. The audit found
ESMFold2, Boltz-2 and `_weight_session` snapshots used for process-local ownership.
Manifest stat records remain strict validation of an existing output generation
for `--resume`; they do not enter JAX namespaces or Tokamax's shared lookup keys.

New ordinary shared namespaces use 0755; private namespaces retain 0700.
Explicitly trusted group installations retain their umask/setgid policy.
Warming warns about shared entries or directories whose modes prevent general
read access. Existing 0700/0600 paths are not changed by prediction: an owner or
administrator can repair the managed installation with
`foldjax setup --shared --apply`.

## 2026-10-18: ESMFold2 managed LM path

**Warm ESMFold2 again after upgrading from 568ee5c1.** Its graph revision is
`2026-10-18-compact-lm-chain64`. This update preserves the other five model
namespaces, including the three `chain64` revisions announced below.

Warming now opens the same backend session as managed prediction. ESMFold2
therefore stages the language model and passes its compact embedding into the
structure graph, as single-input, batch and multi-seed prediction do. The old
warm route passed all language-model layer states and could populate the same
namespace with a different executable. Its larger live tensors also mean the
previous card-specific warm limits, such as 640 tokens on a 4090, must be
measured again on the compact route. No higher capacity is guaranteed by this
change, and memory-admission limits have not been raised without measurements.

The structure graph reserves 64 chain-score slots and crops them to the real
chain extent. Ordinary chain counts share an executable when other shapes and
settings agree; an extent above 64 grows the capacity with a warning.

```bash
foldjax cache warm --model esmfold2 --token-buckets 128..1024 --msa single
```

## 2026-10-18: chain score capacity

**Warm Protenix, OpenDDE and OpenFold3 again after upgrading from 799ca578.**
Their declared graph revision is `2026-10-18-chain64`, and their namespace
changes so that an older warm cannot be mistaken for the current graph.
That revision preserved the AlphaFold 3, Boltz-2 and ESMFold2 namespaces.

Chain confidence arrays now have 64 slots by default. Within the same other
tensor shapes and execution settings, chain counts 1, 5 and 12 share the model
executable. Only a chain extent above 64 grows the score capacity to the next
power of two and emits a warning. Final confidence arrays retain their actual
chain extent. The score reductions avoid unrolling all capacity pairs or
stacking chain-pair PAE-bin tensors.

Use the same serving options as prediction when warming:

```bash
foldjax cache warm --model protenix openfold3 opendde \
  --token-buckets 128..1024 --msa single --fixed-msa 2048
```

The 22/24 GiB device budget remains an admission limit. It is not a new chunk
selection input: the three ports already resolve their automatic chunk widths
from token/sample counts and explicit options. The run record's
`resolved_chunking` field exposes the resulting widths for comparison. Matching
widths do not prove why two physical cards did or did not reuse an executable;
device/runtime, input axes and other compilation settings must also agree.

## Recording future changes

When a model's graph ABI or namespace profile changes, update only that model's
declared revision in `foldjax.cache`. Add a dated note here and name the models
requiring another warm in the commit message. A documentation or test-only
change does not require a revision bump.

An entry in the expected namespace proves that the directory contains a cache
artifact. It does not prove coverage of every shape, kernel, optional input, or
source-dependent XLA key. See [cache warming](cache-warm.md) for the read-only
verification command and its limits.
