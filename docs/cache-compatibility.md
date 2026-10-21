# Compilation cache compatibility

## 2026-10-21: bounded OpenDDE serving profiles and admission

No existing model namespace revision changes. Memory-admission updates,
cuSolver diagnostics and the opt-in ESMC measurement controls do not require
rewarming working default caches for any of the six models.

OpenDDE's optional `--opendde-padding-profile compact` fixes auxiliary targets
within each supported token bucket. **Warm missing OpenDDE compact shapes**
with the same option used for prediction. Padding policy metadata does not
split the outer namespace; JAX distinguishes the concrete input/HLO keys.
Existing entries with identical shapes and execution options remain eligible,
and ordinary OpenDDE padding is unchanged. This bounded profile refuses inputs
that exceed its auxiliary capacities instead of silently growing or truncating
them. See the [profile table and cluster checks](opendde-compact-profile.md).

ESMFold2 retains its default numerical path and compiled signatures. Its
observed 48 GB refusal at 1792 or more tokens is an admission decision, not a
graph change. The new [process-variation controls](esmc-process-variation-2026-10-21.md)
are explicitly labelled experiments; their deterministic or compiled-block
arms are not changes to prediction defaults.

## 2026-10-20: editable and fixed-install source paths

FoldJAX defaults `jax_hlo_source_file_canonicalization_regex` to exactly
`.*/(site-packages|dist-packages|src)/` before lowering. This removes installation
prefixes inside serialized Pallas-Triton kernels, where ordinary HLO metadata
stripping does not reach. An explicit environment value or non-`None` config
value, including the empty string, takes precedence. Model namespace revisions
are unchanged; the embedded kernel bytes determine the executable key.

**Rewarm affected Pallas-Triton executables that were built without this
canonicalization**, including the reported AF3, Boltz-2, Protenix, OpenFold3
and ESMFold2 paths. Entries already built with this exact regex remain eligible
for reuse; do not delete working caches just because the installation moved.
Matching source, weights, shapes, compiler options and GPU/runtime are still
required. This does not promise cache compatibility across arbitrary edits.

Wheels now include the two carried AF3 shell scripts, preserving the editable
runtime key and making its source set identical to a clean wheel. Portable
Tokamax source identities exclude legacy generated extensions and CCD pickles,
as runtime build identities already do. **AF3 rewarm required only if the old
tuning source digest included these generated files.** Clean-source tuning
identities remain unchanged; session stat anchors still detect local mutation.
Weight paths in compilation namespaces retain their existing contract: moving
the installed package does not require moving the shared weight store.

JAX 0.11.1's Pallas keyword-form trace-key explanations also receive a narrow
diagnostic compatibility adapter. It preserves requested miss/debug logs and
does not change actual trace keys, persistent keys or model computations.

## 2026-10-20: OpenDDE automatic capacities and Boltz-2 RTX 4090 allocator

**OpenDDE automatic padding rewarm required when resolved axes change.** Atom
capacity now follows stored atoms on the existing grid; structural capacity
rounds stored structural tokens to 64 / context-parallel alignment. For the
reported 1024-token job, atoms 24576→8192 and structural tokens 2048→1856
select a different JAX shape key. Warm real jobs or matching explicit pins;
synthetic token warming covers its reported concrete shapes only.

The OpenDDE model graph and namespace revision are unchanged. Existing exact
or explicit requests with the same concrete shapes, masks and options remain
valid. The other five model namespaces are unchanged. Boltz-2's confirmed
RTX 4090 default pool fraction changes to 0.8 without changing its keys;
working caches need not be rebuilt for that allocator choice alone.

ESMFold2 keeps `stack` and its existing graph as defaults. Cross-process LM
variation is being diagnosed separately from the storage experiment; no new
numerical default or capacity claim is admitted. See the
[measurement plan](cluster-memory-2026-10-20.md).

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
