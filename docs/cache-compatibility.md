# Compilation cache compatibility

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
