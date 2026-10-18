# Compilation cache compatibility

## 2026-10-18: chain score capacity

**Warm Protenix, OpenDDE and OpenFold3 again after upgrading from 799ca578.**
Their declared graph revision is `2026-10-18-chain64`, and their namespace
changes so that an older warm cannot be mistaken for the current graph.
This update preserves the AlphaFold 3, Boltz-2 and ESMFold2 namespaces.

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
