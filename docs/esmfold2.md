# ESMFold2

The one carried model that is not an evolutionary-trunk predictor, and the
one whose two runs of the same job differ on purpose.

ESMFold2 is the odd one out and worth knowing about before you use it. It has
no evolutionary trunk: it is a diffusion structure head on a linear-recurrence
pair trunk, folding the representations of **ESMC-6B**, a 25 GB protein
language model that upstream distributes separately (`foldjax weights fetch
--model esmfold2` puts the complete bundle where the loader looks). It is also
*random at inference by design* —
random initial pair state, per-loop language-model dropout, sampler noise — so
two seeds give genuinely different structures rather than the same structure
twice. **[Upstream ESMFold2](https://github.com/Biohub/esm/blob/main/esm/models/esmfold2/prepare_input.py)
is an all-biomolecule model**: its released input pipeline supports proteins,
DNA, RNA, small-molecule ligands, and modified residues. FoldJAX ports that
input contract without Torch: standard polymers, CCD and SMILES ligands,
modified residues, and explicit covalent bonds are featurized in NumPy from
the verified Biohub CCD snapshot. Protein-only jobs retain the established
in-package fast path.

The complete released ESMFold2 profile is the compatible default. To run the
explicit structure-network-only variant without downloading ESMC-6B, fetch its
1.36 GB structure+chemistry profile and select the same profile for prediction:

```bash
foldjax weights fetch --model esmfold2 --profile structure-only
foldjax predict --model esmfold2 --input protein.json \
  --profile structure-only
```

`--profile structure-only` sets `no_language_model=true` itself; conflicting
native options are rejected instead of silently overriding the profile.
The older explicit option still selects that managed profile automatically
when weights are omitted. Without either spelling, FoldJAX still requires the complete
structure+ESMC release and never silently falls back to a different model.
An explicit `esmc_weights=/path/to/esmc` also selects the 1.36 GB managed
profile: the released language-model branch still runs, but its weights come
from that path rather than a redundant managed ESMC copy.

## Denoising the samples one at a time

`--option structure_sample_sequential=true` runs the diffusion denoiser once
per sample instead of once for all of them. It is off by default.

What it narrows is one tensor. The diffusion token transformer builds attention
logits of shape `[samples, tokens, tokens, heads]` in float32, inside every
block of every denoiser call, and at the released 16 heads and 3,012 tokens
that is 2.7 GiB at five samples and 17.3 GiB at thirty-two. Sequencing divides
it by the sample count, and narrows the diffusion cache's per-atom conditioning
by the same factor. The blocked atom attention carries a sample axis as well,
but it is linear in atoms rather than quadratic in tokens, so this one tensor
is most of the saving.

What it does not narrow is the trunk, and that matters more than the saving
does. ESMFold2's pair trunk has no sample axis in this port at all: the
recurrence and the coda return one `[tokens, tokens, d_pair]` state, and the
sample count first appears in the structure head. The 45 GiB peak measured at
2,096 residues and the failure at 3,012 are trunk pair tensors, so a run on the
released five-sample schedule should not expect this option to move them. It
earns its place when a caller raises the sample count, not when a long input
will not fit.

The cost is one launch per sample per step in place of one batched launch, and
arithmetic that is the same arithmetic on a narrower array rather than a
bitwise-identical one: a reduction over one row may order itself differently
than over thirty-two. The random stream is untouched. Every draw the sampler
makes -- the initial noise, the per-step augmentation and translation, the
churn normal -- still happens once per step at the full batched width from the
same key, so each sample gets the same noise it would have got batched. The
confidence head's own `confidence_sample_sequential` is a separate stage and
stays on; either may be set without the other. A request that puts more than
one input in a single call is refused rather than run, because telling a
rollout which input it belongs to would mean slicing the pair conditioning
inside the loop this option exists to narrow.

## Managed memory staging

For one semantic input, including a multi-seed sweep of that input, FoldJAX's
managed predictor does not need ESMC-6B and the structure network resident at
the same time. It first loads ESMC with only the small structure-side projection
that combines its hidden layers, synchronizes the resulting compact embedding,
releases ESMC's device arrays, and then loads the structure network. This changes
buffer ownership only; the input, seed, model values, and generated structures
are unchanged.

A batch containing different inputs deliberately keeps the complete model
resident and reuses it. Staging such a batch would trade the memory saving for
re-reading the 25.4 GB ESMC checkpoint for every input. Direct calls to
`foldjax.models.esmfold2.inference.load` likewise still return the complete
model; staging is confined to FoldJAX's managed request session.
