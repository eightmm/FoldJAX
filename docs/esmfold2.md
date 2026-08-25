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
