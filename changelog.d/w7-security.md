### Fixed

- **Boltz-2 no longer answers a run with an earlier run's MSA.** Every common
  job is written as `boltz2_input.yaml`, so every run shares one record id, and
  preprocessing kept an existing `processed/msa/boltz2_input_0.npz`: a second
  job into the same `--output-dir`, or the same job after its `.a3m` was
  edited, silently folded with the first run's alignment, and a record left by
  another job joined the manifest. Each run now builds its `processed/` tree in
  a fresh private directory and then replaces `<output>/processed`, so an
  existing `processed/` is regenerated rather than reused, and a symlink
  planted at that name is replaced instead of written through.

### Security

- **Boltz-2 loads its processed arrays and molecule pickles without arbitrary
  pickle.** `processed/` structures, MSAs and constraints are read with
  `allow_pickle=False` (the absent `pocket` field is no longer stored as a
  pickled `None`, and feature-cache entries written before this change miss
  once and are rebuilt). The CCD molecule pickles beside the weights and the
  `processed/mols/*.pkl` file preprocessing writes go through a restricted
  unpickler that admits only `rdkit.Chem.rdchem.Mol`, as ESMFold2's CCD reader
  already did.
- **The persistent compile cache is used only when no other account can write
  it.** JAX runs a cached executable as found. The cache directory and every
  ancestor must belong to the user (or root) and be writable by neither the
  world nor a group with another member; otherwise FoldJAX warns once and
  compiles without a persistent cache. New cache directories are created
  without group write under any umask. A deliberately shared store opts in
  with `FOLDJAX_TRUST_SHARED_COMPILE_CACHE=1`. This applies to the request
  cache and to the Boltz-2, OpenDDE, OpenFold3 and AlphaFold 3 entry points
  that set the cache themselves. A group-writable, setgid store shared with
  another account (such as a lab's) now misses until that variable is set.
