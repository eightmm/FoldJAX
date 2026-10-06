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
