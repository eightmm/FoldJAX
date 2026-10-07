### Fixed

- **The test suite no longer reads the developer's FoldJAX store.** With
  `FOLDJAX_HOME` unset, a source checkout's `.foldjax/` was the store for every
  test, so a machine with the released `components.cif` ran the CCD bond-atom
  check that CI (no dictionary) skips. Two tests named atoms their residues do
  not have -- `SG` on selenomethionine and `C1` on ATP in `tests/test_job.py`,
  `OG` on threonine in `tests/test_input_ergonomics.py` -- and failed only
  there. An autouse fixture now points `FOLDJAX_HOME` at a per-session tmp
  directory and unsets `PROTENIX_CCD_COMPONENTS_FILE`,
  `PROTENIX_CCD_RDKIT_MOL_FILE` and `PROTENIX_TEMPLATE_MMCIF_DIR`; the atoms are
  corrected, and the bond test also runs against a two-component dictionary.

### Added

- **`real_store` test marker and fixture** for tests gated on released assets.
  `ccd_components`, `alphafold3_runtime`, and the `cpu_parity` and
  `official_parity` markers opt in without it.
