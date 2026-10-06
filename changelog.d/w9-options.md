## Unreleased

### Added

- **Upstream run options FoldJAX lacked**, each as `--option`, at upstream's
  value when omitted and part of the compilation-cache identity when spelled
  (table in [cli](../docs/cli.md#upstream-run-options---option)):
  - Boltz-2: `step_scale`, `subsample_msa`, `num_subsampled_msa`, `method`
    (also keys the feature cache) and `use_potentials` (upstream's
    `--use_potentials` steering configuration, refused with `steering_args`,
    padding, `deterministic=on` and context parallelism).
  - Protenix and OpenDDE: `use_tfg_guidance`, training-free guidance with
    upstream's default guidance mapping on the eager, unrolled sampler;
    OpenDDE reuses Protenix's TFG port, whose upstream it matches. Both
    native CLIs gain `--use-tfg-guidance`; padding, deterministic reductions
    and context parallelism are refused at plan time.
  - AlphaFold 3: `resolve_msa_overlaps`, `fix_standalone_glycans` and
    `conformer_max_iterations` reach `featurise_input` on every route, and
    `ref_max_modified_date` follows `--template-max-date` when set.
  - ESMFold2: `lm_mask_pct`, `msa_column_mask_rate` and `full_depth_msa`.
- **Protenix profiles `base-constraint-v0.5.0`, `mini-default-v0.5.0` and
  `tiny-default-v0.5.0`**: upstream's three other public v0.5.0 checkpoints,
  each in its own storage root and pinned by FoldJAX's SHA-256 of one
  download (the publisher serves none).
