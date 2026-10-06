## Unreleased

### Fixed

- **Protenix and OpenDDE now pair a searched heteromer's alignment.** Under
  `--msa auto` they used to read each chain's own ColabFold `paircomplete`
  alignment as `pairedMsaPath`. Those rows are not aligned across chains, and
  their `>UniRef100_<accession>\t<scores>` headers carry no species for the
  featurizer's `_UNIREF_REGEX`, so a heteromer was paired on the query row
  alone. `msa_pairing="model"` now runs the one ColabFold `pairgreedy` search
  over the complex that Protenix 2.0.0's ColabFold mode and OpenDDE submit
  (`complete` runs `paircomplete`; both accepted, no longer refused). Each
  chain's block is written to `msa/entity_NNNN_pairing.a3m` the way upstream
  Protenix writes it (`>UniRef100_<accession>_<row>/...`), so the species is
  the row number and row *i* of every chain is paired. Blocks of different
  depths are refused, and a caller's `paired_msa` is passed through untouched.
  This changes predictions: heteromers gain paired rows. A monomer or homomer
  no longer reads a per-chain pairing alignment, as in upstream Protenix's
  ColabFold mode. Upstream OpenDDE keeps the server's headers and so pairs
  nothing; FoldJAX gives OpenDDE Protenix's reading.

### Changed

- `foldjax_run.json` `msa_pairing` records `paired_by` (`row`, `species` or
  null) beside requested, resolved and mode.
- AlphaFold 3 keeps its per-chain alignment and still refuses
  `greedy`/`complete`. The docs now say that this alignment pairs no rows of a
  heteromer either: its ColabFold headers carry no UniProt species.
