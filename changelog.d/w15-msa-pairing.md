## Unreleased

### Fixed

- **Protenix now pairs a searched heteromer's alignment.** Under `--msa auto`
  it used to read each chain's own ColabFold `paircomplete` alignment as
  `pairedMsaPath`. Those rows are not aligned across chains, and their
  `>UniRef100_<accession>\t<scores>` headers carry no species for the
  featurizer's `_UNIREF_REGEX`, so a heteromer was paired on the query row
  alone. `msa_pairing="model"` now runs the one ColabFold `pairgreedy` search
  over the complex that Protenix 2.0.0's ColabFold mode submits (`complete`
  runs `paircomplete`; both are accepted, no longer refused). Each chain's
  block is written to `msa/entity_NNNN_pairing.a3m` the way upstream Protenix
  writes it (`>UniRef100_<accession>_<row>/...`), so the species is the row
  number and row *i* of every chain is paired. A caller's `paired_msa` is
  passed through untouched. This changes predictions: heteromers gain paired
  rows. A monomer or homomer no longer reads a per-chain pairing alignment, as
  in upstream Protenix's ColabFold mode, and neither does a heteromer searched
  by a local wrapper (`FOLDJAX_MSA_COMMAND`), which cannot pair a complex and
  now says so.
- **OpenDDE's default now runs what upstream OpenDDE runs**: the same
  `pairgreedy` complex search, with the blocks passed as the server wrote them
  (`opendde/data/msa/msa_service_client.py` `search_and_build_msa`). Its
  species re-pairing therefore pairs only the query row, as upstream's does,
  and the block's rows join each chain's unpaired stack
  (`msa_pair_as_unpair`); a monomer or homomer gets no pairing alignment, as
  upstream writes it a query-only `pairing.a3m`. The manifest records
  `paired_by: species`. An explicit `--msa-pairing greedy`/`complete` opts
  OpenDDE into Protenix's row reading (`paired_by: row`), unmeasured for
  accuracy against upstream.
- A complex pairing search whose blocks differ in depth is refused before it is
  cached (any model): `msa="required"` fails, `auto` folds without the pairing
  and warns.

### Changed

- `foldjax_run.json` `msa_pairing` records `paired_by` (`row`, `species` or
  null) beside requested, resolved and mode; `species` means no row beyond the
  query is paired.
- AlphaFold 3 follows its upstream, which has no reader of ColabFold output: it
  keeps its per-chain alignment and still refuses `greedy`/`complete`. The docs
  now say that this alignment pairs no rows of a heteromer either: its
  ColabFold headers carry no UniProt species.
