### Fixed

- **`capabilities` and `plan` report the sampling each backend runs.**
  `ModelCapabilities.sampling_defaults` read the adapters' native tables, so
  AlphaFold 3 said 10 recycles where its adapter runs 3, OpenFold3 said 4
  where the neutral count is 3 (4 is its trunk passes), and Protenix's steps
  and recycles, ESMFold2's samples and steps and Boltz-2's MSA depth were
  null although each runs a definite value. Both now come from
  `Backend.sampling_resolution`, the translation a run takes: AlphaFold 3
  5/200/3/1,024, Boltz-2 1/200/3/16,384, ESMFold2 32/14/3/1,024 (its released
  `config.json`, or the named checkpoint's), OpenDDE 5/200/10/16,384,
  OpenFold3 5/200/3/1,024, Protenix 5/200/10/16,384 (the model variant's
  schedule, read off the model name). `plan`'s `sampling` shows the value a
  knob runs at rather than the one spelled where a port narrows it, and
  `sampling_source` says `checkpoint` with a value where the checkpoint
  decides. `docs/cli.md` now documents `sampling`, `sampling_source`,
  `generated_input` and `not_checked`, and `docs/model-interface.md`
  `sampling_defaults`.

- **A resumed random-seed run no longer says it drew its seed.** `--resume`
  takes the seed back from the finished run's manifest, but the "has no
  upstream default seed; drew N" line printed anyway, beside the line saying
  the run was reused. It now prints only when the seed was actually drawn.

- **Boltz-2's MSA-server download is streamed under the 1 GiB ceiling.** The
  native client read the whole result archive into memory with no bound; it
  now streams it to disk and refuses more than `MAX_REMOTE_BYTES`, the cap the
  shared search client already applies. A body that breaks off mid-transfer is
  retried like a failed request, and a refused or failed download leaves no
  partial archive to be reused.

- **Protenix refuses a feature archive that cannot say which tokens are
  ligand.** An archive without `token_is_ligand`, `is_ligand` or
  `token_polymer_type` fell back to reading the unknown residue type as
  ligand identity, which every modified residue shares, so chain pTM/ipTM
  scored modified polymers as ligands. `--features` with such an archive now
  stops before any weights load and asks for the job to be re-featurized;
  a trunk-only or confidence-free run, which reads no ligand identity, is
  unaffected.

- **OpenFold3 no longer calls a nucleic-acid or ligand input a fit.** Its peak
  law keys on the token count alone and every point it was fitted on is
  protein-only; 5NPK (DNA gyrase with DNA and ligands, 3,061 tokens) peaked at
  41,260 MiB against the law's 29,876 MiB upper estimate. A run with any real
  non-protein token is now admitted as `unknown` with that reason named, the
  way a padded or float32 run already was; a refusal still binds.

### Changed

- **One job-name rule for every output path.** FoldJAX's layout, the
  Protenix/OpenDDE original-style tree and OpenFold3's native files each
  sanitized a job name their own way; all three now use
  `foldjax._fsutil.safe_job_name`. A name of ASCII letters, digits, `_`, `.`
  and `-` is written exactly as before. What moves: Protenix and OpenDDE keep
  non-ASCII letters (a Korean or Greek job name used to collapse to
  `prediction`, so two such jobs shared one directory) and shorten names over
  120 bytes with a digest; OpenFold3 sanitizes a name with a separator,
  whitespace or a control character instead of failing the finished
  prediction at the writer.

- **Duplicated helpers are one each.** Device identity is
  `cache.device_identity` (unchanged, so every compile-cache namespace keeps
  its digest), and the AlphaFold 3 runner and Boltz-2 parameter sessions key
  on `cache.device_key` built from it instead of their own attribute lists.
  AlphaFold 3's parameter filename families are one table,
  `assets.AF3_PARAMETER_PATTERNS`, read by both the readiness check and the
  adapter's replay of upstream's selector. Boltz-2's feature-cache digest
  streams referenced files through `_fsutil.update_digest_from_file`, with the
  key byte-for-byte what it was, and its representation-archive notice is a
  progress line instead of a `print` to stdout.
