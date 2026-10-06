### Fixed

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
