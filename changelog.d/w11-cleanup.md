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
