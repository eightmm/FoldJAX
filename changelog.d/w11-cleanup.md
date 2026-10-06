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
