### Fixed

- **A resumed random-seed run no longer says it drew its seed.** `--resume`
  takes the seed back from the finished run's manifest, but the "has no
  upstream default seed; drew N" line printed anyway, beside the line saying
  the run was reused. It now prints only when the seed was actually drawn.
