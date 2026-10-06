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

- **The remote MSA client rides out a busy server.** A submission answered
  `RATELIMIT` or `UNKNOWN` is submitted again with backoff, as ColabFold's and
  Boltz's clients do, instead of polling a job that does not exist. A 5xx, a
  dropped connection or a truncated body (`http.client.IncompleteRead`, which
  escaped every caller's `except`) is retried up to five times with
  exponential backoff for searches and template-structure downloads alike; a
  failure that persists is a search error naming the server. Waits print a
  progress line instead of sitting silent, a timeout names the server, and the
  one-hour ceiling is `FOLDJAX_MSA_MAX_WAIT_SECONDS`.

- **Search caches survive concurrent runs and damaged entries.** Two runs on
  one sequence raced to publish, and the loser failed: renaming onto a
  non-empty directory raises `ENOTEMPTY`, which was not caught. Publishing
  now reads the winner's entry, and an advisory lock per cache key makes the
  second run wait and reuse the first one's search instead of asking the
  server again. An MSA, RNA-MSA or template-hits entry that fails its hash or
  completeness check used to fail every later run of that sequence; it is now
  moved aside as `.<key>.damaged` with a warning naming it, and searched
  again. Entries are published with the umask's mode instead of 0700, so a
  group-shared cache is readable by the group, and a failed mmCIF download no
  longer leaves its staging file behind.

- **Uncached sequences share the unpaired MSA ticket.** Twenty unique
  sequences cost forty serial tickets; their unpaired (`env`) searches now go
  to the server together, up to 16 per ticket, as ColabFold's and Boltz's own
  clients submit them, and each query's block is renumbered to the `>101` a
  single-query ticket writes, so the cached files are byte-identical either
  way (the server answering each query independently of its ticket-mates is
  taken from those clients, not re-measured). The per-chain paired search stays
  one ticket per sequence, since what it pairs depends on the ticket.

- **A failed MSA search under `--msa auto` is recorded and keeps what
  succeeded.** One chain's failure dropped every chain's alignment; each chain
  now keeps its own result, and a failed chain is recorded with its `error` in
  `msa_search.json` and in a new `msa_search` field of `foldjax_run.json`
  (`null` when no search was asked for; not part of the resume identity, so
  existing manifests still resume). In the CLI the failure, and a failed
  template search, is a progress line for every input instead of a Python
  warning, which the warning registry printed only for the first identical
  one in a batch; library callers still get the warning.

### Security

- **Boltz-2 loads its processed arrays and molecule pickles without arbitrary
  pickle.** `processed/` structures, MSAs and constraints are read with
  `allow_pickle=False` (the absent `pocket` field is no longer stored as a
  pickled `None`, and feature-cache entries written before this change miss
  once and are rebuilt). The CCD molecule pickles beside the weights and the
  `processed/mols/*.pkl` file preprocessing writes go through a restricted
  unpickler that admits only `rdkit.Chem.rdchem.Mol`, as ESMFold2's CCD reader
  already did.
- **The persistent compile cache is used only when no other account can write
  it.** JAX runs a cached executable as found. The cache directory and every
  ancestor must belong to the user (or root) and be writable by neither the
  world nor a group with another member; otherwise FoldJAX warns once and
  compiles without a persistent cache. New cache directories are created
  without group write under any umask. A deliberately shared store opts in
  with `FOLDJAX_TRUST_SHARED_COMPILE_CACHE=1`. This applies to the request
  cache and to the Boltz-2, OpenDDE, OpenFold3 and AlphaFold 3 entry points
  that set the cache themselves. A group-writable, setgid store shared with
  another account (such as a lab's) now misses until that variable is set.
- **Remote MSA, template and structure servers must be https and are not
  followed through redirects.** `FOLDJAX_MSA_SERVER_URL`,
  `FOLDJAX_TEMPLATE_STRUCTURE_URL`, `--msa-remote-url` and Boltz-2's MSA server
  URL must be `https://`, plain `http://` being accepted for loopback hosts or
  with `FOLDJAX_ALLOW_INSECURE_HTTP=1`. A redirect is now an error instead of
  being followed: urllib resent `Authorization` and API-key headers to whatever
  host it named, `http://` included, and requests strips only `Authorization`.
  A server job id must be `[A-Za-z0-9_-]+` before it is put in a URL, and one
  response or result-archive member is capped at 1 GiB.
- **Search provenance and `foldjax doctor` redact the search setup.** A local
  search command's argv (`FOLDJAX_MSA_COMMAND`, `FOLDJAX_TEMPLATE_COMMAND`)
  and a server URL were written verbatim to the MSA and template caches'
  `provenance.json`, to `template_search.json` and the run manifest's
  `template_search`, and printed by `doctor`; a `--password`/`--api-key`
  argument or URL userinfo now reads `[REDACTED]` there. Cache identities keep
  the raw values, so existing cache entries still hit.
- **Generated files stay inside their output directory.** The
  `template_search/` directory that Protenix and OpenDDE template files are
  written to gets the symlink and containment checks `msa/` already had (a
  planted symlink now fails that chain's template search with the reason
  recorded); the OpenDDE CLI's `--stop-after inputs|trunk` representation
  directory sanitizes the native document's `name` as the structure writer
  does (`"../../x"` escaped `--out`); and an AlphaFold 3 build wheel member
  under `share/libcifpp/` cannot climb out of that directory.
- **Representation archives are written with `allow_pickle=False`**, so an
  object array is refused at write time rather than stored as a pickle.
- **A weight download stops at its registered size.** A server that sent more
  than the registry's byte count was written to disk until it stopped; the
  download now fails at the first byte past it and discards the prefix.
- **A cached template mmCIF is checked before it is used.** A downloaded or
  unpacked structure must name the requested PDB id in its data block (and
  parse, where gemmi is installed); one that does not is fetched or unpacked
  again, and a download naming another entry is refused before it is cached.
  The local mirror (`FOLDJAX_TEMPLATE_MMCIF_DIR`) stays trusted as is.
