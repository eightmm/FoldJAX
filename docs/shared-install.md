# Shared installation

Use one `FOLDJAX_HOME` for an administrator's managed weights, generated runtime,
chemistry assets, and device-specific compilation caches. Inspection does not
create directories or change permissions:

```bash
export FOLDJAX_HOME=/path/to/shared/foldjax
foldjax setup --shared
foldjax setup --shared --json
foldjax setup --shared --apply
foldjax doctor --json
```

Run `--apply` as the owner or administrator. The parent of `FOLDJAX_HOME` must
already exist. The command visits only managed paths; it does not download
weights, build a runtime, compile kernels, or modify unrelated files in the home.

| Managed path | Directories | Files |
| --- | --- | --- |
| Home, `weights/`, `runtime/`, `assets/`, `downloads/`, `compile/` | 0755 | 0644 |
| `templates/` | 0755 | — |
| `msa/`, `templates/hits/`, `templates/mmcif/` publication roots | 1777 | — |
| Existing entries below those publication roots | 0755 | 0644 |

Runtime executables retain execute permission and become 0755. Other regular
files become 0644. The compilation tree includes older
`.tokamax-autotuning-v2` directories and records created as 0700/0600; `--apply`
repairs these to 0755/0644. A reader can reuse measured tuning and existing XLA
artifacts without publishing new compilation results. The existing trust policy
accepts a root-owned cache. Readers of a cache owned by another ordinary account
need to trust that owner and set `FOLDJAX_TRUST_SHARED_COMPILE_CACHE=1`;
`foldjax doctor` reports the compilation-cache trust decision.

Predictions read executable entries from that trusted shared namespace first,
then from the same model/weights/runtime namespace in the user's private cache.
The private root is `$XDG_CACHE_HOME/foldjax/compile`, or
`~/.cache/foldjax/compile` when `XDG_CACHE_HOME` is unset. Set
`FOLDJAX_PRIVATE_COMPILE_CACHE` to put it on another disk, for example when the
shared home filesystem is full. `--cache-dir` selects the shared candidate;
it does not hide the private fallback. `--no-cache` disables both tiers.

A shared hit is used in place. New entries are published to shared storage when
possible, otherwise to the private namespace, including when the requested
shared namespace does not exist or an actual lock/write fails. Private
namespaces must be owned by the current uid with mode 0700, and private files
use 0600. Shared-cache trust never relaxes those checks. Unsafe private paths
are refused rather than repaired automatically. If neither tier can persist
an entry, prediction can still compile and run without saving it. AF3's Tokamax
tuning configurations follow the same read and publication order; only complete,
validated configurations are persisted. XLA's auxiliary autotuning caches bypass
the JAX cache interface, so layered scopes disable those auxiliary caches to
keep native writes out of read-only or full filesystems. JAX executable and
Tokamax configuration caching remain active.

`foldjax_run.json` records JAX activity in `compilation_cache`: `locations`,
per-tier byte `lookups`, successfully deserialized executable `loads`, completed
backend `compiles`, and per-tier `writes`. A byte lookup alone does not prove
an executable was loaded. JAX in-memory cache hits are reported as unobserved.
AF3's `kernel_tuning.calls` separately records tuning sources and shared/private
storage activity. The warm summary includes these records; its older `cache`
directory snapshots describe the shared namespace only.
Each seed's manifest records its own execution. A multi-seed aggregate keeps
the last executed seed's telemetry; an all-resumed request does not attribute
the previous request's activity to the resumed run.

Hidden temporary directories directly under `runtime/cache-warm/` are excluded
from inspection and repair. They contain unfinished predictions and stay 0700;
their count appears as `temporary_directories` in the runtime report. Symlinks
and other hidden files remain subject to the normal safety checks. Warming
removes its scratch output after completion or an exception. With the default
SIGTERM policy, it also unwinds and removes that output before exiting with
status 143; an application's existing signal handler is preserved. SIGKILL or
a machine crash cannot run cleanup, so excluded remnants may remain private.

The sticky publication roots let every account create its own cache entries
while preventing it from deleting another account's entries. Published entry
directories keep their owners and are readable, rather than world writable.
Existing entries owned by another account may need their owner or administrator
to repair them. Symlinks, mounted filesystems below the home, special files, and
multiply linked files needing a permission change are reported and left alone.
Symlinked home paths and ancestors are refused. Thus repair never follows a
link into a different installation or changes a hard-linked outside file.

A trusted Unix group can share write access instead:

```bash
foldjax setup --shared --group GROUP
foldjax setup --shared --group GROUP --apply
export FOLDJAX_TRUST_SHARED_COMPILE_CACHE=1
```

Group mode uses 2775 directories and 0664 files, with 0775 runtime executables,
and changes the group while retaining each entry's owner. The caller must be a
member of the requested existing group or an administrator. Group validation
happens before any changes. All accounts using the group-writable compilation
cache need the trust setting because that cache contains executable code.

Repair changes existing entries. Use `umask 022` for future owner-published
files, or `umask 002` for group writers, so newly generated files retain the
intended read and group-write permissions.

`foldjax doctor` reports whether the home's owner differs from the current user,
read/write access to each managed root, owner-only files, and the features
blocked by permission problems. It recognizes group mode from a setgid home.
Its JSON `shared_install` report includes issues, skipped paths, and per-root
counts; both doctor and ordinary shared setup remain read only.
