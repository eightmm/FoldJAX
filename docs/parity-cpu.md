# The CPU parity regression subset

`tests/parity/` replays stored native captures on CPU, so that a change to a
port's core is caught by a test rather than by the next GPU panel. This
document says what that certifies, what it does not, where the fixtures live,
and why the subset does not run in the CI job that exists today.

The scaffold is in place; the port cases are not. What ships right now is the
marker, the gate, the manifest schema and loader, the fixture resolver and the
fetch helper. Each port adds its own manifest and its own replay test.

## What a passing run certifies -- and what it does not

**Core only.** The fixtures are pre-featurized model inputs plus the native
tape. A test that loads `foldjax-input.npz` and replays it exercises the
model core, not the featurizer: a featurization regression is invisible to it
by construction. Featurizer coverage stays where it is, in the per-port
featurize suites and in the native-input audits described in `PROJECT.md`.

**A third implementation, not the panel.** Every residual recorded in a GPU
panel is a GPU port -- cuEq triangle kernels, bf16 where the port uses it --
against a GPU native run. CPU XLA has no tensor-core bf16 and materialises
converts that the GPU fuses, and fusion decisions are backend-specific. A CPU
replay is therefore a third implementation of the same arithmetic, and its
residual against the same stored capture is its own number. Tolerances in the
manifest come from a CPU calibration run of that exact fixture, recorded with
its wall time and source commit; `tests/parity/_manifest.py` refuses an entry
whose tolerance is below its own calibration residual, and refuses a zero
tolerance (chunk and scan choices are arithmetically equal, not bitwise).

**A regression detector, not an accuracy claim.** The subset answers "does this
checkout still reproduce the run we captured", not "is this model right".
Publisher parity remains the GPU panels' job.

**Only where a capture exists.** A fixture is bound to the capture that
produced it: the manifest records the capture's `provenance.json` path, its
commit, and a `tripwire` naming the tape/featurizer schema. When the checkout
reports a different schema the test fails loudly and says to re-capture -- it
must never fall back to re-featurizing, which would compare the checkout
against itself.

## Two tiers

| tier | what it compares | cost |
| --- | --- | --- |
| **A -- trunk boundary** | port trunk outputs (`s`, `z`, `s_inputs`, ...) against the stored native trunk capture | minutes; the only tier with a plausible CI budget |
| **B -- full tape replay** | coordinates after replaying the whole stored tape (noise, MSA cycles, dropout, augmentation) | measured at roughly 2-12 min per port per case at 76-118 tokens on 4 pinned cores; nightly |

Tier A needs a stored trunk capture. Protenix, OpenBind and Boltz-2 have one in
the master captures; OpenDDE (`raw.npz` holds heads and coordinates) and
ESMFold2 (`tape.npz` holds only the initial pair state) do not, so those two
need a new GPU capture before they get a tier A entry.

Tier B slices to one sample where the port's model API accepts sliced noise
(`sigmas` unchanged, so the tape stays valid). OpenBind's tape is a torch RNG
call log and is not sample-indexed: it replays at n5 or not at all.

## Running it

```
pytest --run-cpu-parity tests/parity          # the subset, fixtures required
pytest -m cpu_parity --collect-only tests/parity   # what the subset contains
pytest -q tests/parity                        # scaffold tests only; the subset is deselected
```

The subset is **deselected by default** by a collection hook in
`tests/parity/conftest.py` -- not by `collect_ignore`, which is the mechanism
that kept 101 Boltz-2 tests from being collected at all, and not by `addopts`,
which `pyproject.toml` deliberately leaves empty. Deselected means imported,
collected and visible in `--collect-only`; only the running is withheld.

Selected, the tests **fail** on a missing or mismatched fixture. They do not
skip: a skipped parity test is indistinguishable from a passing one in a
summary line. The failure message carries the exact fetch command.

`tests/parity/test_gate.py` is not marked `cpu_parity` and therefore always
runs. It asserts, in a subprocess, that `-m cpu_parity` still collects
something, that a default run collects none of it, that every shipped manifest
validates, and that every test a manifest entry names is actually collected --
so a fixture cannot end up certified by a test that no longer exists.

## Where the fixtures live

The repository pack is 22 MB, its largest blob 5.4 MB, and `.gitattributes` is
empty: there is no LFS here. One case per port is roughly 58 MB at five samples
(15-20 MB sliced), so the `.npz` files are **not** committed. What is committed
is `tests/parity/manifest/<port>.json`: for every file, its sha256 and byte
count, plus the capture it came from, the GPU residual for context, the CPU
calibration, the tolerance and how it was set (schema:
[`tests/parity/manifest/README.md`](../tests/parity/manifest/README.md)).

The files themselves go under `$FOLDJAX_PARITY_FIXTURES/<port>/<case>/`,
defaulting to `/home/jaemin/non-project/optimizing/foldjax-bench/parity-fixtures`
-- the capture host's store. Set the environment variable anywhere else.

Fetching is by digest, out of the read-only capture directory:

```
python -m tests.parity.fetch --from <capture dir>/<case>/native-A
```

It matches on content, not on name (captures keep their own layout), compares
sizes before hashing anything (a capture directory also holds checkpoints and
490 MB chemistry assets), never writes under `--from`, and exits non-zero
listing every file it could not find. The resolver verifies size and sha256
again before any path reaches a test.

## CI placement is the maintainer's decision

Stated plainly, because the honest answer is that no home for this exists yet:

- The shipped CI job (`.github/workflows/ci.yml`) is a single 30-minute
  ubuntu-latest job that runs `pytest -q -m 'not network'` with coverage, on a
  runner that has **no model weights**; recent runs use 19-21 minutes of the 30.
- Every tier needs released weights (Boltz-2 4.1 GB, OpenDDE 2.6 GB, Protenix
  1.5 GB, OpenFold3 4.6 GB, ESMFold2 1.4 GB plus 25.4 GB of ESM-C unless the
  language-model embedding is injected from the capture).
- `PROJECT.md` forbids implicit weight downloads, so the weights would have to
  be a deliberate, cached step of whatever job runs this.

So the subset can only live in a **second job, a nightly, or a self-hosted
runner** with a weights cache. Which of those, and who pays for it, is a
maintainer call and is not decided here. Until it is decided, the subset runs
locally on demand, and the guarantee the repository actually carries is the
one `test_gate.py` enforces in the default job: the subset is still there, and
its manifests still validate.

Two further unknowns, so they are not discovered later: the 4-vCPU GitHub
runner multiplier over this 144-core host is unmeasured (plausibly 2-3x), and
tier B's per-port runtimes above come from single-shot timings taken while the
host was shared.

## Adding a case (for a port package)

1. Capture on GPU as usual; keep the capture directory intact.
2. Copy the minimal files into the fixture store by digest, then record them:
   `tests/parity/manifest/<port>.json`, one entry per (case, tier).
3. Run the fixture once on CPU and record `cpu_calibration` (residual, wall
   seconds, source commit). The loader rejects an entry without it -- a
   tolerance taken from a GPU residual is a tolerance for a different
   implementation.
4. Write the test in `tests/parity/`, marked `cpu_parity`, resolving the case
   through the `parity_case` fixture and asserting with the manifest's
   tolerance. Name it in the entry's `test_node_ids`; the gate checks it is
   collected.

```python
import pytest

pytestmark = pytest.mark.cpu_parity


def test_trunk_matches_capture(parity_case) -> None:
    case = parity_case("protenix", "protein_1ubq", tier="A")
    case.assert_tripwire({"tape_schema": current_tape_schema()})
    captured = numpy.load(case.path("trunk.npz"))
    ...
    assert residual <= case.entry.tolerance_value
```
