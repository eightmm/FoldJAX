# Maintaining common model boundaries

This refactor is isolated on `refactor/common-interface`. Commit `2b4fc10` is a
local source snapshot of the preceding in-progress work, not part of the
refactor to merge. Review the commits after that snapshot. Integrating the
refactor requires the preceding interface changes to be present first.

## Where each rule belongs

| Concern | Owner | Backend responsibility |
| --- | --- | --- |
| Comma-separated selectors and `all` | `models/_representations.resolve` | Declare native representation names |
| Stage capability and single-archive validation | `backends/_representations.resolve_representations` | Call `Backend.resolve_representations` in execution/cache planning |
| Input/trunk result metadata | `backends/_representations.representation_result` | Finish native extraction, cropping and memory release first |
| Recycle count conventions and managed defaults | `sampling.RecyclePolicy` | Explicitly select a policy; keep unrelated native options local |
| Boltz/Protenix Tokamax triangle core | `models/_tokamax_attention` | Keep the existing native import paths through compatibility modules |

Full/trunk selectors retain historical behavior: `all` supersedes subsequent
selectors. Input-only requests validate all selectors, including those following
`all`, so a trunk representation cannot silently become an input representation.

Recycle policy preserves both interfaces: `ModelConfig.trunk_passes` counts total
passes, while `PredictionRequest.num_recycles` retains its existing convention.
Explicit native options remain native. Models without an override continue to
use checkpoint/configuration defaults. Generic backends do not receive a built-in
policy merely because they use the same name.

The shared early-stage result helper now carries ESMFold2's padded shape profile
into the public `PredictionResult`, matching its full path and OpenFold3. This is
the intentional metadata correction in an otherwise behavior-preserving refactor.

## Boundaries retained

Shared policy/kernel sources remain part of resume provenance after moving out
of model directories. Requested representation runs also bind their shared
resolver/archive modules; source changes must invalidate a previously saved run.

Native feature schemas, MSA pairing/selection, masks, atom/residue/structural-token
mapping, output cropping and streamed MSA ownership remain with each model.
Bucket selection, bounded JIT ownership and prepared weight sessions already
have shared implementations; this change does not add another runtime manager.

No speed or memory improvement is claimed. Existing kernel operations, defaults,
checkpoints and numerical precision are retained. CPU tests cover selector
validation, cache aliases, count translation, metadata propagation and the
Tokamax layout/mask/scale boundary. The Tokamax test substitutes the kernel call;
it does not claim GPU numerical parity.

## Verification (2026-09-08)

- Complete CPU suite: **5,740 passed, 426 skipped**, no failures (66 warnings).
- After final selector, alias and provenance refinements: **771 affected tests
  passed** (2 warnings), covering common APIs, all six stage routes, cache/resume,
  backend sessions, sampling and the shared attention boundary.
- Ruff, lock consistency and Git whitespace checks passed.
- Both shared Tokamax functions have exactly the same AST as the snapshot.
- Tests used a separate environment/runtime home, CPU affinity limited to two
  CPUs and reduced process priority. No GPU inference or benchmark was launched.

The complete suite preceded the final small refinements; those refinements were
verified by the final affected suite. Skipped tests and GPU/checkpoint validation
are not claimed as passing.
