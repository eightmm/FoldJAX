# OpenBind transition rounding control

**Rejected for full-model use.** The wiring was reverted after the n5 5SAK
failure below. Only the explicitly callable diagnostic helper remains; the
active PairBlock file is byte-identical to the earlier seven-case snapshot.

The pinned upstream SwiGLUTransition uses ordinary Torch LayerNorm and Linear;
its SwiGLU call leaves `use_kernel=False`. A prior FoldJAX docstring incorrectly
implied automatic Triton activation dispatch on CUDA and is corrected.

The experimental control reuses native-faithful ordinary LayerNorm and TF32-RNE
GEMM helpers while retaining the existing JAX SiLU formula. It does not claim
that every remaining activation/accumulation instruction matches native.

On saved real first-PairBlock child inputs (job 1098):

| Transition error | Existing XLA transition | Norm/GEMM control |
| --- | ---: | ---: |
| RMSE | 0.000128406039 | 0.000017975815 |
| Maximum absolute error | 0.015590668 | 0.006374359 |

RMSE improves 7.143-fold. Other child controls remain unchanged: outgoing and
incoming multiplication exact; start/end attention RMSE 1.2644e-5/1.3693e-5.
All saved child output hashes and three-call repeat equality were verified.
Bundle: `openbind-transition-control-20260909-ArrMhC/candidate`.

The trial source native-private backend selected this transition while
cuEq/XLA defaults retained the existing implementation. Formula, rectangular
row/mask, dtype/CP rejection, dispatch and related block tests: 15 passed.
Ruff/diff checks passed. The next accumulated-block control is job 1099,
bundle `openbind-transition-block-20260909-NVhaES`; it uses explicit
`source-native-private` dispatch rather than the legacy operator monkeypatch.

The accumulated first PairBlock control (1099) exited zero: RMSE
0.000236986609 -> 0.000205283987 (1.154-fold improvement), while maximum error
remains 0.033325195. The output archive hash and three-call equality passed.
No native child outputs were injected into that block execution. Full-model
5SAK job 1100 is now queued/running with the original forward tape and n=5;
the historical autotune map is loaded but new entries are allowed, so compiler
choices are not yet a fully frozen single-variable control.

## Full-model rejection

Job 1100 completed all three calls but worsened 5SAK: protein maximum RMSD
1.660694551 Å and ligand 10.224526860 Å, versus 0.062239110/0.193974387 Å
before this trial. Public maximum differences are pLDDT 26.731286894 points,
pTM 0.030830536 and ipTM 0.098421067. All output hashes and repeat equality
passed; raw confidence fields are complete. Thus this is not an incomplete
output mistaken for a prediction. Local child/block improvements did not
translate into acceptable full-model behavior.

The source routing change was reverted, and `cmp` confirmed PairBlock bytes
match `openbind-source-backend-20260909-A3bJ4y`. A regression prevents dispatch
of the rejected transition helper under both XLA and native-private. Root
cause is not isolated: the helper's rounding policy, contexts outside the
observed block, and new compiler choices need discriminating controls before
any future admission. The failed snapshot and artifacts are preserved.

Earlier seven-case/warm results are evidence for their original snapshots,
not for this rejected trial. Review, commit, and push remain pending.
