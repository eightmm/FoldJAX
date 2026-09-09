# Optional OpenBind raw pLDDT output

`released_config(return_plddt_logits=True)` returns the already-computed
atom logits in `Prediction.plddt_logits`. Default is false/None. Atom cropping
includes the new field and NPZ writing preserves it. Requested atom logits count
toward the writer's actual byte budget and may displace pair-logit outputs;
the pre-inference pair budget is an estimate, not a persistence guarantee.

Verification:

- OpenFold3 CPU suite: 881 passed, 360 skipped, 25 warnings. Skipped native/GPU
  checks are not covered by this CPU result.
- Pinned-source full inference: five tests passed; additional direct off/on
  common-output equality regression passed separately.
- Pinned config audit: four passed after declaring the pre-existing
  FoldJAX-only `stop_after_inputs` field; initial omission failure retained.
- Writer budget follow-up: one passed separately; scoped Ruff/diff pass.
- Real-weight 3GCA, n5 fixed tape, jobs 1085/1086: 402 byte-identical autotune
  records; all eleven common outputs exactly equal between off/on and all
  same-callable repeats equal. Only the atom-logit output is added.

Evidence root label: `openbind-raw-plddt-20260909-0YkshM`,
`controlled-bridge.json`. This finite output-return bridge does not prove
native model parity, independent preprocessing or behavior across all shapes.
Native comparison still has nonzero confidence/structure differences.

Independent read-only review returned no blocking finding. Parent addressed
its budget-documentation and direct-comparison-test follow-ups and admits this
narrow code change locally; the review tool did not produce a formal gate
verdict. No whole-project release, private-kernel default or push is implied.
