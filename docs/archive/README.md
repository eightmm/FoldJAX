# Archived engineering notes

Dated measurement reports, panels and plans from the port-validation work of
September 2026. They are kept as the record behind statements in the live
documentation ([model-versions.md](../model-versions.md),
[engineering-notes.md](../engineering-notes.md),
[benchmark.md](../benchmark.md), [cli.md](../cli.md)) and in source comments,
not as user documentation: each describes the tree on the day it was written,
and later changes are not folded back into it. Start from the
[documentation index](../README.md) instead.

Ledger rows the notes cite are in [`docs/EXPERIMENTS.jsonl`](../EXPERIMENTS.jsonl)
and `bench/experiments/`.

| note | subject |
|---|---|
| [af3-closure-2026-09-06.md](af3-closure-2026-09-06.md) | AlphaFold 3 native-precision closure (nine-case panel) |
| [af3-warm-current-2026-09-09.md](af3-warm-current-2026-09-09.md) | AlphaFold 3 current-source warm comparison |
| [benchmark-followup-2026-09-05.md](benchmark-followup-2026-09-05.md) | Fresh multimodal upstream comparisons |
| [boltz2-master-kernel-toggle-2026-09-09.md](boltz2-master-kernel-toggle-2026-09-09.md) | Boltz-2: upstream's own kernel toggle as the scale for the 5SAK residual |
| [boltz2-msa-error-origin-2026-09-09.md](boltz2-msa-error-origin-2026-09-09.md) | Boltz-2: origin of the MSA-stage error; triangle-bias upcast excluded |
| [boltz2-upstream-msa-deletion-regression-2026-09-10.md](boltz2-upstream-msa-deletion-regression-2026-09-10.md) | Boltz-2 upstream zeroes every MSA deletion feature (v2.2.0+); the port reproduces it |
| [boltz-5sak-boundary-followup-2026-09-05.md](boltz-5sak-boundary-followup-2026-09-05.md) | 5SAK boundary isolation and other-model status |
| [boltz-conditioning-fma-2026-09-07.md](boltz-conditioning-fma-2026-09-07.md) | Boltz conditioning: native CUDA normalization arithmetic |
| [boltz-mixed-precision-audit-2026-09-05.md](boltz-mixed-precision-audit-2026-09-05.md) | Boltz-2 mixed-precision boundary audit |
| [boltz-msa-transition-2026-09-07.md](boltz-msa-transition-2026-09-07.md) | Boltz MSA transition: matched-native-input normalization control |
| [boltz-native-amp-2026-09-07.md](boltz-native-amp-2026-09-07.md) | Boltz-2 native AMP repair |
| [boltz-trunk-pair-norm-2026-09-07.md](boltz-trunk-pair-norm-2026-09-07.md) | Boltz trunk: matched-input operator localization |
| [closing-plan-2026-09-10.md](closing-plan-2026-09-10.md) | Closing plan for the 2026-09-10 validation window |
| [cueq-first-validation-2026-09-09.md](cueq-first-validation-2026-09-09.md) | cuEq-first validation decision |
| [esmfold2-master-panel-2026-09-09.md](esmfold2-master-panel-2026-09-09.md) | ESMFold2: seven cases, native floors, kernel-selection noise |
| [esmfold2-msa-entry-observer-2026-09-09.md](esmfold2-msa-entry-observer-2026-09-09.md) | ESMFold2 MSA-entry observer and matched-tape corrections |
| [esmfold2-native-tape-2026-09-07.md](esmfold2-native-tape-2026-09-07.md) | ESMFold2 native tape implementation progress |
| [esmfold2-repeat-recheck-2026-09-09.md](esmfold2-repeat-recheck-2026-09-09.md) | ESMFold2 read-only repeat check |
| [four-way-precision-followup-2026-09-06.md](four-way-precision-followup-2026-09-06.md) | Four-way precision follow-up |
| [interface-inventory-2026-09-09.md](interface-inventory-2026-09-09.md) | Interface inventory surveyed for the unification task |
| [ion-case-1aay-master-2026-09-09.md](ion-case-1aay-master-2026-09-09.md) | The ion-containing case (1AAY) added to the multimodal panel |
| [master-parity-summary-2026-09-09.md](master-parity-summary-2026-09-09.md) | Parity summary, all models, 2026-09-09/10 |
| [model-closure-progress-2026-09-06.md](model-closure-progress-2026-09-06.md) | Sequential native-precision closure progress |
| [model-shapes-defaults-2026-09-08.md](model-shapes-defaults-2026-09-08.md) | Default tensor shapes per model (in Korean) |
| [native-precision-selection-results-2026-09-06.md](native-precision-selection-results-2026-09-06.md) | Native-first precision screening results |
| [openbind-augmentation-chunking-2026-09-07.md](openbind-augmentation-chunking-2026-09-07.md) | OpenFold3/OpenBind sample chunking and augmentation identity |
| [openbind-current-panel-2026-09-09.md](openbind-current-panel-2026-09-09.md) | OpenFold3/OpenBind native-default panel preflight |
| [openbind-independent-input-identity-2026-09-09.md](openbind-independent-input-identity-2026-09-09.md) | OpenFold3/OpenBind independent input identity |
| [openbind-master-cueq-panel-2026-09-09.md](openbind-master-cueq-panel-2026-09-09.md) | OpenFold3/OpenBind cuEq-versus-cuEq panel |
| [openbind-native-private-backend-2026-09-09.md](openbind-native-private-backend-2026-09-09.md) | Experimental OpenBind native-private dispatch |
| [openbind-ordinary-warm-2026-09-09.md](openbind-ordinary-warm-2026-09-09.md) | OpenBind ordinary-RNG warm inference |
| [openbind-private-panel-summary-2026-09-09.md](openbind-private-panel-summary-2026-09-09.md) | OpenBind private-kernel seven-case comparison |
| [openbind-raw-plddt-return-2026-09-09.md](openbind-raw-plddt-return-2026-09-09.md) | Optional OpenBind raw pLDDT output |
| [openbind-tf32-tie-rounding-2026-09-08.md](openbind-tf32-tie-rounding-2026-09-08.md) | OpenBind TF32 projection boundary |
| [openbind-transition-control-2026-09-09.md](openbind-transition-control-2026-09-09.md) | OpenBind transition rounding control |
| [opendde-2k-single-card-2026-09-24.md](opendde-2k-single-card-2026-09-24.md) | OpenDDE at 2,096 residues on one card |
| [opendde-bf16-layernorm-followup-2026-09-06.md](opendde-bf16-layernorm-followup-2026-09-06.md) | OpenDDE BF16 LayerNorm correction |
| [opendde-current-panel-2026-09-09.md](opendde-current-panel-2026-09-09.md) | OpenDDE current-source panel expansion |
| [opendde-master-panel-2026-09-09.md](opendde-master-panel-2026-09-09.md) | OpenDDE: seven cases, native floors, the TF32 policy decision |
| [openfold3-bf16-default-evidence-2026-09-12.md](openfold3-bf16-default-evidence-2026-09-12.md) | Evidence behind OpenFold3's BF16 default |
| [openfold3-streamed-msa-2026-09-08.md](openfold3-streamed-msa-2026-09-08.md) | OpenFold3 host-streamed MSA recycling |
| [precision-selection-protocol-2026-09-06.md](precision-selection-protocol-2026-09-06.md) | Upstream-first correctness and optimization contract |
| [preprocessing-contract-audit.md](preprocessing-contract-audit.md) | Independent preprocessing contract audit (2026-09-05 checkpoint) |
| [protenix-cueq-regression-2026-09-07.md](protenix-cueq-regression-2026-09-07.md) | Protenix BF16/cuEq regression and structural gray zone |
| [protenix-current-panel-2026-09-09.md](protenix-current-panel-2026-09-09.md) | Protenix panel and MC-dropout investigation |
| [protenix-master-panel-2026-09-09.md](protenix-master-panel-2026-09-09.md) | Protenix: native floors first, then the port read three ways |
| [protenix-warm-measurement-preflight-2026-09-09.md](protenix-warm-measurement-preflight-2026-09-09.md) | Protenix warm harness preflight |
| [public-model-panel-2026-09-09.md](public-model-panel-2026-09-09.md) | Six-model default-path comparison |
| [scale-rows-master-2026-09-10.md](scale-rows-master-2026-09-10.md) | FoldJAX against upstream at 1k-5k tokens |
