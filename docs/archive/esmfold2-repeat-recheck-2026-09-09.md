# ESMFold2 F1/F2 read-only repeat check

Collaborator jobs 1002/1003 exited 0. Artifacts
`esmfold2-jax-full-tape-F1-20260909` and
`esmfold2-jax-full-tape-F2-20260909` were compared without modifying them.

All metadata fields agree. Coordinate arrays (5, 3104, 3), all 14 confidence
arrays (including raw logits), and independent JAX language-model hidden states
are equal in dtype, shape and storage bytes. Maximum absolute differences are
zero. This establishes repeat agreement for this pair, not across all inputs
or a causal explanation of earlier instability.

Recorded scope is shared core features, independent JAX LM, no intermediate
injection, n5/seed101. Precision records FP32 checkpoint, BF16 LM/trunk and
highest matmul; conditioning remains `port_policy_not_native_verified`.
`full_model_admission` and `native_input_control` are unset. Therefore this
result is neither native/FoldJAX parity nor independent preprocessing proof,
and the process durations are not warm inference timings.

Verification: both completed queue entries and metadata inspected; all three
NPZ archives read and compared field-by-field, including raw byte equality.

## Bound native comparison

The existing tape reporter successfully compared F1 against its bound native
capture esmfold2-native-full-tape-20260907-5sak-a, checking reference manifest,
checkpoint identities, input/tape hashes and shared feature equality. System-fit
entity maxima are protein 0.42094314 A and ligand 0.11826316 A over five samples
(3073 valid atoms; stored arrays have 3104 slots). Both exceed 0.1 A: repeat
agreement does not close native parity. The legacy native capture lacks recorded
LM output, so independent LM comparison is unavailable; no model admission.
All 25 recorded ESMFold2 model-source hashes match the current model tree.
This is model-source identity, not identity of every shared helper or harness.
