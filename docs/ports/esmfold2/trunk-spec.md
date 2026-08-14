# ESMFold2 trunk: what the torch source actually does

Read off `transformers/models/esmfold2/` at the commit vendored in this
environment, and the reference the JAX port is written against. Kept because
the port's correctness argument is "matches this", and the source lives in a
fork that will move.

## The trunk is a linear recurrence, not AF2 recycling

`_run_one_loop` runs `num_loops + 1` iterations (21 by default). Only `z` is
carried; `z_init`, `lm_z` and `x_inputs` are loop-invariant and re-injected
every step:

    a      = exp(-softplus(parcae_log_delta) * exp(parcae_log_a))     # [1,1,1,256]
    b_mat  = softplus(parcae_log_delta)[:, None] * parcae_b_cont      # [256,256]
    z      = trunc_normal(0, sqrt(2/(5*256)), +/-3 sigma)             # random start
    repeat:
        inject = z_init (+ lm_z when there is no lm_encoder)
        inject = msa_encoder(inject, ...) if msa else inject          # overwrite by default
        inject = inject + lm_encoder(lm_z)                            # 4-block FoldingTrunk
        z      = a * z + parcae_input_norm(inject) @ b_mat.T
        z      = folding_trunk(z, pair_mask)                          # 24 blocks, overwrite

There is no recycled single representation and no recycled coordinates.

## Inference is stochastic by design

Three sources, all active with `model.eval()`:

* `_init_pair_state` draws `z` from a truncated normal every call;
* LM dropout is `F.dropout(..., training=True)` -- forced, not a bug -- at
  `lm_encoder.lm_dropout`, default 0.25;
* the MSA is column-masked once (rate 0.1) and row-subsampled *per iteration*
  to `msa_max_depth` (1024).

A port that produced one fixed answer would not be reproducing this model.

## Traps found while reading

* `TriangleMultiplicativeUpdate` wraps a `TriangleMultiplicativeBlock` in an
  attribute named `_engine`, so every checkpoint key carries that level.
* `proj_bundle` packs four blocks of `c_h`: `[left | right | gate_left |
  gate_right]`. The input gate multiplies, *then* the mask applies, *then* the
  contraction runs in float32 (`routed.float()`), and the output gate comes
  from the same normalised input as the bundle.
* Two transitions with the same shape and different semantics:
  `common.Transition` has the residual **inside**; `modeling.PairTransition`
  does **not**.
* `OuterProductMean` divides *after* the projection by default
  (`Wout(outer) / n_valid`, so the bias is scaled too). The experimental
  module divides before, with the same key names.
* `MSAEncoder`'s final block has no `msa_pair_weighted_averaging` and no
  `msa_transition` -- those keys are absent, not zero.
* `ResIdxAsymIdSymIdEntityIdEncoding` concatenates
  `[rel_pos(66), rel_token(66), same_entity(1), rel_chain(6)]` -- the scalar
  sits between the one-hots -- and the chain branch is inverted relative to
  the other two.
* `ConfidenceHead` treats its `FoldingTrunk` output as a **delta**
  (`pair += trunk(pair)`); the main loop treats it as an overwrite.

## Status

Ported and gated. `tests/models/esmfold2/` holds 60-odd parity tests against
the torch modules these notes describe, plus an end-to-end run on the released
checkpoint. What this file records is what the torch source does; where the
port deliberately departs from it -- the PDB writer's chain field and b-factor
scale -- the departure is stated where it happens.
