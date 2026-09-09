# Upstream Protenix applies MC dropout at inference; the port applies none

Independent reading of the pinned upstream tree
(`upstream-default-n5-20260904/upstream-root/protenix`), prompted by noticing
that another session had begun adding pair-dropout mask plumbing to the port.
This record is the evidence for *why* that is the right direction, established
without reference to that work.

## What upstream does

`protenix/model/protenix.py`, in `main_inference_loop` -- the inference entry
point, not a training path:

```python
# line 404 and line 440
mc_dropout=random.random() < mc_dropout_apply_rate,
```

and the flag lands here, in `get_pairformer_output`:

```python
# lines 240-244
if mc_dropout:
    z = z_init + F.dropout(
        self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z)),
        p=self.configs.mc_dropout_rate,
    )
```

Both rates come from the **released base config**, not a dataclass default:

```python
# configs/configs_base.py, lines 109-110
"mc_dropout_apply_rate": 0.4,
"mc_dropout_rate": 0.4,
```

So at inference, upstream enables Monte-Carlo dropout on roughly 40% of
recycling passes, and on those passes drops 40% of the recycled pair
projection. `torch.nn.functional.dropout` defaults to `training=True`, so eval
mode does not suppress it -- the `mc_dropout` flag is the only gate, and it is
open 40% of the time.

The gate is drawn from Python's **global** `random` module, not from the model
seed, so which cycles get dropout varies run to run unless a caller seeds
`random` explicitly.

Note the separate mechanism that does *not* apply: the pairformer's own
`DropoutRowwise(0.25)` goes through `dropout_add_rowwise(..., self.training)`
(`modules/pairformer.py:178,187,189`) and is correctly inert at inference. Only
the MC path is live.

## What the port does

Nothing. Grepping the committed port for `dropout` outside the ESM feature
builder returns two docstrings that say so explicitly:

```
models/protenix/models/trunk_blocks/trunk.py:107
    """Apply Protenix root-level recycling projections without dropout."""
models/protenix/models/trunk_blocks/embedders.py:368
    """Apply inference-mode MLP ``SubstructureEmbedder`` without dropout."""
```

`data/esm.py`'s `token_dropout` is ESM's masking convention, a deterministic
feature transform, and unrelated.

## Why this matters to the cross-model standard

It reclassifies Protenix-v2's residual. The standard records 0.1611 A on 5SAK
and 0.0335 A on 7st3 and reads them as accumulated arithmetic. They cannot be
only that: one side of the comparison randomly zeroes 40% of a recycled pair
representation on 40% of its cycles and the other side never does.

This puts Protenix in the same class as ESMFold2, for the same reason and with
the same consequence: a single-run port-versus-native number mixes a real
implementation difference with a stochasticity the port does not share, and no
amount of precision work moves it. Either the masks are matched -- which is what
the mask-injection plumbing in flight is for -- or the comparison is made over
distributions with both sides replicated, as was done for ESMFold2 here.

## What this record does not establish

Whether the closure harness's native arm actually routes through
`main_inference_loop` rather than a lower-level entry, and whether any caller
seeds Python's `random` before it. Both are checkable and both change how the
existing Protenix panel numbers should be read. They are named here rather than
guessed.

---

# Correction, same day: the harness already excludes the dropout runs

The section above says this "reclassifies Protenix-v2's residual" and that one
side of the comparison drops 40% of a pair representation while the other never
does. That is wrong for the panel numbers, and `bench/protenix_closure_capture.py`
says so in code I had not read before writing it.

The capture instruments the very call site I quoted and refuses the run when the
flag came up true:

```python
# line 379-381
recorder.mc_dropout = bool(bound.arguments["mc_dropout"])
if recorder.mc_dropout:
    raise ValueError("native selected MC dropout; mask tape is not captured")
```

and the completion check asserts it afterwards:

```python
# line 226-229
if self.mc_dropout is not False or len(self.random_decisions) != 1:
    raise ValueError("MC dropout decision is missing or its mask tape is unsupported")
```

So every captured native arm is one where `random.random()` came up at or above
0.4 and MC dropout did **not** apply. The port having no dropout is the correct
behaviour for those comparisons, and the existing Protenix panel numbers are not
contaminated by an unshared stochasticity. Nothing about 0.1611 A on 5SAK is
explained by this.

## What survives, and it is not nothing

Upstream's shipped inference really does enable MC dropout on roughly 40% of
recycling passes -- `configs_base.py:109-110`, `protenix.py:404,440,240-244`,
with `F.dropout` defaulting to `training=True`. That part of the reading holds.

What follows is a **coverage** statement rather than an accuracy one: the port
implements upstream's no-dropout path and cannot reproduce the other one, and
the audit harness makes that invisible by discarding any native run that takes
it. A user running upstream Protenix gets the dropout path 40% of the time; a
user running the port never does. Whether that matters is a product question
about what the port promises, not a parity defect in what has been measured.

## The lesson, which I had already written down today

Earlier in this session I recorded, after doing the same thing to the AlphaFold 3
config gate: when something looks like a systematic defect, grep for the
allowlist or the guard before calling it one. I read upstream carefully and the
harness not at all, and wrote a reclassification of another model's residual on
that basis. The claim never reached the cross-model standard, which is the only
reason this correction is cheap.
