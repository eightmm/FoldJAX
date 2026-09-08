# OpenDDE's rerun floor swallows the arm that was meant to judge the pin

Status: a control result. It does not admit or reject the precision pin on
numerical grounds, and that is the finding.

## What was being tested

The [precision pin](../src/foldjax/models/opendde/cli/predict.py) states
upstream's TF32 rather than inheriting whatever JAX defaults to. A GEMM-level
probe had already shown unset and `"high"` are bitwise equal on this card
across a square, a thin and a small shape, so the expectation was an end-to-end
arm that reproduced the unpinned coordinates.

## What the control says

Three `bench.run_foldjax` runs of `opendde` on `t0128`, seed 101, five samples,
same weights, same case. Maximum per-atom displacement, no fit:

| Comparison | Maximum | Per sample | Bitwise |
| --- | ---: | --- | ---: |
| Same source, run twice | **0.5220 Å** | 0.098, 0.105, 0.522, 0.045, 0.079 | 0/5 |
| Pinned against unpinned | 0.7414 Å | 0.125, 0.610, 0.741, 0.423, 0.112 | 0/5 |

The two comparisons are the same order. **A single pair of runs at this case
cannot resolve the pin's effect**, and would not have resolved an effect four
times smaller either.

Running the port twice from one unchanged tree already moves an atom half an
angstrom and reproduces nothing bitwise. That is the measurement instrument,
before any change is applied to it.

## Why this matters beyond the pin

The cross-model gate records OpenDDE at 0.0728, 0.0957 and 0.0725 Å on its
three worst cases. Those are an order of magnitude *below* the floor measured
here. They are not smaller errors than this arm's — they are measured a
different way. That gate uses matched tapes, which removes the sampling
draw from the comparison; this harness does not.

So the two numbers are not comparable, and neither is wrong. What is wrong is
using this harness to judge a numerical change: an ordinary-RNG arm at one case
has no power against anything under an angstrom on this model.

Anything asserting an OpenDDE numerical difference needs a matched tape, or
enough repeats to state a floor with the change measured against it. The
project's own practical gate is written that way -- `max(0.05 Å, 3 × R_native)`
over at least three native runs -- and this is what it is protecting against.

## Verdict on the pin

Kept, on the evidence that actually bears on it: at the GEMM level the pin is
bitwise inert on this card, and its justification is that the port stops
depending on a process-global default it never stated. The end-to-end arm is
reported here as unable to confirm or contradict that, rather than quietly
dropped for saying the inconvenient thing.

No claim is made that the pin improves agreement with upstream. That would need
a native OpenDDE reference on this case, which this branch does not have.
