
## The triangle-bias upcast is excluded too, and the source reading is exhausted

Upstream's triangle attention carries an explicit rule for the fused-kernel
path: `if use_kernels and triangle_bias.dtype is not torch.float32:
triangle_bias = triangle_bias.float()`. The run uses the cuEquivariance kernel,
and the port hands `tri_bias` to it without that upcast.

| Arm | RMSE against native `delta_z` |
| --- | ---: |
| shipped | 3.227007e-02 |
| upstream's placement: bias upcast to float32 for the kernel | 3.227007e-02 |
| **control: bias doubled** | **1.664130e+00** |

The control moves the result fifty-two fold. The arm does not move it at all.
Under this configuration the bias already arrives in float32, so upstream's
guard is a no-op here rather than a divergence that matters.

### What the source comparison found, in total

Three operators read against upstream and every explicit divergence measured
with a control in the same design:

| Operator | Finding |
| --- | --- |
| Outer product mean | faithful, including the chunked branch upstream takes above 384 tokens: hidden-axis splitting at width 4, autocast-dtype einsum, accumulation across splits, deferred bias |
| Triangle multiplication | `contraction_precision` defaults to float32, matching upstream's `x.float()` before the einsum |
| Pair weighted averaging | softmax dtype differs in source; inert (control moved 100x) |
| Outer product mean, validity count | dtype differs in source; inert (control moved 42x) |
| Triangle attention, kernel bias | upcast differs in source; inert (control moved 52x) |

The port follows upstream closely enough that every difference visible by
reading is either already matched or does not reach the output on this input.
The residual survives all of it.
