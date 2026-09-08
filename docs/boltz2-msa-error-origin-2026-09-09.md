
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

## The kernel choice is also already the better one

The native capture's own settings record `use_kernels: True`, with
`cuda_autocast_enabled: True` at bfloat16 and `float32_matmul_precision:
highest`. So both sides run fused kernels -- but different ones: upstream its own
Triton, the port cuEquivariance.

Measured at the shipped configuration, as an argument rather than a patch:

| Triangle backend | RMSE against native `delta_z` |
| --- | ---: |
| `cueq` (shipped) | 3.233621e-02 |
| `xla` | 3.270342e-02 |

The fused kernel is 1.1% *closer* to upstream's Triton kernel than the unfused
XLA path. An earlier arm found these equal, but it ran at a float32-parameter
baseline that is not the shipped configuration; at the right baseline they
separate, and the port's default is the better of the two.

## Terminus

Every axis reachable from this port is closed, each on an arm that either passed
through the module's signature or carried a positive control in the same design:

- configuration knobs, parameter dtype, activation dtype, chunk width
- every explicit divergence found by reading upstream's three operators
- the choice of fused kernel

None of them reduces the residual, and the two that move it at all -- lower
matmul precision, blanket bfloat16 -- move it the wrong way.

What is left is not a choice the port exposes. It is the difference between two
*different fused kernel implementations* computing the same operator, and closing
it means reproducing upstream's Triton semantics rather than selecting among the
port's existing paths. That is implementation work of the kind already under way
elsewhere in this repository, not another arm of this harness.

**The cell stands at 1.1761 Å.**
