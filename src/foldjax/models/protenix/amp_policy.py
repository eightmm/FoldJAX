"""Which stages run under the bf16 autocast, written down instead of implied.

Upstream Protenix does not run one mixed-precision policy. Its inference
runner rewrites the policy from the token count before the model is built --
``runner/inference.py:492`` ``update_inference_configs`` -- and the two stages
it moves are the confidence head and the diffusion sampler:

    n_token > 3840          confidence bf16, diffusion bf16
    2560 < n_token <= 3840  confidence bf16, diffusion fp32
    n_token <= 3840         confidence fp32*, diffusion fp32

``*`` except ``protenix-v2``, whose confidence head is bf16 at every size.

Upstream spells it as ``skip_amp.<stage>``, which is the negation of what
happens: ``skip_amp.confidence_head = True`` means that stage runs with
autocast *disabled* -- its arguments cast to fp32 and its matmuls in fp32
(``protenix/model/protenix.py:344``, ``protenix/utils/torch_utils.py:199``).
This module states the positive form, ``<stage>_autocast``, because that is
what a reader has to know to predict a dtype.

The thresholds exist to avoid OOM, so a port that reproduces the trunk but not
the gate is not running the released model at every size: at 3,012 tokens
upstream's confidence head is bf16 and a port that widens everything to fp32
is measuring a configuration upstream never ships. The native captures record
the gate firing -- ``protenix-master-native-20260909-9yET4f`` has
``confidence_head`` skip_amp false at 3,012 tokens and true at 76.

Nothing here touches JAX: the CLI, the backend option table and the model all
resolve the same policy from the same function.
"""

from __future__ import annotations

from typing import NamedTuple

#: Accepted ``--amp-policy`` values. ``auto`` is the released default and
#: reproduces the upstream token gate; the other two pin one policy at every
#: size, which is what an A/B against the gate needs.
AMP_POLICY_CHOICES = ("auto", "fp32", "bf16")

#: The released default, and the value the backend strips from a cache
#: namespace: asking for it explicitly is the same run as not asking.
DEFAULT_AMP_POLICY = "auto"

#: Both bounds are exclusive upstream (``>``), so a job exactly at 2560 or
#: 3840 tokens keeps the lower stage's policy. The boundary tests pin that.
CONFIDENCE_AUTOCAST_ABOVE_TOKENS = 2560
DIFFUSION_AUTOCAST_ABOVE_TOKENS = 3840

#: Model variants whose confidence head is under autocast at every size.
#: Upstream writes this as the ``else`` of its ``n_token > 2560`` branch, so it
#: only changes the answer below the confidence threshold.
CONFIDENCE_AUTOCAST_ALWAYS_MODELS = frozenset({"protenix-v2"})


class AmpPolicy(NamedTuple):
    """Which of the two gated stages run under the bf16 autocast."""

    confidence_autocast: bool
    diffusion_autocast: bool

    def label(self) -> str:
        """A stable one-line spelling for logs and provenance."""
        confidence = "bf16" if self.confidence_autocast else "fp32"
        diffusion = "bf16" if self.diffusion_autocast else "fp32"
        return f"confidence={confidence} diffusion={diffusion}"


def amp_policy_for_tokens(n_token: int, model_name: str | None = None) -> AmpPolicy:
    """The policy upstream's ``update_inference_configs`` would select.

    ``model_name`` only matters below the confidence threshold, where upstream
    keeps ``protenix-v2``'s confidence head under autocast. Unlike upstream
    this does not refuse ``protenix-v2`` above 2,560 tokens: that limit is a
    supported-size rule, and the port already owns it in
    :data:`foldjax.models.protenix.runtime_policy.PROTENIX_V2_MAX_TOKENS`.
    A precision resolver that also raised would give the same limit two
    owners, which is how they drift apart.
    """
    n_token = int(n_token)
    if n_token < 0:
        raise ValueError(f"n_token must not be negative, got {n_token}")
    if n_token > DIFFUSION_AUTOCAST_ABOVE_TOKENS:
        return AmpPolicy(confidence_autocast=True, diffusion_autocast=True)
    if n_token > CONFIDENCE_AUTOCAST_ABOVE_TOKENS:
        return AmpPolicy(confidence_autocast=True, diffusion_autocast=False)
    return AmpPolicy(
        confidence_autocast=model_name in CONFIDENCE_AUTOCAST_ALWAYS_MODELS,
        diffusion_autocast=False,
    )


def requested_amp_policy(
    request: str,
    n_token: int,
    model_name: str | None = None,
) -> AmpPolicy:
    """Resolve one ``--amp-policy`` value against a job's token count."""
    if request == "auto":
        return amp_policy_for_tokens(n_token, model_name)
    if request == "fp32":
        return AmpPolicy(confidence_autocast=False, diffusion_autocast=False)
    if request == "bf16":
        return AmpPolicy(confidence_autocast=True, diffusion_autocast=True)
    choices = ", ".join(AMP_POLICY_CHOICES)
    raise ValueError(f"amp_policy must be one of {choices}; got {request!r}")


def realise_amp_policy(policy: AmpPolicy, *, trunk_is_bf16: bool) -> AmpPolicy:
    """Narrow a requested policy to what the ambient compute dtype allows.

    Upstream's ``skip_amp`` flags decide whether a stage runs *inside* the
    forward's autocast context, and that context is opened with
    ``configs.dtype`` (``runner/inference.py:218-228``). Under ``dtype=fp32``
    there is no autocast to skip, so both stages run fp32 whatever ``skip_amp``
    says. The port's ambient dtype is ``trunk_dtype``, so the same rule applies
    here: with an fp32 trunk the realised policy is fp32 on both stages, and
    ``--amp-policy bf16`` is not a way around that.
    """
    if trunk_is_bf16:
        return policy
    return AmpPolicy(confidence_autocast=False, diffusion_autocast=False)
