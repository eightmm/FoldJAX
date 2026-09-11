"""Which stages run under the bf16 autocast, written down instead of implied.

Two tables live here: what upstream Protenix runs, and what this port runs by
default. They agree on the diffusion sampler and differ on the confidence
head.

Upstream does not run one mixed-precision policy. Its inference runner
rewrites the policy from the token count before the model is built --
``runner/inference.py:492`` ``update_inference_configs`` -- and the two stages
it moves are the confidence head and the diffusion sampler:

    n_token > 3840          confidence bf16, diffusion bf16
    2560 < n_token <= 3840  confidence bf16, diffusion fp32
    n_token <= 2560         confidence fp32*, diffusion fp32

``*`` except ``protenix-v2``, whose confidence head is bf16 at every size.
:func:`amp_policy_for_tokens` is that table and nothing else, and
``--amp-policy upstream`` selects it at every size. That is the spelling a
parity run against a native capture wants, and the reason the thresholds below
are still named constants rather than folded into an inequality.

The port's released default, ``--amp-policy auto``, moves the confidence half
of the gate and leaves the diffusion half exactly where upstream put it:

    n_token > 3840          confidence bf16, diffusion bf16
    n_token <= 3840         confidence bf16, diffusion fp32

The confidence head narrows at every size because the accuracy is equivalent
and the narrower head is cheaper -- not because upstream runs it that way.
Upstream's 2,560 is an OOM heuristic for the configuration it ships, not a
measured accuracy boundary, and the port does not inherit it as one. Below
that threshold the confidence-only change is bracketed by two measured arms at
2,096 tokens on 5DEI: ``fp32`` on one side, and ``bf16`` -- which narrows the
confidence head *and* the sampler, so strictly more than ``auto`` does -- on
the other, at 0.39-0.45 A per-chain from the deposited structure for 11.9%
less wall time and 7.7% less peak memory. Those savings are the two-stage
arm's; ``auto`` narrows one stage and takes the smaller share.

The diffusion half does not move. That stage owns the coordinates, so leaving
it gated keeps every job at or below 3,840 tokens on the arithmetic it ran on
before this default existed, and keeps the confidence head the only thing
``auto`` changed.

Upstream spells it as ``skip_amp.<stage>``, which is the negation of what
happens: ``skip_amp.confidence_head = True`` means that stage runs with
autocast *disabled* -- its arguments cast to fp32 and its matmuls in fp32
(``protenix/model/protenix.py:344``, ``protenix/utils/torch_utils.py:199``).
This module states the positive form, ``<stage>_autocast``, because that is
what a reader has to know to predict a dtype. The native captures record the
upstream gate firing -- ``protenix-master-native-20260909-9yET4f`` has
``confidence_head`` skip_amp false at 3,012 tokens and true at 76.

Nothing here touches JAX: the CLI, the backend option table and the model all
resolve the same policy from the same function.
"""

from __future__ import annotations

from typing import NamedTuple

#: Accepted ``--amp-policy`` values. ``auto`` is the released default and
#: resolves the port's own table; ``upstream`` resolves the native token gate
#: instead; the other two pin one policy at every size, which is what an A/B
#: against either table needs.
AMP_POLICY_CHOICES = ("auto", "upstream", "fp32", "bf16")

#: The released default, and the value the backend strips from a cache
#: namespace: asking for it explicitly is the same run as not asking.
DEFAULT_AMP_POLICY = "auto"

#: Both bounds are exclusive upstream (``>``), so a job exactly at 2560 or
#: 3840 tokens keeps the lower stage's policy. The boundary tests pin that.
#: Only ``DIFFUSION_AUTOCAST_ABOVE_TOKENS`` is reachable under the released
#: default: ``auto`` narrows the confidence head at every size, so
#: ``CONFIDENCE_AUTOCAST_ABOVE_TOKENS`` is read only by
#: :func:`amp_policy_for_tokens`, which exists to state upstream's table.
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

    This is upstream's table, not the port's default -- see
    :func:`default_amp_policy_for_tokens` for the one ``--amp-policy auto``
    resolves. It stays because ``--amp-policy upstream`` has to be able to
    reproduce a native capture's configuration at any size.

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


def default_amp_policy_for_tokens(
    n_token: int, model_name: str | None = None
) -> AmpPolicy:
    """What ``--amp-policy auto`` selects: upstream's gate, confidence half removed.

    The confidence head is under the autocast at every size. The diffusion
    half is read straight off :func:`amp_policy_for_tokens` rather than
    re-spelled, so :data:`DIFFUSION_AUTOCAST_ABOVE_TOKENS` keeps one owner and
    the two tables cannot drift apart on the stage they agree about.

    ``model_name`` is accepted and forwarded for that diffusion half alone. It
    no longer changes the confidence answer: :data:`protenix-v2
    <CONFIDENCE_AUTOCAST_ALWAYS_MODELS>` was the one variant already narrowing
    its head below the gate, and under this table every variant does.
    """
    return AmpPolicy(
        confidence_autocast=True,
        diffusion_autocast=amp_policy_for_tokens(
            n_token, model_name
        ).diffusion_autocast,
    )


def requested_amp_policy(
    request: str,
    n_token: int,
    model_name: str | None = None,
) -> AmpPolicy:
    """Resolve one ``--amp-policy`` value against a job's token count."""
    if request == "auto":
        return default_amp_policy_for_tokens(n_token, model_name)
    if request == "upstream":
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
