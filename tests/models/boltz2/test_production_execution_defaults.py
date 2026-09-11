from __future__ import annotations

import inspect

import pytest

from foldjax.models.boltz2.api import COMPUTE_DTYPES, predict
from foldjax.models.boltz2.models.diffusion.atom import diffusion_transformer_forward
from foldjax.models.boltz2.models.trunk_blocks.pairformer import (
    pairformer_module_forward,
)
from foldjax.models.boltz2.models.trunk_blocks.trunk import boltz2_sample_forward


def test_production_layer_stacks_default_to_memory_stable_scan() -> None:
    assert inspect.signature(pairformer_module_forward).parameters["use_scan"].default
    diffusion_scan = inspect.signature(diffusion_transformer_forward).parameters[
        "use_scan"
    ]
    assert diffusion_scan.default
    assert inspect.signature(boltz2_sample_forward).parameters["use_scan"].default


def test_prediction_defaults_to_the_precision_upstream_ships() -> None:
    """Boltz-2's own predict path is `Trainer(..., precision="bf16-mixed")`.

    A port that defaults to float32 is not being conservative, it is running a
    configuration its reference implementation does not ship -- which is the
    mistake this repository has made before. The trunk takes this dtype; the
    diffusion and confidence modules stay float32 whatever it says, which is
    the same split upstream's autocast draws.
    """
    assert inspect.signature(predict).parameters["compute_dtype"].default == "bfloat16"


def test_the_float32_trunk_is_still_reachable() -> None:
    """The old behaviour has to remain one argument away.

    bf16 costs ~0.002 of reported pLDDT and moves atoms by about three times
    the noise floor of running the same program twice. That is small, but a
    parity harness comparing against a float32 tape needs the exact previous
    program, not something close to it.
    """
    assert {"float32", "bfloat16"} <= set(COMPUTE_DTYPES)


def test_the_matmul_precision_pin_is_a_scope_not_a_latch() -> None:
    """Boltz-2's pin must not follow the caller out of `predict`.

    JAX's matmul precision is process-global. This was a `jax.config.update`
    inside `predict`, so a process that ran Boltz-2 and then anything else --
    another port, a notebook cell, a test collected later -- left that other
    thing on Boltz-2's policy. It went unnoticed while every port pinned the
    same value; the ports no longer do.
    """
    import jax

    from foldjax.execution import resolved_matmul_precision
    from foldjax.models.boltz2.api import MATMUL_PRECISION, _pinned_matmul_precision

    seen = []

    @_pinned_matmul_precision
    def record():
        seen.append(jax.config.jax_default_matmul_precision)

    before = jax.config.jax_default_matmul_precision
    record()

    assert seen == [resolved_matmul_precision(MATMUL_PRECISION)]
    assert jax.config.jax_default_matmul_precision == before


def _dot_precisions(text: str) -> set[str]:
    """The `precision = [...]` attributes one lowered program carries."""

    import re

    return set(re.findall(r"precision = \[([A-Z0-9, ]+)\]", text))


def _scope_program() -> str:
    """A bare float32 dot, lowered wherever the caller put it.

    This stands for every matmul in the graph that carries no `precision=` of
    its own, which is most of them: the diffusion score model, the
    transitions, the einsums, the XLA attention cores, and the
    cuEquivariance triangle-multiplication FFI, which reads the same config.
    """

    import jax
    import jax.numpy as jnp
    import numpy as np

    operand = np.zeros((4, 4), np.float32)
    return jax.jit(lambda a, b: jnp.matmul(a, b)).lower(operand, operand).as_text()


def _triangle_pin_program(dtype: str = "float32") -> str:
    """One triangle-attention layer, called the way the product calls it.

    No `matmul_precision` argument, because `api.predict` passes none: this is
    the second precision surface, the one the neutral knob does not reach.
    Lowered with float32 parameters by default, the arm where the attribute
    has float32 operands to act on.
    """

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models.boltz2.models.triangle.triangle_attention import (
        triangle_attention_forward,
    )

    element = jnp.dtype(dtype)
    rng = np.random.default_rng(0)
    dim, heads, hidden = 8, 2, 4
    inner = heads * hidden

    def weight(*shape):
        return jnp.asarray(rng.standard_normal(shape) * 0.1, dtype=element)

    params = {
        "layer_norm": {"scale": weight(dim), "bias": weight(dim)},
        "linear": {"kernel": weight(dim, heads)},
        "mha": {
            name: {"kernel": weight(dim, inner)}
            for name in ("linear_q", "linear_k", "linear_v", "linear_g")
        }
        | {"linear_o": {"kernel": weight(inner, dim)}},
    }
    return (
        jax.jit(
            triangle_attention_forward,
            static_argnames=("starting", "chunk_size", "triangle_backend"),
        )
        .lower(
            params,
            jnp.zeros((1, 6, 6, dim), jnp.float32),
            jnp.ones((1, 6, 6), jnp.float32),
            starting=True,
            chunk_size=0,
            triangle_backend="xla",
        )
        .as_text()
    )


def _inside(request: str | None, program):
    """Run `program` where a prediction runs it, for one neutral request."""

    from foldjax.execution import matmul_precision_scope
    from foldjax.models.boltz2.api import _pinned_matmul_precision

    @_pinned_matmul_precision
    def call():
        return program()

    with matmul_precision_scope(request):
        return call()


@pytest.mark.parametrize(
    ("request_value", "expected"),
    [(None, "HIGH, HIGH"), ("high", "HIGH, HIGH"), ("highest", "HIGHEST, HIGHEST")],
)
def test_the_shipped_scope_runs_float32_matmuls_at_tensorfloat32(
    request_value: str | None, expected: str
) -> None:
    """Boltz-2 ships TF32, and upstream's float32 is one option away.

    Upstream asks for float32 (`main.py:1096`) and this port followed until
    2026-09-11. The criterion for a default here is accuracy equivalence, not
    agreement with upstream's configuration, and TF32 meets it: measured on
    GPU over the released schedule, warm, 90.98 -> 82.17 s at 1,003 tokens and
    317.91 -> 287.62 s at 2,096, with per-chain RMSD to the deposited 5DEI
    chains 0.34-0.40 A on both arms and a same-index residual between them of
    0.038 A median against a 0.238 A within-set spread.

    The `None` row is the assertion that carries the change; the other two
    keep upstream's arithmetic reachable for the parity harnesses. All three
    read the compiled attribute rather than the configuration string.
    """

    assert _dot_precisions(_inside(request_value, _scope_program)) == {expected}


def test_the_realised_precision_assertion_can_see_the_other_value(monkeypatch) -> None:
    """Power check for the row above: the same probe, the other default.

    Without this, the `None` row could be passing because the probe cannot
    distinguish the two policies at all.
    """

    from foldjax.models.boltz2 import api

    monkeypatch.setattr(api, "MATMUL_PRECISION", "highest")

    assert _dot_precisions(_inside(None, _scope_program)) == {"HIGHEST, HIGHEST"}


def test_omitting_the_default_and_spelling_it_compile_one_program() -> None:
    """An explicit `high` must not be a second program, or a second cache entry."""

    assert _inside(None, _scope_program) == _inside("high", _scope_program)


def _split_by_pin(text: str) -> tuple[set[str], set[str]]:
    """Precisions of the explicitly pinned dots, and of the ones that inherit.

    Inside one triangle-attention layer both kinds sit side by side. The four
    `_linear` projections contract a rank-4 activation against a rank-2
    parameter and carry `precision=` from the op-level string; the score and
    P@V contractions in `_attention_block` are batched over (triangle, head)
    and carry none, so they take the scope. `batching_dims` is what separates
    them in the lowered text.
    """

    import re

    pinned: set[str] = set()
    inherited: set[str] = set()
    for line in text.splitlines():
        match = re.search(r"precision = \[([A-Z0-9, ]+)\]", line)
        if match is None or "dot_general" not in line:
            continue
        (inherited if "batching_dims" in line else pinned).add(match.group(1))
    return pinned, inherited


@pytest.mark.parametrize(
    ("request_value", "scope"),
    [(None, "HIGH, HIGH"), ("high", "HIGH, HIGH"), ("highest", "HIGHEST, HIGHEST")],
)
def test_the_neutral_knob_does_not_move_the_triangle_attention_pin(
    request_value: str | None, scope: str
) -> None:
    """The port has two precision surfaces and the knob reaches only one.

    `triangle_attention_forward` turns its `matmul_precision` string into a
    `jax.lax.Precision` and passes it explicitly to the four projections, so
    those dots keep it whatever `jax.default_matmul_precision` says.
    `api.predict` passes no such string, so they stay at the signature default
    float32 while every dot beside them -- including the two contractions in
    the same function -- follows the scope to TF32.

    That asymmetry is the state the 2026-09-11 GPU measurement was taken in:
    the measured arm moved the neutral knob and nothing else. Pinning it here
    is what keeps the shipped default equal to the arm that was measured, and
    it fails if a future edit wires the two surfaces together -- a real
    change with its own accuracy gate, not a tidy-up.
    """

    pinned, inherited = _split_by_pin(_inside(request_value, _triangle_pin_program))

    assert pinned == {"HIGHEST, HIGHEST"}
    assert inherited == {scope}


def test_wiring_the_two_surfaces_together_cannot_happen_quietly() -> None:
    """The op-level resolver refuses the spelling the neutral knob uses.

    A future edit that fed the resolved neutral value into the op-level string
    would hand it `"high"`, and this is what makes that loud on the first
    prediction instead of silently compiling a program nothing measured. It is
    part of the cost of unifying the surfaces, and it is deliberate.
    """

    from foldjax.models.boltz2.models.triangle.triangle_attention import (
        resolve_matmul_precision,
    )

    with pytest.raises(ValueError, match="Unsupported matmul_precision: 'high'"):
        resolve_matmul_precision("high")


def test_the_op_level_pin_is_inert_on_the_released_trunk() -> None:
    """Its disagreement with the scope has no float32 operand to act on.

    `_cast_trunk_params` narrows every triangle-attention kernel at
    `compute_dtype="bfloat16"`, and `_linear` narrows the float32 pair
    activation to match before the matmul, so all four projections and both
    score contractions are bfloat16 x bfloat16. The attribute is still
    recorded, but it selects between float32 accumulation strategies and there
    is no float32 operand. Under `--option dtype=float32` the same dots are
    float32 and the disagreement is live -- which is why that arm, and not the
    shipped one, is where unifying the surfaces would change arithmetic.
    """

    import re

    released = _inside(None, lambda: _triangle_pin_program("bfloat16"))
    operands = re.findall(r"precision = \[[A-Z0-9, ]+\] : \(([^)]*)\)", released)

    assert operands, "no annotated dot found in the lowered layer"
    assert all("f32" not in operand for operand in operands), operands
    assert all("bf16" in operand for operand in operands), operands

    float32_arm = _inside(None, lambda: _triangle_pin_program("float32"))
    assert "xf32>, tensor<" in float32_arm


def test_the_compiled_path_hands_the_graph_unplaced_features() -> None:
    """Placing the feature dict costs device memory the program never asks for.

    The featurizer emits 78 arrays and the inference graph reads 31. Building
    the dict with `jnp.asarray` put all 78 on the device and kept them there
    for the whole run; `disto_target` -- a training label, f32[N, N, 1, 64] --
    was 246 MiB of that at 1,003 tokens on its own, and the term is quadratic
    in token count. Handing `jax.jit` the NumPy arrays instead lets it drop the
    unread ones before anything is transferred.
    """
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models.boltz2.api import _graph_features

    feats = {"msa": np.zeros((1, 4, 3), np.int64)}

    compiled = _graph_features(feats, place=False)
    assert type(compiled["msa"]) is np.ndarray

    placed = _graph_features(feats, place=True)
    assert isinstance(placed["msa"], jnp.ndarray)

    # Both spellings reach the graph as the same argument, so the traced
    # program -- and therefore every coordinate it produces -- is unchanged.
    assert jnp.asarray(compiled["msa"]).dtype == placed["msa"].dtype
    assert compiled["msa"].shape == placed["msa"].shape


def test_jit_drops_arguments_the_graph_never_reads() -> None:
    """The saving above is JAX's argument pruning, so pin that it still prunes.

    `jax.jit` DCEs unread arguments out of the lowered program and filters them
    out before the rest are placed. If an upgrade ever stopped doing that, the
    unread features would silently be transferred again and only a memory
    benchmark would notice.
    """
    import jax
    import numpy as np

    def graph(read, never_read):
        return read * 2.0

    lowered = jax.jit(graph).lower(
        np.zeros((3,), np.float32), np.zeros((1024,), np.float32)
    )
    signature = lowered.as_text().split("func.func public @main")[1].split("\n")[0]
    assert "1024" not in signature
    assert "3xf32" in signature
