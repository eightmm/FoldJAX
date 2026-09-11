"""Building the template pair embedding inside the template scan.

The embedding used to be built for every template at once and handed to the
scan as its ``xs``; it is now built one template at a time inside the scan
body. That is the same arithmetic in a different place, and the term-by-term
test below shows every term is bitwise unchanged. The compiled dense program
is nevertheless not bitwise: XLA fuses the eight projections into the addition
chain differently once the leading template extent is one, which moves the
float32 result by a single unit in the last place from about 64 tokens up.
``test_fencing_the_projections_restores_exact_equality`` pins that diagnosis,
and the strict xfail below pins the divergence itself, so a future XLA that
stops fusing fails loudly instead of quietly.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.data import compact_zero_template_pair_features
from foldjax.models.openfold3.data.featurize import (
    _ZERO_TEMPLATE_PAIR_FEATURES,
    _ZERO_TEMPLATE_PAIR_MARKER,
)
from foldjax.models.openfold3.models import template_module
from foldjax.models.openfold3.models.primitives import LinearParams, layer_norm, linear
from tests.models.cp_probe_env import inherited_environment
from tests.models.openfold3.test_template_collapse import (
    DISTOGRAM_BINS,
    N_TEMPLATE,
    RESTYPE,
    _embedder_params,
)

#: Below this the whole embedding is one fusion either way and the dense path
#: is bitwise; at and above it XLA's choices diverge. Measured, not chosen:
#: 48 tokens agrees, 64 does not.
_FUSION_DIVERGENCE_TOKENS = 64


# --- the implementation this change replaced, verbatim from 3e9f0f6 ---------
#
# Reconstructed here rather than kept behind a private name in `src`, so the
# comparison runs against the released code and not against a second copy that
# a later edit could quietly drag along with the first.


def _legacy_template_pair_embedder(
    batch: Mapping[str, jnp.ndarray],
    z: jnp.ndarray,
    params,
    *,
    eps: float = 1e-5,
) -> jnp.ndarray:
    restype = batch["template_restype"].astype(z.dtype)
    n_token = restype.shape[-2]
    restype_ti, restype_tj = template_module._project_template_restype(
        restype,
        params.aatype_linear_1,
        params.aatype_linear_2,
    )
    compact_zero_pairs = _ZERO_TEMPLATE_PAIR_MARKER in batch and all(
        name not in batch for name in _ZERO_TEMPLATE_PAIR_FEATURES
    )
    if compact_zero_pairs:
        marker = jnp.asarray(batch[_ZERO_TEMPLATE_PAIR_MARKER]).reshape(())

        def zero_pair_projection(projection: LinearParams) -> jnp.ndarray:
            zero_input = jnp.broadcast_to(marker, (projection.weight.shape[-1],))
            projected = linear(zero_input, projection)
            return jnp.broadcast_to(
                projected,
                (*restype.shape[:-2], n_token, n_token, projected.shape[-1]),
            )

        a = zero_pair_projection(params.dgram_linear)
        a = a + zero_pair_projection(params.pseudo_beta_mask_linear)
    else:
        asym_id = batch["asym_id"]
        same_chain = (asym_id[..., :, None] == asym_id[..., None, :]).astype(z.dtype)
        same_chain = same_chain[..., None, :, :, None]
        pseudo_beta = batch["template_pseudo_beta_mask"]
        pseudo_beta_pair = (
            pseudo_beta[..., :, None] * pseudo_beta[..., None, :]
        )[..., None] * same_chain
        backbone = batch["template_backbone_frame_mask"]
        backbone_pair = (
            backbone[..., :, None] * backbone[..., None, :]
        )[..., None] * same_chain
        unit_vector = batch["template_unit_vector"]
        x, y, w = (unit_vector[..., index] for index in range(3))
        a = linear(batch["template_distogram"], params.dgram_linear)
        a = a + linear(pseudo_beta_pair, params.pseudo_beta_mask_linear)
    a = a + restype_ti
    a = a + restype_tj
    if compact_zero_pairs:
        a = a + zero_pair_projection(params.x_linear)
        a = a + zero_pair_projection(params.y_linear)
        a = a + zero_pair_projection(params.z_linear)
        a = a + zero_pair_projection(params.backbone_mask_linear)
    else:
        a = a + linear(x[..., None], params.x_linear)
        a = a + linear(y[..., None], params.y_linear)
        a = a + linear(w[..., None], params.z_linear)
        a = a + linear(backbone_pair, params.backbone_mask_linear)
    z_embedded = linear(layer_norm(z, params.layer_norm_z, eps=eps), params.linear_z)
    return z_embedded[..., None, :, :, :] + a


def _legacy_template_embedder(
    batch: Mapping[str, jnp.ndarray],
    z: jnp.ndarray,
    params,
    *,
    pair_mask: jnp.ndarray,
    no_heads: int,
    tri_mul_first: bool = True,
    inf: float = 1e9,
    mask_transition: bool = True,
    eps: float = 1e-5,
    chunk_size: int | None = None,
    scan_templates: bool = True,
) -> jnp.ndarray:
    t = _legacy_template_pair_embedder(
        batch, z, params.template_pair_embedder, eps=eps
    )
    n_templ = t.shape[-4]
    template_weights = batch.get("template_padding_mask")
    if template_weights is None:
        template_weights = jnp.ones(t.shape[:-3], dtype=t.dtype)
    else:
        template_weights = jnp.asarray(template_weights, dtype=t.dtype)
        expected = t.shape[:-3]
        if template_weights.shape != expected:
            raise ValueError(
                "template_padding_mask must have shape "
                f"{expected}, got {template_weights.shape}"
            )
    denominator = jnp.clip(jnp.sum(template_weights, axis=-1), min=1.0)
    denominator = denominator[..., None, None, None]
    settings = dict(
        no_heads=no_heads,
        tri_mul_first=tri_mul_first,
        inf=inf,
        mask_transition=mask_transition,
        eps=eps,
        chunk_size=chunk_size,
    )
    if scan_templates and n_templ > 1:
        leading = jnp.moveaxis(t, -4, 0)
        leading_weights = jnp.moveaxis(template_weights, -1, 0)

        def accumulate(total, item):
            one, weight = item
            updated = template_module.template_pair_stack(
                one, params.template_pair_stack, mask=pair_mask, **settings
            )
            return total + updated * weight[..., None, None, None], None

        total, _ = jax.lax.scan(
            accumulate,
            jnp.zeros_like(leading[0]),
            (leading, leading_weights),
        )
        t = total / denominator
    else:
        t = template_module.template_pair_stack(
            t,
            params.template_pair_stack,
            mask=pair_mask[..., None, :, :],
            **settings,
        )
        weights = template_weights[..., :, None, None, None]
        t = jnp.sum(t * weights, axis=-4) / denominator
    return linear(jax.nn.relu(t), params.linear_t)


# --- inputs ----------------------------------------------------------------


def _dense_batch(n_token: int, seed: int = 3) -> dict[str, jnp.ndarray]:
    """Distinct, non-zero templates across two chains.

    Zero-filled templates would send the quadratic dots through ``0 * w`` and
    hide exactly the reduction the comparison is meant to exercise.
    """
    rng = np.random.default_rng(seed)
    restype = np.zeros((1, N_TEMPLATE, n_token, RESTYPE), dtype=np.int32)
    picks = rng.integers(0, RESTYPE, size=(1, N_TEMPLATE, n_token))
    np.put_along_axis(restype, picks[..., None], 1, axis=-1)
    asym_id = np.zeros((1, n_token), dtype=np.int32)
    asym_id[0, n_token // 2 :] = 1
    return {
        "asym_id": jnp.asarray(asym_id),
        "template_restype": jnp.asarray(restype),
        "template_pseudo_beta_mask": jnp.asarray(
            rng.random((1, N_TEMPLATE, n_token), dtype=np.float32)
        ),
        "template_backbone_frame_mask": jnp.asarray(
            rng.random((1, N_TEMPLATE, n_token), dtype=np.float32)
        ),
        "template_distogram": jnp.asarray(
            rng.random((1, N_TEMPLATE, n_token, n_token, DISTOGRAM_BINS), np.float32)
        ),
        "template_unit_vector": jnp.asarray(
            rng.random((1, N_TEMPLATE, n_token, n_token, 3), dtype=np.float32)
        ),
    }


def _compact_batch(n_token: int) -> dict[str, jnp.ndarray]:
    """The released four-row template axis in its compact zero-pair form.

    Deliberately not collapsed: `collapse_identical_templates` would reduce
    these interchangeable rows to one and the scan would never run.
    """
    restype = np.zeros((1, N_TEMPLATE, n_token, RESTYPE), dtype=np.int32)
    restype[..., -1] = 1
    source = compact_zero_template_pair_features(
        {
            "template_restype": restype,
            "template_pseudo_beta_mask": np.zeros((1, N_TEMPLATE, n_token), np.float32),
            "template_backbone_frame_mask": np.zeros(
                (1, N_TEMPLATE, n_token), np.float32
            ),
            "template_distogram": np.zeros(
                (1, N_TEMPLATE, n_token, n_token, DISTOGRAM_BINS), np.float32
            ),
            "template_unit_vector": np.zeros(
                (1, N_TEMPLATE, n_token, n_token, 3), np.float32
            ),
        }
    )
    assert set(source) == {"template_restype", _ZERO_TEMPLATE_PAIR_MARKER}
    return {
        "asym_id": jnp.zeros((1, n_token), dtype=jnp.int32),
        **{name: jnp.asarray(value) for name, value in source.items()},
    }


_BATCHES = {"dense": _dense_batch, "compact": _compact_batch}


def _run(embedder, batch, params, n_token: int, dtype=jnp.float32) -> np.ndarray:
    z = jax.random.normal(
        jax.random.key(11), (1, n_token, n_token, 8), dtype=dtype
    )
    pair_mask = jnp.ones((1, n_token, n_token), dtype=jnp.float32)
    return np.asarray(
        jax.jit(
            lambda b, p, one_z, mask: embedder(
                b, one_z, p, pair_mask=mask, no_heads=2
            )
        )(batch, params, z, pair_mask)
    )


# --- equality --------------------------------------------------------------


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize(
    ("branch", "n_token"),
    [
        ("dense", 5),
        pytest.param(
            "dense",
            96,
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "not bitwise from ~64 tokens up: XLA fuses the eight "
                    "projections into the addition chain differently once the "
                    "leading template extent is one, moving the result one ULP "
                    "on about a ninth of the elements. The arithmetic is "
                    "unchanged -- see the term-by-term and fenced tests below. "
                    "An XPASS means XLA stopped fusing that way; drop the mark."
                ),
            ),
        ),
        ("compact", 5),
        ("compact", 96),
    ],
)
def test_scanned_embedding_matches_the_all_at_once_form(
    branch: str, n_token: int, dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENFOLD3_TRIANGLE_BACKEND", "xla")
    params = _embedder_params()
    batch = _BATCHES[branch](n_token)
    expected = _run(_legacy_template_embedder, batch, params, n_token, dtype)
    actual = _run(template_module.template_embedder, batch, params, n_token, dtype)
    assert actual.tobytes() == expected.tobytes()


def _terms(batch, z, params) -> dict[str, jnp.ndarray]:
    """Each addend of the dense embedding, as its own program output."""
    restype = batch["template_restype"].astype(z.dtype)
    restype_ti, restype_tj = template_module._project_template_restype(
        restype, params.aatype_linear_1, params.aatype_linear_2
    )
    asym_id = batch["asym_id"]
    same_chain = (asym_id[..., :, None] == asym_id[..., None, :]).astype(z.dtype)
    same_chain = same_chain[..., None, :, :, None]
    pseudo_beta = batch["template_pseudo_beta_mask"]
    backbone = batch["template_backbone_frame_mask"]
    unit_vector = batch["template_unit_vector"]
    return {
        "dgram": linear(batch["template_distogram"], params.dgram_linear),
        "pseudo_beta": linear(
            (pseudo_beta[..., :, None] * pseudo_beta[..., None, :])[..., None]
            * same_chain,
            params.pseudo_beta_mask_linear,
        ),
        "backbone": linear(
            (backbone[..., :, None] * backbone[..., None, :])[..., None] * same_chain,
            params.backbone_mask_linear,
        ),
        "restype_ti": jnp.broadcast_to(restype_ti, restype_tj.shape[:-3] + (
            restype.shape[-2], restype.shape[-2], restype_ti.shape[-1])),
        "restype_tj": jnp.broadcast_to(restype_tj, restype_tj.shape[:-3] + (
            restype.shape[-2], restype.shape[-2], restype_tj.shape[-1])),
        "x": linear(unit_vector[..., 0][..., None], params.x_linear),
        "y": linear(unit_vector[..., 1][..., None], params.y_linear),
        "z": linear(unit_vector[..., 2][..., None], params.z_linear),
    }


def test_every_term_is_bitwise_unchanged_when_materialised() -> None:
    """The arithmetic is the same; only XLA's fusion of it differs.

    Each addend is forced out as its own result, so no fusion spans the
    projection and the additions. Under that constraint the two forms agree
    exactly at a size where the fused programs do not.
    """
    n_token = _FUSION_DIVERGENCE_TOKENS + 32
    params = _embedder_params().template_pair_embedder
    batch = _dense_batch(n_token)
    z = jax.random.normal(jax.random.key(11), (1, n_token, n_token, 8), jnp.float32)

    stacked = jax.jit(lambda b, one_z: _terms(b, one_z, params))(batch, z)
    per_template = jax.jit(
        lambda b, one_z: jax.tree.map(
            lambda *pieces: jnp.concatenate(pieces, axis=-4),
            *[
                _terms(template_module._one_template_batch(b, jnp.int32(index)),
                       one_z, params)
                for index in range(N_TEMPLATE)
            ],
        )
    )(batch, z)

    for name, value in stacked.items():
        expected = np.asarray(value).tobytes()
        assert expected == np.asarray(per_template[name]).tobytes(), name


def test_fencing_the_projections_restores_exact_equality() -> None:
    """Names the cause, and is not a remedy: the fence materialises all eight.

    With an optimization barrier on each projection the two forms are bitwise
    equal at a size where, unfenced, they are not. That localises the
    divergence to XLA's fusion of the projections into the addition chain
    rather than to any change in what is computed.
    """
    n_token = _FUSION_DIVERGENCE_TOKENS + 32
    params = _embedder_params().template_pair_embedder
    batch = _dense_batch(n_token)
    z = jax.random.normal(jax.random.key(11), (1, n_token, n_token, 8), jnp.float32)

    def fenced(one_batch, one_z):
        terms = jax.tree.map(
            jax.lax.optimization_barrier, _terms(one_batch, one_z, params)
        )
        total = terms["dgram"] + terms["pseudo_beta"]
        total = total + terms["restype_ti"] + terms["restype_tj"]
        total = total + terms["x"] + terms["y"] + terms["z"] + terms["backbone"]
        embedded = template_module._embed_pair_state(one_z, params, eps=1e-5)
        return embedded[..., None, :, :, :] + total

    stacked = np.asarray(jax.jit(fenced)(batch, z))
    per_template = np.asarray(
        jax.jit(
            lambda b, one_z: jnp.concatenate(
                [
                    fenced(
                        template_module._one_template_batch(b, jnp.int32(index)),
                        one_z,
                    )
                    for index in range(N_TEMPLATE)
                ],
                axis=-4,
            )
        )(batch, z)
    )
    assert stacked.tobytes() == per_template.tobytes()


def test_a_nonfinite_projection_weight_survives_the_scan_constant() -> None:
    """The compact marker's ``0 * NaN`` semantics are not constant-folded away.

    The marker is now read inside the scan body rather than once ahead of it.
    A NaN parameter has to reach the output exactly as it did before, which it
    cannot if XLA has replaced the dynamic marker with a literal zero.
    """
    params = _embedder_params()
    embedder = params.template_pair_embedder
    poisoned = embedder._replace(
        x_linear=LinearParams(weight=embedder.x_linear.weight.at[0, 0].set(jnp.nan))
    )
    params = params._replace(template_pair_embedder=poisoned)
    batch = _compact_batch(5)

    expected = _run(_legacy_template_embedder, batch, params, 5)
    actual = _run(template_module.template_embedder, batch, params, 5)

    assert np.isnan(expected).any(), "the poisoned weight never reached the output"
    assert actual.tobytes() == expected.tobytes()
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))


# --- structure -------------------------------------------------------------


def _scanned_operands(embedder, batch, params, n_token: int) -> list[str]:
    """Every top-level scan operand whose leading axis is the template axis."""
    z = jax.random.normal(jax.random.key(11), (1, n_token, n_token, 8), jnp.float32)
    pair_mask = jnp.ones((1, n_token, n_token), dtype=jnp.float32)
    jaxpr = jax.make_jaxpr(
        lambda b, p, one_z, mask: embedder(b, one_z, p, pair_mask=mask, no_heads=2)
    )(batch, params, z, pair_mask).jaxpr
    operands = []
    for equation in jaxpr.eqns:
        if equation.primitive.name != "scan":
            continue
        for variable in equation.invars:
            shape = getattr(variable.aval, "shape", ())
            if shape and shape[0] == equation.params["length"] == N_TEMPLATE:
                operands.append(variable.aval.str_short())
    assert operands, "no top-level template scan was traced"
    return operands


def test_the_scan_no_longer_carries_the_stacked_embedding() -> None:
    """Re-widening the ``xs`` is what this change exists to prevent."""
    n_token = 16
    params = _embedder_params()
    batch = _dense_batch(n_token)

    legacy = _scanned_operands(_legacy_template_embedder, batch, params, n_token)
    current = _scanned_operands(
        template_module.template_embedder, batch, params, n_token
    )

    stacked = f"float32[{N_TEMPLATE},1,{n_token},{n_token},8]"
    assert stacked in legacy, legacy
    assert stacked not in current, current
    # One index per template and one weight per template, nothing quadratic.
    assert f"int32[{N_TEMPLATE}]" in current, current
    quadratic = [name for name in current if f"{n_token},{n_token}" in name]
    assert not quadratic, quadratic


def test_the_template_axis_table_covers_every_template_feature() -> None:
    """A feature added to the released set must be given its axis here too."""
    assert set(template_module._TEMPLATE_FEATURE_AXES) == {
        *_ZERO_TEMPLATE_PAIR_FEATURES,
        "template_restype",
    }


# --- context parallelism ---------------------------------------------------


_CONTEXT_PARALLEL_PROBE = textwrap.dedent(
    """
    import os
    os.environ["OPENFOLD3_TRIANGLE_BACKEND"] = "xla"

    import jax
    import numpy as np

    from foldjax.models._cp import context_parallel
    from foldjax.models.openfold3.models import template_module
    from tests.models.openfold3.test_template_scan_embedding import (
        _dense_batch, _legacy_template_embedder, _run,
    )
    from tests.models.openfold3.test_template_collapse import _embedder_params

    assert jax.device_count() == 4, jax.devices()
    N_TOKEN = 8
    # The released stack is deep enough to scan its blocks, and `scan_stack`
    # emits one block body however many there are, so the default two-block
    # fixture is both the representative program and a cheap one. Truncating
    # to a single block unrolls instead, and that program really does pay nine
    # extra all-gathers for the moved embedding -- an artefact of the
    # truncation, not of this change.
    params = _embedder_params()
    batch = _dense_batch(N_TOKEN)

    serial_old = _run(_legacy_template_embedder, batch, params, N_TOKEN)
    serial_new = _run(template_module.template_embedder, batch, params, N_TOKEN)
    assert serial_new.tobytes() == serial_old.tobytes(), "serial baseline differs"

    COLLECTIVES = ("all-gather", "all_gather", "all-reduce", "collective-permute")

    def counts(embedder):
        import jax.numpy as jnp
        z = jax.random.normal(jax.random.key(11), (1, N_TOKEN, N_TOKEN, 8), jnp.float32)
        mask = jnp.ones((1, N_TOKEN, N_TOKEN), dtype=jnp.float32)
        compiled = jax.jit(
            lambda b, p, one_z, m: embedder(b, one_z, p, pair_mask=m, no_heads=2)
        ).lower(batch, params, z, mask).compile()
        value = np.asarray(jax.device_get(compiled(batch, params, z, mask)))
        text = compiled.runtime_executable().hlo_modules()[0].to_string().lower()
        return value, {name: text.count(name) for name in COLLECTIVES}

    for layout in ("1d", "2d"):
        jax.clear_caches()
        with context_parallel(4, layout=layout):
            old, old_counts = counts(_legacy_template_embedder)
            jax.clear_caches()
            new, new_counts = counts(template_module.template_embedder)
        assert new.tobytes() == old.tobytes(), (layout, "sharded results differ")
        # The two-dimensional layout reassociates the contractions, so neither
        # path reproduces its own serial bytes; what matters is that the moved
        # embedding does not move the result further than the layout already
        # does.
        np.testing.assert_allclose(new, serial_new, atol=3e-5, rtol=3e-5)
        np.testing.assert_array_equal(
            np.isclose(new, serial_new, atol=0, rtol=0),
            np.isclose(old, serial_old, atol=0, rtol=0),
        )
        regressed = {
            name: (old_counts[name], new_counts[name])
            for name in COLLECTIVES
            if new_counts[name] > old_counts[name]
        }
        assert not regressed, (layout, regressed)
        print(layout, "collectives old", old_counts, "new", new_counts)

    print("OPENFOLD3_TEMPLATE_SCAN_CP_OK")
    """
)


def test_the_scanned_embedding_holds_up_on_four_cpu_devices() -> None:
    """Context parallelism must not pay a collective for the moved embedding.

    This module applies no sharding constraint of its own -- the pair state
    arrives sharded from the trunk and the triangle operations constrain rows
    inside the stack -- so the question is whether slicing replicated template
    features per step forces a reshard the all-at-once form avoided.
    """
    completed = subprocess.run(
        [sys.executable, "-c", _CONTEXT_PARALLEL_PROBE],
        capture_output=True,
        text=True,
        timeout=300,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
            **inherited_environment(),
        },
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "OPENFOLD3_TEMPLATE_SCAN_CP_OK" in completed.stdout
