"""OpenFold3's opt-in fused gated linear unit.

The shipped SwiGLU is two projections and an elementwise product, which is what
upstream runs: its ``SwiGLU`` takes ``use_kernel=False`` and ``SwiGLUTransition``
never passes the flag. ``glu_backend="tokamax"`` computes the same product in
one fused Triton kernel instead. These tests hold the default where upstream
left it, prove the fused path computes the same numbers, and prove it is
actually reached -- a numerical test alone passes just as happily when the
branch is dead.

The kernel is pinned to Triton and raises off a GPU, deliberately, so a card
that cannot run it says so rather than measuring XLA under the fused name.
CPU coverage moves that seam: ``_force_tokamax_xla`` replaces
``tokamax.gated_linear_unit`` with a wrapper that overrides the implementation
and counts the calls. The layout, the orientation and the dtype contract are
what is under test here, and they are identical either way; the Triton
lowering itself is not exercised on CPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tokamax

from foldjax.models.openfold3 import inference
from foldjax.models.openfold3.models import primitives
from foldjax.models.openfold3.models.attention import AttentionParams
from foldjax.models.openfold3.models.attention_pair_bias import (
    AttentionPairBiasParams,
)
from foldjax.models.openfold3.models.pair_block import PairBlockParams
from foldjax.models.openfold3.models.pairformer import (
    PairformerBlockParams,
    PairformerStackParams,
    pairformer_stack,
)
from foldjax.models.openfold3.models.primitives import (
    LayerNormParams,
    LinearParams,
    SwiGLUParams,
    SwiGLUTransitionParams,
    swiglu,
)
from foldjax.models.openfold3.models.sampler import sample_diffusion
from foldjax.models.openfold3.models.triangle import TriangleMultiplicationParams
from foldjax.models.openfold3.models.triangle_attention import (
    TriangleAttentionParams,
)

C, HEADS, N = 8, 2, 6


def _force_tokamax_xla(monkeypatch) -> list[dict]:
    """Run the fused branch through tokamax's XLA implementation, and count."""

    calls: list[dict] = []
    original = tokamax.gated_linear_unit

    def wrapper(*args, **kwargs):
        calls.append(kwargs)
        # Overriding rather than defaulting: the caller passes "triton"
        # explicitly, so a `functools.partial` default would be ignored.
        return original(*args, **{**kwargs, "implementation": "xla"})

    monkeypatch.setattr(tokamax, "gated_linear_unit", wrapper)
    return calls


def _record_swiglu(monkeypatch) -> list[str]:
    """Record the backend every SwiGLU in the model is handed.

    ``swiglu_transition`` and ``conditioned_transition_block`` both resolve
    ``swiglu`` through this module's globals, so one patch sees every site.
    """

    seen: list[str] = []
    original = primitives.swiglu

    def observe(x, params, *, glu_backend="xla"):
        seen.append(glu_backend)
        return original(x, params, glu_backend=glu_backend)

    monkeypatch.setattr(primitives, "swiglu", observe)
    return seen


def _swiglu_params(rng, *, bias=False):
    def arr(*shape):
        return jnp.asarray(rng.normal(size=shape, scale=0.5), dtype=jnp.float32)

    def projection(out, size):
        return LinearParams(weight=arr(out, size), bias=arr(out) if bias else None)

    return SwiGLUParams(linear_a=projection(4 * C, C), linear_b=projection(4 * C, C))


def _stack_params(rng) -> PairformerStackParams:
    """Two real Pairformer blocks: four SwiGLU sites, no upstream checkout."""

    def arr(*shape):
        return jnp.asarray(rng.normal(size=shape, scale=0.5), dtype=jnp.float32)

    def lin(out, size):
        return LinearParams(weight=arr(out, size), bias=arr(out))

    def ln():
        return LayerNormParams(weight=arr(C) * 0.1 + 1.0, bias=arr(C) * 0.1)

    def tri_mult():
        return TriangleMultiplicationParams(
            layer_norm_in=ln(),
            layer_norm_out=ln(),
            linear_a_p=lin(C, C),
            linear_a_g=lin(C, C),
            linear_b_p=lin(C, C),
            linear_b_g=lin(C, C),
            linear_g=lin(C, C),
            linear_z=lin(C, C),
        )

    def attn():
        return AttentionParams(lin(C, C), lin(C, C), lin(C, C), lin(C, C), lin(C, C))

    def tri_att():
        return TriangleAttentionParams(
            layer_norm=ln(),
            linear_z=LinearParams(weight=arr(HEADS, C), bias=None),
            mha=attn(),
        )

    def transition():
        # Bias-free, the way ``map_swiglu`` maps every real checkpoint.
        return SwiGLUTransitionParams(
            layer_norm=ln(),
            swiglu=SwiGLUParams(
                linear_a=LinearParams(weight=arr(2 * C, C), bias=None),
                linear_b=LinearParams(weight=arr(2 * C, C), bias=None),
            ),
            linear_out=lin(C, 2 * C),
        )

    def block():
        return PairformerBlockParams(
            pair_stack=PairBlockParams(
                tri_mul_out=tri_mult(),
                tri_mul_in=tri_mult(),
                tri_att_start=tri_att(),
                tri_att_end=tri_att(),
                pair_transition=transition(),
            ),
            attn_pair_bias=AttentionPairBiasParams(
                layer_norm_a=ln(),
                layer_norm_z=ln(),
                linear_z=LinearParams(weight=arr(HEADS, C), bias=None),
                mha=attn(),
            ),
            single_transition=transition(),
        )

    return PairformerStackParams(blocks=(block(), block()))


def test_the_fused_unit_matches_the_released_path_on_float32(monkeypatch) -> None:
    """Same product, and a control that shows the tolerance can fail.

    The orientation is the part a name cannot settle: ``LinearParams.weight``
    is torch's ``[out, in]`` and the kernel wants ``[in, out]`` per branch with
    the activated branch first. Handing it the branches the other way round is
    a whole unit apart, so a tolerance this tight is measuring something.
    """

    _force_tokamax_xla(monkeypatch)
    rng = np.random.default_rng(11)
    x = jnp.asarray(rng.normal(size=(2, 5, C)), dtype=jnp.float32)
    params = _swiglu_params(rng)

    released = swiglu(x, params)
    fused = swiglu(x, params, glu_backend="tokamax")

    np.testing.assert_allclose(fused, released, rtol=1e-5, atol=1e-6)
    swapped = swiglu(
        x,
        SwiGLUParams(linear_a=params.linear_b, linear_b=params.linear_a),
        glu_backend="tokamax",
    )
    assert float(jnp.max(jnp.abs(swapped - released))) > 1e-2


@pytest.mark.parametrize("scan_blocks", [True, False])
def test_a_real_pairformer_stack_matches_and_actually_calls_the_kernel(
    monkeypatch, scan_blocks
) -> None:
    """Both transitions of both blocks, through the real stack.

    The call count is the tripwire: a run that computes the same numbers while
    calling the kernel zero times is a dead branch, which is exactly what a
    numbers-only test cannot see. Two sites per block -- one pair transition
    and one single transition -- so the scanned body traces two and the
    unrolled two-block stack traces four.
    """

    sites = 2 if scan_blocks else 4
    calls = _force_tokamax_xla(monkeypatch)
    seen = _record_swiglu(monkeypatch)
    rng = np.random.default_rng(5)
    params = _stack_params(rng)
    s = jnp.asarray(rng.normal(size=(N, C), scale=0.5), dtype=jnp.float32)
    z = jnp.asarray(rng.normal(size=(N, N, C), scale=0.5), dtype=jnp.float32)
    single_mask = jnp.ones(N, dtype=jnp.float32)
    pair_mask = jnp.ones((N, N), dtype=jnp.float32)

    def run(backend):
        return pairformer_stack(
            s,
            z,
            params,
            single_mask=single_mask,
            pair_mask=pair_mask,
            no_heads_pair=HEADS,
            no_heads_pair_bias=HEADS,
            glu_backend=backend,
            scan_blocks=scan_blocks,
        )

    released_s, released_z = jax.jit(run, static_argnums=0)("xla")
    assert seen == ["xla"] * sites
    assert calls == []

    seen.clear()
    fused_s, fused_z = jax.jit(run, static_argnums=0)("tokamax")
    assert seen == ["tokamax"] * sites
    assert len(calls) == sites

    # Looser than the leaf comparison on purpose: two blocks of triangle
    # updates and attention carry the two arms' float32 round-off forward, and
    # the fused arm applies `jax.nn.silu` where the released one applies the
    # port's own `silu`. Measured worst element here is 1.8e-05 relative.
    np.testing.assert_allclose(fused_s, released_s, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(fused_z, released_z, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("backend", ["xla", "tokamax"])
def test_predict_hands_every_stage_the_configured_backend(monkeypatch, backend) -> None:
    """The config reaches each network boundary ``predict`` itself calls.

    Real ``predict``, with the expensive stages replaced by recorders; the
    same cut the augmentation wiring test uses. A stage that never receives
    the value runs the released path under a name that says otherwise.
    """

    config = inference.InferenceConfig(
        n_atom=4,
        n_token=2,
        num_samples=3,
        num_steps=2,
        n_query=2,
        n_key=4,
        atom_heads=1,
        token_heads=1,
        no_heads_msa=1,
        no_heads_pair=1,
        no_heads_pair_bias=1,
        max_relative_idx=2,
        max_relative_chain=2,
        num_recycles=1,
        max_atoms_per_token=2,
        plddt_bins=4,
        pae_bins=4,
        pae_bin_max=4.0,
        msa_depth=None,
        glu_backend=backend,
    )
    batch = {
        "atom_mask": jnp.array([[1.0, 1.0, 0.0, 1.0]]),
        "token_mask": jnp.ones((1, 2)),
        "asym_id": jnp.zeros((1, 2)),
    }
    single, pair = jnp.zeros((1, 2, 4)), jnp.zeros((1, 2, 2, 4))
    received: dict[str, str] = {}

    def recorder(name, value):
        def observe(*args, glu_backend="xla", **kwargs):
            received[name] = glu_backend
            return value

        return observe

    monkeypatch.setattr(inference, "trunk", recorder("trunk", (single, single, pair)))
    monkeypatch.setattr(
        inference, "pair_conditioning", recorder("pair_conditioning", pair)
    )
    monkeypatch.setattr(
        inference, "single_conditioning", recorder("single_conditioning", single)
    )

    def denoise(batch, x, *args, glu_backend="xla", **kwargs):
        received["denoise"] = glu_backend
        return x * 0.25

    monkeypatch.setattr(inference, "denoise", denoise)

    class SamplerCompletedError(Exception):
        """Stop before the confidence heads, which need real parameters."""

    def observe(key, schedule, shape, denoiser, **kwargs):
        # Run the real rollout first: the single conditioning and the denoiser
        # are only reached from inside it.
        sample_diffusion(key, schedule, shape, denoiser, **kwargs)
        raise SamplerCompletedError

    monkeypatch.setattr(inference, "sample_diffusion", observe)
    params = SimpleNamespace(trunk=None, diffusion_conditioning=None, denoiser=None)
    with pytest.raises(SamplerCompletedError):
        inference.predict(jax.random.key(3), batch, params, config, None)

    assert received == {
        "trunk": backend,
        "pair_conditioning": backend,
        "single_conditioning": backend,
        "denoise": backend,
    }


@pytest.mark.parametrize("backend", ["xla", "tokamax"])
def test_the_confidence_re_embedding_forwards_the_backend(monkeypatch, backend) -> None:
    """The confidence head runs its own Pairformer, so it needs the value too.

    ``predict``'s own coverage above stops at the sampler, because the heads
    below it need real parameters. This takes the remaining stage on its own.
    """

    from foldjax.models.openfold3.models import heads

    received: list[str] = []

    def observe(s, z, params, *, glu_backend="xla", **kwargs):
        received.append(glu_backend)
        return s, z

    monkeypatch.setattr(heads, "pairformer_stack", observe)
    rng = np.random.default_rng(2)

    def arr(*shape):
        return jnp.asarray(rng.normal(size=shape, scale=0.5), dtype=jnp.float32)

    params = heads.PairformerEmbeddingParams(
        linear_i=LinearParams(weight=arr(C, C), bias=None),
        linear_j=LinearParams(weight=arr(C, C), bias=None),
        linear_distance=LinearParams(weight=arr(C, 5), bias=None),
        pairformer_stack=None,
    )
    heads.pairformer_embedding(
        arr(N, C),
        arr(N, C),
        arr(N, N, C),
        arr(N, 3),
        params,
        single_mask=jnp.ones(N),
        pair_mask=jnp.ones((N, N)),
        no_heads_pair=HEADS,
        no_heads_pair_bias=HEADS,
        min_bin=2.0,
        max_bin=8.0,
        no_bin=5,
        glu_backend=backend,
    )

    assert received == [backend]


def test_the_released_default_is_the_unfused_path() -> None:
    """Nothing about a released run changes: the default is upstream's choice.

    Upstream ships ``use_kernel: bool = False`` on its ``SwiGLU`` and
    ``SwiGLUTransition`` does not pass it, so the fused kernel is a deviation
    from the shipped architecture and has to be asked for by name.
    """

    import inspect

    assert inference.InferenceConfig.__new__.__defaults__ is not None
    config = inference.released_config(n_token=8, n_atom=16)
    assert config.glu_backend == "xla"
    signature = inspect.signature(inference.released_config)
    assert signature.parameters["glu_backend"].default == "xla"
    # Reached without a config at all: a direct caller of the primitive gets
    # the released path too.
    assert (
        inspect.signature(primitives.swiglu).parameters["glu_backend"].default == "xla"
    )


def test_an_unknown_backend_names_the_values_that_exist() -> None:
    from foldjax.backends.openfold3 import OpenFold3Backend

    with pytest.raises(ValueError, match=r"'xla', 'tokamax'"):
        inference.released_config(n_token=8, n_atom=16, glu_backend="triton")
    with pytest.raises(ValueError, match=r"'xla', 'tokamax'"):
        OpenFold3Backend().validate_native_options({"glu_backend": "triton"})


def test_context_parallelism_refuses_the_fused_unit(monkeypatch) -> None:
    """Refused twice: early by the config, and at the point of use.

    A fused kernel cannot be partitioned, and the leaf check is the one a
    hand-built config or a ``_replace`` cannot walk around.
    """

    with pytest.raises(ValueError, match="context parallelism"):
        inference.released_config(
            n_token=8, n_atom=16, cp_shards=4, glu_backend="tokamax"
        )

    rng = np.random.default_rng(3)
    x = jnp.zeros((2, 3, C), jnp.float32)
    params = _swiglu_params(rng)
    monkeypatch.setattr(primitives, "cp_mesh", lambda: object())
    with pytest.raises(ValueError, match="context parallelism"):
        swiglu(x, params, glu_backend="tokamax")
    # The released path is unaffected by the mesh being active.
    assert swiglu(x, params).shape == (2, 3, 4 * C)


def test_the_fused_unit_refuses_a_biased_projection(monkeypatch) -> None:
    """No silent fallback: the kernel has nowhere to put a bias.

    ``map_swiglu`` maps both projections with ``bias=False``, so this cannot
    fire on a real checkpoint; a hand-built one that carries a bias would
    otherwise have it dropped.
    """

    _force_tokamax_xla(monkeypatch)
    rng = np.random.default_rng(7)
    x = jnp.zeros((2, 3, C), jnp.float32)
    with pytest.raises(ValueError, match="bias-free"):
        swiglu(x, _swiglu_params(rng, bias=True), glu_backend="tokamax")


def test_the_backends_literal_tracks_the_shared_module() -> None:
    """The adapter cannot import the shared list: that module imports JAX.

    Cache-directory resolution runs without the model runtime, so the values
    are copied beside the adapter. This is the copy that keeps them equal.
    """

    from foldjax.backends.openfold3 import _GLU_BACKENDS
    from foldjax.models._glu import GLU_BACKENDS

    assert _GLU_BACKENDS == GLU_BACKENDS


def _forwarding_recorder(monkeypatch, module, name, result):
    """Replace one stage with a recorder of the backend it is handed."""

    received: list[str] = []

    def observe(*args, glu_backend="xla", **kwargs):
        received.append(glu_backend)
        return result

    monkeypatch.setattr(module, name, observe)
    return received


@pytest.mark.parametrize("backend", ["xla", "tokamax"])
def test_the_trunk_forwards_the_backend_to_every_stage(monkeypatch, backend) -> None:
    """One recycle, with each stage replaced by a recorder.

    These are one-line forwards, which is exactly the kind that gets dropped:
    the run still produces numbers, and every one of them comes from the
    released path under a name that says otherwise.
    """

    from foldjax.models.openfold3.models import trunk as trunk_module

    rng = np.random.default_rng(13)

    def arr(*shape):
        return jnp.asarray(rng.normal(size=shape, scale=0.5), dtype=jnp.float32)

    s, z = arr(N, C), arr(N, N, C)
    m = arr(2, N, C)
    embedder = _forwarding_recorder(
        monkeypatch, trunk_module, "input_embedder", (s, s, z)
    )
    monkeypatch.setattr(
        trunk_module, "msa_embedder", lambda *a, **k: (m, jnp.ones((2, N)))
    )
    template = _forwarding_recorder(monkeypatch, trunk_module, "template_embedder", z)
    msa = _forwarding_recorder(monkeypatch, trunk_module, "msa_module_stack", z)
    pairformer = _forwarding_recorder(
        monkeypatch, trunk_module, "pairformer_stack", (s, z)
    )
    params = SimpleNamespace(
        input_embedder=None,
        msa_module_embedder=None,
        # Not None, so the template branch is taken.
        template_embedder=object(),
        msa_module=None,
        pairformer_stack=None,
        layer_norm_z=LayerNormParams(weight=arr(C), bias=arr(C)),
        linear_z=LinearParams(weight=arr(C, C), bias=None),
        layer_norm_s=LayerNormParams(weight=arr(C), bias=arr(C)),
        linear_s=LinearParams(weight=arr(C, C), bias=None),
    )
    batch = {"token_mask": jnp.ones(N)}

    trunk_module.trunk(
        batch,
        params,
        num_recycles=1,
        n_query=2,
        n_key=4,
        atom_heads=1,
        n_token=N,
        max_relative_idx=2,
        max_relative_chain=2,
        no_heads_msa=1,
        no_heads_pair=HEADS,
        no_heads_pair_bias=HEADS,
        glu_backend=backend,
    )

    assert embedder == [backend]
    assert template == [backend]
    assert msa == [backend]
    assert pairformer == [backend]


@pytest.mark.parametrize("backend", ["xla", "tokamax"])
def test_the_denoiser_forwards_the_backend_to_every_stage(monkeypatch, backend) -> None:
    """The atom encoder, the token transformer and the atom decoder."""

    from foldjax.models.openfold3.models import denoiser as denoiser_module

    rng = np.random.default_rng(17)

    def arr(*shape):
        return jnp.asarray(rng.normal(size=shape, scale=0.5), dtype=jnp.float32)

    n_atom = 4
    ai, ql, cl, plm = arr(N, C), arr(n_atom, C), arr(n_atom, C), arr(1, 2, 4, C)
    encoder = _forwarding_recorder(
        monkeypatch, denoiser_module, "atom_attention_encoder", (ai, ql, cl, plm)
    )
    transformer = _forwarding_recorder(
        monkeypatch, denoiser_module, "diffusion_transformer", ai
    )
    decoder = _forwarding_recorder(
        monkeypatch, denoiser_module, "atom_attention_decoder", arr(n_atom, 3)
    )
    params = SimpleNamespace(
        atom_attn_enc=None,
        diffusion_transformer=None,
        atom_attn_dec=None,
        layer_norm_s=LayerNormParams(weight=arr(C), bias=arr(C)),
        linear_s=LinearParams(weight=arr(C, C), bias=None),
        layer_norm_a=LayerNormParams(weight=arr(C), bias=arr(C)),
    )
    batch = {
        "atom_mask": jnp.ones(n_atom),
        "token_mask": jnp.ones(N),
    }

    denoiser_module.denoise(
        batch,
        arr(n_atom, 3),
        jnp.asarray(1.0),
        arr(N, C),
        arr(N, C),
        arr(N, N, C),
        params,
        n_query=2,
        n_key=4,
        atom_heads=1,
        token_heads=1,
        n_token=N,
        sigma_data=16.0,
        glu_backend=backend,
    )

    assert encoder == [backend]
    assert transformer == [backend]
    assert decoder == [backend]


@pytest.mark.parametrize("backend", ["xla", "tokamax"])
def test_both_transformer_stacks_forward_the_backend(monkeypatch, backend) -> None:
    """The diffusion transformer and the sequence-local atom transformer.

    Both reach ``conditioned_transition_block``, and the atom stack runs
    inside the rollout, so a dropped forward there is the one that costs the
    most.
    """

    from foldjax.models.openfold3.models import diffusion_transformer as module

    rng = np.random.default_rng(19)

    def arr(*shape):
        return jnp.asarray(rng.normal(size=shape, scale=0.5), dtype=jnp.float32)

    a, s = arr(N, C), arr(N, C)
    received = _forwarding_recorder(
        monkeypatch, module, "conditioned_transition_block", a
    )
    monkeypatch.setattr(module, "ada_attention_pair_bias", lambda *x, **k: a)
    monkeypatch.setattr(module, "cross_attention_pair_bias", lambda *x, **k: a)
    monkeypatch.setattr(module, "layer_norm", lambda x, *a, **k: x)
    block = SimpleNamespace(attention_pair_bias=None, conditioned_transition=None)
    stack = SimpleNamespace(blocks=(block,), layer_norm_z=None)

    module.diffusion_transformer(
        a,
        s,
        arr(N, N, C),
        stack,
        no_heads=HEADS,
        glu_backend=backend,
        scan_blocks=False,
    )
    module.atom_transformer(
        a,
        s,
        arr(1, 2, 4, C),
        stack,
        no_heads=HEADS,
        n_query=2,
        n_key=4,
        glu_backend=backend,
        scan_blocks=False,
    )

    assert received == [backend, backend]


@pytest.mark.parametrize("backend", ["xla", "tokamax"])
def test_the_msa_and_template_stacks_forward_the_backend(monkeypatch, backend) -> None:
    """Both stacks own a pair block, and the MSA stack a transition of its own."""

    from foldjax.models.openfold3.models import msa_module, template_module

    rng = np.random.default_rng(23)

    def arr(*shape):
        return jnp.asarray(rng.normal(size=shape, scale=0.5), dtype=jnp.float32)

    z = arr(N, N, C)
    msa_pair = _forwarding_recorder(monkeypatch, msa_module, "pair_block", z)
    msa_transition = _forwarding_recorder(
        monkeypatch, msa_module, "swiglu_transition", arr(2, N, C)
    )
    monkeypatch.setattr(
        msa_module, "msa_pair_weighted_averaging", lambda *a, **k: arr(2, N, C)
    )
    monkeypatch.setattr(msa_module, "outer_product_mean", lambda *a, **k: z)
    block = SimpleNamespace(
        outer_product_mean=None,
        msa_att_row=object(),
        msa_transition=object(),
        pair_stack=None,
    )
    msa_module.msa_module_stack(
        arr(2, N, C),
        z,
        SimpleNamespace(blocks=(block,)),
        msa_mask=jnp.ones((2, N)),
        pair_mask=jnp.ones((N, N)),
        no_heads_msa=1,
        no_heads_pair=HEADS,
        glu_backend=backend,
    )
    assert msa_pair == [backend]
    assert msa_transition == [backend]

    template_pair = _forwarding_recorder(monkeypatch, template_module, "pair_block", z)
    monkeypatch.setattr(template_module, "layer_norm", lambda x, *a, **k: x)
    template_module.template_pair_stack(
        z,
        SimpleNamespace(blocks=(SimpleNamespace(),), layer_norm=None),
        mask=jnp.ones((N, N)),
        no_heads=HEADS,
        glu_backend=backend,
    )
    assert template_pair == [backend]


@pytest.mark.parametrize("backend", ["xla", "tokamax"])
def test_the_atom_stages_forward_the_backend(monkeypatch, backend) -> None:
    """The input embedder, the atom encoder and the atom decoder.

    Each of them owns a transformer whose blocks carry a conditioned SwiGLU
    transition, and the encoder runs on both the input path and every rollout
    step. The embedders in front are replaced: what is under test is the hop.
    """

    from foldjax.models.openfold3.models import (
        atom_features,
        atomize,
        input_embedders,
    )

    rng = np.random.default_rng(29)

    def arr(*shape):
        return jnp.asarray(rng.normal(size=shape, scale=0.5), dtype=jnp.float32)

    n_atom = 4
    cl, plm = arr(n_atom, C), arr(1, 2, 4, C)
    atom = _forwarding_recorder(monkeypatch, atom_features, "atom_transformer", cl)
    monkeypatch.setattr(
        atom_features, "ref_atom_feature_embedder", lambda *a, **k: (cl, plm)
    )
    monkeypatch.setattr(atom_features, "atom_pair_conditioning", lambda *a, **k: plm)
    # Imported inside the encoder, so the patch has to land on its own module.
    monkeypatch.setattr(
        atomize, "aggregate_atom_feat_to_tokens", lambda *a, **k: arr(N, C)
    )
    monkeypatch.setattr(
        atom_features, "broadcast_token_feat_to_atoms", lambda *a, **k: cl
    )
    monkeypatch.setattr(atom_features, "linear", lambda x, *a, **k: x)
    monkeypatch.setattr(atom_features, "layer_norm", lambda x, *a, **k: x)
    batch = {
        "atom_mask": jnp.ones(n_atom),
        "token_mask": jnp.ones(N),
        "num_atoms_per_token": jnp.ones(N, dtype=jnp.int32),
        "atom_to_token_index": jnp.zeros(n_atom, dtype=jnp.int32),
    }
    params = SimpleNamespace(
        ref_atom_feature_embedder=None,
        noisy_position_embedder=None,
        pair_conditioning=None,
        atom_transformer=None,
        linear_q=None,
        linear_q_in=None,
        linear_q_out=None,
        layer_norm=None,
    )

    atom_features.atom_attention_encoder(
        batch,
        params,
        n_query=2,
        n_key=4,
        no_heads=1,
        n_token=N,
        glu_backend=backend,
    )
    atom_features.atom_attention_decoder(
        batch,
        arr(N, C),
        cl,
        cl,
        plm,
        params,
        n_query=2,
        n_key=4,
        no_heads=1,
        glu_backend=backend,
    )
    assert atom == [backend, backend]

    embedder = _forwarding_recorder(
        monkeypatch,
        input_embedders,
        "atom_attention_encoder",
        (arr(N, C), None, None, None),
    )
    # `linear` is the identity here, so the relpos term has to arrive at the
    # width of the concatenated single input: three C-wide blocks plus one.
    monkeypatch.setattr(
        input_embedders, "relpos_complex", lambda *a, **k: arr(N, N, 3 * C + 1)
    )
    monkeypatch.setattr(input_embedders, "linear", lambda x, *a, **k: x)
    input_embedders.input_embedder(
        {
            "restype": arr(N, C),
            "profile": arr(N, C),
            "deletion_mean": arr(N),
            "token_bonds": arr(N, N),
        },
        SimpleNamespace(
            atom_attn_enc=None,
            linear_s=None,
            linear_z_i=None,
            linear_z_j=None,
            linear_relpos=None,
            linear_token_bonds=None,
        ),
        n_query=2,
        n_key=4,
        atom_heads=1,
        n_token=N,
        max_relative_idx=2,
        max_relative_chain=2,
        glu_backend=backend,
    )
    assert embedder == [backend]
