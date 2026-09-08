"""Context-parallel AlphaFold 3 without editing the vendored publisher source.

The other five FoldJAX ports are this project's own code, so their
context-parallel seams were written in place.  AlphaFold 3 is not: its network
is the publisher's own Haiku source, vendored verbatim at ``._upstream`` and
compared bit-for-bit against the released runtime.  Editing it would spend the
one property that model has.

So the sharded program is built with :func:`haiku.intercept_methods` instead.
Each replacement below is upstream's own ``__call__`` body with sharding
constraints spliced *between* its Haiku calls -- never restructured, never
renamed.  That keeps every parameter path identical to the checkpoint, and it
keeps the diff against upstream readable: what this file adds is exactly the
lines a reviewer can point at.

Two facts decide the shape of the sharded attention, both measured rather than
assumed (2026-09-08, tokamax 0.0.13):

* Haiku parameters cannot be *created* inside ``shard_map`` -- the new
  parameter escapes the transformation as a tracer -- but ``apply`` reads them
  through the enclosing frame and works.  Inference only ever applies.
* ``tokamax.dot_product_attention`` keeps its Triton kernel inside
  ``shard_map`` and costs no collective when the sharded axis is the attention
  *batch* axis.  Sharding the sequence axis instead reconstructs the whole key
  and value; tokamax's public API has no ``k_sharding``, so there is no ring to
  ask for.  Hence the transposed orientation reshards rows to columns rather
  than attending across a sharded sequence.
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Callable, Iterator
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from foldjax.models._cp import (
    cp_mesh,
    cp_row_shards,
    pair_row_spec,
    shard_pair_rows,
)

#: Upstream module classes this file replaces, by class name.  Anything absent
#: runs the publisher's implementation unchanged.  A replacement is called as
#: ``(module, next_f, args, kwargs)`` and may delegate to ``next_f``, which is
#: upstream's own bound method: a replacement that only needs to change where
#: the work runs calls it from inside ``shard_map`` rather than restating it.
_INTERCEPTED: dict[str, Any] = {}


def _replicated(x: jax.Array) -> jax.Array:
    """Gather ``x`` onto every device; identity outside a CP mesh."""

    mesh = cp_mesh()
    if mesh is None:
        return x
    return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, PartitionSpec()))


def _pad_rows(x: jax.Array, pad: int, *, axis: int = 0, value: Any = 0) -> jax.Array:
    if not pad:
        return x
    width = [(0, 0)] * x.ndim
    width[axis] = (0, pad)
    return jnp.pad(x, width, constant_values=value)


def _on_local_rows(
    fn: Callable[..., jax.Array],
    act: jax.Array,
    *replicated: jax.Array,
    extra_row_sharded: tuple[jax.Array, ...] = (),
    pad_values: tuple[Any, ...] = (),
) -> jax.Array:
    """Run ``fn`` on each device's own pair rows, padding up to the mesh.

    ``act`` and every ``extra_row_sharded`` argument are split on axis 0 in the
    order ``fn`` takes them; ``replicated`` arguments follow, whole.  The rows
    are padded to a multiple of the mesh because ``shard_map`` requires the
    sharded axis to divide evenly, and sliced back afterwards -- through a
    constraint, or the partitioner takes the slice as licence to replicate.
    """

    mesh = cp_mesh()
    if mesh is None:
        return fn(act, *extra_row_sharded, *replicated)

    rows = act.shape[0]
    pad = (-rows) % cp_row_shards()
    row_sharded = (act, *extra_row_sharded)
    values = pad_values or (0,) * len(row_sharded)
    padded = tuple(
        _pad_rows(x, pad, value=value)
        for x, value in zip(row_sharded, values, strict=True)
    )
    out = jax.shard_map(
        fn,
        mesh=mesh,
        in_specs=(
            *(pair_row_spec(x.ndim, row_axis=0) for x in padded),
            *(PartitionSpec() for _ in replicated),
        ),
        out_specs=pair_row_spec(act.ndim, row_axis=0),
        check_vma=False,
    )(*padded, *replicated)
    if pad:
        out = shard_pair_rows(jax.lax.slice_in_dim(out, 0, rows, axis=0), row_axis=0)
    return out


def _on_local_tokens(
    fn: Callable[..., jax.Array],
    batched: jax.Array,
    *replicated: jax.Array,
    token_axis: int,
    out_ndim: int = 3,
) -> jax.Array:
    """Split ``batched`` on ``token_axis`` and land the result on pair rows.

    The sibling of :func:`_on_local_rows` for the one module whose split axis
    and output axis are not the same one.
    """

    mesh = cp_mesh()
    if mesh is None:
        return fn(batched, *replicated)

    size = batched.shape[token_axis]
    pad = (-size) % cp_row_shards()
    padded = _pad_rows(batched, pad, axis=token_axis)
    out = jax.shard_map(
        fn,
        mesh=mesh,
        in_specs=(
            pair_row_spec(padded.ndim, row_axis=token_axis),
            *(PartitionSpec() for _ in replicated),
        ),
        out_specs=pair_row_spec(out_ndim, row_axis=0),
        check_vma=False,
    )(padded, *replicated)
    if pad:
        out = shard_pair_rows(jax.lax.slice_in_dim(out, 0, size, axis=0), row_axis=0)
    return out


def _grid_self_attention(module: Any, act: jax.Array, pair_mask: jax.Array):
    """CP form of ``modules.GridSelfAttention.__call__``.

    Upstream's body, with three constraints spliced in:

    * the pair-bias projection is gathered to ``[heads, N, N]``.  This is the
      only tensor the sharded trunk reconstructs, and at four heads it is
      ``32x`` smaller than the pair state it came from;
    * the transposed orientation reshards rows to columns, so the attention
      batch axis is the sharded one in both orientations and the fused kernel
      never sees a split sequence;
    * the row loop runs under ``shard_map`` on whole local rows, because it
      iterates the sharded axis.

    ``chunk_size`` is deliberately taken from the *global* token count.  Inside
    ``shard_map`` the body would see ``N / P`` rows and pick a larger chunk than
    upstream does at the real size.
    """

    from alphafold3.model.components import haiku_modules as hm
    from alphafold3.model.network import modules

    assert len(act.shape) == 3
    assert len(pair_mask.shape) == 2

    pair_mask = jnp.swapaxes(pair_mask, -1, -2)
    act = hm.LayerNorm(name="act_norm")(act)

    nonbatched_bias = hm.Linear(
        module.config.num_head, use_bias=False, name="pair_bias_projection"
    )(act)
    nonbatched_bias = jnp.transpose(nonbatched_bias, [2, 0, 1])
    # Spliced: the bias indexes both token axes and every row needs all of it.
    nonbatched_bias = _replicated(nonbatched_bias)

    num_residues = act.shape[0]

    chunk_size = modules.get_shard_size(
        num_residues, module.global_config.pair_attention_chunk_size
    )

    if module.transpose:
        act = jnp.swapaxes(act, -2, -3)
        # Spliced: an all-to-all, so the attention batch axis is sharded again.
        act = shard_pair_rows(act, row_axis=0)

    pair_mask = pair_mask[:, None, None, :].astype(jnp.bool_)

    act = _attention_on_local_rows(module, act, pair_mask, nonbatched_bias, chunk_size)

    if module.transpose:
        act = jnp.swapaxes(act, -2, -3)
        # Spliced: the all-to-all back, so the caller's pair stays row-sharded.
        act = shard_pair_rows(act, row_axis=0)

    return act


def _attention_on_local_rows(
    module: Any,
    act: jax.Array,
    pair_mask: jax.Array,
    nonbatched_bias: jax.Array,
    chunk_size: int | None,
) -> jax.Array:
    """Run upstream's chunked row loop on each device's own rows."""

    from alphafold3.model.components import mapping

    def local_rows(act_l, pair_mask_l, bias_l):
        return mapping.inference_subbatch(
            module._attention,  # noqa: SLF001 - upstream's own transparent method
            chunk_size,
            batched_args=[act_l, pair_mask_l],
            nonbatched_args=[bias_l],
        )

    # The mask's leading axis is the attention batch, so a padded row is a
    # padded *query*, and ``True`` lets it attend real keys. Both current
    # kernels happen to return a finite uniform average for a fully masked
    # query instead (measured 2026-09-08, xla and triton alike), so this is not
    # repairing a NaN that exists today -- it is declining to depend on a kernel
    # behaviour nothing in the API promises, for a row that is sliced away.
    return _on_local_rows(
        local_rows,
        act,
        nonbatched_bias,
        extra_row_sharded=(pair_mask,),
        pad_values=(0, True),
    )


def _triangle_multiplication(module: Any, act: jax.Array, mask: jax.Array):
    """CP form of ``modules.TriangleMultiplication.__call__``.

    Upstream's body with the gated-linear-unit moved onto local rows and the
    einsum's operands placed explicitly.  The two equations shard differently
    and neither is free:

    * outgoing (``cik,cjk->cij``) keeps output row ``i`` local, but row ``j``
      of the right operand runs over the whole token axis, so that operand is
      gathered -- one full ``[C, N, N]``;
    * incoming (``ckj,cki->cij``) contracts over ``k``, which *is* the sharded
      axis, so each device holds a partial sum and the partitioner reduces
      them; the result arrives replicated and is re-sharded.

    This is Fold-CP's documented one-dimensional concession.  The square Cannon
    schedule is what removes it, and it is not wired here.
    """

    from alphafold3.model.components import haiku_modules as hm

    mask = mask[None, ...]
    num_channels = act.shape[-1]
    equation = {
        "ikc,jkc->ijc": "cik,cjk->cij",
        "kjc,kic->ijc": "ckj,cki->cij",
    }[module.config.equation]

    act = hm.LayerNorm(name="left_norm_input")(act)
    input_act = act

    if module.config.use_glu_kernel:
        weights_projection, _ = hm.haiku_linear_get_params(
            act, num_output=num_channels * 2, name="projection"
        )
        weights_gate, _ = hm.haiku_linear_get_params(
            act,
            num_output=num_channels * 2,
            initializer=module.global_config.final_init,
            name="gate",
        )
        weights_glu = jnp.stack([weights_gate, weights_projection], axis=1)

        # Spliced: the fused kernel is an FFI call the partitioner cannot
        # split, so it runs under `shard_map` on whole local rows instead.
        projection = _on_local_rows(_glu_sigmoid, act, weights_glu)
        projection = jnp.transpose(projection, (2, 0, 1))
        projection *= mask
    else:
        projection = hm.Linear(num_channels * 2, name="projection")(act)
        projection = jnp.transpose(projection, (2, 0, 1))
        projection *= mask

        gate = hm.Linear(
            num_channels * 2,
            name="gate",
            bias_init=1.0,
            initializer=module.global_config.final_init,
        )(act)
        gate = jnp.transpose(gate, (2, 0, 1))
        projection *= jax.nn.sigmoid(gate)

    projection = projection.reshape(num_channels, 2, *projection.shape[1:])
    a, b = jnp.split(projection, 2, axis=1)
    a, b = jnp.squeeze(a, axis=1), jnp.squeeze(b, axis=1)
    # Spliced: the channel axis leads here, so the pair rows are axis 1.
    a = shard_pair_rows(a, row_axis=1)
    b = (
        _replicated(b)
        if module.config.equation == "ikc,jkc->ijc"
        else shard_pair_rows(b, row_axis=1)
    )
    act = jnp.einsum(equation, a, b)
    act = shard_pair_rows(act, row_axis=1)
    act = hm.LayerNorm(name="center_norm", axis=0, param_axis=0)(act)

    act = jnp.transpose(act, (1, 2, 0))
    act = hm.Linear(
        num_channels,
        initializer=module.global_config.final_init,
        name="output_projection",
    )(act)

    gate_out = hm.Linear(
        num_channels,
        name="gating_linear",
        bias_init=1.0,
        initializer=module.global_config.final_init,
    )(input_act)
    act *= jax.nn.sigmoid(gate_out)

    return act


def _outer_product_mean(module: Any, act: jax.Array, mask: jax.Array):
    """CP form of ``modules.OuterProductMean.__call__``.

    This is the one module whose *output* is the sharded pair while its input
    is not.  In ``'abc,ade,cef->bdf'`` the output row ``b`` is the left
    operand's token axis and ``d`` runs over the whole of the right one, so the
    left projection is split on its token axis -- axis 1, not axis 0 -- and the
    right projection stays whole.  It is MSA-shaped rather than pair-shaped, so
    keeping it whole costs a fraction of what the pair would.

    Upstream's chunk loop walks the same axis, which is why this runs under
    ``shard_map``; its width is a config constant here rather than derived from
    the token count, so no global-size correction is needed.
    """

    from alphafold3.model.components import haiku_modules as hm
    from alphafold3.model.components import mapping

    mask = mask[..., None]
    act = hm.LayerNorm(name="layer_norm_input")(act)

    left_act = mask * hm.Linear(
        module.config.num_outer_channel,
        initializer="linear",
        name="left_projection",
    )(act)

    right_act = mask * hm.Linear(
        module.config.num_outer_channel,
        initializer="linear",
        name="right_projection",
    )(act)

    import haiku as hk

    if module.global_config.final_init == "zeros":
        w_init = hk.initializers.Constant(0.0)
    else:
        w_init = hk.initializers.VarianceScaling(scale=2.0, mode="fan_in")

    output_w = hk.get_parameter(
        "output_w",
        shape=(
            module.config.num_outer_channel,
            module.config.num_outer_channel,
            module.num_output_channel,
        ),
        dtype=act.dtype,
        init=w_init,
    )
    output_b = hk.get_parameter(
        "output_b",
        shape=(module.num_output_channel,),
        dtype=act.dtype,
        init=hk.initializers.Constant(0.0),
    )

    def local_tokens(left_l, right_l, output_w_l, output_b_l):
        def compute_chunk(left_act):
            out = jnp.einsum("abc,ade,cef->bdf", left_act, right_l, output_w_l)
            return out + output_b_l

        return mapping.inference_subbatch(
            compute_chunk,
            module.config.chunk_size,
            batched_args=[left_l],
            nonbatched_args=[],
            input_subbatch_dim=1,
            output_subbatch_dim=0,
        )

    # Spliced: the left operand's token axis becomes the output's pair rows.
    act = _on_local_tokens(
        local_tokens, left_act, right_act, output_w, output_b, token_axis=1
    )

    epsilon = 1e-3
    norm = jnp.einsum("abc,adc->bdc", mask, mask)
    return act / (epsilon + norm)


def _msa_attention(module: Any, act: jax.Array, mask: jax.Array, pair_act: jax.Array):
    """CP form of ``modules.MSAAttention.__call__``.

    The pair enters only as a head projection, so the same trick the pair-bias
    uses works here: reduce first, gather the small thing.  ``logits`` is
    ``[heads, N, N]`` against a pair of ``[N, N, 128]``, and everything after
    it is upstream's, on replicated MSA data.  Leaving the MSA replicated is a
    deliberate v1 choice -- the einsum below would otherwise hand the MSA stack
    a token-sharded activation, which is a placement change this layout has not
    measured.

    That replication is a *dependency*, not an observation.  The ``hqk,bkhc``
    contraction below runs over the whole token axis, so a later layout that
    splits the MSA on its tokens breaks here first.
    """

    from alphafold3.model.components import haiku_modules as hm

    act = hm.LayerNorm(name="act_norm")(act)
    pair_act = hm.LayerNorm(name="pair_norm")(pair_act)
    logits = hm.Linear(module.config.num_head, use_bias=False, name="pair_logits")(
        pair_act
    )
    logits = jnp.transpose(logits, [2, 0, 1])
    # Spliced: the projection is row-sharded because the pair was.
    logits = _replicated(logits)
    logits += 1e9 * (jnp.max(mask, axis=0) - 1.0)
    weights = jax.nn.softmax(logits, axis=-1)
    num_channels = act.shape[-1]
    value_dim = num_channels // module.config.num_head
    v = hm.Linear(
        [module.config.num_head, value_dim], use_bias=False, name="v_projection"
    )(act)
    v_avg = jnp.einsum("hqk, bkhc -> bqhc", weights, v)
    v_avg = jnp.reshape(v_avg, v_avg.shape[:-2] + (-1,))
    gate_values = hm.Linear(
        module.config.num_head * value_dim,
        bias_init=1.0,
        initializer="zeros",
        name="gating_query",
    )(act)
    v_avg *= jax.nn.sigmoid(gate_values)

    return hm.Linear(
        num_channels,
        initializer=module.global_config.final_init,
        name="output_projection",
    )(v_avg)


def _glu_sigmoid(act: jax.Array, weights: jax.Array) -> jax.Array:
    import tokamax

    return tokamax.gated_linear_unit(act, weights, activation=jax.nn.sigmoid)


def _transition_block(module: Any, next_f: Callable[..., Any], args, kwargs):
    """Move the pair transition onto local rows, body unchanged.

    ``TransitionBlock`` also serves the single and MSA streams, which this
    layout leaves replicated -- and the MSA's leading axis is alignment depth,
    not tokens, so sharding it on axis 0 would be meaningless.  Upstream names
    the three call sites ``pair_transition``, ``single_transition`` and
    ``msa_transition``; the name is the checkpoint path, so it is a stabler
    discriminator than the shape.

    The row chunking upstream applies from the *caller* is reproduced here at
    the global token count, on local rows.  See :func:`_without_row_chunking`
    for the other half of that.
    """

    from alphafold3.model.components import mapping
    from alphafold3.model.network import modules

    act = args[0] if args else kwargs["act"]
    if cp_mesh() is None or not module.module_name.endswith("pair_transition"):
        return next_f(*args, **kwargs)

    rest = args[1:]
    chunk_size = modules.get_shard_size(
        act.shape[0], module.global_config.pair_transition_shard_spec
    )

    def local_rows(act_l):
        body = lambda x: next_f(x, *rest, **kwargs)  # noqa: E731
        if chunk_size is None:
            return body(act_l)
        return mapping.inference_subbatch(
            body, chunk_size, batched_args=[act_l], nonbatched_args=[]
        )

    return _on_local_rows(local_rows, act)


@contextlib.contextmanager
def _without_row_chunking(module: Any) -> Iterator[None]:
    """Switch an iteration's own transition chunking off for one call.

    ``PairFormerIteration`` and ``EvoformerIteration`` both wrap the pair
    transition in ``mapping.sharded_apply``, whose row loop would slice the
    sharded axis.  :func:`_transition_block` re-applies the same chunk size on
    local rows, so the flag is turned off here rather than the loop rewritten.
    Leaving it on is not a wrong answer -- the partitioner satisfies the loop by
    gathering the pair the layout exists to keep split -- which is why the gate
    for this is a communication comparison rather than a parity check.

    The config is a plain dataclass and the module holds it by reference, so
    this swaps a copy in and puts the original back.
    """

    if not module.config.shard_transition_blocks:
        yield
        return
    original = module.config
    module.config = dataclasses.replace(original, shard_transition_blocks=False)
    try:
        yield
    finally:
        module.config = original


def _pair_former_iteration(module: Any, next_f: Callable[..., Any], args, kwargs):
    """Hold the pair on its rows across one pairformer layer.

    The constraint is what makes the layout survive ``hk.layer_stack``: the
    stack scans this iteration, so the pair is a carry, and a carry that is not
    pinned on both sides drifts back to replicated between layers.
    """

    def take(act):
        return shard_pair_rows(act, row_axis=0)

    if args:
        args = (take(args[0]), *args[1:])
    else:
        kwargs = {**kwargs, "act": take(kwargs["act"])}

    with _without_row_chunking(module):
        out = next_f(*args, **kwargs)

    if module.with_single:
        act, single_act = out
        return take(act), single_act
    return take(out)


def _evoformer_iteration(module: Any, next_f: Callable[..., Any], args, kwargs):
    """The same seam for the MSA stack, whose pair travels in a dict."""

    activations = args[0] if args else kwargs["activations"]
    activations = {
        **activations,
        "pair": shard_pair_rows(activations["pair"], row_axis=0),
    }
    if args:
        args = (activations, *args[1:])
    else:
        kwargs = {**kwargs, "activations": activations}

    with _without_row_chunking(module):
        out = next_f(*args, **kwargs)

    return {**out, "pair": shard_pair_rows(out["pair"], row_axis=0)}


def _spliced(replacement: Callable[..., Any]) -> Callable[..., Any]:
    """Adapt a replacement that restates the body and ignores ``next_f``."""

    def call(module, next_f, args, kwargs):
        del next_f
        return replacement(module, *args, **kwargs)

    return call


_INTERCEPTED["GridSelfAttention"] = _spliced(_grid_self_attention)
_INTERCEPTED["TriangleMultiplication"] = _spliced(_triangle_multiplication)
_INTERCEPTED["OuterProductMean"] = _spliced(_outer_product_mean)
_INTERCEPTED["MSAAttention"] = _spliced(_msa_attention)
_INTERCEPTED["TransitionBlock"] = _transition_block
_INTERCEPTED["PairFormerIteration"] = _pair_former_iteration
_INTERCEPTED["EvoformerIteration"] = _evoformer_iteration


def _interceptor(next_f, args, kwargs, context):
    """Route intercepted ``__call__``s to their context-parallel form."""

    if context.method_name != "__call__" or cp_mesh() is None:
        return next_f(*args, **kwargs)
    replacement = _INTERCEPTED.get(type(context.module).__name__)
    if replacement is None:
        return next_f(*args, **kwargs)
    return replacement(context.module, next_f, args, kwargs)


def context_parallel_model_runner(runner: Any, *, config: Any, model_dir: Any) -> Any:
    """Upstream's ``ModelRunner`` with its single-device commitment removed.

    ``run_alphafold.ModelRunner`` builds ``jax.jit(..., device=self._device)``
    and puts its features on that same device.  A computation committed to one
    device cannot hold a mesh's collectives, so the *commitment* has to go --
    but the placement does not.  Passing a replicated ``NamedSharding`` where
    the runner expects a device leaves ``run_inference`` working unchanged and
    puts the features on every device of the mesh; only the jit is overridden,
    and the vendored runner file stays verbatim like the network beside it.

    The forward function is rebuilt rather than unwrapped because the jit's
    commitment is baked in at construction and there is no accessor for it.
    Those six lines are upstream's, and they are the only ones copied here.
    """

    import functools

    import haiku as hk

    mesh = cp_mesh()
    if mesh is None:
        raise RuntimeError(
            "context_parallel_model_runner needs an active mesh; "
            "enter foldjax.models._cp.context_parallel() first"
        )

    class _ContextParallelModelRunner(runner.ModelRunner):
        @functools.cached_property
        def _model(self):
            @hk.transform
            def forward_fn(batch):
                return runner.model.Model(self._model_config)(batch)

            return functools.partial(jax.jit(forward_fn.apply), self.model_params)

    return _ContextParallelModelRunner(
        config=config,
        device=NamedSharding(mesh, PartitionSpec()),
        model_dir=model_dir,
    )


@contextlib.contextmanager
def context_parallel_modules() -> Iterator[None]:
    """Install the CP replacements for the duration of one ``apply``.

    Must wrap the *apply* call, not ``init``: the replacements read parameters
    inside ``shard_map``, which can read an existing parameter but cannot
    create one.  Outside an active mesh every replacement forwards to upstream,
    so entering this context on one device changes nothing.
    """

    import haiku as hk

    with hk.intercept_methods(_interceptor):
        yield
