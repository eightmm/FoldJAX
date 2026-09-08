"""Context parallelism must not change what AlphaFold 3 computes.

AlphaFold 3's network is the publisher's own Haiku source, vendored verbatim,
so its CP path is installed with ``hk.intercept_methods`` rather than written
into the model.  Two properties have to hold and neither is visible from the
code alone:

* the replacement addresses the same parameters -- it is handed the *serial*
  parameter tree, so a renamed or restructured Haiku call fails outright;
* the sharded program computes the serial answer on a mesh whose size does not
  divide the token count.

A mesh needs more than one device and the device count is fixed at process
start, so the checks run in a subprocess with four forced CPU devices.  The
probes also assert a collective is present: a CP arm that silently fell back to
the serial executable would otherwise pass with an exactly zero difference.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from tests.models.cp_probe_env import inherited_environment

pytestmark = pytest.mark.skipif(
    not __import__("foldjax.models.alphafold3.build", fromlist=["build"]).is_ready(),
    reason="the AlphaFold 3 runtime is not prepared in this store",
)


_PREAMBLE = textwrap.dedent(
    """
    import os

    import haiku as hk
    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel
    from foldjax.models.alphafold3 import _upstream

    _upstream.ensure_registered()

    from alphafold3.model import model_config
    from alphafold3.model.network import modules

    from foldjax.models.alphafold3._cp import context_parallel_modules

    DEVICES = int(os.environ["FOLDJAX_CP_PROBE_DEVICES"])
    # Indivisible by the mesh on purpose: the padding and slicing around
    # `shard_map` is only exercised when the rows do not divide evenly.
    N, C = 13, 16

    GLOBAL = model_config.GlobalConfig(
        bfloat16='none', flash_attention_implementation='xla'
    )

    # Upstream only chunks rows above 1,536 (attention) and 2,048 (transition),
    # so at a probe-sized token count both loops are a single pass and neither
    # CP path is reached. These widths force both at N=13: the transition's
    # chunking is what CP moves from the caller onto local rows, and the
    # attention's is the loop that iterates the sharded axis.
    GLOBAL_CHUNKED = model_config.GlobalConfig(
        bfloat16='none',
        flash_attention_implementation='xla',
        pair_attention_chunk_size=((None, 3),),
        pair_transition_shard_spec=((None, 4),),
    )

    rng = np.random.default_rng(0)
    act = jnp.asarray(rng.normal(size=(N, N, C)).astype(np.float32))
    pair_mask = jnp.asarray(
        (rng.random((N, N)) > 0.15).astype(np.float32)
    )


    def collectives(text):
        return {
            name: text.count(name)
            for name in ('all-gather', 'all-to-all', 'collective-permute')
        }


    def arms(build, *args):
        \"\"\"Serial and CP results, each from its own fresh transform.

        Tracing the same closure twice would hand the CP arm the serial
        executable and make any difference impossible to see.
        \"\"\"
        serial = hk.transform(build())
        params = serial.init(jax.random.PRNGKey(0), *args)
        reference = jax.jit(serial.apply)(params, None, *args)

        sharded = hk.transform(build())
        with context_parallel(DEVICES):
            with context_parallel_modules():
                run = jax.jit(lambda p, *a: sharded.apply(p, None, *a))
                text = run.lower(params, *args).compile().as_text()
                result = run(params, *args)
        return reference, result, collectives(text)
    """
)


_GRID_ATTENTION_PROBE = _PREAMBLE + textwrap.dedent(
    """
    for transpose in (False, True):
        def build(transpose=transpose):
            def fn(act, pair_mask):
                return modules.GridSelfAttention(
                    modules.GridSelfAttention.Config(num_head=4),
                    GLOBAL,
                    transpose=transpose,
                    name='pair_attention',
                )(act, pair_mask)
            return fn

        reference, result, counts = arms(build, act, pair_mask)
        # A general guard, not a gate on the padding value: both kernels return
        # a finite average for a fully masked query, so this does not
        # distinguish the two mask paddings. It catches other NaN sources.
        assert bool(jnp.isfinite(result).all()), transpose
        np.testing.assert_allclose(reference, result, atol=2e-6, rtol=2e-6)

        # The pair-bias projection is gathered in both orientations.
        assert counts['all-gather'] >= 1, (transpose, counts)
        if transpose:
            # And the transposed orientation reshards rows to columns twice.
            moved = counts['all-to-all'] + counts['collective-permute']
            assert moved >= 2, (transpose, counts)
        print(f'GRID_ATTENTION_TRANSPOSE_{transpose}_OK', counts)

    print('GRID_ATTENTION_PARITY_OK')
    """
)


_TRIANGLE_PROBE = _PREAMBLE + textwrap.dedent(
    """
    # Both equations and both projection paths: the fused kernel is the one
    # that has to move onto local rows, and the unfused branch is the fallback
    # that must keep answering the same thing.
    for equation in ('ikc,jkc->ijc', 'kjc,kic->ijc'):
        for use_glu_kernel in (True, False):
            def build(equation=equation, use_glu_kernel=use_glu_kernel):
                def fn(act, pair_mask):
                    return modules.TriangleMultiplication(
                        modules.TriangleMultiplication.Config(
                            equation=equation, use_glu_kernel=use_glu_kernel
                        ),
                        GLOBAL,
                        name='triangle_multiplication',
                    )(act, pair_mask)
                return fn

            reference, result, counts = arms(build, act, pair_mask)
            assert bool(jnp.isfinite(result).all()), (equation, use_glu_kernel)
            np.testing.assert_allclose(reference, result, atol=2e-6, rtol=2e-6)
            # One operand crosses the mesh whichever equation this is: gathered
            # for the outgoing direction, reduced for the incoming one.
            assert sum(counts.values()) >= 1, (equation, counts)
            print(f'TRIANGLE_{equation}_glu{use_glu_kernel}_OK', counts)

    print('TRIANGLE_PARITY_OK')
    """
)


_PAIRFORMER_PROBE = _PREAMBLE + textwrap.dedent(
    """
    # One whole iteration: two triangle multiplications, both attention
    # orientations, and the pair transition whose row chunking CP moves onto
    # local rows. Run once at upstream's widths and once with both row loops
    # forced on, because at probe size upstream takes neither.
    for label, global_config in (('plain', GLOBAL), ('chunked', GLOBAL_CHUNKED)):
        def build(global_config=global_config):
            def fn(act, pair_mask):
                return modules.PairFormerIteration(
                    modules.PairFormerIteration.Config(num_layer=1),
                    global_config,
                    with_single=False,
                    name='trunk_pairformer',
                )(act=act, pair_mask=pair_mask)
            return fn

        reference, result, counts = arms(build, act, pair_mask)
        assert bool(jnp.isfinite(result).all()), label
        np.testing.assert_allclose(reference, result, atol=2e-5, rtol=2e-5)
        assert sum(counts.values()) >= 1, (label, counts)
        print(f'PAIRFORMER_{label}_OK', counts)

    # Turning the caller's transition chunking off is a communication property,
    # not a numerical one: leaving it on still computes the right answer,
    # because the partitioner satisfies the row loop by gathering the pair it
    # was meant to keep split. Parity cannot see that, so compare the two
    # programs directly -- with the interception removed, the same arm must
    # move strictly more.
    import contextlib

    from foldjax.models.alphafold3 import _cp as af3_cp

    # Only the chunking half is removed. Dropping the whole interception would
    # also drop the row seam, and then the two arms differ in two ways.
    original = af3_cp._without_row_chunking
    af3_cp._without_row_chunking = lambda module: contextlib.nullcontext()
    try:
        control_reference, control_result, control = arms(build, act, pair_mask)
    finally:
        af3_cp._without_row_chunking = original

    np.testing.assert_allclose(
        control_reference, control_result, atol=2e-5, rtol=2e-5
    )
    assert sum(counts.values()) < sum(control.values()), (counts, control)
    print('PAIRFORMER_CHUNKING_OFF_MOVES_LESS', counts, control)

    print('PAIRFORMER_PARITY_OK')
    """
)


_EVOFORMER_PROBE = _PREAMBLE + textwrap.dedent(
    """
    # The MSA half of the trunk: the outer product mean, whose *output* is the
    # sharded pair while its input is not, and MSA attention, which reads the
    # pair only as a head projection.
    N_MSA, C_MSA = 5, 16
    msa_act = jnp.asarray(rng.normal(size=(N_MSA, N, C_MSA)).astype(np.float32))
    msa_mask = jnp.asarray((rng.random((N_MSA, N)) > 0.2).astype(np.float32))

    def config(chunk_size):
        return modules.EvoformerIteration.Config(
            outer_product_mean=modules.OuterProductMean.Config(
                chunk_size=chunk_size, num_outer_channel=8
            ),
        )

    # chunk_size 3 does not divide 13 either, so the loop's own tail runs
    # inside the shard as well as the mesh's.
    for label, chunk_size, global_config in (
        ('plain', 128, GLOBAL),
        ('chunked', 3, GLOBAL_CHUNKED),
    ):
        def build(chunk_size=chunk_size, global_config=global_config):
            def fn(msa_act, msa_mask, pair_act, pair_mask):
                out = modules.EvoformerIteration(
                    config(chunk_size), global_config, name='evoformer_iteration'
                )(
                    {'msa': msa_act, 'pair': pair_act},
                    {'msa': msa_mask, 'pair': pair_mask},
                )
                return out['msa'], out['pair']
            return fn

        reference, result, counts = arms(
            build, msa_act, msa_mask, act, pair_mask
        )
        for name, want, got in zip(('msa', 'pair'), reference, result):
            assert bool(jnp.isfinite(got).all()), (label, name)
            np.testing.assert_allclose(want, got, atol=2e-5, rtol=2e-5)
        assert sum(counts.values()) >= 1, (label, counts)
        print(f'EVOFORMER_{label}_OK', counts)

    print('EVOFORMER_PARITY_OK')
    """
)


_LAYER_STACK_PROBE = _PREAMBLE + textwrap.dedent(
    """
    # The trunk does not call the iteration 48 times; it scans it once through
    # `hk.experimental.layer_stack`, with stacked parameters and the pair as a
    # carry. That is where an interceptor can be traced under conditions the
    # single-call probes never reach, and where a carry whose sharding is not
    # pinned on both sides drifts back to replicated between layers.
    LAYERS = 3

    def build():
        def fn(act, pair_mask):
            def layer(act):
                return modules.PairFormerIteration(
                    modules.PairFormerIteration.Config(num_layer=LAYERS),
                    GLOBAL_CHUNKED,
                    with_single=False,
                    name='trunk_pairformer',
                )(act=act, pair_mask=pair_mask)

            return hk.experimental.layer_stack(LAYERS)(layer)(act)
        return fn

    reference, result, counts = arms(build, act, pair_mask)
    assert bool(jnp.isfinite(result).all())
    np.testing.assert_allclose(reference, result, atol=6e-5, rtol=6e-5)

    # One traced body, so the collective counts must look like one layer's, not
    # three. A carry that fell back to replicated between layers would have to
    # re-split it every time round the scan.
    def build_single():
        def fn(act, pair_mask):
            return modules.PairFormerIteration(
                modules.PairFormerIteration.Config(num_layer=1),
                GLOBAL_CHUNKED,
                with_single=False,
                name='trunk_pairformer',
            )(act=act, pair_mask=pair_mask)
        return fn

    _, _, single = arms(build_single, act, pair_mask)
    assert sum(counts.values()) <= sum(single.values()) + 4, (counts, single)
    print('LAYER_STACK_PARITY_OK', counts, 'single-layer:', single)
    """
)


_TRANSITION_SCOPE_PROBE = _PREAMBLE + textwrap.dedent(
    """
    # `TransitionBlock` serves three streams and CP shards only one of them.
    # The MSA's leading axis is alignment depth rather than tokens, so splitting
    # it there would still be arithmetically correct -- parity cannot tell the
    # two apart. What separates them is that the untouched streams must move
    # nothing at all.
    single_act = jnp.asarray(rng.normal(size=(N, C)).astype(np.float32))
    msa_act = jnp.asarray(rng.normal(size=(5, N, C)).astype(np.float32))

    def block(name, global_config=GLOBAL_CHUNKED):
        def build():
            def fn(x):
                return modules.TransitionBlock(
                    modules.TransitionBlock.Config(), global_config, name=name
                )(x)
            return fn
        return build

    for name, value, expected in (
        ('pair_transition', act, 'moves'),
        ('msa_transition', msa_act, 'still'),
        ('single_transition', single_act, 'still'),
    ):
        reference, result, counts = arms(block(name), value)
        np.testing.assert_allclose(reference, result, atol=2e-6, rtol=2e-6)
        if expected == 'still':
            assert sum(counts.values()) == 0, (name, counts)
        else:
            assert sum(counts.values()) >= 1, (name, counts)
        print(f'TRANSITION_{name}_{expected}_OK', counts)

    print('TRANSITION_SCOPE_OK')
    """
)


_BACKEND_CONTEXT_PROBE = _PREAMBLE + textwrap.dedent(
    """
    # The backend enters the mesh and the replacements as one context. Both
    # have to be live at the same time or the sharded program is never built:
    # a mesh with upstream's modules gathers everything back, and the
    # replacements without a mesh forward to upstream.
    from foldjax.backends.alphafold3 import _context_parallel
    from foldjax.models._cp import cp_mesh, cp_shards

    installed = []

    class Probe(hk.Module):
        def __call__(self, x):
            return x

    import foldjax.models.alphafold3._cp as af3_cp
    af3_cp._INTERCEPTED['Probe'] = lambda module, next_f, args, kwargs: (
        installed.append(cp_shards()) or next_f(*args, **kwargs)
    )

    def run():
        return hk.transform(lambda x: Probe(name='probe')(x)).apply(
            {}, None, jnp.zeros((2,))
        )

    with _context_parallel(tuple(jax.devices()[:DEVICES]), 'auto'):
        assert cp_mesh() is not None
        assert cp_shards() == DEVICES
        run()

    assert installed == [DEVICES], installed
    assert cp_mesh() is None
    run()
    assert installed == [DEVICES], 'the replacements outlived the context'
    print('BACKEND_CONTEXT_OK', installed)
    """
)


def _run_probe(source: str, *, devices: int = 4) -> str:
    environment = {
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
        "FOLDJAX_CP_PROBE_DEVICES": str(devices),
        **inherited_environment(),
    }
    # The vendored runtime is resolved from the FoldJAX store, not the source
    # tree, so the child cannot find it without this.
    if "FOLDJAX_HOME" in os.environ:
        environment["FOLDJAX_HOME"] = os.environ["FOLDJAX_HOME"]
    completed = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def test_grid_self_attention_matches_the_unsharded_form() -> None:
    """Both orientations, on a mesh size that does not divide the tokens."""
    assert "GRID_ATTENTION_PARITY_OK" in _run_probe(_GRID_ATTENTION_PROBE)


def test_triangle_multiplication_matches_the_unsharded_form() -> None:
    """Both equations, fused and unfused, on an indivisible token count."""
    assert "TRIANGLE_PARITY_OK" in _run_probe(_TRIANGLE_PROBE)


def test_pairformer_iteration_matches_the_unsharded_form() -> None:
    """The whole pair block, including the transition CP stops chunking."""
    assert "PAIRFORMER_PARITY_OK" in _run_probe(_PAIRFORMER_PROBE)


def test_evoformer_iteration_matches_the_unsharded_form() -> None:
    """The MSA half: outer product mean and the pair-conditioned attention."""
    assert "EVOFORMER_PARITY_OK" in _run_probe(_EVOFORMER_PROBE)


def test_layer_stack_keeps_the_pair_on_its_rows() -> None:
    """The trunk scans one traced body; the seam has to survive that."""
    assert "LAYER_STACK_PARITY_OK" in _run_probe(_LAYER_STACK_PROBE)


def test_only_the_pair_transition_is_sharded() -> None:
    """The single and MSA streams must move nothing under this layout."""
    assert "TRANSITION_SCOPE_OK" in _run_probe(_TRANSITION_SCOPE_PROBE)


def test_cp_options_are_validated_before_anything_runs() -> None:
    """A bad mesh request must fail at the option, not after featurisation."""
    from foldjax.backends.alphafold3 import AlphaFold3Backend

    backend = AlphaFold3Backend()
    backend.validate_native_options({"cp_devices": 4, "cp_layout": "2d"})
    backend.validate_native_options({"cp_devices": 1})

    with pytest.raises(ValueError, match="cp_devices"):
        backend.validate_native_options({"cp_devices": 0})
    with pytest.raises(ValueError, match="cp_devices"):
        backend.validate_native_options({"cp_devices": "four"})
    # Eight devices is the size the deployment note reaches for, and it is
    # exactly the count the square layout cannot take.
    with pytest.raises(ValueError, match="perfect-square"):
        backend.validate_native_options({"cp_devices": 8, "cp_layout": "2d"})
    with pytest.raises(ValueError):
        backend.validate_native_options({"cp_devices": 4, "cp_layout": "3d"})


def test_the_mesh_is_part_of_the_compilation_namespace() -> None:
    """A serial and a sharded run are two programs, not one run two ways."""
    from foldjax.backends.alphafold3 import AlphaFold3Backend

    assert {"cp_devices", "cp_layout"} <= set(AlphaFold3Backend.compile_options)
    assert {"cp_devices", "cp_layout"} <= AlphaFold3Backend.native_options


def test_one_device_keeps_the_publisher_route() -> None:
    """A single-device request must not build a mesh at all.

    The interception is keyed on an active mesh, so proving no mesh exists is
    what proves the serial route still runs the publisher's own modules.  It
    also has to hold for a device object the mesh machinery would reject, which
    is why this passes a bare sentinel.
    """
    from foldjax.backends.alphafold3 import _context_parallel
    from foldjax.models._cp import cp_mesh

    with _context_parallel((object(),), "auto"):
        assert cp_mesh() is None
    assert cp_mesh() is None


def test_the_backend_context_installs_both_halves_and_removes_them() -> None:
    """The mesh and the replacements go in together and come out together."""
    assert "BACKEND_CONTEXT_OK" in _run_probe(_BACKEND_CONTEXT_PROBE)


def test_a_serial_request_keeps_its_compilation_namespace(tmp_path) -> None:
    """Adding a mesh knob must not orphan what the serial route already cached.

    The compilation profile namespaces the persistent Tokamax store, so an
    entry that appears whether or not a mesh was asked for would re-autotune
    every existing AlphaFold 3 shape once.
    """
    import json

    from foldjax.backends.alphafold3 import AlphaFold3Backend
    from foldjax.schema import PredictionRequest

    def request(options: dict) -> PredictionRequest:
        path = tmp_path / "job.json"
        path.write_text(json.dumps({"name": "job", "shape": 8}))
        return PredictionRequest(
            model="alphafold3",
            input=path,
            input_format="native",
            weights=tmp_path,
            output_dir=tmp_path / "out",
            seed=0,
            num_seeds=1,
            cache_dir=None,
            use_compile_cache=False,
            options=options,
            resume=False,
            on_error="stop",
        )

    backend = AlphaFold3Backend()
    serial = backend.cache_profile(request({}))
    assert "cp_devices" not in serial
    assert "cp_layout" not in serial

    sharded = backend.cache_profile(request({"cp_devices": 4, "cp_layout": "1d"}))
    assert sharded["cp_devices"] == 4
    assert sharded["cp_layout"] == "1d"
    assert {
        name: value
        for name, value in sharded.items()
        if name not in ("cp_devices", "cp_layout")
    } == serial
