"""Context parallelism must not change what the ESMFold2 trunk computes.

ESMFold2's pair trunk has triangle multiplicative updates and no triangle
attention, so the one-dimensional layout is constraint-only: no ``shard_map``
and no kernel gating. The square grid is not -- both pair axes go on the mesh,
which puts the contracted axis on a different device from the operand that
needs it, so the contraction runs Cannon's schedule inside a ``shard_map`` and
the pair transition takes its row block inside the same shard.

The property to hold is numerical parity of the pair stacks against the
unsharded program, on a mesh whose size does not divide the token count and on
one it does. The model is stochastic *end to end* (random initial pair state,
per-loop LM dropout), so parity is checked at module level with fixed inputs,
where the computation is deterministic. A mesh needs more than one device and
the device count is fixed at process start, so every mesh check runs in a
subprocess with forced CPU devices: four for the row mesh and the 2x2 grid,
nine for the 3x3 grid, which is the smallest grid that can see a sign error in
the ring schedule at all -- on a side of two ``(x + 1) % 2 == (x - 1) % 2``,
so every hop and skew sign is indistinguishable from its opposite.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from foldjax.models._cp import context_parallel, cp_shards
from tests.models.cp_probe_env import inherited_environment


def test_single_shard_context_is_a_no_op() -> None:
    with context_parallel(1) as mesh:
        assert mesh is None
        assert cp_shards() == 1


def test_shard_count_must_match_the_active_mesh() -> None:
    """The static ``cp_shards`` and the ambient mesh are one decision.

    The guard runs before the model touches anything, so garbage inputs are
    fine.
    """
    from foldjax.models.esmfold2.inference import _run

    with pytest.raises(RuntimeError, match="cp_shards=2"):
        _run(None, {}, {}, None, None, 1, False, 2)


def test_compact_token_bond_static_choice_crosses_the_mesh_context(
    monkeypatch,
) -> None:
    """The host proof and ambient CP route must reach one graph decision."""
    from foldjax.models.esmfold2 import inference

    seen = []

    def fake_predict(*args, **kwargs):
        del args
        seen.append(kwargs["compact_token_bond_encoding"])
        return {}

    monkeypatch.setattr(inference.structure_model, "predict", fake_predict)
    with context_parallel(1):
        inference._run(
            None, {}, {}, None, None, 1, False, compact_token_bond_encoding=True
        )
        # A direct low-level call that does not supply the proof stays generic.
        inference._run(None, {}, {}, None, None, 1, False)

    assert seen == [True, False]


def test_compact_language_model_input_crosses_the_mesh_context(monkeypatch) -> None:
    from foldjax.models.esmfold2 import inference

    seen = []

    def fake_predict(*args, **kwargs):
        del args
        seen.append((kwargs["lm_hidden_states"], kwargs["lm_embedding"]))
        return {}

    embedding = object()
    monkeypatch.setattr(inference.structure_model, "predict", fake_predict)
    with context_parallel(1):
        inference._run(
            None,
            {},
            {},
            embedding,
            None,
            1,
            False,
            compact_lm_input=True,
        )
        inference._run(None, {}, {}, embedding, None, 1, False)

    assert seen == [(None, embedding), (embedding, None)]


_DISTOGRAM_ROUTE_PROBE = textwrap.dedent(
    """
    import jax

    from foldjax.models._cp import context_parallel
    from foldjax.models.esmfold2 import inference

    assert jax.device_count() == 4, jax.devices()
    seen = []
    original = inference.structure_model.predict

    def fake_predict(*args, **kwargs):
        del args
        seen.append(kwargs["return_distogram_logits"])
        return {}

    inference.structure_model.predict = fake_predict
    try:
        with context_parallel(4):
            inference._run(
                None,
                {},
                {},
                None,
                None,
                1,
                False,
                4,
                return_distogram_logits=False,
            )
    finally:
        inference.structure_model.predict = original

    assert seen == [False], seen
    print("CP_DISTOGRAM_ROUTE_OK")
    """
)


def test_distogram_choice_crosses_a_forced_four_device_cpu_mesh() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _DISTOGRAM_ROUTE_PROBE],
        capture_output=True,
        text=True,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
            **inherited_environment(),
        },
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "CP_DISTOGRAM_ROUTE_OK" in completed.stdout


_PARITY_PROBE = textwrap.dedent(
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel
    from foldjax.models.esmfold2.models.trunk import folding_trunk

    assert jax.device_count() == 4, jax.devices()
    # `jax.jit` caches its jaxpr on the *callable*, and the mesh lives in a
    # module global the trace reads, so a second `jit` of the same function
    # object replays the unsharded program -- the comparison below was the
    # unsharded program against itself until `clear_caches` was added here.
    # The giveaway was an exactly-zero difference, which a real `shard_map`
    # cannot produce because it reorders float32 accumulation. Removing these
    # calls makes every parity assertion below vacuous.


    C, N, LAYERS = 8, 13, 2  # 13 rows over 4 shards: uneven-shard path
    rng = np.random.default_rng(0)

    def arr(*shape):
        return jnp.asarray(rng.normal(size=shape, scale=0.5), dtype=jnp.float32)

    params = {}
    for index in range(LAYERS):
        block = f"blocks.{index}"
        for tri in ("tri_mul_out", "tri_mul_in"):
            engine = f"{block}.{tri}._engine"
            params[f"{engine}.norm_start.weight"] = arr(C) * 0.1 + 1.0
            params[f"{engine}.norm_start.bias"] = arr(C) * 0.1
            params[f"{engine}.proj_bundle.weight"] = arr(4 * C, C)
            params[f"{engine}.proj_bundle.bias"] = arr(4 * C)
            params[f"{engine}.norm_mix.weight"] = arr(C) * 0.1 + 1.0
            params[f"{engine}.norm_mix.bias"] = arr(C) * 0.1
            params[f"{engine}.proj_emit.weight"] = arr(C, C)
            params[f"{engine}.proj_emit.bias"] = arr(C)
            params[f"{engine}.proj_gate.weight"] = arr(C, C)
            params[f"{engine}.proj_gate.bias"] = arr(C)
        transition = f"{block}.pair_transition"
        params[f"{transition}.norm.weight"] = arr(C) * 0.1 + 1.0
        params[f"{transition}.norm.bias"] = arr(C) * 0.1
        params[f"{transition}.ffn.w12.weight"] = arr(4 * C, C)
        params[f"{transition}.ffn.w3.weight"] = arr(C, 2 * C)

    pair = arr(1, N, N, C)
    mask_np = rng.random(N) > 0.15
    mask = jnp.asarray((mask_np[:, None] & mask_np[None, :])[None].astype(np.float32))

    def run(pair_in):
        return folding_trunk(pair_in, params, n_layers=LAYERS, mask=mask)

    ref = jax.device_get(jax.jit(run)(pair))
    jax.clear_caches()
    with context_parallel(4):
        got = jax.device_get(jax.jit(run)(pair))
    np.testing.assert_allclose(ref, got, atol=3e-5, rtol=3e-5)
    print("CP_PARITY_OK", float(np.abs(ref - got).max()))
    """
)


def test_context_parallel_matches_the_unsharded_trunk() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _PARITY_PROBE],
        capture_output=True,
        text=True,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
            **inherited_environment(),
        },
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "CP_PARITY_OK" in completed.stdout


#: The fixture both grid probes build: two pair-trunk blocks and one MSA
#: encoder block, with parameters small enough to compile on CPU in seconds
#: and an input that distinguishes every grid coordinate. Written as source
#: because the probes run in a subprocess, with `#` comments rather than
#: docstrings for the same reason.
_FIXTURE = textwrap.dedent(
    r"""
    import os
    import re

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel, cp_grid
    from foldjax.models.esmfold2.models.embedders import msa_encoder_block
    from foldjax.models.esmfold2.models.trunk import folding_trunk

    DEVICES = int(os.environ["FOLDJAX_CP_PROBE_DEVICES"])
    SIDE = int(round(DEVICES ** 0.5))
    assert jax.device_count() == DEVICES, jax.devices()
    assert SIDE * SIDE == DEVICES, DEVICES

    C, C_M, HEADS, HEAD_DIM, OPM_HALF, LAYERS, MSA_ROWS = 8, 8, 2, 3, 4, 2, 6


    def arr(rng, *shape):
        return jnp.asarray(rng.normal(size=shape, scale=0.5), dtype=jnp.float32)


    def triangle_params(rng, params, engine):
        params[engine + ".norm_start.weight"] = arr(rng, C) * 0.1 + 1.0
        params[engine + ".norm_start.bias"] = arr(rng, C) * 0.1
        params[engine + ".proj_bundle.weight"] = arr(rng, 4 * C, C)
        params[engine + ".proj_bundle.bias"] = arr(rng, 4 * C)
        params[engine + ".norm_mix.weight"] = arr(rng, C) * 0.1 + 1.0
        params[engine + ".norm_mix.bias"] = arr(rng, C) * 0.1
        params[engine + ".proj_emit.weight"] = arr(rng, C, C)
        params[engine + ".proj_emit.bias"] = arr(rng, C)
        params[engine + ".proj_gate.weight"] = arr(rng, C, C)
        params[engine + ".proj_gate.bias"] = arr(rng, C)


    def transition_params(rng, params, prefix, width):
        params[prefix + ".norm.weight"] = arr(rng, width) * 0.1 + 1.0
        params[prefix + ".norm.bias"] = arr(rng, width) * 0.1
        params[prefix + ".ffn.w12.weight"] = arr(rng, 4 * width, width)
        params[prefix + ".ffn.w3.weight"] = arr(rng, width, 2 * width)


    def trunk_parameters():
        rng = np.random.default_rng(0)
        params = {}
        for index in range(LAYERS):
            block = "blocks.%d" % index
            for tri in ("tri_mul_out", "tri_mul_in"):
                triangle_params(rng, params, block + "." + tri + "._engine")
            transition_params(rng, params, block + ".pair_transition", C)
        return params


    def msa_parameters(prefix="block"):
        rng = np.random.default_rng(1)
        params = {}
        dot = prefix + "."
        opm = dot + "outer_product_mean"
        params[opm + ".norm.weight"] = arr(rng, C_M) * 0.1 + 1.0
        params[opm + ".norm.bias"] = arr(rng, C_M) * 0.1
        params[opm + ".W.weight"] = arr(rng, 2 * OPM_HALF, C_M)
        params[opm + ".Wout.weight"] = arr(rng, C, OPM_HALF * OPM_HALF)
        params[opm + ".Wout.bias"] = arr(rng, C)
        pwa = dot + "msa_pair_weighted_averaging"
        params[pwa + ".norm_single.weight"] = arr(rng, C_M) * 0.1 + 1.0
        params[pwa + ".norm_single.bias"] = arr(rng, C_M) * 0.1
        params[pwa + ".compute_bias.0.weight"] = arr(rng, C) * 0.1 + 1.0
        params[pwa + ".compute_bias.0.bias"] = arr(rng, C) * 0.1
        params[pwa + ".compute_bias.1.weight"] = arr(rng, HEADS, C)
        params[pwa + ".Wv.weight"] = arr(rng, HEADS * HEAD_DIM, C_M)
        params[pwa + ".Wgate.weight"] = arr(rng, HEADS * HEAD_DIM, C_M)
        params[pwa + ".Wout.weight"] = arr(rng, C_M, HEADS * HEAD_DIM)
        transition_params(rng, params, dot + "msa_transition", C_M)
        for tri in ("tri_mul_out", "tri_mul_in"):
            triangle_params(rng, params, dot + tri + "._engine")
        transition_params(rng, params, dot + "pair_transition", C)
        return params


    def inputs(n):
        # Rank-distinguishable and asymmetric: a ramp that differs for every
        # (i, j) and disagrees with its own transpose, so a tile routed to the
        # wrong grid coordinate cannot agree with the serial answer by
        # accident. A symmetric fixture would pass a transposed schedule.
        rng = np.random.default_rng(7)
        rows = np.arange(n, dtype=np.float32)[:, None]
        cols = np.arange(n, dtype=np.float32)[None, :]
        ramp = (0.03 * rows - 0.017 * cols + 0.004 * rows * cols / n)[..., None]
        pair = jnp.asarray(
            (ramp + rng.normal(size=(n, n, C), scale=0.5))[None].astype(np.float32)
        )
        token = (rng.random(n) > 0.15).astype(np.float32)
        pair_mask = jnp.asarray((token[:, None] * token[None, :])[None])
        msa = jnp.asarray(
            rng.normal(size=(1, n, MSA_ROWS, C_M), scale=0.5).astype(np.float32)
        )
        msa_mask = jnp.asarray(
            (rng.random((1, n, MSA_ROWS)) > 0.1).astype(np.float32)
        )
        return pair, pair_mask, msa, msa_mask


    TRUNK = trunk_parameters()
    MSA = msa_parameters()


    # A fresh closure per arm, every time: `jax.jit` keys its cache on the
    # callable and the mesh lives in a context variable the trace reads, so a
    # reused one replays the first arm's program. That is what made an earlier
    # version of the probe below compare the unsharded program with itself.
    def trunk_program():
        def run(pair_in, mask_in):
            return folding_trunk(pair_in, TRUNK, n_layers=LAYERS, mask=mask_in)

        return run


    def msa_program():
        def run(msa_in, pair_in, msa_mask_in, pair_mask_in):
            return msa_encoder_block(
                msa_in,
                pair_in,
                MSA,
                "block",
                msa_mask=msa_mask_in,
                pair_mask=pair_mask_in,
                is_final=False,
            )

        return run


    def local_shape(value):
        return tuple(
            int(size) for size in next(iter(value.addressable_shards)).data.shape
        )
    """
)


_GRID_PROBE = _FIXTURE + textwrap.dedent(
    r"""
    import contextlib

    from foldjax.models.esmfold2.models import trunk as trunk_module

    # Values in any leading position, so `[1, N, N, C]` and `[N, N, C]` both
    # count, and only values that carry a channel axis: `pair_mask` is
    # `[1, N, N]`, is never pinned, and is legitimately full width.
    VALUE = re.compile(r"\b[a-z0-9]+\[((?:[0-9]+,)*[0-9]+)\]")
    COLLECTIVES = ("all-gather", "all-reduce", "all-to-all", "collective-permute")


    def full_width(text, n):
        found = {}
        for dims in VALUE.findall(text):
            sizes = [int(size) for size in dims.split(",")]
            if len(sizes) >= 3 and sizes[-3] == n and sizes[-2] == n:
                found[dims] = found.get(dims, 0) + 1
        return found


    def collectives(text):
        return {name: text.count(name) for name in COLLECTIVES if text.count(name)}


    # The leak tripwire: the grid's two branches, replaced by something that
    # cannot run. A serial or one-dimensional program that reached either of
    # them would fail here rather than quietly become a fifth program nobody
    # measured -- and both of those layouts have to stay byte-identical.
    @contextlib.contextmanager
    def no_grid_path():
        names = ("_cannon_contract", "_cp_pair_transition")
        originals = {name: getattr(trunk_module, name) for name in names}

        def refuse(*args, **kwargs):
            raise AssertionError("the grid's schedule was reached without a grid")

        for name in names:
            setattr(trunk_module, name, refuse)
        try:
            yield
        finally:
            for name, value in originals.items():
                setattr(trunk_module, name, value)

    for n in (SIDE * 4, 13):
        pair, pair_mask, msa, msa_mask = inputs(n)
        jax.clear_caches()
        with no_grid_path():
            trunk_ref = np.asarray(
                jax.device_get(jax.jit(trunk_program())(pair, pair_mask))
            )
            jax.clear_caches()
            msa_ref = [
                np.asarray(value)
                for value in jax.device_get(
                    jax.jit(msa_program())(msa, pair, msa_mask, pair_mask)
                )
            ]

        for layout in ("1d", "2d"):
            jax.clear_caches()
            guard = no_grid_path() if layout == "1d" else contextlib.nullcontext()
            with guard, context_parallel(DEVICES, layout=layout):
                trunk_value = jax.jit(trunk_program())(pair, pair_mask)
                trunk_value.block_until_ready()
                trunk_local = local_shape(trunk_value)
                trunk_got = np.asarray(jax.device_get(trunk_value))
                trunk_text = (
                    jax.jit(trunk_program())
                    .lower(pair, pair_mask)
                    .compile()
                    .as_text()
                )
                msa_value = jax.jit(msa_program())(msa, pair, msa_mask, pair_mask)
                msa_local = local_shape(msa_value[1])
                msa_got = [
                    np.asarray(value) for value in jax.device_get(msa_value)
                ]
                grid = cp_grid()

            np.testing.assert_allclose(trunk_ref, trunk_got, atol=3e-5, rtol=3e-5)
            for reference, got in zip(msa_ref, msa_got, strict=True):
                np.testing.assert_allclose(reference, got, atol=3e-5, rtol=3e-5)
            assert trunk_got.dtype == trunk_ref.dtype, trunk_got.dtype

            if layout == "1d":
                # The row mesh keeps the constraint-only program; the
                # tripwire above is what said so, and it ran.
                assert grid == (DEVICES, 1), grid
                continue

            assert grid == (SIDE, SIDE), grid
            if n % SIDE == 0:
                # The tile is what says the grid ran. Both token axes are
                # divided, so a device holds (N/side)^2 of the pair state --
                # a quarter of it on four devices, a ninth on nine.
                assert trunk_local == (1, n // SIDE, n // SIDE, C), trunk_local
                assert msa_local == (1, n // SIDE, n // SIDE, C), msa_local
                # Nothing full-width: the whole point of Cannon's schedule is
                # that no operand or partial is ever gathered back to `N`.
                assert not full_width(trunk_text, n), full_width(trunk_text, n)
                counts = collectives(trunk_text)
                assert "all-gather" not in counts, counts
                assert counts.get("collective-permute", 0) > 0, counts
            else:
                # A token count the grid cannot divide is a correctness arm
                # only: `with_sharding_constraint` drops a spec it cannot
                # apply, so the pair state comes back replicated and the tile
                # assertions above would be asserting the serial program.
                assert trunk_local == (1, n, n, C), trunk_local

    print("GRID_PARITY_OK")
    """
)


_SIGN_PROBE = _FIXTURE + textwrap.dedent(
    r"""
    # Every move in Cannon's schedule is +-1 or +-coordinate, and modulo two
    # those are the same move, so a 2x2 grid passes every sign error there is.
    # This probe flips one sign at a time and reports what the mesh it is run
    # on can see. Flipping *both* ring hops is not an error: it walks the k
    # blocks backwards while keeping the two operands paired at every step, so
    # it is an equivalent schedule on any grid and must not be expected to
    # fail.
    from foldjax.models import _cp
    from foldjax.models.esmfold2.models import trunk as trunk_module

    MUTATIONS = {
        "row_skew_sign": {
            "row_skew_perm": lambda side, sign=-1: _cp.row_skew_perm(side, sign=+1)
        },
        "col_skew_sign": {
            "col_skew_perm": lambda side, sign=-1: _cp.col_skew_perm(side, sign=+1)
        },
        "lhs_hop_only": {
            "ring_perm": lambda side, *, axis, delta=1: _cp.ring_perm(
                side,
                axis=axis,
                delta=+1 if axis == _cp.CP_COL_AXIS else delta,
            )
        },
        "both_hops": {
            "ring_perm": lambda side, *, axis, delta=1: _cp.ring_perm(
                side, axis=axis, delta=-delta
            )
        },
    }
    # What each grid size is able to see. The first three break the pairing of
    # the two operands and are real defects; `both_hops` does not.
    BROKEN = {2: (), 3: ("row_skew_sign", "col_skew_sign", "lhs_hop_only")}

    n = SIDE * 4
    pair, pair_mask, _, _ = inputs(n)
    jax.clear_caches()
    reference = np.asarray(jax.device_get(jax.jit(trunk_program())(pair, pair_mask)))

    jax.clear_caches()
    with context_parallel(DEVICES, layout="2d"):
        honest = np.asarray(jax.device_get(jax.jit(trunk_program())(pair, pair_mask)))
    honest_diff = float(np.abs(reference - honest).max())
    assert honest_diff < 3e-5, honest_diff

    for name, patch in MUTATIONS.items():
        originals = {key: getattr(trunk_module, key) for key in patch}
        for key, value in patch.items():
            setattr(trunk_module, key, value)
        try:
            jax.clear_caches()
            with context_parallel(DEVICES, layout="2d"):
                got = np.asarray(
                    jax.device_get(jax.jit(trunk_program())(pair, pair_mask))
                )
        finally:
            for key, value in originals.items():
                setattr(trunk_module, key, value)
        diff = float(np.abs(reference - got).max())
        print("side=%d mutation=%s max_diff=%.3e" % (SIDE, name, diff))
        if name in BROKEN[SIDE]:
            # An order-one error, not a tolerance one: a mis-paired block sums
            # the wrong part of the contracted axis.
            assert diff > 1.0, (name, diff)
        else:
            assert diff < 3e-5, (name, diff)

    print("SIGN_TRAP_OK")
    """
)


_NATIVE_GRID_PROBE = _FIXTURE + textwrap.dedent(
    r"""
    # The native-autocast triangle block on the grid. This path is reachable
    # under a mesh and the reference one is not the only one that needs
    # Cannon: `lm_encoder_params` is gated on the trunk dtype alone
    # (`models/model.py`), so the language-model encoder's trunk keeps native
    # autocast under context parallelism while every other pair stack falls
    # back. Left on the dense einsum it would be correct and gathered.
    from foldjax.models.esmfold2.models import trunk as trunk_module
    from foldjax.models.esmfold2.models.trunk import triangle_multiplicative

    n = SIDE * 4
    pair, pair_mask, _, _ = inputs(n)


    def taken(program, name):
        # Replace one of the grid's two branches with something that cannot
        # run, and require it to be hit. Without this the comparisons in this
        # probe would pass on a program that quietly stayed unsharded.
        original = getattr(trunk_module, name)

        def refuse(*args, **kwargs):
            raise AssertionError("reached")

        setattr(trunk_module, name, refuse)
        try:
            jax.clear_caches()
            with context_parallel(DEVICES, layout="2d"):
                jax.jit(program)(pair, pair_mask)
        except AssertionError as error:
            assert "reached" in str(error), error
        else:
            raise AssertionError("the grid did not reach " + name)
        finally:
            setattr(trunk_module, name, original)

    for outgoing, block in ((True, "tri_mul_out"), (False, "tri_mul_in")):

        def program(pair_in, mask_in, outgoing=outgoing, block=block):
            return triangle_multiplicative(
                pair_in.astype(jnp.bfloat16),
                TRUNK,
                "blocks.0." + block,
                outgoing=outgoing,
                mask=mask_in,
                native_autocast=True,
            )

        jax.clear_caches()
        reference = jax.device_get(jax.jit(program)(pair, pair_mask))
        jax.clear_caches()
        with context_parallel(DEVICES, layout="2d"):
            got = jax.device_get(jax.jit(program)(pair, pair_mask))

        # The block's output boundary is bfloat16 on both arms -- the native
        # policy this path exists to preserve -- and the ring's float32
        # accumulation is narrowed once, where the dense chunk loop narrowed
        # each of its chunks.
        assert reference.dtype == jnp.bfloat16, reference.dtype
        assert got.dtype == jnp.bfloat16, got.dtype
        np.testing.assert_allclose(
            np.asarray(reference, dtype=np.float32),
            np.asarray(got, dtype=np.float32),
            atol=8e-3,
            rtol=8e-3,
        )

        # An exactly-zero difference is the signature of a comparison that
        # never ran the sharded program, so the branch is asked to prove it
        # was taken rather than inferred from the agreement above.
        taken(program, "_cannon_contract")

    # And the whole native stack, which is the composition the language-model
    # encoder actually runs: two blocks of both triangle directions and a pair
    # transition, with the transition's row block inside the same `shard_map`
    # the contraction uses. Each piece is checked above; this is the arm that
    # traces them together, because "each half works" is not evidence that a
    # native transition traces inside a shard.
    def stack(pair_in, mask_in):
        return folding_trunk(
            pair_in.astype(jnp.bfloat16),
            TRUNK,
            n_layers=LAYERS,
            mask=mask_in,
            native_autocast=True,
        )

    jax.clear_caches()
    reference = jax.device_get(jax.jit(stack)(pair, pair_mask))
    jax.clear_caches()
    with context_parallel(DEVICES, layout="2d"):
        value = jax.jit(stack)(pair, pair_mask)
        value.block_until_ready()
        local = local_shape(value)
        got = jax.device_get(value)

    assert got.dtype == jnp.bfloat16, got.dtype
    assert local == (1, n // SIDE, n // SIDE, C), local
    a = np.asarray(reference, dtype=np.float32)
    b = np.asarray(got, dtype=np.float32)
    scale = float(np.abs(a).max())
    # In units of the output's own bfloat16 ULP rather than a decimal
    # tolerance: this stack rounds to bfloat16 at every linear and residual,
    # so its floor is the format, and a tolerance written as a small decimal
    # would be either unmeetable or meaningless. Measured 4.0 ULP on 2x2 and
    # 2.25 on 3x3, against a schedule error -- a mis-paired ring, a direction
    # read the wrong way round -- that moves values by their own magnitude.
    ulp = 2.0 ** (np.floor(np.log2(scale)) - 7)
    difference = float(np.abs(a - b).max())
    assert difference <= 8 * ulp, (difference, ulp, scale)

    taken(stack, "_cannon_contract")
    taken(stack, "_cp_pair_transition")

    print("NATIVE_GRID_OK")
    """
)


_LOCAL_BLOCK_PROBE = _FIXTURE + textwrap.dedent(
    r"""
    # The native triangle prologue's row block, taken inside the shard.
    #
    # At the released 64 rows the block drops itself on every fixture the
    # suite can afford -- a device holds a handful of rows, and a block wider
    # than the local tile divides nothing -- so the sharded path here is the
    # one nothing else reaches. The width is lowered until it fires, which is
    # the only way to run the code a 3,012-token program would run.
    from foldjax.models.esmfold2.models import trunk as trunk_module
    from foldjax.models.esmfold2.models.trunk import folding_trunk

    COLLECTIVES = ("all-gather", "all-reduce", "all-to-all", "collective-permute")


    def collectives(text):
        return {name: text.count(name) for name in COLLECTIVES if text.count(name)}


    def stack():
        def run(pair_in, mask_in):
            return folding_trunk(
                pair_in.astype(jnp.bfloat16),
                TRUNK,
                n_layers=LAYERS,
                mask=mask_in,
                native_autocast=True,
            )

        return run


    def arm(pair, mask, rows, layout):
        # How many times the body is traced is the witness that the block
        # fired: dropped, it runs once per direction per layer; taken, once
        # per block of local rows. Without it both arms would be the dropped
        # program and every assertion below would be comparing it to itself.
        original = trunk_module._AUTOCAST_ROWS
        body = trunk_module._triangle_prologue
        calls = []

        def counted(*args, **kwargs):
            calls.append(args[0].shape)
            return body(*args, **kwargs)

        trunk_module._AUTOCAST_ROWS = rows
        trunk_module._triangle_prologue = counted
        try:
            jax.clear_caches()
            if layout is None:
                value = jax.jit(stack())(pair, mask)
                value.block_until_ready()
                text = jax.jit(stack()).lower(pair, mask).compile().as_text()
            else:
                with context_parallel(DEVICES, layout=layout):
                    value = jax.jit(stack())(pair, mask)
                    value.block_until_ready()
                    text = jax.jit(stack()).lower(pair, mask).compile().as_text()
            return np.asarray(jax.device_get(value), np.float32), text, calls
        finally:
            trunk_module._AUTOCAST_ROWS = original
            trunk_module._triangle_prologue = body


    # 13 rows: 4 shards of 4 after padding on the row mesh, 2x2 tiles of 7
    # after padding on the grid, and neither divides the block of 2.
    n = 13
    pair, pair_mask, _, _ = inputs(n)
    serial, _, serial_calls = arm(pair, pair_mask, 10 ** 6, None)
    scale = float(np.abs(serial).max())
    assert scale > 0.0, scale
    # In units of the output's own bfloat16 ULP: this stack rounds to
    # bfloat16 at every linear, so a decimal tolerance would be meaningless.
    ulp = 2.0 ** (np.floor(np.log2(scale)) - 7)

    for layout in ("1d", "2d"):
        whole, whole_text, whole_calls = arm(pair, pair_mask, 10 ** 6, layout)
        blocked, blocked_text, blocked_calls = arm(pair, pair_mask, 2, layout)
        # The block fired, on rows narrower than the tile the whole arm used.
        assert len(blocked_calls) > len(whole_calls), (
            layout,
            whole_calls,
            blocked_calls,
        )
        assert max(shape[-3] for shape in blocked_calls) <= 2, blocked_calls
        # ... and on *local* rows. The rows handed to the body add up to the
        # padded local tile, not to the global axis: a global block would
        # trace exactly `n` rows per prologue and is the version the
        # partitioner can only serve by gathering.
        traced = sum(shape[-3] for shape in blocked_calls)
        assert traced < len(whole_calls) * n, (layout, traced, blocked_calls)
        # The block must buy its rows without buying communication.
        assert collectives(whole_text) == collectives(blocked_text), (
            layout,
            collectives(whole_text),
            collectives(blocked_text),
        )
        for tag, got in (("whole", whole), ("blocked", blocked)):
            difference = float(np.abs(serial - got).max())
            assert difference <= 8 * ulp, (layout, tag, difference, ulp)
        print(
            "layout=%s calls=%d->%d collectives=%s"
            % (layout, len(whole_calls), len(blocked_calls),
               collectives(blocked_text))
        )

    print("LOCAL_BLOCK_OK")
    """
)


def _run_grid_probe(source: str, devices: int) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
            "FOLDJAX_CP_PROBE_DEVICES": str(devices),
            **inherited_environment(),
        },
        timeout=900,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


@pytest.mark.parametrize("devices", [4, 9])
def test_the_grid_matches_the_unsharded_pair_stacks(devices: int) -> None:
    """2-D parity, the per-device tile, and no gather, on 2x2 and 3x3.

    Both pair stacks are checked -- the trunk, which is triangle updates and a
    transition, and the MSA encoder block, whose outer product mean and pair
    weighted averaging now contract over a *sharded* column axis with no
    hand-written code at all. The row mesh never sharded columns, so that half
    is GSPMD's and has never run on this port before.
    """

    assert "GRID_PARITY_OK" in _run_grid_probe(_GRID_PROBE, devices)


@pytest.mark.parametrize("devices", [4, 9])
def test_the_native_autocast_block_runs_the_grid_schedule_too(devices: int) -> None:
    """The one pair stack that keeps native autocast under a mesh.

    Both directions, and with the branch asked to prove it was taken: the two
    arms agree bitwise at these sizes, which is exactly what a comparison that
    silently replayed the unsharded program would also report.
    """

    assert "NATIVE_GRID_OK" in _run_grid_probe(_NATIVE_GRID_PROBE, devices)


def test_the_prologue_row_block_is_taken_inside_the_shard() -> None:
    """The block fires on local rows, agrees with serial, and adds nothing.

    The row block on a *global* axis is a slice of a sharded one, which the
    partitioner can only serve by moving data -- 104 `all-to-all`s when
    `_cp_pair_transition` measured it. Inside the shard it is free, and the
    census before and after the block is what says so.
    """

    assert "LOCAL_BLOCK_OK" in _run_grid_probe(_LOCAL_BLOCK_PROBE, 4)


@pytest.mark.parametrize("devices", [4, 9])
def test_a_mis_paired_ring_is_caught_only_by_a_side_of_three(devices: int) -> None:
    """The 3x3 grid is the gate; the 2x2 one is here to show it is not.

    Keeping the 2x2 arm is the point: it passes every mutation, which is what
    makes it evidence that a suite with only four devices would certify a
    broken schedule rather than merely miss it.
    """

    assert "SIGN_TRAP_OK" in _run_grid_probe(_SIGN_PROBE, devices)
