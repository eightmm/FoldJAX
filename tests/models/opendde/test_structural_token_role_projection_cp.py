"""The context-parallel role-pair projection must gather one role at a time.

The CP branch of ``_pair_project_by_role`` sums seven terms, one per column
role, and each term needs one ``[N, C_out, C_in]`` slab of row weights. Asking
for those seven slabs in a way XLA can serve at once -- selecting
``pair_block_proj[role]`` and slicing the column role afterwards, or indexing
both roles but leaving the seven terms unrolled -- costs seven times the
weights: ``bf16[512, 7, 384, 384]``, 1,008 MiB per device at 1,024 structural
tokens on a 2x2 grid, and linear in the token count from there.

The property is therefore about the compiled program, not about how the source
spells the loop: no value in the CP program may carry more weights than one
column role's slab. A test that asserted a ``lax.scan`` were present instead
would pass a rewrite that reintroduced the block through some other route, and
would fail a rewrite that kept the property by different means.

A forced device count has to be set before JAX initialises, so the check runs
in a subprocess with four CPU devices: enough for the 1-D layout and for the
smallest perfect square the 2-D layout accepts. The probe uses ``#`` comments
rather than docstrings because it lives inside a triple-quoted literal.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

from tests.models.cp_probe_env import inherited_environment

#: Four devices give a 4-shard 1-D mesh and a 2x2 grid. The probe asserts the
#: local tile shape each layout must produce, so it depends on this count.
_DEVICES = 4

_PROBE = textwrap.dedent(
    r"""
    import re

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel, shard_pair_rows
    from foldjax.models.opendde.models import structural_tokens

    # `N` has to clear 4 * ROLES**2 so that one slab of row weights on the
    # 1-D mesh, N/4 * C_OUT * C_IN, is larger than the replicated
    # [ROLES, ROLES, C_OUT, C_IN] table. Otherwise the table trips the bound
    # and the check has to special-case it by shape, which is the kind of
    # spelling-dependent exemption this test exists to avoid.
    N = 224
    C_IN = 16
    C_OUT = 12
    ROLES = 7

    # A gathered weight tensor is any value whose two trailing dimensions are
    # the projection matrix. C_OUT != C_IN so the pair representation and the
    # weights cannot be confused for one another.
    # `(?:[0-9]+,)*` rather than `[0-9,]*`: the latter also matches a pair
    # tile whose dimensions merely end in the right digits, e.g. [112,112,16].
    WEIGHT = re.compile(r"\b\w+\[((?:[0-9]+,)*" + f"{C_OUT},{C_IN}" + r")\]")
    DEFINITION = re.compile(r"^\s+(?:ROOT\s+)?%?[\w.\-]+ = \S+ ")

    def elements(dims):
        total = 1
        for size in dims.split(","):
            total *= int(size)
        return total

    role = jnp.asarray((np.arange(N) % ROLES).astype(np.int32))
    weights = np.random.default_rng(3).normal(size=(ROLES, ROLES, C_OUT, C_IN))
    weights = weights.astype(np.float32)
    z = np.random.default_rng(4).normal(size=(N, N, C_IN)).astype(np.float32)
    expected = np.einsum(
        "...ijc,ijoc->...ijo",
        z,
        weights[np.asarray(role)[:, None], np.asarray(role)[None, :]],
    )
    z = jnp.asarray(z)
    weights = jnp.asarray(weights)

    for layout, rows in (("1d", 4), ("2d", 2)):
        # A fresh closure per arm: `jax.jit` keys its cache on the callable and
        # the mesh is a context variable the trace reads, so a reused one would
        # replay the first arm's program.
        def program(z_in, role_in, weights_in):
            return structural_tokens._pair_project_by_role(
                shard_pair_rows(z_in), role_in, weights_in
            )

        with context_parallel(_DEVICES, layout=layout):
            compiled = jax.jit(program).lower(z, role, weights).compile()
            value = jax.jit(program)(z, role, weights)
            value.block_until_ready()
            local = tuple(
                int(size)
                for size in next(iter(value.addressable_shards)).data.shape
            )

        # The tripwire: without it a program that quietly ran unsharded would
        # satisfy every assertion below.
        columns = N // rows if layout == "2d" else N
        assert local == (N // rows, columns, C_OUT), (layout, local)
        np.testing.assert_allclose(
            np.asarray(jax.device_get(value)), expected, rtol=1e-5, atol=1e-5
        )

        slab = (N // rows) * C_OUT * C_IN
        text = compiled.as_text()
        gathered = sorted(
            {dims: elements(dims) for dims in WEIGHT.findall(text)}.items(),
            key=lambda item: -item[1],
        )
        # Gathers whose result is larger than the replicated table, i.e. one
        # per structural row. The size bound below cannot see the unrolled
        # `pair_block_proj[role, c]` form, whose seven gathers are each
        # exactly one slab; their number is what gives it away.
        per_row = [
            line
            for line in text.splitlines()
            if DEFINITION.match(line)
            and " gather(" in line
            and any(
                elements(dims) > ROLES * ROLES * C_OUT * C_IN
                for dims in WEIGHT.findall(line.split(" = ")[1].split(" gather(")[0])
            )
        ]
        oversized = [item for item in gathered if item[1] > slab]
        # Non-empty, so a regex that stopped matching cannot pass silently.
        assert gathered, (layout, text[:2000])
        assert not oversized, (layout, slab, oversized)
        assert len(per_row) <= 2, (layout, len(per_row), per_row)
        print(
            f"{layout} local={local} slab={slab} per_row={len(per_row)} "
            f"weights={gathered[:3]}"
        )

    print("ROLE_PROJECTION_CP_OK")
    """
)


def _run_probe(source: str) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": f"--xla_force_host_platform_device_count={_DEVICES}",
            **inherited_environment(),
        },
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def test_the_cp_role_projection_never_gathers_all_seven_roles_at_once() -> None:
    """One column role's weights per term, on the 1-D mesh and the 2x2 grid."""

    assert "ROLE_PROJECTION_CP_OK" in _run_probe(
        _PROBE.replace("_DEVICES", str(_DEVICES))
    )
