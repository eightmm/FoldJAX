"""The trunk must run in the dtype the configuration asks for, not merely say so.

`test_the_released_trunk_is_bfloat16` asserts `settings.trunk_dtype`, and that
assertion passed for as long as anyone looked while the entire 48-layer folding
trunk ran in float32. Weights were cast to bfloat16 by `_cast`; three feature
tensors were not, and one float32 term in
`z_init = z_init + rel_pos + token_bonds_encoding` promoted `z`, which
`z.astype(z_init.dtype)` then carried through every layer.

`test_the_trunk_runs_at_the_dtype_it_was_configured_for` in
`test_checkpoint_load.py` is the authoritative gate: it spies on the real trunk
with the released checkpoint and asserts the dtypes the graph actually built.
It is marked slow and needs gigabytes of weights, so it does not run where they
are absent -- which is most places.

These two are the cheap complement that runs everywhere. They pin the mechanism
rather than the outcome: the promotion rule that let a bfloat16 region run
float32, and the casts in `predict` that stop it. Neither watches the real
trunk, and saying so here is better than letting a reader assume otherwise.
"""

import ast
import inspect
import pathlib

import jax.numpy as jnp

from foldjax.models.esmfold2.models import model as jax_model


def test_a_bfloat16_weight_against_a_float32_input_yields_float32() -> None:
    """The rule the bug rode in on, stated so a reader meets it here first."""
    weight = jnp.zeros((4, 4), jnp.bfloat16)
    narrow = jnp.zeros((2, 4), jnp.bfloat16)
    wide = jnp.zeros((2, 4), jnp.float32)

    assert jnp.dot(narrow, weight).dtype == jnp.bfloat16
    assert jnp.dot(wide, weight).dtype == jnp.float32, (
        "casting the weights is not enough: one float32 activation is all it "
        "takes to widen a region that every setting calls bfloat16"
    )


def _predict_source() -> ast.FunctionDef:
    tree = ast.parse(pathlib.Path(inspect.getfile(jax_model)).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "predict":
            return node
    raise AssertionError("esmfold2's `predict` has been renamed")


def test_every_tensor_entering_the_pair_state_is_cast_to_the_compute_dtype() -> None:
    """`z_init`'s three addends must each reach it at the compute dtype.

    Read from the source because the alternative is loading the checkpoint. A
    rename will fail this test loudly, which is the right trade against a
    float32 trunk shipping silently again.
    """
    source = ast.unparse(_predict_source())
    assert "pair_inputs = x_inputs.astype(compute)" in source, (
        "the inputs embedding must be narrowed before the z_init linears"
    )
    for fragment, what in (
        ("dtype=compute", "the relative position encoding"),
        ("features['token_bonds'].astype(compute)", "the token-bond encoding"),
    ):
        assert fragment in source, f"{what} must be built at the compute dtype"
