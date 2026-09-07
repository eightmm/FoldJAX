"""Native graph identity includes commutative operand order, not only algebra."""

import ast

from foldjax.models.alphafold3.build import source_package


def test_sampler_preserves_native_noise_scaling_operand_order():
    path = source_package() / "model/network/diffusion_head.py"
    tree = ast.parse(path.read_text())
    sample = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "sample"
    )
    step = next(
        node
        for node in sample.body
        if isinstance(node, ast.FunctionDef) and node.name == "apply_denoising_step"
    )
    scaling = [
        node.value
        for node in step.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "noise"
            for target in node.targets
        )
        and isinstance(node.value, ast.BinOp)
        and isinstance(node.value.op, ast.Mult)
    ]
    assert len(scaling) == 1
    assert ast.unparse(scaling[0]) == "noise_scale * noise"
