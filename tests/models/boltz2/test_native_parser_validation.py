"""Native Boltz input validation must survive optimized Python."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path


def _run_parser_check(source: str, *, optimized: bool) -> subprocess.CompletedProcess:
    command = [sys.executable]
    if optimized:
        command.append("-O")
    command.extend(["-c", textwrap.dedent(source)])
    environment = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "JAX_PLATFORM_NAME": "cpu",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    environment.pop("PYTHONOPTIMIZE", None)
    return subprocess.run(
        command, env=environment, capture_output=True, text=True, check=False
    )


def _assert_same_with_optimization(source: str, expected: list[str]) -> None:
    normal = _run_parser_check(source, optimized=False)
    optimized = _run_parser_check(source, optimized=True)

    assert normal.returncode == 0, normal.stderr
    assert optimized.returncode == 0, optimized.stderr
    assert normal.stdout.splitlines() == expected
    assert optimized.stdout == normal.stdout


def test_ligand_representation_validation_survives_python_optimization() -> None:
    source = """
        from foldjax.models.boltz2.data.parse.schema import parse_boltz_schema

        invalid_ligands = [
            {"id": "L"},
            {"id": "L", "smiles": "CC", "ccd": "ATP"},
        ]
        for ligand in invalid_ligands:
            schema = {"version": 1, "sequences": [{"ligand": ligand}]}
            try:
                parse_boltz_schema("invalid", schema, {}, boltz_2=True)
            except AssertionError as error:
                print(f"AssertionError: {error}")
            else:
                raise RuntimeError("invalid ligand representation was accepted")
    """
    message = "AssertionError: Ligand must contain exactly one of 'smiles' or 'ccd'."
    _assert_same_with_optimization(source, [message, message])


def test_ligand_cyclic_validation_survives_python_optimization() -> None:
    source = """
        import foldjax.models.boltz2.data.parse.schema as native_schema

        native_schema.get_mol = lambda *args, **kwargs: object()
        native_schema.compute_3d_conformer = lambda mol: True
        native_schema.parse_ccd_residue = lambda **kwargs: object()

        invalid_ligands = [
            {"id": "L", "ccd": "ATP", "cyclic": True},
            {"id": "L", "smiles": "CC", "cyclic": True},
        ]
        for ligand in invalid_ligands:
            schema = {"version": 1, "sequences": [{"ligand": ligand}]}
            try:
                native_schema.parse_boltz_schema(
                    "invalid", schema, {}, boltz_2=True
                )
            except AssertionError as error:
                print(f"AssertionError: {error}")
            else:
                raise RuntimeError("cyclic ligand was accepted")
    """
    message = "AssertionError: Cyclic flag is not supported for ligands"
    _assert_same_with_optimization(source, [message, message])


def test_nonprotein_msa_id_validation_survives_python_optimization() -> None:
    source = """
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from foldjax.models.boltz2.data.parse.fasta import parse_fasta

        with TemporaryDirectory() as directory:
            directory = Path(directory)
            fasta = directory / "invalid.fasta"
            fasta.write_text(">A|dna|custom_msa\\nAT\\n")
            try:
                parse_fasta(fasta, {}, directory, boltz2=True)
            except AssertionError as error:
                print(f"AssertionError: {error}")
            else:
                raise RuntimeError("non-protein MSA ID was accepted")
    """
    _assert_same_with_optimization(
        source, ["AssertionError: MSA_ID is only allowed for proteins"]
    )


def test_native_parser_invariants_are_not_python_assert_statements() -> None:
    parse_root = (
        Path(__file__).resolve().parents[3] / "src/foldjax/models/boltz2/data/parse"
    )
    remaining = {
        str(path.relative_to(parse_root)): [
            node.lineno
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.Assert)
        ]
        for path in parse_root.glob("*.py")
    }
    assert not {name: lines for name, lines in remaining.items() if lines}
