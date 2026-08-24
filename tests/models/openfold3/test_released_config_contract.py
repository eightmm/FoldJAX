"""The backend's overrides and ``released_config``'s signature must agree.

``foldjax --model openfold3 predict`` builds a dict of overrides and splats it
into :func:`released_config`. Nothing checked that the two sides named the same
things, so adding ``returned_representations``/``stop_after_trunk`` to the
backend without adding them to the constructor made *every* OpenFold3
prediction fail with a ``TypeError`` before the model was reached. This static
contract catches the mismatch without requiring the optional multi-gigabyte p1
checkpoint or running a prediction.
"""

import ast
import inspect
import pathlib

from foldjax.models.openfold3 import inference


def _override_keys() -> set[str]:
    """Every constant key the backend writes into its ``overrides`` dict."""
    source = pathlib.Path(inspect.getfile(inference)).parents[3]
    backend = source / "foldjax" / "backends" / "openfold3.py"
    tree = ast.parse(backend.read_text())
    keys = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "overrides"
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
            ):
                keys.add(target.slice.value)
    return keys


def test_every_backend_override_is_a_released_config_parameter() -> None:
    accepted = set(inspect.signature(inference.released_config).parameters)
    unknown = sorted(_override_keys() - accepted)
    assert not unknown, (
        f"backends/openfold3.py passes {unknown} to released_config(), which "
        "does not accept them; every OpenFold3 prediction would raise TypeError"
    )


def test_the_backend_writes_overrides_this_test_can_see() -> None:
    """Guard the guard: an empty key set would make the test above vacuous."""
    assert len(_override_keys()) >= 4


def test_released_config_carries_the_representation_request() -> None:
    config = inference.released_config(
        n_token=16,
        n_atom=64,
        returned_representations=("pair", "single"),
        stop_after_trunk=True,
    )
    assert config.returned_representations == ("pair", "single")
    assert config.stop_after_trunk is True
