"""Every triangle site resolves through one rule, and none opts out privately.

Three separate bugs in this repo had the same shape: a knob that looked set and
did not reach. A removed port's MSA pair block kept its own copy of a threshold that
gained a byte rule only on the other copy; protenix's confidence head carried
`triangle_attention_backend="xla_jit"` as a parameter default, so changing the
module default moved everything except it; OpenDDE forced both backends to XLA
in a decorator, so an environment variable was the only way in and the
confidence head was closed even to that.

None of those were visible as a failure. The runs completed, the scores were
plausible, and the only symptom was a number in a benchmark. So the invariant
worth asserting is not "the default is cueq" -- that is a value, and values
change -- but "no site holds a private opinion", which is the property that was
actually violated.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

MODELS = Path(__file__).resolve().parent.parent / "src/foldjax/models"


def _module_files(*relative: str) -> list[Path]:
    found: list[Path] = []
    for entry in relative:
        path = MODELS / entry
        found.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])
    return [path for path in found if "__pycache__" not in path.parts]


def _string_defaults(path: Path, parameter: str) -> list[tuple[str, str]]:
    """Every function in `path` whose `parameter` defaults to a string literal."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = node.args
        # Positional and keyword-only defaults line up with the tail of each list.
        pairs = list(
            zip(arguments.args[-len(arguments.defaults) :], arguments.defaults)
        ) + [
            (arg, default)
            for arg, default in zip(arguments.kwonlyargs, arguments.kw_defaults)
            if default is not None
        ]
        for arg, default in pairs:
            if arg.arg != parameter:
                continue
            if isinstance(default, ast.Constant) and isinstance(default.value, str):
                found.append((node.name, default.value))
    return found


@pytest.mark.parametrize(
    "files, parameter",
    [
        (("protenix", "opendde"), "triangle_attention_backend"),
        (("protenix", "opendde"), "confidence_triangle_attention_backend"),
        (("protenix", "opendde"), "trunk_triangle_attention_backend"),
        (("protenix", "opendde"), "structural_triangle_attention_backend"),
    ],
)
def test_no_triangle_site_pins_its_own_backend(files, parameter) -> None:
    """A string default here is a site the shared resolver cannot reach."""
    offenders = [
        f"{path.relative_to(MODELS)}::{function} = {value!r}"
        for path in _module_files(*files)
        for function, value in _string_defaults(path, parameter)
    ]
    assert not offenders, (
        f"{parameter} is pinned to a literal at these sites, so the shared "
        "resolver never sees them:\n  " + "\n  ".join(offenders)
    )


def test_protenix_hands_its_confidence_head_a_backend() -> None:
    """Removing the private default was only half the fix.

    `test_no_triangle_site_pins_its_own_backend` checks that no site *pins* a
    backend, and the confidence head stopped pinning one. But `model.py` then
    called `confidence_head(...)` without passing a backend at all, so the head
    fell through to the module default while the trunk ran whatever was asked
    for -- the same divergence, arrived at from the other direction, and
    invisible to a test that only reads defaults.

    It cost a `f32[2030, 4, 256, 2030]` score tensor: 15.72 GiB inside a
    39.07 GiB temp arena, which killed a 2030-token target *after* the compute
    had finished. Nothing failed earlier because both names resolve to the same
    kernel under the default environment; they part company the moment
    `PROTENIX_TRIANGLE_BACKEND` or a trunk override is set, which is exactly
    when a large job is being rescued.

    Asserted on the call site rather than by running the model, because the
    shapes that make this matter need a GPU and 2000 tokens to reproduce.
    """
    path = MODELS / "protenix/models/model.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "confidence_head"
    ]
    assert calls, "confidence_head is no longer called here; update this test"
    for call in calls:
        passed = {keyword.arg for keyword in call.keywords}
        assert "triangle_attention_backend" in passed, (
            "model.py calls confidence_head without a triangle backend, so the "
            "head takes the module default while the trunk takes the caller's"
        )


def test_opendde_runs_the_fused_kernels_like_upstream() -> None:
    """OpenDDE stopped being the exception.

    Both backends were pinned to XLA here on an OOM measured only at 1,531
    tokens -- 92.21 GiB fused against a 97,887 MiB card. That is real at that
    size and was wrong as a rule: the same model peaks at 10,552 MiB on a
    490-token job and 34,408 on a 970-token one. Upstream runs both kernels
    fused at every size.
    """
    from foldjax.models.opendde.models import model

    source = inspect.getsource(model._with_cueq_triangle_defaults)
    assert '"PROTENIX_TRIANGLE_BACKEND": "cueq_jit"' in source
    assert '"PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND": "cueq"' in source
    # setdefault, not assignment: the largest jobs still need a way to the
    # blocked path, and the process must not inherit either value.
    assert "if name not in os.environ" in source
    assert "os.environ.pop" in source


def test_neither_protenix_resolver_has_an_auto_mode(monkeypatch) -> None:
    """A probe-driven default puts two machines on two kernels under one name.

    Both resolvers briefly read `auto` and fell back to XLA when cuEquivariance
    could not load, which is indistinguishable from the fused path right up
    until the numbers differ. If the kernel cannot load, the import raises.
    """
    from foldjax.models.protenix.models.triangle import triangle

    assert not hasattr(triangle, "_cueq_usable")

    monkeypatch.delenv("PROTENIX_TRIANGLE_BACKEND", raising=False)
    assert triangle._triangle_attention_backend() == "cueq_jit"

    source = inspect.getsource(triangle.triangle_multiplication)
    assert '"PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND", "cueq"' in source
    assert '"auto"' not in source


def test_explicit_environment_still_wins(monkeypatch) -> None:
    """`cueq_jit` is a default, not a lock.

    A card whose arena the fused kernel overflows needs a way out.
    """
    from foldjax.models.protenix.models.triangle import triangle

    monkeypatch.setenv("PROTENIX_TRIANGLE_BACKEND", "xla_jit")
    assert triangle._triangle_attention_backend() == "xla_jit"
