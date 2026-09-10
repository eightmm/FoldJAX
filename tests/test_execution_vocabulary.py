"""One request, five models, one spelling.

The point of the neutral vocabulary is that a caller does not have to know
which port it is talking to. These check that claim from the outside -- by
translating the same request through every backend -- rather than by asserting
the contents of a table against itself.
"""

from __future__ import annotations

import os

import pytest

from foldjax import execution
from foldjax.registry import available_models, get_backend
from foldjax.schema import PredictionRequest


@pytest.fixture
def request_with(tmp_path):
    """A minimal valid request; `PredictionRequest` insists its input exists."""
    job = tmp_path / "job.json"
    job.write_text("{}")

    def build(**options) -> PredictionRequest:
        return PredictionRequest(
            model="boltz2", input=job, output_dir=tmp_path / "out", options=options
        )

    return build


def test_one_dtype_spelling_reaches_every_model_that_has_one(request_with) -> None:
    """`dtype=bfloat16` must not need to know it is `bf16` over there.

    Boltz-2 and Protenix both run a bfloat16 trunk; they spell the option and
    its value differently, which meant a script that switched models had to
    switch vocabulary too.
    """
    spelled = {
        model: get_backend(model).apply_sampling(request_with(dtype="bfloat16"))
        for model in ("boltz2", "protenix", "opendde")
    }

    assert spelled["boltz2"]["compute_dtype"] == "bfloat16"
    assert spelled["protenix"]["trunk_dtype"] == "bf16"
    assert spelled["opendde"]["trunk_dtype"] == "bf16"


def test_one_kernel_spelling_reaches_every_model_that_has_one(request_with) -> None:
    """Same for the fused triangle kernel, which has three native names."""
    assert get_backend("boltz2").apply_sampling(request_with(triangle_kernel="cueq"))[
        "triangle_backend"
    ] == "cueq"
    assert get_backend("protenix").apply_sampling(request_with(triangle_kernel="cueq"))[
        "trunk_triangle_attention_backend"
    ] == "cueq_jit"
    of3 = get_backend("openfold3")
    assert of3.apply_sampling(request_with(triangle_kernel="cueq"))[
        "triangle_kernel"
    ] == "cueq"


def test_a_knob_a_model_does_not_have_is_an_error(request_with) -> None:
    """Not a silent no-op.

    OpenFold3 has no trunk dtype -- upstream runs `precision="32-true"` and a
    bfloat16 trunk destroys the prediction -- and OpenDDE exposes no triangle
    kernel. A request that asked for either and got a float32 cueq run anyway
    would be reporting something it did not measure.
    """
    with pytest.raises(ValueError, match="openfold3 does not support dtype"):
        get_backend("openfold3").apply_sampling(request_with(dtype="bfloat16"))

    with pytest.raises(ValueError, match="opendde does not support triangle_kernel"):
        get_backend("opendde").apply_sampling(request_with(triangle_kernel="cueq"))


def test_a_value_a_model_does_not_have_is_an_error(request_with) -> None:
    """`tokamax` is a Boltz-2 path; asking Protenix for it must not pick one."""
    with pytest.raises(ValueError, match="does not support attention_kernel"):
        get_backend("protenix").apply_sampling(request_with(attention_kernel="tokamax"))


def test_an_unknown_value_names_the_ones_that_exist(request_with) -> None:
    with pytest.raises(ValueError, match=r"dtype must be one of"):
        get_backend("boltz2").apply_sampling(request_with(dtype="fp8"))


def test_a_models_own_spelling_keeps_working_untouched(request_with) -> None:
    """A native name is that port's API, not an alias, and keeps its own values.

    `trunk_dtype=bf16` has to keep meaning what it meant on Protenix. Rewriting
    it through the neutral vocabulary would validate `bf16` against the neutral
    values -- which spell it `bfloat16` -- and break every script and every
    reproduction command in EXPERIMENT_LOG.md, which is the opposite of what an
    alias is for.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", execution.Alias)
        assert get_backend("protenix").apply_sampling(
            request_with(trunk_dtype="bf16")
        )["trunk_dtype"] == "bf16"
        assert get_backend("boltz2").apply_sampling(
            request_with(compute_dtype="bfloat16")
        )["compute_dtype"] == "bfloat16"


def test_another_models_spelling_is_accepted_and_named(request_with) -> None:
    """The point of the alias: `trunk_dtype` on Boltz-2 used to be an error."""
    with pytest.warns(execution.Alias, match="trunk_dtype"):
        options = get_backend("boltz2").apply_sampling(
            request_with(trunk_dtype="bfloat16")
        )

    assert options["compute_dtype"] == "bfloat16"


def test_setting_both_spellings_is_an_error(request_with) -> None:
    """Preferring one silently would change the run without changing the exit."""
    with pytest.raises(ValueError, match="both set"):
        get_backend("protenix").apply_sampling(
            request_with(dtype="bfloat16", trunk_dtype="fp32")
        )


def test_every_backend_declares_a_vocabulary_the_module_knows(request_with) -> None:
    """A typo in a backend's table would otherwise surface as a runtime error."""
    for model in available_models():
        for knob, (native, values) in get_backend(model).execution_options.items():
            assert knob in execution.KNOBS, f"{model}: unknown knob {knob}"
            assert isinstance(native, str) and native
            unknown = set(values) - set(execution.KNOBS[knob])
            assert not unknown, f"{model}.{knob}: not in the vocabulary: {unknown}"


def test_auto_resolves_to_the_fused_kernel_where_one_exists(request_with) -> None:
    """`auto` means the fastest path, and it is the same word everywhere.

    It is not "try cueq, fall back to xla": a silent fallback makes one command
    run two different programs on two machines, which is how a benchmark ends
    up comparing kernels instead of models.
    """
    for model, native in (
        ("boltz2", "triangle_backend"),
        ("protenix", "trunk_triangle_attention_backend"),
        ("openfold3", "triangle_kernel"),
    ):
        resolved = get_backend(model).apply_sampling(
            request_with(triangle_kernel="auto")
        )
        assert resolved[native] in ("cueq", "cueq_jit"), model


def test_openfold3_kernel_selection_restores_the_host_environment(
    monkeypatch,
) -> None:
    from foldjax.backends.openfold3 import _triangle_backend

    name = "OPENFOLD3_TRIANGLE_BACKEND"
    monkeypatch.setenv(name, "host-choice")
    with _triangle_backend("xla"):
        assert os.environ[name] == "xla"
    assert os.environ[name] == "host-choice"

    monkeypatch.delenv(name)
    with pytest.raises(RuntimeError, match="failed"):
        with _triangle_backend("cueq"):
            assert os.environ[name] == "cueq"
            raise RuntimeError("failed")
    assert name not in os.environ


def test_the_deterministic_knob_is_in_the_vocabulary_and_protenix_answers_it(
    request_with,
) -> None:
    """One spelling for "reduction orders that repeat", already renamed.

    Protenix reached this by way of a process-wide
    `XLA_FLAGS=--xla_gpu_deterministic_ops=true`, which no request can carry
    and which reaches every other model in the same process. The neutral name
    is here from the start so the second port that grows the option does not
    invent a second word for it.
    """
    assert execution.KNOBS["deterministic"] == ("off", "on")

    options = get_backend("protenix").apply_sampling(request_with(deterministic="on"))
    assert options["deterministic_ops"] == "on"
    assert "deterministic" not in options


#: Every port, and the native name each one renders `deterministic` as.
#:
#: Protenix and OpenDDE are driven by rendering argv for their own predict
#: parsers, whose flag is `--deterministic-ops`; the other four are called
#: through a Python signature whose parameter is `deterministic`.
_DETERMINISTIC_NATIVE_NAMES = {
    "alphafold3": "deterministic",
    "boltz2": "deterministic",
    "esmfold2": "deterministic",
    "opendde": "deterministic_ops",
    "openfold3": "deterministic",
    "protenix": "deterministic_ops",
}


@pytest.mark.parametrize(
    ("model", "native"), sorted(_DETERMINISTIC_NATIVE_NAMES.items())
)
def test_every_port_answers_the_deterministic_knob(
    request_with, model: str, native: str
) -> None:
    """One request, six models, one spelling -- including this one.

    Repeatability that only one port can be asked for is repeatability a
    cross-model benchmark cannot use: the caller has to know which port it is
    talking to, and the five that cannot answer keep depending on a
    process-wide `XLA_FLAGS`, which reaches every other model in the process.

    Both values, because `off` is the one a bool-shaped port can get wrong:
    `False` is a value its table supplies, not a missing entry.
    `tests/test_execution_knob_coverage.py` pins which shape each port takes;
    what matters here is that asking works and that the two answers differ.
    """
    off = get_backend(model).apply_sampling(request_with(deterministic="off"))
    on = get_backend(model).apply_sampling(request_with(deterministic="on"))

    assert native in off and native in on
    assert off[native] != on[native]
    assert off[native] in ("off", False) and on[native] in ("on", True)
    assert type(off[native]) is type(on[native])


@pytest.mark.parametrize(
    ("model", "native"), sorted(_DETERMINISTIC_NATIVE_NAMES.items())
)
def test_the_option_is_part_of_every_port_s_compile_identity(
    model: str, native: str
) -> None:
    """A repeatable-reduction executable is a different executable.

    Two runs that differ only here compile different programs, so sharing one
    cache namespace would hand a run that asked for deterministic reductions
    the program built without them.
    """
    assert native in get_backend(model).compile_options
