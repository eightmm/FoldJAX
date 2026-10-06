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
        for model in ("boltz2", "protenix", "opendde", "openfold3")
    }

    assert spelled["boltz2"]["compute_dtype"] == "bfloat16"
    assert spelled["protenix"]["trunk_dtype"] == "bf16"
    assert spelled["opendde"]["trunk_dtype"] == "bf16"
    assert spelled["openfold3"]["dtype"] == "bfloat16"


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

    OpenDDE exposes no triangle kernel. A request that asked for one and got
    its only kernel anyway would be reporting something it did not measure.

    OpenFold3 used to be the example on the other side of this test: it had no
    `dtype` at all, because a whole-trunk bfloat16 cast destroys its prediction
    and upstream infers at `precision="32-true"`. It has one now -- a partial
    profile that narrows the token/pair track and leaves the input embedder,
    the denoiser and every head float32 -- so the knob is real and the
    assertion moved to `test_openfold3_dtype_is_a_partial_profile` below.
    """
    with pytest.raises(ValueError, match="opendde does not support triangle_kernel"):
        get_backend("opendde").apply_sampling(request_with(triangle_kernel="cueq"))


def test_openfold3_dtype_is_a_partial_profile(request_with) -> None:
    """One spelling, and a value the port can actually run.

    `float32` stays the default, which is upstream's inference precision; the
    neutral name reaches OpenFold3's own `dtype` rather than a fourth spelling.
    """
    of3 = get_backend("openfold3")
    assert (
        of3.apply_sampling(request_with(dtype="bfloat16"))["dtype"] == "bfloat16"
    )
    assert "dtype" not in of3.apply_sampling(request_with())
    with pytest.raises(ValueError, match=r"dtype must be one of"):
        of3.apply_sampling(request_with(dtype="fp8"))


def test_a_value_a_model_does_not_have_is_an_error(request_with) -> None:
    """`tokamax` is a Boltz-2 and Protenix path; OpenDDE must not pick one.

    OpenDDE reaches the single-attention site through Protenix's own
    primitives, so the kernel would run there. It is still an error, because
    the value has not been measured on OpenDDE and the vocabulary offers what
    a port has measured rather than what its imports make reachable.
    """
    with pytest.raises(ValueError, match="does not support attention_kernel"):
        get_backend("opendde").apply_sampling(request_with(attention_kernel="tokamax"))


def test_the_protenix_fused_attention_value_reaches_its_native_name(
    request_with,
) -> None:
    """`attention_kernel=tokamax` is the neutral spelling of the trunk site."""
    assert get_backend("protenix").apply_sampling(
        request_with(attention_kernel="tokamax")
    )["trunk_single_attention_backend"] == "tokamax"


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
    resolved = get_backend("boltz2").apply_sampling(
        request_with(triangle_kernel="auto")
    )
    assert resolved["triangle_backend"] == "cueq"


def test_protenix_auto_is_the_omitted_kernel(request_with) -> None:
    """`auto` used to pin `cueq_jit`, which no omitted Protenix run selects.

    An omitted trunk triangle attention reaches the runner as None and takes
    the untraced multiplication; `cueq_jit` traces it too. `auto` now passes
    nothing, so it is the omitted run and its cache namespace.
    """
    protenix = get_backend("protenix")
    auto = request_with(triangle_kernel="auto")
    assert "trunk_triangle_attention_backend" not in protenix.apply_sampling(auto)
    assert protenix.cache_profile(auto) == protenix.cache_profile(request_with())
    assert protenix.cache_profile(
        request_with(triangle_kernel="cueq")
    ) != protenix.cache_profile(request_with())


@pytest.mark.parametrize("cp_devices", [1, 4])
@pytest.mark.parametrize("gpu", [False, True])
def test_openfold3_auto_is_the_omitted_kernel(
    request_with, monkeypatch, gpu: bool, cp_devices: int
) -> None:
    """OpenFold3's fastest kernel depends on the platform and the device count.

    `auto` used to pin `cueq`, the attention-only kernel no omitted run
    selects: `cueq-pallas` on a GPU, `cueq-full` elsewhere, `xla` under
    context parallelism. It now passes nothing, so it names the same kernel
    and the same cache namespace as omitting the knob.
    """
    from foldjax.models import _pallas_pair

    monkeypatch.setattr(_pallas_pair, "gpu_process", lambda: gpu)
    monkeypatch.delenv("OPENFOLD3_TRIANGLE_BACKEND", raising=False)
    of3 = get_backend("openfold3")
    auto = request_with(triangle_kernel="auto", cp_devices=cp_devices)
    omitted = request_with(cp_devices=cp_devices)
    assert "triangle_kernel" not in of3.apply_sampling(auto)
    expected = "xla" if cp_devices > 1 else "cueq-pallas" if gpu else "cueq-full"
    assert of3.cache_profile(auto)["triangle_kernel"] == expected
    assert of3.cache_profile(auto) == of3.cache_profile(omitted)


def test_cueq_pallas_is_requestable_where_a_port_runs_it(request_with) -> None:
    """A kernel an omitted OpenFold3 run selects must be nameable in the knob."""
    of3 = get_backend("openfold3")
    assert (
        of3.apply_sampling(request_with(triangle_kernel="cueq-pallas"))[
            "triangle_kernel"
        ]
        == "cueq-pallas"
    )
    for model in ("boltz2", "protenix"):
        with pytest.raises(
            ValueError, match=f"{model} does not support triangle_kernel='cueq-pallas'"
        ):
            get_backend(model).apply_sampling(
                request_with(triangle_kernel="cueq-pallas")
            )


_LONG = {"bfloat16": "bfloat16", "float32": "float32"}
_SHORT = {"bfloat16": "bf16", "float32": "fp32"}

#: (model, option as passed, the native option it lands on, what lands there)
#: through the neutral knob, an alias and each port's own `*_dtype` options.
_WIDTHS = [
    ("boltz2", "dtype", "compute_dtype", _LONG),
    ("boltz2", "compute_dtype", "compute_dtype", _LONG),
    ("boltz2", "diffusion_compute_dtype", "diffusion_compute_dtype", _LONG),
    ("boltz2", "pair_residual_dtype", "pair_residual_dtype", _LONG),
    ("protenix", "dtype", "trunk_dtype", _SHORT),
    ("protenix", "trunk_dtype", "trunk_dtype", _SHORT),
    ("opendde", "dtype", "trunk_dtype", _SHORT),
    ("opendde", "trunk_dtype", "trunk_dtype", _SHORT),
    ("opendde", "confidence_dtype", "confidence_dtype", _SHORT),
    ("opendde", "diffusion_dtype", "diffusion_dtype", _SHORT),
    ("openfold3", "dtype", "dtype", _LONG),
    ("openfold3", "trunk_dtype", "dtype", _LONG),
    ("openfold3", "compute_dtype", "dtype", _LONG),
    ("openfold3", "confidence_dtype", "confidence_dtype", _LONG),
    ("esmfold2", "confidence_dtype", "confidence_dtype", _LONG),
]
_SPELLINGS = {
    "bfloat16": ("bf16", "bfloat16", "BF16"),
    "float32": ("fp32", "float32", "f32"),
}


@pytest.mark.parametrize(
    ("model", "option", "native", "lands"),
    _WIDTHS,
    ids=[f"{model}-{option}" for model, option, _native, _lands in _WIDTHS],
)
def test_every_width_spelling_reaches_every_port(
    tmp_path, model: str, option: str, native: str, lands: dict[str, str]
) -> None:
    """`bf16` and `bfloat16` name one width on every port and in every option.

    The aliases renamed keys but not values, so `trunk_dtype=bf16` was refused
    on OpenFold3 and `confidence_dtype=float32` on OpenDDE. The request is
    also validated, so the port's own check accepts what lands.
    """
    import warnings

    job = tmp_path / "job.json"
    job.write_text("{}")
    weights = tmp_path / "w"
    weights.write_bytes(b"x")
    backend = get_backend(model)
    for width, spellings in _SPELLINGS.items():
        for spelling in spellings:
            request = PredictionRequest(
                model=model,
                input=job,
                input_format="foldjax",
                weights=weights,
                output_dir=tmp_path / "out",
                options={option: spelling},
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", execution.Alias)
                options = backend.apply_sampling(request)
                assert options[native] == lands[width], spelling
                backend.validate_request(request)


def test_a_width_that_is_not_one_passes_through_to_the_port(request_with) -> None:
    """Only width spellings are rewritten; the port still judges the rest."""
    boltz2 = get_backend("boltz2")
    options = boltz2.apply_sampling(request_with(pair_residual_dtype="auto"))
    assert options["pair_residual_dtype"] == "auto"
    with pytest.raises(ValueError, match=r"dtype must be one of"):
        boltz2.apply_sampling(request_with(dtype="float16"))


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


_SWITCH_SPELLINGS = {
    True: (True, 1, "true", "TRUE", "1", "yes", "Yes", "on"),
    False: (False, 0, "false", "False", "0", "no", "NO", "off"),
}


@pytest.mark.parametrize(
    ("model", "option"),
    [
        (model, option)
        for model in available_models()
        for option in sorted(get_backend(model).boolean_options)
    ],
)
def test_every_switch_spelling_reaches_every_port_as_a_bool(
    tmp_path, model: str, option: str
) -> None:
    """One vocabulary for a switch, whatever the port's own validator takes.

    `_strict_boolean` took only a real `bool` and the Protenix/OpenDDE rules
    only `true`/`false`; now every spelling lands as a `bool`, which both
    accept, so the request also validates.
    """
    job = tmp_path / "job.json"
    job.write_text("{}")
    weights = tmp_path / "w"
    weights.write_bytes(b"x")
    backend = get_backend(model)
    for switch, spellings in _SWITCH_SPELLINGS.items():
        for spelling in spellings:
            request = PredictionRequest(
                model=model,
                input=job,
                input_format="foldjax",
                weights=weights,
                output_dir=tmp_path / "out",
                options={option: spelling},
            )
            value = backend.apply_sampling(request)[option]
            assert value is switch, spelling
            backend.validate_request(request)


def test_only_a_declared_switch_is_rewritten(request_with) -> None:
    """`"1"` is a count elsewhere, and an unknown spelling is the port's to refuse."""
    protenix = get_backend("protenix")
    options = protenix.apply_sampling(
        request_with(diffusion_chunk_size="1", use_template="maybe")
    )
    assert options["diffusion_chunk_size"] == "1"
    assert options["use_template"] == "maybe"


def test_the_deterministic_knob_takes_the_switch_spellings(request_with) -> None:
    assert get_backend("boltz2").apply_sampling(request_with(deterministic="yes"))[
        "deterministic"
    ] is True
    assert get_backend("protenix").apply_sampling(request_with(deterministic=False))[
        "deterministic_ops"
    ] == "off"


def test_every_native_switch_is_declared_as_one() -> None:
    """A native option whose released default is a `bool` is a switch.

    Read off the backend module's default tables rather than its validators,
    so a switch added later -- with its released default named the way every
    port names one -- fails here until it joins `boolean_options`.
    """
    import importlib

    for model in available_models():
        backend = get_backend(model)
        module = importlib.import_module(type(backend).__module__)
        native = set(backend.native_options or ())
        defaulted = {
            key
            for name, table in vars(module).items()
            if "DEFAULT" in name and isinstance(table, dict)
            for key, value in table.items()
            if isinstance(key, str) and isinstance(value, bool) and key in native
        }
        assert defaulted <= backend.boolean_options, (model, defaulted)
        assert backend.boolean_options <= native, model
