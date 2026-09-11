"""One census over every option surface, reconciled port by port.

An option in this repository is declared on up to eight surfaces, and nothing
made them agree:

    native_options          what a request may spell natively
    sampling_options        the four neutral sampling knobs, renamed per port
    execution_options       the neutral execution knobs, renamed per port
    compile_options         which names fork the compilation-cache namespace
    the strip table         which spelled values are the released ones
    cache_profile           what actually reaches the namespace digest
    the port's own CLI      the flags an argv-driven adapter renders into
    validate_request        what a planned request is allowed to carry

Each pair of those is a place a name can exist on one side and not the other,
and all three defects this file was written after were exactly that:

* `openfold3` named `confidence_dtype` in `compile_options` and left it out of
  `native_options`, so `validate_request` rejected every managed run that
  passed it. Caught by a GPU row dying in one second.
* `boltz2` named `pair_residual_dtype` in `_RELEASED_COMPILE_DEFAULTS` and left
  it out of `compile_options`, so the strip removed something that had never
  entered the profile and two programs shared one namespace. The CHANGELOG
  claimed the opposite.
* A capability present on one port with no slot in another's vocabulary,
  invisible because nothing compared the vocabularies.

`tests/test_backends.py` already holds the first two as
`test_every_compile_option_is_reachable_through_validate_request` and
`test_every_stripped_default_is_part_of_the_compile_identity`. Those are the
seed and they stay where they are; this file generalises them from two surface
pairs to the whole matrix, and adds the third defect class -- the one that
needs the ports compared against each other rather than each against itself.

Every gap found and deliberately not fixed here is a `pytest.mark.xfail(
strict=True)` naming that specific defect, one parameter per gap, so fixing
one port turns one xfail into a failure rather than closing silently. Nothing
in this file changes behaviour; a fix belongs on a branch that can measure it.

Where a disagreement is legitimate it is an entry in an explicit allow-list
with the code that justifies it, never a softened assertion -- and an entry
whose justification could not be verified says so, the way
`OUTSIDE_THE_IDENTITY` in the seed records that AlphaFold 3's
`kernel_autotuning` exemption rests on a comment rather than a measurement.
"""

from __future__ import annotations

import functools
import importlib
from collections.abc import Mapping
from typing import Any

import pytest

from foldjax import execution
from foldjax.registry import available_models, get_backend
from foldjax.schema import PredictionRequest

#: Every port this census covers. Pinned rather than derived so a seventh
#: backend cannot join the repository and skip the whole file; the coverage
#: test below compares it against the registry.
BACKENDS = ("alphafold3", "boltz2", "esmfold2", "opendde", "openfold3", "protenix")

#: Each port's strip table, which is spelled differently on ESMFold2 because
#: its values are fixed by the port rather than released by an upstream runner.
_STRIP_TABLE_ATTRIBUTE = {
    "alphafold3": "_RELEASED_COMPILE_DEFAULTS",
    "boltz2": "_RELEASED_COMPILE_DEFAULTS",
    "esmfold2": "_FIXED_COMPILE_DEFAULTS",
    "opendde": "_RELEASED_COMPILE_DEFAULTS",
    "openfold3": "_RELEASED_COMPILE_DEFAULTS",
    "protenix": "_RELEASED_COMPILE_DEFAULTS",
}

#: The two ports the adapter drives by rendering argv for their own predict
#: parser. Every other port is called through a Python signature, so its
#: "CLI surface" is that signature and is checked in
#: `tests/test_native_contracts.py` instead.
ARGV_PORTS = ("opendde", "protenix")


# ---------------------------------------------------------------------------
# Surfaces
# ---------------------------------------------------------------------------


def _module(name: str):
    return importlib.import_module(f"foldjax.backends.{name}")


def strip_table(name: str) -> Mapping[str, Any]:
    return getattr(_module(name), _STRIP_TABLE_ATTRIBUTE[name])


def generated_names(backend) -> set[str]:
    """The native names `sampling_options` and `execution_options` produce.

    These are accepted by `validate_request` without appearing in
    `native_options`; `Backend.validate_request` builds the same union.
    """
    names = set(backend.sampling_options.values())
    names.update(native for native, _values in backend.execution_options.values())
    return names


def accepted_names(backend) -> set[str]:
    """Every native option name a planned request is allowed to carry."""
    return generated_names(backend) | set(backend.native_options or ())


@functools.lru_cache(maxsize=None)
def native_parser(port: str):
    """The argparse parser an argv-driven port's `main` builds."""
    from tests._parser_capture import capture_parser

    native = importlib.import_module(f"foldjax.models.{port}.cli.predict")
    return capture_parser(native.main)


def parser_defaults(port: str) -> dict[str, Any]:
    """dest -> the value the native parser resolves an omitted flag to."""
    return {
        action.dest: action.default
        for action in native_parser(port)._actions
        if action.option_strings
    }


@pytest.fixture
def profile_of(tmp_path):
    """Call a backend's `cache_profile` for one spelling of the options.

    `cache_profile` translates and strips but loads no weights and imports no
    model runtime, so this is the one surface that can be read behaviourally
    rather than off a table -- which is the whole point of including it.
    """
    job = tmp_path / "job.json"
    job.write_text("{}")

    def build(name: str, **options) -> dict[str, Any]:
        return get_backend(name).cache_profile(
            PredictionRequest(
                model="boltz2",
                input=job,
                output_dir=tmp_path / "out",
                options=options,
            )
        )

    return build


def spelling_for(backend, native: str, value: Any) -> tuple[dict[str, Any] | None, str]:
    """How a request spells `native=value`, or why it cannot.

    A native name that is also a neutral knob -- `deterministic`, `dtype`,
    `matmul_precision`, `triangle_kernel` -- cannot be passed under its own
    name: `execution.translate` intercepts any key in `KNOBS` and validates it
    against the *neutral* vocabulary, so `deterministic=False` is rejected
    before it reaches the port. Those are spelled through the knob.
    """
    for knob, (name, values) in backend.execution_options.items():
        if name != native:
            continue
        neutral = [key for key, mapped in values.items() if mapped == value]
        if not neutral:
            return None, f"no neutral {knob} value maps to {value!r}"
        return {knob: neutral[0]}, ""
    if native in execution.KNOBS:
        return None, f"{native} is a neutral knob this port does not declare"
    return {native: value}, ""


# ---------------------------------------------------------------------------
# Invariants, as functions that return findings
# ---------------------------------------------------------------------------
#
# Written as functions rather than inline assertions so that each one can be
# called against a deliberately broken input. A conformance test that passes
# on a broken tree is worse than no test, and the only way to know is to break
# the tree; the `_has_power` tests at the bottom do exactly that.


def values_no_port_can_produce(knobs: Mapping[str, tuple[str, ...]] | None = None):
    """Neutral (knob, value) pairs that no backend's table maps to anything.

    A value in `KNOBS` is a promise that some model can run it. A value no
    port maps is not a per-port gap -- those are legitimate and declared model
    by model, which is what `test_a_value_a_model_does_not_have_is_an_error`
    in `tests/test_execution_vocabulary.py` pins -- it is a word in the
    vocabulary that every possible request is refused for.
    """
    table = execution.KNOBS if knobs is None else knobs
    findings = []
    for knob, values in table.items():
        for value in values:
            reachable = any(
                value in get_backend(name).execution_options.get(knob, (None, {}))[1]
                for name in BACKENDS
            )
            if not reachable:
                findings.append((knob, value))
    return findings


def knobs_outside_the_compile_identity(name: str) -> list[str]:
    """Neutral knobs this port declares whose native name never forks a cache.

    A port declares a neutral knob because the knob changes what that port
    runs. Every one of the five in `KNOBS` changes the compiled program --
    `dtype` and `matmul_precision` change operand and accumulator precision,
    the two kernel knobs select a different implementation, and
    `deterministic` asks XLA for different reduction orders -- so a declared
    knob whose native name is absent from `compile_options` hands two
    different programs one namespace.
    """
    backend = get_backend(name)
    compiled = set(backend.compile_options)
    return sorted(
        native
        for _knob, (native, _values) in backend.execution_options.items()
        if native not in compiled
    )


def shared_names_that_disagree() -> dict[str, dict[str, bool]]:
    """Native names two or more ports accept and classify differently.

    This is the third defect class, and the only one that cannot be seen from
    inside one backend: every surface in `boltz2.py` can be self-consistent
    while `diffusion_chunk_size` is compile-relevant on the three other ports
    that have it and not on that one. Nothing compared the vocabularies, so
    nothing noticed.

    Names are matched on the native spelling, which is the conservative
    choice: two ports that spell one concept differently are not compared, so
    this under-reports rather than inventing equivalences.
    """
    ownership: dict[str, dict[str, bool]] = {}
    for name in BACKENDS:
        backend = get_backend(name)
        compiled = set(backend.compile_options)
        for option in accepted_names(backend):
            ownership.setdefault(option, {})[name] = option in compiled
    return {
        option: ports
        for option, ports in sorted(ownership.items())
        if len(ports) > 1 and len(set(ports.values())) > 1
    }


#: Native options an argv port accepts that its flag loop never renders,
#: because the adapter handles them somewhere else. Each entry names the line
#: that does it; all three were read, not assumed.
_RENDERED_OUTSIDE_THE_FLAG_LOOP = {
    # Popped before the loop and rendered as a bare switch, because the native
    # flag takes no value. `src/foldjax/backends/opendde.py:222` pops it,
    # `:241` appends `--include-raw`.
    "opendde": {"include_raw"},
    # `output_format` is in the fixed argv prologue at
    # `src/foldjax/backends/protenix.py:455`, before the loop runs.
    # `cli_args` is the escape hatch: `:486` appends it verbatim, so it is not
    # a flag name at all but a list of them, guarded against colliding with a
    # reserved flag by `_RESERVED_CLI_FLAGS` at `:68`.
    "protenix": {"output_format", "cli_args"},
}


def native_options_no_flag_renders(port: str) -> list[str]:
    """Options an argv port accepts at plan time but cannot render at predict.

    `validate_request` accepts anything in `native_options`, and the adapter
    renders only `_CLI_OPTIONS`; on OpenDDE anything left over reaches
    `raise ValueError(f"unsupported OpenDDE options: ...")` after preprocessing
    has already run. So a name in one and not the other plans clean and dies
    late, which is the same shape as the OpenFold3 `confidence_dtype` defect
    with the two surfaces swapped.

    `tests/test_native_contracts.py` checks the other direction -- that every
    `_CLI_OPTIONS` name is a flag the native parser declares. Both directions
    are needed: that one catches a flag renamed upstream, this one catches an
    option offered to callers that no flag carries.

    Honest about its current reach: both argv ports *derive* `native_options`
    from `_CLI_OPTIONS` (`backends/opendde.py:106`,
    `backends/protenix.py:257`), so today this invariant holds by construction
    and cannot drift. It is here for the day one of them is rewritten as a
    literal set, which is what the four API-driven ports already are -- and
    which is how OpenFold3's `native_options` came to be missing a name its
    `compile_options` had.
    """
    backend = get_backend(port)
    cli = set(getattr(_module(port), "_CLI_OPTIONS"))
    # Generated names are rendered by the same loop (they are `_CLI_OPTIONS`
    # members) except `matmul_precision`, which `Backend.matmul_precision`
    # pops at `src/foldjax/backends/base.py:123` on every port because no
    # model takes it as an argument -- it travels in a ContextVar.
    exempt = cli | _RENDERED_OUTSIDE_THE_FLAG_LOOP[port] | {"matmul_precision"}
    return sorted(set(backend.native_options or ()) - exempt)


def released_defaults_that_fork(port: str, profile_of) -> list[str]:
    """Compile options whose released value is not neutralised anywhere.

    The strip tables exist so that spelling the value the native runner would
    have resolved anyway names the same namespace as omitting it. A port can
    also reach that by resolving the value into the profile unconditionally,
    which is what OpenFold3 does for `cp_layout` and Boltz-2 does for
    `pair_residual_dtype`; either mechanism satisfies the contract.

    A compile option with a real (non-`None`) default on the native parser and
    neither mechanism does not: `--option name=<the released value>` forks the
    namespace from omitting it, for a run that is identical.

    Only the two argv ports are checked, because only for them is the released
    default mechanically knowable -- it is `action.default` on the parser the
    adapter actually feeds. The four API-driven ports are reported as
    undetermined rather than guessed at.
    """
    backend = get_backend(port)
    stripped = set(strip_table(port))
    resolved = set(profile_of(port))
    findings = []
    for name, default in parser_defaults(port).items():
        if name not in backend.compile_options or default is None:
            continue
        if name in stripped or name in resolved:
            continue
        findings.append(name)
    return sorted(findings)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def test_the_census_covers_every_registered_backend() -> None:
    """A seventh port must not join the repository and skip this file."""
    assert set(BACKENDS) == set(available_models())
    assert set(_STRIP_TABLE_ATTRIBUTE) == set(BACKENDS)
    # Every port is reached one way or the other, and no port both ways.
    assert set(ARGV_PORTS) | set(API_PORTS) == set(BACKENDS)
    assert not set(ARGV_PORTS) & set(API_PORTS)


@pytest.mark.parametrize("name", BACKENDS)
def test_every_backend_declares_every_surface(name: str) -> None:
    """`native_options = None` is the open contract, and closes this census.

    Third-party adapters keep it, which is why the base class default is
    `None`; a built-in that took it would make `validate_request` accept any
    misspelling and would silently exempt itself from every reconciliation
    below.
    """
    backend = get_backend(name)
    assert backend.native_options is not None, f"{name} opted out of validation"
    assert backend.sampling_options, f"{name} declares no sampling knobs"
    assert backend.execution_options, f"{name} declares no execution knobs"
    assert backend.compile_options, f"{name} caches on nothing"
    assert strip_table(name), f"{name} strips nothing"


# ---------------------------------------------------------------------------
# The vocabulary against the ports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("knob", "value"),
    [
        pytest.param(
            knob,
            value,
            marks=(
                [
                    pytest.mark.xfail(
                        strict=True,
                        reason=(
                            "GAP: execution.py:73 offers attention_kernel='cueq' "
                            "and no backend's execution_options maps it to "
                            "anything. AlphaFold 3 maps auto/xla, Boltz-2 and "
                            "Protenix map auto/tokamax/xla, OpenDDE maps "
                            "auto/xla; the other two declare no attention "
                            "kernel at all. Every request spelling it is "
                            "refused with 'does not support attention_kernel=', "
                            "so the value is advertised vocabulary no model can "
                            "run. Either a port gains the mapping or the value "
                            "leaves KNOBS -- both are behaviour changes, so "
                            "neither happens here."
                        ),
                    )
                ]
                if (knob, value) == ("attention_kernel", "cueq")
                else []
            ),
        )
        for knob, values in execution.KNOBS.items()
        for value in values
    ],
)
def test_every_value_in_the_vocabulary_is_reachable_on_some_port(
    knob: str, value: str
) -> None:
    """A neutral value no port maps is a word every request is refused for.

    Not the same as a value one port lacks. `tokamax` on OpenDDE is a
    deliberate refusal -- the kernel would run there, but the value has not
    been measured there, and the vocabulary offers what a port has measured.
    That is per-port and legitimate. A value *no* port maps is different: it
    cannot be a measured choice anywhere, because there is nowhere it resolves
    to a program.
    """
    assert (knob, value) not in values_no_port_can_produce()


@pytest.mark.parametrize(
    ("name", "native"),
    [
        pytest.param(
            name,
            "matmul_precision",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    f"GAP: {name} declares the matmul_precision knob and leaves "
                    "it out of compile_options, so a run at 'highest' and a run "
                    "at 'high' share one compilation-cache namespace. This is "
                    "measured, not argued: lowering `a @ a` under each scope "
                    "emits `precision = [HIGH, HIGH]` and `precision = "
                    "[HIGHEST, HIGHEST]` -- two programs (see "
                    "test_the_precision_knob_provably_changes_the_program "
                    "below). AlphaFold 3 and Boltz-2 already list it. The fix "
                    "has two halves on the ports that pin a value of their own: "
                    "openfold3 pins 'high' at models/openfold3/inference.py:549 "
                    "and protenix reads the request at "
                    "models/protenix/models/predict.py:153, so each needs the "
                    "name in compile_options *and* an entry in its strip table "
                    "at that pinned value, or an explicit 'high' forks from an "
                    "omitted one. esmfold2 and opendde call "
                    "resolved_matmul_precision nowhere, so they pin nothing and "
                    "need the compile_options half only."
                ),
            ),
        )
        for name in ("esmfold2", "opendde", "openfold3", "protenix")
    ]
    + [
        pytest.param(name, native)
        for name in BACKENDS
        for _knob, (native, _values) in get_backend(name).execution_options.items()
        if (name, native) not in {
            ("esmfold2", "matmul_precision"),
            ("opendde", "matmul_precision"),
            ("openfold3", "matmul_precision"),
            ("protenix", "matmul_precision"),
        }
    ],
)
def test_a_declared_knob_forks_the_compile_identity(name: str, native: str) -> None:
    """Declaring a knob and not caching on it hands two programs one namespace.

    `tests/test_execution_vocabulary.py` already asserts this for
    `deterministic` on all six ports, with the argument that a
    repeatable-reduction executable is a different executable. The argument is
    not special to that knob: every name in `KNOBS` selects a different
    program, which is why each one is a knob rather than an output-formatting
    option. This is that assertion generalised to the whole table.
    """
    assert native not in knobs_outside_the_compile_identity(name)


def test_the_precision_knob_provably_changes_the_program() -> None:
    """The measurement the xfail reason above rests on, run here.

    The seed test's `OUTSIDE_THE_IDENTITY` entry records that AlphaFold 3's
    `kernel_autotuning` exemption is a comment and not a measurement, and
    leaves it a deferral. This one does not have to be: the two values differ
    in lowered StableHLO on CPU, in this process, in milliseconds. A
    conformance gap justified by running the thing is worth more than one
    justified by reading about it.
    """
    import jax
    import jax.numpy as jnp

    operand = jnp.ones((8, 8), dtype=jnp.float32)
    lowered = {}
    for value in ("high", "highest"):
        with execution.matmul_precision_scope(value):
            lowered[value] = jax.jit(lambda a: a @ a).lower(operand).as_text()

    assert "precision = [HIGH, HIGH]" in lowered["high"]
    assert "precision = [HIGHEST, HIGHEST]" in lowered["highest"]
    assert lowered["high"] != lowered["highest"]


def test_every_alias_renames_to_a_knob_some_port_actually_declares() -> None:
    """An alias for a knob nobody has is a deprecation path to a dead end.

    `ALIASES` exists so an old spelling keeps working. If the neutral name it
    rewrites to is not a knob, or is a knob no port declares, the rewrite
    turns a working native option into a hard error -- the opposite of what
    the alias is for.
    """
    for old, neutral in execution.ALIASES.items():
        assert neutral in execution.KNOBS, f"{old} aliases to a non-knob {neutral!r}"
        owners = [
            name
            for name in BACKENDS
            if get_backend(name).execution_options.get(neutral, (None, {}))[0] == old
        ]
        assert owners, (
            f"{old!r} is the old name for {neutral!r} but is no port's native "
            "spelling of it; an alias nobody owns rewrites a working option "
            "into an error"
        )


# ---------------------------------------------------------------------------
# The ports against each other
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "option",
    [
        pytest.param(
            "diffusion_chunk_size",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "GAP: boltz2 accepts diffusion_chunk_size and leaves it out "
                    "of compile_options; opendde, openfold3 and protenix all "
                    "put it in. It is NOT unconsumed -- a survey said so and "
                    "the source says otherwise: models/boltz2/api.py:578 takes "
                    "it, :1099 resolves it against auto_diffusion_chunk_size, "
                    "and _runner_identity at :522 already forks the retained "
                    "jit wrapper on it. So the port compiles two programs for "
                    "two widths and files them under one namespace. The fix "
                    "should record the *resolved* width rather than the "
                    "spelling, the way cache_profile already does for "
                    "pair_residual_dtype, because :534 proves chunk >= "
                    "multiplicity is the same program as None."
                ),
            ),
        ),
        pytest.param(
            "matmul_precision",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "GAP: alphafold3 and boltz2 cache on matmul_precision; "
                    "esmfold2, opendde, openfold3 and protenix accept it and do "
                    "not. Same defect as the four parameters of "
                    "test_a_declared_knob_forks_the_compile_identity, seen from "
                    "the other direction -- that one asks each port against the "
                    "vocabulary's promise, this one asks the ports against each "
                    "other. Both stay: a port could satisfy one and not the "
                    "other, and it is the disagreement itself that says which "
                    "of the two answers is the mistake."
                ),
            ),
        ),
    ]
    + [
        pytest.param(option)
        for option in sorted(
            {
                option
                for name in BACKENDS
                for option in accepted_names(get_backend(name))
            }
            - {"diffusion_chunk_size", "matmul_precision"}
        )
    ],
)
def test_ports_that_share_a_native_name_agree_about_caching_on_it(
    option: str,
) -> None:
    """The defect class no single backend's own surfaces can show.

    `boltz2.py` is internally consistent about `diffusion_chunk_size`: it is
    in `native_options`, it is validated, it is forwarded, it is consumed.
    Every check that reads one backend passes. The only thing that says it is
    wrong is the other three ports, which all decided the same name changes
    the program.

    Matching on the native spelling deliberately under-reports: two ports that
    call one concept different things are not compared here, so an entry in
    this list is always two ports that agreed on the word and disagreed on
    what it means.
    """
    disagreements = shared_names_that_disagree()
    assert option not in disagreements, (
        f"{option} is compile-relevant on some ports and not others: "
        f"{disagreements.get(option)}"
    )


# ---------------------------------------------------------------------------
# The adapters against the parsers they feed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("port", ARGV_PORTS)
def test_every_native_option_an_argv_port_accepts_can_be_rendered(port: str) -> None:
    """Accepted at plan time and unrenderable at predict time is the same bug.

    OpenFold3's `confidence_dtype` was rejected at `validate_request` for an
    option the model consumed. This is the mirror: an option `validate_request`
    waves through, that the flag loop has no flag for, and that surfaces only
    after featurization has run.
    """
    assert not native_options_no_flag_renders(port)


@pytest.mark.parametrize("port", ARGV_PORTS)
def test_the_native_default_of_every_knob_has_a_neutral_spelling(port: str) -> None:
    """`auto` must be able to name what the port runs when nobody asks.

    A port whose parser defaults `trunk_single_attention_backend` to `xla_jit`
    while the neutral vocabulary can only produce `xla` would make the
    model-neutral request unable to ask for the run the port ships -- which is
    the one every measurement in this repository describes.

    The reverse -- native values with no neutral spelling -- is deliberate and
    not asserted: Protenix's parser also accepts `xla_sdpa` and `cueq`, and
    OpenDDE's accepts `xla_sdpa`, and the vocabulary offers what a port has
    measured rather than what its parser will take.
    """
    backend = get_backend(port)
    defaults = parser_defaults(port)
    for knob, (native, values) in backend.execution_options.items():
        if native not in defaults or defaults[native] is None:
            continue
        assert defaults[native] in set(values.values()), (
            f"{port} runs {native}={defaults[native]!r} by default and no "
            f"neutral {knob} value produces it"
        )


# ---------------------------------------------------------------------------
# cache_profile, read behaviourally rather than off a table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", BACKENDS)
def test_spelling_the_value_the_profile_already_resolved_changes_nothing(
    name: str, profile_of
) -> None:
    """The one surface that cannot be read by comparing declarations.

    Four of the six ports override `cache_profile`, and two of them resolve
    values into the profile that no table mentions -- OpenFold3 writes
    `cp_layout`, `triangle_kernel`, `rng_route`, `representations` and
    `stop_after`; OpenDDE and Protenix write `return_confidence_details`. A
    census that only reconciled tables would not see any of that.

    So: take what the profile resolves an omitted option to, spell exactly
    that back, and require the digest input to be identical. An override that
    normalises the omission and forgets the explicit spelling fails here and
    nowhere else.
    """
    backend = get_backend(name)
    omitted = profile_of(name)
    for option, value in sorted(omitted.items()):
        spelling, why = spelling_for(backend, option, value)
        if spelling is None:
            pytest.skip(f"{name}.{option}: {why}")
        assert profile_of(name, **spelling) == omitted, (
            f"{name} resolves an omitted {option} to {value!r} but spelling "
            f"{spelling} selects a different namespace"
        )


@pytest.mark.parametrize(
    "port",
    [
        pytest.param(
            "protenix",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "GAP: protenix names model_name in compile_options, its "
                    "native parser defaults --model-name to 'auto', and neither "
                    "_RELEASED_COMPILE_DEFAULTS nor the cache_profile override "
                    "neutralises that value. So `--option model_name=auto` and "
                    "omitting it are the same run in two namespaces -- the "
                    "inverse of the pair_residual_dtype defect, which stripped "
                    "a default that had never entered the profile. Every other "
                    "non-None parser default on both argv ports is stripped; "
                    "this is the only one left. The fix is one strip-table "
                    "entry, but 'auto' resolves to a checkpoint-dependent model "
                    "name, so whether the entry should be 'auto' or the "
                    "resolved name is a measurement, not an edit."
                ),
            ),
        ),
        pytest.param("opendde"),
    ],
)
def test_spelling_a_released_default_selects_the_namespace_omitting_it_selects(
    port: str, profile_of
) -> None:
    """The contract the strip tables exist for, asked of the parser.

    The seed test asks whether a stripped name reaches the profile at all.
    This asks the other half: whether every name that reaches the profile and
    has a released value gets stripped. Both halves are needed -- Boltz-2's
    `pair_residual_dtype` failed the first and Protenix's `model_name` fails
    the second, and neither test sees the other's defect.
    """
    assert not released_defaults_that_fork(port, profile_of)


#: Every compile option with no strip-table entry, and why it does not need
#: one. The strip tables neutralise an explicitly spelled released default so
#: it names the namespace an omitted option names; a compile option outside
#: the table needs another reason, and this is the census of those reasons.
#:
#: Four categories, and each entry says which evidence it rests on. Where the
#: justification could not be verified the entry says so rather than being
#: quietly dropped, the way the seed's `OUTSIDE_THE_IDENTITY` records that
#: AlphaFold 3's `kernel_autotuning` exemption is a comment.
_WHY_NO_STRIP_ENTRY = {
    "alphafold3": {
        # `_prediction_buckets` at backends/alphafold3.py:447 resolves an
        # omitted `buckets` to `None` via `options.pop("buckets", ()) or None`.
        # A request cannot spell `None`, so there is no value to alias.
        "buckets": "no released default a request can spell",
        # AlphaFold 3 calls `resolved_matmul_precision` nowhere -- only
        # boltz2/api.py:259, openfold3/inference.py:577 and
        # protenix/models/predict.py:153 do -- so this port pins nothing and
        # an omitted knob is JAX's default rather than a released value.
        "matmul_precision": "no released default a request can spell",
    },
    "boltz2": {
        # models/boltz2/api.py:317 defaults it to `None`.
        "max_msa_depth": "no released default a request can spell",
        # api.py:1090 keeps `None` as `None` rather than resolving a width.
        "token_attention_chunk": "no released default a request can spell",
    },
    "esmfold2": {
        # Injected into every effective request at backends/esmfold2.py:875,
        # so omitted and spelled resolve to the same 9. Proved behaviourally
        # by test_spelling_the_value_the_profile_already_resolved_changes_
        # nothing, which finds `num_recycles` in this port's base profile.
        "num_recycles": "resolved into every profile",
        # VERIFIED, not merely claimed by the comment at esmfold2.py:71-73:
        # the omitted resolution is read out of the checkpoint config at
        # models/esmfold2/models/model.py:365 (`num_diffusion_samples`, with
        # the released `base.num_samples` only as a fallback) and, for the
        # step count, by `diffusion.settings_from_config(config)` on the line
        # below it. `DEFAULTS` names the paper's 32/14 schedule for
        # documentation and is read nowhere for these two keys. So there is
        # no fixed value to strip: aliasing 32 would claim a checkpoint
        # resolves to 32 when the file decides.
        "num_samples": "checkpoint-dependent resolution",
        "num_steps": "checkpoint-dependent resolution",
    },
    "openfold3": {
        # Popped by the override when they equal the released value:
        # backends/openfold3.py:319-325 (all_arrays), :353 (dtype), :361
        # (confidence_dtype, against the request's own dtype) and :378
        # (glu_backend). Read, not assumed.
        "all_arrays": "neutralised in the cache_profile override",
        "confidence_dtype": "neutralised in the cache_profile override",
        "dtype": "neutralised in the cache_profile override",
        "glu_backend": "neutralised in the cache_profile override",
        # Written unconditionally at :363-374, so the resolved value is in
        # every profile and spelling it back is the same digest.
        "cp_devices": "resolved into every profile",
        "cp_layout": "resolved into every profile",
        "triangle_kernel": "resolved into every profile",
        # models/openfold3/inference.py:140 defaults it to `None`.
        "pair_chunk_size": "no released default a request can spell",
        # UNDETERMINED. inference.py:1320 defaults it to the string "auto",
        # resolved by `auto_diffusion_chunk_size(num_samples)` -- which is
        # `None` at the released five samples and a width above it. So the
        # value an omitted option resolves to depends on the sample count,
        # and a flat strip entry cannot express it; it needs the resolve-style
        # treatment `cache_profile` gives Boltz-2's `pair_residual_dtype`.
        # Whether any spelling currently aliases wrongly was not measured.
        "diffusion_chunk_size": "undetermined",
    },
}


#: The four ports called through a Python signature. Only these need the
#: record above: on OpenDDE and Protenix the released default is
#: `action.default` on the parser the adapter feeds, so
#: `released_defaults_that_fork` settles the same question mechanically and
#: every name it passes over has a `None` default. Protenix's `cli_args` is
#: the one compile option on either port with no parser dest at all -- it is a
#: list of flags rather than a flag, and an omitted one is the empty tuple.
API_PORTS = ("alphafold3", "boltz2", "esmfold2", "openfold3")


@pytest.mark.parametrize("name", API_PORTS)
def test_every_compile_option_outside_the_strip_table_has_a_reason(name: str) -> None:
    """Nothing leaves the strip table without a recorded reason.

    The seed asks whether a stripped name reaches the profile. The argv test
    above asks whether a name with a released default gets stripped. This is
    the remainder, for the ports where that second question cannot be asked
    mechanically: every compile option the strip table does not mention, with
    the evidence that it does not need mentioning.

    Adding a compile option and not deciding this question fails here, which
    is the point -- the Boltz-2 `pair_residual_dtype` defect and the Protenix
    `model_name` gap are both cases where the question was never asked.
    """
    outside = {
        option
        for option in get_backend(name).compile_options
        if option not in strip_table(name)
    }
    recorded = set(_WHY_NO_STRIP_ENTRY.get(name, {}))
    assert outside == recorded, (
        f"{name}: {sorted(outside - recorded)} left the strip table with no "
        f"recorded reason; {sorted(recorded - outside)} have a reason and are "
        "no longer outside it"
    )


# ---------------------------------------------------------------------------
# Power checks
# ---------------------------------------------------------------------------
#
# Each invariant, broken on purpose in one backend, must produce a finding
# that names that backend and that option. Without these the file's passing
# parameters prove nothing: an invariant that cannot see a defect passes on a
# tree full of them, which is worse than having no test at all because it also
# says the tree is clean.


def test_the_vocabulary_invariant_notices_a_value_nobody_maps() -> None:
    assert values_no_port_can_produce({"dtype": ("float32", "int4")}) == [
        ("dtype", "int4")
    ]
    # And does not fire on the values that are mapped.
    assert values_no_port_can_produce({"dtype": ("float32", "bfloat16")}) == []


def test_the_compile_identity_invariant_notices_a_dropped_knob(monkeypatch) -> None:
    from foldjax.backends.boltz2 import Boltz2Backend

    assert knobs_outside_the_compile_identity("boltz2") == []
    monkeypatch.setattr(
        Boltz2Backend,
        "compile_options",
        tuple(o for o in Boltz2Backend.compile_options if o != "triangle_backend"),
    )
    assert knobs_outside_the_compile_identity("boltz2") == ["triangle_backend"]


def test_the_cross_port_invariant_notices_a_port_that_stops_agreeing(
    monkeypatch,
) -> None:
    from foldjax.backends.openfold3 import OpenFold3Backend

    assert "glu_backend" not in shared_names_that_disagree()
    monkeypatch.setattr(
        OpenFold3Backend,
        "compile_options",
        tuple(o for o in OpenFold3Backend.compile_options if o != "glu_backend"),
    )
    disagreement = shared_names_that_disagree()["glu_backend"]
    assert disagreement["openfold3"] is False
    assert disagreement["boltz2"] is True


def test_the_render_invariant_notices_an_unrenderable_option(monkeypatch) -> None:
    from foldjax.backends.opendde import OpenDDEBackend

    assert native_options_no_flag_renders("opendde") == []
    monkeypatch.setattr(
        OpenDDEBackend,
        "native_options",
        OpenDDEBackend.native_options | {"invented_option"},
    )
    assert native_options_no_flag_renders("opendde") == ["invented_option"]


def test_the_released_default_invariant_notices_an_unstripped_default(
    monkeypatch, profile_of
) -> None:
    from foldjax.backends import opendde as opendde_module

    assert released_defaults_that_fork("opendde", profile_of) == []
    monkeypatch.setattr(
        opendde_module,
        "_RELEASED_COMPILE_DEFAULTS",
        {
            key: value
            for key, value in opendde_module._RELEASED_COMPILE_DEFAULTS.items()
            if key != "n_queries"
        },
    )
    assert released_defaults_that_fork("opendde", profile_of) == ["n_queries"]


def test_the_profile_invariant_notices_an_override_that_forgets_a_spelling(
    monkeypatch, profile_of
) -> None:
    """Break the resolve-style normalisation and the behavioural test fires.

    OpenFold3 resolves `cp_layout` into every profile. Strip it back out when
    the caller spells it and the two spellings name two namespaces -- which is
    exactly the failure the table-only census cannot reach.
    """
    from foldjax.backends.openfold3 import OpenFold3Backend

    omitted = profile_of("openfold3")
    assert "cp_layout" in omitted
    assert profile_of("openfold3", cp_layout=omitted["cp_layout"]) == omitted

    original = OpenFold3Backend.cache_profile

    def forgetful(self, request):
        profile = original(self, request)
        if "cp_layout" in request.options:
            profile.pop("cp_layout", None)
        return profile

    monkeypatch.setattr(OpenFold3Backend, "cache_profile", forgetful)
    assert profile_of("openfold3", cp_layout=omitted["cp_layout"]) != omitted


def test_the_strip_table_record_notices_an_option_with_no_recorded_reason(
    monkeypatch,
) -> None:
    """Adding a compile option without deciding its released-default story."""
    from foldjax.backends.esmfold2 import ESMFold2Backend

    outside = {
        option
        for option in ESMFold2Backend.compile_options
        if option not in strip_table("esmfold2")
    }
    assert outside == set(_WHY_NO_STRIP_ENTRY["esmfold2"])

    monkeypatch.setattr(
        ESMFold2Backend,
        "compile_options",
        ESMFold2Backend.compile_options + ("invented_option",),
    )
    with pytest.raises(AssertionError, match="invented_option"):
        test_every_compile_option_outside_the_strip_table_has_a_reason("esmfold2")
