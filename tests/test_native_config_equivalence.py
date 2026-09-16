"""The Protenix and OpenDDE adapters build the configuration the parser would.

Both adapters used to run a prediction by rendering the native command line and
handing it back to their own ``argparse`` parser. They now build the parser's
own destination namespace -- a ``PredictionConfig`` -- straight from the
resolved request and call ``run_prediction``. The rendered command is still
produced, because it is the spelling the result records, but nothing executes
from it.

That makes the parser the oracle: for every supported option, the
configuration built from the request must equal the configuration the parser
would have produced from the argv rendered for the same request, field by
field and type by type. ``_native_invocation`` returns both halves from one
pass, so what is compared here is the pair ``predict`` itself uses.

Two things the comparison cannot see on its own, and so are tested separately:

* ``choices=``. The equivalence matrix only feeds values the parser accepts, so
  it can never notice a vocabulary that is no longer checked -- and below the
  parser several of these values are read with ``==`` and no vocabulary at all
  (``runner._load_prepared_params`` branches on ``bf16`` and loads FP32 weights
  for anything else). The specs' vocabularies are pinned to the parser's and
  every one of them is shown to refuse an unknown value.
* Which entry point ran. An adapter that still rendered argv and called
  ``main`` would pass every equality assertion here, so the last tests watch
  the call itself: the runner for an ordinary request, and the parser only for
  Protenix's ``cli_args`` escape hatch, whose arbitrary native flags only that
  parser can turn into option values.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import pytest

from foldjax.backends import opendde as opendde_backend
from foldjax.backends import protenix as protenix_backend
from foldjax.backends.opendde import OpenDDEBackend
from foldjax.backends.protenix import ProtenixBackend
from foldjax.models.opendde import runner as opendde_runner
from foldjax.models.opendde.cli import predict as opendde_predict
from foldjax.models.protenix import runner as protenix_runner
from foldjax.models.protenix.cli import predict as protenix_predict
from foldjax.schema import PaddingConfig, PredictionRequest
from tests._parser_capture import capture_parser


class _Port(NamedTuple):
    """One argv-driven port: its adapter, its parser and its runner."""

    name: str
    backend: Any
    module: Any
    main: Any
    runner: Any


PROTENIX = _Port(
    name="protenix",
    backend=ProtenixBackend,
    module=protenix_backend,
    main=protenix_predict.main,
    runner=protenix_runner,
)
OPENDDE = _Port(
    name="opendde",
    backend=OpenDDEBackend,
    module=opendde_backend,
    main=opendde_predict.main,
    runner=opendde_runner,
)
PORTS = (PROTENIX, OPENDDE)


def _request(tmp_path: Path, port: _Port, **kwargs: Any) -> PredictionRequest:
    """One prediction request for this port, with its weights on disk."""
    input_path = tmp_path / "job.json"
    input_path.write_text("{}", encoding="utf-8")
    weights = tmp_path / f"{port.name}.jax"
    weights.touch()
    fields: dict[str, Any] = {
        "model": port.name,
        "input": input_path,
        "weights": weights,
        "output_dir": tmp_path / "out",
        "seed": 5,
        "cache_dir": tmp_path / "cache",
    }
    fields.update(kwargs)
    return PredictionRequest(**fields)


def _parser_namespace(port: _Port, argv: list[str]) -> dict[str, Any]:
    """The namespace ``main`` would have run this argv as, MSA depth resolved."""
    args = capture_parser(port.main).parse_args(argv)
    args.max_msa_depth = port.runner._resolve_msa_depth(args.max_msa_depth)
    return vars(args)


def _typed(fields: dict[str, Any]) -> dict[str, tuple[Any, type]]:
    """Fields with their exact types.

    ``5 == 5.0`` and ``1 == True``, so comparing values alone would accept a
    float where the parser produced an int -- which is the kind of drift a
    rendered-and-reparsed value could never have.
    """
    return {name: (value, type(value)) for name, value in fields.items()}


def _assert_equivalent(port: _Port, request: PredictionRequest) -> None:
    """The request-built configuration is the parsed-argv one, field by field."""
    invocation = port.backend()._native_invocation(request)
    built = invocation.config_fields
    parsed = _parser_namespace(port, invocation.argv)
    # Construction proves the field set is exactly the parser's: a missing or
    # extra name is a `TypeError` here rather than a silent default.
    assert port.runner.PredictionConfig(**built) == port.runner.PredictionConfig(
        **parsed
    )
    differences = {
        name: (built.get(name), parsed.get(name))
        for name in set(built) | set(parsed)
        if _typed(built).get(name) != _typed(parsed).get(name)
    }
    assert differences == {}


#: One valid value per native option, so every option is exercised explicitly
#: at least once. Values that differ from the parser's default, because a value
#: equal to it cannot tell an applied option from a dropped one.
_PROTENIX_OPTION_VALUES: dict[str, Any] = {
    "amp_policy": "upstream",
    "chunk_policy": "manual",
    "confidence_triangle_attention_backend": "xla_jit",
    "cp_atom_windows": False,
    "cp_devices": 2,
    "cp_layout": "2d",
    "deterministic_ops": "on",
    "diffusion_attention_backend": "xla_jit",
    "diffusion_chunk_size": 2,
    "esm_checkpoint_dir": Path("/tmp/esm-checkpoint"),
    "glu_backend": "tokamax",
    "max_msa_depth": 1024,
    "memory_budget_gib": 12.5,
    "memory_check": "warn",
    "model_name": "protenix-v2",
    "msa_seed": 7,
    "num_recycles": 2,
    "num_samples": 3,
    "num_steps": 7,
    "opm_chunk_size": 16,
    "single_att_q_chunk_size": 32,
    "strict_token_limit": True,
    "token_q_chunk_size": 256,
    "triangle_att_q_chunk_size": 128,
    "triangle_mul_chunk_size": 64,
    "trunk_dtype": "fp32",
    "trunk_single_attention_backend": "tokamax",
    "trunk_triangle_attention_backend": "cueq_jit",
}

_OPENDDE_OPTION_VALUES: dict[str, Any] = {
    "ccd_rdkit_cache": Path("/tmp/components.cif.rdkit_mol.pkl"),
    "chunk_policy": "manual",
    "components_cif": Path("/tmp/components.cif"),
    "confidence_dtype": "fp32",
    "cp_atom_windows": False,
    "cp_devices": 4,
    "cp_layout": "1d",
    "deterministic_ops": "on",
    "diffusion_attention_backend": "xla",
    "diffusion_chunk_size": 2,
    "diffusion_dtype": "bf16",
    "kalign_binary": Path("/tmp/kalign"),
    "max_msa_depth": 1024,
    "n_keys": 64,
    "n_queries": 16,
    "num_recycles": 2,
    "num_samples": 3,
    "num_steps": 7,
    "single_att_q_chunk_size": 32,
    "structural_single_attention_backend": "xla_sdpa",
    "template_mmcif_dir": Path("/tmp/mmcif"),
    "template_obsolete_map": Path("/tmp/obsolete_to_successor.json"),
    "template_release_dates": Path("/tmp/release_date_cache.json"),
    "token_q_chunk_size": 256,
    "triangle_att_q_chunk_size": 128,
    "triangle_mul_chunk_size": 64,
    "trunk_dtype": "fp32",
    "trunk_single_attention_backend": "xla",
    "use_rna_msa": True,
    "use_template": True,
}

#: Requests that are not one option: the shapes the adapter renders specially.
_PROTENIX_REQUESTS: dict[str, dict[str, Any]] = {
    "bare": {},
    # Every option at once except the fused GLU, which this port refuses to
    # combine with a mesh (`validate_native_options`) -- and the mesh is what
    # `cp_devices` here asks for. It is covered alone, below.
    "every-option": {
        "options": {
            name: value
            for name, value in _PROTENIX_OPTION_VALUES.items()
            if name != "glu_backend"
        }
    },
    "fused-glu": {"options": {"glu_backend": "tokamax"}},
    "switches-off": {
        "options": {"cp_atom_windows": True, "strict_token_limit": False}
    },
    # `--option num_samples=3` arrives as JSON's int and a quoted one as text,
    # so the same run can reach the adapter either way and has to land on the
    # parser's int either way. `msa_seed` and `memory_budget_gib` are absent
    # deliberately: text is refused for those two before any of this, by
    # `validate_native_options`.
    "numbers-as-text": {
        "options": {
            "num_samples": "3",
            "cp_devices": "2",
            "max_msa_depth": "2048",
            "opm_chunk_size": "8",
        }
    },
    "paths-as-text": {"options": {"esm_checkpoint_dir": "/tmp/esm-as-text"}},
    "sampling-knobs": {
        "num_samples": 2,
        "num_steps": 11,
        "num_recycles": 3,
        "max_msa_depth": 512,
    },
    "execution-knobs": {
        "options": {
            "dtype": "float32",
            "attention_kernel": "tokamax",
            "triangle_kernel": "cueq",
            "deterministic": "on",
            "matmul_precision": "highest",
        }
    },
    "output-format-both": {"options": {"output_format": "both"}},
    "output-format-npz": {"options": {"output_format": "npz"}},
    "no-compile-cache": {"cache_dir": None, "use_compile_cache": False},
    "representations": {"representations": ("single", "pair")},
    "representations-all": {"representations": "all"},
    "stop-after-trunk": {"representations": "all", "stop_after": "trunk"},
    "stop-after-inputs": {"representations": "all", "stop_after": "inputs"},
    "padding-profile": {"padding": True},
    "padding-explicit": {
        "padding": PaddingConfig(
            tokens=256,
            atoms=6144,
            msa=1024,
            templates=4,
            language_model_tokens=258,
            overflow="exact",
        )
    },
    "padding-and-options": {
        "padding": PaddingConfig(tokens=128, overflow="exact"),
        "options": {"cp_devices": 2, "cp_layout": "1d"},
    },
}

_OPENDDE_REQUESTS: dict[str, dict[str, Any]] = {
    "bare": {},
    "every-option": {"options": dict(_OPENDDE_OPTION_VALUES)},
    "switches-off": {
        "options": {
            "cp_atom_windows": True,
            "use_template": False,
            "use_rna_msa": False,
        }
    },
    "numbers-as-text": {
        "options": {
            "num_samples": "3",
            "cp_devices": "4",
            "max_msa_depth": "2048",
            "n_queries": "16",
        }
    },
    "paths-as-text": {"options": {"components_cif": "/tmp/components-as-text.cif"}},
    "include-raw": {"options": {"include_raw": True}},
    "include-raw-off": {"options": {"include_raw": False}},
    "sampling-knobs": {
        "num_samples": 2,
        "num_steps": 11,
        "num_recycles": 3,
        "max_msa_depth": 512,
    },
    "execution-knobs": {
        "options": {
            "dtype": "float32",
            "attention_kernel": "xla",
            "deterministic": "on",
            "matmul_precision": "highest",
        }
    },
    "no-compile-cache": {"cache_dir": None, "use_compile_cache": False},
    "representations": {"representations": ("single", "structural_pair")},
    "representations-all": {"representations": "all"},
    "stop-after-trunk": {"representations": "all", "stop_after": "trunk"},
    "stop-after-inputs": {"representations": "all", "stop_after": "inputs"},
    # Padding never reached this parser: it arrives as a keyword on the native
    # entry point, so the configuration carries no padding field at all. Kept
    # in the matrix because it must stay that way.
    "padding-profile": {"padding": True},
    "padding-explicit": {
        "padding": PaddingConfig(
            tokens=256, atoms=6144, msa=1280, structural_tokens=512
        )
    },
}

#: Every boolean spelling each port accepts, and what it means. Protenix'
#: switches are rendered as bare flags out of a wide vocabulary; OpenDDE spells
#: its own as `--flag true|false` and takes only those two words.
_PROTENIX_BOOLEANS = (
    (True, True),
    (False, False),
    ("true", True),
    ("false", False),
    ("yes", True),
    ("no", False),
    ("on", True),
    ("off", False),
    ("1", True),
    ("0", False),
    ("", False),
    ("TRUE", True),
    (" false ", False),
)
_OPENDDE_BOOLEANS = (
    (True, True),
    (False, False),
    ("true", True),
    ("false", False),
    ("TRUE", True),
    (" false ", False),
)


@pytest.mark.parametrize("label", sorted(_PROTENIX_REQUESTS))
def test_protenix_builds_the_configuration_its_argv_would_parse_to(
    tmp_path: Path, label: str
) -> None:
    request = _request(tmp_path, PROTENIX, **_PROTENIX_REQUESTS[label])
    _assert_equivalent(PROTENIX, request)


@pytest.mark.parametrize("label", sorted(_OPENDDE_REQUESTS))
def test_opendde_builds_the_configuration_its_argv_would_parse_to(
    tmp_path: Path, label: str
) -> None:
    request = _request(tmp_path, OPENDDE, **_OPENDDE_REQUESTS[label])
    _assert_equivalent(OPENDDE, request)


@pytest.mark.parametrize("option", sorted(_PROTENIX_OPTION_VALUES))
def test_each_protenix_option_alone_reaches_the_same_configuration(
    tmp_path: Path, option: str
) -> None:
    """One option at a time, so a value is never carried by another's presence."""
    value = _PROTENIX_OPTION_VALUES[option]
    _assert_equivalent(
        PROTENIX, _request(tmp_path, PROTENIX, options={option: value})
    )


@pytest.mark.parametrize("option", sorted(_OPENDDE_OPTION_VALUES))
def test_each_opendde_option_alone_reaches_the_same_configuration(
    tmp_path: Path, option: str
) -> None:
    value = _OPENDDE_OPTION_VALUES[option]
    _assert_equivalent(OPENDDE, _request(tmp_path, OPENDDE, options={option: value}))


@pytest.mark.parametrize(("spelling", "meaning"), _PROTENIX_BOOLEANS)
@pytest.mark.parametrize("option", ("cp_atom_windows", "strict_token_limit"))
def test_every_protenix_switch_spelling_means_the_same_in_both_paths(
    tmp_path: Path, option: str, spelling: Any, meaning: bool
) -> None:
    """Both the positive and the negated switch, over the whole vocabulary.

    ``cp_atom_windows`` ships on, so it is the negative flag that has to be
    rendered, and the two spellings answer a falsey value in opposite
    directions -- an omitted flag selects the parser's ``True``. This is the
    pair that has to agree.
    """
    request = _request(tmp_path, PROTENIX, options={option: spelling})
    invocation = ProtenixBackend()._native_invocation(request)
    assert invocation.config_fields[option] is meaning
    _assert_equivalent(PROTENIX, request)


@pytest.mark.parametrize(("spelling", "meaning"), _OPENDDE_BOOLEANS)
@pytest.mark.parametrize("option", ("cp_atom_windows", "use_template", "use_rna_msa"))
def test_every_opendde_switch_spelling_means_the_same_in_both_paths(
    tmp_path: Path, option: str, spelling: Any, meaning: bool
) -> None:
    request = _request(tmp_path, OPENDDE, options={option: spelling})
    invocation = OpenDDEBackend()._native_invocation(request)
    assert invocation.config_fields[option] is meaning
    _assert_equivalent(OPENDDE, request)


@pytest.mark.parametrize("port", PORTS, ids=lambda port: port.name)
def test_the_defaults_table_is_the_parsers_own_namespace(port: _Port) -> None:
    """An omitted option means what the parser says it means.

    The bare request above already compares the whole table through one
    configuration; this names the table, so a drifted default is reported as
    the default it is rather than as one field of a run.
    """
    parser = capture_parser(port.main)
    defaults = {
        action.dest: action.default
        for action in parser._actions
        if action.dest not in {"help", argparse.SUPPRESS}
    }
    defaults["max_msa_depth"] = port.runner._resolve_msa_depth(
        defaults["max_msa_depth"]
    )
    assert _typed(port.module._PARSER_DEFAULTS) == _typed(defaults)
    assert set(port.module._PARSER_DEFAULTS) == set(
        port.runner.PredictionConfig._fields
    )


@pytest.mark.parametrize("port", PORTS, ids=lambda port: port.name)
def test_every_rendered_option_has_a_spec_carrying_the_parsers_vocabulary(
    port: _Port,
) -> None:
    """The specs are the parser's own ``type=`` and ``choices=``, exhaustively.

    Exhaustive in both directions: an option this adapter renders and has no
    spec for would raise `KeyError` at prediction time, and a spec for an
    option it cannot render is a vocabulary nothing checks.
    """
    parser = capture_parser(port.main)
    actions = {action.dest: action for action in parser._actions}
    assert set(port.module._OPTION_SPECS) == set(port.module._CLI_OPTIONS)
    for option, (_coerce, choices) in sorted(port.module._OPTION_SPECS.items()):
        action = actions[option]
        expected = tuple(action.choices) if action.choices is not None else None
        assert choices == expected, option


@pytest.mark.parametrize("port", PORTS, ids=lambda port: port.name)
def test_a_value_outside_the_parsers_vocabulary_is_refused(
    tmp_path: Path, port: _Port
) -> None:
    """The rejection argparse used to make, at the point it used to make it.

    Without this the configuration path is strictly more permissive than the
    command line it replaced: ``--trunk-dtype fp64`` was a parser error, while
    a configuration carrying it runs -- `_load_prepared_params` tests for
    ``bf16`` and loads the FP32 checkpoint for everything else, so the run
    would have used the other precision under a name no port offers.
    """
    checked = 0
    for option, (_coerce, choices) in sorted(port.module._OPTION_SPECS.items()):
        if choices is None:
            continue
        checked += 1
        request = _request(tmp_path, port, options={option: "not-a-choice"})
        with pytest.raises(ValueError, match=option):
            port.backend()._native_invocation(request)
    assert checked > 0


@pytest.mark.parametrize("port", PORTS, ids=lambda port: port.name)
def test_a_malformed_number_is_refused_by_name(tmp_path: Path, port: _Port) -> None:
    """argparse answered these with its whole usage dump; this names the option."""
    request = _request(tmp_path, port, options={"num_samples": "several"})
    with pytest.raises(ValueError, match="num_samples must be an integer"):
        port.backend()._native_invocation(request)


def _double(port: _Port, calls: list[tuple[str, Any]]) -> SimpleNamespace:
    """A stand-in for the imported parser and runner that records the call.

    Both entry points, and the private weight-session ABI beside them: the
    adapters read the loader sentinel off the module they import, and stage 2
    does not retire that.
    """

    def main(argv: Any, **_keywords: Any) -> list[Path]:
        calls.append(("main", tuple(argv)))
        return []

    def run_prediction(config: Any, **_keywords: Any) -> list[Path]:
        calls.append(("run_prediction", config))
        return []

    return SimpleNamespace(
        PREPARED_PARAMS_LOADER_API=True,
        _load_prepared_params=lambda path, dtype: None,
        main=main,
        run_prediction=run_prediction,
        PredictionConfig=port.runner.PredictionConfig,
    )


@pytest.mark.parametrize("port", PORTS, ids=lambda port: port.name)
def test_an_ordinary_request_runs_the_runner_rather_than_the_parser(
    tmp_path: Path, port: _Port, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tripwire for the whole file: equality proves nothing about the call."""
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(
        f"foldjax.backends.{port.name}.import_module",
        lambda _name: _double(port, calls),
    )
    request = _request(tmp_path, port, options={"num_samples": 2})
    result = port.backend().predict(request)

    assert [name for name, _payload in calls] == ["run_prediction"]
    config = calls[0][1]
    assert isinstance(config, port.runner.PredictionConfig)
    assert config.num_samples == 2
    # The command is still rendered, and is still what the result records.
    assert result.raw["argv"] == tuple(
        port.backend()._native_invocation(request).argv
    )


def test_protenix_extra_native_argv_keeps_going_through_the_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``cli_args`` is arbitrary native argv, so the parser is its only reader.

    A configuration built here cannot carry ``--gamma0 0.2``: the adapter owns
    no field for it and would have dropped the override silently. So a
    non-empty ``cli_args`` renders the command and runs ``main``, and the
    parser applies the override on top of exactly the configuration this
    adapter would have built.
    """
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(
        "foldjax.backends.protenix.import_module",
        lambda _name: _double(PROTENIX, calls),
    )
    request = _request(
        tmp_path,
        PROTENIX,
        options={"num_samples": 2, "cli_args": ("--gamma0", "0.2")},
    )
    ProtenixBackend().predict(request)

    invocation = ProtenixBackend()._native_invocation(request)
    assert calls == [("main", tuple(invocation.argv))]
    assert invocation.argv[-2:] == ["--gamma0", "0.2"]

    # The override is not dropped: the parser's configuration differs from the
    # one built here in that field, and in nothing else.
    parsed = _parser_namespace(PROTENIX, invocation.argv)
    built = invocation.config_fields
    differences = {
        name: (built[name], parsed[name])
        for name in parsed
        if _typed(built)[name] != _typed(parsed)[name]
    }
    assert differences == {"gamma0": (None, 0.2)}
    assert built["num_samples"] == 2


def test_protenix_without_extra_argv_needs_no_parser_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty ``cli_args`` is not the escape hatch, it is the ordinary run."""
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(
        "foldjax.backends.protenix.import_module",
        lambda _name: _double(PROTENIX, calls),
    )
    request = _request(tmp_path, PROTENIX, options={"cli_args": ()})
    ProtenixBackend().predict(request)

    assert [name for name, _payload in calls] == ["run_prediction"]


@pytest.mark.parametrize("port", PORTS, ids=lambda port: port.name)
def test_a_session_still_negotiates_the_private_loader_by_keyword(
    tmp_path: Path, port: _Port, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage 2 keeps the weight-session ABI: only the entry point moved.

    The sentinel is still read off the imported parser module and the loader is
    still supplied as the private keyword -- now to ``run_prediction`` -- and
    only while a session has negotiated it.
    """
    keywords: list[dict[str, Any]] = []

    def run_prediction(config: Any, **given: Any) -> list[Path]:
        keywords.append(given)
        return []

    double = SimpleNamespace(
        PREPARED_PARAMS_LOADER_API=True,
        _load_prepared_params=lambda path, dtype: None,
        main=lambda argv, **given: keywords.append(given) or [],
        run_prediction=run_prediction,
        PredictionConfig=port.runner.PredictionConfig,
    )
    monkeypatch.setattr(
        f"foldjax.backends.{port.name}.import_module", lambda _name: double
    )
    first = _request(tmp_path, port)
    second = dataclasses.replace(first, seed=6, output_dir=tmp_path / "out-second")
    backend = port.backend()

    backend.predict(first)
    assert keywords == [{}]

    with backend.session((first, second)):
        backend.predict(first)
    assert list(keywords[1]) == ["_prepared_params_loader"]
