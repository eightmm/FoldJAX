"""The vendored Boltz-2 data layer warns; it never writes to stdout.

Upstream printed its MSA and affinity notices with `print`/`click.echo`.
The CLI redirected stdout for the whole command, but `foldjax.api.predict`
in a notebook or pipeline got the chatter on stdout. Each notice is now a
`UserWarning`, which reaches stderr for both callers and is formatted once
by the CLI.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any

import numpy as np
import pytest

from foldjax.models.boltz2.data import const
from foldjax.models.boltz2.data.feature import featurizerv2
from foldjax.models.boltz2.data.parse.schema import parse_boltz_schema
from tests.models.boltz2.test_msa_deletions import build_tokenized


def _parse(native: dict[str, Any]) -> None:
    """Run Boltz's schema parser past the notice under test.

    With no CCD and no molecule directory the parse stops at the first
    chemistry lookup after the notice, which is not what these tests assert.
    """
    try:
        parse_boltz_schema("job", native, {}, None, boltz_2=True)
    except (KeyError, AttributeError, TypeError, FileNotFoundError, ValueError):
        pass


def test_an_explicit_empty_msa_is_a_warning_not_stdout(capsys) -> None:
    native = {
        "version": 1,
        "sequences": [{"protein": {"id": "A", "sequence": "ACDE", "msa": "empty"}}],
    }
    with pytest.warns(UserWarning, match="Found explicit empty MSA"):
        _parse(native)
    assert capsys.readouterr().out == ""


def test_a_large_affinity_ligand_is_a_warning_not_stdout(capsys) -> None:
    native = {
        "version": 1,
        "sequences": [{"ligand": {"id": "L", "smiles": "C" * 60}}],
        "properties": [{"affinity": {"binder": "L"}}],
    }
    with pytest.warns(UserWarning, match="larger than 56 heavy-atoms"):
        _parse(native)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("mismatch", ["residue", "length"])
def test_an_msa_that_does_not_match_its_chain_is_a_warning_not_stdout(
    capsys, mismatch: str
) -> None:
    data = build_tokenized()
    if mismatch == "residue":
        # Not MET/UNK, so the released code does not quietly take the input.
        data.structure.residues["res_type"][0] = const.token_ids["TRP"]
    else:
        data.structure.chains["res_num"][0] -= 1
        data.tokens = data.tokens[:-1]
    with pytest.warns(UserWarning, match="MSA does not match input sequence"):
        features = featurizerv2.process_msa_features(
            data=data,
            random=np.random.default_rng(42),
            max_seqs_batch=const.max_msa_seqs,
            max_seqs=const.max_msa_seqs,
        )
    # The chain got a dummy single-row alignment, as upstream gives it.
    msa = np.asarray(features["msa"])
    assert msa.shape[0] == 1
    assert capsys.readouterr().out == ""


def test_a_matching_msa_raises_no_warning() -> None:
    data = copy.deepcopy(build_tokenized())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        featurizerv2.process_msa_features(
            data=data,
            random=np.random.default_rng(42),
            max_seqs_batch=const.max_msa_seqs,
            max_seqs=const.max_msa_seqs,
        )
