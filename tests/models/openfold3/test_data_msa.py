"""MSA search attached to a query specification.

The search itself lives in :mod:`foldjax.search` and is tested there. What is
tested here is the part specific to OpenFold3: that the alignments end up under
stems upstream will actually parse, and that the featurizer then sees more than
the query sequence.

A fake backend stands in for the ColabFold server; nothing here touches the network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from foldjax.models.openfold3.data import (
    MAIN_STEM,
    PAIRED_STEM,
    attach_msas,
    featurize_query,
)

# `attach_msas` searches through `foldjax.search`, which is in-package, so these
# no longer skip on a missing sibling. They still need the torch-side data stack
# to featurize what the search returns.
pytestmark = pytest.mark.torch_parity

UBIQUITIN = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)
SECOND = "GSHMASMTGGQQMGRDPNSSSVDKLAAALEHHHHHH"


class _Backend:
    """Returns the query plus one hit, and records what it was asked for."""

    name = "fake"
    version = "1"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, sequence: str, **_options):
        from foldjax.search import MsaPayload

        self.calls.append(sequence)
        hit = "A" * len(sequence)
        return MsaPayload(
            paired=f">query\n{sequence}\n>paired_hit\n{hit}\n",
            unpaired=f">query\n{sequence}\n>main_hit\n{hit}\n",
        )


def _spec(*sequences: str) -> dict:
    return {
        "queries": {
            "q": {
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": [chr(ord("A") + index)],
                        "sequence": sequence,
                    }
                    for index, sequence in enumerate(sequences)
                ]
            }
        }
    }


def test_alignments_land_under_stems_upstream_parses(tmp_path: Path) -> None:
    """The whole point of the adapter: upstream ignores any other filename."""
    backend = _Backend()
    updated = attach_msas(
        _spec(UBIQUITIN), alignment_dir=tmp_path, backend=backend
    )
    chain = updated["queries"]["q"]["chains"][0]

    main = Path(chain["main_msa_file_paths"][0])
    assert main.stem == MAIN_STEM
    assert main.is_file()
    assert UBIQUITIN in main.read_text()
    assert backend.calls == [UBIQUITIN]
    # Paired alignments are opt-in; see test_unpairable_paired_msas_collapse_the_msa.
    assert "paired_msa_file_paths" not in chain

    with_paired = attach_msas(
        _spec(UBIQUITIN), alignment_dir=tmp_path / "p", backend=backend, paired=True
    )
    paired = Path(
        with_paired["queries"]["q"]["chains"][0]["paired_msa_file_paths"][0]
    )
    assert paired.stem == PAIRED_STEM
    assert paired.is_file()


def test_the_original_spec_is_not_mutated(tmp_path: Path) -> None:
    spec = _spec(UBIQUITIN)
    attach_msas(spec, alignment_dir=tmp_path, backend=_Backend())
    assert "main_msa_file_paths" not in spec["queries"]["q"]["chains"][0]


def test_each_chain_gets_its_own_alignments(tmp_path: Path) -> None:
    """Two chains must not overwrite one another's files."""
    backend = _Backend()
    updated = attach_msas(
        _spec(UBIQUITIN, SECOND), alignment_dir=tmp_path, backend=backend
    )
    chains = updated["queries"]["q"]["chains"]
    first = Path(chains[0]["main_msa_file_paths"][0])
    second = Path(chains[1]["main_msa_file_paths"][0])
    assert first != second
    assert UBIQUITIN in first.read_text()
    assert SECOND in second.read_text()
    assert backend.calls == [UBIQUITIN, SECOND]


def test_non_protein_chains_are_left_alone(tmp_path: Path) -> None:
    spec = {
        "queries": {
            "q": {
                "chains": [
                    {
                        "molecule_type": "dna",
                        "chain_ids": ["A"],
                        "sequence": "ACGTACGT",
                    }
                ]
            }
        }
    }
    backend = _Backend()
    updated = attach_msas(spec, alignment_dir=tmp_path, backend=backend)
    assert backend.calls == []
    assert "main_msa_file_paths" not in updated["queries"]["q"]["chains"][0]


def test_the_featurizer_sees_the_alignments(
    openfold3_source: Path, tmp_path: Path
) -> None:
    """End of the chain: without this, the MSA is the query sequence alone."""
    updated = attach_msas(
        _spec(UBIQUITIN), alignment_dir=tmp_path, backend=_Backend()
    )
    with_msa = featurize_query(updated)
    without = featurize_query(_spec(UBIQUITIN))
    assert without["msa"].shape[1] == 1
    assert with_msa["msa"].shape[1] > without["msa"].shape[1]


def test_unpairable_paired_msas_collapse_the_msa(
    openfold3_source: Path, tmp_path: Path
) -> None:
    """Why ``paired`` defaults to off.

    Pairing needs taxonomy annotations in the alignment headers. An alignment that
    cannot be paired does not degrade to "unpaired only" -- upstream returns an MSA
    containing just the query sequence, with no error, which looks exactly like a
    successful run. Measured for both a monomer and a dimer.
    """
    main_only = attach_msas(
        _spec(UBIQUITIN, SECOND), alignment_dir=tmp_path / "main", backend=_Backend()
    )
    with_paired = attach_msas(
        _spec(UBIQUITIN, SECOND),
        alignment_dir=tmp_path / "both",
        backend=_Backend(),
        paired=True,
    )
    assert featurize_query(main_only)["msa"].shape[1] > 1
    assert featurize_query(with_paired)["msa"].shape[1] == 1
