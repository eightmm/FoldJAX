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


class _ComplexBackend(_Backend):
    """Also pairs a complex in one search, one row-aligned block per sequence."""

    def __init__(self) -> None:
        super().__init__()
        self.complex_calls: list[list[str]] = []

    def search_complex(self, sequences):
        from foldjax.search import ComplexPairPayload

        self.complex_calls.append(list(sequences))
        return ComplexPairPayload(
            tuple(
                f">{101 + i}\n{s}\n>paired_hit\n{'C' * len(s)}\n"
                for i, s in enumerate(sequences)
            )
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
    # Paired alignments are opt-in.
    assert "paired_msa_file_paths" not in chain

    # A monomer is never paired, as upstream pairs only >1 distinct sequence.
    monomer = attach_msas(
        _spec(UBIQUITIN),
        alignment_dir=tmp_path / "m",
        backend=_ComplexBackend(),
        paired=True,
    )
    assert "paired_msa_file_paths" not in monomer["queries"]["q"]["chains"][0]

    complex_backend = _ComplexBackend()
    with_paired = attach_msas(
        _spec(UBIQUITIN, SECOND),
        alignment_dir=tmp_path / "p",
        backend=complex_backend,
        paired=True,
    )
    assert complex_backend.complex_calls == [[UBIQUITIN, SECOND]]
    for chain, sequence in zip(
        with_paired["queries"]["q"]["chains"], (UBIQUITIN, SECOND), strict=True
    ):
        paired = Path(chain["paired_msa_file_paths"][0])
        assert paired.stem == PAIRED_STEM
        assert paired.read_text().splitlines()[1] == sequence


def test_paired_needs_a_backend_that_pairs_a_complex(tmp_path: Path) -> None:
    backend = _Backend()
    with pytest.raises(ValueError, match="pairs a complex in one search"):
        attach_msas(
            _spec(UBIQUITIN, SECOND),
            alignment_dir=tmp_path,
            backend=backend,
            paired=True,
        )
    assert backend.calls == []


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


def test_query_ids_never_become_alignment_path_components(tmp_path: Path) -> None:
    spec = _spec(UBIQUITIN)
    spec["queries"]["../../outside"] = spec["queries"].pop("q")

    updated = attach_msas(spec, alignment_dir=tmp_path / "align", backend=_Backend())

    (main,) = updated["queries"]["../../outside"]["chains"][0][
        "main_msa_file_paths"
    ]
    path = Path(main)
    assert path.is_relative_to(tmp_path / "align")
    assert path.parent.parent.name == "query_0000"
    assert "outside" not in path.parts


def test_selected_query_is_the_only_one_searched(tmp_path: Path) -> None:
    spec = _spec(UBIQUITIN)
    spec["queries"]["second"] = _spec(SECOND)["queries"]["q"]
    backend = _Backend()

    updated = attach_msas(
        spec,
        alignment_dir=tmp_path,
        backend=backend,
        query_id="second",
    )

    assert backend.calls == [SECOND]
    assert "main_msa_file_paths" not in updated["queries"]["q"]["chains"][0]
    (main,) = updated["queries"]["second"]["chains"][0]["main_msa_file_paths"]
    assert Path(main).parent.parent.name == "query_0001"


def test_unknown_selected_query_is_rejected_without_search(tmp_path: Path) -> None:
    backend = _Backend()
    with pytest.raises(KeyError, match="not in the specification"):
        attach_msas(
            _spec(UBIQUITIN),
            alignment_dir=tmp_path,
            backend=backend,
            query_id="missing",
        )
    assert backend.calls == []


def test_preexisting_query_directory_symlink_is_refused_before_search(
    tmp_path: Path,
) -> None:
    root = tmp_path / "align"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "query_0000").symlink_to(outside, target_is_directory=True)
    backend = _Backend()

    with pytest.raises(ValueError, match="escapes output root|is a symlink"):
        attach_msas(_spec(UBIQUITIN), alignment_dir=root, backend=backend)

    assert backend.calls == []
    assert not list(outside.iterdir())


def test_existing_alignment_symlink_is_replaced_without_touching_target(
    tmp_path: Path,
) -> None:
    root = tmp_path / "align"
    directory = root / "query_0000" / "chain_0000"
    directory.mkdir(parents=True)
    external = tmp_path / "user-owned.a3m"
    external.write_text("do not replace\n")
    generated = directory / f"{MAIN_STEM}.a3m"
    generated.symlink_to(external)

    updated = attach_msas(
        _spec(UBIQUITIN), alignment_dir=root, backend=_Backend()
    )

    (main,) = updated["queries"]["q"]["chains"][0]["main_msa_file_paths"]
    assert UBIQUITIN in Path(main).read_text()
    assert external.read_text() == "do not replace\n"


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
    assert without["msa"].shape[1] == 2
    assert with_msa["msa"].shape[1] > without["msa"].shape[1]


def test_complex_paired_rows_reach_the_features(tmp_path: Path) -> None:
    """The paired block is kept, as in upstream v0.5.0.

    The pre-v0.5.0 bundled data code dropped precomputed paired rows and returned
    fewer rows than the main-only run; that is why ``paired`` used to be
    documented as collapsing the MSA.
    """
    spec = _spec(UBIQUITIN, SECOND)
    main_only = attach_msas(spec, alignment_dir=tmp_path / "main", backend=_Backend())
    with_paired = attach_msas(
        spec,
        alignment_dir=tmp_path / "both",
        backend=_ComplexBackend(),
        paired=True,
    )
    # query + (query, main hit) without pairing; query + (query, paired hit) +
    # main hit with it.
    assert featurize_query(main_only)["msa"].shape[1] == 3
    assert featurize_query(with_paired)["msa"].shape[1] == 4
