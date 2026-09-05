"""OpenDDE keeps upstream's defaults but exposes both optional input paths.

`use_template=False` and `use_rna_msa=False` are OpenDDE's shipped inference
defaults (config/inference_defaults.py:28-29), so upstream's featurizer never
opens a `templatesPath` or an RNA `unpairedMsaPath`. This port reuses Protenix's
featurizer, which does.

That divergence is the dangerous kind: nothing fails, both runs finish, and the
structures differ only on jobs where a user supplied the very file they cared
about. The port drops the fields at those defaults and warns. Explicit opt-in
retains them, matching the corresponding upstream CLI flags.
"""

from __future__ import annotations

import pytest

from foldjax.models.opendde.data import featurize_json as fj


def _prepare(document: dict, tmp_path):
    return fj._prepare_job(document, base_dir=tmp_path)


def _chain(document: dict, index: int = 0) -> dict:
    return next(iter(document["sequences"][index].values()))


@pytest.fixture(autouse=True)
def _reset_warnings():
    fj._warned_dropped.clear()
    yield
    fj._warned_dropped.clear()


def test_protein_templates_are_dropped_with_a_warning(tmp_path) -> None:
    (tmp_path / "t.cif").write_text("data_x\n")
    document = {
        "sequences": [
            {
                "proteinChain": {
                    "sequence": "ACD",
                    "count": 1,
                    "templatesPath": "t.cif",
                }
            }
        ]
    }

    with pytest.warns(RuntimeWarning, match="use_template=False"):
        prepared = _prepare(document, tmp_path)

    assert "templatesPath" not in _chain(prepared)


def test_rna_alignment_is_dropped_with_a_warning(tmp_path) -> None:
    (tmp_path / "r.a3m").write_text(">q\nACGU\n")
    document = {
        "sequences": [
            {
                "rnaSequence": {
                    "sequence": "ACGU",
                    "count": 1,
                    "unpairedMsaPath": "r.a3m",
                }
            }
        ]
    }

    with pytest.warns(RuntimeWarning, match="use_rna_msa=False"):
        prepared = _prepare(document, tmp_path)

    assert "unpairedMsaPath" not in _chain(prepared)


@pytest.mark.parametrize(
    ("kind", "field", "filename", "options"),
    [
        ("proteinChain", "templatesPath", "t.cif", {"use_template": True}),
        ("rnaSequence", "unpairedMsaPath", "r.a3m", {"use_rna_msa": True}),
    ],
)
def test_explicit_upstream_feature_flag_keeps_the_input(
    tmp_path, kind: str, field: str, filename: str, options: dict
) -> None:
    (tmp_path / filename).write_text("feature\n")
    document = {
        "sequences": [
            {
                kind: {
                    "sequence": "ACGU" if kind == "rnaSequence" else "ACD",
                    "count": 1,
                    field: filename,
                }
            }
        ]
    }

    with NoWarning():
        prepared = fj._prepare_job(document, base_dir=tmp_path, **options)

    assert _chain(prepared)[field] == str(tmp_path / filename)


def test_a_protein_alignment_is_kept(tmp_path) -> None:
    """Only the two fields upstream's flags gate are dropped, not every path.

    An assertion that everything disappears would pass on a version that
    deleted the protein MSA too, which would be a far worse bug than the one
    this file is about.
    """
    (tmp_path / "m.a3m").write_text(">q\nACD\n")
    document = {
        "sequences": [
            {
                "proteinChain": {
                    "sequence": "ACD",
                    "count": 1,
                    "unpairedMsaPath": "m.a3m",
                }
            }
        ]
    }

    with pytest.warns(None) if False else NoWarning():
        prepared = _prepare(document, tmp_path)

    assert _chain(prepared)["unpairedMsaPath"].endswith("m.a3m")


def test_a_job_without_those_fields_warns_about_nothing(tmp_path) -> None:
    document = {"sequences": [{"proteinChain": {"sequence": "ACD", "count": 1}}]}

    with NoWarning():
        _prepare(document, tmp_path)


def test_the_warning_fires_once_per_document_not_once_per_chain(tmp_path) -> None:
    """Ten chains with templates are one problem, not ten lines of output."""
    (tmp_path / "t.cif").write_text("data_x\n")
    chain = {"proteinChain": {"sequence": "ACD", "count": 1, "templatesPath": "t.cif"}}
    document = {"sequences": [dict(chain) for _ in range(10)]}
    document["sequences"] = [
        {"proteinChain": dict(next(iter(entry.values())))}
        for entry in document["sequences"]
    ]

    with pytest.warns(RuntimeWarning) as caught:
        prepared = _prepare(document, tmp_path)

    assert len([w for w in caught if "use_template" in str(w.message)]) == 1
    assert all("templatesPath" not in _chain(prepared, i) for i in range(10))


class NoWarning:
    """`pytest.warns(None)` is an error on modern pytest; this is the intent."""

    def __enter__(self):
        import warnings

        self._context = warnings.catch_warnings(record=True)
        self._caught = self._context.__enter__()
        warnings.simplefilter("always")
        return self._caught

    def __exit__(self, *exc):
        offending = [w for w in self._caught if issubclass(w.category, RuntimeWarning)]
        self._context.__exit__(*exc)
        assert not offending, [str(w.message) for w in offending]
        return False
