"""Viewer-ready outputs: pLDDT in every mmCIF's B-factor, and an AFDB PAE JSON.

`foldjax.output.normalize` adds both from each sample's `confidence_full.npz`.
The B-factor check runs against one real structure per backend writer
(``tests/fixtures/outputs``), so each writer's own mmCIF dialect is read; the
per-writer unit tests in `test_confidence_arrays` additionally assert that each
writer's B-factors already equal the pLDDT it stages.
"""

from __future__ import annotations

import gzip
import inspect
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from foldjax import confidence_arrays
from foldjax.output import (
    MAX_PREDICTED_ALIGNED_ERROR,
    PAE_JSON,
    _ensure_plddt_b_factors,
    normalize,
)
from foldjax.schema import PredictionResult, PredictionSample

gemmi = pytest.importorskip("gemmi")

FIXTURES = Path(__file__).parent / "fixtures" / "outputs"
CASES = sorted(path.name for path in FIXTURES.iterdir() if path.is_dir())
MODELS = ("alphafold3", "boltz2", "esmfold2", "opendde", "openfold3", "protenix")

#: The score each fixture reports whose value is the atom-mean pLDDT, and its
#: scale. Boltz-2's `complex_plddt` is a token mean, so it agrees only roughly.
_MEAN_PLDDT = {
    "boltz2": ("complex_plddt", 100.0, 0.6),
    "esmfold2": ("complex_plddt", 100.0, 0.01),
    "opendde": ("plddt", 1.0, 0.01),
    "openfold3": ("mean_plddt", 1.0, 0.01),
    "protenix": ("plddt", 1.0, 0.01),
}


def _model(case: str) -> str:
    return json.loads((FIXTURES / case / "sample.json").read_text())["model"]


def _copy(case: str, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{case}.cif"
    with gzip.open(FIXTURES / case / "structure.cif.gz", "rb") as source:
        with target.open("wb") as handle:
            shutil.copyfileobj(source, handle)
    return target


def _site(path: Path):
    return gemmi.cif.read(str(path)).sole_block().find_mmcif_category("_atom_site.")


def _b_factors(path: Path) -> np.ndarray:
    return np.asarray([float(v) for v in _site(path).find_column("B_iso_or_equiv")])


def _archive(path: Path, plddt: np.ndarray, *, pae=None, model="openfold3") -> Path:
    arrays = {"atom_plddt": plddt}
    if pae is not None:
        arrays["pae"] = pae
    confidence_arrays.write(
        path, model=model, arrays=arrays, scales={"atom_plddt": "0-1"}
    )
    return path


def test_every_model_states_its_pae_scale() -> None:
    assert set(MAX_PREDICTED_ALIGNED_ERROR) == set(MODELS)


def test_pae_scales_match_each_ports_own_bins() -> None:
    """The last bin centre, recomputed from each port's own code."""
    import jax.numpy as jnp

    from foldjax.models.boltz2.models.heads.confidence import (
        compute_aggregated_metric,
    )
    from foldjax.models.esmfold2.models.heads import categorical_mean
    from foldjax.models.opendde import postprocess
    from foldjax.models.openfold3 import inference as openfold3
    from foldjax.models.openfold3.models.confidence import bin_centers
    from foldjax.models.protenix.models.heads import confidence as protenix

    last_bin = jnp.full((1, 64), -1e9).at[0, -1].set(0.0)
    released = openfold3.released_config(n_token=4, n_atom=4)
    protenix_bins = inspect.signature(protenix.confidence_scores_from_logits)
    # OpenDDE passes its bins as literals at the one call site.
    opendde_call = inspect.getsource(postprocess.opendde_confidence_scores)
    opendde_max = float(re.search(r"pae_max_bin=([\d.]+)", opendde_call)[1])
    opendde_bins = int(re.search(r"pae_no_bins=(\d+)", opendde_call)[1])
    observed = {
        "openfold3": bin_centers(0.0, released.pae_bin_max, released.pae_bins)[-1],
        # The bin count is read off the 64-wide logits.
        "protenix": protenix.get_bin_centers(
            0.0, protenix_bins.parameters["pae_max_bin"].default, 64
        )[-1],
        "opendde": protenix.get_bin_centers(0.0, opendde_max, opendde_bins)[-1],
        "boltz2": compute_aggregated_metric(last_bin, end=32)[0],
        "esmfold2": categorical_mean(last_bin, 0.0, 32.0)[0],
    }
    # AlphaFold 3: 63 breaks over 0..max_error_bin, centres half a step on,
    # plus a catch-all bin one step further (confidence_head.py).
    breaks = np.linspace(0.0, 31.0, 64 - 1)
    step = breaks[1] - breaks[0]
    observed["alphafold3"] = breaks[-1] + step / 2 + step
    for model, value in observed.items():
        assert float(value) == pytest.approx(MAX_PREDICTED_ALIGNED_ERROR[model]), model


@pytest.mark.parametrize("case", CASES)
def test_each_writer_puts_plddt_in_the_b_factor_column(case, tmp_path) -> None:
    cif = _copy(case, tmp_path)
    b_factors = _b_factors(cif)
    model = _model(case)
    assert np.all((b_factors >= 0.0) & (b_factors <= 100.0))
    if model in _MEAN_PLDDT:
        name, factor, tolerance = _MEAN_PLDDT[model]
        score = json.loads((FIXTURES / case / "sample.json").read_text())["scores"]
        assert b_factors.mean() == pytest.approx(score[name] * factor, abs=tolerance)

    # The archive the writer stages holds the same values: nothing is rewritten.
    before = cif.read_bytes()
    archive = _archive(tmp_path / "a.npz", b_factors / 100.0, model=model)
    loaded = confidence_arrays.load_confidence_arrays(archive)
    assert _ensure_plddt_b_factors(cif, loaded) == "native"
    assert cif.read_bytes() == before


@pytest.mark.parametrize("case", ["boltz2_e9_8reh", "alphafold3_e9_8reh"])
def test_a_wrong_column_is_filled_and_nothing_else_moves(case, tmp_path) -> None:
    cif = _copy(case, tmp_path)
    original = gemmi.cif.read(str(cif)).sole_block()
    categories = original.get_mmcif_category_names()
    site = _site(cif)
    tags = [tag for tag in site.tags if not tag.endswith("B_iso_or_equiv")]
    columns = {tag: list(original.find_values(tag)) for tag in tags}
    rng = np.random.default_rng(0)
    plddt = rng.uniform(0.0, 1.0, size=len(site))
    loaded = confidence_arrays.load_confidence_arrays(
        _archive(tmp_path / "a.npz", plddt)
    )

    assert _ensure_plddt_b_factors(cif, loaded) == "filled"
    np.testing.assert_allclose(_b_factors(cif), plddt * 100.0, atol=0.005)
    rewritten = gemmi.cif.read(str(cif)).sole_block()
    assert rewritten.get_mmcif_category_names() == categories
    for tag, values in columns.items():
        assert list(rewritten.find_values(tag)) == values, tag
    # Idempotent: the filled file is now native.
    assert _ensure_plddt_b_factors(cif, loaded) == "native"


def test_a_missing_column_is_added(tmp_path) -> None:
    cif = _copy("protenix_e9_8reh", tmp_path)
    document = gemmi.cif.read(str(cif))
    site = document.sole_block().find_mmcif_category("_atom_site.")
    site.loop.remove_column("_atom_site.B_iso_or_equiv")
    document.write_file(str(cif))
    plddt = np.linspace(0.2, 0.9, len(site))
    loaded = confidence_arrays.load_confidence_arrays(
        _archive(tmp_path / "a.npz", plddt)
    )
    assert _ensure_plddt_b_factors(cif, loaded) == "filled"
    np.testing.assert_allclose(_b_factors(cif), plddt * 100.0, atol=0.005)


def test_token_plddt_reaches_atoms_through_the_token_map(tmp_path) -> None:
    cif = _copy("boltz2_e9_8reh", tmp_path)
    n_atom = len(_site(cif))
    owners = np.arange(n_atom) // 8
    token_plddt = np.linspace(0.3, 0.8, owners.max() + 1)
    confidence_arrays.write(
        tmp_path / "a.npz",
        model="boltz2",
        arrays={"token_plddt": token_plddt, "atom_token_index": owners},
        scales={"token_plddt": "0-1"},
    )
    loaded = confidence_arrays.load_confidence_arrays(tmp_path / "a.npz")
    assert _ensure_plddt_b_factors(cif, loaded) == "filled"
    np.testing.assert_allclose(_b_factors(cif), token_plddt[owners] * 100, atol=0.005)


def test_a_disagreeing_atom_count_is_reported_not_guessed(tmp_path) -> None:
    cif = _copy("openfold3_e9_8reh", tmp_path)
    before = cif.read_bytes()
    loaded = confidence_arrays.load_confidence_arrays(
        _archive(tmp_path / "a.npz", np.full(3, 0.5))
    )
    assert _ensure_plddt_b_factors(cif, loaded).startswith("skipped:")
    assert cif.read_bytes() == before


def _staged_run(tmp_path: Path, case: str, *, pae: np.ndarray | None):
    native = tmp_path / "native"
    cif = _copy(case, native)
    model = _model(case)
    b_factors = _b_factors(cif)
    record = confidence_arrays.write(
        confidence_arrays.staged_path(cif),
        model=model,
        arrays={
            "atom_plddt": b_factors / 100.0,
            **({} if pae is None else {"pae": pae}),
        },
        scales={"atom_plddt": "0-1"},
    )
    result = PredictionResult(
        model=model,
        samples=(
            PredictionSample(
                seed=2, structure_path=cif, metadata={"confidence_arrays": record}
            ),
        ),
        output_dir=tmp_path,
    )
    return normalize(result, job="job")


@pytest.mark.parametrize("case", ["openfold3_e9_8reh", "esmfold2_e9_8reh"])
def test_normalize_writes_an_alphafold_db_pae_json(case, tmp_path) -> None:
    rng = np.random.default_rng(7)
    pae = rng.uniform(0.0, 31.75, size=(11, 11)).astype(np.float32)
    placed = _staged_run(tmp_path, case, pae=pae)

    directory = tmp_path / "seed-2_sample-00"
    document = json.loads((directory / PAE_JSON).read_text())
    assert isinstance(document, list) and len(document) == 1
    (entry,) = document
    assert set(entry) == {"predicted_aligned_error", "max_predicted_aligned_error"}
    assert entry["max_predicted_aligned_error"] == 31.75
    matrix = np.asarray(entry["predicted_aligned_error"])
    assert matrix.shape == (11, 11)
    stored = confidence_arrays.load_confidence_arrays(directory)["pae"]
    np.testing.assert_allclose(matrix, stored.astype(np.float64), atol=0.005)
    assert np.array_equal(matrix, np.round(matrix, 2))

    exports = placed.samples[0].metadata["confidence_arrays"]["exports"]
    assert exports == {"pae_json": PAE_JSON, "plddt_b_factor": "native"}
    assert not list(directory.glob(".foldjax-pae-*"))


def test_no_pae_means_no_json(tmp_path) -> None:
    placed = _staged_run(tmp_path, "opendde_e9_8reh", pae=None)
    assert not (tmp_path / "seed-2_sample-00" / PAE_JSON).exists()
    exports = placed.samples[0].metadata["confidence_arrays"]["exports"]
    assert exports == {"pae_json": None, "plddt_b_factor": "native"}
