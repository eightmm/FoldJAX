"""Every directory FoldJAX writes into stays under the root it was given.

Each generated location is checked twice: a planted symlink is refused before
anything is created, and after creation the resolved path must still be under
the root. The first check is reachable by planting a link. The second guards
the window between the two -- a link swapped in after the first look -- and is
reached here by hiding the planted link from the first check
(`_first_look_misses`), which is what that race looks like from the code.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from foldjax.schema import PredictionOutputError


def _first_look_misses(monkeypatch, link: Path, *, looks: int | None = None) -> None:
    """``link.is_symlink()`` answers False, as it would before a swap.

    ``looks`` limits that to the first so many questions; after them the link
    is seen, as it is by a check made after the swap.
    """
    original = Path.is_symlink
    missed = 0

    def is_symlink(self: Path) -> bool:
        nonlocal missed
        if self == link and (looks is None or missed < looks):
            missed += 1
            return False
        return original(self)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    directory = tmp_path / "outside"
    directory.mkdir()
    return directory


# -- run manifests and output directories (foldjax.api) -----------------------


def test_a_manifest_artifact_path_must_be_relative(tmp_path: Path) -> None:
    from foldjax.api import _resolve_artifact_path

    artifact = tmp_path / "sample.cif"
    artifact.write_text("data_x\n")
    with pytest.raises(PredictionOutputError, match="must be relative"):
        _resolve_artifact_path(tmp_path, str(artifact), allowed_root=tmp_path)


@pytest.mark.parametrize(
    ("hidden", "message"),
    [(False, "artifact path is a symlink"), (True, "escapes its output root")],
    ids=["planted", "swapped-after-the-walk"],
)
def test_a_manifest_artifact_behind_a_symlink_is_refused(
    tmp_path: Path, outside: Path, monkeypatch, hidden: bool, message: str
) -> None:
    from foldjax.api import _resolve_artifact_path

    root = tmp_path / "run"
    root.mkdir()
    (outside / "sample.cif").write_text("data_x\n")
    link = root / "seed-0_sample-00"
    link.symlink_to(outside, target_is_directory=True)
    if hidden:
        _first_look_misses(monkeypatch, link)
    with pytest.raises(PredictionOutputError, match=message):
        _resolve_artifact_path(
            root, "seed-0_sample-00/sample.cif", allowed_root=root
        )


@pytest.mark.parametrize(
    "generated",
    ["../elsewhere", "model/../../elsewhere"],
    ids=["parent", "nested-parent"],
)
def test_a_generated_output_directory_cannot_climb_out_of_its_root(
    tmp_path: Path, generated: str
) -> None:
    """`root/../x` is under `root` to a prefix test, and outside it on disk."""
    from foldjax.api import _prepare_output_directory

    root = tmp_path / "run"
    root.mkdir()
    with pytest.raises(PredictionOutputError, match="escapes its run root"):
        _prepare_output_directory(root / generated, boundary=root)
    assert not (tmp_path / "elsewhere").exists(), "refused after it was created"


def test_a_generated_output_directory_elsewhere_is_refused(tmp_path: Path) -> None:
    from foldjax.api import _prepare_output_directory

    with pytest.raises(PredictionOutputError, match="escapes its run root"):
        _prepare_output_directory(tmp_path / "b", boundary=tmp_path / "a")


@pytest.mark.parametrize(
    ("hidden", "message"),
    [(False, "is a symlink"), (True, "escapes its run root")],
    ids=["planted", "swapped-after-the-walk"],
)
def test_a_generated_output_directory_behind_a_symlink_is_refused(
    tmp_path: Path, outside: Path, monkeypatch, hidden: bool, message: str
) -> None:
    from foldjax.api import _prepare_output_directory

    root = tmp_path / "run"
    root.mkdir()
    link = root / "boltz2"
    link.symlink_to(outside, target_is_directory=True)
    if hidden:
        _first_look_misses(monkeypatch, link)
    with pytest.raises(PredictionOutputError, match=message):
        _prepare_output_directory(link / "job", boundary=root)


# -- canonical sample directories (foldjax.output) ----------------------------


def _structure(path: Path) -> Path:
    path.write_text("data_x\n_entry.id x\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("swapped", [False, True], ids=["planted", "swapped-in"])
def test_a_canonical_sample_directory_symlink_is_refused(
    tmp_path: Path, outside: Path, monkeypatch, swapped: bool
) -> None:
    """Checked before ``mkdir`` and again after it, for a link that arrives late."""
    from foldjax.output import normalize
    from foldjax.schema import PredictionResult, PredictionSample

    run = tmp_path / "run"
    run.mkdir()
    link = run / "seed-0_sample-00"
    link.symlink_to(outside, target_is_directory=True)
    if swapped:
        _first_look_misses(monkeypatch, link, looks=1)
    result = PredictionResult(
        model="protenix",
        samples=(
            PredictionSample(seed=0, structure_path=_structure(run / "native.cif")),
        ),
        output_dir=run,
    )
    with pytest.raises(PredictionOutputError, match="directory is a symlink"):
        normalize(result, job="job")
    assert not any(outside.iterdir())


def test_a_job_level_symlink_cannot_carry_samples_out_of_the_run(
    tmp_path: Path, outside: Path
) -> None:
    """Several jobs get a directory each; a planted one leads elsewhere."""
    from foldjax.output import normalize
    from foldjax.schema import PredictionResult, PredictionSample

    run = tmp_path / "run"
    run.mkdir()
    (run / "first").symlink_to(outside, target_is_directory=True)
    samples = tuple(
        PredictionSample(
            seed=0,
            structure_path=_structure(run / f"{job}.cif"),
            metadata={"job": job, "sample": 0},
        )
        for job in ("first", "second")
    )
    result = PredictionResult(model="protenix", samples=samples, output_dir=run)
    with pytest.raises(PredictionOutputError, match="escapes run root"):
        normalize(result, job="batch")
    assert not list(outside.rglob("*.cif"))


# -- generated native inputs (foldjax.input, foldjax.template_search) ---------


@pytest.fixture
def alignment(tmp_path: Path) -> Path:
    path = tmp_path / "target.a3m"
    path.write_text(">query\nMKTAYIAK\n", encoding="utf-8")
    return path


def _boltz_csv(tmp_path: Path, destination: Path, alignment: Path) -> None:
    from foldjax.input import _boltz_server_csv

    entity = {"paired_msa": str(alignment), "unpaired_msa": str(alignment)}
    _boltz_server_csv(entity, tmp_path, destination, 0)


def _openfold3_msa(tmp_path: Path, destination: Path, alignment: Path) -> None:
    from foldjax.input import _openfold3_msa

    # A stem OpenFold3 does not read, so the alignment is linked under one.
    _openfold3_msa(str(alignment), tmp_path, "unpaired_msa", destination, 0)


@pytest.mark.parametrize("write", [_boltz_csv, _openfold3_msa], ids=["boltz2", "of3"])
@pytest.mark.parametrize(
    ("hidden", "message"),
    [(False, "generated MSA directory is a symlink"), (True, "escapes output root")],
    ids=["planted", "swapped-after-the-check"],
)
def test_a_generated_msa_directory_behind_a_symlink_is_refused(
    tmp_path: Path,
    outside: Path,
    alignment: Path,
    monkeypatch,
    write,
    hidden: bool,
    message: str,
) -> None:
    destination = tmp_path / "native"
    destination.mkdir()
    link = destination / "msa"
    link.symlink_to(outside, target_is_directory=True)
    if hidden:
        _first_look_misses(monkeypatch, link)
    with pytest.raises(ValueError, match=message):
        write(tmp_path, destination, alignment)
    assert not any(outside.iterdir())


@pytest.mark.parametrize(
    ("hidden", "message"),
    [(False, "generated MSA directory is a symlink"), (True, "escapes output root")],
    ids=["planted", "swapped-after-the-check"],
)
def test_openfold3_refuses_a_symlinked_per_entity_msa_directory(
    tmp_path: Path,
    outside: Path,
    alignment: Path,
    monkeypatch,
    hidden: bool,
    message: str,
) -> None:
    destination = tmp_path / "native"
    (destination / "msa").mkdir(parents=True)
    link = destination / "msa" / "entity_0000"
    link.symlink_to(outside, target_is_directory=True)
    if hidden:
        _first_look_misses(monkeypatch, link)
    with pytest.raises(ValueError, match=message):
        _openfold3_msa(tmp_path, destination, alignment)
    assert not any(outside.iterdir())


def test_alphafold3_refuses_a_symlinked_template_directory(
    tmp_path: Path, outside: Path
) -> None:
    from foldjax.input import _alphafold3_template_path

    destination = tmp_path / "native"
    destination.mkdir()
    (destination / "templates").symlink_to(outside, target_is_directory=True)
    template = {"mmcif": str(_structure(tmp_path / "t.cif")), "chain_id": "A"}
    with pytest.raises(ValueError, match="template directory is a symlink"):
        _alphafold3_template_path(template, tmp_path, destination, 0, 0)
    assert not any(outside.iterdir())


def test_openfold3_refuses_a_symlinked_template_directory(
    tmp_path: Path, outside: Path
) -> None:
    from foldjax.input import _openfold3_templates
    from tests.test_template_search import STRUCTURES

    (tmp_path / "t.cif").write_text(STRUCTURES["2abc"])
    destination = tmp_path / "native"
    destination.mkdir()
    (destination / "templates").symlink_to(outside, target_is_directory=True)
    templates = [{"mmcif": "t.cif", "mapping": [(2, 2)], "chain_id": "X"}]
    with pytest.raises(ValueError, match="template directory is a symlink"):
        _openfold3_templates(templates, tmp_path, destination, 0)
    assert not any(outside.iterdir())


@pytest.mark.parametrize(
    ("hidden", "message"),
    [(False, "is a symlink"), (True, "escapes output root")],
    ids=["planted", "swapped-after-the-check"],
)
def test_a_searched_template_directory_behind_a_symlink_is_refused(
    tmp_path: Path, outside: Path, monkeypatch, hidden: bool, message: str
) -> None:
    from foldjax.template_search import _generated_directory

    root = tmp_path / "native"
    root.mkdir()
    link = root / "templates"
    link.symlink_to(outside, target_is_directory=True)
    if hidden:
        _first_look_misses(monkeypatch, link)
    with pytest.raises(ValueError, match=message):
        _generated_directory(root, link)
