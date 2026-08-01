from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import gemmi
import numpy as np
import pytest

import foldjax.models.chai.inference as inference
from foldjax.models.chai.bridge.conformer_io import ConformerData
from foldjax.models.chai.data.input import EntityType, Input
from foldjax.models.chai.data.structure import tokenize_inputs
from foldjax.models.chai.data.templates import (
    build_template_context_from_m8,
    materialize_server_template_hits,
)


def _write_template_cif(path: Path) -> None:
    structure = gemmi.Structure()
    structure.name = "fixture"
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    residue_specs = (
        (
            "ALA",
            (
                ("N", "N", (0.0, 0.0, 0.0)),
                ("CA", "C", (1.0, 0.0, 0.0)),
                ("C", "C", (2.0, 1.0, 0.0)),
                ("CB", "C", (1.0, 1.0, 1.0)),
            ),
        ),
        (
            "CYS",
            (
                ("N", "N", (4.0, 0.0, 0.0)),
                ("CA", "C", (5.0, 0.0, 0.0)),
                ("C", "C", (6.0, 1.0, 0.0)),
                ("CB", "C", (5.0, 1.0, 1.0)),
            ),
        ),
        (
            "GLY",
            (
                ("N", "N", (8.0, 0.0, 0.0)),
                ("CA", "C", (9.0, 0.0, 0.0)),
                ("C", "C", (10.0, 1.0, 0.0)),
            ),
        ),
    )
    for residue_index, (residue_name, atoms) in enumerate(residue_specs, start=1):
        residue = gemmi.Residue()
        residue.name = residue_name
        residue.seqid = gemmi.SeqId(residue_index, " ")
        residue.entity_type = gemmi.EntityType.Polymer
        for atom_name, element, position in atoms:
            atom = gemmi.Atom()
            atom.name = atom_name
            atom.element = gemmi.Element(element)
            atom.pos = gemmi.Position(*position)
            atom.occ = 1.0
            residue.add_atom(atom)
        chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    structure.entities[0].full_sequence = ["ALA", "CYS", "GLY"]
    structure.assign_label_seq_id()
    document = structure.make_mmcif_document()
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as output:
            output.write(document.as_string())
    else:
        document.write_file(str(path))


def test_raw_m8_and_gzipped_cif_build_exact_template_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    _write_template_cif(cif_dir / "fixture.cif.gz")
    m8 = tmp_path / "hits.m8"
    m8.write_text(
        "target\tfixture_A\t100\t3\t0\t0\t1\t3\t1\t3\t1e-20\t100\t0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "foldjax.models.chai.data.templates._align_with_kalign",
        lambda reference, query, executable: (reference, query),
    )

    loaded = build_template_context_from_m8(
        [Input("ACG", EntityType.PROTEIN.value, "target")],
        token_asym_id=np.asarray([1, 1, 1], np.int32),
        token_residue_index=np.asarray([0, 1, 2], np.int32),
        m8_path=m8,
        template_cif_dir=cif_dir,
        cache_dir=tmp_path / "cache",
    )

    context = loaded.context
    np.testing.assert_array_equal(context.template_restype, [[0, 4, 7]])
    np.testing.assert_array_equal(context.template_pseudo_beta_mask, [[1, 1, 1]])
    np.testing.assert_array_equal(context.template_backbone_frame_mask, [[1, 1, 1]])
    reference_positions = np.asarray([[1, 1, 1], [5, 1, 1], [9, 0, 0]], np.float32)
    expected_distances = np.linalg.norm(
        reference_positions[:, None] - reference_positions[None, :], axis=-1
    )
    np.testing.assert_allclose(context.template_distances[0], expected_distances)
    assert context.template_unit_vector.shape == (1, 3, 3, 3)
    assert np.isfinite(context.template_unit_vector).all()
    assert loaded.source_sha256 == loaded.provenance["source_sha256"]
    assert loaded.provenance["template_ids"] == ["fixture_A"]


def test_raw_templates_fail_if_kalign_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_template_cif(tmp_path / "fixture.cif")
    m8 = tmp_path / "hits.m8"
    m8.write_text(
        "target\tfixture_A\t100\t3\t0\t0\t1\t3\t1\t3\t1e-20\t100\t0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("shutil.which", lambda _: None)

    with pytest.raises(RuntimeError, match="kalign"):
        build_template_context_from_m8(
            [Input("ACG", EntityType.PROTEIN.value, "target")],
            token_asym_id=np.asarray([1, 1, 1], np.int32),
            token_residue_index=np.asarray([0, 1, 2], np.int32),
            m8_path=m8,
            template_cif_dir=tmp_path,
            cache_dir=tmp_path / "cache",
        )


def test_raw_templates_reject_malformed_m8_without_silent_fallback(
    tmp_path: Path,
) -> None:
    m8 = tmp_path / "bad.m8"
    m8.write_text("target\tmissing_fields\n", encoding="utf-8")

    with pytest.raises(ValueError, match="m8"):
        build_template_context_from_m8(
            [Input("A", EntityType.PROTEIN.value, "target")],
            token_asym_id=np.asarray([1], np.int32),
            token_residue_index=np.asarray([0], np.int32),
            m8_path=m8,
            template_cif_dir=tmp_path,
            cache_dir=tmp_path / "cache",
        )


def test_server_template_hits_remap_query_ids_and_verify_cached_sources(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    line = "101\tfixture_A\t100\t3\t0\t0\t1\t3\t1\t3\t1e-20\t100\t0\n"
    results = []
    for directory in (first, second):
        hits = directory / "pdb70.m8"
        hits.write_text(line, encoding="utf-8")
        provenance = directory / "provenance.json"
        provenance.write_text(
            json.dumps(
                {
                    "cache_key": directory.name,
                    "files": {
                        "pdb70.m8": {
                            "sha256": hashlib.sha256(line.encode()).hexdigest(),
                            "bytes": len(line.encode()),
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        results.append(
            {
                "templateHitsPath": str(hits),
                "provenancePath": str(provenance),
            }
        )

    combined = materialize_server_template_hits(
        [
            Input("ACG", EntityType.PROTEIN.value, "chain_a"),
            Input("ACG", EntityType.PROTEIN.value, "chain_b"),
        ],
        results,
        cache_dir=tmp_path / "combined",
    )

    assert combined.read_text().splitlines() == [
        line.replace("101\t", "chain_a\t").strip(),
        line.replace("101\t", "chain_b\t").strip(),
    ]
    manifest = json.loads((combined.parent / "provenance.json").read_text())
    assert manifest["source_cache_keys"] == ["first", "second"]
    assert len(manifest["source_sha256"]) == 2

    (first / "pdb70.m8").write_text(line.replace("fixture", "tamper"))
    with pytest.raises(ValueError, match="hash"):
        materialize_server_template_hits(
            [Input("ACG", EntityType.PROTEIN.value, "chain_a")],
            [results[0]],
            cache_dir=tmp_path / "other",
        )


def test_server_template_hits_reject_unexpected_query_id(tmp_path: Path) -> None:
    hits = tmp_path / "pdb70.m8"
    hits.write_text("102\tfixture_A\t100\t3\t0\t0\t1\t3\t1\t3\t1e-20\t100\t0\n")
    provenance = tmp_path / "provenance.json"
    raw = hits.read_bytes()
    provenance.write_text(
        json.dumps(
            {
                "cache_key": "source",
                "files": {
                    "pdb70.m8": {
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "bytes": len(raw),
                    }
                },
            }
        )
    )
    with pytest.raises(ValueError, match="query ID 101"):
        materialize_server_template_hits(
            [Input("ACG", EntityType.PROTEIN.value, "chain_a")],
            [
                {
                    "templateHitsPath": str(hits),
                    "provenancePath": str(provenance),
                }
            ],
            cache_dir=tmp_path / "combined",
        )


@pytest.mark.parametrize(
    "inputs, message",
    [
        (
            [
                Input("ACG", EntityType.PROTEIN.value, "duplicate"),
                Input("ACG", EntityType.PROTEIN.value, "duplicate"),
            ],
            "unique",
        ),
        ([Input("ACG", EntityType.PROTEIN.value, "bad\tname")], "tab|newline"),
    ],
)
def test_server_template_hits_reject_ambiguous_entity_names(
    tmp_path: Path, inputs: list[Input], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        materialize_server_template_hits(inputs, [], cache_dir=tmp_path)


def test_raw_m8_templates_are_wired_into_inference_collation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = [Input("AA", EntityType.PROTEIN.value, "target")]
    conformer = ConformerData(
        position=np.asarray([[0, 0, 0], [1, 0, 0], [2, 1, 0], [1, 1, 1]], np.float32),
        element=np.asarray([7, 6, 6, 6], np.int32),
        charge=np.zeros(4, np.int32),
        atom_names=("N", "CA", "C", "CB"),
        bonds=((0, 1), (1, 2), (1, 3)),
        symmetries=np.arange(4, dtype=np.int64)[:, None],
    )
    structure = tokenize_inputs(inputs, {"ALA": conformer})
    m8 = tmp_path / "hits.m8"
    m8.write_text("", encoding="utf-8")
    received: dict[str, object] = {}

    def build_raw(parsed_inputs: object, **kwargs: object) -> object:
        received.update(parsed_inputs=parsed_inputs, **kwargs)
        return type(
            "RawTemplates",
            (),
            {"context": inference.TemplateContext.empty(1, 2)},
        )()

    monkeypatch.setattr(inference, "build_template_context_from_m8", build_raw)
    _, values = inference._collate_inference_inputs(
        structure,
        inputs,
            inference.InferenceConfig(
                use_esm_embeddings=False,
                template_hits_path=m8,
            template_cif_directory=tmp_path / "cifs",
            template_cache_directory=tmp_path / "cache",
        ),
    )

    assert values["template_restype"].shape == (1, 4, 256)
    assert received["m8_path"] == m8
    assert received["template_cif_dir"] == tmp_path / "cifs"
    np.testing.assert_array_equal(received["token_asym_id"], [1, 1])


@pytest.mark.official_parity
def test_raw_template_features_match_official_chai(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    upstream_chai_dir: Path,
    upstream_chai_python: Path,
) -> None:
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    cif_path = cif_dir / "fixture.cif"
    _write_template_cif(cif_path)
    m8 = tmp_path / "hits.m8"
    m8.write_text(
        "target\tfixture_A\t100\t3\t0\t0\t1\t3\t1\t3\t1e-20\t100\t0\n",
        encoding="utf-8",
    )
    program = r"""
import json
import sys
from pathlib import Path

import gemmi
import torch

from chai_lab.data.dataset.structure.all_atom_residue_tokenizer import (
    AllAtomResidueTokenizer,
)
from chai_lab.data.dataset.templates.context import TemplateContext
from chai_lab.data.dataset.templates.load import LoadedTemplate
from chai_lab.data.parsing.structure.all_atom_entity_data import (
    structure_to_entities_data,
)
from chai_lab.data.parsing.templates.template_hit import TemplateHit
from chai_lab.data.sources.rdkit import RefConformerGenerator

path = Path(sys.argv[1])
structure = gemmi.read_structure(str(path))
entity = structure_to_entities_data(structure, chain_ids=["A"], make_assembly=False)[0]
context = AllAtomResidueTokenizer(RefConformerGenerator()).tokenize_entity(entity)
hit = TemplateHit(
    "target", "ACG", 0, "fixture", "A", 0, 3,
    torch.tensor([0, 4, 7], dtype=torch.int32),
    torch.zeros(3, dtype=torch.uint8), "ACG", path,
)
loaded = LoadedTemplate(torch.arange(3), hit, context)
templates = TemplateContext.from_loaded_templates(3, [loaded])
print(json.dumps({
    "restype": templates.template_restype.tolist(),
    "pseudo": templates.template_pseudo_beta_mask.tolist(),
    "backbone": templates.template_backbone_frame_mask.tolist(),
    "distances": templates.template_distances.tolist(),
    "vectors": templates.template_unit_vector.tolist(),
}))
"""
    output = subprocess.run(
        [str(upstream_chai_python), "-c", program, str(cif_path)],
        cwd=upstream_chai_dir,
        check=True,
        text=True,
        capture_output=True,
    )
    expected = json.loads(output.stdout)
    monkeypatch.setattr(
        "foldjax.models.chai.data.templates._align_with_kalign",
        lambda reference, query, executable: (reference, query),
    )

    actual = build_template_context_from_m8(
        [Input("ACG", EntityType.PROTEIN.value, "target")],
        token_asym_id=np.asarray([1, 1, 1], np.int32),
        token_residue_index=np.asarray([0, 1, 2], np.int32),
        m8_path=m8,
        template_cif_dir=cif_dir,
        cache_dir=tmp_path / "cache",
    ).context

    np.testing.assert_array_equal(actual.template_restype, expected["restype"])
    np.testing.assert_array_equal(actual.template_pseudo_beta_mask, expected["pseudo"])
    np.testing.assert_array_equal(
        actual.template_backbone_frame_mask, expected["backbone"]
    )
    np.testing.assert_allclose(actual.template_distances, expected["distances"])
    np.testing.assert_allclose(actual.template_unit_vector, expected["vectors"])
