"""Parity of the NumPy tensorizer against an external publisher reference."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip(
    "torch",
    reason="OpenFold3 preprocessing parity needs an external PyTorch environment",
)
# Upstream's InferenceDataset pulls in Lightning on the way to the tensorizer.
# Guarding only torch made this file *fail* rather than skip on an environment
# that had torch and not this -- which is what a torch environment provisioned
# from `--extra openfold3-preprocess` looks like.
pytest.importorskip(
    "pytorch_lightning",
    reason="upstream OpenFold3's inference dataset imports pytorch_lightning",
)

from foldjax.models.openfold3.data import MODEL_FEATURES, featurize_query

pytestmark = pytest.mark.torch_parity

_PANEL_CASES = (
    "protein_1ubq",
    "protein_ligand_5sak",
    "protein_rna_1urn",
    "rna_ligand_3gca",
    "protein_dna_7r6r",
    "protein_rna_ligand_3v7e",
    "protein_protein_7st3",
)


@pytest.mark.parametrize("case", _PANEL_CASES)
def test_independent_seven_case_panel(
    openfold3_source: Path, tmp_path: Path, monkeypatch, case: str
) -> None:
    root = os.environ.get("FOLDJAX_OPENBIND_INPUT_PANEL")
    if root is None:
        pytest.skip(
            "set FOLDJAX_OPENBIND_INPUT_PANEL to the native panel work directory"
        )
    path = Path(root) / case / "foldjax/openfold3/inputs/openfold3_input.json"
    assert path.is_file(), f"requested panel input missing: {case}"
    _assert_independent_augmentation(
        json.loads(path.read_text()), tmp_path, monkeypatch, seed=101
    )


def _spec() -> dict:
    return {
        "queries": {
            "query": {
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": ["A"],
                        "sequence": "ACDEFGHIK",
                    }
                ]
            }
        }
    }


def _torch_reference(
    tmp_path: Path,
    spec: dict | None = None,
    *,
    seed: int = 0,
    atom_arrays: list | None = None,
) -> dict[str, np.ndarray]:
    import torch
    from openfold3.core.data.framework.single_datasets.inference import (
        InferenceDataset,
    )
    from openfold3.core.data.pipelines.preprocessing.template import (
        TemplatePreprocessorSettings,
    )
    from openfold3.projects.of3_all_atom.config.dataset_configs import (
        InferenceJobConfig,
        MSASettings,
        TemplateSettings,
    )
    from openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )

    native_spec = _spec() if spec is None else spec
    native_spec = {
        **native_spec,
        "queries": {
            name: {
                **query,
                "chains": [dict(chain) for chain in query["chains"]],
            }
            for name, query in native_spec["queries"].items()
        },
    }
    for query_index, query in enumerate(native_spec["queries"].values()):
        for chain_index, chain in enumerate(query["chains"]):
            if (
                str(chain["molecule_type"]).lower() in {"protein", "rna"}
                and chain.get("main_msa_file_paths") is None
            ):
                directory = tmp_path / f"query_{query_index}" / f"chain_{chain_index}"
                directory.mkdir(parents=True)
                # The native parser ignores basenames outside max_seq_counts.
                dummy = directory / "uniref90_hits.a3m"
                dummy.write_text(f">query\n{chain['sequence']}\n")
                chain["main_msa_file_paths"] = [str(dummy)]
    query_set = InferenceQuerySet.model_validate(native_spec)
    dataset = InferenceDataset(
        InferenceJobConfig(
            query_set=query_set,
            seeds=[seed],
            msa=MSASettings(subsample_main=False),
            template=TemplateSettings(take_top_k=True),
            template_preprocessor_settings=TemplatePreprocessorSettings(),
        )
    )
    # This comparison aligns RDKit's Python stream with the candidate seed scope.
    # Native dataloader worker RNG is not generally initialized by the query seed.
    state = random.getstate()
    try:
        random.seed(seed)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            raw = dataset[0]
    finally:
        random.setstate(state)
    if atom_arrays is not None:
        atom_arrays.append(raw["atom_array"])
    return {
        name: np.asarray(value.detach().cpu())[None, ...]
        for name, value in raw.items()
        if hasattr(value, "detach")
    }


def test_numpy_preprocessing_matches_torch_reference(
    openfold3_source: Path, tmp_path: Path
) -> None:
    actual = featurize_query(_spec(), seed=0)
    expected = _torch_reference(tmp_path)

    for name in MODEL_FEATURES:
        if name == "max_atom_per_token_mask":
            continue
        assert actual[name].shape == expected[name].shape, name
        assert actual[name].dtype == expected[name].dtype, name
        if name == "ref_pos":
            # Both paths intentionally augment reference coordinates independently.
            continue
        if np.issubdtype(actual[name].dtype, np.floating):
            np.testing.assert_allclose(
                actual[name], expected[name], rtol=1e-6, atol=1e-6, err_msg=name
            )
        else:
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


@pytest.mark.parametrize("rna_ligand", [False, True], ids=["protein", "3gca"])
def test_independent_preprocessing_with_native_augmentation_draws(
    openfold3_source: Path, tmp_path: Path, monkeypatch, rna_ligand
) -> None:
    spec = _spec()
    if rna_ligand:
        spec["queries"]["query"]["chains"] = [
            {
                "molecule_type": "rna",
                "chain_ids": ["R"],
                "sequence": "CUGGGUCGCAGUAACCCCAGUUAACAAAACAAG",
            },
            {"molecule_type": "ligand", "chain_ids": ["L"], "ccd_codes": ["PQ0"]},
        ]
    _assert_independent_augmentation(spec, tmp_path, monkeypatch)


def _assert_independent_augmentation(spec, tmp_path, monkeypatch, *, seed=0):
    import torch
    from openfold3.core.data.pipelines.featurization import conformer

    from foldjax.models.openfold3.data import _numpy_featurization as numpy_features

    native_augment = conformer.centre_random_augmentation
    native_randn = torch.randn
    events = []

    def capture(positions, mask, *args, **kwargs):
        draws = []

        def randn(*shape, **options):
            value = native_randn(*shape, **options)
            draws.append(value.detach().cpu().numpy().copy())
            return value

        with monkeypatch.context() as scoped:
            scoped.setattr(torch, "randn", randn)
            result = native_augment(positions, mask, *args, **kwargs)
        assert [value.shape for value in draws] == [(4,), (3,)]
        events.append((positions.numpy().copy(), mask.numpy().copy(), draws))
        return result

    monkeypatch.setattr(conformer, "centre_random_augmentation", capture)
    atom_arrays = []
    expected = _torch_reference(tmp_path, spec, seed=seed, atom_arrays=atom_arrays)
    numpy_augment = numpy_features._augment_reference_positions
    consumed = []

    def replay(positions, mask, rng):
        index = len(consumed)
        assert index < len(events), "unexpected candidate augmentation"
        native_positions, native_mask, draws = events[index]
        np.testing.assert_array_equal(positions, native_positions)
        np.testing.assert_array_equal(mask, native_mask)

        class Draws:
            cursor = 0

            def standard_normal(self, size, dtype):
                value = draws[self.cursor]
                self.cursor += 1
                assert value.shape == (size,) and value.dtype == dtype
                return value.copy()

        tape = Draws()
        result = numpy_augment(positions, mask, tape)
        assert tape.cursor == 2
        consumed.append(index)
        return result

    monkeypatch.setattr(numpy_features, "_augment_reference_positions", replay)
    from foldjax.models.openfold3.data.featurize import featurize_query_with_metadata

    actual, metadata = featurize_query_with_metadata(spec, seed=seed)
    from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms

    # Native dataset bookkeeping is excluded by the portable feature ABI.
    # Assert the complete key difference so a new/missing feature cannot hide
    # behind the smaller MODEL_FEATURES list.
    assert expected.keys() - actual.keys() == {
        "num_paired_seqs",
        "repeated_sample",
        "seed",
        "valid_sample",
    }
    assert actual.keys() - expected.keys() == {"max_atom_per_token_mask"}
    np.testing.assert_array_equal(expected["seed"], [[seed]])
    assert expected["valid_sample"].all()
    assert not expected["repeated_sample"].any()
    native_token_mask = torch.from_numpy(expected["token_mask"])
    expected["max_atom_per_token_mask"] = broadcast_token_feat_to_atoms(
        token_mask=native_token_mask,
        num_atoms_per_token=torch.from_numpy(expected["num_atoms_per_token"]),
        token_feat=native_token_mask,
        max_num_atoms_per_token=23,
    ).numpy()
    native_atoms = atom_arrays[0]
    assert len(native_atoms) > 0
    for field, annotation in (
        ("atom_name", "atom_name"),
        ("element", "element"),
        ("residue_name", "res_name"),
        ("residue_id", "res_id"),
        ("chain_id", "chain_id"),
        ("entity_id", "entity_id"),
        ("molecule_type_id", "molecule_type_id"),
    ):
        np.testing.assert_array_equal(
            getattr(metadata, field), getattr(native_atoms, annotation), err_msg=field
        )
    from biotite.structure import BondType

    native_bonds = native_atoms.bonds.as_array()
    np.testing.assert_array_equal(metadata.bonds, native_bonds[:, :2])
    np.testing.assert_array_equal(
        metadata.bond_type, [BondType(int(code)).name for code in native_bonds[:, 2]]
    )
    assert events and len(consumed) == len(events)
    for name in actual:
        assert actual[name].shape == expected[name].shape, name
        assert actual[name].dtype == expected[name].dtype, name
        if np.issubdtype(actual[name].dtype, np.floating):
            np.testing.assert_allclose(
                actual[name], expected[name], rtol=1e-6, atol=1e-6, err_msg=name
            )
        else:
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


def _direct_template_spec(cif: Path) -> dict:
    sequence = (
        "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    )
    return {
        "queries": {
            "query": {
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": ["A"],
                        "sequence": sequence,
                        "template_cif_paths": [str(cif)],
                        "template_cif_chain_ids": ["A"],
                    }
                ]
            }
        }
    }


def _torch_template_reference(spec: dict) -> dict[str, np.ndarray]:
    from openfold3.core.data.pipelines.featurization.template import (
        featurize_template_structures_of3,
    )
    from openfold3.core.data.primitives.structure.component import (
        BiotiteCCDWrapper,
    )
    from openfold3.core.data.primitives.structure.query import (
        structure_with_ref_mols_from_query,
    )
    from openfold3.core.data.primitives.structure.template import (
        TemplateSliceCollection,
    )
    from openfold3.core.data.primitives.structure.tokenization import (
        add_token_positions,
        get_token_count,
        tokenize_atom_array,
    )
    from openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )

    from foldjax.models.openfold3.data._numpy_featurization import _template_slices

    query_set = InferenceQuerySet.model_validate(spec)
    query = next(iter(query_set.queries.values()))
    atom_array, _reference_molecules = structure_with_ref_mols_from_query(query=query)
    tokenize_atom_array(atom_array)
    add_token_positions(atom_array)
    n_tokens = get_token_count(atom_array)
    slices = _template_slices(query, atom_array, ccd=BiotiteCCDWrapper())
    raw = featurize_template_structures_of3(
        atom_array=atom_array,
        template_slice_collection=TemplateSliceCollection(template_slices=slices),
        n_templates=4,
        n_tokens=n_tokens,
        min_bin=3.25,
        max_bin=50.75,
        n_bins=39,
    )
    return {
        name: np.asarray(value.detach().cpu())[None, ...] for name, value in raw.items()
    }


def test_numpy_direct_template_geometry_matches_torch_reference(
    openfold3_source: Path,
) -> None:
    cif = openfold3_source / "openfold3" / "tests" / "test_data" / "mmcifs" / "1ubq.cif"
    if not cif.is_file():
        pytest.skip(f"no local template fixture at {cif}")

    spec = _direct_template_spec(cif)
    actual = featurize_query(spec, seed=0)
    expected = _torch_template_reference(spec)
    for name in (
        "template_restype",
        "template_pseudo_beta_mask",
        "template_backbone_frame_mask",
        "template_distogram",
        "template_unit_vector",
    ):
        assert actual[name].shape == expected[name].shape, name
        assert actual[name].dtype == expected[name].dtype, name
        if np.issubdtype(actual[name].dtype, np.floating):
            atol = 2e-6 if name == "template_unit_vector" else 1e-6
            np.testing.assert_allclose(
                actual[name], expected[name], rtol=1e-6, atol=atol, err_msg=name
            )
        else:
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
