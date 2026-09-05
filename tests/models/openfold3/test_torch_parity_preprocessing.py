"""Parity of the NumPy tensorizer against an external publisher reference."""

from __future__ import annotations

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
    tmp_path: Path, spec: dict | None = None
) -> dict[str, np.ndarray]:
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
        "queries": {
            name: {
                **query,
                "chains": [dict(chain) for chain in query["chains"]],
            }
            for name, query in native_spec["queries"].items()
        }
    }
    for query_index, query in enumerate(native_spec["queries"].values()):
        for chain_index, chain in enumerate(query["chains"]):
            if (
                str(chain["molecule_type"]).lower() in {"protein", "rna"}
                and chain.get("main_msa_file_paths") is None
            ):
                directory = tmp_path / f"query_{query_index}" / f"chain_{chain_index}"
                directory.mkdir(parents=True)
                dummy = directory / "dummy.a3m"
                dummy.write_text(f">query\n{chain['sequence']}\n")
                chain["main_msa_file_paths"] = [str(dummy)]
    query_set = InferenceQuerySet.model_validate(native_spec)
    raw = InferenceDataset(
        InferenceJobConfig(
            query_set=query_set,
            seeds=[0],
            msa=MSASettings(subsample_main=False),
            template=TemplateSettings(take_top_k=True),
            template_preprocessor_settings=TemplatePreprocessorSettings(),
        )
    )[0]
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


def _direct_template_spec(cif: Path) -> dict:
    sequence = (
        "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHL"
        "VLRLRGG"
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
        name: np.asarray(value.detach().cpu())[None, ...]
        for name, value in raw.items()
    }


def test_numpy_direct_template_geometry_matches_torch_reference(
    openfold3_source: Path,
) -> None:
    cif = (
        openfold3_source
        / "openfold3"
        / "tests"
        / "test_data"
        / "mmcifs"
        / "1ubq.cif"
    )
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
