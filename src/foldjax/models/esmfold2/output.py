"""Writing ESMFold2's samples and confidences to disk.

One file per diffusion sample plus one confidence JSON, which is the shape the
other FoldJAX backends produce and what `foldjax.output.normalize` expects to
tidy afterwards.

The confidence names are upstream's, with one clarification carried into the
file: `plddt` is on the model's own 0-1 scale here, while the b-factor column
of the structures is on the 0-100 scale viewers assume. Reporting one number
under one name on two scales in two places is how a confidence gets misread,
so the JSON says which it is.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from foldjax.models._output_validation import require_finite_coordinates
from foldjax.models.esmfold2.data import pdb

#: Scalars the confidence head returns once per sample.
SAMPLE_SCORES = ("complex_plddt", "complex_iplddt", "ptm", "iptm")

#: Exact union read by :func:`crop_prediction` and the two structure-writer
#: branches.  Managed inference projects its generated feature dictionary to
#: this shallow view after the final serving padding.  Direct writer callers
#: keep their mapping unchanged.
ESMFOLD2_GENERATED_OUTPUT_FEATURE_FIELDS = frozenset(
    {
        "token_attention_mask",
        "atom_attention_mask",
        "atom_to_token",
        "residue_index",
        "res_type",
        "asym_id",
        "ref_atom_name_chars",
        "ref_element",
        "mol_type",
        "token_chain_id_chars",
        "token_residue_name_chars",
    }
)

_BASE_OUTPUT_FEATURE_FIELDS = ESMFOLD2_GENERATED_OUTPUT_FEATURE_FIELDS - {
    "token_chain_id_chars",
    "token_residue_name_chars",
}
_ALL_BIOMOLECULE_OUTPUT_FEATURE_FIELDS = frozenset(
    {"token_chain_id_chars", "token_residue_name_chars"}
)


def project_generated_output_features(
    features: Mapping[str, object],
) -> Mapping[str, object]:
    """Return the bounded writer view of a generated managed feature tree.

    This is deliberately a proof, not a best-effort filter.  FoldJAX's two
    ESMFold2 featurizers produce one batched job, with explicit token and atom
    axes.  If a wrapper supplies an incomplete or differently shaped mapping,
    return that exact object so the historical writer branch and its errors
    remain authoritative.
    """

    try:
        if type(features) is not dict:
            return features
        if not _BASE_OUTPUT_FEATURE_FIELDS <= features.keys():
            return features

        def shape(name: str) -> tuple[int, ...]:
            value_shape = getattr(features[name], "shape", None)
            if value_shape is None:
                raise AttributeError(name)
            return tuple(value_shape)

        token_shape = shape("token_attention_mask")
        atom_shape = shape("atom_attention_mask")
        if (
            len(token_shape) != 2
            or token_shape[0] != 1
            or token_shape[1] < 1
            or len(atom_shape) != 2
            or atom_shape[0] != 1
            or atom_shape[1] < 1
        ):
            return features
        token_fields = ("residue_index", "res_type", "asym_id", "mol_type")
        atom_fields = ("atom_to_token", "ref_element")
        if any(
            shape(name) != token_shape for name in token_fields
        ) or any(shape(name) != atom_shape for name in atom_fields):
            return features
        if shape("ref_atom_name_chars") != (*atom_shape, 4):
            return features

        metadata = _ALL_BIOMOLECULE_OUTPUT_FEATURE_FIELDS & features.keys()
        if metadata and metadata != _ALL_BIOMOLECULE_OUTPUT_FEATURE_FIELDS:
            return features
        if metadata and any(
            len(shape(name)) != 3
            or shape(name)[:2] != token_shape
            or shape(name)[-1] < 1
            for name in metadata
        ):
            return features
    except (AttributeError, KeyError, TypeError, ValueError):
        return features

    return {
        name: features[name]
        for name in ESMFOLD2_GENERATED_OUTPUT_FEATURE_FIELDS
        if name in features
    }


def _numpy(value: object) -> np.ndarray:
    return np.asarray(value)


def sample_scores(
    output: Mapping[str, object], *, token_mask: object | None = None
) -> list[dict[str, float]]:
    """Per-sample confidences, in the order the sampler produced them."""
    lengths = [
        _numpy(output[name]).shape[0] for name in SAMPLE_SCORES if name in output
    ]
    n_samples = lengths[0] if lengths else 1
    scores: list[dict[str, float]] = []
    for index in range(n_samples):
        entry: dict[str, float] = {"sample": index}
        for name in SAMPLE_SCORES:
            if name in output:
                entry[name] = float(_numpy(output[name])[index])
        if "plddt" in output:
            # The per-token mean, masked to real tokens, which is the number
            # people quote as "the pLDDT". It is also what `foldjax.output`
            # orders ESMFold2's samples by -- upstream writes one structure and
            # so publishes no ranking, and this scalar is the only thing the
            # model says about a sample as a whole. That ordering is FoldJAX's
            # choice, recorded here because this is where the number is made.
            plddt = _numpy(output["plddt"])[index]
            if token_mask is not None:
                mask = _numpy(token_mask).astype(bool).reshape(-1)
                if plddt.shape[-1] != mask.size:
                    raise ValueError(
                        "ESMFold2 token mask does not match pLDDT width: "
                        f"{mask.size} versus {plddt.shape[-1]}"
                    )
                plddt = plddt[mask]
            entry["plddt"] = float(plddt.mean()) if plddt.size else 0.0
        scores.append(entry)
    return scores


def crop_prediction(
    output: Mapping[str, object], features: Mapping[str, np.ndarray]
) -> dict[str, object]:
    """Remove masked token/atom suffixes from every public prediction array."""

    token_mask = np.asarray(features["token_attention_mask"]).reshape(-1).astype(bool)
    atom_mask = np.asarray(features["atom_attention_mask"]).reshape(-1).astype(bool)
    n_token = int(token_mask.sum())
    n_atom = int(atom_mask.sum())
    if not np.array_equal(token_mask, np.arange(token_mask.size) < n_token):
        raise ValueError("ESMFold2 token padding must be a suffix")
    if not np.array_equal(atom_mask, np.arange(atom_mask.size) < n_atom):
        raise ValueError("ESMFold2 atom padding must be a suffix")

    token_arrays = {"plddt", "plddt_ca", "residue_index", "entity_id"}
    pair_logit_arrays = {
        "distogram_logits",
        "pae_logits",
        "pde_logits",
    }
    pair_scalar_arrays = {"pae", "pde"}
    atom_arrays = {"atom_pad_mask", "plddt_per_atom"}
    atom_channel_arrays = {
        "sample_atom_coords",
        "plddt_logits",
        "resolved_logits",
    }

    cropped: dict[str, object] = {}
    for name, value in output.items():
        array = value
        if name in token_arrays:
            array = value[..., :n_token]
        elif name in pair_logit_arrays:
            array = value[..., :n_token, :n_token, :]
        elif name in pair_scalar_arrays:
            array = value[..., :n_token, :n_token]
        elif name in atom_arrays:
            array = value[..., :n_atom]
        elif name in atom_channel_arrays:
            array = value[..., :n_atom, :]
        cropped[name] = array
    return cropped


def _confidence_index(
    features: Mapping[str, np.ndarray], *, n_atom: int
) -> dict[str, np.ndarray]:
    """Token and atom index maps in the order the structure writer emits atoms.

    Chain names are the ones the mmCIF carries: the all-biomolecule features'
    own chain ids, else the PDB alphabet the legacy writer uses. Residue
    numbers are the writer's `residue_index + 1`.
    """
    take = lambda name: pdb._drop_batch(np.asarray(features[name]))  # noqa: E731
    token_mask = take("token_attention_mask").astype(bool)
    n_token = int(token_mask.sum())
    if "token_chain_id_chars" in features:
        chains = [
            pdb._decode_text(row) for row in take("token_chain_id_chars")[:n_token]
        ]
    else:
        chains = [
            pdb.CHAIN_ALPHABET[int(asym)]
            if int(asym) < len(pdb.CHAIN_ALPHABET)
            else str(int(asym))
            for asym in take("asym_id")[:n_token]
        ]
    # The chain axis of `pair_chains_iptm` is `one_hot(asym_id, max + 1)`; each
    # entry is named by its first token's chain.
    asym = take("asym_id")[:n_token].astype(np.int64)
    names = {}
    for value, name in zip(asym.tolist(), chains, strict=True):
        names.setdefault(value, name)
    n_chain = int(asym.max()) + 1 if asym.size else 0
    return {
        "token_chain_id": np.asarray(chains),
        "token_residue_index": take("residue_index")[:n_token].astype(np.int32) + 1,
        "atom_token_index": take("atom_to_token")[:n_atom].astype(np.int32),
        "chain_id": np.asarray(
            [names.get(index, str(index)) for index in range(n_chain)]
        ),
    }


def _write_confidence_arrays(
    structure_path: Path,
    cropped: Mapping[str, object],
    index: int,
    n_samples: int,
    index_maps: Mapping[str, np.ndarray],
) -> None:
    """Stage one sample's `confidence_full.npz` beside its structure.

    pLDDT stays on the head's 0-1 scale. `pae`/`pde` are present unless the
    program was compiled with both `return_confidence_logits` and
    `return_expected_errors` off; `pair_chains_iptm` unless a direct caller
    projected auxiliary outputs away.
    """
    from foldjax import confidence_arrays

    def per_sample(name: str, ndim: int) -> np.ndarray | None:
        if name not in cropped:
            return None
        array = _numpy(cropped[name])
        if array.ndim == ndim + 1 and array.shape[0] == n_samples:
            return array[index]
        return array if array.ndim == ndim else None

    chain_pair = per_sample("pair_chains_iptm", 2)
    maps = dict(index_maps)
    if chain_pair is not None and chain_pair.shape[0] != len(maps.get("chain_id", ())):
        chain_pair = None
    unavailable = dict(confidence_arrays.AVAILABILITY["esmfold2"]["unavailable"])
    for name, reason in (
        ("pae", _WITHHELD_ERRORS),
        ("pde", _WITHHELD_ERRORS),
        ("chain_pair_iptm", _WITHHELD_CHAIN_PAIR),
    ):
        if (chain_pair if name == "chain_pair_iptm" else cropped.get(name)) is None:
            unavailable[name] = reason
    confidence_arrays.write(
        confidence_arrays.staged_path(structure_path),
        model="esmfold2",
        arrays={
            "token_plddt": per_sample("plddt", 1),
            "atom_plddt": per_sample("plddt_per_atom", 1),
            "pae": per_sample("pae", 2),
            "pde": per_sample("pde", 2),
            "chain_pair_iptm": chain_pair,
            **maps,
        },
        scales={"token_plddt": "0-1", "atom_plddt": "0-1"},
        sources={
            "token_plddt": "plddt",
            "atom_plddt": "plddt_per_atom",
            "pae": "pae",
            "pde": "pde",
            "chain_pair_iptm": "pair_chains_iptm",
        },
        unavailable=unavailable,
        sample={"sample": index},
    )


_WITHHELD_ERRORS = (
    "the program was compiled with return_confidence_logits and "
    "return_expected_errors both off (--option return_expected_errors=false)"
)
_WITHHELD_CHAIN_PAIR = (
    "the program projected pair_chains_iptm out (return_auxiliary_outputs=False)"
)


def write_prediction_outputs(
    output: Mapping[str, object],
    features: Mapping[str, np.ndarray],
    output_dir: str | Path,
    *,
    name: str,
    plddt_scale: float = 100.0,
) -> dict[str, object]:
    """Write one PDB per sample and one confidence JSON beside them."""
    cropped = crop_prediction(output, features)
    coords = require_finite_coordinates(cropped["sample_atom_coords"], model="ESMFold2")
    if coords.ndim == 2:
        coords = coords[None]
    per_atom = (
        _numpy(cropped["plddt_per_atom"])
        if "plddt_per_atom" in cropped
        else None
    )

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    structures: list[Path] = []
    for index in range(coords.shape[0]):
        # mmCIF, because `foldjax.output.normalize` names every structure
        # `.cif` and edits its data block. A PDB file under that name would be
        # a file whose extension lies about its contents.
        path = directory / f"{name}_sample_{index}.cif"
        path.write_text(
            pdb.to_mmcif(
                coords[index],
                features,
                None if per_atom is None else per_atom[index],
                plddt_scale=plddt_scale,
                name=f"{name}_sample_{index}",
            ),
            encoding="utf-8",
        )
        structures.append(path)

    index_maps = _confidence_index(features, n_atom=coords.shape[1])
    # Each array crosses to the host once, not once per sample.
    confidence_source = {
        name: _numpy(cropped[name])
        for name in ("plddt", "plddt_per_atom", "pae", "pde", "pair_chains_iptm")
        if name in cropped
    }
    for index, path in enumerate(structures):
        _write_confidence_arrays(
            path, confidence_source, index, coords.shape[0], index_maps
        )

    scores = sample_scores(cropped)
    summary = {
        "model": "esmfold2",
        "plddt_scale": "0-1 here; the structures' b-factor column is 0-100",
        "samples": scores,
    }
    scores_path = directory / f"{name}_confidence.json"
    scores_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"structures": structures, "scores": scores_path, "summary": scores}


__all__ = [
    "ESMFOLD2_GENERATED_OUTPUT_FEATURE_FIELDS",
    "SAMPLE_SCORES",
    "crop_prediction",
    "project_generated_output_features",
    "sample_scores",
    "write_prediction_outputs",
]
