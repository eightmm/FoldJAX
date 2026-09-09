"""Finite independent-input and returned-output adapters for Protenix closure.

The mappings below target publisher 4c355be4553512f72453ecbfb65e69f4c35d1413,
inference without labels, guidance, active constraints or serving padding.
Input checks never replace FoldJAX features with publisher values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from bench.af3_closure_capture import save, sha
from bench.native_repeat_policy import RunRecord, calibrate_native_repeats

IDENTITY_FIELDS = (
    "output_atom_chain_id",
    "output_atom_res_id",
    "output_atom_res_name",
    "output_atom_name",
    "output_atom_element",
)
COMMON_FIELDS = frozenset(
    """
asym_id atom_to_tokatom_idx atom_to_token_idx deletion_mean deletion_value
distogram_rep_atom_mask entity_id frame_atom_index has_deletion has_frame
is_ligand mol_id msa profile ref_atom_name_chars ref_charge ref_element ref_mask
ref_pos ref_space_uid residue_index restype sym_id template_aatype
template_atom_mask template_atom_positions template_backbone_frame_mask
template_distogram template_pseudo_beta_mask template_unit_vector token_bonds
token_index d_lm v_lm pad_info.mask_trunked
""".split()
) | frozenset(IDENTITY_FIELDS)
COMPACT_FIELDS = frozenset(
    """
_foldjax_compact_relp _foldjax_relp_residue_bin _foldjax_relp_token_bin
_foldjax_relp_same_entity _foldjax_relp_chain_bin
""".split()
)
PADDING_SCALARS = frozenset(
    ("pad_info.q_pad", "pad_info.k_pad_left", "pad_info.k_pad_right")
)
CONSTRAINT_FIELDS = {
    "constraint_feature.pocket": ("pocket_embedder", 1),
    "constraint_feature.contact": ("contact_embedder", 2),
    "constraint_feature.contact_atom": ("contact_atom_embedder", 2),
    "constraint_feature.substructure": ("substructure_embedder", 4),
}
# protenix/model/loss.py consumes bond_mask/resolution/plddt_m_rep_atom_mask.
# utils/permutation consumes entity_mol_id/mol_atom_index/pae_rep_atom_mask,
# but protenix.py inference supplies no labels or symmetric_permutation.
# MSA counts/deletion_matrix are preprocessing products, not MSAModule inputs;
# its exact msa/profile/deletion_value/has_deletion inputs are compared above.
NATIVE_METADATA = {
    "bond_mask": "training bond loss; inference consumes token_bonds instead",
    "resolution": "training confidence-loss eligibility",
    "plddt_m_rep_atom_mask": "training pLDDT loss representative atoms",
    "pae_rep_atom_mask": "label permutation/training loss; no inference labels",
    "entity_mol_id": "label chain permutation; disabled for inference",
    "mol_atom_index": "label chain permutation; disabled for inference",
    "modified_res_mask": "retained featurizer annotation; no inference model consumer",
    "deletion_matrix": "MSA preprocessing; derived model inputs compared exactly",
    "prot_pair_num_alignments": "MSA preprocessing row-count metadata",
    "prot_paired_num_alignments": "MSA preprocessing row-count alias",
    "prot_unpair_num_alignments": "MSA preprocessing row-count metadata",
    "prot_unpaired_num_alignments": "MSA preprocessing row-count alias",
    "rna_pair_num_alignments": "MSA preprocessing row-count metadata",
    "rna_paired_num_alignments": "MSA preprocessing row-count alias",
    "rna_unpair_num_alignments": "MSA preprocessing row-count metadata",
    "rna_unpaired_num_alignments": "MSA preprocessing row-count alias",
}
NATIVE_DERIVED = frozenset(("msa_mask", "is_protein", "is_dna", "is_rna", "relp"))
FOLDJAX_METADATA = {
    name: (
        "writer/chemistry identity; not a native neural input in guidance-off inference"
    )
    for name in """
atom_copy_id atom_entity_id atom_input_index atom_residue_index
chemical_bond_atom_indices chemical_bond_order chemical_bond_stereo ligand_stereo
output_atom_polymer_type token_ccd_code_chars token_copy_id token_entity_id
token_is_modified token_is_standard_polymer token_polymer_type token_reference_is_mse
""".split()
}
FOLDJAX_DERIVED = frozenset(
    ("token_is_ligand", "covalent_atom_indices", "covalent_token_indices")
)
FLOAT32_INTEGER_FIELDS = frozenset(
    (
        "distogram_rep_atom_mask",
        "ref_atom_name_chars",
        "ref_charge",
        "ref_element",
        "ref_mask",
    )
)


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def flat_features(features):
    result = {}

    def visit(value, key):
        if isinstance(value, Mapping):
            for name, child in value.items():
                visit(child, f"{key}.{name}" if key else str(name))
        else:
            if key in result:
                raise ValueError(f"duplicate flattened feature: {key}")
            result[key] = np.asarray(value)

    visit(features, "")
    return result


def _numeric_valid(array):
    return array.dtype.kind in "biuUS" or (
        array.dtype.kind == "f" and np.isfinite(array).all()
    )


def _dtype_mapping(name, native, candidate):
    if native.dtype == candidate.dtype:
        return "identical"
    if name in IDENTITY_FIELDS and native.dtype.kind == candidate.dtype.kind == "U":
        return "Unicode storage width only; exact text/order"
    if (
        name == "template_aatype"
        and native.dtype == np.int64
        and candidate.dtype == np.int32
    ):
        return "template categorical int64 -> int32; exact values"
    if (
        name == "template_atom_mask"
        and native.dtype == np.int64
        and candidate.dtype == np.bool_
        and np.isin(native, (0, 1)).all()
    ):
        return "binary template mask int64 -> bool"
    if name == "v_lm" and native.dtype == np.bool_ and candidate.dtype == np.float32:
        return "reference-space equality bool -> float32"
    if (
        name in FLOAT32_INTEGER_FIELDS
        and native.dtype == np.int64
        and candidate.dtype == np.float32
    ):
        if not np.array_equal(native.astype(np.float32).astype(np.int64), native):
            raise ValueError(f"integer feature loses values in FP32: {name}")
        if name != "ref_charge" and not np.isin(native, (0, 1)).all():
            raise ValueError(f"nonbinary categorical/mask feature: {name}")
        return "native integer numeric feature -> float32; exact values"
    raise ValueError(f"unmapped dtype: {name}: {native.dtype}/{candidate.dtype}")


def _identity(values):
    columns = [np.asarray(values[name]) for name in IDENTITY_FIELDS]
    if any(value.ndim != 1 or value.shape != columns[0].shape for value in columns):
        raise ValueError("atom identity columns must share one atom axis")
    keys = [
        json.dumps(list(row), separators=(",", ":"))
        for row in zip(*(value.tolist() for value in columns), strict=True)
    ]
    entities = columns[0].tolist()
    if len(keys) != len(set(keys)) or any(not str(label) for label in entities):
        raise ValueError("ambiguous or empty atom/chain identities")
    return keys, [str(label) for label in entities]


def _relative_position(native, candidate):
    n_token = native["asym_id"].shape[0]
    marker = candidate["_foldjax_compact_relp"]
    if marker.shape != () or marker.dtype != np.uint8 or marker.item() != 1:
        raise ValueError("unsupported compact relative-position marker")
    same_chain = native["asym_id"][:, None] == native["asym_id"][None, :]
    same_residue = native["residue_index"][:, None] == native["residue_index"][None, :]
    same_entity = native["entity_id"][:, None] == native["entity_id"][None, :]
    expected = {"_foldjax_relp_same_entity": same_entity.astype(np.uint8)}
    dense_blocks = []
    for field, target, radius, valid in (
        ("residue_index", "residue", 32, same_chain),
        ("token_index", "token", 32, same_chain & same_residue),
        ("sym_id", "chain", 2, same_entity),
    ):
        values = native[field]
        bins = np.where(
            valid,
            np.clip(values[:, None] - values[None, :] + radius, 0, 2 * radius),
            2 * radius + 1,
        ).astype(np.uint8)
        expected[f"_foldjax_relp_{target}_bin"] = bins
        dense_blocks.append(np.eye(2 * radius + 2, dtype=np.float32)[bins])
    dense = np.concatenate(
        (
            dense_blocks[0],
            dense_blocks[1],
            same_entity[..., None].astype(np.float32),
            dense_blocks[2],
        ),
        axis=-1,
    )
    if native["relp"].dtype != np.float32 or native["relp"].shape != (
        n_token,
        n_token,
        139,
    ):
        raise ValueError("native dense relp shape/dtype differs from publisher schema")
    if not np.array_equal(native["relp"], dense):
        raise ValueError(
            "native dense relp differs from independent categorical formula"
        )
    for name, expected_value in expected.items():
        actual = candidate[name]
        if actual.dtype != np.uint8 or not np.array_equal(actual, expected_value):
            raise ValueError(f"compact relative-position mismatch: {name}")


def check_inputs(reference: Path, features: dict) -> dict:
    """Audit independently produced FoldJAX arrays; never mutate/replay inputs."""
    reference = Path(reference)
    native = {}
    files = ("native-input.npz", "native-derived.npz", "native-identity.npz")
    for name in files:
        for key, value in arrays(reference / name).items():
            if key in native and not np.array_equal(native[key], value):
                raise ValueError(f"inconsistent captured native duplicate: {key}")
            native[key] = value
    candidate = flat_features(features)
    native_allowed = (
        COMMON_FIELDS
        | NATIVE_DERIVED
        | PADDING_SCALARS
        | set(NATIVE_METADATA)
        | set(CONSTRAINT_FIELDS)
    )
    candidate_allowed = (
        COMMON_FIELDS | COMPACT_FIELDS | set(FOLDJAX_METADATA) | FOLDJAX_DERIVED
    )
    for label, values, allowed, required in (
        (
            "native",
            native,
            native_allowed,
            COMMON_FIELDS | NATIVE_DERIVED | PADDING_SCALARS,
        ),
        (
            "foldjax",
            candidate,
            candidate_allowed,
            COMMON_FIELDS
            | COMPACT_FIELDS
            | FOLDJAX_DERIVED
            | {"output_atom_polymer_type"},
        ),
    ):
        extra, missing = set(values) - allowed, required - set(values)
        if extra or missing:
            raise ValueError(
                f"unmapped {label} input schema: extra={sorted(extra)}, "
                f"missing={sorted(missing)}"
            )
        for name, value in values.items():
            if not _numeric_valid(value):
                raise ValueError(f"nonfinite/unsupported {label} input: {name}")
    failures, leaves = [], {}
    for name in sorted(COMMON_FIELDS):
        left, right = native[name], candidate[name]
        mapping = _dtype_mapping(name, left, right)
        passed = left.shape == right.shape and np.array_equal(left, right)
        leaves[name] = {
            "passed": bool(passed),
            "native_shape": list(left.shape),
            "foldjax_shape": list(right.shape),
            "native_dtype": str(left.dtype),
            "foldjax_dtype": str(right.dtype),
            "representation": mapping,
        }
        if not passed:
            failures.append(f"shape/value:{name}")
    n_atom, n_token = len(candidate["ref_pos"]), len(candidate["asym_id"])
    if (
        native["msa_mask"].dtype != np.bool_
        or native["msa_mask"].shape != native["msa"].shape
        or not native["msa_mask"].all()
    ):
        failures.append("native MSA mask must be all-valid and match the alignment")
    blocks = (n_atom + 31) // 32
    expected_padding = {
        "pad_info.q_pad": blocks * 32 - n_atom,
        "pad_info.k_pad_left": 48,
        "pad_info.k_pad_right": (blocks - 1) * 32 + 80 - n_atom,
    }
    for name, expected in expected_padding.items():
        value = native[name]
        if (
            value.shape != ()
            or value.dtype.kind not in "iu"
            or value.item() != expected
        ):
            failures.append(f"native 32-query/128-key padding:{name}")
    if candidate["pad_info.mask_trunked"].shape != (blocks, 32, 128):
        failures.append("FoldJAX local-atom padding shape")
    _relative_position(native, candidate)
    polymers = candidate["output_atom_polymer_type"]
    polymer_types = {
        "is_protein": ("polypeptide(L)", "polypeptide(D)"),
        "is_dna": ("polydeoxyribonucleotide",),
        "is_rna": ("polyribonucleotide",),
    }
    allowed_types = {"non-polymer"} | set().union(*map(set, polymer_types.values()))
    if polymers.shape != (n_atom,) or not set(polymers.tolist()) <= allowed_types:
        raise ValueError("unmapped atom polymer type representation")
    for name, spellings in polymer_types.items():
        if not np.array_equal(native[name], np.isin(polymers, spellings)):
            failures.append(f"polymer identity:{name}")
    if not np.array_equal(native["is_ligand"], polymers == "non-polymer"):
        failures.append("polymer identity:is_ligand")
    token_ligand = np.zeros(n_token, bool)
    atom_to_token = candidate["atom_to_token_idx"]
    if (
        atom_to_token.shape != (n_atom,)
        or np.any(atom_to_token < 0)
        or np.any(atom_to_token >= n_token)
    ):
        raise ValueError("invalid atom-to-token identity")
    np.logical_or.at(token_ligand, atom_to_token, candidate["is_ligand"].astype(bool))
    if candidate["token_is_ligand"].dtype != bool or not np.array_equal(
        candidate["token_is_ligand"], token_ligand
    ):
        failures.append("token ligand identity")
    for name in ("covalent_atom_indices", "covalent_token_indices"):
        if candidate[name].shape != (0, 2) or candidate[name].dtype != np.int64:
            failures.append(f"active/unknown covalent route:{name}")
    config_path = reference / "effective-config.json"
    if not config_path.exists():
        config_path = reference / "initial-config.json"
    config = json.loads(config_path.read_text())
    if (
        config["sample_diffusion"]["guidance"]["enable"]
        or config["train_confidence_only"]
    ):
        raise ValueError(
            "native metadata mapping requires inference with guidance disabled"
        )
    for name, (embedder, channels) in CONSTRAINT_FIELDS.items():
        if config["model"]["constraint_embedder"][embedder]["enable"]:
            raise ValueError(
                "active native constraints are outside the finite input mapping"
            )
        if name in native and (
            native[name].shape != (n_token, n_token, channels)
            or np.any(native[name] != 0)
        ):
            failures.append(f"nonempty native constraint:{name}")
    if _identity(native) != _identity(candidate):
        failures.append("atom identity/order")
    return {
        "passed": not failures,
        "failures": failures,
        "leaves": leaves,
        "samples": 5,
        "atoms": n_atom,
        "tokens": n_token,
        "independent_inputs": True,
        "reference_artifacts": {name: sha(reference / name) for name in files},
        "config_sha256": sha(config_path),
        "native_metadata_not_model_inputs": {
            k: v for k, v in NATIVE_METADATA.items() if k in native
        },
        "foldjax_metadata_not_native_model_inputs": {
            k: v for k, v in FOLDJAX_METADATA.items() if k in candidate
        },
        "derived_mappings": [
            "32/128 local atom padding",
            "native dense vs independent compact139 relative position",
            "all-valid native MSA mask",
            "atom polymer and token ligand identity",
            "inactive native constraints",
        ],
        "reference_draw_tape": (
            "not independently replayed; actual resulting features checked exactly"
        ),
    }


RAW = frozenset(("plddt", "pae", "pde", "resolved", "contact_probs"))
SUMMARY = frozenset(
    """
plddt gpde ptm iptm chain_gpde chain_pair_gpde chain_ptm chain_iptm
chain_pair_iptm chain_pair_iptm_global chain_plddt chain_pair_plddt
chain_pair_pae_mean chain_pair_pae_min has_clash disorder ranking_score num_recycles
""".split()
)
FULL = frozenset(
    """
atom_plddt token_pair_pde token_pair_pae atom_coordinate contact_probs
token_has_frame token_asym_id atom_to_token_idx atom_is_polymer
""".split()
)
SUMMARY_ALIASES = {
    name: "summary_" + name
    for name in ("plddt", "gpde", "ptm", "iptm", "ranking_score")
}
FOLDJAX_ONLY_DIAGNOSTICS = frozenset(
    ("has_vdw_clash", "summary_ranking_score_vdw_penalized")
)
TRUNK_TAPS = frozenset(
    ("s_inputs", "s_trunk", "z_trunk", "single_inputs", "single", "pair")
)


def _require_keys(values, expected, name):
    if set(values) != set(expected):
        raise ValueError(
            f"unmapped {name} schema: missing={sorted(set(expected) - set(values))}, "
            f"extra={sorted(set(values) - set(expected))}"
        )


def _exact(left, right, name):
    if (
        left.shape != right.shape
        or left.dtype != right.dtype
        or left.tobytes() != right.tobytes()
    ):
        raise ValueError(f"inconsistent repeated confidence field: {name}")


def _distogram_fp32(values):
    if str(values.dtype) == "bfloat16":
        return values.astype(np.float32)
    if values.dtype != np.float32:
        raise ValueError("distogram storage must be BF16 or exact FP32 widening")
    return values


def validate_confidence(coordinates, confidence):
    if coordinates.ndim != 3 or coordinates.shape[0] != 5 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates require (5, atoms, 3) sample-index pairing")
    if coordinates.dtype != np.float32 or not np.isfinite(coordinates).all():
        raise ValueError("coordinates require finite FP32 values")
    atoms = coordinates.shape[1]
    contacts = confidence["raw.contact_probs"]
    if contacts.ndim != 2 or contacts.shape[0] != contacts.shape[1]:
        raise ValueError("contact probabilities require square token axes")
    tokens = contacts.shape[0]
    chains = confidence["summary.chain_ptm"]
    if chains.ndim != 2 or chains.shape[0] != 5:
        raise ValueError("chain confidence requires five samples")
    count = chains.shape[1]
    if min(atoms, tokens, count) <= 0:
        raise ValueError("empty atom/token/chain axis")
    shapes = {
        "raw.contact_probs": (tokens, tokens),
        "raw.distogram_logits": (tokens, tokens, 64),
        "raw.plddt": (5, atoms, 50),
        "raw.resolved": (5, atoms, 2),
        "raw.pae": (5, tokens, tokens, 64),
        "raw.pde": (5, tokens, tokens, 64),
        "full.atom_plddt": (5, atoms),
        "full.token_pair_pae": (5, tokens, tokens),
        "full.token_pair_pde": (5, tokens, tokens),
        "full.token_has_frame": (5, tokens),
        "full.token_asym_id": (5, tokens),
        "full.atom_to_token_idx": (5, atoms),
        "full.atom_is_polymer": (5, atoms),
    }
    for name in SUMMARY:
        shapes[f"summary.{name}"] = (
            ()
            if name == "num_recycles"
            else (5, count, count)
            if name.startswith("chain_pair_")
            else (5, count)
            if name.startswith("chain_")
            else (5,)
        )
    integer = frozenset(
        (
            "full.token_has_frame",
            "full.token_asym_id",
            "full.atom_to_token_idx",
            "full.atom_is_polymer",
        )
    )
    _require_keys(confidence, shapes, "canonical confidence")
    for name, shape in shapes.items():
        value = confidence[name]
        dtype = (
            np.int64
            if name in integer
            else np.int32
            if name == "summary.num_recycles"
            else np.bool_
            if name == "summary.has_clash"
            else np.float32
        )
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"invalid confidence shape/dtype: {name}")
    if confidence["summary.num_recycles"].item() != 10:
        raise ValueError("confidence adapter requires the ten-recycle native profile")


def canonical_native(raw, distogram):
    expected = set(RAW) | {"coordinate"}
    expected.update(
        f"summary_confidence.{sample}.{name}" for sample in range(5) for name in SUMMARY
    )
    expected.update(
        f"full_data.{sample}.{name}" for sample in range(5) for name in FULL
    )
    _require_keys(raw, expected, "native prediction")
    _require_keys(distogram, {"input", "logits"}, "native distogram boundary")
    confidence = {f"raw.{name}": raw[name] for name in RAW}
    confidence["raw.distogram_logits"] = _distogram_fp32(distogram["logits"])
    for name in SUMMARY:
        values = [raw[f"summary_confidence.{sample}.{name}"] for sample in range(5)]
        if name == "num_recycles":
            if any(
                v.shape != () or v.dtype != np.int64 or v.item() != 10 for v in values
            ):
                raise ValueError(
                    "native confidence requires ten recycles in every sample"
                )
            confidence[f"summary.{name}"] = np.asarray(10, np.int32)
        else:
            confidence[f"summary.{name}"] = np.stack(values)
    for name in FULL:
        values = [raw[f"full_data.{sample}.{name}"] for sample in range(5)]
        if name == "atom_coordinate":
            for sample, value in enumerate(values):
                _exact(value, raw["coordinate"][sample], name)
        elif name == "contact_probs":
            for value in values:
                _exact(value, raw["contact_probs"], name)
        else:
            confidence[f"full.{name}"] = np.stack(values)
    validate_confidence(raw["coordinate"], confidence)
    return raw["coordinate"], confidence


def canonical_foldjax(scored, features):
    features = flat_features(features)
    expected = set(RAW) | {"coordinate", "distogram_logits", "ranking_score"}
    expected.update(SUMMARY_ALIASES.get(name, name) for name in SUMMARY)
    expected.update(("atom_plddt", "token_pair_pae", "token_pair_pde"))
    extras = {
        name: scored[name]
        for name in FOLDJAX_ONLY_DIAGNOSTICS | TRUNK_TAPS
        if name in scored
    }
    _require_keys(scored, expected | set(extras), "FoldJAX prediction")
    _exact(scored["ranking_score"], scored["summary_ranking_score"], "ranking_score")
    confidence = {f"raw.{name}": scored[name] for name in RAW}
    confidence["raw.distogram_logits"] = _distogram_fp32(scored["distogram_logits"])
    confidence.update(
        {f"summary.{name}": scored[SUMMARY_ALIASES.get(name, name)] for name in SUMMARY}
    )
    for name in ("atom_plddt", "token_pair_pae", "token_pair_pde"):
        confidence[f"full.{name}"] = scored[name]
    ligand = features["is_ligand"]
    if ligand.dtype != np.int64 or not np.isin(ligand, (0, 1)).all():
        raise ValueError("native polymer mapping requires binary int64 is_ligand")
    metadata = {
        "token_has_frame": features["has_frame"],
        "token_asym_id": features["asym_id"],
        "atom_to_token_idx": features["atom_to_token_idx"],
        "atom_is_polymer": 1 - ligand,
    }
    confidence.update(
        {f"full.{name}": np.stack([value] * 5) for name, value in metadata.items()}
    )
    for name in FOLDJAX_ONLY_DIAGNOSTICS & set(extras):
        expected_dtype = np.bool_ if name == "has_vdw_clash" else np.float32
        if extras[name].shape != (5,) or extras[name].dtype != expected_dtype:
            raise ValueError(f"malformed FoldJAX-only diagnostic: {name}")
    validate_confidence(scored["coordinate"], confidence)
    return scored["coordinate"], confidence, extras


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _array_digest(value):
    digest = hashlib.sha256()
    digest.update(_digest([str(value.dtype), list(value.shape)]).encode())
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _bundle_digest(value):
    return _digest(
        {name: _array_digest(array) for name, array in sorted(value.items())}
    )


def native_record(root):
    """Return a RunRecord and its content-addressed report for root to persist.

    Calibration remains a separate operation and requires three records with
    exactly identical validated target provenance, including actual tape bytes.
    This adapter does not issue a calibrated pass or treat seed equality as tape.
    """
    root = Path(root)
    completion = json.loads((root / "capture-complete.json").read_text())
    required = {
        "passed": True,
        "samples": 5,
        "steps": 200,
        "cycles": 10,
        "sampler_calls": 1,
        "initial_draws": 1,
        "churn_draws": 200,
        "rotations": 200,
        "translations": 200,
        "msa_calls": 10,
    }
    for name, value in required.items():
        if completion.get(name) != value or type(completion.get(name)) is not type(
            value
        ):
            raise ValueError(f"incomplete/unsupported native capture: {name}")
    decisions = completion["mc_dropout_random_draws"]
    if (
        len(decisions) != 1
        or not isinstance(decisions[0], float)
        or not np.isfinite(decisions[0])
    ):
        raise ValueError("missing actual native MC-dropout decision draw")
    raw, distogram = arrays(root / "prediction.npz"), arrays(root / "distogram.npz")
    coords, confidence = canonical_native(raw, distogram)
    features = arrays(root / "native-input.npz")
    identity = arrays(root / "native-identity.npz")
    keys, entities = _identity(identity)
    if (
        len(keys) != coords.shape[1]
        or len(set(entities)) != confidence["summary.chain_ptm"].shape[1]
    ):
        raise ValueError(
            "native coordinate/confidence atom/chain identity disagreement"
        )
    tape = arrays(root / "sampler-tape.npz")
    tape_shapes = {
        "init_noise": coords.shape,
        "step_noises": (200, *coords.shape),
        "rotations": (200, 5, 3, 3),
        "translations": (200, 5, 3),
        "noise_schedule": (201,),
    }
    _require_keys(tape, tape_shapes, "native sampler tape")
    for name, shape in tape_shapes.items():
        if (
            tape[name].shape != shape
            or tape[name].dtype != np.float32
            or not np.isfinite(tape[name]).all()
        ):
            raise ValueError(f"invalid native sampler tape: {name}")
    msa = arrays(root / "msa-tape.npz")
    _require_keys(
        msa,
        {
            f"{i}.{name}"
            for i in range(10)
            for name in (
                "rows",
                "selected.msa",
                "selected.has_deletion",
                "selected.deletion_value",
            )
        },
        "native MSA tape",
    )
    for cycle in range(10):
        rows = msa[f"{cycle}.rows"]
        if (
            rows.ndim != 1
            or rows.dtype != np.int64
            or rows.size < 1
            or np.any(rows < 0)
            or np.any(rows >= len(features["msa"]))
        ):
            raise ValueError("invalid native MSA row indices")
        for name in ("msa", "has_deletion", "deletion_value"):
            _exact(msa[f"{cycle}.selected.{name}"], features[name][rows], name)
    config = json.loads((root / "effective-config.json").read_text())
    from bench.protenix_dropout_tape import load_dropout_tape

    dropout_masks, dropout_rate = load_dropout_tape(root, completion, config)
    policy = {
        name: value
        for name, value in config.items()
        if name not in {"dump_dir", "load_checkpoint_dir", "input_json_path"}
    }
    provenance = json.loads((root / "provenance.json").read_text())
    local_assets = {
        name: value["sha256"]
        for name, value in provenance["local_assets"].items()
        if name != "checkpoint"
    }
    if (
        provenance["local_assets"]["checkpoint"]["sha256"]
        != provenance["checkpoint_sha256"]
    ):
        raise ValueError("native preflight and executed checkpoint identities differ")
    input_values = {"native." + name: value for name, value in features.items()}
    input_values.update({"identity." + name: value for name, value in identity.items()})
    input_values.update(
        {
            "derived." + name: value
            for name, value in arrays(root / "native-derived.npz").items()
        }
    )
    target = {
        "input": _digest(
            {
                "features": _bundle_digest(input_values),
                "input_json": provenance["input_sha256"],
                "assets": provenance["input_assets"],
                "local_assets": local_assets,
            }
        ),
        "checkpoint": provenance["checkpoint_sha256"],
        "source": _digest(provenance["source"]),
        "runtime": _digest(
            {
                name: provenance[name]
                for name in (
                    "versions",
                    "environment_policy",
                    "snapshot_python_source",
                    "wrapper_sha256",
                    "device",
                    "cuda",
                )
            }
        ),
        "effective_policy": _digest(
            {
                "config": policy,
                "operator": json.loads((root / "operator-policy.json").read_text()),
            }
        ),
        "tape": _digest(
            {
                "sampler": _bundle_digest(tape),
                "msa": _bundle_digest(msa),
                "mc_dropout": decisions,
                **(
                    {
                        "dropout_masks": _array_digest(dropout_masks),
                        "dropout_rate": dropout_rate,
                    }
                    if dropout_masks is not None
                    else {}
                ),
            }
        ),
    }
    artifact_names = (
        "capture-complete.json",
        "provenance.json",
        "effective-config.json",
        "operator-policy.json",
        "prediction.npz",
        "distogram.npz",
        "native-input.npz",
        "native-derived.npz",
        "native-identity.npz",
        "sampler-tape.npz",
        "msa-tape.npz",
    )
    if dropout_masks is not None:
        artifact_names += ("dropout-tape.npz",)
    report = {
        "schema": 1,
        "run_id": root.name,
        "arm": "native",
        "target": target,
        "artifacts": {name: sha(root / name) for name in artifact_names},
        "confidence": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _array_digest(value),
            }
            for name, value in confidence.items()
        },
        "native_leaves_mapped": len(raw),
        "canonical_confidence_leaves": len(confidence),
        "distogram_mapping": (
            "captured native BF16 logits widened exactly to FP32 storage"
        ),
        "calibrated_pass": None,
        "scope": (
            "all native returned leaves plus raw distogram; sample-index pairing; "
            "no writer/crystal/performance admission"
        ),
    }
    record = RunRecord(
        run_id=root.name,
        report_sha256=_digest(report),
        provenance=target,
        coordinates=coords,
        atom_keys=keys,
        entity_labels=entities,
        mask=np.ones(coords.shape[:2], dtype=bool),
        confidence=confidence,
    )
    return record, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inputs = sub.add_parser("inputs")
    inputs.add_argument("reference", type=Path)
    inputs.add_argument("features", type=Path)
    inputs.add_argument("--out", type=Path, required=True)
    native = sub.add_parser("native-record")
    native.add_argument("root", type=Path)
    native.add_argument("--out", type=Path, required=True)
    calibration = sub.add_parser("calibrate")
    calibration.add_argument("roots", type=Path, nargs="+")
    calibration.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "calibrate":
        args.out.mkdir(parents=True, exist_ok=False)
        records = []
        for root in args.roots:
            record, report = native_record(root)
            records.append(record)
            save(args.out / f"{record.run_id}-adapter.json", report)
        result = calibrate_native_repeats(records)
        save(args.out / "calibration.json", result)
        print(
            json.dumps(
                {
                    "calibration_sha256": result["calibration_sha256"],
                    "reference_run_id": result["reference_run_id"],
                    "entity_limits": result["entity_limits"],
                    "confidence_leaves": len(result["confidence_reference_limits"]),
                },
                sort_keys=True,
            )
        )
        return
    if args.command == "inputs":
        result = check_inputs(args.reference, arrays(args.features))
    else:
        _, result = native_record(args.root)
    save(args.out, result)
    if result.get("passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
