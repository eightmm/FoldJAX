"""Strict native-returned OpenDDE output diagnostics; not model closure admission."""

import argparse
import json
from pathlib import Path

import numpy as np

from bench.af3_closure import compare_confidence
from bench.af3_closure_capture import save, sha
from bench.entity_parity import compare_entity_parity
from bench.precision_policy import coordinate_gate

RAW = frozenset(
    "contact_probs plddt pae pde resolved shape_comp_token_pred "
    "shape_comp_global_pred shape_comp_token_mask shape_comp_pair_mean_pred "
    "shape_comp_pair_topk_mean_pred shape_comp_valid_pair_frac_pred".split()
)
SUMMARY = frozenset(
    "plddt gpde ptm iptm chain_gpde chain_pair_gpde chain_ptm chain_iptm "
    "chain_pair_iptm chain_pair_iptm_global chain_plddt chain_pair_plddt "
    "has_clash disorder ranking_score num_recycles".split()
)
FULL = frozenset(
    "atom_plddt token_pair_pde token_pair_pae atom_coordinate contact_probs "
    "token_has_frame token_asym_id atom_to_token_idx atom_is_polymer".split()
)
SHAPE_FLAG = "shape_comp_uses_structural_tokens"
ALIASES = {
    key: f"summary_{key}" for key in ("plddt", "gpde", "ptm", "iptm", "ranking_score")
}


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def require_keys(values, expected):
    if set(values) != set(expected):
        raise ValueError(
            f"unmapped schema: missing={sorted(set(expected) - set(values))}, "
            f"extra={sorted(set(values) - set(expected))}"
        )


def exact(a, b, name):
    if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
        raise ValueError(f"inconsistent duplicate {name}")


def validate_shapes(coords, confidence):
    if coords.ndim != 3 or coords.shape[0] != 5 or coords.shape[2] != 3:
        raise ValueError("coordinate shape must be (5, atoms, 3)")
    atoms = coords.shape[1]
    contact = confidence["raw.contact_probs"]
    if contact.ndim != 2 or contact.shape[0] != contact.shape[1]:
        raise ValueError("contact probabilities must have two equal token axes")
    tokens = contact.shape[0]
    chains = confidence["summary.chain_ptm"]
    if chains.ndim != 2 or chains.shape[0] != 5:
        raise ValueError("chain confidence requires sample and chain axes")
    count = chains.shape[1]
    if (
        min(atoms, tokens, count) < 1
        or coords.dtype != np.float32
        or not np.isfinite(coords).all()
    ):
        raise ValueError("invalid coordinate dimensions/dtype/values")
    if confidence[f"metadata.{SHAPE_FLAG}"].item():
        raise ValueError(
            "structural-token shape-complementarity mapping is not audited"
        )
    expected = {
        "raw.contact_probs": (tokens, tokens),
        "raw.plddt": (5, atoms, 50),
        "raw.resolved": (5, atoms, 2),
        "raw.pae": (5, tokens, tokens, 64),
        "raw.pde": (5, tokens, tokens, 64),
        "raw.shape_comp_token_pred": (5, tokens),
        "raw.shape_comp_token_mask": (5, tokens),
        f"metadata.{SHAPE_FLAG}": (),
        "full.atom_plddt": (5, atoms),
        "full.token_pair_pae": (5, tokens, tokens),
        "full.token_pair_pde": (5, tokens, tokens),
        "full.token_has_frame": (5, tokens),
        "full.token_asym_id": (5, tokens),
        "full.atom_to_token_idx": (5, atoms),
        "full.atom_is_polymer": (5, atoms),
    }
    for key in RAW:
        expected.setdefault(f"raw.{key}", (5,))
    for key in SUMMARY:
        expected[f"summary.{key}"] = (
            ()
            if key == "num_recycles"
            else (5, count, count)
            if key.startswith("chain_pair_")
            else (5, count)
            if key.startswith("chain_")
            else (5,)
        )
    bool_fields = {
        "raw.shape_comp_token_mask",
        f"metadata.{SHAPE_FLAG}",
        "summary.has_clash",
    }
    int_fields = {
        "full.token_has_frame",
        "full.token_asym_id",
        "full.atom_to_token_idx",
        "full.atom_is_polymer",
    }
    require_keys(confidence, expected)
    for key, shape in expected.items():
        dtype = (
            np.bool_
            if key in bool_fields
            else np.int64
            if key in int_fields
            else np.int32
            if key == "summary.num_recycles"
            else np.float32
        )
        if confidence[key].shape != shape or confidence[key].dtype != dtype:
            raise ValueError(f"invalid confidence shape/dtype: {key}")


def canonical_native(raw):
    expected = set(RAW) | {"coordinate", SHAPE_FLAG}
    expected.update(
        f"summary_confidence.{i}.{key}" for i in range(5) for key in SUMMARY
    )
    expected.update(f"full_data.{i}.{key}" for i in range(5) for key in FULL)
    require_keys(raw, expected)
    result = {f"raw.{key}": raw[key] for key in RAW}
    flag = raw[SHAPE_FLAG]
    if flag.shape != (1,) or flag.dtype != np.int64 or flag.item() not in (0, 1):
        raise ValueError("unknown native structural-token flag representation")
    result[f"metadata.{SHAPE_FLAG}"] = np.asarray(bool(flag.item()))
    for key in SUMMARY:
        values = [raw[f"summary_confidence.{i}.{key}"] for i in range(5)]
        if key == "num_recycles":
            if any(
                v.shape != () or v.dtype != np.int64 or v.item() != 10 for v in values
            ):
                raise ValueError("native closure profile requires ten recycles")
            result[f"summary.{key}"] = np.asarray(10, dtype=np.int32)
        else:
            result[f"summary.{key}"] = np.stack(values)
    for key in FULL:
        values = [raw[f"full_data.{i}.{key}"] for i in range(5)]
        if key == "atom_coordinate":
            for i, value in enumerate(values):
                exact(value, raw["coordinate"][i], key)
        elif key == "contact_probs":
            for value in values:
                exact(value, raw[key], key)
        else:
            result[f"full.{key}"] = np.stack(values)
    validate_shapes(raw["coordinate"], result)
    return raw["coordinate"], result


def canonical_foldjax(scored, features):
    expected = set(RAW) | {"coordinate", SHAPE_FLAG, "distogram_logits"}
    expected.update(ALIASES.get(key, key) for key in SUMMARY)
    expected.update({"atom_plddt", "token_pair_pae", "token_pair_pde", "ranking_score"})
    require_keys(scored, expected)
    flag = scored[SHAPE_FLAG]
    if flag.shape != () or flag.dtype != np.bool_:
        raise ValueError("unknown FoldJAX structural-token flag representation")
    cycles = scored["num_recycles"]
    if cycles.shape != () or cycles.dtype != np.int32 or cycles.item() != 10:
        raise ValueError("FoldJAX closure profile requires ten recycles")
    exact(scored["ranking_score"], scored["summary_ranking_score"], "ranking_score")
    result = {f"raw.{key}": scored[key] for key in RAW}
    result[f"metadata.{SHAPE_FLAG}"] = flag
    result.update({f"summary.{key}": scored[ALIASES.get(key, key)] for key in SUMMARY})
    for key in ("atom_plddt", "token_pair_pae", "token_pair_pde"):
        result[f"full.{key}"] = scored[key]
    masks = [
        features[key] for key in ("is_protein", "is_dna", "is_rna") if key in features
    ]
    if not masks:
        raise ValueError("missing confidence polymer masks")
    polymer = np.logical_or.reduce([value.astype(bool) for value in masks])
    if not np.array_equal(polymer, 1 - features["is_ligand"]):
        raise ValueError("native and FoldJAX confidence polymer masks disagree")
    metadata = {
        "token_has_frame": features["has_frame"],
        "token_asym_id": features["asym_id"],
        "atom_to_token_idx": features["atom_to_token_idx"],
        # Native serializes int64 while FoldJAX's scoring consumer uses bool.
        "atom_is_polymer": polymer.astype(np.int64),
    }
    for key, value in metadata.items():
        result[f"full.{key}"] = np.stack([value] * 5)
    validate_shapes(scored["coordinate"], result)
    return scored["coordinate"], result


def identity(values):
    columns = [
        values[f"output_atom_{key}"].tolist()
        for key in ("chain_id", "res_id", "res_name", "name", "element")
    ]
    return list(zip(*columns, strict=True)), columns[0]


def load_arm(root):
    provenance = json.loads((root / "provenance.json").read_text())
    if not json.loads((root / "finished.json").read_text()):
        raise ValueError("unfinished capture")
    if provenance["arm"] == "native":
        coords, confidence = canonical_native(arrays(root / "raw.npz"))
        features = arrays(root / "native-identity.npz")
    elif provenance["arm"] == "foldjax":
        if (
            json.loads((root / "input-audit.json").read_text())["gate_passed"]
            is not True
        ):
            raise ValueError("independent input gate failed")
        features = arrays(root / "foldjax-input.npz")
        coords, confidence = canonical_foldjax(arrays(root / "scored.npz"), features)
    else:
        raise ValueError("unknown capture arm")
    if coords.shape[0] != 5:
        raise ValueError("five paired samples required")
    keys, entities = identity(features)
    if confidence["summary.chain_ptm"].shape[1] != len(set(entities)):
        raise ValueError("confidence chain count differs from atom entity identities")
    return coords, confidence, (keys, entities), provenance


def report(left, right):
    left, right = Path(left), Path(right)
    output = {
        "passed": False,
        "scope": "Instrumented output diagnostic, not public-route/performance closure",
        "exclusions": [
            "FoldJAX-only distogram_logits excluded; native contact_probs compared",
            "No independent observer proves FoldJAX consumed every injected RNG draw",
            "No uninstrumented bridge, writer parity or speed/memory admission",
        ],
    }
    try:
        a, ac, (ak, ae), ap = load_arm(left)
        b, bc, (bk, be), bp = load_arm(right)
        shared = (
            "input_sha256",
            "samples",
            "steps",
            "cycles",
            "seed",
            "trunk_dtype",
            "native_source",
        )
        for key in shared:
            if ap[key] != bp[key]:
                raise ValueError(f"different comparison profile: {key}")
        output["arm_provenance_sha256"] = [
            sha(p / "provenance.json") for p in (left, right)
        ]
        output["precision_policies"] = [
            {key: p.get(key) for key in ("arm", "native_tf32", "jax_matmul_precision")}
            for p in (ap, bp)
        ]
        output["tape"] = {}
        for name in ("tape.npz", "msa.npz"):
            comparison = compare_confidence(
                left / "torch" / name, right / "torch" / name
            )
            comparison["exact"] = comparison["passed"] and all(
                v["bitwise_equal"] for v in comparison["leaves"].values()
            )
            output["tape"][name] = comparison
        output["native_input"] = compare_confidence(
            left / "native-input.npz", right / "native-input.npz"
        )
        native_input_exact = output["native_input"]["passed"] and all(
            leaf["bitwise_equal"] for leaf in output["native_input"]["leaves"].values()
        )
        output["coordinates"] = compare_entity_parity(
            a, b, ak, bk, ae, be, np.ones(a.shape[:2], bool), np.ones(b.shape[:2], bool)
        )
        output["coordinate_gate"] = coordinate_gate(
            output["coordinates"]["entity_rmsd"]
        )
        output["confidence"] = compare_confidence(ac, bc)
        output["passed"] = (
            native_input_exact
            and all(v["exact"] for v in output["tape"].values())
            and output["coordinate_gate"]["coordinate_gate_passed"]
            and output["confidence"]["passed"]
        )
    except (KeyError, OSError, ValueError, TypeError) as error:
        output["error"] = str(error)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.left, args.right)
    save(args.out, result)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "error": result.get("error"),
                "coordinate_gate": result.get("coordinate_gate"),
            }
        )
    )


if __name__ == "__main__":
    main()
