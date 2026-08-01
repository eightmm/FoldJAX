"""Manual contact, docking, and pocket restraint parity tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from foldjax.models.chai.data.features import generate_features
from foldjax.models.chai.data.restraints import parse_restraints_csv

from .test_features import _inputs


def _tensorcode(values: list[str], width: int) -> np.ndarray:
    result = np.full((len(values), width), 255, dtype=np.uint8)
    for row, value in enumerate(values):
        result[row, : len(value)] = np.frombuffer(value.encode("ascii"), np.uint8)
    return result


def _restraint_inputs() -> dict[str, np.ndarray]:
    inputs = _inputs()
    inputs["token_asym_id"] = np.asarray([[1, 1, 2, 0]], np.int32)
    inputs["token_residue_index"] = np.asarray([[0, 1, 0, 0]], np.int32)
    inputs["subchain_id"] = _tensorcode(["A", "A", "B", ""], 4)[None]
    inputs["token_residue_name"] = _tensorcode(["ALA", "ARG", "ASN", ""], 8)[None]
    return inputs


def test_parse_public_contact_and_pocket_csv(tmp_path: Path) -> None:
    path = tmp_path / "restraints.csv"
    path.write_text(
        "restraint_id,chainA,res_idxA,chainB,res_idxB,connection_type,"
        "confidence,min_distance_angstrom,max_distance_angstrom,comment\n"
        "r1,A,A1,B,N1,contact,1.0,0.0,9.5,contact\n"
        "r2,A,,B,N1,pocket,1.0,0.0,12.0,pocket\n",
        encoding="utf-8",
    )
    parsed = parse_restraints_csv(path)
    assert parsed["docking_constraints"] == [None]
    assert parsed["contact_constraints"][0]["left_residue_name"] == "ALA"
    assert parsed["pocket_constraints"][0]["pocket_token_residue_name"] == "ASN"

    inputs = _restraint_inputs()
    inputs.update(parsed)
    features = generate_features(inputs)
    assert features["TokenDistanceRestraint"][0, 0, 2, 0] == 9.5
    assert features["TokenPairPocketRestraint"][0, 2, 0, 0] == 12.0
    assert features["TokenPairPocketRestraint"][0, 2, 1, 0] == 12.0


@pytest.mark.parametrize(
    ("constraints", "message"),
    [
        ([{"left_residue_subchain_id": "Z"}], "missing keys"),
        (
            [
                {
                    "left_residue_subchain_id": "Z",
                    "right_residue_subchain_id": "B",
                    "left_residue_index": 0,
                    "right_residue_index": 0,
                    "left_residue_name": "ALA",
                    "right_residue_name": "ASN",
                    "distance_threshold": 8.0,
                }
            ],
            "subchain",
        ),
    ],
)
def test_invalid_contact_constraints_fail(
    constraints: list[dict], message: str
) -> None:
    inputs = _restraint_inputs()
    inputs["contact_constraints"] = [constraints]
    with pytest.raises(ValueError, match=message):
        generate_features(inputs)


def test_manual_docking_feature() -> None:
    inputs = _restraint_inputs()
    inputs["docking_constraints"] = [
        [
            {
                "subchain_ids": ["A", "B"],
                "noise_sigma": 0.0,
                "dropout_prob": 0.0,
                "atom_center_mask": [
                    np.asarray([True, True]),
                    np.asarray([True]),
                ],
                "atom_center_coords": [
                    np.asarray([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], np.float32),
                    np.asarray([[10.0, 0.0, 0.0]], np.float32),
                ],
            }
        ]
    ]
    feature = generate_features(inputs)["DockingConstraintGenerator"]
    expected = np.asarray(
        [[[0, 2, 3, 5], [2, 0, 2, 5], [3, 2, 0, 5], [5, 5, 5, 5]]],
        np.int64,
    )
    np.testing.assert_array_equal(feature, expected)


@pytest.mark.official_parity
def test_official_manual_docking_parity(
    tmp_path: Path, upstream_chai_dir: Path, upstream_chai_python: Path
) -> None:
    inputs = _restraint_inputs()
    input_path, output_path = tmp_path / "in.npz", tmp_path / "out.npy"
    np.savez(input_path, **inputs)
    script = r"""
import sys, numpy as np, torch
from chai_lab.chai1 import feature_factory
x = np.load(sys.argv[1], allow_pickle=False)
inputs = {k: torch.from_numpy(x[k]) for k in x.files}
inputs["docking_constraints"] = [[{
 "subchain_ids":["A","B"], "noise_sigma":0.0, "dropout_prob":0.0,
 "atom_center_mask":[torch.tensor([True,True]),torch.tensor([True])],
 "atom_center_coords":[torch.tensor([[0.,0.,0.],[5.,0.,0.]]),torch.tensor([[10.,0.,0.]])],
}]]
f = feature_factory.generate({"inputs": inputs})
np.save(sys.argv[2], f["DockingConstraintGenerator"].numpy())
"""
    subprocess.run(
        [upstream_chai_python, "-c", script, input_path, output_path],
        cwd=upstream_chai_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    inputs["docking_constraints"] = [
        [
            {
                "subchain_ids": ["A", "B"],
                "noise_sigma": 0.0,
                "dropout_prob": 0.0,
                "atom_center_mask": [np.asarray([True, True]), np.asarray([True])],
                "atom_center_coords": [
                    np.asarray([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], np.float32),
                    np.asarray([[10.0, 0.0, 0.0]], np.float32),
                ],
            }
        ]
    ]
    actual = generate_features(inputs)["DockingConstraintGenerator"]
    np.testing.assert_array_equal(actual[..., None], np.load(output_path))


@pytest.mark.official_parity
def test_official_manual_restraint_feature_parity(
    tmp_path: Path, upstream_chai_dir: Path, upstream_chai_python: Path
) -> None:
    inputs = _restraint_inputs()
    constraints = {
        "contact_constraints": [
            [
                {
                    "left_residue_subchain_id": "A",
                    "right_residue_subchain_id": "B",
                    "left_residue_index": 0,
                    "right_residue_index": 0,
                    "left_residue_name": "ALA",
                    "right_residue_name": "ASN",
                    "distance_threshold": 9.5,
                }
            ]
        ],
        "pocket_constraints": [
            [
                {
                    "pocket_chain_subchain_id": "A",
                    "pocket_token_subchain_id": "B",
                    "pocket_token_residue_index": 0,
                    "pocket_token_residue_name": "ASN",
                    "pocket_distance_threshold": 12.0,
                }
            ]
        ],
    }
    inputs.update(constraints)
    input_path, output_path = tmp_path / "in.npz", tmp_path / "out.npz"
    np.savez(input_path, **{k: v for k, v in inputs.items() if k not in constraints})
    script = r"""
import sys, numpy as np, torch
from chai_lab.chai1 import feature_factory
x = np.load(sys.argv[1], allow_pickle=False)
inputs = {k: torch.from_numpy(x[k]) for k in x.files}
inputs.update({
 "contact_constraints": [[{
  "left_residue_subchain_id":"A", "right_residue_subchain_id":"B",
  "left_residue_index":0, "right_residue_index":0,
  "left_residue_name":"ALA", "right_residue_name":"ASN",
  "distance_threshold":9.5,
 }]],
 "pocket_constraints": [[{
  "pocket_chain_subchain_id":"A", "pocket_token_subchain_id":"B",
  "pocket_token_residue_index":0, "pocket_token_residue_name":"ASN",
  "pocket_distance_threshold":12.0,
 }]],
})
f = feature_factory.generate({"inputs": inputs})
np.savez(
 sys.argv[2],
 contact=f["TokenDistanceRestraint"].numpy(),
 pocket=f["TokenPairPocketRestraint"].numpy(),
)
"""
    subprocess.run(
        [upstream_chai_python, "-c", script, input_path, output_path],
        cwd=upstream_chai_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    official = np.load(output_path)
    actual = generate_features(inputs)
    np.testing.assert_array_equal(actual["TokenDistanceRestraint"], official["contact"])
    np.testing.assert_array_equal(
        actual["TokenPairPocketRestraint"], official["pocket"]
    )
