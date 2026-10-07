"""Featurize one job on real torch and on the NumPy layer, and compare.

The unit tests pin the places torch and NumPy disagree; this one is the gate
that says the whole featurizer agrees. It runs only when torch is installed
(`--extra boltz-preprocess`), because it needs both backends to compare.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from foldjax.paths import weights_dir

pytestmark = pytest.mark.slow

_JOB = {
    "version": 1,
    "sequences": [{"protein": {"id": ["A"], "sequence": "LSDEDFKAV", "msa": "empty"}}],
}

_ALL_MODALITIES_JOB = {
    "version": 1,
    "sequences": [
        {
            "protein": {
                "id": ["A"],
                "sequence": "ACDEFGHIK",
                "msa": "empty",
            }
        },
        {"dna": {"id": ["D"], "sequence": "ACGTAC"}},
        {"rna": {"id": ["R"], "sequence": "ACGUAC"}},
        {"ligand": {"id": ["C"], "ccd": "ATP"}},
        {"ligand": {"id": ["S"], "smiles": "CC(=O)N"}},
        {"ligand": {"id": ["I"], "ccd": "MG"}},
    ],
}

# The random roto-translation applied to the reference conformers is drawn
# from a NumPy generator seeded by the job seed, and both backends consume the
# same normals. What is left is the rotation itself: torch's float32 einsum
# and NumPy's round differently, 2.4e-7 and 4.8e-7 at most on the two jobs
# below (one or two ulp). A wrong draw moves atoms by angstroms, so a bound
# of 1e-5 still refuses any real difference.
_SEED = 5
TOLERANCE = {"ref_pos": 1e-5}


def _featurize(
    job: Path, out: Path, backend: str, mols: Path, seed: int = _SEED
) -> dict:
    """Featurize in a subprocess with a test-only upstream tensor injection."""
    test_backend = ""
    if backend == "torch":
        test_backend = (
            "import torch, types\n"
            "selector = types.ModuleType(\n"
            "    'foldjax.models.boltz2.data._torch')\n"
            "selector.torch = torch\n"
            "selector.Tensor = torch.Tensor\n"
            "selector.from_numpy = torch.from_numpy\n"
            "selector.BACKEND = 'torch'\n"
            "sys.modules['foldjax.models.boltz2.data._torch'] = selector\n"
        )
    script = "".join(
        (
            "import sys, numpy as np\n",
            test_backend,
            "from pathlib import Path\n",
            "from foldjax.models.boltz2.data._torch import BACKEND\n",
            f"assert BACKEND == {backend!r}, BACKEND\n",
            "from foldjax.models.boltz2.data.featurize import featurize_yaml\n",
            "feats, _, _ = featurize_yaml(\n",
            f"    yaml_path=Path({str(job)!r}), out_dir=Path({str(out)!r}),\n",
            f"    mol_dir=Path({str(mols)!r}), seed={int(seed)})\n",
            f"np.savez({str(out) + '.npz'!r}, **feats)\n",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    return dict(np.load(f"{out}.npz"))


@pytest.mark.parametrize(
    "job_spec",
    [_JOB, _ALL_MODALITIES_JOB],
    ids=["protein", "protein-dna-rna-ccd-smiles-ion"],
)
@pytest.mark.real_store
def test_the_numpy_layer_reproduces_the_torch_featurizer(
    tmp_path: Path, job_spec: dict
) -> None:
    pytest.importorskip("torch")
    mols = weights_dir("boltz2") / "mols"
    if not mols.is_dir():
        pytest.skip("boltz2 molecule directory not fetched")

    job = tmp_path / "job.yaml"
    job.write_text(json.dumps(job_spec))

    with_torch = _featurize(job, tmp_path / "torch", "torch", mols)
    with_numpy = _featurize(job, tmp_path / "numpy", "numpy", mols)

    assert set(with_torch) == set(with_numpy)
    assert with_torch, "featurizer produced nothing"

    mismatched = []
    for name in sorted(with_torch):
        expected, actual = with_torch[name], with_numpy[name]
        if expected.shape != actual.shape:
            mismatched.append(f"{name}: shape {expected.shape} vs {actual.shape}")
        elif expected.dtype != actual.dtype:
            mismatched.append(f"{name}: dtype {expected.dtype} vs {actual.dtype}")
        elif name in TOLERANCE:
            worst = np.max(np.abs(expected.astype(float) - actual.astype(float)))
            if not worst <= TOLERANCE[name]:
                mismatched.append(f"{name}: max|diff| {worst}")
        elif not np.array_equal(expected, actual):
            worst = np.max(np.abs(expected.astype(float) - actual.astype(float)))
            mismatched.append(f"{name}: max|diff| {worst}")
    assert mismatched == []


@pytest.mark.real_store
def test_the_torch_backend_reproduces_ref_pos_per_seed(tmp_path: Path) -> None:
    """Upstream's global-RNG draw is replaced on both backends, torch included."""
    pytest.importorskip("torch")
    mols = weights_dir("boltz2") / "mols"
    if not mols.is_dir():
        pytest.skip("boltz2 molecule directory not fetched")

    job = tmp_path / "job.yaml"
    job.write_text(json.dumps(_JOB))
    first = _featurize(job, tmp_path / "a", "torch", mols)
    second = _featurize(job, tmp_path / "b", "torch", mols)
    other = _featurize(job, tmp_path / "c", "torch", mols, seed=_SEED + 1)

    np.testing.assert_array_equal(first["ref_pos"], second["ref_pos"])
    assert not np.array_equal(first["ref_pos"], other["ref_pos"])
