from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from foldjax.models._output_validation import require_finite_coordinates
from foldjax.models.esmfold2 import output as esmfold2_output
from foldjax.models.esmfold2.data.features import build_features
from foldjax.models.openfold3.inference import Prediction
from foldjax.models.openfold3.output import write_prediction_outputs
from foldjax.models.protenix.data.output import write_protenix_outputs
from tests.models.openfold3.feature_fixture import minimal_features


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_protenix_and_opendde_shared_writer_reject_nonfinite_coordinates(
    tmp_path: Path, bad: float
) -> None:
    output = {
        "coordinate": np.asarray([[[bad, 0.0, 0.0]]], dtype=np.float32),
        "atom_plddt": np.ones((1, 1), dtype=np.float32),
    }
    features = {
        "output_atom_name": np.asarray(["CA"]),
        "output_atom_element": np.asarray(["C"]),
        "output_atom_res_name": np.asarray(["ALA"]),
        "output_atom_chain_id": np.asarray(["A"]),
        "output_atom_res_id": np.asarray([1]),
    }

    with pytest.raises(ValueError, match="Protenix/OpenDDE.*non-finite"):
        write_protenix_outputs(
            tmp_path,
            job_name="bad",
            seed=1,
            output=output,
            features=features,
        )

    assert not list(tmp_path.rglob("*.cif"))


@pytest.mark.parametrize("bad", [float("nan"), float("-inf")])
def test_esmfold2_writer_rejects_nonfinite_public_atoms_before_writing(
    tmp_path: Path, bad: float
) -> None:
    features = build_features([("A", "A", 0, 0)])
    coordinates = np.zeros(
        (1, features["atom_attention_mask"].shape[-1], 3), dtype=np.float32
    )
    coordinates[0, 0, 0] = bad

    with pytest.raises(ValueError, match="ESMFold2.*non-finite"):
        esmfold2_output.write_prediction_outputs(
            {"sample_atom_coords": coordinates},
            features,
            tmp_path / "out",
            name="bad",
        )

    assert not list(tmp_path.rglob("*.cif"))


def test_finite_guard_ignores_masked_serving_padding() -> None:
    coordinates = np.zeros((1, 3, 3), dtype=np.float32)
    coordinates[0, 2, 0] = np.nan

    returned = require_finite_coordinates(
        coordinates,
        model="test",
        atom_mask=np.asarray([1, 1, 0]),
    )

    assert returned is coordinates


def _openfold3_prediction(coordinates: np.ndarray) -> Prediction:
    n_sample, n_atom, _ = coordinates.shape
    return Prediction(
        coordinates=coordinates,
        plddt=np.full((n_sample, n_atom), 0.9, dtype=np.float32),
        ptm=np.full((n_sample,), 0.7, dtype=np.float32),
        iptm=np.full((n_sample,), 0.6, dtype=np.float32),
        chain_pair_iptm=None,
        pae_logits=None,
        pde_logits=None,
        distogram_logits=None,
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_openfold3_direct_writer_rejects_nonfinite_public_atoms_before_writing(
    tmp_path: Path, bad: float
) -> None:
    features = minimal_features(tokens=2, atoms=4)
    features["atom_mask"] = np.asarray([[1, 1, 0, 0]], dtype=np.float32)
    coordinates = np.zeros((1, 4, 3), dtype=np.float32)
    coordinates[0, 0, 0] = bad

    with (
        pytest.warns(RuntimeWarning, match="no exact output metadata"),
        pytest.raises(ValueError, match="OpenFold3.*non-finite"),
    ):
        write_prediction_outputs(
            _openfold3_prediction(coordinates),
            features,
            tmp_path / "out",
        )

    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.rglob("*.cif"))


def test_openfold3_direct_writer_ignores_nonfinite_masked_padding(
    tmp_path: Path,
) -> None:
    features = minimal_features(tokens=2, atoms=4)
    features["atom_mask"] = np.asarray([[1, 1, 0, 0]], dtype=np.float32)
    coordinates = np.zeros((1, 4, 3), dtype=np.float32)
    coordinates[0, 2, 0] = np.nan

    with pytest.warns(RuntimeWarning, match="no exact output metadata"):
        written = write_prediction_outputs(
            _openfold3_prediction(coordinates),
            features,
            tmp_path / "out",
        )

    assert written["structures"][0].is_file()


def test_boltz2_nonfinite_guard_survives_optimized_python(tmp_path: Path) -> None:
    script = f"""
from pathlib import Path
import jax.numpy as jnp
import numpy as np
import foldjax.models.boltz2.api as api
import foldjax.models.boltz2.bridge.native as native
import foldjax.models.boltz2.models.predict as model_predict

root = Path({str(tmp_path)!r})
features = {{
    "atom_pad_mask": np.asarray([[1.0, 0.0]], dtype=np.float32),
    "token_pad_mask": np.ones((1, 1), dtype=np.float32),
    "mol_type": np.zeros((1, 1), dtype=np.int32),
    "affinity_token_mask": np.zeros((1, 1), dtype=np.float32),
}}
api.featurize = lambda **kwargs: (features, "job", root)
native.load_params = lambda path: {{"trunk": {{}}}}
next_coordinates = jnp.asarray([[[0.0, 0.0, 0.0], [jnp.nan, 0.0, 0.0]]])
model_predict.boltz2_predict = lambda *args, **kwargs: {{
    "sample_atom_coords": next_coordinates,
    "plddt": jnp.ones((1, 1)),
}}
api.predict(
    seq=["A"],
    weights=root / "weights",
    mols=root,
    out_dir=root / "masked-padding",
    diffusion_samples=1,
    write_fmt=None,
)
next_coordinates = jnp.asarray([[[jnp.nan, 0.0, 0.0], [0.0, 0.0, 0.0]]])
try:
    api.predict(
        seq=["A"],
        weights=root / "weights",
        mols=root,
        out_dir=root / "out",
        diffusion_samples=1,
        write_fmt=None,
    )
except ValueError as error:
    if "Boltz2 produced non-finite coordinates" not in str(error):
        raise
else:
    raise RuntimeError("optimized Python accepted non-finite Boltz2 coordinates")
"""
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    completed = subprocess.run(
        [sys.executable, "-O", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    assert not list(tmp_path.rglob("*.cif"))
