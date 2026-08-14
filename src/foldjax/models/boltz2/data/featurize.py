"""Featurization: raw YAML job -> model feature dict (no ``import boltz``).

``featurize_yaml`` runs check_inputs -> process_inputs -> PredictionDataset over
one input and returns the batched feature dict, the manifest, and the processed
structures dir. The vendored tensor facade is NumPy-backed, so both this
preprocessor and inference stay inside FoldJAX's NumPy/JAX runtime. ``cache_dir``
memoizes features by a content digest of the input (YAML + referenced
MSA/template files + options).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from pathlib import Path

import numpy as np

from foldjax.models.boltz2.data.module.inferencev2 import PredictionDataset
from foldjax.models.boltz2.data.preprocess import check_inputs, process_inputs
from foldjax.models.boltz2.data.types import Manifest, StructureV2


def _cache_opts(
    use_msa_server: bool,
    msa_server_url: str,
    msa_pairing_strategy: str,
    max_msa_depth: int | None,
) -> tuple:
    """Everything besides the input files that changes the features produced.

    `max_msa_depth` belongs here and was missing. Without it a run that asked
    for a cap got an earlier uncapped run's features straight back from cache,
    and a run that asked for no cap got the capped ones -- on the knob that
    dominates both peak memory and accuracy, with nothing to say it happened.
    """
    return (use_msa_server, msa_server_url, msa_pairing_strategy, max_msa_depth)


def _input_digest(yaml_path: Path, mol_dir: Path, opts: tuple) -> str:
    """Content digest of a featurization request.

    Hashes the YAML text, every existing file it references (MSA a3m/csv,
    template cif/pdb), the mol dir path, and the options. Output-identical
    inputs -> identical digest -> cache hit.
    """
    h = hashlib.sha256()
    text = yaml_path.read_bytes()
    h.update(text)
    h.update(str(mol_dir).encode())
    h.update(repr(opts).encode())
    # Fold in the content of any referenced file that exists on disk. Only
    # consider path-like tokens (contain '/' or a file extension) and short
    # enough to be a real path -- avoids treating long sequence strings as paths.
    #
    # Relative references are resolved against the job file, not the process
    # working directory. Boltz job files name their alignments relative to
    # themselves, so resolving against the CWD found nothing, folded nothing
    # in, and left the digest blind to the alignment: editing an a3m then
    # re-running returned the previous features from cache.
    for tok in re.findall(rb"[\w./@-]+", text):
        s = tok.decode("utf-8", "ignore")
        if len(s) > 255 or ("/" not in s and "." not in s):
            continue
        try:
            p = Path(s)
            if not p.is_absolute():
                beside = yaml_path.parent / p
                p = beside if beside.is_file() else p
            if not p.is_file():
                continue
            with p.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
        except OSError:
            continue
    return h.hexdigest()[:16]


def featurize_yaml(
    yaml_path: Path,
    out_dir: Path,
    mol_dir: Path,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    msa_server_username: str | None = None,
    msa_server_password: str | None = None,
    msa_api_key_header: str | None = None,
    msa_api_key_value: str | None = None,
    cache_dir: Path | None = None,
    max_msa_depth: int | None = None,
) -> tuple[dict[str, np.ndarray], object, Path]:
    """Run preprocessing + featurization for one YAML.

    Returns ``(feats_np, manifest, processed_structures_dir)`` where
    ``feats_np`` maps feature name -> batched numpy array (leading batch dim 1),
    and the structures dir holds the processed ``{id}.npz`` files used for
    structure-file writing. ``manifest`` is None on a cache hit (callers use
    record_id from the structure-dir filename, which is the cached record).
    """
    from foldjax.models.boltz2.data._torch import torch

    assert mol_dir.exists(), f"missing mols dir: {mol_dir}"

    opts = _cache_opts(
        use_msa_server, msa_server_url, msa_pairing_strategy, max_msa_depth
    )
    cache_entry = None
    if cache_dir is not None:
        digest = _input_digest(yaml_path, mol_dir, opts)
        cache_entry = Path(cache_dir) / digest
        feats_file = cache_entry / "feats.npz"
        struct_dir = cache_entry / "structures"
        if feats_file.is_file() and struct_dir.is_dir():
            loaded = np.load(feats_file)
            feats_np = {k: loaded[k] for k in loaded.files}
            return feats_np, None, struct_dir

    data = check_inputs(yaml_path)
    assert data, "check_inputs returned nothing"

    manifest = process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=out_dir / "unused_ccd.pkl",  # boltz2=True uses canonicals
        mol_dir=mol_dir,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        api_key_header=msa_api_key_header,
        api_key_value=msa_api_key_value,
        boltz2=True,
    )

    processed = out_dir / "processed"
    dataset = PredictionDataset(
        manifest=manifest,
        target_dir=processed / "structures",
        msa_dir=processed / "msa",
        mol_dir=mol_dir,
        constraints_dir=processed / "constraints",
        template_dir=processed / "templates",
        extra_mols_dir=processed / "mols",
        max_msa_seqs=max_msa_depth,
    )
    features = dataset[0]

    feats_np: dict[str, np.ndarray] = {}
    for key, value in features.items():
        if key.startswith("_") or key == "record":
            continue
        if not torch.is_tensor(value):
            continue
        feats_np[key] = value.unsqueeze(0).detach().cpu().numpy()

    struct_dir = processed / "structures"
    if cache_entry is not None:
        import shutil

        out_struct = cache_entry / "structures"
        out_struct.mkdir(parents=True, exist_ok=True)
        for rec in manifest.records:
            src = struct_dir / f"{rec.id}.npz"
            if src.is_file():
                shutil.copy2(src, out_struct / f"{rec.id}.npz")
        np.savez(cache_entry / "feats.npz", **feats_np)
        struct_dir = out_struct
    return feats_np, manifest, struct_dir


def featurize_affinity_from_prediction(
    *,
    manifest: Manifest,
    processed_dir: Path,
    mol_dir: Path,
    predicted_coords: np.ndarray,
    atom_pad_mask: np.ndarray,
    out_dir: Path,
    max_msa_depth: int | None = None,
) -> dict[str, np.ndarray]:
    """Build cropped affinity features under the primary stage's MSA cap."""

    from foldjax.models.boltz2.data._torch import torch

    if len(manifest.records) != 1:
        raise ValueError("affinity inference currently requires exactly one record")
    record = manifest.records[0]
    structure = StructureV2.load(processed_dir / "structures" / f"{record.id}.npz")
    coords = np.asarray(predicted_coords).reshape(-1, 3)
    mask = np.asarray(atom_pad_mask).reshape(-1).astype(bool)
    real_coords = coords[mask]
    if real_coords.shape[0] != structure.atoms.shape[0]:
        raise ValueError(
            "predicted atom count does not match processed structure: "
            f"{real_coords.shape[0]} != {structure.atoms.shape[0]}"
        )

    atoms = structure.atoms.copy()
    atoms["coords"] = real_coords
    ensemble_coords = structure.coords.copy()
    if ensemble_coords.shape[0] != real_coords.shape[0]:
        raise ValueError("affinity stage expects one coordinate conformer")
    ensemble_coords["coords"] = real_coords
    predicted_structure = replace(
        structure,
        atoms=atoms,
        coords=ensemble_coords,
    )

    target_dir = Path(out_dir) / "structures"
    record_dir = target_dir / record.id
    record_dir.mkdir(parents=True, exist_ok=True)
    predicted_structure.dump(record_dir / f"pre_affinity_{record.id}.npz")

    dataset = PredictionDataset(
        manifest=manifest,
        target_dir=target_dir,
        msa_dir=processed_dir / "msa",
        mol_dir=mol_dir,
        constraints_dir=processed_dir / "constraints",
        template_dir=processed_dir / "templates",
        extra_mols_dir=processed_dir / "mols",
        override_method="other",
        affinity=True,
        max_msa_seqs=max_msa_depth,
    )
    features = dataset[0]
    feats_np: dict[str, np.ndarray] = {}
    for key, value in features.items():
        if key.startswith("_") or key == "record":
            continue
        if torch.is_tensor(value):
            feats_np[key] = value.unsqueeze(0).detach().cpu().numpy()
        elif isinstance(value, (int, float, np.number)):
            feats_np[key] = np.asarray([value])
    return feats_np
