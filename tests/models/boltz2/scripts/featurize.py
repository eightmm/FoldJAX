"""Featurization via foldjax.models.boltz2.data: raw YAML -> model feature dict.

Provides ``featurize_yaml`` (used by ``scripts/predict.py``): runs
check_inputs -> process_inputs -> PredictionDataset over one input, returning the
feature dict, manifest, and processed-structures dir. The featurization runs
without ``import boltz``. Running this module directly featurizes 1UBQ_A and
checks it against outputs/real_features/1UBQ_A.npz.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from foldjax.models.boltz2.data.featurize import featurize_yaml

ROOT = Path(__file__).resolve().parent.parent
MOL_DIR = ROOT / ".cache" / "boltz" / "mols"
REF_NPZ = ROOT / "outputs" / "real_features" / "1UBQ_A.npz"
RECORD_ID = "1UBQ_A"

PREP_DIR = ROOT / "outputs" / "prep" / "1UBQ_A"
YAML_PATH = PREP_DIR / "1UBQ_A.yaml"
SEQUENCE = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"  # noqa: E501


def write_input_yaml() -> Path:
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    YAML_PATH.write_text(
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        f"      id: A\n"
        f"      sequence: {SEQUENCE}\n"
        "      msa: empty\n"
    )
    return YAML_PATH


def build_features() -> dict[str, np.ndarray]:
    write_input_yaml()
    feats_np, manifest, _ = featurize_yaml(YAML_PATH, PREP_DIR, MOL_DIR)
    records = [r for r in manifest.records if r.id == RECORD_ID]
    assert len(records) == 1, f"record {RECORD_ID} not in manifest"
    return feats_np


def _ref_pos_rigid_match(produced: dict, ref: dict) -> tuple[bool, str]:
    rp_r, rp_o = ref["ref_pos"][0], produced["ref_pos"][0]
    if rp_r.shape != rp_o.shape:
        return False, f"shape {rp_o.shape} vs {rp_r.shape}"
    uid = ref["ref_space_uid"][0]
    res = ref["atom_resolved_mask"][0].astype(bool)

    def pdist(x):
        d = x[:, None, :] - x[None, :, :]
        return np.sqrt((d * d).sum(-1))

    worst = 0.0
    for u in np.unique(uid):
        idx = np.where((uid == u) & res)[0]
        if len(idx) < 2:
            continue
        worst = max(worst, float(np.abs(pdist(rp_r[idx]) - pdist(rp_o[idx])).max()))
    ok = worst < 1e-3
    return ok, f"rigid-invariant: intra-block dist max|d|={worst:.2e} (RNG aug)"


def compare(produced: dict[str, np.ndarray], ref: dict[str, np.ndarray]) -> bool:
    pk, rk = set(produced), set(ref)
    missing = sorted(rk - pk)
    extra = sorted(pk - rk)

    rows = []
    overall = True
    for k in sorted(rk & pk):
        if k == "ref_pos":
            ok, detail = _ref_pos_rigid_match(produced, ref)
            rows.append((k, "PASS" if ok else "FAIL", detail))
            overall = overall and ok
            continue
        a, b = produced[k], ref[k]
        if a.shape != b.shape:
            rows.append((k, "FAIL", f"shape {a.shape} vs {b.shape}"))
            overall = False
            continue
        if np.issubdtype(b.dtype, np.floating):
            close = np.allclose(a, b, atol=1e-4, equal_nan=True)
            detail = "" if close else f"max|d|={np.nanmax(np.abs(a - b)):.3e}"
        else:
            close = np.array_equal(a, b)
            detail = "" if close else "int mismatch"
        rows.append((k, "PASS" if close else "FAIL", detail))
        overall = overall and close

    w = max(len(r[0]) for r in rows) if rows else 10
    print(f"{'key':<{w}}  RESULT  detail")
    print("-" * (w + 25))
    for k, res, detail in rows:
        print(f"{k:<{w}}  {res:<6}  {detail}")

    if missing:
        overall = False
        print(f"\nMISSING keys (in ref, not produced): {missing}")
    if extra:
        print(f"\nEXTRA keys (produced, not in ref): {extra}")

    n_pass = sum(1 for _, r, _ in rows if r == "PASS")
    print(
        f"\n{n_pass}/{len(rows)} matched keys PASS; "
        f"{len(missing)} missing, {len(extra)} extra"
    )
    return overall and not missing


def main() -> None:
    assert REF_NPZ.exists(), f"missing reference npz: {REF_NPZ}"
    ref = dict(np.load(REF_NPZ))
    produced = build_features()
    print(f"produced {len(produced)} keys; reference {len(ref)} keys\n")
    overall = compare(produced, ref)
    print("\nOVERALL:", "PASS" if overall else "FAIL")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
