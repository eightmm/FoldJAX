"""CPU reduction-order diagnostics; no native-kernel or model admission."""

import argparse
from pathlib import Path

import numpy as np

from bench.af3_closure_capture import sha
from bench.boltz_closure_capture import save_new
from bench.boltz_pwa_logits_probe import load_reference, verify_bindings
from bench.boltz_relpos_probe import bf16_round


def ordered_sum(products, lanes, *, contiguous=False):
    if products.dtype != np.float32 or products.shape[-1] != 128:
        raise ValueError("requires 128 FP32 products")
    if lanes not in (1, 2, 4, 8, 16, 32):
        raise ValueError("unsupported diagnostic lane count")
    terms = 128 // lanes
    values = products.reshape(*products.shape[:-1], terms, lanes)
    if contiguous:
        values = products.reshape(*products.shape[:-1], lanes, terms).swapaxes(-1, -2)
    total = np.zeros((*products.shape[:-1], lanes), np.float32)
    for i in range(terms):
        total = total + values[..., i, :]
    offset = lanes // 2
    while offset:
        total = total + total[..., np.arange(lanes) ^ offset]
        offset //= 2
    return total[..., 0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    _, x, w, target, bindings = load_reference(args.reference)
    bindings[Path(__file__).resolve()] = sha(Path(__file__).resolve())
    x, w = bf16_round(x), bf16_round(w)
    results = {}
    for contiguous in (False, True):
        for lanes in (1, 2, 4, 8, 16, 32):
            heads = []
            for h in range(8):
                products = x * w[h]
                actual = bf16_round(ordered_sum(products, lanes, contiguous=contiguous))
                heads.append({
                    "unequal": int(np.count_nonzero(actual != target[..., h])),
                    "max_abs": float(np.max(np.abs(actual - target[..., h]))),
                })
            results[f"{'contiguous' if contiguous else 'strided'}_{lanes}"] = heads
    verify_bindings(bindings)
    save_new(args.out, {
        "scope": "hypothesis discrimination only; not observed cuBLAS order",
        "not_model_parity_admission": True,
        "source_sha256": bindings[Path(__file__).resolve()],
        "reference_sha256": bindings[args.reference / "report.json"],
        "bindings_unchanged": True,
        "profiles": results,
    })


if __name__ == "__main__":
    main()
