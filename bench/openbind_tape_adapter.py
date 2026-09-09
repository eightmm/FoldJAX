"""Fail-closed adapter for pinned OpenBind forward RNG captures.

Only forward draws were recorded: shared input.npz is not evidence that the
independent preprocessing RNG stream matched. This adapter changes no draws.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from bench.boltz_historical_replay import digest, save_new, source_hashes

UPSTREAM_COMMIT = "c4771653c5d0a3ebb0b3af71b05efd64bc44ee86"


def prepare_core_features(features, *, max_atoms_per_token):
    """Add the head-derived mask to a native capture, not a public archive.

    Native head_modules constructs this after the forward input capture using
    broadcast_token_feat_to_atoms. Reuse the port's existing equivalent builder.
    This does not supply writer identity or independent preprocessing evidence.
    """
    from foldjax.models.openfold3.data.featurize import _max_atom_per_token_mask

    mask = np.asarray(features["token_mask"])
    counts = np.asarray(features["num_atoms_per_token"])
    if (
        max_atoms_per_token < 1
        or mask.ndim != 2
        or mask.shape[0] != 1
        or counts.shape != mask.shape
        or not np.isin(mask, (0, 1)).all()
        or not np.issubdtype(counts.dtype, np.integer)
        or np.any(counts < 0)
        or np.any(counts > max_atoms_per_token)
    ):
        raise ValueError("invalid native token counts or mask")
    derived = _max_atom_per_token_mask(features, max_atoms_per_token)
    if "max_atom_per_token_mask" in features and not np.array_equal(
        features["max_atom_per_token_mask"], derived
    ):
        raise ValueError("captured head mask disagrees with token counts")
    return {**features, "max_atom_per_token_mask": derived}


@dataclass(frozen=True)
class ForwardTape:
    msa_indices: np.ndarray
    noise: np.ndarray
    quaternions: np.ndarray
    translations: np.ndarray

    def augmentation(self):
        from foldjax.models.openfold3.models.augmentation import (
            AugmentationTape,
            validate_augmentation_tape,
        )

        return validate_augmentation_tape(
            AugmentationTape(self.quaternions, self.translations),
            steps=self.quaternions.shape[0],
            samples=self.quaternions.shape[1],
            check_values=True,
        )

    def prepare_features(self, features):
        from foldjax.models.openfold3.data.featurize import prepare_msa_cycle_features

        return prepare_msa_cycle_features(
            features,
            num_recycles=self.msa_indices.shape[0],
            no_subsampled=self.msa_indices.shape[1],
            selected_indices=self.msa_indices,
        )


def parse_forward_tape(
    draws, *, msa_mask, n_atom, samples, steps, cycles, msa_depth=1024
):
    mask = np.asarray(msa_mask)
    if mask.ndim != 3 or mask.shape[0] != 1 or not np.isin(mask, (0, 1)).all():
        raise ValueError("requires a finite binary batch-one MSA mask")
    if min(n_atom, samples, steps, cycles, msa_depth, mask.shape[1]) < 1:
        raise ValueError("tape dimensions must be positive")
    keys = list(draws)
    cursor = 0

    def take(name, dtype, shape):
        nonlocal cursor
        key = f"{cursor:06d}_{name}_torch_{dtype}"
        if cursor >= len(keys) or keys[cursor] != key:
            raise ValueError(f"missing or out-of-order native draw {key}")
        value = np.asarray(draws[key])
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise ValueError(f"wrong shape/dtype for native draw {key}")
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite native draw {key}")
        cursor += 1
        return value

    valid = np.flatnonzero(np.any(mask[0] != 0, axis=-1))
    invalid = np.flatnonzero(~np.any(mask[0] != 0, axis=-1))
    selections = []
    for _ in range(cycles):
        requested = int(take("randint", "int64", (1,))[0])
        if requested != msa_depth:
            raise ValueError("native MSA draw differs from pinned fixed depth")
        pool = valid if len(valid) >= requested else invalid
        if len(pool):
            permutation = take("randperm", "int64", (len(pool),))
            if not np.array_equal(np.sort(permutation), np.arange(len(pool))):
                raise ValueError("native randperm is not a permutation")
            selected = (
                pool[permutation[:requested]]
                if len(valid) >= requested
                else np.concatenate(
                    (valid, pool[permutation[: requested - len(valid)]])
                )
            )
        else:
            selected = valid
        selections.append(selected)
    noise = [take("randn", "float32", (1, samples, n_atom, 3))[0]]
    quaternions, translations = [], []
    for _ in range(steps):
        q = take("randn", "float32", (1, samples, 4))[0]
        with np.errstate(over="ignore", invalid="ignore"):
            norms = np.linalg.norm(q, axis=-1)
        if not np.isfinite(norms).all() or np.any(norms == 0):
            raise ValueError("native quaternion has invalid norm")
        quaternions.append(q)
        translations.append(take("randn", "float32", (1, samples, 3))[0])
        noise.append(take("randn_like", "float32", (1, samples, n_atom, 3))[0])
    if cursor != len(keys):
        raise ValueError("unconsumed native draws")
    return ForwardTape(
        np.stack(selections),
        np.stack(noise),
        np.stack(quaternions),
        np.stack(translations),
    )


def load_capture(root):
    runner = yaml.safe_load((root / "runner.yml").read_text())
    if runner["pl_trainer_args"]["precision"] != "32-true":
        raise ValueError("only native FP32 capture is admitted")
    effective = json.loads((root / "effective-model.json").read_text())
    full = json.loads((root / "predictions/model_config.json").read_text())
    validate_native_config(full, effective)
    shared = effective["shared"]
    samples = shared["diffusion"]["no_full_rollout_samples"]
    steps = shared["diffusion"]["no_full_rollout_steps"]
    cycles = shared["num_recycles"] + 1
    if (samples, steps, cycles) != (5, 200, 4):
        raise ValueError("only released n5/200/four-pass capture is admitted")
    with np.load(root / "input.npz", allow_pickle=False) as inputs:
        mask = inputs["msa_mask"]
        atom_mask = inputs["atom_mask"]
        if atom_mask.ndim != 2 or atom_mask.shape[0] != 1:
            raise ValueError("requires batch-one atom mask")
        with np.load(root / "tape.npz", allow_pickle=False) as draws:
            tape = parse_forward_tape(
                draws,
                msa_mask=mask,
                n_atom=atom_mask.shape[1],
                samples=samples,
                steps=steps,
                cycles=cycles,
            )
    return tape, effective


def validate_native_config(full, effective):
    architecture = full["architecture"]
    if architecture["shared"] != effective["shared"]:
        raise ValueError("recorded native shared config disagrees with forward")
    if full["settings"] != effective["settings"]:
        raise ValueError("recorded native settings disagree with forward")
    msa = architecture["msa"]["msa_module_embedder"]
    if (
        msa["subsample_main_msa"],
        msa["subsample_all_msa"],
        msa["min_subsampled_all_msa"],
        msa["max_subsampled_all_msa"],
    ) != (False, True, 1024, 1024):
        raise ValueError("adapter requires native fixed-depth all-MSA selection")
    return architecture


def capture_provenance(root, checkpoint_hash):
    """Never attach original-run provenance or tuning to a different repeat."""
    trace_path = root / "trace.json"
    if trace_path.exists():
        trace = json.loads(trace_path.read_text())
        if trace["native_source"]["commit"] != UPSTREAM_COMMIT:
            raise ValueError("repeat capture upstream provenance mismatch")
        experiment = json.loads(
            (root / "predictions/experiment_config.json").read_text()
        )
        captured_checkpoint = Path(experiment["inference_ckpt_path"])
        if digest(captured_checkpoint) != checkpoint_hash:
            raise ValueError("checkpoint differs from repeat's recorded checkpoint")
        return {
            "kind": "repeat trace plus recorded checkpoint identity",
            "sha256": digest(trace_path),
            "trace": trace,
            "checkpoint_identity_scope": "recorded path rehashed now",
        }
    path = root.parent / "provenance.json"
    provenance = json.loads(path.read_text())
    if provenance["source"]["commit"] != UPSTREAM_COMMIT:
        raise ValueError("capture upstream provenance mismatch")
    if provenance["weights"]["sha256"] != checkpoint_hash:
        raise ValueError("checkpoint differs from native capture")
    return {"kind": "original capture provenance", "sha256": digest(path)}


def required_triangle_kernel(effective):
    memory = effective["settings"]["memory"]["eval"]
    if memory["use_deepspeed_evo_attention"] or memory["use_lma"]:
        raise ValueError("native attention policy is outside this replay adapter")
    triton, cueq = (
        memory["use_triton_triangle_kernels"],
        memory["use_cueq_triangle_kernels"],
    )
    if triton and cueq:
        raise ValueError("ambiguous native triangle kernel flags")
    return "triton" if triton else "cueq" if cueq else "xla"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    upstream = args.upstream_root.resolve(strict=True)

    def git(*command):
        return subprocess.check_output(
            ["git", "-C", str(upstream), *command], text=True
        )

    if git("rev-parse", "HEAD").strip() != UPSTREAM_COMMIT or git("diff", "HEAD", "--"):
        raise ValueError("upstream must be the clean pinned OpenBind source")
    capture = args.capture.resolve(strict=True)
    names = (
        "input.npz",
        "tape.npz",
        "effective-model.json",
        "coordinate.npz",
        "runner.yml",
        "predictions/model_config.json",
        "predictions/experiment_config.json",
    )
    if (capture / "trace.json").exists():
        names += ("trace.json",)
    identities = {name: digest(capture / name) for name in names}
    tape, effective = load_capture(capture)
    candidate_hashes = source_hashes(args.candidate_root.resolve(strict=True))
    checkpoint_hash = digest(args.checkpoint)
    provenance = capture_provenance(capture, checkpoint_hash)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    with (args.out_dir / "adapted-tape.npz").open("xb") as stream:
        np.savez_compressed(
            stream,
            msa_indices=tape.msa_indices,
            noise=tape.noise,
            quaternions=tape.quaternions,
            translations=tape.translations,
        )
    save_new(
        args.out_dir / "manifest.json",
        {
            "scope": (
                "forward-tape adapter only; shared features; no preprocessing proof"
            ),
            "scientific_acceptance": None,
            "native_artifacts": identities,
            "upstream_commit": UPSTREAM_COMMIT,
            "capture_provenance": provenance,
            "checkpoint_sha256": checkpoint_hash,
            "candidate_source": candidate_hashes,
            "adapter_sha256": digest(Path(__file__)),
            "adapted_tape_sha256": digest(args.out_dir / "adapted-tape.npz"),
            "effective_model": effective,
            "required_triangle_kernel": required_triangle_kernel(effective),
            "native_model_config": json.loads(
                (capture / "predictions/model_config.json").read_text()
            ),
            "draw_policy": (
                "raw FP32 quaternion/translation and initial/churn; original order"
            ),
            "schedule": (
                "arrays not captured; native noise_schedule and sample_diffusion "
                "constructor settings recovered from bound model_config.json"
            ),
            "remaining": [
                "runtime tuned chunks",
                "native-equivalent kernel admission",
                "source-isolated model replay and entity/confidence metrics",
            ],
        },
    )
    if identities != {name: digest(capture / name) for name in names}:
        raise RuntimeError("native capture changed during adaptation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
