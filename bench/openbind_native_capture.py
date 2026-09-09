"""Base native OpenBind capture: shared forward inputs, RNG tape, kernel census.

This is the ``--native-wrapper`` module for ``openbind_native_outputs.py``.
It records what the tape adapter replays -- the model feature batch as the
native runner hands it to ``forward`` and every ``torch.randint`` /
``randperm`` / ``randn`` / ``randn_like`` draw made inside that forward, in
call order -- and counts which triangle kernel actually executed. A requested
cuEquivariance kernel is not evidence that it ran: upstream silently falls
back to plain attention below ``CUEQ_TRIATTN_FALLBACK_THRESHOLD`` tokens.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

UP = Path(
    os.environ.get(
        "OPENBIND_UPSTREAM_ROOT",
        "/home/jaemin/non-project/optimizing/openfold3-v050",
    )
)
CHECKPOINT = Path(
    os.environ.get(
        "OPENBIND_NATIVE_CHECKPOINT",
        "/home/jaemin/non-project/optimizing/foldjax-bench/upstream-default-n5-20260904"
        "/upstream-root/openfold3-v050/openfold3_weights/checkpoints"
        "/of3-ob-2025-06-30-174k.pt",
    )
)
SEED = 101
RECORDED_DRAWS = ("randint", "randperm", "randn", "randn_like")


class driver:  # noqa: N801 - attribute name the outputs script expects
    @staticmethod
    def digest_file(path):
        path = Path(path)
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                h.update(block)
        return {"sha256": h.hexdigest(), "bytes": path.stat().st_size}

    @staticmethod
    def source_identity(root):
        def git(*command):
            return subprocess.check_output(
                ["git", "-C", str(root), *command], text=True
            )

        return {
            "commit": git("rev-parse", "HEAD").strip(),
            "tracked_diff_sha256": hashlib.sha256(
                subprocess.check_output(
                    ["git", "-C", str(root), "diff", "--binary", "HEAD", "--"]
                )
            ).hexdigest(),
            "status_sha256": hashlib.sha256(
                git("status", "--porcelain", "--untracked-files=no").encode()
            ).hexdigest(),
        }


class control:  # noqa: N801 - attribute name the outputs script expects
    @staticmethod
    def save(path, value):
        with Path(path).open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, default=_json_default)
            stream.write("\n")


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def flatten(value, prefix=""):
    """Arrays only; nested keys join with '.', sequences by index."""
    import torch

    out = {}
    if isinstance(value, torch.Tensor):
        out[prefix] = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        out[prefix] = value
    elif isinstance(value, dict):
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            out.update(flatten(item, f"{prefix}.{index}" if prefix else str(index)))
    elif hasattr(value, "get_annotation_categories") and hasattr(value, "coord"):
        for name in value.get_annotation_categories():
            out[f"{prefix}.annotation.{name}"] = np.asarray(value.get_annotation(name))
        out[f"{prefix}.coord"] = np.asarray(value.coord)
    return out


def _dtype_name(tensor):
    return str(tensor.dtype).removeprefix("torch.")


class DrawRecorder:
    """Record every module-level RNG draw in call order, tape-key spelled."""

    def __init__(self):
        self.draws = []

    def wrap(self, name, original):
        def recorded(*args, **kwargs):
            value = original(*args, **kwargs)
            self.draws.append((name, value.detach().cpu().numpy().copy(), value))
            return value

        return recorded

    def archive(self):
        return {
            f"{index:06d}_{name}_torch_{_dtype_name(tensor)}": array
            for index, (name, array, tensor) in enumerate(self.draws)
        }


class KernelCensus:
    """Count which attention/multiplication implementation executed."""

    def __init__(self):
        self.calls = {}

    def wrap(self, label, original):
        def counted(*args, **kwargs):
            self.calls[label] = self.calls.get(label, 0) + 1
            return original(*args, **kwargs)

        return counted

    def patches(self):
        from openfold3.core.model.layers import triangular_multiplicative_update as tmu
        from openfold3.core.model.primitives import attention

        targets = [
            (attention, "_attention", "attention.torch"),
            (attention, "_cueq_triangle_attn", "attention.cueq"),
            (attention, "_triton_evo_attn", "attention.triton"),
            (attention, "_deepspeed_evo_attn", "attention.deepspeed"),
            (attention, "cueq_would_fall_back", "attention.cueq_fallback_check"),
            (tmu, "_cueq_triangle_mult", "trimul.cueq"),
            (tmu, "triton_layernorm", "trimul.triton_layernorm"),
            (tmu, "triton_linear_fused", "trimul.triton_linear_fused"),
        ]
        active = []
        for module, name, label in targets:
            original = getattr(module, name, None)
            if original is None:
                continue
            if name == "cueq_would_fall_back":
                def wrapped(*args, _o=original, **kwargs):
                    result = _o(*args, **kwargs)
                    key = "attention.cueq_fallback_" + ("true" if result else "false")
                    self.calls[key] = self.calls.get(key, 0) + 1
                    return result
                active.append(patch.object(module, name, wrapped))
            else:
                active.append(patch.object(module, name, self.wrap(label, original)))
        return active


def capture(input_json, out, precision):
    """Run one native prediction and leave a replayable capture in ``out``."""
    import torch
    from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom
    from openfold3.run_openfold import cli

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runner = out / "runner.yml"
    runner.write_text(
        yaml.safe_dump(
            {
                "model_update": {"presets": ["predict"]},
                "experiment_settings": {"seeds": [SEED]},
                "pl_trainer_args": {"precision": precision},
            },
            sort_keys=True,
        )
    )
    completed = []

    def predict_step(self, batch, batch_idx):
        if not batch.get("valid_sample") or batch.get("repeated_sample"):
            return None
        if completed:
            raise RuntimeError("expected exactly one native query batch")
        completed.append(True)
        seed = batch["seed"].cpu().tolist()
        batch["seed"] = seed
        self.reseed(seed[0])
        config = self.config.to_dict()
        control.save(
            out / "effective-model.json",
            {
                "shared": config["architecture"]["shared"],
                "settings": config["settings"],
            },
        )
        with (out / "input.npz").open("xb") as stream:
            np.savez_compressed(stream, **flatten(batch))
        recorder, census = DrawRecorder(), KernelCensus()
        rng_patches = [
            patch.object(torch, name, recorder.wrap(name, getattr(torch, name)))
            for name in RECORDED_DRAWS
        ]
        torch.cuda.synchronize()
        try:
            for active in (*rng_patches, *census.patches()):
                active.start()
            batch, outputs = self(batch)
            torch.cuda.synchronize()
        finally:
            for active in (*rng_patches, *census.patches()):
                try:
                    active.stop()
                except RuntimeError:
                    pass
        with (out / "tape.npz").open("xb") as stream:
            np.savez_compressed(stream, **recorder.archive())
        coordinates = outputs["atom_positions_predicted"].detach().cpu().numpy()
        if coordinates.ndim != 4 or coordinates.shape[0] != 1:
            raise ValueError(f"unexpected native coordinate shape {coordinates.shape}")
        with (out / "coordinate.npz").open("xb") as stream:
            np.savez_compressed(stream, coordinate=coordinates[0].astype(np.float32))
        control.save(
            out / "kernel-calls.json",
            {
                "calls": census.calls,
                "draws": [
                    {
                        "name": name,
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                    }
                    for name, array, _ in recorder.draws
                ],
                "seed": seed,
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuequivariance": _versions(),
                "scope": (
                    "draw and kernel census inside the single native forward; "
                    "confidence scoring runs after the tape closes"
                ),
            },
        )
        confidence_scores = self._compute_confidence_scores(batch, outputs)
        outputs["confidence_scores"] = confidence_scores
        return batch, outputs

    with patch.object(OpenFold3AllAtom, "predict_step", predict_step):
        cli.main(
            args=[
                "predict",
                "--query_json", str(input_json),
                "--inference_ckpt_path", str(CHECKPOINT),
                "--num_diffusion_samples", "5",
                "--runner_yaml", str(runner),
                "--use_msa_server", "false",
                "--use_templates", "false",
                "--output_dir", str(out / "predictions"),
            ],
            standalone_mode=False,
        )
    if not completed:
        raise RuntimeError("native predict_step never received a valid batch")
    return out


def _versions():
    versions = {}
    for name in ("cuequivariance", "cuequivariance_torch", "cuequivariance_ops_torch"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as error:  # noqa: BLE001 - a census, not a gate
            versions[name] = f"unavailable: {type(error).__name__}"
    return versions
