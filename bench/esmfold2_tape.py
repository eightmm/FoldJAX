"""Capture/classify ESMFold2's native random tape; full replay is pending.

This is development evidence, not a normal benchmark.  Capture requires the
publisher's torch/transformers checkout; replay deliberately imports neither.
Replay fails before weight loading unless the core supports every tape input.
Both commands use ESMFold2's released settings, five samples and seed 101.
The input NPZ is the already-built native feature dictionary, so this compares
the shared core only, not two independent preprocessing implementations.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

SAMPLES = 5
SEED = 101


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: np.asarray(loaded[name]) for name in loaded.files}


def _save_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


def _schedule_steps(model: Any) -> int:
    """Read the native schedule with sample()'s released 256-sigma cap."""
    schedule = model.structure_head.inference_noise_schedule()
    return int((schedule <= 256.0).sum().item())


def _model_features(features: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    from foldjax.models.esmfold2.data.all_atom import OUTPUT_METADATA_FEATURES

    return {
        key: value
        for key, value in features.items()
        if key not in OUTPUT_METADATA_FEATURES
    }


@dataclass(frozen=True)
class TapeShapes:
    batch: int
    tokens: int
    atoms: int
    pair_width: int
    samples: int
    loops: int
    diffusion_steps: int
    msa_depth: int | None
    max_msa_depth: int | None
    msa_column_mask_rate: float


def classify_random_events(
    shapes: TapeShapes,
    *,
    initial_pair: list[np.ndarray],
    dropout: list[np.ndarray],
    rand: list[np.ndarray],
    randperm: list[np.ndarray],
    normal: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """Validate and name forward-only native draws.

    This intentionally accepts no "close enough" shape.  A changed upstream
    random consumer must be inspected before its tape can be replayed.
    """

    pair_shape = (shapes.batch, shapes.tokens, shapes.tokens, shapes.pair_width)
    atom_shape = (shapes.batch * shapes.samples, shapes.atoms, 3)
    rotation_shape = (shapes.batch * shapes.samples, 4)
    translation_shape = (shapes.batch * shapes.samples, 1, 3)
    if len(initial_pair) != 1 or tuple(initial_pair[0].shape) != pair_shape:
        raise ValueError(
            f"expected one initial pair draw {pair_shape}, got {[x.shape for x in initial_pair]}"
        )
    if len(dropout) != shapes.loops or any(
        tuple(x.shape) != pair_shape for x in dropout
    ):
        raise ValueError(
            f"expected {shapes.loops} LM dropout masks {pair_shape}, got {[x.shape for x in dropout]}"
        )
    expected_rand = (
        1 if (shapes.msa_depth or 0) > 1 and shapes.msa_column_mask_rate > 0 else 0
    )
    token_shape = (shapes.batch, shapes.tokens)
    if len(rand) != expected_rand or any(tuple(x.shape) != token_shape for x in rand):
        raise ValueError(
            f"expected {expected_rand} MSA column draws {token_shape}, got {[x.shape for x in rand]}"
        )
    needs_rows = bool(
        shapes.msa_depth
        and shapes.max_msa_depth
        and shapes.msa_depth > shapes.max_msa_depth
    )
    expected_rows = shapes.loops if needs_rows else 0
    row_shape = ((shapes.msa_depth or 0) - 1,)
    if len(randperm) != expected_rows or any(
        tuple(x.shape) != row_shape for x in randperm
    ):
        raise ValueError(
            f"expected {expected_rows} MSA row permutations {row_shape}, got {[x.shape for x in randperm]}"
        )
    expected_normals = 1 + 3 * shapes.diffusion_steps
    if len(normal) != expected_normals:
        raise ValueError(
            f"expected {expected_normals} diffusion normal draws, got {[x.shape for x in normal]}"
        )
    if tuple(normal[0].shape) != atom_shape:
        raise ValueError(
            f"initial diffusion draw must be {atom_shape}, got {normal[0].shape}"
        )
    rotations, translations, churn = normal[1::3], normal[2::3], normal[3::3]
    if any(tuple(x.shape) != rotation_shape for x in rotations):
        raise ValueError("unexpected diffusion rotation draw shape")
    if any(tuple(x.shape) != translation_shape for x in translations):
        raise ValueError("unexpected diffusion translation draw shape")
    if any(tuple(x.shape) != atom_shape for x in churn):
        raise ValueError("unexpected diffusion churn draw shape")
    result = {
        "initial_pair_state": initial_pair[0],
        "lm_dropout_masks": np.stack(dropout),
        "msa_column_keep": np.asarray(rand[0] >= shapes.msa_column_mask_rate)
        if rand
        else np.empty((0,), bool),
        "msa_row_choices": np.stack(
            [
                np.sort(
                    np.concatenate(([0], x[: (shapes.max_msa_depth or 1) - 1] + 1))
                ).astype(np.int64)
                for x in randperm
            ]
        )
        if randperm
        else np.empty((0,), np.int64),
        "diffusion_initial_normal": normal[0],
        "diffusion_rotation_quaternions": np.stack(rotations),
        "diffusion_translations": np.stack(translations),
        "diffusion_churn_normals": np.stack(churn),
    }
    return {name: value for name, value in result.items() if value.size}


class TorchRecorder:
    """Temporary torch patch set that records draws without perturbing RNG."""

    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self.initial_pair: list[np.ndarray] = []
        self.dropout: list[np.ndarray] = []
        self.rand: list[np.ndarray] = []
        self.randperm: list[np.ndarray] = []
        self.normal: list[np.ndarray] = []
        self._undo: list[tuple[Any, str, Any]] = []

    def _patch(self, target: Any, name: str, replacement: Any) -> None:
        self._undo.append((target, name, getattr(target, name)))
        setattr(target, name, replacement)

    @staticmethod
    def _array(value: Any) -> np.ndarray:
        value = value.detach().cpu()
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        return value.numpy().copy()

    def __enter__(self) -> TorchRecorder:
        torch = self.torch
        original_trunc = torch.nn.init.trunc_normal_
        original_dropout = torch.nn.functional.dropout
        original_rand, original_randperm = torch.rand, torch.randperm
        original_randn, original_randn_like = torch.randn, torch.randn_like

        def trunc(tensor: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_trunc(tensor, *args, **kwargs)
            self.initial_pair.append(self._array(result))
            return result

        def dropout(
            input: Any, p: float = 0.5, training: bool = True, inplace: bool = False
        ) -> Any:
            before = (
                torch.cuda.get_rng_state(input.device)
                if input.is_cuda
                else torch.get_rng_state()
            )
            result = original_dropout(input, p=p, training=training, inplace=inplace)
            after = (
                torch.cuda.get_rng_state(input.device)
                if input.is_cuda
                else torch.get_rng_state()
            )
            try:
                if input.is_cuda:
                    torch.cuda.set_rng_state(before, input.device)
                else:
                    torch.set_rng_state(before)
                replay = original_dropout(
                    torch.ones_like(input), p=p, training=training, inplace=False
                )
                if training and p > 0:
                    self.dropout.append(self._array(replay != 0))
            finally:
                if input.is_cuda:
                    torch.cuda.set_rng_state(after, input.device)
                else:
                    torch.set_rng_state(after)
            return result

        def rand(*args: Any, **kwargs: Any) -> Any:
            result = original_rand(*args, **kwargs)
            self.rand.append(self._array(result))
            return result

        def randperm(*args: Any, **kwargs: Any) -> Any:
            result = original_randperm(*args, **kwargs)
            self.randperm.append(self._array(result))
            return result

        def randn(*args: Any, **kwargs: Any) -> Any:
            result = original_randn(*args, **kwargs)
            self.normal.append(self._array(result))
            return result

        def randn_like(*args: Any, **kwargs: Any) -> Any:
            result = original_randn_like(*args, **kwargs)
            self.normal.append(self._array(result))
            return result

        self._patch(torch.nn.init, "trunc_normal_", trunc)
        self._patch(torch.nn.functional, "dropout", dropout)
        self._patch(torch, "rand", rand)
        self._patch(torch, "randperm", randperm)
        self._patch(torch, "randn", randn)
        self._patch(torch, "randn_like", randn_like)
        return self

    def __exit__(self, *unused: object) -> None:
        for target, name, original in reversed(self._undo):
            setattr(target, name, original)


def _capture(args: argparse.Namespace) -> None:
    upstream = args.upstream_source_root.resolve()
    source_file = (
        upstream
        / "src"
        / "transformers"
        / "models"
        / "esmfold2"
        / "modeling_esmfold2.py"
    )
    if not source_file.is_file():
        raise SystemExit(f"not an upstream transformers source root: {source_file}")
    sys.path.insert(0, str(upstream / "src"))
    import torch
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    features = _npz(args.input_features)
    config = json.loads((args.weights / "config.json").read_text())
    model = ESMFold2Model.from_pretrained(str(args.weights), load_esmc=False)
    model.load_esmc(str(args.weights / "esmc"), precision="bf16")
    model = model.to("cuda").eval()
    device_features = {
        name: torch.as_tensor(value, device="cuda")
        for name, value in _model_features(features).items()
    }
    shape = TapeShapes(
        batch=int(features["token_attention_mask"].shape[0]),
        tokens=int(features["token_attention_mask"].shape[1]),
        atoms=int(features["atom_attention_mask"].shape[1]),
        pair_width=int(config["d_pair"]),
        samples=SAMPLES,
        loops=int(config.get("num_loops", 3)) + 1,
        diffusion_steps=_schedule_steps(model),
        msa_depth=None if "msa" not in features else int(features["msa"].shape[1]),
        max_msa_depth=config.get("msa_max_depth", config.get("max_msa_depth", 1024)),
        msa_column_mask_rate=float(config.get("msa_column_mask_rate", 0.1)),
    )
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    with torch.no_grad(), TorchRecorder(torch) as recorder:
        output = model(**device_features, num_diffusion_samples=SAMPLES)
    tape = classify_random_events(
        shape,
        initial_pair=recorder.initial_pair,
        dropout=recorder.dropout,
        rand=recorder.rand,
        randperm=recorder.randperm,
        normal=recorder.normal,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_npz(args.output_dir / "features.npz", features)
    _save_npz(args.output_dir / "tape.npz", tape)
    _save_npz(
        args.output_dir / "upstream_coords.npz",
        {"coords": output["sample_atom_coords"].float().cpu().numpy()},
    )
    _write_metadata(
        args,
        config,
        {
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
        },
    )


def _write_metadata(
    args: argparse.Namespace, config: Mapping[str, Any], versions: Mapping[str, str]
) -> None:
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "samples": SAMPLES,
                "core_only_shared_features": True,
                "input_sha256": _sha256(args.input_features),
                "precision": {
                    "trunk": "float32",
                    "language_model": "bfloat16",
                    "matmul": "highest",
                },
                "tape_sha256": _sha256(
                    args.tape
                    if args.command == "replay"
                    else args.output_dir / "tape.npz"
                ),
                "config": config,
                "versions": dict(versions),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _replay(args: argparse.Namespace) -> None:
    # Keep this function torch-free: the installed FoldJAX package must remain so.
    import inspect

    import jax

    from foldjax.models.esmfold2 import inference
    from foldjax.models.esmfold2.models import model as structure_model

    required = {
        "initial_pair_state", "lm_dropout_masks", "msa_column_keep",
        "msa_row_choices", "diffusion_initial_normal",
        "diffusion_rotation_quaternions", "diffusion_translations",
        "diffusion_churn_normals",
    }
    missing = required - set(inspect.signature(structure_model.predict).parameters)
    if missing:
        raise RuntimeError(
            "ESMFold2 full tape replay is not implemented by this installed core; "
            f"missing inputs: {sorted(missing)}. Capture/classification only."
        )
    jax.config.update("jax_default_matmul_precision", "highest")
    capture = json.loads((args.tape.parent / "metadata.json").read_text())
    if capture["input_sha256"] != _sha256(args.input_features):
        raise ValueError("replay input differs from captured input")
    if capture["tape_sha256"] != _sha256(args.tape):
        raise ValueError("replay tape differs from capture manifest")
    if capture["config"] != json.loads((args.weights / "config.json").read_text()):
        raise ValueError("replay checkpoint configuration differs from capture")
    features, tape = _npz(args.input_features), _npz(args.tape)
    loaded = inference.load(args.weights, dtype="float32", esmc_dtype="bfloat16")
    lm = inference.language_model_states(features, loaded)
    settings = replace(
        structure_model.with_overrides(loaded.settings, num_samples=SAMPLES),
        # The released torch loader keeps the folding trunk at checkpoint FP32.
        trunk_dtype="float32",
    )
    arrays = {
        name: jax.numpy.asarray(value)
        for name, value in _model_features(features).items()
    }
    predict = jax.jit(structure_model.predict, static_argnames=("settings", "n_chains"))
    output = predict(
        jax.random.key(SEED),
        arrays,
        loaded.parameters,
        settings=settings,
        lm_hidden_states=lm,
        n_chains=int(features["asym_id"].max()) + 1,
        **{name: jax.numpy.asarray(value) for name, value in tape.items()},
    )
    jax.block_until_ready(output["sample_atom_coords"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_npz(
        args.output_dir / "jax_coords.npz",
        {
            "coords": np.asarray(output["sample_atom_coords"]),
        },
    )
    _write_metadata(
        args,
        json.loads((args.weights / "config.json").read_text()),
        {"jax": jax.__version__},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("capture", "replay"):
        item = sub.add_parser(name)
        item.add_argument("--input-features", type=Path, required=True)
        item.add_argument("--weights", type=Path, required=True)
        item.add_argument("--output-dir", type=Path, required=True)
        if name == "capture":
            item.add_argument("--upstream-source-root", type=Path, required=True)
        else:
            item.add_argument("--tape", type=Path, required=True)
    args = parser.parse_args()
    (_capture if args.command == "capture" else _replay)(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
