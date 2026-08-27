"""High-level Python API: sequences / YAML -> predicted structure.

Wraps featurization + the JAX sampler so other code (incl. other JAX models)
can call Boltz-2 inference without the CLI. JAX and the heavier chemistry
modules are imported lazily, so ``import foldjax.models.boltz2`` stays cheap.

  import foldjax.models.boltz2
  out = foldjax.models.boltz2.predict(seq=["MKQLED..."], ligand_ccd=["ATP"],
                          weights="outputs/native_weights/boltz2_conf",
                          mols=".cache/boltz/mols")
  out["coords"]       # (n_atom, 3) np.ndarray
  out["plddt"]        # (n_atom,) np.ndarray

For composing with other JAX models, use the lower-level
``boltz2_predict`` (a pure JAX function over params + feature pytree)
and ``foldjax.models.boltz2.load_params`` directly.
"""

from __future__ import annotations

import functools
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from foldjax.models import _capture, _representations
from foldjax.models._output_validation import require_finite_coordinates
from foldjax.models.boltz2.data.featurize import featurize_yaml
from foldjax.models.boltz2.data.job_yaml import build_job_yaml
from foldjax.schema import PaddingConfig

#: Trunk dtypes `predict` accepts, as names rather than `jnp` dtypes so that
#: importing this module does not pull in JAX. fp16 is absent deliberately: the
#: sampler's `inf` constants saturate in half precision and a fully-masked
#: softmax row then gives NaN.
COMPUTE_DTYPES = ("bfloat16", "float32")

#: Matmul precision this port runs under. Boltz-2 is the one model here whose
#: upstream asks for true float32 -- `main.py:1096` is
#: `torch.set_float32_matmul_precision("highest")`, where OpenFold3, Protenix and
#: OpenDDE all select TF32 -- so "highest" is upstream parity, not caution.
MATMUL_PRECISION = "highest"


def _prefix_stable_noise_tape(
    key: Any,
    *,
    multiplicity: int,
    storage_atoms: int,
    target_atoms: int,
    steps: int,
) -> tuple[Any, Any]:
    """Draw at the existing storage stride, then suffix-pad for a bucket.

    Boltz's sampler draws flattened coordinate tensors, so changing the atom
    dimension can move values belonging to later diffusion samples.  Replaying
    these dynamic tape arguments preserves the exact default-path stream,
    including masked storage atoms between real input and a new serving bucket.
    Only the newly appended suffix is zeroed.
    """

    import jax
    import jax.numpy as jnp

    if multiplicity <= 0:
        raise ValueError("noise tape multiplicity must be positive")
    if steps <= 0:
        raise ValueError("noise tape steps must be positive")
    if storage_atoms < 0 or target_atoms < storage_atoms:
        raise ValueError(
            f"noise tape target atoms {target_atoms} is smaller than storage "
            f"atoms {storage_atoms}"
        )

    padding = ((0, 0), (0, target_atoms - storage_atoms), (0, 0))

    def draw(draw_key):
        real = jax.random.normal(
            draw_key,
            (multiplicity, storage_atoms, 3),
            dtype=jnp.float32,
        )
        return jnp.pad(real, padding)

    run_key, init_key = jax.random.split(key)
    init_noise = draw(init_key)
    step_noises = []
    for _ in range(steps):
        run_key, noise_key = jax.random.split(run_key)
        step_noises.append(draw(noise_key))
    return init_noise, jnp.stack(step_noises, axis=0)


def _prefix_rng_is_supported() -> bool:
    """Whether masked padding draws preserve JAX's compact random prefix."""

    from foldjax.models._random import supports_masked_prefix_draw

    return supports_masked_prefix_draw()


def _storage_prefix_size(*, storage_atoms: int, target_atoms: int) -> np.ndarray:
    """Return Boltz's dynamic storage stride after validating its bucket."""

    if storage_atoms < 0 or target_atoms < storage_atoms:
        raise ValueError(
            f"noise target atoms {target_atoms} is smaller than storage atoms "
            f"{storage_atoms}"
        )
    return np.asarray(storage_atoms, dtype=np.int32)


def _pinned_matmul_precision(function):
    """Run `function` with the matmul precision scoped, not latched.

    This used to be a `jax.config.update` inside `predict`, which is
    process-global and stayed set after the call returned: a notebook that ran
    Boltz-2 and then anything else left that other thing in float32. It went
    unnoticed while every port pinned the same value; they no longer do.

    JAX is imported inside the wrapper rather than at module scope, because
    importing this module has to stay cheap and torch-free -- `test_import`
    asserts it.
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        import jax

        from foldjax.execution import resolved_matmul_precision

        with jax.default_matmul_precision(
            resolved_matmul_precision(MATMUL_PRECISION)
        ):
            return function(*args, **kwargs)

    return wrapper


def _graph_features(
    feats: Mapping[str, np.ndarray], *, place: bool
) -> dict[str, object]:
    """Prepare the featurizer's arrays for the model call.

    ``place=False`` (the compiled path) leaves them as NumPy. ``jax.jit`` drops
    every argument the traced program never reads, and it drops them *before*
    the remaining ones are transferred, so a feature the model does not use is
    never copied to the device at all. Placing the dict first defeats that: the
    buffers exist for the whole run whether or not the program takes them.

    Measured at 1,003 tokens: the graph reads 31 of the featurizer's 78 arrays.
    The other 47 are 306 MiB of the 541 MiB that used to be placed, and
    ``disto_target`` -- a training label, ``f32[N, N, 1, 64]``, read by no
    inference path -- is 246 MiB of that on its own. The term is quadratic in
    token count, so it grows with the input.

    Argument dtypes canonicalize identically from a NumPy array and from a
    placed one, so the traced jaxpr and the compiled program are unchanged: the
    StableHLO of the whole L1000 graph hashes the same either way.

    ``place=True`` is for the two callers that cannot use the pruning: steering
    runs eagerly (no jit, so no DCE, and op-by-op dispatch would re-transfer
    per op) and context parallelism places every input at its own ownership
    boundary.
    """

    import jax.numpy as jnp

    convert = jnp.asarray if place else np.asarray
    return {name: convert(value) for name, value in feats.items()}


def featurize(
    *,
    input: str | Path | None = None,
    seq: Sequence[str] = (),
    dna: Sequence[str] = (),
    rna: Sequence[str] = (),
    ligand_ccd: Sequence[str] = (),
    ligand_smiles: Sequence[str] = (),
    mols: str | Path,
    out_dir: str | Path | None = None,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    msa_server_username: str | None = None,
    msa_server_password: str | None = None,
    msa_api_key_header: str | None = None,
    msa_api_key_value: str | None = None,
    feature_cache: str | Path | None = None,
    max_msa_depth: int | None = None,
) -> tuple[dict[str, np.ndarray], str, Path]:
    """Featurize a YAML path or bare entities.

    Returns ``(feats, record_id, struct_dir)``.
    """
    if use_msa_server:
        if msa_server_username is None:
            msa_server_username = os.environ.get("BOLTZ_MSA_USERNAME")
        if msa_server_password is None:
            msa_server_password = os.environ.get("BOLTZ_MSA_PASSWORD")
        if msa_api_key_value is None:
            msa_api_key_value = os.environ.get("MSA_API_KEY_VALUE")
    work = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp())
    if input is None:
        if not (seq or dna or rna or ligand_ccd or ligand_smiles):
            raise ValueError("provide `input` YAML or sequences/ligands")
        work.mkdir(parents=True, exist_ok=True)
        input = work / "job.yaml"
        Path(input).write_text(
            build_job_yaml(
                seq,
                ligand_ccd,
                dna=dna,
                rna=rna,
                ligands_smiles=ligand_smiles,
                use_msa_server=use_msa_server,
            )
        )
    feats, manifest, struct_dir = featurize_yaml(
        Path(input),
        work,
        Path(mols),
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        msa_api_key_header=msa_api_key_header,
        msa_api_key_value=msa_api_key_value,
        cache_dir=Path(feature_cache) if feature_cache is not None else None,
        max_msa_depth=max_msa_depth,
    )
    if manifest is not None:
        record_id = manifest.records[0].id
    else:
        record_id = sorted(struct_dir.glob("*.npz"))[0].stem
    return feats, record_id, struct_dir


def _resolve_cp_layout(layout: str, cp_devices: int) -> str:
    """Resolve the context-parallel layout, expanding ``"auto"``.

    ``"1d"`` splits pair rows only; ``"2d"`` is Fold-CP's square grid, which
    splits both pair axes and needs a perfect-square device count. ``"auto"``
    resolves to ``"1d"`` -- see the comment on that branch for why the grid is
    opt-in rather than automatic.

    An explicit ``"2d"`` on a non-square count is refused here rather than left
    to ``context_parallel``, which is only reached after featurization -- an
    MSA search is a long way to go to be told the device count was the wrong
    shape.
    """
    if layout not in ("auto", "1d", "2d"):
        raise ValueError(f"cp_layout must be one of 'auto', '1d', '2d'; got {layout!r}")
    side = math.isqrt(cp_devices)
    square = cp_devices > 1 and side * side == cp_devices
    if layout == "auto":
        # "auto" stays on the 1-D layout. The square grid is the better design
        # and is gated on CPU meshes at 2x2 and 3x3, but every published
        # measurement of this feature was taken on the 1-D layout, and a
        # default that silently changes the program would make those numbers
        # describe a configuration nobody can reproduce. It flips once the
        # grid has its own GPU evidence; "2d" asks for it explicitly.
        return "1d"
    if layout == "2d" and not square:
        raise ValueError(
            "cp_layout='2d' needs a square cp_devices greater than 1 "
            f"(4, 9, 16, ...); got cp_devices={cp_devices}"
        )
    return layout


def _static_runtime_identity(value: Any) -> Any:
    """Freeze one trace-time value without accepting an unsafe cache collision."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return ("path", str(value.absolute()))
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                (str(key), _static_runtime_identity(child))
                for key, child in sorted(value.items(), key=lambda item: str(item[0]))
            ),
        )
    if isinstance(value, (list, tuple)):
        return (
            type(value).__name__,
            tuple(_static_runtime_identity(child) for child in value),
        )
    try:
        dtype = np.dtype(value)
    except TypeError:
        # An unfamiliar future option gets a per-object key. That may miss a
        # reuse opportunity, but can never replay a graph traced for a
        # semantically different value merely because their reprs collide.
        return ("opaque", type(value).__module__, type(value).__qualname__, id(value))
    return ("dtype", dtype.str, dtype.name)


def _device_runtime_identity(device: Any) -> tuple[Any, ...]:
    return tuple(
        getattr(device, name, None)
        for name in (
            "platform",
            "process_index",
            "id",
            "local_hardware_id",
            "device_kind",
        )
    )


def _parameter_runtime_identity(
    jax_module: Any,
    *,
    cp_devices: int,
    cp_layout: str,
) -> tuple[Any, ...]:
    """Settings that change the loaded dtype or retained mesh placement."""

    devices = tuple(
        _device_runtime_identity(device)
        for device in list(jax_module.devices())[:cp_devices]
    )
    return (
        ("cp_layout", cp_layout),
        ("cp_devices", cp_devices),
        ("devices", devices),
        ("default_device", str(jax_module.config.jax_default_device)),
        ("enable_x64", bool(jax_module.config.jax_enable_x64)),
    )


def _runtime_identity(
    jax_module: Any,
    *,
    cp_devices: int,
    cp_layout: str,
    compile_cache: Path | None,
    parameter_identity: tuple[Any, ...] | None = None,
) -> tuple[Any, ...]:
    """Complete identity of values captured while tracing one model graph."""

    triangle_multiplication_backend = os.environ.get(
        "BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND", "cueq"
    )
    if triangle_multiplication_backend not in {"cueq", "xla"}:
        raise ValueError(
            "BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND must be 'cueq' or "
            f"'xla'; got {triangle_multiplication_backend!r}"
        )
    if cp_devices > 1:
        triangle_multiplication_backend = "xla"
    if parameter_identity is None:
        parameter_identity = _parameter_runtime_identity(
            jax_module,
            cp_devices=cp_devices,
            cp_layout=cp_layout,
        )
    return (
        *parameter_identity,
        ("default_prng_impl", str(jax_module.config.jax_default_prng_impl)),
        (
            "dtype_promotion",
            str(jax_module.config.jax_numpy_dtype_promotion),
        ),
        ("rank_promotion", str(jax_module.config.jax_numpy_rank_promotion)),
        ("matmul_precision", str(jax_module.config.jax_default_matmul_precision)),
        ("triangle_multiplication_backend", triangle_multiplication_backend),
        (
            "compile_cache",
            str(compile_cache.resolve()) if compile_cache is not None else None,
        ),
    )


def _runner_identity(
    *,
    predict_function: Any,
    predict_kwargs: Mapping[str, Any],
    noise_mode: str,
    runtime: tuple[Any, ...],
) -> tuple[Any, ...]:
    """Everything captured by one retained ``jax.jit`` wrapper."""

    return (
        ("model_function", id(predict_function)),
        ("noise_mode", noise_mode),
        ("kwargs", _static_runtime_identity(predict_kwargs)),
        ("runtime", runtime),
    )


@_pinned_matmul_precision
def predict(
    *,
    input: str | Path | None = None,
    seq: Sequence[str] = (),
    dna: Sequence[str] = (),
    rna: Sequence[str] = (),
    ligand_ccd: Sequence[str] = (),
    ligand_smiles: Sequence[str] = (),
    weights: str | Path,
    affinity_weights: str | Path | None = None,
    mols: str | Path,
    out_dir: str | Path | None = None,
    representations: Sequence[str] | str | None = None,
    stop_after: str = "full",
    representations_dir: str | Path | None = None,
    num_steps: int = 200,
    num_recycles: int = 3,
    num_samples: int = 1,
    affinity_num_steps: int = 200,
    affinity_num_samples: int = 5,
    affinity_mw_correction: bool = False,
    seed: int = 0,
    # Upstream's predict path is `Trainer(..., precision="bf16-mixed")`
    # (`main.py:1262`), so float32 was this port inventing a setting Boltz-2
    # does not ship. The trunk runs in this dtype; the diffusion and confidence
    # modules stay float32 either way, which is the same split upstream's
    # autocast draws. Measured at 1,003 tokens over an 8,192-row alignment,
    # five samples of the released schedule: 87.3 s -> 65.0 s warm, 13,035 ->
    # 10,218 MiB peak, and against the deposited structure TM 0.9932 -> 0.9931,
    # the two crossing in both directions per sample. Passing "float32"
    # restores the previous behaviour exactly.
    compute_dtype: str = "bfloat16",
    # False by default: the full-bin pae/pde/plddt logits and the distogram are
    # program outputs nothing in the cif/JSON path reads, and XLA keeps entry
    # outputs resident alongside the temp arena for the whole run -- at 3,012
    # tokens and 5 samples that is >20 GiB of peak for arrays that were thrown
    # away. The summaries (plddt, pae, ptm/iptm, complex_*) are unaffected.
    # Pass True to get the raw logits and pdistogram back in `raw`.
    return_confidence_logits: bool = False,
    attention_backend: str = "xla",
    triangle_backend: str = "cueq",
    glu_backend: str = "xla",
    #: Context parallelism: shard the pair representations across this many
    #: JAX devices (the JAX form of OpenDDE's Fold-CP). Needs that many
    #: visible devices; the default "cueq" triangle kernel resolves to the
    #: blocked XLA path because a fused FFI call cannot be partitioned.
    cp_devices: int = 1,
    #: How those devices are arranged. "1d" splits pair rows only; "2d" is
    #: Fold-CP's square grid, which splits both pair axes and runs the triangle
    #: multiplication as Cannon's algorithm, so no full-width operand is ever
    #: built. "auto" (the default) stays on the 1-D layout until the square
    #: grid has GPU evidence of its own; pass "2d" to ask for the grid, which
    #: then requires a square `cp_devices`.
    cp_layout: str = "auto",
    #: Distribute Boltz-2 atom query windows over CP rows. The required atom
    #: and token alignment is padded automatically and cropped from outputs.
    cp_atom_windows: bool = True,
    steering_args: Mapping[str, object] | None = None,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    msa_server_username: str | None = None,
    msa_server_password: str | None = None,
    msa_api_key_header: str | None = None,
    msa_api_key_value: str | None = None,
    feature_cache: str | Path | None = None,
    compile_cache: str | Path | None = None,
    bucket: bool = False,
    padding: PaddingConfig | None = None,
    write_fmt: str | None = None,
    max_msa_depth: int | None = None,
    _runtime: Any | None = None,
) -> dict[str, Any]:
    """Run end-to-end Boltz-2 inference.

    Pass either ``input`` (a job YAML) or bare ``seq``/``dna``/``rna``/
    ``ligand_ccd``/``ligand_smiles``. Affinity jobs automatically load a
    sibling ``boltz2_aff`` bundle unless ``affinity_weights`` is explicit.
    Returns a dict with ``coords`` (n_atom, 3), ``plddt``, ``record_id``,
    ``raw`` (full model output), and ``out_path`` (if ``write_fmt`` is
    "pdb"/"cif"). Defaults match the Boltz-2 reference.
    """
    import jax
    import jax.numpy as jnp

    from foldjax.models.boltz2.bridge.native import load_params
    from foldjax.models.boltz2.models.predict import boltz2_predict

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if padding is not None and bucket:
        raise ValueError("padding and legacy bucket=True cannot be used together")
    if cp_devices < 1:
        raise ValueError("cp_devices must be positive")
    if not isinstance(cp_atom_windows, bool):
        raise ValueError("cp_atom_windows must be a boolean")
    resolved_cp_layout = _resolve_cp_layout(cp_layout, cp_devices)
    cp_atom_active = cp_devices > 1 and cp_atom_windows and stop_after != "trunk"
    cp_rows = int(math.isqrt(cp_devices)) if resolved_cp_layout == "2d" else cp_devices
    cp_cols = int(math.isqrt(cp_devices)) if resolved_cp_layout == "2d" else 1
    if cp_devices > 1 and glu_backend != "xla":
        raise ValueError(
            "context parallelism requires glu_backend='xla'; a fused GLU "
            "cannot be partitioned"
        )

    feats_np, record_id, struct_dir = featurize(
        input=input,
        seq=seq,
        dna=dna,
        rna=rna,
        ligand_ccd=ligand_ccd,
        ligand_smiles=ligand_smiles,
        mols=mols,
        out_dir=out_dir,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        msa_api_key_header=msa_api_key_header,
        msa_api_key_value=msa_api_key_value,
        feature_cache=feature_cache,
        max_msa_depth=max_msa_depth,
    )

    cache = None
    if compile_cache is not None:
        cache = Path(compile_cache).expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(cache))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

    if compute_dtype not in COMPUTE_DTYPES:
        raise ValueError(
            f"compute_dtype must be one of {COMPUTE_DTYPES}, got {compute_dtype!r}"
        )
    dtype = {"float32": jnp.float32, "bfloat16": jnp.bfloat16}[compute_dtype]
    parameter_identity = _parameter_runtime_identity(
        jax,
        cp_devices=cp_devices,
        cp_layout=resolved_cp_layout,
    )
    runtime_identity = _runtime_identity(
        jax,
        cp_devices=cp_devices,
        cp_layout=resolved_cp_layout,
        compile_cache=cache,
        parameter_identity=parameter_identity,
    )
    confidence_weights = Path(weights)
    affinity_requested = stop_after != "trunk" and bool(
        np.any(feats_np["affinity_token_mask"])
    )
    if _runtime is not None:
        _runtime.prepare(affinity_requested=affinity_requested)
    params = (
        load_params(confidence_weights)
        if _runtime is None
        else _runtime.load_params(
            "primary",
            confidence_weights,
            load_params,
            placement=parameter_identity,
        )
    )
    affinity_model_params = None
    if affinity_requested:
        affinity_path = (
            Path(affinity_weights)
            if affinity_weights is not None
            else confidence_weights.with_name("boltz2_aff")
        )
        if not _native_weights_exist(affinity_path):
            raise FileNotFoundError(
                "affinity weights are required for an affinity job; "
                f"not found: {affinity_path}(.safetensors|.npz)"
            )
        affinity_model_params = (
            load_params(affinity_path)
            if _runtime is None
            else _runtime.load_params(
                "affinity",
                affinity_path,
                load_params,
                placement=parameter_identity,
            )
        )
        if not {"trunk", "affinity"} <= affinity_model_params.keys():
            raise ValueError(
                "native affinity weights use the obsolete head-only format; "
                "rerun scripts/setup.sh to export the complete affinity model"
            )

    steering_active = steering_args is not None and any(
        bool(steering_args.get(key, False))
        for key in (
            "fk_steering",
            "physical_guidance_update",
            "contact_guidance_update",
        )
    )
    if cp_devices > 1:
        if steering_active:
            raise ValueError(
                "context parallelism requires the compiled graph; steering "
                "runs eagerly, so drop steering or cp_devices"
            )
        if affinity_requested:
            raise ValueError(
                "affinity jobs are not supported under context parallelism "
                "yet; run the affinity job on a single device"
            )
    if padding is not None:
        from foldjax.models.boltz2.data.bucket import select_model_features_for_padding

        feats_np = select_model_features_for_padding(
            feats_np, steering_active=steering_active
        )
    elif not steering_active:
        from foldjax.models.boltz2.data.bucket import select_model_features

        feats_np = select_model_features(feats_np)

    original_tokens = int(feats_np["token_pad_mask"].shape[-1])
    original_atoms = int(feats_np["atom_pad_mask"].shape[-1])
    padding_plan = None
    if padding is not None or bucket:
        from foldjax.models.boltz2.data.bucket import (
            resolve_legacy_padding_plan,
            resolve_padding_plan,
        )

        padding_plan = (
            resolve_padding_plan(feats_np, padding)
            if padding is not None
            else resolve_legacy_padding_plan(feats_np)
        )
    if cp_atom_active:
        from foldjax.models.boltz2.data.bucket import (
            align_padding_plan_for_context_parallel,
        )

        padding_plan = align_padding_plan_for_context_parallel(
            feats_np,
            padding_plan,
            cp_rows=cp_rows,
            cp_cols=cp_cols,
        )
    if padding_plan is not None:
        from foldjax.models.boltz2.data.bucket import pad_feats

        feats_np, _ = pad_feats(
            feats_np,
            padding_plan.target["tokens"],
            padding_plan.target["atoms"],
            target_msa=padding_plan.target["msa"],
        )

    if cp_atom_active:
        atom_to_token = np.asarray(feats_np["atom_to_token"])
        atom_valid = np.any(atom_to_token > 0, axis=-1)
        atom_ids = np.argmax(atom_to_token, axis=-1).astype(np.int32)
        feats_np["atom_to_token_ids_global"] = np.where(
            atom_valid, atom_ids, -1
        ).astype(np.int32)
        feats_np["atom_to_token_valid"] = atom_valid

    # Only the eager steering path and the context-parallel path need the
    # features placed; the compiled path lets `jax.jit` prune the ones the
    # graph never reads before anything is transferred (see `_graph_features`).
    feats = _graph_features(feats_np, place=steering_active or cp_devices > 1)
    padding_noise_mode = "none"
    padding_noise_arg = None
    if (
        stop_after != "trunk"
        and padding_plan is not None
        and padding_plan.target["atoms"] > padding_plan.storage["atoms"]
    ):
        if _prefix_rng_is_supported():
            padding_noise_mode = "storage_prefix"
            padding_noise_arg = _storage_prefix_size(
                storage_atoms=padding_plan.storage["atoms"],
                target_atoms=padding_plan.target["atoms"],
            )
        else:
            padding_noise_mode = "tape"
            padding_noise_arg = _prefix_stable_noise_tape(
                jax.random.PRNGKey(seed),
                multiplicity=num_samples,
                storage_atoms=padding_plan.storage["atoms"],
                target_atoms=padding_plan.target["atoms"],
                steps=num_steps,
            )
    wanted_representations = _representations.resolve(
        representations, _representations.specs_for("boltz2")
    )
    predict_kwargs = {
        "recycling_steps": num_recycles,
        "num_sampling_steps": num_steps,
        "augmentation": False,
        "steering_args": steering_args,
        "run_confidence": True,
        "return_representations": wanted_representations,
        "stop_after_trunk": stop_after == "trunk",
        "run_distogram": return_confidence_logits,
        "return_confidence_logits": return_confidence_logits,
        "run_bfactor": True,
        "compute_dtype": dtype,
        "use_scan": True,
        "trunk_use_scan": True,
        "score_use_scan": True,
        "multiplicity": num_samples,
        "atom_context_parallel": cp_atom_active,
        "attention_backend": attention_backend,
        "triangle_backend": (
            # The fused default cannot be partitioned; unset-equivalent
            # resolves to the blocked XLA path under context parallelism.
            "xla" if cp_devices > 1 and triangle_backend == "cueq" else triangle_backend
        ),
        "glu_backend": glu_backend,
        "confidence_sequentially": num_samples > 1,
        "return_pair_chains_iptm": False,
        "recompute_nonpolymer_frames": bool(np.any(feats_np["mol_type"] == 3)),
        # The featurizer always emits template_* arrays, zero-filled when no
        # template was given. Whether they are real is a value question, so it
        # is answered here on concrete arrays rather than inside the jit.
        "use_template": bool(np.any(feats_np.get("template_mask", np.zeros(1)) != 0)),
    }
    primary_static = {
        "use_template": predict_kwargs["use_template"],
        "recompute_nonpolymer_frames": predict_kwargs["recompute_nonpolymer_frames"],
    }
    if padding_noise_mode == "tape":

        def run_model(model_params, model_feats, model_key, init_noise, step_noises):
            return boltz2_predict(
                model_params,
                model_feats,
                model_key,
                init_noise=init_noise,
                step_noises=step_noises,
                **predict_kwargs,
            )

        model_args = (
            params,
            feats,
            jax.random.PRNGKey(seed),
            *padding_noise_arg,
        )
    elif padding_noise_mode == "storage_prefix":

        def run_model(model_params, model_feats, model_key, noise_storage_atoms):
            return boltz2_predict(
                model_params,
                model_feats,
                model_key,
                noise_storage_atoms=noise_storage_atoms,
                **predict_kwargs,
            )

        model_args = (
            params,
            feats,
            jax.random.PRNGKey(seed),
            padding_noise_arg,
        )
    else:

        def run_model(model_params, model_feats, model_key):
            return boltz2_predict(
                model_params, model_feats, model_key, **predict_kwargs
            )

        model_args = (params, feats, jax.random.PRNGKey(seed))

    from foldjax.models._cp import context_parallel, replicate_tree
    from foldjax.models._cp_atom import place_atoms

    if steering_active:
        runner = run_model
    elif _runtime is None:
        runner = jax.jit(run_model)
    else:
        runner = _runtime.jit_runner(
            "primary",
            _runner_identity(
                predict_function=boltz2_predict,
                predict_kwargs=predict_kwargs,
                noise_mode=padding_noise_mode,
                runtime=runtime_identity,
            ),
            run_model,
            jax.jit,
        )
    # A tap records a tracer of the graph being built, so the capture set
    # has to be live while the program is traced, not while it runs.
    with (
        _capture.capturing(wanted_representations),
        context_parallel(cp_devices, layout=resolved_cp_layout),
    ):
        if cp_devices > 1:
            # Place each semantic input at its production ownership boundary.
            # Parameters and RNG keys are replicated, pair inputs may use the
            # pair mesh, and safe Boltz atom features/noise tapes enter already
            # sharded over CP rows. Dense coupled atom/token maps stay replicated.
            placed_args = list(model_args)
            place_params = lambda tree: replicate_tree(  # noqa: E731
                tree,
                shard_pair_features=False,
            )
            placed_args[0] = (
                place_params(placed_args[0])
                if _runtime is None
                else _runtime.place_params(
                    "primary",
                    placed_args[0],
                    placement=parameter_identity,
                    placer=place_params,
                )
            )
            # On the first seed this drops the loader's single-device owner as
            # soon as the mesh-placed tree has replaced it in the session.
            params = placed_args[0]
            placed_args[1] = replicate_tree(
                placed_args[1],
                shard_atom_features=cp_atom_active,
            )
            placed_args[2] = replicate_tree(
                placed_args[2],
                shard_pair_features=False,
            )
            if cp_atom_active and padding_noise_mode == "tape":
                placed_args[3] = place_atoms(placed_args[3], atom_axis=-2)
                placed_args[4] = place_atoms(placed_args[4], atom_axis=-2)
            elif padding_noise_mode == "storage_prefix":
                placed_args[3] = replicate_tree(
                    placed_args[3], shard_pair_features=False
                )
            model_args = tuple(placed_args)
        out = runner(*model_args)
    if stop_after == "trunk":
        # The trunk-only graph intentionally has no sampler or confidence
        # outputs. Crop captured padded representations before touching any
        # coordinate field, persist them, and return immediately.
        from foldjax.models.boltz2.data.bucket import crop_prediction_outputs

        public_tokens = (
            padding_plan.actual["tokens"]
            if padding_plan is not None
            else original_tokens
        )
        public_atoms = (
            padding_plan.actual["atoms"] if padding_plan is not None else original_atoms
        )
        public_out = (
            crop_prediction_outputs(out, public_tokens, public_atoms)
            if padding_plan is not None
            else out
        )
        destination = (
            Path(representations_dir)
            if representations_dir is not None
            else (Path(out_dir) if out_dir is not None else struct_dir.parent)
        )
        archive = _representations.save(
            destination,
            {
                name: public_out[name]
                for name in wanted_representations
                if name in public_out
            },
            _representations.specs_for("boltz2"),
            model="boltz2",
        )
        result: dict[str, Any] = {
            "record_id": record_id,
            "raw": public_out,
            "representations": archive,
        }
        if padding_plan is not None:
            primary_summary = padding_plan.summary()
            primary_summary["static"] = primary_static
            result["padding"] = {"primary": primary_summary}
        return result
    coords_batched = require_finite_coordinates(
        jax.block_until_ready(out["sample_atom_coords"]),
        model="Boltz2",
        atom_mask=feats_np["atom_pad_mask"],
    ).reshape(num_samples, -1, 3)
    plddt_batched = np.asarray(out["plddt"]).reshape(num_samples, -1)

    affinity_padding_plan = None
    if affinity_model_params is not None:
        best_idx = int(np.argmax(np.asarray(out["iptm"])))
        affinity_feats_np = _prepare_affinity_features(
            input_path=Path(input) if input is not None else None,
            struct_dir=struct_dir,
            record_id=record_id,
            mol_dir=Path(mols),
            predicted_coords=coords_batched[best_idx : best_idx + 1],
            atom_pad_mask=feats_np["atom_pad_mask"],
            out_dir=(Path(out_dir) if out_dir is not None else struct_dir.parent)
            / "affinity_stage",
            use_msa_server=use_msa_server,
            msa_server_url=msa_server_url,
            msa_pairing_strategy=msa_pairing_strategy,
            msa_server_username=msa_server_username,
            msa_server_password=msa_server_password,
            msa_api_key_header=msa_api_key_header,
            msa_api_key_value=msa_api_key_value,
            max_msa_depth=max_msa_depth,
        )
        if padding is None:
            from foldjax.models.boltz2.data.bucket import select_model_features

            affinity_feats_np = select_model_features(affinity_feats_np)
        if padding is not None or bucket:
            from foldjax.models.boltz2.data.bucket import (
                pad_feats,
                resolve_legacy_padding_plan,
                resolve_padding_plan,
                select_model_features_for_padding,
            )

            if padding is not None:
                affinity_feats_np = select_model_features_for_padding(
                    affinity_feats_np, steering_active=False
                )
            affinity_padding_plan = (
                resolve_padding_plan(affinity_feats_np, padding)
                if padding is not None
                else resolve_legacy_padding_plan(affinity_feats_np)
            )
            affinity_feats_np, _ = pad_feats(
                affinity_feats_np,
                affinity_padding_plan.target["tokens"],
                affinity_padding_plan.target["atoms"],
                target_msa=affinity_padding_plan.target["msa"],
            )
        # `run_affinity` below is always jitted -- affinity is rejected under
        # context parallelism and never runs the eager steering path -- so this
        # dict can always take the pruning.
        affinity_feats = _graph_features(affinity_feats_np, place=False)
        affinity_kwargs = {
            **predict_kwargs,
            "recycling_steps": 5,
            "num_sampling_steps": affinity_num_steps,
            "steering_args": None,
            "multiplicity": affinity_num_samples,
            "run_bfactor": "bfactor" in affinity_model_params,
            "confidence_sequentially": affinity_num_samples > 1,
            "recompute_nonpolymer_frames": True,
            "affinity_mw_correction": affinity_mw_correction,
            "use_template": (
                bool(np.any(affinity_feats_np.get("template_mask", np.zeros(1)) != 0))
                if padding is not None
                else predict_kwargs["use_template"]
            ),
        }
        affinity_static = {
            "use_template": affinity_kwargs["use_template"],
            "recompute_nonpolymer_frames": affinity_kwargs[
                "recompute_nonpolymer_frames"
            ],
        }
        if padding is not None and affinity_padding_plan is not None:
            if _prefix_rng_is_supported():
                affinity_noise_mode = "storage_prefix"
                affinity_noise_arg = _storage_prefix_size(
                    storage_atoms=affinity_padding_plan.storage["atoms"],
                    target_atoms=affinity_padding_plan.target["atoms"],
                )
            else:
                affinity_noise_mode = "tape"
                affinity_noise_arg = _prefix_stable_noise_tape(
                    jax.random.PRNGKey(seed),
                    multiplicity=affinity_num_samples,
                    storage_atoms=affinity_padding_plan.storage["atoms"],
                    target_atoms=affinity_padding_plan.target["atoms"],
                    steps=affinity_num_steps,
                )

        else:
            affinity_noise_mode = "none"
            affinity_noise_arg = None

        if affinity_noise_mode == "tape":

            def run_affinity(
                model_params, model_feats, model_key, init_noise, step_noises
            ):
                return boltz2_predict(
                    model_params,
                    model_feats,
                    model_key,
                    init_noise=init_noise,
                    step_noises=step_noises,
                    **affinity_kwargs,
                )

            affinity_args = (
                affinity_model_params,
                affinity_feats,
                jax.random.PRNGKey(seed),
                *affinity_noise_arg,
            )
        elif affinity_noise_mode == "storage_prefix":

            def run_affinity(
                model_params, model_feats, model_key, noise_storage_atoms
            ):
                return boltz2_predict(
                    model_params,
                    model_feats,
                    model_key,
                    noise_storage_atoms=noise_storage_atoms,
                    **affinity_kwargs,
                )

            affinity_args = (
                affinity_model_params,
                affinity_feats,
                jax.random.PRNGKey(seed),
                affinity_noise_arg,
            )
        else:

            def run_affinity(model_params, model_feats, model_key):
                return boltz2_predict(
                    model_params, model_feats, model_key, **affinity_kwargs
                )

            affinity_args = (
                affinity_model_params,
                affinity_feats,
                jax.random.PRNGKey(seed),
            )

        affinity_runner = (
            jax.jit(run_affinity)
            if _runtime is None
            else _runtime.jit_runner(
                "affinity",
                _runner_identity(
                    predict_function=boltz2_predict,
                    predict_kwargs=affinity_kwargs,
                    noise_mode=affinity_noise_mode,
                    runtime=runtime_identity,
                ),
                run_affinity,
                jax.jit,
            )
        )
        affinity_out = affinity_runner(*affinity_args)
        out.update(
            {
                key: value
                for key, value in affinity_out.items()
                if key.startswith("affinity_")
            }
        )

    from foldjax.models.boltz2.data.bucket import crop_prediction_outputs

    public_tokens = (
        padding_plan.actual["tokens"] if padding_plan is not None else original_tokens
    )
    public_atoms = (
        padding_plan.actual["atoms"] if padding_plan is not None else original_atoms
    )
    public_out = (
        crop_prediction_outputs(out, public_tokens, public_atoms)
        if padding_plan is not None
        else out
    )
    public_coords_batched = (
        coords_batched[:, :public_atoms] if padding_plan is not None else coords_batched
    )
    public_plddt_batched = (
        plddt_batched[:, :public_tokens] if padding_plan is not None else plddt_batched
    )
    public_coords = (
        public_coords_batched[0] if num_samples == 1 else public_coords_batched
    )
    public_plddt = (
        public_plddt_batched[0] if num_samples == 1 else public_plddt_batched
    )
    result: dict[str, Any] = {
        "coords": public_coords,
        "plddt": public_plddt,
        "record_id": record_id,
        "raw": public_out,
    }
    if padding_plan is not None:
        primary_summary = padding_plan.summary()
        primary_summary["static"] = primary_static
        padding_summary: dict[str, Any] = {"primary": primary_summary}
        if affinity_padding_plan is not None:
            affinity_summary = affinity_padding_plan.summary()
            affinity_summary["static"] = affinity_static
            padding_summary["affinity"] = affinity_summary
        result["padding"] = padding_summary
    result.update(
        {key: value for key, value in public_out.items() if key.startswith("affinity_")}
    )
    if wanted_representations:
        destination = (
            Path(representations_dir)
            if representations_dir is not None
            else (Path(out_dir) if out_dir is not None else struct_dir.parent)
        )
        archive = _representations.save(
            destination,
            {
                name: public_out[name]
                for name in wanted_representations
                if name in public_out
            },
            _representations.specs_for("boltz2"),
            model="boltz2",
        )
        if archive is not None:
            print(f"wrote {archive}")
    if write_fmt is not None:
        from foldjax.models.boltz2.data.write.structure import write_prediction

        dest = Path(out_dir) if out_dir is not None else struct_dir.parent
        dest.mkdir(parents=True, exist_ok=True)
        paths = []
        for model_idx in range(num_samples):
            suffix = "" if num_samples == 1 else f"_model_{model_idx}"
            out_path = dest / f"{record_id}{suffix}.{write_fmt}"
            paths.append(
                write_prediction(
                    structure_npz=struct_dir / f"{record_id}.npz",
                    coords=coords_batched[model_idx],
                    atom_pad_mask=feats_np["atom_pad_mask"].reshape(-1),
                    out_path=out_path,
                    plddts=public_plddt_batched[model_idx],
                    fmt=write_fmt,
                )
            )
        if num_samples == 1:
            result["out_path"] = paths[0]
        else:
            result["out_paths"] = paths
    return result


def _native_weights_exist(path: Path) -> bool:
    from foldjax.models.boltz2.weights import resolve_native_weight_bundle

    return resolve_native_weight_bundle(path) is not None


def _prepare_affinity_features(
    *,
    input_path: Path | None,
    struct_dir: Path,
    record_id: str,
    mol_dir: Path,
    predicted_coords: np.ndarray,
    atom_pad_mask: np.ndarray,
    out_dir: Path,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    msa_server_username: str | None = None,
    msa_server_password: str | None = None,
    msa_api_key_header: str | None = None,
    msa_api_key_value: str | None = None,
    max_msa_depth: int | None = None,
) -> dict[str, np.ndarray]:
    from foldjax.models.boltz2.data.featurize import (
        featurize_affinity_from_prediction,
        featurize_yaml,
    )
    from foldjax.models.boltz2.data.types import Manifest

    processed_dir = struct_dir.parent
    manifest_path = processed_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = Manifest.load(manifest_path)
    else:
        if input_path is None:
            raise ValueError(
                "affinity feature-cache recovery requires the original input YAML"
            )
        _, manifest, regenerated_struct_dir = featurize_yaml(
            input_path,
            out_dir / "source",
            mol_dir,
            use_msa_server=use_msa_server,
            msa_server_url=msa_server_url,
            msa_pairing_strategy=msa_pairing_strategy,
            msa_server_username=msa_server_username,
            msa_server_password=msa_server_password,
            msa_api_key_header=msa_api_key_header,
            msa_api_key_value=msa_api_key_value,
            cache_dir=None,
            max_msa_depth=max_msa_depth,
        )
        processed_dir = regenerated_struct_dir.parent
        if manifest.records[0].id != record_id:
            raise ValueError("affinity re-featurization changed the record id")
    return featurize_affinity_from_prediction(
        manifest=manifest,
        processed_dir=processed_dir,
        mol_dir=mol_dir,
        predicted_coords=predicted_coords,
        atom_pad_mask=atom_pad_mask,
        out_dir=out_dir,
        max_msa_depth=max_msa_depth,
    )
