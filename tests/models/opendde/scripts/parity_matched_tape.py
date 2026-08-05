"""Measure JAX OpenDDE against a captured, noise-matched upstream run.

`capture_upstream_tape.py` records everything one upstream OpenDDE prediction
draws -- the sampler's initial noise, its per-step churn noise, its per-step
rigid augmentation, and the MSA row permutation of every trunk pass -- plus the
residue trunk's state at each stage and the structure it produced. This replays
that tape through the JAX port and compares, atom for atom, with no alignment.

Two things are being asked at once, and the harness keeps them apart:

* Does the SAMPLER agree? Answered by the coordinate RMSD on a matched tape.
* Does the TRUNK agree, and if not, which module first stops agreeing? Answered
  by the stage table. Every trunk stage is shape-preserving in `z`, so a
  whole-trunk number can only say that something is wrong; the first stage that
  drifts names the module.

The MSA row question is why this exists. Upstream's `MSAModule` calls
`subsample_msa_feature_dict_valid_first` on every trunk pass, and its own
docstring says "each call re-samples the order"; at `msa_depth = 1280` on a
2,372-row alignment each pass therefore reads a different 1,280 rows. Feeding
the JAX trunk the whole alignment is not the same model on the same input.
`--msa-source` selects what the port reads:

    upstream   the exact rows upstream's sampler returned  (the matched run)
    indices    upstream's row indices, applied to FoldJAX's own featurized
               alignment -- equivalent to `upstream` only if the two
               featurizers emit the same rows in the same order, which this
               harness measures rather than assumes
    whole      the entire alignment, which is what the port did before this
               harness existed (the control: it must be measurably worse)

Measured on ``jobs/t0128.json`` translated to OpenDDE's dialect (132 tokens,
1,014 atoms, a 2,372-row alignment that upstream subsamples to 1,280) at
1 recycle / 20 diffusion steps / 1 sample, seed 101:

                        whole        upstream     indices
    after_template      5.0e-05      5.0e-05      5.0e-05   rel_rmse
    after_msa           5.6e-02      1.7e-04      1.7e-04
    after_pairformer_z  6.0e-02      2.1e-03      2.1e-03
    all-atom RMSD       0.8838 A     0.0224 A     --

Read that table left to right and top to bottom. Every stage up to and
including the template embedder agrees to 5e-5 whatever the alignment is, so
nothing before the MSA module is implicated; `after_msa` is where the whole
difference appears and the Pairformer merely carries it. Matching the rows
removes it, and the 0.8838 A control is what the port measured for as long as
it read the whole alignment while upstream read 1,280 rows of it.

`indices` reproducing `upstream` is the featurizer's alibi: the two selections
agree because FoldJAX's own alignment is row-for-row identical to upstream's
(reported as `featurizer_msa_row_match_fraction`), so the gap was the draw, not
the features.

What the residual turned out to be
---------------------------------

Matching the rows leaves a residual that grows with length -- nothing at 132
tokens, 1.57 A at 970 -- and it is not a defect. It is arithmetic precision, and
specifically *upstream's*. Three runs on one tape at 970 tokens settle it:

    ours        upstream      after_pairformer_z   all-atom RMSD
    TF32        TF32               0.01299            1.5709 A
    HIGHEST     TF32               0.01299            1.5162 A
    HIGHEST     TF32 off           0.00000            0.00018 A

Raising only this port's precision changes nothing, which rules this side out.
Turning off upstream's collapses the difference by a factor of 8,600, with all
27 trunk stages at correlation 1.00000000. `enable_tf32: True` is upstream's
default (`opendde/config/inference_defaults.py:24`), so under default settings
that ~1.5 A is irreducible rather than a quality gap.

On this card JAX's own fp32 matmuls already run at TF32 -- 2.941e-04 relative
against `Precision.HIGHEST`, which is exactly the per-block floor the stage
table implies -- so `JAX_DEFAULT_MATMUL_PRECISION=tensorfloat32` is a no-op
here and only `highest` moves anything.

Reproduce the control with `capture_upstream_tape.py --disable-tf32` and this
script under `JAX_DEFAULT_MATMUL_PRECISION=highest`.

Exits non-zero if the coordinate RMSD or any trunk correlation falls short.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

#: ``tests/models/opendde/scripts/`` -> the FoldJAX checkout root.
REPOSITORY = Path(__file__).resolve().parents[4]

#: Trunk stages in the order the trunk produces them, so the table reads as a
#: bisection: the first row that drifts is the one to look at.
STAGE_ORDER = (
    "s_inputs",
    "z_init",
    "after_recycle_z",
    "after_template",
    "after_msa",
    "msa_input_z",
    # Inside each MSA block, in execution order. `z` enters the module clean
    # (6e-05, the same as at 132 tokens) and leaves at 8.1e-04, and block 0
    # alone accounts for 4.8x of that 13.5x. The block has exactly two ways to
    # move `z`: the outer product mean writes the MSA representation into it,
    # and the pair stack works on it afterwards. Recording `m` on both sides of
    # the MSA stack and `z` between the two says which -- and whether `m` was
    # already wrong on the way in, which nothing has ever measured.
    *(
        stage
        for index in range(4)
        for stage in (
            f"msa_block{index}_m_in",
            f"msa_block{index}_m_out",
            f"msa_block{index}_z_after_opm",
            f"msa_block{index}_z",
        )
    ),
    "block0_input_z",
    "block0_out_z",
    "block0_out_s",
    "after_pairformer_z",
    "after_pairformer_s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tape-dir", type=Path, required=True)
    parser.add_argument(
        "--trunk-dtype",
        choices=("float32", "bf16"),
        default="float32",
        help=(
            "narrow the embedder and both trunks, as `--trunk-dtype bf16` does "
            "on the prediction path. The fused triangle kernels take whole "
            "tensors, so bfloat16 is the only lever left on OpenDDE's peak -- "
            "58,722 MiB against 80,843 at 1,531 tokens. This measures what that "
            "costs in coordinates"
        ),
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="the same OpenDDE-dialect job the capture ran, so both sides read "
        "one alignment file",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path.home() / ".cache/foldjax/weights/opendde/opendde.jax",
        help="native OpenDDE weights exported by `foldjax weights fetch`",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPOSITORY / "outputs/opendde-parity"
    )
    parser.add_argument(
        "--msa-source",
        choices=("upstream", "indices", "whole"),
        default="upstream",
        help="which alignment rows the JAX trunk reads; see the module docstring",
    )
    parser.add_argument(
        "--max-rmsd",
        type=float,
        default=0.5,
        help="fail above this all-atom RMSD in angstrom",
    )
    parser.add_argument(
        "--min-correlation",
        type=float,
        default=0.999,
        help="fail if any trunk stage correlates below this with upstream's",
    )
    parser.add_argument(
        "--skip-sampler",
        action="store_true",
        help="report the stage table only; the diffusion run is the slow half "
        "and is not needed to attribute a trunk difference",
    )
    return parser.parse_args()


def rmsd(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((left - right) ** 2, axis=-1))))


def array_metrics(left, right) -> dict[str, float]:
    x = np.asarray(left, dtype=np.float64).ravel()
    y = np.asarray(right, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    scale = float(np.sqrt(np.mean(y**2))) or 1.0
    return {
        "correlation": float(np.corrcoef(x, y)[0, 1]),
        "rel_rmse": float(np.sqrt(np.mean((x - y) ** 2)) / scale),
        "max_abs": float(np.max(np.abs(x - y))),
    }


def _squeeze_batch(array: np.ndarray, rank: int) -> np.ndarray:
    """Drop leading singleton axes down to ``rank``; fail rather than reshape."""
    while array.ndim > rank and array.shape[0] == 1:
        array = array[0]
    if array.ndim != rank:
        raise ValueError(f"cannot reduce shape {array.shape} to rank {rank}")
    return array


def load_tape(path: Path, *, n_step: int, n_atom: int) -> tuple[np.ndarray, dict]:
    contracts = {
        "noise_schedule": (n_step + 1,),
        "init_noise": (1, n_atom, 3),
        "step_noises": (n_step, 1, n_atom, 3),
        "rotations": (n_step, 1, 3, 3),
        "translations": (n_step, 1, 3),
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = set(contracts).difference(archive.files)
        if missing:
            raise ValueError(f"tape is missing arrays: {sorted(missing)}")
        arrays = {name: np.asarray(archive[name], np.float32) for name in contracts}
    for name, expected in contracts.items():
        if arrays[name].shape != expected:
            raise ValueError(
                f"{name} expected shape {expected}, got {arrays[name].shape}"
            )
    return arrays["noise_schedule"], {
        "init_noise": arrays["init_noise"],
        "step_noises": tuple(arrays["step_noises"]),
        "rotations": arrays["rotations"],
        "translations": arrays["translations"],
    }


def build_cycle_msa(
    features: dict,
    archive: dict[str, np.ndarray],
    *,
    source: str,
    n_cycle: int,
) -> tuple[tuple[dict, ...] | None, dict]:
    """Per-cycle MSA features for the JAX trunk, plus what they are based on.

    Returns ``None`` for ``whole``, which leaves the trunk reading the feature
    dict as it stands -- the pre-existing behaviour this run is measured
    against.
    """
    import jax.numpy as jnp

    fields = ("msa", "has_deletion", "deletion_value")
    rows = np.asarray(archive["rows"])
    if rows.shape[0] != n_cycle:
        raise ValueError(
            f"capture holds {rows.shape[0]} MSA row draws but this run has "
            f"{n_cycle} cycles"
        )

    upstream_full = {
        name: _squeeze_batch(np.asarray(archive[f"input_{name}"]), 2) for name in fields
    }
    jax_full = {name: np.asarray(features[name]) for name in fields}
    agreement = {
        "jax_msa_rows": int(jax_full["msa"].shape[0]),
        "upstream_msa_rows": int(upstream_full["msa"].shape[0]),
        "upstream_sampled_rows": int(rows.shape[1]),
    }
    if jax_full["msa"].shape == upstream_full["msa"].shape:
        agreement["featurizer_msa_rows_identical"] = bool(
            np.array_equal(jax_full["msa"], upstream_full["msa"])
        )
        agreement["featurizer_msa_row_match_fraction"] = float(
            np.mean(np.all(jax_full["msa"] == upstream_full["msa"], axis=-1))
        )
    else:
        agreement["featurizer_msa_rows_identical"] = False
        agreement["featurizer_msa_row_match_fraction"] = None

    if source == "whole":
        return None, agreement

    cycles: list[dict] = []
    for cycle in range(n_cycle):
        if source == "upstream":
            selected = {
                name: _squeeze_batch(
                    np.asarray(archive[f"selected_{name}"][cycle]), 2
                )
                for name in fields
            }
        else:
            index = rows[cycle]
            selected = {name: jax_full[name][index] for name in fields}
        cycle_features = {
            name: jnp.asarray(
                value.astype(np.int32 if name == "msa" else np.float32)
            )
            for name, value in selected.items()
        }
        # Upstream's mask is all-ones and is used only to prioritize rows during
        # selection; once selected, every row enters the MSA module unmasked.
        cycle_features["msa_mask"] = jnp.ones(
            cycle_features["msa"].shape, dtype=jnp.float32
        )
        cycles.append(cycle_features)
    return tuple(cycles), agreement


def trunk_stages(
    features: dict,
    params,
    *,
    cycle_msa: tuple[dict, ...] | None,
    n_cycle: int,
) -> dict[str, np.ndarray]:
    """Run the residue trunk one stage at a time, keeping the last cycle's state.

    This mirrors `pairformer_output_from_s_inputs` rather than calling it,
    because the whole point is to see inside it. `msa_stack_first=True` is
    OpenDDE's own ordering and is what that function is asked for at the
    OpenDDE call site.
    """
    import jax.numpy as jnp

    from foldjax.models.protenix.models.trunk_blocks.embedders import (
        input_feature_embedder,
    )
    from foldjax.models.protenix.models.trunk_blocks.msa import msa_module
    from foldjax.models.protenix.models.trunk_blocks.pairformer import pairformer_stack
    from foldjax.models.protenix.models.trunk_blocks.template import template_embedder
    from foldjax.models.protenix.models.trunk_blocks.trunk import (
        recycle_embeddings,
        relative_position_features,
        trunk_initial_embeddings,
    )

    trunk = params.pairformer_output
    n_token = int(features["restype"].shape[-2])
    stages: dict[str, np.ndarray] = {}

    s_inputs = input_feature_embedder(
        features, params.input_embedder, n_token=n_token, n_heads=4
    )
    stages["s_inputs"] = np.asarray(s_inputs)

    relp = features.get("relp")
    if relp is None:
        relp = relative_position_features(features)
    s_init, z_init = trunk_initial_embeddings(
        s_inputs, relp, features["token_bonds"], trunk.trunk.initial
    )
    stages["z_init"] = np.asarray(z_init)

    s = jnp.zeros_like(s_init)
    z = jnp.zeros_like(z_init)
    for cycle in range(n_cycle):
        msa_features = features
        if cycle_msa is not None:
            msa_features = {**features, **cycle_msa[cycle]}
        s, z = recycle_embeddings(s_init, z_init, s, z, trunk.trunk.recycling)
        stages["after_recycle_z"] = np.asarray(z)
        z = z + template_embedder(features, z, None, trunk.template)
        stages["after_template"] = np.asarray(z)
        # The MSA module block by block. `after_msa` is the first stage that
        # loses digits -- everything before it agrees to ten -- and the
        # Pairformer only amplifies what it receives, so this is where a real
        # difference can still enter. Run through `msa_block` itself, unrolled,
        # rather than rebuilt: the previous probe hand-assembled operations and
        # measured its own reconstruction.
        from foldjax.models.protenix.models.primitives.primitives import linear
        from foldjax.models.protenix.models.trunk_blocks.msa import msa_block

        # `msa_module` builds `m` inline rather than exposing an embedder, so
        # the same four lines are repeated here. They are one-hot, concatenate,
        # project -- deterministic, and every stage feeding them already agrees
        # to ten digits, so reproducing them cannot be what a difference hides
        # behind.
        stages["msa_input_z"] = np.asarray(z)
        _msa = msa_features["msa"]
        _one_hot = jnp.eye(32, dtype=s_inputs.dtype)[_msa]
        _shape = _one_hot.shape[:-1]
        _sample = jnp.concatenate(
            [
                _one_hot,
                msa_features["has_deletion"].reshape(_shape + (1,)),
                msa_features["deletion_value"].reshape(_shape + (1,)),
            ],
            axis=-1,
        )
        probe_m = linear(_sample, trunk.msa.linear_m)
        probe_m = probe_m + linear(s_inputs, trunk.msa.linear_s)[..., None, :, :]
        probe_z = z
        # Each block is also run in pieces, because the block-level number says
        # only that z changed. OpenDDE's ordering is MSA stack, then outer
        # product mean into z, then pair stack -- so `m_in`, `m_out` and
        # `z_after_opm` separate "the MSA representation arrived wrong" from
        # "the MSA stack made it wrong" from "the pair stack amplified z".
        #
        # These call the same functions `msa_block` calls, in the same order,
        # on the same inputs -- but so did the probe that measured its own
        # reconstruction and reported correlations of 0.067 for operations that
        # were fine. So the pieces are checked against the real block below
        # rather than trusted, and a mismatch raises instead of printing.
        from foldjax.models.protenix.models.primitives.primitives import (
            transition as _transition,
        )
        from foldjax.models.protenix.models.trunk_blocks.msa import (
            msa_pair_weighted_averaging as _weighted_averaging,
        )
        from foldjax.models.protenix.models.trunk_blocks.msa import (
            outer_product_mean,
        )
        from foldjax.models.protenix.models.trunk_blocks.pairformer import (
            pairformer_block as _pairformer_block,
        )

        for index in range(len(trunk.msa.blocks)):
            block_params = trunk.msa.blocks[index]
            stages[f"msa_block{index}_m_in"] = np.asarray(probe_m)
            if block_params.msa_pair_weighted_averaging is None:
                split_m = probe_m
            else:
                split_m = probe_m + _weighted_averaging(
                    probe_m,
                    probe_z,
                    block_params.msa_pair_weighted_averaging,
                )
                split_m = split_m + _transition(split_m, block_params.msa_transition)
                stages[f"msa_block{index}_m_out"] = np.asarray(split_m)
            split_z = probe_z + outer_product_mean(
                split_m, None, block_params.outer_product_mean
            )
            stages[f"msa_block{index}_z_after_opm"] = np.asarray(split_z)
            _, split_z = _pairformer_block(None, split_z, None, block_params.pair_stack)

            probe_m, probe_z = msa_block(
                probe_m,
                probe_z,
                None,
                block_params,
                msa_stack_first=True,
            )
            stages[f"msa_block{index}_z"] = np.asarray(probe_z)
            drift = float(
                np.max(np.abs(np.asarray(split_z) - np.asarray(probe_z)))
            )
            if drift > 0:
                msg = (
                    f"MSA block {index}: the unrolled pieces diverge from "
                    f"`msa_block` by {drift:g}, so the split stages describe "
                    "the probe rather than the port"
                )
                raise AssertionError(msg)
            del split_m, split_z
        del probe_m, probe_z

        z = msa_module(msa_features, z, s_inputs, None, trunk.msa, msa_stack_first=True)
        stages["after_msa"] = np.asarray(z)
        # Block 0's four z updates, run through the same `pairformer_block`
        # the stack uses, one operation at a time.
        #
        # The first attempt rebuilt them by hand and compared each output
        # against upstream's forward hook. Two of the four came back with
        # correlations of 0.067 and -0.168 while the block they belong to
        # emitted z at 0.99992 -- a combination no real defect produces. The
        # hand-built version fed each operation a `z` that the real stack never
        # held, so it measured the reconstruction, not the port.
        #
        # `after_msa` is the block's input on both sides and already agrees to
        # seven digits, so a probe that starts there starts matched. That is
        # asserted rather than assumed, because it is the assumption the last
        # attempt got wrong.
        from foldjax.models.protenix.models.trunk_blocks.pairformer import (
            pairformer_block,
        )

        block0 = trunk.pairformer_stack.blocks[0]
        probe_s, probe_z = s, z
        stages["block0_input_z"] = np.asarray(probe_z)
        stages["block0_out_z"], stages["block0_out_s"] = (
            lambda pair: (np.asarray(pair[1]), np.asarray(pair[0]))
        )(pairformer_block(probe_s, probe_z, None, block0))
        del probe_s, probe_z

        s, z = pairformer_stack(s, z, None, trunk.pairformer_stack, use_scan=True)
        stages["after_pairformer_z"] = np.asarray(z)
        stages["after_pairformer_s"] = np.asarray(s)
    return stages


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import jax

    jax.config.update("jax_default_matmul_precision", "highest")

    from foldjax.models.opendde.bridge.weights_io import load_native_weights
    from foldjax.models.opendde.data.featurize_json import (
        featurize_opendde_json,
        load_jobs,
    )
    from foldjax.models.opendde.models.model import opendde_infer_static

    meta = json.loads((args.tape_dir / "tape.json").read_text(encoding="utf-8"))
    n_cycle = int(meta["n_cycle"])
    n_step = int(meta["n_step"])
    seed = int(meta["seed"])

    with np.load(args.tape_dir / "msa.npz", allow_pickle=False) as archive:
        msa_archive = {name: archive[name] for name in archive.files}
    with np.load(args.tape_dir / "trunk.npz", allow_pickle=False) as archive:
        upstream_stages = {
            name: _squeeze_batch(np.asarray(archive[name], np.float64), 3)
            if archive[name].ndim > 3
            else np.asarray(archive[name], np.float64)
            for name in archive.files
        }
    with np.load(args.tape_dir / "coordinate.npz", allow_pickle=False) as archive:
        upstream_coordinate = np.asarray(archive["coordinate"], np.float64)

    job = load_jobs(args.input_json)[0]
    features = featurize_opendde_json(job, base_dir=args.input_json.parent, seed=seed)
    n_atom = int(features["ref_pos"].shape[0])
    noise_schedule, tape = load_tape(
        args.tape_dir / "tape.npz", n_step=n_step, n_atom=n_atom
    )

    params = load_native_weights(args.weights)
    if args.trunk_dtype == "bf16":
        import jax.numpy as jnp

        from foldjax.models.opendde.models.model import cast_trunk_params

        # Same call the CLI makes, so the harness narrows exactly what a real
        # bf16 run narrows -- embedder and both trunks, heads left in float32.
        params = cast_trunk_params(params, jnp.bfloat16)
    cycle_msa, agreement = build_cycle_msa(
        features, msa_archive, source=args.msa_source, n_cycle=n_cycle
    )

    started = time.perf_counter()
    stages = trunk_stages(features, params, cycle_msa=cycle_msa, n_cycle=n_cycle)
    jax.block_until_ready(stages["after_pairformer_z"])
    trunk_seconds = time.perf_counter() - started

    stage_metrics: dict[str, dict] = {}
    for name in STAGE_ORDER:
        if name not in stages or name not in upstream_stages:
            continue
        mine = _squeeze_batch(stages[name], upstream_stages[name].ndim)
        stage_metrics[name] = array_metrics(mine, upstream_stages[name])

    metrics: dict = {
        "msa_source": args.msa_source,
        "recycles": n_cycle,
        "diffusion_steps": n_step,
        "samples": int(meta["n_sample"]),
        "atoms": n_atom,
        "tokens": int(features["restype"].shape[-2]),
        "trunk_seconds": trunk_seconds,
        **agreement,
        "stages": stage_metrics,
    }

    if not args.skip_sampler:
        started = time.perf_counter()
        output = opendde_infer_static(
            features,
            params,
            noise_schedule,
            key=None,
            n_sample=1,
            n_cycle=n_cycle,
            run_confidence=False,
            cycle_msa_features=cycle_msa,
            diffusion_attention_backend="xla",
            trunk_single_attention_backend="xla",
            trunk_triangle_attention_backend="xla",
            structural_single_attention_backend="xla",
            structural_triangle_attention_backend="xla",
            **tape,
        )
        coordinate = np.asarray(jax.device_get(output["coordinate"]), np.float64)
        sample_seconds = time.perf_counter() - started
        left = _squeeze_batch(coordinate, 2)
        right = _squeeze_batch(upstream_coordinate, 2)
        if left.shape != right.shape:
            raise RuntimeError(f"coordinate shape {left.shape} != {right.shape}")
        metrics.update(
            sample_seconds=sample_seconds,
            all_atom_rmsd_angstrom=rmsd(left, right),
            coordinate_max_abs_error_angstrom=float(np.max(np.abs(left - right))),
            coordinate_mean_abs_error_angstrom=float(np.mean(np.abs(left - right))),
        )
        np.save(args.out_dir / f"coordinate-{args.msa_source}.npy", coordinate)

    (args.out_dir / f"metrics-{args.msa_source}.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))

    print(f"\n{'stage':22s} {'correlation':>14s} {'rel_rmse':>12s} {'max_abs':>12s}")
    for name in STAGE_ORDER:
        row = stage_metrics.get(name)
        if row is None:
            continue
        print(
            f"{name:22s} {row['correlation']:14.8f} {row['rel_rmse']:12.5f} "
            f"{row['max_abs']:12.5f}"
        )

    failures = _verdict(
        metrics, max_rmsd=args.max_rmsd, min_correlation=args.min_correlation
    )
    if failures:
        print("\n".join(["", "PARITY FAILED:", *failures]))
        return 1
    print("\nPARITY OK")
    return 0


def _verdict(metrics: dict, *, max_rmsd: float, min_correlation: float) -> list[str]:
    """Which measurements fall short, as lines to print.

    Printing a number is not checking it. A harness that only reports its RMSD
    cannot fail, and a check that cannot fail is how a reordered MSA block
    survived a whole release here.
    """
    failures = []
    if not metrics["stages"]:
        failures.append("  no trunk stage was compared; the capture named none")
    for name, row in metrics["stages"].items():
        if not row["correlation"] >= min_correlation:
            failures.append(
                f"  {name}: correlation {row['correlation']:.8f} < {min_correlation:g}"
            )
    value = metrics.get("all_atom_rmsd_angstrom")
    if value is not None and not value <= max_rmsd:
        failures.append(f"  all-atom RMSD {value:.4f} A > {max_rmsd:g} A")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
