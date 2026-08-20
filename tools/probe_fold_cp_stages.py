from __future__ import annotations

import os
import runpy
import subprocess
import sys
import textwrap

namespace = runpy.run_path("tests/models/boltz2/test_context_parallel.py")
preamble = namespace["_PREAMBLE"]

probe = preamble + textwrap.dedent(
    """
    from foldjax.models._cp import shard_pair_rows
    from foldjax.models.boltz2.models.primitives.transition import transition_forward
    from foldjax.models.boltz2.models.triangle.triangle_attention_cp import (
        resolve_triangle_attention_chunk,
        resolve_triangle_attention_q_chunk,
    )

    N = 13
    params = {"layers": [layer(), layer()]}
    z = arr(1, N, N, C)
    pair_mask = pair_mask_for(N)

    names = []
    for layer_index in range(2):
        prefix = f"layer{layer_index + 1}"
        names.extend(
            [
                f"{prefix}.mul_out.update",
                f"{prefix}.mul_out.state",
                f"{prefix}.mul_in.update",
                f"{prefix}.mul_in.state",
                f"{prefix}.att_start.update",
                f"{prefix}.att_start.state",
                f"{prefix}.att_end.update",
                f"{prefix}.att_end.state",
                f"{prefix}.transition.update",
                f"{prefix}.transition.state",
            ]
        )

    def staged(z_in):
        outputs = []
        z_work = z_in
        for layer_params in params["layers"]:
            z_work = shard_pair_rows(z_work)
            tri_chunk = resolve_triangle_attention_chunk(
                z_work.shape[1], 128, None
            )
            tri_q_chunk = resolve_triangle_attention_q_chunk(
                z_work.shape[1], None
            )

            update = triangle_multiplication_forward(
                layer_params["tri_mul_out"],
                z_work,
                pair_mask,
                "outgoing",
                chunk_size=128,
            )
            z_work = z_work + update
            outputs.extend((update, z_work))

            update = triangle_multiplication_forward(
                layer_params["tri_mul_in"],
                z_work,
                pair_mask,
                "incoming",
                chunk_size=128,
            )
            z_work = z_work + update
            outputs.extend((update, z_work))

            update = triangle_attention_forward(
                layer_params["tri_att_start"],
                z_work,
                pair_mask,
                starting=True,
                chunk_size=tri_chunk,
                q_chunk_size=tri_q_chunk,
                triangle_backend="xla",
            )
            z_work = z_work + update
            outputs.extend((update, z_work))

            update = triangle_attention_forward(
                layer_params["tri_att_end"],
                z_work,
                pair_mask,
                starting=False,
                chunk_size=tri_chunk,
                q_chunk_size=tri_q_chunk,
                triangle_backend="xla",
            )
            z_work = z_work + update
            outputs.extend((update, z_work))

            update = transition_forward(
                layer_params["transition_z"],
                z_work,
                chunk_size=None,
                row_chunk_size=128,
            )
            z_work = z_work + update
            outputs.extend((update, z_work))
        return tuple(outputs)

    ref = tuple(jax.device_get(value) for value in compiled(staged)(z))
    with context_parallel(DEVICES, layout="2d"):
        got = tuple(jax.device_get(value) for value in compiled(staged)(z))
    assert traced == [None, "2d"], traced

    for name, expected, actual in zip(names, ref, got, strict=True):
        difference = np.abs(actual - expected)
        tolerance = 3e-5 + 3e-5 * np.abs(expected)
        violations = int(np.count_nonzero(difference > tolerance))
        maximum = float(np.max(difference))
        mean = float(np.mean(difference))
        print(
            f"STAGE devices={DEVICES} name={name} "
            f"max_abs={maximum:.9e} mean_abs={mean:.9e} "
            f"violations={violations}"
        )
    """
)

for devices in (4, 9):
    environment = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
        "FOLDJAX_CP_PROBE_DEVICES": str(devices),
    }
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=300,
    )
    print(completed.stdout, end="")
    if completed.returncode:
        print(completed.stderr, file=sys.stderr, end="")
        raise SystemExit(completed.returncode)
