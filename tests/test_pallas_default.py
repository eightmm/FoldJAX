"""The Pallas pair kernels are the GPU default; what omission means now.

Boltz-2 and OpenFold3 run the Pallas-Triton triangle multiplication and pair
transitions when nothing is asked for on a GPU process, and their released
backends everywhere else. Protenix keeps its released backends on every
platform until a component joins `protenix.runtime_policy.PALLAS_DEFAULT`;
the tests here pin both that and that the constant is the whole switch.

Flipping a default turns an omitted option into two programs, one per
platform, so each surface that records a backend must record the one that ran:

* the cache namespace (`cache_profile`) writes the realised `glu_backend` and
  OpenFold3's realised `triangle_kernel`, fed through the released-default
  strip, so an explicit released value keeps the namespace every earlier run
  wrote and an omitted GPU run shares the entry an explicit `pallas` warmed;
* Boltz-2's retained-runner identity records the realised multiplication;
* what reaches the native call is the realised value.

The platform is moved by patching `_pallas_pair.gpu_process`, the one probe
the defaults and the kernels' own refusal read. The kernel census for the
multiplication lives beside the other wiring tests in
`tests/models/test_pallas_pair_ports.py`.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.backends.boltz2 import Boltz2Backend
from foldjax.backends.openfold3 import OpenFold3Backend
from foldjax.backends.protenix import ProtenixBackend
from foldjax.models import _pallas_pair
from foldjax.schema import PredictionRequest


@pytest.fixture(params=[True, False], ids=["gpu", "cpu"])
def gpu(request, monkeypatch):
    monkeypatch.setattr(_pallas_pair, "gpu_process", lambda: request.param)
    for name in (
        "BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND",
        "PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND",
        "OPENFOLD3_TRIANGLE_BACKEND",
    ):
        monkeypatch.delenv(name, raising=False)
    return request.param


def _request(tmp_path: Path, model: str, **options) -> PredictionRequest:
    tmp_path.mkdir(parents=True, exist_ok=True)
    input_path = tmp_path / ("job.yaml" if model == "boltz2" else "job.json")
    input_path.write_text("{}")
    weights = tmp_path / "weights"
    weights.mkdir(exist_ok=True)
    return PredictionRequest(
        model=model,
        input=input_path,
        weights=weights,
        output_dir=tmp_path / "out",
        seed=5,
        cache_dir=tmp_path / "cache",
        options=options,
    )


# --------------------------------------------------------------------------- #
# CPU keeps the released path, which is what CPU parity replays
# --------------------------------------------------------------------------- #


def test_a_cpu_process_realises_every_released_backend(monkeypatch) -> None:
    """Unpatched: the platform this suite and `--run-cpu-parity` run on.

    Every omitted switch resolves to the backend the CPU parity captures were
    calibrated against, so the parity suite needs no pin to keep running it.
    """

    if jax.default_backend() != "cpu":
        pytest.skip("pins what a CPU process realises")
    from foldjax._openfold3_compile import resolve_triangle_kernel
    from foldjax.backends.base import realised_glu_backend
    from foldjax.models.boltz2.models.triangle import triangle as boltz2_triangle
    from foldjax.models.protenix.models.triangle import triangle as protenix_triangle

    for name in (
        "BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND",
        "PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND",
        "OPENFOLD3_TRIANGLE_BACKEND",
    ):
        monkeypatch.delenv(name, raising=False)
    assert _pallas_pair.gpu_process() is False
    assert boltz2_triangle.triangle_multiplication_backend() == "cueq"
    assert protenix_triangle.triangle_multiplication_backend() == "cueq"
    assert resolve_triangle_kernel(None, cp_shards=1) == "cueq-full"
    for released in ("tokamax", "xla"):
        assert realised_glu_backend(None, released=released, serial=True) == released


# --------------------------------------------------------------------------- #
# Identity records the realised backend
# --------------------------------------------------------------------------- #


def test_boltz2_runner_identity_names_the_realised_multiplication(
    gpu, monkeypatch
) -> None:
    import foldjax.models.boltz2.api as native_api

    def identity():
        return native_api._runtime_identity(
            jax, cp_devices=1, cp_layout="1d", compile_cache=None
        )

    omitted = identity()
    assert ("triangle_multiplication_backend", "pallas" if gpu else "cueq") in omitted
    for spelled in ("pallas", "cueq"):
        monkeypatch.setenv("BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND", spelled)
        # One program, one identity: the omitted run and the spelling of what
        # it realises share it; the other spelling is another program.
        assert (identity() == omitted) is (spelled == ("pallas" if gpu else "cueq"))


@pytest.mark.parametrize(
    "backend, released, flips",
    [
        (Boltz2Backend, "tokamax", True),
        (ProtenixBackend, "xla", False),
        (OpenFold3Backend, "xla", True),
    ],
    ids=["boltz2", "protenix", "openfold3"],
)
def test_cache_profile_records_the_realised_glu(
    tmp_path: Path, gpu, backend, released, flips
) -> None:
    adapter = backend()

    def profile(**options):
        return adapter.cache_profile(_request(tmp_path, backend.name, **options))

    realised = "pallas" if gpu and flips else released
    omitted = profile()
    assert omitted.get("glu_backend") == (None if realised == released else realised)
    # The realised value spelled out names the omitted run's namespace.
    assert profile(glu_backend=realised) == omitted
    # The released value spelled out still strips to absence on either
    # platform: the namespace every run recorded before the flip.
    assert "glu_backend" not in profile(glu_backend=released)
    assert profile(glu_backend="pallas")["glu_backend"] == "pallas"


@pytest.mark.parametrize(
    "backend, cp_options",
    [
        (Boltz2Backend, {"cp_devices": 4}),
        (ProtenixBackend, {"cp_devices": 4}),
        (OpenFold3Backend, {"cp_devices": 4}),
    ],
    ids=["boltz2", "protenix", "openfold3"],
)
def test_a_mesh_keeps_the_released_glu_on_a_gpu(
    tmp_path: Path, monkeypatch, backend, cp_options
) -> None:
    monkeypatch.setattr(_pallas_pair, "gpu_process", lambda: True)
    request = _request(tmp_path, backend.name, **cp_options)
    assert "glu_backend" not in backend().cache_profile(request)


def test_openfold3_cache_profile_records_the_realised_triangle_kernel(
    tmp_path: Path, gpu
) -> None:
    adapter = OpenFold3Backend()

    def kernel(**options):
        return adapter.cache_profile(_request(tmp_path, "openfold3", **options))[
            "triangle_kernel"
        ]

    assert kernel() == ("cueq-pallas" if gpu else "cueq-full")
    assert kernel(triangle_kernel="cueq-full") == "cueq-full"
    assert kernel(cp_devices=4) == "xla"


# --------------------------------------------------------------------------- #
# What reaches the native call
# --------------------------------------------------------------------------- #


def _boltz2_native_options(tmp_path: Path, monkeypatch, **options) -> dict:
    mols = tmp_path / "mols"
    mols.mkdir(parents=True, exist_ok=True)
    seen: dict = {}

    def native_predict(**kwargs):
        seen.update(kwargs)
        return {
            "coords": np.zeros((1, 2, 3)),
            "plddt": np.ones((1, 2)),
            "iptm": np.asarray([0.5]),
            "out_path": None,
        }

    monkeypatch.setattr(
        "foldjax.backends.boltz2.import_module",
        lambda name: SimpleNamespace(predict=native_predict),
    )
    Boltz2Backend().predict(_request(tmp_path, "boltz2", mols=mols, **options))
    return seen


@pytest.mark.parametrize(
    "options, on_gpu, on_cpu",
    [
        ({}, "pallas", "tokamax"),
        ({"glu_backend": "tokamax"}, "tokamax", "tokamax"),
        ({"glu_backend": "xla"}, "xla", "xla"),
        # Under a mesh the native API resolves the released `tokamax` to `xla`.
        ({"cp_devices": 2}, "tokamax", "tokamax"),
    ],
    ids=["omitted", "tokamax", "xla", "mesh"],
)
def test_boltz2_adapter_hands_the_native_call_the_realised_glu(
    tmp_path: Path, monkeypatch, gpu, options, on_gpu, on_cpu
) -> None:
    seen = _boltz2_native_options(tmp_path, monkeypatch, **options)
    assert seen["glu_backend"] == (on_gpu if gpu else on_cpu)


@pytest.mark.parametrize(
    "options, on_gpu, on_cpu",
    [
        # Protenix keeps its released GLU on a GPU: `PALLAS_DEFAULT` is empty.
        ({}, "xla", "xla"),
        ({"glu_backend": "pallas"}, "pallas", "pallas"),
        ({"glu_backend": "xla"}, "xla", "xla"),
        ({"glu_backend": "tokamax"}, "tokamax", "tokamax"),
        ({"cp_devices": 2}, "xla", "xla"),
    ],
    ids=["omitted", "pallas", "xla", "tokamax", "mesh"],
)
def test_protenix_adapter_hands_the_native_run_the_realised_glu(
    tmp_path: Path, gpu, options, on_gpu, on_cpu
) -> None:
    invocation = ProtenixBackend()._native_invocation(
        _request(tmp_path, "protenix", **options)
    )
    expected = on_gpu if gpu else on_cpu
    assert invocation.config_fields["glu_backend"] == expected
    # The argv spells it only where it was asked for or differs from the
    # parser's own default, so a CPU run renders the argv it always did.
    rendered = "--glu-backend" in invocation.argv
    assert rendered is ("glu_backend" in options or expected != "xla")


@pytest.mark.parametrize(
    "members",
    [
        frozenset(),
        frozenset({"trimul"}),
        frozenset({"glu"}),
        frozenset({"trimul", "glu"}),
    ],
    ids=["neither", "trimul", "glu", "both"],
)
def test_protenix_pallas_default_is_one_constant(
    tmp_path: Path, monkeypatch, gpu, members
) -> None:
    """Each member of `PALLAS_DEFAULT` flips exactly its own component.

    The shipped value is empty (4,100-token gate, `runtime_policy`), and the
    rest of this module pins that. This proves the constant is the whole
    switch: adding a name flips that component's omitted setting on a GPU --
    in the model's resolver, the cache namespace and the native run alike --
    and nothing off a GPU.
    """

    from foldjax.models.protenix import runtime_policy
    from foldjax.models.protenix.models.triangle import triangle

    assert runtime_policy.PALLAS_DEFAULT == frozenset()
    monkeypatch.setattr(runtime_policy, "PALLAS_DEFAULT", members)
    trimul = "pallas" if gpu and "trimul" in members else "cueq"
    glu = "pallas" if gpu and "glu" in members else "xla"
    assert triangle.triangle_multiplication_backend() == trimul
    request = _request(tmp_path, "protenix")
    invocation = ProtenixBackend()._native_invocation(request)
    assert invocation.config_fields["glu_backend"] == glu
    assert ProtenixBackend().cache_profile(request).get("glu_backend") == (
        None if glu == "xla" else glu
    )
    # The explicit spellings are untouched by the switch.
    monkeypatch.setenv("PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND", "pallas")
    assert triangle.triangle_multiplication_backend() == "pallas"
    spelled = _request(tmp_path, "protenix", glu_backend="pallas")
    assert (
        ProtenixBackend()._native_invocation(spelled).config_fields["glu_backend"]
        == "pallas"
    )


# --------------------------------------------------------------------------- #
# Boltz-2's MSA-site scoping under the default
# --------------------------------------------------------------------------- #


def test_boltz2_msa_scoping_holds_when_pallas_is_the_default(
    tmp_path: Path, monkeypatch
) -> None:
    """The MSA transition and the MSA-module pair transition keep tokamax.

    Both are scoped in `trunk_blocks/msa.py` on the string the adapter hands
    the native call. Reached by default, that string must be the same
    `pallas` an explicit request passes, and the scoping must then hold: the
    MSA layer's pair transition takes the kernel and the other two do not.
    """

    from foldjax.models.boltz2.models.trunk_blocks import msa

    monkeypatch.setattr(_pallas_pair, "gpu_process", lambda: True)
    realised = _boltz2_native_options(tmp_path, monkeypatch)["glu_backend"]
    explicit = _boltz2_native_options(
        tmp_path / "explicit", monkeypatch, glu_backend="pallas"
    )["glu_backend"]
    assert realised == explicit == "pallas"

    no_seq_layer = msa.pairformer_no_seq_layer_forward
    m = jnp.ones((1, 2, 3, 4), jnp.bfloat16)
    z = jnp.ones((1, 3, 3, 4), jnp.bfloat16)
    zero = lambda *a, **k: jnp.zeros_like(z)  # noqa: E731
    fc1 = {"fc1": {"kernel": jnp.zeros((4, 4), jnp.bfloat16)}}
    transitions: list[str] = []

    def transition(params, value, **kwargs):
        transitions.append(kwargs["glu_backend"])
        return jnp.zeros_like(value)

    monkeypatch.setattr(msa, "transition_forward", transition)
    monkeypatch.setattr(msa, "triangle_multiplication_forward", zero)
    monkeypatch.setattr(msa, "triangle_attention_forward", zero)
    monkeypatch.setattr(
        msa, "pair_weighted_averaging_forward", lambda *a, **k: jnp.zeros_like(m)
    )
    monkeypatch.setattr(
        msa,
        "outer_product_mean_forward",
        lambda *a, **k: jnp.zeros((1, 3, 3, 4), jnp.float32),
    )
    handed: list[str] = []
    monkeypatch.setattr(
        msa,
        "pairformer_no_seq_layer_forward",
        lambda params, z, *a, **k: handed.append(k["glu_backend"]) or z,
    )

    # One MSA layer: its MSA transition keeps tokamax, and the pair layer it
    # calls is handed the default unchanged ...
    msa.msa_layer_forward(
        {
            "pair_weighted_averaging": {},
            "msa_transition": fc1,
            "outer_product_mean": {},
            "pairformer_layer": {},
        },
        jnp.zeros((1, 3, 3, 4)),
        m,
        jnp.ones((1, 3, 3)),
        jnp.ones((1, 2, 3)),
        glu_backend=realised,
    )
    assert (transitions, handed) == (["tokamax"], ["pallas"])

    # ... and that pair layer's own transition keeps tokamax as well.
    transitions.clear()
    params = {
        name: {}
        for name in ("tri_mul_out", "tri_mul_in", "tri_att_start", "tri_att_end")
    }
    params["transition_z"] = fc1
    no_seq_layer(params, z, jnp.ones((1, 3, 3)), glu_backend=realised)
    assert transitions == ["tokamax"]
