from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from bench.boltz_msa_probe import (
    FEATURES,
    STAGES,
    Observer,
    full_row_control,
    profile_from_hparams,
    validate_counts,
    validate_operands,
    verify_bound_file,
)


def test_full_row_control_changes_only_row_chunk():
    def operation(*args, **kwargs):
        return args, kwargs

    args, options = full_row_control(operation)(
        "params", "operand", chunk_size=32, row_chunk_size=7, eps=1e-5
    )
    assert args == ("params", "operand")
    assert options == {"chunk_size": 32, "row_chunk_size": 0, "eps": 1e-5}


def _profile():
    return profile_from_hparams(
        {
            "token_s": 384,
            "token_z": 128,
            "msa_args": {
                "msa_s": 64,
                "msa_blocks": 4,
                "msa_dropout": 0.15,
                "z_dropout": 0.25,
                "use_paired_feature": True,
                "use_trifast": True,
                "miniformer_blocks": False,
            },
        }
    )


def _values():
    values = {key: np.zeros((1, 2, 3), np.float32) for key in FEATURES}
    values.update(
        msa=np.zeros((1, 2, 3), np.int64),
        token_pad_mask=np.ones((1, 3), np.float32),
        input_z=np.zeros((1, 3, 3, 128), np.float32),
        emb=np.zeros((1, 3, 384), np.float32),
    )
    return values


def test_pinned_profile_and_full_msa_operands():
    profile = _profile()
    assert profile["msa_blocks"] == 4
    validate_operands(_values(), profile)


@pytest.mark.parametrize("change", ["unknown", "subsample", "unpaired", "dimensions"])
def test_profile_rejects_unreviewed_routes(change):
    h = {"token_s": 384, "token_z": 128, "msa_args": dict(_profile())}
    h["msa_args"].pop("token_s")
    h["msa_args"].pop("token_z")
    if change == "unknown":
        h["msa_args"]["new_publisher_flag"] = True
    elif change == "subsample":
        h["msa_args"]["subsample_msa"] = True
    elif change == "unpaired":
        h["msa_args"]["use_paired_feature"] = False
    else:
        h["token_z"] = 64
    with pytest.raises(ValueError):
        profile_from_hparams(h)


@pytest.mark.parametrize(
    "change", ["unknown", "missing", "shape", "range", "dtype", "nan", "nonbinary"]
)
def test_operands_fail_closed(change):
    v = _values()
    if change == "unknown":
        v["extra"] = np.zeros(1)
    elif change == "missing":
        del v["msa_mask"]
    elif change == "shape":
        v["deletion_value"] = np.zeros((1, 1, 3), np.float32)
    elif change == "range":
        v["msa"][0, 0, 0] = 33
    elif change == "dtype":
        v["input_z"] = v["input_z"].astype(np.float16)
    elif change == "nan":
        v["emb"][0, 0, 0] = np.nan
    else:
        v["msa_mask"][0, 0, 0] = 0.5
    with pytest.raises(ValueError):
        validate_operands(v, _profile())


def test_stage_count_requires_every_actual_layer_and_no_extra():
    counts = Counter({stage: 4 for stage in STAGES})
    validate_counts(counts)
    counts["pwa"] = 3
    with pytest.raises(ValueError, match="stage calls"):
        validate_counts(counts)
    counts["pwa"] = 5
    with pytest.raises(ValueError, match="stage calls"):
        validate_counts(counts)


def test_observer_rejects_wrong_layer_before_writing(tmp_path):
    observer = Observer(tmp_path, "native")
    with pytest.raises(ValueError, match="out-of-order"):
        observer.record("pwa", np.zeros(1), index=1)
    with pytest.raises(ValueError, match="unknown"):
        observer.record("invented", np.zeros(1))
    assert not list(tmp_path.iterdir())


def test_reference_hash_mismatch_is_fatal(tmp_path):
    path = tmp_path / "input"
    path.write_bytes(b"observed")
    with pytest.raises(ValueError, match="identity changed"):
        verify_bound_file(path, "not-the-digest")


def test_jax_callbacks_count_execution_not_scan_traces(tmp_path):
    import jax
    import jax.numpy as jnp

    module = SimpleNamespace()
    module.transition_forward = lambda params, x: x * 0.1
    module.triangle_multiplication_forward = lambda params, z, mask, direction: z * 0.1
    module.triangle_attention_forward = lambda params, z, mask, *, starting: z * 0.1
    module.pair_weighted_averaging_forward = lambda params, m, z, mask: m * 0.1
    module.outer_product_mean_forward = lambda params, m, mask: jnp.ones((1, 2, 2, 128))

    def pair(params, z, mask):
        for direction in ("outgoing", "incoming"):
            z = z + module.triangle_multiplication_forward(params, z, mask, direction)
        for starting in (True, False):
            z = z + module.triangle_attention_forward(
                params, z, mask, starting=starting
            )
        return z + module.transition_forward(params, z)

    module.pairformer_no_seq_layer_forward = pair

    def layer(params, z, m, token_mask, msa_mask):
        m = m + module.pair_weighted_averaging_forward(params, m, z, token_mask)
        m = m + module.transition_forward(params, m)
        z = z + module.outer_product_mean_forward(params, m, msa_mask)
        return module.pairformer_no_seq_layer_forward(params, z, token_mask), m

    module.msa_layer_forward = layer
    observer = Observer(tmp_path, "foldjax")

    def run(z, m):
        def body(carry, unused):
            return module.msa_layer_forward(None, *carry, None, None), None

        return jax.lax.scan(body, (z, m), xs=None, length=4)[0]

    with observer.jax_hooks(module):
        result = jax.jit(run)(jnp.ones((1, 2, 2, 128)), jnp.ones((1, 2, 2, 64)))
        jax.block_until_ready(result)
        jax.effects_barrier()
    validate_counts(observer.counts)
    assert len(observer.artifacts) == 44
    assert module.msa_layer_forward is layer


def test_native_hook_registration_and_removal(tmp_path):
    class Module:
        def __init__(self):
            self.before, self.after = [], []
            self.children = {}

        def register_forward_pre_hook(self, hook):
            self.before.append(hook)
            return SimpleNamespace(remove=lambda: self.before.remove(hook))

        def register_forward_hook(self, hook):
            self.after.append(hook)
            return SimpleNamespace(remove=lambda: self.after.remove(hook))

        def get_submodule(self, name):
            return self.children.setdefault(name, Module())

    model = SimpleNamespace(layers=[Module() for _ in range(4)])
    observer = Observer(tmp_path, "native")
    with observer.native_hooks(model):
        for layer in model.layers:
            for hook in layer.before:
                hook(layer, (np.zeros(1), np.zeros(2)))
            for child in layer.children.values():
                for hook in child.after:
                    assert hook(child, (), np.zeros(2)) is None
            for hook in layer.after:
                assert hook(layer, (), (np.zeros(2), np.zeros(2))) is None
    validate_counts(observer.counts)
    assert all(not layer.after and not layer.before for layer in model.layers)
    assert all(
        not child.after for layer in model.layers for child in layer.children.values()
    )


@pytest.mark.parametrize("compressed", [False, True])
def test_cli_storage_policy_does_not_change_arrays_or_leak(
    monkeypatch, tmp_path, compressed
):
    import sys
    import zipfile

    from bench import boltz_msa_probe as module

    original = np.savez_compressed
    output = tmp_path / "probe.npz"

    def fake_arm(args):
        assert args.compressed is compressed
        module.save_arrays(output, {"x": np.asarray([1.25, -2.5], np.float32)})

    monkeypatch.setattr(module, "foldjax", fake_arm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "foldjax",
            "--reference",
            "input",
            "--out",
            "output",
            "--weights",
            "weights",
        ]
        + (["--compressed"] if compressed else []),
    )
    module.main()
    assert np.savez_compressed is original
    with zipfile.ZipFile(output) as archive:
        assert archive.infolist()[0].compress_type == (
            zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
        )
    with np.load(output, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["x"], [1.25, -2.5])
