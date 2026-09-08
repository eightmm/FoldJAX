import json

import numpy as np
import pytest

from bench import boltz_msa_embedding_probe as probe
from bench.boltz_relpos_probe import bf16_round


def projection_operands():
    return np.zeros((1, 437, 384), np.float32), np.zeros((64, 384), np.float32)


def test_complete_original_projection_shapes():
    emb, weight = projection_operands()
    probe.validate_projection(emb, weight, np.zeros((1, 437, 64), np.float32))


@pytest.mark.parametrize(
    "change",
    [
        "emb_shape",
        "weight_shape",
        "emb_dtype",
        "weight_dtype",
        "nan",
        "inf",
        "overflow",
    ],
)
def test_projection_operands_fail_closed(change):
    emb, weight = projection_operands()
    if change == "emb_shape":
        emb = emb[:, :3]
    elif change == "weight_shape":
        weight = weight.T
    elif change == "emb_dtype":
        emb = emb.astype(np.float16)
    elif change == "weight_dtype":
        weight = weight.astype(np.float64)
    elif change == "nan":
        emb[0, 0, 0] = np.nan
    elif change == "inf":
        weight[0, 0] = np.inf
    else:
        weight[0, 0] = np.finfo(np.float32).max
    with np.errstate(over="ignore"), pytest.raises(ValueError):
        probe.validate_projection(emb, weight)


@pytest.mark.parametrize("change", ["shape", "dtype", "nan", "inf", "not_bf16"])
def test_output_requires_complete_finite_bf16_archive(change):
    emb, weight = projection_operands()
    out = np.zeros((1, 437, 64), np.float32)
    if change == "shape":
        out = out[:, :3]
    elif change == "dtype":
        out = out.astype(np.float64)
    else:
        out[0, 0, 0] = {"nan": np.nan, "inf": np.inf, "not_bf16": 1.001}[change]
    with pytest.raises(ValueError):
        probe.validate_projection(emb, weight, out)


def tiny_reconstruction():
    values = {
        "msa": np.asarray([[[0, 1], [2, 3], [4, 5]]], np.int64),
        "has_deletion": np.asarray([[[0, 1], [1, 0], [1, 1]]], bool),
        "deletion_value": np.asarray([[[0, 0.3], [0.7, 0], [1.1, -0.2]]], np.float32),
        "msa_paired": np.asarray([[[1, 1], [0, 0], [0, 0]]], np.float32),
    }
    rng = np.random.default_rng(31)
    weight = rng.normal(size=(64, 36)).astype(np.float32)
    projection = bf16_round(rng.normal(size=(1, 2, 64)))
    dense = np.concatenate(
        (
            np.eye(33, dtype=np.float32)[values["msa"]],
            np.stack(
                [values[k] for k in ("has_deletion", "deletion_value", "msa_paired")],
                -1,
            ),
        ),
        -1,
    )
    exact = (
        bf16_round(dense).astype(np.float64) @ bf16_round(weight).astype(np.float64).T
    )
    expected = bf16_round(bf16_round(exact.astype(np.float32)) + projection[:, None])
    return values, weight, projection, expected


@pytest.mark.parametrize("chunk", [1, 2, 4])
def test_formula_reconstruction_preserves_order_and_unequal_tail(chunk):
    values, weight, projection, expected = tiny_reconstruction()
    result = probe.reconstruct_embedding(values, weight, projection, expected, chunk)
    assert result["formula_only"] is True
    assert result["elements"] == expected.size
    assert result["comparison"] == {
        "max_abs": 0.0,
        "rmse": 0.0,
        "unequal": 0,
        "values_equal": True,
    }
    assert np.array_equal(values["msa"], np.asarray([[[0, 1], [2, 3], [4, 5]]]))


def test_formula_failure_retains_metric_not_an_admission_gate():
    values, weight, projection, expected = tiny_reconstruction()
    expected[0, 2, 1, 3] += 1
    result = probe.reconstruct_embedding(values, weight, projection, expected, 2)
    assert result["formula_only"]
    assert result["comparison"]["unequal"] == 1
    assert result["comparison"]["max_abs"] == 1
    assert result["comparison"]["rmse"] == pytest.approx(1 / np.sqrt(expected.size))


@pytest.mark.parametrize(
    "change",
    ["id", "id_float", "nonbinary", "nan", "weight", "projection", "zero_chunk"],
)
def test_formula_invalid_contracts(change):
    values, weight, projection, expected = tiny_reconstruction()
    if change == "id":
        values["msa"][0, 0, 0] = 33
    elif change == "id_float":
        values["msa"] = values["msa"].astype(np.float32)
    elif change == "nonbinary":
        values["msa_paired"][0, 0, 0] = 0.5
    elif change == "nan":
        values["deletion_value"][0, 0, 0] = np.nan
    elif change == "weight":
        weight = weight.astype(np.float64)
    elif change == "projection":
        projection[0, 0, 0] = 1.001
    with pytest.raises(ValueError):
        probe.reconstruct_embedding(
            values, weight, projection, expected, 0 if change == "zero_chunk" else 2
        )


def test_suspect_table_labels_exact_sum_as_counterfactual():
    emb, weight = projection_operands()
    out = np.zeros((1, 437, 64), np.float32)
    out[0, 258, 54] = -3.140625
    table = probe.suspected_scalars(emb, weight, out)
    assert [(x["token"], x["channel"]) for x in table] == list(probe.SUSPECTED)
    assert table[2]["observed"] == -3.140625
    assert all(x["cpu_exact_dot_fp32_then_bf16"] == 0 for x in table)
    assert all(x["cpu_value_is_counterfactual_not_native_capture"] for x in table)


def test_candidate_uses_production_linear_full_shape_dtype_and_highest(monkeypatch):
    import jax
    import jax.numpy as jnp

    from foldjax.models.boltz2.models.primitives import _common

    original = _common.linear
    calls = []

    def observe(emb, kernel):
        calls.append((emb.shape, kernel.shape, emb.dtype, kernel.dtype))
        return original(emb, kernel)

    monkeypatch.setattr(_common, "linear", observe)
    emb = jnp.ones((1, 437, 384), jnp.float32)
    kernel = jnp.ones((384, 64), jnp.bfloat16)
    with jax.default_matmul_precision("highest"):
        graph = jax.make_jaxpr(probe.candidate_projection)(emb, kernel)
        result = jax.jit(probe.candidate_projection)(emb, kernel)
    assert calls[0] == ((1, 437, 384), (384, 64), jnp.float32, jnp.bfloat16)
    assert result.dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(result, np.float32), 384)
    dot = next(eq for eq in graph.jaxpr.eqns if eq.primitive.name == "dot_general")
    assert dot.params["precision"] == (jax.lax.Precision.HIGHEST,) * 2
    assert all(var.aval.dtype == jnp.bfloat16 for var in dot.invars)


@pytest.mark.parametrize(
    "profile,extra",
    [
        ("baseline", {}),
        ("split-k-1", {"xla_gpu_experimental_force_split_k": 1}),
        ("no-triton", {"xla_gpu_enable_triton_gemm": False}),
    ],
)
def test_compiler_profiles_change_only_explicit_option(profile, extra):
    expected = {"xla_allow_excess_precision": False, **extra}
    assert probe.candidate_compile_options(profile) == expected
    assert probe.candidate_compile_options("baseline") == {
        "xla_allow_excess_precision": False
    }


@pytest.mark.parametrize("profile", ["split-k-1", "no-triton"])
def test_installed_compiler_accepts_control_options_on_cpu_only(profile):
    import jax
    import jax.numpy as jnp

    result = jax.jit(
        lambda x: x + 1, compiler_options=probe.candidate_compile_options(profile)
    )(jnp.asarray(1))
    assert int(result) == 2


SPLIT_HLO = (
    "ROOT %dot.2 = f32[4,437,64]{2,1,0} dot(%x, %w), lhs_batch_dims={1}, "
    "lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}\n"
    "%reduce.2 = f32[437,64]{1,0} reduce(%partials, %zero), dimensions={0}\n"
    "%fusion = f32[4,437,64]{2,1,0} fusion(%x, %w), "
    'backend_config={"kind":"__triton_nested_gemm_fusion"}\n'
)


def test_saved_split_graph_is_evidence_not_success_from_requested_flag():
    baseline = probe.projection_lowering_evidence(SPLIT_HLO, "baseline")
    assert baseline["visible_split_k_factors"] == [4]
    assert baseline["requested_profile_verified"]
    assert len(baseline["reduction_instructions"]) == 1
    rejected = probe.projection_lowering_evidence(SPLIT_HLO, "split-k-1")
    assert not rejected["requested_profile_verified"]


def test_unsplit_triton_and_external_blas_are_distinct_controls():
    unsplit = (
        "ROOT %dot.1 = bf16[437,64]{1,0} dot(%x, %w), "
        "lhs_contracting_dims={1}, rhs_contracting_dims={0}\n"
        "%fusion = bf16[437,64]{1,0} fusion(%x, %w), "
        'backend_config={"kind":"__triton_gemm"}\n'
    )
    assert probe.projection_lowering_evidence(unsplit, "split-k-1")[
        "requested_profile_verified"
    ]
    assert not probe.projection_lowering_evidence(unsplit, "no-triton")[
        "requested_profile_verified"
    ]
    blas = (
        "ROOT %gemm = bf16[437,64]{1,0} custom-call(%x, %w), "
        'custom_call_target="__cublas$gemm"'
    )
    evidence = probe.projection_lowering_evidence(blas, "no-triton")
    assert evidence["requested_profile_verified"]
    assert evidence["contains_external_blas"]
    assert not evidence["external_blas_internal_split_k_observable"]
    assert not probe.projection_lowering_evidence(blas, "split-k-1")[
        "requested_profile_verified"
    ]


def test_unclassified_or_empty_lowering_cannot_verify_control():
    assert not probe.projection_lowering_evidence("", "split-k-1")[
        "requested_profile_verified"
    ]
    unknown = SPLIT_HLO.replace("[4,437,64]", "[2,2,437,64]")
    assert probe.projection_lowering_evidence(unknown, "split-k-1")[
        "unclassified_dot_instructions"
    ]
    for function in (
        probe.candidate_compile_options,
        lambda p: probe.projection_lowering_evidence("", p),
    ):
        with pytest.raises(ValueError, match="unknown candidate profile"):
            function("inferred-fallback")


def write_reference(tmp_path, monkeypatch):
    # Small row count for archive-validation tests only; production is S4436.
    shape = (1, 3, 437, 64)
    monkeypatch.setattr(probe, "SHAPE", shape)
    root = tmp_path / "msa"
    (root / "layers/00").mkdir(parents=True)
    emb, weight = projection_operands()
    values = {k: np.zeros(shape[:3], np.float32) for k in probe.FEATURES}
    values["msa"] = values["msa"].astype(np.int64)
    values["token_pad_mask"] = np.ones((1, 437), np.float32)
    values["emb"] = emb
    operand_sha = probe.save_array(root / "operands.npz", **values)
    weight_sha = probe.save_array(
        root / "native-weights.npz",
        **{
            "msa_module.s_proj.weight": weight,
            "msa_module.msa_proj.weight": np.zeros((64, 36), np.float32),
        },
    )
    stage_sha = probe.save_array(
        root / "layers/00/input_m.npz", **{"": np.zeros(shape, np.float32)}
    )
    tree = root / "layers/00/input_m.tree.json"
    tree.write_text(
        json.dumps(
            {
                "": {
                    "native_dtype": "torch.bfloat16",
                    "storage_dtype": "float32",
                    "shape": list(shape),
                }
            }
        )
    )
    report = {
        "arm": "native",
        "passed": True,
        "runtime": {
            "float32_matmul_precision": "highest",
            "cuda_allow_tf32": False,
            "bf16_reduced_precision_reduction": True,
        },
        "artifacts": {"operands.npz": operand_sha, "native-weights.npz": weight_sha},
        "stages": {
            "layers/00/input_m": {
                "arrays_sha256": stage_sha,
                "tree_sha256": probe.sha(tree),
            }
        },
        "native_source": {"source.py": "unchanged"},
        "checkpoint_sha256": "checkpoint",
    }
    (root / "report.json").write_text(json.dumps(report))
    return root


def write_projection(tmp_path, bindings):
    root = tmp_path / "projection"
    root.mkdir()
    emb, weight = projection_operands()
    operand_sha = probe.save_array(root / "operands.npz", emb=emb, weight=weight)
    output = np.zeros((1, 437, 64), np.float32)
    output_sha = probe.save_array(root / "outputs.npz", first=output, repeat=output)
    report = {
        "arm": "native",
        "passed": True,
        "runtime": {"control": "none"},
        "reference_files": bindings,
        "artifacts": {"operands.npz": operand_sha, "outputs.npz": output_sha},
    }
    (root / "report.json").write_text(json.dumps(report))
    return root


def test_native_reference_and_projection_archive_binding(tmp_path, monkeypatch):
    root = write_reference(tmp_path, monkeypatch)
    report, bindings, values, weight, _ = probe.load_reference(root)
    projection = write_projection(tmp_path, bindings)
    output, identities = probe.load_native_projection(
        projection, bindings, values["emb"], weight
    )
    assert output.shape == (1, 437, 64)
    assert report["passed"]
    assert set(identities) == {"report.json", "operands.npz", "outputs.npz"}
    with (root / "operands.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="identity changed"):
        probe.verify_files(root, bindings)


@pytest.mark.parametrize(
    "change", ["unpassed", "tf32", "reduction", "shape", "tree_dtype"]
)
def test_reference_policy_and_shape_fail_closed(tmp_path, monkeypatch, change):
    root = write_reference(tmp_path, monkeypatch)
    path = root / "report.json"
    report = json.loads(path.read_text())
    if change == "unpassed":
        report["passed"] = False
    elif change == "tf32":
        report["runtime"]["cuda_allow_tf32"] = True
    elif change == "reduction":
        report["runtime"]["bf16_reduced_precision_reduction"] = False
    else:
        tree_path = root / "layers/00/input_m.tree.json"
        tree = json.loads(tree_path.read_text())
        if change == "shape":
            tree[""]["shape"][1] = 2
        else:
            tree[""]["native_dtype"] = "torch.float32"
        tree_path.write_text(json.dumps(tree))
        report["stages"]["layers/00/input_m"]["tree_sha256"] = probe.sha(tree_path)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        probe.load_reference(root)


@pytest.mark.parametrize(
    "change", ["control", "reference", "operand", "repeat", "negative_zero"]
)
def test_candidate_rejects_control_or_changed_native_projection(
    tmp_path, monkeypatch, change
):
    root = write_reference(tmp_path, monkeypatch)
    _, bindings, values, weight, _ = probe.load_reference(root)
    projection = write_projection(tmp_path, bindings)
    path = projection / "report.json"
    report = json.loads(path.read_text())
    if change == "control":
        report["runtime"]["control"] = "bf16_reduction_disabled"
    elif change == "reference":
        report["reference_files"] = {}
    elif change == "operand":
        with (projection / "operands.npz").open("wb") as stream:
            np.savez(stream, emb=values["emb"] + 1, weight=weight)
        report["artifacts"]["operands.npz"] = probe.sha(projection / "operands.npz")
    else:
        output = np.zeros((1, 437, 64), np.float32)
        repeat = output.copy()
        repeat[0, 0, 0] = -0.0 if change == "negative_zero" else 1.0
        with (projection / "outputs.npz").open("wb") as stream:
            np.savez(stream, first=output, repeat=repeat)
        report["artifacts"]["outputs.npz"] = probe.sha(projection / "outputs.npz")
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        probe.load_native_projection(projection, bindings, values["emb"], weight)


@pytest.mark.parametrize(
    "options",
    [
        ["--mode", "candidate"],
        [
            "--mode",
            "candidate",
            "--reference",
            "ref",
            "--disable-bf16-reduced-precision",
        ],
        ["--mode", "native", "--reference", "ref"],
        ["--mode", "native", "--candidate-profile", "split-k-1"],
    ],
)
def test_cli_rejects_ambiguous_controls_before_framework_import(monkeypatch, options):
    monkeypatch.setattr(
        "sys.argv",
        [
            "probe",
            "--msa-reference",
            "missing",
            "--source-root",
            "missing",
            "--out",
            "missing",
            *options,
        ],
    )
    with pytest.raises(SystemExit) as caught:
        probe.main()
    assert caught.value.code == 2


def test_harness_identity_includes_transitive_local_helpers(tmp_path, monkeypatch):
    for name in ("probe.py", "helper.py"):
        (tmp_path / name).write_text("pass\n")
    monkeypatch.setattr(probe, "__file__", str(tmp_path / "probe.py"))
    before = probe.harness_hashes()
    (tmp_path / "helper.py").write_text("x = 1\n")
    assert set(before) == {"probe.py", "helper.py"}
    assert before != probe.harness_hashes()


def test_cli_cpu_stub_records_evidence_and_does_not_overwrite(tmp_path, monkeypatch):
    root = write_reference(tmp_path, monkeypatch)
    out = tmp_path / "out"
    source = tmp_path / "source"
    monkeypatch.setattr(probe, "source_hashes", lambda *_: {"source.py": "unchanged"})
    called = []

    def native(args, report, values, weight):
        called.append(
            (args.disable_bf16_reduced_precision, values["emb"].shape, weight.shape)
        )
        outputs = [np.zeros((1, 437, 64), np.float32) for _ in range(2)]
        return outputs, {"control": "none"}, {"source.py": "unchanged"}

    monkeypatch.setattr(probe, "native", native)
    monkeypatch.setattr(
        "sys.argv",
        [
            "probe",
            "--mode",
            "native",
            "--msa-reference",
            str(root),
            "--source-root",
            str(source),
            "--out",
            str(out),
        ],
    )
    probe.main()
    result = json.loads((out / "report.json").read_text())
    assert called == [(False, (1, 437, 384), (64, 384))]
    assert result["passed"] and result["capture_complete"]
    assert result["repeat"]["bitwise_equal"]
    assert result["embedding_reconstruction"]["formula_only"]
    assert result["candidate_vs_native"] is None
    assert result["reference_files"]["operands.npz"] == probe.sha(root / "operands.npz")
    with pytest.raises(FileExistsError):
        probe.main()


@pytest.mark.parametrize("lowering_verified", [False, True])
def test_candidate_control_preserves_failed_output_and_lowering(
    tmp_path, monkeypatch, lowering_verified
):
    root = write_reference(tmp_path, monkeypatch)
    _, bindings, _, _, _ = probe.load_reference(root)
    native = write_projection(tmp_path, bindings)
    out = tmp_path / "candidate"
    monkeypatch.setattr(probe, "source_hashes", lambda *_: {"source.py": "unchanged"})

    def candidate(args, *_):
        assert args.candidate_profile == "split-k-1"
        output = np.zeros((1, 437, 64), np.float32)
        output[0, 217, 2] = 1.0
        return (
            [output, output],
            {
                "control": "split-k-1",
                "lowering": {"requested_profile_verified": lowering_verified},
            },
            {"source.py": "unchanged"},
        )

    monkeypatch.setattr(probe, "candidate", candidate)
    monkeypatch.setattr(
        "sys.argv",
        [
            "probe",
            "--mode",
            "candidate",
            "--msa-reference",
            str(root),
            "--source-root",
            str(tmp_path),
            "--reference",
            str(native),
            "--out",
            str(out),
            "--candidate-profile",
            "split-k-1",
        ],
    )
    probe.main()
    result = json.loads((out / "report.json").read_text())
    assert result["capture_complete"] and not result["passed"]
    assert result["candidate_vs_native"]["bitwise_unequal"] == 1
    assert (
        result["runtime"]["lowering"]["requested_profile_verified"] == lowering_verified
    )
    assert result["runtime"]["control"] == "split-k-1"
    with np.load(out / "outputs.npz", allow_pickle=False) as archive:
        assert archive["first"][0, 217, 2] == 1.0
