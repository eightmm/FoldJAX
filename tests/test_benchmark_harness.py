"""Regression gates for benchmark controls and provenance."""

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench import drive, run_gap_ablation, run_upstream
from bench.provenance import (
    CURRENT_RESULT_SCHEMA,
    ArtifactFingerprintError,
    artifact_identity,
    benchmark_identity,
    device_identity,
    execution_identity,
    fingerprint_path,
    foldjax_checkpoint_paths,
    foldjax_effective_environment,
    foldjax_implicit_asset_paths,
    portable_options,
    redact_machine_paths,
    require_unchanged,
    reusable_result,
    reusable_result_file,
    source_identity,
)
from bench.run_upstream import (
    command,
    upstream_checkpoint_paths,
    upstream_git_provenance,
    upstream_implicit_asset_paths,
)
from foldjax.manifest import (
    common_job_dependency_paths,
    native_input_dependency_paths,
)


def _result_identity(tmp_path: Path) -> dict[str, object]:
    job = tmp_path / "job.json"
    checkpoint = tmp_path / "weights.bin"
    job.write_text('{"entities": []}\n')
    checkpoint.write_bytes(b"checkpoint")
    return benchmark_identity(
        impl="foldjax",
        model="opendde",
        case="tiny",
        length=4,
        schedule={"num_samples": 1, "num_steps": 2, "num_recycles": 3},
        seed=101,
        options={"dtype": "float32"},
        artifacts=artifact_identity(
            job=job,
            checkpoints={"model": checkpoint},
        ),
        source={"foldjax": {"sha256": "source"}},
        runtime={"python": {"version": "test"}},
        device={"platform": "cpu"},
        execution={
            "timing_state": "cold-or-unspecified",
            "traced": False,
            "environment": {},
        },
    )


def _successful_record(identity: dict[str, object]) -> dict[str, object]:
    return {
        "schema": CURRENT_RESULT_SCHEMA,
        "identity": identity,
        "impl": identity["impl"],
        "model": identity["model"],
        "case": identity["case"],
        "length": identity["length"],
        "schedule": identity["schedule"],
        "seed": identity["seed"],
        "options": identity["options"],
        "artifacts": identity["artifacts"],
        "source": identity["source"],
        "runtime": identity["runtime"],
        "device": identity["device"],
        "execution": identity["execution"],
        "wall_s": 1.25,
        "peak_mib": 64.0,
        "samples": [{"scores": {"ptm": 0.5}}],
    }


def test_artifact_fingerprints_are_deterministic_and_path_free(
    tmp_path: Path,
) -> None:
    job = tmp_path / "job.json"
    a3m = tmp_path / "query.a3m"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "b.bin").write_bytes(b"second")
    (checkpoint / "nested").mkdir()
    (checkpoint / "nested" / "a.bin").write_bytes(b"first")
    job.write_text(
        json.dumps(
            {
                "entities": [
                    {
                        "type": "protein",
                        "sequence": "ACDE",
                        "unpaired_msa": a3m.name,
                    }
                ]
            }
        )
    )
    a3m.write_text(">query\nACDE\n")

    first = artifact_identity(
        job=job,
        native_input=job,
        checkpoints={"model": checkpoint},
    )
    second = artifact_identity(
        job=job,
        native_input=job,
        checkpoints={"model": checkpoint},
    )

    assert first == second
    assert str(tmp_path) not in json.dumps(first, sort_keys=True)
    assert first["checkpoints"]["model"]["files"] == 2

    (checkpoint / "nested" / "a.bin").write_bytes(b"changed")
    changed = artifact_identity(
        job=job,
        native_input=job,
        checkpoints={"model": checkpoint},
    )
    assert changed != first
    with pytest.raises(ArtifactFingerprintError, match="identity changed"):
        require_unchanged(first, changed)


def test_common_job_fingerprints_every_suffixed_msa_and_template(
    tmp_path: Path,
) -> None:
    unpaired = tmp_path / "query.deep.unpaired.custom.a3m"
    paired = tmp_path / "query.paired.suffix.a3m"
    template = tmp_path / "template.input.mmcif"
    checkpoint = tmp_path / "weights.bin"
    for path, content in (
        (unpaired, ">query\nACDE\n"),
        (paired, ">query\nACDE\n"),
        (template, "data_template\n"),
    ):
        path.write_text(content)
    checkpoint.write_bytes(b"checkpoint")
    job = tmp_path / "job.json"
    job.write_text(
        json.dumps(
            {
                "entities": [
                    {
                        "type": "protein",
                        "sequence": "ACDE",
                        "unpaired_msa": unpaired.name,
                        "paired_msa": paired.name,
                        "templates": [{"mmcif": template.name}],
                    }
                ]
            }
        )
    )

    dependencies = common_job_dependency_paths(job)
    assert dependencies == {
        "entity-0000.unpaired_msa": unpaired,
        "entity-0000.paired_msa": paired,
        "entity-0000.template-0000.mmcif": template,
    }
    identity = artifact_identity(job=job, checkpoints={"model": checkpoint})
    assert set(identity["inputs"]) == {
        "common.job",
        "common.entity-0000.unpaired_msa",
        "common.entity-0000.paired_msa",
        "common.entity-0000.template-0000.mmcif",
    }
    before = copy.deepcopy(identity)
    paired.write_text(">query\nAAAA\n")
    assert artifact_identity(job=job, checkpoints={"model": checkpoint}) != before


def test_generated_native_input_companions_are_fingerprinted_without_paths(
    tmp_path: Path,
) -> None:
    msa = tmp_path / "odd.common.a3m"
    msa.write_text(">query\nACDE\n")
    job = tmp_path / "job.json"
    job.write_text(
        json.dumps(
            {
                "entities": [
                    {
                        "type": "protein",
                        "sequence": "ACDE",
                        "unpaired_msa": msa.name,
                    }
                ]
            }
        )
    )
    native_dir = tmp_path / "native"
    native_dir.mkdir()
    (native_dir / "generated.a3m").symlink_to(msa)
    native = native_dir / "input.json"
    native.write_text(json.dumps({"protein": {"unpairedMsaPath": "generated.a3m"}}))
    checkpoint = tmp_path / "weights.bin"
    checkpoint.write_bytes(b"checkpoint")

    dependencies = native_input_dependency_paths(native)
    assert dependencies == {"reference-0000": msa.resolve()}
    identity = artifact_identity(
        job=job,
        native_input=native,
        checkpoints={"model": checkpoint},
    )
    assert "native.reference-0000" in identity["inputs"]
    assert str(tmp_path) not in json.dumps(identity, sort_keys=True)


def test_artifact_fingerprint_rejects_links_and_special_files(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"weights")
    linked = tmp_path / "linked.bin"
    linked.symlink_to(source)
    with pytest.raises(ArtifactFingerprintError, match="is a symlink"):
        fingerprint_path(linked)

    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "linked.bin").symlink_to(source)
    with pytest.raises(ArtifactFingerprintError, match="entry is a symlink"):
        fingerprint_path(tree)

    fifo = tmp_path / "checkpoint.pipe"
    os.mkfifo(fifo)
    with pytest.raises(ArtifactFingerprintError, match="not a regular file"):
        fingerprint_path(fifo)

    a3m_directory = tmp_path / "msa"
    a3m_directory.mkdir()
    job = tmp_path / "job.json"
    job.write_text(
        json.dumps(
            {
                "entities": [
                    {
                        "type": "protein",
                        "sequence": "ACDE",
                        "unpaired_msa": a3m_directory.name,
                    }
                ]
            }
        )
    )
    with pytest.raises(ArtifactFingerprintError, match="input must be a regular file"):
        artifact_identity(
            job=job,
            checkpoints={"model": source},
        )


def test_boltz_checkpoint_identity_includes_the_selected_index(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "boltz2_conf.safetensors"
    sidecar = tmp_path / "boltz2_conf.safetensors.json"
    checkpoint.write_bytes(b"tensor-payload")
    sidecar.write_text('{"weight": {"dtype": "float32"}}\n')

    selected = foldjax_checkpoint_paths("boltz2", checkpoint)

    assert selected == {"model": checkpoint, "model_index": sidecar}
    assert (
        fingerprint_path(selected["model"])["sha256"]
        != fingerprint_path(selected["model_index"])["sha256"]
    )


def test_boltz_implicit_identity_selects_only_canonical_protein_pickles(
    tmp_path: Path,
) -> None:
    from foldjax.models.boltz2.data.const import canonical_tokens

    weights = tmp_path / "weights" / "boltz2_conf.safetensors"
    weights.parent.mkdir()
    weights.write_bytes(b"weights")
    mols = weights.parent / "mols"
    mols.mkdir()
    for token in canonical_tokens:
        (mols / f"{token}.pkl").write_bytes(token.encode())
    unused = mols / "LIG.pkl"
    unused.write_bytes(b"unused")

    selected = foldjax_implicit_asset_paths("boltz2", weights)

    assert len(selected) == len(canonical_tokens)
    assert set(selected) == {
        f"boltz.canonical-protein.{token}" for token in canonical_tokens
    }
    assert unused not in selected.values()
    before = {label: fingerprint_path(path) for label, path in sorted(selected.items())}
    unused.write_bytes(b"unused changed")
    assert {
        label: fingerprint_path(path) for label, path in sorted(selected.items())
    } == before
    (mols / "ALA.pkl").write_bytes(b"selected changed")
    assert {
        label: fingerprint_path(path) for label, path in sorted(selected.items())
    } != before


def test_alphafold3_implicit_identity_binds_the_managed_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from foldjax.models.alphafold3 import build

    source_package = tmp_path / "source/alphafold3"
    runtime_root = tmp_path / "runtime"
    runtime_package = runtime_root / "alphafold3"
    source = source_package / "model/example.py"
    copied = runtime_package / "model/example.py"
    source.parent.mkdir(parents=True)
    copied.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n")
    copied.write_text("VALUE = 1\n")
    extension = runtime_package / "cpp.test.so"
    extension.write_bytes(b"extension")
    generated = {
        runtime_package / "constants/converters/ccd.pickle": b"ccd",
        runtime_package
        / "constants/converters/chemical_component_sets.pickle": b"sets",
        runtime_root / ".foldjax-extension-complete": b"key\n",
        runtime_root / ".foldjax-complete": b"key\n",
    }
    for path, payload in generated.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    for name in (
        "components.cif",
        "mmcif_ddl.dic",
        "mmcif_ma.dic",
        "mmcif_pdbx.dic",
    ):
        path = runtime_root / "share/libcifpp" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    feedback = runtime_package / "model/__pycache__/example.pyc"
    feedback.parent.mkdir()
    feedback.write_bytes(b"feedback")

    monkeypatch.setattr(build, "ensure_ready", lambda: None)
    monkeypatch.setattr(build, "source_package", lambda: source_package)
    monkeypatch.setattr(build, "runtime_package", lambda: runtime_package)
    monkeypatch.setattr(build, "runtime_root", lambda: runtime_root)
    monkeypatch.setattr(build, "compiled_module", lambda: extension)

    selected = foldjax_implicit_asset_paths("alphafold3", tmp_path / "weights")

    assert copied in selected.values()
    assert extension in selected.values()
    assert all(path in selected.values() for path in generated)
    assert feedback not in selected.values()
    before = {label: fingerprint_path(path) for label, path in selected.items()}
    feedback.write_bytes(b"different feedback")
    assert {label: fingerprint_path(path) for label, path in selected.items()} == before
    copied.write_text("VALUE = 2\n")
    assert {label: fingerprint_path(path) for label, path in selected.items()} != before

    external = tmp_path / "external-libcifpp"
    external.mkdir()
    for name in (
        "components.cif",
        "mmcif_ddl.dic",
        "mmcif_ma.dic",
        "mmcif_pdbx.dic",
    ):
        (external / name).write_bytes(f"external-{name}".encode())
    monkeypatch.setenv("LIBCIFPP_DATA_DIR", str(external))

    external_selected = foldjax_implicit_asset_paths("alphafold3", tmp_path / "weights")

    assert {
        path
        for label, path in external_selected.items()
        if label.startswith("alphafold3.runtime.libcifpp.")
    } == set(external.iterdir())
    assert not any(
        path.is_relative_to(runtime_root / "share/libcifpp")
        for label, path in external_selected.items()
        if label.startswith("alphafold3.runtime.libcifpp.")
    )


def test_alphafold3_explicit_source_binds_external_execution_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from foldjax.models.alphafold3 import build

    checkout = tmp_path / "alphafold3-checkout"
    runner = checkout / "run_alphafold.py"
    package = checkout / "src/alphafold3"
    runner.parent.mkdir(parents=True)
    package.mkdir(parents=True)
    runner.write_text("RUNNER = 1\n")
    package_init = package / "__init__.py"
    package_init.write_text("PACKAGE = 1\n")
    model = package / "model.py"
    model.write_text("MODEL = 1\n")
    feedback = package / "__pycache__/model.pyc"
    feedback.parent.mkdir()
    feedback.write_bytes(b"feedback")
    libcifpp = checkout / "src/share/libcifpp"
    libcifpp.mkdir(parents=True)
    for name in (
        "components.cif",
        "mmcif_ddl.dic",
        "mmcif_ma.dic",
        "mmcif_pdbx.dic",
    ):
        (libcifpp / name).write_bytes(name.encode())
    monkeypatch.delenv("LIBCIFPP_DATA_DIR", raising=False)
    monkeypatch.setattr(
        build,
        "ensure_ready",
        lambda: pytest.fail("explicit source must not prepare the managed runtime"),
    )

    options = {"source": str(checkout)}
    selected = foldjax_implicit_asset_paths(
        "alphafold3",
        tmp_path / "weights",
        options=options,
    )

    assert runner in selected.values()
    assert package_init in selected.values()
    assert model in selected.values()
    assert feedback not in selected.values()
    assert set(libcifpp.iterdir()) <= set(selected.values())
    effective = foldjax_effective_environment(
        {},
        model="alphafold3",
        options=options,
    )
    assert effective["LIBCIFPP_DATA_DIR"] == str(libcifpp)

    before = {label: fingerprint_path(path) for label, path in selected.items()}
    feedback.write_bytes(b"changed feedback")
    assert {label: fingerprint_path(path) for label, path in selected.items()} == before
    model.write_text("MODEL = 2\n")
    assert {label: fingerprint_path(path) for label, path in selected.items()} != before


def test_esmfold2_identity_binds_structure_config_and_selected_esmc_shards(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "weights/model.safetensors"
    weights.parent.mkdir()
    weights.write_bytes(b"structure")
    (weights.parent / "config.json").write_text('{"trunk": {}}\n')
    esmc = weights.parent / "esmc"
    esmc.mkdir()
    (esmc / "config.json").write_text('{"hidden_size": 16}\n')
    shards = ("model-00001.safetensors", "model-00002.safetensors")
    for name in shards:
        (esmc / name).write_bytes(name.encode())
    unused = esmc / "unused.safetensors"
    unused.write_bytes(b"unused")
    (esmc / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "esmc.embed.weight": shards[0],
                    "esmc.transformer.layer.weight": shards[1],
                }
            }
        )
    )

    checkpoints = foldjax_checkpoint_paths("esmfold2", weights)
    selected = foldjax_implicit_asset_paths("esmfold2", weights)

    assert checkpoints == {"model": weights}
    assert weights.parent / "config.json" in selected.values()
    assert esmc / "config.json" in selected.values()
    assert esmc / "model.safetensors.index.json" in selected.values()
    assert all(esmc / name in selected.values() for name in shards)
    assert unused not in selected.values()
    before = {label: fingerprint_path(path) for label, path in selected.items()}
    unused.write_bytes(b"unused changed")
    assert {label: fingerprint_path(path) for label, path in selected.items()} == before
    (esmc / shards[0]).write_bytes(b"selected changed")
    assert {label: fingerprint_path(path) for label, path in selected.items()} != before

    structure_only = foldjax_implicit_asset_paths(
        "esmfold2",
        weights,
        options={"no_language_model": True},
    )
    assert structure_only == {
        "esmfold2.structure_config": weights.parent / "config.json"
    }


def test_openfold3_identity_binds_biotite_runtime_ccd(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from biotite.structure.info import ccd as biotite_ccd

    ccd = tmp_path / "components.bcif"
    ccd.write_bytes(b"ccd generation one")
    monkeypatch.setattr(biotite_ccd, "_CCD_FILE", ccd)

    selected = foldjax_implicit_asset_paths("openfold3", tmp_path / "weights")

    assert selected == {"openfold3.biotite_components": ccd}
    before = fingerprint_path(selected["openfold3.biotite_components"])
    ccd.write_bytes(b"ccd generation two")
    assert fingerprint_path(selected["openfold3.biotite_components"]) != before

    custom = tmp_path / "custom-components.cif"
    custom.write_bytes(b"custom ccd")
    custom_selected = foldjax_implicit_asset_paths(
        "openfold3",
        tmp_path / "weights",
        options={"ccd_file_path": str(custom)},
    )
    assert custom_selected == {"openfold3.biotite_components": custom}


def test_upstream_boltz_implicit_identity_selects_only_canonical_pickles(
    tmp_path: Path,
    monkeypatch,
) -> None:
    upstream_root = tmp_path / "upstreams"
    const = upstream_root / "boltz/src/boltz/data/const.py"
    const.parent.mkdir(parents=True)
    const.write_text('canonical_tokens = ["ALA", "UNK"]\n')
    checkpoint = tmp_path / "cache" / "boltz2_conf.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"weights")
    (checkpoint.parent / "mols.tar").write_bytes(b"archive sentinel")
    (checkpoint.parent / "boltz2_aff.ckpt").write_bytes(b"affinity sentinel")
    mols = checkpoint.parent / "mols"
    mols.mkdir()
    for token in ("ALA", "UNK"):
        (mols / f"{token}.pkl").write_bytes(token.encode())
    unused = mols / "LIG.pkl"
    unused.write_bytes(b"unused")
    native_input = tmp_path / "boltz2_input.yaml"
    native_input.write_text(
        json.dumps(
            {
                "version": 1,
                "sequences": [
                    {
                        "protein": {
                            "id": "A",
                            "sequence": "AX",
                            "msa": "empty",
                        }
                    }
                ],
            }
        )
    )
    monkeypatch.setenv("FOLDJAX_BENCH_ROOT", str(upstream_root))
    monkeypatch.setattr(
        run_upstream,
        "upstream_checkpoint_paths",
        lambda model: {"model": checkpoint},
    )
    monkeypatch.setattr(
        run_upstream,
        "_require_upstream_boltz_runtime_inventory",
        lambda source, tokens: None,
    )

    selected = upstream_implicit_asset_paths(
        "boltz2",
        native_input=native_input,
    )

    assert selected == {
        "boltz.canonical-protein.ALA": mols / "ALA.pkl",
        "boltz.canonical-protein.UNK": mols / "UNK.pkl",
    }
    before = {label: fingerprint_path(path) for label, path in selected.items()}
    unused.write_bytes(b"unused changed")
    assert {label: fingerprint_path(path) for label, path in selected.items()} == before
    (mols / "ALA.pkl").write_bytes(b"selected changed")
    assert {label: fingerprint_path(path) for label, path in selected.items()} != before


@pytest.mark.parametrize(
    ("runtime_source", "runtime_tokens", "matches"),
    [
        ("reviewed", ["ALA", "UNK"], True),
        ("other", ["ALA", "UNK"], False),
        ("reviewed", ["ALA", "SEC", "UNK"], False),
    ],
)
def test_upstream_boltz_runtime_inventory_matches_reviewed_source(
    tmp_path: Path,
    monkeypatch,
    runtime_source: str,
    runtime_tokens: list[str],
    matches: bool,
) -> None:
    reviewed = tmp_path / "reviewed/const.py"
    other = tmp_path / "other/const.py"
    reviewed.parent.mkdir()
    other.parent.mkdir()
    reviewed.write_text('canonical_tokens = ["ALA", "UNK"]\n')
    other.write_text('canonical_tokens = ["ALA", "UNK"]\n')
    selected_source = reviewed if runtime_source == "reviewed" else other
    monkeypatch.setattr(
        run_upstream.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout=json.dumps(
                {"source": str(selected_source), "tokens": runtime_tokens}
            ),
            stderr="",
        ),
    )

    if matches:
        run_upstream._require_upstream_boltz_runtime_inventory(
            reviewed,
            ("ALA", "UNK"),
        )
    else:
        with pytest.raises(RuntimeError, match="reviewed checkout"):
            run_upstream._require_upstream_boltz_runtime_inventory(
                reviewed,
                ("ALA", "UNK"),
            )


@pytest.mark.parametrize(
    "document",
    [
        {
            "version": 1,
            "sequences": [{"ligand": {"id": "L", "ccd": "ATP"}}],
        },
        {
            "version": 1,
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": "ACD",
                        "msa": "empty",
                        "modifications": [{"ccd": "MSE", "position": 1}],
                    }
                }
            ],
        },
    ],
)
def test_upstream_boltz_narrow_assets_reject_noncanonical_jobs(
    tmp_path: Path,
    document: dict[str, object],
) -> None:
    native_input = tmp_path / "boltz2_input.yaml"
    native_input.write_text(json.dumps(document))

    with pytest.raises(ArtifactFingerprintError, match="pure unmodified protein"):
        upstream_implicit_asset_paths("boltz2", native_input=native_input)


@pytest.mark.parametrize("missing", ["mols.tar", "boltz2_aff.ckpt"])
def test_upstream_boltz_assets_require_download_sentinels(
    tmp_path: Path,
    monkeypatch,
    missing: str,
) -> None:
    upstream_root = tmp_path / "upstreams"
    const = upstream_root / "boltz/src/boltz/data/const.py"
    const.parent.mkdir(parents=True)
    const.write_text('canonical_tokens = ["ALA", "UNK"]\n')
    checkpoint = tmp_path / "cache/boltz2_conf.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"weights")
    for name in {"mols.tar", "boltz2_aff.ckpt"} - {missing}:
        (checkpoint.parent / name).write_bytes(b"sentinel")
    native_input = tmp_path / "boltz2_input.yaml"
    native_input.write_text(
        json.dumps(
            {
                "version": 1,
                "sequences": [
                    {
                        "protein": {
                            "id": "A",
                            "sequence": "ACD",
                            "msa": "empty",
                        }
                    }
                ],
            }
        )
    )
    monkeypatch.setenv("FOLDJAX_BENCH_ROOT", str(upstream_root))
    monkeypatch.setattr(
        run_upstream,
        "upstream_checkpoint_paths",
        lambda model: {"model": checkpoint},
    )

    with pytest.raises(ArtifactFingerprintError, match="pre-existing"):
        upstream_implicit_asset_paths("boltz2", native_input=native_input)


@pytest.mark.parametrize(
    "source_text",
    [
        'canonical_tokens = ["ALA"]\ncanonical_tokens = ["ALA", "UNK"]\n',
        'canonical_tokens = ["ALA"]\ncanonical_tokens.append("UNK")\n',
    ],
)
def test_upstream_boltz_assets_reject_dynamic_canonical_inventory(
    tmp_path: Path,
    monkeypatch,
    source_text: str,
) -> None:
    upstream_root = tmp_path / "upstreams"
    const = upstream_root / "boltz/src/boltz/data/const.py"
    const.parent.mkdir(parents=True)
    const.write_text(source_text)
    checkpoint = tmp_path / "cache/boltz2_conf.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"weights")
    (checkpoint.parent / "mols.tar").write_bytes(b"archive sentinel")
    (checkpoint.parent / "boltz2_aff.ckpt").write_bytes(b"affinity sentinel")
    native_input = tmp_path / "boltz2_input.yaml"
    native_input.write_text(
        json.dumps(
            {
                "version": 1,
                "sequences": [
                    {
                        "protein": {
                            "id": "A",
                            "sequence": "ACD",
                            "msa": "empty",
                        }
                    }
                ],
            }
        )
    )
    monkeypatch.setenv("FOLDJAX_BENCH_ROOT", str(upstream_root))
    monkeypatch.setattr(
        run_upstream,
        "upstream_checkpoint_paths",
        lambda model: {"model": checkpoint},
    )

    with pytest.raises(RuntimeError, match="canonical tokens"):
        upstream_implicit_asset_paths("boltz2", native_input=native_input)


def test_upstream_implicit_assets_match_each_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ccd = tmp_path / "components.bcif"
    ccd.write_bytes(b"ccd")
    monkeypatch.setattr(run_upstream, "_upstream_biotite_ccd", lambda repo: ccd)

    assert set(upstream_implicit_asset_paths("protenix")) == {
        "ccd.components",
        "ccd.rdkit_cache",
    }
    assert set(upstream_implicit_asset_paths("opendde")) == {
        "ccd.components",
        "ccd.rdkit_cache",
    }
    assert set(upstream_implicit_asset_paths("openfold3")) == {
        "openfold3.runner_yaml",
        "openfold3.biotite_components",
    }


def test_upstream_openfold3_resolves_biotite_ccd_from_its_venv(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "openfold3-v031"
    repo.mkdir()
    ccd = tmp_path / "venv/components.bcif"
    ccd.parent.mkdir()
    ccd.write_bytes(b"ccd")
    monkeypatch.setattr(
        run_upstream.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout=json.dumps({"ccd": str(ccd)}),
            stderr="",
        ),
    )

    assert run_upstream._upstream_biotite_ccd(repo) == ccd


def test_source_and_execution_identity_are_path_free_and_scoped(
    tmp_path: Path,
) -> None:
    (tmp_path / "src/foldjax").mkdir(parents=True)
    source = tmp_path / "src/foldjax/model.py"
    source.write_text("VALUE = 1\n")
    cache = tmp_path / "src/foldjax/__pycache__"
    cache.mkdir()
    bytecode = cache / "model.pyc"
    bytecode.write_bytes(b"feedback")
    (tmp_path / "bench/peakhook").mkdir(parents=True)
    (tmp_path / "bench/peakhook/sitecustomize.py").write_text("HOOK = True\n")
    for relative in (
        "tests/models/alphafold3/scripts/run_sample_shard_end_to_end_arm.py",
        "bench/drive.py",
        "bench/run_gap_ablation.py",
        "tests/models/opendde/scripts/run_structural_relp_end_to_end_arm.py",
        "bench/openfold3_runner.yml",
        "tests/models/openfold3/scripts/run_confidence_schedule_end_to_end_arm.py",
        "bench/provenance.py",
        "bench/run_foldjax.py",
        "bench/run_upstream.py",
        "bench/spec.py",
        "bench/structures.py",
        "bench/trace.py",
        "pyproject.toml",
        "uv.lock",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)

    first = source_identity(tmp_path)
    bytecode.write_bytes(b"different feedback")
    assert source_identity(tmp_path) == first
    source.write_text("VALUE = 2\n")
    assert source_identity(tmp_path) != first
    assert str(tmp_path) not in json.dumps(first, sort_keys=True)

    execution = execution_identity(
        {
            "CUDA_HOME": str(tmp_path / "cuda"),
            "CUDA_VISIBLE_DEVICES": "GPU-private-uuid",
            "PATH": str(tmp_path / "bin"),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "SERVICE_TOKEN": "must-not-appear",
        },
        timing_state="warm-after-successful-prefill",
        traced=True,
    )
    serialized = json.dumps(execution, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert "GPU-private-uuid" not in serialized
    assert "must-not-appear" not in serialized
    assert execution["timing_state"] == "warm-after-successful-prefill"
    assert execution["traced"] is True


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("JAX_DISABLE_JIT", "1"),
        ("CUDA_DEVICE_ORDER", "PCI_BUS_ID"),
        ("JAX_RANDOM_SEED_OFFSET", "17"),
    ],
)
def test_execution_identity_binds_runtime_control_environment(
    name: str,
    value: str,
) -> None:
    baseline = execution_identity(
        {"JAX_PLATFORMS": "cpu"},
        timing_state="cold-or-unspecified",
        traced=False,
    )
    changed = execution_identity(
        {"JAX_PLATFORMS": "cpu", name: value},
        timing_state="cold-or-unspecified",
        traced=False,
    )

    assert changed != baseline
    assert name in changed["environment"]


def test_driver_and_child_project_the_same_alphafold3_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import foldjax
    from foldjax.models.alphafold3 import build

    runtime_root = tmp_path / "runtime-generation"
    (runtime_root / "share/libcifpp").mkdir(parents=True)
    job = tmp_path / "job.json"
    job.write_text('{"entities": []}\n')
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"weights")
    case = SimpleNamespace(name="tiny", length=4, job=job)
    monkeypatch.delenv("LIBCIFPP_DATA_DIR", raising=False)
    monkeypatch.setattr(build, "runtime_root", lambda: runtime_root)
    monkeypatch.setattr(build, "is_managed_origin", lambda _path: False)
    monkeypatch.setattr(
        foldjax,
        "resolve_request",
        lambda _request: SimpleNamespace(
            input=job,
            weights=weights,
            options={},
        ),
    )
    monkeypatch.setattr(drive, "artifact_identity", lambda **_kwargs: {})
    monkeypatch.setattr(drive, "foldjax_checkpoint_paths", lambda *_args: {})
    monkeypatch.setattr(
        drive, "foldjax_implicit_asset_paths", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(drive, "source_identity", lambda _repo: {"source": 1})
    monkeypatch.setattr(drive, "runtime_identity", lambda: {"runtime": 1})
    monkeypatch.setattr(
        drive,
        "device_identity",
        lambda _environment: {"platform": "cpu"},
    )

    driver_identity = drive._expected_identity(
        "alphafold3",
        "foldjax",
        case,
        tmp_path / "work",
    )
    child_environment = dict(os.environ)
    child_environment["PYTHONPATH"] = str(drive.REPO)
    child_execution = execution_identity(
        foldjax_effective_environment(
            child_environment,
            model="alphafold3",
        ),
        timing_state="warm-after-successful-prefill",
        traced=False,
    )

    assert driver_identity["execution"] == child_execution
    assert "LIBCIFPP_DATA_DIR" in child_execution["environment"]


def test_foldjax_runner_rechecks_live_environment_after_prediction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import jax

    import foldjax
    from bench import run_foldjax

    job = tmp_path / "job.json"
    job.write_text('{"entities": []}\n')
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"weights")
    case = SimpleNamespace(name="tiny", length=4, job=job)
    request = SimpleNamespace(
        input=job,
        weights=weights,
        options={},
    )
    monkeypatch.delenv("JAX_DISABLE_JIT", raising=False)
    monkeypatch.setattr("bench.spec.cases", lambda: (case,))
    monkeypatch.setattr(foldjax, "resolve_request", lambda _request: request)

    def predict(_request):
        monkeypatch.setenv("JAX_DISABLE_JIT", "1")
        return SimpleNamespace(samples=())

    monkeypatch.setattr(foldjax, "predict", predict)
    monkeypatch.setattr(run_foldjax, "artifact_identity", lambda **_kwargs: {})
    monkeypatch.setattr(run_foldjax, "foldjax_checkpoint_paths", lambda *_args: {})
    monkeypatch.setattr(
        run_foldjax,
        "foldjax_implicit_asset_paths",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(run_foldjax, "source_identity", lambda _repo: {"source": 1})
    monkeypatch.setattr(run_foldjax, "runtime_identity", lambda: {"runtime": 1})
    monkeypatch.setattr(
        run_foldjax,
        "device_identity",
        lambda _environment: {"platform": "cpu"},
    )
    monkeypatch.setattr(
        jax,
        "local_devices",
        lambda: [SimpleNamespace(memory_stats=lambda: {})],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench.run_foldjax",
            "--model",
            "opendde",
            "--case",
            "tiny",
            "--output-dir",
            str(tmp_path / "out"),
            "--warmup",
        ],
    )

    with pytest.raises(
        ArtifactFingerprintError,
        match="identity changed during prediction",
    ):
        run_foldjax.main()


def test_device_identity_omits_uuid_and_host_fields(monkeypatch) -> None:
    output = "0, NVIDIA Test GPU, 12.0, 97887, 590.44, GPU-secret-uuid\n"
    monkeypatch.setattr(
        "bench.provenance.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=output,
            stderr="",
        ),
    )

    identity = device_identity({"CUDA_VISIBLE_DEVICES": "GPU-secret-uuid"})

    assert identity == {
        "platform": "cuda",
        "model": "NVIDIA Test GPU",
        "compute_capability": "12.0",
        "total_memory_mib": 97887,
        "driver_version": "590.44",
    }
    assert "uuid" not in json.dumps(identity).lower()


def test_portable_options_hash_paths_and_secrets() -> None:
    options = portable_options(
        {
            "compute_dtype": "bfloat16",
            "mols": "/machine/private/mols",
            "source": "/machine/private/alphafold3",
            "api_token": "secret-value",
        }
    )

    assert options["compute_dtype"] == "bfloat16"
    assert options["mols"] == {
        "sha256": hashlib.sha256(b"/machine/private/mols").hexdigest()
    }
    assert options["source"] == {
        "sha256": hashlib.sha256(b"/machine/private/alphafold3").hexdigest()
    }
    assert "secret-value" not in json.dumps(options)
    diagnostic = redact_machine_paths(
        f"failed under {Path('/machine/private/work') / 'run.log'}",
        {"work": Path("/machine/private/work")},
    )
    assert diagnostic == "failed under <work>/run.log"


def test_skip_existing_requires_a_successful_current_schema_identity(
    tmp_path: Path,
) -> None:
    identity = _result_identity(tmp_path)
    record = _successful_record(identity)
    assert reusable_result(record, expected_identity=identity)

    for field, replacement in (
        ("schema", CURRENT_RESULT_SCHEMA - 1),
        ("schema", True),
        ("failed", True),
        ("returncode", False),
        ("samples", []),
        ("peak_mib", "unknown"),
        ("peak_mib", 0),
        ("wall_s", math.nan),
        ("peak_mib", math.inf),
        ("model", "different"),
    ):
        invalid = copy.deepcopy(record)
        invalid[field] = replacement
        assert not reusable_result(invalid, expected_identity=identity)

    for samples in (
        [{"scores": {}}],
        [{"scores": {"ptm": math.nan}}],
        [{"scores": {"ptm": True}}],
        [{"scores": {"ptm": "0.5"}}],
        [
            {"scores": {"ptm": 0.5}},
            {"scores": {"ptm": 0.6}},
        ],
    ):
        invalid = copy.deepcopy(record)
        invalid["samples"] = samples
        assert not reusable_result(invalid, expected_identity=identity)

    invalid = copy.deepcopy(record)
    invalid["execution"] = {
        **identity["execution"],
        "traced": True,
    }
    assert not reusable_result(invalid, expected_identity=identity)

    different = copy.deepcopy(identity)
    different["seed"] = 102
    assert not reusable_result(record, expected_identity=different)


def test_skip_existing_reads_only_bounded_regular_result_files(
    tmp_path: Path,
) -> None:
    identity = _result_identity(tmp_path)
    row = tmp_path / "result.json"
    row.write_text(json.dumps(_successful_record(identity)))
    assert reusable_result_file(row, expected_identity=identity)

    linked = tmp_path / "linked.json"
    linked.symlink_to(row)
    assert not reusable_result_file(linked, expected_identity=identity)

    fifo = tmp_path / "result.pipe"
    os.mkfifo(fifo)
    assert not reusable_result_file(fifo, expected_identity=identity)


def test_driver_skips_only_an_exact_current_result(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    identity = _result_identity(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    row = results / "opendde-foldjax-tiny.json"
    row.write_text(json.dumps(_successful_record(identity)))
    case = SimpleNamespace(name="tiny", length=4)
    monkeypatch.setattr("bench.spec.cases", lambda: (case,))
    monkeypatch.setattr(drive, "_expected_identity", lambda *args, **kwargs: identity)
    monkeypatch.setattr(
        drive,
        "gpu_idle",
        lambda: pytest.fail("an exact row must not start a GPU run"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench.drive",
            "--models",
            "opendde",
            "--impls",
            "foldjax",
            "--cases",
            "tiny",
            "--results",
            str(results),
            "--work",
            str(tmp_path / "work"),
            "--skip-existing",
        ],
    )

    assert drive.main() == 0
    assert "[skip] opendde-foldjax-tiny.json" in capsys.readouterr().out


def test_driver_reruns_an_identity_mismatch(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    identity = _result_identity(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    row = results / "opendde-foldjax-tiny.json"
    stale = _successful_record(identity)
    stale["identity"] = {**identity, "seed": 102}
    row.write_text(json.dumps(stale))
    case = SimpleNamespace(name="tiny", length=4)
    monkeypatch.setattr("bench.spec.cases", lambda: (case,))
    monkeypatch.setattr(drive, "_expected_identity", lambda *args, **kwargs: identity)
    monkeypatch.setattr(
        drive,
        "gpu_idle",
        lambda: (_ for _ in ()).throw(RuntimeError("rerun reached")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench.drive",
            "--models",
            "opendde",
            "--impls",
            "foldjax",
            "--cases",
            "tiny",
            "--results",
            str(results),
            "--work",
            str(tmp_path / "work"),
            "--skip-existing",
        ],
    )

    with pytest.raises(RuntimeError, match="rerun reached"):
        drive.main()
    assert "[redo] opendde-foldjax-tiny.json" in capsys.readouterr().out
    assert not row.exists()


def test_driver_rechecks_identity_around_a_reusable_row(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    identity = _result_identity(tmp_path)
    changed = copy.deepcopy(identity)
    changed["source"] = {"foldjax": {"sha256": "changed"}}
    results = tmp_path / "results"
    results.mkdir()
    row = results / "opendde-foldjax-tiny.json"
    row.write_text(json.dumps(_successful_record(identity)))
    case = SimpleNamespace(name="tiny", length=4)
    monkeypatch.setattr("bench.spec.cases", lambda: (case,))
    identities = iter((identity, changed))
    monkeypatch.setattr(
        drive,
        "_expected_identity",
        lambda *args, **kwargs: next(identities),
    )
    monkeypatch.setattr(
        drive,
        "gpu_idle",
        lambda: (_ for _ in ()).throw(RuntimeError("rerun reached")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench.drive",
            "--models",
            "opendde",
            "--impls",
            "foldjax",
            "--cases",
            "tiny",
            "--results",
            str(results),
            "--work",
            str(tmp_path / "work"),
            "--skip-existing",
        ],
    )

    with pytest.raises(RuntimeError, match="rerun reached"):
        drive.main()
    assert "[redo]" in capsys.readouterr().out


def test_driver_warmup_failure_records_failure_and_skips_measurement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    results = tmp_path / "results"
    case = SimpleNamespace(name="tiny", length=4)
    monkeypatch.setattr("bench.spec.cases", lambda: (case,))
    monkeypatch.setattr(drive, "gpu_idle", lambda: None)
    measured: list[list[str]] = []

    def fake_run(argv, **kwargs):
        if argv[:2] == ["rm", "-rf"]:
            return subprocess.CompletedProcess(argv, 0, "", "")
        measured.append(argv)
        assert "--warmup" in argv
        return subprocess.CompletedProcess(argv, 7, "", "warmup exploded")

    monkeypatch.setattr(drive.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench.drive",
            "--models",
            "opendde",
            "--impls",
            "foldjax",
            "--cases",
            "tiny",
            "--results",
            str(results),
            "--work",
            str(tmp_path / "work"),
        ],
    )

    assert drive.main() == 0
    assert len(measured) == 1
    row = json.loads((results / "opendde-foldjax-tiny.json").read_text())
    assert row["failed"] is True
    assert row["reason"] == "warm-up failed; measurement not started"
    assert row["stderr_path"] == "opendde-foldjax-tiny.warmup.txt"
    assert not Path(row["stderr_path"]).is_absolute()


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--models", "../outside", "unknown benchmark model"),
        ("--impls", "../../outside", "unknown benchmark implementation"),
    ],
)
def test_driver_rejects_path_shaped_matrix_names_before_cleanup(
    tmp_path: Path,
    monkeypatch,
    capsys,
    flag: str,
    value: str,
    message: str,
) -> None:
    case = SimpleNamespace(name="tiny", length=4)
    monkeypatch.setattr("bench.spec.cases", lambda: (case,))
    victim = tmp_path / "outside"
    victim.mkdir()
    marker = victim / "keep.txt"
    marker.write_text("keep")
    argv = [
        "bench.drive",
        "--models",
        "opendde",
        "--impls",
        "foldjax",
        "--cases",
        "tiny",
        "--results",
        str(tmp_path / "results"),
        "--work",
        str(tmp_path / "work"),
    ]
    argv[argv.index(flag) + 1] = value
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as raised:
        drive.main()
    assert raised.value.code == 2
    assert marker.read_text() == "keep"
    assert message in capsys.readouterr().err


def test_openfold3_upstream_keeps_the_runner_yaml_seed() -> None:
    argv, _cwd, _environment = command(
        "openfold3",
        Path("query.json"),
        Path("output"),
        {"num_samples": 5, "num_steps": 200, "num_recycles": 10},
        101,
    )

    assert "--num_model_seeds" not in argv
    assert argv[argv.index("--use_templates") + 1] == "false"
    assert (
        Path(argv[argv.index("--inference_ckpt_path") + 1])
        == (upstream_checkpoint_paths("openfold3")["model"])
    )
    runner_yaml = Path(argv[argv.index("--runner_yaml") + 1])
    assert "seeds:\n    - 101" in runner_yaml.read_text()


def test_upstream_root_can_be_relocated(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FOLDJAX_BENCH_ROOT", str(tmp_path))

    _argv, cwd, _environment = command(
        "boltz2",
        Path("query.yaml"),
        Path("output"),
        {"num_samples": 5, "num_steps": 200, "num_recycles": 10},
        101,
    )

    assert cwd == tmp_path / "boltz"


def test_upstream_provenance_requires_the_exact_tracked_diff(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "upstream"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    source = repo / "model.py"
    source.write_text("released = True\n")
    subprocess.run(["git", "-C", str(repo), "add", "model.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=FoldJAX test",
            "-c",
            "user.email=foldjax@example.invalid",
            "commit",
            "-qm",
            "initial",
        ],
        check=True,
    )

    clean = upstream_git_provenance(repo)
    assert clean["tracked_dirty"] is False
    assert clean["tracked_diff_sha256"] is None

    # An untracked file can be imported by Python, so provenance is ambiguous
    # even when its present contents look harmless.
    (repo / "profile.json").write_text("{}\n")
    with pytest.raises(RuntimeError, match="has untracked files"):
        upstream_git_provenance(repo)
    (repo / "profile.json").unlink()

    # Git-ignored source, config, executables, and native extensions can all
    # change execution. Exact virtualenv/tool-cache directories are excluded.
    (repo / ".gitignore").write_text("*.so\ngenerated/\n.venv/\ntool-cache/\n.env*\n")
    subprocess.run(["git", "-C", str(repo), "add", ".gitignore"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=FoldJAX test",
            "-c",
            "user.email=foldjax@example.invalid",
            "commit",
            "-qm",
            "ignore native extension",
        ],
        check=True,
    )
    extension = repo / "fast_layer_norm.so"
    extension.write_bytes(b"native-extension")
    native = upstream_git_provenance(repo)
    assert native["ignored_runtime_artifacts"] == [
        {
            "path": "fast_layer_norm.so",
            "kind": "file",
            "size": len(b"native-extension"),
            "sha256": hashlib.sha256(b"native-extension").hexdigest(),
        }
    ]

    linked = repo / "linked_layer_norm.so"
    linked.symlink_to(extension)
    with pytest.raises(RuntimeError, match="runtime artifact is a symlink"):
        upstream_git_provenance(repo)
    linked.unlink()

    generated = repo / "generated"
    generated.mkdir()
    runner = generated / "runner.py"
    config = generated / "profile.yaml"
    executable = generated / "launcher"
    runner.write_text("BACKEND = 'torch'\n")
    config.write_text("dtype: float32\n")
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    for excluded in (repo / ".venv/ignored.py", repo / "tool-cache/ignored.py"):
        excluded.parent.mkdir()
        excluded.write_text("IGNORED = True\n")
    expanded = upstream_git_provenance(repo)["ignored_runtime_artifacts"]
    assert {entry["path"] for entry in expanded} == {
        "fast_layer_norm.so",
        "generated/launcher",
        "generated/profile.yaml",
        "generated/runner.py",
    }
    old_config = next(
        entry for entry in expanded if entry["path"] == "generated/profile.yaml"
    )
    config.write_text("dtype: bfloat16\n")
    changed = upstream_git_provenance(repo)["ignored_runtime_artifacts"]
    new_config = next(
        entry for entry in changed if entry["path"] == "generated/profile.yaml"
    )
    assert new_config["sha256"] != old_config["sha256"]

    linked_config = generated / "linked.py"
    linked_config.symlink_to(runner)
    with pytest.raises(RuntimeError, match="runtime artifact is a symlink"):
        upstream_git_provenance(repo)
    linked_config.unlink()

    special = generated / "special.py"
    os.mkfifo(special)
    with pytest.raises(RuntimeError, match="not a regular file"):
        upstream_git_provenance(repo)
    special.unlink()

    secret = repo / ".env.secret"
    secret.write_text("TOKEN=do-not-hash\n")
    with pytest.raises(RuntimeError, match="secret-like name"):
        upstream_git_provenance(repo)
    secret.unlink()

    # A tracked edit must be reviewed and pinned exactly.
    source.write_text("released = False\n")
    patch = subprocess.run(
        ["git", "-C", str(repo), "diff", "--binary", "--no-ext-diff", "HEAD", "--"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    digest = hashlib.sha256(patch).hexdigest()

    with pytest.raises(RuntimeError, match="has tracked changes"):
        upstream_git_provenance(repo)
    with pytest.raises(RuntimeError, match="does not match"):
        upstream_git_provenance(repo, expected_diff_sha256="0" * 64)

    dirty = upstream_git_provenance(repo, expected_diff_sha256=digest)
    assert dirty["tracked_dirty"] is True
    assert dirty["tracked_diff_sha256"] == digest
    assert dirty["tracked_status"] == [" M model.py"]


@pytest.mark.parametrize(
    ("structure_count", "expected_reason"),
    [
        (0, "exited 0 but produced no structures"),
        (1, "produced 1 structures for 5 requested samples"),
        (4, "produced 4 structures for 5 requested samples"),
        (5, "measurement record has missing or non-finite timing"),
    ],
)
def test_upstream_rejects_nominal_success_without_structure_or_peak(
    tmp_path: Path,
    monkeypatch,
    structure_count: int,
    expected_reason: str,
) -> None:
    job = tmp_path / "job.json"
    job.write_text('{"entities": []}\n')
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint")
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"asset")
    output = tmp_path / "output"
    row = tmp_path / "result.json"
    case = SimpleNamespace(name="tiny", length=4, job=job)

    monkeypatch.setattr("bench.spec.cases", lambda: (case,))
    monkeypatch.setattr(
        run_upstream,
        "command",
        lambda *args: (["runner"], tmp_path, {}),
    )
    monkeypatch.setattr(
        run_upstream,
        "upstream_checkpoint_paths",
        lambda model: {"model": checkpoint},
    )
    monkeypatch.setattr(
        run_upstream,
        "upstream_implicit_asset_paths",
        lambda model, **kwargs: {"asset": asset},
    )
    monkeypatch.setattr(
        run_upstream,
        "upstream_git_provenance",
        lambda *args, **kwargs: {
            "revision": "a" * 40,
            "tracked_dirty": False,
            "tracked_diff_sha256": None,
            "tracked_status": [],
            "ignored_runtime_artifacts": [],
        },
    )
    monkeypatch.setattr(
        run_upstream,
        "upstream_runtime_versions",
        lambda repo: {"python": "test", "distributions": {}},
    )
    monkeypatch.setattr(run_upstream, "source_identity", lambda repo: {"source": 1})
    monkeypatch.setattr(run_upstream, "runtime_identity", lambda: {"runtime": 1})
    monkeypatch.setattr(
        run_upstream,
        "device_identity",
        lambda environment: {"platform": "cpu"},
    )
    monkeypatch.setattr(
        run_upstream,
        "scores",
        lambda model, out: [{"ptm": 0.5} for _ in range(5)],
    )
    if structure_count:
        monkeypatch.setattr(
            run_upstream,
            "produced_structures",
            lambda out: [
                out / f"prediction_{index}.cif" for index in range(structure_count)
            ],
        )
    monkeypatch.setattr(
        run_upstream.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout="",
            stderr="upstream emitted no outputs",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench.run_upstream",
            "--model",
            "opendde",
            "--case",
            "tiny",
            "--job",
            str(job),
            "--output-dir",
            str(output),
            "--json-out",
            str(row),
        ],
    )

    assert run_upstream.main() == 1
    record = json.loads(row.read_text())
    assert record["failed"] is True
    assert expected_reason in record["reason"]
    assert "argv" not in record
    assert "execution_environment" not in record


@pytest.mark.parametrize(
    ("model", "arm", "message"),
    [
        ("opendde", "sharded", "only applies to model 'alphafold3'"),
        ("alphafold3", "dense", "arm must be one of batched, sharded"),
    ],
)
def test_gap_ablation_rejects_mislabeled_arms(
    tmp_path: Path,
    monkeypatch,
    capsys,
    model: str,
    arm: str,
    message: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_gap_ablation",
            "--kind",
            "af3-samples",
            "--arm",
            arm,
            "--model",
            model,
            "--case",
            "t0128",
            "--num-samples",
            "10",
            "--work-root",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        run_gap_ablation.main()
    assert message in capsys.readouterr().err


def test_gap_ablation_records_the_successful_prefill_timing_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command, *, check):
        assert check is True
        calls.append(command)
        if "--json-out" in command:
            output = Path(command[command.index("--json-out") + 1])
            output.write_text("{}\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_gap_ablation.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_gap_ablation",
            "--kind",
            "openfold3-confidence",
            "--arm",
            "serial",
            "--model",
            "openfold3",
            "--case",
            "t0488",
            "--num-samples",
            "5",
            "--work-root",
            str(tmp_path),
        ],
    )

    assert run_gap_ablation.main() == 0
    assert len(calls) == 2
    for argv in calls:
        index = argv.index("--timing-state")
        assert argv[index + 1] == "warm-after-successful-prefill"
        cache_index = argv.index("--cache-dir")
        assert Path(argv[cache_index + 1]) == (
            tmp_path / "openfold3-confidence-serial-t0488-n5/cache"
        )
