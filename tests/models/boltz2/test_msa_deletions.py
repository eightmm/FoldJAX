"""The Boltz-2 `msa_deletions` option: released regression vs restored loop.

Upstream Boltz-2 v2.2.0+ zeroes every MSA deletion feature -- see
`docs/boltz2-upstream-msa-deletion-regression-2026-09-10.md`. The port keeps
that released behaviour by default and reinstates the pre-`04d27c71` loop
behind `msa_deletions=restored`.

The fixture drives `process_msa_features` on a hand-built `Tokenized` stand-in
rather than the full preprocessing pipeline: the mechanism lives entirely in
the MSA path, and the molecule database the pipeline needs is a setup asset the
rest of this directory already skips on.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from foldjax.models.boltz2.data import const
from foldjax.models.boltz2.data.feature import featurizerv2
from foldjax.models.boltz2.data.parse.a3m import _parse_a3m

#: A four-row alignment over a ten-residue query.
#:
#: The query row carries no lowercase insertion, which is exactly what arms the
#: regression: its `del_start == del_end == 0`, so the released loop slices an
#: empty array and every later row slices that empty slice.  Each homolog keeps
#: ten non-lowercase columns so the rows stay aligned to the query.
A3M = (
    ">query\n"
    "ACDEFGHIKL\n"
    ">h1\n"
    "ACDaaEFGHIKL\n"
    ">h2\n"
    "ACDEFgGHIKL\n"
    ">h3\n"
    "AWDEFGHIKppL\n"
)

#: `(seq_idx, res_idx, deletion count)` the a3m parser records for `A3M`.
#: Row order out of `construct_paired_msa` is the sequence order here because
#: no row is annotated with a taxonomy, so no pairing reorders them.
DELETION_RECORDS = ((1, 3, 2), (2, 5, 1), (3, 9, 2))

#: Every array `process_msa_features` returns on the non-affinity path.
MSA_KEYS = (
    "msa",
    "msa_paired",
    "deletion_value",
    "has_deletion",
    "deletion_mean",
    "profile",
    "msa_mask",
)

GOLDEN = Path(__file__).parent / "fixtures/msa_deletions_released.npz"


def build_tokenized() -> SimpleNamespace:
    """A minimal `Tokenized` carrying one protein chain and `A3M`."""

    msa = _parse_a3m(io.StringIO(A3M), taxonomy=None)
    first = msa.sequences[0]
    query = msa.residues[first["res_start"] : first["res_end"]]
    num_residues = len(query)

    residues = np.zeros(num_residues, dtype=[("res_type", np.dtype("i4"))])
    residues["res_type"] = query["res_type"]
    chains = np.zeros(
        1, dtype=[("res_idx", np.dtype("i4")), ("res_num", np.dtype("i4"))]
    )
    chains["res_num"] = num_residues
    tokens = np.zeros(
        num_residues,
        dtype=[("asym_id", np.dtype("i4")), ("res_idx", np.dtype("i4"))],
    )
    tokens["res_idx"] = np.arange(num_residues)

    return SimpleNamespace(
        tokens=tokens,
        structure=SimpleNamespace(chains=chains, residues=residues),
        msa={0: msa},
        record=SimpleNamespace(id="msa-deletions-fixture"),
    )


def msa_features(**kwargs: object) -> dict[str, np.ndarray]:
    """Run `process_msa_features` over the fixture and return NumPy arrays."""

    features = featurizerv2.process_msa_features(
        data=build_tokenized(),
        random=np.random.default_rng(42),
        max_seqs_batch=const.max_msa_seqs,
        max_seqs=const.max_msa_seqs,
        **kwargs,
    )
    return {
        name: value.numpy() if hasattr(value, "numpy") else np.asarray(value)
        for name, value in features.items()
    }


def test_released_zeroes_every_deletion_feature() -> None:
    features = msa_features(msa_deletions="released")

    assert features["msa"].shape == (4, 10)
    assert not features["has_deletion"].any()
    assert not features["deletion_value"].any()
    assert not features["deletion_mean"].any()


def test_restored_recovers_one_value_per_recorded_deletion() -> None:
    features = msa_features(msa_deletions="restored")

    has_deletion = features["has_deletion"].astype(bool)
    assert int(has_deletion.sum()) == len(DELETION_RECORDS)
    assert {
        (int(row), int(token)) for row, token in zip(*np.nonzero(has_deletion))
    } == {(seq_idx, res_idx) for seq_idx, res_idx, _ in DELETION_RECORDS}

    # The released featurizer applies `pi / 2 * atan(d / 3)`, not the
    # `2 / pi` spelling the task description carries; assert the code.
    for seq_idx, res_idx, count in DELETION_RECORDS:
        assert features["deletion_value"][seq_idx, res_idx] == pytest.approx(
            math.pi / 2 * math.atan(count / 3), rel=1e-6
        )
    assert features["deletion_mean"].shape == (10,)
    assert int(np.count_nonzero(features["deletion_mean"])) == len(
        {res_idx for _, res_idx, _ in DELETION_RECORDS}
    )


def test_released_is_the_default() -> None:
    for name, value in msa_features().items():
        np.testing.assert_array_equal(
            value, msa_features(msa_deletions="released")[name], err_msg=name
        )


def test_released_matches_the_pre_change_featurizer_bitwise() -> None:
    """Pin the default against arrays captured before the option existed.

    Captured from the featurizer at `018f3f4` -- the commit that documented the
    regression and predates this option -- so the guard is against that source,
    not against another spelling of the same post-change code.
    """
    golden = np.load(GOLDEN)
    features = msa_features(msa_deletions="released")

    assert set(golden.files) == set(MSA_KEYS) == set(features)
    for name in MSA_KEYS:
        assert features[name].dtype == golden[name].dtype, name
        assert np.array_equal(features[name], golden[name]), name


def test_unknown_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="msa_deletions"):
        msa_features(msa_deletions="fixed")


# --------------------------------------------------------------------------
# Option plumbing: `--option msa_deletions=restored` from the adapter to the
# featurizer call, one test per forwarding site.
# --------------------------------------------------------------------------


def _request(tmp_path: Path, input_format: str = "auto", **options: object):
    from foldjax.schema import PredictionRequest

    job = tmp_path / "job.yaml"
    job.write_text("{}")
    weights = tmp_path / "weights"
    weights.mkdir(exist_ok=True)
    return PredictionRequest(
        model="boltz2",
        input=job,
        weights=weights,
        output_dir=tmp_path / "out",
        seed=5,
        cache_dir=tmp_path / "cache",
        input_format=input_format,
        options=options,
    )


def test_the_backend_declares_the_option_and_its_released_default() -> None:
    from foldjax.backends.boltz2 import _RELEASED_COMPILE_DEFAULTS, Boltz2Backend

    assert "msa_deletions" in Boltz2Backend.native_options
    assert "msa_deletions" in Boltz2Backend.compile_options
    assert _RELEASED_COMPILE_DEFAULTS["msa_deletions"] == "released"


@pytest.mark.parametrize("value", ["fixed", "", "restored ", True, None])
def test_the_backend_rejects_an_unknown_deletion_mode(value: object) -> None:
    from foldjax.backends.boltz2 import Boltz2Backend

    with pytest.raises(ValueError, match="msa_deletions"):
        Boltz2Backend().validate_native_options({"msa_deletions": value})


@pytest.mark.parametrize("value", ["released", "restored"])
def test_the_backend_accepts_both_deletion_modes(value: str) -> None:
    from foldjax.backends.boltz2 import Boltz2Backend

    Boltz2Backend().validate_native_options({"msa_deletions": value})


def test_spelling_the_released_default_reuses_one_cache_namespace(
    tmp_path: Path,
) -> None:
    """`released` is what the native runner resolves to, so naming it is free.

    `restored` produces different features from the same input, so it has to
    take a namespace of its own.
    """
    from foldjax.api import resolve_cache_dir
    from foldjax.backends.boltz2 import Boltz2Backend

    backend = Boltz2Backend()
    omitted = resolve_cache_dir(_request(tmp_path), backend)
    released = resolve_cache_dir(
        _request(tmp_path, msa_deletions="released"), backend
    )
    restored = resolve_cache_dir(
        _request(tmp_path, msa_deletions="restored"), backend
    )

    assert released == omitted
    assert restored != omitted


def test_the_option_reaches_the_native_predict_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from foldjax.backends.boltz2 import Boltz2Backend

    mols = tmp_path / "mols"
    mols.mkdir()
    seen: dict[str, object] = {}

    def native_predict(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {
            "coords": np.zeros((2, 3)),
            "plddt": np.asarray([0.7, 0.8]),
            "iptm": np.asarray([0.6]),
            "out_path": tmp_path / "out" / "job.cif",
        }

    monkeypatch.setattr(
        "foldjax.backends.boltz2.import_module",
        lambda name: SimpleNamespace(predict=native_predict),
    )
    Boltz2Backend().predict(
        _request(tmp_path, mols=mols, msa_deletions="restored")
    )

    assert seen["msa_deletions"] == "restored"


def test_the_option_reaches_featurize_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from foldjax.models.boltz2 import api

    seen: dict[str, object] = {}

    def fake_featurize_yaml(*args: object, **kwargs: object):
        seen.update(kwargs)
        return {}, SimpleNamespace(records=[SimpleNamespace(id="job")]), tmp_path

    monkeypatch.setattr(api, "featurize_yaml", fake_featurize_yaml)
    api.featurize(
        seq=["ACDEFG"],
        mols=tmp_path / "mols",
        out_dir=tmp_path / "work",
        msa_deletions="restored",
    )

    assert seen["msa_deletions"] == "restored"


def test_featurize_yaml_reaches_the_prediction_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from foldjax.models.boltz2.data import featurize as featurize_module

    seen: dict[str, object] = {}

    def fake_dataset(**kwargs: object):
        seen.update(kwargs)
        return {0: {}}

    monkeypatch.setattr(featurize_module, "check_inputs", lambda path: ["job"])
    monkeypatch.setattr(
        featurize_module,
        "process_inputs",
        lambda **kwargs: SimpleNamespace(records=[SimpleNamespace(id="job")]),
    )
    monkeypatch.setattr(featurize_module, "PredictionDataset", fake_dataset)
    job = tmp_path / "job.yaml"
    job.write_text("{}")

    featurize_module.featurize_yaml(
        job, tmp_path / "work", tmp_path, msa_deletions="restored"
    )

    assert seen["msa_deletions"] == "restored"


def test_the_prediction_dataset_reaches_the_featurizer_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from foldjax.models.boltz2.data.module import inferencev2

    monkeypatch.setattr(inferencev2, "load_canonicals", lambda mol_dir: {})
    monkeypatch.setattr(inferencev2, "load_molecules", lambda mol_dir, names: {})
    monkeypatch.setattr(
        inferencev2,
        "load_input",
        lambda **kwargs: SimpleNamespace(extra_mols={}),
    )
    dataset = inferencev2.PredictionDataset(
        manifest=SimpleNamespace(
            records=[SimpleNamespace(id="job", inference_options=None)]
        ),
        target_dir=tmp_path,
        msa_dir=tmp_path,
        mol_dir=tmp_path,
        msa_deletions="restored",
    )
    seen: dict[str, object] = {}
    dataset.tokenizer = SimpleNamespace(
        tokenize=lambda data: SimpleNamespace(
            tokens={"res_name": np.asarray(["ALA"])}
        )
    )
    dataset.featurizer = SimpleNamespace(
        process=lambda *args, **kwargs: seen.update(kwargs) or {}
    )

    dataset[0]

    assert seen["msa_deletions"] == "restored"


def test_the_option_name_is_accepted_by_request_validation(tmp_path: Path) -> None:
    """The `native_options` entry is what keeps `--option` from refusing it.

    Without the declaration the adapter answers "unsupported boltz2 options"
    before any model loads, so this is the behavioural half of the table entry.
    """
    from foldjax.backends.boltz2 import Boltz2Backend

    Boltz2Backend().validate_request(
        _request(tmp_path, input_format="native", msa_deletions="restored")
    )
    with pytest.raises(ValueError, match="unsupported boltz2 options"):
        Boltz2Backend().validate_request(
            _request(tmp_path, input_format="native", msa_deletion="restored")
        )
