import ast
import json
from pathlib import Path

from bench.af3_closure import config_difference_kind
from bench.af3_closure_capture import config_record, save_config


class Config:
    def __init__(self, values):
        self.values = values

    def as_dict(self):
        return self.values


def test_native_full_defaults_share_port_schema_without_mutating_config():
    original = {"num_recycles": 10}
    native = config_record(Config(original))
    assert original == {"num_recycles": 10}
    assert native == config_record(Config({
        **original, "foldjax_stop_after": "full",
        "foldjax_return_representations": [],
    }))


def test_explicit_nondefault_port_execution_is_not_hidden():
    native = config_record(Config({}))
    for values in ({"foldjax_stop_after": "inputs"},
                   {"foldjax_stop_after": "trunk"},
                   {"foldjax_return_representations": ["single"]}):
        assert config_record(Config(values)) != native
        assert all(config_record(Config(values))[k] == v for k, v in values.items())


def test_vendored_defaults_match_capture_after_json_serialization():
    source = Path(__file__).resolve().parents[1] / (
        "src/foldjax/models/alphafold3/_upstream/alphafold3/model/model.py"
    )
    tree = ast.parse(source.read_text())
    model = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Model"
    )
    config = next(
        n for n in model.body if isinstance(n, ast.ClassDef) and n.name == "Config"
    )
    defaults = {
        n.target.id: ast.literal_eval(n.value)
        for n in config.body
        if isinstance(n, ast.AnnAssign)
        and isinstance(n.target, ast.Name)
        and n.target.id.startswith("foldjax_")
    }
    assert defaults == {
        "foldjax_stop_after": "full", "foldjax_return_representations": (),
    }
    raw_native = {"num_recycles": 10}
    raw_port = {**raw_native, **defaults}
    native = json.loads(json.dumps(config_record(Config(raw_native))))
    port = json.loads(json.dumps(config_record(Config(raw_port))))
    assert config_difference_kind(native, port) == "none"
    assert config_difference_kind(raw_native, port) == "allowlisted_extensions_only"
    for stage in ("inputs", "trunk"):
        changed = {**port, "foldjax_stop_after": stage}
        assert config_difference_kind(native, changed) == "other"


def test_synthesized_representation_defaults_are_not_shared():
    first = config_record(Config({}))
    first["foldjax_return_representations"].append("single")
    assert config_record(Config({}))["foldjax_return_representations"] == []


def test_saved_config_retains_raw_schema_and_marks_only_synthesized_fields(tmp_path):
    for name, values in (
        ("config", {"num_recycles": 10}),
        ("effective-config", {
            "num_recycles": 10, "foldjax_stop_after": "inputs",
            "foldjax_return_representations": ("single",),
        }),
    ):
        path = tmp_path / (name + ".json")
        save_config(path, Config(values))
        record = json.loads(path.read_text())
        provenance = json.loads(
            (tmp_path / (name + "-recording.json")).read_text()
        )
        assert provenance["raw_config"] == json.loads(json.dumps(values))
        assert provenance["synthesized_fields"] == sorted(record.keys() - values.keys())
        assert record == json.loads(json.dumps(config_record(Config(values))))
        if name == "effective-config":
            assert provenance["synthesized_fields"] == []
            assert record["foldjax_stop_after"] == "inputs"
