import copy

import pytest

from bench.openbind_native_outputs import configured_backend, triangle_backend_update


@pytest.mark.parametrize("backend", ["cueq", "triton", "xla"])
def test_backend_override_preserves_custom_settings_and_original(backend):
    original = {"settings": {"memory": {"eval": {
        "per_sample_token_cutoff": 750,
        "use_triton_triangle_kernels": True,
    }}}, "other": {"unchanged": [1, 2]}}
    before = copy.deepcopy(original)
    result = triangle_backend_update(original, backend)
    assert original == before
    expected = copy.deepcopy(before)
    expected["settings"]["memory"]["eval"].update({
        "use_cueq_triangle_kernels": backend == "cueq",
        "use_triton_triangle_kernels": backend == "triton",
    })
    assert result == expected


def test_unknown_native_backend_is_rejected():
    with pytest.raises(ValueError, match="unsupported native triangle backend"):
        triangle_backend_update({}, "automatic")


def test_default_config_passes_original_update_unchanged():
    update = object()
    seen = []
    def original(value):
        seen.append(value)
        return value
    assert configured_backend(original, update, None) is update
    assert seen == [update]


@pytest.mark.parametrize("backend", ["cueq", "triton", "xla"])
def test_effective_backend_is_verified(backend):
    class Update:
        custom = {}

        def model_copy(self, *, update):
            result = Update()
            result.custom = update["custom"]
            return result

    configured_backend(lambda value: value.custom, Update(), backend)
    wrong = "cueq" if backend != "cueq" else "triton"
    with pytest.raises(ValueError, match="override was not applied"):
        configured_backend(lambda value: triangle_backend_update({}, wrong),
                           Update(), backend)
