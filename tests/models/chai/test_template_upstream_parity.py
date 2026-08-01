from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from foldjax.models.chai.data.templates import TemplateContext

pytestmark = pytest.mark.official_parity


def _local_context(templates: int, tokens: int, offset: int) -> TemplateContext:
    restype = (
        np.arange(templates * tokens, dtype=np.int32).reshape(templates, tokens)
        + offset
    ) % 31
    pseudo = restype % 2 == 0
    backbone = restype % 3 == 0
    distances = np.arange(templates * tokens * tokens, dtype=np.float32).reshape(
        templates, tokens, tokens
    )
    vectors = np.repeat(distances[..., None], 3, axis=-1)
    return TemplateContext(restype, pseudo, backbone, distances, vectors)


def test_template_pad_merge_and_index_select_match_official_chai(
    upstream_chai_dir: Path, upstream_chai_python: Path
) -> None:
    program = r"""
import json
import torch
from chai_lab.data.dataset.templates.context import TemplateContext

def context(templates, tokens, offset):
    restype = torch.arange(templates * tokens, dtype=torch.int32)
    restype = (restype.reshape(templates, tokens) + offset) % 31
    pseudo = restype % 2 == 0
    backbone = restype % 3 == 0
    distances = torch.arange(templates * tokens * tokens, dtype=torch.float32)
    distances = distances.reshape(templates, tokens, tokens)
    vectors = distances[..., None].repeat(1, 1, 1, 3)
    return TemplateContext(restype, pseudo, backbone, distances, vectors)

left = context(1, 2, 0)
right = context(2, 3, 7).index_select(torch.tensor([2, 0]))
merged = TemplateContext.merge([left, right]).pad(max_templates=4, max_tokens=7)
print(json.dumps({
    "restype": merged.template_restype.tolist(),
    "pseudo": merged.template_pseudo_beta_mask.tolist(),
    "backbone": merged.template_backbone_frame_mask.tolist(),
    "distances": merged.template_distances.tolist(),
    "vectors": merged.template_unit_vector.tolist(),
}))
"""
    output = subprocess.run(
        [str(upstream_chai_python), "-c", program],
        cwd=upstream_chai_dir,
        check=True,
        text=True,
        capture_output=True,
    )
    expected = json.loads(output.stdout)

    left = _local_context(1, 2, 0)
    right = _local_context(2, 3, 7).index_select(np.asarray([2, 0]))
    actual = TemplateContext.merge([left, right]).pad(4, 7)
    np.testing.assert_array_equal(actual.template_restype, expected["restype"])
    np.testing.assert_array_equal(actual.template_pseudo_beta_mask, expected["pseudo"])
    np.testing.assert_array_equal(
        actual.template_backbone_frame_mask, expected["backbone"]
    )
    np.testing.assert_array_equal(actual.template_distances, expected["distances"])
    np.testing.assert_array_equal(actual.template_unit_vector, expected["vectors"])
