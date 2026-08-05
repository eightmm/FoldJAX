"""Collection gate for the vendored ports' own test suites.

Each port arrived with the suite it was developed against. Part of that suite
compares the JAX port against the upstream torch implementation, so it needs an
optional extra the base FoldJAX environment deliberately omits: the whole point
of these ports is that inference runs without torch.

Those modules import their optional dependency at module scope, so they have to
be excluded at collection time rather than skipped inside a test. Install the
named extra to collect and run them.
"""

from __future__ import annotations

from importlib.util import find_spec

# optional import -> the FoldJAX extra that provides it, and the vendored
# modules that fail to import without it
_OPTIONAL_SUITES: dict[str, tuple[str, tuple[str, ...]]] = {
    "torch": (
        "torch-bridge",
        (
            "boltz2/test_affinity_checkpoint_parity.py",
            "boltz2/test_atom_attention_checkpoint_parity.py",
            "boltz2/test_atom_transformer_checkpoint_parity.py",
            "boltz2/test_attention_checkpoint_parity.py",
            "boltz2/test_bfactor_checkpoint_parity.py",
            "boltz2/test_conditioning_checkpoint_parity.py",
            "boltz2/test_confidence_checkpoint_parity.py",
            "boltz2/test_diffusion_conditioning_checkpoint_parity.py",
            "boltz2/test_diffusion_score_checkpoint_parity.py",
            "boltz2/test_diffusion_transformer_checkpoint_parity.py",
            "boltz2/test_distogram_checkpoint_parity.py",
            "boltz2/test_end_to_end_sample_smoke.py",
            "boltz2/test_input_embedder_checkpoint_parity.py",
            "boltz2/test_msa_checkpoint_parity.py",
            "boltz2/test_pairformer_layer_checkpoint_parity.py",
            "boltz2/test_pairformer_module_checkpoint_parity.py",
            "boltz2/test_potentials_parity.py",
            "boltz2/test_transition_checkpoint_parity.py",
            "boltz2/test_transition_forward.py",
            "boltz2/test_transition_mapping.py",
            "boltz2/test_triangle_attention_checkpoint_parity.py",
            "boltz2/test_triangle_checkpoint_parity.py",
            "boltz2/test_trunk_checkpoint_parity.py",
            "boltz2/test_weighted_rigid_align_parity.py",
        ),
    ),
}

collect_ignore: list[str] = []
_skipped: dict[str, str] = {}
for _module, (_extra, _paths) in _OPTIONAL_SUITES.items():
    if find_spec(_module) is None:
        collect_ignore.extend(_paths)
        _skipped[_module] = _extra


def pytest_report_header() -> list[str]:
    """Say out loud which vendored suites were left out, and why."""
    if not _skipped:
        return []
    return [
        f"vendored parity suites not collected: "
        f"{', '.join(f'{m} (--extra {e})' for m, e in sorted(_skipped.items()))}"
    ]
