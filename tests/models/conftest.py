"""Collection gate for the vendored ports' own test suites.

Each port arrived with the suite it was developed against. Part of that suite
compares the JAX port against a publisher reference implementation. Those
dependencies live in a separate development environment: no FoldJAX install
profile contains a second tensor runtime.

Those modules import their optional dependency at module scope, so they have to
be excluded at collection time rather than skipped inside a test. Provision the
named external environment to collect and run them.

The banner that names what was left out is `pytest_report_header` in
`tests/conftest.py`: pytest calls that hook only from the rootdir conftest or a
plugin, so defined here it never ran and the gate hid these suites silently.
"""

from __future__ import annotations

from importlib.util import find_spec

# optional import -> the external environment that provides it, and the vendored
# modules that fail to import without it
_OPTIONAL_SUITES: dict[str, tuple[str, tuple[str, ...]]] = {
    "torch": (
        "external publisher-parity environment",
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

