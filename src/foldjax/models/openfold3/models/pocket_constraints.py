"""Pocket-guided ligand proposals for OpenFold3's partial-diffusion refinement.

A JAX transcription of OpenFold3 v0.5.0 ``core/model/structure/pocket_constraints.py``
and the second rollout in ``diffusion_module.py:SampleDiffusion.forward``
(commit c4771653). After the ordinary rollout, ligand poses are proposed in the
user's pocket -- the parents' own ligand poses plus random rigid placements of
a parent or RDKit conformer around the pocket centroid or one pocket atom --
ranked by pocket-centroid distance and contact, filtered for diversity, written
into the parent structures, jittered, re-noised to the schedule level at
``start_frac`` and denoised again from there.

Upstream builds the proposals in a Python loop with data-dependent sizes. Here
every loop bound is static -- :class:`PocketSamplingConfig` carries the ligand
and pocket atom counts and the settings -- so proposal, ranking and selection
stay inside the compiled program. Two deliberate differences, neither a change
in distribution:

* Random draws come from a JAX key, and every candidate draws both target
  branches (pocket centroid and pocket atom) before one is selected. Upstream
  draws only the branch it takes, from torch's global generator.
* Distances are computed from coordinate differences. ``torch.cdist`` switches
  to the ``|a|^2 + |b|^2 - 2ab`` expansion above 25 rows; the two agree to
  float32 round-off.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models.openfold3.models.augmentation import quat_to_rot


class PocketSamplingConfig(NamedTuple):
    """Static pocket-sampling settings, read on the host from the features.

    Mirrors the scalar ``pocket_sampling_*`` features upstream carries in its
    batch, plus the two atom counts the compiled gathers need.
    """

    n_ligand_atoms: int
    n_pocket_atoms: int
    #: RDKit conformers in ``pocket_sampling_conformer_rels``; ``0`` when
    #: generation failed or was disabled, in which case proposals reuse the
    #: parent's own ligand conformation.
    n_conformers: int
    num_parents: int
    candidates: int
    start_frac: float
    ligand_jitter: float
    center_jitter: float
    surface_jitter: float
    vdw_buffer: float
    diversity_rmsd: float
    contact_distance: float

    def num_parents_for(self, no_rollout_samples: int) -> int:
        return max(1, min(no_rollout_samples, self.num_parents))

    def num_candidates_for(self, no_rollout_samples: int) -> int:
        return max(no_rollout_samples, self.candidates)

    def start_step(self, total_steps: int) -> int:
        """Upstream's refinement start index; Python ``round`` (half to even)."""
        return max(0, min(total_steps - 1, int(round(self.start_frac * total_steps))))


class PocketSamplingInputs(NamedTuple):
    """Per-atom pocket arrays, unbatched: ``[N_atom]`` and ``[conf, lig, 3]``."""

    ligand_atom_mask: jnp.ndarray
    pocket_atom_mask: jnp.ndarray
    vdw_radii: jnp.ndarray
    conformer_rels: jnp.ndarray | None = None


class PocketProposalDraws(NamedTuple):
    """Random draws for every candidate, ``[candidates, ...]``.

    Rows below the parent count are unused (those candidates are the parents'
    own poses). Both target branches are drawn so the arrays have static
    shapes; ``coin < 0.5`` selects the pocket-centroid branch, as upstream's
    ``torch.rand(()) < 0.5``.
    """

    conformer_index: jnp.ndarray
    quaternion: jnp.ndarray
    coin: jnp.ndarray
    center_noise: jnp.ndarray
    surface_index: jnp.ndarray
    surface_noise: jnp.ndarray


def draw_pocket_proposals(
    key: jax.Array, *, candidates: int, n_conformers: int, n_pocket_atoms: int
) -> PocketProposalDraws:
    keys = jax.random.split(key, 6)
    return PocketProposalDraws(
        conformer_index=jax.random.randint(
            keys[0], (candidates,), 0, max(n_conformers, 1)
        ),
        quaternion=jax.random.normal(keys[1], (candidates, 4)),
        coin=jax.random.uniform(keys[2], (candidates,)),
        center_noise=jax.random.normal(keys[3], (candidates, 3)),
        surface_index=jax.random.randint(keys[4], (candidates,), 0, n_pocket_atoms),
        surface_noise=jax.random.normal(keys[5], (candidates, 3)),
    )


def _pairwise_distance(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.sum(jnp.square(a[:, None, :] - b[None, :, :]), axis=-1))


def _score_ligand_pose(
    lig_pose: jnp.ndarray,
    parent: jnp.ndarray,
    protein_mask: jnp.ndarray,
    pocket: jnp.ndarray,
    lig_vdw: jnp.ndarray,
    vdw_radii: jnp.ndarray,
    contact_distance: float,
    vdw_buffer: float,
) -> tuple[jnp.ndarray, ...]:
    """Rank a ligand proposal by site entry.

    Upstream gathers the protein atoms (every real atom outside the
    constrained ligand) into a compact array; here they stay in place under
    ``protein_mask``, which leaves the sum and minimum unchanged.
    """
    d_prot = _pairwise_distance(lig_pose, parent)
    vdw_lower = (lig_vdw[:, None] + vdw_radii[None, :]) * (1.0 - vdw_buffer)
    overlap = jax.nn.relu(vdw_lower - d_prot) * protein_mask[None, :]
    vdw_overlap = jnp.square(overlap).sum()
    min_prot = jnp.min(jnp.where(protein_mask[None, :], d_prot, jnp.inf))

    d_pocket = _pairwise_distance(lig_pose, pocket)
    lig_to_pocket = d_pocket.min(axis=1)
    contact = jnp.square(jax.nn.relu(lig_to_pocket - contact_distance)).mean()
    lig_atoms_in_pocket = (
        (lig_to_pocket < contact_distance).sum().astype(lig_pose.dtype)
    )
    pocket_com_dist = jnp.linalg.norm(lig_pose.mean(axis=0) - pocket.mean(axis=0))
    return pocket_com_dist, vdw_overlap, min_prot, lig_atoms_in_pocket, contact


def _candidate_order(
    pocket_com_dist: jnp.ndarray,
    vdw_overlap: jnp.ndarray,
    min_protein_dist: jnp.ndarray,
    lig_atoms_in_pocket: jnp.ndarray,
    contact_penalty: jnp.ndarray,
) -> jnp.ndarray:
    """Indices in upstream's ``_candidate_sort_key`` order.

    Ascending centroid distance, then more ligand atoms in contact, lower
    contact penalty, lower overlap, larger protein clearance. ``lexsort``
    takes the primary key last; it and Python's ``sort`` are both stable.
    """
    return jnp.lexsort(
        (
            -min_protein_dist,
            vdw_overlap,
            contact_penalty,
            -lig_atoms_in_pocket,
            pocket_com_dist,
        )
    )


def build_pocket_sampling_seeds(
    xl_base: jnp.ndarray,
    atom_mask: jnp.ndarray,
    inputs: PocketSamplingInputs,
    config: PocketSamplingConfig,
    draws: PocketProposalDraws,
) -> jnp.ndarray:
    """Generate ligand seeds in the requested pocket for partial diffusion.

    Args:
        xl_base: ``[samples, N_atom, 3]`` coordinates of the first rollout.
        atom_mask: ``[N_atom]`` real-atom mask.
        inputs: the per-atom pocket arrays.
        config: static settings and counts.
        draws: from :func:`draw_pocket_proposals` with
            ``config.num_candidates_for(samples)`` rows.

    Returns:
        ``[samples, N_atom, 3]``: for each selected proposal, its parent's
        coordinates with the ligand atoms replaced by the proposal.
    """
    samples = xl_base.shape[0]
    dtype = xl_base.dtype
    n_parents = config.num_parents_for(samples)
    n_candidates = config.num_candidates_for(samples)
    lig_mask = inputs.ligand_atom_mask.astype(bool)
    lig_idx = jnp.nonzero(lig_mask, size=config.n_ligand_atoms)[0]
    pocket_idx = jnp.nonzero(
        inputs.pocket_atom_mask.astype(bool), size=config.n_pocket_atoms
    )[0]
    protein_mask = atom_mask.astype(bool) & ~lig_mask
    radii = inputs.vdw_radii.astype(dtype)
    lig_vdw = radii[lig_idx]

    def score(pose: jnp.ndarray, parent: jnp.ndarray):
        return _score_ligand_pose(
            pose,
            parent,
            protein_mask,
            parent[pocket_idx],
            lig_vdw,
            radii,
            config.contact_distance,
            config.vdw_buffer,
        )

    parent_scores = jax.vmap(lambda parent: score(parent[lig_idx], parent)[0])(xl_base)
    parent_order = jnp.argsort(parent_scores)[:n_parents]

    index = jnp.arange(n_candidates)
    parent_slot = parent_order[index % n_parents]
    # Gather the few ligand and pocket atoms per sample first, then per
    # candidate: indexing whole parents by candidate would materialise
    # [candidates, N_atom, 3].
    lig_parent = xl_base[:, lig_idx][parent_slot]
    pocket_parent = xl_base[:, pocket_idx][parent_slot]
    parent_rel = lig_parent - lig_parent.mean(axis=1, keepdims=True)
    if config.n_conformers > 0 and inputs.conformer_rels is not None:
        lig_rel = inputs.conformer_rels.astype(dtype)[draws.conformer_index]
    else:
        lig_rel = parent_rel
    quaternion = draws.quaternion.astype(dtype)
    rot = quat_to_rot(quaternion / jnp.linalg.norm(quaternion, axis=-1, keepdims=True))
    centre_target = pocket_parent.mean(
        axis=1
    ) + config.center_jitter * draws.center_noise.astype(dtype)
    surface_target = jnp.take_along_axis(
        pocket_parent, draws.surface_index[:, None, None], axis=1
    )[:, 0] + config.surface_jitter * draws.surface_noise.astype(dtype)
    target = jnp.where((draws.coin < 0.5)[:, None], centre_target, surface_target)
    proposed = jnp.einsum("cij,ckj->cik", lig_rel, rot) + target[:, None, :]
    poses = jnp.where((index < n_parents)[:, None, None], lig_parent, proposed)

    # One candidate at a time: [ligand, N_atom] distances per candidate stay
    # small, where the vectorised [candidates, ligand, N_atom] would not.
    com, vdw_overlap, min_prot, lig_atoms, contact = jax.lax.map(
        lambda item: score(item[0], xl_base[item[1]]), (poses, parent_slot)
    )
    order = _candidate_order(com, vdw_overlap, min_prot, lig_atoms, contact)

    def select(position, carry):
        chosen, count = carry
        candidate = order[position]
        previous = poses[chosen]
        rmsd = jnp.sqrt(
            jnp.mean(
                jnp.sum(jnp.square(poses[candidate][None] - previous), axis=-1),
                axis=-1,
            )
        )
        diverse = jnp.all(
            (jnp.arange(samples) >= count) | (rmsd >= config.diversity_rmsd)
        )
        take = diverse & (count < samples)
        chosen = jnp.where(
            take, chosen.at[jnp.minimum(count, samples - 1)].set(candidate), chosen
        )
        return chosen, count + take.astype(count.dtype)

    chosen, count = jax.lax.fori_loop(
        0,
        n_candidates,
        select,
        (jnp.zeros((samples,), dtype=order.dtype), jnp.zeros((), dtype=jnp.int32)),
    )
    # Too few diverse proposals: pad with the best-ranked ones, which may
    # repeat a selected pose, as upstream's ``selected.extend`` does.
    slot = jnp.arange(samples)
    fill = order[jnp.clip(slot - count, 0, n_candidates - 1)]
    chosen = jnp.where(slot < count, chosen, fill)

    seeds = xl_base[parent_slot[chosen]]
    return seeds.at[:, lig_idx].set(poses[chosen])


def pocket_refinement_start(
    key: jax.Array,
    seeds: jnp.ndarray,
    inputs: PocketSamplingInputs,
    config: PocketSamplingConfig,
) -> jnp.ndarray:
    """Apply one rigid ligand translation per sample, as upstream does.

    The schedule-level noise is added by the sampler, which starts from these
    coordinates.
    """
    samples = seeds.shape[0]
    jitter = config.ligand_jitter * jax.random.normal(
        key, (samples, 1, 3), dtype=seeds.dtype
    )
    lig = inputs.ligand_atom_mask.astype(bool)[None, :, None]
    return jnp.where(lig, seeds + jitter, seeds)
