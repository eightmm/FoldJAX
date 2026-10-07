"""Runtime training-free guidance engine."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from .config import TFGConfig, validate_features

#: Upstream projects chirality first, then pairwise distances, then the rest
#: (Protenix `tfg/engine.py:87-93`); projection order changes the result.
_PROJECTION_RANK = {"ChiralAtomPotential": 0, "PairwiseDistancePotential": 1}


class TFGEngine:
    def __init__(
        self, cfg: TFGConfig, *, device: Any = None, dtype: Any = None
    ) -> None:
        del device, dtype
        self.cfg = cfg
        self._projection_terms = tuple(
            sorted(cfg.terms, key=lambda term: _PROJECTION_RANK.get(term.name, 2))
        )

    def _energy_and_grad(self, coords, feats, *, t: float, step_i: int):
        energy = jnp.zeros(coords.shape[:-2], dtype=coords.dtype)
        gradient = jnp.zeros_like(coords)
        for term in self.cfg.terms:
            if term.active(step_i):
                term_energy, term_gradient = term.energy_and_grad(coords, feats, t)
                energy = energy + term_energy
                gradient = gradient + term_gradient
        return energy, gradient

    def _energy(self, coords, feats, *, t: float, step_i: int):
        energy = jnp.zeros(coords.shape[:-2], dtype=coords.dtype)
        for term in self.cfg.terms:
            if term.active(step_i):
                energy = energy + term.energy(coords, feats, t)
        return energy

    def _sample_eps(self, key, shape, dtype):
        if self.cfg.eps_std == 0.0:
            return jnp.zeros((1, *shape), dtype=dtype)
        if key is None:
            raise ValueError("a JAX PRNG key is required when TFG mc.std > 0")
        return self.cfg.eps_std * jax.random.normal(
            key, (self.cfg.eps_batch, *shape), dtype=dtype
        )

    def _logp(self, coords, eps, feats, *, t: float, step_i: int):
        energy = self._energy(coords[None, ...] + eps, feats, t=t, step_i=step_i)
        logp = -energy
        return logsumexp(logp, axis=0) - jnp.log(logp.shape[0])

    def _logp_and_grad(self, coords, eps, feats, *, t: float, step_i: int):
        energy, gradient = self._energy_and_grad(
            coords[None, ...] + eps, feats, t=t, step_i=step_i
        )
        logp = -energy
        average = logsumexp(logp, axis=0) - jnp.log(logp.shape[0])
        weights = jax.nn.softmax(logp, axis=0)
        coordinate_gradient = (weights[..., None, None] * -gradient).sum(axis=0)
        return average, coordinate_gradient

    def refine(
        self,
        coords,
        feats: Mapping[str, Any],
        *,
        t: float,
        step_i: int,
        key=None,
        eps=None,
    ):
        """Project the denoised ``coords``, then refine them on log p(x0).

        One pass of upstream's per-outer-step x0 update (Protenix
        `tfg/engine.py:449-475`): projection first, then `inner_steps`
        gradient-ascent steps of size `mu` from the projected coordinates.
        """
        if not self.cfg.enable:
            return coords
        validate_features(feats, self.cfg.terms)
        if eps is None:
            eps = self._sample_eps(key, coords.shape, coords.dtype)
        result = coords + self._project(coords, feats, t=t, step_i=step_i)
        for _ in range(self.cfg.inner_steps):
            if self.cfg.mu == 0.0:
                break
            _, gradient = self._logp_and_grad(result, eps, feats, t=t, step_i=step_i)
            result = result + self.cfg.mu * gradient
        return result

    def _project(self, coords, feats, *, t: float, step_i: int):
        delta = jnp.zeros_like(coords)
        if not self.cfg.enable:
            return delta
        for _ in range(self.cfg.projection_outer_steps):
            for term in self._projection_terms:
                if not term.active(step_i) or not term.enable_projection:
                    continue
                for _ in range(self.cfg.projection_inner_steps):
                    delta = delta + term.project(coords + delta, feats, t)
        return delta

    def project(self, coords, feats, *, step_i: int, num_diffusion_steps: int):
        t = 1.0 - float(step_i) / max(1, num_diffusion_steps)
        return self._project(coords, feats, t=t, step_i=step_i)

    def step(
        self,
        denoise_net,
        *,
        x,
        t_hat,
        c_tau,
        step_scale_eta: float,
        step_i: int,
        num_diffusion_steps: int,
        input_feature_dict: Mapping[str, Any],
        key=None,
        **denoise_kwargs: Any,
    ):
        validate_features(input_feature_dict, self.cfg.terms)
        if self.cfg.outer_steps > 1 and key is None:
            raise ValueError("a JAX PRNG key is required when TFG tfg_outer > 1")
        t = 1.0 - float(step_i) / max(1, num_diffusion_steps)
        eps = self._sample_eps(key, x.shape, x.dtype)

        def denoise(value):
            if denoise_kwargs:
                return denoise_net(
                    x_noisy=value,
                    t_hat_noise_level=t_hat,
                    **denoise_kwargs,
                )
            return denoise_net(value, t_hat)

        def objective(value):
            prediction = denoise(value)
            return jnp.sum(
                self._logp(
                    prediction,
                    eps,
                    input_feature_dict,
                    t=t,
                    step_i=step_i,
                )
            )

        # Upstream's outer loop (Protenix `tfg/engine.py:390-495`): each outer
        # step re-denoises, takes the sampler step, and re-noises the result
        # back to `t_hat` for the next pass; the last pass's step is returned.
        x_work = x
        for outer in range(self.cfg.outer_steps):
            shifted = x_work
            if self.cfg.rho:
                shifted = x_work + self.cfg.rho * jax.grad(objective)(x_work)
            denoised = self.refine(
                denoise(shifted),
                input_feature_dict,
                t=t,
                step_i=step_i,
                eps=eps,
            )
            direction = (shifted - denoised) / t_hat[..., None, None]
            x_next = (
                shifted + step_scale_eta * (c_tau - t_hat)[..., None, None] * direction
            )
            if outer + 1 < self.cfg.outer_steps:
                sigma = jnp.sqrt(t_hat**2 - c_tau**2)
                noise = jax.random.normal(
                    jax.random.fold_in(key, outer + 1), x_next.shape, x_next.dtype
                )
                x_work = x_next + sigma[..., None, None] * noise
        return x_next
