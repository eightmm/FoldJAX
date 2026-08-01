"""Configuration contract for JAX training-free guidance."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from . import potentials


class Schedule:
    def __call__(self, t: float) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class Constant(Schedule):
    value: float

    def __call__(self, t: float) -> float:
        del t
        return float(self.value)


@dataclass(frozen=True)
class ExponentialInterpolation(Schedule):
    start: float
    end: float
    alpha: float = 0.0

    def __call__(self, t: float) -> float:
        if self.alpha == 0.0:
            return float(self.start + (self.end - self.start) * t)
        fraction = math.expm1(self.alpha * t) / math.expm1(self.alpha)
        return float(self.start + (self.end - self.start) * fraction)


def schedule_from_cfg(obj: Any) -> Schedule:
    if isinstance(obj, Schedule):
        return obj
    if isinstance(obj, (int, float)):
        return Constant(float(obj))
    if isinstance(obj, Mapping):
        cfg = dict(obj)
        if "type" not in cfg:
            raise KeyError("Schedule config must contain key 'type'")
        schedule_type = str(cfg["type"]).lower()
        if schedule_type == "const":
            return Constant(float(cfg["value"]))
        if schedule_type == "exp_interpolation":
            return ExponentialInterpolation(
                float(cfg["start"]), float(cfg["end"]), float(cfg.get("alpha", 0.0))
            )
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    raise TypeError(f"Unsupported schedule config type: {type(obj)}")


_REQUIRED_FEATURES = {
    "InterchainBondPotential": {"interchain_bond_index"},
    "PairwiseDistancePotential": {
        "pairwise_distance_index",
        "pairwise_distance_is_bond",
        "pairwise_distance_is_angle",
        "pairwise_distance_upper_bound",
        "pairwise_distance_lower_bound",
        "ref_element",
    },
    "StereoBondPotential": {"stereo_bond_index", "stereo_bond_orientation"},
    "ChiralAtomPotential": {
        "chiral_index",
        "chiral_orientation",
        "asym_id",
        "atom_to_token_idx",
    },
    "PlanarImproperPotential": {
        "planar_improper_index",
        "planar_improper_is_carbonyl",
    },
    "LinearBondPotential": {"linear_triple_bond_index"},
    "ExperimentalTorsionPotential": {
        "experimental_torsion_index",
        "experimental_torsion_force_constant",
        "experimental_torsion_sign",
    },
    "VinaStericPotential": {
        "asym_id",
        "atom_to_token_idx",
        "ref_element",
        "interchain_bond_index",
    },
}


@dataclass
class Term:
    name: str
    interval: int
    weight: Schedule
    param_templates: dict[str, Any]
    _potential: potentials.Potential
    enable_projection: bool = True

    def required_features(self) -> set[str]:
        return set(_REQUIRED_FEATURES.get(self.name, set()))

    def active(self, step_i: int) -> bool:
        return self.interval > 0 and step_i % self.interval == 0

    def _params_at(self, t: float) -> dict[str, Any]:
        return {
            key: value(t) if isinstance(value, Schedule) else value
            for key, value in self.param_templates.items()
        }

    def energy(self, coords, feats, t: float):
        weight = self.weight(t)
        if weight == 0.0:
            return jnp.zeros(coords.shape[:-2], dtype=coords.dtype)
        return self._potential.energy(coords, feats, self._params_at(t)) * weight

    def energy_and_grad(self, coords, feats, t: float):
        weight = self.weight(t)
        if weight == 0.0:
            return jnp.zeros(coords.shape[:-2], coords.dtype), jnp.zeros_like(coords)
        energy, gradient = self._potential.energy_and_grad(
            coords, feats, self._params_at(t)
        )
        return energy * weight, gradient * weight

    def project(self, coords, feats, t: float):
        return self._potential.project(coords, feats, self._params_at(t))


@dataclass(frozen=True)
class TFGConfig:
    enable: bool
    rho: float
    mu: float
    eps_std: float
    eps_batch: int
    outer_steps: int
    inner_steps: int
    projection_outer_steps: int
    projection_inner_steps: int
    terms: tuple[Term, ...]
    log_last_step_energy: bool = False


def _build_terms(raw_terms: Mapping[str, Any] | None) -> tuple[Term, ...]:
    if raw_terms is None:
        return ()
    if not isinstance(raw_terms, Mapping):
        raise TypeError("terms must be a mapping of term_name -> term_config")
    result = []
    for name, raw in raw_terms.items():
        cfg = dict(raw or {})
        if name not in potentials.CLASS_REGISTRY:
            raise KeyError(f"Unknown potential '{name}'")
        interval = int(cfg.pop("interval", 1))
        weight = schedule_from_cfg(cfg.pop("weight", 0.0))
        enable_projection = bool(cfg.pop("enable_projection", True))
        params = {}
        for key, value in cfg.items():
            if isinstance(value, Mapping) and "type" in value:
                params[key] = schedule_from_cfg(value)
            else:
                params[key] = value
        result.append(
            Term(
                name,
                interval,
                weight,
                params,
                potentials.CLASS_REGISTRY[name](),
                enable_projection,
            )
        )
    return tuple(result)


def validate_features(feats: Mapping[str, Any], terms: Iterable[Term]) -> None:
    missing = {
        term.name: sorted(term.required_features() - feats.keys())
        for term in terms
        if term.required_features() - feats.keys()
    }
    if missing:
        raise KeyError(f"TFG is missing required input features: {missing}")


def parse_tfg_config(guidance_cfg: Mapping[str, Any] | None) -> TFGConfig:
    if guidance_cfg is None:
        return TFGConfig(False, 0.0, 0.0, 0.0, 1, 1, 0, 0, 0, ())
    cfg = dict(guidance_cfg)
    allowed = {"enable", "rho", "mu", "mc", "steps", "terms", "log_last_step_energy"}
    extra = set(cfg) - allowed
    if extra:
        raise KeyError(f"Unsupported keys in TFG config: {sorted(extra)}")
    mc = dict(cfg.get("mc", {}))
    steps = dict(cfg.get("steps", {}))
    terms = _build_terms(cfg.get("terms", {}))
    enable = bool(cfg.get("enable", False))
    if enable and not terms:
        raise ValueError("TFG is enabled but no terms are configured")
    return TFGConfig(
        enable=enable,
        rho=float(cfg.get("rho", 0.0)),
        mu=float(cfg.get("mu", 0.0)),
        eps_std=float(mc.get("std", 0.0)),
        eps_batch=max(1, int(mc.get("batch", 1))),
        outer_steps=max(1, int(steps.get("tfg_outer", 1))),
        inner_steps=max(0, int(steps.get("tfg_inner", 10))),
        projection_outer_steps=max(0, int(steps.get("projection_outer", 2))),
        projection_inner_steps=max(0, int(steps.get("projection_inner", 10))),
        terms=terms,
        log_last_step_energy=bool(cfg.get("log_last_step_energy", False)),
    )
