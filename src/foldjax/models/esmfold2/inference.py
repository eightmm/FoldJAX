"""Running ESMFold2 end to end: weights in, structures out.

The one thing worth knowing before calling this is that ESMFold2 is two models.
The 235M-parameter structure network is what this repository ports and what
`weights_dir("esmfold2")` holds; the representations it folds come from
**ESMC-6B**, a separate 12 GB checkpoint that upstream distributes apart from
it. Without ESMC the structure network still runs -- its language-model branch
is simply absent, which is what upstream does when no PLM is loaded -- but it
is not the model anyone benchmarked, so asking for that has to be explicit.

Nothing here imports torch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.esmfold2.bridge import checkpoint as structure_checkpoint
from foldjax.models.esmfold2.bridge import esmc as esmc_checkpoint
from foldjax.models.esmfold2.data import features as featurisation
from foldjax.models.esmfold2.models import esmc as esmc_model
from foldjax.models.esmfold2.models import model as structure_model

#: Where `assets.py` stages the language model beside the structure weights.
ESMC_SUBDIRECTORY = "esmc"


@dataclass(frozen=True)
class LoadedModel:
    """Everything a run needs, with the language model optional."""

    parameters: Mapping[str, jnp.ndarray]
    settings: structure_model.ModelSettings
    esmc_parameters: Mapping[str, jnp.ndarray] | None = None
    esmc_settings: esmc_model.ESMCSettings | None = None

    @property
    def has_language_model(self) -> bool:
        return self.esmc_parameters is not None


def esmc_directory(weights: str | Path) -> Path:
    """Where the language model lives relative to the structure weights."""
    return Path(weights) / ESMC_SUBDIRECTORY


def load(
    weights: str | Path,
    *,
    esmc: str | Path | None = None,
    require_esmc: bool = True,
    dtype: str | None = None,
    esmc_dtype: str | None = "bfloat16",
) -> LoadedModel:
    """Read the checkpoint, and the language model beside it.

    `require_esmc` is on by default and refuses rather than quietly folding a
    different model: a run without ESMC is a legitimate thing to ask for and
    not a thing to get by accident.
    """
    weights = Path(weights)
    parameters = structure_checkpoint.load_parameters(weights, dtype=dtype)
    settings = structure_checkpoint.load_settings(weights)

    directory = Path(esmc) if esmc is not None else esmc_directory(weights)
    if not directory.exists():
        if require_esmc:
            raise FileNotFoundError(
                f"ESMC-6B is not at {directory}. ESMFold2 folds the "
                "representations of a 6B protein language model that upstream "
                "distributes separately (~12 GB); download it into that "
                "directory, or pass require_esmc=False to run the structure "
                "network without it -- which is not the released model."
            )
        return LoadedModel(parameters=parameters, settings=settings)

    return LoadedModel(
        parameters=parameters,
        settings=settings,
        esmc_parameters=esmc_checkpoint.load_parameters(directory, dtype=esmc_dtype),
        esmc_settings=esmc_checkpoint.load_settings(directory),
    )


def language_model_states(
    features: Mapping[str, np.ndarray], model: LoadedModel
) -> jnp.ndarray | None:
    """ESMC's stacked hidden states for these tokens, or `None` without it."""
    if model.esmc_parameters is None or model.esmc_settings is None:
        return None
    return esmc_model.lm_hidden_states(
        np.asarray(features["input_ids"]),
        np.asarray(features["asym_id"]),
        np.asarray(features["residue_index"]),
        np.asarray(features["mol_type"]),
        np.asarray(features["token_attention_mask"]),
        model.esmc_parameters,
        settings=model.esmc_settings,
    )


def predict(
    key: jnp.ndarray,
    features: Mapping[str, np.ndarray],
    model: LoadedModel,
    *,
    num_loops: int | None = None,
    num_samples: int | None = None,
    num_steps: int | None = None,
    msa_max_depth: int | None = None,
    compile_it: bool = True,
) -> dict[str, jnp.ndarray]:
    """One forward over already-built features.

    Compiled by default. Eager JAX dispatches this model an operation at a
    time -- forty-eight trunk layers, four loops, twelve diffusion blocks per
    sampling step -- so the difference is not a tuning detail; `compile_it` is
    there for debugging, where a traced error message is worth the wait.
    """
    settings = structure_model.with_overrides(
        model.settings,
        num_loops=num_loops,
        num_samples=num_samples,
        num_steps=num_steps,
        msa_max_depth=msa_max_depth,
    )
    arrays = {name: jnp.asarray(value) for name, value in features.items()}
    hidden = language_model_states(features, model)
    # Read on the host: it sizes the confidence head's per-chain matrix, and a
    # traced maximum cannot size anything.
    n_chains = int(np.asarray(features["asym_id"]).max()) + 1

    runner = compiled_predict(settings, n_chains) if compile_it else _run
    return runner(key, arrays, model.parameters, hidden, settings, n_chains)


def _run(
    key: jnp.ndarray,
    features: Mapping[str, jnp.ndarray],
    parameters: Mapping[str, jnp.ndarray],
    lm_hidden_states: jnp.ndarray | None,
    settings: structure_model.ModelSettings,
    n_chains: int,
) -> dict[str, jnp.ndarray]:
    return structure_model.predict(
        key,
        features,
        parameters,
        settings=settings,
        lm_hidden_states=lm_hidden_states,
        n_chains=n_chains,
    )


@lru_cache(maxsize=8)
def compiled_predict(
    settings: structure_model.ModelSettings, n_chains: int
) -> Callable[..., dict[str, jnp.ndarray]]:
    """`predict` as one jitted program, cached per settings and chain count.

    Both are static: the settings decide how many layers get traced and
    `n_chains` sizes an output. Everything else -- the features, the weights,
    the key -- is an argument, so a second job of the same shape reuses the
    compilation.
    """
    return jax.jit(_run, static_argnums=(4, 5))


def predict_job(
    key: jnp.ndarray,
    chains: Sequence[tuple[str, str, int, int]],
    alignments: Mapping[int, Path] | None,
    model: LoadedModel,
    **overrides: int | None,
) -> tuple[dict[str, jnp.ndarray], dict[str, np.ndarray]]:
    """Featurise a job and fold it, returning `(output, features)`.

    The features come back because everything downstream -- the structure
    writer, the per-atom confidences -- is indexed by them.
    """
    built = featurisation.build_features(chains, dict(alignments or {}))
    return predict(key, built, model, **overrides), built


def seed_key(seed: int) -> jnp.ndarray:
    """One place that decides what a FoldJAX seed means for this model.

    It means rather more here than elsewhere: the trunk's initial pair state,
    the per-loop language-model dropout and the sampler's noise all come off
    this key, so two seeds give genuinely different structures rather than the
    same structure sampled twice.
    """
    return jax.random.key(seed)


__all__ = [
    "ESMC_SUBDIRECTORY",
    "LoadedModel",
    "esmc_directory",
    "language_model_states",
    "load",
    "predict",
    "predict_job",
    "seed_key",
]
