"""Profile the historical and mapped OpenDDE shape-complementarity stages."""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.opendde.models import shape_complementarity as module


def _block(tree):
    return jax.tree.map(lambda value: value.block_until_ready(), tree)


def _arguments(tokens: int, atoms_per_token: int, samples: int, seed: int):
    atoms = tokens * atoms_per_token
    owner = np.repeat(np.arange(tokens, dtype=np.int64), atoms_per_token)
    representative = np.zeros(atoms, dtype=bool)
    representative[::atoms_per_token] = True
    features = {
        "token_index": np.arange(tokens, dtype=np.int64),
        "atom_to_token_idx": owner,
        "asym_id": (np.arange(tokens) >= tokens // 2).astype(np.int64),
        "distogram_rep_atom_mask": representative,
        "is_protein": np.ones(atoms, dtype=np.float32),
        "is_ligand": np.zeros(atoms, dtype=np.float32),
        "is_dna": np.zeros(atoms, dtype=np.float32),
        "is_rna": np.zeros(atoms, dtype=np.float32),
    }
    rng = np.random.default_rng(seed)
    coordinates = jnp.asarray(
        rng.normal(size=(samples, atoms, 3)), dtype=jnp.float32
    )
    return coordinates, features, jnp.ones(atoms, dtype=bool)


def _stack(entries):
    return {
        name: jnp.stack([entry[name] for entry in entries])
        for name in entries[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--atoms-per-token", type=int, default=4)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4)
    args = parser.parse_args()

    coordinates, features, atom_mask = _arguments(
        args.tokens, args.atoms_per_token, args.samples, args.seed
    )

    def historical():
        return _stack(
            [
                module.compute_shape_complementarity(
                    coordinates[index], features, atom_mask
                )
                for index in range(args.samples)
            ]
        )

    def mapped():
        return module.compute_shape_complementarity_batched(
            coordinates, features, atom_mask
        )

    timings: dict[str, list[float]] = {}
    outputs = {}
    for name, function in (("historical", historical), ("mapped", mapped)):
        _block(function())
        _block(function())
        measured = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            value = _block(function())
            measured.append((time.perf_counter() - start) * 1e3)
        timings[name] = measured
        outputs[name] = value

    differences = {}
    for name in outputs["historical"]:
        old = np.asarray(outputs["historical"][name])
        new = np.asarray(outputs["mapped"][name])
        if old.dtype == bool:
            differences[name] = {"mismatches": int(np.count_nonzero(old != new))}
        else:
            delta = old.astype(np.float64) - new.astype(np.float64)
            differences[name] = {
                "max_abs": float(np.max(np.abs(delta), initial=0.0)),
                "rms": float(np.sqrt(np.mean(delta * delta))),
            }

    resolved = module.resolve_shape_comp_token_features(
        features, np.asarray(atom_mask), args.tokens
    )
    graph_features = module._graph_features(resolved)
    settings = module._settings({})

    def historical_graph(coordinate, mask, graph):
        return _stack(
            [
                module._compute_shape_complementarity_resolved(
                    coordinate[index],
                    mask,
                    graph,
                    n_token=args.tokens,
                    is_structural=resolved.is_structural,
                    settings=settings,
                )
                for index in range(args.samples)
            ]
        )

    def mapped_graph(coordinate, mask, graph):
        return module._compute_shape_complementarity_samples(
            coordinate,
            mask,
            graph,
            n_token=args.tokens,
            is_structural=resolved.is_structural,
            settings=settings,
        )

    compiled = {}
    for name, function in (
        ("historical", historical_graph),
        ("mapped", mapped_graph),
    ):
        lowered = jax.jit(function).lower(
            coordinates, atom_mask, graph_features
        )
        memory = lowered.compile().memory_analysis()
        compiled[name] = {
            "stablehlo_bytes": len(lowered.as_text().encode()),
            "temporary_bytes": memory.temp_size_in_bytes,
            "argument_bytes": memory.argument_size_in_bytes,
            "output_bytes": memory.output_size_in_bytes,
        }

    print(
        json.dumps(
            {
                "backend": jax.default_backend(),
                "device": str(jax.devices()[0]),
                "tokens": args.tokens,
                "atoms": int(coordinates.shape[-2]),
                "samples": args.samples,
                "warm_median_ms": {
                    name: float(np.median(values))
                    for name, values in timings.items()
                },
                "differences": differences,
                "compiled": compiled,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
