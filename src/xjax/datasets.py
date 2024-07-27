"""Provides library of sample datasets."""

import jax
from jax import numpy as jnp

from ._utilities import default_arg

__all__ = [
    "diagonal",
]


def diagonal(*, rng: jax.Array, n: int | None = None) -> tuple[jax.Array, jax.Array]:
    """Generate diagonal dataset where y is 1 if point (x0, x1) is above diagonal."""
    # Defaults
    n = default_arg(n, 10000)

    # RNGs
    x0_rng, x1_rng = jax.random.split(rng, 2)

    # x0
    x0 = jax.random.uniform(x0_rng, shape=(n,))

    # x1
    x1 = jax.random.uniform(x1_rng, shape=(n,))

    # Stack x0, x1 into X
    X = jnp.stack([x0, x1], axis=1)

    # Generate y: 1 if point (x0, x1) is above diagonal
    y = jnp.where(x1 >= x0, 1.0, 0.0)

    return X, y
