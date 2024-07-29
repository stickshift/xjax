"""Provides library of sample datasets."""

import jax
from jax import numpy as jnp

from xjax.tools import default_arg

__all__ = [
    "circle",
    "diagonal",
    "sphere",
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


def circle(*, rng: jax.Array, n: int | None = None) -> tuple[jax.Array, jax.Array]:
    """Generate circle dataset where y is 1 if point (x0, x1) is within 0.4 of center."""
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

    # Generate y: 1 if point (x0, x1) is within 0.4 of center (0.5, 0.5)
    y = jnp.where(jnp.sqrt((x0 - 0.5) ** 2 + (x1 - 0.5) ** 2) <= 0.4, 1.0, 0.0)  # noqa: PLR2004

    return X, y


def sphere(*, rng: jax.Array, n: int | None = None) -> tuple[jax.Array, jax.Array]:
    """Generate sphere dataset where y is 1 if point (x0, x1, x2) is within 0.4 of center."""
    # Defaults
    n = default_arg(n, 10000)

    # RNGs
    x0_rng, x1_rng, x2_rng = jax.random.split(rng, 3)

    # x0
    x0 = jax.random.uniform(x0_rng, shape=(n,))

    # x1
    x1 = jax.random.uniform(x1_rng, shape=(n,))

    # x2
    x2 = jax.random.uniform(x2_rng, shape=(n,))

    # Stack x0, x1, x2 into X
    X = jnp.stack([x0, x1, x2], axis=1)

    # Generate y: 1 if point (x0, x1, x2) is within 0.4 of center (0.5, 0.5, 0.5)
    y = jnp.where(jnp.sqrt((x0 - 0.5) ** 2 + (x1 - 0.5) ** 2 + (x2 - 0.5) ** 2) <= 0.4, 1.0, 0.0)  # noqa: PLR2004

    return X, y
