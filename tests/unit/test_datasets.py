import jax
from jax import numpy as jnp

import xjax


def test_diagonal_dataset():
    #
    # Givens
    #

    # I initialized rng
    rng = jax.random.key(42)

    #
    # Whens
    #

    # I load diagonal dataset
    X, y = xjax.datasets.diagonal(rng=rng)

    #
    # Thens
    #

    # y should be 1 if point (x0, x1) is above diagonal
    for i in range(len(X)):
        if X[i][1] >= X[i][0]:
            assert y[i] == 1.0
        else:
            assert y[i] == 0.0


def test_circle_dataset():
    #
    # Givens
    #

    # I initialized rng
    rng = jax.random.key(42)

    #
    # Whens
    #

    # I load circle dataset
    X, y = xjax.datasets.circle(rng=rng)

    #
    # Thens
    #

    # y should be 1 if point (x0, x1) is within 0.4 from center (0.5, 0.5)
    for i in range(len(X)):
        x0, x1 = X[i][0], X[i][1]

        if jnp.sqrt((x0 - 0.5) ** 2 + (x1 - 0.5) ** 2) <= 0.4:
            assert y[i] == 1.0
        else:
            assert y[i] == 0.0
