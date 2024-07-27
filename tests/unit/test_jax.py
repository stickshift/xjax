from jax import numpy as jnp


def test_arrays():
    #
    # Givens
    #

    #
    # Whens
    #

    # I create array with values 1, 2, 3
    x = jnp.array([1, 2, 3])

    #
    # Thens
    #

    # x should have length 3
    assert len(x) == 3

    # 1, 2, and 3 should be in x
    assert all(v in x for v in [1, 2, 3])
