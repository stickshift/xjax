import jax

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

    # y should be 1 if x1 >= x0
    for i in range(len(X)):
        if X[i][1] >= X[i][0]:
            assert y[i] == 1.0
        else:
            assert y[i] == 0.0
