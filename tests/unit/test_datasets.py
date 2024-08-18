import pytest

import jax
from jax import numpy as jnp

import xjax


def test_diagonal_dataset(rng: jax.Array):
    #
    # Whens
    #

    # I generate diagonal dataset
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


def test_circle_dataset(rng: jax.Array):
    #
    # Whens
    #

    # I generate circle dataset
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


def test_sphere_dataset(rng: jax.Array):
    #
    # Whens
    #

    # I generate sphere dataset
    X, y = xjax.datasets.sphere(rng=rng)

    #
    # Thens
    #

    # y should be 1 if point (x0, x1, x2) is within 0.4 from center (0.5, 0.5, 0.5)
    for i in range(len(X)):
        x0, x1, x2 = X[i][0], X[i][1], X[i][2]

        if jnp.sqrt((x0 - 0.5) ** 2 + (x1 - 0.5) ** 2 + (x2 - 0.5) ** 2) <= 0.4:
            assert y[i] == 1.0
        else:
            assert y[i] == 0.0

@pytest.mark.nlp
def test_freq_word_pair_dataset(rng: jax.Array):
    #
    # Whens
    #

    # I generate the frequent word-pair dataset
    sentences = xjax.datasets.freq_word_pair()

    #
    # Thens
    #

    # The most frequent word pair should appear adjacent to each other
    # At least half the time 
    def contains_word_pair(sentence, word_pair):
        return any(word_pair == sentence[i:i+2] for i in range(len(sentence) - 2 + 1))

    n = 0
    for s in sentences:
        if contains_word_pair(s, ("apple", "banana")):
            n += 1
    assert(n >= len(sentences)/2)