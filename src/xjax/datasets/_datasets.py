"""Provides library of sample datasets."""
from typing import List, Tuple

import jax
from jax import numpy as jnp

from xjax.tools import default_arg

__all__ = [
    "circle",
    "diagonal",
    "sphere",
    "freq_word_pair"
]


def freq_word_pair(*, num_sentences: int | None = None,
           len_sentence: int | None = None,
           vocab: List[str] | None = None,
           freq_pair: Tuple[str, str] | None=None) -> List[List[str]]:
    """Generate a text dataset consisting of sentences where a choen word-pair is frequently adjacent"""

    # Defaults
    num_sentences = default_arg(num_sentences, 5)
    len_sentence = default_arg(len_sentence, 3)
    freq_pair = default_arg(freq_pair, ("apple", "banana"))

    vocab = default_arg(vocab, ["apple", "banana", "grape", "peach", "orange"])

    # Make sure we have enough words 
    assert(len(vocab) >= len_sentence)

    # Start by randomly generating num_sentences/2 sentences
    n = num_sentences//2

    def dfs(sentences, vocab, sentence):

        if len(sentences) == n:
            return

        if len(sentence) == len_sentence-1:
            if f"{freq_pair[0]}-{freq_pair[1]}" in sentence:
                i = sentence.index(f"{freq_pair[0]}-{freq_pair[1]}")
                new_sentence = sentence[:i] + (freq_pair[0], freq_pair[1]) + sentence[i+1:]
                sentences.append(new_sentence)
                return
        
        if len(sentence) == len_sentence and f"{freq_pair[0]}-{freq_pair[1]}" not in sentence:
            sentences.append(sentence)
            return

        for i in range(len(vocab)):
            dfs(sentences, vocab[:i]+vocab[i+1:], sentence + (vocab[i],))

    random_sentences = []
    for i in range(len(vocab)):
        dfs(random_sentences, vocab[:i]+vocab[i+1:], (vocab[i],))

    # Then generate num_sentences - n sentences with the chosen word-pair
    # always adjacent to each other
    n = num_sentences - n

    # Remove the more frequent pair first
    vocab.remove(freq_pair[0])
    vocab.remove(freq_pair[1])

    # Add them to the vocabulary as a single token
    vocab.append(f"{freq_pair[0]}-{freq_pair[1]}")

    freq_sentences = []
    for i in range(len(vocab)):
        dfs(freq_sentences, vocab[:i]+vocab[i+1:], (vocab[i],))

    sentences = random_sentences + freq_sentences

    # Make sure we have generated the correct number of sentences
    assert(len(sentences) == num_sentences)

    return sentences


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
