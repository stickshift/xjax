"""Provides library of sample datasets."""

from ._datasets import circle, diagonal, sphere, freq_word_pair
from .nlp import _nlp as nlp

__all__ = [
    "circle",
    "diagonal",
    "sphere",
    "freq_word_pair",
    "nlp"
]
