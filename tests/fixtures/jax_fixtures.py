import jax
import pytest

import xjax

__all__ = [
    "rng",
]


@pytest.fixture
def rng() -> jax.Array:
    return jax.random.key(xjax.tools.seed())
