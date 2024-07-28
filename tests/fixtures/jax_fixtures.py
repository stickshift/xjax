import jax
import pytest

__all__ = [
    "rng",
]


@pytest.fixture
def rng() -> jax.Array:
    return jax.random.key(42)
