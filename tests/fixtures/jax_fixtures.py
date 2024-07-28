import os

import jax
import pytest

import xjax

__all__ = [
    "enable_parallel_processing",
    "rng",
]


@pytest.fixture(autouse=True)
def enable_parallel_processing():
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"


@pytest.fixture
def rng() -> jax.Array:
    return jax.random.key(xjax.tools.seed())
