import logging

import pytest

__all__ = [
    "disable_loggers",
]


@pytest.fixture(autouse=True)
def disable_loggers(caplog):
    caplog.set_level(logging.WARNING, logger="jax")
