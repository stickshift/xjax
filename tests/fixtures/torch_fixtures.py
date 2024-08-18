import pytest

import torch

__all__ = [
    "device",
]


@pytest.fixture
def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")
