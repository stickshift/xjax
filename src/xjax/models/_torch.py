import logging
from itertools import pairwise
from time import time_ns

from jax import Array
import numpy as np
import torch
from torch import nn, optim, tensor
from torch.utils.data import DataLoader, TensorDataset

from xjax.signals import train_epoch_completed, train_epoch_started
from xjax.tools import default_arg, trace

__all__ = [
    "mlp",
    "predict",
    "train",
]

# Module logger
logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """Basic fully-connected feedforward neural network."""

    def __init__(self, inputs: int, hiddens: list[int], outputs: int):
        super().__init__()

        layers = []
        dims = list(pairwise([inputs, *hiddens, outputs]))
        for i, dim in enumerate(dims):
            layers.append(nn.Linear(dim[0], dim[1]))
            if i < len(dims) - 1:
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, X):
        """Forward pass."""
        return self.layers(X)


def mlp(
    *,
    inputs: int,
    hiddens: list[int] | None = None,
    outputs: int | None = None,
) -> MLP:
    """Create and initialize MLP model."""
    # Defaults
    hiddens = default_arg(hiddens, [])
    outputs = default_arg(outputs, 1)

    # Create model
    model = MLP(inputs=inputs, hiddens=hiddens, outputs=outputs)

    return model


def train[MT: nn.Module](
    model: MT,
    *,
    X: Array,
    y: Array,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
) -> MT:
    """Train torch model."""
    # Defaults
    epochs = default_arg(epochs, 1)
    batch_size = default_arg(batch_size, 1)
    learning_rate = default_arg(learning_rate, 0.01)

    start_time = time_ns()

    # Convert data
    X = tensor(np.array(X), dtype=torch.float32)
    y = tensor(np.array(y), dtype=torch.float32)

    # Batch data
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size)

    # Configure loss function
    loss_fn = nn.BCEWithLogitsLoss()

    # Configure optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Iterate over epochs
    for epoch in range(epochs):
        # Emit signal
        train_epoch_started.send(model, epoch=epoch, elapsed=(time_ns() - start_time))

        # Iterate over batches
        loss = None
        for X_batch, y_batch in loader:
            # Apply model
            logits = model(X_batch)

            # Remove extra dimensions
            logits = logits.squeeze()

            # Compute loss
            loss = loss_fn(logits, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Emit signal
        train_epoch_completed.send(model, epoch=epoch, loss=loss, elapsed=(time_ns() - start_time))

    return model


def predict(model: nn.Module, *, X: Array) -> torch.Tensor:
    with torch.no_grad():
        # Convert data
        X = tensor(np.array(X), dtype=torch.float32)

        # Predict
        logits = model(X)
        y_score = torch.sigmoid(logits)

        # Remove extra dimensions
        y_score = y_score.squeeze()

    return y_score
