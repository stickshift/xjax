from functools import partial
from typing import Any, Mapping, Sequence

from flax import linen as nn
import jax
from jax import Array
from jax import numpy as jnp
import optax

from xjax.signals import train_epoch_completed, train_epoch_started
from xjax.tools import default_arg

__all__ = [
    "mlp",
    "predict",
    "train",
]


type Parameters = Mapping[str, Any]


class MLP(nn.Module):
    """Basic fully-connected feedforward neural network."""

    hiddens: Sequence[int]
    outputs: int

    def setup(self):
        """Construct model."""
        self.hidden_layers = tuple(nn.Dense(n) for n in self.hiddens)
        self.output_layer = nn.Dense(self.outputs)

    def __call__(self, X: Array):
        """Forward pass."""
        logits = X

        # Hidden layers
        for layer in self.hidden_layers:
            logits = nn.relu(layer(logits))

        # Output layer
        logits = self.output_layer(logits)

        return logits


def mlp(
    *,
    rng: Array,
    inputs: int,
    hiddens: Sequence[int] | None = None,
    outputs: int | None = None,
) -> tuple[MLP, Parameters]:
    """Create and initialize MLP model."""
    # Defaults
    hiddens = default_arg(hiddens, [])
    outputs = default_arg(outputs, 1)

    # Create model
    model = MLP(hiddens=hiddens, outputs=outputs)

    # Initialize params
    params = model.init(rng, jnp.empty(shape=(inputs,)))

    return model, params


def train(
    model: nn.Module,
    *,
    params: Parameters,
    X: Array,
    y: Array,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
) -> Parameters:
    """Train flax linen model."""
    # Defaults
    epochs = default_arg(epochs, 1)
    batch_size = default_arg(batch_size, 1)
    learning_rate = default_arg(learning_rate, 0.01)

    # Batch data
    X_batches, y_batches = _batch(X, y, batch_size)

    # Configure loss function
    loss_fn = jax.value_and_grad(jax.jit(partial(_train_step, model)))

    # Configure optimizer
    optimizer = optax.adam(learning_rate=learning_rate)
    optimizer_state = optimizer.init(params)

    # Iterate over epochs
    for epoch in range(epochs):

        # Emit signal
        train_epoch_started.send(model, epoch=epoch)

        # Iterate over batches
        loss = None
        for i in range(len(X_batches)):
            # Compute loss and gradients
            loss, grads = loss_fn(params, X_batches[i], y_batches[i])

            # Compute updates
            updates, optimizer_state = optimizer.update(grads, optimizer_state)

            # Update params
            params = optax.apply_updates(params, updates)

        # Emit signal
        train_epoch_completed.send(model, epoch=epoch, loss=loss)

    return params


def predict(model: nn.Module, *, params: Parameters, X: Array) -> Array:
    # Predict
    y_score = nn.sigmoid(model.apply(params, X))

    # Remove extra dimensions
    y_score = y_score.squeeze()

    return y_score


def _batch(X: Array, y: Array, batch_size: int) -> tuple[list[Array], list[Array]]:
    # Calculate number of complete batches
    n_batches = len(X) // batch_size

    # Truncate and split
    length = n_batches * batch_size
    X_batches = jnp.split(X[0:length], n_batches)
    y_batches = jnp.split(y[0:length], n_batches)

    return X_batches, y_batches


def _train_step(model: nn.Module, params: Parameters, X_batch: Array, y_batch: Array) -> float:
    # Apply model
    logits = model.apply(params, X_batch)

    # Remove extra dimensions
    logits = logits.squeeze()

    # Compute loss
    loss = optax.losses.sigmoid_binary_cross_entropy(logits, y_batch)

    # Validate loss has expected (n,) shape
    assert loss.shape == (len(X_batch),)

    # Average loss across batch
    return loss.mean()
