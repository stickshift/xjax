from time import time
from functools import partial
from typing import Any, Mapping, Sequence, Tuple

import logging

import jax
from jax import Array
from jax import numpy as jnp
import optax

from xjax.signals import train_epoch_completed, train_epoch_started
from xjax.tools import default_arg

__all__ = [
    "sgns",
    "predict",
    "train"
]


Parameters = Mapping[str, Any]


logger = logging.getLogger(__name__)

class SGNS:

    def __call__(self, params, X):

        pos_idxs, neg_idxs = X

        # first look at pos examples
        pos_idxs = jnp.array(pos_idxs)
        input = params[pos_idxs[:, 0], :, 0]
        output = params[pos_idxs[:, 1], :, 1]
        pos_logits = jnp.sum(input*output, axis=1)

        # then look at neg examples
        neg_idxs = jnp.array(neg_idxs)
        input = params[neg_idxs[:, 0], :, 0]
        output = params[neg_idxs[:, 1], :, 1]
        neg_logits = jnp.sum(input*output, axis=1)

        return pos_logits, neg_logits

def sgns(
        *,
        rng: Array,
        vocab_size: int,
        embedding_size: int,
) -> tuple[SGNS, Parameters]:
    """Create and initialize SGNS model."""

    model = SGNS()
    params = jax.random.normal(rng, (vocab_size, embedding_size, 2))

    return model, params

def train(
        model: SGNS,
        *,
        rng: Array,
        params: Parameters,
        X: Array,
        neg_per_pos: int,
        K: int,
        epochs: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
) -> Parameters:
    """Train the SGNS model."""

    epochs = default_arg(epochs, 1)
    batch_size = default_arg(batch_size, 1)
    learning_rate = default_arg(learning_rate, 0.01)

    start_time = time()

    # Set up optimizer 
    optimizer = optax.adam(learning_rate=learning_rate)
    optimizer_state = optimizer.init(params)

    loss_fn = jax.value_and_grad(partial(_loss, model, K))
    step_fn = jax.jit(partial(_step, loss_fn, optimizer))

    num_iter = len(X)//batch_size

    for epoch in range(epochs):
        # Emit signal
        train_epoch_started.send(model, epoch=epoch, elapsed=(time() - start_time))

        # Iterate over batches
        loss = None
        for i in range(num_iter):
            batch = _batch(rng=rng, vocab_size=params.shape[0], batch_size=batch_size, neg_per_pos=neg_per_pos, dataset=X)
            params, optimizer_state, loss = step_fn(optimizer_state, params, batch)
        
        # Emit signal
        train_epoch_completed.send(model, epoch=epoch, loss=loss, elapsed=(time() - start_time))
    
    return params


def _batch(*, rng: Array, vocab_size: int, batch_size: int, neg_per_pos: int, dataset: Sequence) -> Tuple[Sequence]:

  k1, k2 = jax.random.split(rng, 2)

  # Generate positive examples by randomly sampling the dataset
  pos_idxs = jax.random.randint(k1, (batch_size,), minval=0, maxval=len(dataset)-1)

  # Generate negative examples by randomly sampling token-pairs from the vocabulary
  neg_idxs = jax.random.randint(k2, (neg_per_pos*batch_size,2), minval=0, maxval=vocab_size-1) 
  
  return dataset[pos_idxs, :], neg_idxs


def _loss(model, K, params, X_batch) -> float:

    pos_logits, neg_logits = model(params, X_batch)

    # First look at pos samples
    loss = optax.losses.sigmoid_binary_cross_entropy(pos_logits, jnp.ones(pos_logits.size)).mean()

    # Then look at neg samples
    loss += K*optax.losses.sigmoid_binary_cross_entropy(neg_logits, jnp.zeros(neg_logits.size)).mean()

    return loss
  

def _step(loss_fn, optimizer, optimizer_state, params, X_batch):

  loss, grads = loss_fn(params, X_batch)

  # Compute updates
  updates, optimizer_state = optimizer.update(grads, optimizer_state)

  # Update params
  params = optax.apply_updates(params, updates)

  return params, optimizer_state, loss
