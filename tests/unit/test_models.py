import logging

import jax
from pytest import approx
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import xjax
from xjax.signals import train_epoch_completed

# Module logger
logger = logging.getLogger(__name__)


def test_lr_diagonal(rng: jax.Array):
    #
    # Givens
    #

    # I created diagonal dataset
    rng, dataset_rng = jax.random.split(rng)
    X, y = xjax.datasets.diagonal(rng=dataset_rng)

    # I split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    #
    # Whens
    #

    # I create LR model
    model = xjax.models.sklearn.logistic_regression()

    # I train model
    model = xjax.models.sklearn.train(model, X=X_train, y=y_train)

    # I test model
    y_score = xjax.models.sklearn.predict(model, X=X_test)
    auroc = roc_auc_score(y_test, y_score)

    #
    # Thens
    #

    # Model should get perfect score
    assert auroc == approx(1.0, abs=0.001)


def test_xgb_circle(rng: jax.Array):
    #
    # Givens
    #

    # I created circle dataset
    rng, dataset_rng = jax.random.split(rng)
    X, y = xjax.datasets.circle(rng=dataset_rng)

    # I split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    #
    # Whens
    #

    # I create xgb model
    model = xjax.models.sklearn.xgb_classifier()

    # I train model
    model = xjax.models.sklearn.train(model, X=X_train, y=y_train)

    # I test model
    y_score = xjax.models.sklearn.predict(model, X=X_test)
    auroc = roc_auc_score(y_test, y_score)

    #
    # Thens
    #

    # Model should get high score
    assert auroc > 0.9


def test_flax_2x1_diagonal(rng: jax.Array):
    #
    # Givens
    #

    # Hyperparams
    batch_size = 10
    epochs = 10
    learning_rate = 0.01

    # I created diagonal dataset
    rng, dataset_rng = jax.random.split(rng)
    X, y = xjax.datasets.diagonal(rng=dataset_rng)

    # I split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Event collectors
    losses = []

    #
    # Whens
    #

    # I create 2x1 flax model
    rng, model_rng = jax.random.split(rng)
    model, params = xjax.models.flax.mlp(rng=model_rng, inputs=2, outputs=1)

    # I subscribe to epoch completed signal for model
    @train_epoch_completed.connect_via(model)
    def collect_events(sender, epoch, loss, **_):
        losses.append(loss)

        logger.info(f"epoch={epoch}, loss={loss:0.4f}")

    # I train model
    params = xjax.models.flax.train(
        model,
        params=params,
        X=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    # I test model
    y_score = xjax.models.flax.predict(model, params=params, X=X_test)
    auroc = roc_auc_score(y_test, y_score)

    #
    # Thens
    #

    # Model should get perfect score
    assert auroc == approx(1.0, abs=0.001)

    # We should have collected 1 loss for each epoch
    assert len(losses) == epochs


def test_flax_2x5x1_circle(rng: jax.Array):
    #
    # Givens
    #

    # Hyperparams
    batch_size = 10
    epochs = 10
    learning_rate = 0.01

    # I created circle dataset
    rng, dataset_rng = jax.random.split(rng)
    X, y = xjax.datasets.circle(rng=dataset_rng)

    # I split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    #
    # Whens
    #

    # I create 2x5x1 flax model
    rng, model_rng = jax.random.split(rng)
    model, params = xjax.models.flax.mlp(rng=model_rng, inputs=2, hiddens=[5], outputs=1)

    # I log events
    @train_epoch_completed.connect_via(model)
    def collect_events(sender, epoch, loss, **_):
        logger.info(f"epoch={epoch}, loss={loss:0.4f}")

    # I train model
    params = xjax.models.flax.train(
        model,
        params=params,
        X=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    # I test model
    y_score = xjax.models.flax.predict(model, params=params, X=X_test)
    auroc = roc_auc_score(y_test, y_score)

    #
    # Thens
    #

    # Model should get "good" score
    assert auroc > 0.8


def test_torch_2x1_diagonal(rng: jax.Array):
    #
    # Givens
    #

    # Hyperparams
    batch_size = 10
    epochs = 10
    learning_rate = 0.01

    # I created diagonal dataset
    rng, dataset_rng = jax.random.split(rng)
    X, y = xjax.datasets.diagonal(rng=dataset_rng)

    # I split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Event collectors
    losses = []

    #
    # Whens
    #

    # I create 2x1 torch model
    model = xjax.models.torch.mlp(inputs=2, outputs=1)

    # I subscribe to epoch completed signal for model
    @train_epoch_completed.connect_via(model)
    def collect_events(sender, epoch, loss, **_):
        losses.append(loss)

        logger.info(f"epoch={epoch}, loss={loss:0.4f}")

    # I train model
    model = xjax.models.torch.train(
        model,
        X=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    # I test model
    y_score = xjax.models.torch.predict(model, X=X_test)
    auroc = roc_auc_score(y_test, y_score)

    #
    # Thens
    #

    # Model should get perfect score
    assert auroc == approx(1.0, abs=0.001)

    # We should have collected 1 loss for each epoch
    assert len(losses) == epochs


def test_torch_2x5x1_circle(rng: jax.Array):
    #
    # Givens
    #

    # Hyperparams
    batch_size = 10
    epochs = 10
    learning_rate = 0.01

    # I created circle dataset
    rng, dataset_rng = jax.random.split(rng)
    X, y = xjax.datasets.circle(rng=dataset_rng)

    # I split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    #
    # Whens
    #

    # I create 2x5x1 torch model
    model = xjax.models.torch.mlp(inputs=2, hiddens=[5], outputs=1)

    # I log events
    @train_epoch_completed.connect_via(model)
    def collect_events(sender, epoch, loss, **_):
        logger.info(f"epoch={epoch}, loss={loss:0.4f}")

    # I train model
    model = xjax.models.torch.train(
        model,
        X=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    # I test model
    y_score = xjax.models.torch.predict(model, X=X_test)
    auroc = roc_auc_score(y_test, y_score)

    #
    # Thens
    #

    # Model should get "good" score
    assert auroc > 0.8
