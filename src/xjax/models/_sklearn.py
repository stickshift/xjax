"""Sci-Kit Learn models."""

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

__all__ = [
    "logistic_regression",
    "predict",
    "train",
    "xgb_classifier",
]


def logistic_regression(**kwargs) -> LogisticRegression:
    """Create LR model."""
    return LogisticRegression(**kwargs)


def xgb_classifier(**kwargs) -> XGBClassifier:
    """Create xgb model."""
    return XGBClassifier(**kwargs)


def train(model, *, X, y):
    """Train sklearn model."""
    return model.fit(X, y)


def predict(model, *, X):
    """Predict targets for X.

    Returns:
        Prediction probabilities.
    """
    return model.predict_proba(X)[:, 1]
