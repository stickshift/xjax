__all__ = [
    "default_arg",
]


def default_arg(a, default):
    """Shorthand for x = x if x is not None else default."""
    return a if a is not None else default
