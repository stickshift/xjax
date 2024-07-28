from typing import Sequence

from IPython.display import clear_output
from matplotlib import pyplot as plt

__all__ = [
    "loss_history",
]


def loss_history(history: Sequence, elapsed: float):
    """Plot history dynamically."""
    # Lookup current axes
    fig = plt.gcf()
    ax = plt.gca()

    # Clear previous plot
    ax.clear()

    # Plot history
    fig.set_size_inches(5, 3)
    ax.plot(history)
    ax.set_title("Training Loss")
    ax.set_xlabel(f"Epoch ({elapsed:0.1f} elapsed)")

    # Replace cell output
    clear_output(wait=True)
    plt.show()
