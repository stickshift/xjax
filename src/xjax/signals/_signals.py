from blinker import signal

__all__ = [
    "train_epoch_started",
    "train_epoch_completed",
]

train_epoch_started = signal("train_epoch_started")

train_epoch_completed = signal("train_epoch_completed")
