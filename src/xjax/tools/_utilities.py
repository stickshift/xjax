import asyncio as aio
import inspect
import logging
from contextlib import contextmanager
from logging import Logger
from time import time_ns

__all__ = [
    "default_arg",
    "seed",
    "trace",
]

from typing import Optional


def default_arg(a, default):
    """Shorthand for x = x if x is not None else default."""
    return a if a is not None else default


def seed() -> int:
    return time_ns()


@contextmanager
def trace(
    log: Logger,
    prefix: Optional[str] = None,
    log_level: Optional[int] = None,
    elevate_log_level_on_error: Optional[bool] = None,  # noqa: FBT001
):
    # Defaults
    log_level = default_arg(log_level, logging.INFO)
    elevate_log_level_on_error = default_arg(elevate_log_level_on_error, True)

    # Lookup calling functions name
    prefix = default_arg(prefix, inspect.currentframe().f_back.f_back.f_code.co_name)

    # _log_msg(log, log_level, f"{prefix} - started")

    start_time = time_ns()
    try:
        # Return to context code
        yield

        # Context succeeded
        _log_msg(log, log_level, f"{prefix} - completed in {(time_ns() - start_time)/1e6}ms")
    except aio.CancelledError:
        _log_msg(log, log_level, f"{prefix} - cancelled in {(time_ns() - start_time)/1e6}ms")
        raise
    except Exception as e:
        _log_msg(
            log,
            logging.ERROR if elevate_log_level_on_error else log_level,
            f"{prefix} - failed in {(time_ns() - start_time)}ns - {e}",
        )
        raise
    except:  # noqa
        _log_msg(
            log,
            logging.ERROR if elevate_log_level_on_error else log_level,
            f"{prefix} - failed in {(time_ns() - start_time)}ns",
        )
        raise


def _log_msg(log: Logger, log_level: int, msg: str):
    if log_level == logging.DEBUG:
        log.debug(msg)
    elif log_level == logging.INFO:
        log.info(msg)
    elif log_level == logging.WARNING:
        log.warning(msg)
    elif log_level == logging.ERROR:
        log.exception(msg)
    else:
        error_msg = f"Unsupported log_level {log_level}"
        raise ValueError(error_msg)

