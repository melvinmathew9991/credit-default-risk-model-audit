"""Logging setup for the pipeline.

Replaces the bare `print` calls, and - more importantly - the
`try/except Exception: print(e)` blocks that used to swallow failures and let a
function return None, so the real error surfaced much later as an unrelated
AttributeError.
"""

import logging
import os
import sys


def setup_logging(level="INFO", log_file=None):
    """Configure the root logger for a pipeline run.

    Parameters
    ----------
    level : str
        Level name, e.g. "INFO" or "DEBUG".
    log_file : str, optional
        If given, logs go to this file as well as stderr.

    Returns
    -------
    logging.Logger
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)-22s %(message)s", datefmt="%H:%M:%S"
    )

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # third-party noise: hyperopt logs two lines per trial at INFO
    for noisy in ("matplotlib", "numexpr", "hyperopt", "PIL", "shap"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root


def get_logger(name):
    """Module-level logger."""
    return logging.getLogger(name)
