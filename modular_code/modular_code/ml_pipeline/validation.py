"""Time-aware validation for the credit risk model.

Two defects in the original design are addressed here.

**A single validation split doing three jobs.** The original `val` month drove
early stopping in all 50 tuning trials, the Hyperopt objective, and final model
selection, so its AUC was optimistically biased and there was no estimate of
variance. `expanding_window_folds` replaces it with walk-forward folds: each
fold trains on everything up to a month and validates on the next one, which is
how the model will actually be used.

**Customers spanning the splits.** 9,975 `User_id` values repeat in the data;
641 training customers reappeared in validation and 646 in hold-out. A repeat
customer in the evaluation set is partly memorised rather than predicted.
`drop_leaked_users` removes them from the *later* period, so evaluation stays
clean without discarding training data.
"""

import numpy as np

from .logging_utils import get_logger

logger = get_logger(__name__)


def expanding_window_folds(df, months=None, min_train_months=1, time_col="yearmo"):
    """Walk-forward folds over the application month.

    With months [1, 2, 3, 4] and min_train_months=1 this yields

        train [1]        -> validate 2
        train [1, 2]     -> validate 3
        train [1, 2, 3]  -> validate 4

    Parameters
    ----------
    df : DataFrame
        Must contain `time_col`.
    months : list, optional
        Months to use, ascending. Defaults to every month present.
        Pass the tuning months only - keep the hold-out month out of this.
    min_train_months : int
        Months required before the first validation fold.
    time_col : str

    Yields
    ------
    (fold_index, train_months, val_month, train_idx, val_idx)
    """
    if time_col not in df.columns:
        raise KeyError(f"{time_col!r} is required to build time-based folds")

    months = sorted(df[time_col].unique()) if months is None else sorted(months)
    if len(months) <= min_train_months:
        raise ValueError(
            f"need more than {min_train_months} month(s) to build a fold, " f"got {months}"
        )

    for i in range(min_train_months, len(months)):
        train_months = months[:i]
        val_month = months[i]
        train_idx = df.index[df[time_col].isin(train_months)]
        val_idx = df.index[df[time_col] == val_month]
        logger.info(
            "fold %d: train %s (%d rows) -> validate %s (%d rows)",
            i - min_train_months + 1,
            train_months,
            len(train_idx),
            val_month,
            len(val_idx),
        )
        yield i - min_train_months + 1, train_months, val_month, train_idx, val_idx


def drop_leaked_users(train, later, id_col="User_id", name="later"):
    """Remove from `later` any customer already seen in `train`.

    Parameters
    ----------
    train : DataFrame
    later : DataFrame
        The chronologically later split - validation or hold-out.
    id_col : str
    name : str
        Label used in the log message.

    Returns
    -------
    DataFrame
        `later` with the repeat customers removed.
    """
    if id_col not in train.columns or id_col not in later.columns:
        logger.warning("%r not present in both frames - skipping dedup", id_col)
        return later

    seen = set(train[id_col])
    mask = ~later[id_col].isin(seen)
    removed = int((~mask).sum())
    if removed:
        logger.info(
            "dedup: removed %d of %d rows from %s (%.2f%%) whose customer "
            "already appears in training",
            removed,
            len(later),
            name,
            100 * removed / len(later),
        )
    return later.loc[mask].reset_index(drop=True)


def summarise_folds(scores):
    """Mean / std / min across folds, for a tuning objective and the manifest.

    Parameters
    ----------
    scores : list of float

    Returns
    -------
    Dict
    """
    a = np.asarray(scores, dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=0)),
        "min": float(a.min()),
        "max": float(a.max()),
        "n_folds": int(a.size),
    }
