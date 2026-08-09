# imort libraries
import pandas as pd
import numpy as np

from .logging_utils import get_logger

logger = get_logger(__name__)


# Function to read and drop unnecessary columns
def process_data(path, drop_columns):
    """Read data, drop columns and do processing

    Errors are allowed to propagate: a missing file or a missing column is a
    configuration mistake, and returning None here only defers the failure to a
    confusing AttributeError further down the pipeline.

    Parameters
    ----------
    path : String
    drop_columns : List

    Returns
    -------
    df :  DataFrame

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    KeyError
        If any entry of `drop_columns` is not a column of the file.
    """
    # low_memory=False forces single-pass type inference.
    # With the default chunked inference pandas assigns different python
    # types to identical text in `industry` and `work_experience` depending
    # on which 128k-row chunk it lands in (str '0' vs int 0 vs float 0.0).
    # Those become distinct categories downstream, so the encoded feature
    # values would depend on row position in the file.
    df = pd.read_csv(path, low_memory=False)
    logger.info("read %s: %d rows x %d columns", path, len(df), df.shape[1])

    missing = [c for c in drop_columns if c not in df.columns]
    if missing:
        raise KeyError(f"columns to drop are not present in {path}: {missing}")

    df = df.drop(columns=drop_columns)
    logger.info("dropped %s", drop_columns)
    return df


# Function to split the data
def data_split(df, train_max_yearmo=202203, val_yearmo=202204, hold_out_yearmo=202205):
    """Split data in train, val, hold_out on the application month.

    Parameters
    ----------
    df : DataFrame
    train_max_yearmo : int
        Training set is every application month at or before this.
    val_yearmo : int
    hold_out_yearmo : int

    Returns
    -------
    train :  DataFrame,
    val :  DataFrame,
    hold_out :  DataFrame

    Raises
    ------
    KeyError
        If `yearmo` is missing.
    ValueError
        If any split is empty, or the months are not strictly ordered.
    """
    if "yearmo" not in df.columns:
        raise KeyError("`yearmo` column is required to build a time-based split")
    if not (train_max_yearmo < val_yearmo < hold_out_yearmo):
        raise ValueError(
            f"splits must be ordered in time: train<={train_max_yearmo} "
            f"< val=={val_yearmo} < hold_out=={hold_out_yearmo}")

    train = df[df.yearmo <= train_max_yearmo]
    val = df[df.yearmo == val_yearmo]
    hold_out = df[df.yearmo == hold_out_yearmo]

    for name, part in (("train", train), ("val", val), ("hold_out", hold_out)):
        if len(part) == 0:
            raise ValueError(
                f"{name} split is empty - check the yearmo boundaries against "
                f"the data, which covers {sorted(df.yearmo.unique())}")

    logger.info("split rows - train=%d val=%d hold_out=%d",
                len(train), len(val), len(hold_out))
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            hold_out.reset_index(drop=True))
