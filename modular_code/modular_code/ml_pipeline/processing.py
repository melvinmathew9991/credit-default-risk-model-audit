# import
import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from .logging_utils import get_logger

logger = get_logger(__name__)


# Function to create default labels
def create_label(df, dpd, months):
    """Genrate label according to dpd in months,
    returns dataframe with label columns
    Parameters
    ----------
    df : DataFrame
    dpd : Int (30, 60, 90)
    months : Int (1,2,3,4,5,6)

    Returns
    -------
    df : DataFrame

    Raises
    ------
    KeyError
        If any of the required emi_*_dpd columns is missing.
    """
    cols = ["emi_" + str(x) + "_dpd" for x in range(1, months + 1)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"cannot build the label, missing columns: {missing}")

    df["label"] = np.where(df[cols].max(axis=1) >= dpd, 1, 0)
    logger.info(
        "label = dpd%d within %d EMIs -> %d/%d positives (%.2f%%)",
        dpd,
        months,
        int(df["label"].sum()),
        len(df),
        100 * df["label"].mean(),
    )
    return df


# Data quality
#: Columns where a missing value was written as a literal zero, in two spellings.
PLACEHOLDER_COLUMNS = ("industry", "work_experience")
PLACEHOLDER_VALUES = ("0", "0.0", 0, 0.0)


def clean_placeholders(df, columns=PLACEHOLDER_COLUMNS, values=PLACEHOLDER_VALUES):
    """Map placeholder zeros in `columns` to a real missing value.

    `industry` and `work_experience` are 83.9% filled with a placeholder zero
    recorded two different ways - "0" (87,848 rows, 9.75% default rate) and
    "0.0" (32,766 rows, 4.76%). Both mean "not captured", but because they are
    distinct strings the encoder treats them as two categories with very
    different risk, so the model ends up scoring which spelling a record
    happened to be written with. That is a fingerprint of how the file was
    assembled, not an attribute of the borrower, and it cannot be justified to a
    credit committee or a regulator.

    Collapsing both to NaN keeps the legitimate signal ("this field was not
    captured") and discards the provenance artifact.

    Parameters
    ----------
    df : DataFrame
    columns : tuple of str
    values : tuple
        Values to treat as missing.

    Returns
    -------
    df : DataFrame
    """
    for c in columns:
        if c not in df.columns:
            continue
        before = int(df[c].isna().sum())
        df[c] = df[c].where(~df[c].isin(values), np.nan)
        after = int(df[c].isna().sum())
        logger.info(
            "%s: %d placeholder value(s) mapped to missing (null rate " "%.1f%% -> %.1f%%)",
            c,
            after - before,
            100 * before / len(df),
            100 * after / len(df),
        )
    return df


# Features
def derived_features(df):
    """Create Some Features
    ----------
    df : DataFrame

    Returns
    -------
    df : DataFrame

    Raises
    ------
    KeyError
        If a source column for one of the ratios is missing.
    """
    required = ["interest_received", "total_payement", "delinq_2yrs", "number_of_loans"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"cannot derive features, missing columns: {missing}")

    df["interest_received_ratio"] = (
        (df["interest_received"] / df["total_payement"]).replace([np.inf, -np.inf], 0).fillna(0)
    )
    df["total_payement_per_loan"] = (
        (df["total_payement"] / df["number_of_loans"]).replace([np.inf, -np.inf], 0).fillna(0)
    )
    df["delinq_2yrs_ratio"] = (
        (df["delinq_2yrs"] / df["number_of_loans"]).replace([np.inf, -np.inf], 0).fillna(0)
    )

    # These two divide by number_of_loans, which is 0 for ~99.6% of rows; the
    # resulting inf is mapped to 0, leaving a near-constant column. Warn rather
    # than let a dead feature pass unnoticed into the model.
    for c in ["total_payement_per_loan", "delinq_2yrs_ratio"]:
        zero_share = float((df[c] == 0).mean())
        if zero_share > 0.95:
            logger.warning(
                "derived feature %r is %.2f%% zeros - it carries almost " "no information",
                c,
                100 * zero_share,
            )
    return df


# Categorical encoding - target encoding
class categorical_encoding:
    """Target Encoding of categorical variables
    input dataframe, categorical columns, label name, parameters of target_encoder
    """

    def __init__(self, params):
        """
        Parameters
        ----------
        params : Dict
        """
        self.params = params

    def fit(self, df, cat_cols, label):
        """Fitting Encoder
        Parameters
        ----------
        df : DataFrame
        cat_cols : List (Categorical columns)
        label : String
        """
        self.te = ce.target_encoder.TargetEncoder(**self.params)
        self.te.fit(df[cat_cols], df[label])
        # category_encoders renamed `feature_names` -> `get_feature_names_in()` in 2.6;
        # cache the fitted column order so transform works on either version.
        self.feature_names = list(
            getattr(self.te, "feature_names", None) or self.te.get_feature_names_in()
        )

    def transform(self, d):
        """Transforming Data Encode and inplace transform categorical features
        Parameters
        ----------
        d : DataFrame

        Returns
        -------
        d : DataFrame
        """
        d = pd.concat(
            [d.drop(columns=self.feature_names), self.te.transform(d[self.feature_names])], axis=1
        )
        return d


def categorical_transform(train, val, hold_out, id_cols, params=None):
    """
    categorical encoding on train, val, hold_out

    The encoder is fitted on train only and applied to val and hold_out, which
    is what keeps the validation splits free of target leakage.

    ---------
    train: DataFrame
    val: DataFrame
    hold_out: DataFrame
    id_cols: List
    params: Dict, optional
        TargetEncoder parameters; defaults to the project settings.

    Returns:
    train: DataFrame
    val: DataFrame
    hold_out: DataFrame
    target_encoder: categorical_encoding
    """
    if params is None:
        params = {
            "verbose": 0,
            "cols": None,
            "drop_invariant": False,
            "return_df": True,
            "handle_missing": "value",
            "handle_unknown": "value",
            "min_samples_leaf": 5000,
            "smoothing": 1,
        }

    cat_cols = train.drop(columns=id_cols).select_dtypes(include=["category", "object"]).columns
    logger.info("target encoding %d categorical columns: %s", len(cat_cols), list(cat_cols))

    target_encoder = categorical_encoding(params)
    target_encoder.fit(train, cat_cols, "label")

    train = target_encoder.transform(train)
    val = target_encoder.transform(val)
    hold_out = target_encoder.transform(hold_out)

    # With min_samples_leaf=5000 the blending weight underflows to zero for any
    # category smaller than that, pinning it to the prior. A column that comes
    # out constant has been erased, not encoded - say so instead of letting the
    # feature selector silently drop it later.
    for c in cat_cols:
        if train[c].nunique() <= 1:
            logger.warning(
                "%r encoded to a single constant value - the encoder's "
                "min_samples_leaf (%s) is larger than every category in it",
                c,
                params.get("min_samples_leaf"),
            )

    # the fitted encoder is returned so it can be persisted alongside the
    # model - without it the saved model cannot score raw data.
    return train, val, hold_out, target_encoder


# Feature Selection for Random Forest and Decision Tree
def random_forest_zero_importance(df, id_cols, label, params):
    """Finding Zero Importance features using random forest
    ----------
    df : DataFrame
    id_cols : List
    label : String
    params : Dict

    Returns
    -------
    zero_fi : List
    """
    rf = RandomForestClassifier(**params)
    rf.fit(df.drop(columns=id_cols).fillna(0), df["label"])
    fi = pd.DataFrame(
        {"features": df.drop(columns=id_cols).columns, "importance": rf.feature_importances_}
    )
    zero_fi = fi[fi.importance == 0]["features"]
    return zero_fi


def decision_tree_zero_importance(df, id_cols, label, params):
    """Finding Zero Importance features using decision tree
    ----------
    df : DataFrame
    id_cols : List
    label : String
    params : Dict

    Returns
    -------
    zero_fi : List
    """
    dt = DecisionTreeClassifier(**params)
    dt.fit(df.drop(columns=id_cols).fillna(0), df["label"])
    fi = pd.DataFrame(
        {"features": df.drop(columns=id_cols).columns, "importance": dt.feature_importances_}
    )
    zero_fi = fi[fi.importance == 0]["features"]
    return zero_fi


# combine both the functions for feature selection


def select_features(train, val, hold_out, id_cols, rf_params=None, dt_params=None):
    """
    drops the common set of zero importance features from random forest and decision tree
    --------
    train: Dataframe
    val: DataFrame
    hold_out: DataFrame
    id_cols: List
    rf_params: Dict, optional
    dt_params: Dict, optional

    Returns:
    train: Dataframe
    val: Dataframe
    hold_out: Dataframe
    drop_cols: List
        Features dropped from the datasets.
    """
    # random_state is required: without it the zero-importance set - and so
    # the model's feature list - changes from run to run.
    if rf_params is None:
        rf_params = {
            "n_estimators": 250,
            "criterion": "entropy",
            "verbose": False,
            "n_jobs": -1,
            "random_state": 2019,
        }
    if dt_params is None:
        dt_params = {"random_state": 2019}

    if "random_state" not in rf_params or "random_state" not in dt_params:
        logger.warning(
            "feature selection without a random_state is not reproducible - "
            "the surviving feature list can change between runs"
        )

    rf_zero_imp = random_forest_zero_importance(train, id_cols, "label", rf_params)
    dt_zero_imp = decision_tree_zero_importance(train, id_cols, "label", dt_params)
    drop_cols = sorted(set(rf_zero_imp) & set(dt_zero_imp))

    logger.info("zero importance - random forest: %s", sorted(rf_zero_imp))
    logger.info("zero importance - decision tree: %s", sorted(dt_zero_imp))
    logger.info("dropping %d feature(s) flagged by both: %s", len(drop_cols), drop_cols)

    # remove the drop cols
    train = train.drop(drop_cols, axis=1)
    val = val.drop(drop_cols, axis=1)
    hold_out = hold_out.drop(drop_cols, axis=1)

    return train, val, hold_out, drop_cols
