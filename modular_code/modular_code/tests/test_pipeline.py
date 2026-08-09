"""Regression tests for the credit risk pipeline.

These lock down the behaviours that were actually broken or fragile, so that the
same defects cannot come back silently:

  * the target column must never appear in the feature matrix
  * the time-based splits must not overlap
  * the target encoder must be fitted on train only
  * derived features must survive division by zero
  * the csv read must be type-stable regardless of row position

Run with:  python -m pytest tests -q
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_pipeline import evaluation as ev  # noqa: E402
from ml_pipeline import processing, training, utils  # noqa: E402

DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input", "credit_risk_data.csv"
)


# --------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def raw():
    return utils.process_data(DATA, ["gender"])


@pytest.fixture(scope="module")
def splits(raw):
    train, val, hold_out = utils.data_split(raw)
    for d in (train, val, hold_out):
        processing.create_label(d, dpd=60, months=3)
        processing.derived_features(d)
    return train, val, hold_out


# --------------------------------------------------------------------- loading
def test_gender_is_dropped(raw):
    """gender must not be usable as a credit factor."""
    assert "gender" not in raw.columns


def test_read_is_type_stable(raw):
    """Every value in a categorical column must parse to one python type.

    The default chunked read gave `industry` / `work_experience` a mix of str,
    int and float for identical text, which made encoded values depend on the
    row's position in the file.
    """
    for col in ["industry", "work_experience"]:
        kinds = {type(v).__name__ for v in raw[col].dropna()}
        assert kinds == {"str"}, f"{col} parsed as multiple types: {kinds}"


# --------------------------------------------------------------------- splitting
def test_splits_do_not_overlap_in_time(raw):
    train, val, hold_out = utils.data_split(raw)
    assert train.yearmo.max() < val.yearmo.min()
    assert val.yearmo.max() < hold_out.yearmo.min()


def test_splits_cover_all_rows(raw):
    train, val, hold_out = utils.data_split(raw)
    assert len(train) + len(val) + len(hold_out) == len(raw)


# --------------------------------------------------------------------- labelling
def test_label_matches_definition():
    df = pd.DataFrame(
        {
            "emi_1_dpd": [0, 30, 60, 0, 90],
            "emi_2_dpd": [0, 0, 0, 0, 0],
            "emi_3_dpd": [0, 0, 0, 90, 0],
            "emi_4_dpd": [90, 90, 90, 0, 0],  # outside the 3-month window
        }
    )
    processing.create_label(df, dpd=60, months=3)
    # row 1 peaks at 30 within the window; row 0's 90 is in EMI 4 and must not count
    assert df["label"].tolist() == [0, 0, 1, 1, 1]


def test_label_window_is_respected():
    df = pd.DataFrame({f"emi_{i}_dpd": [0] for i in range(1, 7)})
    df.loc[0, "emi_5_dpd"] = 90
    processing.create_label(df, dpd=60, months=3)
    assert df["label"].tolist() == [0]


# --------------------------------------------------------------- derived features
def test_derived_features_survive_zero_denominator():
    df = pd.DataFrame(
        {
            "interest_received": [10.0, 5.0],
            "total_payement": [100.0, 0.0],  # zero denominator
            "delinq_2yrs": [1, 2],
            "number_of_loans": [2, 0],  # zero denominator
        }
    )
    processing.derived_features(df)
    for col in ["interest_received_ratio", "total_payement_per_loan", "delinq_2yrs_ratio"]:
        assert np.isfinite(df[col]).all(), f"{col} contains inf/nan"
    assert df.loc[0, "interest_received_ratio"] == pytest.approx(0.1)
    assert df.loc[1, "interest_received_ratio"] == 0.0


# --------------------------------------------------------------------- encoding
def test_target_encoder_is_fitted_on_train_only(splits):
    """Encoding val/hold_out must not change if their labels change.

    This is the property that makes the encoder leak-free.
    """
    train, val, _ = splits
    id_cols = [
        "User_id",
        "emi_1_dpd",
        "emi_2_dpd",
        "emi_3_dpd",
        "emi_4_dpd",
        "emi_5_dpd",
        "emi_6_dpd",
        "max_dpd",
        "yearmo",
        "label",
    ]
    cat_cols = list(
        train.drop(columns=id_cols).select_dtypes(include=["category", "object"]).columns
    )

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
    enc = processing.categorical_encoding(params)
    enc.fit(train, cat_cols, "label")

    before = enc.transform(val.copy())[cat_cols]
    flipped = val.copy()
    flipped["label"] = 1 - flipped["label"]
    after = enc.transform(flipped)[cat_cols]

    pd.testing.assert_frame_equal(before, after)


def test_encoder_preserves_row_count_and_index(splits):
    train, val, _ = splits
    id_cols = [
        "User_id",
        "emi_1_dpd",
        "emi_2_dpd",
        "emi_3_dpd",
        "emi_4_dpd",
        "emi_5_dpd",
        "emi_6_dpd",
        "max_dpd",
        "yearmo",
        "label",
    ]
    cat_cols = list(
        train.drop(columns=id_cols).select_dtypes(include=["category", "object"]).columns
    )
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
    enc = processing.categorical_encoding(params)
    enc.fit(train, cat_cols, "label")
    out = enc.transform(val.copy())
    assert len(out) == len(val)
    assert out.index.equals(val.index)
    assert not out[cat_cols].isna().any().any()


# --------------------------------------------------------------------- leakage
def test_label_is_not_a_feature():
    """The single most damaging bug: `label` left in the feature matrix."""
    assert "label" in training.id_cols, "label must be listed as a non-feature column"


def test_dpd_columns_are_not_features():
    """The label is derived from these, so they cannot be inputs."""
    for c in ["emi_1_dpd", "emi_2_dpd", "emi_3_dpd", "max_dpd"]:
        assert c in training.id_cols


@pytest.mark.skipif(
    not os.path.exists("output/feature_columns.json"), reason="requires a completed engine.py run"
)
def test_trained_model_has_no_target_leakage():
    with open("output/feature_columns.json") as f:
        features = json.load(f)
    forbidden = {"label", "max_dpd", "User_id", "yearmo"} | {f"emi_{i}_dpd" for i in range(1, 7)}
    assert not (set(features) & forbidden)


# --------------------------------------------------------------------- evaluation
def test_ks_and_auc_agree_on_a_perfect_model():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    m = ev.discrimination_metrics(y, p)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["gini"] == pytest.approx(1.0)
    assert m["ks"] == pytest.approx(1.0)


def test_psi_is_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.uniform(size=20000)
    assert ev.population_stability_index(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_detects_a_shift():
    rng = np.random.default_rng(0)
    a = rng.uniform(size=20000)
    b = rng.uniform(size=20000) ** 3  # heavily shifted toward 0
    assert ev.population_stability_index(a, b) > 0.25


def test_decile_table_is_monotone_and_complete():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=10000)
    y = (rng.uniform(size=10000) < p).astype(int)
    t = ev.decile_table(y, p)
    assert len(t) == 10
    assert t["n"].sum() == 10000
    assert t["bads"].sum() == y.sum()
    assert t["cum_bad_capture"].iloc[-1] == pytest.approx(1.0)
    # riskiest decile must have a higher bad rate than the safest
    assert t["bad_rate"].iloc[0] > t["bad_rate"].iloc[-1]


def test_approval_curve_is_monotone_in_bad_rate():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=5000)
    y = (rng.uniform(size=5000) < p).astype(int)
    c = ev.approval_curve(y, p, steps=10)
    assert c["approval_rate"].is_monotonic_increasing
    # accepting more applicants can only add risk to the book
    assert c["bad_rate_of_approved"].iloc[-1] >= c["bad_rate_of_approved"].iloc[0]
