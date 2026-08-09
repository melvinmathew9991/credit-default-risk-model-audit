"""Tests for walk-forward validation, calibration and the WOE challenger."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_pipeline import woe  # noqa: E402
from ml_pipeline.calibration import Calibrator, calibration_report  # noqa: E402
from ml_pipeline.validation import (drop_leaked_users,  # noqa: E402
                                    expanding_window_folds, summarise_folds)


# ================================================================ validation
@pytest.fixture
def months_df():
    return pd.DataFrame({
        "yearmo": [202201] * 3 + [202202] * 3 + [202203] * 3 + [202204] * 3,
        "User_id": range(12),
    })


def test_folds_are_walk_forward(months_df):
    folds = list(expanding_window_folds(months_df, min_train_months=1))
    assert len(folds) == 3
    assert [f[2] for f in folds] == [202202, 202203, 202204]
    assert [f[1] for f in folds] == [[202201], [202201, 202202],
                                     [202201, 202202, 202203]]


def test_training_window_always_precedes_validation(months_df):
    for _, train_months, val_month, tr_idx, va_idx in expanding_window_folds(months_df):
        assert max(train_months) < val_month
        assert months_df.loc[tr_idx, "yearmo"].max() < months_df.loc[va_idx, "yearmo"].min()
        assert not set(tr_idx) & set(va_idx)


def test_folds_respect_min_train_months(months_df):
    folds = list(expanding_window_folds(months_df, min_train_months=2))
    assert [f[2] for f in folds] == [202203, 202204]


def test_folds_require_enough_months():
    df = pd.DataFrame({"yearmo": [202201, 202201]})
    with pytest.raises(ValueError, match="need more than"):
        list(expanding_window_folds(df, min_train_months=1))


def test_folds_require_the_time_column():
    with pytest.raises(KeyError, match="yearmo"):
        list(expanding_window_folds(pd.DataFrame({"a": [1]})))


def test_hold_out_month_can_be_excluded(months_df):
    folds = list(expanding_window_folds(months_df, months=[202201, 202202, 202203]))
    assert 202204 not in [f[2] for f in folds]


# --------------------------------------------------------------------- dedup
def test_repeat_customers_are_removed_from_the_later_split():
    train = pd.DataFrame({"User_id": [1, 2, 3]})
    later = pd.DataFrame({"User_id": [3, 4, 5]})
    out = drop_leaked_users(train, later)
    assert out["User_id"].tolist() == [4, 5]


def test_dedup_keeps_everything_when_there_is_no_overlap():
    train = pd.DataFrame({"User_id": [1, 2]})
    later = pd.DataFrame({"User_id": [3, 4]})
    assert len(drop_leaked_users(train, later)) == 2


def test_dedup_resets_the_index():
    train = pd.DataFrame({"User_id": [1]})
    later = pd.DataFrame({"User_id": [1, 2, 3]})
    assert drop_leaked_users(train, later).index.tolist() == [0, 1]


def test_summarise_folds():
    s = summarise_folds([0.6, 0.7, 0.8])
    assert s["n_folds"] == 3
    assert s["mean"] == pytest.approx(0.7)
    assert s["min"] == pytest.approx(0.6) and s["max"] == pytest.approx(0.8)


# =============================================================== calibration
@pytest.fixture
def miscalibrated():
    """Scores that rank well but are systematically too high."""
    rng = np.random.default_rng(0)
    n = 8000
    true_p = rng.uniform(0.01, 0.4, size=n)
    y = (rng.uniform(size=n) < true_p).astype(int)
    raw = np.clip(true_p * 2.2, 0, 0.999)      # inflated but monotone in true_p
    return y, raw


@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_calibration_reduces_the_error(miscalibrated, method):
    y, raw = miscalibrated
    cal = Calibrator(method).fit(raw, y)
    rep = calibration_report(y, raw, cal.transform(raw))
    assert rep["after"]["expected_calibration_error"] < rep["before"]["expected_calibration_error"]
    assert rep["brier_improvement"] > 0


@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_calibration_brings_the_mean_onto_the_observed_rate(miscalibrated, method):
    y, raw = miscalibrated
    cal = Calibrator(method).fit(raw, y)
    out = cal.transform(raw)
    assert abs(out.mean() - y.mean()) < abs(raw.mean() - y.mean())


def test_platt_preserves_rank_ordering_exactly(miscalibrated):
    """Platt is strictly monotone, so AUC must be unchanged."""
    y, raw = miscalibrated
    cal = Calibrator("platt").fit(raw, y)
    rep = calibration_report(y, raw, cal.transform(raw))
    assert rep["auc_delta"] == pytest.approx(0.0, abs=1e-9)


def test_isotonic_barely_moves_auc(miscalibrated):
    """Isotonic ties cost a little discrimination - it must stay immaterial."""
    y, raw = miscalibrated
    cal = Calibrator("isotonic").fit(raw, y)
    rep = calibration_report(y, raw, cal.transform(raw))
    assert not rep["auc_materially_changed"]


def test_calibrated_output_is_a_probability(miscalibrated):
    y, raw = miscalibrated
    out = Calibrator("isotonic").fit(raw, y).transform(raw)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_calibrator_must_be_fitted_first():
    with pytest.raises(RuntimeError, match="fitted"):
        Calibrator().transform([0.1, 0.2])


def test_calibrator_rejects_a_single_class():
    with pytest.raises(ValueError, match="one class"):
        Calibrator().fit([0.1, 0.2, 0.3], [0, 0, 0])


def test_unknown_calibration_method():
    with pytest.raises(ValueError, match="isotonic"):
        Calibrator("sigmoidish")


# ======================================================================= WOE
@pytest.fixture
def woe_frame():
    rng = np.random.default_rng(0)
    n = 4000
    strong = rng.choice(["risky", "safe"], size=n)
    noise = rng.choice(list("abcd"), size=n)
    p = np.where(strong == "risky", 0.4, 0.05)
    y = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame({"strong": strong, "noise": noise,
                         "num": rng.normal(size=n)}), y


def test_information_value_separates_signal_from_noise(woe_frame):
    X, y = woe_frame
    enc = woe.WOEEncoder().fit(X, y)
    iv = enc.iv_.copy()
    assert iv["strong"] > 0.3
    assert iv["noise"] < 0.02
    assert iv["num"] < 0.02


def test_woe_sign_follows_default_rate(woe_frame):
    X, y = woe_frame
    enc = woe.WOEEncoder().fit(X, y)
    m = enc.maps_["strong"]
    assert m["risky"] > 0 > m["safe"]


def test_iv_table_is_ranked_and_labelled(woe_frame):
    X, y = woe_frame
    t = woe.WOEEncoder().fit(X, y).iv_table()
    assert t.iloc[0]["feature"] == "strong"
    assert t["iv"].is_monotonic_decreasing
    assert str(t.iloc[0]["strength"]) in {"strong", "suspicious"}


def test_select_applies_the_iv_floor(woe_frame):
    X, y = woe_frame
    enc = woe.WOEEncoder().fit(X, y)
    assert enc.select(min_iv=0.02) == ["strong"]


def test_transform_shape_and_finiteness(woe_frame):
    X, y = woe_frame
    enc = woe.WOEEncoder().fit(X, y)
    out = enc.transform(X)
    assert out.shape == X.shape
    assert np.isfinite(out.to_numpy()).all()


def test_unseen_category_maps_to_neutral_evidence(woe_frame):
    X, y = woe_frame
    enc = woe.WOEEncoder().fit(X, y)
    new = pd.DataFrame({"strong": ["never_seen"], "noise": ["z"], "num": [0.0]})
    out = enc.transform(new)
    assert out["strong"].iloc[0] == 0.0


def test_missing_values_get_their_own_bin():
    X = pd.DataFrame({"f": ["a"] * 100 + ["b"] * 100 + [None] * 100})
    y = np.array([1] * 50 + [0] * 50 + [0] * 100 + [1] * 80 + [0] * 20)
    enc = woe.WOEEncoder(min_bin_frac=0.01).fit(X, y)
    assert woe.MISSING in enc.maps_["f"]


def test_woe_requires_both_classes():
    X = pd.DataFrame({"f": ["a", "b"]})
    with pytest.raises(ValueError, match="both classes"):
        woe.WOEEncoder().fit(X, [0, 0])


def test_logistic_challenger_learns_the_signal(woe_frame):
    X, y = woe_frame
    enc = woe.WOEEncoder().fit(X, y)
    W = enc.transform(X)
    model = woe.fit_logistic_scorecard(W[["strong"]], y)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, model.predict_proba(W[["strong"]])[:, 1]) > 0.7


def test_raw_coefficient_ranking_is_not_trustworthy(woe_frame):
    """A near-constant WOE column can attract a huge, meaningless coefficient.

    This documents *why* coefficient_table takes the WOE matrix: without it the
    ranking can put a pure noise feature first.
    """
    X, y = woe_frame
    W = woe.WOEEncoder().fit(X, y).transform(X)
    t = woe.coefficient_table(woe.fit_logistic_scorecard(W, y), W.columns)
    assert t["coefficient"].abs().is_monotonic_decreasing
    assert set(t.columns) == {"feature", "coefficient"}


def test_contribution_ranking_finds_the_real_driver(woe_frame):
    """|coefficient| * std(WOE) is how much the term actually moves log-odds."""
    X, y = woe_frame
    enc = woe.WOEEncoder().fit(X, y)
    W = enc.transform(X)
    model = woe.fit_logistic_scorecard(W, y)
    t = woe.coefficient_table(model, W.columns, woe_matrix=W)
    assert t.iloc[0]["feature"] == "strong"
    assert t["contribution"].is_monotonic_decreasing
    # the noise features barely move the score at all
    assert t.iloc[0]["contribution"] > 10 * t.iloc[1]["contribution"]


def test_bin_table_is_auditable(woe_frame):
    X, y = woe_frame
    enc = woe.WOEEncoder().fit(X, y)
    t = enc.bin_table("strong")
    assert set(t.columns) == {"bin", "woe"}
    assert len(t) == 2
