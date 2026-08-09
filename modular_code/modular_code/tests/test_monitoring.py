"""Tests for production monitoring.

A monitor that stays green on stable data has proved nothing. Most of these
tests inject a specific failure and assert that the corresponding check fires.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_pipeline.monitoring import (  # noqa: E402
    ALERT,
    OK,
    SKIPPED,
    WARN,
    build_baseline,
    categorical_shares,
    load_baseline,
    numeric_shares,
    open_ended,
    placeholder_rate,
    psi_from_shares,
    quantile_edges,
    run_monitoring,
    save_baseline,
)

FEATURES = ["total_income", "employment_type", "delinq_2yrs"]


def make_population(
    n=4000, seed=0, income_scale=1.0, salaried_share=0.6, placeholder="0", pd_shift=1.0
):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "total_income": rng.lognormal(11, 0.4, size=n) * income_scale,
            "employment_type": rng.choice(
                ["Salaried", "Self - Employeed"], size=n, p=[salaried_share, 1 - salaried_share]
            ),
            "delinq_2yrs": rng.poisson(0.3, size=n),
            "industry": rng.choice([placeholder, "abc", "def"], size=n, p=[0.8, 0.1, 0.1]),
        }
    )
    scores = np.clip(rng.beta(2, 20, size=n) * pd_shift, 1e-4, 0.999)
    labels = (rng.uniform(size=n) < scores).astype(int)
    return df, scores, labels


def status_of(report, name):
    return next(c.status for c in report.checks if c.name == name)


def value_of(report, name):
    return next(c.value for c in report.checks if c.name == name)


# ============================================================ building blocks
def test_psi_is_zero_for_identical_shares():
    s = np.array([0.2, 0.3, 0.5])
    assert psi_from_shares(s, s) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_divergence():
    base = np.array([0.34, 0.33, 0.33])
    mild = psi_from_shares(base, np.array([0.40, 0.30, 0.30]))
    wild = psi_from_shares(base, np.array([0.90, 0.05, 0.05]))
    assert 0 < mild < wild


def test_quantile_edges_are_sorted_and_finite():
    e = quantile_edges(np.random.default_rng(0).normal(size=1000), 10)
    assert np.all(np.diff(e) > 0)
    assert np.isfinite(e).all()


def test_quantile_edges_handle_a_constant_column():
    e = quantile_edges(np.full(100, 5.0), 10)
    assert len(e) >= 2 and np.all(np.diff(e) > 0)


def test_quantile_edges_handle_an_empty_column():
    assert len(quantile_edges(np.array([]), 10)) == 2


def test_open_ended_opens_both_tails_in_the_right_direction():
    """Regression: serialising both infinities as None made the last edge -inf,
    which broke bin monotonicity on reload."""
    e = open_ended([1.0, 2.0, 3.0])
    assert e[0] == -np.inf
    assert e[-1] == np.inf
    assert np.all(np.diff(e) > 0)


def test_numeric_shares_sum_to_one():
    v = np.random.default_rng(0).normal(size=500)
    shares = numeric_shares(v, open_ended(quantile_edges(v, 10)))
    assert shares.sum() == pytest.approx(1.0)


def test_numeric_shares_capture_out_of_range_values():
    edges = open_ended([0.0, 1.0, 2.0])
    shares = numeric_shares([-99, 0.5, 99], edges)
    assert shares.sum() == pytest.approx(1.0)


def test_categorical_shares_pool_unseen_into_other():
    shares = categorical_shares(["a", "a", "zzz"], ["a", "b"])
    assert shares[0] == pytest.approx(2 / 3)
    assert shares[-1] == pytest.approx(1 / 3)  # __OTHER__


def test_placeholder_rate():
    assert placeholder_rate(["0", "0.0", "real", "real"]) == pytest.approx(0.5)
    assert placeholder_rate(["a", "b"]) == 0.0


# ==================================================================== baseline
def test_baseline_records_scores_features_and_outcomes():
    df, scores, labels = make_population()
    b = build_baseline(df, FEATURES, scores, labels)
    assert b["n_rows"] == len(df)
    assert set(b["feature_spec"]) == set(FEATURES)
    assert b["observed_default_rate"] == pytest.approx(labels.mean())
    assert "deciles" in b and "discrimination" in b


def test_baseline_notes_derived_features_it_cannot_track():
    df, scores, labels = make_population()
    b = build_baseline(df, [*FEATURES, "some_derived_ratio"], scores, labels)
    assert b["derived_features_not_directly_monitored"] == ["some_derived_ratio"]


def test_baseline_without_labels_omits_outcome_sections():
    df, scores, _ = make_population()
    b = build_baseline(df, FEATURES, scores)
    assert "observed_default_rate" not in b and "deciles" not in b


def test_baseline_survives_a_json_round_trip(tmp_path):
    """Regression: the stored bin edges must still be usable after reload."""
    df, scores, labels = make_population()
    b = build_baseline(df, FEATURES, scores, labels)
    p = tmp_path / "baseline.json"
    save_baseline(b, str(p))
    reloaded = load_baseline(str(p))

    report = run_monitoring(reloaded, df, scores, labels)
    assert status_of(report, "score_psi") == OK
    assert value_of(report, "score_psi") == pytest.approx(0.0, abs=1e-6)


def test_baseline_is_json_serialisable():
    df, scores, labels = make_population()
    json.dumps(build_baseline(df, FEATURES, scores, labels))


# ================================================================ no drift
def test_identical_population_is_all_clear():
    df, scores, labels = make_population()
    b = build_baseline(df, FEATURES, scores, labels)
    report = run_monitoring(b, df, scores, labels)
    assert report.status == OK
    assert report.alerts == []


# =============================================================== injected drift
def test_score_drift_raises_an_alert():
    df, scores, labels = make_population()
    b = build_baseline(df, FEATURES, scores, labels)
    drifted = np.clip(scores * 4, 1e-4, 0.999)
    report = run_monitoring(b, df, drifted, labels)
    assert status_of(report, "score_psi") == ALERT
    assert report.status == ALERT


def test_numeric_feature_drift_raises_an_alert():
    df, scores, labels = make_population()
    b = build_baseline(df, FEATURES, scores, labels)
    shifted, _, _ = make_population(seed=1, income_scale=6.0)
    report = run_monitoring(b, shifted, scores, labels)
    assert status_of(report, "feature_psi::total_income") == ALERT


def test_categorical_feature_drift_raises_an_alert():
    df, scores, labels = make_population()
    b = build_baseline(df, FEATURES, scores, labels)
    shifted, _, _ = make_population(seed=1, salaried_share=0.05)
    report = run_monitoring(b, shifted, scores, labels)
    assert status_of(report, "feature_psi::employment_type") == ALERT


def test_a_stable_feature_stays_ok_when_another_drifts():
    """Drift must be attributed to the right feature, not smeared across all."""
    df, scores, labels = make_population()
    b = build_baseline(df, FEATURES, scores, labels)
    shifted, _, _ = make_population(seed=1, income_scale=6.0)
    report = run_monitoring(b, shifted, scores, labels)
    assert status_of(report, "feature_psi::delinq_2yrs") == OK


def test_missing_feature_raises_an_alert():
    df, scores, labels = make_population()
    b = build_baseline(df, FEATURES, scores, labels)
    report = run_monitoring(b, df.drop(columns=["total_income"]), scores, labels)
    assert status_of(report, "feature_psi::total_income") == ALERT


def test_changed_placeholder_convention_raises_an_alert():
    """The H1 defect recurring upstream: 'not captured' written a new way.

    PSI on the encoded value would not see this, because the encoder maps an
    unseen category to the prior.
    """
    df, scores, labels = make_population(placeholder="0")
    b = build_baseline(df, [*FEATURES, "industry"], scores, labels)
    changed, _, _ = make_population(seed=0, placeholder="NOT_CAPTURED")
    report = run_monitoring(b, changed, scores, labels)
    assert status_of(report, "placeholder_rate::industry") == ALERT


def test_calibration_drift_raises_an_alert():
    df, scores, labels = make_population()
    b = build_baseline(df, FEATURES, scores, labels)
    # the book goes bad: same scores, twice the defaults
    worse = labels.copy()
    flip = np.flatnonzero(worse == 0)[: int(0.4 * len(worse))]
    worse[flip] = 1
    report = run_monitoring(b, df, scores, worse)
    assert status_of(report, "calibration_ratio") == ALERT


def test_discrimination_decay_raises_an_alert():
    df, scores, labels = make_population()
    b = build_baseline(df, FEATURES, scores, labels)
    random_labels = np.random.default_rng(7).integers(0, 2, size=len(labels))
    report = run_monitoring(b, df, scores, random_labels)
    assert status_of(report, "discrimination_gini") == ALERT


# ============================================================ immature cohorts
def test_outcome_checks_are_reported_as_skipped_not_omitted():
    """A report that quietly drops half its checks is worse than none."""
    df, scores, labels = make_population()
    b = build_baseline(df, FEATURES, scores, labels)
    report = run_monitoring(b, df, scores, labels=None)

    for name in ("calibration_ratio", "discrimination_gini", "decile_bad_rate_drift"):
        assert status_of(report, name) == SKIPPED
    assert not report.labels_available
    assert "three EMIs" in next(c.message for c in report.checks if c.name == "calibration_ratio")


def test_input_checks_still_run_without_labels():
    df, scores, labels = make_population()
    b = build_baseline(df, FEATURES, scores, labels)
    report = run_monitoring(b, df, np.clip(scores * 4, 1e-4, 0.999), labels=None)
    assert status_of(report, "score_psi") == ALERT


def test_single_class_labels_skip_discrimination():
    df, scores, labels = make_population()
    b = build_baseline(df, FEATURES, scores, labels)
    report = run_monitoring(b, df, scores, np.zeros(len(df), dtype=int))
    assert status_of(report, "discrimination_gini") == SKIPPED


# ==================================================================== report
def test_overall_status_is_the_worst_check():
    df, scores, labels = make_population()
    b = build_baseline(df, FEATURES, scores, labels)
    assert run_monitoring(b, df, scores, labels).status == OK
    assert run_monitoring(b, df, np.clip(scores * 4, 1e-4, 0.999), labels).status == ALERT


def test_skipped_checks_do_not_mask_an_ok_status():
    df, scores, labels = make_population()
    b = build_baseline(df, FEATURES, scores, labels)
    assert run_monitoring(b, df, scores, labels=None).status == OK


def test_report_serialises_and_frames():
    df, scores, labels = make_population()
    b = build_baseline(df, FEATURES, scores, labels)
    report = run_monitoring(b, df, scores, labels, period="202205")
    json.dumps(report.to_dict())
    frame = report.to_frame()
    assert set(frame.columns) == {"check", "status", "value", "threshold", "message"}
    assert report.to_dict()["period"] == "202205"


def test_thresholds_are_overridable():
    df, scores, labels = make_population()
    b = build_baseline(df, FEATURES, scores, labels)
    mild = np.clip(scores * 1.5, 1e-4, 0.999)
    assert (
        status_of(
            run_monitoring(b, df, mild, labels, thresholds={"score_psi_alert": 99}), "score_psi"
        )
        != ALERT
    )
    assert (
        status_of(
            run_monitoring(
                b, df, mild, labels, thresholds={"score_psi_warn": 0.0, "score_psi_alert": 1e-9}
            ),
            "score_psi",
        )
        == ALERT
    )


def test_warning_status_exists_between_ok_and_alert():
    df, scores, labels = make_population()
    b = build_baseline(df, FEATURES, scores, labels)
    report = run_monitoring(
        b,
        df,
        np.clip(scores * 4, 1e-4, 0.999),
        labels,
        thresholds={"score_psi_warn": 0.01, "score_psi_alert": 99},
    )
    assert status_of(report, "score_psi") == WARN
