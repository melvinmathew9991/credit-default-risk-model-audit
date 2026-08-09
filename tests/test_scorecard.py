"""Tests for the decisioning layer: points, reason codes and cutoff policy.

Reason codes carry a regulatory obligation, so the properties that make them
correct - that they reflect the direction of risk, that they are never empty on
a decline, and that every model feature has applicant-facing text - are asserted
rather than eyeballed.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_pipeline.scorecard import (  # noqa: E402
    REASON_CODES,
    ScoreScaler,
    choose_cutoff,
    expected_loss,
    policy_table,
    reason_code_summary,
    reason_codes,
    unmapped_features,
)


# ============================================================ points scaling
def test_base_odds_map_to_base_score():
    s = ScoreScaler(pdo=20, base_score=600, base_odds=50)
    assert s.to_points(1 / 51) == pytest.approx(600.0, abs=1e-6)


def test_pdo_is_the_points_that_double_the_odds():
    s = ScoreScaler(pdo=20, base_score=600, base_odds=50, score_range=None)
    at_50 = s.to_points(1 / 51)
    at_100 = s.to_points(1 / 101)  # odds doubled
    assert at_100 - at_50 == pytest.approx(20.0, abs=1e-6)


@pytest.mark.parametrize("pdo", [10, 20, 40])
def test_pdo_is_configurable(pdo):
    s = ScoreScaler(pdo=pdo, score_range=None)
    assert s.to_points(1 / 101) - s.to_points(1 / 51) == pytest.approx(pdo, abs=1e-6)


def test_higher_score_means_lower_risk():
    s = ScoreScaler()
    pts = s.to_points([0.01, 0.05, 0.20, 0.50])
    assert list(pts) == sorted(pts, reverse=True)


def test_points_and_pd_round_trip():
    s = ScoreScaler(score_range=None)
    p = np.array([0.01, 0.05, 0.1, 0.3, 0.6])
    assert s.to_pd(s.to_points(p)) == pytest.approx(p, rel=1e-6)


def test_score_range_is_clamped():
    s = ScoreScaler(score_range=(300, 850))
    assert s.to_points(1e-9) == 850
    assert s.to_points(1 - 1e-9) == 300


def test_extreme_probabilities_do_not_produce_infinities():
    s = ScoreScaler(score_range=None)
    assert np.isfinite(s.to_points([0.0, 1.0])).all()


@pytest.mark.parametrize("kwargs", [{"pdo": 0}, {"pdo": -5}, {"base_odds": 0}])
def test_invalid_scaling_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ScoreScaler(**kwargs)


# =============================================================== reason codes
@pytest.fixture
def shap_matrix():
    # 3 applicants x 4 features
    return np.array(
        [
            [0.9, 0.4, -0.2, 0.1],  # feature 0 dominates
            [-0.5, 0.8, 0.7, -0.1],  # features 1 then 2
            [-0.3, -0.2, -0.1, -0.4],  # nothing pushes toward default
        ]
    ), ["total_income", "delinq_2yrs", "home_type", "role"]


def test_reasons_are_ranked_by_contribution(shap_matrix):
    sv, names = shap_matrix
    r = reason_codes(sv, names, top_n=2)
    assert r.loc[0, "reason_1"] == REASON_CODES["total_income"][1]
    assert r.loc[0, "reason_2"] == REASON_CODES["delinq_2yrs"][1]
    assert r.loc[1, "reason_1"] == REASON_CODES["delinq_2yrs"][1]
    assert r.loc[1, "reason_2"] == REASON_CODES["home_type"][1]


def test_only_features_pushing_toward_default_are_cited(shap_matrix):
    """A feature that helped the applicant is not a reason for declining them."""
    sv, names = shap_matrix
    r = reason_codes(sv, names, top_n=4)
    # row 2: every contribution is negative, so there is nothing to cite
    assert r.loc[2, "reason_1"] == ""
    # row 0 has three positive contributors (0.9, 0.4, 0.1); the fourth is -0.2
    assert r.loc[0, "reason_3"] != ""
    assert r.loc[0, "reason_4"] == ""


def test_reason_slots_are_capped_at_the_feature_count():
    """Asking for more reasons than features must not blow up."""
    r = reason_codes(np.array([[0.5, 0.2]]), ["total_income", "role"], top_n=4)
    assert [c for c in r if c.startswith("reason_")] == ["reason_1", "reason_2"]


def test_contributions_are_reported_and_ordered(shap_matrix):
    sv, names = shap_matrix
    r = reason_codes(sv, names, top_n=2)
    assert r.loc[0, "contribution_1"] > r.loc[0, "contribution_2"]
    assert np.isnan(r.loc[2, "contribution_1"])


def test_codes_accompany_every_reason(shap_matrix):
    sv, names = shap_matrix
    r = reason_codes(sv, names, top_n=2)
    for slot in (1, 2):
        cited = r[f"reason_{slot}"] != ""
        assert (r.loc[cited, f"code_{slot}"] != "").all()


def test_top_n_is_respected(shap_matrix):
    sv, names = shap_matrix
    assert len([c for c in reason_codes(sv, names, top_n=3) if c.startswith("reason_")]) == 3


def test_unknown_feature_gets_the_catch_all_code():
    sv = np.array([[1.0]])
    r = reason_codes(sv, ["some_new_feature"], top_n=1)
    assert r.loc[0, "code_1"] == "R99"


def test_shap_shape_is_validated():
    with pytest.raises(ValueError, match="2-D"):
        reason_codes(np.array([1.0, 2.0]), ["a", "b"])
    with pytest.raises(ValueError, match="feature names"):
        reason_codes(np.zeros((2, 3)), ["a", "b"])


def test_every_v2_model_feature_has_applicant_facing_text():
    """A decline explained as 'other information' is not an acceptable reason."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output_v2",
        "feature_columns_v2.json",
    )
    if not os.path.exists(path):
        pytest.skip("requires a completed engine_v2.py run")
    import json

    with open(path) as fh:
        features = json.load(fh)
    assert unmapped_features(features) == []


def test_reason_summary_counts_citations(shap_matrix):
    sv, names = shap_matrix
    s = reason_code_summary(reason_codes(sv, names, top_n=2))
    assert s["times_cited"].sum() == 4  # rows 0 and 1 cite two each
    assert s["share_of_citations"].sum() == pytest.approx(1.0)


def test_reason_summary_handles_no_declines():
    sv = np.array([[-1.0, -2.0]])
    assert reason_code_summary(reason_codes(sv, ["total_income", "role"])).empty


# ============================================================== cutoff policy
@pytest.fixture
def scored():
    rng = np.random.default_rng(0)
    n = 5000
    p = rng.beta(2, 20, size=n)
    y = (rng.uniform(size=n) < p).astype(int)
    return y, p


def test_policy_table_is_monotone_in_approval(scored):
    y, p = scored
    t = policy_table(y, p, steps=10)
    assert t["approval_rate"].is_monotonic_increasing
    assert t["n_approved"].is_monotonic_increasing
    assert t["pd_cutoff"].is_monotonic_increasing


def test_accepting_more_applicants_worsens_the_book(scored):
    y, p = scored
    t = policy_table(y, p, steps=10)
    assert t["bad_rate_of_book"].iloc[-1] >= t["bad_rate_of_book"].iloc[0]
    assert t["bad_rate_of_book"].iloc[-1] == pytest.approx(y.mean())


def test_full_approval_avoids_nothing(scored):
    y, p = scored
    t = policy_table(y, p, steps=10)
    assert t["bads_declined"].iloc[-1] == 0
    assert t["bad_capture_rate"].iloc[-1] == pytest.approx(0.0)
    assert t["goods_declined"].iloc[-1] == 0


def test_bads_are_conserved(scored):
    y, p = scored
    t = policy_table(y, p, steps=10)
    assert (t["bads_approved"] + t["bads_declined"] == y.sum()).all()


def test_score_cutoffs_are_added_when_a_scaler_is_supplied(scored):
    y, p = scored
    t = policy_table(y, p, scaler=ScoreScaler(), steps=5)
    assert "score_cutoff" in t
    # a looser PD cutoff is a lower score cutoff
    assert t["score_cutoff"].is_monotonic_decreasing


def test_choose_cutoff_respects_the_appetite(scored):
    y, p = scored
    t = policy_table(y, p, steps=20)
    pick = choose_cutoff(t, 0.05)
    assert pick is not None
    assert pick["bad_rate_of_book"] <= 0.05


def test_choose_cutoff_takes_the_largest_book_within_appetite(scored):
    y, p = scored
    t = policy_table(y, p, steps=20)
    pick = choose_cutoff(t, 0.05)
    assert pick["approval_rate"] == t[t.bad_rate_of_book <= 0.05]["approval_rate"].max()


def test_choose_cutoff_returns_none_when_impossible(scored):
    y, p = scored
    assert choose_cutoff(policy_table(y, p, steps=10), 0.0) is None


# =============================================================== expected loss
def test_expected_loss_is_the_product():
    assert expected_loss(0.1, 10000, lgd=0.5) == pytest.approx(500.0)


def test_expected_loss_is_vectorised():
    out = expected_loss([0.1, 0.2], [1000, 2000], lgd=0.5)
    assert out == pytest.approx([50.0, 200.0])


@pytest.mark.parametrize("lgd", [-0.1, 1.5])
def test_invalid_lgd_is_rejected(lgd):
    with pytest.raises(ValueError, match="lgd"):
        expected_loss(0.1, 1000, lgd=lgd)
