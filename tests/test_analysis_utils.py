"""Regression tests for the notebook analysis library (`notebooks/utils.py`).

Each test here pins a defect that was found during the model review.

There used to be a second, byte-identical copy of this library under
`modular_code/lib/`, kept in sync by a test. The restructure removed the
duplicate, so there is now one copy and nothing to keep in sync.
"""

import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "notebooks"))

import utils  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 3000
    p = rng.uniform(0.01, 0.9, size=n)
    y = (rng.uniform(size=n) < p).astype(int)
    return y, p


# ------------------------------------------------------------------ #17
def test_column_helpers_use_public_api():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"], "c": [1, 2]})
    assert set(utils.numeric_columns(df)) == {"a", "c"}
    assert set(utils.categorical_columns(df)) == {"b"}


# ------------------------------------------------------------------ #15, #19
def test_window_roll_rate_does_not_mutate_input():
    df = pd.DataFrame(
        {
            "User_id": [1, 2, 3, 4],
            "max_dpd": [90, 60, 0, 90],
            "emi_1_dpd": [90, 0, 0, 0],
            "emi_2_dpd": [0, 60, 0, 0],
            "emi_3_dpd": [0, 0, 0, 90],
            "emi_4_dpd": [0, 0, 0, 0],
            "emi_5_dpd": [0, 0, 0, 0],
            "emi_6_dpd": [0, 0, 0, 0],
        }
    )
    before = df.columns.tolist()
    out = utils.window_roll_rate(df, 60)
    # the helper must not write a working column back into the caller's frame
    assert df.columns.tolist() == before
    assert list(out.columns) == ["first_default_emi", "users_count", "% of Users"]
    assert out["users_count"].sum() == 3


# ------------------------------------------------------------------ #25
def test_dpd_roll_rate_reports_roll_and_recovery():
    df = pd.DataFrame({"max_dpd": [0] * 50 + [30] * 30 + [60] * 5 + [90] * 15})
    out = utils.dpd_roll_rate(df)
    assert {"roll_rate", "recovery_rate"} <= set(out.columns)
    # 50 of 100 reach dpd30; 20 of those 50 reach dpd60 -> 40% roll, 60% recovery
    assert out.loc[1, "user_count"] == 50
    assert out.loc[2, "user_count"] == 20
    assert out.loc[2, "roll_rate"].startswith("40.0")
    assert out.loc[2, "recovery_rate"].startswith("60.0")


# ------------------------------------------------------------------ #14
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_metric_helpers_accept_any_number_of_datasets(toy, k):
    y, p = toy
    utils.roc_auc([y] * k, [p] * k, [f"d{i}" for i in range(k)])
    utils.pr_auc([y] * k, [p] * k, [f"d{i}" for i in range(k)])
    utils.roc_auc_curve([y] * k, [p] * k, [f"d{i}" for i in range(k)])
    utils.pr_auc_curve([y] * k, [p] * k, [f"d{i}" for i in range(k)])


def test_metric_helpers_reject_mismatched_lengths(toy):
    y, p = toy
    with pytest.raises(ValueError):
        utils.roc_auc([y, y], [p], ["a", "b"])
    with pytest.raises(ValueError):
        utils.roc_auc([y, y], [p, p], ["only_one"])
    with pytest.raises(ValueError):
        utils.roc_auc([], [], [])


def test_metric_helpers_default_names(toy, capsys):
    y, p = toy
    utils.roc_auc([y, y], [p, p])
    assert "dataset_0" in capsys.readouterr().out


# ------------------------------------------------------------------ #13
def test_class_rate_when_first_dataset_is_not_named_train(toy):
    """The original keyed bin creation on the literal name 'Train'."""
    y, p = toy
    utils.class_rate([y, y], [p, p], ["Validation", "Hold Out"])


def test_class_rate_original_call_still_works(toy):
    y, p = toy
    utils.class_rate([y, y, y], [p, p, p], ["Train", "Val", "Hold Out"])


# ------------------------------------------------------------------ #16
def test_score_distribution_runs_without_distplot(toy):
    """sns.distplot was removed in seaborn 0.14."""
    y, p = toy
    utils.score_distribution([y, y], [p, p], ["Train", "Val"])


# ------------------------------------------------------------------ #18
def test_cutoff_score_respects_the_target_default_rate(toy):
    y, p = toy
    target = 0.05
    cutoff = utils.cutoff_score(y, p, target)
    accepted = pd.DataFrame({"y": y, "p": p}).loc[lambda d: d["p"] <= cutoff]
    assert len(accepted) > 0
    assert accepted["y"].mean() <= target + 1e-9


def test_cutoff_score_denominator_counts_accepted_accounts():
    """Cumulative rate must divide by accounts accepted (index + 1), not index."""
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.arange(10) / 10.0
    cutoff = utils.cutoff_score(y, p, 0.11)
    accepted = pd.DataFrame({"y": y, "p": p}).loc[lambda d: d["p"] <= cutoff]
    # 1 bad in the first 10 accounts is exactly 10% - within a 11% appetite
    assert accepted["y"].mean() <= 0.11


# ------------------------------------------------------------------ #21
def test_plot_helpers_close_their_figures(toy):
    y, p = toy
    plt.close("all")
    utils.roc_auc_curve([y], [p], ["a"])
    utils.pr_auc_curve([y], [p], ["a"])
    utils.score_distribution([y], [p], ["a"])
    utils.class_rate([y], [p], ["a"])
    assert plt.get_fignums() == []


# ------------------------------------------------------------------ #20
def test_dead_imports_removed():
    with open(os.path.join(ROOT, "notebooks", "utils.py"), encoding="utf-8") as fh:
        src = fh.read()
    for dead in ["import bisect", "OneHotEncoder", "import random", "import datetime"]:
        assert dead not in src, f"unused import still present: {dead}"


# ------------------------------------------------------------------ #28
def test_analysis_library_has_exactly_one_copy():
    """The duplicate under modular_code/lib/ must not come back.

    Two copies of this library existed and had to be edited in lockstep. The
    restructure removed one; this fails if a second ever reappears.
    """
    copies = [
        p
        for p in glob.glob(os.path.join(ROOT, "**", "utils.py"), recursive=True)
        if os.sep + "ml_pipeline" + os.sep not in p
        and os.sep + "archive" + os.sep not in p
        and os.sep + ".venv" + os.sep not in p
    ]
    assert len(copies) == 1, f"expected one analysis library, found: {copies}"
