"""Model evaluation utilities for the credit default risk model.

The original project performed all evaluation inside the exploratory notebook,
so a run of engine.py produced a model artefact with no recorded evidence of how
good it was. This module makes the evaluation part of the pipeline: every metric
here is computed from the saved model and written to output/ as data, not as a
picture in a notebook.

Metric choices follow standard credit-risk scorecard practice:
  ROC AUC / Gini   - rank ordering power
  KS               - maximum separation between good and bad cumulative curves
  PR AUC           - performance on the minority (defaulter) class
  PSI              - population stability of the score across time periods
  Brier / calib.   - whether the score can be read as a probability of default
  Decile table     - the form in which a credit policy team actually consumes it
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             auc, average_precision_score, brier_score_loss)


# --------------------------------------------------------------------------
# Discrimination
# --------------------------------------------------------------------------
def ks_statistic(y_true, y_score):
    """Kolmogorov-Smirnov statistic: max gap between TPR and FPR curves.

    Returns
    -------
    (ks, threshold) : Tuple[float, float]
    """
    fpr, tpr, thr = roc_curve(y_true, y_score)
    idx = int(np.argmax(tpr - fpr))
    return float(tpr[idx] - fpr[idx]), float(thr[idx])


def discrimination_metrics(y_true, y_score):
    """Full discrimination metric block for one dataset.

    Parameters
    ----------
    y_true : array-like of {0, 1}
    y_score : array-like of float (predicted P(default))

    Returns
    -------
    Dict
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    roc = roc_auc_score(y_true, y_score)
    ks, ks_thr = ks_statistic(y_true, y_score)

    pr, re, _ = precision_recall_curve(y_true, y_score)
    pr0, re0, _ = precision_recall_curve(1 - y_true, 1 - y_score)

    return {
        "n": int(len(y_true)),
        "n_default": int(y_true.sum()),
        "default_rate": float(y_true.mean()),
        "roc_auc": float(roc),
        "gini": float(2 * roc - 1),
        "ks": ks,
        "ks_threshold": ks_thr,
        "pr_auc_class1": float(auc(re, pr)),
        "avg_precision_class1": float(average_precision_score(y_true, y_score)),
        "pr_auc_class0": float(auc(re0, pr0)),
    }


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def calibration_metrics(y_true, y_score, n_bins=10):
    """Brier score plus expected/maximum calibration error.

    A model tuned with pos_bagging_fraction != neg_bagging_fraction no longer
    emits calibrated probabilities, so this is checked explicitly rather than
    assumed.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    bins = pd.qcut(y_score, n_bins, labels=False, duplicates="drop")
    tab = pd.DataFrame({"y": y_true, "p": y_score, "b": bins})
    grp = tab.groupby("b").agg(n=("y", "size"), actual=("y", "mean"),
                               predicted=("p", "mean"))
    gap = (grp["actual"] - grp["predicted"]).abs()

    return {
        "brier": float(brier_score_loss(y_true, y_score)),
        "expected_calibration_error": float((gap * grp["n"]).sum() / grp["n"].sum()),
        "max_calibration_error": float(gap.max()),
        "mean_predicted": float(y_score.mean()),
        "observed_rate": float(y_true.mean()),
        "calibration_ratio": float(y_score.mean() / y_true.mean()) if y_true.mean() else np.nan,
    }, grp.reset_index()


# --------------------------------------------------------------------------
# Stability
# --------------------------------------------------------------------------
def population_stability_index(expected, actual, n_bins=10):
    """PSI of `actual` scores against the `expected` (reference) scores.

    Conventional reading: < 0.10 stable, 0.10-0.25 moderate shift,
    > 0.25 significant shift requiring investigation.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)

    # floor at a small epsilon so empty bins do not produce infinities
    eps = 1e-6
    e = np.clip(e, eps, None)
    a = np.clip(a, eps, None)
    return float(((a - e) * np.log(a / e)).sum())


# --------------------------------------------------------------------------
# Business-facing view
# --------------------------------------------------------------------------
def decile_table(y_true, y_score, n_bins=10):
    """Risk-ranked decile table: the standard scorecard acceptance view.

    Decile 1 is the highest-risk band. `cum_bad_capture` answers "what share of
    all defaulters do I catch if I reject everything down to this band".
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    df = pd.DataFrame({"y": y_true, "p": y_score})
    df["decile"] = pd.qcut(df["p"].rank(method="first", ascending=False),
                           n_bins, labels=range(1, n_bins + 1)).astype(int)

    out = (df.groupby("decile")
             .agg(n=("y", "size"), bads=("y", "sum"),
                  min_score=("p", "min"), max_score=("p", "max"),
                  avg_score=("p", "mean"))
             .reset_index())
    out["goods"] = out["n"] - out["bads"]
    out["bad_rate"] = out["bads"] / out["n"]
    out["cum_bad_capture"] = out["bads"].cumsum() / out["bads"].sum()
    out["cum_pop"] = out["n"].cumsum() / out["n"].sum()
    out["lift"] = out["bad_rate"] / df["y"].mean()
    return out


def approval_curve(y_true, y_score, steps=20):
    """Bad-rate of the accepted book at varying approval rates.

    Applicants are accepted from the safest score upward, which is how a cutoff
    is actually chosen in production.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    order = np.argsort(y_score)          # safest first
    y_sorted = y_true[order]
    s_sorted = y_score[order]

    rows = []
    for rate in np.linspace(1.0 / steps, 1.0, steps):
        k = max(1, int(round(rate * len(y_sorted))))
        rows.append({
            "approval_rate": rate,
            "n_approved": k,
            "cutoff_score": float(s_sorted[k - 1]),
            "bad_rate_of_approved": float(y_sorted[:k].mean()),
            "bads_avoided": int(y_sorted.sum() - y_sorted[:k].sum()),
        })
    return pd.DataFrame(rows)
