"""Probability calibration.

A LightGBM model tuned with `pos_bagging_fraction != neg_bagging_fraction`
resamples the classes at different rates, so its output no longer estimates
P(default) - it only ranks. The original project never checked this and quoted
the raw score as a probability.

Rank ordering (AUC, Gini, KS) is unchanged by any monotone calibration; what
changes is whether the number can be read as a probability, which is what an
expected-loss calculation and a pricing decision need.

The calibrator must be fitted on data the model was not trained on, otherwise it
simply learns the model's training-set optimism.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from .logging_utils import get_logger

logger = get_logger(__name__)


class Calibrator:
    """Monotone map from raw model score to calibrated P(default).

    Parameters
    ----------
    method : {"isotonic", "platt"}
        `isotonic` is non-parametric and flexible but needs a reasonable sample;
        `platt` (a logistic fit on the log-odds) is smoother and safer on small
        calibration sets.
    """

    def __init__(self, method="isotonic"):
        if method not in ("isotonic", "platt"):
            raise ValueError(f"method must be 'isotonic' or 'platt', got {method!r}")
        self.method = method
        self.model = None

    @staticmethod
    def _logit(p, eps=1e-6):
        p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
        return np.log(p / (1 - p))

    def fit(self, scores, y):
        """Fit on out-of-sample scores.

        Parameters
        ----------
        scores : array-like of float
        y : array-like of {0, 1}
        """
        scores = np.asarray(scores, dtype=float)
        y = np.asarray(y)

        if len(np.unique(y)) < 2:
            raise ValueError("calibration set contains only one class")

        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
            self.model.fit(scores, y)
        else:
            self.model = LogisticRegression(C=1e10, solver="lbfgs")
            self.model.fit(self._logit(scores).reshape(-1, 1), y)

        before = brier_score_loss(y, scores)
        after = brier_score_loss(y, self.transform(scores))
        logger.info(
            "calibration (%s) fitted on %d rows: in-sample Brier " "%.5f -> %.5f",
            self.method,
            len(y),
            before,
            after,
        )
        return self

    def transform(self, scores):
        """Map raw scores to calibrated probabilities."""
        if self.model is None:
            raise RuntimeError("Calibrator must be fitted before use")
        scores = np.asarray(scores, dtype=float)
        if self.method == "isotonic":
            return self.model.predict(scores)
        return self.model.predict_proba(self._logit(scores).reshape(-1, 1))[:, 1]

    __call__ = transform


def calibration_report(y, raw, calibrated, n_bins=10):
    """Before/after calibration comparison.

    Confirms that rank ordering is preserved and quantifies the improvement.

    Returns
    -------
    Dict
    """
    y = np.asarray(y, dtype=float)

    def block(p):
        p = np.asarray(p, dtype=float)
        # expected calibration error over equal-count bins
        order = np.argsort(p)
        gaps, weights = [], []
        for chunk in np.array_split(order, n_bins):
            if len(chunk) == 0:
                continue
            gaps.append(abs(y[chunk].mean() - p[chunk].mean()))
            weights.append(len(chunk))
        gaps, weights = np.asarray(gaps), np.asarray(weights, dtype=float)
        return {
            "brier": float(brier_score_loss(y, p)),
            "auc": float(roc_auc_score(y, p)),
            "mean_predicted": float(p.mean()),
            "observed_rate": float(y.mean()),
            "calibration_ratio": float(p.mean() / y.mean()) if y.mean() else np.nan,
            "expected_calibration_error": float((gaps * weights).sum() / weights.sum()),
            "max_calibration_error": float(gaps.max()),
        }

    before, after = block(raw), block(calibrated)

    # Isotonic regression is monotone but not *strictly* monotone: it maps
    # ranges of scores onto flat steps, and tied scores inside a step can no
    # longer be ordered. That costs a little AUC. Platt scaling is strictly
    # monotone and leaves AUC untouched. Report the size of the effect rather
    # than a pass/fail, and flag it only if it is material.
    auc_delta = after["auc"] - before["auc"]
    return {
        "before": before,
        "after": after,
        "auc_delta": auc_delta,
        "auc_materially_changed": bool(abs(auc_delta) > 0.005),
        "brier_improvement": before["brier"] - after["brier"],
        "ece_improvement": (
            before["expected_calibration_error"] - after["expected_calibration_error"]
        ),
    }
