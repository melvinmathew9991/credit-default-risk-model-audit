"""Turning a probability into a credit decision.

A calibrated PD is not yet something a credit function can use. Three things are
missing, and all three are built here:

**Points.** Credit policy is written in score points, not probabilities. The
standard transform is linear in the log-odds and parameterised by PDO - the
number of points that doubles the odds.

**Reason codes.** Where an applicant is declined, they are generally entitled to
be told the principal reasons (ECOA/FCRA in the US; fair-practice expectations
elsewhere). A score with no explanation cannot lawfully decline anybody, which
makes this a deployment blocker rather than a nice-to-have.

**A cutoff.** Choosing where to draw the line is a business decision, but it has
to be made against a table that shows what each cutoff does to approval volume
and to the bad rate of the accepted book.
"""

import numpy as np
import pandas as pd

from .logging_utils import get_logger

logger = get_logger(__name__)


# ===========================================================================
# Points scaling
# ===========================================================================
class ScoreScaler:
    """Linear map from probability of default to score points.

        score = offset + factor * ln(odds),   odds = (1 - pd) / pd

    so a **higher score means lower risk**, the usual convention.

    Parameters
    ----------
    pdo : float
        Points to Double the Odds. 20 is the common default.
    base_score : float
        Score at `base_odds`.
    base_odds : float
        Good:bad odds at `base_score`. 50 means 50 goods per bad.
    score_range : tuple, optional
        Clamp the output to (min, max). None to leave unbounded.

    Examples
    --------
    >>> s = ScoreScaler()
    >>> round(s.to_points(1 / 51), 1)      # odds of exactly 50:1
    600.0
    """

    def __init__(self, pdo=20.0, base_score=600.0, base_odds=50.0,
                 score_range=(300.0, 850.0)):
        if pdo <= 0:
            raise ValueError(f"pdo must be positive, got {pdo}")
        if base_odds <= 0:
            raise ValueError(f"base_odds must be positive, got {base_odds}")
        self.pdo = float(pdo)
        self.base_score = float(base_score)
        self.base_odds = float(base_odds)
        self.score_range = score_range
        self.factor = self.pdo / np.log(2)
        self.offset = self.base_score - self.factor * np.log(self.base_odds)

    def to_points(self, pd_, eps=1e-6):
        """Probability of default -> score points."""
        p = np.clip(np.asarray(pd_, dtype=float), eps, 1 - eps)
        pts = self.offset + self.factor * np.log((1 - p) / p)
        if self.score_range:
            pts = np.clip(pts, *self.score_range)
        return pts

    def to_pd(self, points):
        """Score points -> probability of default (inverse of `to_points`)."""
        pts = np.asarray(points, dtype=float)
        odds = np.exp((pts - self.offset) / self.factor)
        return 1.0 / (1.0 + odds)

    def describe(self):
        return {"pdo": self.pdo, "base_score": self.base_score,
                "base_odds": self.base_odds, "factor": self.factor,
                "offset": self.offset, "score_range": self.score_range}


# ===========================================================================
# Adverse action reason codes
# ===========================================================================
#: Feature -> (code, applicant-facing reason). Wording is deliberately about the
#: applicant's own record, never about a group they belong to.
REASON_CODES = {
    "tier_of_employment":  ("R01", "Employer classification"),
    "work_experience":     ("R02", "Length of employment history"),
    "total_income":        ("R03", "Income relative to amount requested"),
    "employment_type":     ("R04", "Type of employment"),
    "industry":            ("R05", "Employer industry"),
    "role":                ("R06", "Occupation"),
    "home_type":           ("R07", "Housing status"),
    "delinq_2yrs":         ("R08", "Delinquencies on prior credit obligations"),
    "delinq_2yrs_ratio":   ("R09", "Proportion of prior obligations delinquent"),
    "number_of_loans":     ("R10", "Number of prior credit obligations"),
    "dependents":          ("R11", "Number of dependents"),
    "has_social_profile":  ("R12", "Insufficient verifiable profile information"),
    "is_verified":         ("R13", "Profile information could not be verified"),
}

UNMAPPED = ("R99", "Other information in the application")


def reason_codes(shap_values, feature_names, top_n=4, min_contribution=1e-6):
    """Principal reasons a score was adverse, per applicant.

    Uses the SHAP decomposition of the model's log-odds output: for each row,
    the features with a **positive** contribution are the ones that pushed the
    applicant toward default, ranked by how much.

    Parameters
    ----------
    shap_values : ndarray, shape (n_rows, n_features)
        Class-1 SHAP values. Passing class-0 values inverts every reason, so
        the caller must select correctly.
    feature_names : list of str
    top_n : int
        How many reasons to return. Four is the usual regulatory maximum.
    min_contribution : float
        Ignore contributions below this - a reason worth ~0 is not a reason.

    Returns
    -------
    DataFrame
        One row per applicant, columns reason_1..reason_n, code_1..code_n and
        contribution_1..contribution_n.
    """
    shap_values = np.asarray(shap_values, dtype=float)
    if shap_values.ndim != 2:
        raise ValueError(f"expected a 2-D SHAP matrix, got shape {shap_values.shape}")
    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            f"SHAP matrix has {shap_values.shape[1]} columns but "
            f"{len(feature_names)} feature names were supplied")

    # You cannot cite more reasons than there are features. Without this the
    # slot loop indexes past the end of `order` and raises IndexError.
    n_slots = min(top_n, shap_values.shape[1])
    if n_slots < top_n:
        logger.warning("asked for %d reasons but the model has only %d features; "
                       "returning %d", top_n, shap_values.shape[1], n_slots)

    order = np.argsort(-shap_values, axis=1)[:, :n_slots]
    out = {}
    for slot in range(n_slots):
        idx = order[:, slot]
        contrib = shap_values[np.arange(len(shap_values)), idx]
        names = np.array(feature_names)[idx]
        mapped = [REASON_CODES.get(n, UNMAPPED) for n in names]
        blank = contrib <= min_contribution
        out[f"code_{slot + 1}"] = np.where(blank, "", [m[0] for m in mapped])
        out[f"reason_{slot + 1}"] = np.where(blank, "", [m[1] for m in mapped])
        out[f"contribution_{slot + 1}"] = np.where(blank, np.nan, contrib)
    return pd.DataFrame(out)


def unmapped_features(feature_names):
    """Model features with no applicant-facing reason text.

    Anything listed here would be explained to a declined applicant as "other
    information", which is not an acceptable adverse action reason.
    """
    return [f for f in feature_names if f not in REASON_CODES]


def reason_code_summary(reasons):
    """How often each reason is cited - the fair-lending review view.

    A reason appearing in nearly every decline usually means the cutoff is being
    driven by one feature, which is worth knowing before launch.
    """
    cols = [c for c in reasons.columns if c.startswith("reason_")]
    stacked = reasons[cols].to_numpy().ravel()
    stacked = stacked[(stacked != "") & (stacked != None)]  # noqa: E711
    if len(stacked) == 0:
        return pd.DataFrame(columns=["reason", "times_cited", "share_of_citations"])
    s = pd.Series(stacked).value_counts()
    return pd.DataFrame({"reason": s.index, "times_cited": s.to_numpy(),
                         "share_of_citations": s.to_numpy() / s.sum()})


# ===========================================================================
# Cutoff policy
# ===========================================================================
def policy_table(y_true, pd_scores, scaler=None, steps=20):
    """Approve from the safest applicant down; show what each cutoff buys.

    Parameters
    ----------
    y_true : array-like of {0, 1}
    pd_scores : array-like of float
        Calibrated probability of default.
    scaler : ScoreScaler, optional
        Adds the equivalent score-point cutoff.
    steps : int
        Number of approval rates to evaluate.

    Returns
    -------
    DataFrame
    """
    y = np.asarray(y_true)
    p = np.asarray(pd_scores, dtype=float)
    order = np.argsort(p)                     # safest first
    y_s, p_s = y[order], p[order]
    total_bads = int(y.sum())

    rows = []
    for rate in np.linspace(1.0 / steps, 1.0, steps):
        k = max(1, int(round(rate * len(y_s))))
        approved_bads = int(y_s[:k].sum())
        row = {
            "approval_rate": rate,
            "n_approved": k,
            "pd_cutoff": float(p_s[k - 1]),
            "bad_rate_of_book": float(y_s[:k].mean()),
            "bads_approved": approved_bads,
            "bads_declined": total_bads - approved_bads,
            "bad_capture_rate": (total_bads - approved_bads) / total_bads if total_bads else np.nan,
            "goods_declined": int((1 - y_s[k:]).sum()),
        }
        if scaler is not None:
            row["score_cutoff"] = float(scaler.to_points(p_s[k - 1]))
        rows.append(row)
    return pd.DataFrame(rows)


def choose_cutoff(policy, max_bad_rate):
    """Highest approval rate whose accepted book stays within `max_bad_rate`.

    Returns
    -------
    Series, or None if no cutoff satisfies the constraint.
    """
    ok = policy[policy["bad_rate_of_book"] <= max_bad_rate]
    if ok.empty:
        logger.warning("no cutoff achieves a book bad rate at or below %.4f; "
                       "the best available is %.4f",
                       max_bad_rate, policy["bad_rate_of_book"].min())
        return None
    return ok.loc[ok["approval_rate"].idxmax()]


def expected_loss(pd_scores, exposure, lgd=0.45):
    """Expected credit loss = PD x LGD x EAD.

    This model supplies PD only. LGD and EAD are assumptions the user must
    supply and own - the default 45% LGD is the Basel foundation-IRB value for
    unsecured retail exposures and is a placeholder, not an estimate from this
    data.
    """
    p = np.asarray(pd_scores, dtype=float)
    ead = np.asarray(exposure, dtype=float)
    if not 0 <= lgd <= 1:
        raise ValueError(f"lgd must be between 0 and 1, got {lgd}")
    return p * lgd * ead
