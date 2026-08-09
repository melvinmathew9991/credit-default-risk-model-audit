"""Production monitoring for the credit risk model.

`MODEL_CARD.md` section 8 specifies a monitoring plan. This implements it.

The central constraint is timing: a first-payment-default label needs three EMIs
to mature, so for roughly three months after an application is scored there is
no outcome to compare against. Monitoring therefore splits in two:

    input and score checks    available immediately, every month
    outcome checks            available only once the cohort has matured

`run_monitoring` runs whichever it can and says plainly which it skipped. A
monitoring report that silently omits half its checks is worse than none.

Thresholds come from the model card and are overridable, because appetite is a
business decision:

    score PSI            > 0.10 investigate, > 0.25 escalate
    feature PSI          > 0.25 on any live feature
    observed / predicted  outside 0.80 - 1.25
    Gini                 below 0.20
    placeholder rate     any material change in the upstream encoding
"""

import datetime
import json

import numpy as np
import pandas as pd

from .data_contract import PLACEHOLDER_TOKENS
from .evaluation import decile_table, discrimination_metrics
from .logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_THRESHOLDS = {
    "score_psi_warn": 0.10,
    "score_psi_alert": 0.25,
    "feature_psi_warn": 0.10,
    "feature_psi_alert": 0.25,
    "calibration_ratio_low": 0.80,
    "calibration_ratio_high": 1.25,
    "gini_floor": 0.20,
    "placeholder_rate_delta": 0.05,
    "band_share_psi_alert": 0.25,
}

OK, WARN, ALERT, SKIPPED = "OK", "WARN", "ALERT", "SKIPPED"
#: Worst-first, so a report's overall status is just max() over these.
SEVERITY_ORDER = {SKIPPED: 0, OK: 1, WARN: 2, ALERT: 3}


class Check:
    """One monitoring check and its verdict."""

    def __init__(self, name, status, value=None, threshold=None, message=""):
        self.name = name
        self.status = status
        self.value = value
        self.threshold = threshold
        self.message = message

    def to_dict(self):
        return {
            "check": self.name,
            "status": self.status,
            "value": self.value,
            "threshold": self.threshold,
            "message": self.message,
        }

    def __repr__(self):
        v = "" if self.value is None else f" ({self.value:.4f})"
        return f"[{self.status}] {self.name}{v}: {self.message}"


class MonitoringReport:
    def __init__(self, checks, period=None, n_rows=0, labels_available=False):
        self.checks = checks
        self.period = period
        self.n_rows = n_rows
        self.labels_available = labels_available
        self.created_utc = (
            datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
        )

    @property
    def status(self):
        if not self.checks:
            return SKIPPED
        return max((c.status for c in self.checks), key=lambda s: SEVERITY_ORDER[s])

    @property
    def alerts(self):
        return [c for c in self.checks if c.status == ALERT]

    @property
    def warnings(self):
        return [c for c in self.checks if c.status == WARN]

    def to_frame(self):
        return pd.DataFrame([c.to_dict() for c in self.checks])

    def to_dict(self):
        return {
            "created_utc": self.created_utc,
            "period": self.period,
            "n_rows": self.n_rows,
            "labels_available": self.labels_available,
            "overall_status": self.status,
            "checks": [c.to_dict() for c in self.checks],
        }

    def log(self):
        for c in self.checks:
            if c.status == ALERT:
                logger.error("monitoring - %s", c)
            elif c.status == WARN:
                logger.warning("monitoring - %s", c)
            else:
                logger.info("monitoring - %s", c)
        return self

    def __repr__(self):
        return (
            f"MonitoringReport(period={self.period}, status={self.status}, "
            f"{len(self.alerts)} alert(s), {len(self.warnings)} warning(s))"
        )


# ===========================================================================
# Population stability
# ===========================================================================
def quantile_edges(values, n_bins):
    """Finite quantile bin edges, deduplicated.

    Stored finite and made open-ended only at use time - see `open_ended`.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.array([0.0, 1.0])
    edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 2:
        edges = np.array([edges[0] - 1.0, edges[0] + 1.0])
    return edges


def open_ended(edges):
    """Copy of `edges` with the outer bins opened to +/- infinity.

    Applied at use time rather than baked into the stored baseline: JSON has no
    representation for -inf vs +inf that survives a round trip cleanly, and
    collapsing both to the same sentinel silently breaks bin monotonicity.
    """
    e = np.array(edges, dtype=float).copy()
    e[0], e[-1] = -np.inf, np.inf
    return e


def psi_from_shares(expected, actual, eps=1e-6):
    """PSI between two share vectors that are already aligned."""
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    return float(((a - e) * np.log(a / e)).sum())


def numeric_shares(values, edges):
    """Share of `values` falling in each bin defined by `edges`."""
    counts = np.histogram(np.asarray(values, dtype=float), bins=edges)[0]
    total = counts.sum()
    return (counts / total) if total else np.zeros_like(counts, dtype=float)


def categorical_shares(values, categories):
    """Share of `values` in each of `categories`, plus an `__OTHER__` bucket.

    Unseen categories are pooled rather than dropped: a wave of new categories
    is exactly the kind of upstream change monitoring should catch.
    """
    s = pd.Series(values).astype("object").where(pd.notna(pd.Series(values)), "__MISSING__")
    counts = s.value_counts()
    total = counts.sum()
    if not total:
        return np.zeros(len(categories) + 1)
    shares = [counts.get(c, 0) / total for c in categories]
    shares.append(max(0.0, 1.0 - sum(shares)))  # __OTHER__
    return np.asarray(shares, dtype=float)


def placeholder_rate(values):
    """Share of values that look like a missing-value sentinel."""
    s = pd.Series(values).dropna()
    if s.empty:
        return 0.0
    tokens = s.astype(str).str.strip().str.lower()
    return float(tokens.isin(PLACEHOLDER_TOKENS).mean())


# ===========================================================================
# Baseline
# ===========================================================================
def build_baseline(raw, features, pd_scores, labels=None, n_bins=10, model_version=None):
    """Snapshot the training population, against which later months are compared.

    Parameters
    ----------
    raw : DataFrame
        The raw training frame, before encoding - feature drift is more
        interpretable on original values than on target-encoded ones.
    features : list of str
    pd_scores : array-like
        Calibrated PDs the model assigned to `raw`.
    labels : array-like, optional
        Where available, records expected default rate and decile bad rates.
    n_bins : int
    model_version : str, optional

    Returns
    -------
    Dict, JSON-serialisable.
    """
    pd_scores = np.asarray(pd_scores, dtype=float)
    score_edges = quantile_edges(pd_scores, n_bins)

    feature_spec = {}
    not_raw = [f for f in features if f not in raw.columns]
    if not_raw:
        # Derived features are deterministic functions of raw columns, so
        # monitoring their inputs covers them. Say so rather than silently
        # tracking fewer features than the model uses.
        logger.info(
            "%d model feature(s) are derived, not raw, and are monitored via their "
            "inputs instead: %s",
            len(not_raw),
            not_raw,
        )

    for f in features:
        if f not in raw.columns:
            continue
        s = raw[f]
        if pd.api.types.is_numeric_dtype(s):
            clean = s.dropna()
            edges = quantile_edges(clean, n_bins)
            filled = s.fillna(clean.median() if len(clean) else 0)
            feature_spec[f] = {
                "kind": "numeric",
                "edges": [float(e) for e in edges],
                "shares": numeric_shares(filled, open_ended(edges)).tolist(),
                "null_rate": float(s.isna().mean()),
                "placeholder_rate": placeholder_rate(s),
            }
        else:
            # cap the tracked categories: a 9,000-category column would make an
            # unusable baseline, and the __OTHER__ bucket absorbs the tail
            top = s.astype("object").where(s.notna(), "__MISSING__").value_counts().head(20)
            cats = [str(c) for c in top.index]
            feature_spec[f] = {
                "kind": "categorical",
                "categories": cats,
                "shares": categorical_shares(s, cats).tolist(),
                "null_rate": float(s.isna().mean()),
                "placeholder_rate": placeholder_rate(s),
            }

    baseline = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "model_version": model_version,
        "n_rows": int(len(raw)),
        "features": list(features),
        "score": {
            "edges": [float(e) for e in score_edges],
            "shares": numeric_shares(pd_scores, open_ended(score_edges)).tolist(),
            "mean_pd": float(pd_scores.mean()),
        },
        "feature_spec": feature_spec,
        "derived_features_not_directly_monitored": not_raw,
    }

    if labels is not None:
        labels = np.asarray(labels)
        baseline["observed_default_rate"] = float(labels.mean())
        d = decile_table(labels, pd_scores)
        baseline["deciles"] = {
            "bad_rate": d["bad_rate"].tolist(),
            "min_score": d["min_score"].tolist(),
            "max_score": d["max_score"].tolist(),
        }
        baseline["discrimination"] = discrimination_metrics(labels, pd_scores)

    logger.info(
        "baseline built from %d rows, %d features, mean PD %.4f",
        len(raw),
        len(feature_spec),
        baseline["score"]["mean_pd"],
    )
    return baseline


def score_raw(raw, model, encoder, calibrator, features):
    """Calibrated PDs for a raw frame, using the training transformation.

    Shared by the training run and the monthly monitor so the two cannot drift.
    """
    from . import processing

    d = raw.copy()
    processing.clean_placeholders(d)
    processing.derived_features(d)
    d = encoder.transform(d)
    return calibrator.transform(model.predict(d[features]))


def build_baseline_from_raw(raw, model, encoder, calibrator, features, labels=None, **kwargs):
    """Build a baseline from an untouched raw frame.

    The distinction matters: feature statistics - especially `placeholder_rate` -
    must be measured on the data **as received**, not after
    `clean_placeholders` has run. A baseline built from the cleaned frame would
    record a 0% placeholder rate and then alert on every future month, because
    real incoming data still contains the placeholders.
    """
    scores = score_raw(raw, model, encoder, calibrator, features)
    return build_baseline(raw, features, scores, labels=labels, **kwargs)


def save_baseline(baseline, path):
    with open(path, "w") as f:
        json.dump(baseline, f, indent=2)
    logger.info("baseline written to %s", path)


def load_baseline(path):
    with open(path) as f:
        return json.load(f)


# ===========================================================================
# Checks
# ===========================================================================
def run_monitoring(baseline, raw, pd_scores, labels=None, period=None, thresholds=None):
    """Compare a new period against the baseline.

    Parameters
    ----------
    baseline : Dict
    raw : DataFrame
        New period's raw applications.
    pd_scores : array-like
        Calibrated PDs for those applications.
    labels : array-like, optional
        Matured outcomes. Omit when the cohort has not aged three EMIs yet; the
        outcome checks are then reported as SKIPPED rather than quietly dropped.
    period : str, optional
    thresholds : Dict, optional

    Returns
    -------
    MonitoringReport
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    pd_scores = np.asarray(pd_scores, dtype=float)
    checks = []

    # ---- score stability ------------------------------------------------
    edges = open_ended(baseline["score"]["edges"])
    score_psi = psi_from_shares(baseline["score"]["shares"], numeric_shares(pd_scores, edges))
    if score_psi > t["score_psi_alert"]:
        status, msg = ALERT, "score distribution has shifted materially - escalate"
    elif score_psi > t["score_psi_warn"]:
        status, msg = WARN, "score distribution is drifting - investigate"
    else:
        status, msg = OK, "score distribution is stable"
    checks.append(Check("score_psi", status, score_psi, t["score_psi_alert"], msg))

    # ---- mean PD --------------------------------------------------------
    base_mean = baseline["score"]["mean_pd"]
    shift = pd_scores.mean() / base_mean if base_mean else np.nan
    checks.append(
        Check(
            "mean_pd_shift",
            WARN if (np.isfinite(shift) and not 0.8 <= shift <= 1.25) else OK,
            float(shift),
            1.25,
            f"mean PD {pd_scores.mean():.4f} against a baseline of {base_mean:.4f}",
        )
    )

    # ---- feature stability ----------------------------------------------
    for f, spec in baseline["feature_spec"].items():
        if f not in raw.columns:
            checks.append(
                Check(f"feature_psi::{f}", ALERT, None, None, "feature is missing from the input")
            )
            continue

        s = raw[f]
        if spec["kind"] == "numeric":
            fe = open_ended(spec["edges"])
            clean = s.dropna()
            actual = numeric_shares(s.fillna(clean.median() if len(clean) else 0), fe)
        else:
            actual = categorical_shares(s, spec["categories"])

        value = psi_from_shares(spec["shares"], actual)
        if value > t["feature_psi_alert"]:
            status, msg = ALERT, "distribution has shifted materially"
        elif value > t["feature_psi_warn"]:
            status, msg = WARN, "distribution is drifting"
        else:
            status, msg = OK, "stable"
        checks.append(Check(f"feature_psi::{f}", status, value, t["feature_psi_alert"], msg))

        # ---- placeholder convention -------------------------------------
        # If the upstream export changes how it writes "not captured", the
        # encoder silently starts seeing a new category. This is the H1 defect
        # recurring, and it is invisible to a PSI on the encoded value.
        rate = placeholder_rate(s)
        delta = rate - spec["placeholder_rate"]
        if abs(delta) > t["placeholder_rate_delta"]:
            checks.append(
                Check(
                    f"placeholder_rate::{f}",
                    ALERT,
                    rate,
                    spec["placeholder_rate"],
                    f"placeholder share moved {delta:+.2%} (baseline {spec['placeholder_rate']:.2%}) - "
                    f"the upstream encoding convention may have changed",
                )
            )

    # ---- score band volumes ---------------------------------------------
    if "deciles" in baseline:
        band_edges = open_ended(np.unique([0.0] + baseline["deciles"]["max_score"]))
        expected = np.full(len(band_edges) - 1, 1.0 / (len(band_edges) - 1))
        band_psi = psi_from_shares(expected, numeric_shares(pd_scores, band_edges))
        checks.append(
            Check(
                "score_band_volume_psi",
                ALERT if band_psi > t["band_share_psi_alert"] else OK,
                band_psi,
                t["band_share_psi_alert"],
                "share of applicants per score band against an even baseline split",
            )
        )

    # ---- outcome checks --------------------------------------------------
    if labels is None:
        for name in ("calibration_ratio", "discrimination_gini", "decile_bad_rate_drift"):
            checks.append(
                Check(
                    name,
                    SKIPPED,
                    None,
                    None,
                    "no matured outcomes for this cohort yet - a first-payment-default "
                    "label needs three EMIs",
                )
            )
        return MonitoringReport(checks, period, len(raw), labels_available=False).log()

    labels = np.asarray(labels)
    observed = float(labels.mean())
    predicted = float(pd_scores.mean())
    ratio = predicted / observed if observed else np.nan
    in_band = (
        np.isfinite(ratio) and t["calibration_ratio_low"] <= ratio <= t["calibration_ratio_high"]
    )
    checks.append(
        Check(
            "calibration_ratio",
            OK if in_band else ALERT,
            float(ratio),
            t["calibration_ratio_high"],
            f"predicted {predicted:.4f} against observed {observed:.4f}"
            + ("" if in_band else " - the model is out of calibration, recalibrate"),
        )
    )

    if len(np.unique(labels)) < 2:
        checks.append(Check("discrimination_gini", SKIPPED, None, None, "only one class present"))
    else:
        disc = discrimination_metrics(labels, pd_scores)
        checks.append(
            Check(
                "discrimination_gini",
                ALERT if disc["gini"] < t["gini_floor"] else OK,
                disc["gini"],
                t["gini_floor"],
                f"AUC {disc['roc_auc']:.4f}, KS {disc['ks']:.4f}"
                + (
                    ""
                    if disc["gini"] >= t["gini_floor"]
                    else " - discrimination has decayed below the floor"
                ),
            )
        )

        if "deciles" in baseline:
            now = decile_table(labels, pd_scores)["bad_rate"].to_numpy()
            base = np.asarray(baseline["deciles"]["bad_rate"], dtype=float)
            n = min(len(now), len(base))
            drift = float(np.abs(now[:n] - base[:n]).max())
            checks.append(
                Check(
                    "decile_bad_rate_drift",
                    WARN if drift > 0.05 else OK,
                    drift,
                    0.05,
                    "largest absolute change in any decile's bad rate against baseline",
                )
            )

    return MonitoringReport(checks, period, len(raw), labels_available=True).log()
