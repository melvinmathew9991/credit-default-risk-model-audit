"""Weight-of-evidence binning, information value, and a logistic challenger.

This is the incumbent method in credit scoring, and it is the benchmark a
gradient-boosted model has to beat before its extra opacity can be justified.
A regulator or credit committee can read a WOE scorecard line by line; they
cannot read 5,000 boosted trees.

Definitions, per bin `i`, with events = defaults:

    WOE_i = ln( (events_i / total_events) / (non_events_i / total_non_events) )
    IV    = sum_i ( events_i/total_events - non_events_i/total_non_events ) * WOE_i

Conventional IV reading: <0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium,
0.3-0.5 strong, >0.5 suspiciously strong - check for leakage.

Every statistic is computed on training data only and then applied unchanged to
later periods.
"""

import numpy as np
import pandas as pd

from .logging_utils import get_logger

logger = get_logger(__name__)

MISSING = "__MISSING__"


class WOEEncoder:
    """Bin features and replace them with their weight of evidence.

    Parameters
    ----------
    n_bins : int
        Target number of quantile bins for numeric features.
    min_bin_frac : float
        Bins smaller than this share of rows are merged into the neighbouring
        bin, so a WOE is never estimated from a handful of accounts.
    """

    def __init__(self, n_bins=10, min_bin_frac=0.02):
        self.n_bins = n_bins
        self.min_bin_frac = min_bin_frac
        self.maps_ = {}
        self.edges_ = {}
        self.iv_ = {}
        self.kinds_ = {}

    # ------------------------------------------------------------------
    def _bin_numeric(self, s, fit, col=None):
        if fit:
            finite = s.dropna()
            try:
                _, edges = pd.qcut(finite, self.n_bins, retbins=True, duplicates="drop")
            except (ValueError, IndexError):
                edges = np.array([finite.min(), finite.max()])
            edges = np.unique(edges).astype(float)
            if len(edges) < 2:
                edges = np.array([-np.inf, np.inf])
            edges[0], edges[-1] = -np.inf, np.inf
            self.edges_[col] = edges
        edges = self.edges_[col]
        out = pd.cut(s, bins=edges).astype(str)
        return out.where(s.notna(), MISSING)

    @staticmethod
    def _bin_categorical(s):
        return s.astype(str).where(s.notna(), MISSING)

    # ------------------------------------------------------------------
    def fit(self, X, y):
        """Learn bins and WOE values from training data.

        Parameters
        ----------
        X : DataFrame
        y : array-like of {0, 1}
        """
        y = pd.Series(np.asarray(y), index=X.index)
        total_events = float(y.sum())
        total_non = float(len(y) - total_events)
        if total_events == 0 or total_non == 0:
            raise ValueError("need both classes present to compute weight of evidence")

        for col in X.columns:
            s = X[col]
            numeric = pd.api.types.is_numeric_dtype(s)
            self.kinds_[col] = "numeric" if numeric else "categorical"
            binned = (
                self._bin_numeric(s, fit=True, col=col) if numeric else self._bin_categorical(s)
            )

            tab = pd.DataFrame({"bin": binned, "y": y}).groupby("bin")["y"].agg(["count", "sum"])
            tab = self._merge_small(tab, len(y))

            events = tab["sum"].astype(float)
            non_events = (tab["count"] - tab["sum"]).astype(float)

            # 0.5 continuity correction keeps WOE finite in a pure bin
            dist_e = (events + 0.5) / (total_events + 0.5 * len(tab))
            dist_n = (non_events + 0.5) / (total_non + 0.5 * len(tab))

            woe = np.log(dist_e / dist_n)
            self.maps_[col] = woe.to_dict()
            self.iv_[col] = float(((dist_e - dist_n) * woe).sum())

        logger.info(
            "WOE fitted on %d features; IV range %.4f - %.4f",
            len(self.maps_),
            min(self.iv_.values()),
            max(self.iv_.values()),
        )
        return self

    def _merge_small(self, tab, n_rows):
        """Fold bins below min_bin_frac into the next-largest bin."""
        floor = self.min_bin_frac * n_rows
        if (tab["count"] >= floor).all() or len(tab) <= 2:
            return tab
        keep = tab[tab["count"] >= floor]
        small = tab[tab["count"] < floor]
        if keep.empty:
            return tab
        merged = keep.copy()
        target = merged["count"].idxmin()
        merged.loc[target, "count"] += small["count"].sum()
        merged.loc[target, "sum"] += small["sum"].sum()
        return merged

    # ------------------------------------------------------------------
    def transform(self, X):
        """Replace each feature with its WOE value.

        Unseen categories fall back to 0.0, i.e. neutral evidence.
        """
        out = pd.DataFrame(index=X.index)
        for col, mapping in self.maps_.items():
            s = X[col]
            binned = (
                self._bin_numeric(s, fit=False, col=col)
                if self.kinds_[col] == "numeric"
                else self._bin_categorical(s)
            )
            out[col] = binned.map(mapping).astype(float).fillna(0.0)
        return out

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------
    def iv_table(self):
        """Information value per feature, strongest first."""
        t = (
            pd.DataFrame({"feature": list(self.iv_), "iv": list(self.iv_.values())})
            .sort_values("iv", ascending=False)
            .reset_index(drop=True)
        )
        t["strength"] = pd.cut(
            t["iv"],
            [-np.inf, 0.02, 0.1, 0.3, 0.5, np.inf],
            labels=["useless", "weak", "medium", "strong", "suspicious"],
        )
        return t

    def select(self, min_iv=0.02):
        """Features clearing an information-value floor."""
        return [f for f, v in self.iv_.items() if v >= min_iv]

    def bin_table(self, col):
        """The fitted bins and WOE values for one feature - the auditable view."""
        return (
            pd.DataFrame({"bin": list(self.maps_[col]), "woe": list(self.maps_[col].values())})
            .sort_values("woe", ascending=False)
            .reset_index(drop=True)
        )


def fit_logistic_scorecard(woe_train, y_train, C=1.0):
    """Logistic regression on WOE features - the challenger model.

    Coefficients on WOE features are directly interpretable: a positive
    coefficient means the feature's evidence points toward default in the
    direction the WOE encodes.
    """
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
    model.fit(woe_train, y_train)
    logger.info("logistic challenger fitted on %d WOE features", woe_train.shape[1])
    return model


def coefficient_table(model, feature_names, woe_matrix=None):
    """Signed coefficients and, where possible, each feature's real contribution.

    Ranking a WOE scorecard by raw coefficient magnitude is misleading. The WOE
    transform already carries the strength of the evidence, so a feature with
    almost no signal has a near-constant WOE column; the regression can then
    attach a large coefficient to it without changing any prediction. A pure
    noise feature can therefore top a coefficient ranking.

    The meaningful quantity is how much the term actually moves the log-odds:

        contribution = |coefficient| * std(WOE)

    Pass `woe_matrix` to get it, and the table is ranked by that instead.

    Parameters
    ----------
    model : fitted LogisticRegression
    feature_names : iterable of str
    woe_matrix : DataFrame, optional
        The WOE-transformed features the model was fitted on.

    Returns
    -------
    DataFrame
    """
    t = pd.DataFrame({"feature": list(feature_names), "coefficient": model.coef_[0]})

    if woe_matrix is not None:
        sd = woe_matrix[list(feature_names)].std(ddof=0)
        t["woe_std"] = sd.to_numpy()
        t["contribution"] = t["coefficient"].abs() * t["woe_std"].to_numpy()
        return t.sort_values("contribution", ascending=False).reset_index(drop=True)

    return (
        t.assign(abs_coef=lambda d: d.coefficient.abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
        .reset_index(drop=True)
    )
