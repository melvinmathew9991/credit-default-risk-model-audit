"""A checkable contract for the input data.

Every defect found in the model review was present in the file from the first
day and went unnoticed for the life of the project. None of them needed a model
to detect - they needed something to look. This module is that something.

The checks are declarative and dependency-free (deliberately: the project is
pinned to a 2022-era stack, and pandera/great_expectations would drag in
constraints that break those pins).

Four tripwires here exist specifically because of what the review found:

    type stability      identical text parsed as str/int/float depending on the
                        128k-row chunk it landed in (finding H2)
    placeholder spelling  a missing value written as both "0" and "0.0", whose
                        spelling then predicted default (finding H1)
    degenerate columns  number_of_loans is 0 for 99.59% of rows, which made two
                        derived features near-constant (finding M10)
    leakage screen      a single feature reaching Gini 0.50 on its own is a
                        warning sign, not a triumph (finding C1)

Usage
-----
    from ml_pipeline.data_contract import CREDIT_RISK_CONTRACT
    report = CREDIT_RISK_CONTRACT.validate(df)
    report.raise_if_failed()
"""

from enum import Enum

import pandas as pd

from .logging_utils import get_logger

logger = get_logger(__name__)

#: Strings that commonly stand in for "not captured". Two different spellings of
#: the same sentinel in one column is the H1 defect.
PLACEHOLDER_TOKENS = {
    "0",
    "0.0",
    "-1",
    "-1.0",
    "",
    " ",
    "na",
    "n/a",
    "nan",
    "null",
    "none",
    "unknown",
    "missing",
    "?",
}


class Severity(str, Enum):
    ERROR = "ERROR"  # the run must not continue
    WARNING = "WARNING"  # record it, keep going


class Issue:
    def __init__(self, severity, column, check, message):
        self.severity = severity
        self.column = column
        self.check = check
        self.message = message

    def __repr__(self):
        where = self.column or "<dataset>"
        return f"[{self.severity.value}] {where}.{self.check}: {self.message}"


class DataContractError(Exception):
    """Raised when validation finds an ERROR-level issue."""


class ValidationReport:
    """Result of validating a DataFrame against a contract."""

    def __init__(self, issues, n_rows, n_columns):
        self.issues = issues
        self.n_rows = n_rows
        self.n_columns = n_columns

    @property
    def errors(self):
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self):
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @property
    def ok(self):
        return not self.errors

    def to_frame(self):
        return pd.DataFrame(
            [
                {
                    "severity": i.severity.value,
                    "column": i.column,
                    "check": i.check,
                    "message": i.message,
                }
                for i in self.issues
            ]
        )

    def log(self):
        for i in self.warnings:
            logger.warning("data contract - %s", i)
        for i in self.errors:
            logger.error("data contract - %s", i)
        if self.ok and not self.warnings:
            logger.info(
                "data contract: %d rows x %d columns, all checks passed",
                self.n_rows,
                self.n_columns,
            )
        else:
            logger.info(
                "data contract: %d error(s), %d warning(s) over %d rows",
                len(self.errors),
                len(self.warnings),
                self.n_rows,
            )
        return self

    def raise_if_failed(self):
        if self.errors:
            raise DataContractError(
                f"{len(self.errors)} data contract error(s):\n  - "
                + "\n  - ".join(str(i) for i in self.errors)
            )
        return self

    def __repr__(self):
        return (
            f"ValidationReport(rows={self.n_rows}, errors={len(self.errors)}, "
            f"warnings={len(self.warnings)})"
        )


class ColumnSpec:
    """Expectations for one column.

    Parameters
    ----------
    name : str
    kind : {"numeric", "categorical", "any"}
    required : bool
    max_null_rate : float, optional
        Fail if the null share exceeds this.
    warn_null_rate : float, optional
        Warn (do not fail) above this.
    allowed_values : set, optional
    min_value, max_value : float, optional
    min_unique, max_unique : int, optional
    unique : bool
        Every value must be distinct.
    type_stable : bool
        Every non-null value must be the same Python type. Catches the chunked
        csv-parsing defect.
    check_placeholders : bool
        Flag sentinel values, and fail if two spellings of one sentinel appear.
    max_dominant_share : float, optional
        Warn when a single value covers more than this share of rows.
    """

    def __init__(
        self,
        name,
        kind="any",
        required=True,
        max_null_rate=None,
        warn_null_rate=None,
        allowed_values=None,
        min_value=None,
        max_value=None,
        min_unique=None,
        max_unique=None,
        unique=False,
        type_stable=False,
        check_placeholders=False,
        max_dominant_share=None,
    ):
        self.name = name
        self.kind = kind
        self.required = required
        self.max_null_rate = max_null_rate
        self.warn_null_rate = warn_null_rate
        self.allowed_values = allowed_values
        self.min_value = min_value
        self.max_value = max_value
        self.min_unique = min_unique
        self.max_unique = max_unique
        self.unique = unique
        self.type_stable = type_stable
        self.check_placeholders = check_placeholders
        self.max_dominant_share = max_dominant_share

    # ------------------------------------------------------------------
    def validate(self, df):
        out = []
        add = lambda sev, chk, msg: out.append(Issue(sev, self.name, chk, msg))  # noqa: E731

        if self.name not in df.columns:
            if self.required:
                add(Severity.ERROR, "present", "required column is missing")
            return out

        s = df[self.name]
        n = len(s)
        null_rate = float(s.isna().mean()) if n else 0.0

        if self.max_null_rate is not None and null_rate > self.max_null_rate:
            add(
                Severity.ERROR,
                "null_rate",
                f"{null_rate:.2%} null exceeds the limit of {self.max_null_rate:.2%}",
            )
        elif self.warn_null_rate is not None and null_rate > self.warn_null_rate:
            add(
                Severity.WARNING,
                "null_rate",
                f"{null_rate:.2%} null is above the expected {self.warn_null_rate:.2%}",
            )

        if self.kind == "numeric" and not pd.api.types.is_numeric_dtype(s):
            add(Severity.ERROR, "dtype", f"expected numeric, got {s.dtype}")

        if self.type_stable:
            kinds = {type(v).__name__ for v in s.dropna()}
            if len(kinds) > 1:
                add(
                    Severity.ERROR,
                    "type_stable",
                    f"values parse to multiple python types {sorted(kinds)} - read the "
                    f"csv with low_memory=False, otherwise identical text is typed "
                    f"differently depending on its row position",
                )

        if self.unique and s.duplicated().any():
            add(Severity.WARNING, "unique", f"{int(s.duplicated().sum()):,} duplicate value(s)")

        nun = int(s.nunique(dropna=True))
        if self.min_unique is not None and nun < self.min_unique:
            add(
                Severity.ERROR,
                "cardinality",
                f"{nun} distinct value(s), expected at least {self.min_unique}",
            )
        if self.max_unique is not None and nun > self.max_unique:
            add(
                Severity.WARNING,
                "cardinality",
                f"{nun:,} distinct value(s), more than the expected {self.max_unique:,}",
            )
        if nun <= 1:
            add(Severity.ERROR, "constant", "column is constant and carries no information")

        if self.allowed_values is not None:
            unexpected = set(s.dropna().unique()) - set(self.allowed_values)
            if unexpected:
                shown = sorted(map(str, unexpected))[:6]
                add(Severity.ERROR, "allowed_values", f"unexpected value(s): {shown}")

        if pd.api.types.is_numeric_dtype(s):
            if self.min_value is not None and s.min() < self.min_value:
                add(
                    Severity.ERROR,
                    "range",
                    f"minimum {s.min()} is below the allowed {self.min_value}",
                )
            if self.max_value is not None and s.max() > self.max_value:
                add(
                    Severity.ERROR,
                    "range",
                    f"maximum {s.max()} is above the allowed {self.max_value}",
                )

        if self.max_dominant_share is not None and n and nun > 0:
            top = s.value_counts(dropna=False).iloc[0] / n
            if top > self.max_dominant_share:
                add(
                    Severity.WARNING,
                    "degenerate",
                    f"a single value covers {top:.2%} of rows - the column carries "
                    f"almost no information",
                )

        if self.check_placeholders:
            out.extend(self._placeholder_issues(s))

        return out

    def _placeholder_issues(self, s):
        """Sentinel values, and the two-spellings defect."""
        issues = []
        present = {}
        for v in s.dropna().unique():
            token = str(v).strip().lower()
            if token in PLACEHOLDER_TOKENS:
                present[str(v)] = int((s == v).sum())

        if not present:
            return issues

        share = sum(present.values()) / len(s)
        issues.append(
            Issue(
                Severity.WARNING,
                self.name,
                "placeholder",
                f"{share:.2%} of rows hold a missing-value placeholder "
                f"{ {k: f'{v:,}' for k, v in present.items()} } - these should be "
                f"real nulls",
            )
        )

        # "0" and "0.0" are the same sentinel written twice. Distinct spellings
        # become distinct categories, and the encoder then scores which spelling
        # a record happened to carry.
        numeric_spellings = {}
        for raw in present:
            try:
                numeric_spellings.setdefault(float(raw), []).append(raw)
            except ValueError:
                continue
        for spellings in numeric_spellings.values():
            if len(spellings) > 1:
                issues.append(
                    Issue(
                        Severity.ERROR,
                        self.name,
                        "placeholder_spelling",
                        f"the same placeholder is written {len(spellings)} ways "
                        f"{sorted(spellings)} - distinct spellings become distinct "
                        f"categories, so the model can score data provenance rather "
                        f"than the borrower",
                    )
                )
        return issues


class DataContract:
    """A set of column specs plus dataset-level expectations."""

    def __init__(self, columns, min_rows=1, allow_extra_columns=True, name="contract"):
        self.columns = list(columns)
        self.min_rows = min_rows
        self.allow_extra_columns = allow_extra_columns
        self.name = name

    def validate(self, df):
        issues = []

        if len(df) < self.min_rows:
            issues.append(
                Issue(
                    Severity.ERROR,
                    None,
                    "min_rows",
                    f"{len(df):,} rows, expected at least {self.min_rows:,}",
                )
            )

        if not self.allow_extra_columns:
            extra = set(df.columns) - {c.name for c in self.columns}
            if extra:
                issues.append(
                    Issue(
                        Severity.ERROR,
                        None,
                        "unexpected_columns",
                        f"columns not in the contract: {sorted(extra)}",
                    )
                )

        for spec in self.columns:
            issues.extend(spec.validate(df))

        return ValidationReport(issues, len(df), df.shape[1])

    def column(self, name):
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)


# ===========================================================================
# Leakage screen
# ===========================================================================
def screen_for_leakage(
    df, label_col, features=None, gini_threshold=0.35, max_rows=50000, random_state=0
):
    """Warn about single features that predict the target suspiciously well.

    A lone feature reaching Gini 0.35+ on an application scorecard is far more
    often leakage than insight - `received_principal` scored 0.50 here. This is
    a smoke alarm, not a verdict: it is meant to prompt the question "is this
    field actually available at decision time?"

    Returns
    -------
    DataFrame
        feature, auc, gini, flagged - sorted by |gini| descending.
    """
    from sklearn.metrics import roc_auc_score

    if label_col not in df.columns:
        raise KeyError(f"{label_col!r} not in the frame")

    d = df.sample(min(max_rows, len(df)), random_state=random_state) if len(df) > max_rows else df
    y = d[label_col]
    if y.nunique() < 2:
        raise ValueError("need both classes present to screen for leakage")

    if features is None:
        features = [c for c in d.columns if c != label_col and pd.api.types.is_numeric_dtype(d[c])]

    rows = []
    for c in features:
        s = d[c]
        if not pd.api.types.is_numeric_dtype(s) or s.nunique() < 2:
            continue
        x = s.fillna(s.median())
        auc = roc_auc_score(y, x)
        rows.append({"feature": c, "auc": auc, "gini": 2 * auc - 1, "abs_gini": abs(2 * auc - 1)})

    if not rows:
        return pd.DataFrame(columns=["feature", "auc", "gini", "abs_gini", "flagged"])

    out = pd.DataFrame(rows).sort_values("abs_gini", ascending=False).reset_index(drop=True)
    out["flagged"] = out["abs_gini"] >= gini_threshold

    for _, r in out[out.flagged].iterrows():
        logger.warning(
            "leakage screen: %r alone reaches Gini %.4f - confirm it is "
            "observable at decision time",
            r.feature,
            r.abs_gini,
        )
    return out


# ===========================================================================
# The contract for this dataset
# ===========================================================================
DPD_VALUES = {0, 30, 60, 90}

CREDIT_RISK_CONTRACT = DataContract(
    name="credit_risk_data",
    min_rows=1000,
    columns=[
        ColumnSpec("User_id", kind="numeric", unique=True, min_unique=100),
        ColumnSpec(
            "yearmo",
            kind="numeric",
            max_null_rate=0.0,
            min_value=200001,
            max_value=299912,
            min_unique=2,
            max_unique=120,
        ),
        ColumnSpec(
            "employment_type",
            warn_null_rate=0.60,
            max_unique=20,
            type_stable=True,
            check_placeholders=True,
        ),
        ColumnSpec(
            "tier_of_employment",
            warn_null_rate=0.60,
            max_unique=20,
            type_stable=True,
            check_placeholders=True,
        ),
        ColumnSpec(
            "industry",
            warn_null_rate=0.60,
            type_stable=True,
            check_placeholders=True,
            max_dominant_share=0.60,
        ),
        ColumnSpec(
            "role", warn_null_rate=0.10, max_unique=200, type_stable=True, check_placeholders=True
        ),
        ColumnSpec(
            "work_experience",
            warn_null_rate=0.60,
            max_unique=20,
            type_stable=True,
            check_placeholders=True,
            max_dominant_share=0.60,
        ),
        ColumnSpec(
            "home_type",
            warn_null_rate=0.10,
            max_unique=10,
            type_stable=True,
            check_placeholders=True,
        ),
        ColumnSpec("married", required=False, warn_null_rate=0.40, max_unique=10),
        ColumnSpec("has_social_profile", required=False, warn_null_rate=0.40, max_unique=5),
        ColumnSpec("is_verified", required=False, warn_null_rate=0.40, max_unique=5),
        ColumnSpec("pincode", required=False, type_stable=True, check_placeholders=True),
        ColumnSpec("gender", required=False, max_unique=10),
        ColumnSpec("total_income", kind="numeric", max_null_rate=0.05, min_value=0, min_unique=50),
        ColumnSpec("dependents", kind="numeric", max_null_rate=0.05, min_value=0, max_value=30),
        ColumnSpec("delinq_2yrs", kind="numeric", max_null_rate=0.05, min_value=0, max_value=100),
        ColumnSpec(
            "number_of_loans",
            kind="numeric",
            max_null_rate=0.05,
            min_value=0,
            max_value=100,
            max_dominant_share=0.95,
        ),
        ColumnSpec("total_payement", kind="numeric", required=False, min_value=0),
        ColumnSpec("received_principal", kind="numeric", required=False, min_value=0),
        ColumnSpec("interest_received", kind="numeric", required=False, min_value=0),
        *[
            ColumnSpec(f"emi_{i}_dpd", kind="numeric", max_null_rate=0.0, allowed_values=DPD_VALUES)
            for i in range(1, 7)
        ],
        ColumnSpec("max_dpd", kind="numeric", max_null_rate=0.0, allowed_values=DPD_VALUES),
    ],
)


def check_label_consistency(df):
    """`max_dpd` must equal the maximum of the six EMI DPD columns.

    A mismatch means the label and the outcome columns disagree, which would
    quietly corrupt every downstream metric.
    """
    emis = [f"emi_{i}_dpd" for i in range(1, 7)]
    missing = [c for c in emis + ["max_dpd"] if c not in df.columns]
    if missing:
        return [
            Issue(Severity.WARNING, None, "label_consistency", f"cannot check, missing {missing}")
        ]

    recomputed = df[emis].max(axis=1)
    bad = int((recomputed != df["max_dpd"]).sum())
    if bad:
        return [
            Issue(
                Severity.ERROR,
                "max_dpd",
                "label_consistency",
                f"{bad:,} row(s) where max_dpd does not equal the maximum of " f"emi_1..6_dpd",
            )
        ]
    return []


def validate_raw_data(df, contract=CREDIT_RISK_CONTRACT, strict=True):
    """Validate, log, and optionally raise. The single entry point pipelines use.

    Parameters
    ----------
    df : DataFrame
    contract : DataContract
    strict : bool
        Raise on ERROR-level issues.

    Returns
    -------
    ValidationReport
    """
    report = contract.validate(df)
    report.issues.extend(check_label_consistency(df))
    report.log()
    if strict:
        report.raise_if_failed()
    return report
