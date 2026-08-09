"""Data and model governance controls for the credit risk pipeline.

This is banking data used to make lending decisions, so which columns may enter
a model is a controlled decision, not a modelling convenience. This module makes
that decision explicit, machine-readable and enforced: a run that would put a
prohibited attribute into the feature matrix fails before training starts.

Three registers are defined:

    FIELD_REGISTER    one entry per column - classification, PII flag, rationale
    Classification    the permitted / restricted / prohibited / leakage taxonomy
    enforce_policy()  the gate that training must pass

IMPORTANT - this file encodes a *default conservative position*, not legal
advice. The dataset mixes Indian conventions (pincode, EMI) with US ones
(mortgage/rent/own, a 35,000 loan cap), so the applicable regime is ambiguous
and both are referenced. Before any production use, the institution's compliance
and model risk functions must review and sign off every RESTRICTED and
PROHIBITED classification for the actual jurisdiction and product.

References that informed the defaults:
  - ECOA / Regulation B, 12 CFR 1002 (US) - prohibited bases for credit decisions
  - Fair Housing Act (US) - geographic redlining
  - RBI Fair Practices Code and Digital Lending guidelines (India)
  - DPDP Act 2023 (India) - personal data minimisation and purpose limitation
  - SR 11-7 (US Federal Reserve) - model risk management documentation
"""

from enum import Enum
import hashlib
import os


class Classification(str, Enum):
    """How a field may be used."""

    #: Standard credit attribute. May be used as a model feature.
    PERMITTED = "permitted"

    #: May be used only with documented compliance sign-off and fairness
    #: testing - typically alternative data or a possible proxy.
    RESTRICTED = "restricted"

    #: Must never be used as a model feature: a protected characteristic, or a
    #: close proxy for one.
    PROHIBITED = "prohibited"

    #: Not available at decision time. Using it produces an invalid model.
    LEAKAGE = "leakage"

    #: Identifier, target, or bookkeeping column. Not a feature by construction.
    NON_FEATURE = "non_feature"


class Field:
    """One column's governance record."""

    def __init__(self, name, classification, pii=False, rationale=""):
        self.name = name
        self.classification = classification
        self.pii = pii
        self.rationale = rationale

    def __repr__(self):
        return f"Field({self.name!r}, {self.classification.value})"


P, R, X, L, N = (Classification.PERMITTED, Classification.RESTRICTED,
                 Classification.PROHIBITED, Classification.LEAKAGE,
                 Classification.NON_FEATURE)


FIELD_REGISTER = {f.name: f for f in [
    # ---------------------------------------------------------- identifiers
    Field("User_id", N, pii=True,
          rationale="Direct identifier. Join key only; never a feature. Any file "
                    "keyed by it inherits the dataset's confidentiality class."),
    Field("yearmo", N,
          rationale="Application month. Defines the time-based split; not a feature."),

    # ---------------------------------------------------------- target
    Field("label", N,
          rationale="Modelling target: 60+ DPD within the first 3 EMIs."),
    *[Field(f"emi_{i}_dpd", N,
            rationale="Repayment outcome the label is derived from.")
      for i in range(1, 7)],
    Field("max_dpd", N,
          rationale="Repayment outcome the label is derived from."),

    # ---------------------------------------------------------- prohibited
    Field("gender", X, pii=True,
          rationale="Sex is a prohibited basis under ECOA/Reg B 1002.6(b)(9). "
                    "Excluded by the project from the outset."),
    Field("married", X, pii=True,
          rationale="Marital status is a prohibited basis under ECOA/Reg B "
                    "1002.6(b)(8); it may be collected only to determine rights "
                    "and remedies on the specific credit extension, not scored. "
                    "Was a live feature in the original model."),
    Field("pincode", X,
          rationale="Geographic identifier. Area-level attributes are a "
                    "recognised redlining proxy for protected characteristics "
                    "(Fair Housing Act; RBI fair practices). Excluded unless "
                    "disparate-impact tested, which requires demographic data "
                    "this dataset does not contain."),

    # ---------------------------------------------------------- restricted
    Field("dependents", R, pii=True,
          rationale="Number of dependents may be collected and considered for "
                    "repayment capacity under Reg B 1002.5(d)(3), but it is "
                    "adjacent to familial status. Permitted with fairness "
                    "monitoring; contributes almost nothing (univariate Gini "
                    "0.0016), so exclusion is close to costless."),
    Field("has_social_profile", R,
          rationale="Alternative data. Social-media presence can proxy for age "
                    "and national origin. RBI digital lending guidance requires "
                    "explicit justification for non-traditional attributes."),
    Field("is_verified", R,
          rationale="Verification status of the social profile - inherits the "
                    "concerns of has_social_profile."),

    # ---------------------------------------------------------- leakage
    Field("total_payement", L,
          rationale="Repayment on the loan being scored, not prior loans - see "
                    "MODEL_REVIEW.md finding C1. Unavailable at application."),
    Field("received_principal", L,
          rationale="Repayment on the loan being scored. Unavailable at application."),
    Field("interest_received", L,
          rationale="Repayment on the loan being scored. Unavailable at application."),
    Field("interest_received_ratio", L,
          rationale="Derived from interest_received / total_payement."),
    Field("total_payement_per_loan", L,
          rationale="Derived from total_payement."),

    # ---------------------------------------------------------- permitted
    Field("total_income", P,
          rationale="Declared income. Standard capacity attribute."),
    Field("employment_type", P,
          rationale="Salaried vs self-employed. Standard stability attribute."),
    Field("tier_of_employment", P,
          rationale="Employer tier. Standard stability attribute."),
    Field("industry", P,
          rationale="Employer industry. Standard stability attribute. NOTE: 84% "
                    "of values are a placeholder zero - see clean_placeholders()."),
    Field("role", P,
          rationale="Role at employer. Standard stability attribute."),
    Field("work_experience", P,
          rationale="Length of employment. Standard stability attribute. NOTE: "
                    "84% placeholder - see clean_placeholders()."),
    Field("home_type", P,
          rationale="Rent / own / mortgage. A standard, long-accepted credit "
                    "bureau attribute; not a protected basis."),
    Field("delinq_2yrs", P,
          rationale="Prior delinquency count. Core credit history attribute."),
    Field("delinq_2yrs_ratio", P,
          rationale="Derived from delinq_2yrs and number_of_loans."),
    Field("number_of_loans", P,
          rationale="Prior loan count. Core credit history attribute."),
]}


class PolicyViolation(Exception):
    """Raised when a run would breach the feature governance policy."""


def classify(name):
    """Classification of a column, or None if it is not in the register."""
    f = FIELD_REGISTER.get(name)
    return f.classification if f else None


def permitted_features(columns, allow_restricted=True, allow_leakage=False):
    """Filter `columns` down to those the policy allows as model features.

    Parameters
    ----------
    columns : iterable of str
    allow_restricted : bool
        Include RESTRICTED fields. True by default so their contribution can be
        measured; set False to see the cost of dropping them.
    allow_leakage : bool
        Include LEAKAGE fields. Only ever for the documented comparison against
        the original model - never for a candidate model.

    Returns
    -------
    List[str]

    Raises
    ------
    PolicyViolation
        If a column is not in the register at all. New columns must be
        classified before they can be used.
    """
    allowed = {Classification.PERMITTED}
    if allow_restricted:
        allowed.add(Classification.RESTRICTED)
    if allow_leakage:
        allowed.add(Classification.LEAKAGE)

    out, unknown = [], []
    for c in columns:
        cls = classify(c)
        if cls is None:
            unknown.append(c)
        elif cls in allowed:
            out.append(c)

    if unknown:
        raise PolicyViolation(
            f"columns are not in the governance register and cannot be used "
            f"until classified: {unknown}. Add them to FIELD_REGISTER in "
            f"ml_pipeline/governance.py with a rationale.")
    return out


def enforce_policy(feature_names, allow_leakage=False):
    """Gate that training must pass. Raises rather than warns.

    Parameters
    ----------
    feature_names : iterable of str
        The exact feature matrix columns about to be handed to the model.
    allow_leakage : bool
        Escape hatch for the documented leaky-baseline comparison only.

    Raises
    ------
    PolicyViolation
    """
    problems = []
    for c in feature_names:
        cls = classify(c)
        if cls is None:
            problems.append(f"{c}: not in the governance register")
        elif cls is Classification.PROHIBITED:
            problems.append(f"{c}: PROHIBITED - {FIELD_REGISTER[c].rationale}")
        elif cls is Classification.NON_FEATURE:
            problems.append(f"{c}: NON_FEATURE - identifier or target")
        elif cls is Classification.LEAKAGE and not allow_leakage:
            problems.append(f"{c}: LEAKAGE - {FIELD_REGISTER[c].rationale}")

    if problems:
        raise PolicyViolation(
            "feature governance policy violated:\n  - " + "\n  - ".join(problems))
    return list(feature_names)


def pii_columns(columns):
    """Columns in `columns` flagged as personal data.

    Used to keep identifiers out of logs and to label any output file that
    carries them.
    """
    return [c for c in columns
            if c in FIELD_REGISTER and FIELD_REGISTER[c].pii]


def policy_summary():
    """The register grouped by classification, for the run manifest."""
    out = {}
    for f in FIELD_REGISTER.values():
        out.setdefault(f.classification.value, []).append(f.name)
    return {k: sorted(v) for k, v in sorted(out.items())}


def data_fingerprint(path, block=1 << 20):
    """SHA-256 of the input file, for the audit trail.

    Lineage requirement: a run record has to identify exactly which data
    produced the model.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return {"path": os.path.abspath(path),
            "sha256": h.hexdigest(),
            "bytes": os.path.getsize(path)}
