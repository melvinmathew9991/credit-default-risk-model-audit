"""Tests for the data governance controls.

These are the controls that stop a prohibited attribute or a leakage field
reaching a lending model, so they are asserted, not assumed.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_pipeline import governance, processing  # noqa: E402
from ml_pipeline.governance import Classification, PolicyViolation  # noqa: E402


# ------------------------------------------------------------ classification
@pytest.mark.parametrize("col", ["gender", "married", "pincode"])
def test_protected_attributes_are_prohibited(col):
    assert governance.classify(col) is Classification.PROHIBITED


@pytest.mark.parametrize(
    "col",
    [
        "total_payement",
        "received_principal",
        "interest_received",
        "interest_received_ratio",
        "total_payement_per_loan",
    ],
)
def test_post_origination_fields_are_leakage(col):
    assert governance.classify(col) is Classification.LEAKAGE


@pytest.mark.parametrize(
    "col",
    ["total_income", "employment_type", "delinq_2yrs", "home_type", "role", "number_of_loans"],
)
def test_standard_credit_attributes_are_permitted(col):
    assert governance.classify(col) is Classification.PERMITTED


@pytest.mark.parametrize("col", ["User_id", "label", "yearmo", "max_dpd", "emi_1_dpd"])
def test_identifiers_and_target_are_non_features(col):
    assert governance.classify(col) is Classification.NON_FEATURE


def test_every_registered_field_has_a_rationale():
    for name, f in governance.FIELD_REGISTER.items():
        assert f.rationale.strip(), f"{name} has no documented rationale"


# ----------------------------------------------------------------- enforcement
def test_prohibited_feature_blocks_the_run():
    with pytest.raises(PolicyViolation, match="married"):
        governance.enforce_policy(["total_income", "married"])


def test_geographic_proxy_blocks_the_run():
    with pytest.raises(PolicyViolation, match="pincode"):
        governance.enforce_policy(["total_income", "pincode"])


def test_leakage_feature_blocks_the_run_by_default():
    with pytest.raises(PolicyViolation, match="received_principal"):
        governance.enforce_policy(["total_income", "received_principal"])


def test_leakage_allowed_only_with_the_explicit_escape_hatch():
    out = governance.enforce_policy(["total_income", "received_principal"], allow_leakage=True)
    assert "received_principal" in out


def test_target_cannot_be_a_feature():
    with pytest.raises(PolicyViolation, match="label"):
        governance.enforce_policy(["total_income", "label"])


def test_identifier_cannot_be_a_feature():
    with pytest.raises(PolicyViolation, match="User_id"):
        governance.enforce_policy(["total_income", "User_id"])


def test_unregistered_column_blocks_the_run():
    """A new column must be classified before it can be used."""
    with pytest.raises(PolicyViolation, match="governance register"):
        governance.enforce_policy(["total_income", "brand_new_feature"])


def test_permitted_run_passes():
    assert governance.enforce_policy(["total_income", "employment_type"])


# ----------------------------------------------------------------- selection
def test_permitted_features_excludes_prohibited_and_leakage():
    cols = [
        "total_income",
        "married",
        "pincode",
        "received_principal",
        "employment_type",
        "User_id",
        "label",
    ]
    out = governance.permitted_features(cols)
    assert set(out) == {"total_income", "employment_type"}


def test_restricted_can_be_excluded():
    cols = ["total_income", "dependents", "has_social_profile", "is_verified"]
    assert set(governance.permitted_features(cols, allow_restricted=True)) == set(cols)
    assert governance.permitted_features(cols, allow_restricted=False) == ["total_income"]


def test_permitted_features_rejects_unknown_columns():
    with pytest.raises(PolicyViolation, match="not in the governance register"):
        governance.permitted_features(["total_income", "mystery"])


def test_the_permitted_set_survives_enforcement():
    """Whatever permitted_features returns must pass enforce_policy."""
    cols = list(governance.FIELD_REGISTER)
    governance.enforce_policy(governance.permitted_features(cols))


# ----------------------------------------------------------------------- PII
def test_pii_columns_are_flagged():
    pii = governance.pii_columns(
        ["User_id", "gender", "married", "dependents", "total_income", "home_type"]
    )
    assert "User_id" in pii and "total_income" not in pii


def test_policy_summary_covers_every_field():
    summary = governance.policy_summary()
    assert sum(len(v) for v in summary.values()) == len(governance.FIELD_REGISTER)


# ------------------------------------------------------------------- lineage
def test_data_fingerprint_is_stable_and_detects_change(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("a,b\n1,2\n")
    first = governance.data_fingerprint(str(p))
    assert first == governance.data_fingerprint(str(p))
    assert len(first["sha256"]) == 64
    p.write_text("a,b\n1,3\n")
    assert governance.data_fingerprint(str(p))["sha256"] != first["sha256"]


# -------------------------------------------------------------- placeholders
def test_clean_placeholders_maps_both_spellings_to_missing():
    df = pd.DataFrame(
        {
            "work_experience": ["0", "0.0", "5-10", "10+", np.nan],
            "industry": ["0", "0.0", "abc", "def", "ghi"],
        }
    )
    processing.clean_placeholders(df)
    assert df["work_experience"].tolist()[2:4] == ["5-10", "10+"]
    assert df["work_experience"].isna().sum() == 3
    assert df["industry"].isna().sum() == 2


def test_clean_placeholders_ignores_absent_columns():
    df = pd.DataFrame({"other": [1, 2]})
    processing.clean_placeholders(df)
    assert df["other"].tolist() == [1, 2]


def test_clean_placeholders_handles_numeric_zero_variants():
    df = pd.DataFrame({"industry": [0, 0.0, "0", "real"]})
    processing.clean_placeholders(df)
    assert df["industry"].isna().sum() == 3
