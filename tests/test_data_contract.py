"""Tests for the data contract.

The integration tests at the bottom are the point of the module: they assert
that the contract independently rediscovers the defects that took a manual model
review to find.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_pipeline.data_contract import (  # noqa: E402
    CREDIT_RISK_CONTRACT,
    ColumnSpec,
    DataContract,
    DataContractError,
    Severity,
    check_label_consistency,
    screen_for_leakage,
    validate_raw_data,
)

DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "credit_risk_data.csv"
)


def codes(issues):
    return {(i.column, i.check) for i in issues}


# ================================================================ ColumnSpec
def test_missing_required_column_is_an_error():
    issues = ColumnSpec("absent").validate(pd.DataFrame({"other": [1, 2]}))
    assert issues[0].severity is Severity.ERROR
    assert issues[0].check == "present"


def test_missing_optional_column_is_silent():
    assert ColumnSpec("absent", required=False).validate(pd.DataFrame({"a": [1]})) == []


def test_null_rate_error_and_warning():
    df = pd.DataFrame({"a": [1, None, 3, 4, 5, 6, 7, 8, 9, 10]})  # 10% null
    assert ("a", "null_rate") in codes(ColumnSpec("a", max_null_rate=0.05).validate(df))
    warn = ColumnSpec("a", warn_null_rate=0.05).validate(df)
    assert warn[0].severity is Severity.WARNING


def test_null_rate_within_limits_passes():
    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    assert ColumnSpec("a", max_null_rate=0.05).validate(df) == []


def test_numeric_kind_is_enforced():
    df = pd.DataFrame({"a": ["x", "y"]})
    assert ("a", "dtype") in codes(ColumnSpec("a", kind="numeric").validate(df))


def test_type_stability_detects_mixed_python_types():
    """The H2 defect: identical text typed differently by row position."""
    df = pd.DataFrame({"a": ["0", 0, 0.0, "real"]})
    issues = ColumnSpec("a", type_stable=True).validate(df)
    assert ("a", "type_stable") in codes(issues)
    assert [i for i in issues if i.check == "type_stable"][0].severity is Severity.ERROR


def test_type_stability_passes_on_a_clean_column():
    df = pd.DataFrame({"a": ["x", "y", "z"]})
    assert ("a", "type_stable") not in codes(ColumnSpec("a", type_stable=True).validate(df))


def test_duplicates_are_reported_when_uniqueness_is_expected():
    df = pd.DataFrame({"id": [1, 2, 2, 3]})
    issues = ColumnSpec("id", unique=True).validate(df)
    assert ("id", "unique") in codes(issues)


def test_cardinality_bounds():
    df = pd.DataFrame({"a": list("abcdef")})
    assert ("a", "cardinality") in codes(ColumnSpec("a", min_unique=10).validate(df))
    assert ("a", "cardinality") in codes(ColumnSpec("a", max_unique=3).validate(df))
    assert ColumnSpec("a", min_unique=2, max_unique=10).validate(df) == []


def test_constant_column_is_an_error():
    df = pd.DataFrame({"a": [5, 5, 5]})
    issues = ColumnSpec("a").validate(df)
    assert ("a", "constant") in codes(issues)


def test_allowed_values():
    df = pd.DataFrame({"dpd": [0, 30, 45, 90]})
    issues = ColumnSpec("dpd", allowed_values={0, 30, 60, 90}).validate(df)
    assert ("dpd", "allowed_values") in codes(issues)
    assert "45" in issues[0].message


def test_numeric_range():
    df = pd.DataFrame({"a": [-5, 10, 200]})
    assert ("a", "range") in codes(ColumnSpec("a", kind="numeric", min_value=0).validate(df))
    assert ("a", "range") in codes(ColumnSpec("a", kind="numeric", max_value=100).validate(df))


def test_degenerate_column_is_flagged():
    """The M10 defect: number_of_loans is 0 for 99.59% of rows."""
    df = pd.DataFrame({"a": [0] * 99 + [1]})
    issues = ColumnSpec("a", max_dominant_share=0.95).validate(df)
    assert ("a", "degenerate") in codes(issues)


# ============================================================== placeholders
def test_single_placeholder_spelling_is_a_warning():
    df = pd.DataFrame({"a": ["0"] * 50 + ["real"] * 50})
    issues = ColumnSpec("a", check_placeholders=True).validate(df)
    ph = [i for i in issues if i.check == "placeholder"]
    assert ph and ph[0].severity is Severity.WARNING


def test_two_spellings_of_one_placeholder_is_an_error():
    """The H1 defect, caught mechanically."""
    df = pd.DataFrame({"a": ["0"] * 40 + ["0.0"] * 30 + ["real"] * 30})
    issues = ColumnSpec("a", check_placeholders=True).validate(df)
    spelling = [i for i in issues if i.check == "placeholder_spelling"]
    assert spelling and spelling[0].severity is Severity.ERROR
    assert "'0'" in spelling[0].message and "'0.0'" in spelling[0].message


def test_distinct_placeholders_are_not_a_spelling_error():
    """'0' and 'unknown' are different sentinels, not two spellings of one."""
    df = pd.DataFrame({"a": ["0"] * 30 + ["unknown"] * 30 + ["real"] * 40})
    issues = ColumnSpec("a", check_placeholders=True).validate(df)
    assert not [i for i in issues if i.check == "placeholder_spelling"]


def test_clean_column_raises_no_placeholder_issue():
    df = pd.DataFrame({"a": ["alpha", "beta", "gamma"]})
    issues = ColumnSpec("a", check_placeholders=True).validate(df)
    assert not [i for i in issues if i.check.startswith("placeholder")]


# ============================================================== DataContract
def test_min_rows():
    c = DataContract([ColumnSpec("a")], min_rows=10)
    r = c.validate(pd.DataFrame({"a": [1, 2]}))
    assert any(i.check == "min_rows" for i in r.issues)


def test_unexpected_columns_can_be_rejected():
    c = DataContract([ColumnSpec("a")], allow_extra_columns=False)
    r = c.validate(pd.DataFrame({"a": [1, 2], "surprise": [3, 4]}))
    assert any(i.check == "unexpected_columns" for i in r.issues)


def test_extra_columns_allowed_by_default():
    c = DataContract([ColumnSpec("a")])
    r = c.validate(pd.DataFrame({"a": [1, 2], "extra": [3, 4]}))
    assert not any(i.check == "unexpected_columns" for i in r.issues)


def test_contract_column_lookup():
    c = DataContract([ColumnSpec("a")])
    assert c.column("a").name == "a"
    with pytest.raises(KeyError):
        c.column("nope")


# =========================================================== ValidationReport
def test_report_partitions_by_severity():
    c = DataContract([ColumnSpec("missing"), ColumnSpec("a", warn_null_rate=0.0)])
    r = c.validate(pd.DataFrame({"a": [1, None, 3]}))
    assert len(r.errors) >= 1 and len(r.warnings) >= 1
    assert not r.ok


def test_report_is_ok_when_only_warnings():
    c = DataContract([ColumnSpec("a", warn_null_rate=0.0)])
    r = c.validate(pd.DataFrame({"a": [1, None, 3]}))
    assert r.ok and r.warnings


def test_raise_if_failed():
    c = DataContract([ColumnSpec("missing")])
    with pytest.raises(DataContractError, match="missing"):
        c.validate(pd.DataFrame({"a": [1]})).raise_if_failed()


def test_raise_if_failed_is_a_noop_when_clean():
    c = DataContract([ColumnSpec("a")])
    assert c.validate(pd.DataFrame({"a": [1, 2]})).raise_if_failed().ok


def test_report_to_frame():
    c = DataContract([ColumnSpec("missing")])
    f = c.validate(pd.DataFrame({"a": [1]})).to_frame()
    assert list(f.columns) == ["severity", "column", "check", "message"]


# ========================================================= label consistency
def test_label_consistency_detects_a_mismatch():
    df = pd.DataFrame({**{f"emi_{i}_dpd": [0] for i in range(1, 7)}, "max_dpd": [90]})
    assert check_label_consistency(df)[0].severity is Severity.ERROR


def test_label_consistency_passes_when_correct():
    df = pd.DataFrame({**{f"emi_{i}_dpd": [0] for i in range(1, 7)}, "max_dpd": [0]})
    assert check_label_consistency(df) == []


def test_label_consistency_skips_when_columns_absent():
    out = check_label_consistency(pd.DataFrame({"a": [1]}))
    assert out and out[0].severity is Severity.WARNING


# ============================================================ leakage screen
def test_leakage_screen_flags_a_giveaway_feature():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=2000)
    df = pd.DataFrame(
        {"label": y, "leak": y + rng.normal(0, 0.05, size=2000), "noise": rng.normal(size=2000)}
    )
    out = screen_for_leakage(df, "label", gini_threshold=0.35)
    assert out.iloc[0]["feature"] == "leak"
    assert bool(out.iloc[0]["flagged"])
    assert not bool(out.set_index("feature").loc["noise", "flagged"])


def test_leakage_screen_flags_inverse_relationships_too():
    """received_principal has AUC 0.25 - inverse, but just as much of a giveaway."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=2000)
    df = pd.DataFrame({"label": y, "inverse": -y + rng.normal(0, 0.05, size=2000)})
    out = screen_for_leakage(df, "label", gini_threshold=0.35)
    assert out.iloc[0]["auc"] < 0.5
    assert bool(out.iloc[0]["flagged"])


def test_leakage_screen_requires_both_classes():
    df = pd.DataFrame({"label": [0, 0, 0], "x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="both classes"):
        screen_for_leakage(df, "label")


def test_leakage_screen_requires_the_label():
    with pytest.raises(KeyError):
        screen_for_leakage(pd.DataFrame({"x": [1]}), "label")


# ===========================================================================
# Integration: does the contract rediscover the review's findings?
# ===========================================================================
needs_data = pytest.mark.skipif(not os.path.exists(DATA), reason="dataset not present")


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(DATA, low_memory=False)


@needs_data
def test_contract_catches_the_placeholder_spelling_defect(raw):
    """Finding H1, which took a manual investigation to find."""
    report = CREDIT_RISK_CONTRACT.validate(raw)
    spelling = [i for i in report.errors if i.check == "placeholder_spelling"]
    assert {i.column for i in spelling} == {"industry", "work_experience"}


@needs_data
def test_contract_catches_the_degenerate_column(raw):
    """Finding M10: number_of_loans is 0 for 99.59% of rows."""
    report = CREDIT_RISK_CONTRACT.validate(raw)
    assert ("number_of_loans", "degenerate") in codes(report.warnings)


@needs_data
def test_contract_catches_repeat_customers(raw):
    """Finding M7: 9,975 duplicate User_id values."""
    report = CREDIT_RISK_CONTRACT.validate(raw)
    assert ("User_id", "unique") in codes(report.warnings)


@needs_data
def test_leakage_screen_catches_received_principal(raw):
    """Finding C1: the single strongest feature is a post-origination field."""
    d = raw.copy()
    d["label"] = (d[[f"emi_{i}_dpd" for i in range(1, 4)]].max(axis=1) >= 60).astype(int)
    out = screen_for_leakage(
        d,
        "label",
        features=["received_principal", "total_payement", "total_income", "delinq_2yrs"],
        gini_threshold=0.35,
    )
    flagged = set(out[out.flagged]["feature"])
    assert "received_principal" in flagged
    assert "total_income" not in flagged


@needs_data
def test_type_stability_fires_on_the_default_chunked_read():
    """Finding H2 - the check only means something if it catches the bad read."""
    bad = pd.read_csv(DATA)  # default low_memory=True
    issues = ColumnSpec("work_experience", type_stable=True).validate(bad)
    assert ("work_experience", "type_stable") in codes(issues), (
        "the chunked read no longer produces mixed types on this pandas build; "
        "the H2 tripwire needs revisiting"
    )


@needs_data
def test_type_stability_passes_on_the_correct_read(raw):
    assert ("work_experience", "type_stable") not in codes(
        ColumnSpec("work_experience", type_stable=True).validate(raw)
    )


@needs_data
def test_validate_raw_data_raises_in_strict_mode(raw):
    with pytest.raises(DataContractError):
        validate_raw_data(raw, strict=True)


@needs_data
def test_validate_raw_data_reports_without_raising(raw):
    report = validate_raw_data(raw, strict=False)
    assert not report.ok and len(report.errors) == 2
