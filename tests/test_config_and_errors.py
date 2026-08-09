"""Tests for the configuration layer and for error propagation.

Covers review findings #30 (blanket try/except swallowing failures) and #31
(no configuration management).
"""

import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_pipeline import processing, training, utils  # noqa: E402
from ml_pipeline.config import Config  # noqa: E402


# ===================================================================== config
def test_defaults_are_valid():
    cfg = Config()
    assert "label" in cfg.id_cols
    assert cfg.train_max_yearmo < cfg.val_yearmo < cfg.hold_out_yearmo


def test_label_must_be_a_non_feature_column():
    with pytest.raises(ValueError, match="label"):
        Config(id_cols=["User_id", "yearmo", "max_dpd", "emi_1_dpd", "emi_2_dpd", "emi_3_dpd"])


def test_label_source_columns_must_be_excluded():
    with pytest.raises(ValueError, match="emi_3_dpd"):
        Config(
            id_cols=["User_id", "yearmo", "max_dpd", "emi_1_dpd", "emi_2_dpd", "label"],
            label_months=3,
        )


def test_splits_must_be_ordered_in_time():
    with pytest.raises(ValueError, match="ordered in time"):
        Config(train_max_yearmo=202205, val_yearmo=202204)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"label_dpd": 45}, "label_dpd"),
        ({"label_months": 0}, "label_months"),
        ({"label_months": 9}, "label_months"),
        ({"max_evals": 0}, "max_evals"),
    ],
)
def test_invalid_values_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Config(**kwargs)


def test_cli_overrides_win(monkeypatch):
    monkeypatch.delenv("CRD_MAX_EVALS", raising=False)
    cfg = Config.load(overrides={"max_evals": 3})
    assert cfg.max_evals == 3


def test_none_overrides_are_ignored():
    cfg = Config.load(overrides={"max_evals": None})
    assert cfg.max_evals == Config().max_evals


def test_unknown_override_is_rejected():
    with pytest.raises(ValueError, match="unknown config key"):
        Config.load(overrides={"nonsense": 1})


def test_environment_override(monkeypatch):
    monkeypatch.setenv("CRD_MAX_EVALS", "7")
    monkeypatch.setenv("CRD_SEED", "123")
    cfg = Config.load()
    assert cfg.max_evals == 7 and cfg.seed == 123


def test_cli_beats_environment(monkeypatch):
    monkeypatch.setenv("CRD_MAX_EVALS", "7")
    assert Config.load(overrides={"max_evals": 2}).max_evals == 2


def test_json_config_file(tmp_path):
    p = tmp_path / "run.json"
    p.write_text(json.dumps({"max_evals": 4, "seed": 99}))
    cfg = Config.load(path=str(p))
    assert cfg.max_evals == 4 and cfg.seed == 99


@pytest.mark.parametrize("suffix", [".yaml", ".yml"])
def test_yaml_config_file(tmp_path, suffix):
    """The docstring and README both advertise YAML, so it has to work.

    PyYAML is a declared runtime dependency for exactly this reason - the
    feature was documented while the import was unavailable.
    """
    p = tmp_path / f"run{suffix}"
    p.write_text("max_evals: 4\nseed: 99\nlabel_dpd: 30\n")
    cfg = Config.load(path=str(p))
    assert cfg.max_evals == 4
    assert cfg.seed == 99
    assert cfg.label_dpd == 30


def test_yaml_config_is_still_overridden_by_cli(tmp_path):
    p = tmp_path / "run.yaml"
    p.write_text("max_evals: 4\n")
    assert Config.load(path=str(p), overrides={"max_evals": 11}).max_evals == 11


def test_empty_yaml_config_falls_back_to_defaults(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    assert Config.load(path=str(p)).max_evals == Config().max_evals


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_config_file_with_a_byte_order_mark(tmp_path, suffix):
    """Windows editors write a BOM by default.

    Without utf-8-sig the BOM lands inside the first key name and surfaces as
    `unexpected keyword argument '\\ufeffmax_evals'`, which points nowhere near
    the actual cause.
    """
    p = tmp_path / f"bom{suffix}"
    body = "max_evals: 4\n" if suffix == ".yaml" else json.dumps({"max_evals": 4})
    p.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))
    assert Config.load(path=str(p)).max_evals == 4


def test_log_file_follows_output_dir():
    cfg = Config(output_dir="somewhere")
    assert cfg.resolved_log_file == os.path.join("somewhere", "engine_run.log")
    assert cfg.hyperopt_results_path == os.path.join("somewhere", "hyperopt_results.csv")


def test_config_round_trips_to_dict():
    cfg = Config(max_evals=11)
    assert Config(**cfg.to_dict()).max_evals == 11


# ============================================================ error handling
def test_missing_file_raises_not_returns_none():
    """The old code printed the error and returned None."""
    with pytest.raises(FileNotFoundError):
        utils.process_data("does/not/exist.csv", [])


def test_dropping_an_absent_column_raises(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"a": [1]}).to_csv(p, index=False)
    with pytest.raises(KeyError, match="gender"):
        utils.process_data(str(p), ["gender"])


def test_split_without_yearmo_raises():
    with pytest.raises(KeyError, match="yearmo"):
        utils.data_split(pd.DataFrame({"a": [1, 2]}))


def test_empty_split_raises():
    df = pd.DataFrame({"yearmo": [202201, 202202]})
    with pytest.raises(ValueError, match="empty"):
        utils.data_split(df)


def test_split_boundaries_are_configurable():
    df = pd.DataFrame({"yearmo": [202101, 202102, 202103] * 2})
    tr, va, ho = utils.data_split(df, 202101, 202102, 202103)
    assert len(tr) == 2 and len(va) == 2 and len(ho) == 2


def test_create_label_missing_columns_raises():
    with pytest.raises(KeyError, match="emi_"):
        processing.create_label(pd.DataFrame({"a": [1]}), dpd=60, months=3)


def test_derived_features_missing_columns_raises():
    with pytest.raises(KeyError, match="missing columns"):
        processing.derived_features(pd.DataFrame({"interest_received": [1.0]}))


def test_train_lgb_refuses_to_train_on_the_target():
    """Guard against the C2 defect coming back."""
    df = pd.DataFrame({"label": [0, 1] * 20, "x": range(40)})
    with pytest.raises(ValueError, match="label"):
        training.train_lgb(df, df, {"objective": "binary"}, non_feature_cols=["x"])


def test_train_lgb_requires_at_least_one_feature():
    df = pd.DataFrame({"label": [0, 1] * 20})
    with pytest.raises(ValueError, match="no feature columns"):
        training.train_lgb(df, df, {"objective": "binary"}, non_feature_cols=["label"])
