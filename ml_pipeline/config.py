"""Central configuration for the credit risk pipeline.

Everything that was previously hardcoded in engine.py - paths, the label
definition, the split boundaries, seeds, encoder and selector parameters - lives
here, so a run can be described by one object and recorded in the run manifest.

Precedence, lowest to highest:

    dataclass defaults  <  YAML file (--config)  <  environment  <  CLI flags

Environment variables are prefixed `CRD_`, e.g. `CRD_MAX_EVALS=5`.
"""

import json
import os
from dataclasses import asdict, dataclass, field, fields

ENV_PREFIX = "CRD_"


@dataclass
class Config:
    # ---- data ----------------------------------------------------------
    data_path: str = "data/credit_risk_data.csv"
    output_dir: str = "output"
    drop_columns: list = field(default_factory=lambda: ["gender"])

    # Columns that must never enter the feature matrix. 'label' is the target;
    # the emi_*_dpd and max_dpd columns are what the label is derived from.
    id_cols: list = field(
        default_factory=lambda: [
            "User_id",
            "emi_1_dpd",
            "emi_2_dpd",
            "emi_3_dpd",
            "emi_4_dpd",
            "emi_5_dpd",
            "emi_6_dpd",
            "max_dpd",
            "yearmo",
            "label",
        ]
    )

    # ---- label ---------------------------------------------------------
    label_dpd: int = 60
    label_months: int = 3

    # ---- time-based split ----------------------------------------------
    train_max_yearmo: int = 202203
    val_yearmo: int = 202204
    hold_out_yearmo: int = 202205

    # ---- tuning --------------------------------------------------------
    max_evals: int = 50
    seed: int = 7
    num_boost_round: int = 20000
    early_stopping_rounds: int = 50

    # ---- component parameters ------------------------------------------
    encoder_params: dict = field(
        default_factory=lambda: {
            "verbose": 0,
            "cols": None,
            "drop_invariant": False,
            "return_df": True,
            "handle_missing": "value",
            "handle_unknown": "value",
            "min_samples_leaf": 5000,
            "smoothing": 1,
        }
    )

    rf_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 250,
            "criterion": "entropy",
            "verbose": False,
            "n_jobs": -1,
            "random_state": 2019,
        }
    )

    dt_params: dict = field(default_factory=lambda: {"random_state": 2019})

    # ---- logging -------------------------------------------------------
    log_level: str = "INFO"
    # None means "engine_run.log inside output_dir", so the log follows the
    # output directory when that is overridden.
    log_file: str = None

    # --------------------------------------------------------------------
    def __post_init__(self):
        self.validate()

    def validate(self):
        """Fail fast on a configuration that cannot produce a valid model."""
        # Check the scalar ranges first: a bad label_months would otherwise be
        # reported as a missing emi_7_dpd column, which points at the wrong thing.
        if self.label_dpd not in (30, 60, 90):
            raise ValueError(f"label_dpd must be one of 30/60/90, got {self.label_dpd}")
        if not 1 <= self.label_months <= 6:
            raise ValueError(f"label_months must be 1..6, got {self.label_months}")
        if "label" not in self.id_cols:
            raise ValueError(
                "'label' must be listed in id_cols - otherwise the target ends "
                "up in the feature matrix and every metric becomes meaningless"
            )
        for c in ["max_dpd"] + [f"emi_{i}_dpd" for i in range(1, self.label_months + 1)]:
            if c not in self.id_cols:
                raise ValueError(f"{c!r} is used to build the label and must be in id_cols")
        if not (self.train_max_yearmo < self.val_yearmo < self.hold_out_yearmo):
            raise ValueError(
                "splits must be ordered in time: "
                f"train<={self.train_max_yearmo} < val=={self.val_yearmo} "
                f"< hold_out=={self.hold_out_yearmo}"
            )
        if self.max_evals < 1:
            raise ValueError(f"max_evals must be >= 1, got {self.max_evals}")
        return self

    # --------------------------------------------------------------------
    @classmethod
    def load(cls, path=None, overrides=None):
        """Build a Config from defaults, an optional YAML/JSON file, env, and CLI.

        Parameters
        ----------
        path : str, optional
            YAML or JSON file with a subset of the fields.
        overrides : dict, optional
            Values from the CLI. Keys set to None are ignored.

        Returns
        -------
        Config
        """
        values = {}

        if path:
            values.update(cls._read_file(path))

        names = {f.name: f for f in fields(cls)}
        for name, f in names.items():
            env = os.environ.get(ENV_PREFIX + name.upper())
            if env is not None:
                values[name] = cls._coerce(env, f.type)

        for k, v in (overrides or {}).items():
            if v is not None:
                if k not in names:
                    raise ValueError(f"unknown config key: {k}")
                values[k] = v

        return cls(**values)

    @staticmethod
    def _read_file(path):
        # utf-8-sig strips a byte-order mark if one is present and behaves like
        # plain utf-8 otherwise. Windows editors and PowerShell's `Set-Content
        # -Encoding utf8` write a BOM by default, and without this the BOM ends
        # up inside the first key name - "﻿max_evals" - producing a
        # baffling "unexpected keyword argument" error.
        with open(path, encoding="utf-8-sig") as fh:
            text = fh.read()
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml
            except ImportError as e:
                raise ImportError(
                    "PyYAML is required to read a .yaml config; "
                    "use a .json config or `pip install pyyaml`"
                ) from e
            return yaml.safe_load(text) or {}
        return json.loads(text)

    @staticmethod
    def _coerce(raw, type_):
        if type_ is int:
            return int(raw)
        if type_ is float:
            return float(raw)
        if type_ is bool:
            return raw.strip().lower() in ("1", "true", "yes", "y")
        if type_ in (list, dict):
            return json.loads(raw)
        return raw

    def to_dict(self):
        return asdict(self)

    @property
    def hyperopt_results_path(self):
        return os.path.join(self.output_dir, "hyperopt_results.csv")

    @property
    def resolved_log_file(self):
        """Log destination, defaulting to engine_run.log inside output_dir."""
        return self.log_file or os.path.join(self.output_dir, "engine_run.log")

    def path(self, *parts):
        """Path inside the output directory."""
        return os.path.join(self.output_dir, *parts)
