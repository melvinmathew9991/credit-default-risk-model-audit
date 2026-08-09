"""Scoring entry point for the governed v2 model.

    python predict_v2.py --input input/credit_risk_data.csv
    python predict_v2.py --input applications.csv --cutoff-score 560 --reasons

Produces, per applicant: calibrated probability of default, score points, an
approve/decline flag against the cutoff, and - for declines - the principal
reasons, which is what makes a decline communicable to the applicant.

Two things this deliberately refuses to do:

* **Score without calibrating.** The raw booster understates risk by 41% (mean
  predicted PD 0.0510 against 0.0871 observed), so a raw score is not a
  probability and must never be treated as one. There is no flag to skip it.
* **Score with a feature set that breaches policy.** The governance check runs
  on every scoring call, not only at training time.

OUTPUT CONFIDENTIALITY: the scores file is keyed by `User_id` and therefore
carries personal data. It inherits the classification of the source dataset -
see DATA.md - and is subject to the applicable access and retention controls.
"""

import argparse
import json
import os
import pickle
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from ml_pipeline import governance, processing, utils
from ml_pipeline.logging_utils import get_logger, setup_logging
from ml_pipeline.scorecard import (ScoreScaler, reason_code_summary,
                                   reason_codes, unmapped_features)

logger = get_logger("predict_v2")

ARTEFACTS = {
    "model": "model_v2.txt",
    "encoder": "target_encoder_v2.pkl",
    "calibrator": "calibrator_v2.pkl",
    "features": "feature_columns_v2.json",
    "manifest": "run_manifest_v2.json",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="raw application csv")
    p.add_argument("--artifacts", default="output_v2")
    p.add_argument("--output", default=None, help="defaults to <artifacts>/scores_v2.csv")
    p.add_argument("--id-col", default="User_id")
    p.add_argument("--cutoff-score", type=float, default=None,
                   help="approve at or above this score; declines get reasons")
    p.add_argument("--cutoff-pd", type=float, default=None,
                   help="alternative cutoff expressed as a probability of default")
    p.add_argument("--reasons", action="store_true",
                   help="always emit reason codes, not only for declines")
    p.add_argument("--pdo", type=float, default=20.0)
    p.add_argument("--base-score", dest="base_score", type=float, default=600.0)
    p.add_argument("--base-odds", dest="base_odds", type=float, default=50.0)
    p.add_argument("--log-level", dest="log_level", default="INFO")
    return p.parse_args(argv)


def load_artifacts(d):
    """Load the model and everything required to reproduce its inputs."""
    missing = [n for n in ARTEFACTS.values() if not os.path.exists(os.path.join(d, n))]
    if missing:
        raise FileNotFoundError(
            f"missing v2 artefacts in {d!r}: {missing}. Run `python engine_v2.py` first.")

    model = lgb.Booster(model_file=os.path.join(d, ARTEFACTS["model"]))
    with open(os.path.join(d, ARTEFACTS["encoder"]), "rb") as f:
        encoder = pickle.load(f)
    with open(os.path.join(d, ARTEFACTS["calibrator"]), "rb") as f:
        calibrator = pickle.load(f)
    with open(os.path.join(d, ARTEFACTS["features"])) as f:
        features = json.load(f)
    with open(os.path.join(d, ARTEFACTS["manifest"])) as f:
        manifest = json.load(f)
    return model, encoder, calibrator, features, manifest


def prepare(df, encoder, features):
    """Apply the exact training-time transformation, in the same order."""
    d = df.copy()
    processing.clean_placeholders(d)
    processing.derived_features(d)
    d = encoder.transform(d)

    absent = [c for c in features if c not in d.columns]
    if absent:
        raise ValueError(f"input is missing model features: {absent}")

    # the policy gate runs at scoring time too, not just during training
    governance.enforce_policy(features)
    return d


def explain(model, X, features, top_n=4):
    """Class-1 SHAP values for the scored rows."""
    import shap

    if "objective" not in model.params:
        # a booster reloaded from disk has empty params
        model.params["objective"] = "binary"
    values = shap.TreeExplainer(model).shap_values(X[features])
    # binary boosters return [class_0, class_1]; class_0 would invert every reason
    if isinstance(values, list):
        values = values[1]
    return reason_codes(values, features, top_n=top_n)


def main(argv=None):
    args = parse_args(argv)
    setup_logging(args.log_level)

    model, encoder, calibrator, features, manifest = load_artifacts(args.artifacts)
    logger.info("model trained %s on %d features, %d trees",
                manifest["created_utc"], len(features), model.num_trees())

    gaps = unmapped_features(features)
    if gaps:
        logger.warning("no applicant-facing reason text for %s - declines citing "
                       "these would only say 'other information'", gaps)

    raw_df = pd.read_csv(args.input, low_memory=False)
    logger.info("scoring %d applications from %s", len(raw_df), args.input)

    X = prepare(raw_df, encoder, features)

    raw = model.predict(X[features])
    pd_hat = calibrator.transform(raw)          # never optional
    scaler = ScoreScaler(args.pdo, args.base_score, args.base_odds)
    points = scaler.to_points(pd_hat)

    out = pd.DataFrame({
        args.id_col: raw_df[args.id_col] if args.id_col in raw_df else np.arange(len(raw_df)),
        "pd": pd_hat,
        "score": points.round(0).astype(int),
    })

    cutoff_pd = args.cutoff_pd
    if args.cutoff_score is not None:
        cutoff_pd = float(scaler.to_pd(args.cutoff_score))
    if cutoff_pd is not None:
        out["decision"] = np.where(out["pd"] <= cutoff_pd, "APPROVE", "DECLINE")
        n_dec = int((out["decision"] == "DECLINE").sum())
        logger.info("cutoff pd=%.4f (score %.0f): %d approved, %d declined (%.1f%%)",
                    cutoff_pd, scaler.to_points(cutoff_pd),
                    len(out) - n_dec, n_dec, 100 * n_dec / len(out))

    if args.reasons or cutoff_pd is not None:
        reasons = explain(model, X, features)
        if cutoff_pd is not None and not args.reasons:
            # a reason code explains an adverse decision; it means nothing on an
            # approval, so blank those rows out rather than shipping noise
            approved = out["decision"].eq("APPROVE").to_numpy()
            for c in reasons.columns:
                reasons.loc[approved, c] = np.nan if c.startswith("contribution_") else ""
        out = pd.concat([out, reasons], axis=1)

    dest = args.output or os.path.join(args.artifacts, "scores_v2.csv")
    out.to_csv(dest, index=False)

    print("\n" + "=" * 78)
    print(f"Scored {len(out):,} applications -> {dest}")
    print("=" * 78)
    print(f"scaling: PDO={scaler.pdo:.0f}, {scaler.base_score:.0f} points at "
          f"{scaler.base_odds:.0f}:1 odds")
    print(f"\nprobability of default: mean {out['pd'].mean():.4f}  "
          f"median {out['pd'].median():.4f}  "
          f"p5 {out['pd'].quantile(0.05):.4f}  p95 {out['pd'].quantile(0.95):.4f}")
    print(f"score points          : mean {out['score'].mean():.0f}  "
          f"median {out['score'].median():.0f}  "
          f"min {out['score'].min()}  max {out['score'].max()}")

    if "decision" in out:
        print(f"\ndecisions: {out['decision'].value_counts().to_dict()}")
        dec = out[out["decision"] == "DECLINE"]
        if len(dec):
            print("\nprincipal reasons cited across declines:")
            print(reason_code_summary(dec).to_string(
                index=False, float_format=lambda v: f"{v:.3f}"))
            print("\nexample declines:")
            cols = [args.id_col, "score", "pd", "reason_1", "reason_2", "reason_3"]
            print(dec[[c for c in cols if c in dec]].head(5).to_string(
                index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nNOTE: this file is keyed by an identifier and carries personal data. "
          "Handle per DATA.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
