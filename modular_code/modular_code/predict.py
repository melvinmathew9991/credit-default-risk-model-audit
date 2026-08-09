"""Batch scoring entry point.

The original project had no way to apply the trained model to new data: engine.py
saved a booster but discarded the fitted target encoder and the feature list, so
the artefact could not be reproduced against raw input. This script closes that
gap using the artefacts engine.py now persists.

    python predict.py --input input/credit_risk_data.csv --output output/scores.csv

The transformation applied here is exactly the training-time one, in the same
order: drop gender -> derived features -> target encode -> select model features.
"""

import argparse
import json
import os
import pickle

import pandas as pd

from ml_pipeline import processing, utils

OUT = "output"


def load_artifacts(artifact_dir=OUT):
    """Load booster, fitted encoder and feature list written by engine.py."""
    import lightgbm as lgb

    model = lgb.Booster(model_file=os.path.join(artifact_dir, "model.txt"))
    with open(os.path.join(artifact_dir, "target_encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)
    with open(os.path.join(artifact_dir, "feature_columns.json")) as f:
        features = json.load(f)
    with open(os.path.join(artifact_dir, "run_manifest.json")) as f:
        manifest = json.load(f)
    return model, encoder, features, manifest


def score(df, model, encoder, features):
    """Transform raw rows and return P(default).

    Parameters
    ----------
    df : DataFrame  raw input, same schema as the training csv
    model : lgb.Booster
    encoder : processing.categorical_encoding
    features : List[str]  model feature names, in order

    Returns
    -------
    Series of float
    """
    d = df.drop(columns=[c for c in ["gender"] if c in df.columns])
    d = processing.derived_features(d)
    d = encoder.transform(d)

    missing = [c for c in features if c not in d.columns]
    if missing:
        raise ValueError(f"input is missing model features: {missing}")

    return pd.Series(model.predict(d[features]), index=df.index, name="pd_score")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="raw csv to score")
    ap.add_argument("--output", default=os.path.join(OUT, "scores.csv"))
    ap.add_argument("--artifacts", default=OUT)
    ap.add_argument("--id-col", default="User_id")
    args = ap.parse_args()

    model, encoder, features, manifest = load_artifacts(args.artifacts)
    print(f"Loaded model trained {manifest['created_utc']} "
          f"({len(features)} features, {manifest['best_iteration']} trees)")

    df = pd.read_csv(args.input, low_memory=False)
    scores = score(df, model, encoder, features)

    out = pd.DataFrame({args.id_col: df[args.id_col], "pd_score": scores})
    out.to_csv(args.output, index=False)
    print(f"Scored {len(out):,} rows -> {args.output}")
    print(out["pd_score"].describe().to_string())


if __name__ == "__main__":
    main()
