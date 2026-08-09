"""Model monitoring: build a baseline, then check each new period against it.

Implements the monitoring plan in MODEL_CARD.md section 8.

    # once, from the training window
    python monitor.py baseline --input data/credit_risk_data.csv --up-to 202204

    # every month thereafter
    python monitor.py check --input data/credit_risk_data.csv --period 202205

Exit codes, so this can be scheduled and alert on its own:

    0   all checks OK (warnings may be present)
    1   at least one ALERT - the model needs attention
    2   the job could not run

Outcome checks (calibration, discrimination, decile drift) need matured labels.
A first-payment-default label takes three EMIs, so a freshly scored cohort has
none. Pass --no-labels for those months; the outcome checks report SKIPPED
rather than being silently omitted.
"""

import argparse
import json
import os
import pickle
import sys

import lightgbm as lgb
import pandas as pd

from ml_pipeline import processing, utils
from ml_pipeline.logging_utils import get_logger, setup_logging
from ml_pipeline.monitoring import (
    build_baseline_from_raw,
    load_baseline,
    run_monitoring,
    save_baseline,
    score_raw,
)

logger = get_logger("monitor")

BASELINE_FILE = "monitoring_baseline.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("baseline", help="snapshot the training population")
    b.add_argument("--input", default="data/credit_risk_data.csv")
    b.add_argument("--artifacts", default="output_v2")
    b.add_argument(
        "--up-to", type=int, default=202204, help="last application month included in the baseline"
    )

    c = sub.add_parser("check", help="compare a period against the baseline")
    c.add_argument("--input", default="data/credit_risk_data.csv")
    c.add_argument("--artifacts", default="output_v2")
    c.add_argument("--period", type=int, required=True, help="application month to check")
    c.add_argument(
        "--no-labels", action="store_true", help="cohort has not matured; skip the outcome checks"
    )
    c.add_argument("--report", default=None, help="write the report json here")

    for parser in (b, c):
        parser.add_argument("--log-level", dest="log_level", default="INFO")
    return p.parse_args(argv)


def load_model_stack(artifacts):
    model = lgb.Booster(model_file=os.path.join(artifacts, "model_v2.txt"))
    with open(os.path.join(artifacts, "target_encoder_v2.pkl"), "rb") as f:
        encoder = pickle.load(f)
    with open(os.path.join(artifacts, "calibrator_v2.pkl"), "rb") as f:
        calibrator = pickle.load(f)
    with open(os.path.join(artifacts, "feature_columns_v2.json")) as f:
        features = json.load(f)
    return model, encoder, calibrator, features


def load_raw(path, label_dpd=60, label_months=3):
    """Read the file as received - no placeholder cleaning.

    Feature statistics must be measured on the raw data; see
    `monitoring.build_baseline_from_raw`.
    """
    df = utils.process_data(path, [])
    processing.create_label(df, label_dpd, label_months)
    return df


def cmd_baseline(args):
    model, encoder, calibrator, features = load_model_stack(args.artifacts)
    df = load_raw(args.input)
    window = df[df.yearmo <= args.up_to].reset_index(drop=True)
    if window.empty:
        logger.error("no rows at or before %s", args.up_to)
        return 2

    baseline = build_baseline_from_raw(
        window,
        model,
        encoder,
        calibrator,
        features,
        labels=window["label"],
        model_version=os.path.basename(os.path.abspath(args.artifacts)),
    )
    baseline["window"] = {
        "up_to": int(args.up_to),
        "months": sorted(int(m) for m in window.yearmo.unique()),
    }

    dest = os.path.join(args.artifacts, BASELINE_FILE)
    save_baseline(baseline, dest)

    print(f"\nBaseline written to {dest}")
    print(f"  rows           : {baseline['n_rows']:,}")
    print(f"  months         : {baseline['window']['months']}")
    print(f"  features        : {len(baseline['feature_spec'])}")
    print(f"  mean PD        : {baseline['score']['mean_pd']:.4f}")
    print(f"  observed rate  : {baseline['observed_default_rate']:.4f}")
    print(f"  baseline Gini  : {baseline['discrimination']['gini']:.4f}")
    return 0


def cmd_check(args):
    path = os.path.join(args.artifacts, BASELINE_FILE)
    if not os.path.exists(path):
        logger.error("no baseline at %s - run `python monitor.py baseline` first", path)
        return 2
    baseline = load_baseline(path)

    model, encoder, calibrator, features = load_model_stack(args.artifacts)
    df = load_raw(args.input)
    period = df[df.yearmo == args.period].reset_index(drop=True)
    if period.empty:
        logger.error("no rows for period %s", args.period)
        return 2

    scores = score_raw(period, model, encoder, calibrator, features)
    labels = None if args.no_labels else period["label"]
    report = run_monitoring(baseline, period, scores, labels=labels, period=str(args.period))

    print("\n" + "=" * 96)
    print(f"MONITORING {args.period}   n={len(period):,}   overall status: {report.status}")
    print("=" * 96)
    frame = report.to_frame()
    print(f"{'check':<34}{'status':<9}{'value':>10}{'threshold':>11}  message")
    print("-" * 96)
    for _, r in frame.iterrows():
        val = "" if pd.isna(r["value"]) or r["value"] is None else f"{r['value']:.4f}"
        thr = "" if pd.isna(r["threshold"]) or r["threshold"] is None else f"{r['threshold']:.4f}"
        print(f"{r['check']:<34}{r['status']:<9}{val:>10}{thr:>11}  {r['message'][:60]}")
    print("=" * 96)

    if report.alerts:
        print(f"\n{len(report.alerts)} ALERT(S):")
        for c in report.alerts:
            print(f"  - {c.name}: {c.message}")
    if report.warnings:
        print(f"\n{len(report.warnings)} warning(s):")
        for c in report.warnings:
            print(f"  - {c.name}: {c.message}")
    if not report.labels_available:
        print("\nOutcome checks were skipped: this cohort has not matured three EMIs yet.")

    dest = args.report or os.path.join(args.artifacts, f"monitoring_{args.period}.json")
    with open(dest, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"\nReport written to {dest}")

    return 1 if report.alerts else 0


def main(argv=None):
    args = parse_args(argv)
    setup_logging(args.log_level)
    try:
        return cmd_baseline(args) if args.command == "baseline" else cmd_check(args)
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
