"""Where should the cutoff go?

Choosing a cutoff is a business decision, but it should be made against a table
rather than a hunch. This evaluates the v2 model on the hold-out month and shows,
for each approval rate: the score cutoff, the bad rate of the book you would be
left holding, and how many defaults you avoid at the cost of how many good
customers turned away.

    python analysis/cutoff_policy.py --target-bad-rate 0.06

The hold-out is the only honest place to do this: cutoffs picked on training data
look better than they will be.
"""

import argparse
import json
import os
import pickle
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_pipeline import processing, utils  # noqa: E402
from ml_pipeline.config import Config  # noqa: E402
from ml_pipeline.logging_utils import setup_logging  # noqa: E402
from ml_pipeline.scorecard import (ScoreScaler, choose_cutoff, expected_loss,  # noqa: E402
                                   policy_table)
from ml_pipeline.validation import drop_leaked_users  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", default="output_v2")
    ap.add_argument("--target-bad-rate", type=float, default=0.06,
                    help="maximum acceptable bad rate of the approved book")
    ap.add_argument("--avg-exposure", type=float, default=10000.0,
                    help="assumed average exposure per account, for expected loss")
    ap.add_argument("--lgd", type=float, default=0.45,
                    help="assumed loss given default (Basel F-IRB unsecured retail)")
    args = ap.parse_args()

    setup_logging("ERROR")
    cfg = Config()

    model = lgb.Booster(model_file=os.path.join(args.artifacts, "model_v2.txt"))
    with open(os.path.join(args.artifacts, "target_encoder_v2.pkl"), "rb") as f:
        encoder = pickle.load(f)
    with open(os.path.join(args.artifacts, "calibrator_v2.pkl"), "rb") as f:
        calibrator = pickle.load(f)
    with open(os.path.join(args.artifacts, "feature_columns_v2.json")) as f:
        features = json.load(f)

    df = utils.process_data(cfg.data_path, [])
    processing.create_label(df, cfg.label_dpd, cfg.label_months)
    processing.clean_placeholders(df)
    processing.derived_features(df)

    tuning = df[df.yearmo <= cfg.val_yearmo].reset_index(drop=True)
    hold_out = drop_leaked_users(
        tuning, df[df.yearmo == cfg.hold_out_yearmo].reset_index(drop=True),
        name="hold_out")

    X = encoder.transform(hold_out.copy())
    pd_hat = calibrator.transform(model.predict(X[features]))
    scaler = ScoreScaler()

    pol = policy_table(hold_out.label, pd_hat, scaler=scaler, steps=20)
    pol["expected_loss_per_approved"] = [
        expected_loss(r.pd_cutoff, args.avg_exposure, args.lgd) for r in pol.itertuples()]

    print("=" * 104)
    print(f"CUTOFF POLICY - hold-out 202205, n={len(hold_out):,}, "
          f"base bad rate {hold_out.label.mean():.4f}")
    print("=" * 104)
    print(f"{'approve':>8}{'n':>9}{'score':>8}{'pd cutoff':>11}{'book bad':>10}"
          f"{'bads taken':>12}{'bads avoided':>14}{'goods lost':>12}")
    print("-" * 104)
    for _, r in pol.iterrows():
        print(f"{100*r.approval_rate:>7.0f}%{r.n_approved:>9,}{r.score_cutoff:>8.0f}"
              f"{r.pd_cutoff:>11.4f}{r.bad_rate_of_book:>10.4f}"
              f"{r.bads_approved:>12,}{r.bads_declined:>14,}{r.goods_declined:>12,}")
    print("=" * 104)

    pick = choose_cutoff(pol, args.target_bad_rate)
    if pick is None:
        print(f"\nNo cutoff reaches a book bad rate of {args.target_bad_rate:.2%}. "
              f"The best achievable is {pol.bad_rate_of_book.min():.2%} at "
              f"{100*pol.loc[pol.bad_rate_of_book.idxmin(), 'approval_rate']:.0f}% approval.")
    else:
        print(f"\nAt a {args.target_bad_rate:.2%} appetite:")
        print(f"  approve the top {100*pick.approval_rate:.0f}% by score "
              f"({pick.n_approved:,} of {len(hold_out):,})")
        print(f"  cutoff              : {pick.score_cutoff:.0f} points "
              f"(PD {pick.pd_cutoff:.4f})")
        print(f"  book bad rate       : {pick.bad_rate_of_book:.4f} "
              f"vs {hold_out.label.mean():.4f} unscreened "
              f"({100*(1 - pick.bad_rate_of_book/hold_out.label.mean()):.1f}% reduction)")
        print(f"  defaults avoided    : {pick.bads_declined:,} of "
              f"{int(hold_out.label.sum()):,} ({pick.bad_capture_rate:.1%})")
        print(f"  good customers lost : {pick.goods_declined:,}")
        saved = pick.bads_declined * args.lgd * args.avg_exposure
        print(f"\n  Indicative only, at LGD {args.lgd:.0%} and average exposure "
              f"{args.avg_exposure:,.0f}:")
        print(f"    loss avoided on declined defaults ~ {saved:,.0f}")
        print("    (this model estimates PD only; LGD and EAD are assumptions "
              "the user owns)")

    out = os.path.join(args.artifacts, "cutoff_policy.csv")
    pol.to_csv(out, index=False)
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
