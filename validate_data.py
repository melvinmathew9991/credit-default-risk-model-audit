"""Validate an input file against the data contract, before training anything.

    python validate_data.py                            # the project dataset
    python validate_data.py --input new_extract.csv
    python validate_data.py --no-strict                # report, do not fail

Exit code 0 if the contract holds, 1 if it does not — so this can gate a
pipeline or a CI job.

Every check here corresponds to a defect that was found by hand during the model
review and would have been caught on day one had this existed.
"""

import argparse
import os
import sys

import pandas as pd

from ml_pipeline.data_contract import (
    CREDIT_RISK_CONTRACT,
    DataContractError,
    screen_for_leakage,
    validate_raw_data,
)
from ml_pipeline.logging_utils import setup_logging


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", default="data/credit_risk_data.csv")
    p.add_argument(
        "--no-strict", dest="strict", action="store_false", help="report issues without failing"
    )
    p.add_argument("--report", default=None, help="write the issue list to csv")
    p.add_argument("--skip-leakage-screen", action="store_true")
    p.add_argument("--gini-threshold", type=float, default=0.35)
    p.add_argument("--log-level", dest="log_level", default="WARNING")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    setup_logging(args.log_level)

    # low_memory=False is itself part of the contract - see the type_stable check
    df = pd.read_csv(args.input, low_memory=False)

    print("=" * 100)
    print(f"DATA CONTRACT: {args.input}  ({len(df):,} rows x {df.shape[1]} columns)")
    print("=" * 100)

    try:
        report = validate_raw_data(df, CREDIT_RISK_CONTRACT, strict=False)
    except Exception as e:  # a malformed frame should not crash the reporter
        print(f"validation could not run: {e}")
        return 1

    if report.issues:
        frame = report.to_frame()
        for sev in ("ERROR", "WARNING"):
            block = frame[frame.severity == sev]
            if block.empty:
                continue
            print(f"\n{sev}S ({len(block)})")
            print("-" * 100)
            for _, r in block.iterrows():
                where = r.column if r.column else "<dataset>"
                print(f"  {where:<22} {r.check:<22} {r['message']}")
    else:
        print("\nAll checks passed.")

    if not args.skip_leakage_screen:
        emis = [f"emi_{i}_dpd" for i in range(1, 4)]
        if all(c in df.columns for c in emis):
            d = df.copy()
            d["label"] = (d[emis].max(axis=1) >= 60).astype(int)
            exclude = {"label", "max_dpd", "User_id", "yearmo"} | {
                f"emi_{i}_dpd" for i in range(1, 7)
            }
            feats = [
                c for c in d.columns if c not in exclude and pd.api.types.is_numeric_dtype(d[c])
            ]
            screen = screen_for_leakage(
                d, "label", features=feats, gini_threshold=args.gini_threshold
            )
            print(f"\nLEAKAGE SCREEN (single-feature Gini, threshold {args.gini_threshold})")
            print("-" * 100)
            for _, r in screen.iterrows():
                mark = "  <-- CHECK AVAILABILITY AT DECISION TIME" if r.flagged else ""
                print(f"  {r.feature:<26} auc={r.auc:.4f}  gini={r.abs_gini:.4f}{mark}")

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        report.to_frame().to_csv(args.report, index=False)
        print(f"\nIssue list written to {args.report}")

    print("\n" + "=" * 100)
    print(f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)")
    print("=" * 100)

    if args.strict and report.errors:
        print(
            "\nFAILED: the contract is not satisfied. Fix the data, or pass "
            "--no-strict to report without failing."
        )
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except DataContractError as e:
        print(e)
        sys.exit(1)
