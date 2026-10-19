"""
Runs eda/profile.py across all three sources and prints a human-readable summary.

Run with `python -m scripts.run_profile` (not `python scripts/run_profile.py`) so `eda`
resolves as an import.
"""

import os
import sys
from typing import Dict, Any
import pandas as pd

from churn_eda.eda.load import load_customers, load_monthly_agg, load_transactions
from churn_eda.eda.profile import (
    count_outliers,
    count_zero_activity_customers,
    profile_distribution,
    profile_missingness,
    profile_schema,
)


def run_profiling() -> Dict[str, Any]:
    """Load all raw data sources and execute schema, missingness, distribution, and outlier checks."""
    customers = load_customers()
    transactions = load_transactions()
    monthly_agg = load_monthly_agg()

    sources = {
        "customers": customers,
        "transactions": transactions,
        "transactions_monthly_agg": monthly_agg,
    }

    report: Dict[str, Any] = {}

    for name, df in sources.items():
        schema_info = profile_schema(df)
        missingness_info = profile_missingness(df)

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        distributions = {}
        outliers = {}

        for col in numeric_cols:
            distributions[col] = profile_distribution(df[col])
            outliers[col] = count_outliers(df[col], rule="iqr", multiplier=1.5)

        report[name] = {
            "schema": schema_info,
            "missingness": missingness_info,
            "distributions": distributions,
            "outliers": outliers,
        }

    report["zero_activity_customers"] = count_zero_activity_customers(customers, transactions)

    return report


def main() -> None:
    """Execute profiling suite, print formatted output, and save artifact summary."""
    print("Running profiling suite across all three data sources...\n")

    try:
        report = run_profiling()
    except Exception as err:
        print(f"Error executing profile suite: {err}", file=sys.stderr)
        sys.exit(1)

    # Print summary to console
    print("=" * 60)
    print("DATA PROFILING SUMMARY")
    print("=" * 60)

    for source_name in ["customers", "transactions", "transactions_monthly_agg"]:
        info = report[source_name]
        print(f"\n--- SOURCE: {source_name} ---")
        print(f"Rows: {info['schema']['n_rows']}")
        print(f"Columns: {info['schema']['columns']}")
        print(f"Missing Values: {info['missingness']}")

        if info["distributions"]:
            print("\n  Numeric Distributions:")
            for col, dist in info["distributions"].items():
                print(
                    f"    - {col}: mean={dist['mean']}, median={dist['median']}, "
                    f"skew={dist['skew_direction']} (min={dist['min']}, max={dist['max']})"
                )

        if info["outliers"]:
            print("\n  Outlier Counts (IQR 1.5x):")
            for col, out in info["outliers"].items():
                print(
                    f"    - {col}: {out['n_outliers']} outliers "
                    f"(bounds: [{out['bound_low']:.2f}, {out['bound_high']:.2f}])"
                )

    print("\n" + "=" * 60)
    print(f"Zero-Activity Customers Count: {report['zero_activity_customers']}")
    print("=" * 60)

    # Save summary artifact to outputs/artifacts/
    output_dir = os.path.join("outputs", "artifacts")
    os.makedirs(output_dir, exist_ok=True)
    artifact_path = os.path.join(output_dir, "profile_summary.txt")

    with open(artifact_path, "w", encoding="utf-8") as f:
        f.write("DATA PROFILING REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Zero-Activity Customers: {report['zero_activity_customers']}\n\n")

        for source_name in ["customers", "transactions", "transactions_monthly_agg"]:
            info = report[source_name]
            f.write(f"--- SOURCE: {source_name} ---\n")
            f.write(f"Rows: {info['schema']['n_rows']}\n")
            f.write(f"Columns: {info['schema']['columns']}\n")
            f.write(f"Missing Values: {info['missingness']}\n\n")

    print(f"\nProfiling artifact saved successfully to: {artifact_path}")


if __name__ == "__main__":
    main()