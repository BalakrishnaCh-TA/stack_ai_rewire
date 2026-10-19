"""
Time-dimension diagnostics: the monthly transaction-volume trend (non-stationarity check),
and the multi-source agreement check between the row-level table and the pre-aggregated
monthly parquet extract.
"""

import pandas as pd


def monthly_volume_trend(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate `transactions` to one row per calendar month with txn_count and
    total_amount, sorted chronologically, so the trend can be characterized (direction,
    rough magnitude) rather than asserted.
    """
    df = transactions.copy()
    df["txn_timestamp"] = pd.to_datetime(df["txn_timestamp"])
    df["txn_month"] = df["txn_timestamp"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df.groupby("txn_month", as_index=False)
        .agg(
            txn_count=("txn_id" if "txn_id" in df.columns else df.columns[0], "count"),
            total_amount=("amount", "sum") if "amount" in df.columns else ("total_amount", "sum"),
        )
        .sort_values("txn_month")
        .reset_index(drop=True)
    )

    return monthly


def compare_row_level_vs_monthly_agg(
    transactions: pd.DataFrame,
    monthly_agg: pd.DataFrame,
) -> pd.DataFrame:
    """Reaggregate `transactions` to the same (customer_id, txn_month) grain as
    `monthly_agg`, join the two on (customer_id, txn_month), and return a per-month (or
    overall-by-month) comparison of txn_count and total_amount so any disagreement between
    the two sources is a number, not a guess. Do not assume they agree without checking.
    """
    txn_df = transactions.copy()
    txn_df["txn_timestamp"] = pd.to_datetime(txn_df["txn_timestamp"])
    txn_df["txn_month"] = txn_df["txn_timestamp"].dt.to_period("M").dt.to_timestamp().dt.date

    m_agg = monthly_agg.copy()
    m_agg["txn_month"] = pd.to_datetime(m_agg["txn_month"]).dt.date

    # Re-aggregate transactions to customer_id x txn_month grain
    amount_col = "amount" if "amount" in txn_df.columns else "total_amount"
    txn_reagg = (
        txn_df.groupby(["customer_id", "txn_month"], as_index=False)
        .agg(
            db_txn_count=("customer_id", "count"),
            db_total_amount=(amount_col, "sum"),
        )
    )

    # Rename monthly_agg columns for clarity before joining
    m_agg_renamed = m_agg.rename(
        columns={
            "txn_count": "parquet_txn_count",
            "total_amount": "parquet_total_amount",
        }
    )

    # Outer join to capture potential missing records in either source
    merged = pd.merge(
        txn_reagg,
        m_agg_renamed,
        on=["customer_id", "txn_month"],
        how="outer",
    ).fillna(0)

    # Calculate differences per row
    merged["count_diff"] = merged["db_txn_count"] - merged["parquet_txn_count"]
    merged["amount_diff"] = merged["db_total_amount"] - merged["parquet_total_amount"]

    # Roll up per month to give an aggregated comparison per calendar month
    comparison_by_month = (
        merged.groupby("txn_month", as_index=False)
        .agg(
            db_total_txns=("db_txn_count", "sum"),
            parquet_total_txns=("parquet_txn_count", "sum"),
            total_count_diff=("count_diff", lambda x: (x != 0).sum()),
            db_total_amount=("db_total_amount", "sum"),
            parquet_total_amount=("parquet_total_amount", "sum"),
            total_amount_diff=("amount_diff", lambda x: (x.abs() > 1e-4).sum()),
        )
        .sort_values("txn_month")
        .reset_index(drop=True)
    )

    return comparison_by_month