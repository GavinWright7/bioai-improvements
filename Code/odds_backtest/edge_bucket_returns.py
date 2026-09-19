"""
edge_bucket_returns.py
============================================================
Flat 1-unit stake analysis: average realized return by model edge bucket
and cumulative stats by minimum-edge threshold. No bankroll compounding.

Run (from project root):
  python "Code/odds_backtest/edge_bucket_returns.py"
============================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config (edit here)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent


def _find_project_root() -> Path:
    for p in [_SCRIPT_DIR, *_SCRIPT_DIR.parents]:
        if (p / "Code" / "odds_backtest").is_dir() and (p / "gambling").is_dir():
            return p
    for p in [_SCRIPT_DIR, *_SCRIPT_DIR.parents]:
        if (p / "gambling" / "odds_data").is_dir():
            return p
    return _SCRIPT_DIR.parents[2]


_PROJECT_ROOT = _find_project_root()

INPUT_CSV = _PROJECT_ROOT / "gambling" / "matched_predictions_with_odds.csv"
OUTPUT_DIR = _PROJECT_ROOT / "gambling" / "backtest_outputs"

# "raw" → implied prob from odds; "fair" → no-vig fair prob from merged file
PROB_MODE: str = "fair"  # "fair" | "raw"

EDGE_HAIRCUT = 0.02

# Bin edges: buckets are (edges[i], edges[i+1]] for i=0 with edge>0, else (edges[i], edges[i+1]]
EDGE_BUCKETS = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 1.00]

EDGE_THRESHOLDS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]

# Merged file column names (from merge_model_with_odds.py)
COL_PROB = "prob"
COL_CORRECT = "correct"
COL_ODDS = "p1_decimal_odds"
COL_MATCH = "match_id"
COL_RAW = "market_prob_raw_p1"
COL_FAIR = "market_prob_fair_p1"


def main() -> None:
    print("=" * 72)
    print("EDGE BUCKET RETURNS (flat 1-unit stakes, no compounding)")
    print("=" * 72)
    print(f"Input:      {INPUT_CSV}")
    print(f"PROB_MODE:  {PROB_MODE}")
    print(f"EDGE_HAIRCUT: {EDGE_HAIRCUT}")
    print()

    if not INPUT_CSV.is_file():
        print(f"[ERROR] File not found: {INPUT_CSV}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    n_load = len(df)

    market_col = COL_FAIR if PROB_MODE.lower() == "fair" else COL_RAW
    if market_col not in df.columns:
        print(f"[ERROR] Missing column {market_col!r} (PROB_MODE={PROB_MODE})", file=sys.stderr)
        sys.exit(1)

    req = [COL_MATCH, COL_PROB, COL_CORRECT, COL_ODDS, market_col]
    before = len(df)
    df = df.dropna(subset=req)
    print(f"[1] Dropped {before - len(df)} rows with missing required columns")

    df[COL_ODDS] = pd.to_numeric(df[COL_ODDS], errors="coerce")
    df[COL_PROB] = pd.to_numeric(df[COL_PROB], errors="coerce")
    df[market_col] = pd.to_numeric(df[market_col], errors="coerce")
    df[COL_CORRECT] = pd.to_numeric(df[COL_CORRECT], errors="coerce").fillna(0).astype(int)

    before = len(df)
    df = df[df[COL_ODDS] > 1.0]
    print(f"[2] Dropped {before - len(df)} rows with {COL_ODDS} <= 1")

    df["adjusted_model_prob"] = (df[COL_PROB] - EDGE_HAIRCUT).clip(0.0, 1.0)
    before = len(df)
    df = df[(df["adjusted_model_prob"] > 0.0) & (df["adjusted_model_prob"] < 1.0)]
    print(
        f"[3] Dropped {before - len(df)} rows with adjusted_model_prob <= 0 or >= 1 "
        f"(after clip)"
    )

    before = len(df)
    df = df[(df[market_col] >= 0.0) & (df[market_col] <= 1.0)]
    print(f"[4] Dropped {before - len(df)} rows with market_prob outside [0, 1]")

    df["edge"] = df["adjusted_model_prob"] - df[market_col]

    dup_n = df[COL_MATCH].duplicated(keep=False).sum()
    if dup_n > 0:
        print(
            f"[WARN] {df[COL_MATCH].duplicated().sum()} duplicate match_id rows; "
            f"keeping the row with highest edge each."
        )
        df = df.sort_values("edge", ascending=False).drop_duplicates(subset=[COL_MATCH], keep="first")

    before = len(df)
    df = df[df["edge"] > 0].copy()
    print(f"[5] Kept {len(df)} rows with edge > 0 (from {n_load} loaded, after cleaning)")
    print()

    # 1 unit on p1: win → odds - 1, lose → -1
    df["realized_return"] = np.where(
        df[COL_CORRECT].eq(1), df[COL_ODDS] - 1.0, -1.0
    )

    # --- Buckets ---
    bucket_rows: list[dict] = []
    for i in range(len(EDGE_BUCKETS) - 1):
        lo, hi = EDGE_BUCKETS[i], EDGE_BUCKETS[i + 1]
        if i == 0:
            sub = df[(df["edge"] > 0) & (df["edge"] <= hi)]
            label = f"(0, {hi:.2f}]"
        else:
            sub = df[(df["edge"] > lo) & (df["edge"] <= hi)]
            label = "(0.10, 1.00]" if hi >= 1.0 else f"({lo:.2f}, {hi:.2f}]"
        if len(sub) == 0:
            bucket_rows.append(
                {
                    "edge_bucket": label,
                    "n_bets": 0,
                    "wins": 0,
                    "losses": 0,
                    "win_rate": np.nan,
                    "avg_adjusted_model_prob": np.nan,
                    "avg_market_prob": np.nan,
                    "avg_decimal_odds": np.nan,
                    "avg_edge": np.nan,
                    "avg_realized_return": np.nan,
                    "total_profit_1unit": 0.0,
                    "roi_per_bet": np.nan,
                    "std_realized_return": np.nan,
                }
            )
            continue

        ret = sub["realized_return"]
        wins = int((sub[COL_CORRECT] == 1).sum())
        losses = len(sub) - wins
        bucket_rows.append(
            {
                "edge_bucket": label,
                "n_bets": len(sub),
                "wins": wins,
                "losses": losses,
                "win_rate": wins / len(sub),
                "avg_adjusted_model_prob": sub["adjusted_model_prob"].mean(),
                "avg_market_prob": sub[market_col].mean(),
                "avg_decimal_odds": sub[COL_ODDS].mean(),
                "avg_edge": sub["edge"].mean(),
                "avg_realized_return": ret.mean(),
                "total_profit_1unit": ret.sum(),
                "roi_per_bet": ret.mean(),
                "std_realized_return": ret.std(ddof=1) if len(sub) > 1 else np.nan,
            }
        )

    bucket_df = pd.DataFrame(bucket_rows)

    # --- Thresholds (cumulative: edge >= t) ---
    thresh_rows: list[dict] = []
    for t in EDGE_THRESHOLDS:
        sub = df[df["edge"] >= t]
        if len(sub) == 0:
            thresh_rows.append(
                {
                    "min_edge_threshold": t,
                    "n_bets": 0,
                    "avg_return_per_bet": np.nan,
                    "total_profit_1unit": 0.0,
                    "win_rate": np.nan,
                    "avg_odds": np.nan,
                    "avg_edge": np.nan,
                }
            )
            continue
        ret = sub["realized_return"]
        thresh_rows.append(
            {
                "min_edge_threshold": t,
                "n_bets": len(sub),
                "avg_return_per_bet": ret.mean(),
                "total_profit_1unit": ret.sum(),
                "win_rate": (sub[COL_CORRECT] == 1).mean(),
                "avg_odds": sub[COL_ODDS].mean(),
                "avg_edge": sub["edge"].mean(),
            }
        )

    thresh_df = pd.DataFrame(thresh_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out1 = OUTPUT_DIR / "edge_bucket_summary.csv"
    out2 = OUTPUT_DIR / "edge_threshold_summary.csv"
    bucket_df.to_csv(out1, index=False)
    thresh_df.to_csv(out2, index=False)

    # --- Terminal output ---
    print("--- Edge buckets (flat 1-unit, edge > 0 only) ---")
    print(bucket_df.to_string(index=False))
    print()
    print("--- Cumulative by minimum edge (edge >= threshold) ---")
    print(thresh_df.to_string(index=False))
    print()

    valid_t = thresh_df[thresh_df["n_bets"] > 0].copy()
    pos = valid_t[valid_t["avg_return_per_bet"] > 0]
    if pos.empty:
        print("[Insight] No threshold in the grid has positive average return per bet.")
    else:
        smallest = pos.sort_values("min_edge_threshold").iloc[0]
        print(
            f"[Insight] Smallest threshold with positive avg return per bet: "
            f">= {smallest['min_edge_threshold']:.2f} "
            f"(avg return {smallest['avg_return_per_bet']:.4f}, n={int(smallest['n_bets'])})"
        )
        best_avg = valid_t.sort_values("avg_return_per_bet", ascending=False).iloc[0]
        print(
            f"[Insight] Highest avg return per bet: threshold >= {best_avg['min_edge_threshold']:.2f} "
            f"(avg return {best_avg['avg_return_per_bet']:.4f}, n={int(best_avg['n_bets'])})"
        )
        best_profit = valid_t.sort_values("total_profit_1unit", ascending=False).iloc[0]
        print(
            f"[Insight] Highest total profit (1-unit stakes): threshold >= {best_profit['min_edge_threshold']:.2f} "
            f"(total {best_profit['total_profit_1unit']:.2f}, n={int(best_profit['n_bets'])})"
        )

    print()
    print(f"Wrote: {out1}")
    print(f"Wrote: {out2}")
    print("=" * 72)


if __name__ == "__main__":
    main()
