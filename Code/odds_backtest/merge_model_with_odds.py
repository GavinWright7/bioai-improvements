"""
merge_model_with_odds.py
============================================================
Join model predictions (one row per match, p1 vs p2) with normalized
historical odds (winner vs loser) by year + standardized names.

Outputs:
  gambling/matched_predictions_with_odds.csv
  gambling/unmatched_model_examples.csv

Run (from project root):
  python Code/odds_backtest/merge_model_with_odds.py
  python Code/odds_backtest/merge_model_with_odds.py --model path/to/preds.csv
============================================================
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent


def _find_project_root() -> Path:
    """Repo root = directory that contains both Code/odds_backtest/ and gambling/.

    Do not use gambling/odds_data alone: a stray ~/Desktop/gambling/odds_data would
    steal the root and break paths like .../BIOAI improvements/gambling/...
    """
    for p in [_SCRIPT_DIR, *_SCRIPT_DIR.parents]:
        if (p / "Code" / "odds_backtest").is_dir() and (p / "gambling").is_dir():
            return p
    for p in [_SCRIPT_DIR, *_SCRIPT_DIR.parents]:
        if (p / "gambling" / "odds_data").is_dir():
            return p
    return _SCRIPT_DIR.parents[2]


_PROJECT_ROOT = _find_project_root()

# Same location as gambling/odds/odds_backtest/generate_real_predictions.py OUTPUT_FILE
DEFAULT_MODEL_CSV = _PROJECT_ROOT / "gambling" / "odds" / "odds_backtest" / "real_predictions.csv"
DEFAULT_ODDS_CSV = _PROJECT_ROOT / "gambling" / "normalized_odds_matches.csv"
OUT_MATCHED = _PROJECT_ROOT / "gambling" / "matched_predictions_with_odds.csv"
OUT_UNMATCHED = _PROJECT_ROOT / "gambling" / "unmatched_model_examples.csv"


def normalize_raw_name(name: object) -> str:
    """Lightweight cleanup before building a join key (full names vs tennis-data abbreviations)."""
    if pd.isna(name):
        return ""
    s = str(name).strip().lower()
    s = s.replace(",", " ")
    s = s.replace("'", "").replace("’", "").replace("`", "")
    s = re.sub(r"\s+", " ", s)
    # Normalize spaced hyphens to single hyphen (keeps ramos-vinolas style tokens)
    s = re.sub(r"\s*-\s*", "-", s)
    # Remove stray punctuation except word chars, space, dot, hyphen
    s = re.sub(r"[^\w\s\.\-]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_initial_token(tok: str) -> bool:
    """Trailing token is treated as initial(s) if short alphabetic (e.g. s, sw, r)."""
    if not tok.isalpha():
        return False
    return len(tok) <= 3


def to_match_key(normalized: str) -> str:
    """Comparable key: roughly 'surname … + first initial' for both full and abbreviated ATP names."""
    if not normalized:
        return ""
    # Strip periods without inserting spaces so "s.w." → "sw" (not "s w")
    s = re.sub(r"\.", "", normalized)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = [t for t in s.split() if t]
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0].lower()

    last = tokens[-1]
    if _is_initial_token(last):
        surname = " ".join(tokens[:-1])
        initial = last[0]
        return f"{surname} {initial}".lower().strip()

    given = tokens[0]
    rest = tokens[1:]
    initial = given[0] if given else "?"
    if len(rest) >= 2:
        surname = f"{rest[-2]} {rest[-1]}"
    else:
        surname = rest[0]
    return f"{surname} {initial}".lower().strip()


def player_match_key(raw_name: object) -> str:
    """normalize_raw_name → to_match_key."""
    return to_match_key(normalize_raw_name(raw_name))


def _require_cols(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[ERROR] {label} missing columns: {missing}", file=sys.stderr)
        print(f"        Have: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge model predictions with normalized odds.")
    ap.add_argument(
        "--model",
        type=Path,
        default=Path(os.environ.get("MODEL_PREDICTIONS_CSV", DEFAULT_MODEL_CSV)),
        help="Model predictions CSV (match_id, p1_name, p2_name, year, prob, correct, ...)",
    )
    ap.add_argument(
        "--odds",
        type=Path,
        default=Path(os.environ.get("NORMALIZED_ODDS_CSV", DEFAULT_ODDS_CSV)),
        help="Normalized odds CSV from load_and_clean_odds.py",
    )
    ap.add_argument("--out-matched", type=Path, default=OUT_MATCHED)
    ap.add_argument("--out-unmatched", type=Path, default=OUT_UNMATCHED)
    args = ap.parse_args()

    print("=" * 70)
    print("MERGE MODEL PREDICTIONS WITH NORMALIZED ODDS")
    print("=" * 70)
    print(f"Model preds:  {args.model}")
    print(f"Odds:         {args.odds}")
    print(f"Out (match):  {args.out_matched}")
    print(f"Out (miss):   {args.out_unmatched}")
    print()

    if not args.model.is_file():
        print(f"[ERROR] Model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)
    if not args.odds.is_file():
        print(f"[ERROR] Odds file not found: {args.odds}", file=sys.stderr)
        sys.exit(1)

    model = pd.read_csv(args.model)
    odds = pd.read_csv(args.odds)

    _require_cols(
        model,
        ["match_id", "p1_name", "p2_name", "year", "prob", "correct"],
        "Model predictions",
    )
    _require_cols(
        odds,
        ["date", "year", "surface", "winner_name", "loser_name", "winner_odds", "loser_odds", "odds_source"],
        "Normalized odds",
    )

    n_model_raw = len(model)
    # One prediction row per match_id (keep first row if duplicates exist)
    if model["match_id"].duplicated().any():
        n_dup = model["match_id"].duplicated().sum()
        print(f"[WARN] Model has {n_dup} duplicate match_id rows — keeping first occurrence each.")
        model = model.drop_duplicates(subset=["match_id"], keep="first").reset_index(drop=True)

    n_model = len(model)
    print(f"Column names (model): {list(model.columns)}")
    print(f"Column names (odds):  {list(odds.columns)}")
    print()

    # Match keys bridge full names (model) ↔ tennis-data abbreviations (odds)
    model = model.copy()
    odds = odds.copy()
    model["p1_match_key"] = model["p1_name"].map(player_match_key)
    model["p2_match_key"] = model["p2_name"].map(player_match_key)
    if "surface" in model.columns:
        model["surf_norm"] = model["surface"].astype(str).str.lower().str.strip()
    else:
        model["surf_norm"] = ""

    odds["winner_match_key"] = odds["winner_name"].map(player_match_key)
    odds["loser_match_key"] = odds["loser_name"].map(player_match_key)
    odds["odds_surface_norm"] = odds["surface"].astype(str).str.lower().str.strip()

    # Drop model rows with empty keys
    bad_key = (model["p1_match_key"] == "") | (model["p2_match_key"] == "")
    if bad_key.any():
        print(f"[WARN] Dropping {bad_key.sum()} model rows with empty match keys.")
        model = model.loc[~bad_key].copy()

    odds["year"] = pd.to_numeric(odds["year"], errors="coerce").astype("Int64")
    model["year"] = pd.to_numeric(model["year"], errors="coerce").astype("Int64")
    odds["date"] = pd.to_datetime(odds["date"], errors="coerce")

    # Sort odds so "first" match is earliest date when duplicates exist
    odds = odds.sort_values("date", na_position="last").reset_index(drop=True)
    dup_before = len(odds)
    odds = odds.drop_duplicates(
        subset=["year", "winner_match_key", "loser_match_key"], keep="first"
    ).reset_index(drop=True)
    if len(odds) < dup_before:
        print(
            f"[INFO] Dropped {dup_before - len(odds)} duplicate odds rows "
            f"(same year + winner/loser match keys)."
        )

    # Diagnostics: sample normalized strings and keys
    print("[0] Name normalization samples (first 10 model rows)")
    _samp_m = model.head(10)
    for _, r in _samp_m.iterrows():
        print(
            f"    p1={r['p1_name']!r} → {r['p1_match_key']!r} | "
            f"p2={r['p2_name']!r} → {r['p2_match_key']!r}"
        )
    print("\n[0b] First 10 odds rows (winner/loser → match keys)")
    for _, r in odds.head(10).iterrows():
        print(
            f"    W={r['winner_name']!r} → {r['winner_match_key']!r} | "
            f"L={r['loser_name']!r} → {r['loser_match_key']!r}"
        )
    print()

    # Prefer model surface in output; keep odds surface only for tie-breaking
    odds_merge = odds.drop(columns=["surface"], errors="ignore")

    left_cols = [
        "match_id",
        "surface",
        "year",
        "round",
        "p1_name",
        "p2_name",
        "prob",
        "correct",
        "p1_match_key",
        "p2_match_key",
        "surf_norm",
    ]
    left_cols = [c for c in left_cols if c in model.columns]

    # Forward: p1 = winner, p2 = loser
    mf = model[left_cols].merge(
        odds_merge,
        left_on=["year", "p1_match_key", "p2_match_key"],
        right_on=["year", "winner_match_key", "loser_match_key"],
        how="inner",
        suffixes=("", "_odds"),
    )
    mf["match_orientation"] = "forward"
    if not mf.empty:
        mf["surface_score"] = (mf["surf_norm"] == mf["odds_surface_norm"]).astype(int)
        mf = mf.sort_values(
            ["match_id", "surface_score", "date"],
            ascending=[True, False, True],
        ).drop_duplicates(subset=["match_id"], keep="first")

    # Reversed: p1 = loser, p2 = winner
    mr = model[left_cols].merge(
        odds_merge,
        left_on=["year", "p1_match_key", "p2_match_key"],
        right_on=["year", "loser_match_key", "winner_match_key"],
        how="inner",
        suffixes=("", "_odds"),
    )
    mr["match_orientation"] = "reversed"
    if not mr.empty:
        mr["surface_score"] = (mr["surf_norm"] == mr["odds_surface_norm"]).astype(int)
        mr = mr.sort_values(
            ["match_id", "surface_score", "date"],
            ascending=[True, False, True],
        ).drop_duplicates(subset=["match_id"], keep="first")

    # Prefer forward when both exist (same match_id should only appear in one orientation for a real match)
    fwd_ids = set(mf["match_id"].astype(str)) if not mf.empty else set()
    rev_ids = set(mr["match_id"].astype(str)) if not mr.empty else set()
    overlap = fwd_ids & rev_ids
    if overlap:
        print(
            f"[WARN] {len(overlap)} match_ids matched both orientations "
            f"(likely rematches or name collisions). Using forward orientation."
        )
        mr = mr.loc[~mr["match_id"].astype(str).isin(overlap)].copy()

    all_hits = pd.concat([mf, mr], ignore_index=True)
    if not all_hits.empty:
        all_hits = all_hits.sort_values(["match_id", "date"]).reset_index(drop=True)
        all_hits = all_hits.drop_duplicates(subset=["match_id"], keep="first")

    matched_ids = set(all_hits["match_id"].astype(str)) if not all_hits.empty else set()
    matched = all_hits.copy()

    # Compute market fields
    if not matched.empty:
        wo = pd.to_numeric(matched["winner_odds"], errors="coerce")
        lo = pd.to_numeric(matched["loser_odds"], errors="coerce")
        raw_w = 1.0 / wo
        raw_l = 1.0 / lo
        overround = raw_w + raw_l
        fair_w = raw_w / overround
        fair_l = raw_l / overround

        is_forward = matched["match_orientation"].eq("forward")
        p1_dec = np.where(is_forward, wo, lo)
        mkt_raw_p1 = np.where(is_forward, raw_w, raw_l)
        mkt_fair_p1 = np.where(is_forward, fair_w, fair_l)

        matched["p1_decimal_odds"] = p1_dec
        matched["market_prob_raw_p1"] = mkt_raw_p1
        matched["market_prob_fair_p1"] = mkt_fair_p1
        matched["overround"] = overround
        matched["p1_is_winner"] = is_forward.astype(int)

        # Drop merge helpers (keys retained for audit)
        drop_cols = [
            c
            for c in ("surface_score", "surf_norm", "odds_surface_norm")
            if c in matched.columns
        ]
        matched = matched.drop(columns=drop_cols)

    unmatched = model.loc[~model["match_id"].astype(str).isin(matched_ids)].copy()

    if len(unmatched) > 0:
        print("\n[0c] First 10 unmatched model rows (names + match keys)")
        u = unmatched.head(10)
        for _, r in u.iterrows():
            print(
                f"    {r['p1_name']!r} / {r['p2_name']!r} → "
                f"{r['p1_match_key']!r} vs {r['p2_match_key']!r}  (year={r['year']})"
            )

    # Final duplicate guard on match_id
    if not matched.empty:
        if matched["match_id"].duplicated().any():
            print("[WARN] Duplicate match_id in matched output — keeping first row each.")
            matched = matched.drop_duplicates(subset=["match_id"], keep="first")

    # Stats
    n_matched = len(matched)
    n_unmatched = len(unmatched)
    n_fwd = int((matched["match_orientation"] == "forward").sum()) if n_matched else 0
    n_rev = int((matched["match_orientation"] == "reversed").sum()) if n_matched else 0

    print("[1] Summary")
    print(f"    Total model rows (after dedupe): {n_model}")
    if n_model_raw != n_model:
        print(f"    (raw row count before dedupe:   {n_model_raw})")
    print(f"    Matched rows:                    {n_matched}")
    print(f"    Unmatched model rows:            {n_unmatched}")
    print(f"    Match rate:                      {n_matched / max(n_model, 1):.4f}")
    print(f"    Matched orientation 'forward':   {n_fwd}")
    print(f"    Matched orientation 'reversed':  {n_rev}")
    print(f"    Rows missing odds (unmatched): {n_unmatched}")

    if n_matched:
        print("\n[1b] Sample matched rows (up to 5)")
        show_cols = [
            c
            for c in (
                "match_id",
                "year",
                "p1_name",
                "p2_name",
                "p1_match_key",
                "p2_match_key",
                "winner_name",
                "loser_name",
                "winner_match_key",
                "loser_match_key",
                "match_orientation",
                "date",
            )
            if c in matched.columns
        ]
        print(matched[show_cols].head(5).to_string(index=False))

    if n_matched and "odds_source" in matched.columns:
        print("\n[2] odds_source distribution (matched)")
        print(matched["odds_source"].value_counts().to_string())

    args.out_matched.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(args.out_matched, index=False)
    unmatched.to_csv(args.out_unmatched, index=False)

    print(f"\n[3] Wrote matched:   {args.out_matched}")
    print(f"    Wrote unmatched: {args.out_unmatched}")
    print("=" * 70)


if __name__ == "__main__":
    main()
