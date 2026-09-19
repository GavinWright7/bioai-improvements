"""
load_and_clean_odds.py
=============================================================
Load all tennis-data.co.uk style .xlsx files from gambling/odds_data/,
standardize columns, select bookmaker odds by priority, and save a single
clean CSV for model backtesting.

Output columns:
  date, year, surface, winner_name, loser_name, winner_odds, loser_odds, odds_source

Run (from project root):
  python Code/odds_backtest/load_and_clean_odds.py
=============================================================
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths: walk up from this file until gambling/odds_data exists (robust to layout)
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
ODDS_DATA_DIR = _PROJECT_ROOT / "gambling" / "odds_data"
OUTPUT_CSV = _PROJECT_ROOT / "gambling" / "normalized_odds_matches.csv"

# Book priority: (odds_source label, winner decimal col, loser decimal col) — internal names after standardize
BOOK_PRIORITY: list[tuple[str, str, str]] = [
    ("pinnacle", "psw", "psl"),
    ("bet365", "b365w", "b365l"),
    ("max", "maxw", "maxl"),
    ("avg", "avgw", "avgl"),
]

# Map normalized header keys → canonical internal names used above
# Keys are alphanumeric-only lowercase (see _norm_header).
_COLUMN_ALIASES: dict[str, list[str]] = {
    "date": ["date", "matchdate", "startdate", "tourneydate"],
    "surface": ["surface", "court", "courtsurface"],
    "winner_name": ["winner", "winnername", "wname"],
    "loser_name": ["loser", "losername", "lname"],
    "psw": ["psw", "psw", "pinw", "pinnaclew", "pinnaclew"],
    "psl": ["psl", "psl", "pinl", "pinnaclel", "pinnaclel"],
    "b365w": ["b365w", "b365w", "bet365w", "bet365w"],
    "b365l": ["b365l", "b365l", "bet365l", "bet365l"],
    "maxw": ["maxw", "maxw"],
    "maxl": ["maxl", "maxl"],
    "avgw": ["avgw", "avgw"],
    "avgl": ["avgl", "avgl"],
}


def _norm_header(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(col).strip().lower())


def _map_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Rename columns to canonical internal names where recognized."""
    raw_names = list(df.columns)
    norm_to_orig = {_norm_header(c): c for c in df.columns}
    rename_map: dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        for a in aliases:
            na = _norm_header(a)
            if na in norm_to_orig and norm_to_orig[na] not in rename_map.values():
                rename_map[norm_to_orig[na]] = canonical
                break
    out = df.rename(columns=rename_map)
    return out, raw_names


def _safe_float(x: object) -> float:
    v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    if pd.isna(v):
        return np.nan
    return float(v)


def _pick_odds_pair(row: pd.Series) -> tuple[float, float, str]:
    """First valid (decimal>1) book in priority order."""
    for label, wc, lc in BOOK_PRIORITY:
        if wc not in row.index or lc not in row.index:
            continue
        w = _safe_float(row.get(wc, np.nan))
        l = _safe_float(row.get(lc, np.nan))
        if np.isfinite(w) and np.isfinite(l) and w > 1.0 and l > 1.0:
            return w, l, label
    return np.nan, np.nan, ""


def _normalize_display_name(name: object) -> str:
    """Lowercase, strip, collapse internal whitespace (tennis-data display)."""
    if pd.isna(name):
        return ""
    s = str(name).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def load_all_xlsx(odds_dir: Path) -> tuple[pd.DataFrame, list[str] | None]:
    """Load first sheet of each .xlsx; return combined df and raw columns from first file."""
    files = sorted(odds_dir.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No .xlsx files in: {odds_dir}")

    parts: list[pd.DataFrame] = []
    first_raw_cols: list[str] | None = None

    for fp in files:
        try:
            raw = pd.read_excel(fp, sheet_name=0, engine="openpyxl")
        except Exception as e:
            print(f"[WARNING] Skip {fp.name}: {e}", file=sys.stderr)
            continue
        if first_raw_cols is None:
            first_raw_cols = list(raw.columns)
        raw["source_file"] = fp.name
        parts.append(raw)

    if not parts:
        raise RuntimeError("No Excel files could be read successfully.")

    combined = pd.concat(parts, ignore_index=True, sort=False)
    return combined, first_raw_cols


def main() -> None:
    print("=" * 70)
    print("LOAD AND CLEAN TENNIS ODDS (.xlsx → normalized CSV)")
    print("=" * 70)

    if not ODDS_DATA_DIR.is_dir():
        sys.exit(f"[ERROR] Odds folder not found: {ODDS_DATA_DIR}")

    print(f"\nOdds directory: {ODDS_DATA_DIR}")
    print(f"Output file:    {OUTPUT_CSV}\n")

    # --- Load ---
    combined, first_file_raw_cols = load_all_xlsx(ODDS_DATA_DIR)
    rows_loaded = len(combined)
    print(f"[1] Total rows loaded (all files, first sheet each): {rows_loaded:,}")
    if first_file_raw_cols is not None:
        print(f"\n    Column names BEFORE cleaning (first file only):")
        print(f"    {first_file_raw_cols}\n")

    # --- Standardize column names ---
    mapped, _ = _map_columns(combined)
    print(f"    Column names AFTER rename mapping (combined, unique order):")
    print(f"    {list(mapped.columns)}\n")

    required_core = ["winner_name", "loser_name", "date"]
    missing = [c for c in required_core if c not in mapped.columns]
    if missing:
        sys.exit(
            f"[ERROR] After mapping, missing required columns: {missing}\n"
            f"  Available: {list(mapped.columns)}"
        )

    # --- Build working frame ---
    work = pd.DataFrame()
    work["date"] = pd.to_datetime(mapped["date"], errors="coerce")
    if work["date"].notna().sum() == 0:
        alt = pd.to_numeric(mapped["date"], errors="coerce")
        work["date"] = pd.to_datetime(alt.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")

    work["year"] = work["date"].dt.year
    work["surface"] = mapped.get("surface", pd.Series([""] * len(mapped))).astype(str).str.lower().str.strip()
    work["winner_name"] = mapped["winner_name"].map(_normalize_display_name)
    work["loser_name"] = mapped["loser_name"].map(_normalize_display_name)

    # Attach mapped columns needed for odds pick (same index)
    for col in ["psw", "psl", "b365w", "b365l", "maxw", "maxl", "avgw", "avgl"]:
        if col in mapped.columns:
            work[col] = mapped[col]
        else:
            work[col] = np.nan

    rows_before_odds = len(work)

    # --- Select odds ---
    w_odds: list[float] = []
    l_odds: list[float] = []
    src: list[str] = []
    for _, row in work.iterrows():
        wo, lo, s = _pick_odds_pair(row)
        w_odds.append(wo)
        l_odds.append(lo)
        src.append(s)

    work["winner_odds"] = w_odds
    work["loser_odds"] = l_odds
    work["odds_source"] = src

    # Drop rows with no valid book
    dropped_no_book = int((work["odds_source"] == "").sum())
    work = work[work["odds_source"] != ""].copy()

    # Drop invalid decimal odds
    bad = (
        work["winner_odds"].isna()
        | work["loser_odds"].isna()
        | (work["winner_odds"] <= 1.0)
        | (work["loser_odds"] <= 1.0)
    )
    dropped_bad_odds = int(bad.sum())
    work = work[~bad].copy()

    # Drop rows missing essential fields
    bad_names = (work["winner_name"] == "") | (work["loser_name"] == "")
    dropped_names = int(bad_names.sum())
    work = work[~bad_names].copy()

    dropped_date = int(work["date"].isna().sum())
    work = work[work["date"].notna()].copy()

    work["year"] = work["year"].astype("Int64")

    # Final exact column order
    out = work[
        ["date", "year", "surface", "winner_name", "loser_name", "winner_odds", "loser_odds", "odds_source"]
    ].copy()

    final_rows = len(out)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)

    # --- Validation / debug ---
    print("[2] Cleaning summary")
    print(f"    Rows loaded (raw combined):     {rows_loaded:,}")
    print(f"    Dropped (no valid book pair):   {dropped_no_book:,}")
    print(f"    Dropped (odds <= 1 or missing): {dropped_bad_odds:,}")
    print(f"    Dropped (empty winner/loser):   {dropped_names:,}")
    print(f"    Dropped (invalid date):         {dropped_date:,}")
    print(f"    Final usable rows:              {final_rows:,}")

    print("\n[3] odds_source distribution (final)")
    print(out["odds_source"].value_counts(dropna=False).to_string())

    print("\n[4] First 5 rows after cleaning:")
    print(out.head(5).to_string())
    print(f"\n[5] Saved: {OUTPUT_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()
