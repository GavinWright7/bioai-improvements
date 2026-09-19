#!/usr/bin/env python3
"""
Append live-pipeline completed matches (rankings-enriched) into the master
clean_matches schema and write updated_cleaned_matches.csv without touching
the original master file.

Input:  data/processed/completed_matches_2026_with_dates_rankings.csv
Master: cleaned_data/clean_matches.csv (or cleaned_matches.csv)
Outputs:
  - cleaned_data/updated_cleaned_matches.csv (full merge, newest tourney_date first)
  - cleaned_data/new_rows_converted_preview.csv (new rows only, same date sort)
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

SCRAPED_CSV = BASE_DIR / "data" / "processed" / "completed_matches_2026_with_dates_rankings.csv"
OUTPUT_CSV = BASE_DIR / "cleaned_data" / "updated_cleaned_matches.csv"
PREVIEW_CSV = BASE_DIR / "cleaned_data" / "new_rows_converted_preview.csv"

# Pipeline columns we expect on the scraped file (for validation only).
EXPECTED_SCRAPED_COLS: frozenset[str] = frozenset(
    {
        "scraped_at",
        "tournament_name",
        "source_url",
        "surface",
        "round",
        "winner_name",
        "loser_name",
        "score_text",
        "raw_completed_line",
        "match_date",
        "espn_tournament_url",
        "espn_daily_url",
        "round_raw_espn",
        "espn_match_source",
        "date_source",
        "winner_age",
        "loser_age",
        "winner_rank_points",
        "loser_rank_points",
        "winner_official_rank",
        "loser_official_rank",
        "rankings_date",
        "rankings_source",
    }
)

MASTER_COLS: list[str] = [
    "tourney_id",
    "tourney_name",
    "surface",
    "draw_size",
    "tourney_level",
    "indoor",
    "tourney_date",
    "match_num",
    "winner_id",
    "winner_seed",
    "winner_entry",
    "winner_name",
    "winner_hand",
    "winner_ht",
    "winner_ioc",
    "winner_age",
    "winner_rank",
    "winner_rank_points",
    "loser_id",
    "loser_seed",
    "loser_entry",
    "loser_name",
    "loser_hand",
    "loser_ht",
    "loser_ioc",
    "loser_age",
    "loser_rank",
    "loser_rank_points",
    "score",
    "best_of",
    "round",
    "minutes",
    "w_ace",
    "w_df",
    "w_svpt",
    "w_1stIn",
    "w_1stWon",
    "w_2ndWon",
    "w_SvGms",
    "w_bpSaved",
    "w_bpFaced",
    "l_ace",
    "l_df",
    "l_svpt",
    "l_1stIn",
    "l_1stWon",
    "l_2ndWon",
    "l_SvGms",
    "l_bpSaved",
    "l_bpFaced",
]


def resolve_master_csv() -> Path:
    cleaned_dir = BASE_DIR / "cleaned_data"
    for name in ("clean_matches.csv", "cleaned_matches.csv"):
        p = cleaned_dir / name
        if p.exists():
            return p
    return cleaned_dir / "clean_matches.csv"


def is_na(val: object) -> bool:
    if val is None:
        return True
    try:
        return bool(pd.isna(val))
    except (TypeError, ValueError):
        return False


def scalar_str(val: object) -> str:
    if is_na(val):
        return ""
    s = str(val).strip()
    if s.lower() == "nan":
        return ""
    return s


def blank():
    return pd.NA


def normalize_surface(raw: object) -> str:
    s = scalar_str(raw).lower()
    if not s:
        return ""
    mapping = {
        "hard": "Hard",
        "clay": "Clay",
        "grass": "Grass",
        "carpet": "Carpet",
    }
    return mapping.get(s, scalar_str(raw))


def match_date_to_tourney_date(raw: object) -> str:
    s = scalar_str(raw)
    if not s:
        return ""
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"
    if re.fullmatch(r"\d{8}", s):
        return s
    return s


def normalize_round(raw: object):
    """
    Align with master-style round labels: F, SF, QF, R16, R32, R64, etc.
    Blank input stays blank.
    """
    s = scalar_str(raw)
    if not s:
        return blank()
    m = re.fullmatch(r"(r)(\d+)", s, flags=re.I)
    if m:
        return f"R{m.group(2)}"
    u = s.upper()
    if u in ("F", "SF", "QF"):
        return u
    if re.fullmatch(r"R\d+", u):
        return u
    if re.fullmatch(r"RR\d*", u):
        return u
    return s


def normalize_name_for_dedupe(name: object) -> str:
    """
    Dedupe-only: lowercase, strip, strip accents, remove periods, collapse spaces.
    Not used for display columns in the written CSV.
    """
    s = scalar_str(name)
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = s.replace(".", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def infer_best_of(score_text: object) -> int:
    """
    ATP tour events: default best_of=3 unless the score string suggests a
    best-of-five match (four or more set-score tokens).
    """
    s = scalar_str(score_text)
    if not s:
        return 3
    tokens = re.findall(r"\d+\s*-\s*\d+(?:\(\d+\))?", s)
    if len(tokens) >= 4:
        return 5
    return 3


def to_float(val: object):
    if is_na(val):
        return blank()
    s = scalar_str(val)
    if not s:
        return blank()
    try:
        return float(s)
    except ValueError:
        return blank()


def to_rank_value(val: object):
    if is_na(val):
        return blank()
    s = scalar_str(val)
    if not s:
        return blank()
    if re.search(r"[Tt]$", s) or not re.match(r"^[\d.]+$", re.sub(r"[Tt]$", "", s)):
        return s
    try:
        if "." in s.rstrip("Tt"):
            return float(s.rstrip("Tt"))
        return int(float(s.rstrip("Tt")))
    except ValueError:
        return s


def to_age(val: object):
    if is_na(val):
        return blank()
    s = scalar_str(val)
    if not s:
        return blank()
    try:
        return float(s)
    except ValueError:
        return blank()


def key_str_cell(v: object) -> str:
    if is_na(v):
        return ""
    return str(v).strip()


def key_tourney_date(v: object) -> str:
    """Canonical YYYYMMDD string for dedupe (master often loads dates as floats)."""
    if is_na(v):
        return ""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        iv = int(round(float(v)))
        return str(iv)
    s = scalar_str(v)
    if re.fullmatch(r"\d{8}\.0", s):
        return s[:-2]
    return s


def dedupe_key(row: pd.Series) -> tuple:
    return (
        key_str_cell(row.get("tourney_name")),
        key_tourney_date(row.get("tourney_date")),
        normalize_name_for_dedupe(row.get("winner_name")),
        normalize_name_for_dedupe(row.get("loser_name")),
        key_str_cell(row.get("round")),
        key_str_cell(row.get("score")),
    )


def convert_scraped_row_to_master(row: pd.Series) -> dict:
    """
    Explicit mapping from pipeline columns -> MASTER_COLS.

    tournament_name       -> tourney_name
    surface                 -> surface (normalized)
    match_date              -> tourney_date (YYYYMMDD)
    winner_name             -> winner_name  (display preserved)
    loser_name              -> loser_name
    winner_age              -> winner_age
    loser_age               -> loser_age
    winner_official_rank    -> winner_rank
    loser_official_rank     -> loser_rank
    winner_rank_points      -> winner_rank_points
    loser_rank_points       -> loser_rank_points
    score_text              -> score
    round                   -> round (normalized)
    (inferred)              -> best_of (3 or 5 from score_text)
    """
    out: dict = {c: blank() for c in MASTER_COLS}

    tname = scalar_str(row.get("tournament_name"))
    out["tourney_name"] = tname if tname else blank()

    surf = normalize_surface(row.get("surface"))
    out["surface"] = surf if surf else blank()

    tdate = match_date_to_tourney_date(row.get("match_date"))
    out["tourney_date"] = tdate if tdate else blank()

    wn = scalar_str(row.get("winner_name"))
    ln = scalar_str(row.get("loser_name"))
    out["winner_name"] = wn if wn else blank()
    out["loser_name"] = ln if ln else blank()

    out["winner_age"] = to_age(row.get("winner_age"))
    out["loser_age"] = to_age(row.get("loser_age"))
    out["winner_rank"] = to_rank_value(row.get("winner_official_rank"))
    out["loser_rank"] = to_rank_value(row.get("loser_official_rank"))
    out["winner_rank_points"] = to_float(row.get("winner_rank_points"))
    out["loser_rank_points"] = to_float(row.get("loser_rank_points"))

    score = scalar_str(row.get("score_text"))
    out["score"] = score if score else blank()

    out["round"] = normalize_round(row.get("round"))
    out["best_of"] = infer_best_of(row.get("score_text"))

    return out


def validate_scraped_columns(df: pd.DataFrame) -> None:
    missing = sorted(EXPECTED_SCRAPED_COLS - set(df.columns))
    extra = sorted(set(df.columns) - EXPECTED_SCRAPED_COLS)
    if missing:
        print(f"  [WARN] Scraped file missing expected columns: {missing}")
    if extra:
        print(f"  [INFO] Scraped file has extra columns (ignored for mapping): {extra}")


def sort_matches_newest_first(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chronological order with most recent matches first.
    Uses numeric tourney_date (YYYYMMDD). Rows with missing/unparseable dates sort last.
    """
    sdate = pd.to_numeric(df["tourney_date"], errors="coerce")
    mnum = pd.to_numeric(df["match_num"], errors="coerce")
    sorted_df = (
        df.assign(_sdate=sdate, _mnum=mnum)
        .sort_values(
            by=[
                "_sdate",
                "tourney_name",
                "_mnum",
                "round",
                "winner_name",
                "loser_name",
            ],
            ascending=[False, True, True, True, True, True],
            na_position="last",
            kind="mergesort",
        )
        .drop(columns=["_sdate", "_mnum"])
        .reset_index(drop=True)
    )
    return sorted_df


def main() -> None:
    master_path = resolve_master_csv()
    if not master_path.exists():
        print(f"ERROR: Master CSV not found: {master_path}", file=sys.stderr)
        sys.exit(1)
    if not SCRAPED_CSV.exists():
        print(f"ERROR: Scraped CSV not found: {SCRAPED_CSV}", file=sys.stderr)
        sys.exit(1)

    master = pd.read_csv(master_path, low_memory=False)
    missing_master = [c for c in MASTER_COLS if c not in master.columns]
    if missing_master:
        print(
            f"ERROR: Master missing columns: {missing_master}",
            file=sys.stderr,
        )
        sys.exit(1)

    scraped = pd.read_csv(SCRAPED_CSV, low_memory=False)
    validate_scraped_columns(scraped)

    n_master = len(master)
    n_scraped = len(scraped)

    existing_keys: set[tuple] = set()
    for _, r in master.iterrows():
        existing_keys.add(dedupe_key(r))

    converted: list[dict] = []
    for _, srow in scraped.iterrows():
        converted.append(convert_scraped_row_to_master(srow))

    n_converted = len(converted)

    duplicates_found = 0
    rows_to_append: list[dict] = []
    keys_seen: set[tuple] = set(existing_keys)

    for rec in converted:
        s = pd.Series(rec)
        k = dedupe_key(s)
        if k in keys_seen:
            duplicates_found += 1
            continue
        keys_seen.add(k)
        rows_to_append.append(rec)

    rows_added = len(rows_to_append)

    preview_df = pd.DataFrame(rows_to_append, columns=MASTER_COLS)
    preview_df = sort_matches_newest_first(preview_df)

    out = pd.concat([master[MASTER_COLS], preview_df], ignore_index=True)

    if list(out.columns) != MASTER_COLS:
        print("ERROR: Output column order mismatch.", file=sys.stderr)
        sys.exit(1)

    out = sort_matches_newest_first(out)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    preview_df.to_csv(PREVIEW_CSV, index=False)

    print(f"rows in cleaned_matches.csv ({master_path.name}): {n_master}")
    print(f"rows in new file: {n_scraped}")
    print(f"rows converted into master schema: {n_converted}")
    print(f"duplicates found: {duplicates_found}")
    print(f"rows actually added: {rows_added}")
    print(f"final total rows: {len(out)}")
    print("sort order: tourney_date descending (newest first); missing dates last")
    print(f"output file path: {OUTPUT_CSV}")
    print(f"preview file path: {PREVIEW_CSV}")


if __name__ == "__main__":
    main()
