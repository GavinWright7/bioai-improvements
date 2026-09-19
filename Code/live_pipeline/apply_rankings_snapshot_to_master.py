#!/usr/bin/env python3
"""
Leakage-safe rankings enrichment: join each match to the latest ATP snapshot
dated on or before the day before the match.

Reads:  cleaned_data/updated_cleaned_matches.csv
Writes: cleaned_data/master_data.csv
"""
from __future__ import annotations

import re
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = BASE_DIR / "data" / "rankings_snapshots"
MASTER_INPUT = BASE_DIR / "cleaned_data" / "updated_cleaned_matches.csv"
MASTER_OUTPUT = BASE_DIR / "cleaned_data" / "master_data.csv"
UNMATCHED_OUT = SNAPSHOT_DIR / "rankings_snapshot_unmatched_names.csv"

SNAPSHOT_GLOB = "atp_rankings_snapshot_*.csv"


def strip_match_display_name(name: str) -> str:
    s = str(name or "").strip()
    if not s:
        return s
    s = re.sub(r"\s+(RET|WO|W/O)\s*$", "", s, flags=re.I)
    s = re.sub(r"(\s+\d+[-–]\d+)+\s*$", "", s)
    s = re.sub(r"(\s+\d+)+\s*$", "", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_player_name(name: str) -> str:
    s = strip_match_display_name(name)
    if not s or s.lower() == "nan":
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = s.replace(".", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def first_initial_and_last(norm: str) -> tuple[str, str] | None:
    parts = norm.split()
    if len(parts) < 2:
        return None
    initial = parts[0][0] if parts[0] else ""
    last = parts[-1]
    if not initial or not last:
        return None
    return initial, last


def parse_tourney_date(val) -> datetime | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        iv = int(round(float(val)))
        s = str(iv)
        if len(s) == 8:
            return datetime.strptime(s, "%Y%m%d")
        return None
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    if re.fullmatch(r"\d{8}\.0", s):
        s = s[:-2]
    if re.fullmatch(r"\d{8}", s):
        return datetime.strptime(s, "%Y%m%d")
    return None


def snapshot_date_from_path(p: Path) -> datetime | None:
    m = re.search(r"atp_rankings_snapshot_(\d{8})\.csv$", p.name, re.I)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d")


def discover_snapshots() -> list[tuple[datetime, Path]]:
    if not SNAPSHOT_DIR.is_dir():
        return []
    out: list[tuple[datetime, Path]] = []
    for p in sorted(SNAPSHOT_DIR.glob(SNAPSHOT_GLOB)):
        d = snapshot_date_from_path(p)
        if d is not None:
            out.append((d, p))
    out.sort(key=lambda x: x[0])
    return out


def resolve_snapshot_path(match_day: datetime, snapshots: list[tuple[datetime, Path]]) -> Path | None:
    """Latest snapshot with file date <= (match_day - 1 day)."""
    target = (match_day - timedelta(days=1)).date()
    candidates = [(d, p) for d, p in snapshots if d.date() <= target]
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def load_snapshot_dataframe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def build_lookups(df: pd.DataFrame) -> tuple[dict[str, dict], dict[tuple[str, str], list[dict]]]:
    exact: dict[str, dict] = {}
    fb: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for _, row in df.iterrows():
        pn = str(row.get("player_name", "") or "")
        n = normalize_player_name(pn)
        if not n:
            continue
        rec = {
            "player_name": row.get("player_name", ""),
            "official_rank": row.get("official_rank", ""),
            "official_points": row.get("official_points", ""),
            "age": row.get("age", pd.NA),
        }
        exact[n] = rec
        sig = first_initial_and_last(n)
        if sig:
            fb[sig].append(rec)
    return exact, fb


def lookup_player(
    display_name: str,
    exact: dict[str, dict],
    fb: dict[tuple[str, str], list[dict]],
) -> tuple[dict | None, str]:
    n = normalize_player_name(display_name)
    if not n:
        return None, "no_match"
    if n in exact:
        return exact[n], "exact_snapshot_name_match"
    sig = first_initial_and_last(n)
    if not sig:
        return None, "no_match"
    cands = fb.get(sig, [])
    if len(cands) != 1:
        return None, "no_match"
    return cands[0], "fallback_initial_lastname_match"


def points_to_numeric(v) -> float | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = re.sub(r"[^\d]", "", str(v).strip())
    if not s:
        return None
    try:
        return float(int(s))
    except ValueError:
        return None


def age_to_numeric(v) -> float | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def rank_for_master(v) -> object:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return pd.NA
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return pd.NA
    if s.upper().endswith("T"):
        return s
    try:
        return float(s)
    except ValueError:
        return s


def combine_row_method(wm: str, lm: str) -> str:
    if wm == "no_match" or lm == "no_match":
        return "no_match"
    if wm == "exact_snapshot_name_match" and lm == "exact_snapshot_name_match":
        return "exact_snapshot_name_match"
    return "fallback_initial_lastname_match"


def _blank(v) -> bool:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return True
    return str(v).strip() in ("", "nan")


def main() -> None:
    if not MASTER_INPUT.exists():
        print(f"ERROR: Missing master input: {MASTER_INPUT}", file=sys.stderr)
        sys.exit(1)

    snapshots = discover_snapshots()
    if not snapshots:
        print(
            f"ERROR: No snapshots under {SNAPSHOT_DIR} ({SNAPSHOT_GLOB}). "
            "Run scrape_daily_rankings_snapshot.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.read_csv(MASTER_INPUT, low_memory=False)
    n = len(df)

    rank_cols = ["winner_rank", "loser_rank", "winner_rank_points", "loser_rank_points"]
    age_cols = ["winner_age", "loser_age"]
    for c in rank_cols + age_cols:
        if c not in df.columns:
            df[c] = pd.NA

    prov_cols = [
        "winner_rankings_snapshot_date",
        "loser_rankings_snapshot_date",
        "winner_rankings_snapshot_file",
        "loser_rankings_snapshot_file",
        "rankings_join_method",
    ]
    for c in prov_cols:
        if c not in df.columns:
            df[c] = pd.NA

    w_rank = df["winner_rank"].tolist()
    w_pts = df["winner_rank_points"].tolist()
    l_rank = df["loser_rank"].tolist()
    l_pts = df["loser_rank_points"].tolist()
    w_age = df["winner_age"].tolist()
    l_age = df["loser_age"].tolist()

    prov_w_date: list[object] = [pd.NA] * n
    prov_l_date: list[object] = [pd.NA] * n
    prov_w_file: list[object] = [pd.NA] * n
    prov_l_file: list[object] = [pd.NA] * n
    join_method: list[str] = ["no_match"] * n

    lookup_cache: dict[Path, tuple[dict[str, dict], dict[tuple[str, str], list[dict]]]] = {}
    paths_used: set[str] = set()

    w_exact = w_fb = l_exact = l_fb = w_miss = l_miss = 0
    unmatched_rows: list[dict] = []

    for pos in range(n):
        row = df.iloc[pos]
        tdt = parse_tourney_date(row.get("tourney_date"))
        if tdt is None:
            w_miss += 1
            l_miss += 1
            w_rank[pos] = pd.NA
            w_pts[pos] = pd.NA
            l_rank[pos] = pd.NA
            l_pts[pos] = pd.NA
            join_method[pos] = "no_match"
            for side, pname in (
                ("winner", row.get("winner_name", "")),
                ("loser", row.get("loser_name", "")),
            ):
                unmatched_rows.append(
                    {
                        "player_name": pname,
                        "side": side,
                        "tourney_name": row.get("tourney_name", ""),
                        "tourney_date": row.get("tourney_date", ""),
                        "desired_snapshot_date": "",
                        "reason": "invalid_tourney_date",
                    }
                )
            continue

        desired_snap = (tdt - timedelta(days=1)).date().isoformat()
        path = resolve_snapshot_path(tdt, snapshots)
        if path is None:
            w_miss += 1
            l_miss += 1
            w_rank[pos] = pd.NA
            w_pts[pos] = pd.NA
            l_rank[pos] = pd.NA
            l_pts[pos] = pd.NA
            join_method[pos] = "no_match"
            for side, pname in (
                ("winner", row.get("winner_name", "")),
                ("loser", row.get("loser_name", "")),
            ):
                unmatched_rows.append(
                    {
                        "player_name": pname,
                        "side": side,
                        "tourney_name": row.get("tourney_name", ""),
                        "tourney_date": row.get("tourney_date", ""),
                        "desired_snapshot_date": desired_snap,
                        "reason": "no_snapshot_on_or_before_target",
                    }
                )
            continue

        paths_used.add(str(path.resolve()))
        if path not in lookup_cache:
            lookup_cache[path] = build_lookups(load_snapshot_dataframe(path))
        exact, fb = lookup_cache[path]
        snap_dt = snapshot_date_from_path(path)
        snap_label = snap_dt.strftime("%Y-%m-%d") if snap_dt else ""
        rel_file = str(path.relative_to(BASE_DIR))

        prov_w_date[pos] = snap_label
        prov_l_date[pos] = snap_label
        prov_w_file[pos] = rel_file
        prov_l_file[pos] = rel_file

        wrec, wm = lookup_player(str(row.get("winner_name", "")), exact, fb)
        lrec, lm = lookup_player(str(row.get("loser_name", "")), exact, fb)

        if wm == "exact_snapshot_name_match":
            w_exact += 1
        elif wm == "fallback_initial_lastname_match":
            w_fb += 1
        else:
            w_miss += 1
            w_rank[pos] = pd.NA
            w_pts[pos] = pd.NA
            unmatched_rows.append(
                {
                    "player_name": row.get("winner_name", ""),
                    "side": "winner",
                    "tourney_name": row.get("tourney_name", ""),
                    "tourney_date": row.get("tourney_date", ""),
                    "desired_snapshot_date": desired_snap,
                    "reason": "player_not_in_snapshot",
                }
            )

        if lm == "exact_snapshot_name_match":
            l_exact += 1
        elif lm == "fallback_initial_lastname_match":
            l_fb += 1
        else:
            l_miss += 1
            l_rank[pos] = pd.NA
            l_pts[pos] = pd.NA
            unmatched_rows.append(
                {
                    "player_name": row.get("loser_name", ""),
                    "side": "loser",
                    "tourney_name": row.get("tourney_name", ""),
                    "tourney_date": row.get("tourney_date", ""),
                    "desired_snapshot_date": desired_snap,
                    "reason": "player_not_in_snapshot",
                }
            )

        join_method[pos] = combine_row_method(wm, lm)

        if wrec:
            w_rank[pos] = rank_for_master(wrec.get("official_rank"))
            pts = points_to_numeric(wrec.get("official_points"))
            w_pts[pos] = pts if pts is not None else pd.NA
            wa = age_to_numeric(wrec.get("age"))
            if wa is not None and _blank(w_age[pos]):
                w_age[pos] = wa

        if lrec:
            l_rank[pos] = rank_for_master(lrec.get("official_rank"))
            pts = points_to_numeric(lrec.get("official_points"))
            l_pts[pos] = pts if pts is not None else pd.NA
            la = age_to_numeric(lrec.get("age"))
            if la is not None and _blank(l_age[pos]):
                l_age[pos] = la

    df["winner_rank"] = w_rank
    df["winner_rank_points"] = w_pts
    df["loser_rank"] = l_rank
    df["loser_rank_points"] = l_pts
    df["winner_age"] = w_age
    df["loser_age"] = l_age
    df["winner_rankings_snapshot_date"] = prov_w_date
    df["loser_rankings_snapshot_date"] = prov_l_date
    df["winner_rankings_snapshot_file"] = prov_w_file
    df["loser_rankings_snapshot_file"] = prov_l_file
    df["rankings_join_method"] = join_method

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    MASTER_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MASTER_OUTPUT, index=False)
    pd.DataFrame(unmatched_rows).to_csv(UNMATCHED_OUT, index=False)

    print(f"total match rows: {n}")
    print(f"unique snapshot files used: {len(paths_used)}")
    print(f"winner exact matches: {w_exact}")
    print(f"loser exact matches: {l_exact}")
    print(f"winner fallback matches: {w_fb}")
    print(f"loser fallback matches: {l_fb}")
    print(f"winner unmatched: {w_miss}")
    print(f"loser unmatched: {l_miss}")
    print(f"output file path: {MASTER_OUTPUT}")
    print(f"unmatched names: {UNMATCHED_OUT}")


if __name__ == "__main__":
    main()
