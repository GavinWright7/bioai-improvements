#!/usr/bin/env python3
"""
ATP rankings enrichment for rows in completed_matches_2026_with_dates.csv only.

One Playwright load of the rankings table, then matches just the winner/loser names
you need. Overview pages are fetched only for those needed players still missing age
(typically a handful, not thousands).

Outputs:
  data/processed/completed_matches_2026_with_dates_rankings.csv
  data/raw/atp_rankings_snapshot.csv
  data/raw/atp_rankings_unmatched_names.csv
"""
from __future__ import annotations

import re
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_CSV = BASE_DIR / "data" / "processed" / "completed_matches_2026_with_dates.csv"
OUTPUT_CSV = BASE_DIR / "data" / "processed" / "completed_matches_2026_with_dates_rankings.csv"
SNAPSHOT_CSV = BASE_DIR / "data" / "raw" / "atp_rankings_snapshot.csv"
UNMATCHED_CSV = BASE_DIR / "data" / "raw" / "atp_rankings_unmatched_names.csv"

RANKINGS_URL = "https://www.atptour.com/en/rankings/singles?rankRange=0-5000"
RANKINGS_SOURCE = "atp_rankings_singles"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def strip_match_display_name(name: str) -> str:
    s = str(name or "").strip()
    if not s:
        return s
    s = re.sub(r"\s+(RET|WO|W/O)\s*$", "", s, flags=re.I)
    s = re.sub(r"(\s+\d+[-–]\d+)+\s*$", "", s)
    s = re.sub(r"(\s+\d+)+\s*$", "", s)
    return norm_space(s)


def normalize_player_name(name: str) -> str:
    """Lowercase, strip accents, remove periods, collapse spaces."""
    s = strip_match_display_name(name)
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


def parse_rank_cell(text: str) -> str:
    t = norm_space(text)
    if not t:
        return ""
    for part in re.split(r"[\s\n]+", t):
        if re.fullmatch(r"\d{1,4}[T]?", part):
            return part
    m = re.search(r"\b(\d{1,4}[T]?)\b", t)
    return m.group(1) if m else ""


def rank_sort_key(rank_s: str) -> tuple[int, int]:
    """Lower is better; trailing T sorts after same integer."""
    s = rank_s.rstrip("Tt")
    try:
        n = int(s)
    except ValueError:
        n = 999999
    tie = 0 if rank_s.upper().endswith("T") else -1
    return (n, tie)


def overview_url_from_row(tr) -> str:
    a = tr.select_one('td.player a[href*="/overview"]')
    if not a or not a.get("href"):
        return ""
    return urljoin("https://www.atptour.com", a["href"])


def display_name_from_row(tr) -> str:
    sp = tr.select_one("td.player span.lastName")
    if sp and sp.get_text(strip=True):
        return norm_space(sp.get_text())
    a = tr.select_one('td.player a[href*="/overview"]')
    if a:
        return norm_space(a.get_text())
    return ""


def age_from_rankings_row(tr) -> str:
    td = tr.select_one("td.age")
    if not td:
        return ""
    t = norm_space(td.get_text())
    return t if t.isdigit() and 15 <= int(t) <= 55 else ""


def points_from_row(tr) -> str:
    td = tr.select_one("td.points")
    if not td:
        return ""
    t = norm_space(td.get_text())
    digits = re.sub(r"[^\d]", "", t)
    return digits


def detect_rankings_date(html: str) -> str:
    m = re.search(r"\b(20\d{2})\s*\.\s*(\d{2})\s*\.\s*(\d{2})\b", html)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return ""


def dismiss_cookies(page) -> None:
    for sel in ("#onetrust-accept-btn-handler", "button:has-text('Accept All')"):
        try:
            loc = page.locator(sel).first
            if loc.is_visible(timeout=1000):
                loc.click(timeout=3000)
                page.wait_for_timeout(500)
        except Exception:
            pass


def load_rankings_html() -> tuple[str, str]:
    scraped = now_iso()
    print(f"  [RANK] Loading (Playwright, one page): {RANKINGS_URL}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=HEADERS["User-Agent"],
            locale="en-US",
        )
        page = context.new_page()
        try:
            page.goto(RANKINGS_URL, wait_until="load", timeout=120000)
            dismiss_cookies(page)
            try:
                page.wait_for_selector(
                    "a[href*='/rankings-breakdown'][href*='team=singles']",
                    timeout=90000,
                )
            except Exception:
                print("  [WARN] Rankings links slow to appear; continuing…")
            try:
                page.wait_for_load_state("networkidle", timeout=30000)
            except Exception:
                pass
            page.wait_for_timeout(2000)
            html = page.content()
        finally:
            context.close()
            browser.close()
    return html, scraped


@dataclass
class RankingRow:
    player_name: str
    official_rank: str
    official_points: int
    age: str
    profile_url: str
    norm_name: str = ""
    norm_sig: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        self.norm_name = normalize_player_name(self.player_name)
        self.norm_sig = first_initial_and_last(self.norm_name)


def parse_rankings_table(html: str, scraped_at: str, rankings_date: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    raw: list[RankingRow] = []
    for tr in soup.select("tr.lower-row"):
        rtd = tr.select_one("td.rank")
        if not rtd:
            continue
        rank = parse_rank_cell(rtd.get_text())
        if not rank or not re.match(r"^\d", rank):
            continue
        name = display_name_from_row(tr)
        if not name:
            continue
        pts_s = points_from_row(tr)
        if not pts_s.isdigit():
            continue
        pts = int(pts_s)
        age = age_from_rankings_row(tr)
        prof = overview_url_from_row(tr)
        raw.append(
            RankingRow(
                player_name=name,
                official_rank=rank,
                official_points=pts,
                age=age,
                profile_url=prof,
            )
        )

    best_by_norm: dict[str, RankingRow] = {}
    for r in raw:
        k = r.norm_name
        if not k:
            continue
        if k not in best_by_norm or rank_sort_key(r.official_rank) < rank_sort_key(
            best_by_norm[k].official_rank
        ):
            best_by_norm[k] = r

    out: list[dict] = []
    for r in sorted(best_by_norm.values(), key=lambda x: rank_sort_key(x.official_rank)):
        out.append(
            {
                "scraped_at": scraped_at,
                "rankings_date": rankings_date,
                "official_rank": r.official_rank,
                "player_name": r.player_name,
                "age": r.age,
                "official_points": r.official_points,
                "profile_url": r.profile_url,
                "_norm": r.norm_name,
                "_sig": r.norm_sig,
            }
        )
    return out


def build_lookups(rows: list[dict]) -> tuple[dict[str, dict], dict[tuple[str, str], list[dict]]]:
    exact: dict[str, dict] = {}
    fb: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        nk = r["_norm"]
        if nk:
            exact[nk] = r
        sig = r.get("_sig")
        if sig:
            fb[sig].append(r)
    return exact, fb


def resolve_needed_player(
    display_name: str,
    exact: dict[str, dict],
    fb: dict[tuple[str, str], list[dict]],
) -> tuple[dict | None, str]:
    """Returns (record_or_none, method: exact|fallback|none)."""
    n = normalize_player_name(display_name)
    if not n:
        return None, "none"
    if n in exact:
        return exact[n], "exact"
    sig = first_initial_and_last(n)
    if not sig:
        return None, "none"
    cands = fb.get(sig, [])
    if len(cands) != 1:
        return None, "none"
    return cands[0], "fallback"


def fetch_age_overview(session: requests.Session, overview_url: str, timeout: int = 20) -> str:
    if not overview_url or "/players/" not in overview_url:
        return ""
    try:
        r = session.get(overview_url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        html = r.text
    except Exception:
        return ""
    for pat in (
        r"Age[:\s]+(\d{1,2})\b",
        r'"age"\s*:\s*(\d{1,2})\b',
        r"data-age=\"(\d{1,2})\"",
        r"\b(\d{1,2})\s*years?\s*old\b",
    ):
        m = re.search(pat, html, re.I)
        if m:
            a = int(m.group(1))
            if 15 <= a <= 55:
                return str(a)
    return ""


def main() -> None:
    for p in (OUTPUT_CSV.parent, SNAPSHOT_CSV.parent):
        p.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        raise SystemExit(f"Missing input: {INPUT_CSV}")

    # STEP 1
    matches = pd.read_csv(INPUT_CSV)
    n_in = len(matches)
    print(f"[STEP 1] Loaded {n_in} input rows from {INPUT_CSV}")

    # STEP 2
    needed: set[str] = set()
    for col in ("winner_name", "loser_name"):
        for v in matches[col].astype(str):
            s = strip_match_display_name(v)
            if s:
                needed.add(s)
    n_needed = len(needed)
    print(f"[STEP 2] Unique needed players: {n_needed}")

    # STEP 3
    html, scraped_at = load_rankings_html()
    rankings_date = detect_rankings_date(html) or datetime.now(timezone.utc).date().isoformat()
    print(f"  [RANK] rankings_date (best-effort): {rankings_date}")

    ranking_rows = parse_rankings_table(html, scraped_at, rankings_date)
    z = len(ranking_rows)
    print(f"[STEP 3] Rankings rows parsed (after dedupe by name): {z}")

    snap_df = pd.DataFrame(
        [
            {
                "scraped_at": r["scraped_at"],
                "rankings_date": r["rankings_date"],
                "official_rank": r["official_rank"],
                "player_name": r["player_name"],
                "age": r["age"],
                "official_points": r["official_points"],
            }
            for r in ranking_rows
        ]
    )
    snap_df.to_csv(SNAPSHOT_CSV, index=False)
    print(f"  [SNAPSHOT] Wrote {SNAPSHOT_CSV}")

    exact, fb = build_lookups(ranking_rows)

    # STEP 4 — resolve only needed players
    resolved: dict[str, dict | None] = {}
    method_counts = {"exact": 0, "fallback": 0, "none": 0}
    for name in sorted(needed):
        rec, how = resolve_needed_player(name, exact, fb)
        resolved[name] = rec
        method_counts[how if how in method_counts else "none"] += 1

    exact_hits = method_counts["exact"]
    fb_hits = method_counts["fallback"]
    miss = method_counts["none"]
    print(f"[STEP 4] exact matches: {exact_hits}, fallback matches: {fb_hits}, not in table: {miss}")

    # STEP 5 — overview only for needed players missing age, with profile URL, safety: ⊆ needed
    session = requests.Session()
    need_age: list[tuple[str, dict]] = []
    for disp, rec in resolved.items():
        if disp not in needed:
            continue
        if not rec:
            continue
        if rec.get("age"):
            continue
        url = rec.get("profile_url") or ""
        if url:
            need_age.append((disp, rec))

    c_missing = len(need_age)
    print(f"[STEP 5] Needed players still missing age (will fetch overview): {c_missing}")

    overview_fetched = 0
    for disp, rec in need_age:
        assert disp in needed
        age = fetch_age_overview(session, rec["profile_url"])
        if age and age.isdigit():
            rec["age"] = age
        overview_fetched += 1
        if overview_fetched % 5 == 0 or overview_fetched == c_missing:
            print(f"  … overview fetches: {overview_fetched}/{c_missing}")
        time.sleep(0.08)

    # STEP 6 — merge onto matches
    for col in (
        "winner_age",
        "loser_age",
        "winner_rank_points",
        "loser_rank_points",
        "winner_official_rank",
        "loser_official_rank",
        "rankings_date",
        "rankings_source",
    ):
        if col not in matches.columns:
            matches[col] = ""

    unmatched_rows: list[dict] = []
    win_hit = lose_hit = both_hit = 0

    for _, row in matches.iterrows():
        wn = strip_match_display_name(str(row.get("winner_name", "") or ""))
        ln = strip_match_display_name(str(row.get("loser_name", "") or ""))
        tourn = str(row.get("tournament_name", "") or "")
        mdate = str(row.get("match_date", "") or "")

        wr = resolved.get(wn)
        lr = resolved.get(ln)

        if wr:
            win_hit += 1
        else:
            unmatched_rows.append(
                {
                    "player_name": wn,
                    "side": "winner",
                    "tournament_name": tourn,
                    "match_date": mdate,
                }
            )
        if lr:
            lose_hit += 1
        else:
            unmatched_rows.append(
                {
                    "player_name": ln,
                    "side": "loser",
                    "tournament_name": tourn,
                    "match_date": mdate,
                }
            )
        if wr and lr:
            both_hit += 1

    def fill_side(rec: dict | None) -> tuple[object, object, object]:
        if not rec:
            return "", "", ""
        age = rec.get("age") or ""
        if age and not str(age).isdigit():
            age = ""
        pts = rec.get("official_points", "")
        rk = rec.get("official_rank", "")
        if pts != "" and not str(pts).isdigit():
            pts = ""
        return age, pts, rk

    w_ages, w_pts, w_rks = [], [], []
    l_ages, l_pts, l_rks = [], [], []
    for _, row in matches.iterrows():
        wn = strip_match_display_name(str(row.get("winner_name", "") or ""))
        ln = strip_match_display_name(str(row.get("loser_name", "") or ""))
        wa, wp, wrk = fill_side(resolved.get(wn))
        la, lp, lrk = fill_side(resolved.get(ln))
        w_ages.append(wa)
        w_pts.append(wp)
        w_rks.append(wrk)
        l_ages.append(la)
        l_pts.append(lp)
        l_rks.append(lrk)

    matches = matches.copy()
    matches["winner_age"] = w_ages
    matches["loser_age"] = l_ages
    matches["winner_rank_points"] = w_pts
    matches["loser_rank_points"] = l_pts
    matches["winner_official_rank"] = w_rks
    matches["loser_official_rank"] = l_rks
    matches["rankings_date"] = rankings_date
    matches["rankings_source"] = RANKINGS_SOURCE

    matches.to_csv(OUTPUT_CSV, index=False)
    pd.DataFrame(unmatched_rows).drop_duplicates().to_csv(UNMATCHED_CSV, index=False)

    print(f"[STEP 7] Wrote {OUTPUT_CSV}")
    print(f"  Unmatched name rows: {UNMATCHED_CSV}")

    print("\n=== Summary ===")
    print(f"  input rows: {n_in}")
    print(f"  unique needed players: {n_needed}")
    print(f"  rankings rows parsed (deduped): {z}")
    print(f"  exact matches (needed): {exact_hits}")
    print(f"  fallback matches (needed): {fb_hits}")
    print(f"  needed players not in rankings table: {miss}")
    print(f"  overview pages fetched (missing age only): {overview_fetched}")
    print(f"  winner rows matched: {win_hit}")
    print(f"  loser rows matched: {lose_hit}")
    print(f"  rows both players matched: {both_hit}")


if __name__ == "__main__":
    main()
