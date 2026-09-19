#!/usr/bin/env python3
"""
Daily ATP singles rankings snapshot (one dated CSV per run, no overwrites).

Writes: data/rankings_snapshots/atp_rankings_snapshot_YYYYMMDD.csv
"""
from __future__ import annotations

import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

BASE_DIR = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = BASE_DIR / "data" / "rankings_snapshots"
RANKINGS_URL = "https://www.atptour.com/en/rankings/singles?rankRange=0-5000"
RANKINGS_SOURCE = "atp_daily_snapshot"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def snapshot_date_ymd_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def snapshot_date_iso_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def normalize_player_name(name: str) -> str:
    s = str(name or "").strip()
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = s.replace(".", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


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
    s = rank_s.rstrip("Tt")
    try:
        n = int(s)
    except ValueError:
        n = 999999
    tie = 0 if rank_s.upper().endswith("T") else -1
    return (n, tie)


def display_name_from_row(tr) -> str:
    """Prefer full name from the player link; last-name-only breaks master joins."""
    a = tr.select_one('td.player a[href*="/overview"]')
    if a:
        t = norm_space(a.get_text())
        if t:
            return t
    sp = tr.select_one("td.player span.lastName")
    if sp and sp.get_text(strip=True):
        return norm_space(sp.get_text())
    return ""


def age_from_rankings_row(tr) -> float | None:
    td = tr.select_one("td.age")
    if not td:
        return None
    t = norm_space(td.get_text())
    if t.isdigit():
        v = int(t)
        if 15 <= v <= 55:
            return float(v)
    return None


def points_from_row(tr) -> int | None:
    td = tr.select_one("td.points")
    if not td:
        return None
    t = norm_space(td.get_text())
    digits = re.sub(r"[^\d]", "", t)
    if digits.isdigit():
        return int(digits)
    return None


def dismiss_cookies(page) -> None:
    for sel in ("#onetrust-accept-btn-handler", "button:has-text('Accept All')"):
        try:
            loc = page.locator(sel).first
            if loc.is_visible(timeout=1000):
                loc.click(timeout=3000)
                page.wait_for_timeout(500)
        except Exception:
            pass


def load_rankings_html() -> str:
    print(f"  [RANK] Loading (Playwright): {RANKINGS_URL}")
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
    return html


def parse_and_dedupe_rows(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    raw: list[tuple[str, str, int, float | None, str]] = []
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
        pts = points_from_row(tr)
        if pts is None:
            continue
        age = age_from_rankings_row(tr)
        norm = normalize_player_name(name)
        if not norm:
            continue
        raw.append((norm, name, pts, age, rank))

    best: dict[str, tuple[str, int, float | None, str]] = {}
    for norm, display_name, pts, age, rank in raw:
        prev = best.get(norm)
        if prev is None or rank_sort_key(rank) < rank_sort_key(prev[3]):
            best[norm] = (display_name, pts, age, rank)

    rows: list[dict] = []
    for norm in sorted(best.keys(), key=lambda k: rank_sort_key(best[k][3])):
        display_name, pts, age, rank = best[norm]
        rows.append(
            {
                "player_name": display_name,
                "official_rank": rank,
                "official_points": pts,
                "age": age,
                "_norm": norm,
            }
        )
    return rows


def main() -> None:
    ymd = snapshot_date_ymd_utc()
    iso_day = snapshot_date_iso_utc()
    scraped_at = now_iso_utc()
    out_path = SNAPSHOT_DIR / f"atp_rankings_snapshot_{ymd}.csv"

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(
            f"ERROR: Snapshot already exists: {out_path}\n"
            "Refusing to overwrite. Remove the file or run tomorrow.",
            file=sys.stderr,
        )
        sys.exit(1)

    html = load_rankings_html()
    parsed = parse_and_dedupe_rows(html)

    records = []
    for r in parsed:
        age_val = r["age"]
        records.append(
            {
                "snapshot_date": iso_day,
                "scraped_at": scraped_at,
                "player_name": r["player_name"],
                "official_rank": r["official_rank"],
                "official_points": int(r["official_points"]),
                "age": age_val if age_val is not None else pd.NA,
                "rankings_source": RANKINGS_SOURCE,
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)

    print(f"snapshot_date: {iso_day}")
    print(f"rows scraped (deduped by player): {len(df)}")
    print(f"output filename: {out_path}")


if __name__ == "__main__":
    main()
