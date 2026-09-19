#!/usr/bin/env python3
"""
Scrape Completed / Upcoming match lines from each Tennis Abstract tournament page.
Reads:  data/raw/current_atp_tournaments.csv
Writes: data/raw/completed_matches_2026.csv
         data/raw/upcoming_matches_2026.csv

Section extraction: heading-based first, keyword blobs, full-page regions,
then <br>-split plain lines (script-embedded rows) plus round-prefix scan.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup, NavigableString, Tag

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils_live_scrape import (
    DATA_RAW,
    MATCH_LINE_START_RE,
    clean_player_text,
    ensure_data_dirs,
    fetch_html,
    is_valid_singles_match,
    normalize_name,
    now_iso,
    parse_completed_line,
    parse_upcoming_line,
    plain_lines_from_html_br_segments,
)

STRICT_SINGLES_ONLY = True

DEBUG_KEYWORDS = (
    "Completed Matches",
    "Upcoming Matches",
    "R16:",
    "R2:",
    " d. ",
    " vs ",
)


def tournament_debug_slug(tournament_name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", tournament_name.strip().lower()).strip("_")
    return s or "unknown"


def write_tournament_debug_files(
    tournament_name: str, html: str, page_text: str
) -> None:
    slug = tournament_debug_slug(tournament_name)
    html_path = DATA_RAW / f"debug_{slug}_page.html"
    txt_path = DATA_RAW / f"debug_{slug}_page.txt"
    html_path.write_text(html, encoding="utf-8", errors="replace")
    txt_path.write_text(page_text, encoding="utf-8", errors="replace")


def print_keyword_presence(label: str, haystack: str) -> None:
    bits = []
    for kw in DEBUG_KEYWORDS:
        if kw == " vs ":
            found = re.search(r"\s+vs\s+", haystack, re.I) is not None
        else:
            found = kw in haystack
        bits.append(f"{kw!r}={found}")
    print(f"  [debug] keywords in {label}: " + ", ".join(bits))


def full_page_round_match_candidates(page_text: str) -> tuple[list[str], list[str]]:
    """
    Line-by-line: any line starting with a round prefix; partition into
    completed (contains ' d. ') vs upcoming (contains ' vs ').
    """
    completed: list[str] = []
    upcoming: list[str] = []
    seen_c: set[str] = set()
    seen_u: set[str] = set()
    for raw in page_text.splitlines():
        line = raw.strip()
        if not line or not MATCH_LINE_START_RE.match(line):
            continue
        if " d. " in line:
            if line not in seen_c:
                seen_c.add(line)
                completed.append(line)
        elif re.search(r"\s+vs\s+", line, re.I):
            if line not in seen_u:
                seen_u.add(line)
                upcoming.append(line)
    return completed, upcoming


def find_section_text_after_heading(soup: BeautifulSoup, heading_substr: str) -> str:
    """Collect text after a heading that contains heading_substr."""
    heading_substr_l = heading_substr.lower()
    candidates: list[Tag] = []
    for tag_name in ("h1", "h2", "h3", "h4", "h5", "h6", "strong", "b"):
        for el in soup.find_all(tag_name):
            txt = el.get_text(" ", strip=True)
            if heading_substr_l in txt.lower():
                candidates.append(el)

    if not candidates:
        return ""

    el = candidates[0]
    lines: list[str] = []
    for sib in el.next_siblings:
        if isinstance(sib, NavigableString):
            t = str(sib).strip()
            if t:
                lines.append(t)
        elif isinstance(sib, Tag):
            name = sib.name.lower() if sib.name else ""
            if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                break
            if name in ("strong", "b"):
                st = sib.get_text(" ", strip=True)
                if len(st) < 60 and any(
                    x in st.lower()
                    for x in ("completed", "upcoming", "forecast", "draw", "match")
                ):
                    if heading_substr_l not in st.lower():
                        break
            lines.append(sib.get_text("\n", strip=True))

    return "\n".join(lines)


def fallback_blob_for_keyword(full_text: str, keyword: str) -> str:
    """Slice after first occurrence of keyword (use combined visible + br-segment text)."""
    if not full_text:
        return ""
    full = full_text
    low = full.lower()
    kw = keyword.lower()
    if kw not in low:
        return ""
    idx = low.index(kw)
    return full[idx : idx + 80000]


def extract_match_lines(blob: str) -> list[str]:
    """Lines that look like tournament match lines (R32:, QF:, F:, etc.)."""
    out: list[str] = []
    for raw in blob.splitlines():
        line = raw.strip()
        if not line:
            continue
        if MATCH_LINE_START_RE.match(line):
            out.append(line)
    return out


def combined_page_text_for_matching(html: str, soup: BeautifulSoup) -> str:
    """Visible body text plus <br>-split plain lines (includes script-embedded rows)."""
    body = soup.find("body")
    visible = body.get_text("\n", strip=True) if body else ""
    br_plain = "\n".join(plain_lines_from_html_br_segments(html))
    if visible and br_plain:
        return f"{visible}\n{br_plain}"
    return visible or br_plain


def extract_match_lines_from_full_page(html: str) -> tuple[list[str], list[str]]:
    """
    Fallback: scan entire HTML text for Completed / Upcoming regions by keywords,
    then extract match lines from those regions.
    """
    soup = BeautifulSoup(html, "html.parser")
    text = combined_page_text_for_matching(html, soup)
    low = text.lower()

    def region(after: str, until: list[str]) -> str:
        if after not in low:
            return ""
        i0 = low.index(after)
        end = len(text)
        for u in until:
            ui = low.find(u, i0 + len(after))
            if ui != -1 and ui < end:
                end = ui
        return text[i0:end]

    completed_blob = region(
        "completed matches",
        ["upcoming matches", "upcoming match", "forecast"],
    )
    upcoming_blob = region(
        "upcoming matches",
        ["completed matches", "draw", "schedule"],
    )
    if not upcoming_blob:
        upcoming_blob = region("upcoming match", ["completed matches"])

    return extract_match_lines(completed_blob), extract_match_lines(upcoming_blob)


def scrape_one_tournament(
    tournament_name: str,
    tournament_url: str,
    surface: str,
    scraped_at: str,
) -> tuple[list[dict], list[dict], dict]:
    completed: list[dict] = []
    upcoming: list[dict] = []
    stats = {
        "completed_raw": 0,
        "upcoming_raw": 0,
        "completed_parse_skipped": 0,
        "upcoming_parse_skipped": 0,
        "completed_parsed": 0,
        "upcoming_parsed": 0,
        "completed_kept_singles": 0,
        "upcoming_kept_singles": 0,
        "completed_dropped_nonsingles": 0,
        "upcoming_dropped_nonsingles": 0,
    }
    dropped_ex_c: list[str] = []
    dropped_ex_u: list[str] = []
    html = fetch_html(tournament_url)
    if not html:
        return completed, upcoming, stats
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("body")
    visible_page_text = body.get_text("\n", strip=True) if body else ""
    combined_text = combined_page_text_for_matching(html, soup)

    write_tournament_debug_files(tournament_name, html, visible_page_text)
    print_keyword_presence("HTML", html)
    print_keyword_presence("visible page text (soup)", visible_page_text)
    print_keyword_presence("visible + <br> plain segments", combined_text)

    completed_lines: list[str] = []
    upcoming_lines: list[str] = []
    seen_c: set[str] = set()
    seen_u: set[str] = set()

    def add_completed_line(line: str) -> None:
        line = line.strip()
        if line and line not in seen_c:
            seen_c.add(line)
            completed_lines.append(line)

    def add_upcoming_line(line: str) -> None:
        line = line.strip()
        if line and line not in seen_u:
            seen_u.add(line)
            upcoming_lines.append(line)

    for label, is_completed in (
        ("Completed Matches", True),
        ("Upcoming Matches", False),
    ):
        blob = find_section_text_after_heading(soup, label)
        if not blob:
            blob = find_section_text_after_heading(
                soup, "Completed" if is_completed else "Upcoming"
            )
        lines = extract_match_lines(blob)
        for ln in lines:
            if is_completed:
                add_completed_line(ln)
            else:
                add_upcoming_line(ln)

    # Per-section keyword fallback if one side is still empty
    if not seen_c:
        blob_c = fallback_blob_for_keyword(combined_text, "completed matches")
        for ln in extract_match_lines(blob_c):
            add_completed_line(ln)
    if not seen_u:
        blob_u = fallback_blob_for_keyword(combined_text, "upcoming matches")
        if not blob_u:
            blob_u = fallback_blob_for_keyword(combined_text, "upcoming match")
        for ln in extract_match_lines(blob_u):
            add_upcoming_line(ln)

    # Whole-page region fallback only if both sides still empty
    if not seen_c and not seen_u:
        c_fb, u_fb = extract_match_lines_from_full_page(html)
        for ln in c_fb:
            add_completed_line(ln)
        for ln in u_fb:
            add_upcoming_line(ln)

    # Fallback: scan full page text line-by-line for round-prefixed match rows
    fp_c, fp_u = full_page_round_match_candidates(combined_text)
    for ln in fp_c:
        add_completed_line(ln)
    for ln in fp_u:
        add_upcoming_line(ln)

    # Heading/body blobs can mix forecast (vs) rows into "completed" sections; keep only
    # lines that match the result pattern for each bucket before parse.
    def is_completed_candidate_line(line: str) -> bool:
        s = line.strip()
        return bool(s and MATCH_LINE_START_RE.match(s) and " d. " in s)

    def is_upcoming_candidate_line(line: str) -> bool:
        s = line.strip()
        return bool(
            s
            and MATCH_LINE_START_RE.match(s)
            and re.search(r"\s+vs\s+", s, re.I)
            and " d. " not in s
        )

    completed_lines = [l for l in completed_lines if is_completed_candidate_line(l)]
    upcoming_lines = [l for l in upcoming_lines if is_upcoming_candidate_line(l)]
    completed_lines = list(dict.fromkeys(completed_lines))
    upcoming_lines = list(dict.fromkeys(upcoming_lines))

    # Brute-force: same scan ensures we catch lines missed by section/blob logic;
    # full_page_round_match_candidates already covers every line with round + d./vs.
    n_cand_c = len(completed_lines)
    n_cand_u = len(upcoming_lines)
    print(
        f"  [debug] match candidates before parse: completed={n_cand_c}, upcoming={n_cand_u}"
    )
    if n_cand_c:
        print("  [debug] first 5 completed candidates:")
        for ln in completed_lines[:5]:
            print(f"    | {ln[:200]}{'...' if len(ln) > 200 else ''}")
    else:
        print("  [debug] first 5 completed candidates: (none)")
    if n_cand_u:
        print("  [debug] first 5 upcoming candidates:")
        for ln in upcoming_lines[:5]:
            print(f"    | {ln[:200]}{'...' if len(ln) > 200 else ''}")
    else:
        print("  [debug] first 5 upcoming candidates: (none)")

    for line in completed_lines:
        stats["completed_raw"] += 1
        parsed: Optional[dict] = parse_completed_line(line)
        if not parsed:
            stats["completed_parse_skipped"] += 1
            continue
        if not parsed.get("round") or not parsed.get("winner_name") or not parsed.get(
            "loser_name"
        ):
            stats["completed_parse_skipped"] += 1
            continue
        stats["completed_parsed"] += 1
        wn = clean_player_text(str(parsed["winner_name"]))
        loser_n = clean_player_text(str(parsed["loser_name"]))
        if not is_valid_singles_match(wn, loser_n):
            stats["completed_dropped_nonsingles"] += 1
            if len(dropped_ex_c) < 3:
                dropped_ex_c.append(
                    f"{wn!r} vs {loser_n!r} | {line[:160]}{'...' if len(line) > 160 else ''}"
                )
            if STRICT_SINGLES_ONLY:
                continue
        else:
            stats["completed_kept_singles"] += 1
        completed.append(
            {
                "scraped_at": scraped_at,
                "tournament_name": tournament_name,
                "source_url": tournament_url,
                "surface": surface,
                "round": parsed["round"],
                "winner_name": wn,
                "loser_name": loser_n,
                "score_text": parsed.get("score_text", ""),
                "raw_completed_line": line,
            }
        )

    for line in upcoming_lines:
        stats["upcoming_raw"] += 1
        parsed = parse_upcoming_line(line)
        if not parsed:
            stats["upcoming_parse_skipped"] += 1
            continue
        if (
            not parsed.get("round")
            or not parsed.get("player_1_name")
            or not parsed.get("player_2_name")
        ):
            stats["upcoming_parse_skipped"] += 1
            continue
        stats["upcoming_parsed"] += 1
        p1 = clean_player_text(str(parsed["player_1_name"]))
        p2 = clean_player_text(str(parsed["player_2_name"]))
        if not is_valid_singles_match(p1, p2):
            stats["upcoming_dropped_nonsingles"] += 1
            if len(dropped_ex_u) < 3:
                dropped_ex_u.append(
                    f"{p1!r} vs {p2!r} | {line[:160]}{'...' if len(line) > 160 else ''}"
                )
            if STRICT_SINGLES_ONLY:
                continue
        else:
            stats["upcoming_kept_singles"] += 1
        upcoming.append(
            {
                "scraped_at": scraped_at,
                "tournament_name": tournament_name,
                "source_url": tournament_url,
                "surface": surface,
                "round": parsed["round"],
                "player_1_name": p1,
                "player_2_name": p2,
                "raw_upcoming_line": line,
            }
        )

    print(
        "  [singles] completed: "
        f"parsed={stats['completed_parsed']} "
        f"kept={stats['completed_kept_singles']} "
        f"dropped_non_singles={stats['completed_dropped_nonsingles']} "
        f"(strict={STRICT_SINGLES_ONLY})"
    )
    print(
        "  [singles] upcoming: "
        f"parsed={stats['upcoming_parsed']} "
        f"kept={stats['upcoming_kept_singles']} "
        f"dropped_non_singles={stats['upcoming_dropped_nonsingles']} "
        f"(strict={STRICT_SINGLES_ONLY})"
    )
    if dropped_ex_c:
        print("  [singles] first dropped examples (completed):")
        for ex in dropped_ex_c:
            print(f"    | {ex}")
    if dropped_ex_u:
        print("  [singles] first dropped examples (upcoming):")
        for ex in dropped_ex_u:
            print(f"    | {ex}")

    return completed, upcoming, stats


def main() -> None:
    ensure_data_dirs()
    scraped_at = now_iso()
    tour_path = DATA_RAW / "current_atp_tournaments.csv"
    if not tour_path.exists():
        print(f"Missing {tour_path}; run discover_current_atp_tournaments.py first")
        return
    tours = pd.read_csv(tour_path)
    all_completed: list[dict] = []
    all_upcoming: list[dict] = []
    n_scraped = 0
    for _, r in tours.iterrows():
        name = str(r.get("tournament_name", ""))
        url = str(r.get("tournament_url", ""))
        surf = str(r.get("surface", "unknown"))
        if not url or url == "nan":
            continue
        print(f"Scraping tournament: {name}")
        try:
            c, u, st = scrape_one_tournament(name, url, surf, scraped_at)
            all_completed.extend(c)
            all_upcoming.extend(u)
            n_scraped += 1
            print(
                f"  -> rows written: completed={len(c)} upcoming={len(u)} "
                f"| line attempts: completed={st['completed_raw']} upcoming={st['upcoming_raw']} "
                f"| parse_skipped: c={st['completed_parse_skipped']} u={st['upcoming_parse_skipped']}"
            )
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    dc = pd.DataFrame(all_completed)
    if len(dc):
        dc["_w"] = dc["winner_name"].map(normalize_name)
        dc["_l"] = dc["loser_name"].map(normalize_name)
        dc = dc.drop_duplicates(
            subset=["tournament_name", "round", "_w", "_l"], keep="first"
        )
        dc = dc.drop(columns=["_w", "_l"], errors="ignore")

    du = pd.DataFrame(all_upcoming)
    if len(du):
        du["_p1"] = du["player_1_name"].map(normalize_name)
        du["_p2"] = du["player_2_name"].map(normalize_name)
        du = du.drop_duplicates(
            subset=["tournament_name", "round", "_p1", "_p2"], keep="first"
        )
        du = du.drop(columns=["_p1", "_p2"], errors="ignore")

    def is_2026_row(df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series([], dtype=bool)
        u = df["source_url"].astype(str).str.contains("2026", na=False)
        n = df["tournament_name"].astype(str).str.contains("2026", na=False)
        return u | n

    if len(dc):
        m = is_2026_row(dc)
        if m.any():
            dc = dc[m]
        else:
            print(
                "  [WARN] No rows with explicit '2026' in URL/name; keeping all completed rows."
            )
    if len(du):
        m = is_2026_row(du)
        if m.any():
            du = du[m]
        else:
            print(
                "  [WARN] No rows with explicit '2026' in URL/name; keeping all upcoming rows."
            )

    c_path = DATA_RAW / "completed_matches_2026.csv"
    u_path = DATA_RAW / "upcoming_matches_2026.csv"
    dc.to_csv(c_path, index=False)
    du.to_csv(u_path, index=False)

    print("\n=== Summary ===")
    print(f"Tournaments scraped: {n_scraped}")
    print(f"Completed extracted: {len(dc)}")
    print(f"Upcoming extracted: {len(du)}")
    print(f"Saved: {c_path}")
    print(f"Saved: {u_path}")


if __name__ == "__main__":
    main()
