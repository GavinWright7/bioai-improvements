#!/usr/bin/env python3
"""
Append match_date to completed ATP matches using ESPN schedule/scoreboards.
"""
from __future__ import annotations

import random
import re
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_CONFIG = BASE_DIR / "data" / "config"

INPUT_CSV = DATA_RAW / "completed_matches_2026.csv"
OUTPUT_CSV = DATA_PROCESSED / "completed_matches_2026_with_dates.csv"
# Every match row parsed from ESPN (no deduplication).
ESPN_ALL_MATCHES_CSV = DATA_RAW / "espn_all_matches_2026.csv"
# Deduped subset used when merging dates into your completed-matches file.
DAILY_RESULTS_CSV = DATA_RAW / "espn_daily_results_2026.csv"
UNMATCHED_CSV = DATA_RAW / "espn_date_unmatched.csv"
ALIASES_CSV = DATA_CONFIG / "espn_tournament_aliases.csv"

ALLOW_OVERWRITE_DATES = False
# If True, scrape ESPN daily scoreboards through today (for in-progress tournament days).
# If False, only through yesterday (legacy behavior).
INCLUDE_TODAY = True
SLEEP_MIN = 1.5
SLEEP_MAX = 3.5
TIMEOUT = 15

# Embedded in scraped score_text so merge can split winner/loser columns reliably
# (plain concatenation loses the boundary and breaks TA vs ESPN score matching).
SCORE_COL_SEP = " || "

SCHEDULE_URL = "https://www.espn.com/tennis/schedule"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.espn.com/",
}

MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dirs() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    DATA_CONFIG.mkdir(parents=True, exist_ok=True)


def sleep_polite() -> None:
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))


def fetch_html(url: str) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"  [WARN] fetch failed: {url} -> {e}")
        return ""


def normalize_name(name: str) -> str:
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_tournament_name(name: str) -> str:
    s = normalize_name(name)
    s = re.sub(r"\b(atp|open|championships|masters|tennis|presented|by|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ensure_aliases_csv() -> None:
    if ALIASES_CSV.exists():
        return
    df = pd.DataFrame(
        [
            {"alias": "barcelona", "canonical": "barcelona open banc sabadell"},
            {"alias": "munich", "canonical": "bmw open by bitpanda"},
        ]
    )
    df.to_csv(ALIASES_CSV, index=False)


def load_alias_map() -> dict[str, str]:
    ensure_aliases_csv()
    out: dict[str, str] = {}
    try:
        df = pd.read_csv(ALIASES_CSV)
    except Exception:
        return out
    for _, r in df.iterrows():
        a = normalize_tournament_name(str(r.get("alias", "")))
        c = normalize_tournament_name(str(r.get("canonical", "")))
        if a and c:
            out[a] = c
            out[c] = c
    return out


def canonical_tournament(name: str, alias_map: dict[str, str]) -> str:
    n = normalize_tournament_name(name)
    return alias_map.get(n, n)


def normalize_score_text(s: str) -> str:
    if not s:
        return ""
    x = re.sub(r"\s+", " ", s).strip()
    return x


def _ordered_set_score_signature(pairs: list[tuple[int, int]]) -> str:
    """
    Fingerprint from parallel score columns or TA set lines. Preserves set order;
    each set is hi-lo (e.g. 7-6) so it differs from a different 2-set match that
    would collide if we sorted sets lexically (e.g. 7-5 / 6-4 vs 6-4 / 7-5).
    """
    if len(pairs) < 2:
        return ""
    parts: list[str] = []
    for a, b in pairs:
        lo, hi = min(a, b), max(a, b)
        parts.append(f"{hi}-{lo}")
    return "|".join(parts)


def canonical_score_signatures(raw: str) -> list[str]:
    """
    One or more keys so Tennis Abstract scores (e.g. 6-4 7-5) align with ESPN
    text lines (e.g. winner column + loser column -> '6 7 4 5').
    """
    s = normalize_score_text(str(raw or ""))
    if not s:
        return []
    has_ret = bool(re.search(r"\bRET\b", s, re.I))
    base_s = re.sub(r"\s*RET\s*$", "", s, flags=re.I).strip() if has_ret else s

    sigs: list[str] = []
    seen: set[str] = set()

    def push_pairs(pairs: list[tuple[int, int]]) -> None:
        sig = _ordered_set_score_signature(pairs)
        if not sig:
            return
        full = f"{sig}|RET" if has_ret else sig
        if full not in seen:
            seen.add(full)
            sigs.append(full)

    if SCORE_COL_SEP in base_s:
        left, _, right = base_s.partition(SCORE_COL_SEP)
        left, right = left.strip(), right.strip()
        a = [int(x) for x in re.findall(r"\d+", left)]
        b = [int(x) for x in re.findall(r"\d+", right)]
        m = min(len(a), len(b))
        if m >= 2:
            push_pairs(list(zip(a[:m], b[:m])))
        if sigs:
            return sigs
        base_s = f"{left} {right}".strip()

    hy = re.findall(r"(\d+)\s*-\s*(\d+)(?:\(\d+\))?", base_s)
    if hy:
        pairs = [(int(a), int(b)) for a, b in hy]
        sig = _ordered_set_score_signature(pairs)
        if not sig:
            return []
        return [f"{sig}|RET"] if has_ret else [sig]

    toks = [int(x) for x in re.findall(r"\d+", base_s)]
    n = len(toks)

    if n >= 4 and n % 2 == 0:
        h = n // 2
        push_pairs(list(zip(toks[:h], toks[h:])))
    elif n >= 5:
        for k in range(2, n - 1):
            a, b = toks[:k], toks[k:]
            m = min(len(a), len(b))
            if m < 2:
                continue
            push_pairs(list(zip(a[:m], b[:m])))
        # Odd-length ESPN lines often include an extra tiebreak digit; try dropping one token.
        if n % 2 == 1 and n >= 5:
            for drop in range(n):
                t2 = toks[:drop] + toks[drop + 1 :]
                if len(t2) < 4 or len(t2) % 2 != 0:
                    continue
                h2 = len(t2) // 2
                push_pairs(list(zip(t2[:h2], t2[h2:])))

    if not sigs:
        fb = normalize_score_text(base_s)
        if fb:
            sigs.append(f"{fb}|RET" if has_ret else fb)
    return sigs


def alternate_name_keys(display_name: str) -> frozenset[str]:
    """
    Normalized name variants for narrow ESPN vs TA matching.
    Includes: base normalize, hyphen collapsed via normalize_name, shortened 3+ token
    (first two tokens), and reversed order for two-token names (Eastern order).
    """
    raw = str(display_name or "").strip()
    if not raw:
        return frozenset()
    hyphen_flat = normalize_name(raw.replace("-", " "))
    base = normalize_name(raw)
    keys: set[str] = set()
    for k in (base, hyphen_flat):
        if k:
            keys.add(k)
    parts = base.split() if base else []
    if len(parts) >= 3:
        keys.add(" ".join(parts[:2]))
    if len(parts) == 2:
        keys.add(f"{parts[1]} {parts[0]}")
    return frozenset(x for x in keys if x)


def name_tokens_compatible(raw_left: str, raw_right: str) -> bool:
    a = alternate_name_keys(raw_left)
    b = alternate_name_keys(raw_right)
    return bool(a & b)


def row_names_compatible(left_row: pd.Series, espn_rec: dict) -> bool:
    return bool(
        name_tokens_compatible(
            str(left_row.get("winner_name", "") or ""),
            str(espn_rec.get("winner_name", "") or ""),
        )
        and name_tokens_compatible(
            str(left_row.get("loser_name", "") or ""),
            str(espn_rec.get("loser_name", "") or ""),
        )
    )


def build_alt_name_index(right: pd.DataFrame) -> dict[tuple[str, str], list[dict]]:
    """ESPN rows by (canonical tournament, canonical score signature) for alternate-name fallback."""
    idx: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for _, row in right.iterrows():
        rec = row.to_dict()
        tt = str(rec.get("_tt", "") or "").strip()
        raw_score = str(rec.get("score_text", "") or "")
        if not tt or not normalize_score_text(raw_score):
            continue
        for sig in canonical_score_signatures(raw_score):
            if not sig:
                continue
            idx[(tt, sig)].append(rec)
    return dict(idx)


def try_alt_name_match(
    r: pd.Series,
    alt_index: dict[tuple[str, str], list[dict]],
) -> dict | None:
    """
    Fallback: same tournament (_tt) + same canonical score signature (TA vs ESPN formats),
    optional round filter, winner/loser compatible via alternate_name_keys.
    Only accepts a unique hit.
    """
    tt = str(r.get("_tt", "") or "").strip()
    raw_score = str(r.get("score_text", "") or "")
    if not tt or not normalize_score_text(raw_score):
        return None
    sigs = canonical_score_signatures(raw_score)
    if not sigs:
        return None

    candidates: list[dict] = []
    seen_id: set[int] = set()
    for sig in sigs:
        for c in alt_index.get((tt, sig), []):
            oid = id(c)
            if oid in seen_id:
                continue
            seen_id.add(oid)
            candidates.append(c)
    if not candidates:
        return None

    tr_l = str(r.get("_tr", "") or "").strip()
    if tr_l:
        round_pool = [
            c
            for c in candidates
            if str(c.get("_tr", "") or "").strip() == tr_l
        ]
        pool = round_pool if round_pool else candidates
    else:
        pool = candidates

    hits = [c for c in pool if row_names_compatible(r, c)]
    if len(hits) != 1:
        return None

    hit = hits[0]
    tname = str(r.get("tournament_name", "") or "")
    print(
        f"  [ALT NAME MATCH] {tname} | "
        f"{r.get('winner_name')} vs {r.get('loser_name')} "
        f"matched to ESPN row {hit.get('winner_name')} vs {hit.get('loser_name')}"
    )
    return hit


def normalize_round(raw: str) -> tuple[str, str]:
    t = (raw or "").strip()
    u = t.upper()
    mapping = {
        "FINAL": "F",
        "SEMIFINAL": "SF",
        "SEMIFINAL": "SF",
        "QUARTERFINAL": "QF",
        "QUARTERFINALS": "QF",
        "ROUND OF 16": "R16",
        "ROUND OF 32": "R32",
        "ROUND OF 64": "R64",
        "QUALIFYING 1ST ROUND": "Q1",
        "QUALIFYING 2ND ROUND": "Q2",
        "QUALIFYING 3RD ROUND": "Q3",
    }
    if u in mapping:
        return mapping[u], t
    m = re.search(r"\bR(\d{1,3})\b", u)
    if m:
        return f"R{m.group(1)}", t
    if u in {"QF", "SF", "F", "RR"}:
        return u, t
    return "", t


def parse_date_range(text: str) -> tuple[Optional[date], Optional[date]]:
    month_pat = "|".join(MONTHS)
    # April 11 - 19, 2026
    m = re.search(
        rf"\b({month_pat})\s+(\d{{1,2}})\s*-\s*(\d{{1,2}}),\s*(\d{{4}})\b",
        text,
        re.I,
    )
    if m:
        mon = datetime.strptime(m.group(1), "%B").month
        year = int(m.group(4))
        start = date(year, mon, int(m.group(2)))
        end = date(year, mon, int(m.group(3)))
        return start, end
    # April 29 - May 5, 2026
    m = re.search(
        rf"\b({month_pat})\s+(\d{{1,2}})\s*-\s*({month_pat})\s+(\d{{1,2}}),\s*(\d{{4}})\b",
        text,
        re.I,
    )
    if m:
        mon1 = datetime.strptime(m.group(1), "%B").month
        mon2 = datetime.strptime(m.group(3), "%B").month
        year = int(m.group(5))
        start = date(year, mon1, int(m.group(2)))
        end = date(year, mon2, int(m.group(4)))
        return start, end
    return None, None


def scoreboard_base_url(url: str) -> str:
    return re.sub(r"/date/\d{8}$", "", url.rstrip("/"))


def extract_schedule_tournaments() -> list[dict]:
    print(f"Fetching schedule page: {SCHEDULE_URL}")
    html = fetch_html(SCHEDULE_URL)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    tournaments: list[dict] = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full = urljoin(SCHEDULE_URL, href)
        low = full.lower()
        if "/tennis/scoreboard/tournament/" not in low:
            continue
        if "competitiontype/1" not in low:
            continue
        row_text = a.parent.get_text(" ", strip=True) if a.parent else a.get_text(" ", strip=True)
        text = a.get_text(" ", strip=True)
        blob = f"{text} {row_text}".lower()
        if any(x in blob for x in ("women", "wta", "doubles")):
            continue
        key = scoreboard_base_url(full)
        if key in seen:
            continue
        seen.add(key)
        tournaments.append(
            {
                "tournament_name": text or row_text or full,
                "location": "",
                "scoreboard_url": key,
            }
        )
    return tournaments


def _parse_set_scores(score_line: str) -> list[int]:
    """Extract numeric game/point tokens from an ESPN score line (e.g. '6 3 4' or '7^{7}')."""
    s = clean_score_line(score_line)
    return [int(x) for x in re.findall(r"\d+", s)]


def infer_winner_loser_from_scores(p1: str, p2: str, s1: str, s2: str) -> tuple[str, str]:
    """
    ESPN match cards list players in DOM order; the winner is not always player1.
    Compare set-by-set using the parallel score columns when possible.
    """
    a = _parse_set_scores(s1)
    b = _parse_set_scores(s2)
    if not a and not b:
        return p1, p2
    n = min(len(a), len(b))
    w1 = w2 = 0
    for i in range(n):
        if a[i] > b[i]:
            w1 += 1
        elif b[i] > a[i]:
            w2 += 1
    if w1 > w2:
        return p1, p2
    if w2 > w1:
        return p2, p1
    # Tie on completed columns (rare / incomplete) — fall back to total games as weak tiebreak
    g1 = sum(a)
    g2 = sum(b)
    if g1 > g2:
        return p1, p2
    if g2 > g1:
        return p2, p1
    return p1, p2


def is_score_line(s: str) -> bool:
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return False
    if re.fullmatch(r"(?:\d+(?:\^\{\d+\})?\s*)+", s):
        return True
    # Plain space-separated digits as rendered by ESPN get_text (superscripts often stripped)
    if re.fullmatch(r"(?:\d+\s*)+", s):
        return True
    return False


def clean_score_line(s: str) -> str:
    s = s.replace("^{", "(").replace("}", ")")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_status_line(s: str) -> bool:
    u = s.upper().strip()
    return u in {"FINAL", "RETIRED", "WALKOVER", "W/O", "WO", "ABD", "DEF"}


def looks_like_round_line(s: str) -> bool:
    u = s.upper().strip()
    if is_status_line(s):
        return False
    keys = [
        "QUALIFYING 1ST ROUND",
        "QUALIFYING 2ND ROUND",
        "QUALIFYING 3RD ROUND",
        "ROUND OF 128",
        "ROUND OF 64",
        "ROUND OF 32",
        "ROUND OF 16",
        "QUARTERFINAL",
        "QUARTERFINALS",
        "SEMIFINAL",
        "SEMIFINALS",
        "CHAMPIONSHIP",
        "GROUP STAGE",
    ]
    return any(k in u for k in keys)


def normalize_espn_round_label(s: str) -> str:
    u = s.upper().strip()
    mapping = {
        "FINAL": "F",
        "SEMIFINAL": "SF",
        "SEMIFINALS": "SF",
        "QUARTERFINAL": "QF",
        "QUARTERFINALS": "QF",
        "ROUND OF 16": "R16",
        "ROUND OF 32": "R32",
        "ROUND OF 64": "R64",
        "ROUND OF 128": "R128",
        "QUALIFYING 1ST ROUND": "Q1",
        "QUALIFYING 2ND ROUND": "Q2",
        "QUALIFYING 3RD ROUND": "Q3",
        "GROUP STAGE": "RR",
        "CHAMPIONSHIP": "F",
    }
    for key, val in mapping.items():
        if key in u:
            return val
    return ""


def is_noise_line(s: str) -> bool:
    x = s.strip().lower()
    if not x:
        return True
    bad = [
        "skip to main content",
        "latest tennis videos",
        "tennis news",
        "all tennis news",
        "watch tennis on espn",
        "tickets",
        "home",
        "scores",
        "schedule",
        "rankings",
        "players",
        "image",
    ]
    return any(b in x for b in bad)


def is_seed_or_rank_line(s: str) -> bool:
    """ESPN often emits seed / ranking as its own text line between status and names."""
    t = s.strip()
    if re.fullmatch(r"\d{1,3}", t):
        return True
    if re.fullmatch(r"\([\d\s]+\)", t):
        return True
    return False


def looks_like_player_line(s: str) -> bool:
    t = s.strip()
    if not t or is_noise_line(t):
        return False
    if is_status_line(t):
        return False
    if looks_like_round_line(t):
        return False
    if is_score_line(t):
        return False
    if is_seed_or_rank_line(t):
        return False
    if t == "Men's Singles":
        return False
    low = t.lower()
    if "court" in low or "pista" in low or "stadium" in low:
        return False
    if not re.search(r"[A-Za-z]", t):
        return False
    parts = t.split()
    if len(parts) < 2:
        return False
    return True


def _next_player_line(lines: list[str], j: int, limit: int) -> tuple[int, str] | None:
    while j < limit and j < len(lines):
        if looks_like_player_line(lines[j]):
            return j, lines[j]
        j += 1
    return None


def _next_score_line(lines: list[str], j: int, limit: int) -> tuple[int, str] | None:
    while j < limit and j < len(lines):
        if is_score_line(lines[j]):
            return j, lines[j]
        j += 1
    return None


def _next_score_block(lines: list[str], j: int, limit: int) -> tuple[int, str] | None:
    """Join consecutive score-only lines (ESPN sometimes prints one set per line)."""
    if j >= limit or j >= len(lines):
        return None
    parts: list[str] = []
    start_j = j
    while j < limit and j < len(lines) and is_score_line(lines[j]):
        parts.append(lines[j])
        j += 1
    if not parts:
        return None
    combined = clean_score_line(" ".join(parts))
    combined = re.sub(r"\s+", " ", combined).strip()
    return j - 1, combined


def extract_daily_results(tournament_name: str, tournament_url: str, day_url: str, day_iso: str) -> list[dict]:
    html = fetch_html(day_url)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    lines = [re.sub(r"\s+", " ", x).strip() for x in soup.get_text("\n").splitlines()]
    lines = [x for x in lines if x and not is_noise_line(x)]
    results: list[dict] = []
    current_round_raw = ""
    current_round_norm = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        if line == "Men's Singles":
            i += 1
            continue
        if is_status_line(line):
            status = line
            window_end = min(len(lines), i + 1 + 24)
            got_p1 = _next_player_line(lines, i + 1, window_end)
            if not got_p1:
                i += 1
                continue
            j1, p1 = got_p1
            got_s1 = _next_score_block(lines, j1 + 1, window_end)
            if not got_s1:
                i += 1
                continue
            j_s1, s1 = got_s1
            got_p2 = _next_player_line(lines, j_s1 + 1, window_end)
            if not got_p2:
                i += 1
                continue
            j2, p2 = got_p2
            got_s2 = _next_score_block(lines, j2 + 1, window_end)
            if not got_s2:
                i += 1
                continue
            _j_s2, s2 = got_s2
            winner_name, loser_name = infer_winner_loser_from_scores(p1, p2, s1, s2)
            score_text = (
                clean_score_line(s1) + SCORE_COL_SEP + clean_score_line(s2)
            )
            score_text = re.sub(r"\s+", " ", score_text).strip()
            if status.upper() == "RETIRED":
                score_text = f"{score_text} RET".strip()
            results.append(
                {
                    "match_date": day_iso,
                    "tournament_name": tournament_name,
                    "round": current_round_norm,
                    "round_raw_espn": current_round_raw,
                    "winner_name": winner_name,
                    "loser_name": loser_name,
                    "score_text": score_text,
                    "espn_tournament_url": tournament_url,
                    "espn_daily_url": day_url,
                }
            )
            i = _j_s2 + 1
            continue
        if looks_like_round_line(line):
            current_round_raw = line
            current_round_norm = normalize_espn_round_label(line)
            i += 1
            continue
        i += 1
    out: list[dict] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for r in results:
        if any(x in (r.get("winner_name") or "").lower() for x in ("women", "wta")):
            continue
        k = (
            normalize_name(r["winner_name"]),
            normalize_name(r["loser_name"]),
            r["match_date"],
            normalize_tournament_name(r["tournament_name"]),
            str(r.get("round") or ""),
        )
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def build_daily_urls(scoreboard_url: str, start_dt: date, stop_dt: date) -> list[tuple[str, str]]:
    urls: list[tuple[str, str]] = []
    base = scoreboard_base_url(scoreboard_url)
    d = start_dt
    while d <= stop_dt:
        ymd = d.strftime("%Y%m%d")
        urls.append((d.isoformat(), f"{base}/date/{ymd}"))
        d += timedelta(days=1)
    return urls


def scrape_tournament(t: dict, scrape_cutoff: date) -> tuple[list[dict], int]:
    tour_url = scoreboard_base_url(t["scoreboard_url"])
    print(f"\nScraping tournament: {t['tournament_name']}")
    print(f"  Scoreboard URL: {tour_url}")
    html = fetch_html(tour_url)
    if not html:
        return [], 0
    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else (soup.title.get_text(" ", strip=True) if soup.title else t["tournament_name"])
    title = re.sub(r"^\d{4}\s+", "", title)
    title = re.sub(r"\s+Scores$", "", title, flags=re.I)
    text = soup.get_text(" ", strip=True)
    start_dt, end_dt = parse_date_range(text)
    if not start_dt:
        print("  [WARN] Could not parse tournament date range; skipping")
        return [], 0
    scrape_end = min(scrape_cutoff, end_dt or scrape_cutoff)
    if scrape_end < start_dt:
        print("  [INFO] Tournament starts after scrape cutoff; nothing to scrape yet")
        return [], 0

    daily_urls = build_daily_urls(tour_url, start_dt, scrape_end)
    all_rows: list[dict] = []
    pages = 0
    for day_iso, day_url in daily_urls:
        print(f"  Visiting date page: {day_url}")
        rows = extract_daily_results(title, tour_url, day_url, day_iso)
        print(f"    matches found: {len(rows)}")
        all_rows.extend(rows)
        pages += 1
        sleep_polite()
    return all_rows, pages


def add_match_keys(df: pd.DataFrame, alias_map: dict[str, str], winner_col: str, loser_col: str, tournament_col: str, round_col: str) -> pd.DataFrame:
    out = df.copy()
    out["_tw"] = out[winner_col].astype(str).map(normalize_name)
    out["_tl"] = out[loser_col].astype(str).map(normalize_name)
    out["_tt"] = out[tournament_col].astype(str).map(lambda x: canonical_tournament(x, alias_map))
    out["_tr"] = out.get(round_col, "").astype(str).str.upper().str.strip() if round_col in out.columns else ""
    return out


def backfill_match_dates_from_tournament_context(
    completed: pd.DataFrame,
    left: pd.DataFrame,
    espn_df: pd.DataFrame,
    alias_map: dict[str, str],
) -> int:
    """
    Fill match_date for rows the ESPN row-matcher missed by borrowing the calendar
    day implied by peers: same round, same tournament, and/or the same tournament's
    ESPN daily pages that already matched other rows in this file.
    """
    if espn_df is None or len(espn_df) == 0:
        return 0

    er = add_match_keys(
        espn_df,
        alias_map=alias_map,
        winner_col="winner_name",
        loser_col="loser_name",
        tournament_col="tournament_name",
        round_col="round",
    )

    by_tt: dict[str, list] = defaultdict(list)
    for idx in left.index:
        tt = str(left.at[idx, "_tt"] or "").strip()
        if tt:
            by_tt[tt].append(idx)

    filled = 0

    def apply_date(idx: int, date_val: str, reason: str, tourn_label: str) -> None:
        nonlocal filled
        if str(completed.at[idx, "match_date"] or "").strip():
            return
        completed.at[idx, "match_date"] = date_val
        completed.at[idx, "date_source"] = reason
        if not str(completed.at[idx, "espn_match_source"] or "").strip():
            completed.at[idx, "espn_match_source"] = "peer_inference"
        idxs_peer = by_tt.get(str(left.at[idx, "_tt"] or "").strip(), [])
        if not str(completed.at[idx, "espn_tournament_url"] or "").strip():
            for pj in idxs_peer:
                if str(completed.at[pj, "match_date"] or "").strip() != date_val:
                    continue
                u = str(completed.at[pj, "espn_tournament_url"] or "").strip()
                if u:
                    completed.at[idx, "espn_tournament_url"] = u
                    break
        print(
            f"  [DATE BACKFILL] {tourn_label} | "
            f"{completed.at[idx, 'winner_name']} vs {completed.at[idx, 'loser_name']} "
            f"<- {date_val} ({reason})"
        )
        filled += 1

    for tt, idxs in by_tt.items():
        undated = [
            idx
            for idx in idxs
            if not str(completed.at[idx, "match_date"] or "").strip()
        ]
        if not undated:
            continue
        dated_pairs = [
            (idx, str(completed.at[idx, "match_date"] or "").strip())
            for idx in idxs
            if str(completed.at[idx, "match_date"] or "").strip()
        ]
        if not dated_pairs:
            continue

        tourn_label = str(completed.at[idxs[0], "tournament_name"] or "") or tt
        all_dates = [d for _, d in dated_pairs]
        dated_counter = Counter(all_dates)
        espn_md = er.loc[er["_tt"] == tt, "match_date"].astype(str).str.strip()
        espn_md = espn_md[espn_md.astype(bool) & (espn_md != "nan")]
        espn_days = set(espn_md.unique())
        espn_day_counts = Counter(espn_md.tolist())

        round_dates: dict[str, list[str]] = defaultdict(list)
        for idx, d in dated_pairs:
            tr = str(left.at[idx, "_tr"] or "").strip().upper()
            if tr:
                round_dates[tr].append(d)

        for idx in list(undated):
            tr = str(left.at[idx, "_tr"] or "").strip().upper()
            if not tr:
                continue
            dlist = round_dates.get(tr, [])
            if len(dlist) < 1:
                continue
            c = Counter(dlist)
            best, n = c.most_common(1)[0]
            if n == len(dlist) or n >= max(2, 0.5 * len(dlist)):
                apply_date(idx, best, "peer_same_round", tourn_label)

        undated = [
            idx
            for idx in idxs
            if not str(completed.at[idx, "match_date"] or "").strip()
        ]
        if not undated:
            continue

        u_dates = sorted(set(all_dates))
        if len(u_dates) == 1:
            for idx in undated:
                apply_date(idx, u_dates[0], "peer_single_tournament_day", tourn_label)
            continue

        overlap = sorted(espn_days & set(all_dates))
        if len(overlap) == 1:
            for idx in undated:
                apply_date(idx, overlap[0], "peer_espn_day_overlap", tourn_label)
            continue
        if overlap:
            best_ov = max(
                overlap,
                key=lambda d: (espn_day_counts.get(d, 0), dated_counter.get(d, 0)),
            )
            for idx in undated:
                apply_date(idx, best_ov, "peer_espn_busiest_matched_day", tourn_label)
            continue

        mode_d, mode_n = dated_counter.most_common(1)[0]
        if mode_n >= max(2, 0.45 * len(all_dates)):
            for idx in undated:
                apply_date(idx, mode_d, "peer_tournament_date_majority", tourn_label)

    return filled


def main() -> None:
    ensure_dirs()
    alias_map = load_alias_map()

    tournaments = extract_schedule_tournaments()
    if not tournaments:
        print("[ERROR] No ATP tournament scoreboard URLs found from schedule page.")
        return
    print(f"Detected ATP tournaments: {len(tournaments)}")

    scrape_cutoff = (
        date.today()
        if INCLUDE_TODAY
        else date.today() - timedelta(days=1)
    )
    if INCLUDE_TODAY:
        print(
            f"[SCRAPE] Date window: through TODAY ({scrape_cutoff.isoformat()}) "
            f"(INCLUDE_TODAY=True)"
        )
    else:
        print(
            f"[SCRAPE] Date window: through YESTERDAY ({scrape_cutoff.isoformat()}) "
            f"(INCLUDE_TODAY=False)"
        )

    all_espn_rows: list[dict] = []
    pages_scraped = 0
    for i, t in enumerate(tournaments, 1):
        print(f"\n[{i}/{len(tournaments)}] {t['tournament_name']}")
        rows, n_pages = scrape_tournament(t, scrape_cutoff)
        all_espn_rows.extend(rows)
        pages_scraped += n_pages

    scraped_ts = now_iso()
    espn_all = pd.DataFrame(all_espn_rows)
    if len(espn_all):
        espn_all.insert(0, "scraped_at", scraped_ts)
    espn_all.to_csv(ESPN_ALL_MATCHES_CSV, index=False)
    print(f"\nSaved all ESPN matches: {ESPN_ALL_MATCHES_CSV} ({len(espn_all)} rows)")

    espn_df = espn_all.drop(columns=["scraped_at"], errors="ignore") if len(espn_all) else pd.DataFrame()
    if len(espn_df):
        espn_df = espn_df.drop_duplicates(
            subset=["match_date", "tournament_name", "winner_name", "loser_name", "score_text"],
            keep="first",
        )
    espn_df.to_csv(DAILY_RESULTS_CSV, index=False)
    print(f"Saved deduped ESPN rows (for merge): {DAILY_RESULTS_CSV} ({len(espn_df)} rows)")

    if not INPUT_CSV.exists():
        print(f"\n[INFO] No input CSV at {INPUT_CSV}; skipped merging dates into completed matches.")
        print("\n=== Summary ===")
        print(f"Tournaments scraped: {len(tournaments)}")
        print(f"Date pages scraped: {pages_scraped}")
        print(f"ESPN matches found: {len(espn_all)}")
        print(f"All matches CSV: {ESPN_ALL_MATCHES_CSV}")
        print(f"Deduped CSV: {DAILY_RESULTS_CSV}")
        return

    print("\nLoading existing completed matches...")
    completed = pd.read_csv(INPUT_CSV)
    print(f"Input rows: {len(completed)}")

    # Prepare merge
    if "match_date" not in completed.columns:
        completed["match_date"] = ""
    for c in ("espn_tournament_url", "espn_daily_url", "round_raw_espn", "espn_match_source", "date_source"):
        if c not in completed.columns:
            completed[c] = ""

    left = add_match_keys(
        completed,
        alias_map=alias_map,
        winner_col="winner_name",
        loser_col="loser_name",
        tournament_col="tournament_name",
        round_col="round",
    )
    right = add_match_keys(
        espn_df if len(espn_df) else pd.DataFrame(columns=["winner_name", "loser_name", "tournament_name", "round"]),
        alias_map=alias_map,
        winner_col="winner_name",
        loser_col="loser_name",
        tournament_col="tournament_name",
        round_col="round",
    )

    # Build indices for matching priority.
    idx1: dict[tuple[str, str, str, str], dict] = {}
    idx2: dict[tuple[str, str, str], dict] = {}
    for _, r in right.iterrows():
        rec = r.to_dict()
        k1 = (rec["_tt"], rec["_tw"], rec["_tl"], rec["_tr"])
        k2 = (rec["_tt"], rec["_tw"], rec["_tl"])
        idx1.setdefault(k1, rec)
        idx2.setdefault(k2, rec)

    alt_index = build_alt_name_index(right)

    matched = 0
    alt_name_matched = 0
    unmatched_rows: list[dict] = []
    for i, r in left.iterrows():
        existing_date = str(r.get("match_date", "") or "").strip()
        if existing_date and not ALLOW_OVERWRITE_DATES:
            continue
        k1 = (r["_tt"], r["_tw"], r["_tl"], r["_tr"])
        k2 = (r["_tt"], r["_tw"], r["_tl"])
        rec = idx1.get(k1) or idx2.get(k2)
        if not rec:
            rec = try_alt_name_match(r, alt_index)
            if rec:
                alt_name_matched += 1
        if rec:
            completed.at[i, "match_date"] = rec.get("match_date", "")
            completed.at[i, "espn_tournament_url"] = rec.get("espn_tournament_url", "")
            completed.at[i, "espn_daily_url"] = rec.get("espn_daily_url", "")
            completed.at[i, "round_raw_espn"] = rec.get("round_raw_espn", "")
            completed.at[i, "espn_match_source"] = "espn"
            completed.at[i, "date_source"] = "espn_schedule_scoreboard"
            matched += 1
        else:
            unmatched_rows.append(
                {
                    "tournament_name": r.get("tournament_name", ""),
                    "winner_name": r.get("winner_name", ""),
                    "loser_name": r.get("loser_name", ""),
                    "round": r.get("round", ""),
                    "score_text": r.get("score_text", ""),
                }
            )

    peer_filled = backfill_match_dates_from_tournament_context(
        completed, left, espn_df, alias_map
    )
    if peer_filled:
        print(f"\n[DATE BACKFILL] Total rows filled from tournament context: {peer_filled}")

    unmatched_rows = []
    for i, r in left.iterrows():
        if str(completed.at[i, "match_date"] or "").strip():
            continue
        unmatched_rows.append(
            {
                "tournament_name": r.get("tournament_name", ""),
                "winner_name": r.get("winner_name", ""),
                "loser_name": r.get("loser_name", ""),
                "round": r.get("round", ""),
                "score_text": r.get("score_text", ""),
            }
        )

    # Save outputs
    completed.to_csv(OUTPUT_CSV, index=False)
    pd.DataFrame(unmatched_rows).to_csv(UNMATCHED_CSV, index=False)

    print("\n=== Summary ===")
    print(f"Tournaments scraped: {len(tournaments)}")
    print(f"Date pages scraped: {pages_scraped}")
    print(f"ESPN matches found (all): {len(espn_all)}")
    print(f"ESPN rows used for merge (deduped): {len(espn_df)}")
    print(f"Rows matched to CSV: {matched}")
    print(f"  (including alternate-name matches: {alt_name_matched})")
    print(f"  (including peer backfill: {peer_filled})")
    print(f"Rows unmatched: {len(unmatched_rows)}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"All ESPN matches: {ESPN_ALL_MATCHES_CSV}")
    print(f"Deduped ESPN: {DAILY_RESULTS_CSV}")
    print(f"Unmatched: {UNMATCHED_CSV}")


if __name__ == "__main__":
    main()
