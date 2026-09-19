#!/usr/bin/env python3
"""Shared helpers for Tennis Abstract live tournament scraping (matches.py)."""
from __future__ import annotations

import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_RAW = BASE_DIR / "data" / "raw"

TIMEOUT = 20
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.tennisabstract.com/",
}

# Round prefix on a single line, e.g. R32:, QF:, F:, RR:
MATCH_LINE_START_RE = re.compile(
    r"^(?P<round>(?:R\d+|QF|SF|F|RR\d*)):\s*",
    re.I,
)

_TA_LEFT_TAG = re.compile(
    r"^(?:(?:\(\d+\)|\(Q\)|\(WC\)|\(LL\)|\(PR\)|\(SE\)|\(ALT\))\s*)+",
    re.I,
)
_TA_COUNTRY = re.compile(r"\s*\([A-Z]{3}\)\s*$")


def now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_data_dirs() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)


def fetch_html(url: str, *, retries: int = 2, delay_s: float = 0.6) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(delay_s * (attempt + 1))
    print(f"  [WARN] fetch failed: {url} -> {last_err}")
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


def clean_player_text(s: str) -> str:
    if not s:
        return ""
    t = unicodedata.normalize("NFKD", str(s))
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _ta_player_chunk_to_name(chunk: str) -> str:
    s = chunk.strip()
    s = _TA_LEFT_TAG.sub("", s)
    s = _TA_COUNTRY.sub("", s).strip()
    return clean_player_text(s)


def _split_loser_and_score(tail: str) -> tuple[str, str]:
    """
    tail: loser-side chunk + score, after ' d. '
    Score begins at the first 'N-N' games token (set score) or trailing RET/WO.
    """
    s = tail.strip()
    if not s:
        return "", ""

    m_ret = re.search(r"(?i)\s+RET\s*$", s)
    if m_ret and not re.search(r"\d+\s*-\s*\d+", s):
        return s[: m_ret.start()].strip(), "RET"

    m_score = re.search(r"\s(\d+\s*-\s*\d+)", s)
    if not m_score:
        m_wo = re.search(r"(?i)\s+(W/O|WO)\s*$", s)
        if m_wo:
            return s[: m_wo.start()].strip(), m_wo.group(1).upper()
        return s, ""

    i0 = m_score.start(1)
    loser = s[:i0].strip()
    score = s[i0:].strip()
    return loser, score


def parse_completed_line(line: str) -> Optional[dict]:
    s = line.strip()
    m = MATCH_LINE_START_RE.match(s)
    if not m:
        return None
    rnd = m.group("round").upper()
    rest = s[m.end() :]
    if " d. " not in rest:
        return None
    win_chunk, tail = rest.split(" d. ", 1)
    lose_chunk, score = _split_loser_and_score(tail)
    wn = _ta_player_chunk_to_name(win_chunk)
    ln = _ta_player_chunk_to_name(lose_chunk)
    if not wn or not ln:
        return None
    return {
        "round": rnd,
        "winner_name": wn,
        "loser_name": ln,
        "score_text": score,
    }


def parse_upcoming_line(line: str) -> Optional[dict]:
    s = line.strip()
    m = MATCH_LINE_START_RE.match(s)
    if not m:
        return None
    rnd = m.group("round").upper()
    rest = s[m.end() :]
    mvs = re.search(r"\s+vs\s+", rest, re.I)
    if not mvs:
        return None
    p1 = _ta_player_chunk_to_name(rest[: mvs.start()])
    p2 = _ta_player_chunk_to_name(rest[mvs.end() :])
    if not p1 or not p2:
        return None
    return {
        "round": rnd,
        "player_1_name": p1,
        "player_2_name": p2,
    }


def is_valid_singles_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    low_a, low_b = a.lower(), b.lower()
    for tok in ("bye", "tbd", "n/a"):
        if tok in low_a or tok in low_b:
            return False
    if "/" in a or "/" in b:
        return False
    return True


def plain_lines_from_html_br_segments(html: str) -> list[str]:
    """Split raw HTML on <br> and return de-tagged, stripped text segments."""
    parts = re.split(r"<br\s*/?>", html, flags=re.I)
    out: list[str] = []
    for p in parts:
        t = re.sub(r"<[^>]+>", " ", p)
        t = re.sub(r"\s+", " ", t).strip()
        if t:
            out.append(t)
    return out
