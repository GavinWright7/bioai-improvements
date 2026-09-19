"""
analyze_historical_calibration.py
=============================================================
For matches in a configurable year window (default 2022–2025), compute
model P(p1 wins) using the same rule as generate_real_predictions:
  - one row per match
  - p1 / p2 = alphabetical on names (not winner/loser)
  - feature rows swapped so p1 is always in the trained "winner" slot

Builds a calibration table: within each predicted-probability bin, compare
mean predicted prob to actual frequency of p1 winning.

Outputs (same folder as this script, Code/calibrate/):
  - historical_calibration_table.csv
  - historical_calibration_curve.png

Run: python analyze_historical_calibration.py
=============================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from tensorflow import keras
except ImportError:
    keras = None

# ========================= CONFIG =========================
DATA_FILE = Path("/Users/gavinwright/Desktop/BIOAI improvements/cleaned_data/data_with_features.csv")
MODELS_DIR = Path("/Users/gavinwright/Desktop/School/BioAI/BIOAI Conference/Code/Models")
# Writes next to this script: Code/calibrate/
OUTPUT_DIR = Path(__file__).resolve().parent

CALIB_FROM_YEAR = 2022
CALIB_TO_YEAR = 2025
SURFACES = ["hard", "clay", "grass"]
N_BINS = 20
PREDICT_BATCH_SIZE = 4096
SAVE_FULL_PREDICTIONS_CSV: Path | None = None  # e.g. OUTPUT_DIR / "all_predictions.csv" (large)

# --- shared with generate_real_predictions / step3b ---


def safe(frame: pd.DataFrame, col: str, fill: float = 0.0) -> np.ndarray:
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce").fillna(fill).values.astype(np.float32)
    return np.full(len(frame), fill, dtype=np.float32)


def build_hard_features(frame: pd.DataFrame) -> np.ndarray:
    elo_diff = (safe(frame, "winner_elo", 1500.0) - safe(frame, "loser_elo", 1500.0)).astype(np.float32)
    recent_elo_diff = (safe(frame, "winner_recent_elo", 1500.0) - safe(frame, "loser_recent_elo", 1500.0)).astype(
        np.float32
    )
    hard_elo_diff = (safe(frame, "winner_hard_elo", 1500.0) - safe(frame, "loser_hard_elo", 1500.0)).astype(
        np.float32
    )
    rank_diff = (safe(frame, "loser_rank", 2000.0) - safe(frame, "winner_rank", 2000.0)).astype(np.float32)
    rank_points_diff = (safe(frame, "winner_rank_points", 0.0) - safe(frame, "loser_rank_points", 0.0)).astype(
        np.float32
    )
    h2h_diff = np.clip(
        safe(frame, "h2h_winner_wins", 0.0) - safe(frame, "h2h_loser_wins", 0.0), -10, 10
    ).astype(np.float32)
    days_rest_diff = np.clip(
        safe(frame, "winner_days_rest", 7.0) - safe(frame, "loser_days_rest", 7.0), -60, 60
    ).astype(np.float32)
    return np.column_stack(
        [elo_diff, recent_elo_diff, hard_elo_diff, rank_diff, rank_points_diff, h2h_diff, days_rest_diff]
    ).astype(np.float32)


def build_clay_features(frame: pd.DataFrame) -> np.ndarray:
    surf = frame["surface"].astype(str).str.lower().str.strip()
    elo_diff = (safe(frame, "winner_elo", 1500.0) - safe(frame, "loser_elo", 1500.0)).astype(np.float32)
    recent_elo_diff = (safe(frame, "winner_recent_elo", 1500.0) - safe(frame, "loser_recent_elo", 1500.0)).astype(
        np.float32
    )
    gs_elo_diff = (safe(frame, "winner_gs_elo", 1500.0) - safe(frame, "loser_gs_elo", 1500.0)).astype(np.float32)
    hard_elo_diff = (safe(frame, "winner_hard_elo", 1500.0) - safe(frame, "loser_hard_elo", 1500.0)).astype(
        np.float32
    )
    clay_elo_diff = (safe(frame, "winner_clay_elo", 1500.0) - safe(frame, "loser_clay_elo", 1500.0)).astype(
        np.float32
    )
    grass_elo_diff = (safe(frame, "winner_grass_elo", 1500.0) - safe(frame, "loser_grass_elo", 1500.0)).astype(
        np.float32
    )
    surface_elo_diff = np.where(
        surf == "hard", hard_elo_diff, np.where(surf == "clay", clay_elo_diff, grass_elo_diff)
    ).astype(np.float32)
    rank_diff = (safe(frame, "loser_rank", 2000.0) - safe(frame, "winner_rank", 2000.0)).astype(np.float32)
    rank_points_diff = (safe(frame, "winner_rank_points", 0.0) - safe(frame, "loser_rank_points", 0.0)).astype(
        np.float32
    )
    h2h_diff = np.clip(
        safe(frame, "h2h_winner_wins", 0.0) - safe(frame, "h2h_loser_wins", 0.0), -10, 10
    ).astype(np.float32)
    days_rest_diff = np.clip(
        safe(frame, "winner_days_rest", 7.0) - safe(frame, "loser_days_rest", 7.0), -60, 60
    ).astype(np.float32)
    surface_wr_diff = (safe(frame, "winner_surface_winrate", 0.5) - safe(frame, "loser_surface_winrate", 0.5)).astype(
        np.float32
    )
    age_diff = (safe(frame, "winner_age", 25.0) - safe(frame, "loser_age", 25.0)).astype(np.float32)
    return np.column_stack(
        [
            elo_diff,
            recent_elo_diff,
            surface_elo_diff,
            gs_elo_diff,
            rank_diff,
            rank_points_diff,
            h2h_diff,
            days_rest_diff,
            surface_wr_diff,
            age_diff,
        ]
    ).astype(np.float32)


def swap_winner_loser_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = list(out.columns)
    done = set()
    for c in cols:
        if c.startswith("winner_"):
            other = "loser_" + c[len("winner_") :]
            if other in out.columns and c not in done:
                tmp = out[c].copy()
                out[c] = out[other].values
                out[other] = tmp.values
                done.add(c)
                done.add(other)
    for c in cols:
        if c.startswith("w_") and not c.startswith("winner"):
            other = "l_" + c[2:]
            if other in out.columns and c not in done:
                tmp = out[c].copy()
                out[c] = out[other].values
                out[other] = tmp.values
                done.add(c)
                done.add(other)
    return out


def batched_predict(model, X_scaled: np.ndarray, batch_size: int) -> np.ndarray:
    out = []
    for start in range(0, len(X_scaled), batch_size):
        chunk = X_scaled[start : start + batch_size]
        out.append(model.predict(chunk, verbose=0).flatten())
    return np.concatenate(out, axis=0)


def process_surface_hard_clay(
    df: pd.DataFrame,
    surface: str,
    model,
    scaler,
    build_fn,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (probs, correct, orig_indices) for rows with valid features.
    orig_indices maps into df.index positions 0..len(df)-1 if df was reset_index.
    """
    n = len(df)
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    w = df["winner_name"].astype(str)
    l = df["loser_name"].astype(str)
    wl = w.str.lower()
    ll = l.str.lower()
    p1 = np.where(wl <= ll, w, l).astype(str)
    wv = w.to_numpy()
    correct = (p1 == wv).astype(np.int32)
    swap_mask = p1 != wv

    idx = np.arange(n, dtype=np.int64)
    idx_no = idx[~swap_mask]
    idx_sw = idx[swap_mask]

    parts_probs = []
    parts_idx = []

    for sub_idx, do_swap in [(idx_no, False), (idx_sw, True)]:
        if len(sub_idx) == 0:
            continue
        sub = df.iloc[sub_idx].copy()
        if do_swap:
            sub = swap_winner_loser_columns(sub)
        X = build_fn(sub)
        valid = ~np.isnan(X).any(axis=1)
        if not valid.any():
            continue
        sub_idx_v = sub_idx[valid]
        Xv = X[valid]
        Xs = scaler.transform(Xv)
        pred = batched_predict(model, Xs, batch_size)
        parts_probs.append(pred)
        parts_idx.append(sub_idx_v)

    if not parts_probs:
        return np.array([]), np.array([]), np.array([])

    all_pred = np.concatenate(parts_probs)
    all_idx = np.concatenate(parts_idx)
    order = np.argsort(all_idx)
    all_pred = all_pred[order]
    all_idx = all_idx[order]
    correct_out = correct[all_idx]
    return all_pred, correct_out, all_idx


def process_surface_grass(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(df)
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    w = df["winner_name"].astype(str)
    l = df["loser_name"].astype(str)
    wl = w.str.lower()
    ll = l.str.lower()
    p1 = np.where(wl <= ll, w, l).astype(str)
    wv = w.to_numpy()
    correct = (p1 == wv).astype(np.int32)
    swap_mask = p1 != wv

    idx = np.arange(n, dtype=np.int64)
    probs = np.full(n, np.nan, dtype=np.float64)

    for sub_idx, do_swap in [(idx[~swap_mask], False), (idx[swap_mask], True)]:
        if len(sub_idx) == 0:
            continue
        sub = df.iloc[sub_idx].copy()
        if do_swap:
            sub = swap_winner_loser_columns(sub)
        we = safe(sub, "winner_elo", 1500.0)
        le = safe(sub, "loser_elo", 1500.0)
        pr = 1.0 / (1.0 + 10.0 ** ((le - we) / 400.0))
        probs[sub_idx] = pr

    valid = ~np.isnan(probs)
    vi = np.where(valid)[0]
    return probs[vi], correct[vi], vi


def build_calibration_table(prob: np.ndarray, correct: np.ndarray, n_bins: int) -> pd.DataFrame:
    edges = np.linspace(0, 1, n_bins + 1)
    bid = np.digitize(prob, edges[1:-1], right=False)
    bid = np.clip(bid, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        m = bid == b
        cnt = int(m.sum())
        lo, hi = edges[b], edges[b + 1]
        if cnt == 0:
            rows.append(
                {
                    "bin_index": b,
                    "bin_low": lo,
                    "bin_high": hi,
                    "count": 0,
                    "avg_predicted_prob": np.nan,
                    "actual_win_rate": np.nan,
                    "calibration_gap": np.nan,
                }
            )
            continue
        ap = float(prob[m].mean())
        aw = float(correct[m].mean())
        rows.append(
            {
                "bin_index": b,
                "bin_low": lo,
                "bin_high": hi,
                "count": cnt,
                "avg_predicted_prob": round(ap, 6),
                "actual_win_rate": round(aw, 6),
                "calibration_gap": round(aw - ap, 6),
            }
        )
    return pd.DataFrame(rows)


def plot_calibration_curve(table: pd.DataFrame, out_path: Path, n_bins: int) -> None:
    t = table[table["count"] > 0].copy()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    sc = ax.scatter(
        t["avg_predicted_prob"],
        t["actual_win_rate"],
        s=np.sqrt(t["count"].clip(lower=1)) * 3,
        c=t["count"],
        cmap="viridis",
        alpha=0.85,
        edgecolors="navy",
        linewidths=0.5,
    )
    ax.plot(t["avg_predicted_prob"], t["actual_win_rate"], color="steelblue", lw=1, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="Matches in bin")
    ax.set_xlabel("Mean predicted P(p1 wins)", fontsize=12)
    ax.set_ylabel("Actual frequency (p1 won)", fontsize=12)
    ax.set_title(
        f"Historical calibration ({n_bins} equal-width bins)\n"
        "Dots above diagonal → underconfident; below → overconfident",
        fontsize=11,
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    print("=" * 60)
    print("HISTORICAL CALIBRATION ({}–{})".format(CALIB_FROM_YEAR, CALIB_TO_YEAR))
    print("=" * 60)

    if not DATA_FILE.exists():
        sys.exit(f"[ERROR] Data not found: {DATA_FILE}")
    if keras is None:
        sys.exit("[ERROR] TensorFlow/Keras required.")
    th = MODELS_DIR / "surface_thresholds.json"
    if not th.exists():
        sys.exit(f"[ERROR] {th} not found — set MODELS_DIR.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hard_model = keras.models.load_model(str(MODELS_DIR / "model_hard.keras"))
    hard_scaler = joblib.load(str(MODELS_DIR / "scaler_hard.pkl"))
    clay_model = keras.models.load_model(str(MODELS_DIR / "model_clay.keras"))
    clay_scaler = joblib.load(str(MODELS_DIR / "scaler_clay.pkl"))

    df = pd.read_csv(DATA_FILE, low_memory=False)
    df["tourney_date"] = pd.to_numeric(df["tourney_date"], errors="coerce")
    df["year"] = (df["tourney_date"] // 10000).astype(int)
    df["surface"] = df["surface"].astype(str).str.lower().str.strip()

    sub = df[
        (df["year"] >= CALIB_FROM_YEAR)
        & (df["year"] <= CALIB_TO_YEAR)
        & (df["surface"].isin(SURFACES))
    ].copy().reset_index(drop=True)
    print(f"Matches loaded ({CALIB_FROM_YEAR}–{CALIB_TO_YEAR}, hard/clay/grass): {len(sub):,}")

    all_prob: list[np.ndarray] = []
    all_correct: list[np.ndarray] = []
    all_year: list[np.ndarray] = []
    all_surf: list[np.ndarray] = []
    all_mid: list[np.ndarray] = []

    for surf in SURFACES:
        sdf = sub[sub["surface"] == surf].copy().reset_index(drop=True)
        if len(sdf) == 0:
            continue
        print(f"  {surf}: {len(sdf):,} matches …")

        if surf == "grass":
            pr, cr, ix = process_surface_grass(sdf)
        elif surf == "hard":
            pr, cr, ix = process_surface_hard_clay(sdf, surf, hard_model, hard_scaler, build_hard_features, PREDICT_BATCH_SIZE)
        else:
            pr, cr, ix = process_surface_hard_clay(sdf, surf, clay_model, clay_scaler, build_clay_features, PREDICT_BATCH_SIZE)

        if len(pr) == 0:
            continue
        all_prob.append(pr.astype(np.float64))
        all_correct.append(cr.astype(np.int32))
        all_year.append(sdf.iloc[ix]["year"].values)
        all_surf.append(np.full(len(ix), surf, dtype=object))
        mid = (
            sdf.iloc[ix]["tourney_id"].astype(str).values
            + "_"
            + sdf.iloc[ix]["match_num"].astype(str).values
        )
        all_mid.append(mid)

    if not all_prob:
        sys.exit("[ERROR] No predictions produced.")

    prob = np.concatenate(all_prob)
    correct = np.concatenate(all_correct)
    years = np.concatenate(all_year)
    surfs = np.concatenate(all_surf)
    match_ids = np.concatenate(all_mid)

    print(f"Total predictions (valid features): {len(prob):,}")
    print(f"Overall p1 win rate: {correct.mean():.4f}")
    print(f"Mean predicted prob: {prob.mean():.4f}")

    # Brier & log loss (clip for stability)
    eps = 1e-15
    pclip = np.clip(prob, eps, 1 - eps)
    brier = float(np.mean((pclip - correct) ** 2))
    logloss = float(-np.mean(correct * np.log(pclip) + (1 - correct) * np.log(1 - pclip)))
    print(f"Brier score: {brier:.4f}  |  Log loss: {logloss:.4f}")

    calib = build_calibration_table(prob, correct, N_BINS)
    out_csv = OUTPUT_DIR / "historical_calibration_table.csv"
    calib.to_csv(out_csv, index=False)
    print(f"Saved table: {out_csv}")

    out_png = OUTPUT_DIR / "historical_calibration_curve.png"
    plot_calibration_curve(calib, out_png, N_BINS)
    print(f"Saved plot:  {out_png}")

    if SAVE_FULL_PREDICTIONS_CSV is not None:
        full = pd.DataFrame(
            {"match_id": match_ids, "year": years, "surface": surfs, "prob": prob, "correct": correct}
        )
        full.to_csv(SAVE_FULL_PREDICTIONS_CSV, index=False)
        print(f"Saved full predictions: {SAVE_FULL_PREDICTIONS_CSV}")

    print("=" * 60)
    print("NOTE: Years that overlap model training will look better than true")
    print("out-of-sample performance. Interpret test-period bins separately if needed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
