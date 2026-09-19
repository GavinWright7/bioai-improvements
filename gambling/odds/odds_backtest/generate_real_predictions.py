import os
import json
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras

DATA_FILE = "/Users/gavinwright/Desktop/School/BioAI/BIOAI Conference/cleaned_data/data_with_features.csv"
MODELS_DIR = "/Users/gavinwright/Desktop/School/BioAI/BIOAI Conference/Code/Models"
OUTPUT_FILE = "/Users/gavinwright/Desktop/BIOAI improvements/gambling/odds/odds_backtest/real_predictions.csv"

TEST_FROM_YEAR = 2022
TEST_TO_YEAR = 2025

def safe(frame, col, fill=0.0):
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce").fillna(fill).values.astype(np.float32)
    return np.full(len(frame), fill, dtype=np.float32)

def build_hard_features_winner_perspective(frame):
    elo_diff = (safe(frame, "winner_elo", 1500.0) - safe(frame, "loser_elo", 1500.0)).astype(np.float32)
    recent_elo_diff = (safe(frame, "winner_recent_elo", 1500.0) - safe(frame, "loser_recent_elo", 1500.0)).astype(np.float32)
    hard_elo_diff = (safe(frame, "winner_hard_elo", 1500.0) - safe(frame, "loser_hard_elo", 1500.0)).astype(np.float32)
    rank_diff = (safe(frame, "loser_rank", 2000.0) - safe(frame, "winner_rank", 2000.0)).astype(np.float32)
    rank_points_diff = (safe(frame, "winner_rank_points", 0.0) - safe(frame, "loser_rank_points", 0.0)).astype(np.float32)
    h2h_diff = np.clip(safe(frame, "h2h_winner_wins", 0.0) - safe(frame, "h2h_loser_wins", 0.0), -10, 10).astype(np.float32)
    days_rest_diff = np.clip(safe(frame, "winner_days_rest", 7.0) - safe(frame, "loser_days_rest", 7.0), -60, 60).astype(np.float32)

    return np.column_stack([
        elo_diff,
        recent_elo_diff,
        hard_elo_diff,
        rank_diff,
        rank_points_diff,
        h2h_diff,
        days_rest_diff,
    ]).astype(np.float32)

def build_clay_features_winner_perspective(frame):
    surf = frame["surface"].astype(str).str.lower().str.strip()

    elo_diff = (safe(frame, "winner_elo", 1500.0) - safe(frame, "loser_elo", 1500.0)).astype(np.float32)
    recent_elo_diff = (safe(frame, "winner_recent_elo", 1500.0) - safe(frame, "loser_recent_elo", 1500.0)).astype(np.float32)
    gs_elo_diff = (safe(frame, "winner_gs_elo", 1500.0) - safe(frame, "loser_gs_elo", 1500.0)).astype(np.float32)
    hard_elo_diff = (safe(frame, "winner_hard_elo", 1500.0) - safe(frame, "loser_hard_elo", 1500.0)).astype(np.float32)
    clay_elo_diff = (safe(frame, "winner_clay_elo", 1500.0) - safe(frame, "loser_clay_elo", 1500.0)).astype(np.float32)
    grass_elo_diff = (safe(frame, "winner_grass_elo", 1500.0) - safe(frame, "loser_grass_elo", 1500.0)).astype(np.float32)

    surface_elo_diff = np.where(
        surf == "hard",
        hard_elo_diff,
        np.where(surf == "clay", clay_elo_diff, grass_elo_diff)
    ).astype(np.float32)

    rank_diff = (safe(frame, "loser_rank", 2000.0) - safe(frame, "winner_rank", 2000.0)).astype(np.float32)
    rank_points_diff = (safe(frame, "winner_rank_points", 0.0) - safe(frame, "loser_rank_points", 0.0)).astype(np.float32)
    h2h_diff = np.clip(safe(frame, "h2h_winner_wins", 0.0) - safe(frame, "h2h_loser_wins", 0.0), -10, 10).astype(np.float32)
    days_rest_diff = np.clip(safe(frame, "winner_days_rest", 7.0) - safe(frame, "loser_days_rest", 7.0), -60, 60).astype(np.float32)
    surface_wr_diff = (safe(frame, "winner_surface_winrate", 0.5) - safe(frame, "loser_surface_winrate", 0.5)).astype(np.float32)
    age_diff = (safe(frame, "winner_age", 25.0) - safe(frame, "loser_age", 25.0)).astype(np.float32)

    return np.column_stack([
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
    ]).astype(np.float32)

def make_swapped_frame(frame):
    swapped = frame.copy()

    pair_cols = [
        ("winner_id", "loser_id"),
        ("winner_seed", "loser_seed"),
        ("winner_entry", "loser_entry"),
        ("winner_name", "loser_name"),
        ("winner_hand", "loser_hand"),
        ("winner_ht", "loser_ht"),
        ("winner_ioc", "loser_ioc"),
        ("winner_age", "loser_age"),
        ("winner_rank", "loser_rank"),
        ("winner_rank_points", "loser_rank_points"),
        ("winner_elo", "loser_elo"),
        ("winner_hard_elo", "loser_hard_elo"),
        ("winner_clay_elo", "loser_clay_elo"),
        ("winner_grass_elo", "loser_grass_elo"),
        ("winner_gs_elo", "loser_gs_elo"),
        ("winner_recent_elo", "loser_recent_elo"),
        ("h2h_winner_wins", "h2h_loser_wins"),
        ("winner_surface_winrate", "loser_surface_winrate"),
        ("winner_days_rest", "loser_days_rest"),
        ("winner_ace_rate", "loser_ace_rate"),
        ("winner_df_rate", "loser_df_rate"),
        ("winner_1stIn_pct", "loser_1stIn_pct"),
        ("winner_1stWon_pct", "loser_1stWon_pct"),
        ("winner_bpSaved_pct", "loser_bpSaved_pct"),
    ]

    for a, b in pair_cols:
        if a in swapped.columns and b in swapped.columns:
            temp = swapped[a].copy()
            swapped[a] = swapped[b]
            swapped[b] = temp

    return swapped

def normalize_name(x):
    return str(x).strip().lower()

def main():
    print("=" * 60)
    print("GENERATE REAL PREDICTIONS CSV")
    print("=" * 60)

    thresholds_path = os.path.join(MODELS_DIR, "surface_thresholds.json")
    if not os.path.exists(thresholds_path):
        raise FileNotFoundError(f"Thresholds file not found: {thresholds_path}")

    with open(thresholds_path, "r") as f:
        thresholds = json.load(f)

    hard_threshold = float(thresholds["hard"])
    clay_threshold = float(thresholds["clay"])

    print("Loading models...")
    hard_model = keras.models.load_model(os.path.join(MODELS_DIR, "model_hard.keras"))
    hard_scaler = joblib.load(os.path.join(MODELS_DIR, "scaler_hard.pkl"))
    clay_model = keras.models.load_model(os.path.join(MODELS_DIR, "model_clay.keras"))
    clay_scaler = joblib.load(os.path.join(MODELS_DIR, "scaler_clay.pkl"))

    print("Loading data...")
    df = pd.read_csv(DATA_FILE, low_memory=False)

    df["tourney_date"] = pd.to_numeric(df["tourney_date"], errors="coerce")
    df["year"] = (df["tourney_date"] // 10000).astype("Int64")
    df["surface"] = df["surface"].astype(str).str.lower().str.strip()

    test_df = df[
        (df["year"] >= TEST_FROM_YEAR) &
        (df["year"] <= TEST_TO_YEAR) &
        (df["surface"].isin(["hard", "clay", "grass"]))
    ].copy().reset_index(drop=True)

    print(f"Matches in test window: {len(test_df):,}")

    rows = []

    for surface in ["hard", "clay", "grass"]:
        sdf = test_df[test_df["surface"] == surface].copy().reset_index(drop=True)
        if len(sdf) == 0:
            continue

        print(f"Processing {surface}: {len(sdf):,} matches")

        p1_is_winner = (
            sdf["winner_name"].astype(str).str.lower().str.strip() <=
            sdf["loser_name"].astype(str).str.lower().str.strip()
        )

        p1_frame = sdf.copy()
        loser_as_p1_frame = make_swapped_frame(sdf)

        if surface == "grass":
            winner_elo_original = safe(sdf, "winner_elo", 1500.0)
            loser_elo_original = safe(sdf, "loser_elo", 1500.0)

            p1_elo = np.where(p1_is_winner.values, winner_elo_original, loser_elo_original)
            p2_elo = np.where(p1_is_winner.values, loser_elo_original, winner_elo_original)

            probs = 1.0 / (1.0 + 10.0 ** ((p2_elo - p1_elo) / 400.0))
            valid_mask = np.ones(len(sdf), dtype=bool)

        elif surface == "hard":
            X_winner = build_hard_features_winner_perspective(p1_frame)
            X_loser = build_hard_features_winner_perspective(loser_as_p1_frame)

            X_final = np.where(p1_is_winner.values[:, None], X_winner, X_loser)
            valid_mask = ~np.isnan(X_final).any(axis=1)

            X_scaled = hard_scaler.transform(X_final[valid_mask])
            probs_valid = hard_model.predict(X_scaled, verbose=0).flatten().astype(float)

            probs = np.full(len(sdf), np.nan, dtype=float)
            probs[valid_mask] = probs_valid

        else:
            X_winner = build_clay_features_winner_perspective(p1_frame)
            X_loser = build_clay_features_winner_perspective(loser_as_p1_frame)

            X_final = np.where(p1_is_winner.values[:, None], X_winner, X_loser)
            valid_mask = ~np.isnan(X_final).any(axis=1)

            X_scaled = clay_scaler.transform(X_final[valid_mask])
            probs_valid = clay_model.predict(X_scaled, verbose=0).flatten().astype(float)

            probs = np.full(len(sdf), np.nan, dtype=float)
            probs[valid_mask] = probs_valid

        kept = sdf[valid_mask].copy().reset_index(drop=True)
        kept_p1_is_winner = p1_is_winner[valid_mask].reset_index(drop=True)
        kept_probs = probs[valid_mask]

        for i in range(len(kept)):
            row = kept.iloc[i]
            p1_wins = bool(kept_p1_is_winner.iloc[i])

            if p1_wins:
                p1_name = normalize_name(row["winner_name"])
                p2_name = normalize_name(row["loser_name"])
                correct = 1
            else:
                p1_name = normalize_name(row["loser_name"])
                p2_name = normalize_name(row["winner_name"])
                correct = 0

            rows.append({
                "match_id": f"{row['tourney_id']}_{row['match_num']}",
                "surface": row["surface"],
                "year": int(row["year"]),
                "round": row["round"],
                "p1_name": p1_name,
                "p2_name": p2_name,
                "prob": float(kept_probs[i]),
                "correct": correct,
            })

    output_df = pd.DataFrame(rows)

    output_df = output_df.drop_duplicates(subset=["match_id"]).reset_index(drop=True)
    output_df = output_df.sort_values(["year", "surface", "match_id"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    output_df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Rows: {len(output_df):,}")
    print(f"Unique match_id: {output_df['match_id'].nunique():,}")
    print("\nSample:")
    print(output_df.head(10).to_string(index=False))

    print("\nProbability summary:")
    print(output_df["prob"].describe())

    print("\nCorrect rate:")
    print(output_df["correct"].mean())

if __name__ == "__main__":
    main()