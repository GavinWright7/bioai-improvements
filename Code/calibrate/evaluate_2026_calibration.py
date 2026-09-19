import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PREDICTIONS_PATH = "gambling/2026_predictions.csv"
BINS_OUTPUT_PATH = "calibration_2026_bins.csv"
PLOT_OUTPUT_PATH = "calibration_2026_plot.png"
TARGET_YEAR = 2026


def main():
    print("=== 2026 Calibration Evaluation ===")
    print(f"Loading predictions from: {PREDICTIONS_PATH}")

    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(f"Predictions file not found: {PREDICTIONS_PATH}")

    df = pd.read_csv(PREDICTIONS_PATH)

    required_cols = {"prob", "correct", "year"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    df_2026 = df[df["year"] == TARGET_YEAR].copy()
    df_2026["prob"] = pd.to_numeric(df_2026["prob"], errors="coerce")
    df_2026["correct"] = pd.to_numeric(df_2026["correct"], errors="coerce")

    df_2026 = df_2026.dropna(subset=["prob", "correct"])
    df_2026 = df_2026[(df_2026["prob"] > 0.0) & (df_2026["prob"] < 1.0)].copy()

    n = len(df_2026)
    print(f"2026 matches used: {n}")

    if n == 0:
        print("No valid 2026 rows after filtering. Exiting without bin/plot outputs.")
        return

    overall_accuracy = float(df_2026["correct"].mean())
    avg_pred_prob = float(df_2026["prob"].mean())
    print(f"Overall accuracy (mean correct): {overall_accuracy:.4f}")
    print(f"Average predicted probability: {avg_pred_prob:.4f}")

    bin_edges = np.linspace(0.0, 1.0, 11)
    bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}" for i in range(10)]

    df_2026["bin"] = pd.cut(
        df_2026["prob"],
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True,
        right=False,
    )

    grouped = (
        df_2026.groupby("bin", observed=False)
        .agg(
            count=("prob", "size"),
            avg_predicted=("prob", "mean"),
            actual_win_rate=("correct", "mean"),
        )
        .reset_index()
    )
    grouped["gap"] = grouped["actual_win_rate"] - grouped["avg_predicted"]

    bins_df = grouped[grouped["count"] > 0].copy()

    print("\nCalibration by bin")
    print("bin_range | count | avg_predicted | actual_win_rate | gap")
    print("-" * 62)
    for _, row in bins_df.iterrows():
        print(
            f"{row['bin']:<9} | "
            f"{int(row['count']):>5} | "
            f"{row['avg_predicted']:.4f} | "
            f"{row['actual_win_rate']:.4f} | "
            f"{row['gap']:+.4f}"
        )

    mace = float(np.mean(np.abs(df_2026["prob"] - df_2026["correct"])))
    brier = float(np.mean((df_2026["prob"] - df_2026["correct"]) ** 2))
    weighted_abs_bin_gap = float(
        np.average(np.abs(bins_df["gap"]), weights=bins_df["count"])
    )

    print("\nOverall metrics")
    print(f"MACE (mean abs(prob - correct)): {mace:.6f}")
    print(f"Brier score: {brier:.6f}")
    print(f"Weighted average absolute bin gap: {weighted_abs_bin_gap:.6f}")

    bins_df_out = bins_df.rename(columns={"bin": "bin_range"})
    bins_df_out.to_csv(BINS_OUTPUT_PATH, index=False)
    print(f"\nSaved bins table: {BINS_OUTPUT_PATH}")

    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration (y=x)")
    plt.plot(
        bins_df["avg_predicted"],
        bins_df["actual_win_rate"],
        marker="o",
        linestyle="-",
        color="tab:blue",
        label="2026 bins",
    )
    for _, row in bins_df.iterrows():
        plt.annotate(str(int(row["count"])), (row["avg_predicted"], row["actual_win_rate"]))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Average predicted probability")
    plt.ylabel("Actual win rate")
    plt.title("Reliability Plot (2026 Only)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_PATH, dpi=150)
    plt.close()
    print(f"Saved reliability plot: {PLOT_OUTPUT_PATH}")

    mean_signed_gap = float(np.average(bins_df["gap"], weights=bins_df["count"]))
    if mean_signed_gap > 0.01:
        confidence_read = "underconfident"
    elif mean_signed_gap < -0.01:
        confidence_read = "overconfident"
    else:
        confidence_read = "roughly well-calibrated on average"

    print("\nInterpretation")
    print(f"Model confidence tendency: {confidence_read}")
    print(f"Average calibration error (weighted abs bin gap): {weighted_abs_bin_gap:.4f}")
    if weighted_abs_bin_gap <= 0.05:
        usability = "Probabilities look reasonably usable for betting with discipline."
    elif weighted_abs_bin_gap <= 0.10:
        usability = "Probabilities are somewhat usable, but calibration adjustment is recommended."
    else:
        usability = "Probabilities are likely too miscalibrated for reliable betting without recalibration."
    print(f"Betting usability note: {usability}")


if __name__ == "__main__":
    main()
