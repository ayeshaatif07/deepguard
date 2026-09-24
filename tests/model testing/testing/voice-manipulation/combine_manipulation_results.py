"""Reads every voice-manipulation model's saved output (no re-running inference) and writes
a cross-model comparison to Tests/Test-Results/voice-manipulation/combined/.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS_DIR

MODEL_KEYS = ["bothbosu", "propaganda", "goemotions"]
COMBINED_DIR = os.path.join(RESULTS_DIR, "combined")


def best_threshold_accuracy(df):
    """Sweeps prob_manipulative thresholds and returns the best accuracy achievable, plus the
    threshold that achieves it a supplementary diagnostic for miscalibrated model.
    """
    y_true = (df["y_true"] == "manipulative").astype(int).to_numpy()
    probs = df["prob_manipulative"].to_numpy()
    best_acc, best_thresh = 0.0, 0.5
    for t in np.linspace(0.0, 1.0, 101):
        preds = (probs >= t).astype(int)
        acc = (preds == y_true).mean()
        if acc > best_acc:
            best_acc, best_thresh = acc, t
    return best_acc, best_thresh


def main():
    os.makedirs(COMBINED_DIR, exist_ok=True)

    summaries = []
    pred_frames = {}
    for key in MODEL_KEYS:
        model_dir = os.path.join(RESULTS_DIR, key)
        summary_path = os.path.join(model_dir, "metrics_summary.csv")
        pred_path = os.path.join(model_dir, f"predictions_{key}.csv")
        if not os.path.exists(summary_path):
            print(f"Skipping {key}: no metrics_summary.csv found (run test_{key}.py first)")
            continue
        s = pd.read_csv(summary_path)
        s.insert(0, "model_key", key)

        if os.path.exists(pred_path):
            pred_df = pd.read_csv(pred_path)
            pred_frames[key] = pred_df
            best_acc, best_thresh = best_threshold_accuracy(pred_df)
            s["best_threshold_accuracy"] = best_acc
            s["best_threshold"] = best_thresh

        summaries.append(s)

    if not summaries:
        print("No results found - run at least one test_<model>.py first.")
        return

    combined = pd.concat(summaries, ignore_index=True)
    combined_path = os.path.join(COMBINED_DIR, "combined_metrics_summary.csv")
    combined.to_csv(combined_path, index=False)
    print(f"Wrote {combined_path}")

    if len(pred_frames) >= 2:
        keys = list(pred_frames.keys())
        merged = pred_frames[keys[0]][["clip_id", "y_true", "y_pred"]].rename(columns={"y_pred": f"pred_{keys[0]}"})
        for k in keys[1:]:
            merged = merged.merge(
                pred_frames[k][["clip_id", "y_pred"]].rename(columns={"y_pred": f"pred_{k}"}),
                on="clip_id", how="outer"
            )
        pred_cols = [f"pred_{k}" for k in keys]
        merged["all_agree"] = merged[pred_cols].nunique(axis=1) == 1
        merged_path = os.path.join(COMBINED_DIR, "combined_predictions.csv")
        merged.to_csv(merged_path, index=False)
        n_disagree = (~merged["all_agree"]).sum()
        print(f"Wrote {merged_path} ({n_disagree}/{len(merged)} disagreements)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_cols = ["accuracy", "macro_f1", "auc"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(metric_cols))
    width = 0.8 / len(combined)
    for i, (_, row) in enumerate(combined.iterrows()):
        values = [row[c] for c in metric_cols]
        ax.bar(x + i * width, values, width, label=row["model_key"])
    ax.set_xticks(x + width * (len(combined) - 1) / 2)
    ax.set_xticklabels(["Accuracy (0.5 thresh)", "Macro F1", "AUC"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Signal 3 (voice manipulation) — model comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(COMBINED_DIR, "metrics_comparison.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.35
    xpos = np.arange(len(combined))
    ax.bar(xpos - width/2, combined["accuracy"] * 100, width, label="Default (0.5) threshold", color="#ff6b6b")
    if "best_threshold_accuracy" in combined.columns:
        ax.bar(xpos + width/2, combined["best_threshold_accuracy"] * 100, width, label="Best possible threshold", color="#2ab38e")
    ax.set_xticks(xpos)
    ax.set_xticklabels(combined["model_key"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Signal 3 — default vs. best-threshold accuracy (calibration check)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(COMBINED_DIR, "threshold_calibration.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(combined["model_key"], combined["auc"], color="#00b4d8")
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1.05)
    ax.set_title("Signal 3 (voice manipulation) — AUC comparison")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(combined["auc"]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(COMBINED_DIR, "auc_comparison.png"), dpi=150)
    plt.close(fig)

    lines = ["Signal 3 (voice manipulation) — combined model comparison", "=" * 70, ""]
    has_threshold = "best_threshold_accuracy" in combined.columns
    headers = ["Model", "Accuracy", "Macro F1", "AUC"]
    if has_threshold:
        headers += ["Best-thresh acc", "Best thresh"]
    rows = []
    for _, row in combined.iterrows():
        r = [row["model_key"], f"{row['accuracy']:.3f}", f"{row['macro_f1']:.3f}", f"{row['auc']:.3f}"]
        if has_threshold:
            r += [f"{row['best_threshold_accuracy']:.3f}", f"{row['best_threshold']:.2f}"]
        rows.append(r)
    widths = [max(len(str(r[i])) for r in ([headers] + rows)) for i in range(len(headers))]
    fmt = lambda r: " | ".join(str(v).ljust(w) for v, w in zip(r, widths))
    lines.append(fmt(headers))
    lines.append("-+-".join("-" * w for w in widths))
    lines.extend(fmt(r) for r in rows)
    with open(os.path.join(COMBINED_DIR, "metrics.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nCharts + metrics.txt saved to: {COMBINED_DIR}/")


if __name__ == "__main__":
    main()
