"""Reads every voice-clone model's saved output (no re-running inference) and writes a
cross-model comparison to Tests/Test-Results/voice-clone/combined/.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS_DIR

STANDARD_KEYS = ["melodymachine", "mothecreator", "ast_asvspoof"]
FINETUNED_KEYS = ["ast_asvspoof_finetuned"]
COMBINED_DIR = os.path.join(RESULTS_DIR, "combined")

DATASET_LABELS = {
    "melodymachine": "voice_clone_200_sample (official, 100 real/100 fake, 6 TTS platforms)",
    "mothecreator": "voice_clone_200_sample (official, 100 real/100 fake, 6 TTS platforms)",
    "ast_asvspoof": "voice_clone_200_sample (official, 100 real/100 fake, 6 TTS platforms)",
    "ast_asvspoof_finetuned": "disjoint_130_heldout (50 real + 50 original-fake + 30 gTTS, NOT the official 200-sample set)",
}


def main():
    os.makedirs(COMBINED_DIR, exist_ok=True)

    summaries = []
    pred_frames = {}
    for key in STANDARD_KEYS + FINETUNED_KEYS:
        model_dir = os.path.join(RESULTS_DIR, key)
        summary_path = os.path.join(model_dir, "metrics_summary.csv")
        if not os.path.exists(summary_path):
            print(f"Skipping {key}: no metrics_summary.csv found (run its test/finetune script first)")
            continue
        s = pd.read_csv(summary_path)
        s.insert(0, "model_key", key)
        s.insert(1, "dataset", DATASET_LABELS.get(key, "unknown"))
        summaries.append(s)

        # Predictions CSVs use different filenames/column sets for the standard vs fine-tuned
        # groups only merge the standard group's (identical path/y_true/y_pred schema) into
        if key in STANDARD_KEYS:
            pred_path = os.path.join(model_dir, f"predictions_{key}.csv")
            if os.path.exists(pred_path):
                pred_frames[key] = pd.read_csv(pred_path)

    if not summaries:
        print("No results found - run at least one test_<model>.py / finetune_ast_asvspoof.py first.")
        return

    combined = pd.concat(summaries, ignore_index=True, sort=False)
    combined_path = os.path.join(COMBINED_DIR, "combined_metrics_summary.csv")
    combined.to_csv(combined_path, index=False)
    print(f"Wrote {combined_path}")

    # Merge per-clip predictions on "path" for cross-model agreement -
    # standard-group models only (same 200-clip dataset).
    if len(pred_frames) >= 2:
        keys = list(pred_frames.keys())
        merged = pred_frames[keys[0]][["path", "y_true", "y_pred"]].rename(columns={"y_pred": f"pred_{keys[0]}"})
        for k in keys[1:]:
            merged = merged.merge(
                pred_frames[k][["path", "y_pred"]].rename(columns={"y_pred": f"pred_{k}"}),
                on="path", how="outer"
            )
        pred_cols = [f"pred_{k}" for k in keys]
        merged["all_agree"] = merged[pred_cols].nunique(axis=1) == 1
        merged_path = os.path.join(COMBINED_DIR, "combined_predictions.csv")
        merged.to_csv(merged_path, index=False)
        n_disagree = (~merged["all_agree"]).sum()
        print(f"Wrote {merged_path} ({n_disagree}/{len(merged)} disagreements, standard-group models only)")

    # Charts
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_cols = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "auc"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(metric_cols))
    width = 0.8 / len(combined)
    for i, (_, row) in enumerate(combined.iterrows()):
        values = [row[c] for c in metric_cols]
        label = row["model_key"] + (" (different test set)" if row["model_key"] in FINETUNED_KEYS else "")
        ax.bar(x + i * width, values, width, label=label)
    ax.set_xticks(x + width * (len(combined) - 1) / 2)
    ax.set_xticklabels(["Accuracy", "Macro P", "Macro R", "Macro F1", "AUC"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Signal 2 (voice clone) — model comparison\n(fine-tuned model uses a different, disjoint held-out test set)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(COMBINED_DIR, "metrics_comparison.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ab38e" if k not in FINETUNED_KEYS else "#e07b39" for k in combined["model_key"]]
    ax.bar(combined["model_key"], combined["accuracy"] * 100, color=colors)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Signal 2 (voice clone) — accuracy comparison\n(orange = different/disjoint test set, not directly comparable)")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(combined["accuracy"] * 100):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(COMBINED_DIR, "accuracy_comparison.png"), dpi=150)
    plt.close(fig)

    # Inference-time chart: only the standard group reports avg_inference_time (the fine-
    # tuning script doesn't track per-clip inference time) skip rows with no data rather than
    inf_df = combined.dropna(subset=["avg_inference_time"]) if "avg_inference_time" in combined.columns else combined.iloc[0:0]
    if not inf_df.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(inf_df["model_key"], inf_df["avg_inference_time"], color="#00b4d8")
        ax.set_ylabel("Avg inference time per clip (s)")
        ax.set_title("Signal 2 (voice clone) — inference speed comparison")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(COMBINED_DIR, "inference_time_comparison.png"), dpi=150)
        plt.close(fig)

    # metrics.txt - a clean, readable comparison table
    lines = ["Signal 2 (voice clone) — combined model comparison", "=" * 90, ""]
    lines.append("STANDARD GROUP - all evaluated on the same official 200-clip benchmark")
    lines.append("(voice_clone_200_sample.csv, 100 real / 100 fake across 6 TTS platforms):")
    lines.append("-" * 90)
    header = f"{'Model':32s} {'Accuracy':>9s} {'MacroP':>8s} {'MacroR':>8s} {'MacroF1':>8s} {'AUC':>7s} {'AvgInfer':>9s}"
    lines.append(header)
    lines.append("-" * 90)
    for key in STANDARD_KEYS:
        row = combined[combined["model_key"] == key]
        if row.empty:
            continue
        r = row.iloc[0]
        lines.append(f"{key:32s} {r['accuracy']:>9.3f} {r['macro_precision']:>8.3f} {r['macro_recall']:>8.3f} "
                      f"{r['macro_f1']:>8.3f} {r['auc']:>7.3f} {r['avg_inference_time']:>8.3f}s")
    lines.append("")
    lines.append("FINE-TUNED MODEL - evaluated on a SEPARATE, disjoint 130-clip held-out test")
    lines.append("set (50 real + 50 original-fake + 30 gTTS, none seen during fine-tuning) -")
    lines.append("NOT directly comparable to the standard group above (different test data,")
    lines.append("includes gTTS which the standard 200-clip benchmark does not):")
    lines.append("-" * 90)
    for key in FINETUNED_KEYS:
        row = combined[combined["model_key"] == key]
        if row.empty:
            continue
        r = row.iloc[0]
        lines.append(f"{key:32s} {r['accuracy']:>9.3f} {r['macro_precision']:>8.3f} {r['macro_recall']:>8.3f} "
                      f"{r['macro_f1']:>8.3f} {r['auc']:>7.3f} {'n/a':>9s}")
    lines.append("")
    lines.append("See finetune_ast_asvspoof.py and Tests/Test-Results/voice-clone/ast_asvspoof_finetuned/metrics.txt")
    lines.append("for the fine-tuned model's per-category breakdown (real / original-fake / gTTS individually).")

    with open(os.path.join(COMBINED_DIR, "metrics.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nCharts + metrics.txt saved to: {COMBINED_DIR}/")


if __name__ == "__main__":
    main()
