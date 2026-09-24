"""Reads every ASR model's saved output (no re-running inference) and writes a cross-model
comparison to Tests/Test-Results/audiotranscription/combined/.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS_DIR

MODEL_KEYS = ["whisper", "wav2vec2", "distil_whisper", "whisper_small"]
COMBINED_DIR = os.path.join(RESULTS_DIR, "combined")


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
        summaries.append(s)
        if os.path.exists(pred_path):
            pred_frames[key] = pd.read_csv(pred_path)

    if not summaries:
        print("No results found - run at least one test_<model>.py first.")
        return

    combined = pd.concat(summaries, ignore_index=True)
    combined_path = os.path.join(COMBINED_DIR, "combined_metrics_summary.csv")
    combined.to_csv(combined_path, index=False)
    print(f"Wrote {combined_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(combined["model_key"], combined["wer"], color="#2ab38e")
    ax.set_ylabel("Word Error Rate (lower is better)")
    ax.set_title("Signal 3/4 ASR — WER comparison")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(combined["wer"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(COMBINED_DIR, "wer_comparison.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(combined["model_key"], combined["avg_inference_time"], color="#00b4d8")
    ax.set_ylabel("Avg inference time per clip (s)")
    ax.set_title("Signal 3/4 ASR — inference speed comparison")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(COMBINED_DIR, "inference_time_comparison.png"), dpi=150)
    plt.close(fig)

    headers = ["Model", "WER", "CER", "Avg inference", "N"]
    rows = [[row["model_key"], f"{row['wer']:.3f}", f"{row['cer']:.3f}",
             f"{row['avg_inference_time']:.3f}s", str(row["n_evaluated"])]
            for _, row in combined.iterrows()]
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]

    def format_row(cells):
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    lines = ["Signal 3/4 (speech-to-text) — combined model comparison", "=" * 70, ""]
    lines.append(format_row(headers))
    lines.append("  ".join("-" * w for w in widths))
    for r in rows:
        lines.append(format_row(r))
    with open(os.path.join(COMBINED_DIR, "metrics.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nCharts + metrics.txt saved to: {COMBINED_DIR}/")


if __name__ == "__main__":
    main()
