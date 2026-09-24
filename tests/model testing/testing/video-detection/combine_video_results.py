"""Aggregates the per-model outputs already sitting in Tests/Test-Results/<model>/ (from
test_naman.py, test_cvit.py, test_universal.py, test_efficientnet.py.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import common
from common import RESULTS_DIR, compute_metrics, build_summary_rows, build_results_table

OUTPUT_DIR = os.path.join(RESULTS_DIR, "combined")

# Display-friendly names, matching each script's BaseModel.name
MODEL_DISPLAY_NAMES = {
    "naman": "Naman712/Deep-fake-detection",
    "cvit": "CViT2 (Wodajo & Atnafu, CNN+ViT)",
    "universal": "UniversalFakeDetect (CLIP ViT-L/14 + linear probe)",
    "efficientnet": "EfficientNet-B7-NS (selimsef, DFDC Challenge solution)",
    "yermandy": "yermandy/deepfake-detection (CLIP ViT-L/14, LN-tuned)",
}


def discover_models():
    """Finds every model_key under Test-Results/ that has a
    predictions_<model_key>.csv file - i.e. has actually been run."""
    found = {}
    if not os.path.isdir(RESULTS_DIR):
        return found
    for entry in sorted(os.listdir(RESULTS_DIR)):
        model_dir = os.path.join(RESULTS_DIR, entry)
        pred_path = os.path.join(model_dir, f"predictions_{entry}.csv")
        if os.path.isdir(model_dir) and os.path.exists(pred_path):
            found[entry] = pred_path
    return found


def build_combined_metrics_csv(model_preds: dict):
    all_metrics = {}
    for model_key, pred_path in model_preds.items():
        df = pd.read_csv(pred_path)
        name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        metrics = compute_metrics(df)
        all_metrics[name] = metrics

    rows = build_summary_rows(all_metrics)
    if not rows:
        return None, all_metrics

    out = pd.DataFrame(rows)
    # Best-to-worst by AUC (NaN sorts last)
    out = out.sort_values("auc", ascending=False, na_position="last").reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out, all_metrics


def build_combined_predictions_csv(model_preds: dict):
    combined = None
    for model_key, pred_path in model_preds.items():
        df = pd.read_csv(pred_path)[["path", "y_true", "y_score"]]
        df = df.rename(columns={"y_score": f"{model_key}_score"})
        if combined is None:
            combined = df
        else:
            df = df.drop(columns=["y_true"])
            combined = combined.merge(df, on="path", how="outer")

    if combined is None:
        return None

    combined["label"] = combined["y_true"].map({1: "FAKE", 0: "REAL"})
    score_cols = [c for c in combined.columns if c.endswith("_score")]
    cols = ["path", "label", "y_true"] + score_cols
    return combined[cols]


def write_combined_metrics_txt(all_metrics: dict, output_dir: str, model_preds: dict):
    """Plain-text combined report (metrics.txt), the same table build_results_table already
    produces for each individual model's own metrics.txt.
    """
    header_line, sep_line, row_lines, note_lines = build_results_table(all_metrics)

    lines = []
    lines.append("DeepGuard Signal 1 (video) — combined model comparison")
    lines.append("=" * 70)
    lines.append(f"Models compared : {', '.join(sorted(model_preds))}")
    lines.append("")
    lines.append(header_line)
    lines.append(sep_line)
    lines.extend(row_lines)
    if note_lines:
        lines.append("")
        lines.extend(note_lines)

    for name, m in all_metrics.items():
        if m and m.get("per_class"):
            lines.append("")
            lines.append(f"Per-class breakdown [{name}]:")
            pc_headers = ["Class", "Support", "Accuracy", "Precision", "Recall", "F1"]
            pc_rows = []
            for cls_name, cls_m in m["per_class"].items():
                pc_rows.append([
                    cls_name,
                    str(cls_m["support"]),
                    f"{cls_m['accuracy']:.3f}" if not np.isnan(cls_m['accuracy']) else "N/A",
                    f"{cls_m['precision']:.3f}",
                    f"{cls_m['recall']:.3f}",
                    f"{cls_m['f1']:.3f}",
                ])
            pc_widths = [max(len(str(r[i])) for r in ([pc_headers] + pc_rows)) for i in range(len(pc_headers))]
            pc_fmt = lambda r: "  " + " | ".join(str(v).ljust(w) for v, w in zip(r, pc_widths))
            lines.append(pc_fmt(pc_headers))
            lines.append("  " + "-+-".join("-" * w for w in pc_widths))
            for r in pc_rows:
                lines.append(pc_fmt(r))

    for name, m in all_metrics.items():
        if m:
            lines.append("")
            lines.append(f"Confusion matrix [{name}] (rows=true, cols=pred):")
            cm = m["confusion_matrix"]
            cm_headers = ["True \\ Pred", "real", "fake"]
            cm_rows = [["real"] + [str(v) for v in cm[0]], ["fake"] + [str(v) for v in cm[1]]]
            cm_widths = [max(len(str(r[i])) for r in ([cm_headers] + cm_rows)) for i in range(len(cm_headers))]
            cm_fmt = lambda r: "  " + " | ".join(str(v).ljust(w) for v, w in zip(r, cm_widths))
            lines.append(cm_fmt(cm_headers))
            lines.append("  " + "-+-".join("-" * w for w in cm_widths))
            for r in cm_rows:
                lines.append(cm_fmt(r))

    path = os.path.join(output_dir, "metrics.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Combined metrics text report saved to: {path}")


def save_combined_charts(all_metrics: dict, output_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid_metrics = {n: m for n, m in all_metrics.items() if m is not None}
    if not valid_metrics:
        print("No successful model results to plot.")
        return
    names = list(valid_metrics.keys())

    # 1. Grouped bar: Accuracy/Precision/Recall/F1/AUC across all models
    metric_keys = ["accuracy", "precision", "recall", "f1", "auc"]
    x = np.arange(len(metric_keys))
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, name in enumerate(names):
        m = valid_metrics[name]
        values = [m[k] if not (isinstance(m[k], float) and np.isnan(m[k])) else 0 for k in metric_keys]
        ax.bar(x + i * width, values, width, label=name)
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels([k.upper() for k in metric_keys])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Signal 1 (video) — all models compared")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=150)
    plt.close(fig)

    # 2.
    sorted_names = sorted(names, key=lambda n: valid_metrics[n]["accuracy"], reverse=True)
    accuracies_pct = [valid_metrics[n]["accuracy"] * 100 for n in sorted_names]
    n_totals = [valid_metrics[n]["n_evaluated"] for n in sorted_names]
    n_correct = [round(a / 100 * t) for a, t in zip(accuracies_pct, n_totals)]

    fig, ax = plt.subplots(figsize=(9, 6))
    bar_colors = plt.cm.RdYlGn(np.array(accuracies_pct) / 100)
    bars = ax.bar(sorted_names, accuracies_pct, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="Chance (50%)")
    for bar, pct, correct, total in zip(bars, accuracies_pct, n_correct, n_totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{pct:.1f}%\n({correct}/{total})", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Correctly classified (%)")
    ax.set_title("Signal 1 (video) — percentage of correctly classified videos")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "accuracy_percentage.png"), dpi=150)
    plt.close(fig)

    # 3. Inference time comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    times = [valid_metrics[n]["avg_inference_time"] for n in names]
    ax.bar(names, times, color="#2ab38e")
    ax.set_ylabel("Avg inference time per video (s)")
    ax.set_title("Signal 1 (video) — inference speed compared")
    plt.xticks(rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "inference_time_comparison.png"), dpi=150)
    plt.close(fig)

    # 4. ROC overlay
    fig, ax = plt.subplots(figsize=(7, 7))
    plotted_any = False
    for name in names:
        roc = valid_metrics[name].get("roc_curve")
        if roc is None:
            continue
        fpr, tpr = roc
        ax.plot(fpr, tpr, label=f"{name} (AUC={valid_metrics[name]['auc']:.3f})")
        plotted_any = True
    if plotted_any:
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC curve comparison")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "roc_curve_comparison.png"), dpi=150)
    plt.close(fig)

    # 5. Per-class breakdown, one subplot per model, all side by side
    models_with_pc = [n for n in names if valid_metrics[n].get("per_class")]
    if models_with_pc:
        per_class_metric_keys = ["accuracy", "precision", "recall", "f1"]
        fig, axes = plt.subplots(1, len(models_with_pc), figsize=(5.5 * len(models_with_pc), 5), squeeze=False)
        axes = axes[0]
        class_colors = {"Real": "#2ab38e", "Fake": "#e74c3c"}
        for ax, name in zip(axes, models_with_pc):
            per_class = valid_metrics[name]["per_class"]
            xk = np.arange(len(per_class_metric_keys))
            w = 0.8 / len(per_class)
            for i, (cls_name, cls_m) in enumerate(per_class.items()):
                values = [cls_m[k] for k in per_class_metric_keys]
                ax.bar(xk + i * w, values, w, label=f"{cls_name} (n={cls_m['support']})",
                       color=class_colors.get(cls_name))
            ax.set_xticks(xk + w * (len(per_class) - 1) / 2)
            ax.set_xticklabels([k.upper() for k in per_class_metric_keys])
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Score")
            ax.set_title(name, fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle("Per-class breakdown compared — Real vs Fake")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "per_class_comparison.png"), dpi=150)
        plt.close(fig)

    print(f"\nCharts saved to: {output_dir}/")
    print("  - metrics_comparison.png")
    print("  - accuracy_percentage.png")
    print("  - inference_time_comparison.png")
    print("  - roc_curve_comparison.png")
    print("  - per_class_comparison.png")


def main():
    model_preds = discover_models()
    # combine_video_results.py's own output folder would otherwise be
    # picked up as a "model" on a second run - exclude it explicitly.
    model_preds.pop("combined", None)

    if len(model_preds) < 2:
        raise SystemExit(
            f"Found {len(model_preds)} model result(s) in {RESULTS_DIR} - "
            "need at least 2 to combine. Run at least two of the test_*.py "
            "scripts first."
        )

    print(f"Combining results from: {', '.join(model_preds)}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metrics_csv, all_metrics = build_combined_metrics_csv(model_preds)
    metrics_path = os.path.join(OUTPUT_DIR, "combined_metrics_summary.csv")
    metrics_csv.to_csv(metrics_path, index=False)
    print(f"\nCombined metrics summary saved to: {metrics_path}")
    print(metrics_csv[["rank", "model", "accuracy", "precision", "recall", "f1", "auc", "avg_inference_time"]]
          .to_string(index=False))

    preds_csv = build_combined_predictions_csv(model_preds)
    preds_path = os.path.join(OUTPUT_DIR, "combined_predictions.csv")
    preds_csv.to_csv(preds_path, index=False)
    print(f"\nCombined per-video predictions saved to: {preds_path}")

    write_combined_metrics_txt(all_metrics, OUTPUT_DIR, model_preds)
    save_combined_charts(all_metrics, OUTPUT_DIR)


if __name__ == "__main__":
    main()
