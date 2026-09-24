"""Aggregates the per-model outputs already sitting in Tests/Test-Results/image-
detection/<model>/ (from test_9999.py, test_v2.py.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import common
from common import RESULTS_DIR, CLASS_NAMES, compute_metrics

OUTPUT_DIR = os.path.join(RESULTS_DIR, "combined")

# Display-friendly names, matching each script's BaseModel.name
MODEL_DISPLAY_NAMES = {
    "9999": "AI-vs-Deepfake-vs-Real-9999 (SigLIP2)",
    "v2": "AI-vs-Deepfake-vs-Real-v2.0 (SigLIP2)",
    "original": "AI-vs-Deepfake-vs-Real (original ViT)",
}


def discover_models():
    """Finds every model_key under Test-Results/image-detection/ that has a
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
        all_metrics[name] = compute_metrics(df)

    rows = []
    for name, m in all_metrics.items():
        if m is None:
            continue
        row = {k: v for k, v in m.items() if k not in ("per_class", "confusion_matrix", "labels_present")}
        row["model"] = name
        for cls, cls_m in m["per_class"].items():
            for mk, mv in cls_m.items():
                row[f"{cls.lower()}_{mk}"] = mv
        rows.append(row)

    if not rows:
        return None, all_metrics

    out = pd.DataFrame(rows)
    out = out.sort_values("macro_f1", ascending=False, na_position="last").reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out, all_metrics


def build_combined_predictions_csv(model_preds: dict):
    combined = None
    for model_key, pred_path in model_preds.items():
        df = pd.read_csv(pred_path)[["path", "y_true", "y_pred"]]
        df = df.rename(columns={"y_pred": f"{model_key}_pred"})
        if combined is None:
            combined = df
        else:
            df = df.drop(columns=["y_true"])
            combined = combined.merge(df, on="path", how="outer")

    if combined is None:
        return None

    pred_cols = [c for c in combined.columns if c.endswith("_pred")]
    combined["all_agree"] = combined[pred_cols].nunique(axis=1) == 1
    cols = ["path", "y_true"] + pred_cols + ["all_agree"]
    return combined[cols]


def write_combined_metrics_txt(metrics_csv: pd.DataFrame, all_metrics: dict, output_dir: str, n_disagree: int, n_total: int):
    """Plain-text report mirroring each individual model's metrics.txt, but
    covering all models side by side - readable without opening a CSV."""
    lines = []
    lines.append("DeepGuard Signal 1b (image) — combined model comparison")
    lines.append("=" * 70)
    lines.append(f"Models compared : {len(metrics_csv)}")
    lines.append(f"Images per model: {n_total}")
    lines.append(f"Images where all models disagree with each other: {n_disagree}/{n_total}")
    lines.append("")

    headers = ["Rank", "Model", "Accuracy", "Macro P", "Macro R", "Macro F1", "Macro AUC", "Avg time (s)"]
    rows = []
    for _, r in metrics_csv.iterrows():
        rows.append([
            str(r["rank"]), r["model"],
            f"{r['accuracy']:.3f}", f"{r['macro_precision']:.3f}", f"{r['macro_recall']:.3f}",
            f"{r['macro_f1']:.3f}",
            f"{r['macro_auc_ovr']:.3f}" if not np.isnan(r["macro_auc_ovr"]) else "N/A",
            f"{r['avg_inference_time']:.3f}",
        ])
    widths = [max(len(str(row[i])) for row in ([headers] + rows)) for i in range(len(headers))]
    fmt = lambda row: " | ".join(str(v).ljust(w) for v, w in zip(row, widths))
    lines.append(fmt(headers))
    lines.append("-+-".join("-" * w for w in widths))
    lines.extend(fmt(r) for r in rows)

    for name, m in all_metrics.items():
        if m is None:
            continue
        lines.append("")
        lines.append(f"Per-class breakdown [{name}]:")
        pc_headers = ["Class", "Support", "Precision", "Recall", "F1"]
        pc_rows = [[cls, str(cm["support"]), f"{cm['precision']:.3f}", f"{cm['recall']:.3f}", f"{cm['f1']:.3f}"]
                   for cls, cm in m["per_class"].items()]
        pc_widths = [max(len(str(row[i])) for row in ([pc_headers] + pc_rows)) for i in range(len(pc_headers))]
        pc_fmt = lambda row: "  " + " | ".join(str(v).ljust(w) for v, w in zip(row, pc_widths))
        lines.append(pc_fmt(pc_headers))
        lines.append("  " + "-+-".join("-" * w for w in pc_widths))
        lines.extend(pc_fmt(r) for r in pc_rows)
        lines.append(f"  Confusion matrix (rows=true, cols=pred, order={m['labels_present']}):")
        for row in m["confusion_matrix"]:
            lines.append(f"    {row}")

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

    # 1. Grouped bar: Accuracy / Macro Precision / Macro Recall / Macro F1 / AUC
    metric_keys = [
        ("accuracy", "Accuracy"), ("macro_precision", "Macro Precision"),
        ("macro_recall", "Macro Recall"), ("macro_f1", "Macro F1"), ("macro_auc_ovr", "Macro AUC (OvR)"),
    ]
    x = np.arange(len(metric_keys))
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, name in enumerate(names):
        m = valid_metrics[name]
        values = [m[k] if not (isinstance(m.get(k), float) and np.isnan(m.get(k, np.nan))) else 0 for k, _ in metric_keys]
        ax.bar(x + i * width, values, width, label=name)
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels([lbl for _, lbl in metric_keys])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Signal 1b (image) — all models compared")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=150)
    plt.close(fig)

    # 2. Percentage correctly classified, sorted, with correct/total labels
    sorted_names = sorted(names, key=lambda n: valid_metrics[n]["accuracy"], reverse=True)
    accs_pct = [valid_metrics[n]["accuracy"] * 100 for n in sorted_names]
    totals = [valid_metrics[n]["n_evaluated"] for n in sorted_names]
    corrects = [round(a / 100 * t) for a, t in zip(accs_pct, totals)]
    fig, ax = plt.subplots(figsize=(9, 6))
    bar_colors = plt.cm.RdYlGn(np.array(accs_pct) / 100)
    bars = ax.bar(sorted_names, accs_pct, color=bar_colors, edgecolor="black", linewidth=0.5)
    for bar, pct, c, t in zip(bars, accs_pct, corrects, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                 f"{pct:.2f}%\n({c}/{t})", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Correctly classified (%)")
    ax.set_title("Signal 1b (image) — percentage of correctly classified images")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=10, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "accuracy_percentage.png"), dpi=150)
    plt.close(fig)

    # 3. Inference time comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    times = [valid_metrics[n]["avg_inference_time"] for n in names]
    ax.bar(names, times, color="#2ab38e")
    ax.set_ylabel("Avg inference time per image (s)")
    ax.set_title("Signal 1b (image) — inference speed compared")
    plt.xticks(rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "inference_time_comparison.png"), dpi=150)
    plt.close(fig)

    # 4. Confusion matrices side by side
    fig, axes = plt.subplots(1, len(names), figsize=(5.5 * len(names), 5), squeeze=False)
    axes = axes[0]
    for ax, name in zip(axes, names):
        m = valid_metrics[name]
        labels = m["labels_present"]
        cm = np.array(m["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
        ax.set_title(name, fontsize=9)
        for i in range(len(labels)):
            for j in range(len(labels)):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=10)
    fig.suptitle("Confusion matrices compared")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrices_comparison.png"), dpi=150)
    plt.close(fig)

    # 5. Per-class F1 comparison, grouped by class, one bar per model
    fig, ax = plt.subplots(figsize=(9, 6))
    xk = np.arange(len(CLASS_NAMES))
    w = 0.8 / len(names)
    for i, name in enumerate(names):
        per_class = valid_metrics[name]["per_class"]
        values = [per_class.get(c, {}).get("f1", 0) for c in CLASS_NAMES]
        ax.bar(xk + i * w, values, w, label=name)
    ax.set_xticks(xk + w * (len(names) - 1) / 2)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1 compared — all models")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "per_class_f1_comparison.png"), dpi=150)
    plt.close(fig)

    print(f"\nCharts saved to: {output_dir}/")
    print("  - metrics_comparison.png")
    print("  - accuracy_percentage.png")
    print("  - inference_time_comparison.png")
    print("  - confusion_matrices_comparison.png")
    print("  - per_class_f1_comparison.png")


def main():
    model_preds = discover_models()
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
    print(metrics_csv[["rank", "model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_auc_ovr"]]
          .to_string(index=False))

    preds_csv = build_combined_predictions_csv(model_preds)
    preds_path = os.path.join(OUTPUT_DIR, "combined_predictions.csv")
    preds_csv.to_csv(preds_path, index=False)
    n_disagree = (~preds_csv["all_agree"]).sum()
    print(f"\nCombined per-image predictions saved to: {preds_path}")
    print(f"Images where models disagree: {n_disagree}/{len(preds_csv)}")

    write_combined_metrics_txt(metrics_csv, all_metrics, OUTPUT_DIR, n_disagree, len(preds_csv))

    save_combined_charts(all_metrics, OUTPUT_DIR)


if __name__ == "__main__":
    main()
