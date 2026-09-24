"""Shared dataset loading, metrics and reporting for the video-detection benchmark scripts."""

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass

import numpy as np
import pandas as pd

# -------------------------------------------------------------------------- Make the main
# project importable.
TESTING_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_TESTING_ROOT = os.path.abspath(os.path.join(TESTING_ROOT, '..'))
TESTS_ROOT = os.path.abspath(os.path.join(MODEL_TESTING_ROOT, '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_ROOT, '..'))
sys.path.insert(0, PROJECT_ROOT)

from modules.frame_extractor import extract_frames  # noqa: E402

# Graphs/CSVs go in Tests/Test-Results/video-detection/ (Test-Results/ is split into video-
# detection/ and image-detection/ subfolders this common.py module is used only by the video
RESULTS_DIR = os.path.join(MODEL_TESTING_ROOT, 'Test-Results', 'video-detection')
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)


class BaseModel:
    """Every test_<model>.py model wrapper implements this interface."""
    name = "base"

    def load(self):
        raise NotImplementedError

    def score_video(self, video_path, num_frames=20):
        raise NotImplementedError


# -------------------------------------------------------------------------- CSV loading /
# column auto-detection.

PATH_COL_CANDIDATES = ["path", "filename", "file", "video", "video_path", "filepath", "file path"]
LABEL_COL_CANDIDATES = ["label", "class", "target", "y"]


def _normalize_col(c):
    # Collapses "File Path" / "file_path" / "File-Path" all to "filepath" so real-world CSV
    # headers (e.g. the FF++ Kaggle mirror's "File Path") match without requiring users to
    return c.strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def load_dataset_csv(csv_path, video_root=None, limit=None, seed=42):
    df = pd.read_csv(csv_path)
    cols_norm = {_normalize_col(c): c for c in df.columns}
    path_candidates_norm = {_normalize_col(c) for c in PATH_COL_CANDIDATES}
    label_candidates_norm = {_normalize_col(c) for c in LABEL_COL_CANDIDATES}

    path_col = next((cols_norm[c] for c in path_candidates_norm if c in cols_norm), None)
    label_col = next((cols_norm[c] for c in label_candidates_norm if c in cols_norm), None)

    if path_col is None or label_col is None:
        raise ValueError(
            f"Could not auto-detect path/label columns in {csv_path}. "
            f"Found columns: {list(df.columns)}. "
            f"Expected one of {PATH_COL_CANDIDATES} for the video path and "
            f"one of {LABEL_COL_CANDIDATES} for the label."
        )

    def normalize_label(v):
        if isinstance(v, (int, float)):
            return int(v) != 0
        s = str(v).strip().lower()
        return s in ("1", "fake", "true", "deepfake", "synthetic")

    records = []
    for _, row in df.iterrows():
        rel_path = str(row[path_col])
        full_path = os.path.join(video_root, rel_path) if video_root else rel_path
        records.append({
            "path": full_path,
            "is_fake": normalize_label(row[label_col]),
        })

    out = pd.DataFrame(records)

    if limit:
        # Stratified sample so both classes are represented in quick test runs
        real = out[~out["is_fake"]]
        fake = out[out["is_fake"]]
        n_each = max(1, limit // 2)
        real = real.sample(n=min(n_each, len(real)), random_state=seed)
        fake = fake.sample(n=min(n_each, len(fake)), random_state=seed)
        out = pd.concat([real, fake]).sample(frac=1, random_state=seed).reset_index(drop=True)

    return out


# -------------------------------------------------------------------------- Evaluation loop.

@dataclass
class VideoResult:
    path: str
    y_true: int
    y_score: float = float("nan")
    inference_time: float = float("nan")
    error: str = ""


def evaluate_model(model: BaseModel, dataset: pd.DataFrame, num_frames: int):
    print(f"\n{'='*70}\nLoading model: {model.name}\n{'='*70}")
    model.load()

    results = []
    for i, row in dataset.iterrows():
        path = row["path"]
        y_true = int(row["is_fake"])
        if not os.path.exists(path):
            results.append(VideoResult(path=path, y_true=y_true, error="file not found"))
            print(f"  [{i+1}/{len(dataset)}] SKIP (missing file): {path}")
            continue
        try:
            t0 = time.time()
            score = model.score_video(path, num_frames=num_frames)
            dt = time.time() - t0
            results.append(VideoResult(path=path, y_true=y_true, y_score=score, inference_time=dt))
            print(f"  [{i+1}/{len(dataset)}] {os.path.basename(path):40s} "
                  f"true={'FAKE' if y_true else 'REAL':5s} score={score:.3f} ({dt:.2f}s)")
        except Exception as e:
            results.append(VideoResult(path=path, y_true=y_true, error=str(e)))
            print(f"  [{i+1}/{len(dataset)}] ERROR on {path}: {e}")

    found = getattr(model, "faces_found", None)
    missed = getattr(model, "faces_missed", None)
    if found is not None and (found + missed) > 0:
        total = found + missed
        print(f"\nFace detection: {found}/{total} frames had a detected face "
              f"({found/total*100:.1f}%); {missed} fell back to the full frame.")

    return pd.DataFrame([r.__dict__ for r in results])


def compute_metrics(df: pd.DataFrame):
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, roc_auc_score, roc_curve, confusion_matrix,
                                  precision_recall_fscore_support)

    valid = df.dropna(subset=["y_score"])
    if valid.empty:
        return None

    y_true = valid["y_true"].astype(int).to_numpy()
    y_score = valid["y_score"].astype(float).to_numpy()
    y_pred = (y_score >= 0.5).astype(int)

    metrics = {
        "n_evaluated": len(valid),
        "n_skipped": len(df) - len(valid),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "avg_inference_time": valid["inference_time"].mean(),
    }

    # Per-class breakdown (Real=0, Fake/Deepfake=1) precision/recall/F1 computed treating each
    # class as the positive class in turn, plus per-class accuracy (how often that class alone
    class_labels = {0: "Real", 1: "Fake"}
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    per_class = {}
    for idx, cls_name in class_labels.items():
        mask = y_true == idx
        cls_accuracy = float((y_pred[mask] == y_true[mask]).mean()) if mask.any() else float("nan")
        per_class[cls_name] = {
            "support": int(supports[idx]),
            "accuracy": cls_accuracy,
            "precision": float(precisions[idx]),
            "recall": float(recalls[idx]),
            "f1": float(f1s[idx]),
        }
    metrics["per_class"] = per_class

    # AUC needs both classes present
    if len(set(y_true)) == 2:
        auc = roc_auc_score(y_true, y_score)
        # A model scoring below 0.5 AUC is very likely scoring the wrong way round (predicting
        # p(real) instead of p(fake), or vice versa) rather than genuinely anti-correlated
        metrics["auc"] = auc
        metrics["auc_note"] = "score orientation looks inverted (AUC<0.5) - check label mapping" if auc < 0.5 else ""
        fpr, tpr, _ = roc_curve(y_true, y_score)
        metrics["roc_curve"] = (fpr, tpr)
    else:
        metrics["auc"] = float("nan")
        metrics["auc_note"] = "only one class present in evaluated set - AUC undefined"
        metrics["roc_curve"] = None

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics


# -------------------------------------------------------------------------- Reporting:
# terminal table + metrics.txt + metrics_summary.csv + graphs.

def build_summary_rows(all_metrics: dict):
    """Flattens each model's metrics dict into one CSV-friendly row."""
    summary_rows = []
    for name, m in all_metrics.items():
        if m is None:
            continue
        row = {k: v for k, v in m.items() if k not in ("roc_curve", "per_class")}
        row["model"] = name
        row["confusion_matrix"] = str(row["confusion_matrix"])

        for cls_name, cls_m in (m.get("per_class") or {}).items():
            prefix = cls_name.lower()  # "real" / "fake"
            for metric_name, value in cls_m.items():
                row[f"{prefix}_{metric_name}"] = value

        summary_rows.append(row)
    return summary_rows


def build_results_table(all_metrics: dict):
    """Returns (header_line, sep_line, row_lines, note_lines)."""
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "AUC", "Avg Time (s)", "N eval", "N skipped"]
    rows = []
    for name, m in all_metrics.items():
        if m is None:
            rows.append([name, "N/A - no successful predictions", "", "", "", "", "", "", ""])
            continue
        rows.append([
            name,
            f"{m['accuracy']:.3f}",
            f"{m['precision']:.3f}",
            f"{m['recall']:.3f}",
            f"{m['f1']:.3f}",
            f"{m['auc']:.3f}" if not np.isnan(m['auc']) else "N/A",
            f"{m['avg_inference_time']:.2f}",
            str(m['n_evaluated']),
            str(m['n_skipped']),
        ])

    col_widths = [max(len(str(r[i])) for r in ([headers] + rows)) for i in range(len(headers))]

    def fmt_row(r):
        return " | ".join(str(v).ljust(w) for v, w in zip(r, col_widths))

    header_line = fmt_row(headers)
    sep_line = "-+-".join("-" * w for w in col_widths)
    row_lines = [fmt_row(r) for r in rows]

    note_lines = [f"NOTE [{name}]: {m['auc_note']}" for name, m in all_metrics.items() if m and m.get("auc_note")]

    return header_line, sep_line, row_lines, note_lines


def print_results_table(all_metrics: dict):
    header_line, sep_line, row_lines, note_lines = build_results_table(all_metrics)

    print("\n" + "=" * 70)
    print("RESULTS — Signal 1 (video) model evaluation")
    print("=" * 70)
    print(header_line)
    print(sep_line)
    for r in row_lines:
        print(r)

    for n in note_lines:
        print(f"\n  {n}")
    print()


def write_metrics_txt(all_metrics: dict, output_dir: str, model_key: str, csv_path: str,
                       n_videos: int, frames: int, device: str):
    """Plain-text report (metrics.txt) alongside metrics_summary.csv, so the
    numbers are readable without opening a CSV/spreadsheet."""
    header_line, sep_line, row_lines, note_lines = build_results_table(all_metrics)

    lines = []
    lines.append("DeepGuard Signal 1 (video) model evaluation")
    lines.append("=" * 70)
    lines.append(f"Model tested : {model_key}")
    lines.append(f"Dataset CSV  : {csv_path}")
    lines.append(f"Videos       : {n_videos}")
    lines.append(f"Frames/video : {frames}")
    lines.append(f"Device       : {device}")
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
            cm = m['confusion_matrix']
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
    print(f"Metrics text report saved to: {path}")


def save_graphs(all_metrics: dict, output_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid_metrics = {n: m for n, m in all_metrics.items() if m is not None}
    if not valid_metrics:
        print("No successful model results to plot.")
        return

    names = list(valid_metrics.keys())

    # 1. Bar chart: Accuracy / Precision / Recall / F1 / AUC
    metric_keys = ["accuracy", "precision", "recall", "f1", "auc"]
    x = np.arange(len(metric_keys))
    width = 0.8 / len(names)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(names):
        m = valid_metrics[name]
        values = [m[k] if not (isinstance(m[k], float) and np.isnan(m[k])) else 0 for k in metric_keys]
        ax.bar(x + i * width, values, width, label=name)

    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels([k.upper() for k in metric_keys])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Signal 1 model evaluation — classification metrics")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "metrics.png"), dpi=150)
    plt.close(fig)

    # 2. Inference time bar chart
    fig, ax = plt.subplots(figsize=(7, 5))
    times = [valid_metrics[n]["avg_inference_time"] for n in names]
    ax.bar(names, times, color="#2ab38e")
    ax.set_ylabel("Avg inference time per video (s)")
    ax.set_title("Signal 1 model evaluation — inference speed")
    plt.xticks(rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "inference_time.png"), dpi=150)
    plt.close(fig)

    # 3. ROC curve
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
        ax.set_title("ROC curve")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
    plt.close(fig)

    # 4.
    per_class_metric_keys = ["accuracy", "precision", "recall", "f1"]
    models_with_pc = [n for n in names if valid_metrics[n].get("per_class")]
    if models_with_pc:
        fig, axes = plt.subplots(1, len(models_with_pc), figsize=(6 * len(models_with_pc), 5), squeeze=False)
        axes = axes[0]
        class_colors = {"Real": "#2ab38e", "Fake": "#e74c3c"}
        for ax, name in zip(axes, models_with_pc):
            per_class = valid_metrics[name]["per_class"]
            x = np.arange(len(per_class_metric_keys))
            width = 0.8 / len(per_class)
            for i, (cls_name, cls_m) in enumerate(per_class.items()):
                values = [cls_m[k] for k in per_class_metric_keys]
                ax.bar(x + i * width, values, width, label=f"{cls_name} (n={cls_m['support']})",
                       color=class_colors.get(cls_name))
            ax.set_xticks(x + width * (len(per_class) - 1) / 2)
            ax.set_xticklabels([k.upper() for k in per_class_metric_keys])
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Score")
            ax.set_title(name, fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle("Per-class breakdown — Real vs Fake")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "per_class.png"), dpi=150)
        plt.close(fig)

    print(f"\nGraphs saved to: {output_dir}/")
    print("  - metrics.png")
    print("  - inference_time.png")
    print("  - roc_curve.png (only if the evaluated set had both classes)")
    print("  - per_class.png (Real vs Fake breakdown, if per-class data available)")


# -------------------------------------------------------------------------- Shared CLI +
# orchestration, called by each test_<model>.py's main().

def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--csv", required=True, help="Path to the labels CSV (e.g. FF++ c23 metadata).")
    parser.add_argument("--video-root", default=None, help="Directory to prepend to relative video paths in the CSV.")
    parser.add_argument("--frames", type=int, default=20, help="Number of frames to sample per video.")
    parser.add_argument("--limit", type=int, default=None, help="Cap total videos evaluated (stratified real/fake sample) for a quick run.")
    parser.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"], help="Force a device; default auto-detects.")
    parser.add_argument("--output-dir", default=None, help="Where to write graphs/CSVs/metrics.txt. Defaults to Test-Results/<model_key>.")
    return parser


def run_evaluation(model_key: str, model_factory, args):
    """model_key: short name used for the output folder / predictions CSV
    (e.g. "naman"). model_factory: callable(device) -> BaseModel instance."""
    if args.device:
        device = args.device
    else:
        import torch
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = args.output_dir or os.path.join(RESULTS_DIR, model_key)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    dataset = load_dataset_csv(args.csv, video_root=args.video_root, limit=args.limit)
    print(f"Loaded {len(dataset)} labelled videos "
          f"({int(dataset['is_fake'].sum())} fake / {int((~dataset['is_fake']).sum())} real)")

    all_metrics = {}
    model = model_factory(device=device)
    try:
        per_video_df = evaluate_model(model, dataset, num_frames=args.frames)
        per_video_csv = os.path.join(output_dir, f"predictions_{model_key}.csv")
        per_video_df.to_csv(per_video_csv, index=False)
        metrics = compute_metrics(per_video_df)
        all_metrics[model.name] = metrics
    except RuntimeError as e:
        # Expected, actionable failures (e.g. device incompatibility) we
        # raise ourselves in a model's load() - show just the message.
        print(f"\nSKIPPING {model.name}: {e}")
        all_metrics[model.name] = None
    except Exception:
        print(f"\nFATAL error while evaluating {model.name}, skipping it:")
        traceback.print_exc()
        all_metrics[model.name] = None

    print_results_table(all_metrics)

    summary_rows = build_summary_rows(all_metrics)
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
        print(f"Metrics summary saved to: {os.path.join(output_dir, 'metrics_summary.csv')}")

    write_metrics_txt(all_metrics, output_dir, model_key, args.csv, len(dataset), args.frames, device)

    save_graphs(all_metrics, output_dir)
