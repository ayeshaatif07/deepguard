"""Shared utilities for the per-model Signal 3 (voice manipulation scoring) test scripts."""

import argparse
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

TESTING_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TESTS_ROOT = os.path.abspath(os.path.join(TESTING_ROOT, '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_ROOT, '..'))
sys.path.insert(0, PROJECT_ROOT)

CLASS_NAMES = ["non-manipulative", "manipulative"]
RESULTS_DIR = os.path.join(TESTS_ROOT, 'Test-Results', 'voice-manipulation')
DEFAULT_CSV = os.path.join(TESTS_ROOT, 'Datasets', 'voice-manipulation', 'signal3_transcripts_final.csv')
DEFAULT_AUDIO_DIR = os.path.join(TESTS_ROOT, 'Datasets', 'voice-manipulation', 'audio')
os.makedirs(RESULTS_DIR, exist_ok=True)


class BaseModel:
    """Every test_<model>.py model wrapper implements this interface."""
    name = "base"

    def load(self):
        raise NotImplementedError

    def score_text(self, text):
        """Returns {'manipulative': prob, 'non-manipulative': prob}, probabilities summing to
        ~1.0.
        """
        raise NotImplementedError


def load_dataset_csv(csv_path=DEFAULT_CSV, limit=None, seed=42):
    df = pd.read_csv(csv_path)
    required = {"clip_id", "audio_filename", "transcript", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected columns {required} in {csv_path}, found: {list(df.columns)}")

    if limit:
        per_class = max(1, limit // df["label"].nunique())
        parts = []
        for cls in sorted(df["label"].unique()):
            cls_df = df[df["label"] == cls]
            parts.append(cls_df.sample(n=min(per_class, len(cls_df)), random_state=seed))
        df = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

    return df


def evaluate_model(model: BaseModel, dataset: pd.DataFrame):
    print(f"\n{'='*70}\nLoading model: {model.name}\n{'='*70}")
    model.load()

    rows = []
    for i, row in dataset.iterrows():
        text = row["transcript"]
        y_true = row["label"]
        try:
            t0 = time.time()
            probs = model.score_text(text)
            dt = time.time() - t0
            y_pred = max(probs, key=probs.get)
            record = {
                "clip_id": row["clip_id"], "text": text, "y_true": y_true,
                "y_pred": y_pred, "inference_time": dt, "error": "",
            }
            for cls in CLASS_NAMES:
                record[f"prob_{cls}"] = probs.get(cls, float("nan"))
            rows.append(record)
            correct = "OK" if y_pred == y_true else "WRONG"
            short = (text[:40] + "...") if len(text) > 43 else text
            print(f"  [{i+1}/{len(dataset)}] {short:43s} true={y_true:17s} pred={y_pred:17s} [{correct}] ({dt:.3f}s)")
        except Exception as e:
            rows.append({"clip_id": row["clip_id"], "text": text, "y_true": y_true,
                         "y_pred": "", "inference_time": float("nan"), "error": str(e)})
            print(f"  [{i+1}/{len(dataset)}] ERROR on {row['clip_id']}: {e}")

    return pd.DataFrame(rows)


def compute_metrics(df: pd.DataFrame):
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                  confusion_matrix, roc_auc_score)

    valid = df[df["y_pred"] != ""].dropna(subset=["y_pred"])
    if valid.empty:
        return None

    y_true = valid["y_true"].to_numpy()
    y_pred = valid["y_pred"].to_numpy()
    labels_present = [c for c in CLASS_NAMES if c in set(y_true) | set(y_pred)]

    metrics = {
        "n_evaluated": len(valid),
        "n_skipped": len(df) - len(valid),
        "accuracy": accuracy_score(y_true, y_pred),
        "avg_inference_time": valid["inference_time"].mean(),
    }

    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_present, zero_division=0
    )
    metrics["macro_precision"] = float(np.mean(precisions))
    metrics["macro_recall"] = float(np.mean(recalls))
    metrics["macro_f1"] = float(np.mean(f1s))

    per_class = {}
    for i, cls in enumerate(labels_present):
        per_class[cls] = {
            "support": int(supports[i]), "precision": float(precisions[i]),
            "recall": float(recalls[i]), "f1": float(f1s[i]),
        }
    metrics["per_class"] = per_class
    metrics["labels_present"] = labels_present
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels_present).tolist()

    if "prob_manipulative" in valid.columns and len(labels_present) > 1:
        try:
            metrics["auc"] = roc_auc_score((y_true == "manipulative").astype(int), valid["prob_manipulative"])
        except Exception:
            metrics["auc"] = float("nan")
    else:
        metrics["auc"] = float("nan")

    return metrics


def print_results(model_name, metrics):
    print("\n" + "=" * 70)
    print(f"RESULTS — Signal 3 (voice manipulation) model evaluation: {model_name}")
    print("=" * 70)
    if metrics is None:
        print("N/A - no successful predictions")
        return
    print(f"Accuracy:        {metrics['accuracy']:.3f}")
    print(f"Macro Precision: {metrics['macro_precision']:.3f}")
    print(f"Macro Recall:    {metrics['macro_recall']:.3f}")
    print(f"Macro F1:        {metrics['macro_f1']:.3f}")
    auc = metrics.get("auc", float("nan"))
    print(f"AUC:             {auc:.3f}" if not np.isnan(auc) else "AUC:             N/A")
    print(f"Avg inference:   {metrics['avg_inference_time']:.3f}s")
    print(f"N evaluated:     {metrics['n_evaluated']}  (skipped: {metrics['n_skipped']})")

    print("\nPer-class breakdown:")
    headers = ["Class", "Support", "Precision", "Recall", "F1"]
    rows = [[cls, str(m["support"]), f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1']:.3f}"]
            for cls, m in metrics["per_class"].items()]
    widths = [max(len(str(r[i])) for r in ([headers] + rows)) for i in range(len(headers))]
    fmt = lambda r: "  " + " | ".join(str(v).ljust(w) for v, w in zip(r, widths))
    print(fmt(headers))
    print("  " + "-+-".join("-" * w for w in widths))
    for r in rows:
        print(fmt(r))

    print(f"\nConfusion matrix (rows=true, cols=pred, order={metrics['labels_present']}):")
    for row in metrics["confusion_matrix"]:
        print("  " + str(row))
    print()


def write_metrics_txt(model_name, metrics, output_dir, model_key, csv_path, n_clips, device):
    lines = [
        "DeepGuard Signal 3 (voice manipulation) model evaluation",
        "=" * 70,
        f"Model tested : {model_key} ({model_name})",
        f"Dataset CSV  : {csv_path}",
        f"Transcripts  : {n_clips}",
        f"Device       : {device}",
        "",
    ]
    if metrics is None:
        lines.append("N/A - no successful predictions")
    else:
        def table(headers, rows):
            widths = [max(len(str(r[i])) for r in ([headers] + rows)) for i in range(len(headers))]
            fmt = lambda r: " | ".join(str(v).ljust(w) for v, w in zip(r, widths))
            out = [fmt(headers), "-+-".join("-" * w for w in widths)]
            out.extend(fmt(r) for r in rows)
            return out

        auc = metrics.get("auc", float("nan"))
        auc_str = f"{auc:.3f}" if not np.isnan(auc) else "N/A"
        lines.append("Summary:")
        lines.extend(table(
            ["Accuracy", "Macro P", "Macro R", "Macro F1", "AUC", "Avg inference", "N (skipped)"],
            [[f"{metrics['accuracy']:.3f}", f"{metrics['macro_precision']:.3f}",
              f"{metrics['macro_recall']:.3f}", f"{metrics['macro_f1']:.3f}", auc_str,
              f"{metrics['avg_inference_time']:.3f}s",
              f"{metrics['n_evaluated']} ({metrics['n_skipped']})"]]
        ))
        lines.append("")
        lines.append("Per-class breakdown:")
        lines.extend(table(
            ["Class", "Support", "Precision", "Recall", "F1"],
            [[cls, str(m["support"]), f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1']:.3f}"]
             for cls, m in metrics["per_class"].items()]
        ))
        lines.append("")
        labels = metrics["labels_present"]
        lines.append("Confusion matrix (rows=true, cols=pred):")
        lines.extend(table(
            ["True \\ Pred"] + labels,
            [[labels[i]] + [str(v) for v in row] for i, row in enumerate(metrics["confusion_matrix"])]
        ))

    path = os.path.join(output_dir, "metrics.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Metrics text report saved to: {path}")


def save_graphs(model_name, metrics, output_dir):
    if metrics is None:
        print("No successful results to plot.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = metrics["labels_present"]

    fig, ax = plt.subplots(figsize=(7, 5))
    metric_keys = ["precision", "recall", "f1"]
    x = np.arange(len(labels))
    width = 0.25
    for i, mk in enumerate(metric_keys):
        values = [metrics["per_class"][c][mk] for c in labels]
        ax.bar(x + i * width, values, width, label=mk.capitalize())
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"{model_name} — per-class precision/recall/F1")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "per_class.png"), dpi=150)
    plt.close(fig)

    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name} — confusion matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    print(f"\nGraphs saved to: {output_dir}/")
    print("  - per_class.png")
    print("  - confusion_matrix.png")


def build_arg_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to signal3_transcripts_final.csv.")
    parser.add_argument("--limit", type=int, default=None, help="Cap total clips evaluated (stratified sample).")
    parser.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"], help="Force a device; default auto-detects.")
    parser.add_argument("--output-dir", default=None, help="Where to write graphs/CSVs/metrics.txt. Defaults to Test-Results/voice-manipulation/<model_key>.")
    return parser


def run_evaluation(model_key, model_factory, args):
    if args.device:
        device = args.device
    else:
        import torch
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = args.output_dir or os.path.join(RESULTS_DIR, model_key)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    dataset = load_dataset_csv(args.csv, limit=args.limit)
    print(f"Loaded {len(dataset)} labelled transcripts: {dataset['label'].value_counts().to_dict()}")

    model = model_factory(device=device)
    metrics = None
    try:
        per_clip_df = evaluate_model(model, dataset)
        per_clip_df.to_csv(os.path.join(output_dir, f"predictions_{model_key}.csv"), index=False)
        metrics = compute_metrics(per_clip_df)
    except Exception:
        print(f"\nFATAL error while evaluating {model.name}:")
        traceback.print_exc()

    print_results(model.name, metrics)

    if metrics is not None:
        summary_row = {k: v for k, v in metrics.items() if k not in ("per_class", "confusion_matrix", "labels_present")}
        summary_row["model"] = model.name
        for cls, m in metrics["per_class"].items():
            for mk, mv in m.items():
                summary_row[f"{cls.replace('-', '_')}_{mk}"] = mv
        pd.DataFrame([summary_row]).to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
        print(f"Metrics summary saved to: {os.path.join(output_dir, 'metrics_summary.csv')}")

    write_metrics_txt(model.name, metrics, output_dir, model_key, args.csv, len(dataset), device)
    save_graphs(model.name, metrics, output_dir)
