"""Shared utilities for the per-model speech-to-text (ASR) test scripts (test_whisper.py,
test_wav2vec2.py, test_distil_whisper.py).
"""

import argparse
import os
import sys
import time
import traceback

import pandas as pd

TESTING_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TESTS_ROOT = os.path.abspath(os.path.join(TESTING_ROOT, '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_ROOT, '..'))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(TESTS_ROOT, 'Test-Results', 'audiotranscription')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEFAULT_CSV = os.path.join(TESTS_ROOT, 'Datasets', 'librispeech-test-other', 'librispeech_transcripts.csv')
DEFAULT_AUDIO_ROOT = os.path.join(TESTS_ROOT, 'Datasets', 'librispeech-test-other', 'audio')


class BaseModel:
    """Every test_<model>.py model wrapper implements this interface."""
    name = "base"

    def load(self):
        raise NotImplementedError

    def transcribe(self, audio_path):
        """Returns a plain-text transcript string."""
        raise NotImplementedError


def load_dataset_csv(csv_path, audio_root=None, limit=None, seed=42):
    """Reads the signal3_transcripts_final.csv format:
    clip_id,audio_filename,transcript,label,manipulation_tactics only audio_filename.
    """
    df = pd.read_csv(csv_path)
    if "audio_filename" not in df.columns or "transcript" not in df.columns:
        raise ValueError(f"Expected columns 'audio_filename' and 'transcript' in {csv_path}, found: {list(df.columns)}")

    records = []
    for _, row in df.iterrows():
        rel_path = str(row["audio_filename"])
        full_path = os.path.join(audio_root, rel_path) if audio_root else rel_path
        records.append({"path": full_path, "reference": str(row["transcript"])})
    out = pd.DataFrame(records)

    if limit:
        out = out.sample(n=min(limit, len(out)), random_state=seed).reset_index(drop=True)

    return out


def evaluate_model(model: BaseModel, dataset: pd.DataFrame):
    print(f"\n{'='*70}\nLoading model: {model.name}\n{'='*70}")
    model.load()

    rows = []
    for i, row in dataset.iterrows():
        path = row["path"]
        reference = row["reference"]
        if not os.path.exists(path):
            rows.append({"path": path, "reference": reference, "hypothesis": "", "inference_time": float("nan"), "error": "file not found"})
            print(f"  [{i+1}/{len(dataset)}] SKIP (missing file): {path}")
            continue
        try:
            t0 = time.time()
            hypothesis = model.transcribe(path)
            dt = time.time() - t0
            rows.append({"path": path, "reference": reference, "hypothesis": hypothesis, "inference_time": dt, "error": ""})
            print(f"  [{i+1}/{len(dataset)}] {os.path.basename(path):30s} ({dt:.2f}s)")
            print(f"      ref: {reference[:80]}")
            print(f"      hyp: {hypothesis[:80]}")
        except Exception as e:
            rows.append({"path": path, "reference": reference, "hypothesis": "", "inference_time": float("nan"), "error": str(e)})
            print(f"  [{i+1}/{len(dataset)}] ERROR on {path}: {e}")

    return pd.DataFrame(rows)


def compute_metrics(df: pd.DataFrame):
    import jiwer

    valid = df[df["hypothesis"] != ""].dropna(subset=["hypothesis"])
    if valid.empty:
        return None

    # WER needs word-level normalization (case/punctuation stripped, then split into words).
    wer_transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ])
    cer_transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfChars(),
    ])

    references = valid["reference"].tolist()
    hypotheses = valid["hypothesis"].tolist()

    wer = jiwer.wer(references, hypotheses, reference_transform=wer_transform, hypothesis_transform=wer_transform)
    cer = jiwer.cer(references, hypotheses, reference_transform=cer_transform, hypothesis_transform=cer_transform)

    per_clip_wer = []
    for ref, hyp in zip(references, hypotheses):
        try:
            per_clip_wer.append(jiwer.wer(ref, hyp, reference_transform=wer_transform, hypothesis_transform=wer_transform))
        except Exception:
            per_clip_wer.append(float("nan"))
    valid = valid.copy()
    valid["wer"] = per_clip_wer

    return {
        "n_evaluated": len(valid),
        "n_skipped": len(df) - len(valid),
        "wer": wer,
        "cer": cer,
        "avg_inference_time": valid["inference_time"].mean(),
        "per_clip": valid,
    }


def print_results(model_name: str, metrics: dict):
    print("\n" + "=" * 70)
    print(f"RESULTS — Speech-to-text evaluation: {model_name}")
    print("=" * 70)
    if metrics is None:
        print("N/A - no successful transcriptions")
        return
    print(f"WER (Word Error Rate): {metrics['wer']:.3f}  (lower is better)")
    print(f"CER (Char Error Rate): {metrics['cer']:.3f}  (lower is better)")
    print(f"Avg inference:         {metrics['avg_inference_time']:.3f}s")
    print(f"N evaluated:           {metrics['n_evaluated']} (skipped: {metrics['n_skipped']})")


def write_metrics_txt(model_name: str, metrics: dict, output_dir: str, model_key: str,
                       csv_path: str, n_clips: int, device: str):
    lines = []
    lines.append("DeepGuard speech-to-text (ASR) model evaluation")
    lines.append("=" * 70)
    lines.append(f"Model tested : {model_key} ({model_name})")
    lines.append(f"Dataset CSV  : {csv_path}")
    lines.append(f"Clips        : {n_clips}")
    lines.append(f"Device       : {device}")
    lines.append("")
    if metrics is None:
        lines.append("N/A - no successful transcriptions")
    else:
        headers = ["WER", "CER", "Avg inference", "N (skipped)"]
        row = [f"{metrics['wer']:.3f}", f"{metrics['cer']:.3f}", f"{metrics['avg_inference_time']:.3f}s",
               f"{metrics['n_evaluated']} ({metrics['n_skipped']})"]
        widths = [max(len(headers[i]), len(row[i])) for i in range(len(headers))]
        fmt = lambda r: " | ".join(str(v).ljust(w) for v, w in zip(r, widths))
        lines.append(fmt(headers))
        lines.append("-+-".join("-" * w for w in widths))
        lines.append(fmt(row))
        lines.append("")
        lines.append("(lower WER/CER is better)")

    path = os.path.join(output_dir, "metrics.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Metrics text report saved to: {path}")


def save_graphs(model_name: str, metrics: dict, output_dir: str):
    if metrics is None:
        print("No successful results to plot.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(metrics["per_clip"]["wer"].clip(upper=2.0), bins=20, color="#2ab38e", edgecolor="black")
    ax.set_xlabel("Per-clip WER")
    ax.set_ylabel("Count")
    ax.set_title(f"{model_name} — per-clip WER distribution")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "wer_distribution.png"), dpi=150)
    plt.close(fig)

    print(f"\nGraph saved to: {output_dir}/wer_distribution.png")


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to the transcript-labelled CSV.")
    parser.add_argument("--audio-root", default=DEFAULT_AUDIO_ROOT, help="Directory to prepend to relative audio filenames in the CSV.")
    parser.add_argument("--limit", type=int, default=None, help="Cap total clips evaluated (random sample) for a quick run.")
    parser.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"], help="Force a device; default auto-detects.")
    parser.add_argument("--output-dir", default=None, help="Where to write graphs/CSVs/metrics.txt. Defaults to Test-Results/audiotranscription/<model_key>.")
    return parser


def run_evaluation(model_key: str, model_factory, args):
    if args.device:
        device = args.device
    else:
        import torch
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = args.output_dir or os.path.join(RESULTS_DIR, model_key)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    dataset = load_dataset_csv(args.csv, audio_root=args.audio_root, limit=args.limit)
    print(f"Loaded {len(dataset)} labelled clips")

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
        summary_row = {k: v for k, v in metrics.items() if k != "per_clip"}
        summary_row["model"] = model.name
        pd.DataFrame([summary_row]).to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
        print(f"Metrics summary saved to: {os.path.join(output_dir, 'metrics_summary.csv')}")

    write_metrics_txt(model.name, metrics, output_dir, model_key, args.csv, len(dataset), device)
    save_graphs(model.name, metrics, output_dir)
