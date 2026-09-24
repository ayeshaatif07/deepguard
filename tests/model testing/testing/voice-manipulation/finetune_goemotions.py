"""Fine-tunes SamLowe/roberta-base-go_emotions into a direct binary manipulative/non-
manipulative transcript classifier.
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
import common
from common import RESULTS_DIR, DEFAULT_CSV

TESTING_DIR = os.path.dirname(__file__)
WEIGHTS_DIR = os.path.join(TESTING_DIR, "weights", "goemotions_finetuned")

BASE_MODEL_ID = "SamLowe/roberta-base-go_emotions"
SEED = 42
N_TRAIN_PER_CLASS = 20
CLASS_NAMES = ["non-manipulative", "manipulative"]

# The existing production derived-score heuristic (voice_manipulation_scorer.py)
# - re-scored here on the held-out set for a direct, apples-to-apples comparison.
MANIPULATION_EMOTIONS = {"fear", "nervousness", "anger", "annoyance", "disapproval", "disgust"}
CURRENT_PRODUCTION_THRESHOLD = 0.02


def build_train_test_split():
    """Returns (train_items, test_items), each a list of (transcript, label)
    with label in {0: non-manipulative, 1: manipulative}."""
    df = pd.read_csv(DEFAULT_CSV)
    rng = random.Random(SEED)

    train_items, test_items = [], []
    for label_name, label_id in [("manipulative", 1), ("non-manipulative", 0)]:
        rows = df[df["label"] == label_name]["transcript"].tolist()
        assert len(rows) == 30, f"expected 30 {label_name} rows, found {len(rows)}"
        shuffled = rows[:]
        rng.shuffle(shuffled)
        train_rows = shuffled[:N_TRAIN_PER_CLASS]
        test_rows = shuffled[N_TRAIN_PER_CLASS:]
        train_items.extend((t, label_id) for t in train_rows)
        test_items.extend((t, label_id) for t in test_rows)

    rng.shuffle(train_items)
    rng.shuffle(test_items)
    return train_items, test_items


def stratified_internal_split(items, val_frac=0.15, seed=SEED):
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for text, label in items:
        by_label[label].append((text, label))
    train, val = [], []
    for label, group in by_label.items():
        rng.shuffle(group)
        n_val = max(1, int(len(group) * val_frac))
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class TranscriptDataset(Dataset):
    def __init__(self, items, tokenizer, max_length=256):
        self.texts = [t for t, _ in items]
        self.labels = [l for _, l in items]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_length,
                              padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}, self.labels[idx]


def collate(batch):
    encs, labels = zip(*batch)
    keys = encs[0].keys()
    stacked = {k: torch.stack([e[k] for e in encs]) for k in keys}
    return stacked, torch.tensor(labels, dtype=torch.long)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs_manip = [], [], []
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            logits = model(**inputs).logits
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * len(labels)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs_manip.extend(probs[:, 1].cpu().tolist())
    avg_loss = total_loss / len(all_labels)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return avg_loss, acc, all_preds, all_labels, all_probs_manip


def score_with_current_production_heuristic(texts, device):
    """Re-implements the exact production derived-score logic from
    voice_manipulation_scorer.py, using the ORIGINAL (non-fine-tuned) base checkpoint.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_ID)
    model.to(device)
    model.eval()
    id2label = model.config.id2label

    scores = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            emotion_probs = {id2label[i]: float(p) for i, p in enumerate(probs)}
            raw = sum(emotion_probs.get(e, 0.0) for e in MANIPULATION_EMOTIONS)
            scores.append(min(1.0, raw))
    return scores


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SamLowe/roberta-base-go_emotions into a direct binary manipulation classifier.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print(f"Loading base checkpoint: {BASE_MODEL_ID} (replacing its 28-class head with a fresh binary head)")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_ID, num_labels=2, problem_type="single_label_classification",
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    train_pool, test_pool = build_train_test_split()
    print(f"Train pool: {len(train_pool)} (20 manipulative / 20 non-manipulative)")
    print(f"Held-out test pool: {len(test_pool)} (10 manipulative / 10 non-manipulative) - NEVER used in training")

    train_items, val_items = stratified_internal_split(train_pool)
    print(f"Internal split of the training pool: train={len(train_items)}  val={len(val_items)} (for checkpoint selection only)")

    train_ds = TranscriptDataset(train_items, tokenizer)
    val_ds = TranscriptDataset(val_items, tokenizer)
    test_ds = TranscriptDataset(test_pool, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss_sum, n_seen = 0.0, 0
        for inputs, labels in train_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(**inputs).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(labels)
            n_seen += len(labels)
        train_loss = train_loss_sum / n_seen

        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, device)
        dt = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  ({dt:.1f}s)")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nBest internal validation accuracy: {best_val_acc:.3f} - restoring that checkpoint before final held-out test.")
    model.load_state_dict(best_state)
    model.to(device)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    model.save_pretrained(WEIGHTS_DIR)
    tokenizer.save_pretrained(WEIGHTS_DIR)
    print(f"Fine-tuned model saved to: {WEIGHTS_DIR}")

    # ---- Final evaluation on the genuinely held-out 20-transcript test pool ----
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                  confusion_matrix, roc_auc_score)

    _, _, preds, labels, probs_manip = evaluate(model, test_loader, device)
    id_to_class = {0: "non-manipulative", 1: "manipulative"}
    y_true = [id_to_class[l] for l in labels]
    y_pred = [id_to_class[p] for p in preds]
    test_texts = test_ds.texts

    def compute_metrics(y_true, y_pred, probs_pos):
        m = {"accuracy": accuracy_score(y_true, y_pred)}
        try:
            m["auc"] = roc_auc_score([1 if l == "manipulative" else 0 for l in y_true], probs_pos)
        except ValueError:
            m["auc"] = float("nan")
        precisions, recalls, f1s, supports = precision_recall_fscore_support(y_true, y_pred, labels=CLASS_NAMES, zero_division=0)
        m["macro_precision"] = float(np.mean(precisions))
        m["macro_recall"] = float(np.mean(recalls))
        m["macro_f1"] = float(np.mean(f1s))
        per_class = {cls: {"support": int(supports[i]), "precision": float(precisions[i]),
                            "recall": float(recalls[i]), "f1": float(f1s[i])}
                     for i, cls in enumerate(CLASS_NAMES)}
        cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES).tolist()
        return m, per_class, cm

    ft_metrics, ft_per_class, ft_cm = compute_metrics(y_true, y_pred, probs_manip)

    # ---- Same held-out transcripts, scored by the CURRENT production
    # derived-score heuristic (original, non-fine-tuned base checkpoint) ----
    print("\nRe-scoring the same held-out transcripts with the current production heuristic...")
    heuristic_scores = score_with_current_production_heuristic(test_texts, device)
    heuristic_pred_ids = [1 if s >= CURRENT_PRODUCTION_THRESHOLD else 0 for s in heuristic_scores]
    heuristic_y_pred = [id_to_class[p] for p in heuristic_pred_ids]
    heur_metrics, heur_per_class, heur_cm = compute_metrics(y_true, heuristic_y_pred, heuristic_scores)

    output_dir = os.path.join(RESULTS_DIR, "goemotions_finetuned")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("RESULTS - fine-tuned vs current production heuristic (same 20 held-out transcripts)")
    print("=" * 70)
    print(f"{'Metric':<16}{'Fine-tuned':>14}{'Current (heuristic)':>22}")
    for k in ["accuracy", "auc", "macro_precision", "macro_recall", "macro_f1"]:
        print(f"{k:<16}{ft_metrics[k]:>14.3f}{heur_metrics[k]:>22.3f}")

    pd.DataFrame({
        "transcript": test_texts, "y_true": y_true,
        "y_pred_finetuned": y_pred, "prob_manipulative_finetuned": probs_manip,
        "y_pred_heuristic": heuristic_y_pred, "score_heuristic": heuristic_scores,
    }).to_csv(os.path.join(output_dir, "predictions_heldout_test.csv"), index=False)
    pd.DataFrame([{**ft_metrics, "model": "goemotions_finetuned"},
                  {**heur_metrics, "model": "goemotions_current_heuristic"}]) \
        .to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
    pd.DataFrame(history).to_csv(os.path.join(output_dir, "training_history.csv"), index=False)

    def _table(headers, rows):
        widths = [max(len(str(r[i])) for r in ([headers] + rows)) for i in range(len(headers))]
        fmt = lambda r: " | ".join(str(v).ljust(w) for v, w in zip(r, widths))
        out = [fmt(headers), "-+-".join("-" * w for w in widths)]
        out.extend(fmt(r) for r in rows)
        return out

    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write("DeepGuard Signal 3 (voice manipulation) - fine-tuned model evaluation\n")
        f.write("=" * 70 + "\n")
        f.write(f"Base checkpoint: {BASE_MODEL_ID} (28-class head replaced with a fresh binary head)\n")
        f.write("Split: GENUINE disjoint train/test - 20 manipulative + 20 non-manipulative for\n")
        f.write("training (40 total, further split 85/15 internally for checkpoint selection);\n")
        f.write("10 manipulative + 10 non-manipulative for held-out testing (20 total),\n")
        f.write("never touched until this final evaluation.\n")
        f.write(f"Epochs: {args.epochs}  LR: {args.lr}  Batch size: {args.batch_size}\n\n")
        f.write("COMPARISON on the SAME 20 held-out transcripts (fine-tuned vs current production heuristic):\n")
        f.write("\n".join(_table(
            ["Model", "Accuracy", "Macro P", "Macro R", "Macro F1", "AUC"],
            [["Fine-tuned (this run)", f"{ft_metrics['accuracy']:.3f}", f"{ft_metrics['macro_precision']:.3f}",
              f"{ft_metrics['macro_recall']:.3f}", f"{ft_metrics['macro_f1']:.3f}", f"{ft_metrics['auc']:.3f}"],
             ["Current production heuristic", f"{heur_metrics['accuracy']:.3f}", f"{heur_metrics['macro_precision']:.3f}",
              f"{heur_metrics['macro_recall']:.3f}", f"{heur_metrics['macro_f1']:.3f}", f"{heur_metrics['auc']:.3f}"]]
        )) + "\n\n")
        f.write("Fine-tuned model per-class breakdown:\n")
        f.write("\n".join(_table(
            ["Class", "Support", "Precision", "Recall", "F1"],
            [[cls, str(m["support"]), f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1']:.3f}"]
             for cls, m in ft_per_class.items()]
        )) + "\n\n")
        f.write("Fine-tuned model confusion matrix (rows=true, cols=pred):\n")
        f.write("\n".join(_table(
            ["True \\ Pred"] + CLASS_NAMES,
            [[CLASS_NAMES[i]] + [str(v) for v in row] for i, row in enumerate(ft_cm)]
        )) + "\n\n")
        f.write("Current production heuristic confusion matrix, same held-out set (rows=true, cols=pred):\n")
        f.write("\n".join(_table(
            ["True \\ Pred"] + CLASS_NAMES,
            [[CLASS_NAMES[i]] + [str(v) for v in row] for i, row in enumerate(heur_cm)]
        )) + "\n\n")
        f.write("This is a direct, apples-to-apples comparison: identical held-out transcripts,\n")
        f.write("identical base checkpoint, only the scoring approach differs (derived emotion-sum\n")
        f.write("heuristic vs a fine-tuned binary head trained directly on this project's own\n")
        f.write("labelled transcripts).\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist_df = pd.DataFrame(history)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(hist_df["epoch"], hist_df["train_loss"], label="train_loss")
    ax1.plot(hist_df["epoch"], hist_df["val_loss"], label="val_loss (internal)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(alpha=0.3)
    ax1.set_title("Loss")
    ax2.plot(hist_df["epoch"], hist_df["val_acc"], color="green", label="val_accuracy (internal)")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_ylim(0, 1.05); ax2.legend(); ax2.grid(alpha=0.3)
    ax2.set_title("Internal validation accuracy (checkpoint selection only)")
    fig.suptitle("Fine-tuning GoEmotions -> binary manipulation classifier")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curve.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, cm, title in [(axes[0], ft_cm, "Fine-tuned"), (axes[1], heur_cm, "Current heuristic")]:
        cm_arr = np.array(cm)
        im = ax.imshow(cm_arr, cmap="Blues")
        ax.set_xticks(range(len(CLASS_NAMES))); ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=20); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{title} (held-out, n=20)")
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                color = "white" if cm_arr[i, j] > cm_arr.max() / 2 else "black"
                ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center", color=color, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
