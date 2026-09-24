"""Fine-tunes MattyB95/AST-ASVspoof2019-Synthetic-Voice-Detection on project data to close its
gTTS blind spot.
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
from common import CLASS_NAMES, RESULTS_DIR

TESTING_DIR = os.path.dirname(__file__)
WEIGHTS_DIR = os.path.join(TESTING_DIR, "weights", "ast_asvspoof_finetuned")
TESTS_ROOT = os.path.abspath(os.path.join(TESTING_DIR, "..", ".."))

REAL_DIR = os.path.join(TESTS_ROOT, "Datasets", "voice-clone", "audio", "real")
ORIGINAL_FAKE_DIR = os.path.join(TESTS_ROOT, "Datasets", "voice-clone", "audio", "fake")
GTTS_DIR = os.path.join(TESTS_ROOT, "Datasets", "voice-manipulation", "audio")

BASE_MODEL_ID = "MattyB95/AST-ASVspoof2019-Synthetic-Voice-Detection"
SEED = 42
N_REAL_TRAIN = 50
N_ORIGINAL_FAKE_TRAIN = 50
N_GTTS_TRAIN = 30


def build_train_test_split():
    """Returns (train_items, test_items), each a list of (path, label, category) category is
    'real', 'original_fake', or 'gtts'.
    """
    rng = random.Random(SEED)

    real_files = sorted(f for f in os.listdir(REAL_DIR) if f.lower().endswith((".flac", ".wav", ".mp3")))
    fake_files = sorted(f for f in os.listdir(ORIGINAL_FAKE_DIR) if f.lower().endswith((".flac", ".wav", ".mp3")))
    gtts_files = sorted(f for f in os.listdir(GTTS_DIR) if f.lower().endswith(".mp3"))

    real_shuffled = real_files[:]
    fake_shuffled = fake_files[:]
    gtts_shuffled = gtts_files[:]
    rng.shuffle(real_shuffled)
    rng.shuffle(fake_shuffled)
    rng.shuffle(gtts_shuffled)

    real_train, real_test = real_shuffled[:N_REAL_TRAIN], real_shuffled[N_REAL_TRAIN:]
    fake_train, fake_test = fake_shuffled[:N_ORIGINAL_FAKE_TRAIN], fake_shuffled[N_ORIGINAL_FAKE_TRAIN:]
    gtts_train, gtts_test = gtts_shuffled[:N_GTTS_TRAIN], gtts_shuffled[N_GTTS_TRAIN:]

    train_items = (
        [(os.path.join(REAL_DIR, f), "real", "real") for f in real_train]
        + [(os.path.join(ORIGINAL_FAKE_DIR, f), "fake", "original_fake") for f in fake_train]
        + [(os.path.join(GTTS_DIR, f), "fake", "gtts") for f in gtts_train]
    )
    test_items = (
        [(os.path.join(REAL_DIR, f), "real", "real") for f in real_test]
        + [(os.path.join(ORIGINAL_FAKE_DIR, f), "fake", "original_fake") for f in fake_test]
        + [(os.path.join(GTTS_DIR, f), "fake", "gtts") for f in gtts_test]
    )
    rng.shuffle(train_items)
    rng.shuffle(test_items)
    return train_items, test_items


def stratified_internal_split(items, val_frac=0.15, seed=SEED):
    """Splits the TRAINING pool only into an internal train/val split for
    checkpoint selection - never touches the held-out test pool."""
    rng = random.Random(seed)
    by_label = {"real": [], "fake": []}
    for path, label, category in items:
        by_label[label].append((path, label, category))
    train, val = [], []
    for label, group in by_label.items():
        rng.shuffle(group)
        n_val = max(1, int(len(group) * val_frac))
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class AudioClipDataset(Dataset):
    def __init__(self, items, extractor):
        import soundfile as sf
        import librosa

        self.extractor = extractor
        self.waveforms = []
        self.labels = []
        self.categories = []
        self.paths = []
        for path, label, category in items:
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            self.waveforms.append(audio)
            self.labels.append(0 if label == "real" else 1)  # 0=Bonafide, 1=Spoof
            self.categories.append(category)
            self.paths.append(path)

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        features = self.extractor(self.waveforms[idx], sampling_rate=16000, return_tensors="pt")
        return features["input_values"][0], self.labels[idx]


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs_fake = [], [], []
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(input_values=inputs).logits
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * len(labels)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs_fake.extend(probs[:, 1].cpu().tolist())
    avg_loss = total_loss / len(all_labels)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return avg_loss, acc, all_preds, all_labels, all_probs_fake


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MattyB95/AST-ASVspoof2019 with a genuine disjoint train/test split.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CPU proved faster than MPS for this model - see v1's docstring/history
    print(f"Using device: {device}")

    from transformers import AutoFeatureExtractor, ASTForAudioClassification

    print(f"Loading base checkpoint: {BASE_MODEL_ID}")
    extractor = AutoFeatureExtractor.from_pretrained(BASE_MODEL_ID)
    model = ASTForAudioClassification.from_pretrained(BASE_MODEL_ID)
    model.to(device)

    train_pool, test_pool = build_train_test_split()
    print(f"Train pool: {len(train_pool)} (50 real / 50 original-fake / 30 gTTS)")
    print(f"Held-out test pool: {len(test_pool)} (50 real / 50 original-fake / 30 gTTS) - NEVER used in training")

    train_items, val_items = stratified_internal_split(train_pool)
    print(f"Internal split of the training pool: train={len(train_items)}  val={len(val_items)} (for checkpoint selection only)")

    print("Decoding + resampling audio (one-time)...")
    train_ds = AudioClipDataset(train_items, extractor)
    val_ds = AudioClipDataset(val_items, extractor)
    test_ds = AudioClipDataset(test_pool, extractor)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

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
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(input_values=inputs).logits
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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nBest internal validation accuracy: {best_val_acc:.3f} - restoring that checkpoint before final held-out test.")
    model.load_state_dict(best_state)
    model.to(device)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    model.save_pretrained(WEIGHTS_DIR)
    extractor.save_pretrained(WEIGHTS_DIR)
    print(f"Fine-tuned model saved to: {WEIGHTS_DIR}")

    # ---- Final evaluation on the genuinely held-out 130-clip test pool ----
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                  confusion_matrix, roc_auc_score)

    _, _, preds, labels, probs_fake = evaluate(model, test_loader, device)
    id_to_class = {0: "real", 1: "fake"}
    y_true = [id_to_class[l] for l in labels]
    y_pred = [id_to_class[p] for p in preds]
    categories = test_ds.categories
    paths = test_ds.paths

    metrics = {
        "n_evaluated": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score([1 if l == "fake" else 0 for l in y_true], probs_fake),
    }
    precisions, recalls, f1s, supports = precision_recall_fscore_support(y_true, y_pred, labels=CLASS_NAMES, zero_division=0)
    metrics["macro_precision"] = float(np.mean(precisions))
    metrics["macro_recall"] = float(np.mean(recalls))
    metrics["macro_f1"] = float(np.mean(f1s))
    per_class = {cls: {"support": int(supports[i]), "precision": float(precisions[i]),
                        "recall": float(recalls[i]), "f1": float(f1s[i])}
                 for i, cls in enumerate(CLASS_NAMES)}
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES).tolist()

    # Per-category breakdown this is the number that actually answers "does gTTS detection
    # generalize to unseen gTTS clips", not just the pooled accuracy above.
    per_category = {}
    for cat in ["real", "original_fake", "gtts"]:
        idxs = [i for i, c in enumerate(categories) if c == cat]
        cat_true = [y_true[i] for i in idxs]
        cat_pred = [y_pred[i] for i in idxs]
        n_correct = sum(1 for t, p in zip(cat_true, cat_pred) if t == p)
        per_category[cat] = {"n": len(idxs), "n_correct": n_correct, "accuracy": n_correct / len(idxs) if idxs else float("nan")}

    output_dir = os.path.join(RESULTS_DIR, "ast_asvspoof_finetuned")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("RESULTS — fine-tuned MattyB95/AST-ASVspoof2019 (GENUINELY held-out test pool, 130 clips)")
    print("=" * 70)
    print(f"Accuracy:        {metrics['accuracy']:.3f}")
    print(f"Macro Precision: {metrics['macro_precision']:.3f}")
    print(f"Macro Recall:    {metrics['macro_recall']:.3f}")
    print(f"Macro F1:        {metrics['macro_f1']:.3f}")
    print(f"AUC:             {metrics['auc']:.3f}")
    for cls, m in per_class.items():
        print(f"  {cls}: support={m['support']} precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f}")
    print(f"Confusion matrix (rows=true, cols=pred, order={CLASS_NAMES}): {cm}")
    print("\nPer-category breakdown (held-out, never seen in training):")
    for cat, m in per_category.items():
        print(f"  {cat}: {m['n_correct']}/{m['n']} correct ({m['accuracy']*100:.1f}%)")

    pd.DataFrame({"path": paths, "category": categories, "y_true": y_true, "y_pred": y_pred, "prob_fake": probs_fake}) \
        .to_csv(os.path.join(output_dir, "predictions_heldout_test.csv"), index=False)
    pd.DataFrame([{**metrics, "model": "ast_asvspoof_finetuned"}]) \
        .to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
    pd.DataFrame(history).to_csv(os.path.join(output_dir, "training_history.csv"), index=False)

    def _table(headers, rows):
        widths = [max(len(str(r[i])) for r in ([headers] + rows)) for i in range(len(headers))]
        fmt = lambda r: " | ".join(str(v).ljust(w) for v, w in zip(r, widths))
        out = [fmt(headers), "-+-".join("-" * w for w in widths)]
        out.extend(fmt(r) for r in rows)
        return out

    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write("DeepGuard Signal 2 (voice clone) - fine-tuned model evaluation\n")
        f.write("=" * 70 + "\n")
        f.write(f"Base checkpoint: {BASE_MODEL_ID}\n")
        f.write("Split: GENUINE disjoint train/test - 50 real + 50 original-fake + 30 gTTS for\n")
        f.write("training (130 total, further split 85/15 internally for checkpoint selection);\n")
        f.write("50 real + 50 original-fake + 30 gTTS for held-out testing (130 total),\n")
        f.write("never touched until this final evaluation.\n")
        f.write(f"Epochs: {args.epochs}  LR: {args.lr}  Batch size: {args.batch_size}\n\n")
        f.write("POOLED RESULTS (all 130 held-out test clips):\n")
        f.write("\n".join(_table(
            ["Accuracy", "Macro P", "Macro R", "Macro F1", "AUC"],
            [[f"{metrics['accuracy']:.3f}", f"{metrics['macro_precision']:.3f}",
              f"{metrics['macro_recall']:.3f}", f"{metrics['macro_f1']:.3f}", f"{metrics['auc']:.3f}"]]
        )) + "\n\n")
        f.write("Per-class breakdown:\n")
        f.write("\n".join(_table(
            ["Class", "Support", "Precision", "Recall", "F1"],
            [[cls, str(m["support"]), f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1']:.3f}"]
             for cls, m in per_class.items()]
        )) + "\n\n")
        f.write("Confusion matrix (rows=true, cols=pred):\n")
        f.write("\n".join(_table(
            ["True \\ Pred"] + CLASS_NAMES,
            [[CLASS_NAMES[i]] + [str(v) for v in row] for i, row in enumerate(cm)]
        )) + "\n\n")
        f.write("PER-CATEGORY BREAKDOWN (held-out, never seen in training):\n")
        f.write("\n".join(_table(
            ["Category", "Correct", "Total", "Accuracy"],
            [[cat, str(m["n_correct"]), str(m["n"]), f"{m['accuracy']*100:.1f}%"]
             for cat, m in per_category.items()]
        )) + "\n\n")
        f.write("This per-category number is the real answer to \"does this generalize\" -\n")
        f.write("the gtts row specifically reflects performance on gTTS clips the model\n")
        f.write("never saw during training or checkpoint selection, unlike v1's evaluation.\n")

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
    fig.suptitle("Fine-tuning — genuine disjoint train/held-out-test split")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curve.png"), dpi=150)
    plt.close(fig)

    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES))); ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Fine-tuned model — confusion matrix (genuine held-out test, 130 clips)")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            color = "white" if cm_arr[i, j] > cm_arr.max() / 2 else "black"
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center", color=color, fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
