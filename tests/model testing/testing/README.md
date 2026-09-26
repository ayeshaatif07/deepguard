# Model evaluation (comparative benchmarking) suite

Standalone scripts — live outside `app.py` / `modules/` on purpose, so you can
evaluate candidate models *before* wiring one into the live site. Lives under
`tests/model testing/testing/`, a sibling of `tests/model testing/Test-Results/` (outputs) and
`tests/model testing/Datasets/` (frozen sample CSVs) — see the repo-level layout note at
the bottom.

`tests/model testing/testing/` is split into one subfolder per signal:

```
tests/model testing/testing/
├── video-detection/      ← Signal 1.  4 candidates + weights/
├── image-detection/      ← Signal 1b. 3 candidates
├── voice-clone/          ← Signal 2.  3 base candidates + this project's fine-tune
├── audiotranscription/   ← shared ASR. 4 candidates
├── voice-manipulation/   ← Signal 3.  3 base candidates + this project's fine-tune
└── caption-coherence/    ← Signal 4.  4 candidates
```

Each folder holds its own `common.py` (shared loading / metrics / reporting),
one `test_<model>.py` per candidate, and a `combine_*_results.py` that writes
the cross-model comparison. `voice-clone/`, `voice-manipulation/` and
`audiotranscription/` have their own README.md with per-signal detail.

`video-detection/` and `image-detection/` are documented in full below.
Short summary of the rest:

- **voice-clone/ (Signal 2)**: 3 base candidates benchmarked on a 200-clip
  independent sample (100 real / 100 fake, 6 TTS platforms) from
  `garystafford/deepfake-audio-detection`. The newer "V2" fine-tune
  (self-reporting 99.73% on its own eval split) scored only 49.0% accuracy /
  0.537 AUC here, while the older model it was fine-tuned from scored
  79.0% / 0.832 AUC on identical clips. All three base models scored 0/60 on
  gTTS speech - a blind spot, not a calibration issue - so the AST candidate
  was fine-tuned on project data instead (`finetune_ast_asvspoof.py`).
  **`ast_asvspoof_finetuned` is the production model**: 93.8% accuracy /
  0.985 AUC on a disjoint 130-clip held-out set. See
  `modules/voice_detection/voice_clone_scorer.py`.

- **voice-manipulation/ (Signal 3)**: 3 base candidates on 60 hand-built
  transcripts, all miscalibrated at a 0.5 threshold. The GoEmotions candidate
  was fine-tuned with a fresh binary head (`finetune_goemotions.py`).
  **`goemotions_finetuned` is the production model**: 0.950 accuracy /
  0.950 macro F1 / 1.000 AUC on 20 held-out transcripts. See
  `modules/voice_detection/voice_manipulation_scorer.py`.

- **audiotranscription/ (shared ASR)**: 4 candidates on LibriSpeech
  test-other. **Whisper-small adopted** (WER 0.079).

- **caption-coherence/ (Signal 4)**: 4 candidates on 28 caption/transcript
  pairs. **MiniLM adopted** at a calibrated 0.62 threshold.

**video-detection/ — each model has its own separate script file:**

| Script | Model |
|---|---|
| `test_naman.py` | Naman712/Deep-fake-detection |
| `test_cvit.py` | CViT2 (Wodajo & Atnafu, CNN+ViT) |
| `test_universal.py` | UniversalFakeDetect |
| `test_efficientnet.py` | EfficientNet-B7-NS (selimsef DFDC winner) |

Run one at a time, e.g. `python test_naman.py --csv ... --video-root ...` —
not all together. `common.py` (also in `video-detection/`) holds the shared
CSV-loading / metrics / reporting code every script imports from, so results
stay consistent across models, but it isn't runnable on its own.

`image-detection/prepare_image_dataset.py` is separate — it prepares the
image-detection dataset (see near the bottom of this file), not a video model.

## Setup

Uses the same venv as the main app. The benchmark-only packages are in the
testing section of `requirements.txt` at the repo root:

```bash
pip install -r requirements.txt
```

## Get the dataset

FaceForensics++ is obtained through the dataset's own research-access
process run by the Technical University of Munich, and used under the
FaceForensics Terms of Use (non-commercial research and educational use).
Request access at https://github.com/ondyari/FaceForensics, then run the
download script TUM provides. Do not use third-party mirrors: they fall
outside the licence.

The copy used here is already in the repo, together with the frozen
150-clip manifest that every video benchmark reads - see
`tests/model testing/Datasets/video-detection/README.md`. The CSV format is one row per
video with a path and a REAL/FAKE label, e.g.:

```python
import os, csv
root = "/path/to/ff-c23"
rows = []
for dirpath, _, files in os.walk(root):
    for f in files:
        if f.endswith(".mp4"):
            label = "FAKE" if "manipulated_sequences" in dirpath else "REAL"
            rows.append((os.path.relpath(os.path.join(dirpath, f), root), label))
with open("ff_c23_labels.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["path", "label"])
    w.writerows(rows)
```

`tests/model testing/Datasets/video-detection/video_detection_150_official_sample.csv`
records the exact 150-video stratified sample (75 real / 75 fake) used for
every model comparison, pointing into `official_ff_download/`, the copy
obtained through FaceForensics++'s official research-access process at
c23 (HQ). The same 150 clips are also copied to
`tests/model testing/Datasets/video-detection/videos/REAL/` and `videos/FAKE/`, named
`<source>_<clip>.mp4` so each file's origin (youtube, Deepfakes,
Face2Face, FaceShifter, FaceSwap, NeuralTextures, DeepFakeDetection) is
readable from the filename alone.

## Run — one script per model

```bash
cd tests/model testing/testing/video-detection
python test_naman.py \
    --csv ff_c23_labels.csv \
    --video-root /path/to/ff-c23 \
    --frames 20 \
    --limit 100 \
    --device cpu
```

Swap `test_naman.py` for `test_cvit.py`, `test_universal.py`, or
`test_efficientnet.py` to evaluate a different model — same flags, separate
script, separate run.

- `--limit` caps the run to a stratified (equal real/fake) sample — use this
  first for a quick sanity pass before running the full dataset.
- `--device mps` uses Apple Silicon GPU acceleration where supported
  (confirmed working for all four scripts — roughly 1.5-3x faster than CPU
  with identical accuracy, verified by rerunning and comparing metrics).
- First run of `test_cvit.py` downloads a ~1GB checkpoint from HuggingFace;
  `test_efficientnet.py` downloads a ~266MB checkpoint from a GitHub
  Release. Both are cached in `video-detection/weights/` after that.

**Caution:** never run a smoke test (e.g. `--limit 4`) without an explicit
`--output-dir` pointing somewhere disposable — every script defaults to
writing into `tests/model testing/Test-Results/video-detection/<model>/`, which will
silently overwrite real results from a full run. Use
`--output-dir /tmp/smoke-test` or similar for anything exploratory.

## Output

Each run writes to `tests/model testing/Test-Results/video-detection/<model>/` (e.g.
`tests/model testing/Test-Results/video-detection/naman/`):

- **Terminal**: Accuracy / Precision / Recall / F1 / AUC / avg inference time
  for that model, printed live as each video is processed.
- `metrics.txt` — the same results table as a plain-text file (run
  configuration, table, per-class Real/Fake breakdown, confusion matrix) —
  readable without opening a CSV/spreadsheet.
- `metrics_summary.csv` — the same numbers, machine-readable, including
  flattened `real_*` / `fake_*` per-class columns.
- `predictions_<model>.csv` — every video's raw score, so you can audit
  individual misclassifications.
- `metrics.png` — bar chart of Accuracy/Precision/Recall/F1/AUC.
- `inference_time.png` — average inference time per video.
- `roc_curve.png` — ROC curve (only if the evaluated set had both classes).
- `per_class.png` — grouped Real vs Fake bar chart (Accuracy/Precision/Recall/F1).

`combine_video_results.py` reads every model's saved output (no re-running
inference) and writes a cross-model comparison to
`tests/model testing/Test-Results/video-detection/combined/`: `combined_metrics_summary.csv`,
`combined_predictions.csv` (one row per video, one score column per model),
and 5 comparison charts. Run it any time after two or more models have been
benchmarked.

## Models

| Script | Model | Notes |
|---|---|---|
| `test_naman.py` | Naman712/Deep-fake-detection | Tested and rejected. Architecture + weights vendored locally at `weights/naman/` (not in `modules/` — this model isn't in production). Includes face cropping (OpenCV Haar cascade) since the model was trained on face-cropped frames; also uncovered a real inverted-label-index bug during testing. |
| `test_cvit.py` | CViT2 (Wodajo & Atnafu, CNN+ViT) | **Chosen for Signal 1** — wired into the live app as `modules/video_detection/cvit_scorer.py`. The real, weighted model behind the "Wodajo and Atnafu" citation in the report — arXiv:2102.11126. Best F1/recall/speed of all models tested. |
| `test_universal.py` | UniversalFakeDetect (Ojha et al. 2023) | Tested and rejected. Architecture + weights vendored locally at `weights/universal/` (not in `modules/`). Known calibration issue: the fixed 0.5 threshold is miscalibrated for this domain — only AUC is reliable, not Accuracy/Precision/Recall/F1 (see `metrics.txt` for the note). |
| `test_efficientnet.py` | EfficientNet-B7-NS (selimsef DFDC winner) | Tested and rejected. The real "EfficientNet" candidate from the report — actual Kaggle DFDC Challenge winning solution, not the report's separately-proposed (and unweighted/undownloadable) "EfficientNet-B5 + Bi-LSTM" design. Highest accuracy/AUC of all models tested, but slower and lower-recall than CViT2. |

CViT2 was the model selected for Signal 1 — see `modules/video_detection/cvit_scorer.py`
and `tests/model testing/Test-Results/video-detection/` for the full comparison and
decision writeup. `modules/` now contains only the model actually used in
production; every candidate's architecture/weights (including CViT2's own
copy, downloaded separately for testing) live under `tests/model testing/testing/video-detection/weights/`
so this comparison stays reproducible without depending on `modules/`.

## Image detection (Signal 1b)

`image-detection/` benchmarked 3 candidate 3-class models
(`prithivMLmods/AI-vs-Deepfake-vs-Real-9999`, `-v2.0`, and the original
ViT version) on a 600-image sample (200/class) from the gated HuggingFace
dataset `prithivMLmods/AI-vs-Deepfake-vs-Real`. Two of the three showed
data leakage; the original ViT version was chosen for production — see
`image-detection/prepare_image_dataset.py`, `test_9999.py`, `test_v2.py`,
`test_original.py`, `combine_image_results.py`, and
`modules/image_detection/image_scorer.py`.

## Voice clone detection (Signal 2)

See `voice-clone/README.md` - summarised at the top of this file. Signal 3
(voice manipulation scoring) has its own folder and README at
`voice-manipulation/`.

## Repo-level layout

```
deepguard/
├── app.py, modules/, templates/, static/   ← the live Flask app
└── tests/model testing/
    ├── testing/             ← one folder per signal (see the tree above)
    ├── Test-Results/        ← per-model metrics, predictions and graphs,
    │                          plus combined/ per signal
    └── Datasets/
        ├── video-detection/
        │   ├── video_detection_150_official_sample.csv
        │   ├── official_ff_download/      ← official FF++ research-access copy, c23
        │   └── videos/REAL/, videos/FAKE/ ← browsable copies of the 150 sampled clips
        ├── image-detection/
        │   ├── image_detection_600_sample.csv
        │   └── images/Artificial/, images/Deepfake/, images/Real/
        └── voice-clone/
            ├── voice_clone_200_sample.csv
            └── audio/real/, audio/fake/           ← local copies of the 200 sampled clips
```

