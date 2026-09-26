# DeepGuard

A multi-signal deepfake detection system. DeepGuard analyses a video, image, audio
clip or social media post through five independent detection signals and combines
them into a single verdict, so a piece of media is judged on visual, acoustic and
textual evidence rather than on one model's opinion.

Each signal can be run on its own, or a pasted social media link can be pushed
through every signal automatically in one pass.

---

## What It Does

| Capability | Detail |
|---|---|
| Video deepfake detection | Samples 15 evenly spaced frames and scores each for facial manipulation |
| AI-generated image detection | Classifies a still as Real, Artificial or Deepfake |
| Voice clone detection | Flags synthetically generated or cloned speech |
| Voice manipulation scoring | Transcribes speech, then scores it for manipulative and coercive language |
| Caption coherence | Checks whether a post's caption actually matches what is said in its audio |
| Fusion verdict | Ranks all completed signals and surfaces the most concerning as the Overall verdict |
| Social media pipeline | Downloads from YouTube, Facebook or Instagram and runs the chain end to end |
| PDF report | Exports whichever signals have run in the current session |

---

## AI Model Pipelines

Five signals, each backed by its own model. Every production model was selected by
benchmarking candidates against alternatives; see `tests/model testing/`.

| Signal | Task | Production model | Notes |
|---|---|---|---|
| **1** | Video deepfake detection | CViT2 (`Deressa/cvit`) | Convolutional Vision Transformer. Chosen over EfficientNet-B7-NS, which scored higher on accuracy but was slower and had lower recall |
| **1b** | Image detection | `prithivMLmods/AI-vs-Deepfake-vs-Real` | Three-class ViT. Two newer variants were rejected after showing data leakage |
| **2** | Voice clone detection | `ast_asvspoof_finetuned` | Audio Spectrogram Transformer, fine-tuned on project data. 93.8% accuracy. Off-the-shelf candidates failed on real audio |
| **3a** | Speech transcription | `openai/whisper-small` | Adopted after four ASR candidates were compared on LibriSpeech test-other (WER 0.079) |
| **3b** | Voice manipulation | `goemotions_finetuned` | RoBERTa with a fresh binary head fine-tuned on GoEmotions. 0.950 accuracy. Base candidates were miscalibrated at a 0.5 threshold |
| **4** | Caption coherence | `sentence-transformers/all-MiniLM-L6-v2` | Bi-encoder cosine similarity at a calibrated 0.62 threshold |

**Audio is decoded once.** Signals 2 and 3 share a single mono 16 kHz waveform
produced by one `ffmpeg` call, along with the waveform display and in-browser
playback: four consumers, one decode.

**Fusion is rank-based, not a blend.** Each completed signal contributes a
`(rank_key, verdict, score)` triple. Signal 4's polarity is inverted, since a high
coherence score means *safer* content. The highest-ranking triple becomes the
Overall verdict, carrying that signal's own score rather than an average. This
prevents one confidently-safe signal from diluting a genuine detection.

---

## Folder Structure

```
deepguard/
├── app.py                        Flask application: all routes, session handling, JSON API
├── check_install.py              Post-install dependency and device check
├── setup.sh                      Creates the venv and installs dependencies
├── requirements.txt              All Python dependencies (runtime + testing)
│
├── modules/
│   ├── frame_extractor.py        Evenly spaced frame sampling for Signal 1
│   ├── video_detection/          Signal 1  — CViT2 scorer
│   ├── image_detection/          Signal 1b — three-class ViT scorer
│   ├── voice_detection/          Signals 2 and 3 — shared audio utilities, AST and
│   │                             Whisper + RoBERTa scorers
│   ├── caption_coherence/        Signal 4  — MiniLM similarity scoring
│   ├── social_media/             Platform dispatcher and yt-dlp extractors
│   ├── usage_log.py              SQLite aggregate usage logging
│   └── report_generator.py       PDF report generation via ReportLab
│
├── templates/                    base.html shell plus one template per page
├── static/
│   ├── js/app.js                 Uploads, waveform rendering, dashboard updates
│   ├── css/                      Styling and the verdict-badge colour system
│   └── images/                   Logo and background assets
│
└── tests/
    ├── unit testing/             Code correctness (models mocked)
    ├── functionality testing/    End-to-end behaviour (real models)
    ├── model testing/            Model accuracy benchmarks and datasets
    └── performance testing/      Latency, resource usage and scaling
```

Generated at runtime and excluded from version control: `venv/`, `uploads/`,
`usage_log.db`, `__pycache__/`.

---

## Setup Requirements

### System

- **Python 3.13** (developed and tested on 3.13.9)
- **ffmpeg** on `PATH`: required, not installable via pip

```bash
brew install ffmpeg          # macOS
sudo apt install ffmpeg      # Debian/Ubuntu
```

### Audio File Processing

Every audio and video upload is decoded by an `ffmpeg` subprocess into a mono
16 kHz float32 waveform before Signals 2 and 3 run. Without ffmpeg on `PATH`,
voice analysis fails immediately with a clear error while the other signals
continue to work.

Uploads with no audio track are detected and reported rather than crashing:
Signals 2 and 3 are skipped and the video and caption signals still stand.

### User Interface

No frontend build step. The interface is server-rendered Jinja templates plus
vanilla JavaScript, with Chart.js loaded from a CDN for the per-frame score
chart. Results persist across pages via `sessionStorage`, so navigating between
signals does not lose work.

### Model Weights

**Model weights are not in this repository.** Each file exceeds GitHub's 100 MB
limit; the CViT2 checkpoint alone is 1.0 GB.

| Weight | Size | How to obtain |
|---|---|---|
| CViT2 (Signal 1) | ~1.0 GB | Downloaded automatically from HuggingFace on first run |
| Image ViT (Signal 1b) | 327 MB | Downloaded automatically from HuggingFace |
| Whisper-small (Signal 3a) | 926 MB | Downloaded automatically from HuggingFace |
| MiniLM (Signal 4) | 87 MB | Downloaded automatically from HuggingFace |
| `ast_asvspoof_finetuned` (Signal 2) | 329 MB | Produced by `finetune_ast_asvspoof.py` under `tests/model testing/` |
| `goemotions_finetuned` (Signal 3b) | 476 MB | Produced by `finetune_goemotions.py` under `tests/model testing/` |

The first run downloads roughly 2.3 GB and will be slow. The two fine-tuned models
must be trained locally or copied in, because the application will not start
without them: all five models load at startup.

---

## Running the Application

### 1. Create virtual environment

```bash
python3 -m venv venv
```

### 2. Activate virtual environment

```bash
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Verify the install, including the ffmpeg binary:

```bash
python check_install.py
```

Steps 1–3 can be replaced by `./setup.sh`.

### 4. Run the Flask app

```bash
python app.py
```

All five models load at startup, so the first launch takes 30–60 seconds. This is
deliberate: no user request ever pays the loading cost.

To run the interface without model weights, useful for frontend work:

```bash
DEEPGUARD_MOCK=1 python app.py
```

### 5. Open your browser at

```
http://localhost:5001
```

| Page | Route |
|---|---|
| Home | `/` |
| Analyse Video | `/video` |
| Image Detection | `/image` |
| Voice Analysis | `/voice` |
| Caption Check | `/caption` |
| Full Verdict Dashboard | `/verdict` |
| Same dashboard, alias route | `/dashboard` |

---

## Datasets and Licence Terms

Datasets are **not redistributed in this repository.** Each must be obtained from
its own source under its own terms.

### FaceForensics++: video deepfake detection

Obtained through the research-access process run by the Technical University of
Munich, and used under the **FaceForensics Terms of Use: non-commercial research
and educational use only.**

Request access at <https://github.com/ondyari/FaceForensics>, then run the
download script TUM provides. The c23 compression level is used throughout.

> **Do not use third-party mirrors.** Copies redistributed on dataset-sharing
> sites fall outside the licence, regardless of how convenient they are.

### prithivMLmods/AI-vs-Deepfake-vs-Real: image detection

Licensed under the **Apache License 2.0**, and **gated**: accept the terms on the
dataset page while signed in, then authenticate with `huggingface-cli login`.
A 600-image sample (200 per class) is used for benchmarking.

Dataset page: <https://huggingface.co/datasets/prithivMLmods/AI-vs-Deepfake-vs-Real>

Apache 2.0 permits research use and redistribution with attribution, but the
gating is a separate access condition: each user must accept the terms under
their own account, so the images are not redistributed here. The dataset is
itself assembled from subsets of three upstream datasets, whose own terms may
also apply to onward redistribution.

### LibriSpeech test-other: speech transcription

Openly available under **CC BY 4.0**, downloaded automatically by the benchmark
script via the `datasets` library. A 500-clip sample is used.

### Caption coherence pairs

28 caption/transcript pairs collected from public social media posts, used only
to calibrate and verify Signal 4's similarity threshold.

---

## Running Tests

Four suites under `tests/`, each answering a different question. Run all commands
from the project root with the venv active.

### Unit Tests

*Does the code behave correctly?* Models are mocked, so this is fast and needs no
weights or GPU.

**Backend.** 36 tests. `conftest.py` installs lightweight fakes for all five model
classes into `sys.modules` *before* `app.py` is imported, so the eager startup
loading constructs fakes instead of real weights.

```bash
cd "tests/unit testing/backend" && python -m pytest -v
```

Covers the usage-statistics SQLite log, the PDF report's fusion and ranking logic,
Signal 4's empty-input handling, and route validation.

**Frontend.** Jest against `static/js/app.js`.

```bash
cd "tests/unit testing/frontend" && npm install && npm test
```

### Functionality Tests

*Does the feature actually work for a real user?* Black-box integration tests
driving the real Flask app end to end: real routes, real session and cookie
handling, real template rendering, against **real, unmocked models** on real
sample media.

```bash
python -m pytest "tests/functionality testing" -v
```

Each file maps to one functional requirement (FR1–FR9) from the report: upload
validation, each of the five signals, the fusion verdict, dashboard display and
partial-pipeline behaviour. Slower than the unit suite, since all five models load
once per session, but the only suite that proves a feature works end to end.

Requires the datasets, since it runs against real sample media.

### Model Tests

*Is the model accurate?* Per-signal accuracy benchmarking that compares candidate
models and records why each production model was chosen.

```bash
cd "tests/model testing/testing"
```

Each signal has its own scripts and results:

| Area | Candidates compared |
|---|---|
| `video-detection/` | CViT2, EfficientNet-B7-NS, and others on a 150-clip FF++ sample |
| `image-detection/` | Three ViT variants on 600 images; two showed data leakage |
| `voice-clone/` | Off-the-shelf candidates vs. the fine-tuned AST |
| `audiotranscription/` | Four ASR candidates on LibriSpeech test-other |
| `voice-manipulation/` | Base candidates vs. the GoEmotions fine-tune |
| `caption-coherence/` | Four candidates on 28 caption/transcript pairs |

#### Committed Results

**The benchmark output is in the repository; you do not have to re-run anything
to see how the models compared.** Every figure quoted in the report comes from
these files.

```
tests/model testing/Test-Results/<signal>/
├── <candidate>/              One directory per model tried
│   ├── metrics.txt           Accuracy, macro F1, AUC, inference time
│   ├── metrics_summary.csv
│   └── predictions_<name>.csv    Per-item predictions, for auditing
└── combined/                 Cross-candidate comparison
    ├── metrics.txt           The table the model choice was made from
    ├── combined_metrics_summary.csv
    └── *.png                 ROC curves, per-class and accuracy charts
```

Performance results are stored the same way, in
`tests/performance testing/results/` as paired `.txt` and `.png` files per script.

The datasets themselves are **not** committed, since their licences do not allow
redistribution, so re-running a benchmark requires obtaining them first.

See `tests/model testing/testing/README.md` for per-script instructions. First runs
download large checkpoints (~1 GB for CViT2, ~266 MB for EfficientNet).

Requires the datasets.

### Performance Tests

*How fast and how resource-hungry is it?* Not correctness: speed and cost.

```bash
python "tests/performance testing/run_all.py"
python "tests/performance testing/plot_results.py"      # charts from the results
```

| Script | Measures |
|---|---|
| `per_model_latency.py` | Per-signal inference time plus one-time model load time |
| `pipeline_latency.py` | End-to-end request latency through real Flask routing |
| `concurrent_load.py` | Latency and error rate at 1/2/4/8 simultaneous requests |
| `resource_usage.py` | Process CPU and RSS, idle vs. under load |
| `database_scaling.py` | `usage_log.db` insert and query time from 100 to 100,000 rows |

Results are sensitive to machine load. Close other heavy applications before
running; a busy browser alone can inflate inference timings several times over.

---

## Known Limitations

- **TikTok is not supported.** Five extraction approaches were tried and all were
  blocked by TikTok's anti-bot system. Support was removed rather than shipped
  broken.
- **Instagram requires authentication.** A cookies file is needed for the
  Instagram extractor; it is gitignored.
- **Caption coherence needs a caption.** Posts with an empty description, common
  on YouTube Shorts, cannot be scored, since there is nothing to compare the
  transcript against.
- **Single-user local tool.** No cloud API, no authentication, no concurrent-user
  design. The Flask development server is used as-is.

---

## Licence

This project is released under the **Apache License 2.0**; see [`LICENSE`](LICENSE).

Third-party components keep their own terms, which are not superseded by the
above:

| Component | Type | Terms |
|---|---|---|
| `Deressa/cvit` (Signal 1) | Model | No licence stated on the model repository; used here for non-commercial academic evaluation only |
| `prithivMLmods/AI-vs-Deepfake-vs-Real` (Signal 1b) | Model | [Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/) |
| `prithivMLmods/AI-vs-Deepfake-vs-Real` (image benchmark, gated) | Dataset | [Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/) |
| `openai/whisper-small` (Signal 3a) | Model | Apache License 2.0 |
| `sentence-transformers/all-MiniLM-L6-v2` (Signal 4) | Model | Apache License 2.0 |
| FaceForensics++ (Signal 1 benchmark) | Dataset | FaceForensics Terms of Use: non-commercial research and educational use only |
| LibriSpeech test-other (Signal 3a benchmark) | Dataset | CC BY 4.0 |
| `garystafford/deepfake-audio-detection` (Signal 2 benchmark) | Dataset | CC BY 4.0, attribution to Stafford, G. |

The fine-tuned checkpoints (`ast_asvspoof_finetuned`, `goemotions_finetuned`) are
derivative works of their respective base models and inherit those models' terms.
