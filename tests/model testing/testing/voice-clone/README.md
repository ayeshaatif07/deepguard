# Signal 2 model evaluation — voice clone detection

Standalone scripts, mirroring `tests/model testing/testing/video-detection/` and
`tests/model testing/testing/image-detection/` — evaluate candidate models *before*
wiring one into the live site.

```
tests/model testing/testing/voice-clone/
├── prepare_voice_dataset.py   ← downloads + samples the dataset
├── common.py                  ← shared CSV loading / metrics / reporting
├── test_melodymachine.py      ← MelodyMachine/Deepfake-audio-detection-V2
├── test_mothecreator.py       ← mo-thecreator/Deepfake-audio-detection
└── combine_voice_results.py   ← cross-model comparison
```

## Dataset

`prepare_voice_dataset.py` downloads a stratified sample from
[garystafford/deepfake-audio-detection](https://huggingface.co/datasets/garystafford/deepfake-audio-detection)
(public, not gated) — 1,866 FLAC clips, 933 real / 933 synthetic, spanning
6 TTS platforms (Amazon Polly, ElevenLabs, Hexgrad Kokoro, Hume AI,
Luvvoice, Speechify) for the fake class, and 14 YouTube recordings for the
real class:

```bash
cd tests/model testing/testing/voice-clone
python prepare_voice_dataset.py --per-class 100
```

`tests/model testing/Datasets/voice-clone/voice_clone_200_sample.csv` records the exact
200-clip sample (100 real / 100 fake) used for the comparison below.

## Run — one script per model

```bash
python test_mothecreator.py \
    --csv ../../Datasets/voice-clone/voice_clone_200_sample.csv \
    --audio-root ../../Datasets/voice-clone/audio \
    --device mps
```

Swap in `test_melodymachine.py` for the other candidate. Output goes to
`tests/model testing/Test-Results/voice-clone/<model_key>/`: `metrics.txt`,
`metrics_summary.csv`, `predictions_<model>.csv`, `per_class.png`,
`confusion_matrix.png`. `combine_voice_results.py` reads both models'
saved output and writes `tests/model testing/Test-Results/voice-clone/combined/`.

## Models

| Script | Model | Result |
|---|---|---|
| `test_melodymachine.py` | MelodyMachine/Deepfake-audio-detection-V2 | **Tested and rejected.** Self-reports 99.73% accuracy on its own eval split — scored only **49.0% accuracy / 0.537 AUC** on this project's independent 200-clip sample, barely above chance, heavily biased toward predicting "real" (92/100 fake clips misclassified). |
| `test_mothecreator.py` | mo-thecreator/Deepfake-audio-detection | **Benchmark leader, since superseded.** The model V2 was itself fine-tuned from. Scored **79.0% accuracy / 0.832 AUC** on the identical 200 clips — best of the three base candidates. Was the production model for Signal 2 until the fine-tuned AST checkpoint replaced it (see below). |
| `test_ast_asvspoof.py` | MattyB95/AST-ASVspoof2019-Synthetic-Voice-Detection | **Tested; adopted as the fine-tuning base.** Scored **59.5% accuracy / 0.669 AUC** on the same 200 clips — below mo-thecreator, but the only candidate showing any residual signal on gTTS after threshold calibration. |
| `finetune_ast_asvspoof.py` | `ast_asvspoof_finetuned` (this project's own fine-tune) | **Current production model for Signal 2.** Fine-tuned from the AST checkpoint above on project data. On a disjoint 130-clip held-out set: **93.8% accuracy / 0.936 macro F1 / 0.985 AUC**, with per-category accuracy of 98.0% real, 86.0% platform-fake, 100.0% gTTS. Wired into `modules/voice_detection/voice_clone_scorer.py`. |

**Why the fine-tune was needed:** all three base candidates scored **0/60**
on the project owner's own gTTS-synthesised clips — a complete blind spot
rather than a calibration issue, since gTTS was not one of the six
commercial TTS platforms the benchmark corpus was built from. AST was
chosen as the fine-tuning base over the stronger mo-thecreator precisely
because it alone showed a hint of real signal on gTTS after threshold
calibration, i.e. a recoverable blind spot rather than a total one.

> Note: figures on the disjoint 130-clip held-out set are **not** directly
> comparable to the 200-clip benchmark above — different test data.

The gap between the first two is the headline finding here: V2 is a *further*
fine-tune of the chosen model, self-reporting a much higher number on its
own card — but its extra fine-tuning step appears to have overfit to a
narrower training distribution and lost generalization to unseen TTS
platforms, exactly the "self-reported accuracy ≠ real generalization"
lesson learned earlier with the image models (`prithivMLmods/*-9999` and
`-v2.0`) in `tests/model testing/testing/image-detection/`.

## Repo-level layout

See `tests/model testing/testing/README.md` at the top of `tests/model testing/testing/` for the full
repo layout across all signals (video/image/voice).
