# Signal 3 model evaluation — voice manipulation scoring

Rebuilt from scratch against a real, hand-constructed dataset (see
`tests/model testing/Datasets/voice-manipulation/README.md`), mirroring the structure of
`tests/model testing/testing/voice-clone/` and the other signals.

```
tests/model testing/testing/voice-manipulation/
├── common.py                        ← shared CSV loading / metrics / reporting
├── test_bothbosu.py                 ← BothBosu/roberta-scam-classifier-v1
├── test_propaganda.py               ← valurank/distilroberta-propaganda-2class
├── test_goemotions.py               ← SamLowe/roberta-base-go_emotions (derived score)
└── combine_manipulation_results.py  ← cross-model comparison + threshold-calibration check
```

## Dataset

`tests/model testing/Datasets/voice-manipulation/signal3_transcripts_final.csv` — 60
real clips constructed by the project owner (not downloaded), perfectly
balanced (30 manipulative / 30 non-manipulative), each with a transcript,
label, and a hand-annotated `manipulation_tactics` column (18 distinct
tags: urgency, authority, fear, guilt, FOMO, trust exploitation, legal
threat, etc.). Matching mp3 audio for every row lives under
`tests/model testing/Datasets/voice-manipulation/audio/`.

## Models tested — 3 candidates

All 3 are text classifiers, run directly on the `transcript` column (not
the audio — see the top-level project README for where audio-pipeline
verification happens):

| Script | Model | Native labels | Mapping |
|---|---|---|---|
| `test_bothbosu.py` | `BothBosu/roberta-scam-classifier-v1` | `SCAM` / `NON-SCAM` | SCAM→manipulative |
| `test_propaganda.py` | `valurank/distilroberta-propaganda-2class` | `Prop` / `No_Prop` | Prop→manipulative |
| `test_goemotions.py` | `SamLowe/roberta-base-go_emotions` | 28-class multi-label emotion | derived score: sum of fear+nervousness+anger+annoyance+disapproval+disgust probabilities |

`test_goemotions.py` is included specifically because this project's own
preliminary report (Section 2.4.4) cites the **GoEmotions dataset** by
name as the intended training basis for this signal, without pinning an
exact checkpoint ("exact model versions will be confirmed during
implementation") — this is the model that proposal was describing.

## Run

```bash
cd tests/model testing/testing/voice-manipulation
python test_bothbosu.py --device mps
python test_propaganda.py --device mps
python test_goemotions.py --device mps
python combine_manipulation_results.py
```

Output per model goes to `tests/model testing/Test-Results/voice-manipulation/<model_key>/`:
`metrics.txt`, `metrics_summary.csv`, `predictions_<model>.csv`,
`per_class.png`, `confusion_matrix.png`.

## Results

| Model | Accuracy (0.5 threshold) | AUC | Best-threshold accuracy |
|---|---|---|---|
| `bothbosu` | 50.0% | 0.642 | 66.7% (at threshold 0.23) |
| `propaganda` | 50.0% | 0.890 | 81.7% (at threshold 0.07) |
| `goemotions` | 53.3% | **0.936** | **90.0%** (at threshold 0.02) |

**All 3 models showed the same miscalibration pattern already seen once
in this project** (UniversalFakeDetect, in the video-detection benchmark):
a fixed 0.5 threshold on `argmax`/sigmoid output badly underestimates
real accuracy, because none of these models were originally trained or
calibrated for *this* dataset's specific distribution — they all skew
heavily toward predicting the negative class at the default cutoff. AUC
(threshold-independent) and best-possible-threshold accuracy
(`combine_manipulation_results.py`'s calibration sweep) are the
trustworthy numbers here, not the raw 0.5-threshold accuracy.

**`goemotions` is the clear winner on both**, and only 2/60 clips have any
disagreement across all 3 models (see `combined/combined_predictions.csv`).

## Decision

Of the three **base** candidates, `SamLowe/roberta-base-go_emotions`
(derived manipulation score, summed over the 6-emotion "coercive"
cluster) was the winner — best AUC (0.936) and best best-threshold
accuracy (0.900 @ 0.02), *and* it's the model this project's own report
proposal points to. As a derived-score heuristic it required a
recalibrated threshold (~0.02, not the naive 0.5); see
`combine_manipulation_results.py`'s calibration sweep for how that number
was derived.

**Superseded — production now runs `goemotions_finetuned`.** Rather than
ship the derived-score heuristic, the same base checkpoint was fine-tuned
directly on this project's transcripts, replacing its 28-class emotion
head with a fresh binary head (`finetune_goemotions.py`; 10 epochs, LR
2e-05, batch size 4; 40 transcripts for training, a disjoint 20 held
out). On those 20 held-out transcripts the fine-tune scores **0.950
accuracy / 0.950 macro F1 / 1.000 AUC**, against the derived-score
heuristic's **0.750 / 0.749 / 0.860** on the identical set — an
apples-to-apples comparison. The fine-tuned model is wired into
`modules/voice_detection/voice_manipulation_scorer.py` and needs no
threshold recalibration, since its binary head is trained directly on the
target label.

> Caveat: the held-out set is only 20 transcripts, so these figures carry
> correspondingly wide uncertainty and should be re-validated on a larger
> sample.

## Output

`tests/model testing/Test-Results/voice-manipulation/combined/`: `combined_metrics_summary.csv`,
`combined_predictions.csv`, `metrics_comparison.png`,
`threshold_calibration.png`, `auc_comparison.png`, `metrics.txt`.
