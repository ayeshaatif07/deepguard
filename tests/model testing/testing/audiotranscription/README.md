# Speech-to-text (ASR) model evaluation — Signal 3 / Signal 4

A real gap this project had: `openai/whisper-base` was integrated directly
into production (Signal 3's transcription and Signal 4's caption-extraction
pipeline, both in `modules/voice_detection/voice_manipulation_scorer.py`)
without ever being benchmarked against alternatives — unlike every other
model in this project, which went through a test-and-compare process
before being chosen. This folder closes that gap.

```
tests/model testing/testing/audiotranscription/
├── common.py                        ← shared CSV loading / WER / CER / reporting
├── prepare_librispeech.py           ← builds the LibriSpeech test-other subset
├── test_whisper.py                  ← openai/whisper-base
├── test_whisper_small.py            ← openai/whisper-small (current production)
├── test_wav2vec2.py                 ← facebook/wav2vec2-large-960h-lv60-self
├── test_distil_whisper.py           ← distil-whisper/distil-large-v3
└── combine_transcription_results.py ← cross-model comparison
```

## Models tested — 4 candidates

`facebook/mms-1b-all` was tried and dropped (see "Excluded: mms-1b-all"
below) — not included in scripts or results.

| Script | Model | Architecture |
|---|---|---|
| `test_whisper.py` | `openai/whisper-base` | Encoder-decoder transformer (original production) |
| `test_whisper_small.py` | `openai/whisper-small` | Encoder-decoder transformer, 3.3x whisper-base's params (current production) |
| `test_wav2vec2.py` | `facebook/wav2vec2-large-960h-lv60-self` | CTC-based, English-only |
| `test_distil_whisper.py` | `distil-whisper/distil-large-v3` | Distilled Whisper — same lineage, different training/compression |

## Dataset

**`tests/model testing/Datasets/librispeech-test-other/`** — a fixed 500-clip random
sample (`random.sample`, seed 42) drawn from LibriSpeech's `test-other`
split (2,939 clips total, ~328MB) — LibriSpeech's noisier/harder
companion split to `test-clean`, a tougher stress test than clean
single-speaker narration, though still not the actual production domain
(see "Real-clip findings" below). Built by
`prepare_librispeech.py --config other --n 500`.

An earlier pass also built and benchmarked a `test-clean` sample (results
summarized below for context) but that dataset and its results have
since been deleted — `test-other` alone is kept going forward, since it's
the harder/more representative of the two generic LibriSpeech options.

Note on decoding: `datasets`' default audio decoder (`torchcodec`) failed
to load here — it's built against an older FFmpeg ABI (`libavutil.56`)
than the FFmpeg 9.x installed via Homebrew on this machine.
`prepare_librispeech.py` works around this by disabling auto-decode
(`Audio(decode=False)`) and decoding the raw bytes with `soundfile`
directly instead.

## Metrics

- **WER (Word Error Rate)**: the standard ASR accuracy metric — proportion
  of words wrong (substituted, inserted, or deleted), lowercased and
  punctuation-stripped before comparison so formatting differences (e.g.
  "10:30 AM" vs "10.30 a.m.") don't count as errors.
- **CER (Character Error Rate)**: the same idea at character granularity —
  useful for catching models whose output "sounds right" but is built
  from mangled sub-word pieces.
- **Avg inference time** per clip.

A real bug was found and fixed while building this: `jiwer`'s CER
function requires its OWN character-level reduction transform
(`ReduceToListOfListOfChars`), not the word-level one WER uses — an
earlier version of `common.py` applied the wrong transform, which made
`wav2vec2` (whose raw output is unpunctuated ALL-CAPS text) look far worse
on CER (0.824) than it actually was once measured on equal footing (0.040).

## Results — LibriSpeech `test-other` (500 clips, seed 42)

| Model | WER | CER | Avg inference | N |
|---|---|---|---|---|
| whisper (base) | 0.121 | 0.055 | 1.930s | 500 |
| **wav2vec2** | **0.042** | **0.014** | 1.338s | 500 |
| distil_whisper | 0.060 | 0.023 | 4.211s | 500 |
| whisper_small | 0.079 | 0.036 | 3.335s | 500 |

For context, an earlier `test-clean` run (since deleted) showed the same
ranking on the generic-benchmark axis: whisper 0.054, wav2vec2 0.019,
distil_whisper 0.033 WER — `wav2vec2` won decisively on both LibriSpeech
splits, clean and noisy alike.

## Real-clip findings — the LibriSpeech verdict didn't transfer

After swapping production to `wav2vec2` on the strength of the LibriSpeech
numbers above, 4 real YouTube Shorts clips were pulled and transcribed
across models side by side, as a sanity check against actual traffic
(short, informal, sometimes-synthetic-voice, sometimes noisy
social-media audio) rather than LibriSpeech's scripted narration:

| Clip | whisper-base | whisper-small | wav2vec2 |
|---|---|---|---|
| AI-voice, 10s | Clean ("...Bye!") | Clean, identical | Mis-transcribed trailing word ("BY") |
| Noisy Walmart-aisle audio, 19s | Clean | Clean, nearly identical | Badly garbled ("SCA", "GIMMY ONE HINNIT", "MAAR") |
| Loud exclamations, 16s | Clean | Clean | Minor repetition, still usable |
| Casual long vlog, 60s | Clean | Clean, nearly identical | Badly garbled multiple phrases |

`wav2vec2` (CTC-based, no language model) had no way to recover from
noisy/informal acoustic input the way Whisper's decoder-side language
modeling can — it just outputs its best per-frame guess, garbled or not.
LibriSpeech (even `test-other`) is still single-speaker, scripted,
studio-recorded audio; it never exercises this failure mode, which is
exactly why it missed this.

`whisper-small` matched or slightly refined every one of whisper-base's
real-clip transcripts — it fully inherits Whisper's robustness advantage
while also scoring meaningfully better than whisper-base on LibriSpeech
WER (0.079 vs 0.121). It's the strongest candidate found on both axes
that matter for this project: real-world robustness first, generic
accuracy second. See "Recommendation" below.

## Excluded: `facebook/mms-1b-all`

Tried on `test-other`, but abandoned mid-run and removed entirely
(script, cached weights, results) — not due to an accuracy or
architecture problem, but a hardware fit problem specific to this dev
machine: it's a 1-billion-parameter model, the largest tested, running on
an 8GB-RAM Apple Silicon Mac where GPU (MPS) and CPU share the same
unified memory pool. Alongside a live OS and running apps, the working
set didn't fit in RAM, causing heavy swapping (up to 6GB of 7GB
configured swap in use) and wildly inconsistent per-clip latency (some
clips under 1s, others 40-200s+) that made the benchmark impractical to
finish on this hardware. On the (now-deleted) `test-clean` run, finished
before this became a blocker, it scored WER 0.042 / CER 0.012 / 6.5s avg
inference — respectable accuracy, but already the slowest of the four
even without memory pressure. If revisited, it would need either more
RAM headroom or a CUDA machine rather than this setup.

## Recommendation — decided

**Production now runs `openai/whisper-small`**, wired into
`modules/voice_detection/voice_manipulation_scorer.py`. Reasoning, in
priority order for this project (Signal 3's manipulation scoring and
Signal 4's future coherence check are both entirely downstream of
transcript quality — a garbled transcript directly corrupts whatever's
built on top of it, regardless of how a model scores on a generic
benchmark):

1. **Real-world robustness** (most important): whisper-small matched
   whisper-base's clean, usable transcripts on all 4 real clips;
   wav2vec2 broke down on 3 of 4 despite winning LibriSpeech.
2. **Generic accuracy** (useful, not sufficient alone): whisper-small
   beats whisper-base's LibriSpeech WER by ~35% (0.079 vs 0.121).
3. **Inference time** (low priority here): Signal 3/4 run once per
   upload as part of a report, not live/real-time — whisper-small's
   extra ~1.4s/clip over whisper-base is a non-issue.
4. **Parameter count**: not a goal in itself — `mms-1b-all` was the
   largest model tested and neither the most accurate nor usable on this
   hardware, proof that bigger isn't automatically better.

This should still be treated as the best answer found so far, not a
permanently closed question — expanding the original 60-clip
`tests/model testing/Datasets/voice-manipulation/` production-domain dataset (see
`tests/model testing/testing/voice-manipulation/`) with more real examples would be
the highest-value next step for a more confident final answer, since
it's the closest proxy to actual traffic available.

## Output

`tests/model testing/Test-Results/audiotranscription/<model_key>/` per model
(`metrics.txt`, `metrics_summary.csv`, `predictions_<model>.csv`,
`wer_distribution.png`); `tests/model testing/Test-Results/audiotranscription/combined/`
for the cross-model comparison (`combined_metrics_summary.csv`,
`wer_comparison.png`, `inference_time_comparison.png`, `metrics.txt`).
