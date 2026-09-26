# Signal 4 dataset — caption-content coherence

Hand-constructed for this project: the project owner supplied 15 real
social-media URLs (5 YouTube, 5 Instagram, 5 Facebook) known to have a
genuinely coherent transcript/caption pair - i.e. what's said in the
video actually matches what the post's own caption describes. The
mismatched (`Incoherent`) half of the dataset was generated
programmatically from those same 15 clips, not sourced separately - see
"How Incoherent pairs were built" below.

```
Tests/Datasets/caption-coherence/
└── signal4_coherence.csv   ← 28 rows: pair_id, clip_id, url, platform,
                                title, transcript, caption, label,
                                caption_source_clip_id
```

## Source clips

14 of the 15 supplied URLs were usable (5 YouTube, 4 Instagram, 5
Facebook) - one Instagram URL
(`instagram.com/p/DcRZk1GznhJ`) turned out to be a static image post, not
a video (`yt-dlp` reported "No video formats found"), so it has no
audio/transcript to build a pair from and was excluded.

For each usable URL, this project's own extraction pipeline was reused
directly - `modules/social_media/url_extractor.py`'s `download_video()`
for the video/caption, and `voice_manipulation_scorer.transcribe()`
(the same `openai/whisper-small` production model, see
`Tests/testing/audiotranscription/`) for the transcript - so this
dataset's transcripts reflect exactly what production would actually
produce, not a manually-typed idealized transcript.

## How `Incoherent` pairs were built

Every clip's real transcript is paired with a **different clip's real
caption**, via a derangement (a random permutation with no clip mapped to
itself) over the 14 clips, `random.seed(42)` for reproducibility -
matching this project's existing convention for reproducible random
sampling (e.g. the LibriSpeech benchmark subsets in
`Tests/testing/audiotranscription/`). `caption_source_clip_id` on each
`Incoherent` row records exactly which clip the mismatched caption came
from, so every pairing is traceable and auditable, not a black box.

This means every `Incoherent` example is built from two pieces of real,
unedited text (a real transcript, a real caption) - nothing was
synthetically written or edited to force a mismatch. The 14 source clips
span genuinely unrelated topics (an AI baby video, a Walmart skincare
demo, a sitcom clip, English-vocabulary lessons, travel vlogs, a cargo
ship story, jungle animal facts, etc.), so cross-pairing them produces
unambiguous topic mismatches, not borderline/ambiguous cases.

## Balance

**28 rows, perfectly balanced**: 14 `Coherent` / 14 `Incoherent` - one
real pair and one mismatched pair per source clip.

## Limitations

- Small (14 source clips) by deep-learning-benchmark standards, same
  caveat already noted for this project's other hand-built datasets
  (e.g. the 60-clip Signal 3 set) - results should be re-validated as
  more clips are added.
- All `Incoherent` examples come from cross-pairing within the same
  14-clip pool, not independently sourced mismatched pairs - the
  mismatches are real and unambiguous, but the set doesn't include
  "near-miss" pairs (captions that are topically close but subtly
  incorrect), which would be a harder and arguably more realistic
  negative-example category to add later.
- Instagram coverage is 4/5 rather than 5/5 (one supplied URL was an
  image post, not a video).

See `Tests/testing/caption-coherence/` (once built) for how this dataset
is used to benchmark candidate Signal 4 models.
