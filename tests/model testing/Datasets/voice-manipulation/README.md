# Signal 3 dataset — voice manipulation

Hand-constructed by the project owner (not downloaded from an external
source, unlike the video/image/voice-clone datasets), delivered as
`dataset_voice manipulation.zip` and extracted here.

```
Tests/Datasets/voice-manipulation/
├── signal3_transcripts_final.csv   ← 60 rows: clip_id, audio_filename, transcript, label, manipulation_tactics
└── audio/                          ← 60 matching .mp3 files
```

- **60 clips, perfectly balanced**: 30 `manipulative` / 30 `non-manipulative`.
- Every row has real audio (a matching `.mp3` in `audio/`), a transcript,
  a binary label, and a human-annotated `manipulation_tactics` column (18
  distinct tags across the dataset: urgency, authority, fear, guilt, FOMO,
  trust exploitation, legal threat, isolation, false reward, etc.) —
  richer than a flat binary label, useful for future tactic-level
  evaluation beyond the current manipulative/non-manipulative benchmark.

See `Tests/testing/voice-manipulation/README.md` for how this dataset is
used to benchmark candidate models.
