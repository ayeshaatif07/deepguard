# Functionality Testing

Black-box integration tests: each file maps to one functional
requirement from Section 3.1.1 of the report, and drives the real Flask
app end-to-end (real routes, real session/cookie handling, real
template rendering) against the **real, unmocked** models on real
sample media from `../model testing/Datasets/`.

This is deliberately different from `../unit testing/`, which mocks all
5 models to test routing/validation logic in isolation, fast. This
suite is slower (all 5 real models load once per test session, ~30-60s)
but is the only suite that actually proves a feature works end-to-end
for a real user - matching what `../performance testing/` does for
speed instead of correctness.

## Requirement -> test file mapping

| Requirement | File |
|---|---|
| FR1 - accept video/image uploads, reject unsupported types | `test_fr1_upload_validation.py` |
| FR2 - Signal 1 (video deepfake detection) | `test_fr2_video_signal1.py` |
| FR3 - Signal 1b (image detection) | `test_fr3_image_signal1b.py` |
| FR4 - Signal 2 (voice clone detection) | `test_fr4_fr5_voice_signals.py` |
| FR5 - Signal 3 (voice manipulation scoring) | `test_fr4_fr5_voice_signals.py` (same endpoint as FR4) |
| FR6 - Signal 4 (caption-content coherence) | `test_fr6_caption_coherence.py` |
| FR7 - fusion into one Overall verdict | `test_fr7_fusion_verdict.py` |
| FR8 - modular verdict dashboard | `test_fr8_dashboard_display.py` |
| FR9 - stop at any signal, save partial results | `test_fr9_partial_pipeline.py` |

FR7's tests independently recompute DeepGuard's real rank-based fusion
policy (whichever active signal is most concerning wins - see
`templates/verdict.html`, not a weighted average) from each signal's own
real response, then check the rendered dashboard agrees - this is a
genuine correctness check of the fusion logic, not just a smoke test.

FR2 and FR3 don't have their own "persists to dashboard" test - that
would just repeat a weaker version of assertions the FR9 tests already
make (`test_stopping_after_signal1_saves_only_that_result` and
`test_running_a_second_signal_after_stopping_adds_to_not_replaces_the_first`),
so dashboard persistence for Signal 1 and Signal 1b is verified there
instead.

## Running

```bash
cd "deepguard"
"venv/bin/python3" -m pytest "tests/functionality testing" -v
```

Must use `venv/bin/python3` specifically (has all real ML dependencies
installed - see `../performance testing/README.md` for why plain
`python3` on this machine doesn't).

## Output

A `conftest.py` hook writes `results/functionality_testing.txt` on every
run - every test name, its FR mapping, and pass/fail, plus a pass count
- matching the same `results/` convention `../performance testing/`
uses, so results can be pasted into the testing report without
re-running anything.
