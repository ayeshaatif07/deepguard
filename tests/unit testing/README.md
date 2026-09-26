# Unit Testing

Two independent suites, kept separate because they run in different
languages/environments. Both are distinct from `tests/model testing/testing/` (the
model-accuracy benchmarking suite elsewhere in this project) - these
test whether the *code* behaves correctly, not whether a model is
accurate.

## Backend (Python / pytest)

```
cd "backend"
../../venv/bin/python3 -m pytest -v
```

Fast (a few seconds), no GPU, no real model weights loaded - `conftest.py`
installs lightweight fakes for all 5 ML model classes into `sys.modules`
*before* `app.py` is imported, so `app.py`'s own eager model-loading at
startup constructs fakes instead of loading real weights.

36 tests across:
- `test_usage_log.py` - the usage-statistics SQLite log
- `test_report_generator.py` - the PDF report's fusion/ranking logic
- `test_coherence_scorer.py` - Signal 4's empty-input handling (loaded
  directly from disk, bypassing the fake, to test the real class)
- `test_audio_utils.py` - waveform peaks and playable-audio encoding
- `test_url_extractor.py` - platform detection from a pasted URL
- `test_routes.py` - Flask route validation, error paths, and session handling

## Frontend (JavaScript / Jest + jsdom)

```
cd "frontend"
npm install   # first time only
npm test
```

24 tests covering `static/js/app.js`'s DOM/browser-facing functions
(`chipDotClassForVerdict`, `isGenuinePageReload`, `restoreStoredState`,
`renderWaveform`, `seekWaveform`, `showCoherenceResult`,
`resetDashboard`). These are the only functions app.js exports via a
`typeof module !== 'undefined'` guard at the bottom of the file - a
no-op in the browser, active only under Jest/Node. Functions that call
`fetch()` are intentionally not covered here; they need a real
server and belong to a future end-to-end test pass instead.
