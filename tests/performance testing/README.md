# Performance Testing

Sits alongside `model testing/` (per-model accuracy benchmarks) and
`unit testing/` (functional correctness) under `tests/`. This suite
measures **how fast and how resource-hungry** DeepGuard is, not whether
its verdicts are correct.

## What's covered

| Script | Measures |
|---|---|
| `per_model_latency.py` | Per-signal inference time, for each of the 5 real models on a real sample file, plus one-time model load time |
| `pipeline_latency.py` | End-to-end request latency through Flask's real routing/upload/response handling for the image path (`/analyze_image`) and video path (`/analyze`) |
| `concurrent_load.py` | Latency and error rate at increasing concurrency (1/2/4/8 simultaneous requests) against a live server |
| `resource_usage.py` | Process CPU% and RSS memory, idle (models loaded, no requests) vs. under concurrent load |
| `database_scaling.py` | `usage_log.db` insert and dashboard-query (`get_usage_stats`) time as the table grows from 100 to 100,000 rows |

## Why these five

DeepGuard is a standalone, single-user local tool (Section 1.0) with no
cloud API and no expected concurrent-user base, so the priorities differ
from a typical server-hosted app's performance test plan:

- **Per-model and end-to-end latency** matter most directly - they're
  what a user actually experiences per analysis.
- **Concurrent load and resource usage** are tested to characterise
  degradation under *unexpected* simultaneous use (e.g. two browser tabs
  open at once), not to validate a multi-user production capacity
  target that was never a design goal.
- **Database scaling** matters because `usage_log.db` grows for as long
  as the tool is used and is read on every dashboard load, unlike the
  per-analysis media files which are transient.

Throughput/requests-per-second load testing in the sense used for
server-hosted web apps was deliberately left out for the same reason.

## Running

```bash
cd "deepguard"
python "tests/performance testing/run_all.py"
```

Or run any script individually. All of them are runnable from the
project root and resolve their own paths relative to `__file__`, so the
current working directory doesn't matter as long as they're invoked with
`python "tests/performance testing/<script>.py"`.

`per_model_latency.py`, `pipeline_latency.py`, `concurrent_load.py`, and
`resource_usage.py` all load the 5 real production models (no mocks),
so each takes noticeably longer to start than the mocked unit test
suite in `../unit testing/`. `database_scaling.py` does not load any
model and runs quickly.

## Charts

`plot_results.py` turns the saved `.txt` reports into the five PNG
charts in `results/`, one per benchmark. It parses the reports rather
than re-running anything, so the figures always show exactly the numbers
the reports quote, and regenerating them is instant:

```bash
python "tests/performance testing/plot_results.py"
```

`run_all.py` calls it automatically as a final step, so a full suite run
refreshes both the reports and the charts.

## Output

Every script writes its own `.txt` report to `results/`, alongside the
console output, so results can be pasted straight into the testing
report chapter without re-running anything. `plot_results.py` then
writes a matching `.png` chart per report:

| Report | Chart |
|---|---|
| `per_model_latency.txt` | `per_model_latency.png` — model load time, and per-signal inference latency on a log scale (the 0.010s-3.764s spread needs one) |
| `pipeline_latency.txt` | `pipeline_latency.png` — image vs. video request latency, min-p95 whiskers |
| `concurrent_load.txt` | `concurrent_load.png` — mean and p95 against concurrency, annotated with the failure count |
| `database_scaling.txt` | `database_scaling.png` — insert vs. `get_usage_stats()` against row count (log x) |
| `resource_usage.txt` | `resource_usage.png` — CPU% and RSS, idle vs. under load |
