"""Shared helpers for the performance testing suite: timing utilities, summary statistics, and
paths to real sample files already used by the model testing suite.
"""
import os
import statistics
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASETS = os.path.join(PROJECT_ROOT, "tests", "model testing", "Datasets")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SAMPLE_VIDEO = os.path.join(DATASETS, "video-detection", "videos", "REAL", "youtube_033.mp4")
SAMPLE_IMAGE = os.path.join(DATASETS, "image-detection", "images", "Artificial", "0485_a1 (1435).jpg")
SAMPLE_VOICE_CLONE_AUDIO = os.path.join(DATASETS, "voice-clone", "audio", "real", "yt_0011_part_001.flac")
SAMPLE_MANIPULATION_AUDIO = os.path.join(DATASETS, "voice-manipulation", "audio", "CLIP_02_mani.mp3")
SAMPLE_CAPTION_VIDEO = os.path.join(DATASETS, "caption-coherence", "videos", "JeXsKCySpGI.mp4")


def timeit(fn, n=5, warmup=1):
    """Runs fn() `warmup` times (discarded, lets lazy caches/JIT settle), then times it `n`
    times.
    """
    for _ in range(warmup):
        fn()
    runs = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        runs.append(time.perf_counter() - start)
    return summarize(runs)


def summarize(runs):
    runs_sorted = sorted(runs)
    p95_idx = min(len(runs_sorted) - 1, int(round(0.95 * (len(runs_sorted) - 1))))
    return {
        "n": len(runs),
        "mean_s": statistics.mean(runs),
        "median_s": statistics.median(runs),
        "min_s": min(runs),
        "max_s": max(runs),
        "p95_s": runs_sorted[p95_idx],
        "stdev_s": statistics.stdev(runs) if len(runs) > 1 else 0.0,
        "raw_s": runs,
    }


def fmt_row(label, stats):
    return (f"{label:<40} mean={stats['mean_s']:.3f}s  median={stats['median_s']:.3f}s  "
            f"p95={stats['p95_s']:.3f}s  min={stats['min_s']:.3f}s  max={stats['max_s']:.3f}s  "
            f"(n={stats['n']})")


def write_report(filename, lines):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWritten: {path}")
    return path
