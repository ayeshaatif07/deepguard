"""Generates the charts for the performance testing suite from the .txt reports already
written to results/ by the five benchmark scripts.
"""
import os
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

GREEN = "#2ab38e"
RED = "#e74c3c"
BLUE = "#3a7ca5"
GREY = "#8d99ae"

# "<label>   mean=0.544s  median=0.544s  p95=0.550s  min=0.540s  max=0.550s"
STAT_RE = re.compile(
    r"^(?P<label>.*?)\s+mean=(?P<mean>[\d.]+)s\s+median=(?P<median>[\d.]+)s\s+"
    r"p95=(?P<p95>[\d.]+)s\s+min=(?P<min>[\d.]+)s\s+max=(?P<max>[\d.]+)s"
)


def wrap_label(name, width=16):
    """'Signal 3 (Whisper + fine-tuned RoBERTa)' -> a stacked, tick-safe label."""
    m = re.match(r"^(?P<head>.*?)\s*\((?P<tail>.*)\)\s*$", name)
    if not m:
        return textwrap.fill(name, width)
    return m.group("head") + "\n" + textwrap.fill(f"({m.group('tail')})", width)


def read(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return f.read()


def stat_rows(text):
    """Every 'mean=/median=/p95=' line in a report, in file order."""
    rows = []
    for line in text.splitlines():
        m = STAT_RE.match(line.strip())
        if m:
            rows.append({
                "label": m.group("label").strip(),
                "mean": float(m.group("mean")),
                "median": float(m.group("median")),
                "p95": float(m.group("p95")),
                "min": float(m.group("min")),
                "max": float(m.group("max")),
            })
    return rows


def save(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Written: {path}")


def annotate(ax, bars, values, fmt="{:.3f}s"):
    for bar, value in zip(bars, values):
        ax.annotate(fmt.format(value),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", va="bottom", fontsize=8)


# ---------------------------------------------------------------- per-model
def plot_per_model_latency():
    text = read("per_model_latency.txt")

    # Load-time block: "  Signal 4 (all-MiniLM-L6-v2)    7.53s"
    loads = []
    in_block = False
    for line in text.splitlines():
        if line.startswith("Model load time"):
            in_block = True
            continue
        if in_block:
            m = re.match(r"^\s{2,}(?P<label>\S.*?)\s+(?P<secs>[\d.]+)s\s*$", line)
            if m:
                loads.append((m.group("label").strip(), float(m.group("secs"))))
            elif not line.strip():
                if loads:
                    in_block = False

    rows = stat_rows(text)
    short = [r["label"].split(" - ")[0] + "\n" +
             re.sub(r"\s*\(.*\)", "", r["label"].split(" - ")[1]).strip()
             for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    names = [wrap_label(n) for n, _ in loads]
    secs = [s for _, s in loads]
    bars = axes[0].bar(names, secs, color=BLUE)
    annotate(axes[0], bars, secs, "{:.2f}s")
    axes[0].set_title("Model load time (one-time, per app startup)")
    axes[0].set_ylabel("Seconds")
    axes[0].set_ylim(0, max(secs) * 1.18)
    axes[0].tick_params(axis="x", labelsize=8)

    means = [r["mean"] for r in rows]
    p95s = [r["p95"] for r in rows]
    bars = axes[1].bar(short, means, color=GREEN, label="mean")
    axes[1].errorbar(short, means,
                     yerr=[[m - r["min"] for m, r in zip(means, rows)],
                           [p - m for p, m in zip(p95s, means)]],
                     fmt="none", ecolor=GREY, capsize=4, label="min - p95")
    axes[1].set_yscale("log")
    annotate(axes[1], bars, means)
    axes[1].set_title("Per-signal inference latency (log scale, n=5)")
    axes[1].set_ylabel("Seconds (log)")
    axes[1].legend(fontsize=8)
    axes[1].tick_params(axis="x", labelsize=8)

    save(fig, "per_model_latency.png")


# ----------------------------------------------------------------- pipeline
def plot_pipeline_latency():
    rows = stat_rows(read("pipeline_latency.txt"))
    labels = [r["label"].split(" - ")[0] + "\n" + r["label"].split(" - ")[1] for r in rows]
    means = [r["mean"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.bar(labels, means, color=[GREEN, BLUE], width=0.5)
    ax.errorbar(labels, means,
                yerr=[[m - r["min"] for m, r in zip(means, rows)],
                      [r["p95"] - m for m, r in zip(means, rows)]],
                fmt="none", ecolor=GREY, capsize=5, label="min - p95")
    for bar, r in zip(bars, rows):
        ax.annotate(f"{r['mean']:.3f}s",
                    (bar.get_x() + bar.get_width() / 2, r["p95"]),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", va="bottom", fontsize=9)
    ax.set_title("End-to-end request latency (Flask test client, n=5)")
    ax.set_ylabel("Seconds")
    ax.set_ylim(0, max(r["p95"] for r in rows) * 1.2)
    ax.legend(fontsize=8)
    save(fig, "pipeline_latency.png")


# --------------------------------------------------------------- concurrency
def plot_concurrent_load():
    text = read("concurrent_load.txt")
    levels, means, p95s, errors = [], [], [], []
    for line in text.splitlines():
        m = re.match(r"^Concurrency=(\d+)\s+(?P<rest>.*)$", line.strip())
        if not m:
            continue
        stats = STAT_RE.match("x " + m.group("rest"))
        levels.append(int(m.group(1)))
        means.append(float(stats.group("mean")))
        p95s.append(float(stats.group("p95")))
        err = re.search(r"errors=(\d+)/(\d+)", line)
        errors.append((int(err.group(1)), int(err.group(2))) if err else (0, 0))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(levels))
    ax.plot(x, means, marker="o", color=GREEN, linewidth=2, label="mean")
    ax.plot(x, p95s, marker="s", color=BLUE, linestyle="--", linewidth=2, label="p95")
    for i, (mean, p95) in enumerate(zip(means, p95s)):
        ax.annotate(f"{mean:.3f}s", (i, mean), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(c) for c in levels])
    ax.set_xlabel("Concurrent requests")
    ax.set_ylabel("Seconds")
    failed = sum(e[0] for e in errors)
    total = sum(e[1] for e in errors)
    ax.set_title(f"Latency under concurrent load — POST /analyze_image\n"
                 f"({failed}/{total} requests failed)", fontsize=11)
    if p95s and p95s[0] == max(p95s):
        ax.annotate("first request of the run\n(includes cold-start)",
                    (0, p95s[0]), textcoords="offset points", xytext=(24, -6),
                    ha="left", va="top", fontsize=8, color=GREY,
                    arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8))
    ax.set_ylim(0, max(p95s) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    save(fig, "concurrent_load.png")


# ------------------------------------------------------------------ database
def plot_database_scaling():
    text = read("database_scaling.txt")
    series = {}
    for line in text.splitlines():
        m = re.match(r"^\s*(?P<op>.+?) @\s*(?P<rows>[\d,]+) rows\s+(?P<rest>mean=.*)$", line)
        if not m:
            continue
        stats = STAT_RE.match("x " + m.group("rest"))
        op = m.group("op").strip()
        series.setdefault(op, []).append(
            (int(m.group("rows").replace(",", "")), float(stats.group("mean"))))

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {0: GREEN, 1: RED}
    for i, (op, points) in enumerate(series.items()):
        points.sort()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", linewidth=2, color=colors.get(i, BLUE), label=op)
        dy = 8 if i else -14
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}s", (x, y), textcoords="offset points",
                        xytext=(0, dy), ha="center", fontsize=8)
    ax.set_xscale("log")
    ax.margins(y=0.15)
    ax.set_xlabel("Rows in usage_log.db (log scale)")
    ax.set_ylabel("Mean seconds")
    ax.set_title("Database operation scaling with table size (n=10 per point)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    save(fig, "database_scaling.png")


# ------------------------------------------------------------------ resource
def plot_resource_usage():
    text = read("resource_usage.txt")
    phases = []
    current = None
    for line in text.splitlines():
        if line.startswith("Idle baseline"):
            current = {"phase": "Idle"}
            phases.append(current)
        elif line.startswith("Under load"):
            current = {"phase": "Under load\n(4-way concurrent)"}
            phases.append(current)
        elif current is not None:
            cpu = re.match(r"^\s+CPU avg=([\d.]+)%\s+max=([\d.]+)%", line)
            rss = re.match(r"^\s+RSS avg=(\d+) MB\s+max=(\d+) MB", line)
            if cpu:
                current["cpu_avg"], current["cpu_max"] = float(cpu.group(1)), float(cpu.group(2))
            if rss:
                current["rss_avg"], current["rss_max"] = int(rss.group(1)), int(rss.group(2))

    labels = [p["phase"] for p in phases]
    width = 0.35
    x = range(len(phases))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, (avg_key, max_key, title, unit, color) in zip(axes, [
        ("cpu_avg", "cpu_max", "CPU usage (per-process)", "%", GREEN),
        ("rss_avg", "rss_max", "Memory (resident set size)", "MB", BLUE),
    ]):
        avgs = [p[avg_key] for p in phases]
        maxs = [p[max_key] for p in phases]
        b1 = ax.bar([i - width / 2 for i in x], avgs, width, color=color, label="avg")
        b2 = ax.bar([i + width / 2 for i in x], maxs, width, color=GREY, label="max")
        fmt = "{:.1f}" + unit if unit == "%" else "{:.0f} " + unit
        annotate(ax, b1, avgs, fmt)
        annotate(ax, b2, maxs, fmt)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylabel(unit)
        ax.set_title(title)
        ax.set_ylim(0, max(maxs) * 1.2)
        ax.legend(fontsize=8)

    save(fig, "resource_usage.png")


def main():
    plot_per_model_latency()
    plot_pipeline_latency()
    plot_concurrent_load()
    plot_database_scaling()
    plot_resource_usage()


if __name__ == "__main__":
    main()
