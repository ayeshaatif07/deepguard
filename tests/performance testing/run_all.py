"""Runs every performance benchmark in this folder as a separate process, then regenerates the
charts.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCRIPTS = [
    "database_scaling.py",     # fastest, no model loading
    "per_model_latency.py",    # loads models directly, no server
    "pipeline_latency.py",     # loads app.py, Flask test client
    "concurrent_load.py",      # loads app.py, live server on :5055
    "resource_usage.py",       # loads app.py, live server on :5056
]


def main():
    results = {}
    for script in SCRIPTS:
        print("\n" + "=" * 70)
        print(f"Running {script}")
        print("=" * 70)
        proc = subprocess.run([sys.executable, str(HERE / script)])
        results[script] = proc.returncode == 0

    print("\n" + "=" * 70)
    print("Generating charts from results/*.txt")
    print("=" * 70)
    proc = subprocess.run([sys.executable, str(HERE / "plot_results.py")])
    results["plot_results.py"] = proc.returncode == 0

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for script, ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'}  {script}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
