"""Samples this process's CPU% and RAM (RSS) while it is idle (models loaded, no requests)
and while it is under the same concurrent load as concurrent_load.py, using psutil.
"""
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import psutil
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import write_report, SAMPLE_IMAGE

PORT = 5056
BASE_URL = f"http://127.0.0.1:{PORT}"
CONCURRENCY = 4
N_REQUESTS = 16
SAMPLE_INTERVAL_S = 0.2


class Sampler:
    def __init__(self, proc):
        self.proc = proc
        self.samples = []  # list of (cpu_percent, rss_mb)
        self._stop = threading.Event()

    def _run(self):
        self.proc.cpu_percent(interval=None)  # prime the internal counter
        while not self._stop.is_set():
            cpu = self.proc.cpu_percent(interval=SAMPLE_INTERVAL_S)
            rss_mb = self.proc.memory_info().rss / (1024 * 1024)
            self.samples.append((cpu, rss_mb))

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=SAMPLE_INTERVAL_S * 3)

    def summary(self):
        if not self.samples:
            return {"cpu_avg": 0.0, "cpu_max": 0.0, "rss_avg_mb": 0.0, "rss_max_mb": 0.0}
        cpus = [s[0] for s in self.samples]
        rss = [s[1] for s in self.samples]
        return {
            "cpu_avg": sum(cpus) / len(cpus),
            "cpu_max": max(cpus),
            "rss_avg_mb": sum(rss) / len(rss),
            "rss_max_mb": max(rss),
        }


def one_request(image_bytes):
    try:
        requests.post(
            BASE_URL + "/analyze_image",
            files={"image": ("sample.jpg", image_bytes, "image/jpeg")},
            timeout=120,
        )
    except requests.exceptions.RequestException:
        pass


def main():
    print("Importing app.py and starting a live server thread (loads all 5 real models)...")
    import app as flask_app_module
    thread = threading.Thread(
        target=lambda: flask_app_module.app.run(
            host="127.0.0.1", port=PORT, threaded=True, use_reloader=False, debug=False,
        ),
        daemon=True,
    )
    thread.start()
    for _ in range(60):
        try:
            requests.get(BASE_URL + "/", timeout=1)
            break
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    else:
        raise RuntimeError("Server did not start in time")
    print("Server is up.")

    proc = psutil.Process(os.getpid())

    print("\nSampling idle baseline (models loaded, no requests, 5s)...")
    idle_sampler = Sampler(proc)
    idle_sampler.start()
    time.sleep(5)
    idle_sampler.stop()
    idle_stats = idle_sampler.summary()

    with open(SAMPLE_IMAGE, "rb") as f:
        image_bytes = f.read()

    print(f"\nSampling under load ({CONCURRENCY}-way concurrent, {N_REQUESTS} requests)...")
    load_sampler = Sampler(proc)
    load_sampler.start()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = [pool.submit(one_request, image_bytes) for _ in range(N_REQUESTS)]
        for fut in futures:
            fut.result()
    load_sampler.stop()
    load_stats = load_sampler.summary()

    lines = [
        "Resource Usage Under Load (CPU / RAM)",
        "=" * 60,
        "",
        f"Idle baseline (models loaded, no active requests, {5}s sample window):",
        f"  CPU avg={idle_stats['cpu_avg']:.1f}%  max={idle_stats['cpu_max']:.1f}%",
        f"  RSS avg={idle_stats['rss_avg_mb']:.0f} MB  max={idle_stats['rss_max_mb']:.0f} MB",
        "",
        f"Under load ({CONCURRENCY}-way concurrent, {N_REQUESTS} requests to /analyze_image):",
        f"  CPU avg={load_stats['cpu_avg']:.1f}%  max={load_stats['cpu_max']:.1f}%",
        f"  RSS avg={load_stats['rss_avg_mb']:.0f} MB  max={load_stats['rss_max_mb']:.0f} MB",
        "",
        "Notes:",
        "- CPU% is per-process (psutil), can exceed 100% on multi-core machines",
        "  when PyTorch parallelises inference across threads.",
        "- RSS reflects the same process's memory holding all 5 loaded models",
        "  simultaneously (DeepGuard's actual startup behaviour - see app.py),",
        "  not just the one signal being exercised by the load requests.",
    ]
    print("\n" + "\n".join(lines[-9:]))
    write_report("resource_usage.txt", lines)


if __name__ == "__main__":
    main()
