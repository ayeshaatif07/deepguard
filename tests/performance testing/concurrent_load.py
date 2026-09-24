"""Fires simultaneous real HTTP requests at a live DeepGuard server and records latency and
error rate per concurrency level.
"""
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import summarize, fmt_row, write_report, SAMPLE_IMAGE

PORT = 5055
BASE_URL = f"http://127.0.0.1:{PORT}"
CONCURRENCY_LEVELS = [1, 2, 4, 8]
REQUESTS_PER_LEVEL = 8


def start_server():
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
            print("Server is up.")
            return
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    raise RuntimeError("Server did not start in time")


def one_request(image_bytes):
    start = time.perf_counter()
    try:
        resp = requests.post(
            BASE_URL + "/analyze_image",
            files={"image": ("sample.jpg", image_bytes, "image/jpeg")},
            timeout=120,
        )
        ok = resp.status_code in (200, 302)
    except requests.exceptions.RequestException:
        ok = False
    return time.perf_counter() - start, ok


def main():
    start_server()

    with open(SAMPLE_IMAGE, "rb") as f:
        image_bytes = f.read()

    lines = ["Concurrent Load Test", "=" * 60, "", f"Endpoint: POST /analyze_image", ""]

    for concurrency in CONCURRENCY_LEVELS:
        print(f"\nConcurrency level {concurrency}: firing {REQUESTS_PER_LEVEL} requests...")
        latencies = []
        errors = 0
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(one_request, image_bytes) for _ in range(REQUESTS_PER_LEVEL)]
            for fut in as_completed(futures):
                latency, ok = fut.result()
                latencies.append(latency)
                if not ok:
                    errors += 1

        stats = summarize(latencies)
        row = fmt_row(f"Concurrency={concurrency}", stats) + f"  errors={errors}/{REQUESTS_PER_LEVEL}"
        print(row)
        lines.append(row)

    lines.append("")
    lines.append("Notes:")
    lines.append("- Requests fired via Python's `requests` library against a live server")
    lines.append("  (Flask dev server, threaded=True), not the Flask test client, so real")
    lines.append("  socket-level concurrency and Python's GIL-bound inference cost are")
    lines.append("  both reflected in the numbers.")
    lines.append("- DeepGuard is designed as a single-user local tool (Section 1.0); this")
    lines.append("  test characterises degradation under unexpected simultaneous use, it")
    lines.append("  is not evidence toward a multi-user production capacity claim.")
    write_report("concurrent_load.txt", lines)


if __name__ == "__main__":
    main()
