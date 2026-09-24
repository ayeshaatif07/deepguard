"""End-to-end request latency through Flask's own routing, upload parsing and session handling."""
import io
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import timeit, fmt_row, write_report, SAMPLE_VIDEO, SAMPLE_IMAGE

N_RUNS = 5


def main():
    print("Importing app.py (loads all 5 real models eagerly - this takes a while)...")
    import app as flask_app_module
    client = flask_app_module.app.test_client()

    with open(SAMPLE_IMAGE, "rb") as f:
        image_bytes = f.read()
    with open(SAMPLE_VIDEO, "rb") as f:
        video_bytes = f.read()

    def post_image():
        data = {"image": (io.BytesIO(image_bytes), "sample.jpg")}
        resp = client.post("/analyze_image", data=data, content_type="multipart/form-data")
        assert resp.status_code in (200, 302), resp.status_code

    def post_video():
        data = {"video": (io.BytesIO(video_bytes), "sample.mp4")}
        resp = client.post("/analyze", data=data, content_type="multipart/form-data")
        assert resp.status_code in (200, 302), resp.status_code

    print(f"\nTiming end-to-end requests ({N_RUNS} runs each, 1 warmup)...\n")

    stats_image = timeit(post_image, n=N_RUNS)
    print(fmt_row("Image path - POST /analyze_image", stats_image))

    stats_video = timeit(post_video, n=N_RUNS)
    print(fmt_row("Video path - POST /analyze", stats_video))

    lines = [
        "End-to-End Pipeline Latency Benchmark",
        "=" * 60,
        "",
        fmt_row("Image path - POST /analyze_image (Signal 1b only)", stats_image),
        fmt_row("Video path - POST /analyze (Signal 1, full request)", stats_video),
        "",
        "Notes:",
        "- Measured via Flask's own test_client(), which exercises real routing,",
        "  file-upload parsing, session handling, and template rendering around",
        "  the real (non-mocked) model inference - not just the model call itself.",
        "- POST /analyze covers Signal 1 (video deepfake detection) only; the",
        "  audio/caption continuation (Signals 2-4) is a separate user action",
        "  in the real UI (see the branch points in Section 3.5 of the report),",
        "  so it is not included in a single /analyze timing.",
        f"- {N_RUNS} timed repetitions after 1 discarded warmup run.",
    ]
    write_report("pipeline_latency.txt", lines)


if __name__ == "__main__":
    main()
