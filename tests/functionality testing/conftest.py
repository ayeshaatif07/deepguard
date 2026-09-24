"""Shared fixtures for the functionality (integration) test suite."""
import csv
import datetime
import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Maps each test file's stem to the functional requirement(s) it covers,
# matching README.md's table - used to label the generated results report.
_FR_BY_FILE = {
    "test_fr1_upload_validation": "FR1",
    "test_fr2_video_signal1": "FR2",
    "test_fr3_image_signal1b": "FR3",
    "test_fr4_fr5_voice_signals": "FR4/FR5",
    "test_fr6_caption_coherence": "FR6",
    "test_fr7_fusion_verdict": "FR7",
    "test_fr8_dashboard_display": "FR8",
    "test_fr9_partial_pipeline": "FR9",
}

DATASETS = os.path.join(PROJECT_ROOT, "tests", "model testing", "Datasets")

SAMPLE_VIDEO = os.path.join(DATASETS, "video-detection", "videos", "REAL", "youtube_033.mp4")
SAMPLE_IMAGE = os.path.join(DATASETS, "image-detection", "images", "Artificial", "0485_a1 (1435).jpg")
SAMPLE_VOICE_CLONE_AUDIO = os.path.join(DATASETS, "voice-clone", "audio", "real", "yt_0011_part_001.flac")
SAMPLE_MANIPULATION_AUDIO = os.path.join(DATASETS, "voice-manipulation", "audio", "CLIP_02_mani.mp3")
COHERENCE_CSV = os.path.join(DATASETS, "caption-coherence", "signal4_coherence.csv")


@pytest.fixture(scope="session")
def flask_app():
    """Imports the real app.py once for the whole test session (loads all 5 real models the
    expensive part).
    """
    import app as flask_app_module
    flask_app_module.app.config["TESTING"] = True
    return flask_app_module.app


@pytest.fixture
def client(flask_app):
    """A fresh test client (fresh cookie jar -> fresh session) per test."""
    return flask_app.test_client()


@pytest.fixture
def sample_video_bytes():
    with open(SAMPLE_VIDEO, "rb") as f:
        return f.read()


@pytest.fixture
def sample_image_bytes():
    with open(SAMPLE_IMAGE, "rb") as f:
        return f.read()


@pytest.fixture
def sample_voice_clone_bytes():
    with open(SAMPLE_VOICE_CLONE_AUDIO, "rb") as f:
        return f.read()


@pytest.fixture
def sample_manipulation_audio_bytes():
    with open(SAMPLE_MANIPULATION_AUDIO, "rb") as f:
        return f.read()


@pytest.fixture
def coherence_pair():
    """One labelled (transcript, caption) pair from the Signal 4 benchmark dataset."""
    with open(COHERENCE_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    row = next(r for r in rows if r["pair_id"] == "CLIP_01_coherent")
    return row["transcript"], row["caption"]


@pytest.fixture
def incoherent_pair():
    """One real (transcript, caption) pair labelled Incoherent (a
    cross-paired caption from a different clip)."""
    with open(COHERENCE_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    row = next(r for r in rows if r["pair_id"] == "CLIP_01_incoherent")
    return row["transcript"], row["caption"]


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_makereport(item, call):
    if call.when != "call":
        return
    outcome = "PASS" if call.excinfo is None else "FAIL"
    file_stem = os.path.splitext(os.path.basename(item.fspath))[0]
    fr = _FR_BY_FILE.get(file_stem, "?")
    item.session.config._fr_results = getattr(item.session.config, "_fr_results", [])
    item.session.config._fr_results.append((fr, file_stem, item.name, outcome))


def pytest_sessionfinish(session, exitstatus):
    results = getattr(session.config, "_fr_results", [])
    if not results:
        return

    lines = [
        "Functionality Testing Results",
        "=" * 60,
        f"Run at: {datetime.datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    passed = sum(1 for *_, outcome in results if outcome == "PASS")
    for fr, file_stem, test_name, outcome in results:
        lines.append(f"[{outcome:<4}] {fr:<8} {file_stem}::{test_name}")
    lines.append("")
    lines.append(f"Total: {passed}/{len(results)} passed")

    out_path = os.path.join(RESULTS_DIR, "functionality_testing.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWritten: {out_path}")
