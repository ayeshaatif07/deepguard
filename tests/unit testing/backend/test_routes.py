"""Flask route unit tests, using app.test_client()."""
import io

import pytest


# -------------------------------------------------------------------- /analyze_image rejects
# bad input before the model ever runs.

def test_analyze_image_missing_file(client):
    resp = client.post("/analyze_image", data={})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "No image file provided"


def test_analyze_image_empty_filename(client):
    data = {"image": (io.BytesIO(b"data"), "")}
    resp = client.post("/analyze_image", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "No selected file"


def test_analyze_image_unsupported_extension(client):
    data = {"image": (io.BytesIO(b"not really an image"), "notes.txt")}
    resp = client.post("/analyze_image", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.get_json()["error"]


# -------------------------------------------------------------------- /analyze_voice
# NO_AUDIO_TRACK is a specific, user-facing error, distinct from a generic 500.

def test_analyze_voice_no_audio_track(client, monkeypatch, small_wav_bytes):
    def _raise_no_audio_track(*args, **kwargs):
        raise RuntimeError("NO_AUDIO_TRACK")

    monkeypatch.setattr(
        "modules.voice_detection.audio_utils.extract_audio_array",
        _raise_no_audio_track,
    )

    data = {"audio": (io.BytesIO(small_wav_bytes), "clip.wav")}
    resp = client.post("/analyze_voice", data=data, content_type="multipart/form-data")

    assert resp.status_code == 400
    body = resp.get_json()
    assert body["error"] == "NO_AUDIO_TRACK"
    assert "no audio track" in body["message"]


def test_analyze_voice_unsupported_extension(client):
    data = {"audio": (io.BytesIO(b"data"), "clip.xyz")}
    resp = client.post("/analyze_voice", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.get_json()["error"]


# -------------------------------------------------------------------- /analyze_video_url and
# /analyze_url both fail fast on a bad URL, before any network access happens.

def test_analyze_video_url_invalid_url(client):
    resp = client.post("/analyze_video_url", json={"url": "not a url at all"})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "INVALID_URL"


def test_analyze_video_url_unsupported_platform(client):
    resp = client.post("/analyze_video_url", json={"url": "https://www.tiktok.com/@x/video/1"})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "UNSUPPORTED_PLATFORM"


def test_analyze_video_url_missing_url(client):
    resp = client.post("/analyze_video_url", json={})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "No URL provided"


def test_analyze_url_invalid_url(client):
    resp = client.post("/analyze_url", json={"url": "definitely not a url"})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "INVALID_URL"


# -------------------------------------------------------------------- /reset_dashboard and
# /reset_verdict_signals clear exactly the session keys they should, no more, no less.

SIGNAL_KEYS = (
    "signal1_result", "signal1b_result", "signal2_result",
    "signal4_result", "social_media_result",
)


def _seed_session(client, extra_key="some_unrelated_key", extra_value="untouched"):
    with client.session_transaction() as sess:
        for key in SIGNAL_KEYS:
            sess[key] = {"verdict": "Real", "score": 10}
        sess[extra_key] = extra_value


def test_reset_dashboard_clears_all_signal_keys(client):
    _seed_session(client)

    resp = client.get("/reset_dashboard")
    assert resp.status_code == 302  # redirects to /video

    with client.session_transaction() as sess:
        for key in SIGNAL_KEYS:
            assert key not in sess
        assert sess.get("some_unrelated_key") == "untouched"


def test_reset_verdict_signals_clears_session_and_returns_json(client):
    _seed_session(client)

    resp = client.post("/reset_verdict_signals")
    assert resp.status_code == 200
    assert resp.get_json() == {"ok": True}

    with client.session_transaction() as sess:
        for key in SIGNAL_KEYS:
            assert key not in sess


# -------------------------------------------------------------------- /download_report always
# returns a valid PDF, even with nothing tested yet in the session.

def test_download_report_with_empty_session(client):
    resp = client.get("/download_report")
    assert resp.status_code == 200
    assert resp.mimetype == "application/pdf"
    assert resp.data.startswith(b"%PDF")
