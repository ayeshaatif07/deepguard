"""
FR2: Run video frames through a deepfake detection model and return a
confidence score (Signal 1).
"""
import io


def test_video_analysis_returns_well_formed_verdict(client, sample_video_bytes):
    data = {"video": (io.BytesIO(sample_video_bytes), "clip.mp4")}
    resp = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200

    body = resp.get_json()
    assert body["verdict"] in ("Real", "Deepfake")
    assert isinstance(body["score"], int)
    assert 0 <= body["score"] <= 100
    assert isinstance(body["frame_scores"], list) and len(body["frame_scores"]) > 0
    assert all(0.0 <= s <= 1.0 for s in body["frame_scores"])
    assert isinstance(body["inference_time"], (int, float)) and body["inference_time"] >= 0
    assert body["explanation"]
    assert body["filename"] == "clip.mp4"

# Dashboard persistence for this result is covered by
# test_fr9_partial_pipeline.py::test_stopping_after_signal1_saves_only_that_result, which
