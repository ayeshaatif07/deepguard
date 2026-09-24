"""FR3: Classify a still image as Real or AI-generated with a confidence score (Signal 1b)."""
import io


def test_image_analysis_returns_well_formed_verdict(client, sample_image_bytes):
    data = {"image": (io.BytesIO(sample_image_bytes), "photo.jpg")}
    resp = client.post("/analyze_image", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200

    body = resp.get_json()
    assert body["verdict"] in ("Real", "Artificial", "Deepfake")
    assert isinstance(body["score"], int)
    assert 0 <= body["score"] <= 100
    assert body["model_label"] == body["verdict"]
    assert body["explanation"]
    assert body["filename"] == "photo.jpg"

# Dashboard persistence for this result is covered by test_fr9_partial_pipeline.py::test_runni
# ng_a_second_signal_after_stopping_adds_to_not_replaces_the_first, which checks the same
