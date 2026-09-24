"""
FR1: Accept video upload (.mp4, .mov, .webm) and still image upload
(.jpg, .png, .webp).
"""
import io

from conftest import SAMPLE_IMAGE


def test_valid_video_upload_accepted(client, sample_video_bytes):
    data = {"video": (io.BytesIO(sample_video_bytes), "clip.mp4")}
    resp = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200


def test_valid_image_upload_accepted(client, sample_image_bytes):
    data = {"image": (io.BytesIO(sample_image_bytes), "photo.jpg")}
    resp = client.post("/analyze_image", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200


def test_unsupported_image_extension_rejected(client, sample_image_bytes):
    # Same real image bytes, but the filename claims an extension the app
    # doesn't accept for the image endpoint.
    data = {"image": (io.BytesIO(sample_image_bytes), "photo.txt")}
    resp = client.post("/analyze_image", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.get_json()["error"]


def test_missing_video_field_rejected(client):
    resp = client.post("/analyze", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "No video file provided"


def test_missing_image_field_rejected(client):
    resp = client.post("/analyze_image", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "No image file provided"


def test_empty_filename_rejected(client):
    data = {"video": (io.BytesIO(b""), "")}
    resp = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "No selected file"
