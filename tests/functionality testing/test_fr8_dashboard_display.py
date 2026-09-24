"""
FR8: Display a modular verdict dashboard with per-signal breakdown
cards and a fusion bar.
"""
import io


def test_dashboard_shows_empty_state_with_no_signals_run(client):
    dash = client.get("/dashboard").get_data(as_text=True)
    assert "No signals run yet" in dash
    assert "Usage Statistics" in dash  # the all-time log table always renders


def test_dashboard_shows_a_card_per_active_signal(client, sample_video_bytes, sample_image_bytes):
    client.post(
        "/analyze", data={"video": (io.BytesIO(sample_video_bytes), "clip.mp4")},
        content_type="multipart/form-data",
    )
    client.post(
        "/analyze_image", data={"image": (io.BytesIO(sample_image_bytes), "photo.jpg")},
        content_type="multipart/form-data",
    )

    dash = client.get("/dashboard").get_data(as_text=True)
    # All 5 signal cards always render; an inactive one shows "Not run
    # yet". Exactly 2 signals ran here, so exactly 3 must still say that.
    assert dash.count('class="signal-card"') == 5
    assert dash.count("Not run yet") == 3
    assert "Based on 2 active signal" in dash
    # The overall fusion summary must be present alongside the per-signal cards
    assert 'id="dashOverallVerdict"' in dash
    assert 'id="dashOverallScore"' in dash


def test_dashboard_usage_statistics_reflect_logged_analyses(client, sample_image_bytes):
    client.post(
        "/analyze_image", data={"image": (io.BytesIO(sample_image_bytes), "photo.jpg")},
        content_type="multipart/form-data",
    )
    dash = client.get("/dashboard").get_data(as_text=True)
    assert "Usage Statistics" in dash
    assert "Image" in dash  # FEATURES list in modules/usage_log.py
