"""
FR9: Enable users to stop the pipeline at any signal and save the
current results to the dashboard.
"""
import io


def test_stopping_after_signal1_saves_only_that_result(client, sample_video_bytes):
    """A user who runs only the video check (and never continues to audio/caption) must still
    get a valid, self-consistent dashboard driven entirely by that one signal.
    """
    body = client.post(
        "/analyze", data={"video": (io.BytesIO(sample_video_bytes), "clip.mp4")},
        content_type="multipart/form-data",
    ).get_json()

    dash = client.get("/dashboard").get_data(as_text=True)

    # All 5 signal cards always render (see templates/verdict.html); an inactive one shows
    # "Not run yet" instead of a verdict.
    assert dash.count("Not run yet") == 4
    assert "Based on 1 active signal" in dash
    assert "Based on 1 active signals" not in dash  # singular grammar branch

    # With only one active signal, that signal IS the Overall verdict -
    # there's nothing else it could be compared against.
    assert f'id="dashOverallVerdict">Overall: {body["verdict"]}' in dash
    assert f'id="dashOverallScore">{body["score"]}%' in dash


def test_reset_dashboard_clears_all_saved_signals(client, sample_video_bytes):
    client.post(
        "/analyze", data={"video": (io.BytesIO(sample_video_bytes), "clip.mp4")},
        content_type="multipart/form-data",
    )
    assert "No signals run yet" not in client.get("/dashboard").get_data(as_text=True)

    reset_resp = client.get("/reset_dashboard", follow_redirects=False)
    assert reset_resp.status_code == 302  # redirects to /video

    assert "No signals run yet" in client.get("/dashboard").get_data(as_text=True)


def test_running_a_second_signal_after_stopping_adds_to_not_replaces_the_first(
    client, sample_video_bytes, sample_image_bytes
):
    """The user can come back later and add Signal 1b without losing the
    Signal 1 result already saved - both must coexist on the dashboard."""
    video_body = client.post(
        "/analyze", data={"video": (io.BytesIO(sample_video_bytes), "clip.mp4")},
        content_type="multipart/form-data",
    ).get_json()

    dash_after_first = client.get("/dashboard").get_data(as_text=True)
    assert dash_after_first.count("Not run yet") == 4
    assert f'id="dashSig1Status">{video_body["verdict"]}' in dash_after_first

    image_body = client.post(
        "/analyze_image", data={"image": (io.BytesIO(sample_image_bytes), "photo.jpg")},
        content_type="multipart/form-data",
    ).get_json()

    dash_after_second = client.get("/dashboard").get_data(as_text=True)
    assert dash_after_second.count("Not run yet") == 3
    # The first result must still be there, not overwritten by the second.
    assert f'id="dashSig1Status">{video_body["verdict"]}' in dash_after_second
    assert f'id="dashSig2Status">{image_body["verdict"]}' in dash_after_second
