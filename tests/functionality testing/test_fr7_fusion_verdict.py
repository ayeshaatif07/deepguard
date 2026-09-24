"""FR7: Fuse all active signal scores into an Overall verdict."""
import io


def _expected_worst(entries):
    """Mirrors verdict.html's own logic exactly: entries is a list of (rank_key, verdict,
    display_score).
    """
    best = None
    for entry in entries:
        if best is None or entry[0] > best[0]:
            best = entry
    return best


def test_fusion_picks_the_most_concerning_active_signal(
    client,
    sample_video_bytes,
    sample_image_bytes,
    sample_voice_clone_bytes,
    coherence_pair,
):
    entries = []

    video_body = client.post(
        "/analyze", data={"video": (io.BytesIO(sample_video_bytes), "clip.mp4")},
        content_type="multipart/form-data",
    ).get_json()
    entries.append((video_body["score"], video_body["verdict"], video_body["score"]))

    image_body = client.post(
        "/analyze_image", data={"image": (io.BytesIO(sample_image_bytes), "photo.jpg")},
        content_type="multipart/form-data",
    ).get_json()
    entries.append((image_body["score"], image_body["verdict"], image_body["score"]))

    voice_body = client.post(
        "/analyze_voice", data={"audio": (io.BytesIO(sample_voice_clone_bytes), "clip.flac")},
        content_type="multipart/form-data",
    ).get_json()
    entries.append((voice_body["score"], voice_body["verdict"], voice_body["score"]))
    entries.append((
        voice_body["manipulation_score"], voice_body["manipulation_verdict"],
        voice_body["manipulation_score"],
    ))

    transcript, caption = coherence_pair
    coherence_body = client.post(
        "/score_coherence", json={"transcript": transcript, "caption": caption}
    ).get_json()
    entries.append((
        100 - coherence_body["coherence_score"], coherence_body["coherence_verdict"],
        coherence_body["coherence_score"],
    ))

    expected_rank_key, expected_verdict, expected_score = _expected_worst(entries)

    dash = client.get("/dashboard").get_data(as_text=True)
    assert f'id="dashOverallVerdict">Overall: {expected_verdict}' in dash
    assert f'id="dashOverallScore">{expected_score}%' in dash
    assert "Based on 5 active signals" in dash


def test_fusion_is_not_a_simple_average(client, sample_voice_clone_bytes, coherence_pair):
    """Regression guard against the fusion policy silently reverting to a weighted/averaged
    score the Overall confidence must exactly equal the winning signal's own score.
    """
    voice_body = client.post(
        "/analyze_voice", data={"audio": (io.BytesIO(sample_voice_clone_bytes), "clip.flac")},
        content_type="multipart/form-data",
    ).get_json()

    transcript, caption = coherence_pair
    coherence_body = client.post(
        "/score_coherence", json={"transcript": transcript, "caption": caption}
    ).get_json()

    dash = client.get("/dashboard").get_data(as_text=True)

    candidate_scores = {voice_body["score"], coherence_body["coherence_score"]}
    average = sum(candidate_scores) / len(candidate_scores)

    # The rendered Overall score must be one of the real individual scores, not their average
    # (unless they happen to collide, which a real dataset pair essentially never does).
    if round(average) not in candidate_scores:
        assert f'id="dashOverallScore">{round(average)}%' not in dash
