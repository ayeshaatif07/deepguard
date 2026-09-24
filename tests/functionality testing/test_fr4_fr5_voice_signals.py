"""FR4: Analyse the raw audio waveform of a video for voice cloning artefacts (Signal 2)."""
import io


def test_voice_analysis_returns_well_formed_clone_and_manipulation_result(
    client, sample_voice_clone_bytes
):
    data = {"audio": (io.BytesIO(sample_voice_clone_bytes), "clip.flac")}
    resp = client.post("/analyze_voice", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200

    body = resp.get_json()

    # FR4 - Signal 2 (voice clone)
    assert body["verdict"] in ("Real", "Cloned")
    assert isinstance(body["score"], int)
    assert 0 <= body["score"] <= 100

    # FR5 - Signal 3 (manipulation), from the same request
    assert isinstance(body["transcript"], str)
    assert body["manipulation_verdict"] in ("Manipulative", "Non-manipulative")
    assert isinstance(body["manipulation_score"], int)
    assert 0 <= body["manipulation_score"] <= 100

    assert isinstance(body["waveform"], list) and len(body["waveform"]) > 0
    assert body["inference_time"] >= 0


def test_voice_analysis_transcribes_real_speech(client, sample_manipulation_audio_bytes):
    """Uses a real clip from the manipulation-scoring benchmark dataset, which is known to
    contain clear spoken English.
    """
    data = {"audio": (io.BytesIO(sample_manipulation_audio_bytes), "clip.mp3")}
    resp = client.post("/analyze_voice", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200

    body = resp.get_json()
    assert body["transcript"].strip() != ""


def test_voice_analysis_rejects_unsupported_extension(client, sample_voice_clone_bytes):
    data = {"audio": (io.BytesIO(sample_voice_clone_bytes), "clip.xyz")}
    resp = client.post("/analyze_voice", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.get_json()["error"]


def test_voice_analysis_persists_both_signals_for_dashboard(client, sample_voice_clone_bytes):
    data = {"audio": (io.BytesIO(sample_voice_clone_bytes), "clip.flac")}
    body = client.post(
        "/analyze_voice", data=data, content_type="multipart/form-data"
    ).get_json()

    dash = client.get("/dashboard").get_data(as_text=True)
    # dashSig3 = Signal 2 (voice clone), dashSig4 = Signal 3 (manipulation)
    assert f'id="dashSig3Status">{body["verdict"]}' in dash
    assert f'id="dashSig4Status">{body["manipulation_verdict"]}' in dash
