"""
FR6: Compare the spoken transcript to user-supplied caption and hashtags
for semantic incoherence (Signal 4).
"""


def test_coherence_scoring_returns_well_formed_result(client, coherence_pair):
    transcript, caption = coherence_pair
    resp = client.post("/score_coherence", json={"transcript": transcript, "caption": caption})
    assert resp.status_code == 200

    body = resp.get_json()
    assert body["coherence_implemented"] is True
    assert body["coherence_verdict"] in ("Coherent", "Incoherent")
    assert isinstance(body["coherence_score"], int)
    assert 0 <= body["coherence_score"] <= 100


def test_coherence_scoring_handles_missing_transcript_gracefully(client):
    """FR6 must degrade gracefully (not error) when there's nothing to
    compare - e.g. a locally-uploaded file with no post caption."""
    resp = client.post("/score_coherence", json={"transcript": "", "caption": "some caption"})
    assert resp.status_code == 200

    body = resp.get_json()
    assert body["coherence_verdict"] is None
    assert body["coherence_score"] is None
    assert body["coherence_message"]


def test_coherence_scoring_persists_result_for_dashboard(client, incoherent_pair):
    transcript, caption = incoherent_pair
    body = client.post(
        "/score_coherence", json={"transcript": transcript, "caption": caption}
    ).get_json()

    dash = client.get("/dashboard").get_data(as_text=True)
    # dashSig5 = Signal 4 (caption coherence)
    assert f'id="dashSig5Status">{body["coherence_verdict"]}' in dash
