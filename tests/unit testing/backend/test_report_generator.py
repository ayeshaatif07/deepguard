"""Unit tests for modules/report_generator.py's _compute_overall() pure fusion logic with no
I/O, no Flask, no model.
"""
from modules.report_generator import _compute_overall


def test_overall_picks_highest_concern_signal():
    signal1 = {"verdict": "Real", "score": 20}
    signal1b = {"verdict": "Artificial", "score": 45}
    signal2 = {
        "verdict": "Cloned", "score": 94,
        "manipulation_verdict": "Non-manipulative", "manipulation_score": 1,
    }
    signal4 = None

    result = _compute_overall(signal1, signal1b, signal2, signal4)

    assert result["verdict"] == "Cloned"
    assert result["score"] == 94
    assert result["active_count"] == 4  # signal1, signal1b, signal2's clone score, signal2's manipulation score


def test_overall_inverts_coherence_polarity():
    # A LOW coherence score (Incoherent) is the concerning direction, so it must be able to
    # win "worst" even against a middling video score, the same way verdict.html's own ranking
    signal1 = {"verdict": "Real", "score": 30}
    signal4 = {"verdict": "Incoherent", "score": 10}  # very incoherent -> very concerning

    result = _compute_overall(signal1, None, None, signal4)

    assert result["verdict"] == "Incoherent"
    assert result["score"] == 10  # the real coherence score is still displayed, not the inverted rank key
    assert result["active_count"] == 2


def test_overall_with_no_signals_returns_none():
    assert _compute_overall(None, None, None, None) is None


def test_overall_excludes_manipulation_when_not_present():
    # signal2 without a manipulation_verdict (e.g. NO_AUDIO_TRACK case)
    # must not add a phantom manipulation entry to the ranking.
    signal2 = {"verdict": "Real", "score": 5, "manipulation_verdict": None, "manipulation_score": None}
    result = _compute_overall(None, None, signal2, None)

    assert result["active_count"] == 1
    assert result["verdict"] == "Real"
