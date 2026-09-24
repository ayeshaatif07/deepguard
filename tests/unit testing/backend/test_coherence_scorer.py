"""Unit test for the real CaptionCoherenceScorer.score() empty-input branch, bypassing the
conftest fake.
"""
import importlib.util
import os

_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "modules", "caption_coherence", "coherence_scorer.py",
)
_spec = importlib.util.spec_from_file_location("coherence_scorer_real", _PATH)
_real_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_real_module)
CaptionCoherenceScorer = _real_module.CaptionCoherenceScorer


def _bare_scorer():
    return object.__new__(CaptionCoherenceScorer)


def test_score_returns_none_on_empty_transcript():
    scorer = _bare_scorer()
    score, verdict = scorer.score("", "some caption")
    assert score is None
    assert verdict is None


def test_score_returns_none_on_empty_caption():
    scorer = _bare_scorer()
    score, verdict = scorer.score("some transcript", "   ")
    assert score is None
    assert verdict is None


def test_score_returns_none_on_both_missing():
    scorer = _bare_scorer()
    score, verdict = scorer.score(None, None)
    assert score is None
    assert verdict is None
