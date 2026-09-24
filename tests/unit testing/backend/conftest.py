"""Shared pytest fixtures for the backend unit test suite."""
import io
import os
import sys
import types

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


def _install_fake_module(module_path, **attrs):
    """Registers a fake module at `module_path` (e.g. 'modules.image_detection.image_scorer')
    in sys.modules.
    """
    fake = types.ModuleType(module_path)
    for name, value in attrs.items():
        setattr(fake, name, value)
    sys.modules[module_path] = fake
    return fake


class FakeCViTScorer:
    """Stands in for modules.video_detection.cvit_scorer.CViTScorer."""

    def __init__(self, device=None):
        pass

    def score_video_frames(self, frames):
        return 0.10, [0.10] * len(frames)

    def get_verdict(self, avg_score):
        return "Real" if avg_score < 0.5 else "Deepfake"


class FakeImageScorer:
    """Stands in for modules.image_detection.image_scorer.ImageScorer."""

    def __init__(self, device=None):
        pass

    def score_image(self, img):
        return 0.90, "Real"


class FakeVoiceCloneScorer:
    """Stands in for modules.voice_detection.voice_clone_scorer.VoiceCloneScorer."""

    def __init__(self, device=None):
        pass

    def score_array(self, audio_array):
        return 0.10, "Real"


class FakeVoiceManipulationScorer:
    """Stands in for modules.voice_detection.voice_manipulation_scorer.VoiceManipulationScorer."""

    def __init__(self, device=None):
        pass

    def analyze_array(self, audio_array):
        return {
            "transcript": "This is a fake transcript for testing.",
            "manipulation_score": 0.05,
            "verdict": "Non-manipulative",
            "emotions": {"neutral": 0.9, "joy": 0.1},
        }


class FakeCaptionCoherenceScorer:
    """Stands in for modules.caption_coherence.coherence_scorer.CaptionCoherenceScorer."""

    def __init__(self, device=None):
        pass

    def score(self, transcript, caption):
        transcript = (transcript or "").strip()
        caption = (caption or "").strip()
        if not transcript or not caption:
            return None, None
        return 0.80, "Coherent"


_install_fake_module("modules.video_detection.cvit_scorer", CViTScorer=FakeCViTScorer)
_install_fake_module("modules.image_detection.image_scorer", ImageScorer=FakeImageScorer)
_install_fake_module("modules.voice_detection.voice_clone_scorer", VoiceCloneScorer=FakeVoiceCloneScorer)
_install_fake_module("modules.voice_detection.voice_manipulation_scorer", VoiceManipulationScorer=FakeVoiceManipulationScorer)
_install_fake_module("modules.caption_coherence.coherence_scorer", CaptionCoherenceScorer=FakeCaptionCoherenceScorer)

# Now safe to import the real app - its eager model-loading block will
# construct our fakes above instead of loading real weights.
import app as flask_app_module  # noqa: E402


@pytest.fixture(scope="session")
def app():
    flask_app_module.app.config.update(TESTING=True)
    return flask_app_module.app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def isolated_usage_db(tmp_path, monkeypatch):
    """Points modules.usage_log.DB_PATH at a fresh temp file for the duration of one test, so
    tests never read/write the project's real usage_log.db.
    """
    import modules.usage_log as usage_log
    temp_db = tmp_path / "usage_log_test.db"
    monkeypatch.setattr(usage_log, "DB_PATH", str(temp_db))
    return usage_log


@pytest.fixture()
def small_wav_bytes():
    """A minimal, valid, silent WAV file (8kHz mono, ~0.1s) as raw bytes -
    small and fast, no fixture audio file needed on disk."""
    import numpy as np
    import soundfile as sf
    audio = np.zeros(800, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, 8000, format="WAV", subtype="PCM_16")
    return buf.getvalue()
