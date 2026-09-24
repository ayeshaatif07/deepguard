"""Unit tests for modules/voice_detection/audio_utils.py's pure, model-free helpers
compute_waveform_peaks() and encode_playable_audio().
"""
import numpy as np
import pytest

from modules.voice_detection.audio_utils import compute_waveform_peaks, encode_playable_audio


def test_compute_waveform_peaks_silence_is_all_zero():
    silent = np.zeros(16000, dtype=np.float32)
    peaks = compute_waveform_peaks(silent, num_points=100)

    assert len(peaks) == 100
    assert all(p == 0.0 for p in peaks)


def test_compute_waveform_peaks_normalizes_to_loudest_point():
    audio = np.zeros(1000, dtype=np.float32)
    audio[500] = 1.0  # single loud spike, everything else silent
    peaks = compute_waveform_peaks(audio, num_points=10)

    assert len(peaks) == 10
    assert max(peaks) == 1.0  # normalized so the loudest chunk hits exactly 1.0
    assert min(peaks) == 0.0


def test_compute_waveform_peaks_empty_array():
    peaks = compute_waveform_peaks(np.array([], dtype=np.float32), num_points=100)
    assert peaks == [0.0] * 100


def test_encode_playable_audio_returns_data_url_for_short_clip():
    short_clip = np.zeros(8000, dtype=np.float32)  # 0.5s at 16kHz - tiny
    result = encode_playable_audio(short_clip, sample_rate=16000)

    assert result is not None
    assert result.startswith("data:audio/wav;base64,")


def test_encode_playable_audio_returns_none_over_size_cap():
    import modules.voice_detection.audio_utils as audio_utils
    # A clip that would encode to well over the cap - use a tiny cap
    # instead of generating minutes of real audio, to keep this test fast.
    long_clip = np.zeros(16000, dtype=np.float32)  # 1 second
    original_cap = audio_utils.MAX_PLAYBACK_BASE64_BYTES
    audio_utils.MAX_PLAYBACK_BASE64_BYTES = 100  # unrealistically small, forces the cap to trigger
    try:
        result = encode_playable_audio(long_clip, sample_rate=16000)
    finally:
        audio_utils.MAX_PLAYBACK_BASE64_BYTES = original_cap

    assert result is None
