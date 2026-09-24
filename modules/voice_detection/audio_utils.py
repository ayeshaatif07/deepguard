"""Decodes any audio or video upload to a mono 16kHz float32 waveform via ffmpeg.
Also builds the waveform peaks and playable audio the voice page renders.
"""

import base64
import io
import subprocess
import shutil
import numpy as np
import soundfile as sf

TARGET_SR = 16000

# ~4MB cap on the embedded clip, to stay within the browser's sessionStorage quota.
MAX_PLAYBACK_BASE64_BYTES = 4_000_000


def _ffmpeg_available():
    return shutil.which("ffmpeg") is not None


def extract_audio_array(input_path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """Decodes any audio/video file into a mono float32 waveform at target_sr.
    Raises RuntimeError if the file has no audio track.
    """
    if not _ffmpeg_available():
        raise RuntimeError(
            "ffmpeg is not installed or not on PATH - required to decode "
            "audio from uploaded files. Install it with `brew install ffmpeg`."
        )

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn",                      # drop any video stream, audio only
        "-ac", "1",                 # mono
        "-ar", str(target_sr),      # resample
        "-f", "wav",
        "-loglevel", "error",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        stderr = proc.stderr.decode("utf-8", errors="ignore")
        if "does not contain any stream" in stderr or "Output file is empty" in stderr or not proc.stdout:
            raise RuntimeError("NO_AUDIO_TRACK")
        raise RuntimeError(f"ffmpeg failed to decode audio: {stderr[:500]}")

    import io
    audio, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio


def encode_playable_audio(audio: np.ndarray, sample_rate: int = TARGET_SR):
    """Encodes the decoded waveform as a WAV data URI for in-page playback.
    Returns None if the clip would exceed MAX_PLAYBACK_BASE64_BYTES.
    """
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    if len(encoded) > MAX_PLAYBACK_BASE64_BYTES:
        return None
    return f"data:audio/wav;base64,{encoded}"


def compute_waveform_peaks(audio: np.ndarray, num_points: int = 100) -> list:
    """Downsamples the waveform into `num_points` peak amplitudes (0-1) for the waveform display."""
    if audio.size == 0:
        return [0.0] * num_points

    chunks = np.array_split(audio, num_points)
    peaks = np.array([np.abs(chunk).max() if chunk.size else 0.0 for chunk in chunks])
    max_peak = peaks.max()
    if max_peak > 0:
        peaks = peaks / max_peak
    return [round(float(p), 4) for p in peaks]
