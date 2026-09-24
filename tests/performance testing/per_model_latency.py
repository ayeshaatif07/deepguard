"""Times each of the 5 signal models on a real sample file, calling each scorer's production
method directly.
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (
    timeit, fmt_row, write_report,
    SAMPLE_VIDEO, SAMPLE_IMAGE, SAMPLE_VOICE_CLONE_AUDIO,
    SAMPLE_MANIPULATION_AUDIO,
)

N_RUNS = 5


def time_load(label, factory):
    start = time.perf_counter()
    obj = factory()
    elapsed = time.perf_counter() - start
    print(f"  loaded {label} in {elapsed:.2f}s")
    return obj, elapsed


def main():
    lines = ["Per-Model Latency Benchmark", "=" * 60, ""]
    load_times = {}

    print("Loading models (one-time cost per signal)...")

    from modules.video_detection.cvit_scorer import CViTScorer
    from modules.frame_extractor import extract_frames
    cvit, load_times["Signal 1 (CViT2)"] = time_load("CViT2", CViTScorer)

    from modules.image_detection.image_scorer import ImageScorer
    from PIL import Image
    image_scorer, load_times["Signal 1b (prithivMLmods ViT)"] = time_load("ImageScorer", ImageScorer)

    from modules.voice_detection.voice_clone_scorer import VoiceCloneScorer
    clone_scorer, load_times["Signal 2 (fine-tuned AST)"] = time_load("VoiceCloneScorer", VoiceCloneScorer)

    from modules.voice_detection.voice_manipulation_scorer import VoiceManipulationScorer
    manip_scorer, load_times["Signal 3 (Whisper + fine-tuned RoBERTa)"] = time_load(
        "VoiceManipulationScorer", VoiceManipulationScorer)

    from modules.caption_coherence.coherence_scorer import CaptionCoherenceScorer
    coherence_scorer, load_times["Signal 4 (all-MiniLM-L6-v2)"] = time_load(
        "CaptionCoherenceScorer", CaptionCoherenceScorer)

    lines.append("Model load time (one-time, per app startup):")
    for label, secs in load_times.items():
        lines.append(f"  {label:<42} {secs:.2f}s")
    lines.append("")

    print("\nBenchmarking inference (real sample files, "
          f"{N_RUNS} timed runs each after 1 warmup run)...\n")

    # Signal 1 - Video Deepfake Detection (CViT2)
    frames_cache = {}
    def run_signal1():
        if "frames" not in frames_cache:
            frames_cache["frames"] = extract_frames(SAMPLE_VIDEO, num_frames=15)
        cvit.score_video_frames(list(frames_cache["frames"]))
    stats1 = timeit(run_signal1, n=N_RUNS)
    print(fmt_row("Signal 1 - Video Deepfake (CViT2)", stats1))
    lines.append(fmt_row("Signal 1 - Video Deepfake (CViT2)", stats1))

    # Signal 1b - Image Detection
    img = Image.open(SAMPLE_IMAGE)
    stats1b = timeit(lambda: image_scorer.score_image(img), n=N_RUNS)
    print(fmt_row("Signal 1b - Image Detection (prithivMLmods)", stats1b))
    lines.append(fmt_row("Signal 1b - Image Detection (prithivMLmods)", stats1b))

    # Signal 2 - Voice Clone Detection
    stats2 = timeit(lambda: clone_scorer.score_file(SAMPLE_VOICE_CLONE_AUDIO), n=N_RUNS)
    print(fmt_row("Signal 2 - Voice Clone (fine-tuned AST)", stats2))
    lines.append(fmt_row("Signal 2 - Voice Clone (fine-tuned AST)", stats2))

    # Signal 3 - Voice Manipulation Scoring (full: Whisper transcribe + classify)
    stats3 = timeit(lambda: manip_scorer.analyze(SAMPLE_MANIPULATION_AUDIO), n=N_RUNS)
    print(fmt_row("Signal 3 - Voice Manipulation (Whisper+RoBERTa)", stats3))
    lines.append(fmt_row("Signal 3 - Voice Manipulation (Whisper+RoBERTa)", stats3))

    # Signal 4 Caption-Content Coherence (embedding + cosine similarity only; transcript is
    # produced by Signal 3's shared Whisper step in the real pipeline, so it is reused here
    transcript = manip_scorer.transcribe(SAMPLE_MANIPULATION_AUDIO)
    caption = "A short clip discussing everyday topics."
    stats4 = timeit(lambda: coherence_scorer.score(transcript, caption), n=N_RUNS)
    print(fmt_row("Signal 4 - Caption Coherence (all-MiniLM-L6-v2)", stats4))
    lines.append(fmt_row("Signal 4 - Caption Coherence (all-MiniLM-L6-v2)", stats4))

    lines.append("")
    lines.append("Notes:")
    lines.append("- Signal 1 excludes frame extraction (ffmpeg decode), timed separately in pipeline_latency.py.")
    lines.append("- Signal 3's time includes its own Whisper-small transcription step (shared ASR + classification).")
    lines.append("- Signal 4's time is embedding + cosine similarity only, given an already-produced transcript.")
    lines.append(f"- All runs on {os.uname().machine}, {N_RUNS} timed repetitions after 1 discarded warmup run.")

    write_report("per_model_latency.txt", lines)


if __name__ == "__main__":
    main()
