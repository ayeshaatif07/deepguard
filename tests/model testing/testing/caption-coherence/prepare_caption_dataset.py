"""Builds the caption-content coherence dataset used to benchmark the Signal 4 candidates."""

import os
import sys
import csv

TESTING_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(TESTING_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from modules.social_media.youtube_extractor import extract
from modules.voice_detection.voice_manipulation_scorer import VoiceManipulationScorer

# (video_id, transcript_reliable) reliability determined by direct inspection of the
# transcript against the actual video content; see README.md for the reasoning per video.
VIDEOS = [
    ("_uNMZxAOyPw", True),   # kitten playing with yarn - genuine narration
    ("JeXsKCySpGI", False),  # hot coffee tutorial - background music caused a garbled/hallucinated transcript
    ("d52MZJw-hIE", True),   # guitar chords lesson - genuine narration
    ("hovkpjJK8TE", True),   # weather forecast (Urdu) - genuine speech, correctly transcribed non-English
    ("dU0Kn1V0UWM", False),  # pancake recipe - hallucinated number-counting over music
    ("c2OTHeCKsBE", False),  # dogs shopping cart - hallucinated, no real narration
    ("MaRo7Bi4jhY", False),  # car passing sound effect - no real speech at all
    ("awd9l1hKGLs", True),   # LeBron dunk - genuine sports commentary
    ("IAnN47PRTAw", False),  # rain sound (30s) - ambient only, no real speech
    ("1_Cruk3wJ_c", False),  # volcano eruption timelapse - ambient only, no real speech
]

DATASET_DIR = os.path.join(PROJECT_ROOT, "Tests", "Datasets", "caption-coherence")
VIDEOS_DIR = os.path.join(DATASET_DIR, "videos")


def main():
    print("Loading VoiceManipulationScorer (for its Whisper transcription only)...")
    scorer = VoiceManipulationScorer()

    rows = []
    for vid, reliable in VIDEOS:
        url = f"https://www.youtube.com/watch?v={vid}"
        print(f"\n=== {vid} ===")
        try:
            result = extract(url, VIDEOS_DIR)
        except Exception as e:
            print(f"  DOWNLOAD FAILED: {e}")
            continue

        video_path = result["video_path"]
        caption = result["caption"].split("\n")[0][:300]  # first line, capped

        try:
            transcript = scorer.transcribe(video_path)
        except Exception as e:
            transcript = ""
            print(f"  TRANSCRIBE FAILED: {e}")

        final_name = f"{vid}.mp4"
        final_path = os.path.join(VIDEOS_DIR, final_name)
        if video_path != final_path and os.path.exists(video_path):
            os.replace(video_path, final_path)

        print(f"  title: {result['title']}")
        print(f"  caption: {caption[:80]}")
        print(f"  transcript: {transcript[:80]}")

        rows.append({
            "video_id": vid, "file_path": final_name,
            "real_caption": caption, "transcript": transcript,
            "transcript_reliable": reliable,
        })

    print(f"\nDownloaded {len(rows)}/{len(VIDEOS)} videos successfully.")

    final_rows = []
    n = len(rows)
    for i, row in enumerate(rows):
        final_rows.append({
            "video_id": row["video_id"], "file_path": row["file_path"],
            "transcript": row["transcript"], "caption": row["real_caption"],
            "label": "coherent", "transcript_reliable": row["transcript_reliable"],
        })
        other = rows[(i + 1) % n]
        final_rows.append({
            "video_id": row["video_id"], "file_path": row["file_path"],
            "transcript": row["transcript"], "caption": other["real_caption"],
            "label": "mismatched", "transcript_reliable": row["transcript_reliable"],
        })

    csv_path = os.path.join(DATASET_DIR, "caption_coherence_20_sample.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "file_path", "transcript", "caption", "label", "transcript_reliable"])
        w.writeheader()
        w.writerows(final_rows)

    n_reliable = sum(1 for r in final_rows if r["transcript_reliable"])
    print(f"\nWrote {len(final_rows)} rows to {csv_path} ({n_reliable}/{len(final_rows)} with reliable transcripts)")


if __name__ == "__main__":
    main()
