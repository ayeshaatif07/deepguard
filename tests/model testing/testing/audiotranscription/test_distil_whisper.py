"""Benchmarks distil-whisper/distil-large-v3, a distilled Whisper variant."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import common
from common import BaseModel


MODEL_KEY = "distil_whisper"


class DistilWhisperModel(BaseModel):
    name = "distil-whisper/distil-large-v3"

    def __init__(self, device):
        self.device = device
        self.asr = None

    def load(self):
        from transformers import pipeline as hf_pipeline
        pipeline_device = self.device if self.device != "cpu" else -1
        self.asr = hf_pipeline(
            "automatic-speech-recognition", model=self.name, device=pipeline_device,
            chunk_length_s=30, stride_length_s=5,
        )

    def transcribe(self, audio_path):
        # Same repetition-loop mitigation as the production Whisper config, for a fair
        # comparison (not testing distil-whisper at a disadvantage with unmitigated settings).
        out = self.asr(audio_path, generate_kwargs={
            "condition_on_prev_tokens": False,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.3,
        })
        return out["text"].strip()


def main():
    parser = common.build_arg_parser("Evaluate distil-whisper/distil-large-v3 on the voice-manipulation dataset.")
    args = parser.parse_args()
    common.run_evaluation(MODEL_KEY, DistilWhisperModel, args)


if __name__ == "__main__":
    main()
