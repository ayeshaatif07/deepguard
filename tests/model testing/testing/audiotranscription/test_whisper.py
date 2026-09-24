"""Benchmarks openai/whisper-base, the former Signal 3 transcription model."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import common
from common import BaseModel


MODEL_KEY = "whisper"


class WhisperModel(BaseModel):
    name = "openai/whisper-base"

    def __init__(self, device):
        self.device = device
        self.asr = None

    def load(self):
        from transformers import pipeline as hf_pipeline
        pipeline_device = self.device if self.device != "cpu" else -1
        self.asr = hf_pipeline(
            "automatic-speech-recognition", model="openai/whisper-base", device=pipeline_device,
            chunk_length_s=30, stride_length_s=5,
        )

    def transcribe(self, audio_path):
        out = self.asr(audio_path, generate_kwargs={
            "condition_on_prev_tokens": False,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.3,
        })
        return out["text"].strip()


def main():
    parser = common.build_arg_parser("Evaluate openai/whisper-base (production ASR model) on the voice-manipulation dataset.")
    args = parser.parse_args()
    common.run_evaluation(MODEL_KEY, WhisperModel, args)


if __name__ == "__main__":
    main()
