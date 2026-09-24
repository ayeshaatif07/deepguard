"""Benchmarks openai/whisper-small (244M params), the adopted Signal 3 transcription model."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import common
from common import BaseModel


MODEL_KEY = "whisper_small"


class WhisperSmallModel(BaseModel):
    name = "openai/whisper-small"

    def __init__(self, device):
        self.device = device
        self.asr = None

    def load(self):
        from transformers import pipeline as hf_pipeline
        pipeline_device = self.device if self.device != "cpu" else -1
        self.asr = hf_pipeline(
            "automatic-speech-recognition", model="openai/whisper-small", device=pipeline_device,
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
    parser = common.build_arg_parser("Evaluate openai/whisper-small (step up from production whisper-base) on the dataset.")
    args = parser.parse_args()
    common.run_evaluation(MODEL_KEY, WhisperSmallModel, args)


if __name__ == "__main__":
    main()
