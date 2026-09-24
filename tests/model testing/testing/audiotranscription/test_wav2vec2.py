"""Benchmarks facebook/wav2vec2-large-960h-lv60-self, a CTC-based alternative to Whisper."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import common
from common import BaseModel


MODEL_KEY = "wav2vec2"


class Wav2Vec2Model(BaseModel):
    name = "facebook/wav2vec2-large-960h-lv60-self"

    def __init__(self, device):
        self.device = device
        self.asr = None

    def load(self):
        from transformers import pipeline as hf_pipeline
        pipeline_device = self.device if self.device != "cpu" else -1
        self.asr = hf_pipeline("automatic-speech-recognition", model=self.name, device=pipeline_device)

    def transcribe(self, audio_path):
        out = self.asr(audio_path)
        return out["text"].strip()


def main():
    parser = common.build_arg_parser("Evaluate facebook/wav2vec2-large-960h-lv60-self (CTC, English-only) on the voice-manipulation dataset.")
    args = parser.parse_args()
    common.run_evaluation(MODEL_KEY, Wav2Vec2Model, args)


if __name__ == "__main__":
    main()
