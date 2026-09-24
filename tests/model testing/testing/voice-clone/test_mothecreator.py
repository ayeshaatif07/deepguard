"""Benchmarks mo-thecreator/Deepfake-audio-detection, a Wav2Vec2 voice-clone candidate."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import common
from common import BaseModel


MODEL_KEY = "mothecreator"


class MoTheCreatorModel(BaseModel):
    name = "mo-thecreator/Deepfake-audio-detection (Wav2Vec2)"

    def __init__(self, device):
        self.device = device
        self.model = None

    def load(self):
        import torch
        from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

        self.torch = torch
        model_id = "mo-thecreator/Deepfake-audio-detection"
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def score_audio(self, audio_path):
        import soundfile as sf
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        inputs = self.extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
            probs = self.torch.softmax(logits, dim=1).cpu().numpy()[0]

        return {self.id2label[i].lower(): float(p) for i, p in enumerate(probs)}


def main():
    parser = common.build_arg_parser(
        "Evaluate mo-thecreator/Deepfake-audio-detection (Wav2Vec2) on a labelled voice-clone dataset."
    )
    args = parser.parse_args()
    common.run_evaluation(MODEL_KEY, MoTheCreatorModel, args)


if __name__ == "__main__":
    main()
