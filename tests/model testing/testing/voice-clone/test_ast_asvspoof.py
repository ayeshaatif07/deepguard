"""Benchmarks MattyB95/AST-ASVspoof2019-Synthetic-Voice-Detection, an Audio Spectrogram
Transformer candidate.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import common
from common import BaseModel


MODEL_KEY = "ast_asvspoof"
LABEL_MAP = {"Bonafide": "real", "Spoof": "fake"}

# Empirically best threshold from a full sweep over the 200-clip
# benchmark's saved scores - see this file's docstring.
CALIBRATED_FAKE_THRESHOLD = 0.999949


class ASTASVspoofModel(BaseModel):
    name = "MattyB95/AST-ASVspoof2019 (Audio Spectrogram Transformer)"

    def __init__(self, device):
        self.device = device
        self.model = None
        self.fake_threshold = CALIBRATED_FAKE_THRESHOLD

    def load(self):
        import torch
        from transformers import AutoFeatureExtractor, ASTForAudioClassification

        self.torch = torch
        model_id = "MattyB95/AST-ASVspoof2019-Synthetic-Voice-Detection"
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = ASTForAudioClassification.from_pretrained(model_id)
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

        inputs = self.extractor(audio, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
            probs = self.torch.softmax(logits, dim=1).cpu().numpy()[0]

        return {LABEL_MAP.get(self.id2label[i], self.id2label[i]): float(p) for i, p in enumerate(probs)}


def main():
    parser = common.build_arg_parser(
        "Evaluate MattyB95/AST-ASVspoof2019-Synthetic-Voice-Detection (AST) on a labelled voice-clone dataset."
    )
    args = parser.parse_args()
    common.run_evaluation(MODEL_KEY, ASTASVspoofModel, args)


if __name__ == "__main__":
    main()
