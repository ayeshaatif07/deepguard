"""Signal 2 - classifies a speech clip as Real or Cloned.
Uses this project's fine-tuned AST-ASVspoof2019 checkpoint.
"""

import os

import torch
from transformers import AutoFeatureExtractor, ASTForAudioClassification

from modules.voice_detection.audio_utils import extract_audio_array, TARGET_SR

MODEL_DIR = os.path.join(os.path.dirname(__file__), "ast_asvspoof_finetuned")


class VoiceCloneScorer:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available()
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
        else:
            self.device = torch.device(device)

        print(f"Loading fine-tuned AST-ASVspoof2019 (voice clone) from {MODEL_DIR} on device {self.device}...")
        self.extractor = AutoFeatureExtractor.from_pretrained(MODEL_DIR)
        self.model = ASTForAudioClassification.from_pretrained(MODEL_DIR)
        self.model.to(self.device)
        self.model.eval()

        # id2label from the fine-tuned config: {0: 'Bonafide', 1: 'Spoof'}
        # (preserved unchanged from the base checkpoint through fine-tuning)
        self.id2label = self.model.config.id2label

    def score_array(self, audio: "np.ndarray"):
        """Runs inference on a decoded mono 16kHz waveform.
        Returns (clone_probability, verdict) where verdict is "Cloned" or "Real".
        """
        inputs = self.extractor(
            audio, sampling_rate=TARGET_SR, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # Read label indices from id2label rather than assuming order, so a model
        # swap cannot silently invert the verdict.
        label_to_idx = {v: int(k) for k, v in self.id2label.items()}
        spoof_idx = label_to_idx["Spoof"]
        clone_probability = float(probs[spoof_idx])
        verdict = "Cloned" if clone_probability >= 0.5 else "Real"

        return clone_probability, verdict

    def score_file(self, input_path: str):
        """Decodes input_path (any audio/video container ffmpeg supports)
        and scores it. Convenience wrapper around score_array()."""
        audio = extract_audio_array(input_path)
        return self.score_array(audio)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        scorer = VoiceCloneScorer()
        prob, verdict = scorer.score_file(sys.argv[1])
        print(f"Verdict: {verdict} (clone probability: {prob:.1%})")
