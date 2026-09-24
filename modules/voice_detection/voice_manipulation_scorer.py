"""Signal 3 - scores a transcript for manipulative/coercive language.
Whisper-small transcribes, then a fine-tuned GoEmotions RoBERTa binary head classifies.
"""

import os

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from modules.voice_detection.audio_utils import extract_audio_array, TARGET_SR

MODEL_DIR = os.path.join(os.path.dirname(__file__), "goemotions_finetuned")

# Used only by finetune_goemotions.py for its comparison run, not by this scorer.
MANIPULATION_EMOTIONS = {"fear", "nervousness", "anger", "annoyance", "disapproval", "disgust"}
MANIPULATION_THRESHOLD = 0.02


class VoiceManipulationScorer:
    ASR_MODEL_ID = "openai/whisper-small"

    # This fine-tuned head's own label order - see docstring.
    ID2LABEL = {0: "Non-manipulative", 1: "Manipulative"}

    def __init__(self, device=None):
        if device is None:
            device_str = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            device_str = device
        self.device = torch.device(device_str)
        asr_device = 0 if device_str == "mps" else -1

        from transformers import pipeline as hf_pipeline

        print(f"Loading {self.ASR_MODEL_ID} (transcription) on device {device_str}...")
        # chunk_length_s enables Whisper's chunked mode, required for clips over 30s.
        self.asr = hf_pipeline(
            "automatic-speech-recognition", model=self.ASR_MODEL_ID, device=asr_device,
            chunk_length_s=30, stride_length_s=5,
        )

        print(f"Loading fine-tuned GoEmotions (manipulation) from {MODEL_DIR} on device {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        self.model.to(self.device)
        self.model.eval()

    def transcribe(self, audio) -> str:
        """Transcribes a file path or an already-decoded 16kHz mono waveform."""
        # These generate_kwargs suppress Whisper's repetition-loop failure mode.
        out = self.asr(audio, generate_kwargs={
            "condition_on_prev_tokens": False,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.3,
        })
        return out["text"].strip()

    def score_manipulation(self, transcript: str):
        """Returns (manipulation_score 0.0-1.0, verdict, emotions).
        emotions is always empty; the fine-tuned head replaced the 28-class output.
        """
        if not transcript:
            return 0.0, "Non-manipulative", {}
        text = " ".join(transcript.split()[:500])  # stay within the model's token limit
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        manipulation_score = float(probs[1])
        verdict = self.ID2LABEL[1] if manipulation_score >= 0.5 else self.ID2LABEL[0]
        return manipulation_score, verdict, {}

    def analyze(self, input_path: str):
        """Full pipeline on a file path. Returns transcript, emotions, manipulation_score and verdict."""
        audio = extract_audio_array(input_path)
        return self.analyze_array(audio)

    def analyze_array(self, audio):
        """Same as analyze() but takes an already-decoded waveform."""
        transcript = self.transcribe(audio)
        manipulation_score, verdict, emotions = self.score_manipulation(transcript)
        return {
            "transcript": transcript,
            "emotions": emotions,
            "manipulation_score": manipulation_score,
            "verdict": verdict,
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        scorer = VoiceManipulationScorer()
        result = scorer.analyze(sys.argv[1])
        print(f"Transcript: {result['transcript']}")
        print(f"Verdict: {result['verdict']} (score: {result['manipulation_score']:.3f})")
